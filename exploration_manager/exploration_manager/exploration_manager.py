import math

import numpy as np
import rclpy

from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from tf2_ros import Buffer, TransformListener


class ExplorationManager(Node):

    def __init__(self):
        super().__init__('exploration_manager')

        # ============================================================
        # EXPLORATION SETTINGS
        # ============================================================

        self.information_radius = self.declare_parameter(
            'information_radius',
            1.0
        ).value

        # Frontier scoring weights.

        # Information = how useful the frontier is.
        # Distance = how far the robot must travel.
        # Heading = how much the robot must turn.
        self.information_weight = self.declare_parameter(
            'information_weight',
            0.30
        ).value

        self.distance_weight = self.declare_parameter(
            'distance_weight',
            0.40
        ).value

        self.heading_weight = self.declare_parameter(
            'heading_weight',
            0.30
        ).value

        # Small tolerance used to identify the same frontier coordinate.
        self.frontier_match_tolerance = self.declare_parameter(
            'frontier_match_tolerance',
            0.01
        ).value

        # Prefer frontiers requiring no more than this amount of turning.
        # Larger turns are still allowed if no preferred frontier exists.
        self.preferred_turn_angle = math.radians(
            self.declare_parameter(
                'preferred_turn_angle_degrees',
                90.0
            ).value
        )

        self.required_empty_updates = self.declare_parameter(
            'required_empty_updates',
            3
        ).value

        self.map_frame = self.declare_parameter(
            'map_frame',
            'map'
        ).value

        self.robot_frame = self.declare_parameter(
            'robot_frame',
            'base_link'
        ).value

        # ============================================================
        # STORED DATA
        # ============================================================

        # Latest frontier positions received from the frontier detector.
        self.frontier_candidates = []

        # Latest occupancy grid map.
        self.map_msg = None
        self.map_array = None

        # Exact frontier locations that have already failed or succeeded.
        self.failed_regions = []
        self.reached_regions = []

        # Current Nav2 goal.
        self.current_goal = None
        self.navigation_goal_handle = None

        # Used to decide when exploration has finished.
        self.empty_frontier_update_count = 0

        # ============================================================
        # STATE MACHINE
        # ============================================================

        # Possible states:
        #
        # WAITING_FOR_DATA
        # READY
        # GOAL_SELECTED
        # NAVIGATING
        # WAITING_FOR_UPDATE
        # EXPLORATION_COMPLETE
        self.state = 'WAITING_FOR_DATA'

        # ============================================================
        # ROS SUBSCRIPTIONS
        # ============================================================

        # Receive frontier positions continuously from the
        # frontier detector.
        self.frontier_subscription = self.create_subscription(
            PoseArray,
            '/frontier_candidates',
            self.frontier_cb,
            10
        )

        # Receive the occupancy grid map from SLAM.
        self.map_subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_cb,
            10
        )

        # ============================================================
        # TF
        # ============================================================

        # TF gives the robot position and heading in the map.
        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        # ============================================================
        # NAV2
        # ============================================================

        # Used to send navigation goals to Nav2.
        self.navigation_client = ActionClient(
            self,
            NavigateToPose,
            'navigate_to_pose'
        )

        # Run the main exploration logic once every second.
        self.exploration_timer = self.create_timer(
            1.0,
            self.exploration_cb
        )

        self.get_logger().info(
            'Exploration manager started'
        )

    # ================================================================
    # FRONTIER CALLBACK
    # ================================================================

    def frontier_cb(self, msg):

        # Store the newest frontier list every time the
        # frontier detector publishes.
        self.frontier_candidates = list(
            msg.poses
        )

        if self.state == 'EXPLORATION_COMPLETE':
            return

        if self.state == 'WAITING_FOR_UPDATE':

            if self.frontier_candidates:

                self.empty_frontier_update_count = 0
                self.state = 'READY'

                self.get_logger().info(
                    'Updated frontier data received'
                )

            else:

                self.empty_frontier_update_count += 1

                self.get_logger().info(
                    f'Empty frontier update '
                    f'{self.empty_frontier_update_count}/'
                    f'{self.required_empty_updates}'
                )

                if (
                    self.empty_frontier_update_count
                    >= self.required_empty_updates
                ):

                    self.state = 'EXPLORATION_COMPLETE'

                    self.get_logger().info(
                        'Exploration complete'
                    )

            return

        if self.frontier_candidates:
            self.empty_frontier_update_count = 0

        self.check_ready()

    # ================================================================
    # MAP CALLBACK
    # ================================================================

    def map_cb(self, msg):

        self.map_msg = msg

        # Convert the flat OccupancyGrid list into a 2D array.
        self.map_array = np.array(
            msg.data
        ).reshape(
            (
                msg.info.height,
                msg.info.width
            )
        )

        self.check_ready()

    # ================================================================
    # START-UP CHECK
    # ================================================================

    def check_ready(self):

        # Exploration can begin once map and frontier data exist.
        if (
            self.state == 'WAITING_FOR_DATA'
            and self.map_array is not None
            and self.frontier_candidates
        ):

            self.state = 'READY'

            self.get_logger().info(
                'Exploration manager ready'
            )

    # ================================================================
    # ROBOT POSITION
    # ================================================================

    def get_robot_pose(self):

        try:

            # Get robot position relative to the map.
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.robot_frame,
                Time()
            )

            x = transform.transform.translation.x
            y = transform.transform.translation.y

            q = transform.transform.rotation

            # Convert quaternion orientation into yaw.
            yaw = math.atan2(
                2.0 * (
                    q.w * q.z
                    + q.x * q.y
                ),
                1.0 - 2.0 * (
                    q.y ** 2
                    + q.z ** 2
                )
            )

            return x, y, yaw

        except Exception as error:

            self.get_logger().warn(
                f'Could not get robot pose from TF: {error}'
            )

            return None

    # ================================================================
    # INFORMATION GAIN
    # ================================================================

    def information_gain(self, candidate):

        if (
            self.map_msg is None
            or self.map_array is None
        ):
            return 0

        resolution = self.map_msg.info.resolution
        origin = self.map_msg.info.origin

        q = origin.orientation

        # Find rotation of the occupancy grid origin.
        origin_yaw = math.atan2(
            2.0 * (
                q.w * q.z
                + q.x * q.y
            ),
            1.0 - 2.0 * (
                q.y ** 2
                + q.z ** 2
            )
        )

        # Frontier position relative to the map origin.
        dx = (
            candidate.position.x
            - origin.position.x
        )

        dy = (
            candidate.position.y
            - origin.position.y
        )

        # Convert world coordinates into map coordinates.
        local_x = (
            dx * math.cos(origin_yaw)
            + dy * math.sin(origin_yaw)
        )

        local_y = (
            -dx * math.sin(origin_yaw)
            + dy * math.cos(origin_yaw)
        )

        # Convert metres into map cell indices.
        col = math.floor(
            local_x / resolution
        )

        row = math.floor(
            local_y / resolution
        )

        rows, cols = self.map_array.shape

        # Frontier must be inside the occupancy grid.
        if (
            row < 0
            or row >= rows
            or col < 0
            or col >= cols
        ):
            return 0

        radius_cells = math.ceil(
            self.information_radius
            / resolution
        )

        unknown_count = 0

        for check_row in range(
            max(
                0,
                row - radius_cells
            ),
            min(
                rows,
                row + radius_cells + 1
            )
        ):

            for check_col in range(
                max(
                    0,
                    col - radius_cells
                ),
                min(
                    cols,
                    col + radius_cells + 1
                )
            ):

                distance = math.hypot(
                    (
                        check_row - row
                    ) * resolution,
                    (
                        check_col - col
                    ) * resolution
                )

                if (
                    distance
                    <= self.information_radius
                    and
                    self.map_array[
                        check_row,
                        check_col
                    ] == -1
                ):

                    unknown_count += 1

        return unknown_count

    # ================================================================
    # FRONTIER SELECTION
    # ================================================================

    def select_best_candidate(self):

        robot_pose = self.get_robot_pose()

        if robot_pose is None:
            return None

        robot_x, robot_y, robot_yaw = robot_pose

        valid_candidates = []
        distances = []
        information_values = []
        heading_changes = []

        failed_count = 0
        reached_count = 0

        # ------------------------------------------------------------
        # CHECK FRONTIERS
        # ------------------------------------------------------------

        for candidate in self.frontier_candidates:

            # Calculate straight-line distance from the robot.
            distance = math.hypot(
                candidate.position.x
                - robot_x,
                candidate.position.y
                - robot_y
            )

            # --------------------------------------------------------
            # FAILED FRONTIER CHECK
            # --------------------------------------------------------

            same_failed_frontier = any(
                math.hypot(
                    candidate.position.x
                    - failed_x,
                    candidate.position.y
                    - failed_y
                )
                <= self.frontier_match_tolerance

                for failed_x, failed_y
                in self.failed_regions
            )

            if same_failed_frontier:

                failed_count += 1
                continue

            # --------------------------------------------------------
            # REACHED FRONTIER CHECK
            # --------------------------------------------------------

            same_reached_frontier = any(
                math.hypot(
                    candidate.position.x
                    - reached_x,
                    candidate.position.y
                    - reached_y
                )
                <= self.frontier_match_tolerance

                for reached_x, reached_y
                in self.reached_regions
            )

            if same_reached_frontier:

                reached_count += 1
                continue

            # --------------------------------------------------------
            # HEADING CHANGE
            # --------------------------------------------------------

            # Direction from the robot to the frontier.
            goal_heading = math.atan2(
                candidate.position.y
                - robot_y,
                candidate.position.x
                - robot_x
            )

            # Smallest amount the robot would need to rotate.
            heading_change = abs(
                math.atan2(
                    math.sin(
                        goal_heading
                        - robot_yaw
                    ),
                    math.cos(
                        goal_heading
                        - robot_yaw
                    )
                )
            )

            valid_candidates.append(
                candidate
            )

            distances.append(
                distance
            )

            information_values.append(
                self.information_gain(
                    candidate
                )
            )

            heading_changes.append(
                heading_change
            )

        # ------------------------------------------------------------
        # NO VALID FRONTIERS
        # ------------------------------------------------------------

        if not valid_candidates:

            self.get_logger().warn(
                f'No new frontier candidates: '
                f'total={len(self.frontier_candidates)}, '
                f'failed={failed_count}, '
                f'reached={reached_count}'
            )

            return None

        # ------------------------------------------------------------
        # DIRECTION PREFERENCE
        # ------------------------------------------------------------

        # Prefer frontiers requiring no more than a 90 degree turn.
        #
        # This is only a preference.
        # If none exist, frontiers requiring larger turns are allowed.
        preferred_indices = [
            index
            for index, heading_change
            in enumerate(heading_changes)
            if heading_change
            <= self.preferred_turn_angle
        ]

        if preferred_indices:

            valid_candidates = [
                valid_candidates[index]
                for index in preferred_indices
            ]

            distances = [
                distances[index]
                for index in preferred_indices
            ]

            information_values = [
                information_values[index]
                for index in preferred_indices
            ]

            heading_changes = [
                heading_changes[index]
                for index in preferred_indices
            ]

            self.get_logger().info(
                f'Using {len(valid_candidates)} '
                f'forward-preferred frontier candidate(s)'
            )

        else:

            self.get_logger().info(
                'No forward-preferred frontier available; '
                'allowing larger turns'
            )

        # ------------------------------------------------------------
        # NORMALISE VALUES
        # ------------------------------------------------------------

        def normalise(
            values,
            equal_value
        ):

            values = np.array(
                values,
                dtype=float
            )

            minimum = np.min(values)
            maximum = np.max(values)

            # If all values are equal, this factor cannot
            # distinguish between the candidates.
            if maximum == minimum:

                return np.full(
                    len(values),
                    equal_value
                )

            return (
                values - minimum
            ) / (
                maximum - minimum
            )

        information_norm = normalise(
            information_values,
            1.0
        )

        distance_norm = normalise(
            distances,
            0.0
        )

        heading_norm = np.array(
            heading_changes,
            dtype=float
        ) / math.pi

        # ------------------------------------------------------------
        # FRONTIER SCORE
        # ------------------------------------------------------------

        # Information contributes 10%.
        # Distance contributes a 40% penalty.
        # Rotation contributes a 50% penalty.
        scores = (
            self.information_weight
            * information_norm
            -
            self.distance_weight
            * distance_norm
            -
            self.heading_weight
            * heading_norm
        )

        # Select the candidate with the highest score.
        #
        # The highest score is selected even if all scores are negative.
        best_index = int(
            np.argmax(scores)
        )

        candidate = valid_candidates[
            best_index
        ]

        self.get_logger().info(
            f'Selected frontier '
            f'x={candidate.position.x:.2f}, '
            f'y={candidate.position.y:.2f}, '
            f'distance={distances[best_index]:.2f} m, '
            f'turn='
            f'{math.degrees(heading_changes[best_index]):.1f} deg, '
            f'information='
            f'{information_values[best_index]}, '
            f'score={scores[best_index]:.3f}'
        )

        return candidate

    # ================================================================
    # SEND GOAL TO NAV2
    # ================================================================

    def send_navigation_goal(self):

        if self.current_goal is None:
            return False

        if not self.navigation_client.wait_for_server(
            timeout_sec=1.0
        ):

            self.get_logger().warn(
                'NavigateToPose action server unavailable'
            )

            return False

        # Get the robot position when the goal is being sent.
        #
        # This is used to calculate the direction from the robot
        # toward the selected frontier.
        robot_pose = self.get_robot_pose()

        if robot_pose is None:

            self.get_logger().warn(
                'Cannot send navigation goal because '
                'robot pose is unavailable'
            )

            return False

        robot_x, robot_y, _ = robot_pose

        goal = NavigateToPose.Goal()

        goal.pose.header.frame_id = (
            self.map_frame
        )

        goal.pose.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        goal.pose.pose = (
            self.current_goal
        )

        # ------------------------------------------------------------
        # GOAL ORIENTATION
        # ------------------------------------------------------------

        goal_yaw = math.atan2(
            self.current_goal.position.y
            - robot_y,
            self.current_goal.position.x
            - robot_x
        )

        # Convert the desired yaw into a quaternion.
        goal.pose.pose.orientation.x = 0.0
        goal.pose.pose.orientation.y = 0.0

        goal.pose.pose.orientation.z = math.sin(
            goal_yaw / 2.0
        )

        goal.pose.pose.orientation.w = math.cos(
            goal_yaw / 2.0
        )

        self.get_logger().info(
            f'Navigation goal orientation='
            f'{math.degrees(goal_yaw):.1f} deg'
        )

        future = (
            self.navigation_client
            .send_goal_async(goal)
        )

        future.add_done_callback(
            self.navigation_goal_response
        )

        return True

    # ================================================================
    # NAV2 GOAL RESPONSE
    # ================================================================

    def navigation_goal_response(
        self,
        future
    ):

        try:

            goal_handle = future.result()

        except Exception as error:

            self.get_logger().error(
                f'Could not send navigation goal: '
                f'{error}'
            )

            self.current_goal = None
            self.navigation_goal_handle = None
            self.state = 'READY'

            return

        if not goal_handle.accepted:

            self.get_logger().warn(
                'Navigation goal rejected'
            )

            self.fail_current_frontier()

            return

        self.navigation_goal_handle = (
            goal_handle
        )

        self.state = 'NAVIGATING'

        self.get_logger().info(
            'Navigation goal accepted'
        )

        result_future = (
            goal_handle
            .get_result_async()
        )

        result_future.add_done_callback(
            self.navigation_result_cb
        )

    # ================================================================
    # NAVIGATION RESULT
    # ================================================================

    def navigation_result_cb(
        self,
        future
    ):

        try:

            status = (
                future.result().status
            )

        except Exception as error:

            self.get_logger().error(
                f'Could not obtain navigation result: '
                f'{error}'
            )

            self.current_goal = None
            self.navigation_goal_handle = None
            self.state = 'READY'

            return

        # ------------------------------------------------------------
        # SUCCESS
        # ------------------------------------------------------------

        if (
            status
            == GoalStatus.STATUS_SUCCEEDED
        ):

            self.get_logger().info(
                'Frontier reached successfully'
            )

            if self.current_goal is not None:

                reached_position = (
                    self.current_goal.position.x,
                    self.current_goal.position.y
                )

                # Check whether this exact frontier has already
                # been stored as reached.
                duplicate = any(
                    math.hypot(
                        reached_position[0]
                        - old_x,
                        reached_position[1]
                        - old_y
                    )
                    <= self.frontier_match_tolerance

                    for old_x, old_y
                    in self.reached_regions
                )

                if not duplicate:

                    self.reached_regions.append(
                        reached_position
                    )

                    self.get_logger().info(
                        f'Reached frontier recorded: '
                        f'x={reached_position[0]:.2f}, '
                        f'y={reached_position[1]:.2f}'
                    )

            self.current_goal = None
            self.navigation_goal_handle = None

            self.empty_frontier_update_count = 0

            # Wait for updated frontier information before
            # selecting another goal.
            self.state = 'WAITING_FOR_UPDATE'

            return

        # ------------------------------------------------------------
        # ABORTED
        # ------------------------------------------------------------

        if (
            status
            == GoalStatus.STATUS_ABORTED
        ):

            self.get_logger().warn(
                'Navigation failed: ABORTED'
            )

            self.fail_current_frontier()

            return

        # ------------------------------------------------------------
        # OTHER RESULT
        # ------------------------------------------------------------
        self.get_logger().warn(
            f'Navigation ended with status={status}'
        )

        self.fail_current_frontier()

    # ================================================================
    # STORE FAILED FRONTIER
    # ================================================================

    def fail_current_frontier(self):

        if self.current_goal is not None:

            failed_position = (
                self.current_goal.position.x,
                self.current_goal.position.y
            )

            # Check whether this exact failed frontier has
            # already been recorded.
            duplicate = any(
                math.hypot(
                    failed_position[0]
                    - old_x,
                    failed_position[1]
                    - old_y
                )
                <= self.frontier_match_tolerance

                for old_x, old_y
                in self.failed_regions
            )

            if not duplicate:

                self.failed_regions.append(
                    failed_position
                )

                self.get_logger().warn(
                    f'Failed frontier recorded: '
                    f'x={failed_position[0]:.2f}, '
                    f'y={failed_position[1]:.2f}'
                )

        self.current_goal = None
        self.navigation_goal_handle = None

        self.state = 'READY'

    # ================================================================
    # MAIN EXPLORATION LOOP
    # ================================================================

    def exploration_cb(self):

        # Wait for Nav2 to finish the current navigation goal.
        if self.state == 'NAVIGATING':
            return

        # Only select a new frontier when ready.
        if self.state != 'READY':
            return

        candidate = (
            self.select_best_candidate()
        )

        if candidate is None:

            # Wait for updated frontier information.
            self.state = 'WAITING_FOR_UPDATE'
            return

        self.current_goal = (
            candidate
        )

        self.state = 'GOAL_SELECTED'

        # Send the selected frontier to Nav2.
        if not self.send_navigation_goal():

            self.current_goal = None
            self.navigation_goal_handle = None

            self.state = 'READY'


def main(args=None):

    rclpy.init(
        args=args
    )

    node = ExplorationManager()

    rclpy.spin(
        node
    )

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()