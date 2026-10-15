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

        super().__init__(
            'exploration_manager'
        )

        # ============================================================
        # EXPLORATION SETTINGS
        # ============================================================

        self.information_radius = self.declare_parameter(
            'information_radius',
            1.0
        ).value

        # Frontier scoring weights.
        #
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

        # ------------------------------------------------------------
        # MINIMUM GOAL DISTANCE
        # ------------------------------------------------------------

        # Ignore frontier goals already closer than 0.30 m.
        #
        # This prevents very short goals that provide little useful
        # exploration and may already lie within Nav2 goal tolerance.
        self.min_goal_distance = self.declare_parameter(
            'min_goal_distance',
            0.30
        ).value

        # ------------------------------------------------------------
        # FAILED FRONTIER EXCLUSION
        # ------------------------------------------------------------

        # A failed navigation attempt blocks nearby frontier
        # representatives within this radius.
        #
        # This prevents repeated navigation attempts to slightly
        # shifted versions of the same difficult frontier.
        self.failed_frontier_radius = self.declare_parameter(
            'failed_frontier_radius',
            0.40
        ).value

        # Small tolerance used to identify the same exact reached
        # frontier coordinate.
        self.frontier_match_tolerance = self.declare_parameter(
            'frontier_match_tolerance',
            0.01
        ).value

        # Prefer frontiers requiring no more than this amount
        # of turning.
        #
        # Larger turns remain possible when no forward-preferred
        # candidate exists.
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

        # Latest frontier positions from the Frontier Detector.
        self.frontier_candidates = []

        # Latest occupancy-grid map.
        self.map_msg = None
        self.map_array = None

        # Failed navigation locations.
        self.failed_regions = []

        # Successfully reached exact frontier locations.
        self.reached_regions = []

        # Current navigation goal.
        self.current_goal = None

        self.navigation_goal_handle = None

        # Used to decide when exploration has finished.
        self.empty_frontier_update_count = 0

        # ============================================================
        # STATE MACHINE
        # ============================================================

        # WAITING_FOR_DATA
        # READY
        # GOAL_SELECTED
        # NAVIGATING
        # WAITING_FOR_UPDATE
        # EXPLORATION_COMPLETE
        self.state = (
            'WAITING_FOR_DATA'
        )

        # ============================================================
        # ROS SUBSCRIPTIONS
        # ============================================================

        self.frontier_subscription = self.create_subscription(
            PoseArray,
            '/frontier_candidates',
            self.frontier_cb,
            10
        )

        self.map_subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_cb,
            10
        )

        # ============================================================
        # TF
        # ============================================================

        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        # ============================================================
        # NAV2
        # ============================================================

        self.navigation_client = ActionClient(
            self,
            NavigateToPose,
            'navigate_to_pose'
        )

        # Run exploration decision logic once per second.
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

    def frontier_cb(
        self,
        msg
    ):

        self.frontier_candidates = list(
            msg.poses
        )

        if (
            self.state
            == 'EXPLORATION_COMPLETE'
        ):
            return

        if (
            self.state
            == 'WAITING_FOR_UPDATE'
        ):

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

                    self.state = (
                        'EXPLORATION_COMPLETE'
                    )

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

    def map_cb(
        self,
        msg
    ):

        self.map_msg = msg

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

        if (
            self.state
            == 'WAITING_FOR_DATA'
            and
            self.map_array is not None
            and
            self.frontier_candidates
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

            transform = (
                self.tf_buffer
                .lookup_transform(
                    self.map_frame,
                    self.robot_frame,
                    Time()
                )
            )

            x = (
                transform
                .transform
                .translation
                .x
            )

            y = (
                transform
                .transform
                .translation
                .y
            )

            q = (
                transform
                .transform
                .rotation
            )

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

            return (
                x,
                y,
                yaw
            )

        except Exception as error:

            self.get_logger().warn(
                f'Could not get robot pose from TF: '
                f'{error}'
            )

            return None

    # ================================================================
    # INFORMATION GAIN
    # ================================================================

    def information_gain(
        self,
        candidate
    ):

        if (
            self.map_msg is None
            or
            self.map_array is None
        ):
            return 0

        resolution = (
            self.map_msg
            .info
            .resolution
        )

        origin = (
            self.map_msg
            .info
            .origin
        )

        q = origin.orientation

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

        dx = (
            candidate.position.x
            - origin.position.x
        )

        dy = (
            candidate.position.y
            - origin.position.y
        )

        # Convert world coordinates into coordinates relative
        # to the occupancy-grid origin.
        local_x = (
            dx * math.cos(
                origin_yaw
            )
            +
            dy * math.sin(
                origin_yaw
            )
        )

        local_y = (
            -dx * math.sin(
                origin_yaw
            )
            +
            dy * math.cos(
                origin_yaw
            )
        )

        col = math.floor(
            local_x
            / resolution
        )

        row = math.floor(
            local_y
            / resolution
        )

        rows, cols = (
            self.map_array.shape
        )

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

        robot_pose = (
            self.get_robot_pose()
        )

        if robot_pose is None:
            return None

        (
            robot_x,
            robot_y,
            robot_yaw
        ) = robot_pose

        valid_candidates = []

        distances = []

        information_values = []

        heading_changes = []

        failed_count = 0

        reached_count = 0

        too_close_count = 0

        # ------------------------------------------------------------
        # CHECK FRONTIERS
        # ------------------------------------------------------------

        for candidate in self.frontier_candidates:

            # Straight-line distance between robot and frontier.
            distance = math.hypot(
                candidate.position.x
                - robot_x,
                candidate.position.y
                - robot_y
            )

            # --------------------------------------------------------
            # MINIMUM GOAL DISTANCE
            # --------------------------------------------------------

            # Skip frontier goals that are already extremely close
            # to the robot.
            if (
                distance
                < self.min_goal_distance
            ):

                too_close_count += 1

                continue

            # --------------------------------------------------------
            # FAILED FRONTIER CHECK
            # --------------------------------------------------------

            # A previously failed frontier blocks a 0.40 m region.
            #
            # This prevents the detector from moving its representative
            # slightly and causing essentially the same failed goal to
            # be attempted again.
            inside_failed_region = any(
                math.hypot(
                    candidate.position.x
                    - failed_x,
                    candidate.position.y
                    - failed_y
                )
                <= self.failed_frontier_radius

                for (
                    failed_x,
                    failed_y
                ) in self.failed_regions
            )

            if inside_failed_region:

                failed_count += 1

                continue

            # --------------------------------------------------------
            # REACHED FRONTIER CHECK
            # --------------------------------------------------------

            # Successfully reached frontiers still use the very
            # small exact-coordinate tolerance.
            same_reached_frontier = any(
                math.hypot(
                    candidate.position.x
                    - reached_x,
                    candidate.position.y
                    - reached_y
                )
                <= self.frontier_match_tolerance

                for (
                    reached_x,
                    reached_y
                ) in self.reached_regions
            )

            if same_reached_frontier:

                reached_count += 1

                continue

            # --------------------------------------------------------
            # HEADING CHANGE
            # --------------------------------------------------------

            goal_heading = math.atan2(
                candidate.position.y
                - robot_y,
                candidate.position.x
                - robot_x
            )

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
                f'total='
                f'{len(self.frontier_candidates)}, '
                f'too_close={too_close_count}, '
                f'failed={failed_count}, '
                f'reached={reached_count}'
            )

            return None

        # ------------------------------------------------------------
        # DIRECTION PREFERENCE
        # ------------------------------------------------------------

        # Prefer frontiers requiring no more than a 90 degree turn.
        #
        # If none exist, larger turns are still permitted.
        preferred_indices = [
            index
            for (
                index,
                heading_change
            ) in enumerate(
                heading_changes
            )
            if (
                heading_change
                <= self.preferred_turn_angle
            )
        ]

        if preferred_indices:

            valid_candidates = [
                valid_candidates[
                    index
                ]
                for index
                in preferred_indices
            ]

            distances = [
                distances[
                    index
                ]
                for index
                in preferred_indices
            ]

            information_values = [
                information_values[
                    index
                ]
                for index
                in preferred_indices
            ]

            heading_changes = [
                heading_changes[
                    index
                ]
                for index
                in preferred_indices
            ]

            self.get_logger().info(
                f'Using '
                f'{len(valid_candidates)} '
                f'forward-preferred '
                f'frontier candidate(s)'
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

            minimum = np.min(
                values
            )

            maximum = np.max(
                values
            )

            if (
                maximum
                == minimum
            ):

                return np.full(
                    len(values),
                    equal_value
                )

            return (
                values
                - minimum
            ) / (
                maximum
                - minimum
            )

        information_norm = normalise(
            information_values,
            1.0
        )

        distance_norm = normalise(
            distances,
            0.0
        )

        heading_norm = (
            np.array(
                heading_changes,
                dtype=float
            )
            / math.pi
        )

        # ------------------------------------------------------------
        # FRONTIER SCORE
        # ------------------------------------------------------------

        # Information contributes positively.
        #
        # Distance and heading change are penalties.
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

        best_index = int(
            np.argmax(
                scores
            )
        )

        candidate = (
            valid_candidates[
                best_index
            ]
        )

        self.get_logger().info(
            f'Selected frontier '
            f'x={candidate.position.x:.2f}, '
            f'y={candidate.position.y:.2f}, '
            f'distance='
            f'{distances[best_index]:.2f} m, '
            f'turn='
            f'{math.degrees(heading_changes[best_index]):.1f} deg, '
            f'information='
            f'{information_values[best_index]}, '
            f'score='
            f'{scores[best_index]:.3f}'
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

        # Get the current robot position so the final goal
        # orientation follows the approach direction.
        robot_pose = (
            self.get_robot_pose()
        )

        if robot_pose is None:

            self.get_logger().warn(
                'Cannot send navigation goal because '
                'robot pose is unavailable'
            )

            return False

        (
            robot_x,
            robot_y,
            _
        ) = robot_pose

        goal = (
            NavigateToPose.Goal()
        )

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

        goal.pose.pose.orientation.x = 0.0

        goal.pose.pose.orientation.y = 0.0

        goal.pose.pose.orientation.z = (
            math.sin(
                goal_yaw / 2.0
            )
        )

        goal.pose.pose.orientation.w = (
            math.cos(
                goal_yaw / 2.0
            )
        )

        self.get_logger().info(
            f'Navigation goal orientation='
            f'{math.degrees(goal_yaw):.1f} deg'
        )

        future = (
            self.navigation_client
            .send_goal_async(
                goal
            )
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

            goal_handle = (
                future.result()
            )

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
                future.result()
                .status
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

            if (
                self.current_goal
                is not None
            ):

                reached_position = (
                    self.current_goal.position.x,
                    self.current_goal.position.y
                )

                duplicate = any(
                    math.hypot(
                        reached_position[0]
                        - old_x,
                        reached_position[1]
                        - old_y
                    )
                    <= self.frontier_match_tolerance

                    for (
                        old_x,
                        old_y
                    ) in self.reached_regions
                )

                if not duplicate:

                    self.reached_regions.append(
                        reached_position
                    )

                    self.get_logger().info(
                        f'Reached frontier recorded: '
                        f'x='
                        f'{reached_position[0]:.2f}, '
                        f'y='
                        f'{reached_position[1]:.2f}'
                    )

            self.current_goal = None

            self.navigation_goal_handle = None

            self.empty_frontier_update_count = 0

            # Wait for a fresh frontier list before choosing
            # another goal.
            self.state = (
                'WAITING_FOR_UPDATE'
            )

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
            f'Navigation ended with '
            f'status={status}'
        )

        self.fail_current_frontier()

    # ================================================================
    # STORE FAILED FRONTIER
    # ================================================================

    def fail_current_frontier(self):

        if (
            self.current_goal
            is not None
        ):

            failed_position = (
                self.current_goal.position.x,
                self.current_goal.position.y
            )

            # Only avoid storing essentially identical coordinates.
            #
            # Candidate rejection itself uses the larger
            # failed_frontier_radius.
            duplicate = any(
                math.hypot(
                    failed_position[0]
                    - old_x,
                    failed_position[1]
                    - old_y
                )
                <= self.frontier_match_tolerance

                for (
                    old_x,
                    old_y
                ) in self.failed_regions
            )

            if not duplicate:

                self.failed_regions.append(
                    failed_position
                )

                self.get_logger().warn(
                    f'Failed frontier recorded: '
                    f'x='
                    f'{failed_position[0]:.2f}, '
                    f'y='
                    f'{failed_position[1]:.2f}; '
                    f'excluding nearby goals within '
                    f'{self.failed_frontier_radius:.2f} m'
                )

        self.current_goal = None

        self.navigation_goal_handle = None

        self.state = 'READY'

    # ================================================================
    # MAIN EXPLORATION LOOP
    # ================================================================

    def exploration_cb(self):

        # Nav2 owns the current navigation attempt until it
        # returns a result.
        if (
            self.state
            == 'NAVIGATING'
        ):
            return

        # Only select a frontier when ready.
        if (
            self.state
            != 'READY'
        ):
            return

        candidate = (
            self.select_best_candidate()
        )

        if candidate is None:

            # Wait for a fresh frontier update.
            self.state = (
                'WAITING_FOR_UPDATE'
            )

            return

        self.current_goal = (
            candidate
        )

        self.state = (
            'GOAL_SELECTED'
        )

        if not self.send_navigation_goal():

            self.current_goal = None

            self.navigation_goal_handle = None

            self.state = 'READY'


def main(args=None):

    rclpy.init(
        args=args
    )

    node = (
        ExplorationManager()
    )

    rclpy.spin(
        node
    )

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()