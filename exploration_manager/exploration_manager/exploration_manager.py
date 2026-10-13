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

        # Radius used to count unknown map cells around a frontier.
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
            0.35
        ).value

        self.distance_weight = self.declare_parameter(
            'distance_weight',
            0.35
        ).value

        self.heading_weight = self.declare_parameter(
            'heading_weight',
            0.3
        ).value

        # Ignore goals that are already very close to the robot.
        self.min_goal_distance = self.declare_parameter(
            'min_goal_distance',
            0.5
        ).value

        # Ignore areas close to previously failed goals.
        self.failed_region_radius = self.declare_parameter(
            'failed_region_radius',
            0.5
        ).value

        # Ignore areas close to goals already reached.
        self.reached_region_radius = self.declare_parameter(
            'reached_region_radius',
            0.5
        ).value

        # Number of empty frontier updates before exploration ends.
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
        # STUCK DETECTION
        # ============================================================

        # Robot must move at least this distance to count as progress.
        self.stuck_distance = self.declare_parameter(
            'stuck_distance',
            0.15
        ).value

        # Rotation can also count as progress.
        self.stuck_angle = math.radians(
            self.declare_parameter(
                'stuck_angle_degrees',
                20.0
            ).value
        )

        # Cancel the goal if no useful movement occurs for this long.
        self.stuck_timeout = self.declare_parameter(
            'stuck_timeout',
            4.0
        ).value

        # ============================================================
        # STORED DATA
        # ============================================================

        # Latest frontier positions received from the frontier detector.
        self.frontier_candidates = []

        # Latest occupancy grid map.
        self.map_msg = None
        self.map_array = None

        # Areas that should not be selected again.
        self.failed_regions = []
        self.reached_regions = []

        # Current Nav2 goal.
        self.current_goal = None
        self.navigation_goal_handle = None

        # Used to decide when exploration has finished.
        self.empty_frontier_update_count = 0

        # Used to monitor robot movement.
        self.last_progress_pose = None
        self.last_progress_time = None

        # ============================================================
        # STATE MACHINE
        # ============================================================

        # Possible states:
        #
        # WAITING_FOR_DATA
        # READY
        # GOAL_SELECTED
        # NAVIGATING
        # CANCELING
        # WAITING_FOR_UPDATE
        # EXPLORATION_COMPLETE
        self.state = 'WAITING_FOR_DATA'

        # ============================================================
        # ROS SUBSCRIPTIONS
        # ============================================================

        # Receive frontier positions from the frontier detector.
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

        self.frontier_candidates = list(msg.poses)

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

        except Exception:
            return None

    # ================================================================
    # INFORMATION GAIN
    # ================================================================

    def information_gain(self, candidate):

        # Count unknown map cells around the frontier.
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

        dx = (
            candidate.position.x
            - origin.position.x
        )

        dy = (
            candidate.position.y
            - origin.position.y
        )

        local_x = (
            dx * math.cos(origin_yaw)
            + dy * math.sin(origin_yaw)
        )

        local_y = (
            -dx * math.sin(origin_yaw)
            + dy * math.cos(origin_yaw)
        )

        col = math.floor(
            local_x / resolution
        )

        row = math.floor(
            local_y / resolution
        )

        rows, cols = self.map_array.shape

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
            max(0, row - radius_cells),
            min(rows, row + radius_cells + 1)
        ):

            for check_col in range(
                max(0, col - radius_cells),
                min(cols, col + radius_cells + 1)
            ):

                distance = math.hypot(
                    (check_row - row) * resolution,
                    (check_col - col) * resolution
                )

                if (
                    distance <= self.information_radius
                    and self.map_array[
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

        for candidate in self.frontier_candidates:

            # --------------------------------------------------------
            # DISTANCE
            # --------------------------------------------------------

            # Straight-line distance from robot to frontier.
            distance = math.hypot(
                candidate.position.x - robot_x,
                candidate.position.y - robot_y
            )

            # Ignore goals already very close to the robot.
            if distance < self.min_goal_distance:
                continue

            # --------------------------------------------------------
            # FAILED AREAS
            # --------------------------------------------------------

            near_failed_region = any(
                math.hypot(
                    candidate.position.x - failed_x,
                    candidate.position.y - failed_y
                ) < self.failed_region_radius
                for failed_x, failed_y
                in self.failed_regions
            )

            if near_failed_region:
                continue

            # --------------------------------------------------------
            # REACHED AREAS
            # --------------------------------------------------------

            near_reached_region = any(
                math.hypot(
                    candidate.position.x - reached_x,
                    candidate.position.y - reached_y
                ) < self.reached_region_radius
                for reached_x, reached_y
                in self.reached_regions
            )

            if near_reached_region:
                continue

            # --------------------------------------------------------
            # HEADING CHANGE
            # --------------------------------------------------------

            # Direction from the robot to the frontier.
            goal_heading = math.atan2(
                candidate.position.y - robot_y,
                candidate.position.x - robot_x
            )

            heading_change = abs(
                math.atan2(
                    math.sin(
                        goal_heading - robot_yaw
                    ),
                    math.cos(
                        goal_heading - robot_yaw
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

        if not valid_candidates:

            self.get_logger().info(
                'No valid frontier candidates after filtering'
            )

            return None

        # ============================================================
        # NORMALISE VALUES
        # ============================================================

        def normalise(values, equal_value):

            values = np.array(
                values,
                dtype=float
            )

            minimum = np.min(values)
            maximum = np.max(values)

            # If all candidates have the same value,
            # give them the same normalised score.
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

        # ============================================================
        # FRONTIER SCORE
        # ============================================================

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
            f'turn={math.degrees(heading_changes[best_index]):.1f} deg, '
            f'information={information_values[best_index]}, '
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

        # Make sure the orientation quaternion is valid.
        q = goal.pose.pose.orientation

        if (
            q.x == 0.0
            and q.y == 0.0
            and q.z == 0.0
            and q.w == 0.0
        ):
            q.w = 1.0

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
                f'Could not send navigation goal: {error}'
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

        self.last_progress_pose = None
        self.last_progress_time = None

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

            status = future.result().status

        except Exception as error:

            self.get_logger().error(
                f'Could not obtain navigation result: {error}'
            )

            # Do not send another goal while Nav2 state is unknown.
            self.state = 'CANCELING'

            return

        # ------------------------------------------------------------
        # SUCCESS
        # ------------------------------------------------------------

        if status == GoalStatus.STATUS_SUCCEEDED:

            self.get_logger().info(
                'Frontier reached successfully'
            )

            if self.current_goal is not None:

                reached_position = (
                    self.current_goal.position.x,
                    self.current_goal.position.y
                )

                # Avoid storing nearly identical reached positions.
                duplicate = any(
                    math.hypot(
                        reached_position[0] - old_x,
                        reached_position[1] - old_y
                    ) < self.reached_region_radius
                    for old_x, old_y
                    in self.reached_regions
                )

                if not duplicate:

                    self.reached_regions.append(
                        reached_position
                    )

                    self.get_logger().info(
                        f'Reached region recorded: '
                        f'x={reached_position[0]:.2f}, '
                        f'y={reached_position[1]:.2f}'
                    )

            self.current_goal = None
            self.navigation_goal_handle = None

            self.last_progress_pose = None
            self.last_progress_time = None

            self.empty_frontier_update_count = 0

            self.state = 'WAITING_FOR_UPDATE'

            return

        # ------------------------------------------------------------
        # CANCELLED GOAL
        # ------------------------------------------------------------

        if (
            self.state == 'CANCELING'
            and status == GoalStatus.STATUS_CANCELED
        ):

            self.get_logger().warn(
                'Navigation cancellation complete. '
                'Marking frontier as failed.'
            )

            self.fail_current_frontier()

            return

        # ------------------------------------------------------------
        # OTHER FAILURE
        # ------------------------------------------------------------

        if status == GoalStatus.STATUS_ABORTED:
            status_name = 'ABORTED'

        elif status == GoalStatus.STATUS_CANCELED:
            status_name = 'CANCELED'

        else:
            status_name = str(status)

        self.get_logger().warn(
            f'Navigation failed: {status_name}'
        )

        self.fail_current_frontier()

    # ================================================================
    # STUCK DETECTION
    # ================================================================

    def check_stuck(self):

        robot_pose = self.get_robot_pose()

        if robot_pose is None:
            return

        x, y, yaw = robot_pose
        now = self.get_clock().now()

        if self.last_progress_pose is None:

            self.last_progress_pose = (
                x,
                y,
                yaw
            )

            self.last_progress_time = now

            return

        old_x, old_y, old_yaw = (
            self.last_progress_pose
        )

        # Check how far the robot has moved.
        translation = math.hypot(
            x - old_x,
            y - old_y
        )

        # Check how much the robot has rotated.
        heading_change = abs(
            math.atan2(
                math.sin(
                    yaw - old_yaw
                ),
                math.cos(
                    yaw - old_yaw
                )
            )
        )

        # Moving or rotating enough counts as progress.
        if (
            translation >= self.stuck_distance
            or heading_change >= self.stuck_angle
        ):

            self.last_progress_pose = (
                x,
                y,
                yaw
            )

            self.last_progress_time = now

            return

        stationary_time = (
            now
            - self.last_progress_time
        ).nanoseconds / 1e9

        if stationary_time >= self.stuck_timeout:

            self.get_logger().warn(
                f'Robot appears stuck: '
                f'translation={translation:.2f} m, '
                f'rotation={math.degrees(heading_change):.1f} deg, '
                f'time={stationary_time:.1f} s'
            )

            self.cancel_stuck_goal()

    # ================================================================
    # CANCEL STUCK GOAL
    # ================================================================

    def cancel_stuck_goal(self):

        if self.state != 'NAVIGATING':
            return

        if self.navigation_goal_handle is None:

            self.get_logger().warn(
                'Stuck detected but no active navigation goal exists'
            )

            return

        # Change state first so no new goal can be sent.
        self.state = 'CANCELING'

        self.get_logger().warn(
            'Cancelling stuck navigation goal'
        )

        cancel_future = (
            self.navigation_goal_handle
            .cancel_goal_async()
        )

        cancel_future.add_done_callback(
            self.cancel_response_cb
        )

    # ================================================================
    # CANCELLATION RESPONSE
    # ================================================================

    def cancel_response_cb(
        self,
        future
    ):

        try:

            response = future.result()

        except Exception as error:

            self.get_logger().error(
                f'Cancellation request failed: {error}'
            )

            if (
                self.current_goal is not None
                and self.navigation_goal_handle is not None
            ):

                self.state = 'NAVIGATING'

                self.last_progress_pose = None
                self.last_progress_time = None

            return

        if not response.goals_canceling:

            self.get_logger().warn(
                'Nav2 did not accept cancellation. '
                'Keeping current goal active.'
            )

            self.state = 'NAVIGATING'

            self.last_progress_pose = None
            self.last_progress_time = None

            return

        self.get_logger().info(
            'Cancellation accepted by Nav2. '
            'Waiting for cancellation result.'
        )

    # ================================================================
    # STORE FAILED FRONTIER
    # ================================================================

    def fail_current_frontier(self):

        if self.current_goal is not None:

            failed_position = (
                self.current_goal.position.x,
                self.current_goal.position.y
            )

            duplicate = any(
                math.hypot(
                    failed_position[0] - old_x,
                    failed_position[1] - old_y
                ) < self.failed_region_radius
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

        self.last_progress_pose = None
        self.last_progress_time = None

        self.state = 'READY'

    # ================================================================
    # MAIN EXPLORATION LOOP
    # ================================================================

    def exploration_cb(self):

        # While navigating, only monitor progress.
        if self.state == 'NAVIGATING':

            self.check_stuck()
            return

        # Never send another goal while cancellation is pending.
        if self.state == 'CANCELING':
            return

        # Only select a frontier when the manager is ready.
        if self.state != 'READY':
            return

        candidate = self.select_best_candidate()

        if candidate is None:

            self.state = 'WAITING_FOR_UPDATE'
            return

        self.current_goal = candidate

        self.last_progress_pose = None
        self.last_progress_time = None

        self.state = 'GOAL_SELECTED'

        # Send the chosen frontier to Nav2.
        if not self.send_navigation_goal():

            self.current_goal = None
            self.navigation_goal_handle = None

            self.state = 'READY'


def main(args=None):

    rclpy.init(args=args)

    node = ExplorationManager()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()