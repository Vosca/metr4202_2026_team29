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
        # EXPLORATION PARAMETERS
        # ============================================================

        # Radius around each candidate used to estimate information gain.
        self.information_radius = self.declare_parameter(
            'information_radius',
            1.0
        ).value

        
        self.information_weight = self.declare_parameter(
            'information_weight',
            0.6
        ).value

        self.distance_weight = self.declare_parameter(
            'distance_weight',
            0.4
        ).value

    
        self.min_goal_distance = self.declare_parameter(
            'min_goal_distance',
            0.5
        ).value

        # Ignore new candidates close to a previously failed frontier.
        self.failed_region_radius = self.declare_parameter(
            'failed_region_radius',
            0.5
        ).value

        # Ignore new candidates close to a frontier that has
        # already been successfully reached.
        self.reached_region_radius = self.declare_parameter(
            'reached_region_radius',
            0.5
        ).value

        # Number of consecutive empty frontier updates required
        # before exploration is considered complete.
        self.required_empty_updates = self.declare_parameter(
            'required_empty_updates',
            3
        ).value

        # TF coordinate frames.
        self.map_frame = self.declare_parameter(
            'map_frame',
            'map'
        ).value

        self.robot_frame = self.declare_parameter(
            'robot_frame',
            'base_link'
        ).value

        # ============================================================
        # STUCK DETECTION PARAMETERS
        # ============================================================

        # The robot must translate this far...
        self.stuck_distance = self.declare_parameter(
            'stuck_distance',
            0.15
        ).value

        # ...or rotate this much for useful progress to be detected.
        self.stuck_angle = math.radians(
            self.declare_parameter(
                'stuck_angle_degrees',
                20.0
            ).value
        )

        # If neither occurs within this time, cancel the goal.
        self.stuck_timeout = self.declare_parameter(
            'stuck_timeout',
            5.0
        ).value

        # ============================================================
        # STORED EXPLORATION DATA
        # ============================================================

        # Most recent candidate frontier positions.
        self.frontier_candidates = []

        # Most recent occupancy-grid map.
        self.map_msg = None
        self.map_array = None

        # Locations that should no longer be selected.
        self.failed_regions = []
        self.reached_regions = []

        # Current frontier sent to Nav2.
        self.current_goal = None

        self.navigation_goal_handle = None

        self.empty_frontier_update_count = 0

    
        self.last_progress_pose = None
        self.last_progress_time = None

        # ============================================================
        # EXPLORATION STATE MACHINE
        # ============================================================

        # States:
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
        # NAV2 ACTION CLIENT
        # ============================================================

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
    # FRONTIER INPUT
    # ================================================================

    def frontier_cb(self, msg):

        # Always store the newest candidate set.
        self.frontier_candidates = list(msg.poses)

        if self.state == 'EXPLORATION_COMPLETE':
            return

        # After reaching a goal, wait for fresh frontier information.

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

        # Any non-empty update resets the completion counter.
        if self.frontier_candidates:
            self.empty_frontier_update_count = 0

        self.check_ready()

    # ================================================================
    # MAP INPUT
    # ================================================================

    def map_cb(self, msg):

        self.map_msg = msg

        # OccupancyGrid.data is one-dimensional.
        # Reshape it into row/column map form.
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
    # INITIAL READINESS
    # ================================================================

    def check_ready(self):

        # Exploration cannot begin until both:
        # 1. a map exists
        # 2. at least one frontier candidate exists

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
    # ROBOT POSE
    # ================================================================

    def get_robot_pose(self):

        try:

            # Obtain the robot pose in the global map frame.
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.robot_frame,
                Time()
            )

            x = transform.transform.translation.x
            y = transform.transform.translation.y

            q = transform.transform.rotation

            # Convert the quaternion into planar heading (yaw).
            yaw = math.atan2(
                2.0 * (
                    q.w * q.z
                    + q.x * q.y
                ),
                1.0 - 2.0 * (
                    q.y**2
                    + q.z**2
                )
            )

            return x, y, yaw

        except Exception:
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

        # ------------------------------------------------------------
        # Convert the map origin quaternion into yaw.
        # ------------------------------------------------------------

        q = origin.orientation

        origin_yaw = math.atan2(
            2.0 * (
                q.w * q.z
                + q.x * q.y
            ),
            1.0 - 2.0 * (
                q.y**2
                + q.z**2
            )
        )

        # Candidate relative to map origin.
        dx = (
            candidate.position.x
            - origin.position.x
        )

        dy = (
            candidate.position.y
            - origin.position.y
        )

        # Rotate world coordinates into the map frame.
        local_x = (
            dx * math.cos(origin_yaw)
            + dy * math.sin(origin_yaw)
        )

        local_y = (
            -dx * math.sin(origin_yaw)
            + dy * math.cos(origin_yaw)
        )

        # Convert metres into occupancy-grid indices.
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

        # Only cells inside a circular radius are considered.

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

        robot_x, robot_y, _ = robot_pose

        valid_candidates = []
        distances = []
        information_values = []

        # ------------------------------------------------------------
        # FILTER FRONTIERS
        # ------------------------------------------------------------

        for candidate in self.frontier_candidates:

            distance = math.hypot(
                candidate.position.x
                - robot_x,
                candidate.position.y
                - robot_y
            )


            if distance < self.min_goal_distance:
                continue

            # Skip regions that previously failed.

            near_failed_region = any(
                math.hypot(
                    candidate.position.x
                    - failed_x,
                    candidate.position.y
                    - failed_y
                )
                < self.failed_region_radius

                for failed_x, failed_y
                in self.failed_regions
            )

            if near_failed_region:
                continue

            # Also skip regions already successfully reached.

            near_reached_region = any(
                math.hypot(
                    candidate.position.x
                    - reached_x,
                    candidate.position.y
                    - reached_y
                )
                < self.reached_region_radius

                for reached_x, reached_y
                in self.reached_regions
            )

            if near_reached_region:
                continue

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

        if not valid_candidates:

            self.get_logger().info(
                'No valid frontier candidates '
                'after distance/reached/failed filtering'
            )

            return None

        # ------------------------------------------------------------
        # NORMALISE SCORE INPUTS
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

        scores = (
            self.information_weight
            * information_norm
            -
            self.distance_weight
            * distance_norm
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
            f'information={information_values[best_index]}, '
            f'score={scores[best_index]:.3f}'
        )

        return candidate

    # ================================================================
    # SEND NAVIGATION GOAL
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
    # NAV2 GOAL ACCEPTANCE
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

        # Nav2 explicitly rejected this frontier goal.
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

        # Begin monitoring progress from the new goal.
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
    # NAV2 RESULT
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

            # Do not issue another goal while the state of the current
            # Nav2 goal is unknown.
            self.state = 'CANCELING'

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

                duplicate = any(
                    math.hypot(
                        reached_position[0]
                        - old_x,
                        reached_position[1]
                        - old_y
                    )
                    < self.reached_region_radius

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

            # Wait for Frontier Detector to respond to the new map.
            self.state = 'WAITING_FOR_UPDATE'

            return

        # ------------------------------------------------------------
        # CONFIRMED CANCELLATION
        # ------------------------------------------------------------

        if (
            self.state == 'CANCELING'
            and
            status == GoalStatus.STATUS_CANCELED
        ):

            self.get_logger().warn(
                'Navigation cancellation complete. '
                'Marking frontier as failed.'
            )

            self.fail_current_frontier()

            return

        # ------------------------------------------------------------
        # OTHER NAVIGATION FAILURE
        # ------------------------------------------------------------

        if (
            status
            == GoalStatus.STATUS_ABORTED
        ):
            status_name = 'ABORTED'

        elif (
            status
            == GoalStatus.STATUS_CANCELED
        ):
            status_name = 'CANCELED'

        else:
            status_name = str(status)

        self.get_logger().warn(
            f'Navigation failed: '
            f'{status_name}'
        )

        # The Nav2 action has ended, so it is now safe to select
        # another goal.
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

            self.last_progress_time = (
                now
            )

            return

        old_x, old_y, old_yaw = (
            self.last_progress_pose
        )

        # Distance travelled since the previous meaningful movement.
        translation = math.hypot(
            x - old_x,
            y - old_y
        )

        # Wrapped heading difference in the range [-pi, pi].
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

        if (
            translation
            >= self.stuck_distance
            or
            heading_change
            >= self.stuck_angle
        ):

            self.last_progress_pose = (
                x,
                y,
                yaw
            )

            self.last_progress_time = (
                now
            )

            return

        stationary_time = (
            now
            - self.last_progress_time
        ).nanoseconds / 1e9

        if (
            stationary_time
            >= self.stuck_timeout
        ):

            self.get_logger().warn(
                f'Robot appears stuck: '
                f'translation={translation:.2f} m, '
                f'rotation='
                f'{math.degrees(heading_change):.1f} deg, '
                f'time={stationary_time:.1f} s'
            )

            self.cancel_stuck_goal()

    # ================================================================
    # SAFE STUCK-GOAL CANCELLATION
    # ================================================================

    def cancel_stuck_goal(self):

        # Only an actively navigating goal may be cancelled here.
        if self.state != 'NAVIGATING':
            return

        if self.navigation_goal_handle is None:

            self.get_logger().warn(
                'Stuck detected but no active '
                'navigation goal handle exists'
            )

            return

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
                f'Cancellation request failed: '
                f'{error}'
            )

            if (
                self.current_goal is not None
                and
                self.navigation_goal_handle
                is not None
            ):

                self.state = 'NAVIGATING'

                self.last_progress_pose = None
                self.last_progress_time = None

            return

      
        if not response.goals_canceling:

            self.get_logger().warn(
                'Nav2 did not accept the cancellation request. '
                'Keeping the current navigation active.'
            )

            self.state = 'NAVIGATING'

            self.last_progress_pose = None
            self.last_progress_time = None

            return

      
        # navigation_result_cb() must receive STATUS_CANCELED first.
       
        self.get_logger().info(
            'Cancellation accepted by Nav2. '
            'Waiting for the navigation result to confirm cancellation.'
        )

    # ================================================================
    # FAILED FRONTIER
    # ================================================================

    def fail_current_frontier(self):

        if self.current_goal is not None:

            failed_position = (
                self.current_goal.position.x,
                self.current_goal.position.y
            )

            duplicate = any(
                math.hypot(
                    failed_position[0]
                    - old_x,
                    failed_position[1]
                    - old_y
                )
                < self.failed_region_radius

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


        if self.state == 'NAVIGATING':

            self.check_stuck()
            return

        
        # No goal selection occurs while a cancellation is pending.
      
        if self.state == 'CANCELING':
            return

        # All other non-READY states also wait.
        if self.state != 'READY':
            return

        # ------------------------------------------------------------
        # SELECT NEXT FRONTIER
        # ------------------------------------------------------------

        candidate = (
            self.select_best_candidate()
        )

        if candidate is None:

            # Wait for another frontier/map update rather than
            # repeatedly running the exact same selection every second.
            self.state = 'WAITING_FOR_UPDATE'
            return

        self.current_goal = (
            candidate
        )

        self.last_progress_pose = None
        self.last_progress_time = None

        self.state = 'GOAL_SELECTED'

        # ------------------------------------------------------------
        # SEND TO NAV2
        # ------------------------------------------------------------

        if not self.send_navigation_goal():

            self.current_goal = None
            self.navigation_goal_handle = None

            self.state = 'READY'


# ================================================================
# MAIN
# ================================================================

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