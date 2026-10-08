import math

import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.time import Time
from rclpy.action import ActionClient

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration

from geometry_msgs.msg import PoseArray, Point
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan

from nav2_msgs.action import (
    NavigateToPose,
    Spin,
    DriveOnHeading,
)

from tf2_ros import Buffer, TransformListener


class ExplorationManager(Node):

    def __init__(self):
        super().__init__('exploration_manager')

        # ============================================================
        # EXPLORATION PARAMETERS
        # ============================================================

        # Radius around a frontier used to count nearby unknown cells.
        self.information_radius = self.declare_parameter(
            'information_radius',
            1.0
        ).value

        # Frontier scoring:
        # score = information benefit - travel cost
        self.information_weight = self.declare_parameter(
            'information_weight',
            0.6
        ).value

        self.distance_weight = self.declare_parameter(
            'distance_weight',
            0.4
        ).value

        # Prevent repeated selection of frontiers close to known failures.
        self.failed_region_radius = self.declare_parameter(
            'failed_region_radius',
            0.5
        ).value

        # Require several empty frontier updates before declaring
        # exploration complete.
        self.required_empty_updates = self.declare_parameter(
            'required_empty_updates',
            3
        ).value

        # Coordinate frames.
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

        # Robot must either move this distance...
        self.stuck_distance = self.declare_parameter(
            'stuck_distance',
            0.15
        ).value

        # ...or rotate this much for progress to count.
        self.stuck_angle = math.radians(
            self.declare_parameter(
                'stuck_angle_degrees',
                20.0
            ).value
        )

        # If neither occurs for this period, recovery starts.
        self.stuck_timeout = self.declare_parameter(
            'stuck_timeout',
            5.0
        ).value

        # Skip a nearby target when translation stalls, even while turning.
        self.near_goal_radius = self.declare_parameter(
            'near_goal_radius', 0.30
        ).value
        self.near_goal_stall_distance = self.declare_parameter(
            'near_goal_stall_distance', 0.10
        ).value
        self.near_goal_stall_timeout = self.declare_parameter(
            'near_goal_stall_timeout', 10.0
        ).value

        # Only one recovery attempt per frontier by default.
        self.max_recovery_attempts = self.declare_parameter(
            'max_recovery_attempts',
            1
        ).value

        # ============================================================
        # RECOVERY PARAMETERS
        # ============================================================

        # Short movement used to create space away from obstacles.
        self.recovery_distance = self.declare_parameter(
            'recovery_distance',
            0.30
        ).value

        self.recovery_speed = self.declare_parameter(
            'recovery_speed',
            0.08
        ).value

        # If no LiDAR direction is this clear, recovery is abandoned.
        self.recovery_min_clearance = self.declare_parameter(
            'recovery_min_clearance',
            0.50
        ).value

        # ============================================================
        # STORED DATA
        # ============================================================

        self.frontier_candidates = []

        self.map_msg = None
        self.map_array = None

        self.latest_scan = None

        # Coordinates of frontiers that were attempted and failed.
        self.failed_regions = []

        # Current selected frontier.
        self.current_goal = None

        # Stored Nav2 goal handle allows active navigation to be cancelled.
        self.navigation_goal_handle = None

        # Used when determining whether exploration has finished.
        self.empty_frontier_update_count = 0

        # Stores robot pose when useful progress was last detected.
        # Format: (x, y, yaw)
        self.last_progress_pose = None
        self.last_progress_time = None

        self.near_goal_anchor = None
        self.near_goal_since = None

        # Number of recovery attempts made for the current frontier.
        self.recovery_attempts = 0

        # ============================================================
        # STATE MACHINE
        # ============================================================

        self.state = 'WAITING_FOR_DATA'

        # Possible states:
        #
        # WAITING_FOR_DATA
        # READY
        # GOAL_SELECTED
        # NAVIGATING
        # SKIPPING_FRONTIER
        # RECOVERING
        # WAITING_FOR_UPDATE
        # EXPLORATION_COMPLETE

        # ============================================================
        # ROS SUBSCRIPTIONS
        # ============================================================

        # Frontier detector output.
        self.frontier_subscription = self.create_subscription(
            PoseArray,
            '/frontier_candidates',
            self.frontier_cb,
            10
        )

        # Occupancy grid map.
        self.map_subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_cb,
            10
        )

        # Latest LiDAR scan used only during recovery.
        self.scan_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_cb,
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
        # NAV2 ACTION CLIENTS
        # ============================================================

        self.navigation_client = ActionClient(
            self,
            NavigateToPose,
            'navigate_to_pose'
        )

        self.spin_client = ActionClient(
            self,
            Spin,
            'spin'
        )

        self.drive_client = ActionClient(
            self,
            DriveOnHeading,
            'drive_on_heading'
        )

        # ============================================================
        # MAIN TIMER
        # ============================================================

        # Runs exploration management once every second.
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

        self.frontier_candidates = list(
            msg.poses
        )

        if self.state == 'EXPLORATION_COMPLETE':
            return

        # After reaching a frontier, wait for the detector to update.
        if self.state == 'WAITING_FOR_UPDATE':

            if self.frontier_candidates:

                self.empty_frontier_update_count = 0
                self.state = 'READY'

                self.get_logger().info(
                    'New frontier data received'
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

        # Reset empty counter when frontiers exist.
        if self.frontier_candidates:
            self.empty_frontier_update_count = 0

        # Exploration starts once map + frontiers both exist.
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
    # MAP CALLBACK
    # ================================================================

    def map_cb(self, msg):

        self.map_msg = msg

        # OccupancyGrid arrives as a flat list.
        # Convert it into a 2D numpy array.
        self.map_array = np.array(
            msg.data
        ).reshape(
            (
                msg.info.height,
                msg.info.width
            )
        )

        if (
            self.state == 'WAITING_FOR_DATA'
            and self.frontier_candidates
        ):

            self.state = 'READY'

            self.get_logger().info(
                'Exploration manager ready'
            )

    # ================================================================
    # LIDAR CALLBACK
    # ================================================================

    def scan_cb(self, msg):

        # Only the newest scan is needed.
        self.latest_scan = msg

    # ================================================================
    # ROBOT POSE FROM TF
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

            # Convert quaternion orientation to planar yaw angle.
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

        if self.map_msg is None:
            return 0

        resolution = self.map_msg.info.resolution
        origin = self.map_msg.info.origin

        q = origin.orientation

        # Orientation of occupancy-grid origin.
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

        # Position relative to map origin.
        dx = (
            candidate.position.x
            - origin.position.x
        )

        dy = (
            candidate.position.y
            - origin.position.y
        )

        # Convert world coordinates into map-local coordinates.
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

        # Count unknown cells inside a circular region.
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

        for candidate in self.frontier_candidates:

            # Ignore frontiers near previously failed regions.
            failed = any(
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

            if failed:
                continue

            valid_candidates.append(
                candidate
            )

            distances.append(
                math.hypot(
                    candidate.position.x
                    - robot_x,
                    candidate.position.y
                    - robot_y
                )
            )

            information_values.append(
                self.information_gain(
                    candidate
                )
            )

        if not valid_candidates:
            return None

        # ------------------------------------------------------------
        # Normalisation helper
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

        # Higher information is good.
        # Greater distance is bad.
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
            f'distance={distances[best_index]:.2f}, '
            f'information={information_values[best_index]}, '
            f'score={scores[best_index]:.3f}'
        )

        return candidate

    # ================================================================
    # SEND CURRENT FRONTIER TO NAV2
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

        # Ensure the orientation quaternion is valid.
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
    # NAVIGATION GOAL RESPONSE
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

            self.start_recovery(
                cancel_navigation=False
            )

            return

        if not goal_handle.accepted:

            self.get_logger().warn(
                'Navigation goal rejected'
            )

            self.start_recovery(
                cancel_navigation=False
            )

            return

        # Store active goal so it can be cancelled if stuck.
        self.navigation_goal_handle = (
            goal_handle
        )

        self.state = 'NAVIGATING'

        self.near_goal_anchor = None
        self.near_goal_since = None

        # Start a new progress-monitoring window.
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
            lambda future: self.navigation_result_cb(future, goal_handle)
        )

    # ================================================================
    # NAVIGATION RESULT
    # ================================================================

    def navigation_result_cb(
        self,
        future,
        goal_handle=None
    ):

        # A result from an older goal must not affect the current target.
        if goal_handle is not None and goal_handle is not self.navigation_goal_handle:
            return

        try:

            status = (
                future.result().status
            )

        except Exception as error:

            self.get_logger().error(
                f'Navigation result error: '
                f'{error}'
            )

            # Do not start another action if termination cannot be confirmed.
            if self.state == 'SKIPPING_FRONTIER':
                return

            self.start_recovery(
                cancel_navigation=False
            )

            return

        if self.state == 'SKIPPING_FRONTIER':
            if status in (
                GoalStatus.STATUS_CANCELED,
                GoalStatus.STATUS_SUCCEEDED,
                GoalStatus.STATUS_ABORTED,
            ):
                self.fail_current_frontier()
            return

        # A cancellation during recovery is intentional.
        if (
            self.state == 'RECOVERING'
            and
            status == GoalStatus.STATUS_CANCELED
        ):
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

            self.current_goal = None
            self.navigation_goal_handle = None

            self.last_progress_pose = None
            self.last_progress_time = None

            self.recovery_attempts = 0

            self.empty_frontier_update_count = 0

            self.state = 'WAITING_FOR_UPDATE'

            return

        # ------------------------------------------------------------
        # FAILURE
        # ------------------------------------------------------------

        self.get_logger().warn(
            'Navigation failed - attempting recovery'
        )

        # Do not immediately blacklist the frontier.
        # Recovery gets one attempt first.
        self.start_recovery(
            cancel_navigation=False
        )

    # ================================================================
    # STUCK DETECTION
    # ================================================================

    def check_stuck(self):

        robot_pose = self.get_robot_pose()

        if robot_pose is None:
            self.near_goal_anchor = None
            self.near_goal_since = None
            return

        x, y, yaw = robot_pose

        now = self.get_clock().now()

        # Check before the general watchdog: rotation must not reset this timer.
        if self.check_near_goal_stall(x, y, now):
            return

        # First measurement after starting/restarting navigation.
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

        # Translational movement.
        translation = math.hypot(
            x - old_x,
            y - old_y
        )

        # Wrapped angle difference.
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

        # Either translation OR meaningful turning counts as progress.
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

        # If there has been no useful movement for long enough,
        # activate recovery.
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

            self.start_recovery(
                cancel_navigation=True
            )

    def check_near_goal_stall(self, x, y, now):
        """Skip a target after sustained lack of translation nearby."""
        if self.current_goal is None or math.hypot(
            x - self.current_goal.position.x,
            y - self.current_goal.position.y,
        ) > self.near_goal_radius:
            self.near_goal_anchor = None
            self.near_goal_since = None
            return False

        if self.near_goal_anchor is None or math.hypot(
            x - self.near_goal_anchor[0], y - self.near_goal_anchor[1]
        ) >= self.near_goal_stall_distance:
            self.near_goal_anchor = (x, y)
            self.near_goal_since = now
            return False

        elapsed = (now - self.near_goal_since).nanoseconds / 1e9
        if elapsed < self.near_goal_stall_timeout:
            return False

        handle = self.navigation_goal_handle
        if handle is None:
            return False

        self.get_logger().warn(
            f'Stalled near frontier for {elapsed:.1f} s; '
            'cancelling navigation and skipping this region'
        )
        self.state = 'SKIPPING_FRONTIER'

        def cancellation_failed(reason):
            if (
                self.navigation_goal_handle is handle
                and self.state == 'SKIPPING_FRONTIER'
            ):
                self.get_logger().warn(f'Could not cancel frontier: {reason}')
                self.state = 'NAVIGATING'
                self.near_goal_anchor = None
                self.near_goal_since = None
                self.last_progress_pose = None
                self.last_progress_time = None

        def cancelled(future):
            try:
                response = future.result()
                if not response.goals_canceling:
                    cancellation_failed('cancellation rejected')
            except Exception as error:
                cancellation_failed(str(error))
            # Acceptance is not completion. The navigation result callback
            # records the failed region only after the action has terminated.

        try:
            handle.cancel_goal_async().add_done_callback(cancelled)
        except Exception as error:
            cancellation_failed(str(error))
        return True

    # ================================================================
    # RECOVERY
    # ================================================================

    def start_recovery(
        self,
        cancel_navigation
    ):

        # Prevent multiple simultaneous recovery attempts.
        if self.state == 'RECOVERING':
            return

        if self.current_goal is None:
            return

        # Frontier already received its allowed recovery attempt.
        if (
            self.recovery_attempts
            >= self.max_recovery_attempts
        ):

            self.get_logger().warn(
                'Maximum recovery attempts reached. '
                'Frontier considered unreachable.'
            )

            self.fail_current_frontier()

            return

        # Recovery requires current LiDAR data.
        if self.latest_scan is None:

            self.get_logger().warn(
                'No LiDAR scan available for recovery.'
            )

            self.fail_current_frontier()

            return

        # ============================================================
        # FIND A CLEAR DIRECTION FROM LIDAR
        # ============================================================

        ranges = np.array(
            self.latest_scan.ranges,
            dtype=float
        )

        if len(ranges) == 0:

            self.fail_current_frontier()
            return

        # Positive infinity generally means no obstacle was detected
        # within the sensor range.
        ranges[
            np.isposinf(ranges)
        ] = self.latest_scan.range_max

        # Invalid values should not appear as clear space.
        ranges[
            np.isnan(ranges)
        ] = 0.0

        ranges[
            np.isneginf(ranges)
        ] = 0.0

        ranges[
            ranges
            < self.latest_scan.range_min
        ] = 0.0

        ranges[
            ranges
            > self.latest_scan.range_max
        ] = self.latest_scan.range_max

        # ------------------------------------------------------------
        # Average neighbouring LiDAR rays.
        #
        # This prevents one noisy high range value from being selected
        # as the recovery direction.
        # ------------------------------------------------------------

        angle_increment = abs(
            self.latest_scan.angle_increment
        )

        if angle_increment == 0.0:

            self.get_logger().warn(
                'Invalid LiDAR angle increment.'
            )

            self.fail_current_frontier()

            return

        # Approx. +/- 10 degree averaging region.
        window = max(
            1,
            int(
                math.radians(10.0)
                / angle_increment
            )
        )

        averaged = np.zeros_like(
            ranges
        )

        for offset in range(
            -window,
            window + 1
        ):

            averaged += np.roll(
                ranges,
                offset
            )

        averaged /= (
            2 * window + 1
        )

        best_index = int(
            np.argmax(averaged)
        )

        best_clearance = float(
            averaged[best_index]
        )

        if (
            best_clearance
            < self.recovery_min_clearance
        ):

            self.get_logger().warn(
                f'No sufficiently clear recovery '
                f'direction found. '
                f'Best clearance='
                f'{best_clearance:.2f} m'
            )

            self.fail_current_frontier()

            return

        # LaserScan angles are measured relative to the robot.
        recovery_heading = (
            self.latest_scan.angle_min
            +
            best_index
            * self.latest_scan.angle_increment
        )

        self.recovery_attempts += 1

        self.state = 'RECOVERING'

        self.get_logger().warn(
            f'Recovery attempt '
            f'{self.recovery_attempts}/'
            f'{self.max_recovery_attempts}: '
            f'heading='
            f'{math.degrees(recovery_heading):.1f} deg, '
            f'clearance='
            f'{best_clearance:.2f} m'
        )

        # ============================================================
        # RECOVERY CALLBACKS
        #
        # Nested callbacks keep the recovery logic together rather
        # than spreading it across many separate functions.
        # ============================================================

        def recovery_failed(
            message
        ):

            self.get_logger().warn(
                message
            )

            self.fail_current_frontier()

        # ------------------------------------------------------------
        # DRIVE RESULT
        # ------------------------------------------------------------

        def drive_result(
            result_future
        ):

            try:

                status = (
                    result_future
                    .result()
                    .status
                )

            except Exception:

                recovery_failed(
                    'Could not get recovery '
                    'drive result.'
                )

                return

            if (
                status
                != GoalStatus.STATUS_SUCCEEDED
            ):

                recovery_failed(
                    'Recovery drive failed.'
                )

                return

            self.get_logger().info(
                'Recovery successful. '
                'Retrying same frontier.'
            )

            # Keep current_goal unchanged so the same frontier is retried.
            self.navigation_goal_handle = None

            self.last_progress_pose = None
            self.last_progress_time = None

            self.state = 'GOAL_SELECTED'

            if not self.send_navigation_goal():

                self.fail_current_frontier()

        # ------------------------------------------------------------
        # DRIVE GOAL RESPONSE
        # ------------------------------------------------------------

        def drive_response(
            drive_future
        ):

            try:

                handle = (
                    drive_future.result()
                )

            except Exception:

                recovery_failed(
                    'Could not send '
                    'recovery drive.'
                )

                return

            if not handle.accepted:

                recovery_failed(
                    'Recovery drive rejected.'
                )

                return

            handle.get_result_async().add_done_callback(
                drive_result
            )

        # ------------------------------------------------------------
        # START DRIVE
        # ------------------------------------------------------------

        def start_drive():

            if not self.drive_client.wait_for_server(
                timeout_sec=1.0
            ):

                recovery_failed(
                    'DriveOnHeading '
                    'action server unavailable.'
                )

                return

            goal = DriveOnHeading.Goal()

            # Robot has already rotated toward free space,
            # so positive x means drive forward in that direction.
            goal.target = Point(
                x=float(
                    self.recovery_distance
                ),
                y=0.0,
                z=0.0
            )

            goal.speed = float(
                self.recovery_speed
            )

            expected_time = (
                self.recovery_distance
                /
                max(
                    self.recovery_speed,
                    0.01
                )
            )

            goal.time_allowance = Duration(
                sec=int(
                    math.ceil(
                        expected_time + 3.0
                    )
                )
            )

            # Keep collision checking enabled.
            if hasattr(
                goal,
                'disable_collision_checks'
            ):

                goal.disable_collision_checks = False

            future = (
                self.drive_client
                .send_goal_async(goal)
            )

            future.add_done_callback(
                drive_response
            )

        # ------------------------------------------------------------
        # SPIN RESULT
        # ------------------------------------------------------------

        def spin_result(
            result_future
        ):

            try:

                status = (
                    result_future
                    .result()
                    .status
                )

            except Exception:

                recovery_failed(
                    'Could not get recovery '
                    'spin result.'
                )

                return

            if (
                status
                != GoalStatus.STATUS_SUCCEEDED
            ):

                recovery_failed(
                    'Recovery spin failed.'
                )

                return

            # Rotation succeeded.
            # Now drive a short distance into the open area.
            start_drive()

        # ------------------------------------------------------------
        # SPIN GOAL RESPONSE
        # ------------------------------------------------------------

        def spin_response(
            spin_future
        ):

            try:

                handle = (
                    spin_future.result()
                )

            except Exception:

                recovery_failed(
                    'Could not send '
                    'recovery spin.'
                )

                return

            if not handle.accepted:

                recovery_failed(
                    'Recovery spin rejected.'
                )

                return

            handle.get_result_async().add_done_callback(
                spin_result
            )

        # ------------------------------------------------------------
        # START SPIN
        # ------------------------------------------------------------

        def start_spin():

            if not self.spin_client.wait_for_server(
                timeout_sec=1.0
            ):

                recovery_failed(
                    'Spin action server unavailable.'
                )

                return

            goal = Spin.Goal()

            # Relative rotation toward clearest LiDAR direction.
            goal.target_yaw = float(
                recovery_heading
            )

            goal.time_allowance = Duration(
                sec=6
            )

            # Never disable collision checking for recovery.
            if hasattr(
                goal,
                'disable_collision_checks'
            ):

                goal.disable_collision_checks = False

            future = (
                self.spin_client
                .send_goal_async(goal)
            )

            future.add_done_callback(
                spin_response
            )

        # ============================================================
        # BEGIN RECOVERY
        # ============================================================

        # If Nav2 is still actively navigating, cancel that goal first
        # so navigation and recovery do not fight each other.
        if (
            cancel_navigation
            and self.navigation_goal_handle
            is not None
        ):

            cancel_future = (
                self.navigation_goal_handle
                .cancel_goal_async()
            )

            cancel_future.add_done_callback(
                lambda _: start_spin()
            )

        else:

            start_spin()

    # ================================================================
    # FAILED FRONTIER
    # ================================================================

    def fail_current_frontier(self):

        if self.current_goal is not None:

            x = self.current_goal.position.x
            y = self.current_goal.position.y

            # Avoid recording the same failed area repeatedly.
            duplicate = any(
                math.hypot(
                    x - failed_x,
                    y - failed_y
                )
                < self.failed_region_radius

                for failed_x, failed_y
                in self.failed_regions
            )

            if not duplicate:

                self.failed_regions.append(
                    (
                        x,
                        y
                    )
                )

                self.get_logger().warn(
                    f'Failed frontier recorded: '
                    f'x={x:.2f}, '
                    f'y={y:.2f}'
                )

        # Reset goal-specific state.
        self.current_goal = None
        self.navigation_goal_handle = None

        self.last_progress_pose = None
        self.last_progress_time = None

        self.recovery_attempts = 0

        # Return to exploration and choose another frontier.
        self.state = 'READY'

    # ================================================================
    # MAIN EXPLORATION LOOP
    # ================================================================

    def exploration_cb(self):

        # While navigating, do not choose another frontier.
        # Only check whether useful progress is occurring.
        if self.state == 'NAVIGATING':

            self.check_stuck()
            return

        # Recovery owns control while it is active.
        if self.state == 'RECOVERING':
            return

        # Only select goals from READY state.
        if self.state != 'READY':
            return

        candidate = (
            self.select_best_candidate()
        )

        if candidate is None:
            return

        self.current_goal = (
            candidate
        )

        # New frontier receives a fresh recovery allowance.
        self.recovery_attempts = 0

        self.last_progress_pose = None
        self.last_progress_time = None

        self.state = 'GOAL_SELECTED'

        if not self.send_navigation_goal():

            self.current_goal = None
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
