import math

import numpy as np

import rclpy

#----------------------------------------------------------------
import hashlib
import time
#----------------------------------------------------------------


from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import ComputePathToPose, NavigateToPose
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
            2.0
        ).value

        # Frontier scoring weights.
        #
        # Information = how useful the frontier is.
        # Distance = how far the robot must travel.
        # Heading = how much the robot must turn.
        self.information_weight = self.declare_parameter(
            'information_weight',
            0.20
        ).value

        self.distance_weight = self.declare_parameter(
            'distance_weight',
            0.30
        ).value

        self.distance_decay = self.declare_parameter(
            'distance_decay',
            2.0
        ).value

        self.path_search_initial_radius = self.declare_parameter(
            'path_search_initial_radius',
            3.0
        ).value

        self.path_search_expansion_factor = self.declare_parameter(
            'path_search_expansion_factor',
            2.0
        ).value

        if self.path_search_initial_radius <= 0.0:
            raise ValueError(
                'path_search_initial_radius must be greater than zero'
            )

        if self.path_search_expansion_factor <= 1.0:
            raise ValueError(
                'path_search_expansion_factor must be greater than one'
            )

        self.heading_weight = self.declare_parameter(
            'heading_weight',
            0.50
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
                45.0
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

        self.path_candidates = []
        self.path_lengths = []
        self.path_planned = []
        self.path_candidate_distances = []
        self.path_request_indices = []
        self.path_candidate_index = 0
        self.path_request_generation = 0
        self.path_search_radius = self.path_search_initial_radius

        # Used to decide when exploration has finished.
        self.empty_frontier_update_count = 0

        # ============================================================
        # STATE MACHINE
        # ============================================================

        # WAITING_FOR_DATA
        # READY
        # GOAL_SELECTED
        # NAVIGATING
        # PLANNING_PATHS
        # WAITING_FOR_MAP
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

        #------------------------------------------------------------
        self.last_map_time = None
        self.last_map_change_time = None
        self.last_map_signature = None
        self.post_goal_scan_stamp = None
        self.required_frontier_stamp = None

        self.scan_count = 0

        self.diagnostic_timer = self.create_timer(
            5.0,
            self.diagnose_topics
        )
        #------------------------------------------------------------

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

        self.path_client = ActionClient(
            self,
            ComputePathToPose,
            'compute_path_to_pose'
        )

        # Run exploration decision logic once per second.
        self.exploration_timer = self.create_timer(
            1.0,
            self.exploration_cb
        )

        self.get_logger().info(
            'Exploration manager started'
        )


    def diagnose_topics(self):
        now = time.monotonic()

        if self.last_map_time is None:
            self.get_logger().warning('No /map message received yet')
        elif (now - self.last_map_time) > 5.0:
            self.get_logger().warning(
                f'/map is stale: {now - self.last_map_time:.1f} s since last message'
            )
        elif self.last_map_change_time is not None:
            self.get_logger().info(
                'Last /map content change was '
                f'{now - self.last_map_change_time:.1f} s ago'
            )
    # ================================================================
    # FRONTIER CALLBACK
    # ================================================================

    def frontier_cb(
        self,
        msg
    ):

        if (
            self.state == 'WAITING_FOR_UPDATE'
            and self.required_frontier_stamp is not None
        ):
            candidate_stamp = (
                msg.header.stamp.sec * 1_000_000_000
                + msg.header.stamp.nanosec
            )

            if candidate_stamp < self.required_frontier_stamp:
                self.get_logger().debug(
                    'Ignoring frontier candidates older than the '
                    'post-scan map'
                )
                return

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

    def map_cb(self, msg):
        #------------------------------------------------------------
        self.last_map_time = time.monotonic()

        map_signature = hashlib.blake2b(
            np.asarray(msg.data, dtype=np.int8).tobytes(),
            digest_size=8
        ).digest()

        if map_signature != self.last_map_signature:
            self.last_map_signature = map_signature
            self.last_map_change_time = self.last_map_time
            self.get_logger().info('/map contents changed')
        #------------------------------------------------------------

        self.map_msg = msg

        self.map_array = np.array(msg.data).reshape(
            (
                msg.info.height,
                msg.info.width
            )
        )

        if self.state == 'WAITING_FOR_MAP':
            map_stamp = (
                msg.header.stamp.sec * 1_000_000_000
                + msg.header.stamp.nanosec
            )

            if (
                self.post_goal_scan_stamp is not None
                and map_stamp >= self.post_goal_scan_stamp
            ):
                self.required_frontier_stamp = map_stamp
                self.empty_frontier_update_count = 0
                self.state = 'WAITING_FOR_UPDATE'
                self.get_logger().info(
                    'Received /map stamped at or after the scan; '
                    'waiting for matching frontier candidates'
                )
            else:
                self.get_logger().debug(
                    'Ignoring /map older than the post-goal scan'
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

    def select_best_candidate(
        self,
        candidates=None,
        path_lengths=None
    ):

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

        candidates = (
            self.frontier_candidates
            if candidates is None
            else candidates
        )

        unreachable_count = 0

        for candidate_index, candidate in enumerate(candidates):

            straight_line_distance = math.hypot(
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
                straight_line_distance
                < self.min_goal_distance
            ):

                too_close_count += 1

                continue

            if path_lengths is not None:
                path_distance = path_lengths[candidate_index]

                if path_distance is None:
                    unreachable_count += 1
                    continue
            else:
                path_distance = straight_line_distance

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
                path_distance
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
                f'reached={reached_count}, '
                f'unreachable={unreachable_count}'
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

        distance_reward = normalise(
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
        # Distance is an exponentially decaying reward; heading
        # change remains a penalty.
        scores = (
            self.information_weight
            * information_norm

            -

            self.distance_weight
            * distance_reward

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

    def start_path_planning(self):
        if not self.path_client.wait_for_server(timeout_sec=0.0):
            self.get_logger().warn(
                'ComputePathToPose action server unavailable; '
                'will retry'
            )
            return

        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            return

        robot_x, robot_y, _ = robot_pose
        self.path_candidates = list(self.frontier_candidates)
        self.path_lengths = [None] * len(self.path_candidates)
        self.path_planned = [False] * len(self.path_candidates)
        self.path_candidate_distances = [
            math.hypot(
                candidate.position.x - robot_x,
                candidate.position.y - robot_y
            )
            for candidate in self.path_candidates
        ]
        self.path_candidate_index = 0
        self.path_search_radius = self.path_search_initial_radius
        self.path_request_generation += 1
        self.state = 'PLANNING_PATHS'

        self.get_logger().info(
            f'Planning paths for candidates within '
            f'{self.path_search_radius:.2f} m'
        )
        self.plan_candidates_within_radius(
            self.path_request_generation
        )

    def plan_candidates_within_radius(self, generation):
        if generation != self.path_request_generation:
            return

        self.path_request_indices = [
            index
            for index, distance in enumerate(
                self.path_candidate_distances
            )
            if (
                distance <= self.path_search_radius
                and not self.path_planned[index]
            )
        ]
        self.path_candidate_index = 0

        if not self.path_request_indices:
            self.finish_path_planning(generation)
            return

        self.get_logger().info(
            f'Planning {len(self.path_request_indices)} new path(s) '
            f'within {self.path_search_radius:.2f} m'
        )
        self.request_next_path(generation)

    def request_next_path(self, generation):
        if generation != self.path_request_generation:
            return

        if self.path_candidate_index >= len(self.path_request_indices):
            self.finish_path_planning(generation)
            return

        candidate_index = self.path_request_indices[
            self.path_candidate_index
        ]
        candidate = self.path_candidates[candidate_index]

        goal = ComputePathToPose.Goal()
        goal.goal = PoseStamped()
        goal.goal.header.frame_id = self.map_frame
        goal.goal.header.stamp = self.get_clock().now().to_msg()
        goal.goal.pose = candidate
        goal.planner_id = ''
        goal.use_start = False

        future = self.path_client.send_goal_async(goal)
        future.add_done_callback(
            lambda result_future: self.path_goal_response(
                result_future,
                candidate_index,
                generation
            )
        )

    def path_goal_response(
        self,
        future,
        candidate_index,
        generation
    ):
        if generation != self.path_request_generation:
            return

        try:
            goal_handle = future.result()
        except Exception as error:
            self.get_logger().error(
                f'ComputePathToPose request failed: {error}'
            )
            self.path_candidate_complete(
                candidate_index,
                None,
                generation
            )
            return

        if not goal_handle.accepted:
            self.get_logger().warn(
                f'Path request rejected for candidate {candidate_index}'
            )
            self.path_candidate_complete(
                candidate_index,
                None,
                generation
            )
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda result: self.path_result_cb(
                result,
                candidate_index,
                generation
            )
        )

    def path_result_cb(
        self,
        future,
        candidate_index,
        generation
    ):
        if generation != self.path_request_generation:
            return

        try:
            wrapped_result = future.result()
        except Exception as error:
            self.get_logger().error(
                f'Could not obtain path for candidate '
                f'{candidate_index}: {error}'
            )
            self.path_candidate_complete(
                candidate_index,
                None,
                generation
            )
            return

        if (
            wrapped_result.status != GoalStatus.STATUS_SUCCEEDED
            or not wrapped_result.result.path.poses
        ):
            self.get_logger().warn(
                f'No valid path found for candidate {candidate_index}'
            )
            self.path_candidate_complete(
                candidate_index,
                None,
                generation
            )
            return

        poses = wrapped_result.result.path.poses
        path_length = sum(
            math.hypot(
                next_pose.pose.position.x - pose.pose.position.x,
                next_pose.pose.position.y - pose.pose.position.y
            )
            for pose, next_pose in zip(poses, poses[1:])
        )
        self.path_candidate_complete(
            candidate_index,
            path_length,
            generation
        )

    def path_candidate_complete(
        self,
        candidate_index,
        path_length,
        generation
    ):
        if generation != self.path_request_generation:
            return

        self.path_lengths[candidate_index] = path_length
        self.path_planned[candidate_index] = True
        # if path_length is None:
        #     self.get_logger().info(
        #         f'Candidate {candidate_index}: path unavailable'
        #     )
        # else:
        #     self.get_logger().info(
        #         f'Candidate {candidate_index}: planned path '
        #         f'{path_length:.2f} m'
        #     )

        self.path_candidate_index += 1
        self.request_next_path(generation)

    def finish_path_planning(self, generation):
        if generation != self.path_request_generation:
            return

        planned_indices = [
            index
            for index, planned in enumerate(self.path_planned)
            if planned
        ]
        candidate = self.select_best_candidate(
            candidates=[
                self.path_candidates[index]
                for index in planned_indices
            ],
            path_lengths=[
                self.path_lengths[index]
                for index in planned_indices
            ]
        )

        if candidate is None:
            farthest_candidate = max(
                self.path_candidate_distances,
                default=0.0
            )
            if self.path_search_radius < farthest_candidate:
                old_radius = self.path_search_radius
                self.path_search_radius = min(
                    self.path_search_radius
                    * self.path_search_expansion_factor,
                    farthest_candidate
                )
                self.get_logger().info(
                    f'No usable candidate within {old_radius:.2f} m; '
                    f'expanding search radius to '
                    f'{self.path_search_radius:.2f} m'
                )
                self.plan_candidates_within_radius(generation)
                return

            self.state = 'WAITING_FOR_UPDATE'
            self.get_logger().warn(
                'No reachable frontier candidates at any distance; '
                'waiting for updated frontiers'
            )
            return

        self.current_goal = candidate
        self.state = 'GOAL_SELECTED'

        if not self.send_navigation_goal():
            self.current_goal = None
            self.navigation_goal_handle = None
            self.state = 'READY'

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

            self.last_progress_pose = None
            self.last_progress_time = None

            self.empty_frontier_update_count = 0

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

        self.start_path_planning()


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
