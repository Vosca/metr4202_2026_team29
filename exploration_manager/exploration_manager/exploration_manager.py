import math

import numpy as np
import rclpy

from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import PoseArray
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from tf2_ros import Buffer, TransformListener


class ExplorationManager(Node):

    def __init__(self):
        super().__init__('exploration_manager')

        # Store the most recent frontier candidates
        self.frontier_candidates = []

        # Store the latest occupancy grid map
        self.map_array = None
        self.map_msg = None

        # Radius around each frontier candidate used
        # to estimate how much unknown space is nearby
        self.information_radius = self.declare_parameter(
            'information_radius',
            1.0
        ).value

        # Weight given to information gain when
        # scoring frontier candidates
        self.information_weight = self.declare_parameter(
            'information_weight',
            0.6
        ).value

        # Weight given to travel distance when
        # scoring frontier candidates
        self.distance_weight = self.declare_parameter(
            'distance_weight',
            0.4
        ).value

        # Radius around a failed goal that should
        # temporarily be avoided
        self.failed_region_radius = self.declare_parameter(
            'failed_region_radius',
            0.5
        ).value

        # Number of consecutive empty frontier updates
        # required before exploration is considered complete
        self.required_empty_updates = int(
            self.declare_parameter(
                'required_empty_updates',
                3
            ).value
        )

        # Make sure at least one empty update is required
        self.required_empty_updates = max(
            1,
            self.required_empty_updates
        )

        # Count consecutive empty frontier updates
        self.empty_frontier_update_count = 0

        # Store failed navigation goal locations
        self.failed_regions = []

        # Store the current exploration state
        self.state = 'WAITING_FOR_DATA'

        # Store the currently selected frontier goal
        self.current_goal = None

        # Store the active Nav2 goal handle
        self.current_goal_handle = None

        # Store the future used to obtain the
        # final Nav2 navigation result
        self.navigation_result_future = None

        # Subscribe to the frontier candidates produced by
        # the frontier detector
        self.candidate_subscription = self.create_subscription(
            PoseArray,
            '/frontier_candidates',
            self.frontier_candidates_callback,
            10
        )

        # Subscribe to the SLAM occupancy grid
        self.map_subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        # TF buffer stores coordinate transforms
        self.tf_buffer = Buffer()

        # TF listener receives transforms and stores them
        # in the TF buffer
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        # Action client used to send exploration goals
        # to the Nav2 navigation system
        self.navigation_client = ActionClient(
            self,
            NavigateToPose,
            'navigate_to_pose'
        )

        # Run the exploration check once per second
        self.exploration_timer = self.create_timer(
            1.0,
            self.exploration_callback
        )

        self.get_logger().info(
            'Exploration manager node started'
        )

    def frontier_candidates_callback(self, msg):

        # Replace the previous frontier list with the
        # newest set of frontier candidates
        self.frontier_candidates = list(msg.poses)

        self.get_logger().info(
            f'Received {len(self.frontier_candidates)} '
            f'frontier candidates'
        )

        # Once exploration has been declared complete,
        # keep the state terminal
        if self.state == 'EXPLORATION_COMPLETE':
            return

        # After successful navigation, wait for fresh
        # frontier information before selecting another goal
        if self.state == 'WAITING_FOR_UPDATE':

            # If new frontiers exist, exploration continues
            if self.frontier_candidates:

                # A non-empty update breaks the sequence
                # of consecutive empty updates
                self.empty_frontier_update_count = 0

                self.state = 'READY'

                self.get_logger().info(
                    'New frontier data received after '
                    'navigation. Exploration is ready.'
                )

                return

            # No frontiers were found in this update
            self.empty_frontier_update_count += 1

            self.get_logger().info(
                f'Empty frontier update '
                f'{self.empty_frontier_update_count}/'
                f'{self.required_empty_updates}'
            )

            # Require several consecutive empty updates
            # before declaring exploration complete
            if (
                self.empty_frontier_update_count
                >= self.required_empty_updates
            ):

                self.state = 'EXPLORATION_COMPLETE'

                self.get_logger().info(
                    'Exploration complete: no frontier '
                    'candidates remain.'
                )

            return

        # If frontiers exist during normal operation,
        # reset any previous empty-update count
        if self.frontier_candidates:
            self.empty_frontier_update_count = 0

        # Check whether all required exploration data
        # has now been received
        self.update_ready_state()

    def map_callback(self, msg):

        # Store the complete map message because its
        # resolution and origin are needed later
        self.map_msg = msg

        # Convert the 1D OccupancyGrid data into
        # a 2D NumPy array
        self.map_array = np.array(
            msg.data
        ).reshape(
            (
                msg.info.height,
                msg.info.width
            )
        )

        self.get_logger().info(
            f'Received map: '
            f'{msg.info.width} x {msg.info.height} cells'
        )

        # Check whether all required exploration data
        # has now been received
        self.update_ready_state()

    def update_ready_state(self):

        # Initial exploration can begin only when both:
        # 1. frontier candidates are available
        # 2. an occupancy map has been received
        if (
            self.state == 'WAITING_FOR_DATA'
            and self.frontier_candidates
            and self.map_array is not None
        ):
            self.state = 'READY'

            self.get_logger().info(
                'Map and frontier data received. '
                'Exploration manager is ready.'
            )

    def get_robot_position(self):

        try:
            # Find the robot position relative to the map frame
            transform = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                Time()
            )

            robot_x = transform.transform.translation.x
            robot_y = transform.transform.translation.y

            return robot_x, robot_y

        except Exception:
            # TF may not yet be available
            return None

    def calculate_distance(
        self,
        candidate,
        robot_x,
        robot_y
    ):

        # Calculate straight-line distance from the robot
        # to the frontier candidate
        distance = math.hypot(
            candidate.position.x - robot_x,
            candidate.position.y - robot_y
        )

        return distance

    def world_to_cell(self, world_x, world_y):

        # Cannot perform the conversion without map metadata
        if self.map_msg is None:
            return None

        origin = self.map_msg.info.origin
        resolution = self.map_msg.info.resolution

        # Get the yaw rotation of the map origin
        orientation = origin.orientation

        yaw = math.atan2(
            2.0 * (
                orientation.w * orientation.z
                + orientation.x * orientation.y
            ),
            1.0 - 2.0 * (
                orientation.y ** 2
                + orientation.z ** 2
            )
        )

        # Position relative to the map origin
        delta_x = world_x - origin.position.x
        delta_y = world_y - origin.position.y

        # Rotate the world position into the map frame
        local_x = (
            delta_x * math.cos(yaw)
            + delta_y * math.sin(yaw)
        )

        local_y = (
            -delta_x * math.sin(yaw)
            + delta_y * math.cos(yaw)
        )

        # Convert metres into grid indices
        col = int(
            math.floor(
                local_x / resolution
            )
        )

        row = int(
            math.floor(
                local_y / resolution
            )
        )

        # Reject coordinates outside the map
        if (
            row < 0
            or row >= self.map_msg.info.height
            or col < 0
            or col >= self.map_msg.info.width
        ):
            return None

        return row, col

    def calculate_information_gain(self, candidate):

        # Convert candidate world coordinates back
        # into occupancy-grid coordinates
        cell = self.world_to_cell(
            candidate.position.x,
            candidate.position.y
        )

        if cell is None:
            return 0

        row, col = cell

        resolution = self.map_msg.info.resolution

        # Convert the information radius from metres
        # into grid cells
        radius_cells = int(
            math.ceil(
                self.information_radius / resolution
            )
        )

        rows, cols = self.map_array.shape

        unknown_count = 0

        # Search cells around the frontier candidate
        for check_row in range(
            max(0, row - radius_cells),
            min(rows, row + radius_cells + 1)
        ):
            for check_col in range(
                max(0, col - radius_cells),
                min(cols, col + radius_cells + 1)
            ):

                # Calculate physical distance from the
                # candidate to this map cell
                row_distance = (
                    check_row - row
                ) * resolution

                col_distance = (
                    check_col - col
                ) * resolution

                distance = math.hypot(
                    row_distance,
                    col_distance
                )

                # Only count cells inside the circular
                # information radius
                if distance > self.information_radius:
                    continue

                # Unknown occupancy-grid cells have value -1
                if self.map_array[
                    check_row,
                    check_col
                ] == -1:
                    unknown_count += 1

        return unknown_count

    def is_failed_region(self, candidate):

        # Compare this candidate with every previously
        # failed navigation location
        for failed_x, failed_y in self.failed_regions:

            distance = math.hypot(
                candidate.position.x - failed_x,
                candidate.position.y - failed_y
            )

            # Treat nearby candidates as belonging to
            # the same failed region
            if distance < self.failed_region_radius:
                return True

        return False

    def record_failed_goal(self, candidate):

        # Do not store multiple almost-identical
        # failed goal locations
        if self.is_failed_region(candidate):
            return

        failed_x = candidate.position.x
        failed_y = candidate.position.y

        self.failed_regions.append(
            (
                failed_x,
                failed_y
            )
        )

        self.get_logger().warn(
            f'Recorded failed frontier region: '
            f'x={failed_x:.2f} m, '
            f'y={failed_y:.2f} m'
        )

    def normalise_values(self, values, equal_value):

        values = np.array(
            values,
            dtype=float
        )

        minimum = np.min(values)
        maximum = np.max(values)

        # If every candidate has the same value,
        # this quantity should not favour any candidate
        if maximum == minimum:
            return np.full(
                len(values),
                equal_value,
                dtype=float
            )

        return (
            values - minimum
        ) / (
            maximum - minimum
        )

    def select_best_candidate(self):

        # Cannot select anything if there are no frontiers
        if not self.frontier_candidates:
            return None

        # Remove candidates that are close to previously
        # failed navigation goals
        valid_candidates = []

        for candidate in self.frontier_candidates:

            if self.is_failed_region(candidate):

                self.get_logger().info(
                    f'Skipping failed region: '
                    f'x={candidate.position.x:.2f} m, '
                    f'y={candidate.position.y:.2f} m'
                )

                continue

            valid_candidates.append(candidate)

        # Stop if all current candidates are within
        # previously failed regions
        if not valid_candidates:

            self.get_logger().warn(
                'No valid frontier candidates remain '
                'after failed-region filtering'
            )

            return None

        robot_position = self.get_robot_position()

        # Cannot evaluate candidates until robot pose is known
        if robot_position is None:
            return None

        robot_x, robot_y = robot_position

        distances = []
        information_gains = []

        # Calculate raw values for every valid
        # frontier candidate
        for candidate in valid_candidates:

            distance = self.calculate_distance(
                candidate,
                robot_x,
                robot_y
            )

            information_gain = (
                self.calculate_information_gain(
                    candidate
                )
            )

            distances.append(distance)

            information_gains.append(
                information_gain
            )

        # Normalise information gain to values
        # between 0 and 1
        normalised_information = self.normalise_values(
            information_gains,
            1.0
        )

        # Normalise distance to values
        # between 0 and 1
        normalised_distance = self.normalise_values(
            distances,
            0.0
        )

        best_candidate = None
        best_score = -float('inf')
        best_distance = None
        best_information = None

        # Calculate the utility score for every
        # valid candidate
        for index, candidate in enumerate(
            valid_candidates
        ):

            score = (
                self.information_weight
                * normalised_information[index]
                -
                self.distance_weight
                * normalised_distance[index]
            )

            self.get_logger().info(
                f'Candidate: '
                f'x={candidate.position.x:.2f} m, '
                f'y={candidate.position.y:.2f} m, '
                f'distance={distances[index]:.2f} m, '
                f'information={information_gains[index]} cells, '
                f'score={score:.3f}'
            )

            # Keep the candidate with the highest score
            if score > best_score:

                best_score = score
                best_candidate = candidate
                best_distance = distances[index]
                best_information = information_gains[index]

        return (
            best_candidate,
            best_score,
            best_distance,
            best_information
        )

    def send_navigation_goal(self, candidate):

        # Check whether the Nav2 NavigateToPose
        # action server is currently available
        if not self.navigation_client.wait_for_server(
            timeout_sec=1.0
        ):

            self.get_logger().warn(
                'Nav2 NavigateToPose action server '
                'is not available'
            )

            return False

        # Create a new Nav2 navigation goal
        goal_msg = NavigateToPose.Goal()

        # The frontier coordinates are expressed
        # relative to the SLAM map
        goal_msg.pose.header.frame_id = 'map'

        goal_msg.pose.header.stamp = (
            self.get_clock().now().to_msg()
        )

        # Copy the selected frontier pose into
        # the Nav2 goal
        goal_msg.pose.pose = candidate

        # Check that the goal orientation contains
        # a valid quaternion
        orientation = goal_msg.pose.pose.orientation

        if (
            orientation.x == 0.0
            and orientation.y == 0.0
            and orientation.z == 0.0
            and orientation.w == 0.0
        ):
            # Identity quaternion represents zero rotation
            orientation.w = 1.0

        self.get_logger().info(
            f'Sending navigation goal: '
            f'x={candidate.position.x:.2f} m, '
            f'y={candidate.position.y:.2f} m'
        )

        # Send the navigation request asynchronously
        send_goal_future = (
            self.navigation_client.send_goal_async(
                goal_msg
            )
        )

        # Run navigation_goal_response() when Nav2
        # responds to the request
        send_goal_future.add_done_callback(
            self.navigation_goal_response
        )

        return True

    def navigation_goal_response(self, future):

        try:
            # Retrieve the Nav2 goal handle
            goal_handle = future.result()

        except Exception as error:

            self.get_logger().error(
                f'Failed to send navigation goal: {error}'
            )

            self.state = 'READY'

            return

        # Nav2 may reject a requested goal before
        # navigation actually begins
        if not goal_handle.accepted:

            self.get_logger().warn(
                'Navigation goal was rejected by Nav2'
            )

            # A rejected goal is treated as an
            # unusable frontier location
            if self.current_goal is not None:

                self.record_failed_goal(
                    self.current_goal
                )

            self.current_goal = None
            self.current_goal_handle = None

            # Allow another frontier to be selected
            self.state = 'READY'

            return

        self.get_logger().info(
            'Navigation goal accepted by Nav2'
        )

        # Store the goal handle so the active
        # navigation request can be tracked
        self.current_goal_handle = goal_handle

        # The robot is now travelling toward
        # the selected frontier
        self.state = 'NAVIGATING'

        self.get_logger().info(
            'Exploration state changed to NAVIGATING'
        )

        # Ask Nav2 for the final navigation result
        self.navigation_result_future = (
            goal_handle.get_result_async()
        )

        # Run navigation_result_callback() when
        # navigation finishes
        self.navigation_result_future.add_done_callback(
            self.navigation_result_callback
        )

    def navigation_result_callback(self, future):

        try:
            result = future.result()

        except Exception as error:

            self.get_logger().error(
                f'Failed to obtain navigation result: {error}'
            )

            # Treat an unknown result failure as a
            # failed navigation attempt
            if self.current_goal is not None:

                self.record_failed_goal(
                    self.current_goal
                )

            self.current_goal = None
            self.current_goal_handle = None
            self.navigation_result_future = None

            self.state = 'READY'

            return

        status = result.status

        # Nav2 reports STATUS_SUCCEEDED when the
        # robot successfully reaches the goal
        if status == GoalStatus.STATUS_SUCCEEDED:

            self.get_logger().info(
                'Navigation goal succeeded'
            )

            # Clear the completed goal
            self.current_goal = None
            self.current_goal_handle = None
            self.navigation_result_future = None

            # Begin a new empty-frontier sequence for
            # the newly explored location
            self.empty_frontier_update_count = 0

            # Do not immediately reuse an old frontier list.
            # Wait until the frontier detector publishes
            # another set based on the updated map.
            self.state = 'WAITING_FOR_UPDATE'

            self.get_logger().info(
                'Waiting for updated frontier data'
            )

            return

        # Any non-success result is treated as a failed
        # navigation attempt for frontier selection
        if status == GoalStatus.STATUS_CANCELED:

            status_name = 'CANCELED'

        elif status == GoalStatus.STATUS_ABORTED:

            status_name = 'ABORTED'

        else:

            status_name = f'STATUS_{status}'

        self.get_logger().warn(
            f'Navigation goal failed: {status_name}'
        )

        # Store the failed region so the same physical
        # area is not repeatedly selected
        if self.current_goal is not None:

            self.record_failed_goal(
                self.current_goal
            )

        # Clear the completed navigation request
        self.current_goal = None
        self.current_goal_handle = None
        self.navigation_result_future = None

        # The current frontier list may already contain
        # alternative candidates. Failed-region filtering
        # will prevent the failed location being reused.
        self.state = 'READY'

        self.get_logger().info(
            'Exploration state returned to READY'
        )

    def exploration_callback(self):

        # Only select a new frontier when the exploration
        # manager is ready for a new goal
        if self.state != 'READY':
            return

        result = self.select_best_candidate()

        if result is None:
            return

        (
            best_candidate,
            best_score,
            best_distance,
            best_information
        ) = result

        # Store the selected frontier as the current goal
        self.current_goal = best_candidate

        self.get_logger().info(
            f'Selected frontier: '
            f'x={best_candidate.position.x:.2f} m, '
            f'y={best_candidate.position.y:.2f} m, '
            f'distance={best_distance:.2f} m, '
            f'information={best_information} cells, '
            f'score={best_score:.3f}'
        )

        # Prevent another frontier from being selected
        # while this goal is being sent to Nav2
        self.state = 'GOAL_SELECTED'

        self.get_logger().info(
            'Exploration state changed to GOAL_SELECTED'
        )

        # Send the selected frontier to Nav2
        goal_sent = self.send_navigation_goal(
            self.current_goal
        )

        # If Nav2 is currently unavailable, return
        # to READY so another attempt can be made later
        if not goal_sent:

            self.state = 'READY'

            self.get_logger().info(
                'Exploration state returned to READY'
            )


def main(args=None):

    rclpy.init(args=args)

    node = ExplorationManager()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
