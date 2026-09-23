import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.action import ActionClient

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose

from tf2_ros import Buffer, TransformListener


class ExplorationManager(Node):

    def __init__(self):
        super().__init__('exploration_manager')

        # Set the frontier scoring parameters
        self.information_radius = self.declare_parameter(
            'information_radius', 1.0
        ).value

        self.information_weight = self.declare_parameter(
            'information_weight', 0.6
        ).value

        self.distance_weight = self.declare_parameter(
            'distance_weight', 0.4
        ).value

        # Set the failed goal and completion parameters
        self.failed_region_radius = self.declare_parameter(
            'failed_region_radius', 0.5
        ).value

        self.required_empty_updates = self.declare_parameter(
            'required_empty_updates', 3
        ).value

        # Set the coordinate frames
        self.map_frame = self.declare_parameter(
            'map_frame', 'map'
        ).value

        self.robot_frame = self.declare_parameter(
            'robot_frame', 'base_link'
        ).value

        # Store the latest frontier and map information
        self.frontier_candidates = []
        self.map_msg = None
        self.map_array = None

        # Store failed frontier locations
        self.failed_regions = []

        # Store the currently selected navigation goal
        self.current_goal = None

        # Keep track of empty frontier updates
        self.empty_frontier_update_count = 0

        # Start by waiting for map and frontier data
        self.state = 'WAITING_FOR_DATA'

        # Subscribe to the frontier candidates
        self.frontier_subscription = self.create_subscription(
            PoseArray,
            '/frontier_candidates',
            self.frontier_candidates_cb,
            10
        )

        # Subscribe to the occupancy grid map
        self.map_subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_cb,
            10
        )

        # Set up the TF listener to find the robot position
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        # Set up the Nav2 action client
        self.navigation_client = ActionClient(
            self,
            NavigateToPose,
            'navigate_to_pose'
        )

        # Run the exploration manager once per second
        self.exploration_timer = self.create_timer(
            1.0,
            self.exploration_cb
        )

        self.get_logger().info(
            'Exploration manager started'
        )


    def frontier_candidates_cb(self, msg):

        # Store the newest set of frontier candidates
        self.frontier_candidates = list(msg.poses)

        # Do nothing once exploration is complete
        if self.state == 'EXPLORATION_COMPLETE':
            return

        # After reaching a frontier, wait for fresh frontier data
        if self.state == 'WAITING_FOR_UPDATE':

            # Continue exploring if new frontiers are available
            if self.frontier_candidates:
                self.empty_frontier_update_count = 0
                self.state = 'READY'

                self.get_logger().info(
                    'New frontier data received'
                )

                return

            # Count consecutive empty frontier updates
            self.empty_frontier_update_count += 1

            self.get_logger().info(
                f'Empty frontier update '
                f'{self.empty_frontier_update_count}/'
                f'{self.required_empty_updates}'
            )

            # Stop exploration after enough empty updates
            if (
                self.empty_frontier_update_count
                >= self.required_empty_updates
            ):
                self.state = 'EXPLORATION_COMPLETE'

                self.get_logger().info(
                    'Exploration complete'
                )

            return

        # Reset the empty counter if frontiers are available
        if self.frontier_candidates:
            self.empty_frontier_update_count = 0

        self.check_ready_state()


    def map_cb(self, msg):

        # Store the newest map message
        self.map_msg = msg

        # Convert the OccupancyGrid list to a 2D array
        self.map_array = np.array(
            msg.data
        ).reshape(
            (
                msg.info.height,
                msg.info.width
            )
        )

        self.check_ready_state()


    def check_ready_state(self):

        # Exploration can start once both the map and
        # frontier candidates have been received
        if (
            self.state == 'WAITING_FOR_DATA'
            and self.map_array is not None
            and self.frontier_candidates
        ):
            self.state = 'READY'

            self.get_logger().info(
                'Exploration manager ready'
            )


    def get_robot_position(self):

        try:
            # Find the robot position relative to the map
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.robot_frame,
                Time()
            )

            robot_x = transform.transform.translation.x
            robot_y = transform.transform.translation.y

            return robot_x, robot_y

        except Exception:
            return None


    def calculate_distance(
        self,
        candidate,
        robot_x,
        robot_y
    ):

        # Calculate the straight-line distance from the
        # robot to the frontier candidate
        distance = math.hypot(
            candidate.position.x - robot_x,
            candidate.position.y - robot_y
        )

        return distance


    def world_to_cell(
        self,
        world_x,
        world_y
    ):

        if self.map_msg is None:
            return None

        resolution = self.map_msg.info.resolution
        origin = self.map_msg.info.origin

        orientation = origin.orientation

        yaw = math.atan2(
            2.0 * (
                orientation.w * orientation.z
                + orientation.x * orientation.y
            ),
            1.0 - 2.0 * (
                orientation.y**2
                + orientation.z**2
            )
        )

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        delta_x = world_x - origin.position.x
        delta_y = world_y - origin.position.y

        local_x = (
            delta_x * cos_yaw
            + delta_y * sin_yaw
        )

        local_y = (
            -delta_x * sin_yaw
            + delta_y * cos_yaw
        )

        col = math.floor(
            local_x / resolution
        )

        row = math.floor(
            local_y / resolution
        )

        if (
            row < 0
            or row >= self.map_msg.info.height
            or col < 0
            or col >= self.map_msg.info.width
        ):
            return None

        return row, col


    def calculate_information_gain(self, candidate):

        cell = self.world_to_cell(
            candidate.position.x,
            candidate.position.y
        )

        if cell is None:
            return 0

        row, col = cell
        resolution = self.map_msg.info.resolution

        radius_cells = math.ceil(
            self.information_radius / resolution
        )

        rows, cols = self.map_array.shape
        unknown_count = 0

        for check_row in range(
            max(0, row - radius_cells),
            min(rows, row + radius_cells + 1)
        ):
            for check_col in range(
                max(0, col - radius_cells),
                min(cols, col + radius_cells + 1)
            ):

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

                if distance > self.information_radius:
                    continue

                if (
                    self.map_array[
                        check_row,
                        check_col
                    ] == -1
                ):
                    unknown_count += 1

        return unknown_count


    def is_failed_region(self, candidate):

        for failed_x, failed_y in self.failed_regions:

            distance = math.hypot(
                candidate.position.x - failed_x,
                candidate.position.y - failed_y
            )

            if distance < self.failed_region_radius:
                return True

        return False


    def record_failed_goal(self, candidate):

        if candidate is None:
            return

        if self.is_failed_region(candidate):
            return

        failed_position = (
            candidate.position.x,
            candidate.position.y
        )

        self.failed_regions.append(
            failed_position
        )

        self.get_logger().warn(
            f'Failed frontier recorded at '
            f'x={failed_position[0]:.2f}, '
            f'y={failed_position[1]:.2f}'
        )


    def normalise_values(
        self,
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

        normalised = (
            values - minimum
        ) / (
            maximum - minimum
        )

        return normalised


    def select_best_candidate(self):

        # Remove candidates near failed regions
        valid_candidates = []

        for candidate in self.frontier_candidates:

            if self.is_failed_region(candidate):
                continue

            valid_candidates.append(
                candidate
            )

        if not valid_candidates:
            return None

        robot_position = self.get_robot_position()

        if robot_position is None:
            return None

        robot_x, robot_y = robot_position

        distances = []
        information_gains = []

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

            distances.append(
                distance
            )

            information_gains.append(
                information_gain
            )

        normalised_information = self.normalise_values(
            information_gains,
            1.0
        )

        normalised_distance = self.normalise_values(
            distances,
            0.0
        )

        best_candidate = None
        best_score = -float('inf')
        best_distance = None
        best_information = None

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
                f'Candidate '
                f'x={candidate.position.x:.2f}, '
                f'y={candidate.position.y:.2f}, '
                f'distance={distances[index]:.2f}, '
                f'information={information_gains[index]}, '
                f'score={score:.3f}'
            )

            if score > best_score:
                best_candidate = candidate
                best_score = score
                best_distance = distances[index]
                best_information = information_gains[index]

        return (
            best_candidate,
            best_score,
            best_distance,
            best_information
        )


    def send_navigation_goal(self, candidate):

        if not self.navigation_client.wait_for_server(
            timeout_sec=1.0
        ):
            self.get_logger().warn(
                'Nav2 action server unavailable'
            )

            return False

        goal_msg = NavigateToPose.Goal()

        goal_msg.pose.header.frame_id = (
            self.map_frame
        )

        goal_msg.pose.header.stamp = (
            self.get_clock().now().to_msg()
        )

        goal_msg.pose.pose = candidate

        orientation = goal_msg.pose.pose.orientation

        if (
            orientation.x == 0.0
            and orientation.y == 0.0
            and orientation.z == 0.0
            and orientation.w == 0.0
        ):
            orientation.w = 1.0

        send_goal_future = (
            self.navigation_client.send_goal_async(
                goal_msg
            )
        )

        send_goal_future.add_done_callback(
            self.navigation_goal_response
        )

        return True


    def navigation_goal_response(self, future):

        try:
            goal_handle = future.result()

        except Exception as error:

            self.get_logger().error(
                f'Could not send navigation goal: {error}'
            )

            self.current_goal = None
            self.state = 'READY'

            return

        if not goal_handle.accepted:

            self.get_logger().warn(
                'Navigation goal rejected'
            )

            self.record_failed_goal(
                self.current_goal
            )

            self.current_goal = None
            self.state = 'READY'

            return

        self.state = 'NAVIGATING'

        self.get_logger().info(
            'Navigation goal accepted'
        )

        result_future = (
            goal_handle.get_result_async()
        )

        result_future.add_done_callback(
            self.navigation_result_cb
        )


    def navigation_result_cb(self, future):

        try:
            status = future.result().status

        except Exception as error:

            self.get_logger().error(
                f'Could not get navigation result: {error}'
            )

            self.record_failed_goal(
                self.current_goal
            )

            self.current_goal = None
            self.state = 'READY'

            return

        if status == GoalStatus.STATUS_SUCCEEDED:

            self.get_logger().info(
                'Frontier reached successfully'
            )

            self.current_goal = None
            self.empty_frontier_update_count = 0
            self.state = 'WAITING_FOR_UPDATE'

            return

        if status == GoalStatus.STATUS_ABORTED:
            status_name = 'ABORTED'

        elif status == GoalStatus.STATUS_CANCELED:
            status_name = 'CANCELED'

        else:
            status_name = str(status)

        self.get_logger().warn(
            f'Navigation failed: {status_name}'
        )

        self.record_failed_goal(
            self.current_goal
        )

        self.current_goal = None
        self.state = 'READY'


    def exploration_cb(self):

        if self.state != 'READY':
            return

        result = self.select_best_candidate()

        if result is None:
            return

        (
            candidate,
            score,
            distance,
            information_gain
        ) = result

        self.current_goal = candidate
        self.state = 'GOAL_SELECTED'

        self.get_logger().info(
            f'Selected frontier '
            f'x={candidate.position.x:.2f}, '
            f'y={candidate.position.y:.2f}, '
            f'distance={distance:.2f}, '
            f'information={information_gain}, '
            f'score={score:.3f}'
        )

        goal_sent = self.send_navigation_goal(
            candidate
        )

        if not goal_sent:
            self.current_goal = None
            self.state = 'READY'


def main(args=None):

    rclpy.init(args=args)

    node = ExplorationManager()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
