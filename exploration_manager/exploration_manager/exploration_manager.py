import math

import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import PoseArray
from nav_msgs.msg import OccupancyGrid
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

        # Store the current exploration state
        self.state = 'WAITING_FOR_DATA'

        # Store the currently selected frontier goal
        self.current_goal = None

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

        # Exploration can begin only when both:
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

        robot_position = self.get_robot_position()

        # Cannot evaluate candidates until robot pose is known
        if robot_position is None:
            return None

        robot_x, robot_y = robot_position

        distances = []
        information_gains = []

        # First calculate the raw values for every
        # frontier candidate
        for candidate in self.frontier_candidates:

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

        # Calculate the utility score for every candidate
        for index, candidate in enumerate(
            self.frontier_candidates
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
        # while this goal is active
        self.state = 'GOAL_SELECTED'

        self.get_logger().info(
            'Exploration state changed to GOAL_SELECTED'
        )


def main(args=None):

    rclpy.init(args=args)

    node = ExplorationManager()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
