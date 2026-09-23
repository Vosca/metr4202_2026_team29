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
        # resolution and origin will be needed later
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

    def calculate_distance(self, candidate, robot_x, robot_y):

        # Calculate straight-line distance from the robot
        # to the frontier candidate
        distance = math.hypot(
            candidate.position.x - robot_x,
            candidate.position.y - robot_y
        )

        return distance

    def select_nearest_candidate(self):

        # Cannot select anything if there are no frontiers
        if not self.frontier_candidates:
            return None

        robot_position = self.get_robot_position()

        # Cannot compare distances until the robot pose is known
        if robot_position is None:
            return None

        robot_x, robot_y = robot_position

        nearest_candidate = None
        nearest_distance = float('inf')

        # Compare every current frontier candidate
        for candidate in self.frontier_candidates:

            distance = self.calculate_distance(
                candidate,
                robot_x,
                robot_y
            )

            if distance < nearest_distance:
                nearest_distance = distance
                nearest_candidate = candidate

        return nearest_candidate, nearest_distance

    def exploration_callback(self):

        # Only select a new frontier when the exploration
        # manager is ready for a new goal
        if self.state != 'READY':
            return

        result = self.select_nearest_candidate()

        if result is None:
            return

        nearest_candidate, nearest_distance = result

        # Store the selected frontier as the current goal
        self.current_goal = nearest_candidate

        self.get_logger().info(
            f'Selected frontier: '
            f'x={nearest_candidate.position.x:.2f} m, '
            f'y={nearest_candidate.position.y:.2f} m, '
            f'distance={nearest_distance:.2f} m'
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
