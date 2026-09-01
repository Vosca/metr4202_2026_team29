import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np


class FrontierDetector(Node):

    def __init__(self):
        super().__init__('frontier_detector')

        self.map_subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

    def map_callback(self, msg):

        # Convert list to 2D array
        map_array = np.array(msg.data).reshape((msg.info.height, msg.info.width))

        frontiers = self.find_frontiers(map_array)

        self.get_logger().info(f'Found {len(frontiers)} frontier cells')

        

    def find_frontiers(self, map_array):

        frontiers = []

        # Get the dimensions of the map
        rows, cols = map_array.shape

        for row in range(1, rows - 1):
            for col in range(1, cols - 1):

                # If the cell is not free, skip it
                if map_array[row, col] != 0:
                    continue

                # Get surrounding 3x3 cells
                neighbours = map_array[row - 1:row + 2, col - 1:col + 2]

                # If our cell is next to an unknown cell, it's a frontier
                if np.any(neighbours == -1):
                    frontiers.append((row, col))

        return frontiers


def main(args=None):
    rclpy.init(args=args)

    node = FrontierDetector()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()