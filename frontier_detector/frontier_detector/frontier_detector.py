import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np

# message imports
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


class FrontierDetector(Node):

    def __init__(self):
        super().__init__('frontier_detector')

        # subcribe to the map topic
        self.map_subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        # publish the frontier cells as markers
        self.frontier_publisher = self.create_publisher(
            Marker,
            '/frontier_cells',
            10
        )

    def map_callback(self, msg):

        # Convert list to 2D array
        map_array = np.array(msg.data).reshape((msg.info.height, msg.info.width))

        frontiers = self.find_frontiers(map_array)

        self.get_logger().info(f'Found {len(frontiers)} frontier cells')

        self.publish_frontiers(frontiers, msg)

        

    def find_frontiers(self, map_array):

        frontiers = []

        # Get the dimensions of the map
        rows, cols = map_array.shape

        # Iterate through the map, skipping the border cells
        for row in range(1, rows - 1):
            for col in range(1, cols - 1):

                # If the cell is not free, skip it
                if map_array[row, col] != 0:
                    continue

                # Get surrounding 3x3 cells
                neighbours = map_array[row - 1:row + 2, col - 1:col + 2]

                # Count unknown neighbouring cells
                unknown_count = np.sum(neighbours == -1)

                # Only keep strong frontier cells
                if unknown_count >= 3:
                    frontiers.append((row, col))

        return frontiers
    
    # Publish the frontiers as a Marker message
    def publish_frontiers(self, frontiers, msg):

        marker = Marker()

        # Set the header information
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()

        # Set the namespace and id for the marker
        marker.ns = 'frontiers'
        marker.id = 0

        # Set the type and action for the marker
        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        # Size of each point
        marker.scale.x = 0.05
        marker.scale.y = 0.05

        # Colour: red
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # for every frontier cell, create a Point and add it to the marker
        for row, col in frontiers:

            point = Point()

            # Convert the row and column indices to world coordinates
            point.x = (
                msg.info.origin.position.x
                + (col + 0.5) * msg.info.resolution
            )

            point.y = (
                msg.info.origin.position.y
                + (row + 0.5) * msg.info.resolution
            )

            # Set a small z value to make the points visible above the map
            point.z = 0.05

            # Add the point to the marker
            marker.points.append(point)

        # Publish the marker
        self.frontier_publisher.publish(marker)


def main(args=None):
    rclpy.init(args=args)

    node = FrontierDetector()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()