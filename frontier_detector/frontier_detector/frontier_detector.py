import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid


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
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution

        self.get_logger().info(
            f'Map received: {width} x {height}, '
            f'resolution = {resolution:.3f} m'
        )


def main(args=None):
    rclpy.init(args=args)

    node = FrontierDetector()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()