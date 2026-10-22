import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformException
from tf2_ros import TransformListener
import tf2_ros
import yaml

def main(args=None):
    rclpy.init(args=args)
    detector = PoseGeneratorNode()
    rclpy.spin(detector)

if __name__ == '__main__':
    main()

class PoseGeneratorNode(Node):
    def __init__(self):
        super().__init__('pose_generator_node')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        try:
            # Look up the transform from the camera frame to the marker frame
            yaml_str = self.tf_buffer.all_frames_as_yaml()
            data = yaml.safe_load(yaml_str)
            self.get_logger().info(f'Printing markers:')
            for key,value in data.items():
                if 'marker' in key:
                    self.get_logger().info(f'Marker frame: {key}, Transform: {value}')
                    try:
                        marker_pose = PoseStamped()
                        marker_pose.header.frame_id = key
                        marker_pose.header.stamp = self.get_clock().now().to_msg()
                        marker_pose.pose.position.x = 0
                        marker_pose.pose.position.y = 0
                        transform = self.tf_buffer.lookup_transform('map', key, rclpy.time.Time())
                        self.get_logger().info(f'Transform from {key} to map: {transform}')
                    except TransformException as e:
                        self.get_logger().error(f'Could not get transform from {key} to map: {e}')
            

        except Exception as e:
            self.get_logger().error(f'Could not get transform: {e}')