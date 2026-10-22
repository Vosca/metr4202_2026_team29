"""
Goal is to locate any Aruco markers in the given image and publish their poses
Will subscribe to the camera image that is generated and publish the pose of visible markers
"""

import numpy as np
import rclpy
from rclpy.node import Node

import cv2
from cv_bridge import CvBridge

from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

def main(args=None):
    rclpy.init(args=args)
    detector = ArucoDetectorNode()
    rclpy.spin(detector)

if __name__ == '__main__':
    main()

class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__("aruco_detector_node")

        # Set up the ArUco Detector uisng OpenCV
        self.marker_size = 0.1 # in metres
        self.aruco_dict_id = cv2.aruco.DICT_6X6_250
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.aruco_dict_id)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # Camera parameters - might need more later, will find out
        self.curr_image = None
        self.info_msg = None
        self.camera_header = None
        self.camera_frame_id = None
        self.cam_matrix = None
        self.distorsion = None

        self.bridge = CvBridge()

        # Previously identified ArUco markers, updated every second
        self.markers = None

        # Subcriptions to the CameraInfo and Image topics
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.camera_info_cb, 10
        )

        self.image_sub = self.create_subscription (
            Image, '/camera/image_raw', self.image_cb, 10
        )

        self.timer = self.create_timer(1.0, self.image_processing)

        self.marker_pub = self.create_publisher(
            Marker, '/aruco_markers', 10
        )


    def camera_info_cb(self, info_msg):
        self.info_msg = info_msg
        self.camera_header = self.info_msg.header
        self.cam_matrix = np.reshape(np.array(self.info_msg.k), (3, 3))
        self.distorsion = np.array(self.info_msg.d)
        self.markers = self.initialise_marker()
        self.node.destroy_subscription(self.camera_info_sub)
        self.camera_info_sub = None


    def image_cb(self, image_msg):
        self.curr_image = image_msg

    def image_processing(self):
        if self.curr_image == None:
            self.get_logger().info("No image detected yet...")
            return

        #Convert the image to the CV2 image
        cv2_image = self.bridge.imgmsg_to_cv2(self.curr_image, desired_encoding='passthrough')

        #Detect any Aruco markers in the image
        corners, ids, rejected = cv2.aruco.detectMarkers(cv2_image, self.aruco_dict, parameters=self.aruco_params)

        #Create a list of aruco marker poses relative to camera
        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners,self.marker_size,self.cam_matrix, self.distorsion)
            # Each marker has a pose relative to the camera, which is added to the tf2 tree
            for i, id in enumerate(ids):
                self.get_logger().info("Detected marker with ID: {}".format(id[0]))

                marker_point = Point()
                marker_point.x = tvecs[i][0][0]
                marker_point.y = tvecs[i][0][1]
                marker_point.z = tvecs[i][0][2]
                self.markers.points.append(marker_point)

                """# Orientation of the transform from rvecs
                rotation_matrix = np.eye(4)
                rotation_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i][0]))[0]
                quaternion = tf_transformations.quaternion_from_matrix(rotation_matrix)
                marker.pose.rotation.x = quaternion[0]
                marker.pose.rotation.y = quaternion[1]
                marker.pose.rotation.z = quaternion[2]
                marker.pose.rotation.w = quaternion
                """

        self.marker_pub.publish(self.markers)

    def initialise_marker(self):
        m = Marker()
        m.header = self.camera_header
        m.ns = "ArUco Markers"
        m.id = id
        m.type = 7 #Sphere List
        m.action = 0 #Add
        m.pose.position.x = 0
        m.pose.position.y = 0
        m.pose.position.z = 0
        m.pose.orientation.x = 0
        m.pose.orientation.y = 0
        m.pose.orientation.z = 0
        m.pose.orientation.w = 1
        m.scale.x = 0.1
        m.scale.y = 0.1
        m.scale.z = 0.1
        m.color.r = 0.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.color.a = 1.0
        m.lifetime = 0
        m.frame_locked = True
               