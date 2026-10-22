"""
Goal is to locate any Aruco markers in the given image and publish their poses
Will subscribe to the camera image that is generated and publish the pose of visible markers
"""

import numpy as np
import rclpy
from rclpy.node import Node

import cv2
from cv_bridge import CvBridge

import tf_transformations

from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import TransformStamped

from tf2_ros import TransformBroadcaster

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
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        # Camera parameters - might need more later, will find out
        self.img_msg = None
        self.camera_frame_id = None
        self.cam_matrix = None
        self.distorsion = None

        self.bridge = CvBridge()

       
        # Subcriptions to the CameraInfo and Image topics
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.camera_info_cb, 10
        )

        self.image_sub = self.create_subscription (
            Image, '/camera/image_raw', self.image_cb, 10
        )

        # Publishing the transforms for the markers
        self.tf_broadcaster = TransformBroadcaster(self)


    def camera_info_cb(self, info_msg):
        self.img_msg = info_msg
        self.camera_header = self.img_msg.header
        self.cam_matrix = np.reshape(np.array(self.img_msg.k), (3, 3))
        self.distorsion = np.array(self.img_msg.d)


    def image_cb(self, image_msg):
        #Convert the image to the CV2 image
        cv2_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')

        #Detect any Aruco markers in the image
        corners, ids, rejected = cv2.aruco.detectMarkers(cv2_image, self.aruco_dict, parameters=self.aruco_params)

        #Create a list of aruco marker poses relative to camera
        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners,self.marker_size,self.cam_matrix, self.distorsion)
            # Each marker has a pose relative to the camera, which is added to the tf2 tree
            for i, id in enumerate(ids):
                self.get_logger().info("Detected marker with ID: {}".format(id[0]))
                marker_transform = TransformStamped()
                # Parent frame is the camera
                marker_transform.header = self.camera_header

                # Child frame corresponds to this marker
                marker_transform.child_frame_id = f"marker_{id}"

                # Translation of the transform from tvecss
                marker_transform.transform.translation.x = tvecs[i][0][0]
                marker_transform.transform.translation.y = tvecs[i][0][1]
                marker_transform.transform.translation.z = tvecs[i][0][2]

                # Orientation of the transform from rvecs
                rotation_matrix = np.eye(4)
                rotation_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i][0]))[0]
                quaternion = tf_transformations.quaternion_from_matrix(rotation_matrix)
                marker_transform.transform.rotation.x = quaternion[0]
                marker_transform.transform.rotation.y = quaternion[1]
                marker_transform.transform.rotation.z = quaternion[2]
                marker_transform.transform.rotation.w = quaternion[3]

                self.tf_broadcaster.sendTransform(marker_transform)
               