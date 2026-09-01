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
from geometry_msgs.msg import PoseArray, Pose



class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__("aruco_detector_node")

        # Set up the ArUco Detector uisng OpenCV
        self.marker_size = 0.1 # in metres
        self.aruco_dict_id = "DICT_6X6_250"
        self.aruco_dict = cv2.aruco.Dictionary_get(self.aruco_dict_id)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        # Camera parameters - might need more later, will find out
        self.img_msg = None
        self.cam_matrix = None
        self.distorsion = None

        # CvBridge converts ROS2 Image to CV2 image
        self.bridge = CvBridge()
        
        # Subcriptions to the CameraInfo and Image topics
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/image_raw', self.camera_info_cb, 10
        )

        self.image_sub = self.create_subscription (
            Image, '/camera/camera_info', self.image_cb, 10
        )

        # Publishing a list of Aruco Marker Poses


    def camera_info_cb(self, info_msg):
        self.img_msg = info_msg
        self.cam_matrix = np.reshape(np.array(self.img_msg.k), (3, 3))
        self.distorsion = np.array(self.img_msg.d)


    def image_cb(self, image_msg):
        #Convert the image to the CV2 image
        cv2_image = bridge.imgmsg_to_cv2(image_msg)

        #Detect any Aruco markers in the image
        corners, ids, rejected = cv2.aruco.detectMarkers(cv2_image, self.aruco_dict, parameters=self.aruco_params)

        #Create a list of aruco marker poses relative to camera
        if ids is not None:
            rvecs, tvecs = cv2.aruco.estimatePoseSingleMarkers(corners,self.marker_size,self.cam_matrix, self.distorsion)


