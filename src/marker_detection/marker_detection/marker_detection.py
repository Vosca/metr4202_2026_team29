"""
Goal is to locate any Aruco markers in the given image and publish their poses
Will subscribe to the camera image that is generated and publish the pose of visible markers
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

import cv2
from cv_bridge import CvBridge

from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import Pose, PoseStamped, PoseArray
from visualization_msgs.msg import Marker, MarkerArray
import tf_transformations
import tf2_ros
from tf2_geometry_msgs import do_transform_pose
import math

def main(args=None):
    rclpy.init(args=args)
    detector = ArucoDetectorNode()
    rclpy.spin(detector)

if __name__ == '__main__':
    main()

class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__("aruco_detector_node")

        # Marker Parameters
        self.marker_size = 0.1 # in metres

        # Set up the ArUco Detector uisng OpenCV
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        detector_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

        # Camera parameters - might need more later, will find out
        self.curr_image = None
        self.camera_frame_id = None
        self.cam_matrix = None
        self.distorsion = None

        self.bridge = CvBridge()

        # Previously identified ArUco markers, updated every second
        self.marker_poses = PoseArray()
        self.marker_ids = []

        # Subcriptions to the CameraInfo and Image topics
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.camera_info_cb, 10
        )

        self.image_sub = self.create_subscription (
            Image, '/camera/image_raw', self.image_cb, 10
        )

        self.create_timer(1.0, self.display_markers)
        self.create_timer(0.5, self.image_processing)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/aruco_markers', 10
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)


    def camera_info_cb(self, info_msg):
        self.camera_frame_id = info_msg.header.frame_id
        self.cam_matrix = np.reshape(np.array(info_msg.k), (3, 3))
        self.distorsion = np.array(info_msg.d)
        self.destroy_subscription(self.camera_info_sub)
        self.camera_info_sub = None


    def image_cb(self, image_msg): #Should scrub every image and update local memory
        self.curr_image = image_msg
        
    def image_processing(self):
        #Convert the image to the CV2 image
        if self.curr_image == None:
            return 
        
        cv2_image = self.bridge.imgmsg_to_cv2(self.curr_image, desired_encoding='passthrough')

        #Detect any Aruco markers in the image
        corners, ids, rejected = self.detector.detectMarkers(cv2_image)
        marker_points = np.array([
            [-self.marker_size / 2 , self.marker_size / 2, 0],
            [self.marker_size / 2 , self.marker_size / 2, 0],
            [self.marker_size / 2 , -self.marker_size / 2, 0],
            [-self.marker_size / 2 , -self.marker_size / 2, 0]
        ], dtype=np.float32)
        #Create a list of aruco marker poses relative to camera
        if ids is not None:
            transform = self.tf_buffer.lookup_transform(
                'map',
                self.camera_frame_id,
                Time(),
                rclpy.duration.Duration(seconds=1)
            )

            
            for i, c in enumerate(corners):
                corner_points = c.reshape((4, 2))

                success, rvec, tvec = cv2.solvePnP(
                    marker_points,
                    corner_points,
                    self.cam_matrix,
                    self.distorsion,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )

                if success:
                    #self.get_logger().info("Detected marker with ID: {}".format(ids[0]))
                    aruco_pose = self.create_pose(tvec, rvec)
                    pose_out = do_transform_pose(aruco_pose, transform)
                    self.check_existing_poses(pose_out, ids[i])

    def create_pose(self, tvec, rvec):
        p = Pose()
        p.position.x = float(tvec[0])
        p.position.y = float(tvec[1])
        p.position.z = float(tvec[2])

        rotation_matrix = np.eye(4)
        rotation_matrix[0:3, 0:3], _ = cv2.Rodrigues(np.array(rvec, dtype=np.float32))
        quaternion = tf_transformations.quaternion_from_matrix(rotation_matrix)
        p.orientation.x = quaternion[0]
        p.orientation.y = quaternion[1]
        p.orientation.z = quaternion[2]
        p.orientation.w = quaternion[3]
        return p

    def check_existing_poses(self, candidate, candidate_id):
        if candidate_id in self.marker_ids:
            return
        
        for existing_pose in self.marker_poses.poses:
            #If the candidate is too close, return
            dx = candidate.position.x - existing_pose.position.x
            dy = candidate.position.y - existing_pose.position.y
            dz = candidate.position.z - existing_pose.position.z
            dist_3d = math.sqrt(dx**2 + dy**2 + dz**2)
            if dist_3d < 0.05:
                return
        
        self.marker_poses.poses.append(candidate)
        self.marker_ids.append(candidate_id)
        self.get_logger().info("Adding one")

    def create_marker_from_pose(self, marker_id, pose):
        m = Marker()
        m.header.frame_id = 'map'
        m.ns = "ArUco Marker"
        m.id = marker_id
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose = pose
        # Scale is 10cm x 10cm x 10cm
        m.scale.x = 0.1
        m.scale.y = 0.1
        m.scale.z = 0.1
        # RED
        m.color.r = 0.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.color.a = 1.0

        m.lifetime = Duration().to_msg()
        m.frame_locked = False

        return m

    def display_markers(self):
        aruco_markers = MarkerArray();
        for i, pose in enumerate(self.marker_poses.poses):
            marker = self.create_marker_from_pose(i, pose)
            aruco_markers.markers.append(marker)

        self.marker_pub.publish(aruco_markers)
        self.get_logger().info("Number: {}".format(len(aruco_markers.markers)))