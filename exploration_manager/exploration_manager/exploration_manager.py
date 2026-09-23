import math

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener


class ExplorationManager(Node):
    def __init__(self):
        super().__init__('exploration_manager')

        self.information_radius = float(
            self.declare_parameter('information_radius', 1.0).value
        )
        self.information_weight = float(
            self.declare_parameter('information_weight', 0.6).value
        )
        self.distance_weight = float(
            self.declare_parameter('distance_weight', 0.4).value
        )
        self.failed_region_radius = float(
            self.declare_parameter('failed_region_radius', 0.5).value
        )
        self.required_empty_updates = max(
            1,
            int(self.declare_parameter('required_empty_updates', 3).value),
        )
        self.map_frame = str(
            self.declare_parameter('map_frame', 'map').value
        )
        self.robot_frame = str(
            self.declare_parameter('robot_frame', 'base_link').value
        )

        self.frontier_candidates = []
        self.failed_regions = []
        self.map_msg = None
        self.map_array = None
        self.current_goal = None
        self.empty_frontier_update_count = 0
        self.state = 'WAITING_FOR_DATA'

        self.create_subscription(
            PoseArray,
            '/frontier_candidates',
            self.frontier_candidates_callback,
            10
        )
        self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        self.navigation_client = ActionClient(
            self,
            NavigateToPose,
            'navigate_to_pose'
        )

        self.create_timer(
            1.0,
            self.exploration_callback
        )

        self.get_logger().info(
            'Exploration manager started'
        )

    def frontier_candidates_callback(self, msg):
        self.frontier_candidates = list(msg.poses)

        if self.state == 'EXPLORATION_COMPLETE':
            return

        if self.state == 'WAITING_FOR_UPDATE':
            if self.frontier_candidates:
                self.empty_frontier_update_count = 0
                self.state = 'READY'
                self.get_logger().info(
                    'Fresh frontiers received'
                )
            else:
                self.empty_frontier_update_count += 1

                self.get_logger().info(
                    f'Empty frontier update '
                    f'{self.empty_frontier_update_count}/'
                    f'{self.required_empty_updates}'
                )

                if (
                    self.empty_frontier_update_count
                    >= self.required_empty_updates
                ):
                    self.state = 'EXPLORATION_COMPLETE'

                    self.get_logger().info(
                        'Exploration complete'
                    )

            return

        if self.frontier_candidates:
            self.empty_frontier_update_count = 0

        self.update_ready_state()

    def map_callback(self, msg):
        self.map_msg = msg

        self.map_array = np.asarray(
            msg.data,
            dtype=np.int16
        ).reshape(
            msg.info.height,
            msg.info.width
        )

        self.update_ready_state()

    def update_ready_state(self):
        if (
            self.state == 'WAITING_FOR_DATA'
            and self.map_array is not None
            and self.frontier_candidates
        ):
            self.state = 'READY'

            self.get_logger().info(
                'Exploration manager ready'
            )

    def get_robot_position(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.robot_frame,
                Time()
            )

            position = transform.transform.translation

            return position.x, position.y

        except Exception:
            return None

    @staticmethod
    def calculate_distance(
        candidate,
        robot_x,
        robot_y
    ):
        return math.hypot(
            candidate.position.x - robot_x,
            candidate.position.y - robot_y
        )

    def world_to_cell(self, x, y):
        if (
            self.map_msg is None
            or self.map_msg.info.resolution <= 0.0
        ):
            return None

        info = self.map_msg.info
        origin = info.origin
        q = origin.orientation

        yaw = math.atan2(
            2.0 * (
                q.w * q.z
                + q.x * q.y
            ),
            1.0 - 2.0 * (
                q.y * q.y
                + q.z * q.z
            )
        )

        dx = x - origin.position.x
        dy = y - origin.position.y

        local_x = (
            dx * math.cos(yaw)
            + dy * math.sin(yaw)
        )
        local_y = (
            -dx * math.sin(yaw)
            + dy * math.cos(yaw)
        )

        col = math.floor(
            local_x / info.resolution
        )
        row = math.floor(
            local_y / info.resolution
        )

        if (
            0 <= row < info.height
            and 0 <= col < info.width
        ):
            return row, col

        return None

    def calculate_information_gain(self, candidate):
        cell = self.world_to_cell(
            candidate.position.x,
            candidate.position.y
        )

        if cell is None:
            return 0

        row, col = cell
        resolution = self.map_msg.info.resolution
        radius = max(
            0.0,
            self.information_radius
        )

        radius_cells = math.ceil(
            radius / resolution
        )

        rows, cols = self.map_array.shape

        row_start = max(
            0,
            row - radius_cells
        )
        row_end = min(
            rows,
            row + radius_cells + 1
        )
        col_start = max(
            0,
            col - radius_cells
        )
        col_end = min(
            cols,
            col + radius_cells + 1
        )

        region = self.map_array[
            row_start:row_end,
            col_start:col_end
        ]

        row_grid, col_grid = np.ogrid[
            row_start:row_end,
            col_start:col_end
        ]

        radius_mask = (
            (
                (row_grid - row)
                * resolution
            ) ** 2
            +
            (
                (col_grid - col)
                * resolution
            ) ** 2
            <= radius ** 2
        )

        return int(
            np.count_nonzero(
                (region == -1)
                & radius_mask
            )
        )

    def is_failed_region(self, candidate):
        return any(
            math.hypot(
                candidate.position.x - failed_x,
                candidate.position.y - failed_y
            ) < self.failed_region_radius
            for failed_x, failed_y
            in self.failed_regions
        )

    def record_failed_goal(self, candidate):
        if (
            candidate is None
            or self.is_failed_region(candidate)
        ):
            return

        position = (
            candidate.position.x,
            candidate.position.y
        )

        self.failed_regions.append(
            position
        )

        self.get_logger().warn(
            f'Blacklisted failed frontier at '
            f'x={position[0]:.2f}, '
            f'y={position[1]:.2f}'
        )

    @staticmethod
    def normalise(values, equal_value):
        values = np.asarray(
            values,
            dtype=float
        )

        span = np.ptp(values)

        if span == 0.0:
            return np.full(
                values.shape,
                equal_value,
                dtype=float
            )

        return (
            values - np.min(values)
        ) / span

    def select_best_candidate(self):
        valid_candidates = [
            candidate
            for candidate
            in self.frontier_candidates
            if not self.is_failed_region(candidate)
        ]

        if not valid_candidates:
            return None

        robot_position = self.get_robot_position()

        if robot_position is None:
            return None

        distances = np.asarray(
            [
                self.calculate_distance(
                    candidate,
                    *robot_position
                )
                for candidate
                in valid_candidates
            ]
        )

        information = np.asarray(
            [
                self.calculate_information_gain(
                    candidate
                )
                for candidate
                in valid_candidates
            ],
            dtype=float
        )

        scores = (
            self.information_weight
            * self.normalise(
                information,
                1.0
            )
            -
            self.distance_weight
            * self.normalise(
                distances,
                0.0
            )
        )

        best_index = int(
            np.argmax(scores)
        )

        for index, candidate in enumerate(
            valid_candidates
        ):
            self.get_logger().info(
                f'Candidate '
                f'x={candidate.position.x:.2f}, '
                f'y={candidate.position.y:.2f}, '
                f'd={distances[index]:.2f}, '
                f'info={int(information[index])}, '
                f'score={scores[index]:.3f}'
            )

        return (
            valid_candidates[best_index],
            float(scores[best_index]),
            float(distances[best_index]),
            int(information[best_index])
        )

    def send_navigation_goal(self, candidate):
        if not self.navigation_client.wait_for_server(
            timeout_sec=1.0
        ):
            self.get_logger().warn(
                'Nav2 navigate_to_pose unavailable'
            )

            return False

        goal = NavigateToPose.Goal()

        goal.pose.header.frame_id = (
            self.map_frame
        )
        goal.pose.header.stamp = (
            self.get_clock().now().to_msg()
        )
        goal.pose.pose = candidate

        q = goal.pose.pose.orientation

        if (
            q.x == q.y == q.z == q.w == 0.0
        ):
            q.w = 1.0

        future = (
            self.navigation_client
            .send_goal_async(goal)
        )

        future.add_done_callback(
            self.navigation_goal_response
        )

        return True

    def navigation_goal_response(self, future):
        try:
            goal_handle = future.result()

        except Exception as error:
            self.get_logger().error(
                f'Goal request failed: {error}'
            )

            self.current_goal = None
            self.state = 'READY'

            return

        if not goal_handle.accepted:
            self.get_logger().warn(
                'Nav2 rejected frontier goal'
            )

            self.record_failed_goal(
                self.current_goal
            )

            self.current_goal = None
            self.state = 'READY'

            return

        self.state = 'NAVIGATING'

        self.get_logger().info(
            'Nav2 goal accepted'
        )

        result_future = (
            goal_handle.get_result_async()
        )

        result_future.add_done_callback(
            self.navigation_result_callback
        )

    def navigation_result_callback(self, future):
        try:
            status = future.result().status

        except Exception as error:
            self.get_logger().error(
                f'Navigation result failed: {error}'
            )

            self.record_failed_goal(
                self.current_goal
            )

            self.current_goal = None
            self.state = 'READY'

            return

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(
                'Frontier reached'
            )

            self.current_goal = None
            self.empty_frontier_update_count = 0
            self.state = 'WAITING_FOR_UPDATE'

            return

        status_name = {
            GoalStatus.STATUS_CANCELED:
                'CANCELED',
            GoalStatus.STATUS_ABORTED:
                'ABORTED'
        }.get(
            status,
            str(status)
        )

        self.get_logger().warn(
            f'Navigation failed: {status_name}'
        )

        self.record_failed_goal(
            self.current_goal
        )

        self.current_goal = None
        self.state = 'READY'

    def exploration_callback(self):
        if self.state != 'READY':
            return

        result = self.select_best_candidate()

        if result is None:
            return

        (
            candidate,
            score,
            distance,
            information
        ) = result

        self.current_goal = candidate
        self.state = 'GOAL_SELECTED'

        self.get_logger().info(
            f'Selected '
            f'x={candidate.position.x:.2f}, '
            f'y={candidate.position.y:.2f}, '
            f'd={distance:.2f}, '
            f'info={information}, '
            f'score={score:.3f}'
        )

        if not self.send_navigation_goal(
            candidate
        ):
            self.current_goal = None
            self.state = 'READY'


def main(args=None):
    rclpy.init(args=args)

    node = ExplorationManager()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
