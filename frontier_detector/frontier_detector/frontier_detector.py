import math
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Pose, PoseArray
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from visualization_msgs.msg import Marker


class FrontierDetector(Node):
    NEIGHBOURS = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1)
    ]

    def __init__(self):
        super().__init__('frontier_detector')

        self.min_unknown_neighbours = max(
            1,
            int(
                self.declare_parameter(
                    'min_unknown_neighbours',
                    3
                ).value
            )
        )

        self.min_frontier_size = max(
            1,
            int(
                self.declare_parameter(
                    'min_frontier_size',
                    3
                ).value
            )
        )

        self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.candidate_publisher = (
            self.create_publisher(
                PoseArray,
                '/frontier_candidates',
                10
            )
        )

        self.frontier_marker_publisher = (
            self.create_publisher(
                Marker,
                '/frontier_cells',
                10
            )
        )

        self.representative_marker_publisher = (
            self.create_publisher(
                Marker,
                '/frontier_representatives',
                10
            )
        )

        self.get_logger().info(
            'Frontier detector started'
        )

    def map_callback(self, msg):
        if (
            msg.info.width == 0
            or msg.info.height == 0
        ):
            self.publish_candidates(
                [],
                msg
            )

            return

        grid = np.asarray(
            msg.data,
            dtype=np.int16
        ).reshape(
            msg.info.height,
            msg.info.width
        )

        frontier_cells = (
            self.find_frontier_cells(grid)
        )

        clusters = [
            cluster
            for cluster
            in self.cluster_frontiers(
                frontier_cells
            )
            if len(cluster)
            >= self.min_frontier_size
        ]

        representatives = (
            self.select_representatives(
                clusters
            )
        )

        self.publish_candidates(
            representatives,
            msg
        )

        self.publish_markers(
            frontier_cells,
            representatives,
            msg
        )

        self.get_logger().info(
            f'Frontiers: '
            f'{len(frontier_cells)} cells, '
            f'{len(clusters)} clusters, '
            f'{len(representatives)} candidates'
        )

    def find_frontier_cells(self, grid):
        rows, cols = grid.shape

        unknown = (
            grid == -1
        ).astype(
            np.uint8
        )

        padded = np.pad(
            unknown,
            1
        )

        unknown_neighbours = np.zeros_like(
            grid,
            dtype=np.uint8
        )

        for row_offset, col_offset in self.NEIGHBOURS:
            unknown_neighbours += padded[
                1 + row_offset:
                1 + row_offset + rows,
                1 + col_offset:
                1 + col_offset + cols
            ]

        mask = (
            (grid == 0)
            &
            (
                unknown_neighbours
                >= self.min_unknown_neighbours
            )
        )

        return [
            tuple(cell)
            for cell
            in np.argwhere(mask)
        ]

    def cluster_frontiers(self, frontier_cells):
        unvisited = set(
            frontier_cells
        )

        clusters = []

        while unvisited:
            start = unvisited.pop()

            queue = deque(
                [start]
            )

            cluster = [
                start
            ]

            while queue:
                row, col = queue.popleft()

                for (
                    row_offset,
                    col_offset
                ) in self.NEIGHBOURS:

                    neighbour = (
                        row + row_offset,
                        col + col_offset
                    )

                    if neighbour in unvisited:
                        unvisited.remove(
                            neighbour
                        )
                        queue.append(
                            neighbour
                        )
                        cluster.append(
                            neighbour
                        )

            clusters.append(
                cluster
            )

        return clusters

    @staticmethod
    def select_representatives(clusters):
        representatives = []

        for cluster in clusters:
            mean_row = (
                sum(
                    row
                    for row, _
                    in cluster
                )
                / len(cluster)
            )

            mean_col = (
                sum(
                    col
                    for _, col
                    in cluster
                )
                / len(cluster)
            )

            representatives.append(
                min(
                    cluster,
                    key=lambda cell: (
                        (
                            cell[0]
                            - mean_row
                        ) ** 2
                        +
                        (
                            cell[1]
                            - mean_col
                        ) ** 2
                    )
                )
            )

        return representatives

    @staticmethod
    def cell_to_world(
        row,
        col,
        info
    ):
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

        local_x = (
            col + 0.5
        ) * info.resolution

        local_y = (
            row + 0.5
        ) * info.resolution

        return (
            origin.position.x
            + local_x * math.cos(yaw)
            - local_y * math.sin(yaw),
            origin.position.y
            + local_x * math.sin(yaw)
            + local_y * math.cos(yaw)
        )

    def create_pose(
        self,
        row,
        col,
        info
    ):
        x, y = self.cell_to_world(
            row,
            col,
            info
        )

        pose = Pose()

        pose.position.x = x
        pose.position.y = y
        pose.orientation.w = 1.0

        return pose

    def publish_candidates(
        self,
        representatives,
        map_msg
    ):
        msg = PoseArray()

        msg.header = map_msg.header

        if not msg.header.frame_id:
            msg.header.frame_id = 'map'

        msg.poses = [
            self.create_pose(
                row,
                col,
                map_msg.info
            )
            for row, col
            in representatives
        ]

        self.candidate_publisher.publish(
            msg
        )

    def create_marker(
        self,
        cells,
        map_msg,
        namespace,
        size,
        red,
        green,
        blue
    ):
        marker = Marker()

        marker.header = map_msg.header

        if not marker.header.frame_id:
            marker.header.frame_id = 'map'

        marker.ns = namespace
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = size
        marker.scale.y = size
        marker.color.r = red
        marker.color.g = green
        marker.color.b = blue
        marker.color.a = 1.0

        for row, col in cells:
            x, y = self.cell_to_world(
                row,
                col,
                map_msg.info
            )

            point = Point()

            point.x = x
            point.y = y
            point.z = 0.05

            marker.points.append(
                point
            )

        return marker

    def publish_markers(
        self,
        frontier_cells,
        representatives,
        map_msg
    ):
        self.frontier_marker_publisher.publish(
            self.create_marker(
                frontier_cells,
                map_msg,
                'frontier_cells',
                0.06,
                1.0,
                1.0,
                0.0
            )
        )

        self.representative_marker_publisher.publish(
            self.create_marker(
                representatives,
                map_msg,
                'frontier_representatives',
                0.18,
                1.0,
                0.0,
                0.0
            )
        )


def main(args=None):
    rclpy.init(args=args)

    node = FrontierDetector()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
