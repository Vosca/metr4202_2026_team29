import math
from collections import deque

import numpy as np
import rclpy

from geometry_msgs.msg import Point, Pose, PoseArray
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from visualization_msgs.msg import Marker


class FrontierDetector(Node):

    def __init__(self):
        super().__init__('frontier_detector')

        # Minimum number of unknown neighbours required
        # for a free cell to be considered a frontier
        self.min_unknown_neighbours = int(
            self.declare_parameter(
                'min_unknown_neighbours',
                3
            ).value
        )

        # Minimum number of connected frontier cells
        # required for a cluster to be retained
        self.min_frontier_size = int(
            self.declare_parameter(
                'min_frontier_size',
                3
            ).value
        )

        # Make sure both parameters remain valid
        self.min_unknown_neighbours = max(
            1,
            self.min_unknown_neighbours
        )

        self.min_frontier_size = max(
            1,
            self.min_frontier_size
        )

        # Subscribe to the occupancy grid produced
        # by SLAM
        self.map_subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        # Publish one representative pose for each
        # valid frontier cluster
        self.candidate_publisher = self.create_publisher(
            PoseArray,
            '/frontier_candidates',
            10
        )

        # RViz marker showing every detected
        # frontier cell
        self.frontier_marker_publisher = (
            self.create_publisher(
                Marker,
                '/frontier_cells',
                10
            )
        )

        # RViz marker showing the representative
        # navigation candidates
        self.representative_marker_publisher = (
            self.create_publisher(
                Marker,
                '/frontier_representatives',
                10
            )
        )

        self.get_logger().info(
            'Frontier detector node started'
        )

    def map_callback(self, msg):

        # Ignore invalid maps, but still publish an
        # empty candidate list so downstream nodes
        # receive an explicit update
        if (
            msg.info.width == 0
            or msg.info.height == 0
        ):
            self.publish_candidates(
                [],
                msg
            )

            return

        # OccupancyGrid data arrives as a one-dimensional
        # list. Convert it to row-column form.
        map_array = np.array(
            msg.data,
            dtype=np.int16
        ).reshape(
            (
                msg.info.height,
                msg.info.width
            )
        )

        # Step 1:
        # Identify all known-free cells that border
        # sufficiently many unknown cells
        frontier_cells = self.find_frontier_cells(
            map_array
        )

        # Step 2:
        # Group neighbouring frontier cells into
        # connected frontier clusters
        frontier_clusters = self.cluster_frontiers(
            frontier_cells
        )

        # Step 3:
        # Remove tiny frontier clusters that are more
        # likely to be isolated map noise
        valid_clusters = [
            cluster
            for cluster in frontier_clusters
            if len(cluster) >= self.min_frontier_size
        ]

        # Step 4:
        # Select exactly one actual frontier cell
        # to represent each valid cluster
        representatives = (
            self.select_cluster_representatives(
                valid_clusters
            )
        )

        # Step 5:
        # Publish representatives for the
        # exploration manager
        self.publish_candidates(
            representatives,
            msg
        )

        # Publish RViz visualisation markers
        self.publish_frontier_markers(
            frontier_cells,
            representatives,
            msg
        )

        self.get_logger().info(
            f'Frontiers: '
            f'{len(frontier_cells)} cells, '
            f'{len(valid_clusters)} clusters, '
            f'{len(representatives)} candidates'
        )

    def find_frontier_cells(self, map_array):

        rows, cols = map_array.shape

        frontier_cells = []

        # Check every cell in the occupancy grid
        for row in range(rows):

            for col in range(cols):

                # A frontier candidate must itself be
                # known free space
                if map_array[row, col] != 0:
                    continue

                # Determine the local 3 x 3 neighbourhood,
                # clipped safely at map boundaries
                row_start = max(
                    0,
                    row - 1
                )

                row_end = min(
                    rows,
                    row + 2
                )

                col_start = max(
                    0,
                    col - 1
                )

                col_end = min(
                    cols,
                    col + 2
                )

                neighbourhood = map_array[
                    row_start:row_end,
                    col_start:col_end
                ]

                # Unknown OccupancyGrid cells have
                # the value -1
                unknown_count = np.count_nonzero(
                    neighbourhood == -1
                )

                # A known-free cell becomes a frontier
                # when it borders enough unknown cells
                if (
                    unknown_count
                    >= self.min_unknown_neighbours
                ):
                    frontier_cells.append(
                        (
                            row,
                            col
                        )
                    )

        return frontier_cells

    def cluster_frontiers(self, frontier_cells):

        # Use a set so membership checks are efficient
        unvisited = set(
            frontier_cells
        )

        clusters = []

        # 8-connected neighbourhood
        neighbour_offsets = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1)
        ]

        # Continue until every frontier cell has
        # been assigned to a connected cluster
        while unvisited:

            start_cell = unvisited.pop()

            queue = deque(
                [start_cell]
            )

            cluster = [
                start_cell
            ]

            while queue:

                row, col = queue.popleft()

                for (
                    row_offset,
                    col_offset
                ) in neighbour_offsets:

                    neighbour = (
                        row + row_offset,
                        col + col_offset
                    )

                    # If the neighbour is another
                    # unvisited frontier cell, it belongs
                    # to this connected component
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

    def select_cluster_representatives(
        self,
        frontier_clusters
    ):

        representatives = []

        for cluster in frontier_clusters:

            # Calculate the centroid of all cells in
            # this connected frontier cluster
            mean_row = sum(
                row
                for row, _ in cluster
            ) / len(cluster)

            mean_col = sum(
                col
                for _, col in cluster
            ) / len(cluster)

            # Do not use the centroid directly because
            # it may lie in occupied or unknown space.
            #
            # Instead select the ACTUAL frontier cell
            # closest to the cluster centroid.
            representative = min(
                cluster,
                key=lambda cell: (
                    (
                        cell[0] - mean_row
                    ) ** 2
                    +
                    (
                        cell[1] - mean_col
                    ) ** 2
                )
            )

            representatives.append(
                representative
            )

        return representatives

    def cell_to_world(
        self,
        row,
        col,
        map_info
    ):

        resolution = map_info.resolution

        origin = map_info.origin

        # Position of the centre of the grid cell
        # relative to the occupancy-grid origin
        local_x = (
            col + 0.5
        ) * resolution

        local_y = (
            row + 0.5
        ) * resolution

        orientation = origin.orientation

        # Extract the yaw angle from the map-origin
        # quaternion
        yaw = math.atan2(
            2.0 * (
                orientation.w * orientation.z
                + orientation.x * orientation.y
            ),
            1.0 - 2.0 * (
                orientation.y ** 2
                + orientation.z ** 2
            )
        )

        # Rotate and translate from grid coordinates
        # into world coordinates
        world_x = (
            origin.position.x
            + local_x * math.cos(yaw)
            - local_y * math.sin(yaw)
        )

        world_y = (
            origin.position.y
            + local_x * math.sin(yaw)
            + local_y * math.cos(yaw)
        )

        return world_x, world_y

    def create_candidate_pose(
        self,
        row,
        col,
        map_info
    ):

        world_x, world_y = self.cell_to_world(
            row,
            col,
            map_info
        )

        pose = Pose()

        pose.position.x = world_x
        pose.position.y = world_y
        pose.position.z = 0.0

        # Use a valid identity quaternion.
        #
        # This gives zero yaw for now.
        # A later refinement can orient the robot
        # toward the unknown side of the frontier.
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0
        pose.orientation.w = 1.0

        return pose

    def publish_candidates(
        self,
        representatives,
        map_msg
    ):

        candidate_msg = PoseArray()

        # The occupancy grid from slam_toolbox should
        # normally already use the map frame
        candidate_msg.header = map_msg.header

        # Protect against a map message with an
        # unexpectedly empty frame ID
        if not candidate_msg.header.frame_id:
            candidate_msg.header.frame_id = 'map'

        for row, col in representatives:

            pose = self.create_candidate_pose(
                row,
                col,
                map_msg.info
            )

            candidate_msg.poses.append(
                pose
            )

        # IMPORTANT:
        #
        # Publish on EVERY map update.
        #
        # If representatives is empty, this still publishes
        # a PoseArray with zero poses. The exploration manager
        # uses consecutive empty updates to determine when
        # exploration is complete.
        self.candidate_publisher.publish(
            candidate_msg
        )

    def create_marker(
        self,
        cells,
        map_msg,
        namespace,
        marker_id,
        point_size,
        red,
        green,
        blue
    ):

        marker = Marker()

        marker.header = map_msg.header

        if not marker.header.frame_id:
            marker.header.frame_id = 'map'

        marker.ns = namespace
        marker.id = marker_id

        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0

        marker.scale.x = point_size
        marker.scale.y = point_size

        marker.color.r = red
        marker.color.g = green
        marker.color.b = blue
        marker.color.a = 1.0

        for row, col in cells:

            world_x, world_y = self.cell_to_world(
                row,
                col,
                map_msg.info
            )

            point = Point()

            point.x = world_x
            point.y = world_y
            point.z = 0.05

            marker.points.append(
                point
            )

        return marker

    def publish_frontier_markers(
        self,
        frontier_cells,
        representatives,
        map_msg
    ):

        # Small points show every detected
        # frontier cell
        frontier_marker = self.create_marker(
            frontier_cells,
            map_msg,
            'frontier_cells',
            0,
            0.06,
            1.0,
            1.0,
            0.0
        )

        # Larger points show the candidate supplied
        # to the exploration manager for each cluster
        representative_marker = self.create_marker(
            representatives,
            map_msg,
            'frontier_representatives',
            0,
            0.18,
            1.0,
            0.0,
            0.0
        )

        self.frontier_marker_publisher.publish(
            frontier_marker
        )

        self.representative_marker_publisher.publish(
            representative_marker
        )


def main(args=None):

    rclpy.init(args=args)

    node = FrontierDetector()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
