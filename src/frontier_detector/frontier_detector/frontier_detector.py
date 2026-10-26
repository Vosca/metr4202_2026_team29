import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose, PoseArray


class FrontierDetector(Node):

    def __init__(self):
        """Initialise the frontier detection parameters, ROS 2 subscribers and publishers."""

        super().__init__('frontier_detector')

        # Minimum number of unknown neighbours for cell to be a frontier
        self.min_unknown_neighbours = self.declare_parameter(
            'min_unknown_neighbours', 3
        ).value

        # Minimum number of cells in a frontier cluster to be kept
        self.min_frontier_size = self.declare_parameter(
            'min_frontier_size', 4
        ).value

        # Radius in metres to count nearby frontier cells for scoring
        self.scoring_radius = self.declare_parameter(
            'scoring_radius', 0.25
        ).value

        # Minimum distance in metres between selected representatives
        self.min_goal_spacing = self.declare_parameter(
            'min_goal_spacing', 0.3
        ).value

        # Minimum distance in metres from occupied cells for a candidate
        self.min_obstacle_clearance = self.declare_parameter(
            'min_obstacle_clearance', 0.3
        ).value

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

        # Publish the selected representatives as markers
        self.representative_publisher = self.create_publisher(
            Marker,
            '/frontier_representatives',
            10
        )

        # Send the same representatives to the exploration manager
        self.candidate_publisher = self.create_publisher(
            PoseArray,
            '/frontier_candidates',
            10
        )

    def map_callback(self, msg):
        """Process incoming map updates to detect, filter and publish frontiers."""

        # Convert list to 2D array
        map_array = np.array(msg.data).reshape((msg.info.height, msg.info.width))

        # Find the frontier cells in the map
        raw_frontiers = self.find_frontiers(map_array)

        # Keep frontiers in large enough clusters
        filtered_frontiers = self.filter_small_frontiers(
            raw_frontiers,
            self.min_frontier_size
        )

        # Select few representative frontier cells
        representatives = self.select_representatives(
            filtered_frontiers,
            map_array,
            msg.info.resolution
        )

        # Log the number of frontiers found, filtered and selected
        self.get_logger().info(
            f'Found {len(raw_frontiers)} frontier cells, '
            f'kept {len(filtered_frontiers)} after filtering, '
            f'selected {len(representatives)} representatives'
        )

        # Publish the frontiers and representatives as markers
        self.publish_frontiers(filtered_frontiers, msg)
        self.publish_representatives(representatives, msg)

    def find_frontiers(self, map_array):
        """Find free cells with enough unknown neighbours to be considered frontiers."""

        frontiers = []

        # Get the dimensions of the map
        rows, cols = map_array.shape

        # Iterate through the map, including the border cells
        for row in range(rows):
            for col in range(cols):

                # If the cell is not free, skip it
                if map_array[row, col] != 0:
                    continue

                # Keep the neighbourhood within the map bounds
                row_start = max(0, row - 1)
                row_end = min(rows, row + 2)
                col_start = max(0, col - 1)
                col_end = min(cols, col + 2)

                # Get surrounding 3x3 cells, clipped at the map edges
                neighbours = map_array[
                    row_start:row_end,
                    col_start:col_end
                ]

                # Count unknown neighbouring cells
                unknown_count = np.sum(neighbours == -1)

                # Only keep strong frontier cells
                if unknown_count >= self.min_unknown_neighbours:
                    frontiers.append((row, col))

        return frontiers

    def filter_small_frontiers(self, frontiers, min_size):
        """Group connected frontier cells and remove groups smaller than the minimum size."""

        unvisited_cells = set(frontiers)
        kept = set()

        # Go through the unvisited frontier cells
        while unvisited_cells:

            start = unvisited_cells.pop()
            component = [start]
            cells_to_check = [start]

            while cells_to_check:

                # get the cell coordinates
                row, col = cells_to_check.pop()

                # Check the 8 neighbouring cells
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):

                        if dr == 0 and dc == 0:
                            continue

                        # Check the neighbouring cells
                        neighbour = (row + dr, col + dc)

                        # Add connected frontier cells to the group and mark them as visited
                        if neighbour in unvisited_cells:
                            unvisited_cells.remove(neighbour)
                            component.append(neighbour)
                            cells_to_check.append(neighbour)

            if len(component) >= min_size:
                kept.update(component)

        # Keep the cells in their original order
        result = []

        for cell in frontiers:
            if cell in kept:
                result.append(cell)

        return result

    def has_obstacle_clearance(self, row, col, map_array, resolution):
        """Check if a candidate cell has sufficient clearance from occupied cells."""

        # Convert the clearance distance from metres to grid cells
        clearance_cells = math.ceil(self.min_obstacle_clearance / resolution)

        rows, cols = map_array.shape

        # Check surrounding cells within the clearance distance
        for check_row in range(
            max(0, row - clearance_cells),
            min(rows, row + clearance_cells + 1)
        ):
            for check_col in range(
                max(0, col - clearance_cells),
                min(cols, col + clearance_cells + 1)
            ):

                row_distance = (check_row - row) * resolution
                col_distance = (check_col - col) * resolution
                distance = math.hypot(row_distance, col_distance)

                # Reject candidates too close to occupied cells
                if (distance <= self.min_obstacle_clearance
                    and map_array[check_row, check_col] > 50
                ):
                    return False

        return True

    def select_representatives(self, frontiers, map_array, resolution):
        """Score frontier cells and select spaced representatives with obstacle clearance."""

        # Stop if there are no frontier cells
        if not frontiers:
            return []

        # Store the cell coordinates and scores
        cells = np.array(frontiers, dtype=float)
        scores = np.zeros(len(frontiers), dtype=int)

        # Count nearby frontier cells for each candidate
        for index in range(len(frontiers)):
            row, col = frontiers[index]

            # Convert the distances from grid cells to metres
            row_distances = (cells[:, 0] - row) * resolution
            col_distances = (cells[:, 1] - col) * resolution

            distances = np.hypot(row_distances, col_distances)

            is_close = distances <= self.scoring_radius
            scores[index] = np.count_nonzero(is_close)

        # Try the highest scores first, keeping the original order for ties
        sorted_indices = np.argsort(-scores, kind='stable')

        representatives = []

        # Check each candidate against the selected points
        for index in sorted_indices:

            row, col = frontiers[index]

            # Skip candidates that are too close to occupied cells
            if not self.has_obstacle_clearance(row, col, map_array, resolution):
                continue

            too_close = False

            # Check the distance to each selected representative
            for selected_row, selected_col in representatives:

                row_distance = (row - selected_row) * resolution
                col_distance = (col - selected_col) * resolution

                distance = math.hypot(row_distance, col_distance)

                if distance < self.min_goal_spacing:
                    too_close = True
                    break

            # Keep the candidate if there is enough space
            if not too_close:
                representatives.append(
                    frontiers[index]
                )

        return representatives

    def publish_frontiers(self, frontiers, msg):
        """Publish the filtered frontier cells as an RViz marker."""

        marker = self.create_point_marker(frontiers, msg)

        # Set the frontier namespace, size and colour
        marker.ns = 'frontiers'
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        self.frontier_publisher.publish(marker)

    def publish_representatives(self, representatives, msg):
        """Publish representative markers and candidate poses for the exploration manager."""

        marker = self.create_point_marker(representatives, msg)

        # Set the representative namespace, size and colour
        marker.ns = 'frontier_representatives'
        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        self.representative_publisher.publish(marker)

        # Create a PoseArray to send the representatives to the exploration manager
        candidates = PoseArray()
        candidates.header = marker.header

        # Create a Pose for each representative and add it to the PoseArray
        for point in marker.points:

            pose = Pose()
            pose.position.x = point.x
            pose.position.y = point.y
            pose.orientation.w = 1.0

            candidates.poses.append(pose)

        # Publish empty updates too, so the manager can detect completion
        self.candidate_publisher.publish(candidates)

    def create_point_marker(self, cells, msg):
        """Convert map cell coordinates to world coordinates and create an RViz marker."""

        marker = Marker()

        # Set the header information
        marker.header.frame_id = msg.header.frame_id
        marker.header.stamp = (self.get_clock().now().to_msg())

        marker.id = 0

        # Set the type and action for the marker
        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        # Make the points fully visible
        marker.color.a = 1.0

        # Get the rotation of the map origin
        orientation = msg.info.origin.orientation

        yaw = math.atan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y**2 + orientation.z**2))

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        # Create a point for each cell
        for row, col in cells:

            point = Point()

            # Get the cell centre relative to the map origin
            local_x = (col + 0.5) * msg.info.resolution

            local_y = (row + 0.5) * msg.info.resolution

            # Convert the row and column indices to world coordinates
            point.x = (msg.info.origin.position.x
                + local_x * cos_yaw - local_y * sin_yaw)

            point.y = (msg.info.origin.position.y
                + local_x * sin_yaw + local_y * cos_yaw)

            # Small z value to make the points visible above the map
            point.z = 0.05

            # Add the point to the marker
            marker.points.append(point)

        return marker


def main(args=None):
    """Initialise ROS 2, run the frontier detector node and shut down when finished."""

    rclpy.init(args=args)

    node = FrontierDetector()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

# Generative AI Declaration:
# ChatGPT, OpenAI Codex, and GitHub Copilot were used to assist with
# code development, debugging, and documentation.