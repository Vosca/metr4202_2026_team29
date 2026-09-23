import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    # Locate the installed exploration_manager package
    package_share = get_package_share_directory(
        'exploration_manager'
    )

    # Build the path to the shared parameter file
    parameter_file = os.path.join(
        package_share,
        'config',
        'exploration_params.yaml'
    )

    # Frontier detector node
    frontier_detector_node = Node(
        package='frontier_detector',
        executable='frontier_detector',
        name='frontier_detector',
        output='screen',
        parameters=[
            parameter_file
        ]
    )

    # Exploration manager node
    exploration_manager_node = Node(
        package='exploration_manager',
        executable='exploration_manager',
        name='exploration_manager',
        output='screen',
        parameters=[
            parameter_file
        ]
    )

    return LaunchDescription(
        [
            frontier_detector_node,
            exploration_manager_node
        ]
    )
