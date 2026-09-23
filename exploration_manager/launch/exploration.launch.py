import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(
        get_package_share_directory('exploration_manager'),
        'config',
        'exploration_params.yaml'
    )

    return LaunchDescription([
        Node(
            package='frontier_detector',
            executable='frontier_detector',
            name='frontier_detector',
            output='screen',
            parameters=[params]
        ),
        Node(
            package='exploration_manager',
            executable='exploration_manager',
            name='exploration_manager',
            output='screen',
            parameters=[params]
        )
    ])
