import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    parameter_file = os.path.join(
        get_package_share_directory(
            'exploration_manager'
        ),
        'config',
        'exploration_params.yaml'
    )

    exploration_manager = Node(
        package='exploration_manager',
        executable='exploration_manager',
        name='exploration_manager',
        output='screen',
        parameters=[
            parameter_file
        ]
    )

    return LaunchDescription([
        exploration_manager
    ])
