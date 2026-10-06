import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    use_sim_time = LaunchConfiguration('use_sim_time')

    parameter_file = os.path.join(
        get_package_share_directory(
            'exploration_manager'
        ),
        'config',
        'exploration_params.yaml'
    )

    frontier_detector = Node(
        package='frontier_detector',
        executable='frontier_detector',
        name='frontier_detector',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    exploration_manager = Node(
        package='exploration_manager',
        executable='exploration_manager',
        name='exploration_manager',
        output='screen',
        parameters=[
            parameter_file,
            {'use_sim_time': use_sim_time}
        ]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use the simulation clock'
        ),
        frontier_detector,
        exploration_manager
    ])
