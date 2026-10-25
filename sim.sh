#!/bin/bash

colcon build --packages-select exploration_manager frontier_detector marker_detection

set -e

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v gnome-terminal >/dev/null 2>&1; then
    echo "gnome-terminal is required to launch each node in a separate terminal." >&2
    exit 1
fi

gnome-terminal --title="nav2" -- bash -lc '
    cd "$1" || exit
    source install/setup.bash || exit
    ros2 launch nav2_bringup navigation_launch.py use_sim_time:=True
    status=$?
    echo "nav2 exited with status $status."
    read -r -p "Press Enter to close this terminal..."
' bash "$script_dir" &

gnome-terminal --title="SLAM" -- bash -lc '
    cd "$1" || exit
    source install/setup.bash || exit
    ros2 launch slam_toolbox online_async_launch.py slam_params_file:=slam.cfg.yaml
    status=$?
    echo "SLAM exited with status $status."
    read -r -p "Press Enter to close this terminal..."
' bash "$script_dir" &

gnome-terminal --title="RViz" -- bash -lc '
    cd "$1" || exit
    source install/setup.bash || exit
    ros2 launch nav2_bringup rviz_launch.py
    status=$?
    echo "RViz exited with status $status."
    read -r -p "Press Enter to close this terminal..."
' bash "$script_dir" &

gnome-terminal --title="Exploration manager" -- bash -lc '
    cd "$1" || exit
    source install/setup.bash || exit
    ros2 run exploration_manager exploration_manager
    status=$?
    echo "Exploration manager exited with status $status."
    read -r -p "Press Enter to close this terminal..."
' bash "$script_dir" &

gnome-terminal --title="Frontier detector" -- bash -lc '
    cd "$1" || exit
    source install/setup.bash || exit
    ros2 run frontier_detector frontier_detector
    status=$?
    echo "Frontier detector exited with status $status."
    read -r -p "Press Enter to close this terminal..."
' bash "$script_dir" &

gnome-terminal --title="Marker detection" -- bash -lc '
cd "$1" || exit
source install/setup.bash || exit
ros2 run marker_detection marker_detection --ros-args -p focal_length_scalar:=1.0
status=$?
echo "Marker_detection exited with status $status."
read -r -p "Press Enter to close this terminal..."
' bash "$script_dir" &

echo "Please start gazebo world in new terminal"

wait