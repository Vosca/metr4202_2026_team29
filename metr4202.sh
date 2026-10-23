#!/bin/bash

colcon build --packages-select exploration_manager frontier_detector marker_detection

set -e

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v gnome-terminal >/dev/null 2>&1; then
    echo "gnome-terminal is required to launch each node in a separate terminal." >&2
    exit 1
fi

gnome-terminal --title="Man I Love Chips" -- bash -lc '
    ros2 launch turtlebot3_navigation2 navigation2.launch.py use_sim_time:=False params_file:=params_live.yaml slam:=True
    status=$?
    echo "exited with status $status."
    read -r -p "Press Enter to close this terminal..."
' bash "$script_dir &

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
ros2 run marker_detection marker_detection
status=$?
echo "Marker_detection exited with status $status."
read -r -p "Press Enter to close this terminal..."
' bash "$script_dir" &

wait
