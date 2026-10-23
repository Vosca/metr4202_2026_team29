# metr4202 2026 Team 29
Collaboration space for Group 29 work on the Turtlebot for METR4202

## Setup
1. Clone repository into workspace:
```
cd ~/ws/src;
git clone https://github.com/Vosca/metr4202_2026_team29.git;
cd ..;
colcon build --packages-select exploration_manager frontier_detector;
```

## Testing
1. Launch Turtlebot3 in gazebo:
```
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```
2. Launch Nav2:
```
ros2 launch nav2_bringup navigation_launch.py use_sim_time:=True
```
3. Launch SLAM:
```
ros2 launch slam_toolbox online_async_launch.py slam_params_file:=slam.cfg.yaml
```
4. Launch Rvis:
```
ros2 launch nav2_bringup rviz_launch.py
```
5. Run explorer manager:
```
ros2 run exploration_manager exploration_manager
```
6. Run Frontier detector:
```
ros2 run frontier_detector frontier_detector
```
