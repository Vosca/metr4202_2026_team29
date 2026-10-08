# metr4202-g29
Collaboration space for Group 29 work on the Turtlebot for METR4202

## Setup
1. Clone repository into workspace:
```
cd ~/ws/src;
git clone https://github.com/Vosca/metr4202_2026_team29.git;
cd ..;
colcon build;
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
ros2 launch slam_toolbox online_async_launch.py
```
4. Launch Rvis:
```
ros2 launch nav2_bringup rviz_launch.py
```
5. Run explorer manager:
```
ros2 run exploration_manager exploration_manager
```
6. Run Frontier detector??:
```
ros2 run frontier_detector frontier_detector
```

## Task Breakdown:
 - Creating measurable criteria
 - Ensure ROS Gazebo environment is working
 - Finding frontiers of the map
 - Exploration strategy, which frontiers do we navigate to?
 - Navigation algorithm
 - Identifying the markers location
 - Testing

## Team Responsibilities:
 - Michael - ArUco Markers
 - Robin - Testing
 - Mitch - Finding map frontiers
 - Thomas - Exploration and Navigation

### Useful existing solutions
Frontier Detection: https://github.com/AniArka/Autonomous-Explorer-and-Mapper-ros2-nav2
ArUco Markers : https://github.com/JMU-ROBOTICS-VIVA/ros2_aruco/tree/main
