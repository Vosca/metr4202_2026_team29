# metr4202 2026 Team 29
Collaboration space for Group 29 work on the Turtlebot for METR4202

## Setup
1. Clone repository into workspace:
```
cd ~/ws;
git clone https://github.com/Vosca/metr4202_2026_team29.git;
```

## Simulation:
1. Launch Turtlebot3 in gazebo:
```
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```
2. Run sim
```
chmod +x sim.sh
./sim.sh
```

## Real world:
1. Run metr4202.sh
```
chmod +x metr4202.sh
./metr4202.sh
```

Note: Must be on Ubuntu-22.04 to use .sh files, does not work on WSL.
