# RTC Dash Node Launch Guide

This README provides instructions on how to launch the `dash.py` node in your ROS 2 environment.

## Prerequisites

Before launching the `dash.py` node, ensure the following:

1. **ROS 2 Setup**: ROS 2 (e.g., Foxy,) is installed on your system.
2. **Workspace Build**: Your ROS 2 workspace should be built and sourced correctly. If you haven't done that yet, follow the steps below:

   ```bash
   cd ~/ros2_ws  # Navigate to your workspace
   colcon build  # Build the workspace
   source install/setup.bash  # Source the workspace

## Launch the file

*Customize arguments based on your environment
```bash
ros2 launch racetrack_challenge dash.py kp:=0.7 ki:=0.2 kd:=0.1
```


# RTC Object Detection Launch Guide

Launch the file with camera, rtab map, and database path for the rtab map.
```bash
ros2 launch racetrack_challenge launch_rtc_challenge.py \
  launch_camera:=true \
  launch_rtab:=true \
  database_path:="/home/nvidia/ros2_ws/src/loop/racetrack_challenge/racetrack_map2.db"
```
Optionally launch rviz using ```rviz2```