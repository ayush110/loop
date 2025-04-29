# Autolap Launch File

This launch file starts three core nodes from the `autolap` package:

- **`person_follower`**: Follows detected people using perception inputs.
- **`lane_follower`**: Follows lane markings for path alignment.
- **`navigation_controller`**: Makes navigation decisions and publishes control commands.

## 📦 Prerequisites

- ROS 2 (tested with **Humble** or newer)
- The `autolap` package is built and sourced in your workspace
- All dependencies are installed
- Executables (`person_follower`, `lane_follower`, `navigation_controller`) are correctly registered in `setup.py` and `package.xml`

## 🛠️ Setup

### 1. Build your Workspace

```bash
cd ~/your_ros2_ws
colcon build
source install/setup.bash
```

### 2. Run the Launch File

```bash 
ros2 launch autolap autolap_launch.py
```


### Nodes Launched
| Node Name               | Executable             | Description                                        |
|-------------------------|------------------------|----------------------------------------------------|
| `PersonFollower`       | `person_follower`      | Tracks and follows detected people                 |
| `LaneFollowerNode`     | `lane_follower`        | Follows detected lane lines                        |
| `NavigationController` | `navigation_controller`| Combines perception inputs and sends control commands|
