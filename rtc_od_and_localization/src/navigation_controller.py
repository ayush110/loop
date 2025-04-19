#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist, Point
from rclpy.qos import QoSProfile
from nav_msgs.msg import Odometry
import math
import tf2_ros


class GoalNavigationNode(Node):

    def __init__(self):
        super().__init__('goal_navigation_node')

        # Parameters for controlling the robot
        self.goal_tolerance = 0.3  # 30 cm tolerance
        self.kp = 0.5  # Proportional constant for PID or simple controller
        self.max_speed = 0.5  # Max linear speed in m/s
        self.max_angular_velocity = 1.0  # Max angular velocity in rad/s

        # Subscribers
        self.goal_subscriber = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_pose_callback,
            QoSProfile(depth=10)
        )

        # Odometry subscriber (using RTAB-Map or any localization)
        self.odom_subscriber = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            QoSProfile(depth=10)
        )

        self.obstacle_subscriber = self.create_subscription(
            Point,
            '/obstacle_data',
            self.obstacle_callback,
            QoSProfile(depth=10)
        )

        # Publisher for movement commands (Twist)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', QoSProfile(depth=10))

        # TF listener to get the robot’s pose (from RTAB-Map or other localization system)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Current robot pose and goal
        self.current_pose = None
        self.goal_pose = None

        # Obstacle data (initially empty)
        self.obstacles = []

    def goal_pose_callback(self, msg: PoseStamped):
        self.get_logger().info(f"Received goal pose: {msg.pose.position.x}, {msg.pose.position.y}")
        self.goal_pose = msg

    def odom_callback(self, msg: Odometry):
        self.current_pose = msg.pose.pose
        if self.goal_pose is not None:
            self.navigate_to_goal()

    def obstacle_callback(self, msg: Point):
        # received every time we have a new obstacle
        self.obstacles = self.process_obstacle_data(msg)


    def navigate_to_goal(self):
        if self.current_pose is None or self.goal_pose is None:
            return

        # Get current position and goal position
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        goal_x = self.goal_pose.pose.position.x
        goal_y = self.goal_pose.pose.position.y

        # Calculate the distance to the goal
        distance = math.sqrt((goal_x - current_x)**2 + (goal_y - current_y)**2)

        # If we're within tolerance, stop
        if distance < self.goal_tolerance:
            self.get_logger().info("Goal reached!")
            self._stop_robot()
            return

        # Calculate angle to the goal (heading)
        angle_to_goal = math.atan2(goal_y - current_y, goal_x - current_x)

        # Start with basic steering (no obstacle avoidance)
        angular_velocity = 2 * (angle_to_goal - self._get_robot_heading())

        # Now, calculate obstacle avoidance adjustments:
        for obstacle in self.obstacles:
            obs_x, obs_y = obstacle
            # Calculate the distance to the obstacle
            dist_to_obs = math.sqrt((obs_x - current_x)**2 + (obs_y - current_y)**2)

            # If the obstacle is within a critical distance (e.g., 1 meter), adjust steering
            if dist_to_obs < 1.0:  # 1 meter threshold
                # Calculate the angle to the obstacle
                angle_to_obs = math.atan2(obs_y - current_y, obs_x - current_x)

                # Calculate steering adjustment: If obstacle is near, steer away
                angular_velocity += 1.0 / dist_to_obs * (self._get_robot_heading() - angle_to_obs)

        # Apply some damping to prevent oscillations
        angular_velocity = max(min(angular_velocity, self.max_angular_velocity), -self.max_angular_velocity)

        # Calculate linear speed to goal (keep it proportional)
        linear_speed = min(self.kp * distance, self.max_speed)

        # Create a Twist message to control movement
        cmd_msg = Twist()
        cmd_msg.linear.x = linear_speed
        cmd_msg.angular.z = angular_velocity

        # Publish the command
        self.cmd_pub.publish(cmd_msg)

    def _get_robot_heading(self):
        """Get the robot’s heading (yaw) from the current pose (using quaternion to euler conversion)."""
        if self.current_pose is None:
            return 0.0
        orientation = self.current_pose.orientation
        euler = self._quaternion_to_euler(orientation)
        return euler[2]  # yaw angle

    def _quaternion_to_euler(self, quaternion):
        pass

    def _stop_robot(self):
        """Send a stop command to the robot."""
        stop_msg = Twist()
        self.cmd_pub.publish(stop_msg)


def main(args=None):
    rclpy.init(args=args)
    node = GoalNavigationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
