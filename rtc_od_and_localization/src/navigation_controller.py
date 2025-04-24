#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist, Point
from rclpy.qos import QoSProfile
from nav_msgs.msg import Odometry
import math
import tf2_ros
import tf_transformations

from zed_interfaces.msg import ObjectsStamped  # (FOR USE ON REAL ROBOT)


class BicycleModelNavigationNode(Node):

    def __init__(self):
        super().__init__("bicycle_model_navigation_node")

        # Parameters for controlling the robot
        self.goal_tolerance = 0.8  # 30 cm tolerance
        self.kp = 0.05  # Proportional constant for the PID controller
        self.max_speed = 0.5  # Max linear speed in m/s
        self.max_angular_velocity = 0.2  # Max angular velocity in rad/s
        self.robot_length = (
            0.3  # Length of the robot (distance between front and rear axles)
        )

        # Subscribers
        self.goal_subscriber = self.create_subscription(
            PoseStamped, "/goal_pose", self.goal_pose_callback, QoSProfile(depth=10)
        )

        # self.obstacle_subscriber = self.create_subscription(
        #     ObjectsStamped,
        #     "/processed_obstacles",
        #     self.obstacle_callback,
        #     QoSProfile(depth=10),
        # )

        # Timer for regular replanning (e.g., every 0.2 seconds)
        self.timer = self.create_timer(0.01, self.timer_callback)

        # Publisher for movement commands (Twist)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", QoSProfile(depth=10))

        # Publisher for goal completed
        self.goal_completed_pub = self.create_publisher(
            PoseStamped, "/goal_reached", QoSProfile(depth=10)
        )

        # TF listener to get the robot’s pose (from RTAB-Map or other localization system)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Current robot pose and goal
        self.goal_reached = False
        self.current_pose = None
        self.goal_pose = None

        # Obstacle data (initially empty)
        self.obstacles = []

    def goal_pose_callback(self, msg: PoseStamped):
        self.get_logger().info(
            f"Received goal pose: {msg.pose.position.x}, {msg.pose.position.y}"
        )
        self.goal_pose = msg

    def timer_callback(self):
        new_pose = self.update_current_pose_from_tf()
        if new_pose:
            self.current_pose = new_pose

        if self.goal_pose is not None:
            self.navigate_to_goal()

    def obstacle_callback(self, msg: ObjectsStamped):
        # received every time we have a new obstacle
        # use this to update the obstacle list
        self.obstacles = self.process_obstacle_data(msg)

        if not self.obstacles:
            return

        self.update_current_pose_from_tf()

        if self.goal_pose is not None:
            self.navigate_to_goal()

    def process_obstacle_data(self, msg: ObjectsStamped):
        # Convert obstacle point to a list of obstacles with (x, y) coordinates
        obstacles = []
        for obj in msg.objects:
            obstacles.append((obj.position[0], obj.position[1]))
        return obstacles

    def update_current_pose_from_tf(self):
        try:
            # try zed camera center and just use odometry
            trans = self.tf_buffer.lookup_transform(
                "map", "base_link", rclpy.time.Time()
            )
            pose = PoseStamped()
            pose.pose.position.x = trans.transform.translation.x
            pose.pose.position.y = trans.transform.translation.y
            pose.pose.position.z = trans.transform.translation.z
            pose.pose.orientation = trans.transform.rotation
            return pose.pose
        except Exception as e:
            self.get_logger().warn(f"Failed to get transform: {str(e)}")
            return None

    def navigate_to_goal(self):
        if self.current_pose is None or self.goal_pose is None or self.goal_reached:
            return

        # Get current position and goal position
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        goal_x = self.goal_pose.pose.position.x
        goal_y = self.goal_pose.pose.position.y

        # Calculate the distance to the goal
        distance = math.sqrt((goal_x - current_x) ** 2 + (goal_y - current_y) ** 2)
        self.get_logger().info(f"Distance: {distance}")

        # If we're within tolerance, stop
        if distance < self.goal_tolerance:
            self.get_logger().info("Goal reached!")
            self.goal_completed_pub.publish(self.goal_pose)
            self._stop_robot()

            time.sleep(0.3)
            self.destroy_node()
            rclpy.shutdown()
            return

        # Calculate angle to the goal (heading)
        angle_to_goal = math.atan2(goal_y - current_y, goal_x - current_x)
        heading = self._get_robot_heading()  # Should return yaw angle

        # angle_diff = math.atan2(
        #     math.sin(angle_to_goal - heading), math.cos(angle_to_goal - heading)
        # )
        # angular_velocity = self.kp * 0  # You can tune this gain
        # self.get_logger().info(
        #     f"Angle to goal: {math.degrees(angle_diff):.2f}°, Angular velocity: {angular_velocity:.2f}"
        # )
        # for obstacle in self.obstacles:
        #     obs_x, obs_y = obstacle
        #     # Calculate the distance to the obstacle
        #     dist_to_obs = math.sqrt((obs_x - current_x) ** 2 + (obs_y - current_y) ** 2)

        #     # If the obstacle is within a critical distance (e.g., 1 meter), adjust steering
        #     if dist_to_obs < 1.0:  # 1 meter threshold
        #         # Calculate the angle to the obstacle
        #         angle_to_obs = math.atan2(obs_y - current_y, obs_x - current_x)

        #         # Calculate steering adjustment: If obstacle is near, steer away
        #         angular_velocity += (
        #             1.0 / dist_to_obs * (self._get_robot_heading() - angle_to_obs)
        #         )
        # angular_velocity = max(
        #     min(angular_velocity, self.max_angular_velocity), -self.max_angular_velocity
        # )

        # Calculate linear speed to goal (keep it proportional)
        linear_speed = min(self.kp * distance, self.max_speed)

        # Create a Twist message to control movement
        cmd_msg = Twist()
        cmd_msg.linear.x = linear_speed

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
        """Convert quaternion to Euler angles (roll, pitch, yaw)."""
        return tf_transformations.euler_from_quaternion(
            [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        )

    def _stop_robot(self):
        """Send a stop command to the robot."""
        stop_msg = Twist()
        self.cmd_pub.publish(stop_msg)


def main(args=None):
    rclpy.init(args=args)
    node = BicycleModelNavigationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
