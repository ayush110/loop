import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
import time


class SteeringController:
    def __init__(self, kp=0.5, ki=0.0, kd=0.1, ema_alpha=0.2):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.ema_alpha = ema_alpha  # Smoothing factor for EMA

        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_output = 0.0  # EMA state

    def compute_command(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0

        raw_output = self.kp * error + self.ki * self.integral + self.kd * derivative

        # Apply EMA filter
        self.filtered_output = (
            self.ema_alpha * raw_output + (1 - self.ema_alpha) * self.filtered_output
        )

        self.prev_error = error
        return self.filtered_output


class NavigationController(Node):
    def __init__(self):
        super().__init__("navigation_controller")

        self.person_error = 0.0
        self.lane_error = 0.0

        self.person_sub = self.create_subscription(
            Float32, "/person_lateral_error", self.person_callback, 10
        )
        self.lane_sub = self.create_subscription(
            Float32, "/lane_center_error", self.lane_callback, 10
        )
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.timer = self.create_timer(0.1, self.control_loop)  # 10Hz

        # Weighting for combining person and lane errors
        self.kp_person = 0.3
        self.kp_lane = 0.7

        # Create PID controller for steering
        self.pid = SteeringController(kp=1.21, ki=0.02, kd=0.3)

        # Keep track of time between control loop calls
        self.last_time = time.time()

    def person_callback(self, msg):
        self.person_error = msg.data

    def lane_callback(self, msg):
        self.lane_error = msg.data

    def control_loop(self):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        twist = Twist()
        twist.linear.x = 0.2  # constant forward speed

        # Combine errors
        combined_error = (
            self.kp_person * self.person_error + self.kp_lane * self.lane_error
        )

        # Pass combined error into PID controller
        twist.angular.z = self.pid.compute_command(combined_error, dt)

        self.cmd_pub.publish(twist)

        self.get_logger().info(
            f"Person error: {self.person_error:.2f}, Lane error: {self.lane_error:.2f}, Combined error: {combined_error:.2f}, Angular z: {twist.angular.z:.2f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = NavigationController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
