import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import time

class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = None

    def update_params(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def compute(self, error):
        current_time = time.time()
        dt = current_time - self.last_time if self.last_time else 0.1
        self.last_time = current_time

        self.integral += error * dt
        derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        self.last_error = error

        return self.kp * error + self.ki * self.integral + self.kd * derivative

class Dash(Node):
    def __init__(self):
        super().__init__('straight_line_pid')

        # Declare tunable parameters
        self.declare_parameter('kp', 1.5)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 0.2)

        self.pid = PID(
            self.get_parameter('kp').value,
            self.get_parameter('ki').value,
            self.get_parameter('kd').value
        )

        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(
            Odometry,
            '/zed/zed_node/odom',
            self.odom_callback,
            10
        )

        self.timer = self.create_timer(0.1, self.control_loop)

        self.current_y = 0.0
        self.add_on_set_parameters_callback(self.param_callback)

    def odom_callback(self, msg):
        self.current_y = msg.pose.pose.position.y

    def control_loop(self):
        error = -self.current_y
        correction = self.pid.compute(error)

        cmd = Twist()
        cmd.linear.x = 0.2
        cmd.angular.z = correction

        self.publisher.publish(cmd)

    def param_callback(self, params):
        for param in params:
            if param.name in ['kp', 'ki', 'kd']:
                # Update PID params live
                kp = self.get_parameter('kp').value
                ki = self.get_parameter('ki').value
                kd = self.get_parameter('kd').value
                self.pid.update_params(kp, ki, kd)
                self.get_logger().info(f"Updated PID: kp={kp}, ki={ki}, kd={kd}")
        return rclpy.parameter.ParameterEventHandler.Result(successful=True)

def main(args=None):
    rclpy.init(args=args)
    node = Dash()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


"""ros2 run autolap Dash \
  --ros-args \
  -p kp:=2.0 \
  -p ki:=0.05 \
  -p kd:=0.3

  rqt_plot /zed/zed_node/odom/pose/pose/position/y
"""