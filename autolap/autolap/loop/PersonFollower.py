import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32  # <-- NEW
from zed_interfaces.msg import ObjectsStamped
import math

class PersonFollower(Node):
    def __init__(self):
        super().__init__('person_follower')
        self.subscription = self.create_subscription(
            ObjectsStamped,
            '/zed/zed_node/obj_det/objects',
            self.listener_callback,
            10)
        #self.velocity_publisher = self.create_publisher(Twist, '/person_cmd_vel', 10)
        self.error_publisher = self.create_publisher(Float32, '/person_lateral_error', 10)  # <-- NEW

        self.target_distance = 1.0
        self.max_speed = 0.3
        self.kp_linear = 0.5
        self.kp_angular = 0.5

    def listener_callback(self, msg):
        for obj in msg.objects:
            if obj.label == "Person":
                x, y, z = obj.position[0], obj.position[1], obj.position[2]
                distance = math.sqrt(x**2 + y**2 + z**2)

                # Lateral error = left/right position (y)
                lateral_error = y

                # Publish lateral error
                error_msg = Float32()
                error_msg.data = lateral_error
                self.error_publisher.publish(error_msg)

                # (optional) still publish a cmd_vel if you want for debug
                # cmd = Twist()
                # cmd.linear.x = 0.5
                # cmd.angular.z = self.kp_angular * lateral_error
                # self.velocity_publisher.publish(cmd)

                self.get_logger().info(f"Person lateral error: {lateral_error:.2f}")
                break

def main(args=None):
    rclpy.init(args=args)
    node = PersonFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
