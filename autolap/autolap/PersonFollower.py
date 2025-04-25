import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from zed_interfaces.msg import ObjectsStamped
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import math

class PersonFollower(Node):

    def __init__(self):
        super().__init__('person_follower')
        self.subscription = self.create_subscription(
            ObjectsStamped,
            '/zed/zed_node/obj_det/objects',
            self.listener_callback,
            10)
        self.velocity_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self.target_distance = 1.0  # Desired following distance in meters
        self.max_speed = 0.3        # Max linear speed
        self.kp_linear = 0.5        # Proportional gain for linear speed
        self.kp_angular = 0.5       # Proportional gain for angular speed

        # Marker publisher
        self.marker_publisher = self.create_publisher(MarkerArray, '/person_marker', 10)

    def listener_callback(self, msg):
        for obj in msg.objects:
            if obj.label == "Person":
                x, y, z = obj.position[0], obj.position[1], obj.position[2]
                distance = math.sqrt(x**2 + y**2 + z**2)

                error = distance - self.target_distance

                cmd = Twist()
                # Proportional control for forward speed
                cmd.linear.x = 0.5

                # Turn toward the person if they are off to the side
                cmd.angular.z = self.kp_angular * y  # x is left-right offset

                self.velocity_publisher.publish(cmd)

                

                self.get_logger().info(
                    f"Following person: dist={distance:.2f}, speed={cmd.linear.x:.2f}, turn={cmd.angular.z:.2f}"
                )
                break  # Follow the first detected person

def main(args=None):
    rclpy.init(args=args)
    node = PersonFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
