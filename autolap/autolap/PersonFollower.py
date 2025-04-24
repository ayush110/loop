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
        self.kp_angular = 1.0       # Proportional gain for angular speed

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
                cmd.linear.x = max(min(self.kp_linear * error, self.max_speed), -self.max_speed)

                # Turn toward the person if they are off to the side
                cmd.angular.z = -self.kp_angular * x  # x is left-right offset

                self.velocity_publisher.publish(cmd)

                # Create a marker for the person's position
                marker = Marker()
                marker.header.frame_id = "zed_camera_frame"  # Use the correct frame ID for your camera
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "person_tracking"
                marker.id = 0  # ID for the marker
                marker.type = Marker.SPHERE  # Use a sphere to represent the person
                marker.action = Marker.ADD
                marker.pose.position.x = x
                marker.pose.position.y = y
                marker.pose.position.z = z
                marker.scale.x = 0.1  # Adjust size of the sphere
                marker.scale.y = 0.1
                marker.scale.z = 0.1
                marker.color.a = 1.0  # Fully opaque
                marker.color.r = 1.0  # Red color
                marker.color.g = 0.0
                marker.color.b = 0.0

                # Publish the marker
                marker_array = MarkerArray()
                marker_array.markers.append(marker)
                self.marker_publisher.publish(marker_array)

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
