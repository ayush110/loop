import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class LaneDetectionOpenCV(Node):
    def __init__(self):
        super().__init__('lane_detection_opencv')

        # ROS2 parameters
        self.declare_parameter('throttle_gain', 0.3)
        self.declare_parameter('input_shape', [120, 160])
        self.declare_parameter('camera_topic', '/zed/zed_node/rgb_raw/image_raw_color')

        # Get parameters
        self.throttle = self.get_parameter('throttle_gain').value
        self.input_shape = self.get_parameter('input_shape').value
        camera_topic = self.get_parameter('camera_topic').value

        # Bridge for ROS <-> OpenCV
        self.bridge = CvBridge()

        # Create subscribers/publishers
        self.subscription = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            10
        )
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
       
        self.get_logger().info("Lane Detection Node initialized.")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            height, width, _ = cv_image.shape

            # Preprocess image
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)

            # Region of Interest
            mask = np.zeros_like(edges)
            roi_vertices = np.array([
                [(0, height), (width, height), (width, int(height * 0.6)), (0, int(height * 0.6))]
            ])
            cv2.fillPoly(mask, roi_vertices, 255)
            roi = cv2.bitwise_and(edges, mask)

            # Hough Line Detection
            lines = cv2.HoughLinesP(roi, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=100)

            left_x = []
            right_x = []

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    slope = (y2 - y1) / (x2 - x1 + 1e-6)
                    if abs(slope) < 0.5:
                        continue
                    if slope < 0:
                        left_x.extend([x1, x2])
                    else:
                        right_x.extend([x1, x2])

            # Estimate lane center
            if left_x and right_x:
                left_avg = np.mean(left_x)
                right_avg = np.mean(right_x)
                lane_center_x = (left_avg + right_avg) / 2
            elif left_x:
                lane_center_x = np.mean(left_x) + 100
            elif right_x:
                lane_center_x = np.mean(right_x) - 100
            else:
                lane_center_x = width / 2  # fallback

            # Compute vectors
            bottom_center = np.array([width / 2, height])
            lookahead = np.array([lane_center_x, height * 0.6])
            heading_vector = np.array([0, -1])  # pointing up
            lane_vector = lookahead - bottom_center

            # Compute steering angle
            angle = self.get_angle_between_vectors(heading_vector, lane_vector)

            # Create and publish Twist
            twist_msg = Twist()
            twist_msg.linear.x = self.throttle
            twist_msg.angular.z = float(angle)
            self.cmd_pub.publish(twist_msg)

            self.get_logger().info(f"Steering angle: {angle:.3f}")

        except Exception as e:
            self.get_logger().error(f"Image processing error: {e}")

    def get_angle_between_vectors(self, v1, v2):
        unit_v1 = v1 / (np.linalg.norm(v1) + 1e-6)
        unit_v2 = v2 / (np.linalg.norm(v2) + 1e-6)
        angle = np.arctan2(unit_v2[1], unit_v2[0]) - np.arctan2(unit_v1[1], unit_v1[0])
        return angle

def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionOpenCV()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
