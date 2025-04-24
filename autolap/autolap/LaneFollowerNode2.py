import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class LaneFollowerNode2(Node):
    def __init__(self):
        super().__init__('lane_follower')

        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, '/zed/zed_node/left/image_rect_color', self.rgb_image_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/zed/zed_node/depth/depth_registered', self.depth_image_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.image_pub = self.create_publisher(Image, '/lane_detection/image', 10)

        self.latest_depth_image = None

        self.get_logger().info('Started Node')

    def depth_image_callback(self, msg):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.latest_depth_image = depth_image
        except Exception as e:
            self.get_logger().error(f"Depth CV bridge error: {e}")

    def rgb_image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"RGB CV bridge error: {e}")
            return

        cropped = self.crop_bottom_half(cv_image)
        y_offset = cv_image.shape[0] - cropped.shape[0]

        processed_image, lane_center, used_side = self.process_image(cropped, self.latest_depth_image, y_offset)

        angle = self.calculate_steering_angle(lane_center, cropped.shape[1])

        if used_side == 'left':
            angle = abs(angle)  # steer right
        elif used_side == 'right':
            angle = -abs(angle)  # steer left

        self.get_logger().info(f'Steering Angle: {angle:.2f} (from {used_side} lane)')

        twist_msg = Twist()
        twist_msg.linear.x = 0.2
        twist_msg.angular.z = angle
        self.cmd_pub.publish(twist_msg)

        try:
            debug_img_msg = self.bridge.cv2_to_imgmsg(processed_image, encoding='bgr8')
            debug_img_msg.header.stamp = msg.header.stamp
            debug_img_msg.header.frame_id = "zed_left_camera"
            self.image_pub.publish(debug_img_msg)
        except Exception as e:
            self.get_logger().warn(f"Failed to publish debug image: {e}")

    def crop_bottom_half(self, image, ratio=0.5):
        height = image.shape[0]
        start_row = int(height * (1 - ratio))
        return image[start_row:, :]

    def isolate_white(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 160])
        upper_white = np.array([180, 80, 255])
        return cv2.inRange(hsv, lower_white, upper_white)

    def get_depth_at_point(self, depth_image, x, y, y_offset=0):
        y_original = y + y_offset
        if depth_image is None or y_original >= depth_image.shape[0] or x >= depth_image.shape[1]:
            return None
        depth = depth_image[y_original, x]
        return float(depth) if not np.isnan(depth) and depth > 0.1 else None

    def process_image(self, image, depth_image=None, y_offset=0):
        white_mask = self.isolate_white(image)
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.dilate(cleaned, kernel, iterations=1)

        edges = cv2.Canny(cleaned, 30, 100)
        height, width = edges.shape

        mask = np.zeros_like(edges)
        polygon = np.array([[
            (int(width * 0.1), height),
            (int(width * 0.9), height),
            (int(width * 0.9), 0),
            (int(width * 0.1), 0),
        ]], dtype=np.int32)
        cv2.fillPoly(mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=30, minLineLength=30, maxLineGap=300)

        output_image = np.copy(image)
        lane_center = width // 2

        left_lines = []
        right_lines = []
        left_depths = []
        right_depths = []

        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    slope = (y2 - y1) / (x2 - x1 + 1e-6)
                    if abs(slope) < 0.3 or abs(slope) > 2.0:
                        continue

                    depth1 = self.get_depth_at_point(depth_image, x1, y1, y_offset)
                    depth2 = self.get_depth_at_point(depth_image, x2, y2, y_offset)
                    avg_depth = (depth1 + depth2) / 2 if depth1 and depth2 else None
                    if avg_depth is None:
                        continue

                    if slope < -0.1:
                        left_lines.append((x1, y1, x2, y2))
                        left_depths.append(avg_depth)
                    elif slope > 0.1:
                        right_lines.append((x1, y1, x2, y2))
                        right_depths.append(avg_depth)

            left_lane_center = self.calculate_weighted_center(left_lines, left_depths)
            right_lane_center = self.calculate_weighted_center(right_lines, right_depths)

            used_side = None
            if left_lane_center is not None and right_lane_center is not None:
                left_avg_depth = np.mean(left_depths)
                right_avg_depth = np.mean(right_depths)
                if left_avg_depth < right_avg_depth:
                    lane_center = left_lane_center
                    used_side = 'left'
                else:
                    lane_center = right_lane_center
                    used_side = 'right'
            elif left_lane_center is not None:
                lane_center = left_lane_center
                used_side = 'left'
            elif right_lane_center is not None:
                lane_center = right_lane_center
                used_side = 'right'

            for x1, y1, x2, y2 in left_lines:
                cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            for x1, y1, x2, y2 in right_lines:
                cv2.line(output_image, (x1, y1), (x2, y2), (0, 0, 255), 3)

            cv2.line(output_image, (int(lane_center), 0), (int(lane_center), height), (0, 0, 255), 2)
            cv2.putText(output_image, f"Lane center: {int(lane_center)}", (int(lane_center) + 10, height - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(output_image, (int(lane_center), height - 30), 10, (255, 0, 0), -1)

            return output_image, lane_center, used_side

        return output_image, lane_center, None

    def calculate_weighted_center(self, lines, depths):
        if not lines or not depths:
            return None
        weighted_x_sum = 0
        total_weight = 0
        for (x1, y1, x2, y2), depth in zip(lines, depths):
            weight = 1 / depth
            weighted_x_sum += (x1 + x2) / 2 * weight
            total_weight += weight
        return weighted_x_sum / total_weight if total_weight > 0 else None

    def calculate_steering_angle(self, lane_center, image_width):
        error = lane_center - image_width // 2
        max_error = image_width // 2
        return error / max_error

def main(args=None):
    rclpy.init(args=args)
    node = LaneFollowerNode2()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
