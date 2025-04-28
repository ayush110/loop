import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from std_msgs.msg import Float32  # <-- NEW

class LaneFollowerNode(Node):
    def __init__(self):
        super().__init__('lane_follower')

        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, '/zed/zed_node/left/image_rect_color', self.image_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.image_pub = self.create_publisher(Image, '/lane_detection/image', 10)
        self.depth_sub = self.create_subscription(Image, '/zed/zed_node/depth/depth_registered', self.depth_image_callback, 10)
        self.latest_depth_image = None

        self.error_pub = self.create_publisher(Float32, '/lane_center_error', 10)  # <-- NEW


        self.get_logger().info(f'Started Node')

    def enhance_brightness(image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(1)
        enhanced_lab = cv2.merge((cl,a,b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def fit_lane_line(self, lines, image_height):
        if not lines:
            return None

        x_coords = []
        y_coords = []
        for x1, y1, x2, y2 in lines:
            x_coords += [x1,x2]
            y_coords += [y1,y2]

        if len(x_coords) < 2:
            return None

        poly = np.polyfit(y_coords, x_coords, 1)
        m, b = poly

        y1 = image_height
        y2 = int(image_height * 0.5)

        x1 = int(m*y1+b)
        x2 = int(m*y2+b)

        return (x1,y1,x2,y2)


    def crop_bottom_half(self, image, ratio=0.6):
        height = image.shape[0]
        start_row = int(height * (1 - ratio))
        return image[start_row:, :]

    def isolate_white(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds for white color in HSV
        # Adjust the values based on environment
        lower_white = np.array([0, 0, 160])
        upper_white = np.array([180, 50, 255])
        return cv2.inRange(hsv, lower_white, upper_white)

    def is_horizontal_or_vertical(self, x1, y1, x2, y2, threshold=10):
        angle = abs(math.degrees(math.atan2(y2-y1, x2-x1)))
        return angle < threshold or angle > 180

    def process_image(self, image):
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

        left_x = None
        right_x = None

        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if self.is_horizontal_or_vertical(x1, y1, x2, y2):
                        continue
                    slope = (y2 - y1) / (x2 - x1 + 1e-6)
                    if slope < -0.1:
                        left_lines.append((x1, y1, x2, y2))
                    elif slope > 0.1:
                        right_lines.append((x1, y1, x2, y2))

            def average_x(lines):
                if not lines:
                    return None
                return np.mean([x1 + x2 for x1, _, x2, _ in lines]) / 2

            left_x = average_x(left_lines)
            right_x = average_x(right_lines)


            left_fit = self.fit_lane_line(left_lines, height)
            right_fit = self.fit_lane_line(right_lines, height)

            if left_fit:
                cv2.line(output_image, (left_fit[0], left_fit[1]), (left_fit[2], left_fit[3]), (0, 255, 0), 3)
                left_x = (left_fit[0] + left_fit[2]) / 2
            if right_fit:
                cv2.line(output_image, (right_fit[0], right_fit[1]), (right_fit[2], right_fit[3]), (0, 255, 0), 3)
                left_x = (right_fit[0] + right_fit[2]) / 2
            
            if left_x is not None and right_x is not None:
                lane_center = (left_x + right_x) / 2

            if left_x is not None:
                cv2.line(output_image, (int(left_x), 0), (int(left_x), height), (255, 0, 0), 2)
            if right_x is not None:
                cv2.line(output_image, (int(right_x), 0), (int(right_x), height), (255, 0, 0), 2)

        # cv2.line(output_image, (int(lane_center), 0), (int(lane_center), height), (0, 0, 255), 2)
        # cv2.putText(output_image, f"Lane center: {int(lane_center)}",
        #             (int(lane_center) + 10, height - 50),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        # cv2.circle(output_image, (int(lane_center), height - 30), 10, (255, 0, 0), -1)

        return output_image, lane_center, left_x, right_x

    def calculate_steering_angle(self, lane_center, image_width):
        error = lane_center - image_width // 2
        max_error = image_width // 2
        normalized_error = error / max_error
        return normalized_error  # return normalized error now!

    def get_depth_at_point(self, x, y):
        if self.latest_depth_image is None:
            return None
        h, w = self.latest_depth_image.shape
        x = np.clip(int(x), 0, w-1)
        y = np.clip(int(y), 0, h-1)
        depth = float(self.latest_depth_image[y,x])
        self.get_logger().info(f"Latest Depth Image: {depth}")
        return depth

    def depth_image_callback(self, msg):
        try:
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().error(f"Depth image conversion error: {e}")    

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV bridge error: {e}")
            return
        cv_image = cv2.convertScaleAbs(cv_image, alpha=1.0, beta=40)
        cropped = self.crop_bottom_half(cv_image)
        processed_image, lane_center, left_x, right_x = self.process_image(cropped)


        angle = self.calculate_steering_angle(lane_center, cropped.shape[1])

        
        # Manual steering angle adjustment based on depth and testing
        try:
            height = cropped.shape[0]
            left_depth = self.get_depth_at_point(left_x, height-30) if left_x is not None else None
            right_depth = self.get_depth_at_point(right_x, height-30) if right_x is not None else None

            if left_depth and not right_depth:
                angle -= 0.2
            elif not left_depth and right_depth: 
                angle += 0.2
            elif left_depth and right_depth:
                if left_depth < right_depth:
                    angle -= 0.15
                elif left_depth > right_depth:
                    angle += 0.15

            #self.get_logger().info(f"Left Depth: {left_depth}")
            #self.get_logger().info(f"Right Depth: {right_depth}")
            
        except Exception as e:
            self.get_logger().error(f"Error {e}")

        # Publish the lane center error
        error_msg = Float32()
        error_msg.data = angle
        self.error_pub.publish(error_msg)

        self.get_logger().info(f"Lane center normalized error: {angle:.2f}")

        try:
            debug_img_msg = self.bridge.cv2_to_imgmsg(processed_image, encoding='bgr8')
            debug_img_msg.header.stamp = msg.header.stamp
            debug_img_msg.header.frame_id = "zed_left_camera"
            self.image_pub.publish(debug_img_msg)
        except Exception as e:
            self.get_logger().warn(f"Failed to publish debug image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = LaneFollowerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
