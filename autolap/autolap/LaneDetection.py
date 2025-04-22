import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_node')
        
        # Subscribe to the camera topic
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',  # Adjust based on your actual camera topic
            self.image_callback,
            10
        )
        
        # Publisher for the processed image
        self.publisher = self.create_publisher(Image, '/lane_detection/image', 10)
        
        # Publisher for the Twist message (steering control)
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # CvBridge to convert between ROS Image messages and OpenCV images
        self.bridge = CvBridge()

    def image_callback(self, msg):
        # Convert the incoming ROS image message to a cv2 image
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # Apply perspective transformation to adjust for the low camera angle
        transformed_image = self.perspective_transform(cv_image)
        
        # Process the transformed image for lane detection
        processed_image, lane_center = self.process_image(transformed_image)
        
        # Calculate the steering command based on the lane center
        steering_angle = self.calculate_steering_angle(lane_center, transformed_image.shape[1])
        
        # Publish the steering command as a Twist message
        self.publish_steering_command(steering_angle)
        
        # Convert the processed image back to ROS format
        msg_out = self.bridge.cv2_to_imgmsg(processed_image, "bgr8")
        
        # Publish the processed image
        self.publisher.publish(msg_out)

    def perspective_transform(self, image):
        # Same perspective transformation function as before
        height, width = image.shape[:2]

        src_pts = np.float32([
            [width * 0.1, height],
            [width * 0.9, height],
            [width * 0.4, height * 0.6],
            [width * 0.6, height * 0.6]
        ])

        dst_pts = np.float32([
            [width * 0.2, height],
            [width * 0.8, height],
            [width * 0.2, 0],
            [width * 0.8, 0]
        ])

        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_image = cv2.warpPerspective(image, matrix, (width, height))

        return warped_image

    def process_image(self, image):
        # Convert image to HSV and apply color masking (for red and white lanes)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 25, 255])

        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        mask = cv2.bitwise_or(mask_red1, mask_red2)
        mask = cv2.bitwise_or(mask, mask_white)

        blurred = cv2.GaussianBlur(mask, (5, 5), 0)

        edges = cv2.Canny(blurred, 50, 150)

        # Region of Interest (ROI) to focus on the lane area
        height, width = edges.shape
        mask_roi = np.zeros_like(edges)
        
        polygon = np.array([[(100, height), (width-100, height), (width//2, height//2)]], dtype=np.int32)
        cv2.fillPoly(mask_roi, polygon, 255)
        
        masked_edges = cv2.bitwise_and(edges, mask_roi)

        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=200)

        # Draw lane lines on the image
        output_image = np.copy(image)
        lane_center = width // 2  # Default center if no lanes are found

        if lines is not None:
            # Find the average position of the lanes to calculate the center
            left_lines = []
            right_lines = []
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if x1 < width // 2 and x2 < width // 2:
                        left_lines.append((x1, y1, x2, y2))
                    elif x1 > width // 2 and x2 > width // 2:
                        right_lines.append((x1, y1, x2, y2))
            
            # Compute lane center (average of the left and right lanes)
            if left_lines and right_lines:
                left_x = np.mean([x1 for x1, y1, x2, y2 in left_lines])
                right_x = np.mean([x1 for x1, y1, x2, y2 in right_lines])
                lane_center = (left_x + right_x) / 2

            # Draw the lane lines
            for line in left_lines + right_lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

        return output_image, lane_center

    def calculate_steering_angle(self, lane_center, image_width):
        # Calculate how far the lane center is from the image center
        error = lane_center - image_width // 2
        
        # Steering angle: Simple proportional controller
        steering_angle = -error * 0.005  # You may need to fine-tune the factor (0.005)
        return steering_angle

    def publish_steering_command(self, steering_angle):
        # Create a Twist message
        twist = Twist()
        
        # Set the angular velocity (steering)
        twist.angular.z = steering_angle
        
        # Optionally set linear velocity if you want to drive forward
        twist.linear.x = 0.5  # Constant speed (you can adjust this)
        
        # Publish the Twist message
        self.cmd_vel_publisher.publish(twist)

def main(args=None):
    # Initialize the ROS 2 system
    rclpy.init(args=args)

    # Create and spin the lane detection node
    lane_detection_node = LaneDetectionNode()
    rclpy.spin(lane_detection_node)

    # Cleanup after the node is shutdown
    lane_detection_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
