import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import onnxruntime as ort

class onnx(Node):
    def __init__(self):
        super().__init__('lane_follower')

        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, '/zed/left/image_rect_color', self.image_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.image_pub = self.create_publisher(Image, '/lane_detection/image', 10)

        self.session = ort.InferenceSession('./culane_res18.onnx')  # Update with actual model path
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.image_width = None

    def preprocess_for_onnx(self, image):
        resized = cv2.resize(image, (256, 256))
        normalized = resized.astype(np.float32) / 255.0
        transposed = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(transposed, axis=0)

    def postprocess_output(self, output, original_shape):
        mask = output.squeeze() > 0.5
        mask = mask.astype(np.uint8) * 255
        mask = cv2.resize(mask, (original_shape[1], original_shape[0]))
        return mask

    def process_image(self, image):
        input_tensor = self.preprocess_for_onnx(image)
        onnx_output = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        mask = self.postprocess_output(onnx_output, image.shape[:2])

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = image.copy()
        lane_center = image.shape[1] // 2

        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                lane_center = cx
                cv2.circle(overlay, (cx, cy), 10, (0, 0, 255), -1)

        cv2.putText(overlay, f"Lane center: {lane_center}",
                    (lane_center + 10, image.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(overlay, (lane_center, 0), (lane_center, image.shape[0]), (0, 0, 255), 2)

        return overlay, lane_center

    def calculate_steering_angle(self, lane_center, image_width):
        error = lane_center - image_width // 2
        max_error = image_width // 2
        return error / max_error * 0.3

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV bridge error: {e}")
            return

        cropped = cv_image  # optional: use self.crop_bottom_half(cv_image)
        self.image_width = cropped.shape[1]
        processed_image, lane_center = self.process_image(cropped)
        angle = self.calculate_steering_angle(lane_center, cropped.shape[1])

        twist_msg = Twist()
        twist_msg.linear.x = 0.2  # constant speed
        twist_msg.angular.z = -angle  # adjust if necessary
        self.cmd_pub.publish(twist_msg)

        try:
            debug_img_msg = self.bridge.cv2_to_imgmsg(processed_image, encoding='bgr8')
            debug_img_msg.header.stamp = msg.header.stamp
            debug_img_msg.header.frame_id = "zed_left_camera"
            self.image_pub.publish(debug_img_msg)
        except Exception as e:
            self.get_logger().warn(f"Failed to publish debug image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = onnx()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


#https://drive.google.com/file/d/1Evy24fruMDpeU73Vlqm8_nSOHFWeDgTO/view