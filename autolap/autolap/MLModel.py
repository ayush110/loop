import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import tensorflow as tf
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class MLModel(Node):
    def __init__(self):
        super().__init__('ml_model')
       
        # Load TensorRT-optimized model
        self.model = tf.saved_model.load('/home/nvidia/ros2_ws/src/loop/autolap/model.trt')
        self.predict = self.model.signatures['serving_default']
       
        # Initialize CV bridge
        self.bridge = CvBridge()
       
        # ROS2 parameters
        self.declare_parameter('throttle_gain', 0.3)
        self.declare_parameter('input_shape', [120, 160])
        self.declare_parameter('camera_topic', '/zed/zed_node/rgb_raw/image_raw_color')

        self.prev_steering = 0.0
        self.prev_throttle = 0.0
        self.alpha = 0.2  # Smoothing factor (0 = more smoothing, 1 = no smoothing)
       
        # Get parameters
        self.throttle = self.get_parameter('throttle_gain').value
        self.input_shape = self.get_parameter('input_shape').value
        camera_topic = self.get_parameter('camera_topic').value
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        # Create subscribers/publishers
        self.subscription = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            qos_profile
        )
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
       
        self.get_logger().info("Node initialized. Waiting for camera images...")

    def preprocess_image(self, cv_image):
        # Resize and normalize
        img = cv2.resize(cv_image, tuple(self.input_shape[::-1]))
        img = img.astype(np.float32) / 255.0
        return img

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV

            self.get_logger().info(f"Steering")

            cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
           
            # Preprocess
            processed = self.preprocess_image(cv_img)
           
            # Convert to tensor
            input_tensor = tf.convert_to_tensor(
                np.expand_dims(processed, 0),
                dtype=tf.float32
            )
           
            # TensorRT inference
            output = self.predict(input_tensor)

            # Extract model output
            # Assumes output is like {'output_0': [[steering, throttle]]}
            throttle = float(output['n_outputs1'].numpy()[0][0])
            steering = float(output['n_outputs0'].numpy()[0][0])

            # Smooth using exponential moving average
            # steering = self.alpha * steering + (1 - self.alpha) * self.prev_steering
            # throttle = self.alpha * throttle + (1 - self.alpha) * self.prev_throttle

            # # Update previous values
            # self.prev_steering = steering
            # self.prev_throttle = throttle

            self.get_logger().info(f"Predicted -> Steering: {steering:.4f}, Throttle: {throttle:.4f}")

            # Publish Twist message
            twist_msg = Twist()
            twist_msg.linear.x = throttle
            twist_msg.angular.z = steering
            self.cmd_pub.publish(twist_msg)
            
        except Exception as e:
            self.get_logger().error(f"Processing error: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = MLModel()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

