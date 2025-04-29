import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import tensorflow as tf
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import time

class EnhancedMLModel(Node):
    def __init__(self):
        super().__init__('enhanced_ml_model')
        
        # Improved model loading
        self.model = self.load_tensorrt_model('/path/to/model.engine')
        
        # Enhanced parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('throttle_gain', 0.3),
                ('steering_gain', 1.0),
                ('input_shape', [256, 512]),  # Common lane detection resolution
                ('camera_topic', '/zed/zed_node/rgb_raw/image_raw_color'),
                ('debug_mode', False)
            ]
        )
        
        # Image preprocessing config
        self.crop_ratio = 0.6  # Crop upper portion of image
        self.mean = [0.485, 0.456, 0.406]  # ImageNet normalization
        self.std = [0.229, 0.224, 0.225]
        
        # Control parameters
        self.steering_filter = ExponentialFilter(alpha=0.3)
        self.throttle_filter = ExponentialFilter(alpha=0.5)
        self.min_throttle = 0.1  # Maintain minimum speed
        
        # Initialize components
        self.bridge = CvBridge()
        self.last_inference_time = time.time()
        
        # QoS configuration
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # ROS2 setup
        self.subscription = self.create_subscription(
            Image,
            self.get_parameter('camera_topic').value,
            self.image_callback,
            qos_profile
        )
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Debug tools
        if self.get_parameter('debug_mode').value:
            self.debug_pub = self.create_publisher(Image, '/debug_image', 10)

    def load_tensorrt_model(self, model_path):
        """Proper TensorRT model loading"""
        # Use TensorRT's Python API or tf.experimental.tensorrt.ExperimentalConverter
        # Implementation depends on your model conversion method
        raise NotImplementedError("Implement proper TensorRT loading")

    def preprocess_image(self, cv_image):
        """Enhanced preprocessing pipeline"""
        # 1. Crop sky region
        h, w = cv_image.shape[:2]
        cv_image = cv_image[int(h*(1-self.crop_ratio)):h, :]
        
        # 2. Resize with aspect ratio preservation
        target_h, target_w = self.get_parameter('input_shape').value
        scale = min(target_h/h, target_w/w)
        cv_image = cv2.resize(cv_image, None, fx=scale, fy=scale)
        
        # 3. Normalize using ImageNet stats
        cv_image = cv_image.astype(np.float32) / 255.0
        cv_image = (cv_image - self.mean) / self.std
        
        # 4. Add batch dimension
        return np.expand_dims(cv_image, axis=0)

    def postprocess_output(self, raw_output):
        """Convert model output to control signals"""
        steering = raw_output['steering_output'][0][0]
        throttle = raw_output['throttle_output'][0][0]
        
        # Apply gains and limits
        steering = np.clip(steering * self.get_parameter('steering_gain').value, -1.0, 1.0)
        throttle = np.clip(throttle * self.get_parameter('throttle_gain').value, 
                          self.min_throttle, 1.0)
        
        return steering, throttle

    def image_callback(self, msg):
        try:
            # Check frame rate
            current_time = time.time()
            dt = current_time - self.last_inference_time
            self.last_inference_time = current_time
            
            # Convert image
            cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # Preprocess
            processed = self.preprocess_image(cv_img)
            
            # Inference
            output = self.model(processed)
            
            # Postprocess
            steering, throttle = self.postprocess_output(output)
            
            # Apply temporal filtering
            steering = self.steering_filter.update(steering)
            throttle = self.throttle_filter.update(throttle)
            
            # Publish commands
            twist_msg = Twist()
            twist_msg.linear.x = float(throttle)
            twist_msg.angular.z = float(steering)
            self.cmd_pub.publish(twist_msg)
            
            # Debugging
            if self.get_parameter('debug_mode').value:
                self.publish_debug_image(cv_img, steering, throttle)
                
            self.get_logger().info(
                f"Steering: {steering:.2f} | Throttle: {throttle:.2f} | DT: {dt*1000:.1f}ms",
                throttle_dd=True
            )

        except Exception as e:
            self.get_logger().error(f"Processing error: {str(e)}")

    def publish_debug_image(self, cv_img, steering, throttle):
        """Create annotated debug image"""
        debug_img = cv_img.copy()
        
        # Add steering indicator
        cv2.putText(debug_img, f"Steering: {steering:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add throttle indicator
        cv2.putText(debug_img, f"Throttle: {throttle:.2f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Publish debug image
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug_img, 'bgr8'))

class ExponentialFilter:
    """Smoothing filter for control signals"""
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.value = 0.0
        
    def update(self, new_value):
        self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

def main(args=None):
    rclpy.init(args=args)
    node = EnhancedMLModel()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
