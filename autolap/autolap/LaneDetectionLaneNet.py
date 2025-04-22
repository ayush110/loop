import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import tensorflow as tf
import os
import sys

# === Add LaneNet repo to path ===
LANENET_PATH = '/home/mobilerobotics-008/ros2_ws/src/lanenet-lane-detection'  # ← Change this
sys.path.append(os.path.join(LANENET_PATH, 'model'))
sys.path.append(os.path.join(LANENET_PATH, 'config'))

from lanenet_merge_model import LaneNet
from config import global_config

CFG = global_config.cfg


class LaneNetDetector:
    def __init__(self):
        self.input_tensor = tf.placeholder(tf.float32, [1, 256, 512, 3], name='input_tensor')
        self.net = LaneNet(phase='test', cfg=CFG)
        self.binary_seg_ret, self.instance_seg_ret = self.net.inference(input_tensor=self.input_tensor, name='lanenet_model')

        self.sess = tf.Session()
        saver = tf.train.Saver()

        checkpoint_path = '/home/mobilerobotics-008/ros2_ws/src/autolap/BiseNetV2_LaneNet_Tusimple_Model_Weights/tusimple_lanenet.ckpt.data-00000-of-00001'  # ← Change this
        saver.restore(self.sess, checkpoint_path)
        print('[LaneNet] Model loaded.')

    def detect(self, frame):
        img = cv2.resize(frame, (512, 256))
        img = img / 127.5 - 1.0
        img = np.expand_dims(img, 0)

        binary_seg, _ = self.sess.run([self.binary_seg_ret, self.instance_seg_ret],
                                      feed_dict={self.input_tensor: img})
        binary_img = (np.squeeze(binary_seg) * 255).astype(np.uint8)
        binary_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        return cv2.resize(binary_img, (frame.shape[1], frame.shape[0]))


class LaneNetROSNode(Node):
    def __init__(self):
        super().__init__('lanenet_ros_node')
        self.bridge = CvBridge()
        self.detector = LaneNetDetector()

        self.subscription = self.create_subscription(
            Image,
            '/zed/zed_node/rgb/image_rect_color',
            self.image_callback,
            10)
        
        self.publisher = self.create_publisher(Image, '/lanenet/output_image', 10)
        self.get_logger().info('LaneNet node started.')

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            output = self.detector.detect(frame)
            output_msg = self.bridge.cv2_to_imgmsg(output, encoding='bgr8')
            self.publisher.publish(output_msg)
        except Exception as e:
            self.get_logger().error(f'Error in image callback: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = LaneNetROSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
