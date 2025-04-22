#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped


class Localization(Node):

    def __init__(self):
        super().__init__("localization_to_tf")

        # Create a TransformBroadcaster to broadcast the transforms
        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscribe to RTAB-Map localization pose
        self.localization_subscriber = self.create_subscription(
            PoseWithCovarianceStamped,
            "/rtabmap/localization_pose",
            self.localization_callback,
            10,
        )

    def localization_callback(self, msg: PoseWithCovarianceStamped):
        # Get the pose from the message
        pose = msg.pose.pose
        translation = pose.position
        rotation = pose.orientation

        # Create a TransformStamped message
        t = TransformStamped()

        # Set header for the transform
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"  # The parent frame is 'map'
        t.child_frame_id = "base_link"  # Set this to the camera frame

        # Set the translation and rotation from the pose message
        t.transform.translation.x = translation.x
        t.transform.translation.y = translation.y
        t.transform.translation.z = translation.z
        t.transform.rotation = rotation

        # Broadcast the transform
        self.tf_broadcaster.sendTransform(t)
        self.get_logger().info(
            f"Broadcasting TF: {t.child_frame_id} -> {t.header.frame_id}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = Localization()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
