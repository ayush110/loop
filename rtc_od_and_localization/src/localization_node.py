#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarian-ceStamped
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped


class Localization(Node):
    def __init__(self):
        super().__init__("localization_to_tf")

        # Subscribe to RTAB-Map localization pose
        self.localization_subscriber = self.create_subscription(
            PoseWithCovarianceStamped,
            "/rtabmap/localization_pose",
            self.localization_callback,
            10,
        )

        # Publish the pose as PoseStamped (you can use it in Detector node)
        self.pose_pub = self.create_publisher(PoseStamped, "/localization_pose", 10)

    def localization_callback(self, msg: PoseWithCovarianceStamped):
        """
        Convert localization pose to PoseStamped and publish it.
        """
        pose = msg.pose.pose
        translation = pose.position
        rotation = pose.orientation

        pose_stamped = PoseStamped()
        pose_stamped.header = Header()
        pose_stamped.header.stamp = msg.header.stamp
        pose_stamped.header.frame_id = "map"

        pose_stamped.pose.position = translation
        pose_stamped.pose.orientation = rotation

        # Publish the pose
        self.pose_pub.publish(pose_stamped)
        self.get_logger().info(f"Published PoseStamped: {pose_stamped}")


def main(args=None):
    rclpy.init(args=args)
    node = Localization()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
