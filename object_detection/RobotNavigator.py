#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist, PointStamped
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
import tf2_geometry_msgs
from geometry_msgs.msg import TransformStamped
from zed_interfaces.msg import ObjectsStamped
import csv
import math
from sklearn.cluster import DBSCAN
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray  # Import Marker and MarkerArray

class RobotNavigator(Node):
    def __init__(self):
        super().__init__("robot_navigator")

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscriptions
        self.localization_subscriber = self.create_subscription(
            PoseWithCovarianceStamped,
            "/rtabmap/localization_pose",
            self.localization_callback,
            10,
        )
        self.object_detection_subscriber = self.create_subscription(
            ObjectsStamped,
            "/zed/zed_node/obj_det/objects",
            self.object_detection_callback,
            10,
        )

        # Publisher
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # RViz Marker publisher
        self.marker_publisher = self.create_publisher(MarkerArray, '/detected_objects_markers', 10)

        # Movement variables
        self.goal_distance = 2.5  # meters
        self.current_distance = 0.0
        self.start_position = None

        # Object detection storage: (label, x, y, z, confidence)
        self.detected_objects = []

        # CSV file
        self.output_file = '/tmp/detected_objects.csv'

        # Timer
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.moving = True

    def localization_callback(self, msg: PoseWithCovarianceStamped):
        """Localization updates"""
        pose = msg.pose.pose
        translation = pose.position
        rotation = pose.orientation

        # Broadcast TF
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "base_link"
        t.transform.translation.x = translation.x
        t.transform.translation.y = translation.y
        t.transform.translation.z = translation.z
        t.transform.rotation = rotation

        self.tf_broadcaster.sendTransform(t)

        # Save start position
        if self.start_position is None:
            self.start_position = (translation.x, translation.y)
            self.get_logger().info(f"Start position set at ({translation.x:.2f}, {translation.y:.2f})")

        # Update current distance traveled
        dx = translation.x - self.start_position[0]
        dy = translation.y - self.start_position[1]
        self.current_distance = math.sqrt(dx**2 + dy**2)

    def object_detection_callback(self, msg: ObjectsStamped):
        """Collect and transform detected objects to map frame and publish markers."""
        marker_array = MarkerArray()

        for obj in msg.objects:
            if obj.label in ["Person", "Vehicle"]:
                try:
                    # Create a PointStamped for the object's position
                    object_point = PointStamped()
                    object_point.header.frame_id = msg.header.frame_id
                    object_point.header.stamp = msg.header.stamp
                    object_point.point.x = obj.position.x
                    object_point.point.y = obj.position.y
                    object_point.point.z = obj.position.z

                    # Transform the point to 'map' frame
                    transform = self.tf_buffer.lookup_transform(
                        "map",
                        object_point.header.frame_id,
                        rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=0.5)
                    )
                    map_point = tf2_geometry_msgs.do_transform_point(object_point, transform)

                    confidence = obj.confidence / 100.0
                    self.detected_objects.append((obj.label, map_point.point.x, map_point.point.y, map_point.point.z, confidence))

                    self.get_logger().info(
                        f"Detected {obj.label} at MAP frame ({map_point.point.x:.2f}, {map_point.point.y:.2f}, {map_point.point.z:.2f}) conf={confidence:.2f}"
                    )

                    # Create a marker for the detected object
                    marker = Marker()
                    marker.header.frame_id = "map"
                    marker.header.stamp = self.get_clock().now().to_msg()
                    marker.ns = "detected_objects"
                    marker.id = len(marker_array.markers)  # Unique ID for each marker
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD
                    marker.pose.position.x = map_point.point.x
                    marker.pose.position.y = map_point.point.y
                    marker.pose.position.z = map_point.point.z
                    marker.scale.x = marker.scale.y = marker.scale.z = 0.2  # Size of the sphere
                    marker.color.r = 1.0  # Red color
                    marker.color.a = 1.0  # Alpha (opacity)
                    marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()

                    marker_array.markers.append(marker)

                except Exception as e:
                    self.get_logger().warn(f"Transform failed for object: {e}")

        # Publish markers for the detected objects
        if marker_array.markers:
            self.marker_publisher.publish(marker_array)
            self.get_logger().info(f"Published {len(marker_array.markers)} markers to RViz.")

    def timer_callback(self):
        """Timer event to move or stop."""
        if not self.moving:
            return

        # Calculate the remaining distance to the goal
        remaining_distance = self.goal_distance - self.current_distance
        
        # Proportional speed control: scale speed based on remaining distance
        max_speed = 0.2  # Maximum speed (in meters per second)
        k = 1.0  # Proportional constant, you can adjust this value to control how quickly the robot slows down

        # Calculate speed based on remaining distance
        speed = k * remaining_distance
        # Limit speed to a maximum value to avoid overshooting
        speed = min(speed, max_speed)

        if remaining_distance > 0.3:  # If not within stopping range
            # Move forward
            cmd = Twist()
            cmd.linear.x = speed
            self.cmd_vel_publisher.publish(cmd)
        else:
            # Goal reached or near enough
            self.moving = False
            self.get_logger().info(f"Reached goal at {self.current_distance:.2f} meters. Stopping robot.")

            cmd = Twist()
            cmd.linear.x = 0.0
            self.cmd_vel_publisher.publish(cmd)

            self.process_and_save_objects()


    def process_and_save_objects(self):
        """Cluster objects and save top 3 based on confidence."""
        if len(self.detected_objects) == 0:
            self.get_logger().warn("No objects detected to cluster.")
            return

        # Prepare data for clustering
        positions = np.array([[x, y, z] for (_, x, y, z, _) in self.detected_objects])

        # DBSCAN clustering
        clustering = DBSCAN(eps=1.0, min_samples=1).fit(positions)
        labels = clustering.labels_

        # Group points by cluster
        clusters = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = {"points": [], "confidences": [], "labels": []}
            clusters[label]["points"].append(positions[idx])
            clusters[label]["confidences"].append(self.detected_objects[idx][4])
            clusters[label]["labels"].append(self.detected_objects[idx][0])

        # Compute average confidence for each cluster
        cluster_confidences = []
        for label, data in clusters.items():
            avg_conf = sum(data["confidences"]) / len(data["confidences"])
            cluster_confidences.append((avg_conf, label))

        # Sort clusters by average confidence
        cluster_confidences.sort(reverse=True)

        # Save top 3 clusters
        with open(self.output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Label", "Center X", "Center Y", "Center Z"])

            for idx, (avg_conf, cluster_label) in enumerate(cluster_confidences[:3]):
                center = np.mean(clusters[cluster_label]["points"], axis=0)
                label = clusters[cluster_label]["labels"][0]  # Take first label
                writer.writerow([label, center[0], center[1], center[2]])

        self.get_logger().info(f"Saved top {min(3, len(clusters))} clusters to {self.output_file}.")

def main(args=None):
    rclpy.init(args=args)
    node = RobotNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
