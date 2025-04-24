#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String, ColorRGBA
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point
from visualization_msgs.msg import Marker, MarkerArray
from zed_interfaces.msg import ObjectsStamped, Object
from threading import Lock
from sklearn.cluster import DBSCAN

from itertools import combinations


class Detector(Node):
    def __init__(self):
        super().__init__("object_detector")

        self.CONF_THRESHOLD = 20.0
        self.MIN_ASSOCIATION_DISTANCE = 0.8  # meters
        self.MAX_VIEW_DISTANCE = 6
        self.MARKER_SIZE = 0.5  # meters

        self.SUPPORTED_OBJECTS = ["Person", "Vehicle"]
        self.detected_objects = {label: [] for label in self.SUPPORTED_OBJECTS}
        self.obstacle_mutex = Lock()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.obstacle_pub = self.create_publisher(
            ObjectsStamped, "/processed_obstacles", 10
        )

        self.log_pub = self.create_publisher(String, "/detected_objects_log", 10)
        self.detected_obstacles_pub = self.create_publisher(
            MarkerArray, "/detected_objects_markers", 10
        )
        self.filtered_obstacles_pub = self.create_publisher(
            MarkerArray, "/filtered_objects_markers", 10
        )

        self.create_subscription(
            ObjectsStamped,
            "/zed/zed_node/obj_det/objects",
            self.object_detection_callback,
            10,
        )

        self.create_subscription(
            PoseStamped, "/goal_reached", self.goal_reached_callback, 10
        )

    def object_detection_callback(self, msg: ObjectsStamped):
        with self.obstacle_mutex:
            for obj in msg.objects:
                if obj.label not in self.SUPPORTED_OBJECTS:
                    continue
                if obj.confidence < self.CONF_THRESHOLD:
                    self.get_logger().info(
                        f"Filtering object: {obj.label} confidence is {obj.confidence} < {self.CONF_THRESHOLD}"
                    )
                    continue

                distance_from_camera = np.linalg.norm(obj.position)
                if distance_from_camera > self.MAX_VIEW_DISTANCE:
                    self.get_logger().info(
                        f"Filtering object: {obj.label} distance is {distance_from_camera} > {self.MAX_VIEW_DISTANCE}"
                    )
                    continue

                position_map = self._transform_point_in_map(obj.position)
                if position_map is None or np.isnan(position_map).any():
                    continue

                self.merge_static_obstacle(obj.label, position_map, obj.confidence)

                # self.detected_objects = self.non_maximum_suppression(
                #     self.detected_objects
                # )

            # change to either publish all received objects or just the best ones
            self.publish_detected_obstacle_markers()

            self.published_unfiltered_obstacles()

    def merge_static_obstacle(self, label, position, confidence, merge_existing=False):
        obj_list = self.detected_objects[label]

        if merge_existing:
            for entry in obj_list:
                dist = np.linalg.norm(entry["avg_position"] - position)
                if dist < self.MIN_ASSOCIATION_DISTANCE:
                    # Merge into existing tracked object
                    entry["position_sum"] += position
                    entry["confidence_total"] += confidence
                    entry["count"] += 1

                    entry["confidence"] = float(
                        entry["confidence_total"] / entry["count"]
                    )
                    entry["avg_position"] = entry["position_sum"] / entry["count"]
                    return

        # No close object found, create new one
        obj_list.append(
            {
                "position_sum": np.array(position, dtype=np.float32),
                "avg_position": np.array(position, dtype=np.float32),
                "count": 1,
                "confidence_total": confidence,
                "confidence": confidence,
            }
        )

    # def non_maximum_suppression(self, objects_by_label):
    #     final_objects = {}

    #     for label, objects in objects_by_label.items():
    #         if not objects:
    #             continue

    #         # Sort by confidence descending
    #         sorted_objects = sorted(objects, key=lambda o: -o["confidence"])

    #         kept = []
    #         while sorted_objects:
    #             current = sorted_objects.pop(0)
    #             kept.append(current)

    #             # Remove all others that are too close to `current`
    #             sorted_objects = [
    #                 o
    #                 for o in sorted_objects
    #                 if np.linalg.norm(o["avg_position"] - current["avg_position"])
    #                 >= self.MIN_ASSOCIATION_DISTANCE
    #             ]

    #         final_objects[label] = kept

    #     return final_objects

    def goal_reached_callback(self, msg):
        # now filter_top_3() returns exactly what we need for CSV
        with self.obstacle_mutex:
            filtered_objects = self.offline_filter_obstacles()

        with open("detected_objects.csv", "w") as f:
            f.write("class,x,y,z\n")
            for obj in filtered_objects:
                x, y, z = obj["position"]
                f.write(f"{obj['class']},{x:.3f},{y:.3f},{z:.3f}\n")

        self.publish_offline_filtered_obstacle_markers(filtered_objects)
        self.get_logger().info("Filtered objects saved to detected_objects.csv")

        # exit the node and everything
        self.destroy_node()
        rclpy.shutdown()

    def offline_filter_obstacles(self):
        all_detections = []

        for label, entries in self.detected_objects.items():
            for e in entries:
                all_detections.append(
                    {
                        "label": label,
                        "position": e["avg_position"],
                        "confidence": e["confidence"],
                    }
                )

        if not all_detections:
            self.get_logger().warning("No detections to cluster.")
            return []

        positions = np.array([d["position"] for d in all_detections])
        confidences = np.array([d["confidence"] for d in all_detections])
        labels = [d["label"] for d in all_detections]

        # DBSCAN clustering
        clustering = DBSCAN(eps=0.8, min_samples=1).fit(positions)
        cluster_ids = clustering.labels_

        clusters = {}
        for i, cluster_id in enumerate(cluster_ids):
            if cluster_id not in clusters:
                clusters[cluster_id] = {
                    "positions": [],
                    "confidences": [],
                    "labels": [],
                }
            clusters[cluster_id]["positions"].append(positions[i])
            clusters[cluster_id]["confidences"].append(confidences[i])
            clusters[cluster_id]["labels"].append(labels[i])

        merged_detections = []
        for cluster_id, data in clusters.items():
            pos_array = np.array(data["positions"])
            conf_array = np.array(data["confidences"])
            label_array = data["labels"]

            total_conf = conf_array.sum()
            weighted_pos = np.average(pos_array, axis=0, weights=conf_array)
            majority_label = max(set(label_array), key=label_array.count)

            merged_detections.append(
                {
                    "class": majority_label,
                    "position": weighted_pos,
                    "confidence": total_conf,
                }
            )

        # Try to find 2 Person + 1 Vehicle combo with highest confidence
        people = [d for d in merged_detections if d["class"] == "Person"]
        vehicles = [d for d in merged_detections if d["class"] == "Vehicle"]

        if len(people) >= 2 and len(vehicles) >= 1:
            best_score = -1.0
            best_triple = None

            for p1, p2 in combinations(people, 2):
                for v in vehicles:
                    total_conf = p1["confidence"] + p2["confidence"] + v["confidence"]
                    if total_conf > best_score:
                        best_score = total_conf
                        best_triple = [p1, p2, v]

            return best_triple

        # Fallback: return top-3 by confidence
        self.get_logger().error("Did not find 2P+1V in clusters, returning top-3")
        # TODO: CHANGE THIS BACK TO 3
        return sorted(merged_detections, key=lambda d: -d["confidence"])  # [:3]

    def published_unfiltered_obstacles(self):
        output_msg = ObjectsStamped()
        output_msg.header.stamp = self.get_clock().now().to_msg()
        output_msg.header.frame_id = "map"

        for label, obj_list in self.detected_objects.items():
            for entry in obj_list:
                obj = Object()
                obj.label = label
                obj.position = np.array(entry["avg_position"], dtype=np.float32)
                obj.confidence = float(entry["confidence"])
                output_msg.objects.append(obj)

        self.obstacle_pub.publish(output_msg)

    def publish_detected_obstacle_markers(self):
        marker_array = MarkerArray()
        marker_id = 0
        selected_objects = []

        for label, detections in self.detected_objects.items():
            for detection in detections:
                selected_objects.append(
                    (label, detection["avg_position"], detection["confidence"])
                )

        # Publish only selected markers
        for label, position, confidence in selected_objects:
            x, y, z = position
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = label
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = self.MARKER_SIZE / 2.0
            marker.scale.y = self.MARKER_SIZE / 2.0
            marker.scale.z = self.MARKER_SIZE / 2.0
            marker.color = (
                ColorRGBA(r=0.0, g=1.0, b=0.0, a=confidence / 100.0)  # green for Person
                if label == "Person"
                else ColorRGBA(
                    r=0.0, g=0.0, b=1.0, a=confidence / 100.0
                )  # blue for Vehicle
            )
            marker.lifetime = Duration(sec=2)
            marker_array.markers.append(marker)
            marker_id += 1

        self.detected_obstacles_pub.publish(marker_array)

    def publish_offline_filtered_obstacle_markers(self, filtered_obstacles):
        marker_array = MarkerArray()
        marker_id = 0
        for obstacle in filtered_obstacles:
            position = obstacle["position"]
            label = obstacle["class"]
            confidence = obstacle["confidence"]

            # Create a marker for each detected object
            x, y, z = position
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = label
            marker.id = marker_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = self.MARKER_SIZE
            marker.scale.y = self.MARKER_SIZE
            marker.scale.z = self.MARKER_SIZE
            marker.color = (
                ColorRGBA(r=0.0, g=1.0, b=0.0, a=confidence / 100.0)  # green for Person
                if label == "Person"
                else ColorRGBA(
                    r=0.0, g=0.0, b=1.0, a=confidence / 100.0
                )  # blue for Vehicle
            )
            marker.lifetime = Duration(sec=2)
            marker_array.markers.append(marker)
            marker_id += 1

        self.filtered_obstacles_pub.publish(marker_array)

    def _transform_point_in_map(self, point, from_frame="base_link", to_frame="map"):
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(to_frame, from_frame, now)

            point_stamped = PointStamped()
            point_stamped.header.stamp = now.to_msg()
            point_stamped.header.frame_id = from_frame
            point_stamped.point.x = float(point[0])
            point_stamped.point.y = float(point[1])
            point_stamped.point.z = float(point[2])

            transformed = do_transform_point(point_stamped, trans)
            return np.array(
                [transformed.point.x, transformed.point.y, transformed.point.z],
                dtype=np.float32,
            )

        except Exception as e:
            self.get_logger().warn(f"TF transform failed: {e}")
            return None


def main(args=None):
    rclpy.init(args=args)
    node = Detector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
