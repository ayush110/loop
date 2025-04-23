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


class Detector(Node):
    def __init__(self):
        super().__init__("object_detector")

        self.CONF_THRESHOLD = 20.0
        self.MIN_ASSOCIATION_DISTANCE = 0.8  # meters
        self.MAX_VIEW_DISTANCE = 3
        self.MARKER_SIZE = 0.5  # meters

        self.SUPPORTED_OBJECTS = ["Person", "Vehicle"]
        self.detected_objects = {label: [] for label in self.SUPPORTED_OBJECTS}
        self.obstacle_mutex = Lock()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.obstacle_pub = self.create_publisher(
            ObjectsStamped, "/processed_obstacles", 10
        )
        self.obstacle_timer = self.create_timer(
            0.5, self.published_unfiltered_obstacles
        )

        self.log_pub = self.create_publisher(String, "/detected_objects_log", 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, "/detected_objects_markers", 10
        )
        self.marker_timer = self.create_timer(
            1.0, self.publish_detected_objects_markers
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
                if not position_map or np.isnan(position_map).any():
                    return

                self.get_logger().error(f"position in map: {position_map}")
                self.merge_static_obstacle(obj.label, position_map, obj.confidence)

                # self.detected_objects = self.non_maximum_suppression(
                #     self.detected_objects
                # )

            # change to either publish all received objects or just the best ones
            self.published_unfiltered_obstacles()

    def merge_static_obstacle(self, label, position, confidence):
        obj_list = self.detected_objects[label]

        for entry in obj_list:
            dist = np.linalg.norm(entry["avg_position"] - position)
            if dist < self.MIN_ASSOCIATION_DISTANCE:
                # Merge into existing tracked object
                entry["position_sum"] += position
                entry["confidence_total"] += confidence
                entry["count"] += 1

                self.get_logger().warn(f"COUNT: {entry['count']}")

                entry["confidence"] = float(entry["confidence_total"] / entry["count"])
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

    def non_maximum_suppression(self, objects_by_label):
        final_objects = {}

        for label, objects in objects_by_label.items():
            if not objects:
                continue

            # Sort by confidence descending
            sorted_objects = sorted(objects, key=lambda o: -o["confidence"])

            kept = []
            while sorted_objects:
                current = sorted_objects.pop(0)
                kept.append(current)

                # Remove all others that are too close to `current`
                sorted_objects = [
                    o
                    for o in sorted_objects
                    if np.linalg.norm(o["avg_position"] - current["avg_position"])
                    >= self.MIN_ASSOCIATION_DISTANCE
                ]

            final_objects[label] = kept

        return final_objects

    def goal_reached_callback(self, msg):
        filtered_objects = []

        # Get top 1 Car
        if "Car" in self.detected_objects:
            cars = sorted(
                self.detected_objects["Car"],
                key=lambda o: -(o["confidence"]),
            )
            if cars:
                car = cars[0]
                filtered_objects.append(
                    {
                        "class": "Car",
                        "position": car["avg_position"],
                        "confidence": car["confidence"],
                    }
                )

        # Get top 2 People
        if "Person" in self.detected_objects:
            people = sorted(
                self.detected_objects["Person"],
                key=lambda o: -(o["confidence"]),
            )
            for person in people[:2]:
                filtered_objects.append(
                    {
                        "class": "Person",
                        "position": person["avg_position"],
                        "confidence": person["confidence"],
                    }
                )

        with open("detected_objects.csv", "w") as f:
            f.write("class,x,y,z\n")
            for obj in filtered_objects:
                x, y, z = obj["position"]
                f.write(f"{obj['class']},{x},{y},{z}\n")

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

    def publish_detected_objects_markers(self):
        marker_array = MarkerArray()
        marker_id = 0
        selected_objects = []

        # # Top 1 Car
        # if "Car" in self.detected_objects:
        #     cars = sorted(
        #         self.detected_objects["Car"],
        #         key=lambda o: -(o["confidence_total"] / o["count"]),
        #     )
        #     if cars:
        #         selected_objects.append(("Car", cars[0]["avg_position"]))

        # # Top 2 People
        # if "Person" in self.detected_objects:
        #     people = sorted(
        #         self.detected_objects["Person"],
        #         key=lambda o: -(o["confidence_total"] / o["count"]),
        #     )
        #     for person in people[:2]:
        #         selected_objects.append(("Person", person["avg_position"]))

        for label, detections in self.detected_objects.items():
            for detection in detections:
                selected_objects.append(
                    (label, detection["avg_position"], detection["confidence"])
                )

                self.get_logger().warn(
                    f"{detection['avg_position']}, {detection['confidence']}"
                )

        # Publish only selected markers
        for label, position, confidence in selected_objects:
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
                )  # blue for Car
            )
            marker.lifetime = Duration(sec=2)
            marker_array.markers.append(marker)
            marker_id += 1

        self.marker_pub.publish(marker_array)

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
