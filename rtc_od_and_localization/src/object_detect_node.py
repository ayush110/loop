#!/usr/bin/env python3


from math import sqrt, cos, sin, pi, atan2
from math import pi, log, exp

import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String, Header, ColorRGBA
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
from geometry_msgs.msg import (
    Twist,
    PoseStamped,
    Point,
    PointStamped,
    PoseWithCovarianceStamped,
)
from sensor_msgs.msg import LaserScan
from builtin_interfaces.msg import Time, Duration
from tf2_ros import Buffer, TransformListener
from threading import Thread, Lock
from zed_interfaces.msg import ObjectsStamped, Object  # (FOR USE ON REAL ROBOT)
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point
from visualization_msgs.msg import Marker, MarkerArray


class Detector(Node):

    def __init__(self):

        super().__init__("object_detector")
        #############param declaration ##########################################################

        self.CONF_THRESHOLD = 0.5
        self.MIN_ASSOCIATION_DISTANCE = 0.3  # m
        self.MARKER_SIZE = 0.2  # cube side length in meters

        ################################Data Objects#############################################

        self.SUPPORTED_OBJECTS = ["Person", "Bag"]
        self.tracked_objects = {}
        self.detected_objects = {"Person": [], "Bag": []}
        self.obstacle_mutex = Lock()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        ################################Publisher##################################################
        # pub to something for constraints if needed
        self.object_tracks_pub = self.create_publisher(
            ObjectsStamped, "/processed_obstacles", 10
        )
        self.obstacle_timer = self.create_timer(0.5, self.publish_processed_obstacles)

        self.log_pub = self.create_publisher(String, "/detected_objects_log", 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, "/detected_objects_markers", 10
        )
        self.marker_timer = self.create_timer(
            1.0, self.publish_detected_objects_markers
        )

        ########################Subcriber#####################################################
        # sub to zed2
        self.camera_detector_subscriber = self.create_subscription(
            ObjectsStamped,
            "/zed/zed_node/obj_det/objects",
            self.object_detection_callback,
            10,
        )

        # sub to goal reached
        self.goal_reached_subscriber = self.create_subscription(
            PoseStamped, "/goal_reached", self.goal_reached_callback, 10
        )

    def object_detection_callback(self, msg: ObjectsStamped):
        with self.obstacle_mutex:
            for obj in msg.objects:
                if obj.label not in self.SUPPORTED_OBJECTS:
                    continue
                if obj.confidence < self.CONF_THRESHOLD:
                    continue

                position_map = self._transform_point_in_map(obj.position)
                if position_map is None:
                    continue

                self.get_logger().info(f"merging object detection: {obj}")

                self._merge_static_obstacle(obj.label, position_map, obj.confidence)

            self._publish_static_obstacles()

    def _merge_static_obstacle(self, label, position, confidence):
        obj_list = self.detected_objects[label]
        for entry in obj_list:
            dist = np.linalg.norm(entry["avg_position"] - position)
            if dist < self.MIN_ASSOCIATION_DISTANCE:
                entry["position_sum"] += position
                entry["confidence_total"] += confidence
                entry["count"] += 1
                entry["avg_position"] = entry["position_sum"] / entry["count"]
                return

        # New obstacle
        self.detected_objects[label].append(
            {
                "position_sum": position,
                "count": 1,
                "avg_position": position,
                "confidence_total": confidence,
            }
        )

    def _publish_static_obstacles(self):
        output_msg = ObjectsStamped()
        output_msg.header.stamp = self.get_clock().now().to_msg()
        output_msg.header.frame_id = "map"

        for label, obj_list in self.detected_objects.items():
            for entry in obj_list:
                obj = Object()
                obj.label = label
                obj.position.x = entry["avg_position"][0]
                obj.position.y = entry["avg_position"][1]
                obj.position.z = entry["avg_position"][2]
                obj.confidence = entry["confidence_total"] / entry["count"]
                output_msg.objects.append(obj)

        self.object_pub.publish(output_msg)

    def _update_object_tracks(self, object_msg: Object, replace=False):
        if object_msg.confidence < self.CONF_THRESHOLD:
            self.get_logger().info(f"Detection too low-conf: {object_msg.confidence}")
            return

        # get obstacle position in map frame
        position_map = self._transform_point_in_map(object_msg.position)
        if not position_map:
            return

        # Replace=True Either we update the tracks with the most recent object info (query by track id)
        if (
            replace
            and object_msg.tracking_available
            and object_msg.track_id in self.tracked_objects
        ):
            # Use track_id to update existing tracked object
            self.tracked_objects[object_msg.track_id] = {
                "position": position_map,
                "label": object_msg.label,
                "confidence": object_msg.confidence,
            }

            # if obj.tracking_state == 3: then tracking is terminating, then mature to detected objects
            return

        # Replace=False, use some sort of tracking method for static objects based on distance
        # Check if object_msg of same class is already stored
        obj_list = self.detected_objects[object_msg.label]
        for object_msg in obj_list:
            dist = np.linalg.norm(object_msg["avg_position"] - position_map)
            if dist < self.MIN_ASSOCIATION_DISTANCE:
                object_msg["position_sum"] += position_map
                object_msg["confidence_total"] += object_msg.confidence
                object_msg["count"] += 1
                object_msg["avg_position"] = (
                    object_msg["position_sum"] / object_msg["count"]
                )
                return

        # No nearby object_msg found, create new entry
        self.tracked_objects[object_msg.label].append(
            {
                "position_sum": position_map,
                "count": 1,
                "avg_position": position_map,
                "confidence_total": object_msg.confidence,
            }
        )

    def goal_reached_callback(self, msg):
        # now we need to write to csv 3 objects, class, position
        all_objects = []
        for label, obj_list in self.tracked_objects.items():
            for obj in obj_list:
                all_objects.append(
                    {
                        "class": label,
                        "position": obj["avg_position"],
                        "confidence": obj["confidence_total"],
                    }
                )

        # Sort by confidence and pick top 3
        all_objects.sort(key=lambda x: -x["confidence"])
        top3 = all_objects[:3]

        # Write to CSV
        with open("detected_objects.csv", "w") as f:
            f.write("class,x,y,z\n")
            for obj in top3:
                x, y, z = obj["position"]
                f.write(f"{obj['class']},{x},{y},{z}\n")

    def publish_detected_objects_markers(self):
        marker_array = MarkerArray()
        marker_id = 0

        for label, obj_list in self.detected_objects.items():
            for obj in obj_list:
                x, y, z = obj["avg_position"]
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = label
                marker.id = marker_id
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.pose.position.x = x
                marker.pose.position.y = y
                marker.pose.position.z = z
                marker.pose.orientation.w = 1.0
                marker.scale.x = self.MARKER_SIZE
                marker.scale.y = self.MARKER_SIZE
                marker.scale.z = self.MARKER_SIZE
                marker.color = (
                    ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)
                    if label == "person"
                    else ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.8)
                )
                marker.lifetime = Duration(sec=2)
                marker_array.markers.append(marker)
                marker_id += 1

        self.marker_pub.publish(marker_array)

    def publish_top_objects_log(self):
        all_objects = []
        for label, obj_list in self.detected_objects.items():
            for obj in obj_list:
                avg_confidence = obj["confidence_total"] / obj["count"]
                all_objects.append(
                    {
                        "class": label,
                        "position": obj["avg_position"],
                        "confidence": avg_confidence,
                    }
                )

        all_objects.sort(key=lambda x: -x["confidence"])
        top3 = all_objects[:3]

        log_msg = "Top Detected Objects:\n"
        for i, obj in enumerate(top3):
            x, y, z = obj["position"]
            log_msg += f"{i+1}) {obj['class']} at ({x:.2f}, {y:.2f}, {z:.2f}) with confidence {obj['confidence']:.2f}\n"

        self.log_pub.publish(String(data=log_msg))

    def _transform_point_in_map(
        self, point, from_frame="zed_camera_center", to_frame="map"
    ):
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(
                to_frame, from_frame, now, timeout=rclpy.duration.Duration(seconds=0.5)
            )

            point_stamped = PointStamped()
            point_stamped.header.stamp = now.to_msg()
            point_stamped.header.frame_id = from_frame
            point_stamped.point.x = point[0]
            point_stamped.point.y = point[1]
            point_stamped.point.z = point[2]

            transformed = do_transform_point(point_stamped, trans)
            return np.array(
                [transformed.point.x, transformed.point.y, transformed.point.z]
            )

        except Exception as e:
            self.get_logger().warn(f"TF transform failed: {e}")
            return None


def main(args=None):

    rclpy.init(args=args)
    node = Detector()

    rclpy.spin(node)
    node.node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
