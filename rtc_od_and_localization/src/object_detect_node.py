#!/usr/bin/env python3


from math import sqrt, cos, sin, pi, atan2
from math import pi, log, exp

import numpy as np
import rclpy

# import tf_transformations as tr

from rclpy.node import Node
from rclpy.qos import QoSProfile,ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String, Header, ColorRGBA
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
from geometry_msgs.msg import Twist, PoseStamped, Point, PointStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from builtin_interfaces.msg import Time, Duration
from tf2_ros import Buffer, TransformListener
from threading import Thread, Lock
from zed_msgs.msg import ObjectsStamped # from zed_interfaces.msg import ObjectStamped (FOR USE ON REAL ROBOT)
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point        

class Detector(Node):
    
    def __init__(self):
      
        super().__init__('object_detector')
        #############param declaration ##########################################################

        self.CONV_THRESHOLD = 0.5
        self.MIN_ASSOCIATION_DISTANCE = 0.3 # m

        ################################Data Objects#############################################
        
        self.detected_objects = {
            'person': [],
            'backpack': []
        }
        self.localization_mutex = Lock()
        self.obstacle_mutex = Lock()

        self.global_pose = None
        ################################Publisher##################################################

        # pub to something for constraints if needed
        self.log_pub = self.create_publisher(String, '/detected_objects_log', 10)

        ########################Subcriber#####################################################

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)


        # sub to zed2
        self.camera_detector_subscriber = self.create_subscription(
            ObjectsStamped,
            '/zed/zed_node/obj_det/objects',
            self.object_detection_callback,
            10)
        
        # sub to goal reached

    def object_detection_callback(self, msg: ObjectsStamped):
        self.obstacle_mutex.acquire()
        for obj in msg.objects:
            if obj.label in ['person', 'backpack']: #and obj.tracking_state == 1:  # only if actively tracked
                if obj.confidence < self.CONF_THRESHOLD:
                    self.get_logger().info(f"Detection too low-conf: {obj.confidence}")
                    continue  # skip low-confidence objects
                
                # Get position in camera frame
                position = np.array(obj.position)

                # Transform to map frame using tf2
                position_map = self._transform_point_in_map(position)
                
                # Check if object of same class is already stored
                obj_list = self.detected_objects[obj.label]
    
                for obj in obj_list:
                    dist = np.linalg.norm(obj['avg_position'] - position_map)
                    if dist < self.MIN_ASSOCIATION_DISTANCE:
                        obj['position_sum'] += position_map
                        obj['confidence_total'] += obj.confidence
                        obj['count'] += 1
                        obj['avg_position'] = obj['position_sum'] / obj['count']
                        return

                # No nearby object found, create new entry
                self.detected_objects[obj.label].append({
                    'position_sum': position_map,
                    'count': 1,
                    'avg_position': position_map,
                    'confidence_total': obj.confidence
                })
        self.obstacle_mutex.release()

    def localization_callback(self, msg: PoseWithCovarianceStamped):
        self.localization_mutex.acquire()
        self.global_pose = msg
        self.localization_mutex.release()

    def goal_reached_callback(self, msg):
        # now we need to write to csv 3 objects, class, position
        all_objects = []
        for label, obj_list in self.detected_objects.items():
            for obj in obj_list:
                all_objects.append({
                    'class': label,
                    'position': obj['avg_position'],
                    'confidence': obj['confidence_total']
                })

        # Sort by confidence and pick top 3
        all_objects.sort(key=lambda x: -x['confidence'])
        top3 = all_objects[:3]

        # Write to CSV
        with open('detected_objects.csv', 'w') as f:
            f.write("class,x,y,z\n")
            for obj in top3:
                x, y, z = obj['position']
                f.write(f"{obj['class']},{x},{y},{z}\n")

    def publish_top_objects_log(self):
        all_objects = []
        for label, obj_list in self.detected_objects.items():
            for obj in obj_list:
                all_objects.append({
                    'class': label,
                    'position': obj['avg_position'],
                    'confidence': obj['confidence_total']
                })

        all_objects.sort(key=lambda x: -x['confidence'])
        top3 = all_objects[:3]

        log_msg = "Top Detected Objects:\n"
        for i, obj in enumerate(top3):
            x, y, z = obj['position']
            log_msg += f"{i+1}) {obj['class']} at ({x:.2f}, {y:.2f}, {z:.2f}) with confidence {obj['confidence']:.2f}\n"

        self.log_pub.publish(String(data=log_msg))


    def _transform_point_in_map(self, point, from_frame='zed_camera_center', to_frame='map'):
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(
                to_frame,
                from_frame,
                now,
                timeout=rclpy.duration.Duration(seconds=0.5)
            )

            point_stamped = PointStamped()
            point_stamped.header.stamp = now.to_msg()
            point_stamped.header.frame_id = from_frame
            point_stamped.point.x = point[0]
            point_stamped.point.y = point[1]
            point_stamped.point.z = point[2]

            transformed = do_transform_point(point_stamped, trans)
            return np.array([transformed.point.x, transformed.point.y, transformed.point.z])

        except Exception as e:
            self.get_logger().warn(f"TF transform failed: {e}")
            return None
   
    def _update_or_add_object(self, label, new_pos, confidence):
        obj_list = self.detected_objects[label]
        
        for obj in obj_list:
            dist = np.linalg.norm(obj['avg_position'] - new_pos)
            if dist < self.MIN_ASSOCIATION_DISTANCE:
                obj['position_sum'] += new_pos
                obj['confidence_total'] += confidence
                obj['count'] += 1
                obj['avg_position'] = obj['position_sum'] / obj['count']
                return

        # No nearby object found, create new entry
        self.detected_objects[label].append({
            'position_sum': new_pos,
            'count': 1,
            'avg_position': new_pos,
            'confidence_total': confidence
        })

def main(args=None):
    
    rclpy.init(args=args)
    node = Detector()    
   
    rclpy.spin(node)
    node.node.destroy_node()
    rclpy.shutdown()
            
if __name__ == '__main__':
    main()


