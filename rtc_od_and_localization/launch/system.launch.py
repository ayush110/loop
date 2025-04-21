import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def launch_camera_if_requested(context, *args, **kwargs):
    launch_camera = LaunchConfiguration("launch_camera").perform(context)

    if launch_camera.lower() == "true":
        # Only import and look up here, not at the top of the file
        from ament_index_python.packages import get_package_share_directory

        zed_launch_file = os.path.join(
            get_package_share_directory("zed_wrapper"), "launch", "zed_camera.launch.py"
        )

        return [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(zed_launch_file),
                launch_arguments={"camera_model": "zed2"}.items(),
            )
        ]
    return []


def launch_rtab_if_requested(context, *args, **kwargs):
    launch_rtab = LaunchConfiguration("launch_rtab").perform(context)
    database_path = LaunchConfiguration("database_path").perform(context)

    if launch_rtab.lower() == "true":
        from ament_index_python.packages import get_package_share_directory

        rtab_launch_file = os.path.join(
            get_package_share_directory("rtabmap_launch"),
            "launch",
            "rtabmap.launch.py",
        )

        return [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(rtab_launch_file),
                launch_arguments={
                    "localization": "true",
                    "rgb_topic": "/zed/zed_node/rgb/image_rect_color",
                    "depth_topic": "/zed/zed_node/depth/depth_registered",
                    "camera_info_topic": "/zed/zed_node/rgb/camera_info",
                    "odom_topic": "/zed/zed_node/odom",
                    "visual_odometry": "false",
                    "frame_id": "zed_camera_link",
                    "approx_sync": "true",
                    "wait_imu_to_init": "true",
                    "rgbd_sync": "true",
                    "approx_rgbd_sync": "true",
                    "imu_topic": "/zed/zed_node/imu/data",
                    "qos": "0",
                    "rviz": "true",
                    "rtabmapviz": "false",
                    "database_path": database_path,
                    "initial_pose": "0 0 0 0 0 0",
                }.items(),
            )
        ]
    return []


def generate_launch_description():
    """
    ros2 launch zed_wrapper rtabmap.launch.py \
    localization:=true \
    rgb_topic:=/zed/zed_node/rgb/image_rect_color \
    depth_topic:=/zed/zed_node/depth/depth_registered \
    camera_info_topic:=/zed/zed_node/rgb/camera_info \
    odom_topic:=/zed/zed_node/odom \
    visual_odometry:=false \
    frame_id:=zed_camera_link \
    approx_sync:=true \
    wait_imu_to_init:=true \
    rgbd_sync:=true \
    approx_rgbd_sync:=true \
    imu_topic:=/zed/zed_node/imu/data \
    qos:=0 \
    rviz:=true \
    rtabmapviz:=false \
    database_path:="/home/nvidia/<path to saved map>" \
    initial_pose:="0 0 0 0 0 0"
    
    parameterized on database_path
    """
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "launch_camera",
                default_value="false",
                description="Whether to launch the ZED camera",
            ),
            DeclareLaunchArgument(
                "launch_rtab",
                default_value="false",
                description="Whether to launch RTAB-Map localization",
            ),
            DeclareLaunchArgument(
                "database_path",
                default_value="/home/nvidia/ros2_ws/src/loop/rtc_od_and_localization/racetrack_map2.db",
                description="Path to saved RTAB-Map database",
            ),
            Node(
                package="rtc_od_and_localization",
                executable="object_detect_node.py",
                name="object_detect_node",
                output="screen",
            ),
            Node(
                package="rtc_od_and_localization",
                executable="localization_node.py",
                name="localization_node",
                output="screen",
            ),
            Node(
                package="rtc_od_and_localization",
                executable="navigation_controller.py",
                name="navigation_controller",
                output="screen",
            ),
            Node(
                package="obj_det_visualizer",
                executable="obj_visualizer",
                name="obj_visualizer",
                output="screen",
            ),
            # run rtab localization
            # Include the camera launch file if requested
            OpaqueFunction(function=launch_camera_if_requested),
            OpaqueFunction(function=launch_rtab_if_requested),
        ]
    )
