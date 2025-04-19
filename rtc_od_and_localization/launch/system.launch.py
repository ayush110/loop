import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def launch_camera_if_requested(context, *args, **kwargs):
    launch_camera = LaunchConfiguration('launch_camera').perform(context)

    if launch_camera.lower() == 'true':
        # Only import and look up here, not at the top of the file
        from ament_index_python.packages import get_package_share_directory

        zed_launch_file = os.path.join(
            get_package_share_directory('zed_wrapper'),
            'launch',
            'zed_camera.launch.py'
        )

        return [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(zed_launch_file),
                launch_arguments={'camera_model': 'zed2'}.items()
            )
        ]
    return []


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'launch_camera',
            default_value='false',
            description='Whether to launch the ZED camera'
        ),
        Node(
            package='rtc_od_and_localization',
            executable='object_detect_node.py',
            name='object_detect_node',
            output='screen',
        ),
        Node(
            package='rtc_od_and_localization',
            executable='localization_node.py',
            name='localization_node',
            output='screen',
        ),

        Node(
            package='rtc_od_and_localization',
            executable='navigation_controller.py',
            name='navigation_controller',
            output='screen',
        ),
        # Node(
        #     package='obj_det_visualizer',
        #     executable='obj_visualizer',
        #     name='obj_visualizer',
        #     output='screen'
        # ),
        OpaqueFunction(function=launch_camera_if_requested)
    ])
