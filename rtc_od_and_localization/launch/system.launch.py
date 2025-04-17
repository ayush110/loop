import os
import launch
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import TextSubstitution
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory



def generate_launch_description():
        object_detector_node= Node(
            package='rtc_od_and_localization',
            executable='object_detect_node.py',
            name='object_detect_node',
            output='screen',
        )
        zed_wrapper_dir = get_package_share_directory('zed_wrapper')
        zed_launch_file = os.path.join(
            zed_wrapper_dir,'launch',
            'zed_camera.launch.py'
        )
        zed_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(zed_launch_file),
            launch_arguments={'camera_model': 'zed2'}.items()
        )

        obj_visualizer_node = Node(
            package='obj_det_visualizer',
            executable='obj_visualizer',
            name='obj_visualizer',
            output='screen'
        )


        ld = LaunchDescription()
        ld.add_action(object_detector_node)
        ld.add_action(obj_visualizer_node)
        ld.add_action(zed_launch)

        return ld
