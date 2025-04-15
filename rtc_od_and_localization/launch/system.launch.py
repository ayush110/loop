import os
import launch
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.substitutions import TextSubstitution



def generate_launch_description():
        occupany_grid_map= Node(
            package='rtc_od_and_localization',
            executable='object_detect_node.py',
            name='object_detect_node',
            output='screen',
        )

        ld = LaunchDescription()
        ld.add_action(occupany_grid_map)

        return ld
