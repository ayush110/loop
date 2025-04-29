import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription(
        [
            # Declare launch arguments for kp, ki, and kd
            DeclareLaunchArgument('kp', default_value='0.5', description='Proportional gain'),
            DeclareLaunchArgument('ki', default_value='0.1', description='Integral gain'),
            DeclareLaunchArgument('kd', default_value='0.0', description='Derivative gain'),

            # Launch the rtc_dash node with the specified arguments
            Node(
                package="racetrack_challenge",  # Replace with your package name
                executable="dash.py",  # Ensure this matches the script name
                name="Dash",
                output="screen",
                parameters=[
                    {"kp": LaunchConfiguration('kp')},
                    {"ki": LaunchConfiguration('ki')},
                    {"kd": LaunchConfiguration('kd')},
                ],
            ),
        ]
    )
