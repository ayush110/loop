import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, Node
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # Declare arguments to allow customization at launch
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (gazebo) clock if true'
        ),

        # Launch the PersonFollower node
        Node(
            package='autolap',  # Name of your package
            executable='person_follower',  # The executable to run
            name='person_follower',
            output='screen',
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        ),

        # Launch the LaneFollower node
        Node(
            package='autolap',
            executable='lane_follower',
            name='lane_follower',
            output='screen',
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        ),

        # Launch the NavigationController node
        Node(
            package='autolap',
            executable='navigation_controller',
            name='navigation_controller',
            output='screen',
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        ),
        
        LogInfo(
            msg="All nodes launched successfully!"
        )
    ])
