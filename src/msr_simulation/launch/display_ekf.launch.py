from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('msr_simulation')
    urdf_path = os.path.join(pkg_share, 'urdf', 'puzzlebot.urdf.xacro')
    rviz_path = os.path.join(pkg_share, 'rviz', 'msr_map_ekf.rviz')

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot1_state_pub',
            namespace='robot1',
            parameters=[{
                'robot_description': Command(['xacro ', urdf_path]),
                'frame_prefix': 'robot1_'
            }],
        ),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot2_state_pub',
            namespace='robot2',
            parameters=[{
                'robot_description': Command(['xacro ', urdf_path]),
                'frame_prefix': 'robot2_'
            }],
        ),

        # EKF broadcaster for Robot 1 (odom → base_link)
        Node(
            package='msr_simulation',
            executable='odom_tf_broadcaster',
            name='odom_tf_broadcaster',
            namespace='robot1',
            output='screen',
            parameters=[
                {'base_frame_id': 'base_link'},
                {'odom_frame_id': 'odom'},
                {'odom_topic': '/odom'}
            ]
        ),

        # EKF broadcaster for Robot 2 (ekf_odom → robot2_base_link)
        Node(
            package='msr_simulation',
            executable='ekf_odom_tf_broadcaster',
            name='ekf_odom_tf_broadcaster_2',
            namespace='robot2',
            output='screen',
            parameters=[
                {'base_frame_id': 'robot2_base_link'},
                {'odom_frame_id': 'ekf_odom'},
                {'odom_topic': '/ekf_odom'}
            ]
        ),

        # RViz
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_path],
            output='screen'
        )
    ])
