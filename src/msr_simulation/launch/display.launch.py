from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_share = get_package_share_directory('msr_simulation')
    urdf_path = os.path.join(pkg_share, 'urdf', 'puzzlebot.urdf.xacro')
    rviz_path = os.path.join(pkg_share, 'rviz', 'msr_map.rviz')

    # Read URDF as a string safely
    with open(urdf_path, 'r') as urdf_file:
        urdf_content = urdf_file.read()

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'robot_description': urdf_content}]
        ),
        # Node(
        #     package='joint_state_publisher',
        #     executable='joint_state_publisher',
        #     name='joint_state_publisher'
        # ),
        Node(
            package='msr_simulation',
            executable='odom_tf_broadcaster',
            name='odom_tf_broadcaster',
            output='screen'
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_path],
            output='screen'
        )
    ])
