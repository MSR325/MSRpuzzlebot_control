from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # -------------------------
    # Nodos del lado izquierdo (namespace 'left')
    # -------------------------
    detection_fsm = Node(
        name='detection_fsm',
        package='motor_control',
        executable='detection_fsm',
        emulate_tty=True,
        output='screen',
    )

    line_follower = Node(
        name='line_follower_cam2',
        package='motor_control',
        executable='line_follower_cam2',
        emulate_tty=True,
        output='screen',
    )

    return LaunchDescription([
        detection_fsm,
        line_follower,
    ])
