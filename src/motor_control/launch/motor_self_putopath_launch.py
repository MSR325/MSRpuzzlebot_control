from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    left_motor_node = Node(
        name='left_motor_controller',
        package='motor_control',
        executable='left_motor_controller',
        emulate_tty=True,
        output='screen',
        parameters=[{
            'kP': 0.02,
            'kI': 0.2,
            'kD': 0.00005,
            'sample_time': 0.018,
            'min_output': -0.4,
            'max_output': 0.4,
            'filter_alpha': 0.05,
            'deadzone_threshold': 1.0
        }]
    )

    right_motor_node = Node(
        name='right_motor_controller',
        package='motor_control',
        executable='right_motor_controller',
        emulate_tty=True,
        output='screen',
        parameters=[{
            'kP': 0.02,
            'kI': 0.2,
            'kD': 0.00005,
            'sample_time': 0.018,
            'min_output': -0.4,
            'max_output': 0.4,
            'filter_alpha': 0.05,
            'deadzone_threshold': 1.0
        }]
    )

    odom_node = Node(
        name='odometry',
        package='motor_control',
        executable='odometry',
        emulate_tty=True,
        output='screen',
        parameters=[{
            'wheel_radius': 0.05,
            'wheel_separation': 0.19,
            'sample_time': 0.018,
        }]
    )

    inverse_kinematics_node = Node(
        name='inverse_kinematics',
        package='motor_control',
        executable='inverse_kinematics',
        emulate_tty=True,
        output='screen',
        parameters=[{
            'wheel_radius': 0.05,
            'wheel_separation': 0.19,
            'sample_time': 0.018,
            'kP_pos': 0.5,
            'kI_pos': 0.0,
            'kD_pos': 0.0007,
            'kP_ori': 0.5,
            'kI_ori': 0.0,
            'kD_ori': 0.0,
            'heading_threshold': 0.05,
            'position_threshold': 0.02,
            'desired_x': 0.0,
            'desired_y': 0.0,
            'desired_yaw': 0.0
        }]
    )

    path_generator_node = Node(
        name='path_generator',
        package='motor_control',
        executable='path_generator',
        emulate_tty=True,
        output='screen',
        parameters=[{
            'sample_time': 0.018,
            'waypoints_json': '[]'
        }]
    )

    cmd_mux_node = Node(
        name='cmd_vel_mux',
        package='motor_control',
        executable='cmd_vel_mux',
        emulate_tty=True,
        output='screen'
    )

    return LaunchDescription([
        left_motor_node,
        right_motor_node,
        odom_node,
        inverse_kinematics_node,
        cmd_mux_node,
        path_generator_node,
    ])
