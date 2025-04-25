from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    left_motor_node = Node(
        namespace='left',
        name='motor_sys',
        package='motor_control',
        executable='dc_motor',
        emulate_tty=True,
        output='screen',
        parameters=[{
            'sample_time': 0.018,
            'armature_inductance_La': 3.3e-3,
            'armature_resistance_Ra': 0.5,
            'motor_const_Ka': 0.018,
            'back_emf_const_Kb': 0.06,
            'motor_inertia_Jm': 0.0005,
            'motor_friction_b': 0.0027,
            'load_torque_tau_c': 0.0,
        }]
    )

    left_ctrl_node = Node(
        namespace='left',
        name='ctrl',
        package='motor_control',
        executable='ctrl',
        emulate_tty=True,
        output='screen',
        parameters=[{
            'sample_time': 0.018,
            'kP': 75.0,
            'kI': 0.5,
            'kD': 20.0,
            'La': 3.3e-3,
            'Ra': 0.5,
            'Ka': 0.018,
            'Kb': 0.06,
            'Jm': 0.0005,
            'b': 0.0027,
            'tau_c': 0.0,
            'vmin': -6.0,
            'vmax': 6.0
        }]
    )

    right_motor_node = Node(
        namespace='right',
        name='motor_sys',
        package='motor_control',
        executable='dc_motor',
        emulate_tty=True,
        output='screen',
        parameters=[{
            'sample_time': 0.018,
            'armature_inductance_La': 3.3e-3,
            'armature_resistance_Ra': 0.5,
            'motor_const_Ka': 0.018,
            'back_emf_const_Kb': 0.06,
            'motor_inertia_Jm': 0.0005,
            'motor_friction_b': 0.0027,
            'load_torque_tau_c': 0.0,
        }]
    )

    right_ctrl_node = Node(
        namespace='right',
        name='ctrl',
        package='motor_control',
        executable='ctrl',
        emulate_tty=True,
        output='screen',
        parameters=[{
            'sample_time': 0.018,
            'kP': 75.0,
            'kI': 0.5,
            'kD': 20.0,
            'La': 3.3e-3,
            'Ra': 0.5,
            'Ka': 0.018,
            'Kb': 0.06,
            'Jm': 0.0005,
            'b': 0.0027,
            'tau_c': 0.0,
            'vmin': -6.0,
            'vmax': 6.0
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

    # Nodo path_generator sin waypoints iniciales
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
        left_ctrl_node,
        right_motor_node,
        right_ctrl_node,
        odom_node,
        inverse_kinematics_node,
        cmd_mux_node,
        path_generator_node,
    ])
