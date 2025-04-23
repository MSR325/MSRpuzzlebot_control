from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Instancias para el lado izquierdo (left)
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

    left_sp_node = Node(
        namespace='left',
        name='sp_gen',
        package='motor_control',
        executable='set_point',
        emulate_tty=True,
        output='screen',
        parameters=[{
            'signal_type': "square",
            'amplitude': 2.0,
            'omega': 1.0,
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
            'vmax':  6.0
        }]
    )

    # Instancias para el lado derecho (right)
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

    right_sp_node = Node(
        namespace='right',
        name='sp_gen',
        package='motor_control',
        executable='set_point',
        emulate_tty=True,
        output='screen',
        parameters=[{
            'signal_type': "square",
            'amplitude': 2.0,
            'omega': 1.0,
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
            'vmax':  6.0
        }]
    )

    # Nodo de control de trayectoria (global)
    traj_node = Node(
        name='trajectory_controller',
        package='motor_control',
        executable='trajectory',
        emulate_tty=True,
        output='screen',
        parameters=[{
            'wheel_radius': 0.05,
            'wheel_separation': 0.15,
            'sample_time': 0.018,
        }]
    )

    ld = LaunchDescription([
        left_motor_node,
        left_sp_node,
        left_ctrl_node,
        right_motor_node,
        right_sp_node,
        right_ctrl_node,
        traj_node,
    ])

    return ld
