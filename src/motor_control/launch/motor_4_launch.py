from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # -------------------------
    # Nodos del lado izquierdo (namespace 'left')
    # -------------------------
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

    # -------------------------
    # Nodos del lado derecho (namespace 'right')
    # -------------------------
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

    # -------------------------
    # Nodos globales (sin namespace)
    # -------------------------
    # Nodo de control de trayectoria (cinemática directa)
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

    # Nodo de odometría: integra las velocidades de las ruedas para publicar la pose actual.
    # Se asume que el entry point definido en setup.py es 'odometry' (no 'odometry_node')
    odom_node = Node(
        name='odometry',
        package='motor_control',
        executable='odometry',
        emulate_tty=True,
        output='screen',
        parameters=[{
            'wheel_radius': 0.05,
            'wheel_separation': 0.15,
            'sample_time': 0.018,
        }]
    )

    # Nodo de cinemática inversa: calcula los setpoints de las ruedas a partir de la pose deseada y la actual.
    inverse_kinematics_node = Node(
        name='inverse_kinematics',
        package='motor_control',
        executable='inverse_kinematics',
        emulate_tty=True,
        output='screen',
        parameters=[{
            'wheel_radius': 0.05,
            'wheel_separation': 0.15,
            'sample_time': 0.018,
            'kP_pos': 1.0,
            'kP_ori': 1.0,
        }]
    )

    return LaunchDescription([
        left_motor_node,
        left_ctrl_node,
        right_motor_node,
        right_ctrl_node,
        traj_node,
        odom_node,
        inverse_kinematics_node,
    ])
