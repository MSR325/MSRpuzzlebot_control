from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    # Iniciar el servidor de Gazebo (gzserver) con el plugin para ROS
    gazebo_server = ExecuteProcess(
        cmd=['gzserver', '--verbose', '-s', 'libgazebo_ros_factory.so'],
        output='screen'
    )

    # Iniciar el cliente gráfico de Gazebo (gzclient) para visualizar la simulación
    gazebo_client = ExecuteProcess(
        cmd=['gzclient', '--verbose'],
        output='screen'
    )

    # Spawnear el modelo en Gazebo usando spawn_entity.py del paquete gazebo_ros
    spawn_robot = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'gazebo_ros', 'spawn_entity.py',
            '-entity', 'puzzlebot_model',
            '-file', '/home/idmx/ros2_ws_2/src/motor_control/puzzlebot/model.sdf',
            '-x', '0', '-y', '0', '-z', '0.5'
        ],
        output='screen'
    )

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
            # PID para posición
            'kP_pos': 0.5,
            'kI_pos': 0.0,
            'kD_pos': 0.0007,
            # PID para orientación
            'kP_ori': 0.5,
            'kI_ori': 0.0,
            'kD_ori': 0.0,
            # Umbral para el control secuencial de yaw
            'heading_threshold': 0.05,
            'position_threshold': 0.02,
            # Pose deseada
            'desired_x': 0.0,
            'desired_y': 0.0,
            'desired_yaw': 0.0
        }]
    )

    return LaunchDescription([
        gazebo_server,
        gazebo_client,
        spawn_robot,
        left_motor_node,
        left_ctrl_node,
        right_motor_node,
        right_ctrl_node,
        odom_node,
        inverse_kinematics_node,
    ])
