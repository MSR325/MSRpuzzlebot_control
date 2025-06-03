from launch import LaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import ExecuteProcess, IncludeLaunchDescription, GroupAction
from launch_ros.actions import PushRosNamespace
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():


    state_estimation_node = Node(
        name='state_estimation',
        package='msr_localization',
        executable='state_estimation',
        emulate_tty=False,
        output='screen',
        parameters=[{
            'wheel_radius': 0.05,
            'wheel_separation': 0.19,
            'sample_time': 0.018,
            'base_frame_id': 'base_link',
            'odom_frame_id': 'odom'
        }]
    )

    inverse_kinematics_node = Node(
        name='inverse_kinematics',
        package='motor_control',
        executable='inverse_kinematics',
        emulate_tty=False,
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
        emulate_tty=False,
        output='screen'
    )

    cmd_mux_node = Node(
        name='cmd_vel_mux',
        package='motor_control',
        executable='cmd_vel_mux',
        emulate_tty=False,
        output='screen'
    )

    static_tf_pub_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_pub_map_to_odom',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        output='screen'
    )

    ekf_static_tf_pub_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_ekf_odom_static',
        arguments=['0', '0', '0',   # xyz
                '0', '0', '0',   # rpy
                'map', 'ekf_odom'],
        output='screen'
    )

    pose_saver_node = Node(
        name='pose_saver',
        package='msr_simulation',
        executable='pose_saver',
        emulate_tty=False,
        output='screen'
    )

    # # --- Robot 2 functional nodes (in /robot2 namespace) ---
    # robot2_nodes = GroupAction([
    #     PushRosNamespace('robot2'),

    #     Node(
    #         name='state_estimation_2',
    #         package='msr_localization',
    #         executable='state_estimation',
    #         emulate_tty=False,
    #         output='screen',
    #         parameters=[{
    #             'wheel_radius': 0.05,
    #             'wheel_separation': 0.19,
    #             'sample_time': 0.018,
    #             'base_frame_id': 'robot2_base_link',
    #             'odom_frame_id': 'ekf_odom'
    #         }]
    #     ),

    #     # Add any other robot2-specific nodes here if needed...
    # ])

    # --- Include display launch (visualization + TFs) ---
    display_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('msr_simulation'),
                'launch',
                'display_ekf.launch.py'
            )
        )
    )

    # # --- Start tmux terminals ---
    # open_tmux_terminals = ExecuteProcess(
    #     cmd=[
    #         'gnome-terminal',
    #         '--',
    #         'bash', '-c', 
    #         os.path.join(
    #             get_package_share_directory('msr_simulation'),
    #             'run',
    #             'run_simulation.sh'
    #         )
    #     ],
    #     shell=False
    # )

    # --- Assemble everything ---
    return LaunchDescription([
        state_estimation_node,
        inverse_kinematics_node,
        path_generator_node,
        cmd_mux_node,
        static_tf_pub_node,
        ekf_static_tf_pub_node,
        pose_saver_node,
        # robot2_nodes,
        display_launch,
        # open_tmux_terminals
    ])
