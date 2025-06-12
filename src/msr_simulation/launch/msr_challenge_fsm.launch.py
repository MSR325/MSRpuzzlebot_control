from launch import LaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from launch.actions import IncludeLaunchDescription
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

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
        output='screen'
    )

    detection_fsm_node = Node(
        name='detection_fsm',
        package='motor_control',
        executable='detection_fsm',
        emulate_tty=True,
        output='screen'
    )

    cmd_mux_node = Node(
        name='cmd_vel_mux',
        package='motor_control',
        executable='cmd_vel_mux',
        emulate_tty=True,
        output='screen'
    )

    line_follow_node = Node(
        name='line_follow_node',
        package='line_follow_msr',
        executable='line_follower_obesidad_I_for_launch',
        emulate_tty=True,
        output='screen'
    )

    pose_saver_node = Node(
        name='pose_saver',
        package='msr_simulation',
        executable='pose_saver',
        emulate_tty=True,
        output='screen'
    )

    undistort_frames_node = Node(
        name='undistort_frames',
        package='msr_camera_calibration',
        executable='undistort_frames',
        emulate_tty=True,
        output='screen'
    )
    
    crossroad_detection_node = Node(
        name='crossroad_detection_node',
        package='line_follow_msr',
        executable='crossroad_detection_for_launch',
        emulate_tty=True,
        output='screen'
    )

    display_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('msr_simulation'),
                'launch',
                'display.launch.py'
            )
        )
    )

    # yolo_launch = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(
    #             get_package_share_directory('msrsigndet'),
    #             'launch',
    #             'pc.launch.py'
    #         )
    #     )
    # )

    yolo_map_event = Node(
        name='yolo_map_event',
        package='msr_detection',
        executable='yolo_map_event',
        emulate_tty=True,
        output='screen'
    )

    turn_manager_node = Node(
        name='turn_manager_node',
        package='line_follow_msr',
        executable='turn_manager',
        emulate_tty=True,
        output='screen'
    )

    curve_control_node = Node(
        name='curve_control_node',
        package='motor_control',
        executable='curve_control',
        emulate_tty=True,
        output='screen'
    )

    open_tmux_terminals = ExecuteProcess(
        cmd=[
            'gnome-terminal',
            '--',
            'bash', '-c', 
            os.path.join(
                get_package_share_directory('msr_simulation'),
                'run',
                'run_simulation.sh'
            )
        ],
        shell=False
    )

    static_tf_pub_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_pub_map_to_odom',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        output='screen'
    )

    return LaunchDescription([
        odom_node,
        inverse_kinematics_node,
        cmd_mux_node, 
        display_launch,
        detection_fsm_node,
        line_follow_node,
        open_tmux_terminals,
        path_generator_node,
        static_tf_pub_node,
        pose_saver_node,
        undistort_frames_node,
        crossroad_detection_node,
        # yolo_launch,
        yolo_map_event,
        turn_manager_node,
        curve_control_node,
    ])
