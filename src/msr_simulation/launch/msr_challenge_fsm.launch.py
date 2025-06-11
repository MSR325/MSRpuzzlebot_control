from launch import LaunchDescription
from launch_ros.actions import Node, LifecycleNode
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import ExecuteProcess, RegisterEventHandler, TimerAction
from launch.actions import IncludeLaunchDescription
from launch.event_handlers import OnProcessStart
from launch_ros.events.lifecycle import ChangeState
from launch_ros.event_handlers import OnStateTransition
from lifecycle_msgs.msg import Transition
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    # Regular nodes (non-lifecycle)
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

    detection_fsm_node = Node(
        name='detection_fsm',
        package='motor_control',
        executable='detection_fsm',
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

    yolo_map_event = Node(
        name='yolo_map_event',
        package='msr_detection',
        executable='yolo_map_event',
        emulate_tty=True,
        output='screen'
    )

    static_tf_pub_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_pub_map_to_odom',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        output='screen'
    )

    # ========== LIFECYCLE NODES ==========

    # Line Follower (Lifecycle)
    line_follower_node = LifecycleNode(
        name='line_follower',
        namespace='',
        package='line_follow_msr',
        executable='line_follower_centroid',  # Your updated lifecycle executable
        emulate_tty=True,
        output='screen',
        parameters=[{
            'Kp_x': 0.01,
            'Kp_ang': 0.016,
            'xVel': 0.13,
            'ang_e_thrsh': 2.0,
            'min_area_param': 500,
            'length_threshold': 100,
            'homography_matrix_path': 'data/homography3.npy',
            'warp_width': 200,
            'warp_height': 200,
            'max_missing_center': 5
        }]
    )

    # Trajectory Controller (Lifecycle) - Renamed from inverse_kinematics
    trajectory_controller_node = LifecycleNode(
        name='trajectory_controller',
        namespace='',
        package='motor_control',
        executable='inverse_kinematics',  # Your updated lifecycle executable
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
            'waypoint_arrival_tolerance': 0.05,
            'use_path_following': True,
            'lookahead_distance': 0.1,
            'desired_x': 0.0,
            'desired_y': 0.0,
            'desired_yaw': 0.0
        }]
    )

    # Turn Manager (Regular node - manages the lifecycle nodes)
    turn_manager_node = Node(
        name='turn_manager',
        package='line_follow_msr',
        executable='turn_manager',  # Your updated turn manager
        emulate_tty=True,
        output='screen',
        parameters=[{
            'line_follower_node': 'line_follower',
            'trajectory_controller_node': 'trajectory_controller',
            'homography_matrix_path': 'data/homography_after_calib_firstsegment_decent.npy',
            'waypoint_count': 5,
            'arrival_tolerance': 0.02,
            'flip_warp_x': False,
            'flip_warp_y': False
        }]
    )

    # ========== LIFECYCLE STATE MANAGEMENT ==========

    # Configure line follower
    configure_line_follower = ExecuteProcess(
        cmd=['ros2', 'lifecycle', 'set', '/line_follower', 'configure'],
        shell=False
    )

    # Configure trajectory controller  
    configure_trajectory_controller = ExecuteProcess(
        cmd=['ros2', 'lifecycle', 'set', '/trajectory_controller', 'configure'],
        shell=False
    )

    # Activate line follower (start in line following mode)
    activate_line_follower = ExecuteProcess(
        cmd=['ros2', 'lifecycle', 'set', '/line_follower', 'activate'],
        shell=False
    )

    # Keep trajectory controller in configured state (inactive)
    # It will be activated by TurnManager when needed

    # ========== DELAYED LIFECYCLE STARTUP ==========

    # Configure lifecycle nodes after a delay to ensure they're ready
    configure_lifecycle_timer = TimerAction(
        period=3.0,  # Wait 3 seconds for nodes to start
        actions=[
            configure_line_follower,
            configure_trajectory_controller,
        ]
    )

    # Activate line follower after configuration
    activate_lifecycle_timer = TimerAction(
        period=5.0,  # Wait 5 seconds total
        actions=[
            activate_line_follower,
        ]
    )

    # ========== EXTERNAL LAUNCHES ==========

    display_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('msr_simulation'),
                'launch',
                'display.launch.py'
            )
        )
    )

    yolo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('msrsigndet'),
                'launch',
                'pc.launch.py'
            )
        )
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

    # ========== OPTIONAL: Lifecycle Manager Node ==========
    # You can use this instead of manual configuration if preferred
    
    # lifecycle_manager_node = Node(
    #     package='nav2_lifecycle_manager',
    #     executable='lifecycle_manager',
    #     name='lifecycle_manager',
    #     output='screen',
    #     parameters=[{
    #         'node_names': ['line_follower', 'trajectory_controller'],
    #         'autostart': True,
    #         'bond_timeout': 4.0,
    #     }]
    # )

    return LaunchDescription([
        # Regular nodes
        odom_node,
        detection_fsm_node,
        pose_saver_node,
        undistort_frames_node,
        crossroad_detection_node,
        yolo_map_event,
        static_tf_pub_node,
        
        # Lifecycle nodes
        line_follower_node,
        trajectory_controller_node,
        turn_manager_node,
        
        # External launches
        display_launch,
        yolo_launch,
        open_tmux_terminals,
        
        # Lifecycle configuration (delayed)
        configure_lifecycle_timer,
        activate_lifecycle_timer,
        
        # Optional: Use lifecycle manager instead of manual commands
        # lifecycle_manager_node,
    ])