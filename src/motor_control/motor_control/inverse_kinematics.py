#!/usr/bin/env python3
import rclpy
from rclpy_lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path as PathMsg
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, String, Bool
import math
from rcl_interfaces.msg import SetParametersResult
import transforms3d as tfe
from geometry_msgs.msg import Twist

def saturate(value, min_val, max_val):
    return max(min(value, max_val), min_val)

class InverseKinematics(LifecycleNode):
    def __init__(self):
        super().__init__('inverse_kinematics')

        # Parámetros de cinemática y controlador
        self.declare_parameter('wheel_radius', 0.05)
        self.declare_parameter('wheel_separation', 0.173)
        self.declare_parameter('sample_time', 0.018)

        # PID para posición
        self.declare_parameter('kP_pos', 0.1)
        self.declare_parameter('kI_pos', 0.0)
        self.declare_parameter('kD_pos', 0.05)

        # PID para orientación
        self.declare_parameter('kP_ori', 0.5)
        self.declare_parameter('kI_ori', 0.0)
        self.declare_parameter('kD_ori', 0.05)

        # Umbrales para cambio de modo
        self.declare_parameter('heading_threshold', 0.1)
        self.declare_parameter('position_threshold', 0.1)
        self.declare_parameter('waypoint_arrival_tolerance', 0.05)

        # Parámetros de seguimiento de trayectoria
        self.declare_parameter('use_path_following', True)
        self.declare_parameter('lookahead_distance', 0.1)

        # Deadzones internas
        self.orientation_deadzone = 0.02
        self.position_deadzone = 0.02
        self.lin_vel_mag_saturation = 0.5
        self.ang_vel_mag_saturation = 3.0
        self.color_flag_multiplier = 1.0

        # Variables de estado
        self.current_pose = None
        self.current_path = None
        self.current_waypoint_idx = 0
        self.target_pose = None

        # Variables de control PID
        self.integral_error_pos = 0.0
        self.prev_error_pos = 0.0
        self.integral_error_ori = 0.0
        self.prev_error_ori = 0.0

        # Lifecycle-managed resources
        self.control_timer = None
        self.publishers = {}
        self.subscriptions = {}

        self.get_logger().info('InverseKinematics lifecycle node created')

    # ====================== LIFECYCLE CALLBACKS ======================

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Configure the node - load parameters and initialize resources"""
        self.get_logger().info('Configuring InverseKinematics...')

        try:
            # Load parameters
            self._load_parameters()

            # Create publishers
            self.publishers['left_setpoint'] = self.create_publisher(
                Float32, 'left/set_point', 10)
            self.publishers['right_setpoint'] = self.create_publisher(
                Float32, 'right/set_point', 10)
            self.publishers['cmd_vel'] = self.create_publisher(
                Twist, '/cmd_vel', 10)
            self.publishers['current_yaw'] = self.create_publisher(
                Float32, 'current_yaw', 10)
            self.publishers['completed_curve'] = self.create_publisher(
                Bool, '/completed_curve', 10)

            # Create subscriptions
            self.subscriptions['odom'] = self.create_subscription(
                Odometry, '/odom', self.current_pose_callback, 10)
            self.subscriptions['fsm_action'] = self.create_subscription(
                Float32, '/fsm_action', self.color_flag_callback, 10)
            self.subscriptions['waypoints'] = self.create_subscription(
                PathMsg, '/turn_manager/waypoints', self.waypoints_callback, 10)

            # Add parameter callback
            self.add_on_set_parameters_callback(self.parameter_callback)

            self.get_logger().info('✅ InverseKinematics configured successfully')
            return TransitionCallbackReturn.SUCCESS

        except Exception as e:
            self.get_logger().error(f'❌ Configuration failed: {e}')
            return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Activate the node - start control loop"""
        self.get_logger().info('🚀 Activating InverseKinematics...')

        try:
            # Start control timer
            self.control_timer = self.create_timer(self.sample_time, self.timer_callback)

            # Reset control state
            self._reset_control_state()

            # Activate publishers
            for pub in self.publishers.values():
                pub.on_activate(state)

            self.get_logger().info('✅ InverseKinematics activated - trajectory following ACTIVE')
            return TransitionCallbackReturn.SUCCESS

        except Exception as e:
            self.get_logger().error(f'❌ Activation failed: {e}')
            return TransitionCallbackReturn.FAILURE

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Deactivate the node - stop control loop"""
        self.get_logger().info('🛑 Deactivating InverseKinematics...')

        try:
            # Stop the robot immediately
            self._stop_robot()

            # Stop control timer
            if self.control_timer is not None:
                self.control_timer.cancel()
                self.control_timer = None

            # Deactivate publishers
            for pub in self.publishers.values():
                pub.on_deactivate(state)

            # Reset trajectory state
            self.current_path = None
            self.current_waypoint_idx = 0
            self.target_pose = None

            self.get_logger().info('✅ InverseKinematics deactivated - trajectory following STOPPED')
            return TransitionCallbackReturn.SUCCESS

        except Exception as e:
            self.get_logger().error(f'❌ Deactivation failed: {e}')
            return TransitionCallbackReturn.FAILURE

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Cleanup resources"""
        self.get_logger().info('🧹 Cleaning up InverseKinematics...')

        try:
            # Destroy publishers
            for name, pub in self.publishers.items():
                self.destroy_publisher(pub)
            self.publishers.clear()

            # Destroy subscriptions
            for name, sub in self.subscriptions.items():
                self.destroy_subscription(sub)
            self.subscriptions.clear()

            self.get_logger().info('✅ InverseKinematics cleaned up')
            return TransitionCallbackReturn.SUCCESS

        except Exception as e:
            self.get_logger().error(f'❌ Cleanup failed: {e}')
            return TransitionCallbackReturn.FAILURE

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Shutdown the node"""
        self.get_logger().info('🔴 Shutting down InverseKinematics...')
        
        # Ensure robot is stopped
        self._stop_robot()
        
        return TransitionCallbackReturn.SUCCESS

    # ====================== HELPER METHODS ======================

    def _load_parameters(self):
        """Load all parameters"""
        self.wheel_radius = self.get_parameter('wheel_radius').value
        self.wheel_separation = self.get_parameter('wheel_separation').value
        self.sample_time = self.get_parameter('sample_time').value

        self.kP_pos = self.get_parameter('kP_pos').value
        self.kI_pos = self.get_parameter('kI_pos').value
        self.kD_pos = self.get_parameter('kD_pos').value

        self.kP_ori = self.get_parameter('kP_ori').value
        self.kI_ori = self.get_parameter('kI_ori').value
        self.kD_ori = self.get_parameter('kD_ori').value

        self.heading_threshold = self.get_parameter('heading_threshold').value
        self.position_threshold = self.get_parameter('position_threshold').value
        self.waypoint_arrival_tolerance = self.get_parameter('waypoint_arrival_tolerance').value
        
        self.use_path_following = self.get_parameter('use_path_following').value
        self.lookahead_distance = self.get_parameter('lookahead_distance').value

    def _reset_control_state(self):
        """Reset PID control state"""
        self.integral_error_pos = 0.0
        self.prev_error_pos = 0.0
        self.integral_error_ori = 0.0
        self.prev_error_ori = 0.0
        self.current_waypoint_idx = 0

    def _stop_robot(self):
        """Send stop commands to robot"""
        try:
            # Stop wheel setpoints
            if 'left_setpoint' in self.publishers:
                self.publishers['left_setpoint'].publish(Float32(data=0.0))
            if 'right_setpoint' in self.publishers:
                self.publishers['right_setpoint'].publish(Float32(data=0.0))
            
            # Stop cmd_vel
            if 'cmd_vel' in self.publishers:
                stop_twist = Twist()
                self.publishers['cmd_vel'].publish(stop_twist)
                
            self.get_logger().info('🛑 Robot stopped')
        except Exception as e:
            self.get_logger().warn(f'Error stopping robot: {e}')

    # ====================== CALLBACK METHODS ======================

    def current_pose_callback(self, msg: Odometry):
        """Update current pose"""
        self.current_pose = msg
    
    def color_flag_callback(self, msg: Float32):
        """Update speed multiplier"""
        self.color_flag_multiplier = msg.data

    def waypoints_callback(self, msg: PathMsg):
        """Receive new waypoint path from TurnManager"""
        if self.get_current_state().id == LifecycleState.PRIMARY_STATE_ACTIVE:
            self.current_path = msg
            self.current_waypoint_idx = 0
            self.get_logger().info(f'📍 Received new path with {len(msg.poses)} waypoints')
        
    def timer_callback(self):
        """Main control loop"""
        # Only run if we're in the ACTIVE state
        if self.get_current_state().id != LifecycleState.PRIMARY_STATE_ACTIVE:
            return

        if self.current_pose is None:
            return

        # Get current pose
        cur_x = self.current_pose.pose.pose.position.x
        cur_y = self.current_pose.pose.pose.position.y
        cur_yaw = self.quaternion_to_yaw(self.current_pose.pose.pose.orientation)

        # Publish current yaw
        if 'current_yaw' in self.publishers:
            self.publishers['current_yaw'].publish(Float32(data=cur_yaw))

        # Determine target based on mode
        if self.use_path_following and self.current_path is not None:
            target = self._get_path_target(cur_x, cur_y)
        else:
            # Single point mode (backwards compatibility)
            target = {
                'x': self.get_parameter('desired_x').value if self.has_parameter('desired_x') else 0.0,
                'y': self.get_parameter('desired_y').value if self.has_parameter('desired_y') else 0.0,
                'yaw': self.get_parameter('desired_yaw').value if self.has_parameter('desired_yaw') else 0.0
            }

        if target is None:
            # No valid target - stop
            self._stop_robot()
            return

        # Run control algorithm
        self._execute_control(cur_x, cur_y, cur_yaw, target)

    def _get_path_target(self, cur_x, cur_y):
        """Get target pose from current path"""
        if not self.current_path or not self.current_path.poses:
            return None

        # Check if we've reached the current waypoint
        if self.current_waypoint_idx < len(self.current_path.poses):
            waypoint = self.current_path.poses[self.current_waypoint_idx]
            wp_x = waypoint.pose.position.x
            wp_y = waypoint.pose.position.y
            
            distance_to_waypoint = math.hypot(cur_x - wp_x, cur_y - wp_y)
            
            if distance_to_waypoint < self.waypoint_arrival_tolerance:
                self.current_waypoint_idx += 1
                self.get_logger().info(f'📍 Reached waypoint {self.current_waypoint_idx-1}')
                
                # Check if we've completed the path
                if self.current_waypoint_idx >= len(self.current_path.poses):
                    self.get_logger().info('🎯 Path completed!')
                    if 'completed_curve' in self.publishers:
                        self.publishers['completed_curve'].publish(Bool(data=True))
                    return None

        # Get current target waypoint
        if self.current_waypoint_idx < len(self.current_path.poses):
            waypoint = self.current_path.poses[self.current_waypoint_idx]
            
            # For path following, we want to point towards the waypoint
            target_yaw = math.atan2(
                waypoint.pose.position.y - cur_y,
                waypoint.pose.position.x - cur_x
            )
            
            return {
                'x': waypoint.pose.position.x,
                'y': waypoint.pose.position.y,
                'yaw': target_yaw
            }
        
        return None

    def _execute_control(self, cur_x, cur_y, cur_yaw, target):
        """Execute PID control to reach target"""
        dt = self.sample_time
        
        # Target position
        des_x, des_y, des_yaw = target['x'], target['y'], target['yaw']

        # Position error
        err_x = des_x - cur_x
        err_y = des_y - cur_y
        err_d = math.hypot(err_x, err_y)

        # Heading target
        if err_d > self.position_threshold:
            heading_target = math.atan2(err_y, err_x)
        else:
            heading_target = des_yaw

        error_heading = self.normalize_angle(heading_target - cur_yaw)

        # PID orientation control
        self.integral_error_ori += error_heading * dt
        d_ori = (error_heading - self.prev_error_ori) / dt
        self.prev_error_ori = error_heading
        pid_ori = (self.kP_ori * error_heading +
                   self.kI_ori * self.integral_error_ori +
                   self.kD_ori * d_ori)

        # Linear velocity decision
        if abs(error_heading) > self.heading_threshold:
            V_d = 0.0
            # Anti-windup for position
            self.integral_error_pos = 0.0
            self.prev_error_pos = err_d
        else:
            # PID position control
            self.integral_error_pos += err_d * dt
            d_pos = (err_d - self.prev_error_pos) / dt
            self.prev_error_pos = err_d
            pid_pos = (self.kP_pos * err_d +
                       self.kI_pos * self.integral_error_pos +
                       self.kD_pos * d_pos)
            V_d = pid_pos * math.cos(error_heading)

        # Apply internal deadzones
        if abs(error_heading) < self.orientation_deadzone:
            pid_ori = 0.0
            self.integral_error_ori = 0.0

        if err_d < self.position_deadzone:
            V_d = 0.0
            self.integral_error_pos = 0.0

        omega_d = pid_ori

        # Saturate velocities
        V_d = saturate(V_d, -self.lin_vel_mag_saturation, self.lin_vel_mag_saturation)
        V_d = V_d * self.color_flag_multiplier

        # Minimum angular velocity threshold
        if abs(omega_d) > 1e-4 and abs(omega_d) < 0.3:
            omega_d = 0.3 * (1 if omega_d > 0 else -1)

        omega_d = saturate(omega_d, -self.ang_vel_mag_saturation, self.ang_vel_mag_saturation)
        omega_d = omega_d * self.color_flag_multiplier

        # Calculate wheel setpoints
        left_setpoint = (V_d - (self.wheel_separation / 2.0) * omega_d) / self.wheel_radius
        right_setpoint = (V_d + (self.wheel_separation / 2.0) * omega_d) / self.wheel_radius

        # Publish commands
        if 'left_setpoint' in self.publishers:
            self.publishers['left_setpoint'].publish(Float32(data=left_setpoint))
        if 'right_setpoint' in self.publishers:
            self.publishers['right_setpoint'].publish(Float32(data=right_setpoint))

        # Publish twist
        if 'cmd_vel' in self.publishers:
            twist_msg = Twist()
            twist_msg.linear.x = V_d
            twist_msg.angular.z = omega_d
            self.publishers['cmd_vel'].publish(twist_msg)

    # ====================== UTILITY METHODS ======================

    def quaternion_to_yaw(self, q):
        """Convert quaternion to yaw angle"""
        q_array = [q.w, q.x, q.y, q.z]
        _, _, yaw = tfe.euler.quat2euler(q_array)
        return yaw

    def normalize_angle(self, angle):
        """Normalize angle to [-π, π]"""
        return math.atan2(math.sin(angle), math.cos(angle))

    def parameter_callback(self, params):
        """Handle parameter updates"""
        for param in params:
            name = param.name
            val = param.value

            if name == 'wheel_radius':
                if val <= 0.0:
                    return SetParametersResult(successful=False, reason="wheel_radius must be > 0")
                self.wheel_radius = val

            elif name == 'wheel_separation':
                if val <= 0.0:
                    return SetParametersResult(successful=False, reason="wheel_separation must be > 0")
                self.wheel_separation = val

            elif name == 'sample_time':
                if val <= 0.0:
                    return SetParametersResult(successful=False, reason="sample_time must be > 0")
                self.sample_time = val
                # Recreate timer with new period if active
                if self.control_timer is not None:
                    self.control_timer.destroy()
                    self.control_timer = self.create_timer(self.sample_time, self.timer_callback)

            elif name in ('kP_pos', 'kI_pos', 'kD_pos', 'kP_ori', 'kI_ori', 'kD_ori'):
                setattr(self, name, val)
            elif name in ('heading_threshold', 'position_threshold', 'waypoint_arrival_tolerance'):
                setattr(self, name, val)
            elif name in ('use_path_following', 'lookahead_distance'):
                setattr(self, name, val)

        return SetParametersResult(successful=True)


def main(args=None):
    rclpy.init(args=args)
    node = InverseKinematics()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 Shutting down InverseKinematics...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()