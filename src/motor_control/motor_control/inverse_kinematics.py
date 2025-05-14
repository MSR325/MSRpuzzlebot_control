#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, String
import math
from rcl_interfaces.msg import SetParametersResult
import transforms3d as tfe
from geometry_msgs.msg import Twist

def saturate(value, min_val, max_val):
    return max(min(value, max_val), min_val)

class InverseKinematics(Node):
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

        # Poses deseadas
        self.declare_parameter('desired_x', 0.0)
        self.declare_parameter('desired_y', 0.0)
        self.declare_parameter('desired_yaw', 0.0)

        # Deadzones internas
        self.orientation_deadzone = 0.02
        self.position_deadzone = 0.02
        self.lin_vel_mag_saturation = 0.15
        self.ang_vel_mag_saturation = 0.8
        self.color_flag_multiplier = 1.0

        # Carga de parámetros
        self.wheel_radius      = self.get_parameter('wheel_radius').value
        self.wheel_separation  = self.get_parameter('wheel_separation').value
        self.sample_time       = self.get_parameter('sample_time').value

        self.kP_pos = self.get_parameter('kP_pos').value
        self.kI_pos = self.get_parameter('kI_pos').value
        self.kD_pos = self.get_parameter('kD_pos').value

        self.kP_ori = self.get_parameter('kP_ori').value
        self.kI_ori = self.get_parameter('kI_ori').value
        self.kD_ori = self.get_parameter('kD_ori').value

        self.heading_threshold  = self.get_parameter('heading_threshold').value
        self.position_threshold = self.get_parameter('position_threshold').value

        self.desired_x   = self.get_parameter('desired_x').value
        self.desired_y   = self.get_parameter('desired_y').value
        self.desired_yaw = self.get_parameter('desired_yaw').value

        # Variables internas de PID
        self.integral_error_pos = 0.0
        self.prev_error_pos     = 0.0
        self.integral_error_ori = 0.0
        self.prev_error_ori     = 0.0

        # Subscripciones y publicaciones
        self.current_pose_sub = self.create_subscription(Odometry, 'odom', self.current_pose_callback, 10)
        self.color_flag_sub = self.create_subscription(Float32, '/fsm_action', self.color_flag_callback, 10)
        
        self.left_setpoint_pub  = self.create_publisher(Float32, 'left/set_point', 10)
        self.right_setpoint_pub = self.create_publisher(Float32, 'right/set_point', 10)
        self.cmd_vel_pub        = self.create_publisher(Twist,   'ik_cmd_vel',   10)
        self.current_yaw_pub    = self.create_publisher(Float32, 'current_yaw',  10)

        self.current_pose = None
        self.timer = self.create_timer(self.sample_time, self.timer_callback)
        self.get_logger().info("Inverse Kinematics Node with position/orientation DeadZone")

        self.add_on_set_parameters_callback(self.parameter_callback)

    def current_pose_callback(self, msg: Odometry):
        self.current_pose = msg
    
    def color_flag_callback(self, msg: Float32):
        # self.get_logger().info(f"Received multiplier: {msg.data}")
        self.color_flag_multiplier = msg.data
        
    def timer_callback(self):
        if self.current_pose is None:
            return

        dt = self.sample_time
        # Deseados
        des_x, des_y, des_yaw = self.desired_x, self.desired_y, self.desired_yaw
        # Actuales
        cur_x = self.current_pose.pose.pose.position.x
        cur_y = self.current_pose.pose.pose.position.y
        cur_yaw = self.quaternion_to_yaw(self.current_pose.pose.pose.orientation)

        # Publicar yaw actual
        yaw_msg = Float32(data=cur_yaw)
        self.current_yaw_pub.publish(yaw_msg)

        # Error de posición en plano
        err_x = des_x - cur_x
        err_y = des_y - cur_y
        err_d = math.hypot(err_x, err_y)

        # Objetivo de orientación
        if err_d > self.position_threshold:
            heading_target = math.atan2(err_y, err_x)
        else:
            heading_target = des_yaw

        error_heading = self.normalize_angle(heading_target - cur_yaw)

        # PID orientación
        self.integral_error_ori += error_heading * dt
        d_ori = (error_heading - self.prev_error_ori) / dt
        self.prev_error_ori = error_heading
        pid_ori = (self.kP_ori * error_heading +
                   self.kI_ori * self.integral_error_ori +
                   self.kD_ori * d_ori)

        # Decidir velocidad lineal
        if abs(error_heading) > self.heading_threshold:
            V_d = 0.0
            # Anti-windup posición
            self.integral_error_pos = 0.0
            self.prev_error_pos     = err_d
        else:
            # PID posición
            self.integral_error_pos += err_d * dt
            d_pos = (err_d - self.prev_error_pos) / dt
            self.prev_error_pos = err_d
            pid_pos = (self.kP_pos * err_d +
                       self.kI_pos * self.integral_error_pos +
                       self.kD_pos * d_pos)
            V_d = pid_pos * math.cos(error_heading)

        # Aplicar deadzone interna
        if abs(error_heading) < self.orientation_deadzone:
            pid_ori = 0.0
            self.integral_error_ori = 0.0

        if err_d < self.position_deadzone:
            V_d = 0.0
            self.integral_error_pos = 0.0

        omega_d = pid_ori

        # Saturar velocidad lineal a ±0.23 m/s
        V_d = saturate(V_d, -self.lin_vel_mag_saturation, self.lin_vel_mag_saturation)
        V_d = V_d * self.color_flag_multiplier
        # Garantizar mínimo angular de ±0.3 rad/s si hay comando pequeño
        if abs(omega_d) > 1e-4 and abs(omega_d) < 0.3:
            omega_d = 0.3 * (1 if omega_d > 0 else -1)

        # Saturar velocidad angular a ±1.2 rad/s
        omega_d = saturate(omega_d, -self.ang_vel_mag_saturation, self.ang_vel_mag_saturation)
        omega_d = omega_d * self.color_flag_multiplier

        # Cálculo de setpoints de ruedas
        left_setpoint  = (V_d - (self.wheel_separation / 2.0) * omega_d) / self.wheel_radius
        right_setpoint = (V_d + (self.wheel_separation / 2.0) * omega_d) / self.wheel_radius

        # Publicar setpoints
        self.left_setpoint_pub.publish(Float32(data=left_setpoint))
        self.right_setpoint_pub.publish(Float32(data=right_setpoint))

        # Publicar Twist
        twist_msg = Twist()
        twist_msg.linear.x  = V_d
        twist_msg.angular.z = omega_d
        self.cmd_vel_pub.publish(twist_msg)

    def quaternion_to_yaw(self, q):
        q_array = [q.w, q.x, q.y, q.z]
        _, _, yaw = tfe.euler.quat2euler(q_array)
        return yaw

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def parameter_callback(self, params):
        for param in params:
            name = param.name
            val  = param.value

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
                self.timer.destroy()
                self.timer = self.create_timer(self.sample_time, self.timer_callback)

            elif name in ('kP_pos','kI_pos','kD_pos'):
                setattr(self, name, val)
            elif name in ('kP_ori','kI_ori','kD_ori'):
                setattr(self, name, val)
            elif name == 'heading_threshold':
                self.heading_threshold = val
            elif name == 'position_threshold':
                self.position_threshold = val
            elif name == 'desired_x':
                self.desired_x = val
            elif name == 'desired_y':
                self.desired_y = val
            elif name == 'desired_yaw':
                self.desired_yaw = val

        return SetParametersResult(successful=True)

def main(args=None):
    rclpy.init(args=args)
    node = InverseKinematics()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
