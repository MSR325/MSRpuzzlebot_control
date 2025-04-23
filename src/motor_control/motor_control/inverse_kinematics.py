#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
import math
from rcl_interfaces.msg import SetParametersResult
import transforms3d as tfe
from geometry_msgs.msg import Twist

def saturate(value, min_val, max_val):
    return max(min(value, max_val), min_val)

class InverseKinematics(Node):
    def __init__(self):
        super().__init__('inverse_kinematics')

        # Declarar parámetros de cinemática y del controlador
        self.declare_parameter('wheel_radius', 0.05)       # [m]
        self.declare_parameter('wheel_separation', 0.15)   # [m]
        self.declare_parameter('sample_time', 0.018)       # [s]

        # PID para posición
        self.declare_parameter('kP_pos', 0.1)
        self.declare_parameter('kI_pos', 0.0)
        self.declare_parameter('kD_pos', 0.05)

        # PID para orientación
        self.declare_parameter('kP_ori', 0.5)
        self.declare_parameter('kI_ori', 0.0)
        self.declare_parameter('kD_ori', 0.05)

        # Umbral para giro puro vs avance + giro
        self.declare_parameter('heading_threshold', 0.1)   # [rad]

        # Umbral para usar desired_yaw en lugar del vector a la meta
        self.declare_parameter('position_threshold', 0.1)  # [m]

        # Parámetros de la pose deseada
        self.declare_parameter('desired_x', 0.0)
        self.declare_parameter('desired_y', 0.0)
        self.declare_parameter('desired_yaw', 0.0)

        # -------------------------------------------------------------------------------------
        # NUEVO: Agregamos dos variables para las ZONAS MUERTAS (dead zones):
        # 1) para error de orientación
        # 2) para error de posición
        self.orientation_deadzone = 0.02  # rad (ej. ~1.15 grados)
        self.position_deadzone = 0.02    # m   (ajústalo a tu gusto)
        # -------------------------------------------------------------------------------------

        # Leer parámetros
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

        self.desired_x = self.get_parameter('desired_x').value
        self.desired_y = self.get_parameter('desired_y').value
        self.desired_yaw = self.get_parameter('desired_yaw').value

        # Variables internas para PID
        self.integral_error_pos = 0.0
        self.prev_error_pos = 0.0
        self.integral_error_ori = 0.0
        self.prev_error_ori = 0.0

        # Subscripción a la pose actual
        self.current_pose_sub = self.create_subscription(
            Odometry, 'odom', self.current_pose_callback, 10
        )

        # Publicadores de setpoints
        self.left_setpoint_pub = self.create_publisher(Float32, 'left/set_point', 10)
        self.right_setpoint_pub = self.create_publisher(Float32, 'right/set_point', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'ik_cmd_vel', 10)
        
        # Publicador para visualizar el yaw actual
        self.current_yaw_pub = self.create_publisher(Float32, 'current_yaw', 10)

        # Timer
        self.timer = self.create_timer(self.sample_time, self.timer_callback)
        self.get_logger().info("Inverse Kinematics Node with position/orientation DeadZone")
        self.current_pose = None

        # Callback dinámico
        self.add_on_set_parameters_callback(self.parameter_callback)

    def current_pose_callback(self, msg: Odometry):
        self.current_pose = msg

    def timer_callback(self):
        if self.current_pose is None:
            return

        dt = self.sample_time

        # Pose deseada
        des_x = self.desired_x
        des_y = self.desired_y
        des_yaw = self.desired_yaw

        # Pose actual
        cur_x = self.current_pose.pose.pose.position.x
        cur_y = self.current_pose.pose.pose.position.y
        cur_yaw = self.quaternion_to_yaw(self.current_pose.pose.pose.orientation)

        # Publicar yaw actual
        yaw_msg = Float32()
        yaw_msg.data = cur_yaw
        self.current_yaw_pub.publish(yaw_msg)

        # Cálculo de error de posición
        err_x = des_x - cur_x
        err_y = des_y - cur_y
        err_d = math.sqrt(err_x**2 + err_y**2)

        # Determinar heading_target
        if err_d > self.position_threshold:
            heading_target = math.atan2(err_y, err_x)
        else:
            heading_target = des_yaw

        # Error de orientación
        error_heading = self.normalize_angle(heading_target - cur_yaw)

        # ================= PID ORIENTACIÓN =================
        self.integral_error_ori += error_heading * dt
        d_ori = (error_heading - self.prev_error_ori) / dt
        self.prev_error_ori = error_heading
        pid_ori = (self.kP_ori * error_heading +
                   self.kI_ori * self.integral_error_ori +
                   self.kD_ori * d_ori)

        # Control secuencial: si error_heading grande => no avanza
        if abs(error_heading) > self.heading_threshold:
            # SOLO gira
            V_d = 0.0
            self.integral_error_pos = 0.0
            self.prev_error_pos = err_d
        else:
            # ================= PID POSICIÓN =================
            self.integral_error_pos += err_d * dt
            d_pos = (err_d - self.prev_error_pos) / dt
            self.prev_error_pos = err_d
            pid_pos = (self.kP_pos * err_d +
                       self.kI_pos * self.integral_error_pos +
                       self.kD_pos * d_pos)
            V_d = pid_pos * math.cos(error_heading)

        # ============= ZONA MUERTA DE ORIENTACIÓN ============
        # Si el error de orientación < orientation_deadzone => anular
        if abs(error_heading) < self.orientation_deadzone:
            error_heading = 0.0
            pid_ori = 0.0
            self.integral_error_ori = 0.0

        # ============= ZONA MUERTA DE POSICIÓN ================
        # Si la distancia err_d < position_deadzone => anular velocidad lineal
        if err_d < self.position_deadzone:
            V_d = 0.0
            self.integral_error_pos = 0.0

        # Salida de PID orient -> omega
        omega_d = pid_ori

        # Saturación
        V_d = saturate(V_d, -0.3, 0.3)
        omega_d = saturate(omega_d, -1.0, 1.0)

        # Cinemática inversa
        left_setpoint = (V_d - (self.wheel_separation / 2.0) * omega_d) / self.wheel_radius
        right_setpoint = (V_d + (self.wheel_separation / 2.0) * omega_d) / self.wheel_radius

        # Publicar
        l_msg = Float32()
        l_msg.data = left_setpoint
        self.left_setpoint_pub.publish(l_msg)

        r_msg = Float32()
        r_msg.data = right_setpoint
        self.right_setpoint_pub.publish(r_msg)

        # Publicar también el Twist con (v, ω) para el robot real
        twist_msg = Twist()
        twist_msg.linear.x = V_d      # m/s
        twist_msg.angular.z = omega_d # rad/s
        self.cmd_vel_pub.publish(twist_msg)

    def quaternion_to_yaw(self, q):
        # Reordena [w, x, y, z] para transforms3d
        q_array = [q.w, q.x, q.y, q.z]
        roll, pitch, yaw = tfe.euler.quat2euler(q_array)
        return yaw

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'wheel_radius':
                if param.value <= 0.0:
                    return SetParametersResult(False, "wheel_radius must be > 0")
                self.wheel_radius = param.value

            elif param.name == 'wheel_separation':
                if param.value <= 0.0:
                    return SetParametersResult(False, "wheel_separation must be > 0")
                self.wheel_separation = param.value

            elif param.name == 'sample_time':
                if param.value <= 0.0:
                    return SetParametersResult(False, "sample_time must be > 0")
                self.sample_time = param.value
                self.timer.destroy()
                self.timer = self.create_timer(self.sample_time, self.timer_callback)

            elif param.name == 'kP_pos':
                self.kP_pos = param.value
            elif param.name == 'kI_pos':
                self.kI_pos = param.value
            elif param.name == 'kD_pos':
                self.kD_pos = param.value

            elif param.name == 'kP_ori':
                self.kP_ori = param.value
            elif param.name == 'kI_ori':
                self.kI_ori = param.value
            elif param.name == 'kD_ori':
                self.kD_ori = param.value

            elif param.name == 'heading_threshold':
                self.heading_threshold = param.value
            elif param.name == 'position_threshold':
                self.position_threshold = param.value

            elif param.name == 'desired_x':
                self.desired_x = param.value
            elif param.name == 'desired_y':
                self.desired_y = param.value
            elif param.name == 'desired_yaw':
                self.desired_yaw = param.value

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
