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

class CubicControl(Node):
    def __init__(self):
        super().__init__('cubic_control')

        # Parámetros de cinemática
        self.declare_parameter('wheel_radius',       0.05)
        self.declare_parameter('wheel_separation',   0.15)
        self.declare_parameter('sample_time',        0.018)

        # Umbrales y deadzones
        self.declare_parameter('heading_threshold',   0.1)
        self.declare_parameter('position_threshold',  0.1)
        self.declare_parameter('orientation_deadzone',0.02)
        self.declare_parameter('position_deadzone',   0.02)

        # Poses deseadas
        self.declare_parameter('desired_x',    0.0)
        self.declare_parameter('desired_y',    0.0)
        self.declare_parameter('desired_yaw',  0.0)

        # Ganancias cúbicas
        self.declare_parameter('k_pos', 1.0)    # [m/s per (m³)]
        self.declare_parameter('k_ori', 2.0)    # [rad/s per (rad³)]

        # Carga de parámetros
        self.wheel_radius        = self.get_parameter('wheel_radius').value
        self.wheel_separation    = self.get_parameter('wheel_separation').value
        self.sample_time         = self.get_parameter('sample_time').value

        self.heading_threshold   = self.get_parameter('heading_threshold').value
        self.position_threshold  = self.get_parameter('position_threshold').value
        self.orientation_deadzone= self.get_parameter('orientation_deadzone').value
        self.position_deadzone   = self.get_parameter('position_deadzone').value

        self.desired_x           = self.get_parameter('desired_x').value
        self.desired_y           = self.get_parameter('desired_y').value
        self.desired_yaw         = self.get_parameter('desired_yaw').value

        self.k_pos               = self.get_parameter('k_pos').value
        self.k_ori               = self.get_parameter('k_ori').value

        # Subscripciones y publicaciones
        self.current_pose_sub = self.create_subscription(
            Odometry, 'odom', self.current_pose_callback, 10
        )
        self.left_setpoint_pub  = self.create_publisher(Float32, 'left/set_point',  10)
        self.right_setpoint_pub = self.create_publisher(Float32, 'right/set_point', 10)
        self.cmd_vel_pub        = self.create_publisher(Twist,   '/pseudo_cmd_vel',10)
        self.current_yaw_pub    = self.create_publisher(Float32, 'current_yaw',     10)

        self.current_pose = None
        self.timer = self.create_timer(self.sample_time, self.timer_callback)
        self.get_logger().info("CubicControl Node iniciado con ley cúbica y publicación en /pseudo_cmd_vel")

        self.add_on_set_parameters_callback(self.parameter_callback)

    def current_pose_callback(self, msg: Odometry):
        self.current_pose = msg

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
        self.current_yaw_pub.publish(Float32(data=cur_yaw))

        # Error de posición planar
        err_x = des_x - cur_x
        err_y = des_y - cur_y
        err_d = math.hypot(err_x, err_y)

        # Objetivo de orientación
        if err_d > self.position_threshold:
            heading_target = math.atan2(err_y, err_x)
        else:
            heading_target = des_yaw
        error_heading = self.normalize_angle(heading_target - cur_yaw)

        # --- Ley cúbica ---
        # Velocidad lineal
        if err_d < self.position_deadzone or abs(error_heading) > self.heading_threshold:
            V_d = 0.0
        else:
            V_d = -self.k_pos * (err_d ** 3) * math.cos(error_heading)

        # Velocidad angular
        if abs(error_heading) < self.orientation_deadzone:
            omega_d = 0.0
        else:
            omega_d = -self.k_ori * (error_heading ** 3)

        # Saturaciones
        V_d     = saturate(V_d,   -0.23,  0.23)
        if abs(omega_d) > 1e-4 and abs(omega_d) < 0.3:
            omega_d = 0.3 * (1 if omega_d > 0 else -1)
        omega_d = saturate(omega_d, -1.2, 1.2)

        # Setpoints de ruedas (opcional)
        left_sp  = (V_d - (self.wheel_separation/2.0)*omega_d) / self.wheel_radius
        right_sp = (V_d + (self.wheel_separation/2.0)*omega_d) / self.wheel_radius

        self.left_setpoint_pub.publish(Float32(data=left_sp))
        self.right_setpoint_pub.publish(Float32(data=right_sp))

        # Publicar pseudo comando
        twist = Twist()
        twist.linear.x  = V_d
        twist.angular.z = omega_d
        self.cmd_vel_pub.publish(twist)

    def quaternion_to_yaw(self, q):
        q_arr = [q.w, q.x, q.y, q.z]
        _, _, yaw = tfe.euler.quat2euler(q_arr)
        return yaw

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def parameter_callback(self, params):
        for p in params:
            name, val = p.name, p.value
            if name == 'wheel_radius' and val > 0.0:
                self.wheel_radius = val
            elif name == 'wheel_separation' and val > 0.0:
                self.wheel_separation = val
            elif name == 'sample_time' and val > 0.0:
                self.sample_time = val
                self.timer.destroy()
                self.timer = self.create_timer(self.sample_time, self.timer_callback)
            elif name in ('heading_threshold','position_threshold',
                          'orientation_deadzone','position_deadzone',
                          'desired_x','desired_y','desired_yaw',
                          'k_pos','k_ori'):
                setattr(self, name, val)
        return SetParametersResult(successful=True)

def main(args=None):
    rclpy.init(args=args)
    node = CubicControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
