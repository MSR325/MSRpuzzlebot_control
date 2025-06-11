#!/usr/bin/env python3
"""
ik_path_follower.py  –  Pure-Pursuit + PID desacoplado
— sin ramp limiter, con límite de curvatura y flujo de waypoints —
junio 2025
"""

import rclpy, math
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Bool
from rcl_interfaces.msg import SetParametersResult

class PathFollower(Node):
    def __init__(self):
        super().__init__('path_follower')

        # ---- DECLARACIÓN DE PARÁMETROS DINÁMICOS ----
        params = {
            'lookahead':   0.05,    # [m] distancia look-ahead
            'v_max':       0.17,    # [m/s] velocidad lineal máxima
            'w_max':       0.50,    # [rad/s] velocidad angular máxima
            'arrival_tol': 0.02,    # [m] tolerancia para considerar WP alcanzado
            'wheel_base':  0.175,   # [m] distancia entre ejes
            # PID para control lineal
            'kp_lin':      1.0,
            'ki_lin':      0.1,
            'kd_lin':      0.01,
            # PID para control angular
            'kp_ang':      2.0,
            'ki_ang':      0.1,
            'kd_ang':      0.01,
        }
        for name, default in params.items():
            self.declare_parameter(name, default)

        # sincroniza valores en variables internas
        self._sync_params()
        # callback para reaccionar a cambios en tiempo real
        self.add_on_set_parameters_callback(self._on_param_update)

        # ---- ESTADO PARA PID ----
        self.lin_int      = 0.0
        self.lin_prev_err = 0.0
        self.ang_int      = 0.0
        self.ang_prev_err = 0.0
        self.last_time    = self.get_clock().now()

        # ---- WAYPOINTS & POSE ----
        self.current_path = []   # lista de PoseStamped
        self.pose         = None # se llena desde /odom

        # ---- SUSCRIPCIONES & PUBLICADORES ----
        self.create_subscription(Path,     '/turn_manager/waypoints', self._path_cb, 10)
        self.create_subscription(Odometry, '/odom',                  self._odom_cb,  40)
        self.cmd_pub  = self.create_publisher(Twist, 'ik_cmd_vel',       10)
        self.done_pub = self.create_publisher(Bool,  'completed_curve', 10)

        # ---- BUCLE DE CONTROL 100 Hz ----
        self.create_timer(0.01, self.control_loop)
        self.get_logger().info('🚀 ik_path_follower listo (100 Hz, PID+PP).')

    def _sync_params(self):
        p = self.get_parameter
        self.lookahead   = p('lookahead').value
        self.v_max       = p('v_max').value
        self.w_max       = p('w_max').value
        self.arrival_tol = p('arrival_tol').value
        self.wheel_base  = p('wheel_base').value
        # curvátura máxima física = 2 / wheel_base
        self.kappa_max   = 2.0 / self.wheel_base

        # ganancias PID
        self.kp_lin, self.ki_lin, self.kd_lin = (
            p('kp_lin').value, p('ki_lin').value, p('kd_lin').value
        )
        self.kp_ang, self.ki_ang, self.kd_ang = (
            p('kp_ang').value, p('ki_ang').value, p('kd_ang').value
        )

    def _on_param_update(self, params):
        self._sync_params()
        return SetParametersResult(successful=True)

    def _path_cb(self, msg: Path):
        self.current_path = list(msg.poses)
        self.get_logger().info(f'Nueva trayectoria: {len(self.current_path)} WPs')

    def _odom_cb(self, msg: Odometry):
        self.pose = msg.pose.pose

    def control_loop(self):
        if self.pose is None:
            return

        now = self.get_clock().now()
        dt  = (now - self.last_time).nanoseconds * 1e-9
        if dt <= 0.0:
            return
        self.last_time = now

        x   = self.pose.position.x
        y   = self.pose.position.y
        yaw = self._yaw_from_quat(self.pose.orientation)

        # 1) Pop waypoints alcanzados
        while self.current_path and \
              self._dist(self.current_path[0].pose.position, x, y) < self.arrival_tol:
            self.current_path.pop(0)

        # 2) Si no quedan WPs → detener y señal de done
        if not self.current_path:
            self.cmd_pub.publish(Twist())
            self.done_pub.publish(Bool(data=True))
            # reset integradores
            self.lin_int = self.ang_int = 0.0
            return

        # 3) Seleccionar objetivo > lookahead
        target = self.current_path[0].pose.position
        for wp in self.current_path:
            if self._dist(wp.pose.position, x, y) > self.lookahead:
                target = wp.pose.position
                break

        dx       = target.x - x
        dy       = target.y - y
        dist_err = math.hypot(dx, dy)
        ang_err  = self._wrap(math.atan2(dy, dx) - yaw)

        # 4) PID lineal
        self.lin_int     += dist_err * dt
        lin_der         = (dist_err - self.lin_prev_err) / dt
        v_cmd           = (self.kp_lin * dist_err +
                           self.ki_lin * self.lin_int +
                           self.kd_lin * lin_der)
        self.lin_prev_err = dist_err

        # 5) PID angular
        self.ang_int      += ang_err * dt
        ang_der          = (ang_err - self.ang_prev_err) / dt
        w_cmd            = (self.kp_ang * ang_err +
                            self.ki_ang * self.ang_int +
                            self.kd_ang * ang_der)
        self.ang_prev_err = ang_err

        # 6) Saturación + limitación de curvatura
        v_cmd = max(-self.v_max, min(self.v_max, v_cmd))
        w_cmd = max(-self.w_max, min(self.w_max, w_cmd))
        κ_lim = abs(v_cmd * self.kappa_max)
        w_cmd = max(-min(self.w_max, κ_lim), min(min(self.w_max, κ_lim), w_cmd))

        # 7) Publicar Twist
        twist = Twist()
        twist.linear.x  = v_cmd
        twist.angular.z = w_cmd
        self.cmd_pub.publish(twist)

    @staticmethod
    def _yaw_from_quat(q):
        return math.atan2(
            2*(q.w*q.z + q.x*q.y),
            1 - 2*(q.y*q.y + q.z*q.z)
        )

    @staticmethod
    def _wrap(angle):
        return (angle + math.pi) % (2*math.pi) - math.pi

    @staticmethod
    def _dist(p, x, y):
        return math.hypot(p.x - x, p.y - y)


def main(args=None):
    rclpy.init(args=args)
    node = PathFollower()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
