#!/usr/bin/env python3
"""
ik_path_follower.py  –  Pure-Pursuit **sin ramp-limiter**
junio 2025
"""

import rclpy, math, numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Bool
from rcl_interfaces.msg import SetParametersResult


class PathFollower(Node):
    # --------------------------------------------------  INIT
    def __init__(self):
        super().__init__('path_follower')

        # ---------------- parámetros dinámicos ----------------
        self.declare_parameter('lookahead',      0.05)   # m
        self.declare_parameter('v_ref',          0.175)  # m/s
        self.declare_parameter('v_max',          0.50)   # m/s
        self.declare_parameter('w_max',          4.00)   # rad/s
        self.declare_parameter('arrival_tol',    0.02)   # m

        self._sync_params()
        self.add_on_set_parameters_callback(self._on_param_update)

        # ---------------- estado ----------------
        self.current_path: list[PoseStamped] = []
        self.pose: PoseStamped | None = None

        # ---------------- I/O ----------------
        self.create_subscription(Path,     '/turn_manager/waypoints',
                                  self._path_cb,   10)
        self.create_subscription(Odometry, '/odom',
                                  self._odom_cb,   40)
        self.cmd_pub  = self.create_publisher(Twist, 'cmd_vel', 10)
        self.done_pub = self.create_publisher(Bool,  'completed_curve', 10)

        # ---------------- loop 100 Hz ----------------
        self.control_timer = self.create_timer(0.01, self.control_loop)
        self.get_logger().info('🚀 ik_path_follower listo (100 Hz, sin ramp).')

    # --------------------------------------------------  PARAM SYNC
    def _sync_params(self):
        p = self.get_parameter
        self.lookahead   = p('lookahead').value
        self.v_ref       = p('v_ref').value
        self.v_max       = p('v_max').value
        self.w_max       = p('w_max').value
        self.arrival_tol = p('arrival_tol').value

    def _on_param_update(self, params):
        self._sync_params()
        return SetParametersResult(successful=True)

    # --------------------------------------------------  CBs
    def _path_cb(self, msg: Path):
        self.current_path = list(msg.poses)
        self.get_logger().info(f'Nueva trayectoria: {len(self.current_path)} WPs')

    def _odom_cb(self, msg: Odometry):
        self.pose = msg.pose.pose

    # --------------------------------------------------  PURE-PURSUIT LOOP
    def control_loop(self):
        if self.pose is None or not self.current_path:
            return

        # 1) Pose actual
        x   = self.pose.position.x
        y   = self.pose.position.y
        yaw = self._yaw_from_quat(self.pose.orientation)

        # 2) Elimina WPs ya alcanzados
        while (self.current_path and
               self._dist_xy(self.current_path[0].pose.position, x, y)
               < self.lookahead / 2):
            self.current_path.pop(0)

        if not self.current_path:
            # ✅ Terminado
            self.cmd_pub.publish(Twist())           # stop
            self.done_pub.publish(Bool(data=True))  # evento
            return

        # 3) Punto objetivo = primer WP con dist > lookahead
        target = self.current_path[0].pose.position
        for wp in self.current_path:
            if self._dist_xy(wp.pose.position, x, y) > self.lookahead:
                target = wp.pose.position
                break

        # 4) Geometría Pure-Pursuit
        dx    = target.x - x
        dy    = target.y - y
        ld    = math.hypot(dx, dy)
        alpha = self._wrap(math.atan2(dy, dx) - yaw)

        v_cmd = self.v_ref
        w_cmd = 2.0 * v_cmd * math.sin(alpha) / self.lookahead

        # 5) Saturar velocidades
        v_cmd = max(-self.v_max, min(self.v_max, v_cmd))
        w_cmd = max(-self.w_max, min(self.w_max, w_cmd))

        # 6) Publicar comando (sin ramp)
        twist = Twist()
        twist.linear.x  = v_cmd
        twist.angular.z = w_cmd
        self.cmd_pub.publish(twist)

        # 7) Detectar fin de trayectoria
        last_wp = self.current_path[-1].pose.position
        if self._dist_xy(last_wp, x, y) < self.arrival_tol:
            self.current_path.clear()  # siguiente ciclo enviará stop

    # --------------------------------------------------  UTILS
    @staticmethod
    def _yaw_from_quat(q):
        return math.atan2(2 * (q.w * q.z + q.x * q.y),
                          1 - 2 * (q.y * q.y + q.z * q.z))

    @staticmethod
    def _wrap(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    @staticmethod
    def _dist_xy(p, x, y):
        return math.hypot(p.x - x, p.y - y)


# ------------------------------------------------------  MAIN
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
