#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32MultiArray, Bool
from nav_msgs.msg import Path as PathMsg
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from custom_interfaces.srv import SwitchPublisher
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from ament_index_python.packages import get_package_share_directory
from pathlib import Path
from geometry_msgs.msg import Twist
from std_msgs.msg import Int16  # for enable_line_follow


class TurnManager(Node):
    def __init__(self):
        super().__init__('turn_manager')

        # ---------- ROI warp (300×300 px) y escala física ----------
        self.warp_w          = 300          # px
        self.warp_h          = 300          # px
        self.x_meter_range   = 0.30         # 30 cm de lado a lado
        self.y_meter_range   = 0.264        # 26.4 cm de profundidad
        self.y_offset_m      = 0.15         # warp inicia 15 cm delante del robot

        # ---------- parámetros ROS ----------
        self.declare_parameter('homography_matrix_path', 'data/homography_after_calib_firstsegment_decent.npy')
        self.declare_parameter('waypoint_count',    5)
        self.declare_parameter('arrival_tolerance', 0.02)
        self.declare_parameter('flip_warp_x', False)
        self.declare_parameter('flip_warp_y', False)

        rel_path = self.get_parameter('homography_matrix_path').value
        pkg_share = get_package_share_directory('line_follow_msr')
        h_path = str((Path(pkg_share) / rel_path).resolve())

        self.num_wp = self.get_parameter('waypoint_count').value
        self.tol    = self.get_parameter('arrival_tolerance').value
        self.flip_x = self.get_parameter('flip_warp_x').value
        self.flip_y = self.get_parameter('flip_warp_y').value

        # ---------- homografías ----------
        try:
            H_img2warp = np.load(h_path)             # imagen → warp
            self.H_warp2img = np.linalg.inv(H_img2warp)  # warp → imagen
            self.get_logger().info(f'✅ Homography matrix loaded from: {h_path}')
        except Exception as e:
            self.get_logger().error(f'❌ Failed to load homography: {e}')
            self.H_warp2img = np.eye(3)

        # ---------- estado ----------
        self.current_event     = None
        self.processing        = False
        self.current_waypoints = []
        self.pose              = None
        self.stability_frames  = 4
        self._centroid_buffer  = []   # will store last few (y_fwd, x_lat)
        self.variance_threshold= 0.05

        # ---------- QoS cámara ----------
        sensor_qos = QoSProfile(
            history     = QoSHistoryPolicy.KEEP_LAST,
            reliability = QoSReliabilityPolicy.BEST_EFFORT,
            depth       = 10,
        )

        # ---------- I/O ROS ----------
        self.create_subscription(String,         '/fsm_event',            self.event_cb, 10)
        self.create_subscription(Int32MultiArray,'/crossroad_centroids',  self.cross_cb, 10)
        self.create_subscription(Odometry,       '/odom',                 self.odom_cb, 10)
        self.bridge = CvBridge()
        self.create_subscription(Image,          '/image_raw',            self.image_cb, sensor_qos)

        self.wp_pub  = self.create_publisher(PathMsg,  '/turn_manager/waypoints', 10)
        self.img_pub = self.create_publisher(Image, '/turn_manager/debug_image', 10)
        self.vel_pub = self.create_publisher(Twist, '/ik_cmd_vel', 10)
        self.enable_pub = self.create_publisher(Int16, '/line_follow_enable', 10)

        self.cli = self.create_client(SwitchPublisher, 'switch_cmd_source')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Esperando servicio switch_cmd_source...")

        self.create_subscription(Bool, '/completed_curve', self.curve_done_cb, 10)
        self.get_logger().info("TurnManager listo")

    # --------------------------------------------------  CALLBACKS BÁSICOS
    def event_cb(self, msg: String):
        if msg.data != self.current_event:
            self._centroid_buffer.clear()   # brand-new manoeuvre
        self.current_event = msg.data
        self.get_logger().info(f"[FSM] Evento: {self.current_event}")


    def odom_cb(self, msg: Odometry):
        self.pose = msg.pose.pose.position        # x = adelante, y = izquierda

    # --------------------------------------------------  CROSSROAD → WAYPOINTS
    def cross_cb(self, msg: Int32MultiArray): 

        if self.processing is True or self.pose is None or self.current_event is None:
            return
        
        Hx, Hy, Lx, Ly, Rx, Ry = msg.data
        if self.current_event == 'FORWARD':
            if Hx >= 9999: return
            px_w, py_w = Hx, Hy
        elif self.current_event == 'LEFT_TURN':
            if Lx >= 9999: return
            px_w, py_w = Lx, Ly
        elif self.current_event == 'RIGHT_TURN':
            if Rx >= 9999: return
            px_w, py_w = Rx, Ry

        # ---------- aplicar flips coherentes ----------
        if self.flip_x:
            px_w = self.warp_w - 1 - px_w
        if self.flip_y:
            py_w = self.warp_h - 1 - py_w

        y_fwd =  py_w * (self.y_meter_range / self.warp_h)   # metres forward
        x_lat = (px_w - self.warp_w/2) * (self.x_meter_range / self.warp_w)  # metres left(+)/right(-)

        self._centroid_buffer.append((x_lat, y_fwd))
        if len(self._centroid_buffer) > self.stability_frames:
            self._centroid_buffer.pop(0)

        # only proceed if the buffer is "stable"
        if len(self._centroid_buffer) < self.stability_frames:
            return
        var_u = np.var([c[0] for c in self._centroid_buffer])
        var_v = np.var([c[1] for c in self._centroid_buffer])
        if max(var_u, var_v) > self.variance_threshold:      # centroid variance threshold, tune as needed
            return                       # still wobbling → wait

        self.current_waypoints = self.build_waypoints(target_x = y_fwd,
                                                      target_y = x_lat,
                                                      event    = self.current_event)
        self.get_logger().info("🕒 Esperando 1.5 segundos antes de ejecutar la curva...")
        timer = self.create_timer(
            1.5, self._start_trajectory_once, callback_group=None, 
            timer_period_sec=0.0  # ignored, safe to leave
        )
        timer.cancel()  # cancel immediately to prevent reuse
        
        self.processing = True
        self._centroid_buffer.clear()

    def _start_trajectory_once(self):
        if not self.processing:
            return  # already canceled or reset
        
            # ❌ Desactiva el seguidor de línea
        self.enable_pub.publish(Int16(data=0))
        
        self.publish_path()
        self.call_switch('ik')
        self.get_logger().info("🚀 Trayectoria activada después del retraso")
        
        # prevent multiple timer firings (ROS2 doesn't guarantee oneshot behavior by default)
        self.processing = 'started'


    # --------------------------------------------------  IMAGE DEBUG
    def image_cb(self, msg: Image):
        if not (self.processing and self.current_waypoints):
            return
        cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        h_img, w_img = cv_img.shape[:2]

        # ---------- métricas → pix warp (vector) ----------
        pts_warp = np.empty((len(self.current_waypoints), 1, 2), dtype=np.float32)
        for k, ps in enumerate(self.current_waypoints):
            u = ps.pose.position.y * (self.warp_w/self.x_meter_range) + self.warp_w/2
            v = (ps.pose.position.x - self.y_offset_m) * (self.warp_h/self.y_meter_range)
            if self.flip_x: u = self.warp_w - 1 - u
            if self.flip_y: v = self.warp_h - 1 - v
            pts_warp[k,0] = (u, v)

        # ---------- warp → imagen ----------
        pts_img = cv2.perspectiveTransform(pts_warp, self.H_warp2img)\
                        .round().astype(int).reshape(-1,2)

        # ---------- dibujar ----------
        for k, (u,v) in enumerate(pts_img):
            if 0 <= u < w_img and 0 <= v < h_img:
                if k:
                    cv2.line(cv_img, tuple(pts_img[k-1]), (u,v), (0,0,255), 1)
                cv2.circle(cv_img, (u,v), 5, (0,0,255), -1)
                cv2.putText(cv_img, str(k+1), (u+5, v-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        self.img_pub.publish(self.bridge.cv2_to_imgmsg(cv_img, 'bgr8',
                                                       header=msg.header))

    # --------------------------------------------------  GENERAR BÉZIER
    def build_waypoints(self, target_x, target_y, event):
        now = self.get_clock().now().to_msg()

        # pose inicial ajustada 0.15 m adelante
        x0 = self.pose.x + self.y_offset_m
        y0 = self.pose.y

        # objetivo con mismo offset
        x3 = target_x + self.y_offset_m
        y3 = target_y

        vx, vy = x3 - x0, y3 - y0
        d      = math.hypot(vx, vy)

        if d < 1e-3:
            ctrl = ((x0+x3)/2, (y0+y3)/2)
        else:
            if event == 'LEFT_TURN':
                perpx, perpy = -vy/d,  vx/d
            elif event == 'RIGHT_TURN':
                perpx, perpy =  vy/d, -vx/d
            else:
                perpx, perpy = 0.0, 0.0
            R = min(d, 0.10)          # radio máx 25 cm
            ctrl = ((x0+x3)/2 + perpx*R,
                    (y0+y3)/2 + perpy*R)

        P0, P1, P2 = (x0,y0), ctrl, (x3,y3)
        wayps = []
        for i in range(self.num_wp):
            t  = i/(self.num_wp-1)
            bx = (1-t)**2*P0[0] + 2*(1-t)*t*P1[0] + t**2*P2[0]
            by = (1-t)**2*P0[1] + 2*(1-t)*t*P1[1] + t**2*P2[1]
            ps = PoseStamped()
            ps.header.frame_id = 'odom'
            ps.header.stamp    = now
            ps.pose.position.x = bx
            ps.pose.position.y = by
            wayps.append(ps)
        return wayps

    # --------------------------------------------------  PUBLICAR PATH
    def publish_path(self):
        path = PathMsg()
        path.header.frame_id = 'odom'
        path.header.stamp    = self.get_clock().now().to_msg()
        path.poses           = self.current_waypoints
        self.wp_pub.publish(path)
        self.get_logger().info(f"Path publicado ({len(path.poses)} WPs)")

    # --------------------------------------------------  CAMBIAR MUX
    def call_switch(self, src):
        req = SwitchPublisher.Request(); req.active_source = src
        self.cli.call_async(req)

    # --------------------------------------------------  CALLBACK → CURVE DONE
    def curve_done_cb(self, msg: Bool):
        if msg.data and self.processing:
            self.call_switch('line')
            self.processing = False
            self.get_logger().info("🟢 Trayectoria completada → line follower activado")

            # 🛑 Publish zero velocity to stop the robot
            stop_twist = Twist()
            self.vel_pub.publish(stop_twist)

            # ✅ Reactivate the line follower
            self.enable_pub.publish(Int16(data=1))

# ------------------------------------------------------  MAIN
def main(args=None):
    rclpy.init(args=args)
    node = TurnManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
