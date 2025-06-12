#!/usr/bin/env python3
import rclpy, cv2, math, numpy as np
from rclpy.node         import Node
from rclpy.qos          import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg       import String, Int32MultiArray, Bool, Int16
from nav_msgs.msg       import Path as PathMsg, Odometry
from geometry_msgs.msg  import PoseStamped, Twist
from sensor_msgs.msg    import Image
from cv_bridge          import CvBridge
from pathlib            import Path
from ament_index_python.packages import get_package_share_directory

class TurnManager(Node):
    # ------------------------------------------------------------------ init
    def __init__(self):
        super().__init__('turn_manager')

        # ---------- ROI warp (300×300 px) y escala física ----------
        self.warp_w          = 300          # px
        self.warp_h          = 300          # px
        self.x_meter_range   = 0.30         # 30 cm de lado a lado
        self.y_meter_range   = 0.264        # 26.4 cm de profundidad
        self.y_offset_m      = 0.15         # warp inicia 15 cm delante del robot
        h_path = 'src/line_follow_msr/data/homography_after_calib_firstsegment_decent.npy'
        H_img2warp         = np.load(h_path)          # imagen → warp
        self.H_warp2img    = np.linalg.inv(H_img2warp)# warp   → imagen
        self._centroid_buffer   = []
        self.stability_frames   = 4         # how many detections to accumulate
        self.variance_threshold = 0.05      # maximum allowed variance (meters²)

        self.flip_x = False
        self.flip_y = False

        # --- state --------------------------------------------------------
        self.once_timer        = None
        self.current_event     = None
        self.processing        = False
        self.current_waypoints = []
        self.pose              = None
        self.stability_frames  = 4
        self._centroid_buffer  = []
        self.variance_threshold= 0.05

        self.declare_parameter('waypoint_count', 5)
        self.num_wp = self.get_parameter('waypoint_count').value

        self.bridge = CvBridge()

        # --- SUBSCRIBERS --------------------------------------------------
        self.create_subscription(String,         '/fsm_event',           self.event_cb,   10)
        self.create_subscription(Int32MultiArray,'/crossroad_centroids', self.cross_cb,   10)
        self.create_subscription(Odometry,       '/odom',                self.odom_cb,    10)
        self.create_subscription(Bool,           '/completed_curve',     self.curve_done_cb,10)
        self.create_subscription(Int32MultiArray,'/crossroad_detected',  self.crossroad_detected_cb, 10)
        sensor_qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST,
                                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                depth=10)
        self.create_subscription(Image,          '/image_raw',           self.image_cb,   sensor_qos)

        # --- PUBLISHERS ---------------------------------------------------
        self.wp_pub        = self.create_publisher(PathMsg, '/turn_manager/waypoints', 10)
        self.enable_pub    = self.create_publisher(Int16,   '/line_follow_enable',     10)
        self.line_vel_pub  = self.create_publisher(Twist,   '/line_cmd_vel',           10)
        self.img_pub = self.create_publisher(Image, '/turn_manager/debug_image',10)

        self.get_logger().info("TurnManager initialized.")

    # ================================================================ CBs ===
    def crossroad_detected_cb(self, msg: Int32MultiArray):
        found_h, found_left, found_right = msg.data
        if (found_h or found_left or found_right) and not self.processing:
            self.enable_pub.publish(Int16(data=0))
            

    def event_cb(self, msg: String):
        valid_events = {'LEFT_TURN', 'RIGHT_TURN', 'FORWARD'}
        
        if self.once_timer:
            self.once_timer.cancel(); self.once_timer = None
        if msg.data not in valid_events:
            return  
        if msg.data != self.current_event:
            self._centroid_buffer.clear()
        self.current_event = msg.data

    def odom_cb(self, msg: Odometry):
        self.pose = msg.pose.pose.position

    def cross_cb(self, msg: Int32MultiArray):
        if self.processing or self.pose is None or self.current_event is None:
            return

        Hx, Hy, Lx, Ly, Rx, Ry = msg.data
        if   self.current_event=='FORWARD':   px_w, py_w = (Hx,Hy) if Hx<9999 else (None,None)
        elif self.current_event=='LEFT_TURN': px_w, py_w = (Lx,Ly) if Lx<9999 else (None,None)
        else:                                 px_w, py_w = (Rx,Ry) if Rx<9999 else (None,None)
        if px_w is None:  return

        # ---------- aplicar flips coherentes ----------
        if self.flip_x:
            px_w = self.warp_w - 1 - px_w
        if self.flip_y:
            py_w = self.warp_h - 1 - py_w

        # --- Convert centroid to metric coordinates
        y_fwd = py_w * (self.y_meter_range / self.warp_h)  # meters forward
        x_lat = (px_w - self.warp_w / 2) * (self.x_meter_range / self.warp_w)  # meters lateral

        # --- Update buffer
        self._centroid_buffer.append((x_lat, y_fwd))
        if len(self._centroid_buffer) > self.stability_frames:
            self._centroid_buffer.pop(0)

        # --- Check if buffer is full
        if len(self._centroid_buffer) < self.stability_frames:
            return

        # --- Variance check
        var_x = np.var([c[0] for c in self._centroid_buffer])
        var_y = np.var([c[1] for c in self._centroid_buffer])
        if max(var_x, var_y) > self.variance_threshold:
            return  # still wobbling → wait

        self.current_waypoints = self.build_waypoints(x_lat, y_fwd, self.current_event)
        self.processing = True
        self._start_trajectory_once_once()
        self._centroid_buffer.clear()

    def _start_trajectory_once_once(self):
        self.enable_pub.publish(Int16(data=0))
        self.publish_path()
        twist = Twist()
        twist.linear.x  = 0.0
        twist.angular.z = 0.0
        self.line_vel_pub.publish(twist)

    def publish_path(self):
        path = PathMsg()
        path.header.frame_id = 'odom'
        path.header.stamp    = self.get_clock().now().to_msg()
        path.poses           = self.current_waypoints
        self.wp_pub.publish(path)
        self.get_logger().info(f"Path publicado ({len(path.poses)} WPs) para evento: {self.current_event}")

    def curve_done_cb(self, msg: Bool):
        if msg.data and self.processing:
            self.processing = False
            self.enable_pub.publish(Int16(data=1))
            self.current_event = None
            self.get_logger().info("✅ Curva completada, reactivando seguidor de línea")

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

        self.img_pub.publish(self.bridge.cv2_to_imgmsg(cv_img, 'bgr8',header=msg.header))

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
    
def main(args=None):
    rclpy.init(args=args)
    node = TurnManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
