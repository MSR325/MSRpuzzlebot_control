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

        # --- state --------------------------------------------------------
        self.once_timer        = None
        self.current_event     = None
        self.processing        = False
        self._awaiting_done    = False
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
        # NEW: ask TrajectoryCommander to switch cmd source
        self.cmd_req_pub   = self.create_publisher(String,  '/cmd_source_request',     10)

        self.get_logger().info("TurnManager initialized.")

    # ================================================================ CBs ===
    def crossroad_detected_cb(self, msg: Int32MultiArray):
        found_h, found_left, found_right = msg.data
        if (found_h or found_left or found_right) and not self.processing:
            self.get_logger().info(f"🚦 Dotted line → switch to IK")
            self.enable_pub.publish(Int16(data=0))
            self.cmd_req_pub.publish(String(data='ik'))
            self.processing = True

    def event_cb(self, msg: String):
        if self.once_timer:
            self.once_timer.cancel(); self.once_timer = None
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

        # metric conversion etc. (unchanged) ...
        # -----------------------------------------------------------
        # when ready:
        self.current_waypoints = self.build_waypoints(x_lat, y_fwd, self.current_event)
        self.processing, self._awaiting_done = True, True
        self._centroid_buffer.clear()
        self._start_trajectory_once_once()

    def _start_trajectory_once_once(self):
        self.enable_pub.publish(Int16(data=0))
        self.publish_path()
        self.cmd_req_pub.publish(String(data='ik'))
        self.line_vel_pub.publish(Twist())

    def publish_path(self):
        path = PathMsg(); path.header.frame_id = 'odom'
        path.poses = self.current_waypoints
        self.wp_pub.publish(path)

    def curve_done_cb(self, msg: Bool):
        if msg.data and self.processing and self._awaiting_done:
            self._awaiting_done = False
            self.cmd_req_pub.publish(String(data='line'))
            self.processing = False
            self.enable_pub.publish(Int16(data=1))

    # ---------- stub functions (image_cb, build_waypoints) stay as-is -----

def main(args=None):
    rclpy.init(args=args)
    node = TurnManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
