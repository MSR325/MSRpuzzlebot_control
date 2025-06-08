#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np


class LineFollowerCentroid(Node):
    """
    Nodo ROS2 para seguimiento de línea en pista con 3 marcas (L, M, R).
    Se combina:
      • Buffer de ~n~ mediciones de la línea central (M) para suprimir jitter.
      • Control P sobre la línea “elegida” usando la última posición conocida de M.
      • Umbral adicional de longitud mínima de contorno para filtrar líneas muy cortas.
      • Pipeline mejorado de combinación de ROI binarios en una máscara completa.
      • Visualización de la línea ajustada y el punto de intersección.
    """

    def __init__(self):
        super().__init__('line_follower_centroid')
        self.bridge = CvBridge()

        # Parámetros dinámicos
        self.declare_parameter('Kp_x', 0.01)
        self.declare_parameter('Kp_ang', 0.019)
        self.declare_parameter('xVel', 0.15)
        self.declare_parameter('ang_e_thrsh', 2.0)
        self.declare_parameter('min_area_param', 500)
        self.declare_parameter('length_threshold', 100)
        self.declare_parameter('homography_matrix_path',
                               '/home/idmx/ros2_ws_2/src/line_follow_msr/data/homography3.npy')
        self.declare_parameter('warp_width', 200)
        self.declare_parameter('warp_height', 200)
        self.declare_parameter('max_missing_center', 5)
        self.add_on_set_parameters_callback(self.parameter_update_callback)

        p = self.get_parameter
        self.Kp_x = float(p('Kp_x').value)
        self.Kp_ang = float(p('Kp_ang').value)
        self.xVel = float(p('xVel').value)
        self.ang_e_thrsh = float(p('ang_e_thrsh').value)
        self.min_area_param = int(p('min_area_param').value)
        self.length_threshold = float(p('length_threshold').value)
        self.max_missing_center = int(p('max_missing_center').value)

        homography_path = p('homography_matrix_path').value
        warp_w = int(p('warp_width').value)
        warp_h = int(p('warp_height').value)
        self.warp_size = (warp_w, warp_h)

        # Estado
        self.latest_frame = None
        self.middle_line_buffer = []
        self.middle_line_buffer_size = 10
        self.last_valid_middle_x = None
        self.x_diff_threshold = 50
        self.last_center_x = None

        # Cargar homografía
        try:
            self.homography_matrix = np.load(homography_path)
            self.get_logger().info('✅ Homography loaded')
        except Exception:
            self.homography_matrix = None
            self.get_logger().warn('❌ Homography not found, skipping')

        # Publishers / subscribers
        self.pub_cmd = self.create_publisher(Twist, '/line_cmd_vel', 10)
        self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.create_subscription(Float32, '/fsm_action', self.fsm_action_callback, 10)

        # Ventanas y Trackbars
        cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Binary_Full', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
        cv2.createTrackbar('Blur', 'Controls', 24, 31, lambda x: None)
        cv2.createTrackbar('Morph', 'Controls', 5, 51, lambda x: None)
        cv2.createTrackbar('Block', 'Controls', 75, 101, lambda x: None)
        cv2.createTrackbar('Bias', 'Controls', 13, 50, lambda x: None)
        cv2.createTrackbar('MinArea', 'Controls', self.min_area_param, 2000, lambda x: None)

        # Timer
        self.create_timer(1.0/50.0, self.timer_callback)
        self.get_logger().info('Line Follower Node Started')

    def parameter_update_callback(self, params):
        successful = True
        for pp in params:
            try:
                if pp.name == 'Kp_x': self.Kp_x = float(pp.value)
                elif pp.name == 'Kp_ang': self.Kp_ang = float(pp.value)
                elif pp.name == 'xVel': self.xVel = float(pp.value)
                elif pp.name == 'ang_e_thrsh': self.ang_e_thrsh = float(pp.value)
                elif pp.name == 'min_area_param':
                    self.min_area_param = int(pp.value)
                    cv2.setTrackbarPos('MinArea','Controls', self.min_area_param)
                elif pp.name == 'length_threshold':
                    self.length_threshold = float(pp.value)
            except Exception as e:
                self.get_logger().error(f'Param update error {pp.name}: {e}')
                successful = False
        return rclpy.parameter.ParameterEventHandlerResult(successful=successful)

    def fsm_action_callback(self, msg: Float32):
        self.color_flag_multiplier = msg.data

    def image_callback(self, msg: Image):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'CvBridge error: {e}')
            self.latest_frame = None

    def preprocess_region(self, region):
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        v = hsv[:,:,2]
        v = cv2.createCLAHE(2.0,(8,8)).apply(v)
        bk = cv2.getTrackbarPos('Block','Controls')
        bk = max(3, bk | 1)
        blur_k = cv2.getTrackbarPos('Blur','Controls') | 1
        M = cv2.GaussianBlur(v,(blur_k,blur_k),0)
        thresh = cv2.adaptiveThreshold(M,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV,bk,
                                       cv2.getTrackbarPos('Bias','Controls'))
        mk = cv2.getTrackbarPos('Morph','Controls') | 1
        kernel = np.ones((mk,mk),np.uint8)
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        e = cv2.erode(opened,kernel,iterations=1)
        d = cv2.dilate(e,kernel,iterations=2)
        return d

    def timer_callback(self):
        if self.latest_frame is None: return
        fr = self.latest_frame.copy()
        h,w,_ = fr.shape
        if self.homography_matrix is not None:
            warp = cv2.warpPerspective(fr, self.homography_matrix, self.warp_size)
        else:
            warp = fr

        # construir máscara binaria completa
        twelve = warp.shape[1]//12
        b_full = np.zeros((warp.shape[0],warp.shape[1]),dtype=np.uint8)
        left  = self.preprocess_region(warp[:,0:5*twelve])
        mid   = self.preprocess_region(warp[:,2*twelve:10*twelve])
        right = self.preprocess_region(warp[:,7*twelve:])
        b_full[:,0:5*twelve]   = left
        b_full[:,2*twelve:10*twelve] = mid
        b_full[:,7*twelve:]    = right
        cv2.imshow('Binary_Full', b_full)

        # filtrar contornos válidos
        cnts,_ = cv2.findContours(b_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lines = []
        for c in cnts:
            A = cv2.contourArea(c)
            if A < self.min_area_param: continue
            l = cv2.arcLength(c,False)
            if l < self.length_threshold: continue
            lines.append(c)

        # elegir la mejor línea por proximidad a last_center_x
        if self.last_center_x is None:
            self.last_center_x = w//2
        best_contour = None
        best_dist = None
        for c in lines:
            vx,vy,x0,y0 = cv2.fitLine(c,cv2.DIST_L2,0,0.01,0.01).flatten()
            yb = warp.shape[0] - 1
            t = (yb - y0) / vy
            xb = int(x0 + vx * t)
            d = abs(xb - self.last_center_x)
            if best_dist is None or d < best_dist:
                best_dist = d
                best_contour = c
                best_x = xb
                best_ang = np.degrees(np.arctan2(vy, vx))

        # dibujar detección y línea ajustada
        det_vis = warp.copy()
        # línea roja: centro del FOV
        cv2.line(det_vis, (w//2, 0), (w//2, h), (0,0,255), 2)
        if best_contour is not None:
            # puntos de fitLine
            vx,vy,x0,y0 = cv2.fitLine(best_contour,cv2.DIST_L2,0,0.01,0.01).flatten()
            pt1 = (int(x0 - vx*100), int(y0 - vy*100))
            pt2 = (int(x0 + vx*100), int(y0 + vy*100))
            cv2.line(det_vis, pt1, pt2, (0,255,0), 2)  # línea ajustada
            cv2.circle(det_vis, (best_x, h-1), 5, (0,255,0), -1)  # punto inferior
            # actualizar última posición conocida
            self.last_center_x = best_x

        cv2.imshow('Detection', det_vis)

        # control P
        twist = Twist()
        if best_contour is not None:
            errx = best_x - w//2
            if best_ang > 0:
                erang = 90.0 - best_ang
            else:
                erang = -90.0 - best_ang
            if abs(erang) < self.ang_e_thrsh:
                erang = 0.0
            twist.linear.x  = self.xVel * self.color_flag_multiplier
            twist.angular.z = (self.Kp_ang * erang - self.Kp_x * errx) * self.color_flag_multiplier
        else:
            twist.linear.x  = -0.10 * self.color_flag_multiplier
            twist.angular.z = 0.0
        self.pub_cmd.publish(twist)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = LineFollowerCentroid()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__=='__main__':
    main()
