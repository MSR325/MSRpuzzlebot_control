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
    def __init__(self):
        super().__init__('line_follower_centroid')
        self.bridge = CvBridge()

        # —————— Declaración de parámetros dinámicos ——————
        # Control (se mantiene la ley de control original P)
        self.declare_parameter('Kp_x', 0.01)
        self.declare_parameter('Kp_ang', 0.019)
        self.declare_parameter('xVel', 0.15)
        self.declare_parameter('ang_e_thrsh', 2.0)

        # Visión: umbral mínimo de área para filtrar contornos
        self.declare_parameter('min_area_param', 500)

        # Homografía
        self.declare_parameter('homography_matrix_path',
            '/home/idmx/ros2_ws_2/src/line_follow_msr/data/homography3.npy')
        self.declare_parameter('warp_width', 200)
        self.declare_parameter('warp_height', 200)

        # Cantidad máxima de frames sin línea central antes de fallback
        self.declare_parameter('max_missing_center', 5)

        # Callback para reaccionar a cambios de parámetros en caliente
        self.add_on_set_parameters_callback(self.parameter_update_callback)

        # Leer valores iniciales
        p = self.get_parameter
        self.Kp_x = p('Kp_x').value
        self.Kp_ang = p('Kp_ang').value
        self.xVel = p('xVel').value
        self.ang_e_thrsh = p('ang_e_thrsh').value

        self.min_area_param = int(p('min_area_param').value)

        homography_path = p('homography_matrix_path').value
        warp_w = int(p('warp_width').value)
        warp_h = int(p('warp_height').value)
        self.warp_size = (warp_w, warp_h)

        self.max_missing_center = int(p('max_missing_center').value)

        # Variables de estado
        self.color_flag_multiplier = 1.0
        self.frames_without_center = 0
        self.latest_frame = None

        # Cargar homografía
        self.homography_matrix = None
        self.selecting_points = False
        self.src_points = []

        try:
            self.homography_matrix = np.load(homography_path)
            self.get_logger().info('✅ Homography matrix loaded from file')
        except Exception:
            self.get_logger().warn('❌ Homography file not found. Activando selección manual.')
            self.selecting_points = True

        # Publicadores y suscripciones
        self.publisher = self.create_publisher(Twist, '/line_cmd_vel', 10)
        self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.create_subscription(Float32, '/fsm_action', self.fsm_action_callback, 10)

        # Ventana de trackbars para parámetros de visión
        cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
        cv2.createTrackbar('Blur Kernel', 'Controls', 24, 31, lambda x: None)
        cv2.createTrackbar('Morph Kernel', 'Controls', 5, 51, lambda x: None)
        cv2.createTrackbar('Block Size', 'Controls', 76, 101, lambda x: None)
        cv2.createTrackbar('C (Bias)', 'Controls', 13, 50, lambda x: None)
        cv2.createTrackbar('Min Area', 'Controls', self.min_area_param, 2000, lambda x: None)

        # Timer para ejecutar el lazo de control a 50 Hz
        timer_period = 1.0 / 100.0
        self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info('Line Follower Node Started (control a 50 Hz)')

    def parameter_update_callback(self, params):
        """Actualiza parámetros dinámicos en tiempo real"""
        for p in params:
            if p.name == 'Kp_x' and p.type_ == Parameter.Type.DOUBLE:
                self.Kp_x = p.value
                self.get_logger().info(f'Kp_x actualizado a {self.Kp_x}')
            elif p.name == 'Kp_ang' and p.type_ == Parameter.Type.DOUBLE:
                self.Kp_ang = p.value
                self.get_logger().info(f'Kp_ang actualizado a {self.Kp_ang}')
            elif p.name == 'xVel' and p.type_ == Parameter.Type.DOUBLE:
                self.xVel = p.value
                self.get_logger().info(f'xVel actualizado a {self.xVel}')
            elif p.name == 'ang_e_thrsh' and p.type_ == Parameter.Type.DOUBLE:
                self.ang_e_thrsh = p.value
                self.get_logger().info(f'ang_e_thrsh actualizado a {self.ang_e_thrsh}')
            elif p.name == 'min_area_param' and p.type_ in (Parameter.Type.INTEGER, Parameter.Type.DOUBLE):
                self.min_area_param = int(p.value)
                cv2.setTrackbarPos('Min Area', 'Controls', self.min_area_param)
                self.get_logger().info(f'min_area_param actualizado a {self.min_area_param}')
            elif p.name == 'homography_matrix_path' and p.type_ == Parameter.Type.STRING:
                self.get_logger().warn('Cambio de ruta homografía no aplicado dinámicamente.')
            elif p.name in ('warp_width', 'warp_height'):
                self.get_logger().warn('Cambio de tamaño de warp no aplicado dinámicamente.')
            elif p.name == 'max_missing_center' and p.type_ in (Parameter.Type.INTEGER, Parameter.Type.DOUBLE):
                self.max_missing_center = int(p.value)
                self.get_logger().info(f'max_missing_center actualizado a {self.max_missing_center}')
        return rclpy.parameter.ParameterEventHandlerResult(successful=True)

    def fsm_action_callback(self, msg: Float32):
        self.color_flag_multiplier = msg.data
        self.get_logger().info(f"Updated color_flag_multiplier to: {self.color_flag_multiplier}")

    def image_callback(self, msg):
        """Guarda la última imagen recibida y retorna"""
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def preprocess_region(self, region, blur_k, block_size, c_bias, morph_k):
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        v_equalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(v)
        blur_k = max(1, blur_k | 1)
        blurred = cv2.GaussianBlur(v_equalized, (blur_k, blur_k), 0)
        block_size = max(3, block_size | 1)
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            c_bias
        )
        morph_k = max(1, morph_k | 1)
        kernel = np.ones((morph_k, morph_k), np.uint8)
        binary = cv2.erode(binary, kernel, iterations=1)
        binary = cv2.dilate(binary, kernel, iterations=2)
        return binary

    def timer_callback(self):
        """Lazo principal de percepción y control a 50 Hz"""
        frame = self.latest_frame
        if frame is None:
            return

        # ————— BEV con homografía —————
        if self.homography_matrix is not None:
            warped = cv2.warpPerspective(frame, self.homography_matrix, self.warp_size)
            warped = cv2.flip(warped, 1)
        else:
            warped = frame

        overlay = warped.copy()
        h, w, _ = warped.shape

        # ————— División en 3 ROIs verticales —————
        twelve_div = w // 12
        roi_left = warped[:, 0:5*twelve_div]
        roi_middle = warped[:, 2*twelve_div:10*twelve_div]
        roi_right = warped[:, 7*twelve_div:]

        # ————— Leer parámetros de trackbar —————
        blur_k = max(1, cv2.getTrackbarPos('Blur Kernel', 'Controls') | 1)
        morph_k = max(1, cv2.getTrackbarPos('Morph Kernel', 'Controls') | 1)
        block_size = max(3, cv2.getTrackbarPos('Block Size', 'Controls') | 1)
        c_bias = cv2.getTrackbarPos('C (Bias)', 'Controls')
        min_area_tb = cv2.getTrackbarPos('Min Area', 'Controls')

        # ————— Preprocesar cada ROI —————
        binary_left   = self.preprocess_region(roi_left,   blur_k, block_size, c_bias, morph_k)
        binary_middle = self.preprocess_region(roi_middle, blur_k, block_size, c_bias, morph_k)
        binary_right  = self.preprocess_region(roi_right,  blur_k, block_size, c_bias, morph_k)

        # Umbral mínimo combinado
        min_area = max(self.min_area_param, min_area_tb)

        # ————— Detectar líneas en cada ROI —————
        line_l = self.detect_line_in_roi(binary_left,   0,                  overlay, (255,   0,   0), "L", min_area)
        line_m = self.detect_line_in_roi(binary_middle, 2*twelve_div,      overlay, (255, 255,   0), "M", min_area)
        line_r = self.detect_line_in_roi(binary_right,  7*twelve_div,      overlay, (255,   0,   0), "R", min_area)

        # ————— Mostrar texto de debug en overlay —————
        x_l = line_l["x_global"] if line_l else "-"
        x_m = line_m["x_global"] if line_m else "-"
        x_r = line_r["x_global"] if line_r else "-"
        a_l = f'{line_l["angle"]:.1f}' if line_l else "-"
        a_m = f'{line_m["angle"]:.1f}' if line_m else "-"
        a_r = f'{line_r["angle"]:.1f}' if line_r else "-"
        text_x = f"x: L={x_l} M={x_m} R={x_r}"
        text_a = f"ang: L={a_l} M={a_m} R={a_r}"
        cv2.putText(overlay, text_x, (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(overlay, text_a, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
        center_x = w // 2
        cv2.line(overlay, (center_x, 0), (center_x, h), (0, 0, 255), 2)

        # ————— Mostrar ventanas de debug —————
        if self.selecting_points:
            cv2.imshow("ROI", warped)
        resized_overlay = cv2.resize(overlay, None, fx=3.0, fy=3.0)
        cv2.imshow("Overlay", resized_overlay)
        stacked_rois = stack_with_dividers([binary_left, binary_middle, binary_right])
        cv2.imshow("Left | Middle | Right", stacked_rois)

        # ————— Lógica de selección de “línea elegida” con rescate temporal —————
        chosen = None
        if line_m:
            chosen = line_m
            self.frames_without_center = 0
        else:
            self.frames_without_center += 1
            if self.frames_without_center <= self.max_missing_center:
                # Verificar si las laterales están prácticamente verticales (ángulo cercano a 90°)
                cand_l = (line_l and abs(line_l["angle"] - 90.0) < 10.0)
                cand_r = (line_r and abs(line_r["angle"] - 90.0) < 10.0)
                if cand_l and cand_r:
                    x_virtual     = (line_l["x_global"] + line_r["x_global"]) / 2.0
                    angle_virtual = (line_l["angle"]   + line_r["angle"]  ) / 2.0
                    chosen = {"x_global": x_virtual, "angle": angle_virtual}
                elif cand_l:
                    chosen = line_l
                elif cand_r:
                    chosen = line_r
                # Si ninguna lateral califica, chosen queda en None hasta que termine el margen de rescate
            else:
                chosen = None  # Caerá a fallback

        # ————— Control usando la ley actual (sin PID, como solicitaste) —————
        if chosen:
            # Cálculo de errores lateral y angular
            x_error = chosen["x_global"] - center_x
            line_angle = chosen["angle"]
            if line_angle > 0:
                angle_error = 90.0 - line_angle
            else:
                angle_error = -90.0 - line_angle
            if abs(angle_error) < self.ang_e_thrsh:
                angle_error = 0.0

            # Ley de control P como estaba antes
            linear_speed  = self.xVel
            angular_speed = self.Kp_ang * angle_error - self.Kp_x * x_error

            # Guardar para fallback
            self.last_angular = angular_speed
            self.last_linear  = linear_speed

            # Publicar
            twist = Twist()
            twist.linear.x  = linear_speed * self.color_flag_multiplier
            twist.angular.z = angular_speed * self.color_flag_multiplier
            self.publisher.publish(twist)

            self.get_logger().info(
                f"[Control] ang_err={angle_error:.2f}, x_err={x_error:.2f}, "
                f"v={linear_speed:.3f}, ω={angular_speed:.3f}"
            )
        else:
            # ————— Fallback: mantener última dirección angular conocida, avanzar despacio —————
            twist = Twist()
            twist.linear.x  = 0.05 * self.color_flag_multiplier
            twist.angular.z = getattr(self, 'last_angular', 0.0)
            self.publisher.publish(twist)
            self.get_logger().warn("No line detected → fallback control")

        cv2.waitKey(1)

    def detect_line_in_roi(self, binary_roi, roi_x_offset, overlay=None, color=(255, 255, 0), label="", min_area=150):
        """Detecta el contorno más grande en binary_roi, ajusta línea y devuelve x_global y ángulo"""
        contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > min_area]
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        vx, vy, x0, y0 = cv2.fitLine(largest, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        y_bottom = binary_roi.shape[0] - 1
        if abs(vy) < 1e-6:
            return None
        t = (y_bottom - y0) / vy
        x_bottom = int(x0 + vx * t)
        global_x = roi_x_offset + x_bottom

        if overlay is not None:
            # Dibujar contorno desplazado y la línea ajustada
            offset_contour = [np.array([[pt[0][0] + roi_x_offset, pt[0][1]]]) for pt in largest]
            offset_contour = np.array(offset_contour, dtype=np.int32)
            cv2.drawContours(overlay, [offset_contour], -1, (0, 255, 0), 2)
            pt1 = (roi_x_offset + int(x0 - vx * 50), int(y0 - vy * 50))
            pt2 = (roi_x_offset + int(x0 + vx * 50), int(y0 + vy * 50))
            cv2.line(overlay, pt1, pt2, color, 2)

        return {
            "x_global": global_x,
            "angle": np.degrees(np.arctan2(vy, vx)),
            "length": cv2.arcLength(largest, closed=False),
            "contour": largest
        }


def select_points(event, x, y, flags, param):
    """Callback para seleccionar puntos si no existe la homografía en disco"""
    node = param
    if event == cv2.EVENT_LBUTTONDOWN and node.selecting_points:
        node.src_points.append([x, y])
        node.get_logger().info(f"Selected point: ({x},{y})")
        if len(node.src_points) == 4:
            pts = np.array(node.src_points, dtype=np.float32)
            dst = np.array([
                [0, 0],
                [node.warp_size[0] - 1, 0],
                [node.warp_size[0] - 1, node.warp_size[1] - 1],
                [0, node.warp_size[1] - 1]
            ], dtype=np.float32)
            node.homography_matrix = cv2.getPerspectiveTransform(pts, dst)
            node.selecting_points = False
            np.save(node.get_parameter('homography_matrix_path').value, node.homography_matrix)
            node.get_logger().info("✅ Homography computed and saved.")


def stack_with_dividers(imgs, divider_thickness=3, divider_color=255):
    """
    Apila horizontalmente imágenes en escala de grises con divisor vertical.
    """
    h = imgs[0].shape[0]
    div = np.full((h, divider_thickness), divider_color, dtype=np.uint8)
    result = imgs[0]
    for img in imgs[1:]:
        result = np.hstack((result, div, img))
    return result


def main(args=None):
    rclpy.init(args=args)
    node = LineFollowerCentroid()
    cv2.namedWindow("ROI")
    cv2.setMouseCallback("ROI", select_points, node)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
