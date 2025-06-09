#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Int16
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
from enum import Enum


"""
Relevant DATA: with the homography_msr1 and 2 and the camera pointing fully downwards:
- The closest ROI starts to see the line 16 cm in front of the puzzlebot, ends at 40 cm
- The second ROI starts to see the line 39 cm in front of the puzzlebot, ends at 60 cm
- 

"""


class Mode(Enum):
    NORMAL = 0
    CROSSING = 1
    TURNING = 2


class LineFollowerCentroid(Node):
    """
    Nodo ROS2 para seguimiento de línea en pista con 3 marcas (L, M, R).
    Se combina:
      • Se mantiene un buffer de ~n~ mediciones de la línea central (M) para suprimir jitter
        (si el salto excede el umbral, M se da por perdida inmediatamente).
      • Control P sobre la línea “elegida” usando la última posición conocida de M para reenfocar
        la detección en curvas.
      • Umbral adicional de longitud mínima de contorno para filtrar líneas muy cortas.
    """

    def __init__(self):
        super().__init__('line_follower_centroid')
        self.bridge = CvBridge()

        # —————— Parámetros dinámicos ——————
        # Control P
        self.declare_parameter('Kp_x',             0.01)
        self.declare_parameter('Kp_ang',           0.019)
        self.declare_parameter('xVel',             0.15)
        self.declare_parameter('ang_e_thrsh',      2.0)

        # ---------- Curvature control ----------
        self.declare_parameter('radius_kp',           0.004)
        self.declare_parameter('mid_radius_thrshld',  65.0)

        # Read initial values
        self.radius_kp          = float(self.get_parameter('radius_kp').value)
        self.mid_radius_thrshld = float(self.get_parameter('mid_radius_thrshld').value)

        # Runtime storage for the current middle-line radius
        self.detected_mid_radius = float('inf')
        self.detected_mid_radius_secondary = float('inf')

        # Umbral mínimo de área para filtrar contornos
        self.declare_parameter('min_area_param',   500)

        # Umbral mínimo de longitud del contorno (para descartar contornos pequeños)
        self.declare_parameter('length_threshold', 100)

        # Homografía (BEV)
        self.declare_parameter('homography_matrix_path',
            'src/line_follow_msr/data/homography_msr1.npy')
        self.declare_parameter('secondary_homography_path', 'src/line_follow_msr/data/homography_msr2.npy')

        self.declare_parameter('warp_width',       200)
        self.declare_parameter('warp_height',      200)

        # Cantidad máxima de frames sin línea central antes de fallback (se ignora por ahora)
        self.declare_parameter('max_missing_center', 5)

        # Callback para reaccionar a cambios de parámetros en caliente
        self.add_on_set_parameters_callback(self.parameter_update_callback)

        # Leer valores iniciales
        p = self.get_parameter
        self.Kp_x             = float(p('Kp_x').value)
        self.Kp_ang           = float(p('Kp_ang').value)
        self.xVel             = float(p('xVel').value)
        self.ang_e_thrsh      = float(p('ang_e_thrsh').value)
        self.min_area_param   = int(p('min_area_param').value)
        self.length_threshold = float(p('length_threshold').value)
        self.max_missing_center = int(p('max_missing_center').value)

        # Ruta y tamaño del warp
        homography_path = p('homography_matrix_path').value
        warp_w = int(p('warp_width').value)
        warp_h = int(p('warp_height').value)
        self.warp_size = (warp_w, warp_h)

        # ————— Variables de estado generales ——————
        self.color_flag_multiplier = 1.0
        self.latest_frame = None

        # Historial de detecciones (por compatibilidad, aunque ya no se usa en FSM)
        self.last_detections = []
        self.max_history = 10

        # Buffer para supresión de jitter en la línea media (M)
        self.middle_line_buffer      = []
        self.middle_line_buffer_size = 10
        self.last_valid_middle_x     = None
        self.x_diff_threshold        = 50   # tolerancia para jitter en M

        # Variable de “última posición conocida de la M real”
        self.last_center_x = None

        #
        self.mode = Mode.NORMAL
        self.cross_start_odom = None
        self.turn_target_deg = 0.0  # parámetro opcional
        self.has_stopbar = False


        # Cargar homografía (BEV)
        self.homography_matrix = None
        self.selecting_points  = False
        self.src_points        = []
        try:
            self.homography_matrix = np.load(homography_path)
            self.get_logger().info('✅ Homography matrix loaded from file')
        except Exception:
            self.get_logger().warn('❌ Homography file not found. Activando selección manual.')
            self.selecting_points = True

        secondary_path = self.get_parameter('secondary_homography_path').value
        try:
            self.secondary_homography = np.load(secondary_path)
            self.get_logger().info('✅ Secondary homography loaded from file')
        except Exception:
            self.secondary_homography = None
            self.get_logger().warn('❌ Secondary homography file not found.')


        # ————— Publicadores y suscriptores ——————
        self.publisher = self.create_publisher(Twist, '/line_cmd_vel', 10)
        self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.create_subscription(Float32, '/fsm_action', self.fsm_action_callback, 10)


        # ————— Ventana de trackbars (parámetros de visión) ——————
        cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
        cv2.createTrackbar('Blur Kernel', 'Controls', 24, 31, lambda x: None)
        cv2.createTrackbar('Morph Kernel', 'Controls', 5, 51, lambda x: None)
        cv2.createTrackbar('Block Size', 'Controls', 76, 101, lambda x: None)
        cv2.createTrackbar('C (Bias)', 'Controls', 13, 50, lambda x: None)
        cv2.createTrackbar('Min Area', 'Controls', self.min_area_param, 2000, lambda x: None)

        # Timer a 50 Hz para percepción + control
        timer_period = 1.0 / 50.0
        self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info('Line Follower Node Started (50 Hz control)')

    def parameter_update_callback(self, params):
        """Actualiza parámetros dinámicos en tiempo real (con try/except para no morir)."""
        successful = True
        for pp in params:
            try:
                if pp.name == 'Kp_x' and pp.type_ == Parameter.Type.DOUBLE:
                    self.Kp_x = float(pp.value)
                    self.get_logger().info(f'Kp_x actualizado a {self.Kp_x}')
                elif pp.name == 'Kp_ang' and pp.type_ == Parameter.Type.DOUBLE:
                    self.Kp_ang = float(pp.value)
                    self.get_logger().info(f'Kp_ang actualizado a {self.Kp_ang}')
                elif pp.name == 'xVel' and pp.type_ == Parameter.Type.DOUBLE:
                    self.xVel = float(pp.value)
                    self.get_logger().info(f'xVel actualizado a {self.xVel}')
                elif pp.name == 'ang_e_thrsh' and pp.type_ == Parameter.Type.DOUBLE:
                    self.ang_e_thrsh = float(pp.value)
                    self.get_logger().info(f'ang_e_thrsh actualizado a {self.ang_e_thrsh}')
                elif pp.name == 'min_area_param' and pp.type_ in (Parameter.Type.INTEGER, Parameter.Type.DOUBLE):
                    self.min_area_param = int(pp.value)
                    cv2.setTrackbarPos('Min Area', 'Controls', self.min_area_param)
                    self.get_logger().info(f'min_area_param actualizado a {self.min_area_param}')
                elif pp.name == 'length_threshold' and pp.type_ in (Parameter.Type.INTEGER, Parameter.Type.DOUBLE):
                    self.length_threshold = float(pp.value)
                    self.get_logger().info(f'length_threshold actualizado a {self.length_threshold}')
                elif pp.name == 'max_missing_center' and pp.type_ in (Parameter.Type.INTEGER, Parameter.Type.DOUBLE):
                    self.max_missing_center = int(pp.value)
                    self.get_logger().info(f'max_missing_center actualizado a {self.max_missing_center}')
                elif pp.name in ('warp_width', 'warp_height'):
                    self.get_logger().warn('Cambio de tamaño de warp no aplicado dinámicamente.')
                elif pp.name == 'homography_matrix_path' and pp.type_ == Parameter.Type.STRING:
                    self.get_logger().warn('Cambio de ruta homografía no aplicado dinámicamente.')
                elif pp.name == 'radius_kp' and pp.type_ == Parameter.Type.DOUBLE:
                    self.radius_kp = float(pp.value)
                    self.get_logger().info(f'radius_kp actualizado a {self.radius_kp}')
                elif pp.name == 'mid_radius_thrshld' and pp.type_ == Parameter.Type.DOUBLE:
                    self.mid_radius_thrshld = float(pp.value)
                    self.get_logger().info(f'mid_radius_thrshld actualizado a {self.mid_radius_thrshld}')

                else:
                    self.get_logger().warn(f'Parámetro {pp.name} no gestionado.')
                    successful = False
            except Exception as e:
                self.get_logger().error(f'Error actualizando {pp.name}: {e}')
                successful = False

        return rclpy.parameter.ParameterEventHandlerResult(successful=successful)

    def fsm_action_callback(self, msg: Float32):
        """Permite variar el multiplicador de velocidad desde un FSM externo."""
        self.color_flag_multiplier = msg.data
        self.get_logger().info(f"Updated color_flag_multiplier to: {self.color_flag_multiplier}")

    def image_callback(self, msg: Image):
        """Guarda la última imagen recibida para procesarla en timer_callback."""
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Error al convertir imagen ROS→CV2: {e}")
            self.latest_frame = None

    def preprocess_region(self, region, blur_k, block_size, c_bias, morph_k):
        """Contrast‐limited → blur → adaptive‐threshold → morfología."""
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        v_equalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(v)
        blur_k = max(1, blur_k | 1)
        blurred = cv2.GaussianBlur(v_equalized, (blur_k, blur_k), 0)
        block_size = max(3, block_size | 1)
        binary = cv2.adaptiveThreshold(
            blurred, 255,
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
        frame = self.latest_frame
        if frame is None:
            return

        # ========== ETAPA 1: VISIÓN ==========  
        self.update_trackbars()                      # blur_k, morph_k, etc.
        warped, overlay, binary_rois = self.process_main_homography(frame)
        lines = self.detect_lines(binary_rois, overlay)
        self.update_secondary_overlay(frame)         # opcional, para vista lejana

        # ========== ETAPA 2: CRUCE DETECTADO ==========  
        punteado_izq = self.is_dotted_column(binary_rois["left"])
        punteado_der = self.is_dotted_column(binary_rois["right"])
        stopbar = any(self.is_stopbar(l["contour"]) for l in lines.values() if l)

        if self.mode == Mode.NORMAL:
            if punteado_izq and punteado_der and stopbar:
                self.get_logger().info("🛑 Cruce detectado → CROSSING")
                self.mode = Mode.CROSSING
                self.cross_start_odom = self.get_current_odom_x()
                return

        # ========== ETAPA 3: FSM de comportamiento ==========  
        if self.mode == Mode.CROSSING:
            dist = abs(self.get_current_odom_x() - self.cross_start_odom)
            if dist >= 0.25:
                self.get_logger().info("📍 En centro del cruce. Iniciando giro.")
                self.mode = Mode.TURNING
                self.turn_start_yaw = self.get_current_yaw()
                self.turn_target_deg = self.turn_start_yaw + 90.0  # ejemplo
                return
            else:
                self.publish_cmd(0.1, 0.0)
                return

        if self.mode == Mode.TURNING:
            yaw_error = self.turn_target_deg - self.get_current_yaw()
            if abs(yaw_error) < 5.0:
                self.get_logger().info("✅ Giro terminado. Reanudando seguimiento.")
                self.mode = Mode.NORMAL
            else:
                self.publish_cmd(0.0, 0.4 * np.sign(yaw_error))
            return

        # ========== ETAPA 4: CONTROL DE LINEA NORMAL ==========  
        chosen_label, chosen_line = self.select_best_line(lines)
        if chosen_line is not None:
            twist = self.compute_twist(chosen_line, chosen_label)
            self.publisher.publish(twist)
        else:
            self.publish_cmd(-0.10, 0.0)
            self.get_logger().warn("No line detected → fallback")
        
        cv2.imshow("Overlay", cv2.resize(overlay, None, fx=3.0, fy=3.0))
        cv2.waitKey(1)



    def detect_line_in_roi(self, binary_roi, roi_x_offset, overlay=None, color=(255,255,0), label="", min_area=150):
        """
        Detecta el contorno más grande en binary_roi (área > min_area), ajusta recta con fitLine,
        filtra si el contorno ajustado es muy corto (< length_threshold), y devuelve:
          • x_global: coordenada X en toda la imagen warp
          • angle:  ángulo en grados
          • length: longitud aproximada del contorno (arcLength)
          • contour: puntos del contorno
        Si no hay un contorno suficientemente grande o muy corto, devuelve None.
        """
        contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Filtrar por área mínima
        contours = [c for c in contours if cv2.contourArea(c) > min_area]
        if not contours:
            return None

        # Tomar el contorno de mayor área
        largest = max(contours, key=cv2.contourArea)
        # Ajustar recta
        vx, vy, x0, y0 = cv2.fitLine(largest, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        y_bottom = binary_roi.shape[0] - 1
        if abs(vy) < 1e-6:
            return None

        t = (y_bottom - y0) / vy
        x_bottom = int(x0 + vx * t)
        global_x = roi_x_offset + x_bottom

        # Obtener longitud aproximada del contorno para filtrar contornos cortos
        length = cv2.arcLength(largest, closed=False)
        if length < self.length_threshold:
            return None

        # Dibujar en overlay (si se pide)
        if overlay is not None:
            # green contour
            offset_contour = [np.array([[pt[0][0] + roi_x_offset, pt[0][1]]])
                                for pt in largest]
            offset_contour = np.array(offset_contour, dtype=np.int32)
            cv2.drawContours(overlay, [offset_contour], -1, (0, 255, 0), 2)
            # fitted yellow line
            pt1 = (roi_x_offset + int(x0 - vx * 50), int(y0 - vy * 50))
            pt2 = (roi_x_offset + int(x0 + vx * 50), int(y0 + vy * 50))
            cv2.line(overlay, pt1, pt2, color, 2)

            # -------------- curvature (only for middle ROI) --------------
            if label in ["M", "M2"]:
                try:
                    (xc, yc), R = cv2.minEnclosingCircle(largest)
                    if label == "M":
                        self.detected_mid_radius = float(R)
                    else:
                        self.detected_mid_radius_secondary = float(R)
                    # Draw circle & radius label
                    center = (int(xc) + roi_x_offset, int(yc))
                    cv2.circle(overlay, center, int(R), (0, 255, 255), 1)
                    cv2.putText(overlay, f"M R:{R:.1f}", (roi_x_offset + 5, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                except cv2.error:
                    self.detected_mid_radius = float('inf')

        return {
            "x_global": global_x,
            "angle":    np.degrees(np.arctan2(vy, vx)),
            "length":   length,
            "contour":  largest
        }
    
    def is_dotted_column(self, binary_roi, min_segments=4, aspect_thresh=2.5):
        """Detecta si hay una columna vertical de contornos alargados (línea punteada)."""
        contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        verticals = 0
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if h / max(w, 1) > aspect_thresh:
                verticals += 1
        return verticals >= min_segments

    # Detectar líneas horizontales (barras de detención)
    def is_stopbar(self, contour, angle_thresh_deg=10):
        vx, vy, _, _ = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        angle = np.degrees(np.arctan2(vy, vx))
        return abs(angle) < angle_thresh_deg
    
    def update_trackbars(self):
        self.blur_k      = max(1, cv2.getTrackbarPos('Blur Kernel', 'Controls') | 1)
        self.morph_k     = max(1, cv2.getTrackbarPos('Morph Kernel', 'Controls') | 1)
        self.block_size  = max(3, cv2.getTrackbarPos('Block Size', 'Controls') | 1)
        self.c_bias      = cv2.getTrackbarPos('C (Bias)', 'Controls')
        min_area_tb      = cv2.getTrackbarPos('Min Area', 'Controls')
        self.min_area    = max(self.min_area_param, min_area_tb)

    def process_main_homography(self, frame):
        if self.homography_matrix is not None:
            warped = cv2.warpPerspective(frame, self.homography_matrix, self.warp_size)
            warped = cv2.flip(warped, 1)
        else:
            warped = frame.copy()

        overlay = warped.copy()
        h, w, _ = warped.shape
        self.center_frame_x = w // 2
        twelve_div = w // 12

        rois = {
            "left":   warped[:, 0:5 * twelve_div],
            "middle": warped[:, 2 * twelve_div : 10 * twelve_div],
            "right":  warped[:, 7 * twelve_div : ]
        }

        binary_rois = {
            key: self.preprocess_region(roi, self.blur_k, self.block_size, self.c_bias, self.morph_k)
            for key, roi in rois.items()
        }

        return warped, overlay, binary_rois

    def detect_lines(self, binary_rois, overlay):
        twelve_div = overlay.shape[1] // 12
        return {
            "L": self.detect_line_in_roi(binary_rois["left"], 0, overlay, (255, 0, 0), "L", self.min_area),
            "M": self.detect_line_in_roi(binary_rois["middle"], 2 * twelve_div, overlay, (255, 255, 0), "M", self.min_area),
            "R": self.detect_line_in_roi(binary_rois["right"], 7 * twelve_div, overlay, (255, 0, 0), "R", self.min_area)
        }

    def select_best_line(self, lines):
        candidatas = []
        for label, line in lines.items():
            if line is not None:
                area = float(cv2.contourArea(line["contour"]))
                candidatas.append((label, line, area))

        if self.last_center_x is None:
            self.last_center_x = float(self.center_frame_x)

        if not candidatas:
            return None, None
        elif len(candidatas) == 1:
            chosen_label, chosen_line, _ = candidatas[0]
        else:
            mejor_dist = None
            for label, line, area in candidatas:
                x_i = float(line["x_global"])
                dist = abs(x_i - self.last_center_x)
                if mejor_dist is None or dist < mejor_dist:
                    mejor_dist = dist
                    chosen_label, chosen_line = label, line

        if chosen_label == 'M':
            self.last_center_x = float(chosen_line["x_global"])

        return chosen_label, chosen_line

    def compute_twist(self, line, label):
        x_error   = line["x_global"] - self.center_frame_x
        theta     = line["angle"]
        angle_error = 90.0 - theta if theta > 0 else -90.0 - theta
        if abs(angle_error) < self.ang_e_thrsh:
            angle_error = 0.0

        linear_speed  = self.xVel
        angular_speed = self.Kp_ang * angle_error - self.Kp_x * x_error

        if label == 'M' and self.detected_mid_radius < self.mid_radius_thrshld:
            radius_term   = self.radius_kp * self.detected_mid_radius
            angular_speed -= radius_term

        twist = Twist()
        twist.linear.x  = linear_speed * self.color_flag_multiplier
        twist.angular.z = angular_speed * self.color_flag_multiplier
        self.get_logger().info(
            f"[{label}] x_err={x_error:.1f}, ang_err={angle_error:.1f}, R={self.detected_mid_radius:.1f}, ω={angular_speed:.3f}")
        return twist

    def publish_cmd(self, v, w):
        twist = Twist()
        twist.linear.x = v * self.color_flag_multiplier
        twist.angular.z = w * self.color_flag_multiplier
        self.publisher.publish(twist)

    def update_secondary_overlay(self, frame):
        if self.secondary_homography is None:
            return

        warped2 = cv2.warpPerspective(frame, self.secondary_homography, self.warp_size)
        warped2 = cv2.flip(warped2, 1)
        overlay2 = warped2.copy()
        h2, w2, _ = warped2.shape
        twelve_div2 = w2 // 12

        roi_l2 = warped2[:, 0:5 * twelve_div2]
        roi_m2 = warped2[:, 2 * twelve_div2 : 10 * twelve_div2]
        roi_r2 = warped2[:, 7 * twelve_div2 :]

        binary_l2 = self.preprocess_region(roi_l2, self.blur_k, self.block_size, self.c_bias, self.morph_k)
        binary_m2 = self.preprocess_region(roi_m2, self.blur_k, self.block_size, self.c_bias, self.morph_k)
        binary_r2 = self.preprocess_region(roi_r2, self.blur_k, self.block_size, self.c_bias, self.morph_k)

        self.detect_line_in_roi(binary_l2, 0, overlay2, (0, 0, 255), "L2", self.min_area)
        self.detect_line_in_roi(binary_m2, 2 * twelve_div2, overlay2, (0, 255, 255), "M2", self.min_area)
        self.detect_line_in_roi(binary_r2, 7 * twelve_div2, overlay2, (0, 0, 255), "R2", self.min_area)

        cv2.imshow("Overlay Secondary", cv2.resize(overlay2, None, fx=3.0, fy=3.0))
        stacked_rois2 = stack_with_dividers([binary_l2, binary_m2, binary_r2])
        cv2.imshow("Left2 | Middle2 | Right2", stacked_rois2)




def select_points(event, x, y, flags, param):
    """
    Callback para seleccionar 4 puntos si no había homografía cargada.
    Una vez tenemos 4, computa getPerspectiveTransform y lo guarda en disco.
    """
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
    Apila en horizontal imágenes en escala de grises (1-channel) con divisores verticales.
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
