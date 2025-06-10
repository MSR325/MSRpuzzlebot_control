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
from ament_index_python.packages import get_package_share_directory
from pathlib import Path

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
        self.declare_parameter('Kp_ang',           0.016)
        self.declare_parameter('xVel',             0.13)
        self.declare_parameter('ang_e_thrsh',      2.0)

        # Umbral mínimo de área para filtrar contornos
        self.declare_parameter('min_area_param',   500)

        # Umbral mínimo de longitud del contorno (para descartar contornos pequeños)
        self.declare_parameter('length_threshold', 100)

        # Homografía (BEV)
        self.declare_parameter('homography_matrix_path',
            'data/homography3.npy')
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

        # Cargar homografía (BEV)
        self.homography_matrix = None
        self.selecting_points  = False
        self.src_points        = []
        try:
            pkg_share = get_package_share_directory('line_follow_msr')
            rel_path = self.get_parameter('homography_matrix_path').value
            path = str((Path(pkg_share) / rel_path).resolve())
            self.homography_matrix = np.load(path)
            self.get_logger().info(f'✅ Homography matrix loaded from: {path}')
        except Exception as e:
            self.get_logger().error(f'❌ Failed to load homography: {e}')


        # ————— Publicadores y suscriptores ——————
        self.publisher = self.create_publisher(Twist, '/line_cmd_vel', 10)
        self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.create_subscription(Float32, '/fsm_action', self.fsm_action_callback, 10)
        self.activate_sub = self.create_subscription(Int16, '/line_follow_enable', self.activate_line_follow_callback, 10)
        self.active_line = 0
        self.windows_open = False

        self.trackbar_defaults = {
            'Blur Kernel':    24,
            'Morph Kernel':   5,
            'Block Size':     76,
            'C (Bias)':       13,
            'Min Area':       self.min_area_param
        }


        # ————— Ventana de trackbars (parámetros de visión) ——————
        if (self.active_line):
            cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
            cv2.createTrackbar('Blur Kernel', 'Controls', self.trackbar_defaults['Blur Kernel'], 31, lambda x: None)
            cv2.createTrackbar('Morph Kernel', 'Controls', self.trackbar_defaults['Morph Kernel'], 51, lambda x: None)
            cv2.createTrackbar('Block Size', 'Controls', self.trackbar_defaults['Block Size'], 101, lambda x: None)
            cv2.createTrackbar('C (Bias)', 'Controls', self.trackbar_defaults['C (Bias)'], 50, lambda x: None)
            cv2.createTrackbar('Min Area', 'Controls', self.trackbar_defaults['Min Area'], 2000, lambda x: None)


        # Timer a 50 Hz para percepción + control
        timer_period = 1.0 / 50.0
        self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info('Line Follower Node Started (50 Hz control)')

    def activate_line_follow_callback(self, msg):
        self.active_line = msg.data
        self.get_logger().info(f"line follow node state: {self.active_line}")

        if self.active_line:
            self.create_control_window()
        else:
            self.destroy_control_window()

    def create_control_window(self):
        if not self.windows_open:
            cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
            cv2.createTrackbar('Blur Kernel', 'Controls', 24, 31, lambda x: None)
            cv2.createTrackbar('Morph Kernel', 'Controls', 5, 51, lambda x: None)
            cv2.createTrackbar('Block Size', 'Controls', 76, 101, lambda x: None)
            cv2.createTrackbar('C (Bias)', 'Controls', 13, 50, lambda x: None)
            cv2.createTrackbar('Min Area', 'Controls', self.min_area_param, 2000, lambda x: None)
            self.windows_open = True
            self.get_logger().info("🎛️ Trackbars created.")

    def destroy_control_window(self):
        if self.windows_open:
            for win in ['Controls', 'Overlay', 'Left | Middle | Right', 'ROI']:
                try:
                    cv2.destroyWindow(win)
                except Exception:
                    pass
            cv2.waitKey(1)  # Let OpenCV flush pending GUI events
            self.windows_open = False
            self.get_logger().info("❌ All OpenCV windows closed.")


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
        # self.get_logger().info(f"Updated color_flag_multiplier to: {self.color_flag_multiplier}")

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
        
        if self.active_line == 0:
            return

        """Lazo principal de percepción + selección de la M real + control P a 50 Hz."""
        frame = self.latest_frame
        if frame is None:
            return

        # ————— Aplico homografía para BEV (si está disponible) —————
        if self.homography_matrix is not None:
            warped = cv2.warpPerspective(frame, self.homography_matrix, self.warp_size)
            warped = cv2.flip(warped, 1)
        else:
            warped = frame.copy()

        overlay = warped.copy()
        h, w, _ = warped.shape
        center_frame_x = w // 2

        # ————— Divido en 3 ROIs verticales: Left / Middle / Right —————
        twelve_div   = w // 12
        roi_left     = warped[:, 0:5 * twelve_div]
        roi_middle   = warped[:, 2 * twelve_div : 10 * twelve_div]
        roi_right    = warped[:, 7 * twelve_div : ]

        # ————— Lectura de trackbars —————
        blur_k = max(1, self.safe_get_trackbar('Blur Kernel', 'Controls') | 1)
        morph_k = max(1, self.safe_get_trackbar('Morph Kernel', 'Controls') | 1)
        block_size = max(3, self.safe_get_trackbar('Block Size', 'Controls') | 1)
        c_bias = self.safe_get_trackbar('C (Bias)', 'Controls')
        min_area_tb = self.safe_get_trackbar('Min Area', 'Controls')
        min_area    = max(self.min_area_param, min_area_tb)

        # ————— Preprocesado para cada ROI —————
        binary_left   = self.preprocess_region(roi_left,   blur_k, block_size, c_bias, morph_k)
        binary_middle = self.preprocess_region(roi_middle, blur_k, block_size, c_bias, morph_k)
        binary_right  = self.preprocess_region(roi_right,  blur_k, block_size, c_bias, morph_k)

        # ————— Detecto la línea en cada ROI (x_global, ángulo, longitud, contorno) —————
        line_l = self.detect_line_in_roi(binary_left,   0,             overlay, (255,   0,   0), "L", min_area)
        line_m = self.detect_line_in_roi(binary_middle, 2 * twelve_div, overlay, (255, 255,   0), "M", min_area)
        line_r = self.detect_line_in_roi(binary_right,  7 * twelve_div, overlay, (255,   0,   0), "R", min_area)

        # ————— Dibujar debug en overlay (coordenadas X y ángulos) —————
        x_l = line_l["x_global"] if line_l else "-"
        x_m = line_m["x_global"] if line_m else "-"
        x_r = line_r["x_global"] if line_r else "-"
        a_l = f'{line_l["angle"]:.1f}' if line_l else "-"
        a_m = f'{line_m["angle"]:.1f}' if line_m else "-"
        a_r = f'{line_r["angle"]:.1f}' if line_r else "-"
        text_x = f"x: L={x_l}  M={x_m}  R={x_r}"
        text_a = f"ang: L={a_l}°  M={a_m}°  R={a_r}°"
        cv2.putText(overlay, text_x, (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(overlay, text_a, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
        cv2.line(overlay, (center_frame_x, 0), (center_frame_x, h), (0, 0, 255), 2)

        # ————— Mostrar ventanas de debug —————
                # Display windows
        if (self.active_line == 1):
            if self.selecting_points:
                cv2.imshow("ROI", warped)
            resized_overlay = cv2.resize(overlay, None, fx=3.0, fy=3.0)
            cv2.imshow("Overlay", resized_overlay)
            stacked_rois = stack_with_dividers([binary_left, binary_middle, binary_right])
            cv2.imshow("Left | Middle | Right", stacked_rois)
            self.windows_open = True
            cv2.waitKey(1)
        elif self.windows_open:
            cv2.destroyWindow('ROI')
            cv2.destroyWindow('Binary Mask')
            cv2.destroyWindow('Left | Middle | Right')
            self.windows_open = False
            

        # ————— Construyo lista de candidatas: (“L”/“M”/“R”, dict, área) —————
        candidatas = []
        if line_l is not None:
            area_L = float(cv2.contourArea(line_l["contour"]))
            candidatas.append(("L", line_l, area_L))
        if line_m is not None:
            area_M = float(cv2.contourArea(line_m["contour"]))
            candidatas.append(("M", line_m, area_M))
        if line_r is not None:
            area_R = float(cv2.contourArea(line_r["contour"]))
            candidatas.append(("R", line_r, area_R))

        # ————— Guardo en historial (últimos frames) —————
        detected = set()
        if line_l: detected.add('L')
        if line_m: detected.add('M')
        if line_r: detected.add('R')
        self.last_detections.append(detected.copy())
        if len(self.last_detections) > self.max_history:
            self.last_detections.pop(0)

        # ————— Inicializo last_center_x si no está definido aún —————
        if self.last_center_x is None:
            self.last_center_x = float(center_frame_x)

        # ————— Elijo “línea real” según cercanía a last_center_x —————
        chosen_label = None
        chosen_line  = None

        if len(candidatas) == 0:
            # No hay ninguna línea → fallback
            chosen_line  = None
            chosen_label = None

        elif len(candidatas) == 1:
            # Sólo una candidata → la tomamos como real
            chosen_label = candidatas[0][0]
            chosen_line  = candidatas[0][1]
            if chosen_label == 'M':
                # Actualizo la última posición conocida de M
                self.last_center_x = float(chosen_line["x_global"])

        else:
            # Hay 2 o 3 candidatas: elijo la más cercana a last_center_x
            mejor_dist = None
            for etiqueta, linea, area in candidatas:
                x_i = float(linea["x_global"])
                dist = abs(x_i - self.last_center_x)
                if (mejor_dist is None) or (dist < mejor_dist):
                    mejor_dist    = dist
                    chosen_line   = linea
                    chosen_label  = etiqueta

            if chosen_label == 'M':
                # Si es M, actualizo last_center_x
                self.last_center_x = float(chosen_line["x_global"])
            # Si es L o R, dejo last_center_x igual (para forzar re-búsqueda de M real)
        
        # ————— Control P sobre la chosen_line o fallback completo —————
        if chosen_line is not None:
            linear_speed = 0.0
            angular_speed = 0.0

            x_error   = chosen_line["x_global"] - center_frame_x
            theta     = chosen_line["angle"]
            if theta > 0:
                angle_error = 90.0 - theta
            else:
                angle_error = -90.0 - theta
            if abs(angle_error) < self.ang_e_thrsh:
                angle_error = 0.0

            if (self.active_line):
                linear_speed  = self.xVel
                angular_speed = self.Kp_ang * angle_error - self.Kp_x * x_error
            else:
                angle_error = 0
                x_error = 0

            twist = Twist()
            twist.linear.x  = linear_speed * self.color_flag_multiplier
            twist.angular.z = angular_speed * self.color_flag_multiplier
            if (self.active_line):
                self.publisher.publish(twist)
            
            # self.get_logger().info(
            #     f"[{chosen_label}] x_err={x_error:.1f}, ang_err={angle_error:.1f}, "
            #     f"v={linear_speed:.3f}, ω={angular_speed:.3f}"
            # )
        else:
            # Fallback total: retroceder suavemente
            twist = Twist()
            twist.linear.x  = -0.10 * self.color_flag_multiplier
            twist.angular.z = 0.0
            if (self.active_line):
                self.publisher.publish(twist)
            self.get_logger().warn("No line válida → fallback completo")

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
            offset_contour = [
                np.array([[pt[0][0] + roi_x_offset, pt[0][1]]]) for pt in largest
            ]
            offset_contour = np.array(offset_contour, dtype=np.int32)
            cv2.drawContours(overlay, [offset_contour], -1, (0,255,0), 2)
            pt1 = (roi_x_offset + int(x0 - vx * 50), int(y0 - vy * 50))
            pt2 = (roi_x_offset + int(x0 + vx * 50), int(y0 + vy * 50))
            cv2.line(overlay, pt1, pt2, color, 2)

        return {
            "x_global": global_x,
            "angle":    np.degrees(np.arctan2(vy, vx)),
            "length":   length,
            "contour":  largest
        }

    def safe_get_trackbar(self, name, window):
        try:
            return cv2.getTrackbarPos(name, window)
        except cv2.error:
            return self.trackbar_defaults.get(name, 0)  # fallback if missing from dict


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
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
