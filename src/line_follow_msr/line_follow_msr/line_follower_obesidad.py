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
      • FSM de 5 estados (RECTA, CURVA_IZQ, CURVA_DER, LINEA_UNICA, FALLBACK)
        basado en cuántas líneas se detectan y sus ángulos.
      • Buffer de ~n~ mediciones de la línea central (M) para suprimir jitter
        (si el salto excede el umbral, M se da por perdida inmediatamente).
      • Control P sobre la línea “elegida” en cada estado.
      • Umbral adicional de longitud mínima de contorno para filtrar líneas muy cortas.
    """

    # ---- Definición de los diferentes estados ----
    ESTADO_RECTO       = 0
    ESTADO_CURVA_IZQ   = 1
    ESTADO_CURVA_DER   = 2
    ESTADO_LINEA_UNICA = 3
    ESTADO_FALLBACK    = 4

    def __init__(self):
        super().__init__('line_follower_centroid')
        self.bridge = CvBridge()

        # —————— Parámetros dinámicos ——————
        # Control P
        self.declare_parameter('Kp_x',             0.01)
        self.declare_parameter('Kp_ang',           0.019)
        self.declare_parameter('xVel',             0.15)
        self.declare_parameter('ang_e_thrsh',      2.0)

        # Umbral mínimo de área para filtrar contornos
        self.declare_parameter('min_area_param',   500)

        # Umbral mínimo de longitud del contorno (para descartar contornos pequeños)
        self.declare_parameter('length_threshold', 50)

        # Homografía (BEV)
        self.declare_parameter('homography_matrix_path',
            '/home/idmx/ros2_ws_2/src/line_follow_msr/data/homography3.npy')
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
        self.frames_without_center = 0
        self.latest_frame = None

        # Historial de detecciones para FSM
        self.last_detections = []
        self.max_history = 10

        # Buffer para supresión de jitter en la línea media (M)
        self.middle_line_buffer      = []
        self.middle_line_buffer_size = 10
        self.last_valid_middle_x     = None
        self.x_diff_threshold        = 50   # ↑ aumentado a 30 px para curvas

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
                else:
                    # Parámetro no gestionado → marcamos no exitoso
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
        """Lazo principal de percepción + FSM + control P a 50 Hz."""
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

        # ————— Divido en 3 ROIs verticales: Left / Middle / Right —————
        twelve_div = w // 12
        roi_left   = warped[:, 0:5 * twelve_div]
        roi_middle = warped[:, 2 * twelve_div : 10 * twelve_div]
        roi_right  = warped[:, 7 * twelve_div : ]

        # ————— Lectura de trackbars —————
        blur_k      = max(1, cv2.getTrackbarPos('Blur Kernel', 'Controls') | 1)
        morph_k     = max(1, cv2.getTrackbarPos('Morph Kernel', 'Controls') | 1)
        block_size  = max(3, cv2.getTrackbarPos('Block Size', 'Controls') | 1)
        c_bias      = cv2.getTrackbarPos('C (Bias)', 'Controls')
        min_area_tb = cv2.getTrackbarPos('Min Area', 'Controls')
        min_area    = max(self.min_area_param, min_area_tb)

        # ————— Preprocesado para cada ROI —————
        binary_left   = self.preprocess_region(roi_left,   blur_k, block_size, c_bias, morph_k)
        binary_middle = self.preprocess_region(roi_middle, blur_k, block_size, c_bias, morph_k)
        binary_right  = self.preprocess_region(roi_right,  blur_k, block_size, c_bias, morph_k)

        # ————— Detecto la línea en cada ROI (x_global, ángulo, longitud, contorno) —————
        line_l = self.detect_line_in_roi(binary_left,   0,               overlay, (255,   0,   0), "L", min_area)
        line_m = self.detect_line_in_roi(binary_middle, 2 * twelve_div,  overlay, (255, 255,   0), "M", min_area)
        line_r = self.detect_line_in_roi(binary_right,  7 * twelve_div,  overlay, (255,   0,   0), "R", min_area)

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
        center_x = w // 2
        cv2.line(overlay, (center_x, 0), (center_x, h), (0, 0, 255), 2)

        # ————— Mostrar ventanas de debug —————
        if self.selecting_points:
            cv2.imshow("ROI", warped)
        resized_overlay = cv2.resize(overlay, None, fx=3.0, fy=3.0)
        cv2.imshow("Overlay", resized_overlay)
        stacked_rois = stack_with_dividers([binary_left, binary_middle, binary_right])
        cv2.imshow("Left | Middle | Right", stacked_rois)

        # ————— Construyo el conjunto de líneas detectadas (tags “L”, “M”, “R”) —————
        detected = set()
        if line_l: detected.add('L')
        if line_m: detected.add('M')
        if line_r: detected.add('R')

        # ————— Guardo en historial (últimos frames) —————
        self.last_detections.append(detected.copy())
        if len(self.last_detections) > self.max_history:
            self.last_detections.pop(0)

        # ————— Determino el estado FSM según cuántas líneas y sus combinaciones —————
        estado = self.ESTADO_FALLBACK
        if len(detected) == 3:
            estado = self.ESTADO_RECTO
        elif len(detected) == 2:
            if {'L','M'} <= detected:
                estado = self.ESTADO_CURVA_IZQ
            elif {'M','R'} <= detected:
                estado = self.ESTADO_CURVA_DER
            else:
                # Caso raro “L+R sin M” → lo tratamos como recto
                estado = self.ESTADO_RECTO
        elif len(detected) == 1:
            estado = self.ESTADO_LINEA_UNICA
        else:
            estado = self.ESTADO_FALLBACK

        # ————— FSM: elijo “línea base” para control según el estado —————
        chosen = None
        chosen_label = None

        if estado == self.ESTADO_RECTO:
            # En recta siempre preferimos la línea media (M)
            chosen = line_m
            chosen_label = 'M'
            self.frames_without_center = 0

        elif estado == self.ESTADO_CURVA_IZQ:
            # Curva a la izquierda: prefiero M, si no, uso L
            if line_m:
                chosen = line_m
                chosen_label = 'M'
                self.frames_without_center = 0
            else:
                chosen = line_l
                chosen_label = 'L'

        elif estado == self.ESTADO_CURVA_DER:
            # Curva a la derecha: prefiero M, si no, uso R
            if line_m:
                chosen = line_m
                chosen_label = 'M'
                self.frames_without_center = 0
            else:
                chosen = line_r
                chosen_label = 'R'

        elif estado == self.ESTADO_LINEA_UNICA:
            # Sólo hay una línea visible: inferimos si antes hubo {L,M} o {M,R}
            self.frames_without_center += 1
            inferido = None
            for past in reversed(self.last_detections):
                if len(past) == 2:
                    if past == {'L','M'} and detected == {'L'}:
                        inferido = 'L'
                    elif past == {'M','R'} and detected == {'R'}:
                        inferido = 'R'
                    break

            if inferido == 'L' and line_l:
                chosen = line_l
                chosen_label = 'L'
            elif inferido == 'R' and line_r:
                chosen = line_r
                chosen_label = 'R'
            else:
                # Si no infiero nada, intento usar M
                if line_m:
                    chosen = line_m
                    chosen_label = 'M'
                else:
                    # Si ninguna, uso lateral casi vertical (±20° ahora en vez de ±10°)
                    cand_l = (line_l and abs(line_l["angle"] - 90.0) < 20.0)
                    cand_r = (line_r and abs(line_r["angle"] - 90.0) < 20.0)
                    if cand_l:
                        chosen = line_l
                        chosen_label = 'L'
                    elif cand_r:
                        chosen = line_r
                        chosen_label = 'R'
                    else:
                        chosen = None

        elif estado == self.ESTADO_FALLBACK:
            chosen = None

        # ————— Supresión de jitter en línea M (sin drift frames, umbral aumentado) —————
        use_middle = False
        if chosen_label == 'M' and line_m:
            current_x = line_m["x_global"]
            self.middle_line_buffer.append(current_x)
            if len(self.middle_line_buffer) > self.middle_line_buffer_size:
                self.middle_line_buffer.pop(0)

            recent_avg = (
                np.mean(self.middle_line_buffer[-3:]) 
                if len(self.middle_line_buffer) >= 3 
                else current_x
            )
            x_diff = abs(current_x - recent_avg)

            if x_diff < self.x_diff_threshold:
                use_middle = True
                self.last_valid_middle_x = current_x
            else:
                # Si supera el umbral inmediato, declaramos M como “perdida”
                use_middle = False

        # ————— Decido la línea final a usar (final_line) aprovechando buffer/fallback —————
        final_line  = None
        final_label = None

        if chosen_label == 'M':
            if use_middle:
                final_line  = line_m
                final_label = 'M'
            else:
                # M se perdió → busco lateral más cercana a last_valid_middle_x
                candidates = []
                if line_l and (self.last_valid_middle_x is not None):
                    candidates.append(('L', line_l,
                        abs(line_l['x_global'] - self.last_valid_middle_x)))
                if line_r and (self.last_valid_middle_x is not None):
                    candidates.append(('R', line_r,
                        abs(line_r['x_global'] - self.last_valid_middle_x)))
                if candidates:
                    chosen_side = min(candidates, key=lambda x: x[2])
                    final_label = chosen_side[0]
                    final_line  = chosen_side[1]
                else:
                    final_line  = None
                    final_label = None
        else:
            # Si elegí L o R explícitamente, lo acepto
            if chosen_label in ('L','R') and chosen is not None:
                final_line  = chosen
                final_label = chosen_label
            else:
                final_line  = None
                final_label = None

        # ————— Control P sobre la línea “final_line” seleccionada —————
        if final_line:
            # Cálculo de errores lateral y angular
            x_error = final_line["x_global"] - center_x
            line_angle = final_line["angle"]
            if line_angle > 0:
                angle_error = 90.0 - line_angle
            else:
                angle_error = -90.0 - line_angle
            if abs(angle_error) < self.ang_e_thrsh:
                angle_error = 0.0

            linear_speed  = self.xVel
            angular_speed = self.Kp_ang * angle_error - self.Kp_x * x_error

            twist = Twist()
            twist.linear.x  = linear_speed * self.color_flag_multiplier
            twist.angular.z = angular_speed * self.color_flag_multiplier
            self.publisher.publish(twist)

            self.get_logger().info(
                f"[{final_label}] x_err={x_error:.1f}, ang_err={angle_error:.1f}, "
                f"v={linear_speed:.3f}, ω={angular_speed:.3f}"
            )
        else:
            # ————— Fallback total: retroceder suavemente —————
            twist = Twist()
            twist.linear.x  = -0.10 * self.color_flag_multiplier
            twist.angular.z = 0.0
            self.publisher.publish(twist)
            self.get_logger().warn("No line válida → fallback completo")

        cv2.waitKey(1)

    def detect_line_in_roi(self, binary_roi, roi_x_offset, overlay=None, color=(255,255,0), label="", min_area=150):
        """
        Detecta el contorno más grande en binary_roi (área > min_area), ajusta recta con fitLine,
        filtra si el contorno ajustado es muy corto (< length_threshold), y devuelve:
          • x_global: coordinada X en toda la imagen warp
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
