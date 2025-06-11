#!/usr/bin/env python3

import rclpy
from rclpy_lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Int16
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
from ament_index_python.packages import get_package_share_directory
from pathlib import Path

class LineFollowerCentroid(LifecycleNode):
    """
    Lifecycle-enabled ROS2 node for line following on track with 3 marks (L, M, R).
    Uses lifecycle states to cleanly start/stop line following behavior.
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

        # Cantidad máxima de frames sin línea central antes de fallback
        self.declare_parameter('max_missing_center', 5)

        # ————— Variables de estado generales ——————
        self.color_flag_multiplier = 1.0
        self.latest_frame = None

        # Historial de detecciones
        self.last_detections = []
        self.max_history = 10

        # Buffer para supresión de jitter en la línea media (M)
        self.middle_line_buffer      = []
        self.middle_line_buffer_size = 10
        self.last_valid_middle_x     = None
        self.x_diff_threshold        = 50

        # Variable de "última posición conocida de la M real"
        self.last_center_x = None

        # Lifecycle-specific variables
        self.control_timer = None
        self.publisher = None
        self.image_sub = None
        self.fsm_action_sub = None
        self.windows_open = False

        # Cargar homografía (BEV)
        self.homography_matrix = None
        self.selecting_points = False
        self.src_points = []

        self.trackbar_defaults = {
            'Blur Kernel':    24,
            'Morph Kernel':   5,
            'Block Size':     76,
            'C (Bias)':       13,
            'Min Area':       500
        }

        self.get_logger().info('LineFollowerCentroid lifecycle node created')

    # ====================== LIFECYCLE CALLBACKS ======================

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Configure the node - load parameters and initialize resources"""
        self.get_logger().info('Configuring LineFollowerCentroid...')

        try:
            # Read parameter values
            p = self.get_parameter
            self.Kp_x = float(p('Kp_x').value)
            self.Kp_ang = float(p('Kp_ang').value)
            self.xVel = float(p('xVel').value)
            self.ang_e_thrsh = float(p('ang_e_thrsh').value)
            self.min_area_param = int(p('min_area_param').value)
            self.length_threshold = float(p('length_threshold').value)
            self.max_missing_center = int(p('max_missing_center').value)

            # Warp parameters
            homography_path = p('homography_matrix_path').value
            warp_w = int(p('warp_width').value)
            warp_h = int(p('warp_height').value)
            self.warp_size = (warp_w, warp_h)

            # Load homography matrix
            try:
                pkg_share = get_package_share_directory('line_follow_msr')
                rel_path = self.get_parameter('homography_matrix_path').value
                path = str((Path(pkg_share) / rel_path).resolve())
                self.homography_matrix = np.load(path)
                self.get_logger().info(f'✅ Homography matrix loaded from: {path}')
            except Exception as e:
                self.get_logger().error(f'❌ Failed to load homography: {e}')
                return TransitionCallbackReturn.FAILURE

            # Create publishers and subscribers (but don't activate them yet)
            self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
            self.image_sub = self.create_subscription(
                Image, '/image_raw', self.image_callback, 10)
            self.fsm_action_sub = self.create_subscription(
                Float32, '/fsm_action', self.fsm_action_callback, 10)

            # Add parameter callback
            self.add_on_set_parameters_callback(self.parameter_update_callback)

            self.get_logger().info('✅ LineFollowerCentroid configured successfully')
            return TransitionCallbackReturn.SUCCESS

        except Exception as e:
            self.get_logger().error(f'❌ Configuration failed: {e}')
            return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Activate the node - start control loop and open debug windows"""
        self.get_logger().info('🚀 Activating LineFollowerCentroid...')

        try:
            # Start the control timer at 50 Hz
            timer_period = 1.0 / 50.0
            self.control_timer = self.create_timer(timer_period, self.timer_callback)

            # Create control windows
            self.create_control_window()

            # Reset state variables
            self.last_center_x = None
            self.middle_line_buffer.clear()
            self.last_detections.clear()

            # Activate the publisher
            self.publisher.on_activate(state)

            self.get_logger().info('✅ LineFollowerCentroid activated - line following ACTIVE')
            return TransitionCallbackReturn.SUCCESS

        except Exception as e:
            self.get_logger().error(f'❌ Activation failed: {e}')
            return TransitionCallbackReturn.FAILURE

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Deactivate the node - stop control loop and close windows"""
        self.get_logger().info('🛑 Deactivating LineFollowerCentroid...')

        try:
            # Stop the robot immediately
            if self.publisher is not None:
                stop_twist = Twist()
                self.publisher.publish(stop_twist)
                self.get_logger().info('🛑 Robot stopped')

            # Stop the control timer
            if self.control_timer is not None:
                self.control_timer.cancel()
                self.control_timer = None

            # Close debug windows
            self.destroy_control_window()

            # Deactivate publisher
            if self.publisher is not None:
                self.publisher.on_deactivate(state)

            self.get_logger().info('✅ LineFollowerCentroid deactivated - line following STOPPED')
            return TransitionCallbackReturn.SUCCESS

        except Exception as e:
            self.get_logger().error(f'❌ Deactivation failed: {e}')
            return TransitionCallbackReturn.FAILURE

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Cleanup resources"""
        self.get_logger().info('🧹 Cleaning up LineFollowerCentroid...')

        try:
            # Destroy all resources
            if self.publisher is not None:
                self.destroy_publisher(self.publisher)
                self.publisher = None

            if self.image_sub is not None:
                self.destroy_subscription(self.image_sub)
                self.image_sub = None

            if self.fsm_action_sub is not None:
                self.destroy_subscription(self.fsm_action_sub)
                self.fsm_action_sub = None

            # Close any remaining windows
            self.destroy_control_window()

            self.get_logger().info('✅ LineFollowerCentroid cleaned up')
            return TransitionCallbackReturn.SUCCESS

        except Exception as e:
            self.get_logger().error(f'❌ Cleanup failed: {e}')
            return TransitionCallbackReturn.FAILURE

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Shutdown the node"""
        self.get_logger().info('🔴 Shutting down LineFollowerCentroid...')
        
        # Ensure robot is stopped
        if self.publisher is not None:
            try:
                stop_twist = Twist()
                self.publisher.publish(stop_twist)
            except:
                pass

        # Close all windows
        self.destroy_control_window()
        
        return TransitionCallbackReturn.SUCCESS

    # ====================== CONTROL AND VISION METHODS ======================

    def create_control_window(self):
        """Create OpenCV control windows for debugging"""
        if not self.windows_open:
            try:
                cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
                cv2.createTrackbar('Blur Kernel', 'Controls', 24, 31, lambda x: None)
                cv2.createTrackbar('Morph Kernel', 'Controls', 5, 51, lambda x: None)
                cv2.createTrackbar('Block Size', 'Controls', 76, 101, lambda x: None)
                cv2.createTrackbar('C (Bias)', 'Controls', 13, 50, lambda x: None)
                cv2.createTrackbar('Min Area', 'Controls', self.min_area_param, 2000, lambda x: None)
                self.windows_open = True
                self.get_logger().info("🎛️ Debug windows created")
            except Exception as e:
                self.get_logger().warn(f"Could not create debug windows: {e}")

    def destroy_control_window(self):
        """Close all OpenCV windows"""
        if self.windows_open:
            try:
                for win in ['Controls', 'Overlay', 'Left | Middle | Right', 'ROI']:
                    try:
                        cv2.destroyWindow(win)
                    except:
                        pass
                cv2.waitKey(1)
                self.windows_open = False
                self.get_logger().info("❌ Debug windows closed")
            except Exception as e:
                self.get_logger().warn(f"Error closing windows: {e}")

    def parameter_update_callback(self, params):
        """Update dynamic parameters in real-time"""
        successful = True
        for pp in params:
            try:
                if pp.name == 'Kp_x' and pp.type_ == Parameter.Type.DOUBLE:
                    self.Kp_x = float(pp.value)
                    self.get_logger().info(f'Kp_x updated to {self.Kp_x}')
                elif pp.name == 'Kp_ang' and pp.type_ == Parameter.Type.DOUBLE:
                    self.Kp_ang = float(pp.value)
                    self.get_logger().info(f'Kp_ang updated to {self.Kp_ang}')
                elif pp.name == 'xVel' and pp.type_ == Parameter.Type.DOUBLE:
                    self.xVel = float(pp.value)
                    self.get_logger().info(f'xVel updated to {self.xVel}')
                elif pp.name == 'ang_e_thrsh' and pp.type_ == Parameter.Type.DOUBLE:
                    self.ang_e_thrsh = float(pp.value)
                    self.get_logger().info(f'ang_e_thrsh updated to {self.ang_e_thrsh}')
                elif pp.name == 'min_area_param' and pp.type_ in (Parameter.Type.INTEGER, Parameter.Type.DOUBLE):
                    self.min_area_param = int(pp.value)
                    if self.windows_open:
                        cv2.setTrackbarPos('Min Area', 'Controls', self.min_area_param)
                    self.get_logger().info(f'min_area_param updated to {self.min_area_param}')
                elif pp.name == 'length_threshold' and pp.type_ in (Parameter.Type.INTEGER, Parameter.Type.DOUBLE):
                    self.length_threshold = float(pp.value)
                    self.get_logger().info(f'length_threshold updated to {self.length_threshold}')
                else:
                    self.get_logger().warn(f'Parameter {pp.name} not handled.')
                    successful = False
            except Exception as e:
                self.get_logger().error(f'Error updating {pp.name}: {e}')
                successful = False

        return rclpy.parameter.ParameterEventHandlerResult(successful=successful)

    def fsm_action_callback(self, msg: Float32):
        """Allow varying speed multiplier from external FSM"""
        self.color_flag_multiplier = msg.data

    def image_callback(self, msg: Image):
        """Save latest received image for processing in timer_callback"""
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Error converting ROS→CV2 image: {e}")
            self.latest_frame = None

    def preprocess_region(self, region, blur_k, block_size, c_bias, morph_k):
        """Contrast-limited → blur → adaptive-threshold → morphology"""
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
        """Main perception + line selection + P control loop at 50 Hz"""
        # Only run if we're in the ACTIVE state
        if self.get_current_state().id != LifecycleState.PRIMARY_STATE_ACTIVE:
            return

        frame = self.latest_frame
        if frame is None:
            return

        # Apply homography for BEV (if available)
        if self.homography_matrix is not None:
            warped = cv2.warpPerspective(frame, self.homography_matrix, self.warp_size)
            warped = cv2.flip(warped, 1)
        else:
            warped = frame.copy()

        overlay = warped.copy()
        h, w, _ = warped.shape
        center_frame_x = w // 2

        # Divide into 3 vertical ROIs: Left / Middle / Right
        twelve_div = w // 12
        roi_left = warped[:, 0:5 * twelve_div]
        roi_middle = warped[:, 2 * twelve_div : 10 * twelve_div]
        roi_right = warped[:, 7 * twelve_div : ]

        # Read trackbar values
        blur_k = max(1, self.safe_get_trackbar('Blur Kernel', 'Controls') | 1)
        morph_k = max(1, self.safe_get_trackbar('Morph Kernel', 'Controls') | 1)
        block_size = max(3, self.safe_get_trackbar('Block Size', 'Controls') | 1)
        c_bias = self.safe_get_trackbar('C (Bias)', 'Controls')
        min_area_tb = self.safe_get_trackbar('Min Area', 'Controls')
        min_area = max(self.min_area_param, min_area_tb)

        # Preprocess each ROI
        binary_left = self.preprocess_region(roi_left, blur_k, block_size, c_bias, morph_k)
        binary_middle = self.preprocess_region(roi_middle, blur_k, block_size, c_bias, morph_k)
        binary_right = self.preprocess_region(roi_right, blur_k, block_size, c_bias, morph_k)

        # Detect line in each ROI
        line_l = self.detect_line_in_roi(binary_left, 0, overlay, (255, 0, 0), "L", min_area)
        line_m = self.detect_line_in_roi(binary_middle, 2 * twelve_div, overlay, (255, 255, 0), "M", min_area)
        line_r = self.detect_line_in_roi(binary_right, 7 * twelve_div, overlay, (255, 0, 0), "R", min_area)

        # Debug display
        if self.windows_open:
            self._display_debug_info(overlay, line_l, line_m, line_r, center_frame_x, h, w,
                                   binary_left, binary_middle, binary_right, warped)

        # Build candidates list
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

        # Initialize last_center_x if not defined yet
        if self.last_center_x is None:
            self.last_center_x = float(center_frame_x)

        # Choose "real line" based on proximity to last_center_x
        chosen_label = None
        chosen_line = None

        if len(candidatas) == 0:
            chosen_line = None
            chosen_label = None
        elif len(candidatas) == 1:
            chosen_label = candidatas[0][0]
            chosen_line = candidatas[0][1]
            if chosen_label == 'M':
                self.last_center_x = float(chosen_line["x_global"])
        else:
            # Choose closest to last_center_x
            mejor_dist = None
            for etiqueta, linea, area in candidatas:
                x_i = float(linea["x_global"])
                dist = abs(x_i - self.last_center_x)
                if (mejor_dist is None) or (dist < mejor_dist):
                    mejor_dist = dist
                    chosen_line = linea
                    chosen_label = etiqueta

            if chosen_label == 'M':
                self.last_center_x = float(chosen_line["x_global"])

        # P control on chosen_line or complete fallback
        if chosen_line is not None:
            x_error = chosen_line["x_global"] - center_frame_x
            theta = chosen_line["angle"]
            if theta > 0:
                angle_error = 90.0 - theta
            else:
                angle_error = -90.0 - theta
            if abs(angle_error) < self.ang_e_thrsh:
                angle_error = 0.0

            linear_speed = self.xVel
            angular_speed = self.Kp_ang * angle_error - self.Kp_x * x_error

            twist = Twist()
            twist.linear.x = linear_speed * self.color_flag_multiplier
            twist.angular.z = angular_speed * self.color_flag_multiplier
            
            # Only publish if we're in active state
            if self.get_current_state().id == LifecycleState.PRIMARY_STATE_ACTIVE:
                self.publisher.publish(twist)
        else:
            # Fallback: back up slowly
            twist = Twist()
            twist.linear.x = -0.10 * self.color_flag_multiplier
            twist.angular.z = 0.0
            
            if self.get_current_state().id == LifecycleState.PRIMARY_STATE_ACTIVE:
                self.publisher.publish(twist)
            self.get_logger().warn("No valid line → complete fallback")

    def _display_debug_info(self, overlay, line_l, line_m, line_r, center_frame_x, h, w,
                           binary_left, binary_middle, binary_right, warped):
        """Display debug information in OpenCV windows"""
        try:
            # Draw debug text
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

            # Show windows
            if self.selecting_points:
                cv2.imshow("ROI", warped)
            resized_overlay = cv2.resize(overlay, None, fx=3.0, fy=3.0)
            cv2.imshow("Overlay", resized_overlay)
            stacked_rois = stack_with_dividers([binary_left, binary_middle, binary_right])
            cv2.imshow("Left | Middle | Right", stacked_rois)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().warn(f"Debug display error: {e}")

    def detect_line_in_roi(self, binary_roi, roi_x_offset, overlay=None, color=(255,255,0), label="", min_area=150):
        """Detect largest contour in binary_roi, fit line, filter short contours"""
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

        length = cv2.arcLength(largest, closed=False)
        if length < self.length_threshold:
            return None

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
            "angle": np.degrees(np.arctan2(vy, vx)),
            "length": length,
            "contour": largest
        }

    def safe_get_trackbar(self, name, window):
        try:
            return cv2.getTrackbarPos(name, window)
        except cv2.error:
            return self.trackbar_defaults.get(name, 0)


def stack_with_dividers(imgs, divider_thickness=3, divider_color=255):
    """Stack grayscale images horizontally with vertical dividers"""
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
    except KeyboardInterrupt:
        node.get_logger().info("🛑 Shutting down LineFollowerCentroid...")
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()