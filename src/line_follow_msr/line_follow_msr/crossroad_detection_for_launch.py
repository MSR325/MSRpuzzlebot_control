import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from std_msgs.msg import Int32MultiArray, Int16
from ament_index_python.packages import get_package_share_directory
from pathlib import Path
'''
ESTE NODO VE A PARTIR DE 15 cm efrente del robot y termina de ver a los 15 + 26.4 cm aprox (alto)
(ancho) aprox 30 cm de lado a lado
'''

class CrossroadDetector(Node):
    def __init__(self):
        super().__init__('crossroad_detector')
        self.bridge = CvBridge()

        self.active_crossroad_detection = 0

        # Parameters
        self.declare_parameter('homography_matrix_path', 'data/homography_after_calib_firstsegment_decent.npy')
        self.declare_parameter('warp_width', 300)
        self.declare_parameter('warp_height', 300)
        self.declare_parameter('min_segments', 4)
        self.declare_parameter('y_alignment_thresh_px', 100)  # vertical threshold in pixels
        self.declare_parameter('vertical_angle_thresh', 20)  # vertical threshold in pixels

        # --- add parameters near the top of CrossroadDetector.__init__
        self.declare_parameter('x_meter_range', 0.30)
        self.declare_parameter('y_meter_range', 0.264)
        self.x_meter_range = self.get_parameter('x_meter_range').value
        self.y_meter_range = self.get_parameter('y_meter_range').value


        self.min_segments = self.get_parameter('min_segments').value
        self.y_alignment_thresh_px = self.get_parameter('y_alignment_thresh_px').value
        self.vertical_angle_thresh = self.get_parameter('vertical_angle_thresh').value
        self.homography_matrix = None
        self.warp_size = (
            self.get_parameter('warp_width').value,
            self.get_parameter('warp_height').value
        )

        # Load homography
        try:
            pkg_share = get_package_share_directory('line_follow_msr')
            rel_path = self.get_parameter('homography_matrix_path').value
            path = str((Path(pkg_share) / rel_path).resolve())
            self.homography_matrix = np.load(path)
            self.get_logger().info(f'✅ Homography matrix loaded from: {path}')
        except Exception as e:
            self.get_logger().error(f'❌ Failed to load homography: {e}')

        # Subscribers
        self.crossroad_detect_enable_sub = self.create_subscription(Int16, '/crossroad_detect_enable', self.enable_callback, 10)
        self.image_sub = self.create_subscription(Image, '/image_undistorted', self.image_callback, 10)
        self.crossroad_decision = 0  # default = go straight
        self.decision_sub = self.create_subscription(Int16, '/crossroad_decision', self.decision_callback, 10)

        # Publishers 
        self.flags_pub = self.create_publisher(Int32MultiArray, '/crossroad_detected', 10)
        self.centroids_pub = self.create_publisher(Int32MultiArray, '/crossroad_centroids', 10)

        self.windows_open = False
        
    def enable_callback(self, msg: Int16):
        self.active_crossroad_detection = msg.data
        state = "✅ ENABLED" if msg.data == 1 else "⛔ DISABLED"
        self.get_logger().info(f"Crossroad detection is now {state}")

        if msg.data == 1:
            self.create_control_window()
        else:
            self.destroy_control_window()

    def safe_get(self, name, default):
        try:
            return cv2.getTrackbarPos(name, 'Controls')
        except cv2.error:
            return default

    def destroy_control_window(self):
        """Closes all OpenCV GUI windows used by the node."""
        if self.windows_open:
            window_names = [
                'Controls',
                'Crossroad Debug'
            ]
            for name in window_names:
                try:
                    cv2.destroyWindow(name)
                except Exception:
                    pass
            cv2.waitKey(1)  # Flush GUI event queue
            self.windows_open = False
            self.get_logger().info("🧼 All OpenCV windows destroyed.")



    def create_control_window(self):
        if self.windows_open:
            return  # already open

        cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
        cv2.createTrackbar('Blur Kernel', 'Controls', 28, 31, lambda x: None)
        cv2.createTrackbar('Morph Kernel', 'Controls', 10, 31, lambda x: None)
        cv2.createTrackbar('Block Size', 'Controls', 97, 201, lambda x: None)
        cv2.createTrackbar('C (Bias)', 'Controls', 16, 30, lambda x: None)
        cv2.createTrackbar('Min Aspect X x10', 'Controls', 8, 100, lambda x: None)
        cv2.createTrackbar('Max Aspect X x10', 'Controls', 30, 100, lambda x: None)
        cv2.createTrackbar('Min Aspect Y x10', 'Controls', 4, 100, lambda x: None)
        cv2.createTrackbar('Max Aspect Y x10', 'Controls', 7, 100, lambda x: None)
        cv2.createTrackbar('Min Segments', 'Controls', int(self.min_segments), 8, lambda x: None)
        cv2.createTrackbar('Horizontal ang Thrshld', 'Controls', 20, 90, lambda x: None)
        cv2.createTrackbar('Y Align Threshold', 'Controls', int(self.y_alignment_thresh_px), 200, lambda x: None)
        cv2.createTrackbar('Vertical ang Thrshld', 'Controls', int(self.vertical_angle_thresh), 90, lambda x: None)
        cv2.createTrackbar('X Align Threshold', 'Controls', 40, 200, lambda x: None)
        cv2.createTrackbar('Min Width', 'Controls', 2, 300, lambda x: None)
        cv2.createTrackbar('Max Width', 'Controls', 115, 300, lambda x: None)
        cv2.createTrackbar('Min Height', 'Controls', 25, 300, lambda x: None)
        cv2.createTrackbar('Max Height', 'Controls', 89, 300, lambda x: None)
        cv2.createTrackbar('Min Area', 'Controls', 756, 5000, lambda x: None)
        cv2.createTrackbar('Max Area', 'Controls', 1781, 10000, lambda x: None)
        cv2.createTrackbar('Residual px', 'Controls', 5, 20, lambda x: None)
        cv2.createTrackbar('Scale', 'Controls', 2, 10, lambda x: None)

        self.windows_open = True


    # ---------------------------------------------------------------------------
    # 2.  🔧  Pre-processing with kernel cache
    # ---------------------------------------------------------------------------
    def preprocess(self, img, tb):
        """Threshold + morphology.  Kernel is rebuilt only if size changes."""
        # ---- Blur + adaptive threshold ----
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(hsv[:, :, 2])
        blurred = cv2.GaussianBlur(v_eq, (tb['blur_k'], tb['blur_k']), 0)
        binary  = cv2.adaptiveThreshold(blurred, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV,
                                        tb['block_sz'], tb['c_bias'])

        # ---- Kernel caching ----
        if not hasattr(self, '_prev_morph_k'):
            self._prev_morph_k = -1          # first-time init
            self._morph_kernel = None

        if tb['morph_k'] != self._prev_morph_k:
            self._morph_kernel = np.ones((tb['morph_k'], tb['morph_k']), np.uint8)
            self._prev_morph_k = tb['morph_k']

        binary = cv2.erode(binary,  self._morph_kernel, iterations=1)
        binary = cv2.dilate(binary, self._morph_kernel, iterations=2)
        return binary
    
    # ---------------------------------------------------------------------------
    # 1.  🔧  Track-bar caching ­– read once per frame
    # ---------------------------------------------------------------------------
    def _snapshot_trackbar_state(self):
        d = {}

        # ——— Preprocessing ———
        d['blur_k']   = max(1, self.safe_get('Blur Kernel', 28) | 1)
        d['morph_k']  = max(1, self.safe_get('Morph Kernel', 10) | 1)
        d['block_sz'] = max(3, self.safe_get('Block Size', 97) | 1)
        d['c_bias']   = self.safe_get('C (Bias)', 16)

        # ——— Contour Geometry ———
        d['min_w']    = self.safe_get('Min Width', 2)
        d['max_w']    = self.safe_get('Max Width', 115)
        d['min_h']    = self.safe_get('Min Height', 25)
        d['max_h']    = self.safe_get('Max Height', 89)
        d['area_min'] = self.safe_get('Min Area', 756)
        d['area_max'] = self.safe_get('Max Area', 1781)

        # ——— Aspect Ratios ———
        d['aspect_min_x'] = self.safe_get('Min Aspect X x10', 8) / 10.0
        d['aspect_max_x'] = self.safe_get('Max Aspect X x10', 30) / 10.0
        d['aspect_min_y'] = self.safe_get('Min Aspect Y x10', 4) / 10.0
        d['aspect_max_y'] = self.safe_get('Max Aspect Y x10', 7) / 10.0

        # ——— Alignment / Angles ———
        d['y_align']      = self.safe_get('Y Align Threshold', 100)
        d['x_align']      = self.safe_get('X Align Threshold', 40)
        d['ang_thr_h']    = self.safe_get('Horizontal ang Thrshld', 20)
        d['ang_thr_v']    = self.safe_get('Vertical ang Thrshld', 20)
        d['min_segments'] = self.safe_get('Min Segments', self.min_segments)

        return d


    def is_dotted_row(self, binary_roi, overlay, roi_offset_y, tb):
        contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = []

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)

            if not (tb['area_min'] <= area <= tb['area_max']):
                continue

            aspect = w / max(h, 1)
            if not (tb['min_w'] <= w <= tb['max_w'] and
                    tb['min_h'] <= h <= tb['max_h'] and
                    tb['aspect_min_x'] <= aspect <= tb['aspect_max_x']):
                continue

            offset_c = c + np.array([[[0, roi_offset_y]]])
            cv2.drawContours(overlay, [offset_c], -1, (0, 255, 255), 1)

            M = cv2.moments(c)
            if M['m00'] == 0:
                continue

            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00']) + roi_offset_y
            valid.append({'cx': cx, 'cy': cy, 'rect': (x, y, w, h)})

        if len(valid) < tb['min_segments'] - 1:
            return False, None, (0.0, 0.0)

        ys = np.array([p['cy'] for p in valid])
        if ys.ptp() > tb['y_align']:
            return False, None, (0.0, 0.0)

        # Decide horizontal counting direction
        if self.crossroad_decision == 2:          # turn-left → count left→right
            valid.sort(key=lambda p:  p['cx'])     # ascending X
        else:                                      # straight (0) or turn-right (1)
            valid.sort(key=lambda p: -p['cx'])     # descending X (right→left)

        # Fit line ONCE
        pts = np.float32([(p['cx'], p['cy']) for p in valid])
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)

        res_px = self.safe_get('Residual px', 5)  
        keep = []
        for p in valid:
            # distance from point to line = |(vy)*(x - x0) - (vx)*(y - y0)|
            dist = abs(vy*(p['cx']-x0) - vx*(p['cy']-y0))
            if dist < res_px:
                keep.append(p)
        valid = keep
        if len(valid) < tb['min_segments'] - 1:
            return False, None, (0.0, 0.0)
        # refit after pruning
        pts = np.float32([(p['cx'], p['cy']) for p in valid])
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)

        angle = abs(np.degrees(np.arctan2(vy, vx)))
        if angle > tb['ang_thr_h']:
            return False, None, (0.0, 0.0)

        for p in valid:
            p['vx'] = vx[0]
            p['vy'] = vy[0]

        center_pt = self._center_between_third_and_fourth(valid)
        if center_pt:
            cv2.circle(overlay, center_pt, 4, (0, 255, 0), -1)

        pt1 = (int(x0 - vx * 100), int(y0 - vy * 100))
        pt2 = (int(x0 + vx * 100), int(y0 + vy * 100))
        cv2.line(overlay, pt1, pt2, (0, 255, 0), 1)

        return True, center_pt, (vx[0], vy[0])

    def is_dotted_column(self, binary_roi, overlay, roi_offset_x, tb):
        contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = []

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)

            if not (tb['area_min'] <= area <= tb['area_max']):
                continue

            aspect = w / max(h, 1)
            if not (tb['min_w'] <= w <= tb['max_w'] and
                    tb['min_h'] <= h <= tb['max_h'] and
                    tb['aspect_min_y'] <= aspect <= tb['aspect_max_y']):
                continue

            offset_c = c + np.array([[[roi_offset_x, 0]]])
            cv2.drawContours(overlay, [offset_c], -1, (255, 0, 255), 1)

            M = cv2.moments(c)
            if M['m00'] == 0:
                continue

            cx = int(M['m10'] / M['m00']) + roi_offset_x
            cy = int(M['m01'] / M['m00'])
            valid.append({'cx': cx, 'cy': cy, 'rect': (x, y, w, h)})

        # ---------- Minimum segment check ----------
        if len(valid) < tb['min_segments'] - 1:
            return False, None

        # ---------- X-alignment filtering ----------
        xs = np.array([p['cx'] for p in valid])
        mean_x = np.mean(xs)
        valid = [p for p in valid if abs(p['cx'] - mean_x) < tb['x_align']]

        if len(valid) < tb['min_segments'] - 1:
            return False, None

        # ---------- SORT bottom → top (crucial!) ----------
        valid.sort(key=lambda p: -p['cy'])      # largest Y (bottom) first

        # ---------- Fit line once ----------
        pts = np.float32([[p['cx'], p['cy']] for p in valid])
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        angle = abs(np.degrees(np.arctan2(vy, vx)))
        if abs(angle - 90) > tb['ang_thr_v']:
            return False, None

        for p in valid:
            p['vx'] = vx[0]
            p['vy'] = vy[0]

        # ---------- Pick centre between 3rd & 4th dash ----------
        center_pt = self._center_between_third_and_fourth(valid)
        if center_pt:
            cv2.circle(overlay, center_pt, 4, (0, 128, 255), -1)

        # ---------- Debug line ----------
        pt1 = (int(x0 - vx * 100), int(y0 - vy * 100))
        pt2 = (int(x0 + vx * 100), int(y0 + vy * 100))
        cv2.line(overlay, pt1, pt2, (0, 200, 255), 1)

        return True, center_pt


    def _center_between_third_and_fourth(self, sorted_pts):
        if len(sorted_pts) >= 4:
            p3 = np.array([sorted_pts[2]['cx'], sorted_pts[2]['cy']])
            p4 = np.array([sorted_pts[3]['cx'], sorted_pts[3]['cy']])
            return tuple(((p3 + p4) / 2).astype(int))

        elif len(sorted_pts) == 3:
            r = sorted_pts[2]['rect']
            vx = sorted_pts[2].get('vx', 0)
            vy = sorted_pts[2].get('vy', 1)  # fallback a vertical si no hay

            if abs(vx) > abs(vy):
                # Horizontal → borde derecho del rectángulo
                x = r[0] + r[2]
                y = r[1] + r[3] // 2
            else:
                # Vertical → borde superior del rectángulo
                x = r[0] + r[2] // 2
                y = r[1]
            return (x, y)

        elif len(sorted_pts) == 2:
            p1 = np.array([sorted_pts[0]['cx'], sorted_pts[0]['cy']])
            p2 = np.array([sorted_pts[1]['cx'], sorted_pts[1]['cy']])
            return tuple(((p1 + p2) / 2).astype(int))

        else:
            return None

    def image_callback(self, msg):
        if not self.active_crossroad_detection:
            return  # Do nothing if detection is disabled
        
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        warped = cv2.warpPerspective(frame, self.homography_matrix, self.warp_size) \
            if self.homography_matrix is not None else frame.copy()

        tb = self._snapshot_trackbar_state()
        binary = self.preprocess(warped, tb)
        h, w = binary.shape
        mid_x = w // 2

        # --- Line detection ---
        found_h, center_h, h_vec = self.is_dotted_row(binary, warped, roi_offset_y=0, tb=tb)
        found_left, center_left = self.is_dotted_column(binary[:, :mid_x], warped, roi_offset_x=0, tb=tb)
        found_right, center_right = self.is_dotted_column(binary[:, mid_x:], warped, roi_offset_x=mid_x, tb=tb)

        # ----- fabricate missing side centroid if needed -----
        Lx = Ly = Rx = Ry = 9999           # defaults
        if found_h:
            Hx, Hy = center_h
            # copy existing detections first
            if found_left:
                Lx, Ly = center_left
            if found_right:
                Rx, Ry = center_right

            # need LEFT ?
            if self.crossroad_decision == 2 and not found_left:
                Lx, Ly = self._estimate_side(center_h, h_vec, 'left')
                cv2.circle(warped, (Lx, Ly), 5, (255, 128, 0), -1)  # orange = estimated left
                found_left = True
            # need RIGHT ?
            if self.crossroad_decision == 1 and not found_right:
                Rx, Ry = self._estimate_side(center_h, h_vec, 'right')
                cv2.circle(warped, (Rx, Ry), 5, (255, 0, 128), -1)  # pink = estimated right
                found_right = True
        else:
            Hx = Hy = 9999    # keep interface identical

        # --- Detection flags ---
        flags = [int(found_h), int(found_left), int(found_right)]
        self.flags_pub.publish(Int32MultiArray(data=flags))
        self.centroids_pub.publish(Int32MultiArray(data=[Hx, Hy, Lx, Ly, Rx, Ry]))

        # --- Logging ---
        if found_h:
            self.get_logger().info("🚦 CROSSROAD DETECTED by horizontal dotted line!")
        if found_left:
            self.get_logger().info("🚦 CROSSROAD DETECTED by vertical LEFT dotted line!")
        if found_right:
            self.get_logger().info("🚦 CROSSROAD DETECTED by vertical RIGHT dotted line!")
        if found_h and (found_left or found_right):
            self.get_logger().info("🚦 FULL CROSSROAD (T or X) DETECTED!")

        # --- Overlay text ---
        messages = [
            "Horizontal dotted line detected ✅" if found_h else "Horizontal dotted line not found ❌",
            "Vertical LEFT dotted line detected ✅" if found_left else "Vertical LEFT dotted line not found ❌",
            "Vertical RIGHT dotted line detected ✅" if found_right else "Vertical RIGHT dotted line not found ❌",
        ]
        for i, text in enumerate(messages):
            cv2.putText(warped, text, (10, 20 + 20*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # --- Draw average center only if two lines are detected ---
        if found_h and (found_left or found_right):
            cx_list = [center_h[0]]
            cy_list = [center_h[1]]
            if found_left:
                cx_list.append(center_left[0])
                cy_list.append(center_left[1])
            if found_right:
                cx_list.append(center_right[0])
                cy_list.append(center_right[1])

            cross_center = (sum(cx_list) // len(cx_list), sum(cy_list) // len(cy_list))
            cv2.circle(warped, cross_center, 6, (0, 255, 255), -1)
            self.get_logger().info(f"🚦 Centro del cruce (aprox): {cross_center}")

        # --- Show image ---
        scale = 1.0 + max(1, self.safe_get('Scale', 2) | 1) / 10.0
        cv2.imshow("Crossroad Debug", cv2.resize(warped, None, fx=scale, fy=scale))
        cv2.waitKey(1)


    def decision_callback(self, msg: Int16):
        self.crossroad_decision = msg.data
        self.get_logger().info(f'📍 crossroad_decision set to: {self.crossroad_decision}')

    def _remap(self, pt):
        """Return centroid unchanged; only substitutes 9999 if not detected."""
        return (pt if pt is not None else (9999, 9999))
    
    # ---------- helper (add to the class) ----------
    def _estimate_side(self, center_h, vec, side):
        """
        centre_h : (u,v) of horizontal line centre in warp px
        vec      : (vx,vy) direction of horizontal line (not normalised)
        side     : 'left'  or 'right'
        Returns  : (est_u, est_v) in warp px
        """
        vx, vy = vec
        norm = math.hypot(vx, vy)
        if norm == 0:                       # degenerate
            return (9999, 9999)

        fwd   = np.array([vx, vy]) / norm          # direction ALONG the horizontal dash row
        left  = np.array([-vy, vx]) / norm         # 90° CCW

        # offsets in metres (positive forward, +left / -right)
        offset_m = 0.15 * fwd + 0.15 * ( left if side == 'left' else -left )

        # convert to pixels (warp frame: u = x-axis, v = y-axis downwards)
        du = offset_m[0] * self.warp_size[0] / self.x_meter_range
        dv = offset_m[1] * self.warp_size[1] / self.y_meter_range

        est_u = int(center_h[0] + du)
        est_v = int(center_h[1] + dv)

        # clip to image bounds
        est_u = max(0, min(self.warp_size[0]-1, est_u))
        est_v = max(0, min(self.warp_size[1]-1, est_v))
        return (est_u, est_v)



def main(args=None):
    rclpy.init(args=args)
    node = CrossroadDetector()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

#celso piña