#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from ament_index_python.packages import get_package_share_directory
import os

class LineFollowerCentroid(Node):
    def __init__(self):
        super().__init__('line_follower_centroid')
        self.bridge = CvBridge()

        self.publisher = self.create_publisher(Twist, '/line_cmd_vel', 10)
        self.subscription = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        
        self.get_logger().info('Line Follower Node Started')

        # Create control window and trackbars
        cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
        # cv2.createTrackbar('Threshold', 'Controls', 95, 255, lambda x: None)
        cv2.createTrackbar('Blur Kernel', 'Controls', 24, 31, lambda x: None)
        cv2.createTrackbar('Morph Kernel', 'Controls', 5, 51, lambda x: None)
        cv2.createTrackbar('Block Size', 'Controls', 76, 101, lambda x: None)
        cv2.createTrackbar('C (Bias)', 'Controls', 13, 50, lambda x: None)
        cv2.createTrackbar('Min Area', 'Controls', 500, 1000, lambda x: None)

        self.load_from_file = True  # Set to False to use manual selection
        
        pkg_path = get_package_share_directory('line_follow_msr')
        self.homography_matrix_path = os.path.join(pkg_path, 'data', 'homography3.npy')

        self.homography_matrix = None
        self.warp_size = (200, 200)  # output size of warped image
        self.selecting_points = False
        self.src_points = []

        # Controller gains
        self.Kp_x = 0.01
        self.xVel = 0.15
        self.Kp_ang = 0.022
        self.ang_e_thrsh = 2
        self.detected_mid_radius = 0
        self.mid_radius_thrshld = 90
        self.radius_kp = 0.002

        self.middle_line_buffer = []
        self.middle_line_buffer_size = 50
        self.last_valid_middle_x = None
        self.x_diff_threshold = 20  # pixels
        self.drift_frame_count = 0
        self.max_drift_frames = 10

        self.in_fallback_mode      = False      # are we currently following a side line?
        self.fallback_target       = None       # "LEFT" or "RIGHT"
        self.fallback_ang_scale    = 2.0        # >1.0 gives a ‘snap’ turn
        self.mid_reacq_tol         = 15         # px; how centred a line must be to exit fallback


        if self.load_from_file:
            try:
                self.homography_matrix = np.load(self.homography_matrix_path)
                self.selecting_points = False
                print("✅ Homography matrix loaded from file:")
                print(self.homography_matrix)
            except FileNotFoundError:
                print("❌ Homography file not found. Falling back to manual point selection.")
                self.selecting_points = True


        self.color_flag_multiplier = 1.0
        self.create_subscription(Float32, '/fsm_action', self.fsm_action_callback, 10)

    def fsm_action_callback(self, msg: Float32):
        self.color_flag_multiplier = msg.data
        self.get_logger().info(f"Updated color_flag_multiplier to: {self.color_flag_multiplier}")

    def preprocess_region(self, region, blur_k, block_size, c_bias, morph_k):
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        
        # Apply CLAHE for contrast enhancement
        v_equalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(v)

        # Gaussian blur (kernel must be odd and >= 1)
        blur_k = max(1, blur_k | 1)  # force odd
        blurred = cv2.GaussianBlur(v_equalized, (blur_k, blur_k), 0)

        # Adaptive threshold
        block_size = max(3, block_size | 1)  # must be odd and >= 3
        binary = cv2.adaptiveThreshold( blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c_bias )

        # Morphological filtering
        morph_k = max(1, morph_k | 1)  # force odd
        kernel = np.ones((morph_k, morph_k), np.uint8)
        binary = cv2.erode(binary, kernel, iterations=1)
        binary = cv2.dilate(binary, kernel, iterations=2)

        return binary


    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        if self.homography_matrix is not None:
            warped = cv2.warpPerspective(frame, self.homography_matrix, self.warp_size)
            warped = cv2.flip(warped, 1)
        else:
            warped = frame

        overlay = warped.copy()
        h, w, _ = warped.shape

        third = w // 3
        twelve_div = w // 12
        roi_middle_start = 2 * twelve_div
        roi_middle_end = 10 * twelve_div
        roi_left = warped[:, 0:5 * twelve_div]
        roi_middle = warped[:, roi_middle_start:roi_middle_end]
        roi_right = warped[:, 7 * twelve_div:]

        blur_k = max(1, cv2.getTrackbarPos('Blur Kernel', 'Controls') | 1)
        morph_k = max(1, cv2.getTrackbarPos('Morph Kernel', 'Controls') | 1)
        block_size = max(3, cv2.getTrackbarPos('Block Size', 'Controls') | 1)
        c_bias = cv2.getTrackbarPos('C (Bias)', 'Controls')
        min_area = cv2.getTrackbarPos('Min Area', 'Controls')

        binary_left = self.preprocess_region(roi_left, blur_k, block_size, c_bias, morph_k)
        binary_middle = self.preprocess_region(roi_middle, blur_k, block_size, c_bias, morph_k)
        binary_right = self.preprocess_region(roi_right, blur_k, block_size, c_bias, morph_k)

        line_l = self.detect_line_in_roi(binary_left, 0, overlay, (255, 0, 0), "L", min_area)
        line_m = self.detect_line_in_roi(binary_middle, roi_middle_start, overlay, (255, 255, 0), "M", min_area)
        line_r = self.detect_line_in_roi(binary_right, 7 * twelve_div, overlay, (255, 0, 0), "R", min_area)

        center_x = w // 2
        h_text = overlay.shape[0] - 25
        cv2.putText(overlay, f"x: L={line_l['x_global'] if line_l else '-'} M={line_m['x_global'] if line_m else '-'} R={line_r['x_global'] if line_r else '-'}", (10, h_text), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(overlay, f"ang: L={line_l['angle']:.1f}" if line_l else "-" + f" M={line_m['angle']:.1f}" if line_m else "-" + f" R={line_r['angle']:.1f}" if line_r else "-", (10, h_text + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.line(overlay, (center_x, 0), (center_x, h), (0, 0, 255), 2)

        if self.selecting_points:
            cv2.imshow("ROI", warped)
        cv2.imshow("Overlay", cv2.resize(overlay, None, fx=3.0, fy=3.0))
        cv2.imshow("Left | Middle | Right", stack_with_dividers([binary_left, binary_middle, binary_right]))

        # --- Fallback active check ---
        if self.in_fallback_mode:
            target_line = line_l if self.fallback_target == "LEFT" else line_r
            if target_line:
                twist = Twist()
                x_err = target_line["x_global"] - center_x
                ang_err = 90.0 - target_line["angle"] if target_line["angle"] > 0 else -90.0 - target_line["angle"]
                if abs(ang_err) < self.ang_e_thrsh:
                    ang_err = 0.0
                twist.linear.x = self.xVel
                twist.angular.z = (self.Kp_ang * ang_err - self.Kp_x * x_err) * self.fallback_ang_scale
                self.publisher.publish(twist)
                self.get_logger().info(f"[↩ Fallback-{self.fallback_target}] x_err={x_err:.1f}, ang_err={ang_err:.1f}")

                if roi_middle_start <= target_line["x_global"] <= roi_middle_end:
                    self.get_logger().info("[✅ EXIT Fallback] Side line reached middle ROI.")
                    self.in_fallback_mode = False
                    self.fallback_target = None
                    self.middle_line_buffer.clear()
                    self.drift_frame_count = 0
            else:
                self.get_logger().info("[↩ Fallback] Target side line not visible.")
            cv2.waitKey(1)
            return

        # --- Middle line logic ---
        if line_m:
            x_now = line_m["x_global"]
            self.middle_line_buffer.append(x_now)
            if len(self.middle_line_buffer) > self.middle_line_buffer_size:
                self.middle_line_buffer.pop(0)

            avg_x = np.mean(self.middle_line_buffer[-3:]) if len(self.middle_line_buffer) >= 3 else x_now
            x_diff = abs(x_now - avg_x)
            self.get_logger().info(f"[🟢 LINE M DETECTED] x={x_now:.1f}, avg3={avg_x:.1f}, x_diff={x_diff:.1f}")

            if x_diff < self.x_diff_threshold:
                self.last_valid_middle_x = x_now
                self.drift_frame_count = 0
                x_error = x_now - center_x
                angle_error = 90.0 - line_m["angle"] if line_m["angle"] > 0 else -90.0 - line_m["angle"]
                if abs(angle_error) < self.ang_e_thrsh:
                    angle_error = 0.0
                twist = Twist()
                twist.linear.x = self.xVel
                twist.angular.z = self.Kp_ang * angle_error - self.Kp_x * x_error

                # --- curvature bonus if the line is “bendy” ---
                radius_term = 0.0
                if self.detected_mid_radius < self.mid_radius_thrshld:
                    radius_term = self.radius_kp * self.detected_mid_radius
                    twist.angular.z = twist.angular.z - radius_term
                    self.get_logger().info(f"[🧭 FOLLOWING MIDDLE] x_err={x_error:.1f}, ang_err={angle_error:.1f}, "f"R={self.detected_mid_radius:.1f}, radius_term={radius_term:.3f}")
                
                self.publisher.publish(twist)
                self.get_logger().info(f"[🧭 FOLLOWING MIDDLE] x_err={x_error:.1f}, ang_err={angle_error:.1f}")
            else:
                self.drift_frame_count += 1
                self.get_logger().warn(f"[⚠️ MIDDLE DRIFT] x_diff={x_diff:.1f} frame_count={self.drift_frame_count}")
                if self.drift_frame_count >= self.max_drift_frames:
                    self.get_logger().warn("[🚨 SWITCHING TO FALLBACK]")
                    candidates = []
                    if line_l:
                        candidates.append(("LEFT", line_l, abs(line_l["x_global"] - self.last_valid_middle_x)))
                    if line_r:
                        candidates.append(("RIGHT", line_r, abs(line_r["x_global"] - self.last_valid_middle_x)))
                    if candidates:
                        self.fallback_target, _, _ = min(candidates, key=lambda x: x[2])
                        self.in_fallback_mode = True
                        return
                else:
                    line_m["x_global"] = self.last_valid_middle_x

        else:
            twist = Twist()
            twist.linear.x = -0.1 * self.color_flag_multiplier
            twist.angular.z = 0.0
            self.publisher.publish(twist)
            self.get_logger().warn("[🚫 NO LINES] Emergency reverse.")
        cv2.waitKey(1)


    def detect_line_in_roi(self, binary_roi, roi_x_offset, overlay=None, color=(255, 255, 0), label="", min_area=150):

        contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        if not contours:
            return None  # No line

        largest = max(contours, key=cv2.contourArea)
        vx, vy, x0, y0 = cv2.fitLine(largest, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        y_bottom = binary_roi.shape[0] - 1
        if abs(vy) < 1e-6:  # avoid division by zero
            return None  # or handle it as a vertical line
        t = (y_bottom - y0) / vy
        x_bottom = int(x0 + vx * t)
        global_x = roi_x_offset + x_bottom

        if overlay is not None:
            # Draw contour
            # Offset contour points horizontally by ROI origin
            offset_contour = [np.array([[pt[0][0] + roi_x_offset, pt[0][1]]]) for pt in largest]
            offset_contour = np.array(offset_contour, dtype=np.int32)
            cv2.drawContours(overlay, [offset_contour], -1, (0, 255, 0), 2)  # 🟩 Green


            # --- Estimate curvature using circle fitting ---
            if label == "M":  # Only draw for middle
                try:
                    (x_c, y_c), self.detected_mid_radius = cv2.minEnclosingCircle(largest)
                    center = (int(x_c) + roi_x_offset, int(y_c))
                    self.detected_mid_radius = float(self.detected_mid_radius)

                    # Draw the circle
                    cv2.circle(overlay, center, int(self.detected_mid_radius), (0, 255, 255), 1)  # yellow

                    # Draw curvature annotation
                    cv2.putText(overlay, f"{label} R: {self.detected_mid_radius:.1f}", (roi_x_offset + 5, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                except cv2.error as e:
                    print(f"⚠️ Circle fitting failed: {e}")
                    self.detected_mid_radius = float('inf')  # Treat as straight line

            # Draw line
            pt1 = (roi_x_offset + int(x0 - vx * 50), int(y0 - vy * 50))
            pt2 = (roi_x_offset + int(x0 + vx * 50), int(y0 + vy * 50))
            cv2.line(overlay, pt1, pt2, color, 2)
            # Annotate
            # angle_deg = np.degrees(np.arctan2(vy, vx))
            # cv2.putText(overlay, f"{label} x: {global_x}", (roi_x_offset + 5, 20),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            # cv2.putText(overlay, f"{label} ang: {angle_deg:.1f}", (roi_x_offset + 5, 35),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return {
            "x_global": global_x,
            "angle": np.degrees(np.arctan2(vy, vx)),
            "length": cv2.arcLength(largest, closed=False),
            "contour": largest
        }


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

def select_points(event, x, y, flags, param):
    node = param
    if event == cv2.EVENT_LBUTTONDOWN and node.selecting_points:
        node.src_points.append([x, y])
        print(f"Selected point: ({x},{y})")
        if len(node.src_points) == 4:
            node.src_points = np.array(node.src_points, dtype=np.float32)
            node.dst_points = np.array([
                [0, 0],
                [node.warp_size[0]-1, 0],
                [node.warp_size[0]-1, node.warp_size[1]-1],
                [0, node.warp_size[1]-1]
            ], dtype=np.float32)
            node.homography_matrix = cv2.getPerspectiveTransform(node.src_points, node.dst_points)
            node.selecting_points = False
            print("Homography matrix H:")
            print(node.homography_matrix)

            # Save to file
            np.save(node.homography_matrix_path, node.homography_matrix)
            print(f"✅ Homography saved to file: {node.homography_matrix_path}")
            print("Homography computed. Now warping every frame.")

def stack_with_dividers(imgs, divider_thickness=3, divider_color=255):
    """
    Stack grayscale images horizontally with vertical dividers.
    imgs: list of images (all same height and 1-channel)
    """
    h = imgs[0].shape[0]
    div = np.full((h, divider_thickness), divider_color, dtype=np.uint8)  # white line

    result = imgs[0]
    for img in imgs[1:]:
        result = np.hstack((result, div, img))
    return result



if __name__ == '__main__':
    main()