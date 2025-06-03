#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Int16
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import math
from .pid_controller import PIDController

class LineFollowerCentroid(Node):
    def __init__(self):
        super().__init__('line_follower')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.activate_sub = self.create_subscription(Int16, '/line_follow_enable', self.activate_line_follow_callback, 10)
        self.publisher = self.create_publisher(Twist, 'line_cmd_vel', 10)
        self.bridge = CvBridge()

        self.get_logger().info('Robust Line Follower Node Started')

        # Control parameters
        self.color_flag_multiplier = 1.0
        self.active_line = 0
        self.windows_open = False

        # PID Controller
        self.max_angular_vel = 0.5
        self.max_linear_vel = 0.2
        self.align_threshold = 0.3
        self.angular_pid = PIDController(
            kp=0.8,
            ki=0.0,
            kd=0.15,
            setpoint=0.0,
            output_limits=(-self.max_angular_vel, self.max_angular_vel)
        )

        # Fallback control
        self.last_angular = 0.0
        self.last_linear = 0.05

        # Bias correction for curve transitions
        self.last_angle = 0.0
        self.transition_frames = 0
        self.straight_counter = 0

        self.create_subscription(Float32, '/fsm_action', self.fsm_action_callback, 10)

        cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
        cv2.createTrackbar('Threshold', 'Controls', 95, 255, lambda x: None)
        cv2.createTrackbar('Blur Kernel', 'Controls', 11, 31, lambda x: None)
        cv2.createTrackbar('Morph Kernel', 'Controls', 2, 31, lambda x: None)
        cv2.createTrackbar('Block Size', 'Controls', 39, 51, lambda x: None)
        cv2.createTrackbar('C (Bias)', 'Controls', 12, 20, lambda x: None)


    def activate_line_follow_callback(self, msg):
        self.active_line = msg.data
        self.get_logger().info(f"line follow node state: {self.active_line}")


    def adaptive_threshold(self, frame):
        """ Apply adaptive thresholding for better line detection """
        # gray = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2GRAY)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_k = max(1, cv2.getTrackbarPos('Blur Kernel', 'Controls') | 1)
        gray = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
        block_size = max(3, cv2.getTrackbarPos('Block Size', 'Controls') | 1)
        c_bias = cv2.getTrackbarPos('C (Bias)', 'Controls')
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c_bias)
        return mask

    def get_line_mask(self, frame):
        """ Create mask focusing on bottom image portion """
        mask = self.adaptive_threshold(frame)
        # Focus on bottom 40% of image
        mask[:int(frame.shape[0] * 0.6), :] = 0
        # Morphological operations
        morph_k = max(1, cv2.getTrackbarPos('Morph Kernel', 'Controls') | 1)
        mask = cv2.erode(mask, np.ones((morph_k, morph_k), np.uint8), iterations=3)
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=5)
        return mask
    
    def get_contour_line_info(self, contour):
        """ Extract information from contour """
        vx, vy, cx, cy = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        projections = [((pt[0][0] - cx) * vx + (pt[0][1] - cy) * vy) for pt in contour]
        min_proj = min(projections)
        max_proj = max(projections)
        pt1 = (int(cx + vx * min_proj), int(cy + vy * min_proj))
        pt2 = (int(cx + vx * max_proj), int(cy + vy * max_proj))
        angle = math.degrees(math.atan2(vy, vx)) - 90 * np.sign(math.degrees(math.atan2(vy, vx)))
        length = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
        return pt1, pt2, angle, cx, cy, length

    def get_line_candidates(self, frame):
        """Find potential line contours"""
        mask = self.get_line_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > 800]
        lines = [self.get_contour_line_info(c) for c in contours]
        return [(c, l) for c, l in zip(contours, lines) if l[5] > 50]  # Length > 50

    def get_best_line(self, frame, drawing_frame=None):
        """Select the best line candidate"""
        frame_height, frame_width = frame.shape[:2]

        def line_key(line_data):
            _, _, angle, cx, _, _ = line_data[1]
            max_angle = 80
            angle = max(min(angle, max_angle), -max_angle)
            ref_x = (frame_width / 2) + (angle / max_angle) * (frame_width / 2)
            return abs(cx - ref_x)

        lines = self.get_line_candidates(frame)
        if lines:
            lines = sorted(lines, key=line_key)
            best = lines[0]
            if drawing_frame is not None:
                cv2.drawContours(drawing_frame, [best[0]], -1, (0, 255, 0), 2)
                for c, _ in lines[1:]:
                    cv2.drawContours(drawing_frame, [c], -1, (0, 0, 255), 2)
            return best
        return None

    def calculate_control_commands(self, frame, drawing_frame=None):
        """Calculate throttle and steering commands"""
        line = self.get_best_line(frame, drawing_frame)
        linear_vel, angular_vel = 0.0, 0.0
        frame_height, frame_width = frame.shape[:2]

        if line:
            contour, (pt1, pt2, angle, cx, cy, length) = line
            x, y, w, h = cv2.boundingRect(contour)
            
            # Detect straight sections and curve transitions
            is_straight = abs(angle) < 10
            was_curve = abs(self.last_angle) >= 15
            is_transition = is_straight and was_curve
            
            if is_straight:
                self.straight_counter += 1
            else:
                self.straight_counter = 0
            
            # Handle transition from curve to straight
            if is_transition:
                self.transition_frames = 12
                self.angular_pid.reset()
            
            # Calculate center position
            if is_straight:
                # For straight sections, use ROI-based centroid calculation
                roi_height = int(frame_height * 0.25)
                roi_y_start = frame_height - roi_height
                
                contour_roi = []
                for point in contour:
                    if point[0][1] >= roi_y_start:
                        contour_roi.append(point)
                
                center_x_roi = x + w // 2  # Backup
                if len(contour_roi) > 10:
                    contour_roi = np.array(contour_roi)
                    M_roi = cv2.moments(contour_roi)
                    if M_roi["m00"] != 0:
                        center_x_roi = int(M_roi["m10"] / M_roi["m00"])
                
                # Full contour centroid
                M_full = cv2.moments(contour)
                center_x_full = x + w // 2  # Backup
                if M_full["m00"] != 0:
                    center_x_full = int(M_full["m10"] / M_full["m00"])
                
                # Weighted average: more weight to ROI
                center_x = int(0.7 * center_x_roi + 0.3 * center_x_full)
            else:
                # For curves, use bounding box center
                center_x = x + w // 2
            
            # Normalize position (-1 to 1)
            normalized_x = (center_x - (frame_width / 2)) / (frame_width / 2)
            
            # Apply bias correction during transitions
            if self.transition_frames > 0:
                if self.transition_frames > 8:
                    strength = 0.25  # Very aggressive initially
                elif self.transition_frames > 4:
                    strength = 0.15  # Aggressive
                else:
                    strength = 0.08  # Gentle
                    
                bias_correction = -strength  # Bias towards left
                normalized_x += bias_correction
                self.transition_frames -= 1
                
                if drawing_frame is not None:
                    cv2.putText(drawing_frame, f"CORRECTION: {strength:.3f}", (10, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            # Continuous correction for straight sections
            elif is_straight and self.straight_counter > 2:
                continuous_correction = -0.04
                normalized_x += continuous_correction
                
                if drawing_frame is not None:
                    cv2.putText(drawing_frame, "CONTINUOUS CORRECTION", (10, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Small deadband for straight sections
            if is_straight and abs(normalized_x) < 0.015:
                normalized_x = 0.0
            
            # PID control for angular velocity
            angular_vel = self.angular_pid.compute(normalized_x)
            
            # Update angle history
            self.last_angle = angle

            # Calculate linear velocity based on alignment
            alignment = 1 - abs(normalized_x)
            if alignment > self.align_threshold:
                velocity_factor = (alignment - self.align_threshold) / (1 - self.align_threshold)
                linear_vel = self.max_linear_vel * velocity_factor
            else:
                linear_vel = 0.05  # Minimum speed

            # Save for fallback
            self.last_angular = angular_vel
            self.last_linear = linear_vel

            # Visualization
            if drawing_frame is not None:
                cv2.line(drawing_frame, (center_x, 0), (center_x, frame_height), (255, 0, 0), 2)
                cv2.line(drawing_frame, (frame_width//2, 0), (frame_width//2, frame_height), (0, 255, 255), 1)
                cv2.putText(drawing_frame, f"Linear: {linear_vel:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(drawing_frame, f"Angular: {math.degrees(angular_vel):.2f} deg", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(drawing_frame, f"Straight cnt: {self.straight_counter}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            # Fallback mode - no line detected
            self.get_logger().warn('No line detected! Using fallback control...')
            angular_vel = self.last_angular
            linear_vel = 0.05  # Slow forward
            
            if drawing_frame is not None:
                cv2.putText(drawing_frame, "FALLBACK: No line", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return linear_vel, angular_vel

    def image_callback(self, msg):
        """Enhanced image callback with advanced processing"""
        if self.active_line != 1:
            # If not active, send stop command
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.publisher.publish(twist)
            
            # Close windows if they were open
            if self.windows_open:
                cv2.destroyWindow('ROI Enhanced')
                cv2.destroyWindow('Line Mask')
                cv2.destroyWindow('Control Overlay')
                self.windows_open = False
            return

        # Convert ROS image to OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        height, width, _ = frame.shape

        # Define ROI (bottom portion of image)
        y_start = int(height * 0.6)  # Bottom 40%
        y_end = height
        x_start = int(width * 0.1)   # Central 80%
        x_end = int(width * 0.9)

        roi = frame[y_start:y_end, x_start:x_end]
        
        # Create visualization frames
        roi_display = roi.copy()
        mask_display = self.get_line_mask(roi)
        
        # Calculate control commands
        linear_vel, angular_vel = self.calculate_control_commands(roi, roi_display)
        
        # Apply color multiplier from FSM
        linear_vel *= self.color_flag_multiplier
        angular_vel *= self.color_flag_multiplier

        # Create and publish twist message
        twist = Twist()
        twist.linear.x = linear_vel
        twist.angular.z = angular_vel

        self.get_logger().info(f'Publishing: linear_x={linear_vel:.3f}, angular_z={angular_vel:.3f}')
        self.publisher.publish(twist)

        # Display debug windows
        cv2.imshow('ROI Enhanced', roi_display)
        cv2.imshow('Line Mask', mask_display)
        
        # Create control overlay
        overlay = np.zeros((200, 400, 3), dtype=np.uint8)
        cv2.putText(overlay, f"Linear Vel: {linear_vel:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(overlay, f"Angular Vel: {angular_vel:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(overlay, f"Multiplier: {self.color_flag_multiplier:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(overlay, f"Transition: {self.transition_frames}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(overlay, f"Straight: {self.straight_counter}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow('Control Overlay', overlay)
        
        cv2.waitKey(1)
        self.windows_open = True

    def fsm_action_callback(self, msg: Float32):
        self.color_flag_multiplier = msg.data
        self.get_logger().info(f"Updated color_flag_multiplier to: {self.color_flag_multiplier}")

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