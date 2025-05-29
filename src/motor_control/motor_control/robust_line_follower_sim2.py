#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

class LineFollower(Node):
    def __init__(self):
        super().__init__('line_follower')
        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image, '/Image', self.image_callback, 10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info('Line Follower Node Started')

        # tuning parameters
        self.Kp_ang = 0.04               # angular proportional gain
        self.max_linear = 0.1            # max forward speed (m/s)
        self.min_linear = 0.01           # min forward speed so it never stalls
        self.color_flag_multiplier = 1.0  # speed adjustment factor

        # Create control window and trackbars
        cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
        cv2.createTrackbar('Threshold', 'Controls', 146, 255, lambda x: None)
        cv2.createTrackbar('Blur Kernel', 'Controls', 3, 31, lambda x: None)
        cv2.createTrackbar('Morph Kernel', 'Controls', 0, 31, lambda x: None)

        # For color flag adjustment
        self.create_subscription(Float32, '/fsm_action', self.color_flag_callback, 10)

    def image_callback(self, msg):
        try:
            # Convert ROS image to OpenCV BGR
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            height, width, _ = frame.shape

            # Define ROI (bottom sixth vertically, center half horizontally)
            y_start = int(height * 5/6)
            y_end = height
            x_start = int(width * 0.25)
            x_end = int(width * 0.75)
            roi = frame[y_start:y_end, x_start:x_end]

            # Read and sanitize trackbars
            thresh_val = cv2.getTrackbarPos('Threshold', 'Controls')
            blur_k = cv2.getTrackbarPos('Blur Kernel', 'Controls')
            morph_k = cv2.getTrackbarPos('Morph Kernel', 'Controls')
            blur_k = max(1, blur_k | 1)       # ensure odd ≥1
            morph_k = max(1, morph_k | 1)     # ensure odd ≥1

            # Enhanced preprocessing pipeline
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2]
            
            # CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            v_equalized = clahe.apply(v_channel)
            
            # Dynamic gamma correction
            mean_val = np.mean(v_equalized)
            gamma = 1.5 if mean_val < 90 else (0.8 if mean_val > 160 else 1.0)
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
            v_corrected = cv2.LUT(v_equalized, table)
            
            # Blur and threshold
            blurred = cv2.GaussianBlur(v_corrected, (blur_k, blur_k), 0)
            _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
            
            # Morphological operations
            kernel = np.ones((morph_k, morph_k), np.uint8)
            binary = cv2.dilate(binary, kernel, iterations=1)
            binary = cv2.erode(binary, kernel, iterations=1)

            # Histogram analysis
            histogram = np.sum(binary, axis=0)
            if np.max(histogram) == 0:
                self.get_logger().warn('No line detected! Reversing...')
                twist = Twist()
                twist.linear.x = -0.05 * self.color_flag_multiplier
                twist.angular.z = 0.0
                self.publisher.publish(twist)
                time.sleep(0.5)
                return

            # Find line and calculate error
            line_x = int(np.argmax(histogram))
            center_x = (x_end - x_start) // 2
            offset = line_x - center_x
            self.get_logger().info(f'Offset: {offset}')

            # Control calculations
            ang_z = -self.Kp_ang * float(offset)
            
            # Linear velocity scales down as error grows
            error_norm = min(abs(offset) / float(center_x), 1.0)
            linear_x = self.max_linear * (1.0 - error_norm)
            linear_x = max(self.min_linear, linear_x)
            
            # Apply color flag multiplier
            linear_x *= self.color_flag_multiplier
            ang_z *= self.color_flag_multiplier

            # Publish command
            twist = Twist()
            twist.linear.x = linear_x
            twist.angular.z = ang_z
            self.get_logger().warning(f'Publishing: linear_x={linear_x:.3f}, angular_z={ang_z:.3f}')
            self.publisher.publish(twist)

            # Visualization
            overlay = roi.copy()
            cv2.line(overlay, (line_x, 0), (line_x, overlay.shape[0]), (0,255,0), 2)
            cv2.line(overlay, (center_x, 0), (center_x, overlay.shape[0]), (0,0,255), 2)
            
            debug_img = np.zeros((roi.shape[0]*2, roi.shape[1], 3), dtype=np.uint8)
            debug_img[0:roi.shape[0], :] = roi
            debug_img[roi.shape[0]:, :] = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            cv2.putText(debug_img, f"Offset: {offset}", (10,30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            cv2.imshow('Debug', debug_img)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error in image processing: {str(e)}')

    def color_flag_callback(self, msg):
        self.color_flag_multiplier = msg.data
        self.get_logger().info(f"Updated color_flag_multiplier to: {self.color_flag_multiplier}")

def main(args=None):
    rclpy.init(args=args)
    node = LineFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()