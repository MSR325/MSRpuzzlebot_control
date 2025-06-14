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

class LineFollowerCentroid(Node):
    def __init__(self):
        super().__init__('line_follower_centroid')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/image_raw', self.image_callback_crips, 10)
        self.activate_sub = self.create_subscription(Int16, '/line_follow_enable', self.activate_line_follow_callback, 10)
        self.publisher = self.create_publisher(Twist, 'line_cmd_vel', 10)
        self.get_logger().info('Line Follower Node Started')

        # Create control window and trackbars
        # cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
        # cv2.createTrackbar('Threshold', 'Controls', 95, 255, lambda x: None)
        # cv2.createTrackbar('Blur Kernel', 'Controls', 0, 31, lambda x: None)
        # cv2.createTrackbar('Morph Kernel', 'Controls', 0, 31, lambda x: None)
        self.color_flag_multiplier = 1.0
        self.active_line = 0
        self.windows_open = False
        self.create_subscription(Float32, '/fsm_action', self.fsm_action_callback, 10)

    def activate_line_follow_callback(self, msg):
        self.active_line = msg.data
        self.get_logger().info(f"line follow node state: {self.active_line}")

    def image_callback(self, msg):
        # Convert ROS image to OpenCV BGR
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        height, width, _ = frame.shape

        # vertical bounds (bottom third)
        y_start = int(height * 5/6)
        # y_start = int(height * 0.0)
        y_end   = height
        # horizontal bounds (central 50% of width)
        x_start = int(width * 0.25)
        # x_start = int(width * 0.0)
        x_end   = int(width * 0.75)
        # x_end   = int(width)

        # extract centered ROI
        roi = frame[y_start:y_end, x_start:x_end]


        # Read trackbar positions
        thresh_val = 95
        blur_k = 0
        morph_k = 0
        # thresh_val = cv2.getTrackbarPos('Threshold', 'Controls')
        # blur_k = cv2.getTrackbarPos('Blur Kernel', 'Controls')
        # morph_k = cv2.getTrackbarPos('Morph Kernel', 'Controls')

        # Ensure odd values for kernels
        blur_k = blur_k if blur_k % 2 == 1 else blur_k + 1
        blur_k = max(1, blur_k)
        morph_k = morph_k if morph_k % 2 == 1 else morph_k + 1
        morph_k = max(1, morph_k)

        # Preprocessing: grayscale + blur
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

        # Binary inverse threshold for black line detection
        _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)

        # Morphological operations to remove noise
        kernel = np.ones((morph_k, morph_k), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)
        binary = cv2.erode(binary, kernel, iterations=1)

        # Histogram across columns
        histogram = np.sum(binary, axis=0)

        # Check if no line is detected
        if np.max(histogram) == 0:
            self.get_logger().warn('No line detected! Reversing...')
            twist = Twist()
            twist.linear.x = -0.05  # Reverse speed
            twist.angular.z = 0.0
            if (self.active_line == 1):
                self.publisher.publish(twist)
            else:
                twist.linear.x = 0.0  
                twist.angular.z = 0.0
                self.publisher.publish(twist)

            time.sleep(0.5)  # Wait for 0.5 seconds
            # rclpy.spin_once(self, timeout_sec=1.0)  # Wait for 1 second
            return

        # Detect the peak white line
        line_x = int(np.argmax(histogram))  # Find the column with the highest intensity

        # # Detect top-3 peaks for parallel lines
        # if len(histogram) >= 3:
        #     # indices of three largest values
        #     peak_idxs = np.argpartition(histogram, -3)[-3:]
        #     peaks = np.sort(peak_idxs)
        #     # Use middle peak as reference
        #     line_x = int(peaks[1])
        # else:
        #     # fallback to single line
        #     line_x = int(np.argmax(histogram))

        # Compute offset from center
        center_x = (x_end - x_start) // 2
        offset = line_x - center_x
        # self.get_logger().debug(f'Peaks: {peaks if len(histogram)>=3 else [line_x]}')
        self.get_logger().info(f'Offset: {offset}')

        # Proportional controller for angular velocity
        Kp = 0.005
        ang_z = -Kp * float(offset)
        # Deadband to avoid jitter
        if abs(offset) < 15:
            ang_z = 0.0
            linear_x = 0.15
        else:
            linear_x = 0.08

        # Apply color multiplier from /fsm_action
        linear_x *= self.color_flag_multiplier
        ang_z *= self.color_flag_multiplier

        # Add limits to linear and angular velocities
        # MAX_LINEAR_VELOCITY = 0.2  # Maximum linear velocity
        # MIN_LINEAR_VELOCITY = -0.2  # Minimum linear velocity
        # MAX_ANGULAR_VELOCITY = 0.5  # Maximum angular velocity
        # MIN_ANGULAR_VELOCITY = -0.5  # Minimum angular velocity

        # Clamp velocities
        # linear_x = max(MIN_LINEAR_VELOCITY, min(MAX_LINEAR_VELOCITY, linear_x))
        # ang_z = max(MIN_ANGULAR_VELOCITY, min(MAX_ANGULAR_VELOCITY, ang_z))

        # Publish command
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = ang_z

        self.get_logger().warning(f'Publishing: linear_x={linear_x}, angular_z={ang_z}')
        if (self.active_line == 1):
            self.publisher.publish(twist)
        else:
            twist.linear.x = 0.0  
            twist.angular.z = 0.0
            self.publisher.publish(twist)

        # Overlay: draw detected and center lines
        overlay = roi.copy()
        # detected line in green
        cv2.line(overlay,    (line_x, 0), (line_x, overlay.shape[0]), (0, 255, 0), 2)
        # center line in red
        cv2.line(overlay, (center_x, 0), (center_x, overlay.shape[0]), (0, 0, 255), 2)

        # Display windows
        if (self.active_line == 1):
            cv2.imshow('ROI', roi)
            cv2.imshow('Binary Mask', binary)
            cv2.imshow('Overlay', overlay)
            cv2.waitKey(1)
            self.windows_open = True
        elif self.windows_open:
            cv2.destroyWindow('ROI')
            cv2.destroyWindow('Binary Mask')
            cv2.destroyWindow('Overlay')
            self.windows_open = False

    def image_callback_crips(self, msg):
        # Convert ROS image to OpenCV BGR
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        height, width, _ = frame.shape

        y_start = int(height * 5/6)
        y_end   = height
        x_start = int(width * 0.25)
        x_end   = int(width * 0.75)

        roi = frame[y_start:y_end, x_start:x_end]

        # Robust preprocessing: HSV → V-channel
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]

        # CLAHE (adaptive contrast enhancement)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        v_equalized = clahe.apply(v_channel)

        # Gamma correction based on brightness
        mean_val = np.mean(v_equalized)
        gamma = 1.5 if mean_val < 90 else (0.8 if mean_val > 160 else 1.0)
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
        v_corrected = cv2.LUT(v_equalized, table)

        # Gaussian blur and threshold
        blur_k = 5
        thresh_val = 146
        morph_k = 5

        blurred = cv2.GaussianBlur(v_corrected, (blur_k, blur_k), 0)
        _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)

        # Morphological ops
        kernel = np.ones((morph_k, morph_k), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)
        binary = cv2.erode(binary, kernel, iterations=1)

        histogram = np.sum(binary, axis=0)

        if np.max(histogram) == 0:
            self.get_logger().warn('No line detected! Reversing...')
            twist = Twist()
            twist.linear.x = -0.05
            twist.angular.z = 0.0
            twist.linear.x *= self.color_flag_multiplier
            twist.angular.z *= self.color_flag_multiplier
            if self.active_line == 1:
                self.publisher.publish(twist)
            else:
                self.publisher.publish(Twist())  # stop
            time.sleep(0.5)
            return

        line_x = int(np.argmax(histogram))
        center_x = (x_end - x_start) // 2
        offset = line_x - center_x
        self.get_logger().info(f'Offset: {offset}')

        # Control
        Kp = 0.005
        ang_z = -Kp * float(offset)
        if abs(offset) < 15:
            ang_z = 0.0
            linear_x = 0.15
        else:
            linear_x = 0.08

        linear_x *= self.color_flag_multiplier
        ang_z *= self.color_flag_multiplier

        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = ang_z

        self.get_logger().warning(f'Publishing: linear_x={linear_x}, angular_z={ang_z}')
        if self.active_line == 1:
            self.publisher.publish(twist)
        else:
            self.publisher.publish(Twist())

        # Visualization
        overlay = roi.copy()
        cv2.line(overlay, (line_x, 0), (line_x, overlay.shape[0]), (0, 255, 0), 2)
        cv2.line(overlay, (center_x, 0), (center_x, overlay.shape[0]), (0, 0, 255), 2)

        if self.active_line == 1:
            cv2.imshow('ROI', roi)
            cv2.imshow('Binary Mask', binary)
            cv2.imshow('Overlay', overlay)
            cv2.waitKey(1)
            self.windows_open = True
        elif self.windows_open:
            cv2.destroyWindow('ROI')
            cv2.destroyWindow('Binary Mask')
            cv2.destroyWindow('Overlay')
            self.windows_open = False

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