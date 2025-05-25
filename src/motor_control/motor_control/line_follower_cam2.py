#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

class LineFollower(Node):
    def __init__(self):
        super().__init__('line_follower')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, '/image_raw', self.image_callback, 10)
        self.color_flag_sub = self.create_subscription(
            Float32, '/fms_detection', self.color_flag_callback, 10)  
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info('Line Follower Node Started')

        self.color_flag_multiplier = 1.0

        # Create control window and trackbars
        cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
        cv2.createTrackbar('Threshold', 'Controls', 144, 255, lambda x: None)
        cv2.createTrackbar('Blur Kernel', 'Controls', 3, 31, lambda x: None)
        cv2.createTrackbar('Morph Kernel', 'Controls', 0, 31, lambda x: None)

    def image_callback(self, msg):
        # Convert ROS image to OpenCV BGR
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        height, width, _ = frame.shape

        # vertical bounds (bottom third)
        y_start = int(height * 4/5)
        # y_start = int(height * 0.0)
        y_end   = height
        # horizontal bounds (central 50% of width)
        x_start = int(width * 0.2)
        # x_start = int(width * 0.0)
        x_end   = int(width * 0.8)
        # x_end   = int(width)

        # extract centered ROI
        roi = frame[y_start:y_end, x_start:x_end]


        # Read trackbar positions
        # thresh_val = 110
        # blur_k = 3
        # morph_k = 0
        thresh_val = cv2.getTrackbarPos('Threshold', 'Controls')
        blur_k = cv2.getTrackbarPos('Blur Kernel', 'Controls')
        morph_k = cv2.getTrackbarPos('Morph Kernel', 'Controls')

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
            self.publisher.publish(twist)
            time.sleep(0.5)  # Wait for 0.5 seconds
            # rclpy.spin_once(self, timeout_sec=1.0)  # Wait for 1 second
            return

        # Detect the peak white line
        line_x = int(np.argmax(histogram))  # Find the column with the highest intensity

        # Compute offset from center
        center_x = (x_end - x_start) // 2
        offset = line_x - center_x
        # self.get_logger().debug(f'Peaks: {peaks if len(histogram)>=3 else [line_x]}')
        self.get_logger().info(f'Offset: {offset}')

        # Proportional controller for angular velocity
        Kp = 0.008
        ang_z = -Kp * float(offset)
        # Deadband to avoid jitter
        if abs(offset) < 15:
            ang_z = 0.0
            linear_x = 0.1
        else:
            linear_x = 0.05

        # Publish command
        twist = Twist()
        twist.linear.x = linear_x * self.color_flag_multiplier
        twist.angular.z = ang_z * self.color_flag_multiplier
        self.get_logger().warning(f'Publishing: linear_x={linear_x}, angular_z={ang_z}')
        self.publisher.publish(twist)

        # Overlay: draw detected and center lines
        overlay = roi.copy()
        # detected line in green
        cv2.line(overlay,    (line_x, 0), (line_x, overlay.shape[0]), (0, 255, 0), 2)
        # center line in red
        cv2.line(overlay, (center_x, 0), (center_x, overlay.shape[0]), (0, 0, 255), 2)

        # Display windows
        cv2.imshow('ROI', roi)
        cv2.imshow('Binary Mask', binary)
        cv2.imshow('Overlay', overlay)
        cv2.waitKey(1)

    def color_flag_callback(self, msg):
        # Get the color flag from the message
        self.color_flag_multiplier = msg.data
        

def main(args=None):
    rclpy.init(args=args)
    node = LineFollower()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
