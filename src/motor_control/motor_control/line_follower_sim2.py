#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
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
        self.max_linear = 0.1           # max forward speed (m/s)
        self.min_linear = 0.01          # min forward speed so it never stalls

        # Create control window and trackbars
        cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
        cv2.createTrackbar('Threshold', 'Controls', 146, 255, lambda x: None)
        cv2.createTrackbar('Blur Kernel', 'Controls', 3, 31, lambda x: None)
        cv2.createTrackbar('Morph Kernel', 'Controls', 0, 31, lambda x: None)

    def image_callback(self, msg):
        # Convert ROS image to OpenCV BGR
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        height, width, _ = frame.shape

        # define ROI: bottom sixth vertically, center half horizontally
        y_start = int(height * 5/6)
        y_end   = height
        x_start = int(width * 0.25)
        x_end   = int(width * 0.75)
        roi = frame[y_start:y_end, x_start:x_end]

        # read and sanitize trackbars
        thresh_val = cv2.getTrackbarPos('Threshold', 'Controls')
        blur_k = cv2.getTrackbarPos('Blur Kernel', 'Controls')
        morph_k = cv2.getTrackbarPos('Morph Kernel', 'Controls')
        blur_k = max(1, blur_k | 1)       # ensure odd ≥1
        morph_k = max(1, morph_k | 1)     # ensure odd ≥1

        # preprocess
        gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
        _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((morph_k, morph_k), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)
        binary = cv2.erode(binary, kernel, iterations=1)

        # histogram and check
        histogram = np.sum(binary, axis=0)
        if np.max(histogram) == 0:
            self.get_logger().warn('No line detected! Reversing...')
            twist = Twist()
            twist.linear.x = -0.05
            twist.angular.z = 0.0
            self.publisher.publish(twist)
            time.sleep(0.5)
            return

        # find line and error
        line_x   = int(np.argmax(histogram))
        center_x = (x_end - x_start) // 2
        offset   = line_x - center_x
        self.get_logger().info(f'Offset: {offset}')

        # angular velocity
        ang_z = -self.Kp_ang * float(offset)

        # linear velocity scales down as error grows:
        error_norm = min(abs(offset) / float(center_x), 1.0)
        linear_x   = self.max_linear * (1.0 - error_norm)
        linear_x   = max(self.min_linear, linear_x)

        # publish
        twist = Twist()
        twist.linear.x  = linear_x
        twist.angular.z = ang_z
        self.get_logger().warning(f'Publishing: linear_x={linear_x:.3f}, angular_z={ang_z:.3f}')
        self.publisher.publish(twist)

        # overlay for debug
        overlay = roi.copy()
        cv2.line(overlay,    (line_x, 0),          (line_x, overlay.shape[0]), (0,255,0), 2)
        cv2.line(overlay,    (center_x, 0),        (center_x, overlay.shape[0]), (0,0,255), 2)
        cv2.imshow('ROI', roi)
        cv2.imshow('Binary Mask', binary)
        cv2.imshow('Overlay', overlay)
        cv2.waitKey(1)

    def color_flag_callback(self, msg):
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
