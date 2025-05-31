# 

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

class LineFollowerCentroid(Node):
    def __init__(self):
        super().__init__('line_follower_centroid')
        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image, '/image_raw', self.image_callback, 10)
        
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info('Line Follower Node Started')

        # Create control window and trackbars
        cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
        cv2.createTrackbar('Threshold', 'Controls', 95, 255, lambda x: None)
        cv2.createTrackbar('Blur Kernel', 'Controls', 11, 31, lambda x: None)
        cv2.createTrackbar('Morph Kernel', 'Controls', 2, 31, lambda x: None)
        cv2.createTrackbar('Block Size', 'Controls', 39, 51, lambda x: None)
        cv2.createTrackbar('C (Bias)', 'Controls', 12, 20, lambda x: None)

        self.load_from_file = True  # Set to False to use manual selection
        self.homography_matrix_path = "src/motor_control/data/homography.npy"

        self.homography_matrix = None
        self.warp_size = (200, 200)  # output size of warped image
        self.selecting_points = True
        self.src_points = []

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
        v_equalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(v)

        blurred = cv2.GaussianBlur(v_equalized, (blur_k, blur_k), 0)

        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size, c_bias
        )

        kernel = np.ones((morph_k, morph_k), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)
        binary = cv2.erode(binary, kernel, iterations=1)

        return binary
    
    def image_callback(self, msg):
        # Convert ROS image to OpenCV BGR
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Warp full image if homography is ready
        if self.homography_matrix is not None:
            warped = cv2.warpPerspective(frame, self.homography_matrix, self.warp_size)
        else:
            warped = frame

        # Split warped image into 3 vertical ROIs
        h, w, _ = warped.shape
        third = w // 3
        twelve_div = w // 12
        roi_left = warped[:, 0:5*twelve_div]
        roi_middle_start = 2 * twelve_div
        roi_middle_end = 10 * twelve_div
        roi_middle = warped[:, roi_middle_start:roi_middle_end]
        roi_right = warped[:, 7*twelve_div:]

        # Trackbar parameters
        blur_k = max(1, cv2.getTrackbarPos('Blur Kernel', 'Controls') | 1)
        morph_k = max(1, cv2.getTrackbarPos('Morph Kernel', 'Controls') | 1)
        block_size = max(3, cv2.getTrackbarPos('Block Size', 'Controls') | 1)
        c_bias = cv2.getTrackbarPos('C (Bias)', 'Controls')

        # Preprocess only middle ROI for control logic
        binary_middle = self.preprocess_region(roi_middle, blur_k, block_size, c_bias, morph_k)

        # Find line position in middle ROI only
        line_middle = int(np.argmax(np.sum(binary_middle, axis=0)))
        global_line_x = roi_middle_start + line_middle  # <-- correct this line
        center_x = w // 2
        offset = global_line_x - center_x

        # Velocity control logic
        Kp = 0.005
        ang_z = -Kp * float(offset)
        linear_x = 0.15 if abs(offset) < 15 else 0.08
        linear_x *= self.color_flag_multiplier
        ang_z *= self.color_flag_multiplier

        # Publish Twist command
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = ang_z
        self.publisher.publish(twist)
        self.get_logger().info(f"Offset: {offset}, Publishing: linear_x={linear_x}, angular_z={ang_z}")

        # Debug: Preprocess left and right only for display
        binary_left = self.preprocess_region(roi_left, blur_k, block_size, c_bias, morph_k)
        binary_right = self.preprocess_region(roi_right, blur_k, block_size, c_bias, morph_k)

        # Optional: visualize peak positions in log
        peak_l = int(np.argmax(np.sum(binary_left, axis=0)))
        peak_r = int(np.argmax(np.sum(binary_right, axis=0)))
        self.get_logger().info(f"Peaks: L={peak_l}, M={line_middle}, R={peak_r}")

        # Overlay display
        overlay = warped.copy()
        cv2.line(overlay, (global_line_x, 0), (global_line_x, h), (0, 255, 0), 2)
        cv2.line(overlay, (center_x, 0), (center_x, h), (0, 0, 255), 2)

        # Show views
        cv2.imshow("ROI", warped)
        cv2.imshow("Overlay", overlay)
        cv2.imshow("Left", binary_left)
        cv2.imshow("Middle", binary_middle)
        cv2.imshow("Right", binary_right)
        cv2.waitKey(1)



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

            



if __name__ == '__main__':
    main()