import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
plt.ion()


class CheckerboardImageSaver(Node):
    def __init__(self):
        super().__init__('checkerboard_image_saver')
        self.checkerboard = (9, 6)
        self.image_index = 0
        self.latest_image = None
        self.latest_gray = None
        self.latest_found = False
        self.latest_raw = None
        self.bridge = CvBridge()

        self.calib_imgs_data = "src/msr_camera_calibration/data/calib_imgs/test2"
        self.detected_checkerboards_data = "src/msr_camera_calibration/data/detected_checkerboards"
        os.makedirs(self.calib_imgs_data, exist_ok=True)
        os.makedirs(self.detected_checkerboards_data, exist_ok=True)

        # ROS 2 subscription
        self.subscription = self.create_subscription(Image,'/image_raw',self.image_callback,10)

        # Matplotlib figure
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        self.get_logger().info("📷 Subscribed to /image_raw. Detects and auto-saves checkerboards. Press 'q' to quit.")

        # Periodic checker
        self.timer = self.create_timer(0.03, self.process_loop)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_image = cv_image.copy()
            self.latest_raw = cv_image.copy()
        except Exception as e:
            self.get_logger().error(f"❌ cv_bridge error: {e}")

    def on_key_press(self, event):
        if event.key == 'q':
            self.get_logger().info("👋 Shutting down...")
            plt.close('all')
            rclpy.shutdown()

    def process_loop(self):
        if self.latest_image is None:
            return

        frame = self.latest_image.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, self.checkerboard, None)
        self.latest_found = found

        if found:
            corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            cv2.drawChessboardCorners(frame, self.checkerboard, corners, found)
            self.latest_image = frame.copy()

            # Auto-save
            raw_fname = os.path.join(self.calib_imgs_data, f"calib_live_{self.image_index}.png")
            drawn_fname = os.path.join(self.detected_checkerboards_data, f"checkerboard_live_{self.image_index}.png")
            cv2.imwrite(raw_fname, self.latest_raw)
            # cv2.imwrite(drawn_fname, self.latest_image)

            self.get_logger().info(f"💾 Auto-saved raw:      {raw_fname}")
            # self.get_logger().info(f"🟩 Auto-saved detected: {drawn_fname}")
            self.image_index += 1
        else:
            cv2.putText(frame, "❌ No checkerboard detected", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            self.latest_image = frame.copy()

        # Convert BGR to RGB for matplotlib
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.ax.clear()
        self.ax.imshow(rgb)
        self.ax.set_title("Checkerboard Detection (press 'q' to quit)")
        self.ax.axis('off')
        plt.pause(0.001)


def main(args=None):
    rclpy.init(args=args)
    node = CheckerboardImageSaver()
    rclpy.spin(node)

if __name__ == '__main__':
    main()