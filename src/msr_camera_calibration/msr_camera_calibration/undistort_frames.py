import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os

class UndistortNode(Node):
    def __init__(self):
        super().__init__('undistort_node')

        # === Load camera parameters ===
        calib_path = 'src/msr_camera_calibration/data/calibration_data/calibration_data2.npz'
        if not os.path.exists(calib_path):
            self.get_logger().error(f"❌ Calibration file not found: {calib_path}")
            rclpy.shutdown()
            return

        data = np.load(calib_path)
        self.K = data['K']
        self.dist = data['dist']
        self.get_logger().info("📦 Calibration parameters loaded")

        # === Image bridge and subscriptions ===
        self.bridge = CvBridge()
        self.subscription = self.create_subscription( Image, '/image_raw', self.image_callback, 10)
        self.publisher = self.create_publisher( Image, '/image_undistorted', 10)

        self.new_K = None
        self.map1 = None
        self.map2 = None
        self.frame_size = None

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Initialize undistort map only once
        if self.map1 is None:
            h, w = frame.shape[:2]
            self.frame_size = (w, h)
            self.new_K, _ = cv2.getOptimalNewCameraMatrix(self.K, self.dist, (w, h), 1, (w, h))
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                self.K, self.dist, None, self.new_K, (w, h), cv2.CV_16SC2)
            self.get_logger().info("🗺 Undistort maps initialized")

        # Undistort and publish
        undistorted = cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
        msg_out = self.bridge.cv2_to_imgmsg(undistorted, encoding='bgr8')
        self.publisher.publish(msg_out)

        # Optional preview
        # cv2.imshow("Undistorted", undistorted)
        # cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = UndistortNode()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
