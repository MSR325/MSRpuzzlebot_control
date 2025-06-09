#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import os

class HomographySaver(Node):
    def __init__(self):
        super().__init__('homography_saver')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, '/image_undistorted', self.image_callback, 10)

        self.latest_frame = None
        self.homography = None
        self.src_points_scaled = []
        self.output_path = 'homography_master2.npy'
        self.warp_size = (300, 300)

        # Scaling factor for UI
        self.scale = 3.0

        cv2.namedWindow("Camera")
        cv2.setMouseCallback("Camera", self.mouse_callback)

        self.get_logger().info("Click 4 points on the scaled image to compute homography.")

    def mouse_callback(self, event, x, y, flags, param=None):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Scale point back to original coordinates
            x_orig = x / self.scale
            y_orig = y / self.scale
            self.src_points_scaled.append([x_orig, y_orig])
            self.get_logger().info(f"Point selected (scaled back): ({x_orig:.1f}, {y_orig:.1f})")

            if len(self.src_points_scaled) == 4:
                pts_src = np.array(self.src_points_scaled, dtype=np.float32)
                pts_dst = np.array([
                    [0, 0],
                    [self.warp_size[0] - 1, 0],
                    [self.warp_size[0] - 1, self.warp_size[1] - 1],
                    [0, self.warp_size[1] - 1]
                ], dtype=np.float32)

                self.homography = cv2.getPerspectiveTransform(pts_src, pts_dst)
                np.save(self.output_path, self.homography)
                self.get_logger().info(f"✅ Homography saved to {self.output_path}")

    def image_callback(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return

        if self.latest_frame is not None:
            frame = self.latest_frame.copy()
            scaled_frame = cv2.resize(frame, None, fx=self.scale, fy=self.scale)

            # Draw selected points (scaled)
            for pt in self.src_points_scaled:
                pt_scaled = (int(pt[0] * self.scale), int(pt[1] * self.scale))
                cv2.circle(scaled_frame, pt_scaled, 5, (0, 255, 0), -1)

            cv2.imshow("Camera", scaled_frame)

            if self.homography is not None:
                warped = cv2.warpPerspective(frame, self.homography, self.warp_size)
                cv2.imshow("Warped View", warped)

            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = HomographySaver()
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
