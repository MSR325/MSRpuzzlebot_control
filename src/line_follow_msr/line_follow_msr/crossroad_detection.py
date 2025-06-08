import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

'''
ESTE PINCHE NODO VE A PARTIR DE 15 cm efrente del robot y termina de ver a los 15 + 26.4 cm aprox (alto)
(ancho) aprox 30 cm de lado a lado
'''

class CrossroadDetector(Node):
    def __init__(self):
        super().__init__('crossroad_detector')
        self.bridge = CvBridge()

        # Parameters
        self.declare_parameter('homography_matrix_path', 'src/line_follow_msr/data/homography_after_calib_firstsegment_decent.npy')
        self.declare_parameter('warp_width', 300)
        self.declare_parameter('warp_height', 300)
        self.declare_parameter('min_segments', 6)
        self.declare_parameter('y_alignment_thresh_px', 100)  # vertical threshold in pixels
        self.declare_parameter('vertical_angle_thresh', 20)  # vertical threshold in pixels

        self.min_segments = self.get_parameter('min_segments').value
        self.y_alignment_thresh_px = self.get_parameter('y_alignment_thresh_px').value
        self.vertical_angle_thresh = self.get_parameter('vertical_angle_thresh').value
        self.homography_matrix = None
        self.warp_size = (
            self.get_parameter('warp_width').value,
            self.get_parameter('warp_height').value
        )

        # Load homography
        try:
            path = self.get_parameter('homography_matrix_path').value
            self.homography_matrix = np.load(path)
            self.get_logger().info('✅ Homography matrix loaded')
        except Exception as e:
            self.get_logger().error(f'Failed to load homography: {e}')

        # Subscriber
        self.subscription = self.create_subscription(Image, '/image_undistorted', self.image_callback, 10)

        # Trackbars
        cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
        cv2.createTrackbar('Blur Kernel', 'Controls', 28, 31, lambda x: None)
        cv2.createTrackbar('Morph Kernel', 'Controls', 7, 31, lambda x: None)
        cv2.createTrackbar('Block Size', 'Controls', 68, 101, lambda x: None)
        cv2.createTrackbar('C (Bias)', 'Controls', 16, 30, lambda x: None)
        # Horizontal aspect (X-direction alignment)
        cv2.createTrackbar('Min Aspect X x10', 'Controls', 10, 100, lambda x: None)
        cv2.createTrackbar('Max Aspect X x10', 'Controls', 30, 100, lambda x: None)

        # Vertical aspect (Y-direction alignment)
        cv2.createTrackbar('Min Aspect Y x10', 'Controls', 4, 100, lambda x: None)
        cv2.createTrackbar('Max Aspect Y x10', 'Controls', 8, 100, lambda x: None)

        cv2.createTrackbar('Min Segments', 'Controls', int(self.min_segments), 8, lambda x: None)
        cv2.createTrackbar('Y Align Threshold', 'Controls', int(self.y_alignment_thresh_px), 200, lambda x: None)
        cv2.createTrackbar('Vertical ang Thrshld', 'Controls', int(self.vertical_angle_thresh), 90, lambda x: None)
        cv2.createTrackbar('Min Width', 'Controls', 2, 300, lambda x: None)
        cv2.createTrackbar('Max Width', 'Controls', 115, 300, lambda x: None)
        cv2.createTrackbar('Min Height', 'Controls', 25, 300, lambda x: None)
        cv2.createTrackbar('Max Height', 'Controls', 89, 300, lambda x: None)
        cv2.createTrackbar('Scale', 'Controls', 2, 10, lambda x: None)

    def preprocess(self, img):
        blur_k = max(1, cv2.getTrackbarPos('Blur Kernel', 'Controls') | 1)
        morph_k = max(1, cv2.getTrackbarPos('Morph Kernel', 'Controls') | 1)
        block_size = max(3, cv2.getTrackbarPos('Block Size', 'Controls') | 1)
        c_bias = cv2.getTrackbarPos('C (Bias)', 'Controls')

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        v_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(v)
        blurred = cv2.GaussianBlur(v_eq, (blur_k, blur_k), 0)
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, block_size, c_bias)
        kernel = np.ones((morph_k, morph_k), np.uint8)
        binary = cv2.erode(binary, kernel, iterations=1)
        binary = cv2.dilate(binary, kernel, iterations=2)
        return binary

    def is_dotted_row(self, binary_roi, overlay, roi_offset_y):

        self.min_segments = cv2.getTrackbarPos('Min Segments', 'Controls')
        self.y_alignment_thresh_px = cv2.getTrackbarPos('Y Align Threshold', 'Controls')
        min_w = cv2.getTrackbarPos('Min Width', 'Controls')
        max_w = cv2.getTrackbarPos('Max Width', 'Controls')
        min_h = cv2.getTrackbarPos('Min Height', 'Controls')
        max_h = cv2.getTrackbarPos('Max Height', 'Controls')
        min_aspect = cv2.getTrackbarPos('Min Aspect X x10', 'Controls') / 10.0
        max_aspect = cv2.getTrackbarPos('Max Aspect X x10', 'Controls') / 10.0

        contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_centroids = []

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / max(h, 1)
            if min_w <= w <= max_w and min_h <= h <= max_h and min_aspect <= aspect <= max_aspect:
                # Draw contour
                offset_contour = c + np.array([[[0, roi_offset_y]]])
                cv2.drawContours(overlay, [offset_contour], -1, (0, 255, 255), 1)
                M = cv2.moments(c)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00']) + roi_offset_y
                    valid_centroids.append((cx, cy))

        if len(valid_centroids) >= self.min_segments:
            ys = np.array([c[1] for c in valid_centroids])
            if np.max(ys) - np.min(ys) <= self.y_alignment_thresh_px:
                # Fit a line through centroids
                centroids_np = np.array(valid_centroids, dtype=np.float32)
                vx, vy, x0, y0 = cv2.fitLine(centroids_np, cv2.DIST_L2, 0, 0.01, 0.01)
                pt1 = (int(x0 - vx * 100), int(y0 - vy * 100))
                pt2 = (int(x0 + vx * 100), int(y0 + vy * 100))
                cv2.line(overlay, pt1, pt2, (0, 255, 0), 1)
                return True

        return False

    def is_dotted_column(self, binary_roi, overlay, roi_offset_x):
        self.min_segments = cv2.getTrackbarPos('Min Segments', 'Controls')
        self.y_alignment_thresh_px = cv2.getTrackbarPos('Y Align Threshold', 'Controls')
        self.vertical_angle_thresh = cv2.getTrackbarPos('Vertical ang Thrshld', 'Controls')
        min_w = cv2.getTrackbarPos('Min Width', 'Controls')
        max_w = cv2.getTrackbarPos('Max Width', 'Controls')
        min_h = cv2.getTrackbarPos('Min Height', 'Controls')
        max_h = cv2.getTrackbarPos('Max Height', 'Controls')

        # Fixed aspect ratio limits for vertical detection
        min_aspect = cv2.getTrackbarPos('Min Aspect Y x10', 'Controls') / 10.0
        max_aspect = cv2.getTrackbarPos('Max Aspect Y x10', 'Controls') / 10.0

        contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_centroids = []

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / max(h, 1)
            if min_w <= w <= max_w and min_h <= h <= max_h and min_aspect <= aspect <= max_aspect:
                offset_contour = c + np.array([[[roi_offset_x, 0]]])
                cv2.drawContours(overlay, [offset_contour], -1, (255, 0, 255), 1)
                M = cv2.moments(c)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00']) + roi_offset_x
                    cy = int(M['m01'] / M['m00'])
                    valid_centroids.append((cx, cy))

        if len(valid_centroids) >= self.min_segments-1:
            centroids_np = np.array(valid_centroids, dtype=np.float32)
            vx, vy, x0, y0 = cv2.fitLine(centroids_np, cv2.DIST_L2, 0, 0.01, 0.01)
            angle = np.arctan2(vy, vx) * 180 / np.pi  # in degrees

            if abs(abs(angle) - 90) <= self.vertical_angle_thresh:  # adjustable threshold
                centroids_np = np.array(valid_centroids, dtype=np.float32)
                vx, vy, x0, y0 = cv2.fitLine(centroids_np, cv2.DIST_L2, 0, 0.01, 0.01)
                pt1 = (int(x0 - vx * 100), int(y0 - vy * 100))
                pt2 = (int(x0 + vx * 100), int(y0 + vy * 100))
                cv2.line(overlay, pt1, pt2, (0, 200, 255), 1)
                return True

        return False
    

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if self.homography_matrix is not None:
            warped = cv2.warpPerspective(frame, self.homography_matrix, self.warp_size)
        else:
            warped = frame.copy()

        binary = self.preprocess(warped)
        height, width = binary.shape

        # ——— Detect Horizontal Dotted Line ———
        y_start = 0
        y_end = height
        middle_region = binary[y_start: y_end, :]
        dotted_horizontal = self.is_dotted_row(middle_region, warped, roi_offset_y=y_start)

        # ——— Detect Vertical Dotted Line ———
        x_start = 0
        x_end = width
        vertical_region = binary[:, x_start:x_end]
        dotted_vertical = self.is_dotted_column(vertical_region, warped, roi_offset_x=x_start)

        # ——— Display status text ———
        status_texts = []
        if dotted_horizontal:
            status_texts.append("Horizontal dotted line detected ✅")
            self.get_logger().info("🚦 CROSSROAD DETECTED by horizontal dotted line!")
        else:
            status_texts.append("Horizontal dotted line not found ❌")

        if dotted_vertical:
            status_texts.append("Vertical dotted line detected ✅")
            self.get_logger().info("🚦 CROSSROAD DETECTED by vertical dotted line!")
        else:
            status_texts.append("Vertical dotted line not found ❌")

        if dotted_horizontal and dotted_vertical:
            status_texts.append("🚦 FULL CROSSROAD DETECTED!")
            self.get_logger().info("🚦 FULL CROSSROAD (T or X) DETECTED!")

        for i, text in enumerate(status_texts):
            cv2.putText(warped, text, (10, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # ——— Show output ———
        scale = 1.0 + max(1, cv2.getTrackbarPos('Scale', 'Controls') | 1) / 10.0
        cv2.imshow("Crossroad Debug", cv2.resize(warped, None, fx=scale, fy=scale))
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = CrossroadDetector()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
