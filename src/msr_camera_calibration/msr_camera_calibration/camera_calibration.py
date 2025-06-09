import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import glob
import os

class CameraCalibrator(Node):
    def __init__(self):
        super().__init__('camera_calibrator')
        self.checkerboard = (9, 6)
        self.square_size = 15
        self.image_glob = "src/msr_camera_calibration/data/calib_imgs/test2/calib_*.png"
        self.camera_params_path = self.get_unique_path("src/msr_camera_calibration/data/calibration_data", "calibration_data", ".npz")

        self.calibrate()

    def get_unique_path(self, dir_path, base_name, ext):
        os.makedirs(dir_path, exist_ok=True)
        i = 0
        while True:
            suffix = f"{i}" if i > 0 else ""
            candidate = os.path.join(dir_path, f"{base_name}{suffix}{ext}")
            if not os.path.exists(candidate):
                return candidate
            i += 1


    def calibrate(self):
        objp = np.zeros((self.checkerboard[0]*self.checkerboard[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:self.checkerboard[0], 0:self.checkerboard[1]].T.reshape(-1,2)
        objp *= self.square_size

        objpoints = []
        imgpoints = []

        images = glob.glob(self.image_glob)
        self.get_logger().info(f"🧮 Found {len(images)} images to process.")

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard, None)

            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                objpoints.append(objp)
                imgpoints.append(corners2)

                cv2.drawChessboardCorners(img, self.checkerboard, corners2, ret)
                cv2.imshow('Corners', img)
                cv2.waitKey(100)

                self.get_logger().info(f"✅ Checkerboard found in: {fname}")
            else:
                self.get_logger().warn(f"❌ No checkerboard found in: {fname}")

        cv2.destroyAllWindows()
        ret, K, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        if ret:
            self.get_logger().info("✅ Calibration successful!")
            np.savez(self.camera_params_path, K=K, dist=dist)

            img = cv2.imread(images[0])
            h, w = img.shape[:2]
            new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))
            undistorted = cv2.undistort(img, K, dist, None, new_K)
            cv2.imshow("Undistorted", undistorted)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            self.get_logger().info(f"🎯 K:\n{K}\n📏 Dist:\n{dist.ravel()}")
        else:
            self.get_logger().error("❌ Calibration failed.")

def main(args=None):
    rclpy.init(args=args)
    node = CameraCalibrator()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
