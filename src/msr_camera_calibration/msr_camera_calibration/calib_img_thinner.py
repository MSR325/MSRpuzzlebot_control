import rclpy
from rclpy.node import Node
import os
import glob

class CalibImageThinner(Node):
    def __init__(self):
        super().__init__('calib_image_thinner')

        self.declare_parameter('folder_path', 'src/msr_camera_calibration/data/calib_imgs/test1')
        self.declare_parameter('pattern', 'calib_*.png')
        self.declare_parameter('nth', 2)

        folder = self.get_parameter('folder_path').value
        pattern = self.get_parameter('pattern').value
        nth = self.get_parameter('nth').value

        full_glob = os.path.join(folder, pattern)
        files = sorted(glob.glob(full_glob))

        self.get_logger().info(f"📂 Found {len(files)} images")
        keep_count = 0
        delete_count = 0

        for i, f in enumerate(files):
            if i % nth == 0:
                keep_count += 1
                continue
            try:
                os.remove(f)
                delete_count += 1
            except Exception as e:
                self.get_logger().error(f"❌ Failed to delete {f}: {e}")

        self.get_logger().info(f"✅ Kept {keep_count} images, deleted {delete_count}")

        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = CalibImageThinner()
    # rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
