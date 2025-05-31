#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Int16
from geometry_msgs.msg import TransformStamped
from tf2_ros import Buffer, TransformListener
import tf_transformations
import os
import json
import yaml
import time


class PoseSaver(Node):
    def __init__(self):
        super().__init__('pose_saver')

        # Parameters
        self.declare_parameter('pose_save_frequency', 2.0)  # Hz
        self.declare_parameter('trajectory_file_name', 'default_trajectory')
        self.add_on_set_parameters_callback(self.param_callback)

        # Subscriber to enable/disable saving
        self.activate_sub = self.create_subscription(Int16, '/pose_saver_enable', self.activate_callback, 10)
        self.active = False

        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Timer setup
        self.timer_period = 1.0 / self.get_parameter('pose_save_frequency').get_parameter_value().double_value
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        # Where to save the file
        self.save_directory = 'src/msr_simulation/data/trajectories'
        os.makedirs(self.save_directory, exist_ok=True)

        self.saved_poses = []

        self.get_logger().info("PoseSaver node ready.")

    def param_callback(self, params):
        for param in params:
            if param.name == 'pose_save_frequency' and param.type_ == param.Type.DOUBLE:
                self.timer.cancel()
                self.timer_period = 1.0 / param.value
                self.timer = self.create_timer(self.timer_period, self.timer_callback)
                self.get_logger().info(f"Updated save frequency to {param.value} Hz.")
            elif param.name == 'trajectory_file_name' and param.type_ == param.Type.STRING:
                self.get_logger().info(f"Updated filename to '{param.value}.yaml'")
        return SetParametersResult(successful=True)

    def activate_callback(self, msg: Int16):
        if msg.data == 1 and not self.active:
            self.get_logger().info("🟢 Pose saving activated.")
            self.active = True
        elif msg.data == 0 and self.active:
            self.get_logger().info("🔴 Pose saving deactivated. Saving to file...")
            self.active = False
            self.save_to_yaml()

    def timer_callback(self):
        if not self.active:
            return

        try:
            now = rclpy.time.Time()
            tf: TransformStamped = self.tf_buffer.lookup_transform(
                'map', 'base_link', now, timeout=rclpy.duration.Duration(seconds=0.5)
            )

            t = tf.transform.translation
            q = tf.transform.rotation
            _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])

            pose = {
                'x': round(t.x, 3),
                'y': round(t.y, 3),
                'theta': round(yaw, 4)
            }

            self.saved_poses.append(pose)
            self.get_logger().info(f"✅ Saved pose: {pose}")

        except Exception as e:
            self.get_logger().warn(f"Transform error: {e}")

    def save_to_yaml(self):
        if not self.saved_poses:
            self.get_logger().warn("No poses collected. Nothing to save.")
            return

        file_name = self.get_parameter('trajectory_file_name').get_parameter_value().string_value
        file_path = os.path.join(self.save_directory, f"{file_name}.yaml")

        yaml_data = {
            '/path_generator': {
                'ros__parameters': {
                    'sample_time': round(self.timer_period, 4),
                    'waypoints_json': json.dumps(self.saved_poses, indent=4)
                }
            }
        }

        with open(file_path, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False)

        self.get_logger().info(f"📁 Trajectory saved to: {file_path}")
        self.saved_poses = []


def main(args=None):
    rclpy.init(args=args)
    node = PoseSaver()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
