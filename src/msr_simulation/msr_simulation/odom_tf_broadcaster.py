#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

class OdomTFBroadcaster(Node):
    def __init__(self):
        super().__init__('odom_tf_broadcaster')

        # Declare and read parameters
        self.declare_parameter("odom_topic", "odom")
        self.declare_parameter("odom_frame_id", "odom")
        self.declare_parameter("base_frame_id", "base_link")

        self.odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        self.odom_frame_id = self.get_parameter("odom_frame_id").get_parameter_value().string_value
        self.base_frame_id = self.get_parameter("base_frame_id").get_parameter_value().string_value

        self.get_logger().info(f"📡 Subscribing to odometry topic: {self.odom_topic}")
        self.get_logger().info(f"🌐 Broadcasting TF: {self.odom_frame_id} → {self.base_frame_id}")

        self.br = TransformBroadcaster(self)
        self.sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)

    def odom_callback(self, msg):
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = self.odom_frame_id
        t.child_frame_id = self.base_frame_id

        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z

        t.transform.rotation = msg.pose.pose.orientation

        self.br.sendTransform(t)

def main():
    rclpy.init()
    node = OdomTFBroadcaster()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
