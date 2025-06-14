import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

class OdomTFBroadcaster(Node):
    def __init__(self):
        super().__init__('ekf_odom_tf_broadcaster')

        # Declare and get parameter
        self.declare_parameter("odom_topic", "ekf_odom")
        topic_name = self.get_parameter("odom_topic").get_parameter_value().string_value

        self.get_logger().info(f"📡 Subscribing to odometry topic: {topic_name}")

        self.br = TransformBroadcaster(self)
        self.sub = self.create_subscription(Odometry, topic_name, self.odom_callback, 10)

    def odom_callback(self, msg):
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = msg.header.frame_id
        t.child_frame_id = msg.child_frame_id

        # ✅ Manual assignment of Vector3 fields
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z

        t.transform.rotation = msg.pose.pose.orientation

        self.br.sendTransform(t)
        # self.get_logger().info(f"Published TF: {t.header.frame_id} → {t.child_frame_id}")

def main():
    rclpy.init()
    node = OdomTFBroadcaster()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
