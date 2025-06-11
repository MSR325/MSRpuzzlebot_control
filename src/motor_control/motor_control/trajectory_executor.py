#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import math


class SimpleTrajectoryExecutor(Node):
    def __init__(self):
        super().__init__('trajectory_executor')

        # Parameters
        self.declare_parameter('v_lin', 0.15)
        self.declare_parameter('w_gain', 1.5)
        self.declare_parameter('waypoint_tol', 0.02)

        self.v_lin = self.get_parameter('v_lin').value
        self.w_gain = self.get_parameter('w_gain').value
        self.tol = self.get_parameter('waypoint_tol').value

        # State
        self.path = []
        self.pose = None
        self.target_index = 0

        # I/O
        self.create_subscription(Path, '/turn_manager/waypoints', self.path_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 20)

        self.cmd_pub = self.create_publisher(Twist, '/ik_cmd_vel', 10)
        self.done_pub = self.create_publisher(Bool, '/completed_curve', 10)

        self.timer = self.create_timer(0.01, self.follow_path)
        self.get_logger().info('🚗 SimpleTrajectoryExecutor ready (no Pure Pursuit)')

    def path_cb(self, msg: Path):
        self.path = list(msg.poses)
        self.target_index = 0
        self.get_logger().info(f'Received {len(self.path)} waypoints.')

    def odom_cb(self, msg: Odometry):
        self.pose = msg.pose.pose

    def follow_path(self):
        if self.pose is None or not self.path or self.target_index >= len(self.path):
            return

        # Get current position
        x = self.pose.position.x
        y = self.pose.position.y
        yaw = self._yaw_from_quat(self.pose.orientation)

        # Target waypoint
        goal = self.path[self.target_index].pose.position
        dx = goal.x - x
        dy = goal.y - y
        dist = math.hypot(dx, dy)

        if dist < self.tol:
            self.target_index += 1
            if self.target_index >= len(self.path):
                self.cmd_pub.publish(Twist())
                self.done_pub.publish(Bool(data=True))
                self.get_logger().info("✅ Trajectory completed")
            return

        # Compute heading to goal
        heading = math.atan2(dy, dx)
        yaw_error = self._wrap(heading - yaw)

        # Constant linear velocity, proportional angular
        twist = Twist()
        twist.linear.x = self.v_lin
        twist.angular.z = self.w_gain * yaw_error
        self.cmd_pub.publish(twist)

    @staticmethod
    def _yaw_from_quat(q):
        return math.atan2(2 * (q.w * q.z + q.x * q.y),
                          1 - 2 * (q.y**2 + q.z**2))

    @staticmethod
    def _wrap(angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi


def main(args=None):
    rclpy.init(args=args)
    node = SimpleTrajectoryExecutor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
