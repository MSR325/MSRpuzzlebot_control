import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from math import pi, sqrt, fabs

class SquareTrajectoryUpdater(Node):
    def __init__(self):
        super().__init__('square_trajectory_updater')

        # Target positions and yaw
        self.points = [
            (1.0, 0.0, 0.0),
            (1.0, 1.0, pi / 2),
            (0.0, 1.0, pi),
            (0.0, 0.0, -pi / 2)
        ]
        self.current_index = 0

        # Current robot state
        self.curr_x = None
        self.curr_y = None
        self.curr_yaw = None

        # Thresholds
        self.position_threshold = 0.02  # meters
        self.yaw_threshold = 0.05        # radians

        # Subscriptions
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Float32, '/current_yaw/data', self.yaw_callback, 10)

        # Timer to periodically check if goal is reached
        self.timer = self.create_timer(0.5, self.check_and_update)

        self.get_logger().info("Square trajectory node started!")

    def odom_callback(self, msg):
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y

    def yaw_callback(self, msg):
        self.curr_yaw = msg.data

    def check_and_update(self):
        if self.current_index >= len(self.points):
            self.get_logger().info("Square trajectory complete.")
            self.destroy_timer(self.timer)
            return

        if self.curr_x is None or self.curr_y is None or self.curr_yaw is None:
            return  # Wait for all data to be available

        goal_x, goal_y, goal_yaw = self.points[self.current_index]

        if self.reached_goal(goal_x, goal_y, goal_yaw):
            self.get_logger().info(f"✅ Reached: ({goal_x:.2f}, {goal_y:.2f}, {goal_yaw:.2f})")
            self.current_index += 1
            if self.current_index < len(self.points):
                next_x, next_y, next_yaw = self.points[self.current_index]
                self.update_parameters(next_x, next_y, next_yaw)

    def reached_goal(self, goal_x, goal_y, goal_yaw):
        dx = goal_x - self.curr_x
        dy = goal_y - self.curr_y
        dyaw = self.angle_diff(goal_yaw, self.curr_yaw)
        return sqrt(dx**2 + dy**2) < self.position_threshold and fabs(dyaw) < self.yaw_threshold

    def update_parameters(self, x, y, yaw):
        self.set_parameters([
            Parameter('desired_x', Parameter.Type.DOUBLE, x),
            Parameter('desired_y', Parameter.Type.DOUBLE, y),
            Parameter('desired_yaw', Parameter.Type.DOUBLE, yaw)
        ])
        self.get_logger().info(f"➡️ New goal: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}")

    def angle_diff(self, a, b):
        # Wrap between [-pi, pi]
        d = a - b
        while d > pi:
            d -= 2 * pi
        while d < -pi:
            d += 2 * pi
        return d

def main(args=None):
    rclpy.init(args=args)
    node = SquareTrajectoryUpdater()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
