import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.srv import SetParameters
from math import pi, sqrt, fabs
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry

class SquareTrajectoryUpdater(Node):
    def __init__(self):
        super().__init__('square_trajectory_updater')

        self.points = [
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0)
        ]

        self.settling_timer = None

        self.current_index = -1  # Start before first goal
        self.target_node_name = '/inverse_kinematics'
        self.param_client = self.create_client(SetParameters, f'{self.target_node_name}/set_parameters')

        # Current state
        self.curr_x = None
        self.curr_y = None
        self.curr_yaw = None

        # Thresholds
        self.position_threshold = 0.02
        self.yaw_threshold = 0.05

        self.last_sent = None
        self.goal_active = False

        # Subscriptions
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Float32, '/current_yaw', self.yaw_callback, 10)

        # Timer: continuously checks and updates goals
        self.timer = self.create_timer(0.5, self.check_and_update)

        self.get_logger().info("📍 Position-based Square Trajectory Updater started")

    def odom_callback(self, msg):
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        self.get_logger().info(f"[ODOM] x={self.curr_x:.2f}, y={self.curr_y:.2f}")

    def yaw_callback(self, msg):
        self.curr_yaw = msg.data
        self.get_logger().info(f"[YAW] yaw={self.curr_yaw:.2f}")
        
    def check_and_update(self):
        print("asdfasdfasdf")
        if self.curr_x is None or self.curr_y is None or self.curr_yaw is None:
            print("ptm")
            return

        if self.current_index >= len(self.points) - 1 and self.goal_active is False:
            print("ptmaaaa2")
            self.get_logger().info("✅ All goals completed.")
            self.destroy_timer(self.timer)
            return

        if self.goal_active:
            goal_x, goal_y, goal_yaw = self.points[self.current_index]
            dx = goal_x - self.curr_x
            dy = goal_y - self.curr_y
            dyaw = self.angle_diff(goal_yaw, self.curr_yaw)

            dist_error = sqrt(dx**2 + dy**2)
            yaw_error = fabs(dyaw)

            self.get_logger().info(
                f"📍 At x={self.curr_x:.2f}, y={self.curr_y:.2f}, yaw={self.curr_yaw:.2f} → "
                f"Goal: x={goal_x:.2f}, y={goal_y:.2f}, yaw={goal_yaw:.2f} | "
                f"Errors: dist={dist_error:.3f}, yaw={yaw_error:.3f}"
            )

            if dist_error < self.position_threshold and yaw_error < self.yaw_threshold:
                if self.settling_timer is None:
                    self.get_logger().info(f"✅ Goal {self.current_index} reached. Waiting to settle...")
                    self.settling_timer = self.create_timer(1.0, self.settle_and_continue)  # 2 second delay


        if not self.goal_active and self.current_index < len(self.points) - 1:
            self.current_index += 1
            x, y, yaw = self.points[self.current_index]
            self.set_inverse_kinematics_parameters(x, y, yaw)

    def settle_and_continue(self):
        self.get_logger().info("⏳ Settling complete. Moving to next goal.")
        self.goal_active = False
        if self.settling_timer:
            self.settling_timer.cancel()
            self.settling_timer = None


    def angle_diff(self, a, b):
        d = a - b
        while d > pi:
            d -= 2 * pi
        while d < -pi:
            d += 2 * pi
        return d

    def set_inverse_kinematics_parameters(self, x, y, yaw):
        if self.last_sent == (x, y, yaw):
            return
        self.last_sent = (x, y, yaw)
        self.goal_active = True

        self.get_logger().info(f"➡️ Sending goal: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}")

        if not self.param_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('❌ Parameter service not available!')
            return

        request = SetParameters.Request()
        request.parameters = [
            Parameter(name='desired_x', value=x).to_parameter_msg(),
            Parameter(name='desired_y', value=y).to_parameter_msg(),
            Parameter(name='desired_yaw', value=yaw).to_parameter_msg()
        ]

        future = self.param_client.call_async(request)
        future.add_done_callback(self.parameter_response_callback)

    def parameter_response_callback(self, future):
        try:
            result = future.result()
            if result and result.results and result.results[0].successful:
                self.get_logger().info("✅ Parameters updated successfully")
            else:
                self.get_logger().error("❌ Failed to update parameters!")
        except Exception as e:
            self.get_logger().error(f"⚠️ Exception in parameter callback: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = SquareTrajectoryUpdater()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
