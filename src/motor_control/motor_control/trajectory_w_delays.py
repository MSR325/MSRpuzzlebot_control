import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.srv import SetParameters
from math import pi
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
        self.current_index = 0
        self.target_node_name = '/inverse_kinematics'
        self.param_client = self.create_client(SetParameters, f'{self.target_node_name}/set_parameters')

        # Current robot state
        self.curr_x = None
        self.curr_y = None
        self.curr_yaw = None


        # Thresholds
        self.position_threshold = 0.02  # meters
        self.yaw_threshold = 0.05        # radians

        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Float32, '/current_yaw/data', self.yaw_callback, 10)

        self.update_interval = 5.0
        self.timer = self.create_timer(self.update_interval, self.timer_callback)

        self.get_logger().info("🕒 Timer-based Square Trajectory Updater started")

        # Timer to periodically check if goal is reached
        self.timer = self.create_timer(0.5, self.check_and_update)

        self.first_goal_sent = False
        self.last_sent = None

        # Send first goal after 1s to ensure service is up
        self.create_timer(1.0, self.send_initial_goal)

    def odom_callback(self, msg):
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y

    def yaw_callback(self, msg):
        self.curr_yaw = msg.data

    def send_initial_goal(self):
        if not self.first_goal_sent:
            self.first_goal_sent = True
            x, y, yaw = self.points[0]
            self.set_inverse_kinematics_parameters(x, y, yaw)

    def timer_callback(self):
        self.current_index += 1

        if self.current_index >= len(self.points):
            self.get_logger().info("✅ Finished sending all goals.")
            self.destroy_timer(self.timer)
            return

        x, y, yaw = self.points[self.current_index]
        self.set_inverse_kinematics_parameters(x, y, yaw)

    def set_inverse_kinematics_parameters(self, x, y, yaw):
        if self.last_sent == (x, y, yaw):
            return  # Skip duplicate goal
        self.last_sent = (x, y, yaw)

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