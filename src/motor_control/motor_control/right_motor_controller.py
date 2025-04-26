#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from rcl_interfaces.msg import SetParametersResult
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

class RightMotorController(Node):
    def __init__(self):
        super().__init__('right_motor_controller')

        # Parámetros configurables
        self.declare_parameter('kP', 0.02)
        self.declare_parameter('kI', 0.2)
        self.declare_parameter('kD', 0.00005)
        self.declare_parameter('sample_time', 0.018)
        self.declare_parameter('min_output', -0.4)
        self.declare_parameter('max_output', 0.4)
        self.declare_parameter('filter_alpha', 0.05)

        # Obtener valores
        self.kP = self.get_parameter('kP').value
        self.kI = self.get_parameter('kI').value
        self.kD = self.get_parameter('kD').value
        self.sample_time = self.get_parameter('sample_time').value
        self.min_output = self.get_parameter('min_output').value
        self.max_output = self.get_parameter('max_output').value
        self.alpha = self.get_parameter('filter_alpha').value

        # Variables internas
        self.set_point = 0.0
        self.raw_speed = 0.0
        self.filtered_speed = 0.0
        self.integral = 0.0
        self.previous_error = 0.0

        # QoS para encoders
        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT

        # ROS Interfaces
        self.create_subscription(Float32, '/right/set_point', self.set_point_callback, 10)
        self.create_subscription(Float32, '/VelocityEncR', self.feedback_callback, qos_profile)
        self.cmd_pub = self.create_publisher(Float32, '/ControlR', 10)
        self.filtered_pub = self.create_publisher(Float32, '/VelocityEncR_filtered', 10)

        self.timer = self.create_timer(self.sample_time, self.control_loop)
        self.add_on_set_parameters_callback(self.parameter_callback)

        self.get_logger().info('Right Motor Controller with filter and debug topic initialized')

    def set_point_callback(self, msg):
        self.set_point = msg.data

    def feedback_callback(self, msg):
        self.raw_speed = msg.data
        self.filtered_speed = (
            self.alpha * self.raw_speed + (1 - self.alpha) * self.filtered_speed
        )
        self.filtered_pub.publish(Float32(data=self.filtered_speed))

    def control_loop(self):
        error = self.set_point - self.filtered_speed
        self.integral += error * self.sample_time
        derivative = (error - self.previous_error) / self.sample_time
        self.previous_error = error

        output = self.kP * error + self.kI * self.integral + self.kD * derivative
        output = max(self.min_output, min(self.max_output, output))

        self.cmd_pub.publish(Float32(data=output))

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'kP':
                self.kP = param.value
            elif param.name == 'kI':
                self.kI = param.value
            elif param.name == 'kD':
                self.kD = param.value
            elif param.name == 'sample_time' and param.value > 0.0:
                self.sample_time = param.value
                self.timer.cancel()
                self.timer = self.create_timer(self.sample_time, self.control_loop)
            elif param.name == 'min_output':
                self.min_output = param.value
            elif param.name == 'max_output':
                self.max_output = param.value
            elif param.name == 'filter_alpha':
                self.alpha = param.value
        return SetParametersResult(successful=True)

def main(args=None):
    rclpy.init(args=args)
    node = RightMotorController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
