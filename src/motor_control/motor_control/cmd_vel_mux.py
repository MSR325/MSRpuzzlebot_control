#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from custom_interfaces.srv import SwitchPublisher

class CmdVelMux(Node):
    def __init__(self):
        super().__init__('cmd_vel_mux')

        # Subscripciones a los topics de los distintos publicadores
        self.teleop_sub = self.create_subscription(Twist, 'teleop_cmd_vel', self.teleop_callback, 10)
        self.ik_sub = self.create_subscription(Twist, 'ik_cmd_vel', self.ik_callback, 10)
        self.line_sub = self.create_subscription(Twist, 'line_cmd_vel', self.line_callback, 10)

        # Publicadores unificados
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.puzzlebot_cmd_vel_pub = self.create_publisher(Twist, '/puzzlebot/cmd_vel', 10)

        # Servicio para cambiar la fuente activa
        self.active_source = 'teleop'
        self.srv = self.create_service(SwitchPublisher, 'switch_cmd_source', self.switch_callback)

        # Últimos mensajes recibidos por fuente
        self.last_teleop_msg = Twist()
        self.last_ik_msg = Twist()
        self.last_line_msg = Twist()

        # Ramp acceleration
        self.prev_linear = 0.0
        self.prev_angular = 0.0
        self.max_linear_accel = 0.5     # m/s^2
        self.max_angular_accel = 2.8    # rad/s^2
        self.sample_time = 0.018        # s

        # Temporizador de publicación
        self.timer = self.create_timer(self.sample_time, self.timer_callback)

        self.get_logger().info("Velocity Mux Node with ramp-up started.")

    def teleop_callback(self, msg: Twist):
        self.last_teleop_msg = msg

    def ik_callback(self, msg: Twist):
        self.last_ik_msg = msg

    def line_callback(self, msg: Twist):
        self.last_line_msg = msg

    def switch_callback(self, request, response):
        if request.active_source in ['teleop', 'ik', 'line']:
            self.active_source = request.active_source
            response.success = True
            response.message = f"Active source changed to {self.active_source}"
            self.get_logger().info(response.message)
        else:
            response.success = False
            response.message = "Invalid source. Valid options: 'teleop', 'ik', 'line'."
            self.get_logger().warn(response.message)
        return response

    def ramp_velocity(self, current, target, dt, max_accel):
        delta = target - current
        max_delta = max_accel * dt
        delta = max(min(delta, max_delta), -max_delta)
        return current + delta

    def timer_callback(self):
        msg = Twist()
        if self.active_source == 'teleop':
            msg = self.last_teleop_msg
        elif self.active_source == 'ik':
            msg = self.last_ik_msg
        elif self.active_source == 'line':
            msg = self.last_line_msg

        # Aplicar ramping a velocidades
        linear = self.ramp_velocity(self.prev_linear, msg.linear.x, self.sample_time, self.max_linear_accel)
        angular = self.ramp_velocity(self.prev_angular, msg.angular.z, self.sample_time, self.max_angular_accel)

        self.prev_linear = linear
        self.prev_angular = angular

        twist_out = Twist()
        twist_out.linear.x = linear
        twist_out.angular.z = angular

        self.cmd_vel_pub.publish(twist_out)
        self.puzzlebot_cmd_vel_pub.publish(twist_out)


def main(args=None):
    rclpy.init(args=args)   
    node = CmdVelMux()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
