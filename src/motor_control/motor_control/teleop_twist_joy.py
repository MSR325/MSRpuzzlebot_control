#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist

class TeleopTwistJoy(Node):
    def __init__(self):
        super().__init__('teleop_twist_joy')

        # Parámetros
        self.enable_button = 5  # Botón RB típico
        self.axis_linear = 1    # Eje vertical del stick izquierdo
        self.axis_angular = 0   # Eje horizontal del stick derecho
        self.scale_linear = 0.6
        self.scale_angular = 5.0

        self.joy_sub = self.create_subscription(Joy, 'joy', self.joy_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, 'teleop_cmd_vel', 10)

    def joy_callback(self, msg):
        twist = Twist()

        if len(msg.buttons) > self.enable_button and msg.buttons[self.enable_button]:
            # Habilitado: convertir entradas a velocidades
            twist.linear.x = self.scale_linear * msg.axes[self.axis_linear]
            twist.angular.z = self.scale_angular * msg.axes[self.axis_angular]
        else:
            # Si no se presiona el botón, detener
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = TeleopTwistJoy()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
