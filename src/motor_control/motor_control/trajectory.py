#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import Twist
import numpy as np 
import math

def saturate(value, min_val, max_val):
    return max(min(value, max_val), min_val)

class Trajectory(Node): 
    def __init__(self):
        super().__init__('ctrl')

        # Declarar parámetros para el PID y características del robot
        self.declare_parameter('wheel_radius', 0.05)       # 5 cm por defecto
        self.declare_parameter('wheel_separation', 0.15)     # 15 cm por defecto
        self.declare_parameter('sample_time', 0.02)

        # Robot parameters 
        self.wheel_radius = self.get_parameter('wheel_radius').value
        self.wheel_separation = self.get_parameter('wheel_separation').value
        self.sample_time = self.get_parameter('sample_time').value

        # Mensajes de salida
        self.robot_speed_msg = Float32()
        self.robot_omega_msg = Float32() 

        # Variables de velocidad de las ruedas y orientación del robot
        self.left_speed = 0.0
        self.right_speed = 0.0
        self.theta = 0.0  # Orientación del robot en radianes

        # Variables para aproximar derivadas (si fuera necesario)
        self.prev_setpoint = None
        self.prev_w_dot_ref = 0.0
        self.prev_time = None

        # Subscripciones a las velocidades de las ruedas
        self.left_speed_sub = self.create_subscription(
            Float32, 'left/motor_speed_y', self.left_speed_callback, 10)
        self.right_speed_sub = self.create_subscription(
            Float32, 'right/motor_speed_y', self.right_speed_callback, 10)

        # Publicadores de velocidades y Twist para Gazebo
        self.linear_vel_pub = self.create_publisher(Float32, 'linear_velocity', 10)
        self.angular_vel_pub = self.create_publisher(Float32, 'angular_velocity', 10)
        self.global_vel_x_pub = self.create_publisher(Float32, 'global_velocity_x', 10)
        self.global_vel_y_pub = self.create_publisher(Float32, 'global_velocity_y', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'puzzlebot/cmd_vel', 10)

        # Timer para actualizar la trayectoria
        self.timer = self.create_timer(self.sample_time, self.timer_callback)
        self.get_logger().info("Trajectory Controller Node Started")

    def left_speed_callback(self, msg: Float32):
        self.left_speed = msg.data

    def right_speed_callback(self, msg: Float32):
        self.right_speed = msg.data

    def timer_callback(self):
        # Calcular la velocidad lineal y angular usando la cinemática diferencial:
        # v = (r/2) * (ω_left + ω_right)
        # ω = (r / L) * (ω_right - ω_left)
        linear_velocity = (self.wheel_radius / 2.0) * (self.left_speed + self.right_speed)
        angular_velocity = (self.wheel_radius / self.wheel_separation) * (self.right_speed - self.left_speed)

        # Integrar la velocidad angular para obtener la orientación (θ)
        self.theta += angular_velocity * self.sample_time

        # Calcular velocidades globales
        global_vel_x = linear_velocity * np.cos(self.theta)
        global_vel_y = linear_velocity * np.sin(self.theta)

        # Saturar las velocidades para evitar comandos excesivos:
        # Velocidad lineal máxima: 0.3 m/s, velocidad angular máxima: 1.0 rad/s
        linear_velocity = saturate(linear_velocity, -0.3, 0.3)
        angular_velocity = saturate(angular_velocity, -1.0, 1.0)

        # Publicar los valores saturados
        linear_msg = Float32()
        linear_msg.data = linear_velocity
        self.linear_vel_pub.publish(linear_msg)

        angular_msg = Float32()
        angular_msg.data = angular_velocity
        self.angular_vel_pub.publish(angular_msg)

        gx_msg = Float32()
        gx_msg.data = global_vel_x
        self.global_vel_x_pub.publish(gx_msg)

        gy_msg = Float32()
        gy_msg.data = global_vel_y
        self.global_vel_y_pub.publish(gy_msg)

        twist_msg = Twist()
        twist_msg.linear.x = linear_velocity
        twist_msg.angular.z = angular_velocity
        self.cmd_vel_pub.publish(twist_msg)

    def parameters_callback(self, params):
        for param in params:
            if param.name == 'wheel_radius':
                if param.value <= 0.0:
                    return SetParametersResult(successful=False, reason="wheel_radius debe ser > 0")
                self.wheel_radius = param.value
                self.get_logger().info(f"wheel_radius actualizado a: {self.wheel_radius}")
            elif param.name == 'wheel_separation':
                if param.value <= 0.0:
                    return SetParametersResult(successful=False, reason="wheel_separation debe ser > 0")
                self.wheel_separation = param.value
                self.get_logger().info(f"wheel_separation actualizado a: {self.wheel_separation}")
            elif param.name == 'sample_time':
                if param.value <= 0.0:
                    return SetParametersResult(successful=False, reason="sample_time debe ser > 0")
                self.sample_time = param.value
                self.timer.destroy()
                self.timer = self.create_timer(self.sample_time, self.timer_callback)
                self.get_logger().info(f"sample_time actualizado a: {self.sample_time}")
        return SetParametersResult(successful=True)
    
def main(args=None):
    rclpy.init(args=args)
    node = Trajectory()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
