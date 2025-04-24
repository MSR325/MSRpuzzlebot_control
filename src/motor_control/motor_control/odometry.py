#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math
import numpy as np
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

class OdometryNode(Node):
    def __init__(self):
        super().__init__('odometry_node')
        # Declarar parámetros
        self.declare_parameter('wheel_radius', 0.05)       # [m]
        self.declare_parameter('wheel_separation', 0.19)    # [m]173
        self.declare_parameter('sample_time', 0.018)         # [s]
        # Parámetro para el ruido (sigma²)
        self.declare_parameter('sigma_squared', 0.1)

        # Leer parámetros iniciales
        self.wheel_radius = self.get_parameter('wheel_radius').value
        self.wheel_separation = self.get_parameter('wheel_separation').value
        self.sample_time = self.get_parameter('sample_time').value
        self.sigma_squared = self.get_parameter('sigma_squared').value

        # Variables de pose
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        
        # Matriz de covarianza del estado [x, y, theta]
        self.Sig = np.zeros((3, 3))
        
        # Velocidades de las ruedas
        self.left_speed = 0.0
        self.right_speed = 0.0

        # Crear un perfil de QoS con BEST_EFFORT para las suscripciones a los encoders
        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT
        
        # Subscripciones a las velocidades de las ruedas
        self.left_speed_sub = self.create_subscription(
            Float32, 'VelocityEncL', self.left_speed_callback, qos_profile)  # left/motor_speed_y VelocityEncL
        self.right_speed_sub = self.create_subscription(
            Float32, 'VelocityEncR', self.right_speed_callback, qos_profile)  # right/motor_speed_y VelocityEncR
        
        # Publicador de odometría
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        # Publicadores de velocidades globales
        self.global_vel_x_pub = self.create_publisher(Float32, 'global_velocity_x', 10)
        self.global_vel_y_pub = self.create_publisher(Float32, 'global_velocity_y', 10)
        
        # Timer para integración
        self.timer = self.create_timer(self.sample_time, self.timer_callback)
        
        # Callback para actualización dinámica de parámetros
        self.add_on_set_parameters_callback(self.parameter_callback)
        
        self.get_logger().info("Odometry Node Started")
    
    def left_speed_callback(self, msg: Float32):
        self.left_speed = msg.data
        
    def right_speed_callback(self, msg: Float32):
        self.right_speed = msg.data
        
    def timer_callback(self):
        dt = self.sample_time
        R = self.wheel_radius
        L = self.wheel_separation

        # Calcular velocidades locales:
        # V = (r/2) * (ω_left + ω_right)
        # ω = (r / L) * (ω_right - ω_left)
        V = (R / 2.0) * (self.left_speed + self.right_speed)
        omega = (R / L) * (self.right_speed - self.left_speed)
        
        # Integrar para obtener la pose (usando integración simple)
        self.x += V * math.cos(self.theta) * dt
        self.y += V * math.sin(self.theta) * dt
        self.theta += omega * dt

        # Propagación de la covarianza:
        H = np.array([[1, 0, -dt * V * math.sin(self.theta)],
                      [0, 1,  dt * V * math.cos(self.theta)],
                      [0, 0, 1]])
        
        dH = np.array([[0.5 * dt * R * math.cos(self.theta), 0.5 * dt * R * math.cos(self.theta)],
                       [0.5 * dt * R * math.sin(self.theta), 0.5 * dt * R * math.sin(self.theta)],
                       [dt * R / L, -dt * R / L]])
        
        K = np.array([[self.sigma_squared * abs(self.left_speed), 0],
                      [0, self.sigma_squared * abs(self.right_speed)]])
        
        Q = dH @ K @ dH.T
        
        self.Sig = H @ self.Sig @ H.T + Q

        global_vel_x = V * math.cos(self.theta)
        global_vel_y = V * math.sin(self.theta)

        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_link"
        
        odom_msg.pose.pose.position.x = self.x
        odom_msg.pose.pose.position.y = self.y
        odom_msg.pose.pose.position.z = 0.0
        
        qz = math.sin(self.theta / 2.0)
        qw = math.cos(self.theta / 2.0)
        odom_msg.pose.pose.orientation.z = qz
        odom_msg.pose.pose.orientation.w = qw
        
        odom_msg.pose.covariance[0] = self.Sig[0, 0]
        odom_msg.pose.covariance[1] = self.Sig[0, 1]
        odom_msg.pose.covariance[6] = self.Sig[1, 0]
        odom_msg.pose.covariance[7] = self.Sig[1, 1]
        odom_msg.pose.covariance[35] = self.Sig[2, 2]

        odom_msg.twist.twist.linear.x = V
        odom_msg.twist.twist.angular.z = omega

        self.odom_pub.publish(odom_msg)

        gx_msg = Float32()
        gx_msg.data = global_vel_x
        self.global_vel_x_pub.publish(gx_msg)

        gy_msg = Float32()
        gy_msg.data = global_vel_y
        self.global_vel_y_pub.publish(gy_msg)
        
    def parameter_callback(self, params):
        for param in params:
            if param.name == 'wheel_radius':
                if param.value <= 0.0:
                    return SetParametersResult(successful=False, reason="wheel_radius must be > 0")
                self.wheel_radius = param.value
                self.get_logger().info(f"wheel_radius updated to: {self.wheel_radius}")
            elif param.name == 'wheel_separation':
                if param.value <= 0.0:
                    return SetParametersResult(successful=False, reason="wheel_separation must be > 0")
                self.wheel_separation = param.value
                self.get_logger().info(f"wheel_separation updated to: {self.wheel_separation}")
            elif param.name == 'sample_time':
                if param.value <= 0.0:
                    return SetParametersResult(successful=False, reason="sample_time must be > 0")
                self.sample_time = param.value
                self.get_logger().info(f"sample_time updated to: {self.sample_time}")
                self.timer.destroy()
                self.timer = self.create_timer(self.sample_time, self.timer_callback)
        return SetParametersResult(successful=True)
        
def main(args=None):
    rclpy.init(args=args)
    node = OdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
        
if __name__ == '__main__':
    main()
