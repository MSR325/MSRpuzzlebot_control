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
from sensor_msgs.msg import JointState

class OdometryNode(Node):
    def __init__(self):
        super().__init__('simulation_odometry_node')
        # Declarar parámetros
        self.declare_parameter('wheel_radius', 0.01)       # [m]
        self.declare_parameter('wheel_separation', 0.19)    # [m]173
        self.declare_parameter('sample_time', 0.005)         # [s]
        # Parámetro para el ruido (sigma²)
        self.declare_parameter('sigma_squared', 0.1)

        # Leer parámetros iniciales
        self.wheel_radius = self.get_parameter('wheel_radius').value
        self.wheel_separation = self.get_parameter('wheel_separation').value
        self.sample_time = self.get_parameter('sample_time').value
        self.sigma_squared = self.get_parameter('sigma_squared').value

        # Velocidad en X y angular (locales)
        self.V_forward = 0.0
        self.omega = 0.0

        # Variables de pose
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        
        # Matriz de covarianza del estado [x, y, theta]
        self.Sig = np.zeros((3, 3))
        
        # Posiciones de las ruedas
        self.left_wheel_pos = 0.0  # [rad]
        self.right_wheel_pos = 0.0  # [rad]


        # Crear un perfil de QoS con BEST_EFFORT para las suscripciones a los encoders
        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT
        
        # Subscripciones a las velocidades de las ruedas
        # self.left_speed_sub = self.create_subscription(
        #     Float32, 'VelocityEncL', self.left_speed_callback, qos_profile)  # left/motor_speed_y VelocityEncL
        # self.right_speed_sub = self.create_subscription(
        #     Float32, 'VelocityEncR', self.right_speed_callback, qos_profile)  # right/motor_speed_y VelocityEncR
        self.teleop_vel_sub = self.create_subscription( Twist, '/cmd_vel', self.teleop_vel_sub_callback, qos_profile)  # left/motor_speed_y VelocityEncL
        
        # Publicador de odometría
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        # Publicadores de velocidades globales
        self.global_vel_x_pub = self.create_publisher(Float32, 'global_velocity_x', 10)
        self.global_vel_y_pub = self.create_publisher(Float32, 'global_velocity_y', 10)
        
        # Timer para integración
        self.timer = self.create_timer(self.sample_time, self.timer_callback)
        
        # Callback para actualización dinámica de parámetros
        self.add_on_set_parameters_callback(self.parameter_callback)
        
        self.joint_state_pub = self.create_publisher(JointState, 'joint_states', 10)

        self.get_logger().info("Odometry Node Started")

    def teleop_vel_sub_callback(self, msg: Twist):
        self.V_forward = msg.linear.x
        self.omega = msg.angular.z

    # def left_speed_callback(self, msg: Float32):
    #     self.left_speed = msg.data
        
    # def right_speed_callback(self, msg: Float32):
    #     self.right_speed = msg.data
        
    def timer_callback(self):
        dt = self.sample_time

        # Integrate pose
        self.x += self.V_forward * math.cos(self.theta) * dt
        self.y += self.V_forward * math.sin(self.theta) * dt
        self.theta += self.omega * dt

        global_vel_x = self.V_forward * math.cos(self.theta)
        global_vel_y = self.V_forward * math.sin(self.theta)

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

        # Meaning: z, roll, pitch are invalid (9999), x, y, yaw are trusted (small variance)
        odom_msg.twist.twist.linear.x = self.V_forward
        odom_msg.twist.twist.angular.z = self.omega

        # Publish global velocity if needed
        gx_msg = Float32()
        gx_msg.data = global_vel_x
        self.global_vel_x_pub.publish(gx_msg)

        gy_msg = Float32()
        gy_msg.data = global_vel_y
        self.global_vel_y_pub.publish(gy_msg)

        # Compute wheel rotation based on robot linear/angular velocity
        v_l = self.V_forward - (self.wheel_separation / 2.0) * self.omega
        v_r = self.V_forward - (self.wheel_separation / 2.0) * self.omega

        # Integrate to get wheel positions
        self.left_wheel_pos += (v_l / self.wheel_radius) * dt
        self.right_wheel_pos += (v_r / self.wheel_radius) * dt

        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.name = ['left_wheel_joint', 'right_wheel_joint']
        joint_state.position = [self.left_wheel_pos, self.right_wheel_pos]

        self.odom_pub.publish(odom_msg)
        self.joint_state_pub.publish(joint_state)



        
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
