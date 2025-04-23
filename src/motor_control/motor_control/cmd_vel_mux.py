#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
# Suponemos que has definido un servicio custom llamado SwitchPublisher
# que contiene:
#   Request: string active_source
#   Response: bool success, string message
from custom_interfaces.srv import SwitchPublisher

class CmdVelMux(Node):
    def __init__(self):
        super().__init__('cmd_vel_mux')
        
        # Subscripciones a los topics de los distintos publicadores (tipo Twist)
        self.teleop_sub = self.create_subscription(Twist, 'teleop_cmd_vel', self.teleop_callback, 10)
        self.ik_sub = self.create_subscription(Twist, 'ik_cmd_vel', self.ik_callback, 10)
        self.pseudo_sub = self.create_subscription(Twist, 'pseudo_cmd_vel', self.pseudo_callback, 10)
        
        # Publicadores únicos a /cmd_vel y /puzzlebot/cmd_vel (tipo Twist)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.puzzlebot_cmd_vel_pub = self.create_publisher(Twist, '/puzzlebot/cmd_vel', 10)
        
        # Servicio para cambiar la fuente activa
        self.active_source = 'teleop'  # Valor por defecto
        self.srv = self.create_service(SwitchPublisher, 'switch_cmd_source', self.switch_callback)
        
        # Variables para almacenar el último mensaje recibido de cada fuente
        self.last_teleop_msg = None
        self.last_ik_msg = None
        self.last_pseudo_msg = None
        
        # Temporizador para publicar periódicamente en /cmd_vel
        self.timer = self.create_timer(0.018, self.timer_callback)

        self.get_logger().info("Velocity Mux Node started.")
    
    def teleop_callback(self, msg: Twist):
        self.last_teleop_msg = msg

    def ik_callback(self, msg: Twist):
        self.last_ik_msg = msg

    def pseudo_callback(self, msg: Twist):
        self.last_pseudo_msg = msg

    def switch_callback(self, request, response):
        # Se espera que request.active_source sea 'teleop', 'ik' o 'pseudo'
        if request.active_source in ['teleop', 'ik', 'pseudo']:
            self.active_source = request.active_source
            response.success = True
            response.message = f"Active source changed to {self.active_source}"
            self.get_logger().info(response.message)
        else:
            response.success = False
            response.message = "Invalid source. Valid options: 'teleop', 'ik', 'pseudo'."
            self.get_logger().warn(response.message)
        return response

    def timer_callback(self):
        # Selecciona el mensaje según la fuente activa
        msg_to_publish = None
        if self.active_source == 'teleop':
            msg_to_publish = self.last_teleop_msg
        elif self.active_source == 'ik':
            msg_to_publish = self.last_ik_msg
        elif self.active_source == 'pseudo':
            msg_to_publish = self.last_pseudo_msg

        if msg_to_publish is not None:
            self.cmd_vel_pub.publish(msg_to_publish)
            self.puzzlebot_cmd_vel_pub.publish(msg_to_publish)

def main(args=None):
    rclpy.init(args=args)
    node = CmdVelMux()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
