import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Float32, Int16
from rcl_interfaces.msg import SetParametersResult

from enum import Enum

class MovementState(Enum): 
    STOP = 0 
    CAUTION = 1 
    ADVANCE = 2 

class DetectionFSM(Node): 
    def __init__(self):
        super().__init__('detection_fsm')

        # Inicializa Variable de Control 
        self.color_flag_multiplier = 0.0 
        self.state = MovementState.STOP 

        # Suscripcion 
        self.color_flag_sub = self.create_subscription(String, 'color_flag', self.color_flag_callback, 10)
        self.activate_sub = self.create_subscription(Int16, '/detection_fsm_enable', self.activate_detection_fsm_callback, 10)
        
        # Publisher 
        self.fsm_action_pub = self.create_publisher(Float32, '/fsm_action', 10)
        self.active_detection_fsm = 1

        self.get_logger().info("Nodo FSM de Deteccion Iniciado")

    def activate_detection_fsm_callback(self, msg):
        self.active_detection_fsm = msg.data
        self.get_logger().info(f"detection fsm node state: {self.active_detection_fsm}")

    def transition(self, color_flag: str): 
        if self.state == MovementState.STOP: 
            if color_flag == "green": 
                self.state = MovementState.ADVANCE
        elif self.state == MovementState.ADVANCE:
            if color_flag == "yellow": 
                self.state = MovementState.CAUTION
            elif color_flag == "red": 
                self.state = MovementState.STOP
        elif self.state == MovementState.CAUTION: 
            if color_flag == "red":
                self.state = MovementState.STOP
            elif color_flag == "green": 
                self.state = MovementState.ADVANCE 

    def actuate(self):
        if (self.active_detection_fsm == 1): 
            if self.state == MovementState.STOP: 
                self.color_flag_multiplier = 0.0
                self.get_logger().info("RED detected -> Stop")    
            elif self.state == MovementState.CAUTION: 
                self.color_flag_multiplier = 0.5
                self.get_logger().info("YELLOW detected -> Caution")
            elif self.state == MovementState.ADVANCE: 
                self.color_flag_multiplier = 1.0
                self.get_logger().info("GREEN detected -> Advance")
        else:
            self.color_flag_multiplier = 1.0
        
        msg = Float32()
        msg.data = self.color_flag_multiplier
        self.fsm_action_pub.publish(msg)

    def color_flag_callback(self, msg: String): 
        color_flag = msg.data.lower()
        self.transition(color_flag)
        self.actuate()




def main(args=None):
    rclpy.init(args=args)
    node = DetectionFSM()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
