#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32, Int16
import time

class DetectionFSM(Node):
    def __init__(self):
        super().__init__('detection_fsm')
        self.active_detection_fsm = 1
        self.current_flags = set()
        self.last_detection_time = time.time()  # Track last time a signal was seen

        self.pub = self.create_publisher(Float32, '/fsm_action', 10)
        self.create_subscription(String, '/color_flag', self.cb_color, 10)
        self.create_subscription(String, '/fsm_event', self.cb_event, 10)
        self.create_subscription(Int16, '/detection_fsm_enable', self.cb_enable, 10)

        self.create_timer(0.5, self.timer_step)
        self.get_logger().info("🚦 FSM (con timeout de 2s) iniciada")

    def cb_enable(self, msg: Int16):
        self.active_detection_fsm = msg.data
        self.get_logger().info(f"FSM enable = {self.active_detection_fsm}")

    def cb_color(self, msg: String):
        color = msg.data.strip().upper()
        self.update_flags(color)

    def cb_event(self, msg: String):
        event = msg.data.strip().upper()
        self.update_flags(event)

    def update_flags(self, new_flag: str):
            if new_flag in ("NONE", "RIGHT_TURN", "LEFT_TURN", "ROUNDABOUT"):               
                return
            self.current_flags.clear()
            self.current_flags.add(new_flag)
            self.last_detection_time = time.time()

    def timer_step(self):
        # Timeout: if >2.0s since last detection, clear flags
        if time.time() - self.last_detection_time > 2.0:
            if self.current_flags:
                self.get_logger().info("⌛ Sin señales recientes. Restableciendo estado a ADVANCE.")
            self.current_flags.clear()

        action = 1.0  # Default
        label = "🟩 ADVANCE"

        if self.active_detection_fsm != 1:
            self.pub.publish(Float32(data=1.0))
            return

        if "RED" in self.current_flags or "STOP" in self.current_flags:
            action = 0.0
            label = "🟥 STOP"
        elif any(flag in self.current_flags for flag in ("YELLOW", "CHALANES", "GIVE_WAY")):
            action = 0.5
            label = "🟨 CAUTION"

        # self.get_logger().info(f"{label} ← {self.current_flags}")
        self.pub.publish(Float32(data=action))

def main(args=None):
    rclpy.init(args=args)
    node = DetectionFSM()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()