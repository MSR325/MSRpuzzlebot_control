#!/usr/bin/env python3
# ───────────────────────────────────────────────────────────────
#  CrossroadFSM  |  Autor: ChatGPT-o3 + Iván  |  Jun-2025
#  Lógica: DETECTAR CRUCE → STOP → ESPERAR SEÑAL → EJECUTAR
#          FORWARD  : 4 s lineal
#          LEFT/RIGHT : 2 s lineal  + 4 s lineal+angular
# ───────────────────────────────────────────────────────────────
import rclpy, time
from enum import Enum
from rclpy.node import Node
from std_msgs.msg import String, Int16, Int32MultiArray
from geometry_msgs.msg import Twist

LINEAR_VEL  = 0.15      # m/s
ANGULAR_VEL = 0.526     # rad/s (magnitud fija +/-)

class State(Enum):
    FOLLOW_LINE = 0
    WAIT_EVENT  = 1
    EXECUTE     = 2

class CrossroadFSM(Node):
    def __init__(self):
        super().__init__('crossroad_fsm')

        # ── Publishers ──────────────────────────────────────────
        self.pub_enable  = self.create_publisher(Int16, '/line_follow_enable', 10)
        self.pub_cmd_vel = self.create_publisher(Twist,  '/line_cmd_vel',      10)

        # ── Subscribers ─────────────────────────────────────────
        self.create_subscription(Int32MultiArray, '/crossroad_detected',
                                 self.cb_crossroad, 10)
        self.create_subscription(String, '/fsm_event',
                                 self.cb_event,      10)

        # ── Timer (20 Hz) ───────────────────────────────────────
        self.create_timer(0.05, self.timer_step)

        # ── Estado interno ──────────────────────────────────────
        self.state          = State.FOLLOW_LINE
        self.last_event     = None
        self.exec_start_t   = 0.0
        self.crossroad_flag = False

        self._send_enable(0)
        self.get_logger().info("🚦 Crossroad FSM inicializada (20 Hz)")

    # ─────────────── Callbacks ────────────────────────────────
    def cb_crossroad(self, msg: Int32MultiArray):
        self.crossroad_flag = any(msg.data)   # flanco de cruce

    def cb_event(self, msg: String):
        ev = msg.data.strip().upper()
        if ev in ("FORWARD", "LEFT_TURN", "RIGHT_TURN") and self.state is State.WAIT_EVENT:
            self.last_event = ev
            self.get_logger().info(f"📣 Señal recibida: {ev}")

    # ─────────────── Bucle principal ──────────────────────────
    def timer_step(self):
        now = time.time()

        # ── FOLLOW_LINE ────────────────────────────────────────
        if self.state is State.FOLLOW_LINE:
            if self.crossroad_flag:
                self._send_enable(0)          # pausa seguidor
                self._send_twist(0.0, 0.0)    # stop inmediato
                self.last_event = None
                self.state = State.WAIT_EVENT
                self.crossroad_flag = False
                self.get_logger().info("✋ Crucero detectado → WAIT_EVENT")
            return

        # ── WAIT_EVENT ─────────────────────────────────────────
        if self.state is State.WAIT_EVENT:
            if self.last_event is not None:
                self.exec_start_t = now
                self.state = State.EXECUTE
                self.get_logger().info(f"▶ Ejecutando maniobra: {self.last_event}")
            return

        # ── EXECUTE ────────────────────────────────────────────
        if self.state is State.EXECUTE:
            dt = now - self.exec_start_t

            # Maniobra para FORWARD
            if self.last_event == "FORWARD":
                if dt <= 4.0:
                    self._send_twist(LINEAR_VEL, 0.0)
                else:
                    self._finish_execute()
                return

            # Maniobra para LEFT/RIGHT
            if self.last_event in ("LEFT_TURN", "RIGHT_TURN"):
                if dt <= 1.0:
                    # Fase 1: impulso recto
                    self._send_twist(LINEAR_VEL, 0.0)
                elif dt <= 6.0:
                    # Fase 2: avance + giro
                    ang =  ANGULAR_VEL if self.last_event == "LEFT_TURN"  else -ANGULAR_VEL
                    self._send_twist(LINEAR_VEL, ang)
                else:
                    self._finish_execute()
                return

    # ─────────────── Utilidades ───────────────────────────────
    def _finish_execute(self):
        # Arranque breve y re-habilita follower
        self._send_twist(LINEAR_VEL, 0.0)
        self._send_enable(1)
        self.last_event = None
        self.state = State.FOLLOW_LINE
        self.get_logger().info("✅ Maniobra completada → FOLLOW_LINE")

    def _send_enable(self, flag: int):
        self.pub_enable.publish(Int16(data=flag))

    def _send_twist(self, lin: float, ang: float):
        msg = Twist()
        msg.linear.x  = lin
        msg.angular.z = ang
        self.pub_cmd_vel.publish(msg)

# ────────────────────────── main ──────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = CrossroadFSM()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
