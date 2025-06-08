#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from enum import Enum
from std_msgs.msg import String, Float32, Int16
from vision_msgs.msg import Detection2DArray

# --- DEFINICIÓN DE ESTADOS DE MOVIMIENTO ---
class MovementState(Enum):
    STOP    = 0
    CAUTION = 1
    ADVANCE = 2

# --- DEFINICIÓN DE EVENTOS CON PRIORIDAD ---
class TrafficEvent(Enum):
    ROUNDABOUT    = 11.0
    COLOR_RED     = 10.0
    COLOR_YELLOW  = 10.0
    COLOR_GREEN   = 10.0
    STOP          = 9.0
    GIVE_WAY      = 7.0
    RIGHT_TURN    = 5.0
    LEFT_TURN     = 5.0
    CHALANES      = 3.0
    FORWARD       = 2.0

class DetectionFSM(Node):
    def __init__(self):
        super().__init__('detection_fsm_v4')

        # Estado y prioridad actual
        self.state          = MovementState.STOP
        self.evento_actual  = TrafficEvent.FORWARD

        # Multiplicador de salida
        self.color_flag_multiplier = 0.0

        # FSM activable/desactivable
        self.active_detection_fsm = 0

        # Publishers & Subscribers
        self.fsm_action_pub = self.create_publisher(Float32, '/fsm_action', 10)

        # 1) Semáforo: color_flag ("green"/"yellow"/"red")
        self.create_subscription(
            String, '/color_flag', self.color_flag_callback, 10
        )
        # 2) Señales varias: evento unificado desde el mapeador YOLO
        self.create_subscription(
            String, '/fsm_event', self.fsm_event_callback, 10
        )
        # 3) Habilitación de la FSM
        self.create_subscription(
            Int16, '/detection_fsm_enable', self.activate_detection_fsm_callback, 10
        )

        self.get_logger().info("FSM v4 iniciada (esperando eventos)...")

    # ------------------------------
    # Callbacks de entrada
    # ------------------------------
    def activate_detection_fsm_callback(self, msg: Int16):
        self.active_detection_fsm = msg.data
        self.get_logger().info(f"FSM activada={self.active_detection_fsm}")

    def color_flag_callback(self, msg: String):
        color = msg.data.strip().lower()
        if color == "red":
            self.actualizar_evento(TrafficEvent.COLOR_RED)
        elif color == "yellow":
            self.actualizar_evento(TrafficEvent.COLOR_YELLOW)
        elif color == "green":
            self.actualizar_evento(TrafficEvent.COLOR_GREEN)

    def fsm_event_callback(self, msg: String):
        nombre = msg.data.strip().upper()
        # Ignora colores aquí; ya los maneja color_flag_callback
        if nombre in ("RED", "YELLOW", "GREEN"):
            return

        try:
            nuevo = TrafficEvent[nombre]
        except KeyError:
            self.get_logger().warn(f"Evento no mapeado: '{nombre}'")
            return

        self.actualizar_evento(nuevo)

    # ------------------------------
    # Lógica de prioridad y transición
    # ------------------------------
    def actualizar_evento(self, nuevo_evento: TrafficEvent):
        if nuevo_evento.value >= self.evento_actual.value:
            self.evento_actual = nuevo_evento
            self.transition(nuevo_evento)

    def transition(self, evento: TrafficEvent):
        # Mapea evento → estado
        if evento in (TrafficEvent.COLOR_RED, TrafficEvent.STOP):
            self.state = MovementState.STOP

        elif evento in (
            TrafficEvent.COLOR_YELLOW,
            TrafficEvent.GIVE_WAY,
            TrafficEvent.CHALANES,
            #TrafficEvent.RIGHT_TURN,
            #TrafficEvent.LEFT_TURN,
            #TrafficEvent.ROUNDABOUT
        ):
            self.state = MovementState.CAUTION

        elif evento in (TrafficEvent.COLOR_GREEN, TrafficEvent.FORWARD):
            self.state = MovementState.ADVANCE

        # Publica la acción tras cada transición
        self.actuate()

    # ------------------------------
    # Actuador que envía el multiplicador
    # ------------------------------
    def actuate(self):
        # Si la FSM está desactivada, siempre velocidad plena
        if self.active_detection_fsm != 1:
            out = 1.0

        else:
            if self.state == MovementState.STOP:
                out = 0.0
                etiqueta = "🟥 STOP"
            elif self.state == MovementState.CAUTION:
                out = 0.5
                etiqueta = "🟨 CAUTION"
            else:  # ADVANCE
                out = 1.0
                etiqueta = "🟩 ADVANCE"

            self.get_logger().info(f"{etiqueta} ← Dominante: {self.evento_actual.name}")

        # Publica el multiplicador
        msg = Float32()
        msg.data = out
        self.fsm_action_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = DetectionFSM()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
