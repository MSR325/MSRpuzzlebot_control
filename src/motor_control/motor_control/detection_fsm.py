#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from enum import Enum
from std_msgs.msg import String, Float32, Int16
from vision_msgs.msg import Detection2DArray
import time

class MovementState(Enum):
    STOP    = 0
    CAUTION = 1
    ADVANCE = 2

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

        # Estado inicial
        self.state          = MovementState.STOP
        self.evento_actual  = TrafficEvent.FORWARD
        self.active_detection_fsm = 1

        # Timestamp para watchdog
        self.last_event_time = time.time() 

        # Publicador y subscripciones
        self.fsm_action_pub = self.create_publisher(Float32, '/fsm_action', 10)
        self.create_subscription(String, '/color_flag',
                                 self.color_flag_callback, 10)
        self.create_subscription(String, '/fsm_event',
                                 self.fsm_event_callback, 10)
        # Opcional: suscripción directa a detecciones
        self.create_subscription(Detection2DArray, '/detections',
                                 self.detections_callback, 10)
        self.create_subscription(Int16, '/detection_fsm_enable',
                                 self.activate_detection_fsm_callback, 10)

        # Timer: publica actuate() y chequea watchdog cada 0.1 s
        self.create_timer(0.1, self.timer_callback)

        self.get_logger().info("FSM v4 iniciada (esperando eventos)...")

    # Callbacks de entrada
    def activate_detection_fsm_callback(self, msg: Int16):
        self.active_detection_fsm = msg.data
        self.get_logger().info(f"FSM activada={self.active_detection_fsm}")

    def color_flag_callback(self, msg: String):
        self.last_event_time = time.time()
        color = msg.data.strip().lower()
        mapping = {
            "red": TrafficEvent.COLOR_RED,
            "yellow": TrafficEvent.COLOR_YELLOW,
            "green": TrafficEvent.COLOR_GREEN
        }
        if color in mapping:
            self.actualizar_evento(mapping[color])

    def fsm_event_callback(self, msg: String):
        nombre = msg.data.strip().upper()
        if nombre in ("RED", "YELLOW", "GREEN"):
            return
        try:
            nuevo = TrafficEvent[nombre]
        except KeyError:
            self.get_logger().warn(f"Evento no mapeado: '{nombre}'")
            return
        self.last_event_time = time.time()
        self.actualizar_evento(nuevo)

    def detections_callback(self, msg: Detection2DArray):
        # Si no hay detecciones -> no actualizamos timestamp
        if not msg.detections:
            return
        self.last_event_time = time.time()
        # (Aquí podrías procesar msg.detections si lo necesitas)

    # Lógica de prioridad y transición
    def actualizar_evento(self, nuevo_evento: TrafficEvent):
        if nuevo_evento.value >= self.evento_actual.value:
            self.evento_actual = nuevo_evento
            self.transition(nuevo_evento)

    def transition(self, evento: TrafficEvent):
        if evento in (TrafficEvent.COLOR_RED, TrafficEvent.STOP):
            self.state = MovementState.STOP
        elif evento in (
            TrafficEvent.COLOR_YELLOW,
            TrafficEvent.GIVE_WAY,
            TrafficEvent.CHALANES,
        ):
            self.state = MovementState.CAUTION
        else:  # GREEN, FORWARD, etc.
            self.state = MovementState.ADVANCE
        # self.actuate()

    # Actuador
    def actuate(self):
        if self.active_detection_fsm != 1:
            out = 1.0
            etiqueta = "🟢 FSM OFF"
        else:
            if self.state == MovementState.STOP:
                out = 0.0; etiqueta = "🟥 STOP"
            elif self.state == MovementState.CAUTION:
                out = 0.5; etiqueta = "🟨 CAUTION"
            else:
                out = 1.0; etiqueta = "🟩 ADVANCE"
            self.get_logger().info(f"{etiqueta} ← Dominante: {self.evento_actual.name}")

        msg = Float32(data=out)
        self.fsm_action_pub.publish(msg)

    # Timer para publicación periódica y watchdog
    def timer_callback(self):
        ahora = time.time()
        # Si no hubo eventos en los últimos 0.2 s → default FORWARD
        if ahora - self.last_event_time > 2.0:
            self.evento_actual = TrafficEvent.FORWARD
            self.state = MovementState.ADVANCE
        self.actuate()

def main(args=None):
    rclpy.init(args=args)
    node = DetectionFSM()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
