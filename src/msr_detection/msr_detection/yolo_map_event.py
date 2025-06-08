#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String

class YOLOEventRelay(Node):
    """
    Nodo que recibe detecciones YOLO (Detection2DArray) y repubblica
    cada evento crudo con el nombre correcto en '/fsm_event'.
    """
    def __init__(self):
        super().__init__('yolo_event_relay')

        # Diccionario original de clases YOLOv10
        self.classes = {
            "0": "forward",
            "1": "giveaway",
            "2": "right",
            "3": "roundabout",
            "4": "traffic light",
            "5": "stop",
            "6": "construction",
            "7": "left"
        }

        # Mapeo de nombre de clase a evento FSM (nombres corregidos)
        self.class_to_event = {
            "forward":       "FORWARD",
            "giveaway":      "GIVE_WAY",
            "right":         "RIGHT_TURN",
            "roundabout":    "ROUNDABOUT",
            "traffic light": "TRAFFIC_LIGHT",
            "stop":          "STOP",
            "construction":  "CHALANES",
            "left":          "LEFT_TURN"
        }

        # Umbral mínimo de confianza para reenviar
        self.min_conf = 0.6

        # Suscripción a detecciones YOLO
        self.sub = self.create_subscription(
            Detection2DArray,
            '/yolo/detections',
            self.cb_relay,
            10
        )
        # Publicador único de eventos para la FSM
        self.pub = self.create_publisher(String, '/fsm_event', 10)

        self.get_logger().info("yolo_event_relay iniciado con mapeo de nombres corregidos.")

    def cb_relay(self, msg: Detection2DArray):
        for det in msg.detections:
            if not det.results:
                continue
            res = det.results[0]

            # Filtra por confianza
            if res.hypothesis.score < self.min_conf:
                continue

            # Obtiene el nombre de la clase y el evento corregido
            clase_name = self.classes.get(res.hypothesis.class_id)
            evento = self.class_to_event.get(clase_name)

            if evento:
                self.pub.publish(String(data=evento))
                self.get_logger().debug(
                    f"Reenviado evento: {evento} (clase '{clase_name}', conf={res.hypothesis.score:.2f})"
                )

def main(args=None):
    rclpy.init(args=args)
    node = YOLOEventRelay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
