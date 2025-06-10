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
        valid_detections = []
        
        for det in msg.detections:
            if not det.results:
                continue
            res = det.results[0]

            # confidence filtering
            if res.hypothesis.score < self.min_conf:
                continue

            # mapping
            clase_name = self.classes.get(res.hypothesis.class_id)
            evento = self.class_to_event.get(clase_name)

            if evento:
                #calculate area
                bbox = det.bbox
                area = bbox.size_x * bbox.size_y
                
                valid_detections.append({
                    'detection': det,
                    'result': res,
                    'clase_name': clase_name,
                    'evento': evento,
                    'area': area
                })
        
        # publish on max area detection
        if valid_detections:
            largest_detection = max(valid_detections, key=lambda x: x['area'])
            
            self.pub.publish(String(data=largest_detection['evento']))
            self.get_logger().debug(
                f"evento mapeado xdxdxd: {largest_detection['evento']} "
                f"(clase '{largest_detection['clase_name']}', "
                f"conf={largest_detection['result'].hypothesis.score:.2f}, "
                f"area={largest_detection['area']:.2f})"
            )

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
