#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from rcl_interfaces.srv import GetParameters, SetParameters
from rcl_interfaces.msg import SetParametersResult
from rclpy.parameter import Parameter as RclpyParameter
import json, math, sys

class PathGenerator(Node):
    def __init__(self):
        super().__init__('path_generator')

        # Declarar parámetros
        self.declare_parameter('sample_time', 0.018)
        self.declare_parameter('waypoints_json', '[]')

        # Inicializar variables
        self.sample_time = self.get_parameter('sample_time').get_parameter_value().double_value
        self.waypoints = self._parse_waypoints(self.get_parameter('waypoints_json').get_parameter_value().string_value)
        self.current_pose = None
        self.current_index = 0
        self.first_timer_run = True

        # Subscripciones y servicios
        self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.add_on_set_parameters_callback(self.dynamic_reconfigure_callback)

        self.get_param_cli = self.create_client(GetParameters, '/inverse_kinematics/get_parameters')
        self.set_param_cli = self.create_client(SetParameters, '/inverse_kinematics/set_parameters')

        self.arrival_tolerance = 0.1

        # Esperar conexión a servicios
        if not self.get_param_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("No pude conectar al servicio get_parameters de inverse_kinematics")
        else:
            self.arrival_tolerance = self._fetch_arrival_tolerance(self.arrival_tolerance)

        if not self.set_param_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("No pude conectar al servicio set_parameters de inverse_kinematics")
            sys.exit(1)

        # Iniciar primer waypoint si hay
        if self.waypoints:
            self._set_next_waypoint()

        # Timer principal
        self.create_timer(self.sample_time, self.timer_callback)

        # Mensaje inicial
        self.get_logger().info("PathGenerator inicializado y listo. ✅")

    def odom_callback(self, msg: Odometry):
        self.current_pose = msg.pose.pose.position

    def timer_callback(self):
        if self.first_timer_run:
            self.get_logger().info("Esperando waypoints y odometría... 🚀")
            self.first_timer_run = False

        if self.current_pose is None or self.current_index >= len(self.waypoints):
            return

        wp = self.waypoints[self.current_index]
        dx = wp['x'] - self.current_pose.x
        dy = wp['y'] - self.current_pose.y
        if math.hypot(dx, dy) < self.arrival_tolerance:
            self.get_logger().info(f"Llegado a waypoint {self.current_index}")
            self.current_index += 1
            if self.current_index < len(self.waypoints):
                self._set_next_waypoint()
            else:
                self.get_logger().info("Todos los waypoints completados 🚩")

    def dynamic_reconfigure_callback(self, params):
        results = []
        for param in params:
            if param.name == 'waypoints_json':
                try:
                    self.waypoints = self._parse_waypoints(param.value)
                    self.current_index = 0
                    if self.waypoints:
                        self.get_logger().info("Waypoints recargados dinámicamente ✅")
                        self._set_next_waypoint()
                    else:
                        self.get_logger().warn("Waypoints vacíos tras recarga 🛑")
                    results.append(SetParametersResult(successful=True))
                except Exception as e:
                    self.get_logger().error(f"Error procesando waypoints_json: {e}")
                    results.append(SetParametersResult(successful=False))

            elif param.name == 'sample_time':
                try:
                    self.sample_time = float(param.value)
                    results.append(SetParametersResult(successful=True))
                except Exception as e:
                    self.get_logger().error(f"sample_time inválido: {e}")
                    results.append(SetParametersResult(successful=False))
            else:
                results.append(SetParametersResult(successful=False))
        return results

    def _parse_waypoints(self, json_input):
        if isinstance(json_input, str):
            try:
                return json.loads(json_input)
            except json.JSONDecodeError:
                self.get_logger().error("JSON inválido en waypoints_json. Iniciando con arreglo vacío 🛑")
                return []
        elif isinstance(json_input, list):
            return json_input
        else:
            self.get_logger().error("waypoints_json no es ni string ni lista. Ignorando.")
            return []

    def _fetch_arrival_tolerance(self, default):
        req = GetParameters.Request()
        req.names = ['position_threshold']
        future = self.get_param_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() and future.result().values:
            val = future.result().values[0].double_value
            self.get_logger().info(f"Usando position_threshold={val:.3f} como arrival_tolerance")
            return val
        self.get_logger().warn(f"No pude leer position_threshold; usando {default}")
        return default

    def _set_next_waypoint(self):
        if self.current_index >= len(self.waypoints):
            self.get_logger().warn("Índice fuera de rango al intentar setear waypoint")
            return

        wp = self.waypoints[self.current_index]
        params = [
            RclpyParameter('desired_x', RclpyParameter.Type.DOUBLE, wp['x']).to_parameter_msg(),
            RclpyParameter('desired_y', RclpyParameter.Type.DOUBLE, wp['y']).to_parameter_msg(),
        ]
        req = SetParameters.Request()
        req.parameters = params
        future = self.set_param_cli.call_async(req)

        def on_result(fut):
            try:
                if fut.result():
                    self.get_logger().info(f"Waypoint {self.current_index} enviado → x={wp['x']}, y={wp['y']}")
                else:
                    self.get_logger().error("Fallo al setear parámetros en inverse_kinematics")
            except Exception as e:
                self.get_logger().error(f"Excepción al setear parámetros: {e}")

        future.add_done_callback(on_result)


def main(args=None):
    rclpy.init(args=args)
    node = PathGenerator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
