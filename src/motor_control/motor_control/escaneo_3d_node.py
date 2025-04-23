#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

import cv2
import numpy as np
import os
import time
import math

try:
    # Para la visualización 3D (requiere OpenCV con viz)
    from cv2 import viz
    HAS_CV_VIZ = True
except ImportError:
    HAS_CV_VIZ = False


class Escaneo3DNode(Node):
    def __init__(self):
        super().__init__('escaneo_3d_node')

        # --- Publisher para mover la base ---
        self.cmd_vel_pub = self.create_publisher(Twist, 'ik_cmd_vel', 10)

        # --- Parámetros de escaneo ---
        self.total_images = 5               # Número de capturas a realizar
        self.total_rotation = 2.0 * math.pi # Rotación total (2*pi = 360°)
        self.angular_speed = 0.5            # Vel. angular en rad/s
        self.wait_per_step = 2.0            # Espera tras rotar antes de tomar foto (s)
        self.capture_delay = 0.5            # Espera corta post-rotación antes de capturar
        self.radians_per_image = self.total_rotation / self.total_images
        self.images_captured = 0
        self.is_scanning = False

        # --- Para reconstrucción 3D ---
        # Calibración *hipotética* de la cámara (ajusta a tu setup)
        fx = 800.0
        fy = 800.0
        cx = 320.0
        cy = 240.0
        self.K = np.array([[fx, 0,   cx],
                           [0,  fy,  cy],
                           [0,   0,   1 ]], dtype=np.float32)

        # Directorio donde guardamos las fotos
        self.output_dir = 'capturas_scanner'
        os.makedirs(self.output_dir, exist_ok=True)

        self.img_index = 0
        self.image_paths = []  # Rutas de las imágenes capturadas

        # --- Captura de cámara local con OpenCV ---
        # Ajusta el índice si necesitas otra cámara
        self.cap = cv2.VideoCapture(0, cv2.CAP_ANY)
        if not self.cap.isOpened():
            self.get_logger().error("No se pudo abrir la cámara local.")
        else:
            self.get_logger().info("Cámara local detectada.")

        # --- Timer para iniciar el escaneo tras 3 seg ---
        self.start_timer = self.create_timer(3.0, self.start_scanning_once)

    def start_scanning_once(self):
        """Se llama 1 sola vez, inicia el proceso de escaneo."""
        if not self.is_scanning:
            self.is_scanning = True
            self.get_logger().info("Iniciando escaneo 3D...")
            self.scan_sequence()

    def scan_sequence(self):
        """Bucle principal de escaneo: rota, captura, repite."""
        if self.images_captured >= self.total_images:
            self.is_scanning = False
            self.get_logger().info("Escaneo finalizado. Iniciando reconstrucción 3D...")
            self.do_reconstruction_3d()
            return

        rotation_needed = self.radians_per_image
        time_to_rotate = abs(rotation_needed / self.angular_speed)

        # Publicar vel. angular
        twist_msg = Twist()
        twist_msg.angular.z = self.angular_speed if rotation_needed >= 0 else -self.angular_speed
        self.cmd_vel_pub.publish(twist_msg)

        self.get_logger().info(f"Rotando {rotation_needed:.2f} rad -> ~{time_to_rotate:.2f} s.")
        time.sleep(time_to_rotate)

        # Detener
        twist_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(twist_msg)

        # Espera corta para estabilizar
        time.sleep(self.capture_delay)

        # Capturar imagen de la cámara local
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Error al leer de la cámara local.")
        else:
            filename = os.path.join(self.output_dir, f"scan_{self.img_index:03d}.jpg")
            cv2.imwrite(filename, frame)
            self.image_paths.append(filename)
            self.get_logger().info(f"Imagen guardada: {filename}")
            self.img_index += 1

        # Espera un tiempo adicional antes del siguiente paso
        time.sleep(self.wait_per_step)

        self.images_captured += 1
        self.scan_sequence()

    def do_reconstruction_3d(self):
        """
        Reconstrucción 3D muy simplificada usando la primera y la última imagen.
        Guarda nube_resultante.ply y, si OpenCV tiene viz, la muestra.
        """
        if len(self.image_paths) < 2:
            self.get_logger().info("No hay suficientes imágenes para reconstruir.")
            return

        img1 = cv2.imread(self.image_paths[0])
        img2 = cv2.imread(self.image_paths[-1])
        if img1 is None or img2 is None:
            self.get_logger().error("No se pudo leer alguna imagen para la reconstrucción.")
            return

        # 1) Detector de características (ORB)
        detector = cv2.ORB_create(nfeatures=2000)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        kp1, desc1 = detector.detectAndCompute(gray1, None)
        kp2, desc2 = detector.detectAndCompute(gray2, None)

        if desc1 is None or desc2 is None:
            self.get_logger().error("No se encontraron descriptores en las imágenes.")
            return

        # 2) Emparejamiento
        matches_knn = bf.knnMatch(desc1, desc2, k=2)
        good_matches = []
        ratio_thresh = 0.75
        for m,n in matches_knn:
            if m.distance < ratio_thresh*n.distance:
                good_matches.append(m)

        self.get_logger().info(f"Matches buenos: {len(good_matches)}")

        if len(good_matches) < 8:
            self.get_logger().info("Muy pocos matches para estimar pose.")
            return

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        # 3) Hallar matriz esencial y recuperar R, t
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        inliers, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.K)

        self.get_logger().info(f"Inliers recoverPose: {inliers}")

        # 4) Triangular
        P1 = np.dot(self.K, np.hstack((np.eye(3), np.zeros((3,1)))))
        P2 = np.dot(self.K, np.hstack((R, t)))
        pts1_hom = pts1.T
        pts2_hom = pts2.T

        points_4d = cv2.triangulatePoints(P1, P2, pts1_hom, pts2_hom)
        points_3d = (points_4d[:3] / points_4d[3]).T  # Nx3

        # 5) Guardar nube .ply
        ply_path = os.path.join(self.output_dir, "nube_resultante.ply")
        with open(ply_path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {points_3d.shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("end_header\n")
            for p in points_3d:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")

        self.get_logger().info(f"Nube de puntos guardada en: {ply_path}")

        # 6) Visualizar con cv2.viz (si está disponible)
        if HAS_CV_VIZ:
            self.visualize_cloud_opencv(points_3d)
        else:
            self.get_logger().info("OpenCV se compiló sin módulo 'viz'. Usa MeshLab/CloudCompare para ver la nube.")

    def visualize_cloud_opencv(self, points_3d):
        """
        Visualiza la nube de puntos en una ventana 3D con el módulo cv2.viz
        """
        self.get_logger().info("Abriendo visualización 3D con cv2.viz ...")
        viz3d = viz.Viz3d("Nube 3D")
        viz3d.showWidget("ejes", viz.WCoordinateSystem())

        # Convertir a tipo float32 para WCloud
        pts_32 = points_3d.astype(np.float32)
        cloud_widget = viz.WCloud(pts_32, color=(255,255,255))
        viz3d.showWidget("nube", cloud_widget)

        viz3d.spin()


def main(args=None):
    rclpy.init(args=args)
    node = Escaneo3DNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()  # Cerrar la cámara
        rclpy.shutdown()


if __name__ == '__main__':
    main()
