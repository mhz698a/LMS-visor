# camera_engine.py
# Clase para gestionar la cámara OAK-D y el flujo de frames usando DepthAI.
# Diseñado para estudiantes: maneja la inicialización y obtención de imágenes.

import depthai as dai
import cv2
import time

class CameraEngine:
    """
    Esta clase se encarga de gestionar la captura de video, priorizando la cámara OAK-D
    pero con soporte para webcam estándar como respaldo (fallback).
    """
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps

        # Estado
        self.device = None
        self.q_rgb = None
        self.cap_webcam = None
        self.is_oak = False
        self.is_running = False

    def _setup_oak_pipeline(self):
        pipeline = dai.Pipeline()
        # Crear nodo de cámara de color
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setPreviewSize(self.width, self.height)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setFps(self.fps)

        # Crear salida XLink
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)
        return pipeline

    def start(self, mode="OAK-D"):
        """
        Inicia la cámara seleccionada.
        mode: "OAK-D", "Webcam 0", "Webcam 1", etc. o "Webcam"
        """
        if self.is_running: return True

        if mode == "OAK-D":
            try:
                pipeline = self._setup_oak_pipeline()
                self.device = dai.Device(pipeline)
                self.q_rgb = self.device.getOutputQueue("rgb", maxSize=4, blocking=False)
                self.is_oak = True
                self.is_running = True
                print("Cámara OAK-D iniciada correctamente.")
                return True
            except Exception as e:
                print(f"No se pudo iniciar OAK-D: {e}")
                return False
        else:
            # Extraer el índice de la cámara si viene especificado
            cam_idx = 0
            if " " in mode:
                try:
                    cam_idx = int(mode.split()[-1])
                except ValueError:
                    cam_idx = 0

            self.cap_webcam = cv2.VideoCapture(cam_idx)
            if self.cap_webcam.isOpened():
                self.is_oak = False
                self.is_running = True
                print(f"Webcam {cam_idx} iniciada correctamente.")
                return True
            else:
                print(f"Error: No se pudo abrir la webcam {cam_idx}.")
                self.cap_webcam = None
                return False

    def get_frame(self):
        """Obtiene el frame de la fuente activa."""
        if not self.is_running: return None

        if self.is_oak:
            try:
                in_rgb = self.q_rgb.tryGet()
                if in_rgb is not None:
                    frame = in_rgb.getCvFrame()
                    return cv2.flip(frame, 1)
            except Exception:
                return None
        else:
            if self.cap_webcam:
                ret, frame = self.cap_webcam.read()
                if ret:
                    # Redimensionar para consistencia
                    frame = cv2.resize(frame, (self.width, self.height))
                    return cv2.flip(frame, 1)
        return None

    def stop(self):
        """Cierra cualquier fuente de video activa."""
        if self.device:
            self.device.close()
            self.device = None
        if self.cap_webcam:
            self.cap_webcam.release()
            self.cap_webcam = None
        self.is_running = False
        self.is_oak = False
        print("Cámara detenida.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
