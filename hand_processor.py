# hand_processor.py
# Clase para procesar landmarks de la mano con MediaPipe.
# Extrae características como ángulos, distancias, dirección y rotación.

import math
import numpy as np
import os
from types import SimpleNamespace
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import RunningMode, HandLandmarker, HandLandmarkerOptions

# Umbrales para determinar si un dedo está extendido
EXTENDED_ANGLE_THRESH = 150
CURLED_ANGLE_THRESH = 110

class HandProcessor:
    """
    Gestiona el modelo MediaPipe HandLandmarker y realiza cálculos geométricos
    sobre los puntos (landmarks) detectados en la mano.
    """
    def __init__(self, model_path, num_hands=1, min_conf=0.7):
        self.model_path = model_path
        self.num_hands = num_hands
        self.min_conf = min_conf
        self.detector = self._init_detector()
        self.last_result = None
        self._prev_smoothed = {}
        self.smooth_alpha = 0.7

    def _init_detector(self):
        """Inicializa el detector de MediaPipe en modo LIVE_STREAM."""
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=RunningMode.LIVE_STREAM,
            result_callback=self._callback,
            num_hands=self.num_hands,
            min_hand_detection_confidence=self.min_conf,
            min_tracking_confidence=self.min_conf
        )
        return HandLandmarker.create_from_options(options)

    def _callback(self, result, output_image, timestamp_ms):
        """Callback asíncrono donde MediaPipe entrega los resultados."""
        self.last_result = result

    def reset(self):
        """Limpia los resultados previos y el suavizado."""
        self.last_result = None
        self._prev_smoothed = {}

    def set_num_hands(self, num_hands):
        """Cambia dinámicamente el número máximo de manos a detectar."""
        if self.num_hands != num_hands:
            self.num_hands = num_hands
            if hasattr(self, "detector") and self.detector:
                try:
                    self.detector.close()
                except Exception as e:
                    print(f"Error al cerrar el detector anterior: {e}")
            self.detector = self._init_detector()
            self.reset()

    def detect(self, frame_bgr, timestamp_ms):
        """Envía un frame para detección asíncrona."""
        rgb = cv2_to_mp_rgb(frame_bgr)
        self.detector.detect_async(rgb, timestamp_ms)

    def get_hand_landmarks(self, hand_idx=0):
        """Obtiene los landmarks suavizados de una mano específica."""
        if not self.last_result or not self.last_result.hand_landmarks:
            return None
        if hand_idx >= len(self.last_result.hand_landmarks):
            return None
        
        lands = self.last_result.hand_landmarks[hand_idx]
        return self._smooth_landmarks(hand_idx, lands)

    def _smooth_landmarks(self, hand_index, lands):
        """Aplica un filtro de suavizado para reducir el ruido (jitter)."""
        coords = [SimpleNamespace(x=ld.x, y=ld.y, z=ld.z) for ld in lands]
        prev = self._prev_smoothed.get(hand_index)
        if prev is None:
            self._prev_smoothed[hand_index] = coords
            return coords
        
        alpha = self.smooth_alpha
        out = []
        for p, c in zip(prev, coords):
            nx = alpha * p.x + (1 - alpha) * c.x
            ny = alpha * p.y + (1 - alpha) * c.y
            nz = alpha * p.z + (1 - alpha) * c.z
            out.append(SimpleNamespace(x=nx, y=ny, z=nz))
        self._prev_smoothed[hand_index] = out
        return out

    # --- Cálculos Geométricos ---

    @staticmethod
    def angle_pts(a, b, c):
        """Calcula el ángulo en el punto 'b' formado por los puntos a, b, c."""
        ba = np.array([a.x - b.x, a.y - b.y])
        bc = np.array([c.x - b.x, c.y - b.y])
        dot = np.dot(ba, bc)
        mag = np.linalg.norm(ba) * np.linalg.norm(bc)
        if mag == 0: return 0.0
        return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))

    @staticmethod
    def distance(a, b):
        """Distancia euclidiana entre dos puntos."""
        return math.hypot(a.x - b.x, a.y - b.y)

    def get_palm_size(self, lands):
        """Usa la distancia entre la muñeca (0) y la base del dedo medio (9) como referencia de tamaño."""
        return self.distance(lands[0], lands[9])

    def get_hand_direction(self, lands):
        """
        Determina la dirección de la mano (Arriba, Abajo, Izquierda, Derecha).
        Retorna tanto el nombre como el vector unitario.
        
        Explicación para estudiantes: 
        Calculamos la diferencia entre la muñeca (punto 0) y el nudillo del dedo medio (punto 9).
        Esto nos da un vector que apunta 'hacia adelante' de la palma.
        """
        wrist = lands[0]
        middle_mcp = lands[9]
        dx = middle_mcp.x - wrist.x
        dy = middle_mcp.y - wrist.y
        
        mag = math.hypot(dx, dy)
        vector = (dx/mag, dy/mag) if mag > 0 else (0, 0)
        
        if abs(dx) > abs(dy):
            label = "Derecha" if dx > 0 else "Izquierda"
        else:
            label = "Abajo" if dy > 0 else "Arriba"
            
        return {"label": label, "vector": vector}

    def get_hand_rotation(self, lands):
        """
        Calcula la rotación de la mano en el plano 2D (en grados).
        0 grados es hacia arriba.
        """
        wrist = lands[0]
        middle_mcp = lands[9]
        angle = math.degrees(math.atan2(middle_mcp.y - wrist.y, middle_mcp.x - wrist.x))
        # Ajustar para que "Arriba" sea 0 o 90 dependiendo de convención.
        # atan2 retorna -180 a 180.
        return angle

    def get_finger_states(self, lands):
        """Determina si cada dedo está extendido o no."""
        return {
            "thumb": self.distance(lands[4], lands[17]) > self.get_palm_size(lands) * 0.9, # Simplificado para pulgar
            "index": self.angle_pts(lands[5], lands[6], lands[8]) > EXTENDED_ANGLE_THRESH,
            "middle": self.angle_pts(lands[9], lands[10], lands[12]) > EXTENDED_ANGLE_THRESH,
            "ring": self.angle_pts(lands[13], lands[14], lands[16]) > EXTENDED_ANGLE_THRESH,
            "pinky": self.angle_pts(lands[17], lands[18], lands[20]) > EXTENDED_ANGLE_THRESH,
        }

    def get_finger_curls(self, lands):
        """Retorna el ángulo de curvatura de cada dedo (en la articulación PIP)."""
        return {
            "index": self.angle_pts(lands[5], lands[6], lands[8]),
            "middle": self.angle_pts(lands[9], lands[10], lands[12]),
            "ring": self.angle_pts(lands[13], lands[14], lands[16]),
            "pinky": self.angle_pts(lands[17], lands[18], lands[20]),
        }

def cv2_to_mp_rgb(frame_bgr):
    """Convierte un frame de OpenCV (BGR) a un objeto Image de MediaPipe (RGB)."""
    import cv2
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image(image_format=ImageFormat.SRGB, data=rgb)
