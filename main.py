#!py -3.10
import sys
import os
import time
import cv2
import numpy as np
import threading
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QComboBox,
                             QStatusBar, QFrame, QGroupBox, QTextEdit, QSlider,
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                             QTabWidget, QCheckBox)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal, QSize, QDateTime, QStandardPaths
from PyQt6.QtGui import QImage, QPixmap, QFont, QKeyEvent, QGuiApplication, QWheelEvent

# Importación de nuestros módulos personalizados
from camera_engine import CameraEngine
from hand_processor import HandProcessor
from gesture_logic import GestureLogic
from tracker import HandTracker
from recorder import GestureRecorder
from training.train_static import train
from training.train_motion import train_motion

class GestureGuideViewer(QGraphicsView):
    """Visor de imágenes con soporte para zoom y desplazamiento (pan)."""
    def __init__(self, image_path):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            print(f"Error: No se pudo cargar la imagen de guía en {image_path}")
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)

        # Configuración para desplazamiento y zoom
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(Qt.GlobalColor.black)

        self.zoom_factor = 1.15

    def wheelEvent(self, event: QWheelEvent):
        """Maneja el zoom con la rueda del ratón."""
        if event.angleDelta().y() > 0:
            self.scale(self.zoom_factor, self.zoom_factor)
        else:
            self.scale(1 / self.zoom_factor, 1 / self.zoom_factor)

    def reset_view(self):
        """Ajusta la imagen al tamaño del visor."""
        if not self.pixmap_item.pixmap().isNull():
            self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

class GestureGuidePanel(QWidget):
    """Panel lateral que contiene el visor de la guía de señas."""
    def __init__(self, image_path):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)

        self.viewer = GestureGuideViewer(image_path)
        self.layout.addWidget(self.viewer)

        self.btn_reset = QPushButton("Restablecer Zoom")
        self.btn_reset.clicked.connect(self.viewer.reset_view)
        self.layout.addWidget(self.btn_reset)

        self.setMaximumWidth(500)
        self.setMinimumWidth(300)

class LogWidget(QTextEdit):
    """Widget de logs con soporte para colores y fondo negro."""
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setStyleSheet("""
            background-color: black;
            color: white;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 12px;
            border: 1px solid #444;
        """)

    def append_log(self, message, mode="info"):
        """Añade un mensaje al log con el color correspondiente."""
        timestamp = QDateTime.currentDateTime().toString("hh:mm:ss")
        color = "white"
        if mode == "success": color = "#00FF00" # Verde
        elif mode == "warning": color = "#FFFF00" # Amarillo
        elif mode == "error": color = "#FF0000" # Rojo

        html_msg = f"<span style='color: #888;'>[{timestamp}]</span> "
        html_msg += f"<span style='color: {color};'>{message}</span>"
        self.append(html_msg)
        # Auto-scroll al final
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

class TrainingThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, train_func):
        super().__init__()
        self.train_func = train_func

    def run(self):
        try:
            self.train_func(progress_callback=self.progress.emit)
            self.finished.emit(True, "¡Entrenamiento completado!")
        except Exception as e:
            self.finished.emit(False, f"Error: {e}")

class HandAppQT(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema de Señas LSM - PyQt6")
        self.setMinimumSize(1000, 700)

        # Inicialización de componentes (Lógica)
        self.model_path = "hand_landmarker.task"
        self.camera = CameraEngine()
        self.processor = HandProcessor(self.model_path, num_hands=1)
        self.logic = GestureLogic()
        self.tracker = HandTracker()
        self._init_ui() # Inicializar UI antes para tener el log_widget
        self.recorder = GestureRecorder(log_callback=self.log_widget.append_log)

        # Estado
        self.running_camera = False
        self.current_static_letter = "---"
        self.current_motion_letter = "---"
        self.recognition_source = "---"
        self.manual_letter = None
        self.target_motion_letter = None
        self.last_timestamp_ms = -1
        self.start_time_ns = time.perf_counter_ns()

        # Timer para el loop principal (aprox 30 FPS)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def _init_ui(self):
        # Widget principal y Layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # --- PANEL DE GUÍA DE SEÑAS (OCULTO POR DEFECTO) ---
        self.guide_panel = GestureGuidePanel("img/20260127_091127.jpg")
        self.guide_panel.setVisible(False)
        main_layout.addWidget(self.guide_panel, stretch=1)

        # --- PANEL IZQUIERDO: Video y Logs ---
        left_container = QVBoxLayout()

        # Área de Video
        self.video_label = QLabel("Cámara Desconectada")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white; border: 2px solid #333;")
        self.video_label.setMinimumSize(640, 480)
        left_container.addWidget(self.video_label, stretch=4)

        # Área de Logs
        self.log_widget = LogWidget()
        self.log_widget.setMaximumHeight(200)
        left_container.addWidget(QLabel("Logs del Sistema:"))
        left_container.addWidget(self.log_widget, stretch=1)

        # Status Bar inferior para mensajes rápidos
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Listo")

        main_layout.addLayout(left_container, stretch=3)

        # --- PANEL DERECHO: Controles ---
        sidebar = QVBoxLayout()

        # Grupo de Cámara
        cam_group = QGroupBox("Control de Cámara")
        cam_layout = QVBoxLayout()

        cam_layout.addWidget(QLabel("Seleccionar Fuente:"))
        self.combo_cam_source = QComboBox()
        self.combo_cam_source.addItems(["OAK-D", "Webcam"])
        cam_layout.addWidget(self.combo_cam_source)

        self.btn_refresh_cams = QPushButton("Actualizar Cámaras")
        self.btn_refresh_cams.clicked.connect(self.refresh_cameras)
        cam_layout.addWidget(self.btn_refresh_cams)

        self.btn_cam = QPushButton("Conectar Cámara")
        self.btn_cam.setFixedHeight(40)
        self.btn_cam.clicked.connect(self.toggle_camera)
        cam_layout.addWidget(self.btn_cam)
        cam_group.setLayout(cam_layout)
        sidebar.addWidget(cam_group)

        # Grupo de Reconocimiento
        rec_group = QGroupBox("Reconocimiento")
        rec_layout = QVBoxLayout()

        self.lbl_letter = QLabel("Letra: ---")
        self.lbl_letter.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        self.lbl_letter.setStyleSheet("color: #00FF00;")
        rec_layout.addWidget(self.lbl_letter)

        self.lbl_source = QLabel("Origen: ---")
        rec_layout.addWidget(self.lbl_source)

        self.lbl_motion = QLabel("Movimiento: ---")
        self.lbl_motion.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.lbl_motion.setStyleSheet("color: #00FFFF;")
        rec_layout.addWidget(self.lbl_motion)

        rec_group.setLayout(rec_layout)
        sidebar.addWidget(rec_group)

        # Grupo de Grabación
        record_group = QGroupBox("Grabación / Datos")
        record_layout = QVBoxLayout()

        record_layout.addWidget(QLabel("Seleccionar Letra Manual:"))

        letter_nav_layout = QHBoxLayout()
        self.btn_prev_letter = QPushButton("<")
        self.btn_prev_letter.setFixedWidth(30)
        self.btn_prev_letter.clicked.connect(self.prev_letter)

        self.combo_letter = QComboBox()
        self.combo_letter.addItems(["NINGUNA"] + [chr(i) for i in range(ord('A'), ord('Z') + 1)] + [str(i) for i in range(11)])
        self.combo_letter.currentTextChanged.connect(self.on_letter_changed)

        self.btn_next_letter = QPushButton(">")
        self.btn_next_letter.setFixedWidth(30)
        self.btn_next_letter.clicked.connect(self.next_letter)

        letter_nav_layout.addWidget(self.btn_prev_letter)
        letter_nav_layout.addWidget(self.combo_letter)
        letter_nav_layout.addWidget(self.btn_next_letter)
        record_layout.addLayout(letter_nav_layout)

        self.btn_record_static = QPushButton("Grabar Estático (Enter)")
        self.btn_record_static.clicked.connect(self.record_static)
        record_layout.addWidget(self.btn_record_static)

        self.lbl_motion_target = QLabel("Movimiento Pendiente: Ninguno")
        record_layout.addWidget(self.lbl_motion_target)

        self.btn_record_motion = QPushButton("Grabar Movimiento (F12)")
        self.btn_record_motion.clicked.connect(self.record_motion)
        record_layout.addWidget(self.btn_record_motion)

        record_group.setLayout(record_layout)

        # Grupo de Entrenamiento
        train_group = QGroupBox("Modelo ML")
        train_layout = QVBoxLayout()
        self.btn_train = QPushButton("Entrenar Modelo (F11)")
        self.btn_train.clicked.connect(self.start_training)
        train_layout.addWidget(self.btn_train)

        self.btn_train_motion = QPushButton("Entrenar Movimiento")
        self.btn_train_motion.clicked.connect(self.start_motion_training)
        train_layout.addWidget(self.btn_train_motion)

        self.train_progress_lbl = QLabel("")
        self.train_progress_lbl.setWordWrap(True)
        train_layout.addWidget(self.train_progress_lbl)

        train_group.setLayout(train_layout)

        # Grupo de Configuración Visual
        visual_group = QGroupBox("Configuración Visual")
        visual_layout = QVBoxLayout()

        self.chk_two_hands = QCheckBox("Permitir detección de dos manos")
        self.chk_two_hands.setChecked(False)
        self.chk_two_hands.toggled.connect(self.on_two_hands_toggled)
        visual_layout.addWidget(self.chk_two_hands)

        visual_layout.addWidget(QLabel("Longitud de Estela:"))
        self.sld_trail_len = QSlider(Qt.Orientation.Horizontal)
        self.sld_trail_len.setRange(5, 100)
        self.sld_trail_len.setValue(self.tracker.max_len)
        self.sld_trail_len.valueChanged.connect(self.update_trail_len)
        visual_layout.addWidget(self.sld_trail_len)
        visual_group.setLayout(visual_layout)

        # Grupo de Utilidades
        util_group = QGroupBox("Utilidades")
        util_layout = QVBoxLayout()

        self.btn_toggle_guide = QPushButton("Mostrar/Ocultar Guía de señas")
        self.btn_toggle_guide.clicked.connect(self.toggle_guide)
        util_layout.addWidget(self.btn_toggle_guide)

        self.btn_screenshot = QPushButton("Capturar Pantalla (Full)")
        self.btn_screenshot.clicked.connect(self.take_full_screenshot)
        util_layout.addWidget(self.btn_screenshot)
        util_group.setLayout(util_layout)

        # --- CREAR PESTAÑAS (QTabWidget) ---
        self.tabs = QTabWidget()

        # Pestaña 1: Grabacion/Datos
        tab_recording = QWidget()
        rec_tab_layout = QVBoxLayout(tab_recording)
        rec_tab_layout.addWidget(record_group)
        rec_tab_layout.addStretch()
        self.tabs.addTab(tab_recording, "Grabacion/Datos")

        # Pestaña 2: Modelos ML
        tab_models = QWidget()
        models_tab_layout = QVBoxLayout(tab_models)
        models_tab_layout.addWidget(train_group)
        models_tab_layout.addStretch()
        self.tabs.addTab(tab_models, "Modelos ML")

        # Pestaña 3: Utilidades
        tab_utils = QWidget()
        utils_tab_layout = QVBoxLayout(tab_utils)
        utils_tab_layout.addWidget(visual_group)
        utils_tab_layout.addWidget(util_group)
        utils_tab_layout.addStretch()
        self.tabs.addTab(tab_utils, "Utilidades")

        # Añadir las pestañas al sidebar
        sidebar.addWidget(self.tabs)
        sidebar.addStretch()
        main_layout.addLayout(sidebar, stretch=1)

        # Escanear cámaras disponibles al arrancar
        self.refresh_cameras()

    def toggle_guide(self):
        """Muestra u oculta el panel lateral de la guía de señas."""
        is_visible = self.guide_panel.isVisible()
        self.guide_panel.setVisible(not is_visible)
        if not is_visible:
            self.status_bar.showMessage("Guía de señas mostrada")
            # Ajustar la imagen la primera vez que se muestra
            QTimer.singleShot(50, self.guide_panel.viewer.reset_view)
        else:
            self.status_bar.showMessage("Guía de señas oculta")

    def refresh_cameras(self):
        """Escanea e identifica las cámaras web disponibles en el sistema."""
        self.log_widget.append_log("Buscando cámaras conectadas...", "info")
        indices = []
        try:
            for i in range(5):
                cap = cv2.VideoCapture(i)
                if cap is not None:
                    if cap.isOpened():
                        indices.append(i)
                        cap.release()
        except Exception as e:
            self.log_widget.append_log(f"Error al escanear cámaras: {e}", "error")

        current_selection = self.combo_cam_source.currentText()

        self.combo_cam_source.clear()
        self.combo_cam_source.addItem("OAK-D")
        for idx in indices:
            self.combo_cam_source.addItem(f"Webcam {idx}")

        found_idx = self.combo_cam_source.findText(current_selection)
        if found_idx >= 0:
            self.combo_cam_source.setCurrentIndex(found_idx)
        else:
            self.combo_cam_source.setCurrentIndex(0)

        self.log_widget.append_log(f"Búsqueda finalizada. Cámaras detectadas: {len(indices)}", "success")

    def toggle_camera(self):
        if not self.running_camera:
            mode = self.combo_cam_source.currentText()
            if self.camera.start(mode=mode):
                self.running_camera = True
                self.btn_cam.setText("Desconectar Cámara")
                self.combo_cam_source.setEnabled(False)
                self.status_bar.showMessage(f"Cámara {mode} conectada")
                self.log_widget.append_log(f"Cámara {mode} conectada correctamente.", "success")
                # Resetear procesador para evitar estados antiguos
                self.processor.reset()
                self.tracker.clear_all()
            else:
                self.status_bar.showMessage("Error: No se pudo conectar la cámara")
        else:
            self.camera.stop()
            self.running_camera = False
            self.btn_cam.setText("Conectar Cámara")
            self.combo_cam_source.setEnabled(True)
            self.video_label.clear()
            self.video_label.setText("Cámara Desconectada")
            self.status_bar.showMessage("Cámara desconectada")
            self.log_widget.append_log("Cámara desconectada.", "warning")
            self.current_static_letter = "---"
            self.current_motion_letter = "---"
            self.recognition_source = "---"
            self.lbl_letter.setText("Letra: ---")
            self.lbl_source.setText("Origen: ---")
            self.lbl_motion.setText("Movimiento: ---")

    def prev_letter(self):
        idx = self.combo_letter.currentIndex()
        count = self.combo_letter.count()
        new_idx = (idx - 1) % count
        self.combo_letter.setCurrentIndex(new_idx)

    def next_letter(self):
        idx = self.combo_letter.currentIndex()
        count = self.combo_letter.count()
        new_idx = (idx + 1) % count
        self.combo_letter.setCurrentIndex(new_idx)

    def on_letter_changed(self, text):
        if text == "NINGUNA":
            self.manual_letter = None
        else:
            self.manual_letter = text
        self.status_bar.showMessage(f"Letra manual: {self.manual_letter}")

    def update_frame(self):
        if not self.running_camera:
            return

        frame = self.camera.get_frame()
        if frame is None:
            return

        # Lógica de procesamiento
        curr_ms = (time.perf_counter_ns() - self.start_time_ns) // 1_000_000
        if curr_ms <= self.last_timestamp_ms:
            curr_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = curr_ms

        self.processor.detect(frame, curr_ms)

        # Obtener todas las manos detectadas
        detected_hands = []
        if self.processor.last_result and self.processor.last_result.hand_landmarks:
            for i in range(len(self.processor.last_result.hand_landmarks)):
                lands = self.processor.get_hand_landmarks(i)
                if lands:
                    detected_hands.append((i, lands))

        has_primary_hand = False

        # Dibujar landmarks de todas las manos detectadas y procesar
        for hand_idx, lands in detected_hands:
            self._draw_landmarks_cv(frame, lands)

            if hand_idx == 0:
                has_primary_hand = True
                props = self.logic.extract_properties(lands, self.processor)
                detected, source = self.logic.recognize_static(props, lands)
                self.current_static_letter = detected if detected else "---"
                self.recognition_source = source if source else "---"

                # Triggers de movimiento
                motion_target, fingers_to_track = self.logic.get_trigger_info(self.current_static_letter)
                if motion_target:
                    self.target_motion_letter = motion_target
                    self.tracker.set_active_fingers(fingers_to_track)
                    self.lbl_motion_target.setText(f"Pendiente (F12): {motion_target}")
                elif not self.recorder.recording:
                    self.target_motion_letter = None
                    self.tracker.set_active_fingers([])
                    self.lbl_motion_target.setText("Pendiente (F12): Ninguno")

                self.tracker.update(lands, frame.shape, hand_idx=0)

                # Reconocimiento de movimiento continuo
                if self.tracker.active_fingers:
                    m_detected, m_conf = self.logic.recognize_motion(self.tracker.histories)
                    if m_detected and m_conf > 0.7:
                        self.current_motion_letter = f"{m_detected} ({m_conf:.2f})"
                    else:
                        self.current_motion_letter = "---"
                else:
                    self.current_motion_letter = "---"

                if self.recorder.recording:
                    record_data = {
                        "letter": self.recorder.current_letter,
                        "landmarks": [{"x": l.x, "y": l.y, "z": l.z} for l in lands],
                        "direction": props["direction"],
                        "rotation": props["rotation"],
                        "tracked_fingers": self.tracker.active_fingers,
                        "props": props
                    }
                    self.recorder.add_frame(record_data)
            else:
                self.tracker.update(lands, frame.shape, hand_idx=hand_idx)

        if not has_primary_hand:
            self.current_static_letter = "---"
            self.current_motion_letter = "---"
            self.recognition_source = "---"

        self.recorder.update()
        self.tracker.draw_trails(frame)

        # Actualizar UI
        self.lbl_letter.setText(f"Letra: {self.current_static_letter}")
        self.lbl_source.setText(f"Origen: {self.recognition_source}")
        self.lbl_motion.setText(f"Movimiento: {self.current_motion_letter}")

        if self.recorder.recording:
            rem = self.recorder.get_remaining_time()
            self.status_bar.showMessage(f"GRABANDO {self.recorder.current_letter}: {rem:.1f}s")

        # Convertir frame de OpenCV a QImage para mostrar en PyQt
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def _draw_landmarks_cv(self, frame, lands):
        CONNECTIONS = [
            (0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12), (9,13),(13,14),(14,15),(15,16),
            (13,17),(17,18),(18,19),(19,20), (0,17)
        ]
        h, w = frame.shape[:2]
        for start, end in CONNECTIONS:
            p1 = (int(lands[start].x * w), int(lands[start].y * h))
            p2 = (int(lands[end].x * w), int(lands[end].y * h))
            cv2.line(frame, p1, p2, (0, 255, 0), 2)
        for lm in lands:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 0, 255), -1)

    def record_static(self):
        letter = self.manual_letter or (self.current_static_letter if self.current_static_letter != "---" else None)
        if letter:
            self.log_widget.append_log(f"Iniciando grabación estática para letra '{letter}'...")
            self.recorder.start_recording(letter, is_motion=False, duration=1.5)
        else:
            self.status_bar.showMessage("Selecciona una letra primero")
            self.log_widget.append_log("Error: Intento de grabación sin letra seleccionada.", "error")

    def record_motion(self):
        if self.target_motion_letter:
            self.recorder.start_recording(self.target_motion_letter, is_motion=True)
        else:
            self.status_bar.showMessage("No hay gesto disparador activo")

    def on_two_hands_toggled(self, checked):
        num_hands = 2 if checked else 1
        self.processor.set_num_hands(num_hands)
        self.status_bar.showMessage(f"Detección de dos manos: {'Activada' if checked else 'Desactivada'}")
        self.log_widget.append_log(f"Configuración de dos manos cambiada a: {'Activada' if checked else 'Desactivada'}", "info")

    def update_trail_len(self, val):
        self.tracker.set_max_len(val)
        self.status_bar.showMessage(f"Longitud de estela: {val}")

    def take_full_screenshot(self):
        screen = QGuiApplication.primaryScreen()
        if screen:
            # GrabWindow(0) captura la pantalla completa en la mayoría de plataformas
            screenshot = screen.grabWindow(0)

            # Ruta de Documentos
            docs_path = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
            save_dir = os.path.join(docs_path, "Capturas_LSM")

            # Crear directorio si no existe
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            timestamp = QDateTime.currentDateTime().toString("yyyyMMdd_hhmmss")
            filename = f"screenshot_{timestamp}.png"
            full_path = os.path.join(save_dir, filename)

            if screenshot.save(full_path, "PNG"):
                self.log_widget.append_log(f"Captura guardada: {filename}", "success")
                self.status_bar.showMessage(f"Captura guardada en {save_dir}")
            else:
                self.log_widget.append_log("Error al guardar captura", "error")

    def start_motion_training(self):
        self.btn_train_motion.setEnabled(False)
        self.train_progress_lbl.setText("Entrenando Movimiento...")
        self.log_widget.append_log("Iniciando entrenamiento del modelo de Movimiento...", "info")
        self.thread = TrainingThread(train_motion)
        self.thread.progress.connect(self.on_training_progress)
        self.thread.finished.connect(self.on_motion_training_finished)
        self.thread.start()

    def on_motion_training_finished(self, success, msg):
        self.btn_train_motion.setEnabled(True)
        self.train_progress_lbl.setText(msg)
        self.status_bar.showMessage(msg)
        self.log_widget.append_log(msg, "success" if success else "error")
        if success:
            self.logic.reload()
            self.log_widget.append_log("Modelo de movimiento y base de datos recargados.", "success")

    def start_training(self):
        self.btn_train.setEnabled(False)
        self.train_progress_lbl.setText("Entrenando...")
        self.log_widget.append_log("Iniciando entrenamiento del modelo MLP...", "info")
        self.thread = TrainingThread(train)
        self.thread.progress.connect(self.on_training_progress)
        self.thread.finished.connect(self.on_training_finished)
        self.thread.start()

    def on_training_progress(self, msg):
        self.train_progress_lbl.setText(msg)
        # Evitar saturar el log si el mensaje es muy frecuente, pero aquí son epochs
        self.log_widget.append_log(msg, "info")

    def on_training_finished(self, success, msg):
        self.btn_train.setEnabled(True)
        self.train_progress_lbl.setText(msg)
        self.status_bar.showMessage(msg)
        self.log_widget.append_log(msg, "success" if success else "error")
        if success:
            self.logic.reload()
            self.log_widget.append_log("Modelo y base de datos recargados.", "success")

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        # a-z
        if Qt.Key.Key_A <= key <= Qt.Key.Key_Z:
            char = chr(key).upper()
            self.combo_letter.setCurrentText(char)

        elif Qt.Key.Key_0 <= key <= Qt.Key.Key_9:
            char = chr(key)
            self.combo_letter.setCurrentText(char)

        elif key == Qt.Key.Key_Return or key == Qt.Key.Key_Enter:
            self.record_static()

        elif key == Qt.Key.Key_F11:
            self.start_training()

        elif key == Qt.Key.Key_F12:
            self.record_motion()

        elif key == Qt.Key.Key_Q:
            self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HandAppQT()
    window.show()
    sys.exit(app.exec())
