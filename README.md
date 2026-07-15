# LMS-Visor

![Screenshot del Sistema](https://github.com/user-attachments/assets/a9a0524e-b278-4ab4-acf3-1ddcb28737f3)

### 1. Objetivo
LMS-Visor es una aplicación de escritorio en Python para capturar video, detectar landmarks de la mano, reconocer señas y registrar muestras para entrenamiento. La interfaz está construida con PyQt6 y trabaja con una cámara OAK-D o una webcam estándar.

### 2. Componentes principales
- `main.py`: orquesta la interfaz, la cámara, el reconocimiento, la grabación y el entrenamiento.
- `camera_engine.py`: abstrae la fuente de video.
- `hand_processor.py`: ejecuta la detección de manos con MediaPipe HandLandmarker.
- `gesture_logic.py`: contiene la lógica de reconocimiento estático y de movimiento.
- `tracker.py`: mantiene las trayectorias de los dedos activos.
- `recorder.py`: guarda muestras estáticas y de movimiento.
- `pencil.py`: utilidad separada para visualizar rastros y pruebas de seguimiento.
- `gestures.json`: base de datos de gestos estáticos.
- `motion_gestures.json`: base de datos de gestos dinámicos.
- `models/`: modelos entrenados y mapeo de clases.

### 3. Requisitos
- Python 3.10
- `opencv-python`
- `depthai`
- `mediapipe`
- `numpy`
- `keyboard`
- `PyQt6`
- `scikit-learn`
- `joblib`
- Archivo `hand_landmarker.task` en la raíz del proyecto

### 4. Instalación
1. Clonar el repositorio.
2. Crear y activar un entorno virtual.
3. Instalar dependencias.
4. Verificar que exista `hand_landmarker.task`.
5. Ejecutar `main.py`.

### 5. Flujo de uso
1. Abrir la aplicación.
2. Seleccionar la fuente de cámara.
3. Conectar la cámara.
4. Observar la detección de mano en pantalla.
5. Elegir una letra manual o dejar que el sistema la detecte.
6. Guardar una muestra estática o iniciar una grabación de movimiento.
7. Entrenar el modelo cuando haya suficientes muestras.

### 6. Reconocimiento
El sistema usa una lógica híbrida:
- primero intenta clasificar con MLP,
- si la confianza no alcanza el umbral, cae a heurísticas,
- y después usa comparación estadística sobre la base de datos JSON.

### 7. Entrenamiento
- `F11` inicia el entrenamiento del modelo estático.
- El modelo se recarga en caliente cuando termina.
- También existe un botón para entrenar movimiento.

### 8. Grabación
- Las muestras estáticas duran 1.5 segundos.
- Las muestras de movimiento duran 5 segundos por defecto.
- La grabación descarta buffers inválidos con pocos frames útiles.

### 9. Controles
- `A` a `Z`: selecciona letra manual.
- `Enter`: graba muestra estática.
- `F11`: entrena el modelo.
- `F12`: graba movimiento.
- `Q`: cierra la aplicación.

### 10. Utilidades
- La pantalla completa puede capturarse desde la interfaz.
- Las capturas se guardan en `Documentos/Capturas_LSM/`.
- El panel de guía de señas puede mostrarse u ocultarse.
- El control visual de estelas permite ajustar la longitud del rastro.

### 11. Notas técnicas
- `camera_engine.py` soporta OAK-D y webcam.
- `hand_processor.py` usa detección asíncrona con MediaPipe.
- `gesture_logic.py` incluye disparadores como `I -> J`, `P -> K`, `N -> Ñ`, `Q -> Q`, `X -> X` y `D -> Z`.
- `tracker.py` dibuja estelas por dedo.
- `pencil.py` incluye callbacks de teclado para alternar dedos y salir con `ESC`.

