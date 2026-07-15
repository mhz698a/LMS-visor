# MANUAL DE USUARIO - LMS-Visor
## Sistema de Reconocimiento y Grabación de Lengua de Señas Mexicana (LSM)

¡Bienvenido al Manual de Usuario de **LMS-Visor**! Este documento está diseñado de forma pedagógica para ayudarte a comprender el funcionamiento del sistema, la estructura de su interfaz gráfica y todas las funcionalidades avanzadas disponibles para el aprendizaje, grabación y entrenamiento de gestos.

---

## 1. Introducción
LMS-Visor es una aplicación interactiva desarrollada en Python con **PyQt6**, **MediaPipe** y **Scikit-learn** que permite:
* Capturar video en tiempo real desde una cámara **OAK-D** o cualquier **Webcam** estándar.
* Detectar los puntos de referencia (*landmarks*) de una o dos manos.
* Reconocer gestos estáticos y dinámicos (movimientos) de la Lengua de Señas Mexicana mediante un sistema híbrido de Inteligencia Artificial (MLP), heurísticas y bases de datos estadísticas.
* Registrar nuevas muestras de señas para expandir el vocabulario de manera ilimitada.

---

## 2. Estructura de la Interfaz Gráfica

La interfaz de usuario está organizada de manera modular para facilitar su uso. Se divide principalmente en tres secciones:

### A. Panel de Visualización (Izquierda Superior)
* **Pantalla de Video:** Muestra la señal de la cámara seleccionada. Cuando la cámara está desconectada, se muestra un fondo negro con el texto "Cámara Desconectada". Al conectarse, muestra el flujo de video con el esqueleto de la mano dibujado en tiempo real (puntos rojos y conexiones verdes) y las estelas de seguimiento de colores según el dedo.
* **Historial de Texto en Pantalla:** En la parte inferior central de la pantalla de video se muestra el historial del texto reconocido (por ejemplo: "A B C D 1 2 3"). Cuenta con una sombra negra de alto contraste para garantizar su lectura sobre cualquier fondo de video.

### B. Consola de Logs del Sistema (Izquierda Inferior)
* Un widget de texto especializado con fondo negro que reporta detalladamente las acciones del sistema con códigos de color:
  * **Blanco (Info):** Información general del sistema.
  * **Verde (Éxito):** Operaciones exitosas (conexión de cámara, guardado de capturas, etc.).
  * **Amarillo (Advertencia):** Desconexiones o avisos de configuración.
  * **Rojo (Error):** Errores críticos.

### C. Panel Lateral de Controles (Derecha)
Para mantener la interfaz limpia y organizada, los controles se dividen en paneles fijos permanentes y pestañas interactivas:

#### I. Paneles Fijos (Siempre Visibles)
1. **Control de Cámara:**
   * **Seleccionar Fuente:** Menú desplegable que incluye la cámara "OAK-D" y todas las webcams detectadas.
   * **Actualizar Cámaras:** Escanea dinámicamente el sistema para identificar nuevos dispositivos de video conectados.
   * **Conectar / Desconectar Cámara:** Inicia o detiene el flujo de video.
2. **Reconocimiento:**
   * **Letra:** Muestra el gesto estático reconocido en tiempo real y con letra grande de color verde.
   * **Origen:** Indica qué método de clasificación identificó el gesto (ej: *MLP*, *Heurística* o *DB*).
   * **Movimiento:** Muestra el gesto dinámico detectado y su confianza de predicción.

#### II. Pestañas de Configuración (QTabWidget)
* **Pestaña 1: "Grabacion/Datos"**
  * Control de selección de letra/gesto manual usando botones circulares `<` y `>` o el menú desplegable.
  * **Abrir Gestor de Gestos:** Abre una ventana emergente para añadir o quitar palabras y frases al catálogo de grabación.
  * **Grabar Estático (Enter):** Inicia la grabación de puntos fijos por 1.5 segundos.
  * **Grabar Movimiento (F12):** Inicia la grabación de un gesto dinámico (por ejemplo, para la letra 'J' o 'Z') por 5 segundos.
* **Pestaña 2: "Modelos ML"**
  * **Entrenar Modelo (F11):** Entrena el modelo MLP estático utilizando las muestras de `gestures.json`.
  * **Entrenar Movimiento:** Entrena el modelo MLP dinámico utilizando las muestras de `motion_gestures.json`.
  * Reporta el progreso del entrenamiento en tiempo real.
* **Pestaña 3: "Utilidades"**
  * **Configuración Visual:**
    * **Permitir detección de dos manos:** Activa/desactiva la detección simultánea de ambas manos en la cámara.
    * **Mostrar historial en pantalla:** Muestra u oculta el texto acumulado en la parte inferior del video.
    * **Evitar duplicados seguidos:** Si está activo, impide registrar consecutivamente la misma letra a menos que se retire la mano de la pantalla por un momento.
    * **Limpiar Historial:** Vacía por completo la línea de texto del historial.
    * **Longitud de Estela:** Slider para regular la longitud de la estela de color que dejan los dedos activos.
  * **Utilidades Generales:**
    * **Mostrar/Ocultar Guía de señas:** Despliega un panel lateral izquierdo con una imagen guía interactiva del abecedario en LSM. Permite hacer zoom con la rueda del mouse y desplazar la imagen haciendo click y arrastrando.
    * **Capturar Pantalla (Full):** Guarda una captura de pantalla completa en formato PNG dentro de la carpeta `Documentos/Capturas_LSM/`.

---

## 3. Funcionalidades Detalladas

### 3.1 Gestor de Gestos Personalizados
Este módulo permite ampliar libremente el catálogo de gestos aceptables sin modificar el código fuente:
1. Dirígete a la pestaña **"Grabacion/Datos"** y presiona **"Abrir Gestor de Gestos"**.
2. Verás la lista de señas cargada desde `custom_gestures_list.json`.
3. **Añadir:** Escribe una palabra, número o frase en el campo de entrada. El sistema la convertirá automáticamente a mayúsculas. Presiona "Añadir".
4. **Eliminar:** Selecciona un elemento personalizado y presiona "Eliminar".
5. **Protección:** Por seguridad, las letras base de la `A` a la `Z` y los números del `0` al `10` **no pueden ser eliminados**, garantizando que el abecedario y la numeración base siempre estén disponibles.
6. Presiona **"Guardar y Cerrar"** para actualizar la lista de grabación.

### 3.2 Historial de Texto Inteligente (Escritura Estable)
El sistema incluye un algoritmo de escritura robusto para simular un teclado virtual en pantalla:
* **Filtro de Estabilidad:** Un gesto debe sostenerse de manera idéntica durante **10 frames consecutivos** (aprox. 0.3 segundos) antes de añadirse al historial para evitar errores por movimientos de transición.
* **Evitación de Duplicados Seguidos:** Si está activa, evita escribir la misma letra múltiples veces por error mientras mantienes el gesto.
* **Detección de Pausa (Paso Libre para duplicados):** Si deseas escribir dos letras iguales seguidas (ej: "EE"), simplemente retira la mano del campo de visión de la cámara por **1 segundo** y vuelve a realizar el gesto; el sistema detectará la pausa y permitirá registrar el duplicado perfectamente.

### 3.3 Selección y Detección de Múltiples Cámaras
El sistema escanea activamente los puertos de video al inicio:
* Si tienes varias cámaras o webcams conectadas, se listarán como `"Webcam 0"`, `"Webcam 1"`, etc.
* Puedes cambiar la cámara seleccionada en el combo de fuente y conectar/desconectar sobre la marcha.
* Puedes presionar **"Actualizar Cámaras"** en cualquier momento para refrescar la lista de dispositivos de video si conectaste una nueva cámara USB sin cerrar la aplicación.

### 3.4 Detección Dinámica de Dos Manos
* Al marcar la casilla **"Permitir detección de dos manos"**, el motor MediaPipe se reconfigura internamente para rastrear hasta 2 manos de forma simultánea.
* El sistema dibujará el esqueleto de ambas manos en pantalla.
* Para garantizar consistencia, el análisis principal de gestos, la grabación de muestras y el rastro de estelas activas se centrará en la **mano primaria (primera detectada)**, permitiendo que la segunda mano se visualice libremente en la interfaz.

---

## 4. Tabla de Atajos de Teclado (Teclas Rápidas)

| Tecla | Acción |
| :--- | :--- |
| **`A` - `Z`** | Selecciona la letra correspondiente en el combobox de grabación manual. |
| **`0` - `9`** | Selecciona el número correspondiente en el combobox de grabación manual. |
| **`Enter` / `Return`** | Inicia la grabación de una muestra estática (duración: 1.5 segundos). |
| **`F11`** | Inicia el entrenamiento en segundo plano del modelo MLP estático. |
| **`F12`** | Inicia la grabación de un gesto con movimiento (duración: 5 segundos). |
| **`Q`** | Cierra la aplicación de manera segura. |

---
*Este manual ha sido elaborado con un enfoque pedagógico para facilitar la comprensión técnico-práctica tanto de estudiantes como de usuarios finales.*
