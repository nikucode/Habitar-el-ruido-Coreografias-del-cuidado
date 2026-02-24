# CONEXIÓN / PRESENCIA / EXCESO
# obra audiovisual reactiva con OpenCV
#
# - Usa cámara para detectar presencia humana
# - Mezcla imagen real + video pregrabado
# - Introduce glitch / datamosh como exceso informacional
# - La cercanía transforma el comportamiento del sistema
#
# Autoría colectiva / proceso artístico


# Importación de librerías

import cv2  # OpenCV: visión por computador (cámara, video, imagen)
import random  # Generación de números aleatorios (glitch orgánico)
import time  # Control temporal
import numpy as np  # Operaciones numéricas eficientes

# configuración de archivos

VIDEO_PATH = "videos/v1.mp4"
# video que representa el flujo de información / exceso digital

CASCADE_PATH = "haarcascade_frontalface_default.xml"
# clasificador Haar para detección de rostros (presencia humana)

# Parámetros del glitch / datamosh

BASE_BLOCK_SIZE = 62
# tamaño de los bloques que se desplazan (fragmentación de imagen)

STICKINESS = 0.88
# qué tan pegajosos son los desplazamientos:
# valores altos = memoria / inercia visual

FREEZE_PROB = 0.003
UNFREEZE_PROB = 0.01
# probabilidades de congelar / liberar bloques
# simboliza fijación o liberación de información

# parámetros temporales (ritmo)

NORMAL_DURATION = 11.0  # segundos de imagen "estable"
DATAMOSH_DURATION = 4.0  # segundos de exceso / glitch

FADE_IN_SPEED = 0.25  # rapidez con que entra el glitch
FADE_OUT_SPEED = 0.15  # rapidez con que se disuelve

# inicialización de detectores

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)  # detector de rostros entrenado

cam = cv2.VideoCapture(0)  # cámara en vivo (0 = cámara por defecto)


video = cv2.VideoCapture(VIDEO_PATH)  # video de fondo (memoria / ruido digital)

# variables de estado afectivo
prev_cam_frame = None

stillness_time = 0

presence_time = 0  # tiempo acumulado de presencia humana

decay_time = 0  # persistencia de la presencia cuando el cuerpo se va

# memorias visuales del glitch

prev_video_frame = None  # frame anterior (memoria)
block_memory = {}  # desplazamientos por bloque
frozen_blocks = set()  # bloques congelados

# control de estados del sistema

datamosh_active = False  # ¿estamos en modo exceso?

datamosh_strength = 0.0  # intensidad continua del glitch (0–1)

state_start_time = time.time()
current_state_duration = 0

target_duration = NORMAL_DURATION  # duración del estado actual

video_speed_accumulator = (
    0.0  # control de velocidad del video (ralentización por presencia)
)

# funciones auxiliares


def read_video_frame(speed, target_size):
    """
    Lee frames del video a velocidad variable.
    Más presencia humana = video más lento.
    """
    global video_speed_accumulator

    video_speed_accumulator += speed
    if video_speed_accumulator < 1.0:
        return None

    video_speed_accumulator = 0.0

    ret, frame = video.read()
    if not ret:
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = video.read()
        if not ret:
            return None

    return cv2.resize(frame, target_size)


def update_datamosh_state():
    """
    Alterna entre estados:
    - normal
    - exceso (datamosh)
    siguiendo un ritmo interno
    """
    global datamosh_active, state_start_time
    global target_duration, current_state_duration

    now = time.time()
    current_state_duration = now - state_start_time

    if current_state_duration >= target_duration:
        datamosh_active = not datamosh_active
        state_start_time = now
        current_state_duration = 0

        if datamosh_active:
            target_duration = DATAMOSH_DURATION
        else:
            target_duration = NORMAL_DURATION


# Bucle principal (sistema vivo)

while True:
    # captura de cámara

    ret, cam_frame = cam.read()
    if not ret:
        break

    cam_frame = cv2.flip(cam_frame, 1)
    h, w, _ = cam_frame.shape
    target_size = (w, h)

    # detección de presencia humana

    gray = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
    )

    # visualizacion de deteccion facial
    for x, y, w_face, h_face in faces:
        cv2.rectangle(
            cam_frame,
            (x, y),
            (x + w_face, y + h_face),
            (0, 0, 255),  # cuadro rojo
            2,
        )
    num_faces = len(faces)

    mode = "conexion"  # modo por defecto
    # detección de quietud (comparando con frame anterior)
    if prev_cam_frame is not None:
        movement = np.mean(
            np.abs(cam_frame.astype(np.int16) - prev_cam_frame.astype(np.int16))
        )
    else:
        movement = 999  # primer frame, asumimos movimiento

    if movement < 3 and num_faces == 1:
        stillness_time += 1
    else:
        # romper el reposo debe ser rápido
        stillness_time = 0

    if mode == "reposo" and stillness_time > 180:
        final = np.zeros_like(final)  # pantalla negra

    prev_cam_frame = cam_frame.copy()

    # quietud como cuidado(lo agregamos o no?? nose si toy 100 de acuerdo)
    if num_faces == 1 and presence_time > 120:
        datamosh_strength *= 0.7

    # actualización de presencia
    presence_gain = min(num_faces, 4) * 0.8

    if num_faces > 0:
        presence_gain = min(num_faces, 4) * 0.8

        if mode in ["conexion", "intensidad"]:
            # solo crece si no estamos cuidando el silencio
            presence_time = min(presence_time + presence_gain, 400)
        elif mode == "reposo":
            # en reposo, el tiempo se estabiliza
            presence_time = max(presence_time - 1.0, 80)
    else:
        presence_time = max(0, presence_time - 2.0)

    intensity = min(presence_time / 240, 1.0)  # intensidad emocional / informacional

    BLOCK_SIZE = int(BASE_BLOCK_SIZE + intensity * 20)

    # estado conceptual (exceso / conexion / saturacion)
    if num_faces == 0:
        mode = "exceso"  # el mundo sigue aunque no estés

    elif stillness_time > 120 and num_faces == 1 and movement < 2:
        mode = "reposo"  # cuidado consciente sostenido

    elif num_faces == 1 and movement < 6:
        mode = "conexion"  # vínculo presente pero vivo

    elif num_faces >= 3 or movement > 15:
        mode = "saturacion"  # demasiadas señales

    else:
        mode = "intensidad"  # zona intermedia
    # modulacion emocional del glitch

    if mode == "reposo":
        datamosh_target = 0.05
    elif mode == "conexion":
        datamosh_target = 0.25
    elif mode == "intensidad":
        datamosh_target = 0.6
    else:  # saturacion
        datamosh_target = 1.0

    # exceso basal del sistema (el mundo nunca esta en cero)
    BASE_EXCESS = 0.15
    datamosh_target = max(datamosh_target, BASE_EXCESS)

    # acercamiento suave al estado emocional

    datamosh_strength += (datamosh_target - datamosh_strength) * 0.05
    datamosh_strength = np.clip(datamosh_strength, 0.0, 1.0)

    datamosh_strength = np.clip(datamosh_strength, 0.0, 1.0)

    # fatiga del sistema
    fatigue = min(presence_time / 300, 1.0)
    datamosh_strength *= 1.0 - fatigue * 0.4

    # lectura del video base + datamosh basado en codecs reales

    video_speed = max(0.08, 1.0 - intensity * 0.75)

    # repetición de frames (tipo P-frame)
    if random.random() < 0.15 * datamosh_strength and prev_video_frame is not None:
        video_frame = prev_video_frame.copy()
    else:
        # salto aleatorio en el video (error de compresión)
        if random.random() < 0.08 * datamosh_strength:
            rand_pos = random.randint(0, int(video.get(cv2.CAP_PROP_FRAME_COUNT) - 1))
            video.set(cv2.CAP_PROP_POS_FRAMES, rand_pos)

        new_video_frame = read_video_frame(video_speed, target_size)

        if new_video_frame is not None:
            video_frame = new_video_frame
        else:
            video_frame = (
                prev_video_frame.copy()
                if prev_video_frame is not None
                else cam_frame.copy()
            )

    # artefactos tipo compresión

    if random.random() < 0.20 * datamosh_strength:
        noise = (np.random.randn(h, w, 3) * 30).astype(np.uint8)
        video_frame = cv2.addWeighted(video_frame, 0.85, noise, 0.15, 0)

    # desplazamiento global (vector motion)
    if random.random() < 0.12 * datamosh_strength:
        dx = random.randint(-50, 50)
        dy = random.randint(-50, 50)
        video_frame = np.roll(video_frame, dx, axis=1)
        video_frame = np.roll(video_frame, dy, axis=0)
    # aplicación del datamosh

    if prev_video_frame is not None and datamosh_strength > 0.01:
        for y in range(0, h - BLOCK_SIZE, BLOCK_SIZE):
            for x in range(0, w - BLOCK_SIZE, BLOCK_SIZE):
                key = (x, y)

                if key not in block_memory or random.random() < 0.02:
                    block_memory[key] = [
                        random.randint(-BLOCK_SIZE * 2, BLOCK_SIZE * 2),
                        random.randint(-BLOCK_SIZE * 2, BLOCK_SIZE * 2),
                    ]

                if key not in frozen_blocks:
                    block_memory[key][0] *= STICKINESS
                    block_memory[key][1] *= STICKINESS

                if random.random() < FREEZE_PROB * datamosh_strength:
                    frozen_blocks.add(key)
                if random.random() < UNFREEZE_PROB:
                    frozen_blocks.discard(key)

                dx, dy = block_memory[key]
                dx = int(dx * datamosh_strength)
                dy = int(dy * datamosh_strength)

                src_x = np.clip(x + dx, 0, w - BLOCK_SIZE)
                src_y = np.clip(y + dy, 0, h - BLOCK_SIZE)

                video_frame[y : y + BLOCK_SIZE, x : x + BLOCK_SIZE] = prev_video_frame[
                    src_y : src_y + BLOCK_SIZE, src_x : src_x + BLOCK_SIZE
                ]

    # deriva orgánica
    if random.random() < 0.02 * datamosh_strength:
        offset = random.randint(-10, 10)
        video_frame = np.roll(video_frame, offset, axis=0)

    prev_video_frame = video_frame.copy()

    # mezcla final cámara + video

    opacity = 0.6 + 0.3 * np.sin(presence_time * 0.02)

    # distorsion del cuerpo (cámara)
    cam_distorted = cam_frame.copy()

    if mode in ["intensidad", "saturacion"]:
        # jitter vertical (pérdida de estabilidad corporal)
        if random.random() < 0.25:
            offset = random.randint(-20, 20)
            cam_distorted = np.roll(cam_distorted, offset, axis=0)

        # ruido corporal
        noise = (np.random.randn(h, w, 3) * 10).astype(np.uint8)
        cam_distorted = cv2.addWeighted(cam_distorted, 0.9, noise, 0.1, 0)

        # pequeño blur (despersonalización)
        cam_distorted = cv2.GaussianBlur(cam_distorted, (7, 7), 0)

    # atmósfera según estado
    if mode == "reposo":
        cam_distorted = cv2.GaussianBlur(cam_distorted, (15, 15), 0)

    elif mode == "conexion":
        cam_distorted = cv2.GaussianBlur(cam_distorted, (5, 5), 0)

    elif mode == "intensidad":
        cam_distorted = cv2.GaussianBlur(cam_distorted, (3, 3), 0)

    # saturación ya tiene su propio caos

    final = cv2.addWeighted(cam_distorted, 1.0 - opacity, video_frame, opacity, 0)

    cv2.putText(
        final,
        f"{mode}",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    cv2.imshow("conexion", final)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    # detección de cierre dde ventana
    if cv2.getWindowProperty("conexion", cv2.WND_PROP_VISIBLE) < 1:
        break

# liberación de recursos

cam.release()
video.release()
cv2.destroyAllWindows()
