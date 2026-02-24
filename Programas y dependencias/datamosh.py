import cv2
import random
import time
import numpy as np

# ---------------- CONFIG ----------------
VIDEO_PATH = "videos/Download (3).mp4"
CASCADE_PATH = "haarcascade_frontalface_default.xml"

BLOCK_SIZE = 24
STICKINESS = 0.88
FREEZE_PROB = 0.003
UNFREEZE_PROB = 0.01

# timing
NORMAL_DURATION = 11.0      # 7s
DATAMOSH_DURATION = 4.0    # 4s

# Velocidades de transición
FADE_IN_SPEED = 0.25       # activación rápida
FADE_OUT_SPEED = 0.15      # desactivación rápida
# ----------------------------------------

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

cam = cv2.VideoCapture(0)
video = cv2.VideoCapture(VIDEO_PATH)

presence_time = 0
decay_time = 0

prev_video_frame = None
block_memory = {}
frozen_blocks = set()

# Empieza con el video normal 
datamosh_active = False
datamosh_strength = 0.0
state_start_time = time.time()
current_state_duration = 0

# Duración fija según el estado
target_duration = NORMAL_DURATION  # Empezamos con 7s de normal

video_speed_accumulator = 0.0

# ----------------------------------------

def read_video_frame(speed, target_size):
    global video_speed_accumulator

    video_speed_accumulator += speed
    if video_speed_accumulator < 1.0:
        return None

    video_speed_accumulator = 0.0

    ret, frame = video.read()
    if not ret or frame is None:
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = video.read()
        if not ret:
            return None

    return cv2.resize(frame, target_size)

def update_datamosh_state():
    global datamosh_active, state_start_time, target_duration, current_state_duration
    
    now = time.time()
    current_state_duration = now - state_start_time
    
    # Si ha pasado el tiempo exacto del estado actual, cambiar
    if current_state_duration >= target_duration:
        # Cambiar estado
        datamosh_active = not datamosh_active
        
        # Reiniciar temporizador
        state_start_time = now
        current_state_duration = 0
        
        
        if datamosh_active:
            target_duration = DATAMOSH_DURATION  # 4s d datamosh
        else:
            target_duration = NORMAL_DURATION    # 7s normal

# ----------------------------------------

while True:
    ret, cam_frame = cam.read()
    if not ret:
        break

    cam_frame = cv2.flip(cam_frame, 1)
    h, w, _ = cam_frame.shape
    target_size = (w, h)

    gray = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, 1.2, 5, minSize=(60, 60)
    )

    if len(faces) > 0:
        presence_time = min(presence_time + 2, 300)
        decay_time = min(decay_time + 2, 300)
    else:
        presence_time = max(0, presence_time - 2)
        decay_time = max(0, decay_time - 1)

    intensity = min(presence_time / 120, 1.0)

    # -------- ACTUALIZAR ESTADO DATAMOSH --------
    update_datamosh_state()

    # -------- FADE SEGÚN ESTADO --------
    if datamosh_active:
        # Fade in rápido cuando está activo
        datamosh_strength += (1.0 - datamosh_strength) * FADE_IN_SPEED
    else:
        # Fade out rápido cuando está inactivo
        datamosh_strength += (0.0 - datamosh_strength) * FADE_OUT_SPEED
    
    datamosh_strength = np.clip(datamosh_strength, 0.0, 1.0)

    # -------- VELOCIDAD VIDEO --------
    video_speed = 1.0 - intensity * 0.75
    video_speed = max(0.08, video_speed)

    new_video_frame = read_video_frame(video_speed, target_size)
    if new_video_frame is not None:
        video_frame = new_video_frame
    elif prev_video_frame is not None:
        video_frame = prev_video_frame.copy()
    else:
        video_frame = cam_frame.copy()

    # -------- DATAMOSHING CON INERCIA --------
    if prev_video_frame is not None and datamosh_strength > 0.01:
        for y in range(0, h - BLOCK_SIZE, BLOCK_SIZE):
            for x in range(0, w - BLOCK_SIZE, BLOCK_SIZE):
                key = (x, y)

                if key not in block_memory or random.random() < 0.02:
                    dx = random.randint(-BLOCK_SIZE * 2, BLOCK_SIZE * 2)
                    dy = random.randint(-BLOCK_SIZE * 2, BLOCK_SIZE * 2)
                    block_memory[key] = [dx, dy]

                if key not in frozen_blocks:
                    dx, dy = block_memory[key]
                    dx = int(dx * STICKINESS)
                    dy = int(dy * STICKINESS)
                    block_memory[key] = [dx, dy]

                if random.random() < FREEZE_PROB * datamosh_strength:
                    frozen_blocks.add(key)
                if random.random() < UNFREEZE_PROB:
                    frozen_blocks.discard(key)

                dx, dy = block_memory[key]

                dx = int(dx * datamosh_strength)
                dy = int(dy * datamosh_strength)

                src_x = np.clip(x + dx, 0, w - BLOCK_SIZE)
                src_y = np.clip(y + dy, 0, h - BLOCK_SIZE)

                video_frame[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE] = \
                    prev_video_frame[src_y:src_y+BLOCK_SIZE,
                                     src_x:src_x+BLOCK_SIZE]

    prev_video_frame = video_frame.copy()

    # -------- MEZCLA FINAL --------
    opacity = min((presence_time + decay_time) / 240, 1.0)

    final = cv2.addWeighted(
        cam_frame, 1.0 - opacity,
        video_frame, opacity,
        0
    )

    cv2.imshow("conexion", final)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
video.release()
cv2.destroyAllWindows()
