import cv2
import mediapipe as mp
import numpy as np
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = Path("hand_landmarker.task")


def download_model_if_needed():
    if MODEL_PATH.exists():
        return
    print(f"Downloading model to {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")


def get_hand_orientation(landmarks):
    """Zistí, či je ruka pravá alebo ľavá (podľa x pozície malíčka vs ukazováka)."""
    return "right" if landmarks[17].x < landmarks[5].x else "left"


def is_finger_up(landmarks, tip_idx, pip_idx):
    """Jednoduchá a spoľahlivá detekcia – špička je vyššie ako kĺb."""
    return landmarks[tip_idx].y < landmarks[pip_idx].y


def is_thumb_up(landmarks):
    """Thumbs up – palec natiahnutý nahor (rovnaká logika ako ostatné prsty)."""
    return landmarks[4].y < landmarks[3].y and landmarks[4].y < landmarks[2].y


def is_thumb_down(landmarks):
    """Thumbs down – palec natiahnutý nadol."""
    return landmarks[4].y > landmarks[3].y and landmarks[4].y > landmarks[2].y

def detect_gesture(landmarks):
    index_up = is_finger_up(landmarks, 8, 6)
    middle_up = is_finger_up(landmarks, 12, 10)
    ring_up = is_finger_up(landmarks, 16, 14)
    pinky_up = is_finger_up(landmarks, 20, 18)

    thumb_up = is_thumb_up(landmarks)
    thumb_down = is_thumb_down(landmarks)

    # Fuck you – iba prostredník
    is_fuck = middle_up and not index_up and not ring_up and not pinky_up

    # Peace
    is_peace = index_up and middle_up and not ring_up and not pinky_up

    # Handmetal
    is_handmetal = index_up and pinky_up and not middle_up and not ring_up

    # Fist
    is_fist = not index_up and not middle_up and not ring_up and not pinky_up and not thumb_up and not thumb_down

    # Thumbs
    is_thumbs_up = thumb_up #and not index_up and not middle_up and not ring_up and not pinky_up
    is_thumbs_down = thumb_down and not index_up and not middle_up and not ring_up and not pinky_up

    if is_fuck:
        return "Fuck you"
    if is_peace:
        return "Peace"
    if is_handmetal:
        return "Handmetal"
    if is_thumbs_up:
        return "Thumbs up"
    if is_thumbs_down:
        return "Thumbs down"
    if is_fist:
        return "Fist"
    return "Unknown"

def draw_hand(image, landmarks):
    """Jednoduché kreslenie ruky."""
    h, w = image.shape[:2]
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),          # palec
        (0, 5), (5, 6), (6, 7), (7, 8),          # ukazovák
        (0, 9), (9, 10), (10, 11), (11, 12),     # prostredník
        (0, 13), (13, 14), (14, 15), (15, 16),   # prstenník
        (0, 17), (17, 18), (18, 19), (19, 20),   # malíček
        (5, 9), (9, 13), (13, 17),               # dlane
    ]
    for start, end in connections:
        x1, y1 = int(landmarks[start].x * w), int(landmarks[start].y * h)
        x2, y2 = int(landmarks[end].x * w), int(landmarks[end].y * h)
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (x, y), 4, (0, 0, 255), -1)


def main():
    download_model_if_needed()

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    landmarker = HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Hand gesture detection started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                gesture = detect_gesture(hand_landmarks)
                draw_hand(frame, hand_landmarks)

                # Zobrazenie gesta
                cv2.putText(
                    frame,
                    gesture,
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    3,
                    cv2.LINE_AA,
                )

        cv2.imshow("Hand Gestures", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
