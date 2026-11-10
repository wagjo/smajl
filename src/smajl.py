"""
Real-time facial emotion detection (Happy / Sad / Neutral)
using MediaPipe Tasks API (FaceLandmarker) + geometric rules.

Works with USB webcam on PC.
Compatible with latest MediaPipe versions.
"""

import cv2
import mediapipe as mp
import numpy as np
import urllib.request
from pathlib import Path

# Model download URL (official Google model)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_PATH = Path("face_landmarker.task")


def download_model_if_needed():
    """Download the Face Landmarker model if it doesn't exist."""
    if MODEL_PATH.exists():
        return
    print(f"Downloading model to {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")


def euclidean(p1, p2) -> float:
    """Euclidean distance between two landmarks (normalized coordinates)."""
    return float(np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y])))


def classify_emotion(landmarks) -> tuple[str, dict]:
    """
    Vylepšená detekcia Happy / Sad / Neutral pomocou viacerých landmarkov.
    """
    # Oči
    L_EYE_TOP, L_EYE_BOT = 159, 145
    R_EYE_TOP, R_EYE_BOT = 386, 374
    L_EYE_CENTER, R_EYE_CENTER = 33, 263

    # Ústa
    MOUTH_LEFT, MOUTH_RIGHT = 61, 291
    LIP_UP, LIP_DOWN = 13, 14
    # Kútiky úst
    MOUTH_CORNER_L, MOUTH_CORNER_R = 61, 291
    # Spodné body okolo kútikov (na lepšie zistenie smeru kútikov)
    MOUTH_LOWER_L, MOUTH_LOWER_R = 146, 375

    # Obočie
    BROW_L_IN, BROW_L_OUT = 55, 46
    BROW_R_IN, BROW_R_OUT = 285, 276

    io = euclidean(landmarks[L_EYE_CENTER], landmarks[R_EYE_CENTER])
    if io < 1e-6:
        return "Neutral", {}

    # Základné metriky
    mouth_width = euclidean(landmarks[MOUTH_LEFT], landmarks[MOUTH_RIGHT]) / io
    mouth_open = euclidean(landmarks[LIP_UP], landmarks[LIP_DOWN]) / io

    eye_open_L = euclidean(landmarks[L_EYE_TOP], landmarks[L_EYE_BOT]) / io
    eye_open_R = euclidean(landmarks[R_EYE_TOP], landmarks[R_EYE_BOT]) / io
    eye_open = 0.5 * (eye_open_L + eye_open_R)

    # Kútiky úst – či idú dole (sad)
    # Porovnáme y kútika s hornou perou a aj so spodným bodom
    left_corner_down = landmarks[MOUTH_CORNER_L].y - landmarks[LIP_UP].y
    right_corner_down = landmarks[MOUTH_CORNER_R].y - landmarks[LIP_UP].y
    corners_down = (left_corner_down + right_corner_down) / 2

    # Ďalšia kontrola cez spodné body
    left_lower_diff = landmarks[MOUTH_LOWER_L].y - landmarks[MOUTH_CORNER_L].y
    right_lower_diff = landmarks[MOUTH_LOWER_R].y - landmarks[MOUTH_CORNER_R].y
    lower_support = (left_lower_diff + right_lower_diff) / 2

    # Obočie (vnútorné body vyššie = smutok)
    brow_L = landmarks[BROW_L_IN].y - landmarks[L_EYE_TOP].y
    brow_R = landmarks[BROW_R_IN].y - landmarks[R_EYE_TOP].y
    brow_raise = -(brow_L + brow_R) / 2

    # --- Klasifikácia ---
    if mouth_width > 0.46 and mouth_open > 0.028:
        label = "Happy"
    elif (
        corners_down > 0.010
        and mouth_open < 0.032
        and mouth_width < 0.42
    ) or (
        brow_raise > 0.012
        and mouth_open < 0.035
        and corners_down > 0.005
    ):
        label = "Sad"
    else:
        label = "Neutral"

    details = {
        "mouth_width": round(mouth_width, 3),
        "mouth_open": round(mouth_open, 3),
        "eye_open": round(eye_open, 3),
        "corners_down": round(corners_down, 3),
        "lower_support": round(lower_support, 3),
        "brow_raise": round(brow_raise, 3),
    }
    return label, details


def draw_landmarks(image, face_landmarks):
    """Draw key landmarks used for emotion detection."""
    h, w = image.shape[:2]

    # Body, ktoré kreslíme
    points = {
        # Oči
        33: (255, 0, 0),    # ľavé oko center
        263: (255, 0, 0),   # pravé oko center
        159: (0, 255, 255), # ľavé oko top
        145: (0, 255, 255), # ľavé oko bot
        386: (0, 255, 255), # pravé oko top
        374: (0, 255, 255), # pravé oko bot
        # Ústa
        61: (0, 255, 0),   # ľavý kútik
        291: (0, 255, 0),  # pravý kútik
        13: (0, 200, 255), # horná pera
        14: (0, 200, 255), # spodná pera
        146: (255, 100, 0), # lower L
        375: (255, 100, 0), # lower R
        # Obočie
        55: (255, 0, 255),  # brow L in
        285: (255, 0, 255), # brow R in
    }

    for idx, color in points.items():
        lm = face_landmarks[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (x, y), 3, color, -1)
        # Malý text s indexom (voliteľné)
        #cv2.putText(image, str(idx), (x + 4, y - 4),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)


def main():
    download_model_if_needed()

    # MediaPipe Tasks API
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    landmarker = FaceLandmarker.create_from_options(options)

    # Open USB camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Starting emotion detection... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Convert BGR -> RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect landmarks
        result = landmarker.detect(mp_image)

        if result.face_landmarks:
            for face_landmarks in result.face_landmarks:
                # Classify emotion
                label, metrics = classify_emotion(face_landmarks)

                # Color based on emotion
                if label == "Happy":
                    color = (0, 255, 0)  # green
                elif label == "Sad":
                    color = (0, 0, 255)  # red
                else:
                    color = (255, 255, 0)  # cyan

                # Display emotion
                cv2.putText(
                    frame,
                    label,
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3,
                    color,
                    3,
                    cv2.LINE_AA,
                )

                # Debug metrics
                dbg = (
                    f"mw:{metrics.get('mouth_width', 0)}  "
                    f"mo:{metrics.get('mouth_open', 0)}  "
                    f"eo:{metrics.get('eye_open', 0)}"
                )
                dbg2 = (
                    f"cd:{metrics.get('corners_down', 0)}  "
                    f"ls:{metrics.get('lower_support', 0)}  "
                    f"br:{metrics.get('brow_raise', 0)}"
                )
                cv2.putText(frame, dbg, (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, dbg2, (20, 115),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

                # Draw landmarks
                draw_landmarks(frame, face_landmarks)

        cv2.imshow("Emotion Detection (Happy / Sad / Neutral)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
