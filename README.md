# Detektor emócií tváre (MediaPipe Tasks API)

Detekcia emócií tváre v reálnom čase (Happy / Sad / Neutral) pomocou moderného **MediaPipe Tasks API** [FaceLandmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker#python) a OpenCV.

Určené pre PC s USB webkamerou. Funguje s najnovšími verziami MediaPipe.

## Inštalácia

```bash
# Odporúčané – vytvorenie a aktivácia virtuálneho prostredia
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Inštalácia závislostí
pip install -e .
```

Alebo priamo:

```bash
pip install opencv-python mediapipe numpy
```

## Spustenie

```bash
python emotion_detector.py
```

Pri prvom spustení si skript automaticky stiahne potrebný model (`face_landmarker.task`).

Stlač **q** na ukončenie.

## Požiadavky

- Python 3.9+
- USB webkamera
- Závislosti uvedené v `pyproject.toml`

## Ladenie

Ak klasifikácia nie je dosť presná, uprav prahy vo funkcii `classify_emotion()` podľa debug hodnôt zobrazených na obrazovke (`mw`, `mo`, `eo`, `cd`, `ls`, `br`).

## Poznámky

Táto verzia používa nové MediaPipe Tasks API namiesto starého (už nepodporovaného) `mp.solutions` API.