"""Detección de landmarks con MediaPipe Tasks (las mismas redes que usa la web)."""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

from .features import N_HAND_LM, empty_frame

MODELOS_DIR = Path(__file__).resolve().parents[1] / "modelos_mediapipe"
URLS = {
    "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task",
    "pose": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}


def descargar_modelos(destino: Path = MODELOS_DIR) -> dict[str, Path]:
    destino.mkdir(parents=True, exist_ok=True)
    rutas = {}
    for nombre, url in URLS.items():
        ruta = destino / Path(url).name
        if not ruta.exists():
            print(f"Descargando {ruta.name} ...")
            urllib.request.urlretrieve(url, ruta)
        rutas[nombre] = ruta
    return rutas


class DetectorLandmarks:
    """Envuelve Hand Landmarker (2 manos) + Pose Landmarker en modo IMAGE."""

    def __init__(self, min_conf: float = 0.3):
        import mediapipe as mp  # import tardío: solo se necesita para extraer
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        rutas = descargar_modelos()
        self.hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(rutas["hand"])),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=min_conf,
                min_hand_presence_confidence=min_conf,
            ))
        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(rutas["pose"])),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=min_conf,
                min_pose_presence_confidence=min_conf,
            ))

    def detectar_archivo(self, ruta: str | Path) -> tuple[dict, float]:
        img = self._mp.Image.create_from_file(str(ruta))
        aspect = img.width / img.height
        return self.detectar(img), aspect

    def detectar(self, img) -> dict:
        fr = empty_frame()
        hr = self.hands.detect(img)
        manos = []
        for lms, hd in zip(hr.hand_landmarks, hr.handedness):
            cat = hd[0]
            arr = np.array([[l.x, l.y, l.z] for l in lms], np.float32)
            manos.append((cat.score, cat.category_name, arr))
        asignar_manos(fr, manos)

        pr = self.pose.detect(img)
        if pr.pose_landmarks:
            lms = pr.pose_landmarks[0]
            fr["pose"] = np.array(
                [[l.x, l.y, l.z, l.visibility if l.visibility is not None else 1.0]
                 for l in lms], np.float32)
            fr["pose_present"] = np.float32(1)
        return fr

    def close(self):
        self.hands.close()
        self.pose.close()


def asignar_manos(fr: dict, manos: list[tuple[float, str, np.ndarray]]) -> None:
    """Coloca cada mano en su ranura (0=Right, 1=Left).

    Si MediaPipe etiqueta ambas manos igual, la de menor confianza va a la
    ranura libre. La web (``features.ts``) aplica exactamente esta regla.
    """
    for score, nombre, arr in sorted(manos, key=lambda m: -m[0]):
        slot = 0 if nombre == "Right" else 1
        if fr["hand_present"][slot] > 0.5:
            slot = 1 - slot
            if fr["hand_present"][slot] > 0.5:
                continue
        fr["hands"][slot] = arr[:N_HAND_LM]
        fr["hand_present"][slot] = 1.0
