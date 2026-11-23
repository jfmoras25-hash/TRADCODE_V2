"""Representación de los datos y extracción de características.

Este archivo define el "contrato" entre el entrenamiento (Python) y el
prototipo web (TypeScript, ``web/src/lib/features.ts``). Ambos DEBEN producir
exactamente el mismo vector para los mismos landmarks; ``tests/test_paridad.py``
lo verifica.

Datos crudos por cuadro (salida de MediaPipe Hand Landmarker + Pose Landmarker):
    hands         float32 [2, 21, 3]  ranuras [Right, Left] según la etiqueta
                                       de lateralidad de MediaPipe; x, y en
                                       [0, 1] relativos a la imagen, z relativo.
    hand_present  float32 [2]          1 si la mano de esa ranura fue detectada.
    pose          float32 [33, 4]      x, y, z, visibilidad.
    pose_present  float32 []           1 si se detectó el cuerpo.
    aspect        float                ancho / alto de la imagen.

Vector de características por cuadro (N_FEATURES = 147):
    [0:63]    mano ranura 0 (Right): 21 puntos relativos a la muñeca,
              divididos por la distancia muñeca -> MCP del dedo medio.
    [63:126]  mano ranura 1 (Left), igual.
    [126:128] presencia de cada mano.
    [128:142] 7 puntos del cuerpo (nariz, hombros, codos, muñecas) en (x, y)
              relativos al centro de los hombros y divididos por el ancho de
              hombros.
    [142:146] posición (x, y) de cada muñeca de mano respecto al cuerpo
              (misma normalización) -> codifica DÓNDE se hace la seña.
    [146]     presencia del cuerpo.
"""
from __future__ import annotations

import numpy as np

N_HAND_LM = 21
N_POSE_LM = 33
SEQ_LEN = 6  # LSC70 usa transiciones de 6 cuadros
POSE_IDX = [0, 11, 12, 13, 14, 15, 16]  # nariz, hombros, codos, muñecas
N_FEATURES = 63 * 2 + 2 + len(POSE_IDX) * 2 + 4 + 1  # 147
FEATURE_VERSION = 1
EPS = 1e-6
CLIP = 10.0

# Pares izquierda/derecha del modelo de pose de MediaPipe (para espejar).
POSE_MIRROR_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
    (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30),
    (31, 32),
]
POSE_MIRROR_PERM = np.arange(N_POSE_LM)
for _a, _b in POSE_MIRROR_PAIRS:
    POSE_MIRROR_PERM[_a], POSE_MIRROR_PERM[_b] = _b, _a


def empty_frame() -> dict:
    return {
        "hands": np.zeros((2, N_HAND_LM, 3), np.float32),
        "hand_present": np.zeros(2, np.float32),
        "pose": np.zeros((N_POSE_LM, 4), np.float32),
        "pose_present": np.float32(0),
    }


def frame_features(hands, hand_present, pose, pose_present, aspect) -> np.ndarray:
    """Vector de N_FEATURES características para un cuadro."""
    out = np.zeros(N_FEATURES, np.float32)
    sxyz = np.array([aspect, 1.0, aspect], np.float32)

    for s in range(2):
        if hand_present[s] > 0.5:
            p = hands[s].astype(np.float32) * sxyz
            rel = p - p[0]
            scale = float(np.linalg.norm(rel[9]))
            if scale < EPS:
                scale = float(np.max(np.linalg.norm(rel, axis=1)))
            rel = rel / max(scale, EPS)
            out[s * 63:(s + 1) * 63] = rel.reshape(-1)
    out[126:128] = (np.asarray(hand_present) > 0.5).astype(np.float32)

    if pose_present > 0.5:
        sxy = np.array([aspect, 1.0], np.float32)
        P = pose[:, :2].astype(np.float32) * sxy
        center = (P[11] + P[12]) / 2.0
        sw = max(float(np.linalg.norm(P[11] - P[12])), EPS)
        out[128:142] = ((P[POSE_IDX] - center) / sw).reshape(-1)
        for s in range(2):
            if hand_present[s] > 0.5:
                w = hands[s, 0, :2].astype(np.float32) * sxy
                out[142 + 2 * s:144 + 2 * s] = (w - center) / sw
        out[146] = 1.0

    np.clip(out, -CLIP, CLIP, out=out)
    return out


def sequence_features(hands, hand_present, pose, pose_present, aspect) -> np.ndarray:
    """Arreglos con eje temporal T al inicio -> [T, N_FEATURES]."""
    T = hands.shape[0]
    return np.stack([
        frame_features(hands[t], hand_present[t], pose[t], pose_present[t], aspect)
        for t in range(T)
    ])


def batch_features(data: dict) -> np.ndarray:
    """``data`` con arreglos [N, T, ...] -> [N, T, N_FEATURES] (vectorizado).

    Matemáticamente idéntico a ``frame_features`` aplicado cuadro a cuadro;
    se usa en el entrenamiento porque es ~100 veces más rápido.
    """
    hands = data["hands"].astype(np.float32)
    hp = (data["hand_present"] > 0.5)
    pose = data["pose"].astype(np.float32)
    pp = (data["pose_present"] > 0.5)
    N, T = hands.shape[:2]
    a = data["aspect"].astype(np.float32).reshape(N, 1, 1, 1)
    out = np.zeros((N, T, N_FEATURES), np.float32)

    H = hands.copy()
    H[..., 0] *= a
    H[..., 2] *= a
    rel = H - H[..., 0:1, :]
    sc = np.linalg.norm(rel[..., 9, :], axis=-1)
    alt = np.max(np.linalg.norm(rel, axis=-1), axis=-1)
    sc = np.where(sc < EPS, alt, sc)
    rel = rel / np.maximum(sc, EPS)[..., None, None]
    rel *= hp[..., None, None]
    out[..., 0:126] = rel.reshape(N, T, 126)
    out[..., 126:128] = hp

    a2 = data["aspect"].astype(np.float32).reshape(N, 1, 1)
    P = pose[..., :2].copy()
    P[..., 0] *= a2
    center = (P[..., 11, :] + P[..., 12, :]) / 2.0
    sw = np.maximum(np.linalg.norm(P[..., 11, :] - P[..., 12, :], axis=-1), EPS)
    pts = (P[..., POSE_IDX, :] - center[..., None, :]) / sw[..., None, None]
    W = hands[..., 0, :2].copy()
    W[..., 0] *= a2
    wr = (W - center[..., None, :]) / sw[..., None, None]
    wr *= hp[..., None]
    out[..., 128:142] = pts.reshape(N, T, 14) * pp[..., None]
    out[..., 142:146] = wr.reshape(N, T, 4) * pp[..., None]
    out[..., 146] = pp
    np.clip(out, -CLIP, CLIP, out=out)
    return out


def mirror_raw(hands, hand_present, pose, pose_present):
    """Espeja horizontalmente una secuencia cruda [T, ...].

    En LSC una persona zurda hace las señas en espejo; espejar es una
    aumentación válida y hace al modelo independiente de la mano dominante y
    de si la cámara invierte la imagen.
    """
    h = hands.copy()
    h[..., 0] = 1.0 - h[..., 0]
    h = h[:, ::-1].copy()            # intercambia ranuras Right <-> Left
    hp = hand_present[:, ::-1].copy()
    p = pose.copy()
    p[..., 0] = 1.0 - p[..., 0]
    p = p[:, POSE_MIRROR_PERM].copy()
    return h, hp, p, pose_present.copy()
