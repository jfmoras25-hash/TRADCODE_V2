"""Aumentación de datos sobre landmarks crudos (vectorizada por lote).

Simula lo que cambia entre el dataset y la webcam de un usuario:
mano dominante (espejo), posición/tamaño/inclinación frente a la cámara,
ruido del detector, cuadros donde se pierde una mano o el cuerpo, y
velocidad distinta al hacer la seña.
"""
from __future__ import annotations

import numpy as np

from .features import POSE_MIRROR_PERM


def aumentar_lote(d: dict, rng: np.random.Generator, p_espejo: float = 0.5) -> dict:
    h = d["hands"].copy()
    hp = d["hand_present"].copy()
    p = d["pose"].copy()
    pp = d["pose_present"].copy()
    a = d["aspect"].astype(np.float32)
    N, T = h.shape[:2]

    # 1) espejo horizontal (mano dominante izquierda/derecha)
    m = rng.random(N) < p_espejo
    if m.any():
        h[m, ..., 0] = 1.0 - h[m, ..., 0]
        h[m] = h[m][:, :, ::-1]
        hp[m] = hp[m][:, :, ::-1]
        p[m, ..., 0] = 1.0 - p[m, ..., 0]
        p[m] = p[m][:, :, POSE_MIRROR_PERM]

    # 2) transformación afín igual para toda la secuencia
    ang = np.deg2rad(rng.uniform(-12, 12, N)).astype(np.float32)
    esc = rng.uniform(0.85, 1.15, N).astype(np.float32)
    tx, ty = (rng.uniform(-0.08, 0.08, (2, N))).astype(np.float32)
    c, s = np.cos(ang) * esc, np.sin(ang) * esc

    def afin(xy, nd):
        shp = (N,) + (1,) * nd
        X = (xy[..., 0] - 0.5) * a.reshape(shp)
        Y = xy[..., 1] - 0.5
        Xn = c.reshape(shp) * X - s.reshape(shp) * Y
        Yn = s.reshape(shp) * X + c.reshape(shp) * Y
        xy[..., 0] = Xn / a.reshape(shp) + 0.5 + tx.reshape(shp)
        xy[..., 1] = Yn + 0.5 + ty.reshape(shp)

    afin(h, 3)
    h[..., 2] *= esc.reshape(N, 1, 1, 1)
    afin(p, 2)

    # 3) ruido del detector
    h[..., :3] += rng.normal(0, 0.004, h[..., :3].shape).astype(np.float32)
    p[..., :2] += rng.normal(0, 0.004, p[..., :2].shape).astype(np.float32)

    # 4) pérdidas de detección
    caida = rng.random((N, T, 2)) < 0.04
    hp = np.where(caida, 0.0, hp).astype(np.float32)
    sin_cuerpo = rng.random(N) < 0.10  # usuario muy cerca de la cámara
    pp[sin_cuerpo] = 0.0

    # 5) variación temporal: velocidad y "congelar" un cuadro
    idx = np.tile(np.arange(T), (N, 1))
    vel = rng.random(N) < 0.3
    if vel.any():
        f = rng.uniform(0.7, 1.0, vel.sum())
        off = rng.uniform(0, 1, vel.sum()) * (T - 1) * (1 - f)
        idx[vel] = np.clip(np.round(np.arange(T)[None] * f[:, None] + off[:, None]), 0, T - 1)
    cong = rng.random(N) < 0.2
    if cong.any():
        k = rng.integers(1, T, cong.sum())
        filas = np.where(cong)[0]
        idx[filas, k] = idx[filas, k - 1]
    filas = np.arange(N)[:, None]
    return {
        "hands": h[filas, idx], "hand_present": hp[filas, idx],
        "pose": p[filas, idx], "pose_present": pp[filas, idx], "aspect": a,
    }


def espejar_lote(d: dict) -> dict:
    """Copia espejada de todo el lote (aumentación para modelos clásicos)."""
    h = d["hands"].copy()
    h[..., 0] = 1.0 - h[..., 0]
    p = d["pose"].copy()
    p[..., 0] = 1.0 - p[..., 0]
    return {
        "hands": h[:, :, ::-1].copy(),
        "hand_present": d["hand_present"][:, :, ::-1].copy(),
        "pose": p[:, :, POSE_MIRROR_PERM].copy(),
        "pose_present": d["pose_present"].copy(),
        "aspect": d["aspect"].copy(),
    }
