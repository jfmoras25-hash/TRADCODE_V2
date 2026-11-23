"""Genera landmarks SINTÉTICOS con el mismo formato que LSC70 extraído.

Solo sirve para probar que el pipeline funciona de punta a punta antes de
tener el dataset real. NO usar sus resultados en el documento.

Uso:
    python scripts/generar_datos_sinteticos.py --personas 30
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML))
from lsc.dataset import guardar_npz, vacio  # noqa: E402
from lsc.features import SEQ_LEN, mirror_raw  # noqa: E402

CLASES = (list("ABCDEFGHIJKLMNÑOPQRSTUVWXYZ") + [str(i) for i in range(1, 11)]
          + ["HOLA", "GRACIAS", "POR_FAVOR", "BIEN", "NOMBRE", "AYUDA", "SI", "NO", "ADIOS", "COMO_ESTAS"])


def mano(curl, abre, rot, tam, centro, aspect, rng):
    """21 puntos (x, y, z) normalizados a la imagen."""
    pts = [np.zeros(3)]
    bases = np.array([[-0.35, -0.25], [-0.2, -0.85], [0.0, -0.9], [0.2, -0.85], [0.38, -0.7]])
    largos = np.array([0.35, 0.4, 0.42, 0.4, 0.32])
    for f in range(5):
        dir_ = np.arctan2(bases[f, 1], bases[f, 0]) + (f - 2) * abre
        p = np.array([bases[f, 0], bases[f, 1], 0.0])
        pts.append(p.copy())
        for j in range(3):
            ang = dir_ + curl[f] * (j + 1) * 0.9
            p = p + largos[f] * np.array([np.cos(ang), np.sin(ang), -0.3 * curl[f]])
            pts.append(p.copy())
    pts = np.array(pts)[:21]
    c, s = np.cos(rot), np.sin(rot)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    pts = pts @ R.T * tam + rng.normal(0, 0.01, (21, 3)) * tam
    out = np.empty_like(pts)
    out[:, 0] = centro[0] + pts[:, 0] / aspect
    out[:, 1] = centro[1] + pts[:, 1]
    out[:, 2] = pts[:, 2] * 0.3
    return out.astype(np.float32)


def pose(cx, cy, ancho, aspect, wr_r, wr_l, rng):
    P = np.zeros((33, 4), np.float32)
    P[:, 3] = 0.9
    P[:, :2] = [cx, cy]
    P[0, :2] = [cx, cy - 0.6 * ancho]
    for i in range(1, 11):
        P[i, :2] = [cx + rng.normal(0, 0.02), cy - 0.6 * ancho]
    P[11, :2] = [cx + ancho / 2 / aspect, cy]   # hombro izq. (aparece a la derecha de la imagen)
    P[12, :2] = [cx - ancho / 2 / aspect, cy]
    P[15, :2], P[16, :2] = wr_l, wr_r
    P[13, :2] = (P[11, :2] + P[15, :2]) / 2 + [0.03, 0.08]
    P[14, :2] = (P[12, :2] + P[16, :2]) / 2 + [-0.03, 0.08]
    P[17:23, :2] = np.repeat(P[15:17, :2], 3, axis=0)
    P[23:, :2] = [cx, cy + 1.2 * ancho]
    P[:, :2] += rng.normal(0, 0.004, (33, 2))
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", type=int, default=30)
    ap.add_argument("--repeticiones", type=int, default=2)
    ap.add_argument("--salida", type=Path, default=ML / "datos_prueba" / "sinteticos.npz")
    ap.add_argument("--semilla", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.semilla)

    # prototipo por clase
    proto = []
    for k, c in enumerate(CLASES):
        proto.append({
            "curl": rng.uniform(0, 1.6, 5), "abre": rng.uniform(0, 0.25), "rot": rng.uniform(-0.8, 0.8),
            "lugar": rng.uniform([-1.2, -1.4], [1.2, 0.6]),          # respecto a hombros, en anchos
            "mov": rng.normal(0, 0.5, 2) if (c in "JÑZ" or len(c) > 2) else np.zeros(2),
            "dos": len(c) > 2 and rng.random() < 0.5,
        })
    n = len(CLASES) * a.personas * a.repeticiones
    d = vacio(n)
    i = 0
    for per in range(a.personas):
        aspect = 4 / 3
        cx, cy = rng.uniform(0.4, 0.6), rng.uniform(0.55, 0.7)
        ancho = rng.uniform(0.28, 0.4)
        tam = ancho * rng.uniform(0.38, 0.48)
        zurdo = rng.random() < 0.12
        for k, c in enumerate(CLASES):
            pr = proto[k]
            for _ in range(a.repeticiones):
                curl = np.clip(pr["curl"] + rng.normal(0, 0.18, 5), 0, 2)
                rot = pr["rot"] + rng.normal(0, 0.15)
                lugar = pr["lugar"] + rng.normal(0, 0.15, 2)
                for t in range(SEQ_LEN):
                    f = t / (SEQ_LEN - 1)
                    off = (lugar + pr["mov"] * f) * ancho
                    wr = np.array([cx + off[0] / aspect, cy + off[1]])
                    d["hands"][i, t, 0] = mano(curl, pr["abre"], rot + 0.3 * f * np.linalg.norm(pr["mov"]),
                                               tam, wr, aspect, rng)
                    d["hand_present"][i, t, 0] = rng.random() > 0.03
                    wl = np.array([cx + 0.25, cy + 0.35])
                    if pr["dos"]:
                        wl = np.array([cx - off[0] / aspect, cy + off[1]])
                        d["hands"][i, t, 1] = mano(curl[::-1], pr["abre"], -rot, tam, wl, aspect, rng)
                        d["hand_present"][i, t, 1] = 1
                    d["pose"][i, t] = pose(cx, cy, ancho, aspect, wr, wl, rng)
                    d["pose_present"][i, t] = 1
                if zurdo:
                    h, hp, p, pp = mirror_raw(d["hands"][i], d["hand_present"][i], d["pose"][i], d["pose_present"][i])
                    d["hands"][i], d["hand_present"][i], d["pose"][i], d["pose_present"][i] = h, hp, p, pp
                d["aspect"][i] = aspect
                d["label"][i] = c
                d["group"][i] = f"Sint{per:02d}"
                d["source"][i] = "sintetico"
                i += 1
    guardar_npz(d, a.salida)
    print(f"Guardado {a.salida}: {n} secuencias, {len(CLASES)} clases, {a.personas} personas")


if __name__ == "__main__":
    main()
