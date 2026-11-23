"""Genera casos de prueba para verificar que features.ts == features.py.

Escribe web/src/lib/paridad.casos.json; la prueba ``npm test`` de la web lo lee.
"""
import json
import sys
from pathlib import Path

import numpy as np

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML))
from lsc.augment import aumentar_lote  # noqa: E402
from lsc.features import frame_features  # noqa: E402

rng = np.random.default_rng(7)
casos = []
for i in range(60):
    T = 6
    d = {
        "hands": rng.uniform(0.1, 0.9, (1, T, 2, 21, 3)).astype(np.float32),
        "hand_present": (rng.random((1, T, 2)) < 0.7).astype(np.float32),
        "pose": rng.uniform(0.1, 0.9, (1, T, 33, 4)).astype(np.float32),
        "pose_present": (rng.random((1, T)) < 0.8).astype(np.float32),
        "aspect": np.array([rng.choice([4 / 3, 16 / 9, 3 / 4])], np.float32),
    }
    d["hands"][..., 2] -= 0.5
    if i % 3 == 0:
        d = aumentar_lote(d, rng)
    aspect = float(d["aspect"][0])
    for t in range(T):
        h, hp, p, pp = d["hands"][0, t], d["hand_present"][0, t], d["pose"][0, t], d["pose_present"][0, t]
        casos.append({
            "aspect": aspect,
            "cuadro": {
                "manos": [h[s].tolist() if hp[s] > 0.5 else None for s in range(2)],
                "pose": p.tolist() if pp > 0.5 else None,
            },
            "esperado": frame_features(h, hp, p, pp, aspect).tolist(),
        })

dest = ML.parent / "web" / "src" / "lib" / "paridad.casos.json"
dest.write_text(json.dumps(casos))
print(f"{len(casos)} casos -> {dest}")
