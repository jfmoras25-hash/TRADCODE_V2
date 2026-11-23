"""Ida y vuelta: JSON con el formato que descarga la app -> importador -> mismas características."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML))
from lsc.dataset import cargar_capturas_json, cargar_npz  # noqa: E402
from lsc.features import batch_features  # noqa: E402


def test_ida_y_vuelta():
    d = cargar_npz([ML / "datos_prueba" / "sinteticos.npz"])
    idx = np.arange(0, len(d["label"]), 97)[:20]
    muestras = []
    for i in idx:
        cuadros = []
        for t in range(6):
            cuadros.append({
                "manos": [d["hands"][i, t, s].tolist() if d["hand_present"][i, t, s] else None for s in range(2)],
                "pose": d["pose"][i, t].tolist() if d["pose_present"][i, t] else None,
            })
        muestras.append({"etiqueta": d["label"][i], "persona": "Prueba", "aspect": float(d["aspect"][i]),
                         "fecha": "2026-11-16", "cuadros": cuadros})
    doc = {"formato": "lsc-capturas", "version": 1, "seq_len": 6, "version_caracteristicas": 1, "muestras": muestras}
    with tempfile.TemporaryDirectory() as tmp:
        ruta = Path(tmp) / "c.json"
        ruta.write_text(json.dumps(doc))
        c = cargar_capturas_json(ruta)
    sub = {k: d[k][idx] for k in d}
    sub["hand_present"] = (sub["hand_present"] > 0.5).astype(np.float32)
    assert np.abs(batch_features(c) - batch_features(sub)).max() < 1e-5
    assert list(c["label"]) == list(sub["label"])
    assert set(c["group"]) == {"cap_prueba"}


if __name__ == "__main__":
    test_ida_y_vuelta()
    print("OK")
