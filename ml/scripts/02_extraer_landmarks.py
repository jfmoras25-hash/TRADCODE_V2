"""Paso 2: extraer landmarks de manos y cuerpo de cada imagen de LSC70.

Corre MediaPipe Hand Landmarker (2 manos) y Pose Landmarker sobre cada cuadro y
guarda los puntos crudos en ``datos/landmarks_lsc70.npz``. Se puede
interrumpir y reanudar: lo ya procesado queda en un archivo de caché.

Uso:
    python scripts/02_extraer_landmarks.py --raiz datos/LSC70
    python scripts/02_extraer_landmarks.py --raiz datos/LSC70 --procesos 4 --limite 200
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML))
from lsc.dataset import descubrir_lsc70, guardar_npz, indices_remuestreo, vacio  # noqa: E402
from lsc.features import SEQ_LEN  # noqa: E402

_det = None


def _init(min_conf):
    global _det
    from lsc.landmarks import DetectorLandmarks
    _det = DetectorLandmarks(min_conf=min_conf)


def _procesar(args):
    clave, etiqueta, grupo, seccion, rutas = args
    cuadros, aspect = [], 4 / 3
    for j in indices_remuestreo(len(rutas), SEQ_LEN):
        fr, aspect = _det.detectar_archivo(rutas[j])
        cuadros.append({
            "h": np.round(fr["hands"], 5).tolist(),
            "hp": fr["hand_present"].tolist(),
            "p": np.round(fr["pose"], 5).tolist() if fr["pose_present"] else None,
        })
    return {"clave": clave, "etiqueta": etiqueta, "grupo": grupo,
            "seccion": seccion, "aspect": aspect, "cuadros": cuadros}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raiz", type=Path, required=True)
    ap.add_argument("--secciones", nargs="*", default=["LSC70AN", "LSC70W"],
                    help="LSC70ANH (recortes de 120x120) se omite por defecto")
    ap.add_argument("--salida", type=Path, default=ML / "datos" / "landmarks_lsc70.npz")
    ap.add_argument("--procesos", type=int, default=2)
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--limite", type=int, default=0, help="procesar solo N secuencias (prueba)")
    a = ap.parse_args()

    muestras = descubrir_lsc70(a.raiz, a.secciones)
    if a.limite:
        muestras = muestras[:: max(1, len(muestras) // a.limite)][: a.limite]
    if not muestras:
        sys.exit("No se encontraron secuencias; ejecute primero 01_inspeccionar_dataset.py")

    cache = a.salida.with_suffix(".cache.jsonl")
    hechas = {}
    if cache.exists():
        for linea in cache.read_text(encoding="utf-8").splitlines():
            if linea.strip():
                r = json.loads(linea)
                hechas[r["clave"]] = r
    pendientes = [m for m in muestras if m.clave not in hechas]
    print(f"{len(muestras)} secuencias, {len(hechas)} en caché, {len(pendientes)} por procesar")

    t0 = time.time()
    trabajos = [(m.clave, m.etiqueta, m.grupo, m.seccion, [str(c) for c in m.cuadros])
                for m in pendientes]
    if trabajos:
        from lsc.landmarks import descargar_modelos
        descargar_modelos()  # una sola vez, antes de crear los procesos
        with open(cache, "a", encoding="utf-8") as fcache, \
                Pool(a.procesos, initializer=_init, initargs=(a.min_conf,)) as pool:
            for i, r in enumerate(pool.imap_unordered(_procesar, trabajos, chunksize=4), 1):
                fcache.write(json.dumps(r) + "\n")
                hechas[r["clave"]] = r
                if i % 50 == 0 or i == len(trabajos):
                    v = i / (time.time() - t0)
                    print(f"  {i}/{len(trabajos)}  ({v:.1f} seq/s, faltan ~{(len(trabajos) - i) / v / 60:.1f} min)")

    claves = [m.clave for m in muestras]
    d = vacio(len(claves))
    sin_manos = 0
    for i, k in enumerate(claves):
        r = hechas[k]
        for t, c in enumerate(r["cuadros"]):
            d["hands"][i, t] = np.asarray(c["h"], np.float32)
            d["hand_present"][i, t] = np.asarray(c["hp"], np.float32)
            if c["p"] is not None:
                d["pose"][i, t] = np.asarray(c["p"], np.float32)
                d["pose_present"][i, t] = 1
        d["aspect"][i] = r["aspect"]
        d["label"][i] = r["etiqueta"]
        d["group"][i] = r["grupo"]
        d["source"][i] = "lsc70:" + r["seccion"]
        sin_manos += d["hand_present"][i].sum() == 0
    guardar_npz(d, a.salida)

    hp = d["hand_present"].max(axis=2).mean()
    pp = d["pose_present"].mean()
    print(f"\nGuardado {a.salida}  ({len(claves)} secuencias)")
    print(f"Cuadros con al menos una mano: {hp:.1%} | con cuerpo: {pp:.1%}")
    print(f"Secuencias sin ninguna mano detectada: {sin_manos} (se descartan al entrenar)")


if __name__ == "__main__":
    main()
