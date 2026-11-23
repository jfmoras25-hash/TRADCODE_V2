"""Convierte capturas descargadas desde la app (modo "Capturar") a .npz.

Úselo para agregar señas que no están en LSC70 o para sumar ejemplos
grabados con su propia cámara (mejora mucho la demo en vivo).

Uso:
    python scripts/importar_capturas.py capturas_jorge.json capturas_ana.json
    -> crea ml/datos/capturas.npz, que 03_entrenar_comparar.py carga solo.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML))
from lsc.dataset import cargar_capturas_json, concatenar, guardar_npz  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archivos", nargs="+", type=Path)
    ap.add_argument("--salida", type=Path, default=ML / "datos" / "capturas.npz")
    a = ap.parse_args()
    d = concatenar([cargar_capturas_json(r) for r in a.archivos])
    guardar_npz(d, a.salida)
    print(f"Guardado {a.salida}: {len(d['label'])} secuencias")
    print("Por seña:", dict(sorted(Counter(d["label"]).items())))
    print("Por persona:", dict(Counter(d["group"])))


if __name__ == "__main__":
    main()
