"""Paso 1: revisar que el dataset LSC70 se lee bien ANTES de extraer.

Uso:
    python scripts/01_inspeccionar_dataset.py --raiz datos/LSC70
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lsc.dataset import descubrir_lsc70  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raiz", type=Path, required=True, help="carpeta donde se descomprimió LSC70")
    ap.add_argument("--secciones", nargs="*", default=None,
                    help="p. ej. LSC70AN LSC70W (por defecto todas)")
    a = ap.parse_args()

    muestras = descubrir_lsc70(a.raiz, a.secciones)
    if not muestras:
        print("No se encontraron imágenes. Revise la ruta --raiz.")
        return
    print(f"Secuencias encontradas: {len(muestras)}")
    print("Por sección:", dict(Counter(m.seccion or "(sin sección)" for m in muestras)))
    print("Voluntarios:", len({m.grupo for m in muestras}))
    print("Cuadros por secuencia:", dict(sorted(Counter(len(m.cuadros) for m in muestras).items())))
    por_etq = Counter(m.etiqueta for m in muestras)
    print(f"Señas distintas: {len(por_etq)}")
    for etq, n in sorted(por_etq.items()):
        print(f"  {etq!r:>14}: {n}")
    print("\nEjemplos:")
    for m in muestras[:: max(1, len(muestras) // 5)][:5]:
        print(f"  [{m.seccion}] {m.grupo} / {m.etiqueta}: {[c.name for c in m.cuadros]}")


if __name__ == "__main__":
    main()
