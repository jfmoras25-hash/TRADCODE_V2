"""Evaluador local del Random Forest recuperado, sin conexión a TRADCODE."""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier

MODELO = Path(__file__).with_name("modelo_senas.pkl")
SHA256_MODELO = "f93dab3dc5c5a7fe666d322dd4ec56e16d61dfa277bd057e7b72ac251d169a2e"
UMBRAL_ANTIGUO = 0.70


def normalizar(coordenadas, mano="derecha") -> np.ndarray:
    """Recibe 63 coordenadas crudas intercaladas x, y, z en orden MediaPipe."""
    puntos = np.asarray(coordenadas, dtype=np.float64)
    if puntos.shape != (63,) or not np.isfinite(puntos).all():
        raise ValueError("Se necesitan exactamente 63 coordenadas finitas.")
    if mano not in ("derecha", "izquierda"):
        raise ValueError("La mano debe ser derecha o izquierda.")
    puntos = puntos.reshape(21, 3).copy()
    if mano == "izquierda":
        puntos[:, 0] = 2 * puntos[0, 0] - puntos[:, 0]
    puntos = puntos - puntos[0]
    escala = np.linalg.norm(puntos[9])
    if escala > 0:
        puntos = puntos / escala
    return puntos.ravel()


def mano_de_cuadro(manos):
    """Selecciona Right o Left siguiendo la prioridad del traductor antiguo."""
    if len(manos) != 2:
        raise ValueError("El cuadro debe contener dos ranuras: derecha e izquierda.")
    for lado, puntos in zip(("derecha", "izquierda"), manos):
        if puntos is None:
            continue
        puntos = np.asarray(puntos, dtype=np.float64)
        if puntos.shape != (21, 3):
            raise ValueError(f"La mano {lado} debe contener 21 triples x, y, z.")
        return puntos.ravel(), lado
    raise ValueError("El cuadro no contiene ninguna mano detectada.")


class ReconocedorLegado:
    def __init__(self):
        if sklearn.__version__ != "1.9.1":
            raise RuntimeError("Este archivo fue guardado con scikit-learn 1.9.1.")
        contenido = MODELO.read_bytes()
        if hashlib.sha256(contenido).hexdigest() != SHA256_MODELO:
            raise ValueError("El archivo no coincide con el modelo antiguo verificado.")
        self.bosque = pickle.loads(contenido)
        if not isinstance(self.bosque, RandomForestClassifier) or self.bosque.n_features_in_ != 63:
            raise ValueError("Se esperaba el Random Forest de 63 características.")

    def puntuaciones(self, coordenadas, mano="derecha"):
        probabilidades = self.bosque.predict_proba([normalizar(coordenadas, mano)])[0]
        return [{"etiqueta": str(e), "p": float(p)} for e, p in zip(self.bosque.classes_, probabilidades)]

    def evaluar(self, coordenadas, mano="derecha"):
        mejor = max(self.puntuaciones(coordenadas, mano), key=lambda c: c["p"])
        confianza = mejor["p"]
        return {
            "etiqueta": mejor["etiqueta"],
            "confianza": confianza,
            "aceptada_con_umbral_antiguo": confianza >= UMBRAL_ANTIGUO,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    origen = parser.add_mutually_exclusive_group(required=True)
    origen.add_argument("--coordenadas", type=Path, help="JSON con un vector de 63 valores. Usa - para stdin.")
    origen.add_argument("--captura", type=Path, help="JSON de Descargar JSON en TRADCODE.")
    origen.add_argument("--clases", action="store_true")
    parser.add_argument("--mano", choices=("derecha", "izquierda"), default="derecha", help="Solo para --coordenadas.")
    parser.add_argument("--muestra", type=int, default=1, help="Número de muestra en el JSON, desde 1.")
    parser.add_argument("--cuadro", type=int, default=4, help="Número de cuadro de la muestra, desde 1.")
    args = parser.parse_args()

    try:
        rf = ReconocedorLegado()
        if args.clases:
            resultado = {"clases": rf.bosque.classes_.tolist()}
        elif args.captura:
            doc = json.loads(args.captura.read_text(encoding="utf-8-sig"))
            if doc.get("formato") != "lsc-capturas":
                raise ValueError("El JSON no tiene el formato de capturas de TRADCODE.")
            if args.muestra < 1 or args.cuadro < 1:
                raise ValueError("Los números de muestra y cuadro empiezan en 1.")
            muestra = doc["muestras"][args.muestra - 1]
            puntos, mano = mano_de_cuadro(muestra["cuadros"][args.cuadro - 1]["manos"])
            resultado = {**rf.evaluar(puntos, mano), "mano": mano, "etiqueta_anotada": muestra["etiqueta"]}
        else:
            texto = sys.stdin.read() if str(args.coordenadas) == "-" else args.coordenadas.read_text(encoding="utf-8-sig")
            resultado = rf.evaluar(json.loads(texto), args.mano)
    except (OSError, ValueError, RuntimeError, KeyError, IndexError, TypeError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(json.dumps(resultado, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
