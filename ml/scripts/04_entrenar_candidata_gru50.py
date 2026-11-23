"""Entrena y evalúa la candidata GRU de 50 señas sin escribir en web/public."""
import csv
import json
import runpy
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

ML = Path(__file__).resolve().parents[1]
RAIZ = ML.parent
sys.path[:0] = [str(ML), str(RAIZ)]

from backend.capturas import Capturas  # noqa: E402
from lsc.dataset import cargar_capturas_json, cargar_npz, concatenar, subconjunto  # noqa: E402
from lsc.features import batch_features  # noqa: E402

NUEVAS = ("ADIOS", "MI", "NOMBRE_ES")
SEMILLA = 42
EPOCAS = 30
DESTINO = ML / "candidatas" / "gru_50"
ACTUAL = RAIZ / "web" / "public" / "model"
CAMPOS = ("hands", "hand_present", "pose", "pose_present", "aspect")


def leer_capturas():
    partes = []
    nombres = []
    firmas = set()
    for archivo in sorted((ML / "datos" / "capturas").glob("*.json")):
        doc = Capturas.model_validate_json(archivo.read_text(encoding="utf-8"))
        for i, muestra in enumerate(doc.muestras):
            if muestra.etiqueta not in NUEVAS:
                continue
            datos = muestra.model_dump(mode="json", exclude={"fecha"})
            firma = json.dumps(datos, sort_keys=True)
            if firma in firmas:
                raise ValueError(f"Captura duplicada: {archivo.name}")
            firmas.add(firma)
            partes.append(subconjunto(cargar_capturas_json(archivo), [i]))
            nombres.append({"archivo": archivo.name, "indice": i, "etiqueta": muestra.etiqueta})
    capturas = concatenar(partes)
    assert {e: int((capturas["label"] == e).sum()) for e in NUEVAS} == dict.fromkeys(NUEVAS, 10)
    return capturas, nombres


def inferir(ruta, datos, config):
    opciones = ort.SessionOptions()
    opciones.intra_op_num_threads = 2
    opciones.inter_op_num_threads = 1
    sesion = ort.InferenceSession(str(ruta), opciones, providers=["CPUExecutionProvider"])
    x = batch_features({k: datos[k] for k in CAMPOS})
    pred = sesion.run([config["salida"]], {config["entrada"]: x})[0]
    assert pred.shape == (len(x), len(config["etiquetas"])) and np.isfinite(pred).all()
    np.testing.assert_allclose(pred.sum(1), 1, atol=1e-5)
    individual = sesion.run([config["salida"]], {config["entrada"]: x[:1]})[0]
    np.testing.assert_allclose(individual[0], pred[0], atol=1e-5)
    return np.asarray(config["etiquetas"])[pred.argmax(1)]


def metricas(y, pred, etiquetas):
    report = classification_report(y, pred, labels=etiquetas, output_dict=True, zero_division=0)
    return {"exactitud": float(accuracy_score(y, pred)), "f1_macro": report["macro avg"]["f1-score"],
            "clases": {e: report[e] for e in etiquetas}}


def csv_matriz(ruta, y, pred, etiquetas):
    matriz = confusion_matrix(y, pred, labels=etiquetas)
    with ruta.open("w", encoding="utf-8-sig", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["real / predicción", *etiquetas])
        escritor.writerows([e, *fila] for e, fila in zip(etiquetas, matriz))


def main():
    inicio = time.perf_counter()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    funciones = runpy.run_path(str(ML / "scripts" / "03_entrenar_comparar.py"))
    anterior = json.loads((ACTUAL / "config.json").read_text(encoding="utf-8"))
    originales = anterior["etiquetas"]
    etiquetas = sorted(set(originales) | set(NUEVAS))
    assert len(originales) == 47 and len(etiquetas) == 50
    base = cargar_npz([ML / "datos" / "landmarks_lsc70.npz"])
    caps, origen = leer_capturas()
    total = len(base["label"]) + len(caps["label"])
    assert total == 3314 and set(base["label"]) == set(originales)
    base = subconjunto(base, base["hand_present"].any(axis=(1, 2)))
    manos = caps["hand_present"].any(axis=2).sum(axis=1)
    assert (manos > 0).all() and len(set(caps["group"])) == 1

    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=.15, random_state=SEMILLA)
                  .split(base["label"], groups=base["group"]))
    assert not (set(base["group"][tr]) & set(base["group"][va]))
    rng = np.random.default_rng(SEMILLA)
    grupos_train, grupos_val = [], []
    detalle = {}
    for etiqueta in NUEVAS:
        aptas = np.flatnonzero((caps["label"] == etiqueta) & (manos >= 3))
        orden = rng.permutation(aptas)
        assert len(orden) >= 4
        evaluacion, entrenamiento = orden[:2], orden[2:]
        # Repetir solo muestras del entrenamiento reduce el desbalance; cada
        # ocurrencia pasa por la aumentación del proyecto en cada época.
        grupos_train.extend(np.resize(entrenamiento, 24))
        grupos_val.extend(evaluacion)
        detalle[etiqueta] = {
            "entrenamiento": [origen[i] for i in entrenamiento],
            "validacion": [origen[i] for i in evaluacion],
            "entrenamiento_por_epoca_con_repeticion": 24,
        }

    d_tr = concatenar([subconjunto(base, tr), subconjunto(caps, grupos_train)])
    d_va = concatenar([subconjunto(base, va), subconjunto(caps, grupos_val)])
    y_tr = np.array([etiquetas.index(e) for e in d_tr["label"]])
    y_va = np.array([etiquetas.index(e) for e in d_va["label"]])
    assert len(np.unique(y_tr)) == len(np.unique(y_va)) == 50
    excluidas = [{**origen[i], "cuadros_con_manos": int(manos[i])} for i in np.flatnonzero(manos < 3)]
    print(f"Antes de filtros: {total}; LSC70 útil: {len(base['label'])}; excluidas por calidad: {excluidas}")
    print(f"Entrenamiento: {len(tr)} originales + {len(grupos_train)} con repetición; validación: {len(va)} originales + 6 nuevas")
    DESTINO.mkdir(parents=True, exist_ok=False)

    t0 = time.perf_counter()
    modelo, historial = funciones["entrenar_red"](
        "GRU", d_tr, y_tr, len(etiquetas), EPOCAS, SEMILLA, d_va, y_va)
    tiempo_entrenamiento = time.perf_counter() - t0
    print(f"Entrenamiento terminado en {tiempo_entrenamiento:.1f} s. Exportando…")
    mejor = max(range(EPOCAS), key=lambda i: (historial["f1_val"][i], i))
    funciones["exportar_onnx"](modelo, etiquetas, DESTINO, {
        "modelo": "GRU", "f1_validacion": round(historial["f1_val"][mejor], 4), "comparacion": [],
        "personas": len(set(base["group"])) + 1, "secuencias": len(base["label"]) + int((manos >= 3).sum()),
    })
    config = json.loads((DESTINO / "config.json").read_text(encoding="utf-8"))
    assert config["etiquetas"] == etiquetas and set(config["etiquetas"]) - set(originales) == set(NUEVAS)
    grafo = onnx.load(DESTINO / "modelo.onnx")
    onnx.checker.check_model(grafo)
    assert any(n.op_type == "GRU" for n in grafo.graph.node)
    pred = inferir(DESTINO / "modelo.onnx", d_va, config)
    pred_anterior = inferir(ACTUAL / "modelo.onnx", subconjunto(base, va), anterior)
    reales = d_va["label"]
    metricas_total = metricas(reales, pred, etiquetas)
    metricas_anteriores = metricas(reales[:len(va)], pred_anterior, originales)
    metricas_originales = metricas(reales[:len(va)], pred[:len(va)], originales)
    nuevas_sobre_originales = int(np.isin(pred[:len(va)], NUEVAS).sum())

    with (DESTINO / "metricas_por_clase.csv").open("w", encoding="utf-8-sig", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(("clase", "precision_candidata", "recall_candidata", "f1_candidata", "soporte_candidata",
                           "precision_anterior_originales", "recall_anterior_originales", "f1_anterior_originales",
                           "precision_candidata_originales", "recall_candidata_originales", "f1_candidata_originales",
                           "delta_f1_originales"))
        campos = ("precision", "recall", "f1-score", "support")
        for e in etiquetas:
            total_clase = metricas_total["clases"][e]
            fila = [e, *(total_clase[c] for c in campos)]
            if e in originales:
                v0 = metricas_anteriores["clases"][e]
                v1 = metricas_originales["clases"][e]
                fila += [*(v0[c] for c in campos[:3]), *(v1[c] for c in campos[:3]), v1["f1-score"] - v0["f1-score"]]
            escritor.writerow(fila)

    csv_matriz(DESTINO / "matriz_confusion.csv", reales, pred, etiquetas)
    csv_matriz(DESTINO / "matriz_confusion_anterior.csv", reales[:len(va)], pred_anterior, originales)
    indices = {etiqueta: i for i, etiqueta in enumerate(etiquetas)}
    funciones["graficar_confusion"](y_va, [indices[e] for e in pred], etiquetas,
                                 DESTINO / "matriz_confusion.png", "GRU candidata · validación de 50 señas")
    funciones["graficar_curvas"](historial, DESTINO / "curvas_entrenamiento.png", "GRU 50")
    resumen = {
        "semilla": SEMILLA, "epocas": EPOCAS, "epoca_exportada": mejor + 1, "hilos_cpu": 2,
        "secuencias_antes_de_filtros": total, "lsc70_sin_manos": 3284 - len(base["label"]),
        "capturas_excluidas_por_calidad": excluidas, "reparto_capturas": detalle,
        "lsc70_entrenamiento": len(tr), "lsc70_validacion": len(va),
        "participantes_lsc70_entrenamiento": len(set(base["group"][tr])),
        "participantes_lsc70_validacion": len(set(base["group"][va])),
        "validacion_nuevas": "Muestras distintas de un mismo participante; evaluación no independiente por persona.",
        "candidata_total": metricas_total, "anterior_en_lsc70": metricas_anteriores,
        "candidata_en_lsc70": metricas_originales, "predicciones_nuevas_en_lsc70": nuevas_sobre_originales,
        "historial": historial, "segundos_entrenamiento": tiempo_entrenamiento,
        "segundos_totales": time.perf_counter() - inicio,
    }
    (DESTINO / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ONNX con 50 probabilidades verificado. Resultados: {DESTINO}")
    for k in ("anterior_en_lsc70", "candidata_en_lsc70", "candidata_total"):
        print(k, {m: resumen[k][m] for m in ("exactitud", "f1_macro")})


if __name__ == "__main__":
    main()
