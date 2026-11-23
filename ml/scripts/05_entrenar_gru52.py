"""Audita y entrena manualmente GRU 52 a partir de la ONNX de GRU 50."""
import argparse
import csv
import hashlib
import json
import runpy
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper
from pydantic import ValidationError
from sklearn.model_selection import GroupShuffleSplit

ML = Path(__file__).resolve().parents[1]
RAIZ = ML.parent
sys.path[:0] = [str(ML), str(RAIZ)]

from backend.capturas import Capturas  # noqa: E402
from lsc.dataset import cargar_capturas_json, cargar_npz, concatenar, subconjunto  # noqa: E402
from lsc.features import FEATURE_VERSION, N_FEATURES, SEQ_LEN, batch_features  # noqa: E402
from lsc.models import REDES  # noqa: E402

ANTERIOR = ML / "candidatas" / "gru_50"
NUEVAS = ("DEDO_MEDIO", "HIJUEPUTA")
PREVIAS = ("ADIOS", "MI", "NOMBRE_ES")
SEMILLA = 42
EPOCAS = 30
LR_MAX = 8e-4
CLASE_OBJETIVO = 48


def sha256(ruta):
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def leer_json(ruta):
    return json.loads(ruta.read_text(encoding="utf-8"))


def escribir_json(ruta, obj):
    ruta.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def leer_capturas():
    aptas = {}
    excluidas = []
    firmas = {}
    encontradas = Counter()
    fuentes = (("capturas", PREVIAS), ("capturas_hibrido", NUEVAS))
    for carpeta, etiquetas in fuentes:
        for archivo in sorted((ML / "datos" / carpeta).glob("*.json")):
            origen = {"carpeta": carpeta, "archivo": archivo.name}
            try:
                documento = leer_json(archivo)
                muestras = documento["muestras"]
                if not isinstance(muestras, list) or not muestras:
                    raise ValueError("El JSON debe contener una lista de muestras")
            except (OSError, ValueError, KeyError, TypeError) as error:
                excluidas.append({**origen, "motivo": str(error)})
                continue
            validadas = []
            for indice, muestra in enumerate(muestras):
                referencia = {**origen, "indice": indice}
                etiqueta = muestra.get("etiqueta") if isinstance(muestra, dict) else None
                if isinstance(etiqueta, str):
                    encontradas[etiqueta] += 1
                try:
                    cabecera = {**documento, "muestras": [muestra]}
                    valida = Capturas.model_validate(cabecera).muestras[0]
                except ValidationError as error:
                    fallos = [f"{e['loc']}: {e['msg']}" for e in error.errors(include_input=False)]
                    excluidas.append({**referencia, "etiqueta": etiqueta, "motivo": "; ".join(fallos)})
                    continue
                if valida.etiqueta not in etiquetas:
                    excluidas.append({**referencia, "etiqueta": valida.etiqueta,
                                      "motivo": "Etiqueta fuera del conjunto solicitado para esta carpeta"})
                    continue
                cuadros_con_manos = sum(any(c.manos) for c in valida.cuadros)
                referencia.update(etiqueta=valida.etiqueta, cuadros_con_manos=cuadros_con_manos,
                                  archivo_sha256=sha256(archivo))
                if cuadros_con_manos < 3:
                    excluidas.append({**referencia, "motivo": "Menos de tres cuadros con manos"})
                    continue
                degenerada = any(np.linalg.norm(np.subtract(mano[9], mano[0])) < 1e-6
                                 for cuadro in valida.cuadros for mano in cuadro.manos if mano is not None)
                if degenerada:
                    excluidas.append({**referencia, "motivo": "Muñeca y base del dedo medio coincidentes"})
                    continue
                validadas.append((referencia, valida.fecha))
            # El importador original lee el archivo entero. Si una muestra inválida
            # impide hacerlo, se informa el archivo en vez de introducir datos parciales.
            if not validadas:
                continue
            try:
                datos = cargar_capturas_json(archivo)
            except (AttributeError, ValueError, KeyError, IndexError, TypeError) as error:
                excluidas.append({**origen, "motivo": f"El importador no pudo leer el archivo: {error}"})
                continue
            for referencia, fecha in validadas:
                indice = referencia["indice"]
                fila = subconjunto(datos, [indice])
                etiqueta = referencia["etiqueta"]
                if fila["label"][0] != etiqueta or not np.isfinite(batch_features(crudos(fila))).all():
                    excluidas.append({**referencia, "motivo": "Etiqueta o características incompatibles"})
                    continue
                firma = hashlib.sha256(b"".join(arr.tobytes() for arr in crudos(fila).values())).hexdigest()
                if firma in firmas:
                    excluidas.append({**referencia, "motivo": "Landmarks duplicados", "original": firmas[firma]})
                    continue
                firmas[firma] = referencia
                aptas[carpeta, archivo.name, indice] = (fila, referencia, fecha)
    return aptas, excluidas, encontradas


def repartir(aptas, resumen_anterior):
    entrenamiento = []
    validacion = []
    detalle = {}
    for etiqueta in PREVIAS:
        registro = resumen_anterior["reparto_capturas"][etiqueta]
        tr, va = ([aptas["capturas", i["archivo"], i["indice"]] for i in registro[grupo]]
                  for grupo in ("entrenamiento", "validacion"))
        if any(fila["label"][0] != etiqueta for fila, _, _ in tr + va):
            raise ValueError(f"Captura con etiqueta distinta al reparto de GRU 50: {etiqueta}")
        detalle[etiqueta] = (tr, va)
    for etiqueta in NUEVAS:
        filas = sorted((r for r in aptas.values() if r[1]["etiqueta"] == etiqueta),
                       key=lambda r: (r[2], r[1]["archivo"], r[1]["indice"]))
        if len(filas) < 5:
            raise ValueError(f"{etiqueta} tiene {len(filas)} muestras aptas. Se requieren al menos 5.")
        grupos = sorted({str(fila["group"][0]) for fila, _, _ in filas})
        if len(grupos) == 1:
            n_val = max(2, len(filas) // 5)
            detalle[etiqueta] = (filas[:-n_val], filas[-n_val:])
        else:
            tr, va = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEMILLA)
                          .split(filas, groups=[fila["group"][0] for fila, _, _ in filas]))
            detalle[etiqueta] = ([filas[i] for i in tr], [filas[i] for i in va])
    for tr, va in detalle.values():
        entrenamiento.extend(tr)
        validacion.extend(va)
    personas_tr = {str(fila["group"][0]) for fila, _, _ in entrenamiento}
    return entrenamiento, validacion, {
        e: {
            "entrenamiento": [ref for _, ref, _ in tr],
            "validacion": [ref for _, ref, _ in va],
            "personas_validacion_tambien_en_entrenamiento": sorted(
                {str(fila["group"][0]) for fila, _, _ in va} & personas_tr),
        }
        for e, (tr, va) in detalle.items()
    }


def sesion_onnx(ruta):
    opciones = ort.SessionOptions()
    opciones.intra_op_num_threads = 2
    opciones.inter_op_num_threads = 1
    return ort.InferenceSession(str(ruta), opciones, providers=["CPUExecutionProvider"])


def iniciar_desde_50(grafo, config, etiquetas, x):
    """Invierte el empaquetado ONNX de una GRU r,z,n de PyTorch."""
    tensores = {v.name: numpy_helper.to_array(v) for v in grafo.graph.initializer}
    gru, = (n for n in grafo.graph.node if n.op_type == "GRU")
    proyeccion, = (n for n in grafo.graph.node if n.op_type == "MatMul")
    atributos = {a.name: onnx.helper.get_attribute_value(a) for a in gru.attribute}
    if atributos.get("direction") != b"bidirectional" or atributos.get("linear_before_reset") != 1:
        raise ValueError("La GRU ONNX no usa las convenciones de la GRU bidireccional de PyTorch")

    torch.manual_seed(SEMILLA)
    modelo = REDES["GRU"](N_FEATURES, len(etiquetas), SEQ_LEN)
    estado = modelo.state_dict()
    for clave in ("norm.media", "norm.desv", "proy.0.bias"):
        estado[clave].copy_(torch.tensor(tensores["m." + clave]))
    estado["proy.0.weight"].copy_(torch.tensor(tensores[proyeccion.input[1]].T))
    for direccion, sufijo in enumerate(("", "_reverse")):
        w, r, b = (tensores[nombre][direccion] for nombre in gru.input[1:4])
        bias_w, bias_r = np.split(b, 2)
        for clave, valor in zip(("weight_ih", "weight_hh", "bias_ih", "bias_hh"), (w, r, bias_w, bias_r)):
            z, reset, candidato = np.split(valor, 3, axis=0)
            estado[f"rnn.{clave}_l0{sufijo}"].copy_(torch.tensor(np.concatenate([reset, z, candidato])))
    for clave in ("cabeza.1.weight", "cabeza.1.bias"):
        estado[clave][:len(config["etiquetas"])].copy_(torch.tensor(tensores["m." + clave]))
    modelo.load_state_dict(estado)
    sesion = sesion_onnx(ANTERIOR / "modelo.onnx")
    probabilidades_50 = sesion.run([config["salida"]], {config["entrada"]: x})[0]
    with torch.no_grad():
        logits_50 = modelo.eval()(torch.from_numpy(x))[:, :len(config["etiquetas"])]
        reconstruidas = logits_50.softmax(dim=1).numpy()
    np.testing.assert_allclose(reconstruidas, probabilidades_50, atol=1e-6, rtol=1e-5)
    return modelo, probabilidades_50, float(np.max(np.abs(reconstruidas - probabilidades_50)))


funciones = runpy.run_path(str(ML / "scripts" / "03_entrenar_comparar.py"))
herramientas = runpy.run_path(str(ML / "scripts" / "04_entrenar_candidata_gru50.py"))
crudos = funciones["crudos"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auditar", action="store_true", help="Valida las capturas sin entrenar ni guardar")
    parser.add_argument("--salida", default="gru_52", help="Nombre nuevo dentro de ml/candidatas")
    args = parser.parse_args()
    if not args.salida.startswith("gru_52") or not args.salida.replace("_", "").isalnum():
        parser.error("--salida debe ser un nombre de carpeta nuevo con prefijo gru_52")
    destino = ML / "candidatas" / args.salida
    if not args.auditar and destino.exists():
        parser.error(f"{destino} ya existe. Usa un nuevo nombre con --salida.")

    aptas, excluidas, encontradas = leer_capturas()
    validas = Counter(r[1]["etiqueta"] for r in aptas.values())
    print(json.dumps({"encontradas": encontradas, "validas": validas, "excluidas": excluidas},
                     ensure_ascii=False, indent=2), flush=True)
    if args.auditar:
        return

    reloj = time.perf_counter()
    config = leer_json(ANTERIOR / "config.json")
    resumen_anterior = leer_json(ANTERIOR / "resumen.json")
    originales = config["etiquetas"]
    if (config["modelo"], config["version_caracteristicas"], config["n_caracteristicas"], config["seq_len"]) != (
            "GRU", FEATURE_VERSION, N_FEATURES, SEQ_LEN) or len(set(originales)) != 50 or set(originales) & set(NUEVAS):
        raise ValueError("La configuración de la GRU de 50 clases es incompatible")
    etiquetas = [*originales, *NUEVAS]
    base = cargar_npz([ML / "datos" / "landmarks_lsc70.npz"])
    if set(base["label"]) | set(PREVIAS) != set(originales):
        raise ValueError("El dataset y las tres capturas previas no cubren exactamente las 50 clases")
    total_lsc70 = len(base["label"])
    base = subconjunto(base, base["hand_present"].any(axis=(1, 2)))
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=.15, random_state=SEMILLA)
                  .split(base["label"], groups=base["group"]))
    assert len(tr) == resumen_anterior["lsc70_entrenamiento"] and len(va) == resumen_anterior["lsc70_validacion"]
    assert not set(base["group"][tr]) & set(base["group"][va])
    cap_tr, cap_va, detalle = repartir(aptas, resumen_anterior)
    d_tr = concatenar([subconjunto(base, tr), *(fila for fila, _, _ in cap_tr)])
    d_va = concatenar([subconjunto(base, va), *(fila for fila, _, _ in cap_va)])
    indices = {e: i for i, e in enumerate(etiquetas)}
    y_tr = np.array([indices[e] for e in d_tr["label"]])
    y_va = np.array([indices[e] for e in d_va["label"]])
    assert set(y_tr) == set(y_va) == set(indices.values())

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    x_val = batch_features(crudos(d_va))
    grafo = onnx.load(ANTERIOR / "modelo.onnx")
    onnx.checker.check_model(grafo)
    modelo, prob50, diferencia = iniciar_desde_50(grafo, config, etiquetas, x_val)
    conteos = np.bincount(y_tr, minlength=len(etiquetas))
    pesos = np.ones(len(etiquetas), dtype=np.float32)
    for etiqueta in (*PREVIAS, *NUEVAS):
        i = indices[etiqueta]
        pesos[i] = max(1, CLASE_OBJETIVO / conteos[i])
    print(f"Iniciando ajuste desde GRU 50: {len(y_tr)} muestras train, {len(y_va)} val; "
          f"diferencia al recuperar pesos: {diferencia:.2g}", flush=True)
    destino.mkdir(parents=True, exist_ok=False)
    inicio = time.perf_counter()
    modelo, historial = funciones["entrenar_red"](
        "GRU", d_tr, y_tr, len(etiquetas), EPOCAS, SEMILLA, d_va, y_va,
        modelo_inicial=modelo, lr_max=LR_MAX, pesos_clase=torch.from_numpy(pesos))
    segundos_entrenamiento = time.perf_counter() - inicio
    mejor = max(range(EPOCAS), key=lambda i: (historial["f1_val"][i], i))
    escribir_json(destino / "historial.json", historial)
    torch.save(modelo.state_dict(), destino / "pesos.pt")
    funciones["exportar_onnx"](modelo, etiquetas, destino, {
        "modelo": "GRU", "nombres": config.get("nombres", {}),
        "f1_validacion": round(historial["f1_val"][mejor], 4),
        "personas": len(set(base["group"]) | set(d_tr["group"]) | set(d_va["group"])),
        "secuencias": len(y_tr) + len(y_va),
    })
    nueva = leer_json(destino / "config.json")
    assert nueva["etiquetas"] == etiquetas
    onnx.checker.check_model(destino / "modelo.onnx")
    pred = herramientas["inferir"](destino / "modelo.onnx", d_va, nueva)
    reales = d_va["label"]
    metricas = herramientas["metricas"]
    resultados = metricas(reales, pred, etiquetas)
    mascara = np.isin(reales, originales)
    anterior = metricas(reales[mascara], np.asarray(originales)[prob50[mascara].argmax(1)], originales)
    actuales = metricas(reales[mascara], pred[mascara], originales)
    errores = Counter((str(real), str(predicha)) for real, predicha in zip(reales, pred) if real != predicha)
    herramientas["csv_matriz"](destino / "matriz_confusion.csv", reales, pred, etiquetas)
    funciones["graficar_confusion"](y_va, [indices[e] for e in pred], etiquetas,
                                 destino / "matriz_confusion.png", "GRU 52 · Validación")
    funciones["graficar_curvas"](historial, destino / "curvas_entrenamiento.png", "GRU 52")
    with (destino / "metricas_por_clase.csv").open("w", encoding="utf-8-sig", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(("etiqueta", "entrenamiento", "validacion", "precision", "recall", "f1"))
        for i, e in enumerate(etiquetas):
            m = resultados["clases"][e]
            escritor.writerow((e, conteos[i], int(m["support"]), m["precision"], m["recall"], m["f1-score"]))
    with (destino / "errores_validacion.csv").open("w", encoding="utf-8-sig", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(("real", "prediccion", "casos"))
        escritor.writerows([*par, n] for par, n in errores.most_common())
    escribir_json(destino / "resumen.json", {
        "origen": "ml/candidatas/gru_50",
        "origen_sha256": {n: sha256(ANTERIOR / n) for n in ("modelo.onnx", "config.json", "resumen.json")},
        "exportacion_sha256": {n: sha256(destino / n) for n in ("modelo.onnx", "config.json")},
        "paridad_al_recuperar_pesos": diferencia,
        "semilla": SEMILLA, "epocas": EPOCAS, "epoca_exportada": mejor + 1, "hilos_cpu": 2,
        "lote": 64, "lr_max": LR_MAX, "lr_scheduler": "OneCycleLR", "optimizador": "AdamW",
        "weight_decay": 1e-4, "label_smoothing": 0.1, "grad_norm_max": 2.0,
        "peso_por_etiqueta": dict(zip(etiquetas, pesos.tolist())),
        "clases": len(etiquetas), "capturas_encontradas": encontradas, "capturas_validas": validas,
        "capturas_excluidas": excluidas, "reparto_capturas": detalle,
        "lsc70_muestras_sin_manos": total_lsc70 - len(base["label"]),
        "lsc70_personas_entrenamiento": sorted(set(base["group"][tr])),
        "lsc70_personas_validacion": sorted(set(base["group"][va])),
        "muestras_entrenamiento": len(y_tr), "muestras_validacion": len(y_va),
        "muestras_entrenamiento_por_etiqueta": dict(zip(etiquetas, conteos.tolist())),
        "validacion": resultados, "gru_50_en_las_50_clases": anterior, "gru_52_en_las_50_clases": actuales,
        "segundos_entrenamiento": segundos_entrenamiento, "segundos_totales": time.perf_counter() - reloj,
    })
    print(f"{destino.name}: {len(etiquetas)} clases, {EPOCAS} épocas, mejor {mejor + 1}. "
          f"Exactitud {resultados['exactitud']:.2%}, F1 macro {resultados['f1_macro']:.2%}.", flush=True)


if __name__ == "__main__":
    main()
