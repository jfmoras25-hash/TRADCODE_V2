"""Paso 3: comparar algoritmos, entrenar el modelo final y exportarlo a ONNX.

1. Validación cruzada agrupada por persona (GroupKFold): ninguna persona
   aparece a la vez en entrenamiento y prueba, así se mide cómo funciona con
   alguien NUEVO frente a la cámara (que es lo que pasa en la demo).
2. Compara kNN, SVM, Random Forest, MLP, CNN-1D, LSTM y GRU.
3. Entrena la mejor red con todos los datos y la exporta a
   ``web/public/model/`` (modelo.onnx + config.json) para la app.
4. Guarda tablas y gráficas en ``ml/resultados/`` para el documento.

Uso:
    python scripts/03_entrenar_comparar.py                  # todo, 5 folds
    python scripts/03_entrenar_comparar.py --rapido         # prueba rápida
    python scripts/03_entrenar_comparar.py --solo-final --modelo-final GRU
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ML = Path(__file__).resolve().parents[1]
RAIZ = ML.parent
sys.path.insert(0, str(ML))

from lsc.augment import aumentar_lote, espejar_lote  # noqa: E402
from lsc.dataset import cargar_npz, subconjunto  # noqa: E402
from lsc.features import FEATURE_VERSION, N_FEATURES, SEQ_LEN, batch_features  # noqa: E402

CAMPOS_CRUDOS = ("hands", "hand_present", "pose", "pose_present", "aspect")


def crudos(d, idx=None):
    return {k: (d[k] if idx is None else d[k][idx]) for k in CAMPOS_CRUDOS}


# ----------------------------------------------------------------- redes

def entrenar_red(nombre, d_tr, y_tr, n_cls, epocas, semilla, d_va=None, y_va=None,
                 *, modelo_inicial=None, lr_max=3e-3, pesos_clase=None):
    import torch
    from sklearn.metrics import f1_score
    from torch import nn

    from lsc.models import REDES

    torch.manual_seed(semilla)
    rng = np.random.default_rng(semilla)
    modelo = modelo_inicial
    if modelo is None:
        modelo = REDES[nombre](N_FEATURES, n_cls, SEQ_LEN)
        modelo.norm.ajustar(batch_features(aumentar_lote(crudos(d_tr), rng)))
    opt = torch.optim.AdamW(modelo.parameters(), lr=2e-3, weight_decay=1e-4)
    pasos = epocas * int(np.ceil(len(y_tr) / 64))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr_max, total_steps=pasos)
    perdida = nn.CrossEntropyLoss(weight=pesos_clase, label_smoothing=0.1)
    perdida_val = nn.CrossEntropyLoss(label_smoothing=0.1)
    yt = torch.from_numpy(y_tr).long()
    Xva = torch.from_numpy(batch_features(crudos(d_va))) if d_va is not None else None

    hist = {"perdida_ent": [], "acc_ent": [], "perdida_val": [], "acc_val": [], "f1_val": []}
    mejor = (-1.0, None)
    for ep in range(epocas):
        modelo.train()
        X = torch.from_numpy(batch_features(aumentar_lote(crudos(d_tr), rng)))
        orden = torch.from_numpy(rng.permutation(len(yt)))
        tot, ok, n = 0.0, 0, 0
        for i in range(0, len(orden), 64):
            b = orden[i:i + 64]
            if len(b) < 2:
                continue
            logits = modelo(X[b])
            loss = perdida(logits, yt[b])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 2.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
            ok += (logits.argmax(1) == yt[b]).sum().item()
            n += len(b)
        hist["perdida_ent"].append(tot / n)
        hist["acc_ent"].append(ok / n)
        if Xva is not None:
            modelo.eval()
            with torch.no_grad():
                lv = modelo(Xva)
                pv = lv.argmax(1).numpy()
                hist["perdida_val"].append(perdida_val(lv, torch.from_numpy(y_va).long()).item())
            hist["acc_val"].append(float((pv == y_va).mean()))
            f1 = f1_score(y_va, pv, average="macro")
            hist["f1_val"].append(float(f1))
            if f1 >= mejor[0]:
                mejor = (f1, {k: v.clone() for k, v in modelo.state_dict().items()})
    if mejor[1] is not None:
        modelo.load_state_dict(mejor[1])
    modelo.eval()
    return modelo, hist


def predecir_red(modelo, d):
    import torch
    with torch.no_grad():
        return modelo(torch.from_numpy(batch_features(crudos(d)))).argmax(1).numpy()


def latencia_ms(fn, x, n=100):
    fn(x)
    t = time.perf_counter()
    for _ in range(n):
        fn(x)
    return (time.perf_counter() - t) / n * 1000


# ----------------------------------------------------------------- gráficas

AZUL, NARANJA = "#2a78d6", "#eb6834"
TINTA, TINTA2, GRILLA = "#0b0b0b", "#52514e", "#e4e3df"


def _estilo(plt):
    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": GRILLA, "axes.labelcolor": TINTA2,
        "xtick.color": TINTA2, "ytick.color": TINTA2, "axes.titlecolor": TINTA,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 160,
    })


def graficar_comparacion(tabla, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _estilo(plt)
    tabla = sorted(tabla, key=lambda r: r["f1_media"])
    nombres = [f'{r["modelo"]}  ({r["tipo"]})' for r in tabla]
    f1 = [r["f1_media"] * 100 for r in tabla]
    err = [r["f1_desv"] * 100 for r in tabla]
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(tabla) + 1.2))
    ax.barh(nombres, f1, xerr=err, color=AZUL, height=0.55,
            error_kw={"ecolor": TINTA2, "elinewidth": 1, "capsize": 3})
    for i, v in enumerate(f1):
        ax.text(v + err[i] + 0.8, i, f"{v:.1f}", va="center", color=TINTA, fontsize=9)
    ax.set_xlim(0, 105)
    ax.set_xlabel("F1 macro (%) — media ± desviación entre folds, personas no vistas")
    ax.xaxis.grid(True, color=GRILLA, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Comparación de algoritmos", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def graficar_confusion(y, p, etiquetas, ruta, titulo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    _estilo(plt)
    cm = confusion_matrix(y, p, labels=range(len(etiquetas))).astype(float)
    cm /= np.maximum(cm.sum(1, keepdims=True), 1)
    n = len(etiquetas)
    lado = max(6, 0.28 * n + 2)
    fig, ax = plt.subplots(figsize=(lado + 1, lado))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), etiquetas, rotation=90, fontsize=7 if n > 30 else 9)
    ax.set_yticks(range(n), etiquetas, fontsize=7 if n > 30 else 9)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Seña real")
    ax.spines[:].set_visible(False)
    if n <= 30:
        for i in range(n):
            for j in range(n):
                if cm[i, j] >= 0.05:
                    ax.text(j, i, f"{cm[i, j]:.2f}"[1:] if cm[i, j] < 1 else "1",
                            ha="center", va="center", fontsize=7,
                            color="white" if cm[i, j] > 0.55 else TINTA)
    fig.colorbar(im, ax=ax, fraction=0.04, label="Proporción de la fila")
    ax.set_title(titulo, loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def graficar_f1_clases(reporte, ruta, titulo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _estilo(plt)
    filas = sorted(reporte, key=lambda r: r["f1"])
    fig, ax = plt.subplots(figsize=(7, 0.22 * len(filas) + 1.2))
    ax.barh([r["clase"] for r in filas], [r["f1"] * 100 for r in filas], color=AZUL, height=0.6)
    ax.set_xlim(0, 100)
    ax.set_xlabel("F1 (%) por seña")
    ax.tick_params(axis="y", labelsize=7 if len(filas) > 30 else 9)
    ax.xaxis.grid(True, color=GRILLA, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(titulo, loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def graficar_curvas(hist, ruta, nombre):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _estilo(plt)
    ep = np.arange(1, len(hist["perdida_ent"]) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    a1.plot(ep, hist["perdida_ent"], color=AZUL, lw=2, label="Entrenamiento")
    a1.plot(ep, hist["perdida_val"], color=NARANJA, lw=2, label="Validación")
    a1.set_title("Pérdida (entropía cruzada)", loc="left", fontweight="bold")
    a2.plot(ep, np.array(hist["acc_ent"]) * 100, color=AZUL, lw=2, label="Entrenamiento")
    a2.plot(ep, np.array(hist["acc_val"]) * 100, color=NARANJA, lw=2, label="Validación")
    a2.set_title("Exactitud (%)", loc="left", fontweight="bold")
    from matplotlib.ticker import MaxNLocator
    for a in (a1, a2):
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
        a.set_xlabel("Época")
        a.yaxis.grid(True, color=GRILLA, linewidth=0.6)
        a.legend(frameon=False)
    fig.suptitle(f"Entrenamiento del modelo final ({nombre})", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


# ----------------------------------------------------------------- export

def exportar_onnx(modelo, etiquetas, ruta_dir, info):
    import onnxruntime as ort
    import torch

    from lsc.models import ConSoftmax
    ruta_dir.mkdir(parents=True, exist_ok=True)
    envoltura = ConSoftmax(modelo).eval()
    ejemplo = torch.randn(1, SEQ_LEN, N_FEATURES)
    ruta = ruta_dir / "modelo.onnx"
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Exportador clásico (TorchScript): genera un solo archivo con los pesos
        # y redes GRU/LSTM compatibles con onnxruntime-web. Ver requirements.txt.
        torch.onnx.export(
            envoltura, (ejemplo,), str(ruta), input_names=["entrada"],
            output_names=["probabilidades"], opset_version=17, dynamo=False,
            dynamic_axes={"entrada": {0: "lote"}, "probabilidades": {0: "lote"}})
    # verificación: ONNX Runtime == PyTorch
    x = torch.randn(4, SEQ_LEN, N_FEATURES)
    with torch.no_grad():
        ref = envoltura(x).numpy()
    sess = ort.InferenceSession(str(ruta), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"entrada": x.numpy()})[0]
    dif = float(np.abs(out - ref).max())
    assert dif < 1e-4, f"ONNX difiere de PyTorch: {dif}"
    config = {
        "version_caracteristicas": FEATURE_VERSION,
        "seq_len": SEQ_LEN,
        "n_caracteristicas": N_FEATURES,
        "entrada": "entrada",
        "salida": "probabilidades",
        "etiquetas": list(etiquetas),
        "nombres": {},  # opcional: {"HOLA": "Hola"} para mostrar otro texto
        **info,
    }
    (ruta_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exportado {ruta} ({ruta.stat().st_size / 1024:.0f} KB, dif. ONNX/PyTorch {dif:.1e})")


# ----------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", nargs="*", type=Path, default=None,
                    help="archivos .npz (por defecto todos los de ml/datos/)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epocas", type=int, default=60)
    ap.add_argument("--hilos", type=int, default=os.cpu_count() or 1, help="hilos de CPU para PyTorch")
    ap.add_argument("--modelos", nargs="*", default=None, help="subconjunto de modelos a comparar")
    ap.add_argument("--min-muestras", type=int, default=10, help="descarta señas con menos muestras")
    ap.add_argument("--excluir", nargs="*", default=[], help="señas a excluir")
    ap.add_argument("--modelo-final", default="auto", help="auto = la red con mejor F1")
    ap.add_argument("--solo-final", action="store_true", help="omite la comparación")
    ap.add_argument("--rapido", action="store_true", help="3 folds y 20 épocas (para probar)")
    ap.add_argument("--resultados", type=Path, default=ML / "resultados")
    ap.add_argument("--destino-web", type=Path, default=RAIZ / "web" / "public" / "model")
    ap.add_argument("--semilla", type=int, default=42)
    a = ap.parse_args()
    if a.hilos < 1:
        ap.error("--hilos debe ser al menos 1")
    if a.rapido:
        a.folds, a.epocas = 3, 20

    import torch
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit

    from lsc.models import REDES, modelos_clasicos, n_parametros
    torch.set_num_threads(a.hilos)

    rutas = a.datos or sorted((ML / "datos").glob("*.npz"))
    if not rutas:
        sys.exit("No hay datos. Ejecute 02_extraer_landmarks.py (o generar_datos_sinteticos.py para probar).")
    d = cargar_npz(rutas)
    print("Archivos:", ", ".join(r.name for r in rutas), f"-> {len(d['label'])} secuencias")

    con_mano = d["hand_present"].reshape(len(d["label"]), -1).max(1) > 0
    d = subconjunto(d, con_mano)
    cuenta = {e: int((d["label"] == e).sum()) for e in np.unique(d["label"])}
    validas = sorted(e for e, n in cuenta.items() if n >= a.min_muestras and e not in a.excluir)
    descartadas = sorted(set(cuenta) - set(validas))
    if descartadas:
        print("Señas descartadas (pocas muestras o excluidas):", descartadas)
    d = subconjunto(d, np.isin(d["label"], validas))
    etiquetas = validas
    y = np.array([etiquetas.index(e) for e in d["label"]])
    grupos = d["group"].astype(str)
    print(f"{len(y)} secuencias | {len(etiquetas)} señas | {len(set(grupos))} personas")
    a.resultados.mkdir(parents=True, exist_ok=True)

    tabla, oof = [], {}
    if not a.solo_final:
        clasicos = modelos_clasicos(a.semilla)
        nombres = a.modelos or (list(clasicos) + list(REDES))
        folds = min(a.folds, len(set(grupos)))
        gkf = GroupKFold(n_splits=folds)
        F = batch_features(crudos(d))
        res = {n: {"acc": [], "f1": [], "lat": [], "params": None} for n in nombres}
        oof = {n: np.full(len(y), -1) for n in nombres}
        for k, (tr, te) in enumerate(gkf.split(F, y, grupos), 1):
            print(f"\n=== Fold {k}/{folds}: {len(set(grupos[tr]))} personas ent. / {len(set(grupos[te]))} prueba")
            Fm = batch_features(espejar_lote(crudos(d, tr)))
            Xtr = np.concatenate([F[tr], Fm]).reshape(2 * len(tr), -1)
            ytr = np.concatenate([y[tr], y[tr]])
            Xte = F[te].reshape(len(te), -1)
            for n in nombres:
                t0 = time.time()
                if n in clasicos:
                    from sklearn.base import clone
                    m = clone(clasicos[n]).fit(Xtr, ytr)
                    p = m.predict(Xte)
                    lat = latencia_ms(m.predict, Xte[:1], 20)
                else:
                    m, _ = entrenar_red(n, subconjunto(d, tr), y[tr], len(etiquetas), a.epocas, a.semilla + k)
                    p = predecir_red(m, subconjunto(d, te))
                    xt = torch.from_numpy(F[te[:1]])
                    with torch.no_grad():
                        lat = latencia_ms(m, xt, 100)
                    res[n]["params"] = n_parametros(m)
                oof[n][te] = p
                res[n]["acc"].append(accuracy_score(y[te], p))
                res[n]["f1"].append(f1_score(y[te], p, average="macro"))
                res[n]["lat"].append(lat)
                print(f"  {n:<13} acc={res[n]['acc'][-1]:.3f}  F1={res[n]['f1'][-1]:.3f}  ({time.time() - t0:.0f}s)")

        for n in nombres:
            r = res[n]
            tabla.append({
                "modelo": n, "tipo": "clásico" if n in clasicos else "red neuronal",
                "acc_media": float(np.mean(r["acc"])), "acc_desv": float(np.std(r["acc"])),
                "f1_media": float(np.mean(r["f1"])), "f1_desv": float(np.std(r["f1"])),
                "latencia_ms": float(np.mean(r["lat"])), "parametros": r["params"],
            })
        tabla.sort(key=lambda r: -r["f1_media"])
        lineas = ["| Modelo | Tipo | Exactitud | F1 macro | Latencia (ms/muestra) | Parámetros |",
                  "|---|---|---|---|---|---|"]
        for r in tabla:
            lineas.append(
                f"| {r['modelo']} | {r['tipo']} | {r['acc_media'] * 100:.1f} ± {r['acc_desv'] * 100:.1f} % | "
                f"{r['f1_media'] * 100:.1f} ± {r['f1_desv'] * 100:.1f} % | {r['latencia_ms']:.2f} | "
                f"{r['parametros'] if r['parametros'] else '—'} |")
        (a.resultados / "comparacion_modelos.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")
        (a.resultados / "comparacion_modelos.json").write_text(json.dumps(tabla, indent=2, ensure_ascii=False), encoding="utf-8")
        graficar_comparacion(tabla, a.resultados / "comparacion_modelos.png")
        print("\n" + "\n".join(lineas))

    # ---------------- modelo final
    redes_tabla = [r for r in tabla if r["modelo"] in REDES]
    final = a.modelo_final if a.modelo_final != "auto" else (redes_tabla[0]["modelo"] if redes_tabla else "GRU")
    print(f"\n=== Modelo final: {final}")

    if final in oof:
        graficar_confusion(y, oof[final], etiquetas, a.resultados / "matriz_confusion.png",
                           f"Matriz de confusión — {final} (validación cruzada por persona)")
        rep = classification_report(y, oof[final], labels=range(len(etiquetas)),
                                    target_names=etiquetas, output_dict=True, zero_division=0)
        filas = [{"clase": e, "precision": rep[e]["precision"], "recall": rep[e]["recall"],
                  "f1": rep[e]["f1-score"], "soporte": rep[e]["support"]} for e in etiquetas]
        with open(a.resultados / "reporte_por_sena.csv", "w", encoding="utf-8") as f:
            f.write("sena,precision,recall,f1,soporte\n")
            for r in filas:
                f.write(f"{r['clase']},{r['precision']:.4f},{r['recall']:.4f},{r['f1']:.4f},{r['soporte']}\n")
        graficar_f1_clases(filas, a.resultados / "f1_por_sena.png", f"F1 por seña — {final}")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=a.semilla)
    tr, va = next(gss.split(y, y, grupos))
    epocas_final = max(a.epocas, 30) if not a.rapido else a.epocas
    modelo, hist = entrenar_red(final, subconjunto(d, tr), y[tr], len(etiquetas), epocas_final,
                                a.semilla, subconjunto(d, va), y[va])
    graficar_curvas(hist, a.resultados / "curvas_entrenamiento.png", final)
    mejor_f1 = max(hist["f1_val"])
    print(f"Validación (personas no vistas): F1 macro = {mejor_f1:.3f}")

    info = {
        "modelo": final,
        "f1_validacion": round(mejor_f1, 4),
        "comparacion": tabla,
        "personas": len(set(grupos)),
        "secuencias": int(len(y)),
        "fecha": datetime.now().isoformat(timespec="seconds"),
    }
    exportar_onnx(modelo, etiquetas, a.destino_web, info)
    (a.resultados / "resumen.json").write_text(json.dumps(
        {**info, "etiquetas": etiquetas, "historial_final": hist}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Resultados en {a.resultados}")


if __name__ == "__main__":
    main()
