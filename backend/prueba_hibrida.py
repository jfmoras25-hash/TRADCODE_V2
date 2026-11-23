"""Servidor experimental de TRADCODE con GRU 52 y Random Forest en el puerto 8011."""
import argparse
import hashlib
import json
import re
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from ml.legado_rf.evaluar import ReconocedorLegado, mano_de_cuadro

from .capturas import Capturas, Mano, guardar_capturas
from .main import RAIZ

DIST = RAIZ / "web" / "dist-hibrido"
CANDIDATAS = RAIZ / "ml" / "candidatas"
GRU = CANDIDATAS / "gru_52"
ARCHIVOS_MODELO = {"config.json": "application/json", "modelo.onnx": "application/octet-stream"}
CAPTURAS_PRUEBA = RAIZ / "ml" / "datos" / "capturas_hibrido"


class Manos(BaseModel):
    model_config = ConfigDict(extra="forbid")
    manos: tuple[Mano | None, Mano | None]


@asynccontextmanager
async def iniciar(app: FastAPI):
    if not (DIST / "index.html").is_file():
        raise RuntimeError("Compila web con npm run build:hibrido.")
    gru = app.state.gru
    manifiesto = gru / "exportacion.json"
    # Las versiones de entrenamiento locales anteriores solo tienen el informe completo.
    metadata = manifiesto if manifiesto.is_file() else gru / "resumen.json"
    verificacion = json.loads(metadata.read_text(encoding="utf-8"))
    config = json.loads((gru / "config.json").read_text(encoding="utf-8"))
    base = json.loads((CANDIDATAS / "gru_50" / "config.json").read_text(encoding="utf-8"))
    etiquetas_base = base["etiquetas"]
    esperadas = [*etiquetas_base, "DEDO_MEDIO", "HIJUEPUTA"]
    if (config["modelo"] != "GRU" or len(set(esperadas)) != 52 or config["etiquetas"] != esperadas
            or verificacion["clases"] != 52):
        raise RuntimeError("El puerto 8011 requiere la GRU de 52 clases derivada de GRU 50.")
    for nombre in ARCHIVOS_MODELO:
        firma = hashlib.sha256((gru / nombre).read_bytes()).hexdigest()
        if firma != verificacion["exportacion_sha256"][nombre]:
            raise RuntimeError(f"{nombre} no coincide con el resultado validado del entrenamiento.")
    app.state.config = config
    app.state.bosque = ReconocedorLegado()
    yield


app = FastAPI(title="TRADCODE · Prueba híbrida", lifespan=iniciar)
app.state.gru = GRU


def exigir_origen_local(request: Request):
    origen = request.headers.get("origin")
    if origen and origen != f"{request.url.scheme}://{request.url.netloc}":
        raise HTTPException(403, "La solicitud debe venir de la aplicación local")


@app.get("/", include_in_schema=False)
def inicio():
    return FileResponse(DIST / "inicio.html")


@app.get("/api/estado")
def estado(request: Request):
    clases = request.app.state.bosque.bosque.classes_
    config = request.app.state.config
    return {"modelo": config["modelo"], "categorias": len(config["etiquetas"]), "modo": "prueba_hibrida",
            "version_modelo": request.app.state.gru.name, "etiquetas": config["etiquetas"],
            "letras_rf": [str(e) for e in clases if len(e) == 1 and e in "ABCDEFGHIJKLMNÑOPQRSTUVWXYZ"]}


@app.get("/model/{nombre}", include_in_schema=False)
def servir_modelo(nombre: str, request: Request):
    if nombre not in ARCHIVOS_MODELO:
        raise HTTPException(404)
    return FileResponse(request.app.state.gru / nombre, media_type=ARCHIVOS_MODELO[nombre],
                        headers={"Cache-Control": "no-cache"})


@app.post("/api/letras")
def predecir_letras(entrada: Manos, request: Request):
    exigir_origen_local(request)
    try:
        coordenadas, mano = mano_de_cuadro(entrada.manos)
        return request.app.state.bosque.puntuaciones(coordenadas, mano)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@app.post("/api/capturas")
def capturar(documento: Capturas, request: Request):
    exigir_origen_local(request)
    return guardar_capturas(documento, CAPTURAS_PRUEBA)


app.mount("/", StaticFiles(directory=DIST, html=True, check_dir=False), name="frontend")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=GRU.name, help="Carpeta validada dentro de ml/candidatas")
    args = parser.parse_args()
    if not re.fullmatch(r"gru_52\w*", args.version):
        parser.error("--version debe identificar una versión de GRU 52")
    app.state.gru = CANDIDATAS / args.version
    uvicorn.run(app, host="127.0.0.1", port=8011)
