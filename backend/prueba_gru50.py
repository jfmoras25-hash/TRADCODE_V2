"""Instancia local de prueba para la GRU de 50 clases en el puerto 8011."""
import json
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .capturas import Capturas, guardar_capturas
from .main import FRONTEND, RAIZ, inicio

CANDIDATA = RAIZ / "ml" / "candidatas" / "gru_50"
CAPTURAS_PRUEBA = CANDIDATA / "capturas_prueba"
NUEVAS = {"ADIOS", "MI", "NOMBRE_ES"}


@asynccontextmanager
async def comprobar_archivos(app: FastAPI):
    if not (FRONTEND / "index.html").is_file():
        raise RuntimeError("Compila el frontend con npm run build en web/ antes de iniciar la prueba.")
    config = json.loads((CANDIDATA / "config.json").read_text(encoding="utf-8"))
    activa = json.loads((FRONTEND / "model" / "config.json").read_text(encoding="utf-8"))
    etiquetas = config["etiquetas"]
    anteriores = set(activa["etiquetas"])
    if (
        config["modelo"] != "GRU"
        or len(etiquetas) != 50
        or len(anteriores) != 47
        or set(etiquetas) != anteriores | NUEVAS
        or not (CANDIDATA / "modelo.onnx").is_file()
    ):
        raise RuntimeError("La candidata debe contener las 47 etiquetas activas y las tres nuevas.")
    app.state.categorias = len(etiquetas)
    yield


app = FastAPI(title="TRADCODE · Prueba GRU 50", lifespan=comprobar_archivos)
app.add_api_route("/", inicio, methods=["GET"], include_in_schema=False)


@app.get("/api/estado")
def estado(request: Request):
    return {"aplicacion": "TRADCODE", "modelo": "GRU", "categorias": request.app.state.categorias,
            "modo": "prueba_gru_50"}


# Estas rutas deben registrarse antes del frontend estático, que incluye el modelo de 47 clases.
@app.api_route("/model/config.json", methods=["GET", "HEAD"], include_in_schema=False)
def configuracion():
    return FileResponse(CANDIDATA / "config.json", headers={"Cache-Control": "no-store"})


@app.api_route("/model/modelo.onnx", methods=["GET", "HEAD"], include_in_schema=False)
def modelo():
    return FileResponse(CANDIDATA / "modelo.onnx", headers={"Cache-Control": "no-store"})


@app.post("/api/capturas")
def capturar(documento: Capturas, request: Request):
    origen = request.headers.get("origin")
    if origen and origen != f"{request.url.scheme}://{request.url.netloc}":
        raise HTTPException(403, "La captura debe venir de la aplicación local")
    return guardar_capturas(documento, CAPTURAS_PRUEBA)


app.mount("/", StaticFiles(directory=FRONTEND, html=True, check_dir=False), name="frontend")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8011)
