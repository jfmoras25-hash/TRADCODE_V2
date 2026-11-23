"""Servidor local de TRADCODE. La inferencia GRU se ejecuta en el navegador."""
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

if __package__:
    from .capturas import Capturas, guardar_capturas
else:
    from capturas import Capturas, guardar_capturas

RAIZ = Path(__file__).resolve().parents[1]
FRONTEND = RAIZ / "web" / "dist"
CAPTURAS = RAIZ / "ml" / "datos" / "capturas"

app = FastAPI(title="TRADCODE")


@app.get("/", include_in_schema=False)
def inicio():
    return FileResponse(FRONTEND / "inicio.html")


@app.get("/api/estado")
def estado():
    try:
        config = json.loads((FRONTEND / "model" / "config.json").read_text(encoding="utf-8"))
        listo = config["modelo"] == "GRU" and (FRONTEND / "model" / "modelo.onnx").is_file()
        return {"aplicacion": "TRADCODE", "modelo": "GRU" if listo else None,
                "categorias": len(config["etiquetas"]) if listo else 0}
    except (OSError, ValueError, KeyError, TypeError):
        return {"aplicacion": "TRADCODE", "modelo": None, "categorias": 0}


@app.post("/api/capturas")
def capturar(documento: Capturas, request: Request):
    origen = request.headers.get("origin")
    if origen and origen != f"{request.url.scheme}://{request.url.netloc}":
        raise HTTPException(403, "La captura debe venir de la aplicación local")
    return guardar_capturas(documento, CAPTURAS)


app.mount("/", StaticFiles(directory=FRONTEND, html=True, check_dir=False), name="frontend")

if __name__ == "__main__":
    if not (FRONTEND / "index.html").is_file():
        raise SystemExit("Compila el frontend con npm run build en web/ antes de iniciar TRADCODE.")
    uvicorn.run(app, host="127.0.0.1", port=8010)
