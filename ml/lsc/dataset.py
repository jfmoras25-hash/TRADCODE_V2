"""Lectura de LSC70, capturas propias (JSON de la web) y archivos .npz."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .features import N_HAND_LM, N_POSE_LM, SEQ_LEN

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
RE_PERSONA = re.compile(r"per[\s_-]*0*(\d+)", re.IGNORECASE)
RE_CUADRO = re.compile(r"^(.*?)[\s_-]*(\d+)_*$")
CAMPOS = ("hands", "hand_present", "pose", "pose_present")


@dataclass
class Muestra:
    clave: str
    etiqueta: str
    grupo: str           # persona/voluntario -> se usa para validar por grupos
    seccion: str
    cuadros: list[Path] = field(default_factory=list)


def indices_remuestreo(n: int, T: int = SEQ_LEN) -> list[int]:
    """Índices de T cuadros equiespaciados tomados de n cuadros."""
    if n <= 0:
        return []
    return [int(np.floor(i * (n - 1) / max(T - 1, 1) + 0.5)) for i in range(T)]


def _persona(partes: tuple[str, ...], base: str) -> str | None:
    for p in reversed(partes):
        m = RE_PERSONA.fullmatch(p.strip()) or RE_PERSONA.match(p.strip())
        if m:
            return f"Per{int(m.group(1)):02d}"
    m = RE_PERSONA.match(base)
    return f"Per{int(m.group(1)):02d}" if m else None


def descubrir_lsc70(raiz: Path, secciones: list[str] | None = None) -> list[Muestra]:
    """Agrupa las imágenes de LSC70 en secuencias.

    Estructura documentada por los autores: subconjunto / voluntario / seña /
    cuadro, con nombres tipo ``Per + Número + Etiqueta + NúmeroCuadro.jpg``.
    El código es tolerante: la seña es la carpeta que contiene las imágenes
    (o, si esa carpeta es la del voluntario, lo que queda del nombre del
    archivo), y el voluntario es el componente que coincide con ``PerNN``.
    """
    raiz = Path(raiz)
    secc = {s.upper() for s in secciones} if secciones else None
    grupos: dict[tuple, list[tuple[int, Path]]] = defaultdict(list)

    for ruta in sorted(raiz.rglob("*")):
        if ruta.suffix.lower() not in IMG_EXT or not ruta.is_file():
            continue
        partes = ruta.relative_to(raiz).parts[:-1]
        seccion = next((p for p in partes if p.upper().startswith("LSC70")), "")
        if secc is not None and seccion.upper() not in secc:
            continue
        m = RE_CUADRO.match(ruta.stem)
        base, num = (m.group(1), int(m.group(2))) if m else (ruta.stem, 0)
        grupos[(ruta.parent, base)].append((num, ruta))

    muestras = []
    for (carpeta, base), archivos in grupos.items():
        partes = carpeta.relative_to(raiz).parts
        seccion = next((p for p in partes if p.upper().startswith("LSC70")), "")
        persona = _persona(partes, base) or (partes[-2] if len(partes) > 1 else "desconocido")
        if RE_PERSONA.fullmatch(carpeta.name.strip()):
            etiqueta = RE_PERSONA.sub("", base, count=1).strip(" _-") or base
        else:
            etiqueta = carpeta.name.strip()
        archivos.sort()
        muestras.append(Muestra(
            clave=str(Path(*partes) / base),
            etiqueta=etiqueta,
            grupo=persona,
            seccion=seccion,
            cuadros=[r for _, r in archivos],
        ))
    muestras.sort(key=lambda m: m.clave)
    return muestras


# --------------------------------------------------------------------------
# Contenedor en memoria: dict de arreglos numpy con eje N al inicio.
# --------------------------------------------------------------------------

def vacio(n: int, T: int = SEQ_LEN) -> dict:
    return {
        "hands": np.zeros((n, T, 2, N_HAND_LM, 3), np.float32),
        "hand_present": np.zeros((n, T, 2), np.float32),
        "pose": np.zeros((n, T, N_POSE_LM, 4), np.float32),
        "pose_present": np.zeros((n, T), np.float32),
        "aspect": np.ones(n, np.float32),
        "label": np.array([""] * n, dtype=object),
        "group": np.array([""] * n, dtype=object),
        "source": np.array([""] * n, dtype=object),
    }


def guardar_npz(data: dict, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    out = {k: (v.astype(str) if v.dtype == object else v) for k, v in data.items()}
    np.savez_compressed(ruta, **out)


def cargar_npz(rutas: list[Path]) -> dict:
    partes = []
    for r in rutas:
        z = np.load(r, allow_pickle=False)
        partes.append({k: (z[k].astype(object) if z[k].dtype.kind == "U" else z[k])
                       for k in z.files})
    return concatenar(partes)


def concatenar(partes: list[dict]) -> dict:
    partes = [p for p in partes if len(p["label"])]
    if not partes:
        return vacio(0)
    return {k: np.concatenate([p[k] for p in partes]) for k in partes[0]}


def subconjunto(data: dict, idx) -> dict:
    return {k: v[idx] for k, v in data.items()}


# --------------------------------------------------------------------------
# Capturas hechas con el modo "Capturar" de la app web
# --------------------------------------------------------------------------

def cargar_capturas_json(ruta: Path, T: int = SEQ_LEN) -> dict:
    doc = json.loads(Path(ruta).read_text(encoding="utf-8"))
    if doc.get("formato") != "lsc-capturas":
        raise ValueError(f"{ruta} no es un archivo de capturas de la app")
    muestras = doc["muestras"]
    d = vacio(len(muestras), T)
    for i, m in enumerate(muestras):
        cuadros = m["cuadros"]
        for t, j in enumerate(indices_remuestreo(len(cuadros), T)):
            c = cuadros[j]
            for s, mano in enumerate(c.get("manos") or [None, None]):
                if mano:
                    d["hands"][i, t, s] = np.asarray(mano, np.float32)[:N_HAND_LM, :3]
                    d["hand_present"][i, t, s] = 1
            if c.get("pose"):
                d["pose"][i, t] = np.asarray(c["pose"], np.float32)[:N_POSE_LM, :4]
                d["pose_present"][i, t] = 1
        d["aspect"][i] = float(m.get("aspect", 4 / 3))
        d["label"][i] = str(m["etiqueta"]).strip()
        d["group"][i] = "cap_" + str(m.get("persona") or "anonimo").strip().lower()
        d["source"][i] = "captura"
    return d
