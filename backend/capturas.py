"""Validación y almacenamiento local de capturas compatibles con ml/scripts/importar_capturas.py."""
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StringConstraints, field_validator

Punto3 = tuple[FiniteFloat, FiniteFloat, FiniteFloat]
Punto4 = tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]
Mano = Annotated[list[Punto3], Field(min_length=21, max_length=21)]
Pose = Annotated[list[Punto4], Field(min_length=33, max_length=33)]
Etiqueta = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=30)]
Persona = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]
escritura = Lock()


class Cuadro(BaseModel):
    model_config = ConfigDict(extra="forbid")
    manos: tuple[Mano | None, Mano | None]
    pose: Pose | None


class Muestra(BaseModel):
    model_config = ConfigDict(extra="forbid")
    etiqueta: Etiqueta
    persona: Persona
    aspect: Annotated[FiniteFloat, Field(gt=0, le=100)]
    fecha: Annotated[str, Field(max_length=40)]
    cuadros: Annotated[list[Cuadro], Field(min_length=6, max_length=6)]

    @field_validator("cuadros")
    @classmethod
    def comprobar_manos(cls, cuadros: list[Cuadro]) -> list[Cuadro]:
        if not any(any(cuadro.manos) for cuadro in cuadros):
            raise ValueError("La muestra debe contener al menos una mano")
        return cuadros


class Capturas(BaseModel):
    model_config = ConfigDict(extra="forbid")
    formato: Literal["lsc-capturas"]
    version: Literal[1]
    seq_len: Literal[6]
    version_caracteristicas: Literal[1]
    muestras: Annotated[list[Muestra], Field(min_length=1, max_length=50)]


def guardar_capturas(documento: Capturas, carpeta: Path) -> dict[str, int]:
    """Un JSON por muestra, con deduplicación incluso tras reiniciar el servidor."""
    cabecera = documento.model_dump(mode="json", exclude={"muestras"})
    guardadas = 0
    duplicadas = 0
    with escritura:
        carpeta.mkdir(parents=True, exist_ok=True)
        for muestra in documento.muestras:
            datos = muestra.model_dump(mode="json")
            identidad = {clave: valor for clave, valor in datos.items() if clave != "fecha"}
            huella = hashlib.sha256(json.dumps(identidad, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            destino = carpeta / f"captura_{huella}.json"
            if destino.exists():
                duplicadas += 1
                continue
            temporal = None
            try:
                with NamedTemporaryFile(mode="w", encoding="utf-8", dir=carpeta, suffix=".tmp", delete=False) as archivo:
                    temporal = Path(archivo.name)
                    json.dump({**cabecera, "muestras": [datos]}, archivo, ensure_ascii=False)
                temporal.replace(destino)
            finally:
                if temporal is not None:
                    temporal.unlink(missing_ok=True)
            guardadas += 1
    return {"guardadas": guardadas, "duplicadas": duplicadas}
