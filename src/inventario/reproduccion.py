"""Reproducción de un pedido publicado a partir de lo versionado.

Con el código en el commit del manifiesto y los datos de DVC de esa versión
(`git checkout <commit>` + `dvc checkout` o `dvc pull`), `reproducir_pedido`:

1. comprueba que las features, los datos crudos y el modelo en disco tengan las mismas
   huellas que cuando se emitió el pedido;
2. vuelve a pronosticar con ese modelo y esas features y regenera el pedido;
3. compara byte a byte el pedido regenerado con el publicado.

No usa MLflow ni la bandeja: solo lo que DVC restaura.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from inventario.almacen import cargar_tabla
from inventario.predecir import (
    cargar_modelo_local,
    generar_pedido,
    guardar_pedido,
    guardar_pronostico,
    pronosticar_con_cuantiles,
)
from inventario.versionado import cargar_manifiesto, md5_archivo, md5_directorio, resolver_base


@dataclass
class ResultadoReproduccion:
    clave: str
    modelo_version: str
    commit: str | None
    verificaciones: dict[str, bool] = field(default_factory=dict)
    md5_publicado: str = ""
    md5_reproducido: str = ""
    filas: int = 0
    # Pronóstico diario por cuantil: solo en manifiestos que lo incluyen (sección 11).
    tiene_pronostico: bool = False
    md5_pronostico_publicado: str = ""
    md5_pronostico_reproducido: str = ""

    @property
    def entradas_coinciden(self) -> bool:
        return all(self.verificaciones.values())

    @property
    def identico(self) -> bool:
        pedido = self.md5_publicado == self.md5_reproducido != ""
        pronostico = not self.tiene_pronostico or (
            self.md5_pronostico_publicado == self.md5_pronostico_reproducido != ""
        )
        return self.entradas_coinciden and pedido and pronostico


def reproducir_pedido(clave: str, base: str | Path | None = None) -> ResultadoReproduccion:
    """Regenera el pedido `clave` (p. ej. "2017-07-20" o "2017-07-20__conocimiento_2017-08-15")
    y lo compara con el publicado."""
    base = resolver_base(base)
    ruta_manifiesto = base / "data" / "predictions" / f"pedido_{clave}.json"
    if not ruta_manifiesto.exists():
        raise FileNotFoundError(f"No hay manifiesto para el pedido {clave}: {ruta_manifiesto}")
    m = cargar_manifiesto(ruta_manifiesto)

    resultado = ResultadoReproduccion(
        clave=clave, modelo_version=m["modelo"]["version"], commit=m["codigo"]["commit"]
    )
    ruta_features = base / m["features"]["ruta"]
    ruta_modelo = base / m["modelo"]["ruta"]
    ruta_raw = base / m["datos_crudos"]["ruta"]
    resultado.verificaciones = {
        "features": ruta_features.exists() and md5_archivo(ruta_features) == m["features"]["md5"],
        "datos_crudos": ruta_raw.is_dir() and md5_directorio(ruta_raw) == m["datos_crudos"]["md5"],
        "modelo": ruta_modelo.is_dir() and md5_directorio(ruta_modelo) == m["modelo"]["md5"],
    }
    ruta_publicado = base / m["pedido"]["ruta"]
    resultado.md5_publicado = md5_archivo(ruta_publicado) if ruta_publicado.exists() else ""
    if "pronostico" in m:
        resultado.tiene_pronostico = True
        ruta_pronostico = base / m["pronostico"]["ruta"]
        resultado.md5_pronostico_publicado = md5_archivo(ruta_pronostico) if ruta_pronostico.exists() else ""
    if not (resultado.verificaciones["features"] and resultado.verificaciones["modelo"]):
        return resultado  # sin las entradas exactas no tiene sentido regenerar

    corte = pd.Timestamp(m["ds"])
    version = m["modelo"]["version"]
    pronostico = pronosticar_con_cuantiles(cargar_modelo_local(ruta_modelo), cargar_tabla(ruta_features), corte)
    pedido = generar_pedido(pronostico, corte, modelo_version=version)
    with tempfile.TemporaryDirectory() as tmp:
        regenerado = guardar_pedido(pedido, corte, ruta=Path(tmp) / "pedido.csv")
        resultado.md5_reproducido = md5_archivo(regenerado)
        if resultado.tiene_pronostico:
            diario = guardar_pronostico(pronostico, Path(tmp) / "pronostico.csv", modelo_version=version)
            resultado.md5_pronostico_reproducido = md5_archivo(diario)
    resultado.filas = len(pedido)
    return resultado
