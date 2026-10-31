"""Persistencia de los datos entre etapas del pipeline (parquet).

Las escrituras reemplazan por completo la salida de la misma fecha: re-ejecutar una
etapa deja exactamente el mismo resultado (idempotencia) y nunca mezcla restos de una
ejecución anterior.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import pandas as pd

from inventario.ingesta import DatosBandeja, normalizar_fecha

FUENTES = ("ventas", "petroleo", "feriados", "tiendas")


def guardar_datos(datos: DatosBandeja, directorio: str | Path) -> Path:
    """Guarda cada fuente como parquet en `directorio`, reemplazando su contenido."""
    directorio = Path(directorio)
    temporal = directorio.with_name(directorio.name + ".tmp")
    shutil.rmtree(temporal, ignore_errors=True)
    temporal.mkdir(parents=True)

    for fuente in FUENTES:
        df = getattr(datos, fuente)
        if df is not None:
            df.to_parquet(temporal / f"{fuente}.parquet", index=False)
    (temporal / "meta.json").write_text(
        json.dumps({"fecha_corte": str(datos.fecha_corte.date())}), encoding="utf-8"
    )

    shutil.rmtree(directorio, ignore_errors=True)
    temporal.rename(directorio)
    return directorio


def cargar_datos(directorio: str | Path) -> DatosBandeja:
    directorio = Path(directorio)
    meta_ruta = directorio / "meta.json"
    if not meta_ruta.exists():
        raise FileNotFoundError(f"No hay datos guardados en {directorio} (¿se ejecutó la etapa anterior?)")
    meta = json.loads(meta_ruta.read_text(encoding="utf-8"))

    tablas = {}
    for fuente in FUENTES:
        ruta = directorio / f"{fuente}.parquet"
        tablas[fuente] = pd.read_parquet(ruta) if ruta.exists() else None
    if tablas["ventas"] is None or tablas["petroleo"] is None or tablas["feriados"] is None:
        raise FileNotFoundError(f"Faltan fuentes obligatorias en {directorio}")
    return DatosBandeja(fecha_corte=normalizar_fecha(meta["fecha_corte"]), **tablas)


def recortar_ventas(datos: DatosBandeja, dias: int) -> DatosBandeja:
    """Conserva solo los últimos `dias` días de ventas hasta el corte (inclusive)."""
    desde = datos.fecha_corte - pd.Timedelta(days=dias)
    ventas = datos.ventas.loc[datos.ventas["date"].isna() | (datos.ventas["date"] > desde)]
    return replace(datos, ventas=ventas.reset_index(drop=True))


def guardar_tabla(df: pd.DataFrame, ruta: str | Path) -> Path:
    """Escribe un parquet de forma atómica (archivo temporal + reemplazo)."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_name(ruta.name + ".tmp")
    df.to_parquet(temporal, index=False)
    os.replace(temporal, ruta)
    return ruta


def cargar_tabla(ruta: str | Path) -> pd.DataFrame:
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta} (¿se ejecutó la etapa anterior?)")
    return pd.read_parquet(ruta)
