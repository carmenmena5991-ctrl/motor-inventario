"""Simulación del sistema de origen: genera bandeja_origen/ a partir de los CSV de Kaggle.

Cada fila recibe una fecha de entrega según config/bandeja.yaml (frecuencia y retraso de
su fuente) y se escribe un archivo por fuente y fecha de entrega:

    <destino>/<fuente>/<fuente>_<AAAA-MM-DD>.csv

Todo lo entregado hasta `fin_carga_historica` forma la carga histórica inicial (un solo
archivo por fuente con esa fecha). En el periodo simulado se agregan, de forma
reproducible (semilla fija), las incidencias configuradas:

- entregas tardías: los datos de un día llegan 1-3 días después de lo normal;
- correcciones: algunas filas llegan con un valor erróneo y su valor real llega después;
- entregas corruptas: un archivo trae una venta negativa y la fuente lo reenvía válido.

La última versión entregada de cada fila coincide con el dato original de Kaggle.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from inventario.ingesta import CLAVES, FUENTES

RUTA_CONFIG_DEFAULT = Path(__file__).resolve().parents[2] / "config" / "bandeja.yaml"
ARCHIVOS_ANTIGUOS = ("train.csv", "oil.csv", "holidays_events.csv", "stores.csv", "transactions.csv")


def cargar_config_bandeja(ruta: str | Path | None = None) -> dict[str, Any]:
    with open(ruta or RUTA_CONFIG_DEFAULT, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dia_habil(fechas: pd.Series) -> pd.Series:
    """Mueve sábados y domingos al lunes siguiente."""
    dia = fechas.dt.day_of_week
    return fechas + pd.to_timedelta(np.where(dia >= 5, 7 - dia, 0), unit="D")


def _entrega_normal(df: pd.DataFrame, conf: dict[str, Any], fin_historico: pd.Timestamp) -> pd.Series:
    frecuencia = conf["frecuencia"]
    if frecuencia == "unica":
        return pd.Series(fin_historico, index=df.index)
    if frecuencia == "mensual":
        meses = int(conf.get("meses_anticipacion", 1))
        return (df["date"].dt.to_period("M") - meses).dt.start_time
    entrega = df["date"] + pd.Timedelta(days=int(conf.get("retraso_dias", 0)))
    if frecuencia == "dias_habiles":
        entrega = _dia_habil(entrega)
    elif frecuencia != "diaria":
        raise ValueError(f"Frecuencia desconocida: {frecuencia}")
    return entrega


def _sortear_dias(rng: np.random.Generator, dias: pd.DatetimeIndex, probabilidad: float) -> pd.DatetimeIndex:
    return dias[rng.random(len(dias)) < probabilidad]


def generar_bandeja(
    origen: str | Path, destino: str | Path, config: dict[str, Any] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Genera la bandeja. Devuelve (entregas, incidencias) como DataFrames de resumen.

    Es idempotente: con la misma configuración y semilla produce exactamente los mismos
    archivos, y reemplaza los que hubiera de una ejecución anterior.
    """
    config = config or cargar_config_bandeja()
    origen, destino = Path(origen), Path(destino)
    fin_historico = pd.Timestamp(config["periodo"]["fin_carga_historica"])
    fin_simulacion = pd.Timestamp(config["periodo"]["fin_simulacion"])
    rng = np.random.default_rng(int(config.get("semilla", 0)))
    incidencias_conf = config.get("incidencias", {}) or {}
    dias_simulados = pd.date_range(fin_historico + pd.Timedelta(days=1), fin_simulacion, freq="D")

    filas_por_fuente: dict[str, pd.DataFrame] = {}
    incidencias: list[dict[str, Any]] = []

    for fuente in FUENTES:
        conf = config["fuentes"][fuente]
        df = pd.read_csv(origen / conf["archivo"])
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        df["_entrega"] = _entrega_normal(df, conf, fin_historico)

        # Entregas tardías: todos los datos de un día llegan más tarde.
        tardias = incidencias_conf.get("entregas_tardias", {})
        if fuente in tardias.get("fuentes", []):
            minimo, maximo = tardias["dias_extra"]
            for dia in _sortear_dias(rng, dias_simulados, tardias["probabilidad"]):
                extra = int(rng.integers(minimo, maximo + 1))
                filas = df["date"] == dia
                if filas.any():
                    df.loc[filas, "_entrega"] += pd.Timedelta(days=extra)
                    if conf["frecuencia"] == "dias_habiles":
                        df.loc[filas, "_entrega"] = _dia_habil(df.loc[filas, "_entrega"])
                    incidencias.append({"tipo": "entrega_tardia", "fuente": fuente, "fecha_datos": dia,
                                        "fecha_entrega": df.loc[filas, "_entrega"].iloc[0], "filas": int(filas.sum())})

        df["_orden"] = 0
        extras: list[pd.DataFrame] = []

        # Entregas corruptas: el archivo de un día trae una venta negativa y se reenvía.
        corruptas = incidencias_conf.get("entregas_corruptas", {})
        dias_corruptos: set[pd.Timestamp] = set()
        if fuente in corruptas.get("fuentes", []):
            for dia in _sortear_dias(rng, dias_simulados, corruptas["probabilidad"]):
                archivo = df["_entrega"] == dia
                reenvio = dia + pd.Timedelta(days=int(corruptas["dias_reenvio"]))
                if not archivo.any() or reenvio > fin_simulacion or reenvio in dias_corruptos:
                    continue
                dias_corruptos.add(dia)
                valido = df.loc[archivo].assign(_entrega=reenvio, _orden=1)
                extras.append(valido)
                indice = df.index[archivo][int(rng.integers(archivo.sum()))]
                df.loc[indice, "sales"] = -abs(df.loc[indice, "sales"]) - 1
                incidencias.append({"tipo": "entrega_corrupta", "fuente": fuente, "fecha_datos": dia,
                                    "fecha_entrega": dia, "filas": int(archivo.sum())})
                incidencias.append({"tipo": "reenvio", "fuente": fuente, "fecha_datos": dia,
                                    "fecha_entrega": reenvio, "filas": int(archivo.sum())})

        # Correcciones: filas entregadas con error cuyo valor real llega después.
        correcciones = incidencias_conf.get("correcciones", {})
        if fuente in correcciones.get("fuentes", []):
            fmin, fmax = correcciones["factor_error"]
            dmin, dmax = correcciones["dias_despues"]
            for dia in _sortear_dias(rng, dias_simulados, correcciones["probabilidad"]):
                del_dia = df.index[(df["date"] == dia) & (df["sales"] > 0)]
                entrega_original = df.loc[del_dia, "_entrega"].max() if len(del_dia) else None
                if entrega_original is None or entrega_original in dias_corruptos:
                    continue
                cuando = entrega_original + pd.Timedelta(days=int(rng.integers(dmin, dmax + 1)))
                n = max(1, int(round(len(del_dia) * correcciones["fraccion_filas"])))
                if cuando > fin_simulacion or cuando in dias_corruptos:
                    continue
                elegidas = rng.choice(del_dia, size=n, replace=False)
                extras.append(df.loc[elegidas].assign(_entrega=cuando, _orden=1))
                df.loc[elegidas, "sales"] = (
                    df.loc[elegidas, "sales"] * rng.uniform(fmin, fmax, size=n)
                ).round(3)
                incidencias.append({"tipo": "correccion", "fuente": fuente, "fecha_datos": dia,
                                    "fecha_entrega": cuando, "filas": n})

        todo = pd.concat([df, *extras], ignore_index=True)
        todo.loc[todo["_entrega"] <= fin_historico, "_entrega"] = fin_historico
        todo = todo.loc[todo["_entrega"] <= fin_simulacion]
        filas_por_fuente[fuente] = todo

    resumen = []
    for fuente, todo in filas_por_fuente.items():
        carpeta = destino / fuente
        carpeta.mkdir(parents=True, exist_ok=True)
        for viejo in carpeta.glob(f"{fuente}_*.csv"):
            viejo.unlink()
        columnas = [c for c in todo.columns if not c.startswith("_")]
        orden = [c for c in CLAVES[fuente] if c in columnas]
        for entrega, grupo in todo.groupby("_entrega"):
            salida = grupo.sort_values(["_orden", *orden], kind="stable")[columnas].copy()
            if "date" in salida.columns:
                salida["date"] = salida["date"].dt.strftime("%Y-%m-%d")
            ruta = carpeta / f"{fuente}_{entrega.date()}.csv"
            temporal = ruta.with_name(ruta.name + ".tmp")
            salida.to_csv(temporal, index=False)
            os.replace(temporal, ruta)
            resumen.append({"fuente": fuente, "fecha_entrega": entrega.date(), "filas": len(salida)})

    for antiguo in ARCHIVOS_ANTIGUOS:  # formato plano anterior a la simulación por entregas
        (destino / antiguo).unlink(missing_ok=True)

    incidencias_df = pd.DataFrame(incidencias, columns=["tipo", "fuente", "fecha_datos", "fecha_entrega", "filas"])
    return pd.DataFrame(resumen), incidencias_df.sort_values(["fecha_datos", "tipo"]).reset_index(drop=True)
