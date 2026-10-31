"""Ingesta de la bandeja de entrada (bandeja_origen/) con corte por fecha lógica.

Para evitar data leakage, todas las fuentes se recortan a la información disponible
hasta la fecha de emisión (`fecha_corte`, normalmente el `logical_date`/`ds` de Airflow).

Convención de la bandeja: cada fuente puede llegar como un único archivo o como varias
particiones que comparten prefijo (p. ej. ``train.csv`` o ``train_2017-08-01.csv``):

- ``train*.csv``           -> ventas (date, store_nbr, family, sales, onpromotion)
- ``oil*.csv``             -> petróleo (date, dcoilwtico)
- ``holidays_events*.csv`` -> feriados (date, type, locale, locale_name, description, transferred)
- ``stores*.csv``          -> tiendas (store_nbr, city, state, type, cluster) [opcional, sin fecha]
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

BANDEJA_DEFAULT = Path("bandeja_origen")

PATRONES = {
    "ventas": "train*.csv",
    "petroleo": "oil*.csv",
    "feriados": "holidays_events*.csv",
    "tiendas": "stores*.csv",
}

CLAVES = {
    "ventas": ["date", "store_nbr", "family"],
    "petroleo": ["date"],
    "feriados": ["date", "type", "locale", "locale_name", "description"],
    "tiendas": ["store_nbr"],
}


@dataclass(frozen=True)
class DatosBandeja:
    ventas: pd.DataFrame
    petroleo: pd.DataFrame
    feriados: pd.DataFrame
    tiendas: pd.DataFrame | None
    fecha_corte: pd.Timestamp


def normalizar_fecha(fecha: str | date | datetime | pd.Timestamp) -> pd.Timestamp:
    """Convierte cualquier representación de fecha (incluido un datetime con zona horaria
    de Airflow) a un Timestamp sin hora ni zona."""
    ts = pd.Timestamp(fecha)
    return pd.Timestamp(ts.date())


def _leer_fuente(bandeja: Path, fuente: str, obligatoria: bool = True) -> pd.DataFrame | None:
    archivos = sorted(bandeja.glob(PATRONES[fuente]))
    if not archivos:
        if obligatoria:
            raise FileNotFoundError(f"No hay archivos '{PATRONES[fuente]}' en {bandeja}")
        return None

    df = pd.concat([pd.read_csv(a) for a in archivos], ignore_index=True)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # Si una partición se reenvía, prevalece la última (orden alfabético de archivo).
    claves = [c for c in CLAVES[fuente] if c in df.columns]
    if claves:
        df = df.drop_duplicates(subset=claves, keep="last")
    return df


def _recortar(df: pd.DataFrame, hasta: pd.Timestamp) -> pd.DataFrame:
    # Las fechas no parseables (NaT) se conservan para que validacion las detecte.
    mascara = df["date"].isna() | (df["date"] <= hasta)
    return df.loc[mascara].sort_values("date", kind="stable").reset_index(drop=True)


def leer_bandeja(
    fecha_corte: str | date | datetime | pd.Timestamp,
    bandeja: str | Path = BANDEJA_DEFAULT,
    dias_calendario_futuro: int = 0,
) -> DatosBandeja:
    """Lee la bandeja y devuelve solo la información disponible a `fecha_corte` (inclusive).

    `dias_calendario_futuro` permite incluir feriados posteriores al corte: el calendario
    oficial se publica por adelantado, así que no es leakage y es necesario para predecir
    el horizonte. Ventas y petróleo nunca pasan de `fecha_corte`.

    La función no modifica la bandeja: llamarla varias veces con los mismos argumentos
    devuelve el mismo resultado.
    """
    bandeja = Path(bandeja)
    if not bandeja.is_dir():
        raise FileNotFoundError(f"No existe la bandeja de entrada: {bandeja}")
    if dias_calendario_futuro < 0:
        raise ValueError("dias_calendario_futuro debe ser >= 0")

    corte = normalizar_fecha(fecha_corte)

    ventas = _recortar(_leer_fuente(bandeja, "ventas"), corte)
    petroleo = _recortar(_leer_fuente(bandeja, "petroleo"), corte)
    feriados = _recortar(
        _leer_fuente(bandeja, "feriados"), corte + pd.Timedelta(days=dias_calendario_futuro)
    )
    tiendas = _leer_fuente(bandeja, "tiendas", obligatoria=False)

    if ventas["date"].notna().sum() == 0:
        raise ValueError(f"La bandeja no tiene ventas anteriores o iguales a {corte.date()}")

    return DatosBandeja(ventas, petroleo, feriados, tiendas, corte)
