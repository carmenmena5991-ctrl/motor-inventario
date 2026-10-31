"""Controles de calidad de datos.

Cada función devuelve el mismo DataFrame que recibe (sin modificarlo) si es válido, o
lanza `DatosInvalidosError` para detener el pipeline antes de contaminar los datos.
Al no tener efectos secundarios, son idempotentes.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from inventario.ingesta import DatosBandeja

COLUMNAS_REQUERIDAS = {
    "ventas": ["date", "store_nbr", "family", "sales", "onpromotion"],
    "petroleo": ["date", "dcoilwtico"],
    "feriados": ["date", "type", "locale", "locale_name", "transferred"],
    "tiendas": ["store_nbr", "city", "state"],
}


class DatosInvalidosError(ValueError):
    """Los datos de entrada no superan los controles de calidad."""


def _exigir_columnas(df: pd.DataFrame, columnas: Iterable[str], nombre: str) -> None:
    faltantes = [c for c in columnas if c not in df.columns]
    if faltantes:
        raise DatosInvalidosError(f"[{nombre}] faltan columnas requeridas: {faltantes}")


def _exigir_no_nulos(df: pd.DataFrame, columnas: Iterable[str], nombre: str) -> None:
    for col in columnas:
        n = int(df[col].isna().sum())
        if n:
            raise DatosInvalidosError(f"[{nombre}] {n} valores nulos o no parseables en '{col}'")


def _exigir_numerico_no_negativo(
    df: pd.DataFrame, col: str, nombre: str, permitir_nulos: bool = False
) -> None:
    valores = pd.to_numeric(df[col], errors="coerce")
    no_numericos = valores.isna() & df[col].notna()
    if no_numericos.any():
        raise DatosInvalidosError(f"[{nombre}] {int(no_numericos.sum())} valores no numéricos en '{col}'")
    if not permitir_nulos and valores.isna().any():
        raise DatosInvalidosError(f"[{nombre}] {int(valores.isna().sum())} valores nulos en '{col}'")
    negativos = int((valores < 0).sum())
    if negativos:
        raise DatosInvalidosError(f"[{nombre}] {negativos} valores negativos en '{col}'")


def validar_ventas(df: pd.DataFrame) -> pd.DataFrame:
    nombre = "ventas"
    _exigir_columnas(df, COLUMNAS_REQUERIDAS[nombre], nombre)
    if df.empty:
        raise DatosInvalidosError("[ventas] el DataFrame está vacío")
    _exigir_no_nulos(df, ["date", "store_nbr", "family"], nombre)
    _exigir_numerico_no_negativo(df, "sales", nombre)
    _exigir_numerico_no_negativo(df, "onpromotion", nombre)
    duplicados = int(df.duplicated(subset=["date", "store_nbr", "family"]).sum())
    if duplicados:
        raise DatosInvalidosError(f"[ventas] {duplicados} filas duplicadas por (date, store_nbr, family)")
    return df


def validar_petroleo(df: pd.DataFrame) -> pd.DataFrame:
    """El precio puede faltar algunos días (feriados de mercado), pero nunca ser negativo."""
    nombre = "petroleo"
    _exigir_columnas(df, COLUMNAS_REQUERIDAS[nombre], nombre)
    _exigir_no_nulos(df, ["date"], nombre)
    _exigir_numerico_no_negativo(df, "dcoilwtico", nombre, permitir_nulos=True)
    return df


def validar_feriados(df: pd.DataFrame) -> pd.DataFrame:
    nombre = "feriados"
    _exigir_columnas(df, COLUMNAS_REQUERIDAS[nombre], nombre)
    _exigir_no_nulos(df, ["date", "type", "locale"], nombre)
    return df


def validar_tiendas(df: pd.DataFrame) -> pd.DataFrame:
    nombre = "tiendas"
    _exigir_columnas(df, COLUMNAS_REQUERIDAS[nombre], nombre)
    _exigir_no_nulos(df, ["store_nbr"], nombre)
    if df["store_nbr"].duplicated().any():
        raise DatosInvalidosError("[tiendas] store_nbr duplicado")
    return df


def validar_datos(datos: DatosBandeja) -> DatosBandeja:
    """Valida todas las fuentes de la bandeja y comprueba que no haya leakage temporal."""
    validar_ventas(datos.ventas)
    validar_petroleo(datos.petroleo)
    validar_feriados(datos.feriados)
    if datos.tiendas is not None:
        validar_tiendas(datos.tiendas)

    for nombre, df in (("ventas", datos.ventas), ("petroleo", datos.petroleo)):
        if (df["date"] > datos.fecha_corte).any():
            raise DatosInvalidosError(
                f"[{nombre}] hay registros posteriores a la fecha de corte {datos.fecha_corte.date()}"
            )
    return datos
