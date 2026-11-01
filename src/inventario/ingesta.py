"""Ingesta de la bandeja de entrada (bandeja_origen/) según lo entregado a cada fecha.

La bandeja simula el sistema de origen: un archivo por fuente y por fecha de entrega,

    bandeja_origen/<fuente>/<fuente>_<AAAA-MM-DD>.csv

donde la fecha del nombre es la FECHA DE ENTREGA (no la fecha de los datos). Un archivo
puede traer datos de días anteriores (entregas tardías) o volver a enviar filas ya
entregadas (correcciones). Reglas de lectura:

1. Solo se usan las entregas con fecha <= `fecha_conocimiento` (por defecto, la fecha de
   corte). Así, un backfill al 1-ago no ve una corrección que llegó el 10-ago.
2. Cada archivo se valida por separado; un archivo inválido se pone en cuarentena (se
   excluye y se registra) en lugar de detener todos los runs futuros.
3. Si la misma fila llega varias veces, prevalece la entrega más reciente (corrección).
4. Ventas, transacciones y petróleo se recortan a fechas de datos <= fecha de corte. Los
   feriados entregados pueden traer fechas futuras: el calendario se publica por adelantado.
5. Solo las ventas son obligatorias; si una fuente secundaria no ha llegado, se devuelve
   vacía y el pipeline continúa.

La bandeja es un sistema externo: este módulo la lee, pero nunca la modifica.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd

BANDEJA_DEFAULT = Path("bandeja_origen")

FUENTES = ("ventas", "transacciones", "petroleo", "feriados", "tiendas")

COLUMNAS_REQUERIDAS = {
    "ventas": ["date", "store_nbr", "family", "sales", "onpromotion"],
    "transacciones": ["date", "store_nbr", "transactions"],
    "petroleo": ["date", "dcoilwtico"],
    "feriados": ["date", "type", "locale", "locale_name", "description", "transferred"],
    "tiendas": ["store_nbr", "city", "state"],
}

CLAVES = {
    "ventas": ["date", "store_nbr", "family"],
    "transacciones": ["date", "store_nbr"],
    "petroleo": ["date"],
    "feriados": ["date", "type", "locale", "locale_name", "description"],
    "tiendas": ["store_nbr"],
}

# Fuentes cuyos datos no pueden ser posteriores a la fecha de corte.
CON_CORTE = ("ventas", "transacciones", "petroleo")

PATRON_ARCHIVO = re.compile(r"^(?P<fuente>[a-z]+)_(?P<fecha>\d{4}-\d{2}-\d{2})\.csv$")


class DatosNoDisponiblesError(RuntimeError):
    """La bandeja todavía no tiene la información necesaria para la fecha solicitada."""


@dataclass(frozen=True)
class DatosBandeja:
    ventas: pd.DataFrame
    petroleo: pd.DataFrame
    feriados: pd.DataFrame
    tiendas: pd.DataFrame | None
    fecha_corte: pd.Timestamp
    transacciones: pd.DataFrame | None = None
    fecha_conocimiento: pd.Timestamp | None = None
    entregas: tuple[str, ...] = ()
    cuarentena: tuple[tuple[str, str], ...] = ()
    correcciones: dict[str, int] = field(default_factory=dict)


def normalizar_fecha(fecha: str | date | datetime | pd.Timestamp) -> pd.Timestamp:
    """Convierte cualquier representación de fecha (incluido un datetime con zona horaria
    de Airflow) a un Timestamp sin hora ni zona."""
    ts = pd.Timestamp(fecha)
    return pd.Timestamp(ts.date())


def listar_entregas(bandeja: Path, fuente: str, hasta: pd.Timestamp) -> list[tuple[pd.Timestamp, Path]]:
    """Archivos de `fuente` entregados hasta `hasta`, en orden de entrega."""
    directorio = Path(bandeja) / fuente
    if not directorio.is_dir():
        return []
    entregas = []
    for ruta in directorio.iterdir():
        coincide = PATRON_ARCHIVO.match(ruta.name)
        if not coincide or coincide["fuente"] != fuente:
            continue
        fecha = pd.Timestamp(coincide["fecha"])
        if fecha <= hasta:
            entregas.append((fecha, ruta))
    return sorted(entregas, key=lambda e: (e[0], e[1].name))


def _vacia(fuente: str) -> pd.DataFrame:
    columnas = {c: pd.Series(dtype="object") for c in COLUMNAS_REQUERIDAS[fuente]}
    if "date" in columnas:
        columnas["date"] = pd.Series(dtype="datetime64[ns]")
    return pd.DataFrame(columnas)


def _leer_fuente(
    entregas: list[tuple[pd.Timestamp, Path]], fuente: str
) -> tuple[pd.DataFrame, list[tuple[str, str]], int]:
    """Concatena las entregas válidas. Devuelve (datos, cuarentena, filas corregidas)."""
    from inventario import validacion  # import diferido: validacion importa este módulo

    validar = getattr(validacion, f"validar_{fuente}")
    partes, cuarentena = [], []
    for _, ruta in entregas:
        df = pd.read_csv(ruta)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        try:
            validar(df)
        except validacion.DatosInvalidosError as error:
            cuarentena.append((f"{fuente}/{ruta.name}", str(error)))
            continue
        partes.append(df)

    if not partes:
        return _vacia(fuente), cuarentena, 0

    datos = pd.concat(partes, ignore_index=True)
    total = len(datos)
    datos = datos.drop_duplicates(subset=CLAVES[fuente], keep="last").reset_index(drop=True)
    return datos, cuarentena, total - len(datos)


def leer_bandeja(
    fecha_corte: str | date | datetime | pd.Timestamp,
    bandeja: str | Path = BANDEJA_DEFAULT,
    fecha_conocimiento: str | date | datetime | pd.Timestamp | None = None,
) -> DatosBandeja:
    """Lee lo entregado hasta `fecha_conocimiento` con datos hasta `fecha_corte`.

    - Uso normal: `fecha_conocimiento` = `fecha_corte` (lo que se sabía al emitir).
    - Reprocesamiento: `fecha_conocimiento` > `fecha_corte` incorpora correcciones
      entregadas después, sin usar datos posteriores al corte.

    Lanza `DatosNoDisponiblesError` si todavía no se ha entregado ninguna venta hasta el
    corte. No modifica la bandeja: llamarla varias veces devuelve el mismo resultado.
    """
    bandeja = Path(bandeja)
    if not bandeja.is_dir():
        raise FileNotFoundError(f"No existe la bandeja de entrada: {bandeja}")

    corte = normalizar_fecha(fecha_corte)
    conocimiento = normalizar_fecha(fecha_conocimiento) if fecha_conocimiento is not None else corte
    if conocimiento < corte:
        raise ValueError("fecha_conocimiento no puede ser anterior a fecha_corte")

    tablas: dict[str, pd.DataFrame] = {}
    usadas: list[str] = []
    cuarentena: list[tuple[str, str]] = []
    correcciones: dict[str, int] = {}
    for fuente in FUENTES:
        entregas = listar_entregas(bandeja, fuente, conocimiento)
        usadas += [f"{fuente}/{ruta.name}" for _, ruta in entregas]
        df, aisladas, corregidas = _leer_fuente(entregas, fuente)
        cuarentena += aisladas
        correcciones[fuente] = corregidas
        if fuente in CON_CORTE:
            df = df.loc[df["date"] <= corte]
        if "date" in df.columns:
            df = df.sort_values("date", kind="stable")
        tablas[fuente] = df.reset_index(drop=True)

    if tablas["ventas"].empty:
        raise DatosNoDisponiblesError(
            f"La bandeja no tiene ventas entregadas hasta {conocimiento.date()} con fecha <= {corte.date()}"
        )

    return DatosBandeja(
        ventas=tablas["ventas"],
        petroleo=tablas["petroleo"],
        feriados=tablas["feriados"],
        tiendas=tablas["tiendas"] if not tablas["tiendas"].empty else None,
        fecha_corte=corte,
        transacciones=tablas["transacciones"],
        fecha_conocimiento=conocimiento,
        entregas=tuple(usadas),
        cuarentena=tuple(cuarentena),
        correcciones=correcciones,
    )


def exigir_ventas_del_dia(datos: DatosBandeja) -> DatosBandeja:
    """Comprueba que la bandeja ya recibió las ventas de la fecha de corte.

    Si la última venta es anterior al corte (entrega tardía o en cuarentena), procesar esa
    fecha usaría datos viejos como si fueran del día; en ese caso se lanza
    `DatosNoDisponiblesError` (los DAGs lo convierten en una tarea omitida).
    """
    ultima = datos.ventas["date"].max()
    if pd.isna(ultima) or ultima < datos.fecha_corte:
        hasta = "ninguna fecha" if pd.isna(ultima) else str(ultima.date())
        raise DatosNoDisponiblesError(
            f"La bandeja tiene ventas hasta {hasta}; aún no hay datos de {datos.fecha_corte.date()}"
        )
    return datos
