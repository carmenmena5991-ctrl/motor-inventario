"""Pronóstico con el modelo @Champions y recomendación de pedido."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from inventario.config import cargar_costos
from inventario.entrenar import (
    ALIAS_CAMPEON,
    MODELO_REGISTRADO_DEFAULT,
    columnas_modelo,
    resolver_tracking_uri,
)
from inventario.features import CLAVES, LAGS, VENTANAS, construir_features
from inventario.ingesta import BANDEJA_DEFAULT, DatosBandeja, leer_bandeja
from inventario.validacion import validar_datos

DIRECTORIO_PREDICCIONES = Path("data/predictions")


def cargar_modelo_champions(
    nombre_modelo: str = MODELO_REGISTRADO_DEFAULT,
    alias: str = ALIAS_CAMPEON,
    tracking_uri: str | None = None,
):
    """Descarga desde MLflow la versión del modelo que tiene el alias indicado."""
    mlflow.set_tracking_uri(resolver_tracking_uri(tracking_uri))
    return mlflow.pyfunc.load_model(f"models:/{nombre_modelo}@{alias}")


def crear_filas_futuras(
    ventas: pd.DataFrame,
    fecha_corte: pd.Timestamp,
    horizonte: int,
    promociones: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Una fila por (día del horizonte, tienda, familia) con `sales` vacío.

    `promociones` (date, store_nbr, family, onpromotion) permite informar promociones ya
    planificadas; si no se entrega, se asume que no hay promoción.
    """
    combinaciones = ventas[CLAVES].drop_duplicates()
    fechas = pd.date_range(fecha_corte + pd.Timedelta(days=1), periods=horizonte, freq="D")
    futuras = combinaciones.merge(pd.DataFrame({"date": fechas}), how="cross")
    futuras["sales"] = np.nan
    if promociones is not None:
        futuras = futuras.merge(promociones[["date", *CLAVES, "onpromotion"]], on=["date", *CLAVES], how="left")
        futuras["onpromotion"] = futuras["onpromotion"].fillna(0)
    else:
        futuras["onpromotion"] = 0
    return futuras[["date", *CLAVES, "sales", "onpromotion"]]


def dias_historia_necesarios(horizonte: int) -> int:
    """Días de historia que alimentan los lags y medias móviles del horizonte."""
    return max(*LAGS, *VENTANAS) + horizonte


def preparar_features_prediccion(
    datos: DatosBandeja,
    horizonte: int,
    promociones: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Features de la historia reciente más las filas futuras del horizonte (sales vacío)."""
    desde = datos.fecha_corte - pd.Timedelta(days=dias_historia_necesarios(horizonte))
    historia = datos.ventas.loc[datos.ventas["date"] > desde]
    futuras = crear_filas_futuras(historia, datos.fecha_corte, horizonte, promociones)
    return construir_features(
        pd.concat([historia, futuras], ignore_index=True),
        datos.petroleo,
        datos.feriados,
        datos.tiendas,
        horizonte=horizonte,
    )


def pronosticar_desde_features(modelo: Any, features: pd.DataFrame, fecha_corte: pd.Timestamp) -> pd.DataFrame:
    """Aplica el modelo a las filas posteriores al corte."""
    a_predecir = features.loc[features["date"] > fecha_corte]
    if a_predecir.empty:
        raise ValueError(f"No hay filas posteriores a {fecha_corte.date()} para pronosticar")
    prediccion = np.clip(np.asarray(modelo.predict(a_predecir[columnas_modelo()]), dtype=float), 0, None)

    resultado = a_predecir[["date", *CLAVES]].copy()
    resultado["pronostico"] = prediccion
    return resultado.reset_index(drop=True)


def pronosticar(
    modelo: Any,
    datos: DatosBandeja,
    horizonte: int,
    promociones: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Pronóstico diario (cuantil objetivo) para los `horizonte` días posteriores al corte."""
    features = preparar_features_prediccion(datos, horizonte, promociones)
    return pronosticar_desde_features(modelo, features, datos.fecha_corte)


def generar_pedido(pronostico: pd.DataFrame, fecha_corte: pd.Timestamp) -> pd.DataFrame:
    """Pedido sugerido por tienda y familia = suma del pronóstico del horizonte, redondeada
    hacia arriba a unidades enteras.

    Sumar cuantiles diarios da una cobertura algo mayor que el cuantil de la demanda total
    (supuesto conservador ante quiebres, que cuestan más que la merma).
    """
    pedido = (
        pronostico.groupby(CLAVES, as_index=False)
        .agg(pronostico_horizonte=("pronostico", "sum"), dias_horizonte=("date", "nunique"))
    )
    pedido["pedido_sugerido"] = np.ceil(pedido["pronostico_horizonte"].round(6)).astype(int)
    pedido.insert(0, "fecha_emision", fecha_corte.date())
    return pedido


def guardar_pedido(pedido: pd.DataFrame, fecha_corte: pd.Timestamp, directorio: str | Path = DIRECTORIO_PREDICCIONES) -> Path:
    """Escribe data/predictions/pedido_<fecha>.csv. Sobrescribe el archivo de esa fecha, de
    modo que re-ejecutar el mismo día no duplica resultados."""
    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)
    ruta = directorio / f"pedido_{fecha_corte.date()}.csv"
    pedido.to_csv(ruta, index=False)
    return ruta


def ejecutar_prediccion(
    fecha_corte,
    bandeja: str | Path = BANDEJA_DEFAULT,
    ruta_costos: str | Path | None = None,
    directorio_salida: str | Path = DIRECTORIO_PREDICCIONES,
    modelo: Any = None,
    promociones: pd.DataFrame | None = None,
    **kwargs_modelo,
) -> Path:
    """Pipeline completo de inferencia (para Airflow): ingesta -> validación -> @Champions ->
    pronóstico -> pedido sugerido en disco."""
    config = cargar_costos(ruta_costos)
    horizonte = int(config["parametros_pedido"]["horizonte_dias"])
    datos = validar_datos(leer_bandeja(fecha_corte, bandeja, dias_calendario_futuro=horizonte))
    modelo = modelo if modelo is not None else cargar_modelo_champions(**kwargs_modelo)
    pronostico = pronosticar(modelo, datos, horizonte, promociones)
    return guardar_pedido(generar_pedido(pronostico, datos.fecha_corte), datos.fecha_corte, directorio_salida)
