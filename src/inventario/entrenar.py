"""Entrenamiento del Baseline y del modelo de ML con registro en MLflow.

Ambos candidatos se evalúan en los últimos `evaluacion.dias_historicos_comparar` días.
El de menor Costo Total de Inventario se registra en el Model Registry y recibe el
alias `@Champions`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from inventario.config import cargar_costos
from inventario.evaluacion import evaluar
from inventario.features import COLUMNAS_CATEGORICAS, VENTANAS, construir_features, nombres_features
from inventario.ingesta import BANDEJA_DEFAULT, leer_bandeja
from inventario.validacion import validar_datos

TRACKING_URI_DEFAULT = "http://mlflow:5000"
EXPERIMENTO_DEFAULT = "motor_inventario"
MODELO_REGISTRADO_DEFAULT = "motor_inventario"
ALIAS_CAMPEON = "Champions"

HIPERPARAMETROS_DEFAULT = {
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
    "random_state": 42,
}


# MLflow 3 serializa los modelos sklearn con skops, que solo carga tipos declarados como
# confiables. Estos son los que usan nuestros dos candidatos; son seguros porque los
# artefactos los genera este mismo pipeline y viven en nuestro servidor MLflow. No añadir
# tipos sin revisarlos.
TIPOS_CONFIABLES_SKOPS = [
    "inventario.entrenar.BaselinePromedioMovil",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "sklearn.utils.validation.check_array",
    "functools.partial",
]


def columnas_modelo() -> list[str]:
    return COLUMNAS_CATEGORICAS + nombres_features()


def resolver_tracking_uri(tracking_uri: str | None = None) -> str:
    return tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", TRACKING_URI_DEFAULT)


class BaselinePromedioMovil(RegressorMixin, BaseEstimator):
    """Promedio móvil histórico por tienda y familia, inflado por el factor de seguridad.

    Usa la columna `media_movil_{ventana}` de features.py, que ya está desplazada por el
    horizonte (solo información disponible al emitir).
    """

    def __init__(self, ventana: int = max(VENTANAS), factor_seguridad: float = 0.15):
        self.ventana = ventana
        self.factor_seguridad = factor_seguridad

    def fit(self, X: pd.DataFrame, y=None):
        columna = f"media_movil_{self.ventana}"
        if columna not in X.columns:
            raise KeyError(f"Falta la columna {columna}")
        self.columna_ = columna
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        base = pd.to_numeric(X[self.columna_], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return np.clip(base * (1 + self.factor_seguridad), 0, None)


def construir_modelo_ml(cuantil: float, hiperparametros: dict[str, Any] | None = None) -> Pipeline:
    """HistGradientBoostingRegressor con pérdida cuantílica sobre el cuantil objetivo."""
    hp = {**HIPERPARAMETROS_DEFAULT, **(hiperparametros or {})}
    numericas = nombres_features()
    preprocesado = ColumnTransformer(
        [
            (
                "categoricas",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
                COLUMNAS_CATEGORICAS,
            ),
            ("numericas", "passthrough", numericas),
        ],
        remainder="drop",
    )
    es_categorica = [True] * len(COLUMNAS_CATEGORICAS) + [False] * len(numericas)
    regresor = HistGradientBoostingRegressor(
        loss="quantile", quantile=cuantil, categorical_features=es_categorica, **hp
    )
    return Pipeline([("preprocesado", preprocesado), ("regresor", regresor)])


def dividir_temporal(features: pd.DataFrame, dias_validacion: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Últimos `dias_validacion` días con ventas observadas -> validación; el resto -> entrenamiento."""
    con_objetivo = features.loc[features["sales"].notna()]
    fecha_max = con_objetivo["date"].max()
    limite = fecha_max - pd.Timedelta(days=dias_validacion)
    entrenamiento = con_objetivo.loc[con_objetivo["date"] <= limite]
    validacion = con_objetivo.loc[con_objetivo["date"] > limite]
    if entrenamiento.empty or validacion.empty:
        raise ValueError(
            f"Historia insuficiente para separar {dias_validacion} días de validación "
            f"({con_objetivo['date'].min().date()} a {fecha_max.date()})"
        )
    return entrenamiento, validacion


@dataclass
class ResultadoEntrenamiento:
    ganador: str
    version: str
    run_id: str
    metricas: dict[str, dict[str, float]] = field(default_factory=dict)


def entrenar(
    features: pd.DataFrame,
    config: dict[str, Any] | None = None,
    tracking_uri: str | None = None,
    experimento: str = EXPERIMENTO_DEFAULT,
    nombre_modelo: str = MODELO_REGISTRADO_DEFAULT,
    hiperparametros: dict[str, Any] | None = None,
    alias: str = ALIAS_CAMPEON,
    etiquetas: dict[str, str] | None = None,
) -> ResultadoEntrenamiento:
    """Entrena Baseline y ML, registra todo en MLflow y asigna `@alias` al de menor costo.

    Re-ejecutar con los mismos datos crea nuevos runs y una nueva versión registrada (es lo
    esperado en MLflow), pero los modelos y el ganador resultantes son los mismos.
    """
    config = config or cargar_costos()
    pedido = config["parametros_pedido"]
    dias_validacion = int(config["evaluacion"]["dias_historicos_comparar"])

    entrenamiento, validacion = dividir_temporal(features, dias_validacion)
    columnas = columnas_modelo()
    X_tr, y_tr = entrenamiento[columnas], entrenamiento["sales"].to_numpy(dtype=float)
    X_va, y_va = validacion[columnas], validacion["sales"].to_numpy(dtype=float)

    candidatos = {
        "baseline_promedio_movil": BaselinePromedioMovil(
            factor_seguridad=float(pedido["factor_seguridad_default"])
        ),
        "hist_gradient_boosting": construir_modelo_ml(float(pedido["cuantil_objetivo"]), hiperparametros),
    }

    mlflow.set_tracking_uri(resolver_tracking_uri(tracking_uri))
    mlflow.set_experiment(experimento)

    metricas: dict[str, dict[str, float]] = {}
    modelos_logueados: dict[str, Any] = {}
    run_ids: dict[str, str] = {}

    with mlflow.start_run(run_name="entrenamiento") as run_padre:
        mlflow.set_tags(etiquetas or {})
        mlflow.log_params(
            {
                "horizonte_dias": pedido["horizonte_dias"],
                "cuantil_objetivo": pedido["cuantil_objetivo"],
                "dias_validacion": dias_validacion,
                "costo_quiebre_stock": config["costos"]["quiebre_stock"],
                "costo_merma": config["costos"]["merma"],
                "filas_entrenamiento": len(X_tr),
                "filas_validacion": len(X_va),
            }
        )

        for nombre, modelo in candidatos.items():
            with mlflow.start_run(run_name=nombre, nested=True) as run_hijo:
                modelo.fit(X_tr, y_tr)
                pred = np.clip(modelo.predict(X_va), 0, None)
                metricas[nombre] = evaluar(y_va, pred, config)

                parametros = modelo.get_params(deep=False) if nombre.startswith("baseline") else (
                    modelo.named_steps["regresor"].get_params()
                )
                mlflow.log_params({"modelo": nombre, **{k: v for k, v in parametros.items() if k != "categorical_features"}})
                mlflow.log_metrics(metricas[nombre])
                modelos_logueados[nombre] = mlflow.sklearn.log_model(
                    modelo,
                    name="modelo",
                    signature=infer_signature(X_va.head(100), pred[:100]),
                    skops_trusted_types=TIPOS_CONFIABLES_SKOPS,
                )
                run_ids[nombre] = run_hijo.info.run_id

        ganador = min(metricas, key=lambda n: metricas[n]["costo_total_inventario"])
        mlflow.set_tag("modelo_ganador", ganador)
        mlflow.log_metric("costo_total_inventario_ganador", metricas[ganador]["costo_total_inventario"])

    version = mlflow.register_model(modelos_logueados[ganador].model_uri, nombre_modelo)
    cliente = MlflowClient()
    cliente.set_registered_model_alias(nombre_modelo, alias, version.version)
    cliente.set_model_version_tag(nombre_modelo, version.version, "modelo", ganador)
    cliente.set_model_version_tag(
        nombre_modelo, version.version, "costo_total_inventario", str(metricas[ganador]["costo_total_inventario"])
    )

    return ResultadoEntrenamiento(
        ganador=ganador, version=str(version.version), run_id=run_padre.info.run_id, metricas=metricas
    )


def ejecutar_entrenamiento(
    fecha_corte,
    bandeja: str | Path = BANDEJA_DEFAULT,
    ruta_costos: str | Path | None = None,
    dias_historia: int | None = None,
    **kwargs,
) -> ResultadoEntrenamiento:
    """Pipeline completo: ingesta -> validación -> features -> entrenamiento (para Airflow).

    `dias_historia` limita las ventas a los últimos N días antes del corte (None = toda la
    historia). Sirve para acotar memoria y tiempo con el dataset completo.
    """
    from inventario.almacen import recortar_ventas

    config = cargar_costos(ruta_costos)
    datos = leer_bandeja(fecha_corte, bandeja)
    if dias_historia is not None:
        datos = recortar_ventas(datos, dias_historia)
    datos = validar_datos(datos)
    features = construir_features(
        datos.ventas,
        datos.petroleo,
        datos.feriados,
        datos.tiendas,
        horizonte=int(config["parametros_pedido"]["horizonte_dias"]),
    )
    etiquetas = {
        "fecha_corte": str(datos.fecha_corte.date()),
        "dias_historia": str(dias_historia or "todos"),
        **kwargs.pop("etiquetas", {}),
    }
    return entrenar(features, config, etiquetas=etiquetas, **kwargs)
