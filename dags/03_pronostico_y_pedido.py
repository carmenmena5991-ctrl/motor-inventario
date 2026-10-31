"""DAG 03 - Pronóstico y pedido sugerido.

Descarga de MLflow el modelo con alias @Champions, pronostica los próximos días
(`parametros_pedido.horizonte_dias` de config/costos.yaml, 7 por defecto) a partir de
data/processed/<ds>/features.parquet y guarda la recomendación en
data/predictions/pedido_<ds>.csv (se sobrescribe por fecha: idempotente).

Lo dispara el DAG 02 con conf={"ds": ...}. El servidor MLflow se toma de
MLFLOW_TRACKING_URI (por defecto http://mlflow:5000).
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pendulum
from airflow.sdk import Param, dag, get_current_context, task

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ZONA = "America/Guayaquil"


@dag(
    dag_id="03_pronostico_y_pedido",
    description="Pronóstico con el modelo @Champions y pedido sugerido",
    schedule=None,
    start_date=pendulum.datetime(2017, 1, 1, tz=ZONA),
    catchup=False,
    default_args={"owner": "inventario", "retries": 2, "retry_delay": timedelta(minutes=5)},
    params={"ds": Param(None, type=["null", "string"], description="Fecha lógica YYYY-MM-DD")},
    tags=["inventario", "prediccion"],
)
def pronostico_y_pedido():
    @task
    def resolver_fecha() -> str:
        from inventario.orquestacion import resolver_fecha as resolver

        return resolver(get_current_context())

    @task
    def pronosticar_y_generar_pedido(ds: str) -> str:
        import pandas as pd

        from inventario.almacen import cargar_tabla
        from inventario.orquestacion import rutas
        from inventario.predecir import (
            cargar_modelo_champions,
            generar_pedido,
            guardar_pedido,
            pronosticar_desde_features,
        )

        r = rutas(ds)
        corte = pd.Timestamp(ds)
        features = cargar_tabla(r.features)
        modelo = cargar_modelo_champions()
        pronostico = pronosticar_desde_features(modelo, features, corte)
        return str(guardar_pedido(generar_pedido(pronostico, corte), corte, r.predicciones))

    pronosticar_y_generar_pedido(resolver_fecha())


pronostico_y_pedido()
