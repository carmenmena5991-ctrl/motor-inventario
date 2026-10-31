"""DAG 02 - Preparación y features.

Lee los datos validados de data/processed/<ds>/validado/, construye las variables
explicativas (lags 7/14/28, medias móviles por tienda y familia, promociones, feriados,
petróleo, calendario) para la historia reciente y las filas del horizonte, y las guarda en
data/processed/<ds>/features.parquet. La escritura reemplaza la de la misma fecha, así que
re-ejecutar es idempotente. Al terminar dispara el DAG 03.

Lo dispara el DAG 01 con conf={"ds": ...}; también se puede lanzar a mano con ese conf.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pendulum
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sdk import Param, dag, get_current_context, task

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ZONA = "America/Guayaquil"
FECHA_XCOM = "{{ ti.xcom_pull(task_ids='resolver_fecha') }}"


@dag(
    dag_id="02_preparacion_y_features",
    description="Features de series de tiempo a partir de los datos validados",
    schedule=None,
    start_date=pendulum.datetime(2017, 1, 1, tz=ZONA),
    catchup=False,
    default_args={"owner": "inventario", "retries": 2, "retry_delay": timedelta(minutes=5)},
    params={"ds": Param(None, type=["null", "string"], description="Fecha lógica YYYY-MM-DD")},
    tags=["inventario", "features"],
)
def preparacion_y_features():
    @task
    def resolver_fecha() -> str:
        from inventario.orquestacion import resolver_fecha as resolver

        return resolver(get_current_context())

    @task
    def construir_features(fecha: str) -> str:
        from inventario.almacen import cargar_datos, guardar_tabla
        from inventario.config import cargar_costos
        from inventario.orquestacion import rutas
        from inventario.predecir import preparar_features_prediccion

        r = rutas(fecha)
        horizonte = int(cargar_costos()["parametros_pedido"]["horizonte_dias"])
        features = preparar_features_prediccion(cargar_datos(r.validado), horizonte)
        return str(guardar_tabla(features, r.features))

    features = construir_features(resolver_fecha())

    disparar = TriggerDagRunOperator(
        task_id="disparar_pronostico",
        trigger_dag_id="03_pronostico_y_pedido",
        trigger_run_id=f"desde_02__{FECHA_XCOM}",
        conf={"ds": FECHA_XCOM},
        reset_dag_run=True,
        wait_for_completion=False,
    )
    features >> disparar


preparacion_y_features()
