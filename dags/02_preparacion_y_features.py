"""DAG 02 - Preparación y features.

Lee los datos validados de data/processed/<ds>/validado/, construye las variables
explicativas (lags 7/14/28, medias móviles por tienda y familia, promociones, feriados,
petróleo, calendario) para la historia reciente y las filas del horizonte, y las guarda en
data/processed/<ds>/features.parquet. La escritura reemplaza la de la misma fecha, así que
re-ejecutar es idempotente. Al terminar dispara el DAG 03.

Lo dispara el DAG 01 con conf={"ds": ..., "conocimiento": ...}; también se puede lanzar a
mano con ese conf.
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


def _xcom(clave: str) -> str:
    return "{{ ti.xcom_pull(task_ids='resolver_fecha', key='" + clave + "') }}"


@dag(
    dag_id="02_preparacion_y_features",
    description="Features de series de tiempo a partir de los datos validados",
    schedule=None,
    start_date=pendulum.datetime(2017, 1, 1, tz=ZONA),
    catchup=False,
    default_args={"owner": "inventario", "retries": 2, "retry_delay": timedelta(minutes=5)},
    params={
        "ds": Param(None, type=["null", "string"], description="Fecha lógica YYYY-MM-DD"),
        "conocimiento": Param(None, type=["null", "string"], description="Fecha de conocimiento (reproceso)"),
    },
    tags=["inventario", "features"],
)
def preparacion_y_features():
    @task(multiple_outputs=True)
    def resolver_fecha() -> dict[str, str]:
        from inventario.orquestacion import resolver_conocimiento, resolver_fecha as resolver, rutas

        contexto = get_current_context()
        r = rutas(resolver(contexto), conocimiento=resolver_conocimiento(contexto))
        return {"ds": r.ds, "conocimiento": r.conocimiento or "", "clave": r.clave}

    @task
    def construir_features(fecha: str, conocimiento: str) -> str:
        from inventario.almacen import cargar_datos, guardar_tabla
        from inventario.config import cargar_costos
        from inventario.orquestacion import rutas
        from inventario.predecir import preparar_features_prediccion

        r = rutas(fecha, conocimiento=conocimiento or None)
        horizonte = int(cargar_costos()["parametros_pedido"]["horizonte_dias"])
        features = preparar_features_prediccion(cargar_datos(r.validado), horizonte)
        return str(guardar_tabla(features, r.features))

    fecha = resolver_fecha()
    features = construir_features(fecha["ds"], fecha["conocimiento"])

    disparar = TriggerDagRunOperator(
        task_id="disparar_pronostico",
        trigger_dag_id="03_pronostico_y_pedido",
        trigger_run_id="desde_02__" + _xcom("clave"),
        conf={"ds": _xcom("ds"), "conocimiento": _xcom("conocimiento")},
        reset_dag_run=True,
        wait_for_completion=False,
    )
    features >> disparar


preparacion_y_features()
