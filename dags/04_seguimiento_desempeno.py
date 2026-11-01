"""DAG 04 - Seguimiento del desempeño.

Compara los pedidos ya publicados (data/predictions/pedido_<fecha>.csv) con las ventas
reales que han llegado a bandeja_origen/ y calcula el error del pronóstico (RMSE, MAE) y
el costo real de inventario (quiebres y merma según config/costos.yaml).

En la fecha ds se evalúan los pedidos emitidos en los `evaluacion.dias_historicos_comparar`
días que terminan en ds - horizonte, que son los que ya tienen todo su horizonte observado.
El resumen se guarda en data/predictions/seguimiento_<ds>.csv (sobrescribe por fecha) y se
versiona con DVC (tarea `versionar`).

Programación: diaria a las 07:00, después del DAG 01, con la misma semántica de ds.
Si la bandeja todavía no tiene ventas de ds, el run se omite (skipped).
"""

from __future__ import annotations

import logging
import sys
from datetime import timedelta
from pathlib import Path

import pendulum
from airflow.sdk import Param, dag, get_current_context, task
from airflow.timetables.interval import CronDataIntervalTimetable

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ZONA = "America/Guayaquil"
log = logging.getLogger(__name__)


@dag(
    dag_id="04_seguimiento_desempeno",
    description="Error y costo real de los pedidos publicados frente a las ventas reales",
    schedule=CronDataIntervalTimetable("0 7 * * *", timezone=ZONA),
    start_date=pendulum.datetime(2017, 1, 1, tz=ZONA),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "inventario", "retries": 2, "retry_delay": timedelta(minutes=5)},
    params={"ds": Param(None, type=["null", "string"], description="Fecha lógica YYYY-MM-DD (opcional)")},
    tags=["inventario", "seguimiento"],
)
def seguimiento_desempeno():
    @task
    def resolver_fecha() -> str:
        from inventario.orquestacion import resolver_fecha as resolver

        return resolver(get_current_context())

    @task
    def evaluar_pedidos_publicados(fecha: str) -> str | None:
        from airflow.sdk.exceptions import AirflowSkipException

        from inventario.ingesta import DatosNoDisponiblesError
        from inventario.orquestacion import rutas
        from inventario.seguimiento import ejecutar_seguimiento

        r = rutas(fecha)
        try:
            resultado = ejecutar_seguimiento(fecha, r.bandeja, r.predicciones, ruta_salida=r.seguimiento)
        except DatosNoDisponiblesError as error:
            raise AirflowSkipException(str(error)) from error
        if resultado.empty:
            log.info("No hay pedidos con horizonte completo para evaluar a %s", fecha)
            return None
        log.info("Seguimiento a %s:\n%s", fecha, resultado.to_string(index=False))
        return str(r.seguimiento)

    # Mismo pool que los DAGs 03 y 05: DVC no admite dos operaciones a la vez en el repo.
    @task(pool="versionado")
    def versionar(_seguimiento: str | None) -> dict:
        from dataclasses import asdict

        from inventario.versionado import versionar_con_dvc

        resultado = versionar_con_dvc()
        log.info("Versionado con DVC: %s", resultado)
        return asdict(resultado)

    versionar(evaluar_pedidos_publicados(resolver_fecha()))


seguimiento_desempeno()
