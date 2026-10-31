"""DAG 01 - Ingesta y validación de la bandeja de entrada.

Lee bandeja_origen/ con corte en la fecha lógica (sin data leakage), guarda la foto en
data/raw/<ds>/, ejecuta los controles de calidad y deja los datos aprobados en
data/processed/<ds>/validado/. Al terminar dispara el DAG 02 con la misma fecha.

Programación: todos los días a las 06:00 (America/Guayaquil) con intervalos de datos:
el run de las 06:00 del día D procesa ds = D-1, el último día con ventas completas.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pendulum
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sdk import Param, dag, get_current_context, task
from airflow.timetables.interval import CronDataIntervalTimetable

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ZONA = "America/Guayaquil"
FECHA_XCOM = "{{ ti.xcom_pull(task_ids='resolver_fecha') }}"


@dag(
    dag_id="01_ingesta_y_validacion",
    description="Ingesta de bandeja_origen/ por fecha lógica y controles de calidad",
    schedule=CronDataIntervalTimetable("0 6 * * *", timezone=ZONA),
    start_date=pendulum.datetime(2017, 1, 1, tz=ZONA),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "inventario", "retries": 2, "retry_delay": timedelta(minutes=5)},
    params={"ds": Param(None, type=["null", "string"], description="Fecha lógica YYYY-MM-DD (opcional)")},
    tags=["inventario", "ingesta"],
)
def ingesta_y_validacion():
    @task
    def resolver_fecha() -> str:
        from inventario.orquestacion import resolver_fecha as resolver

        return resolver(get_current_context())

    @task
    def ingestar(fecha: str) -> str:
        from inventario.almacen import guardar_datos, recortar_ventas
        from inventario.config import cargar_costos
        from inventario.ingesta import leer_bandeja
        from inventario.orquestacion import rutas
        from inventario.predecir import dias_historia_necesarios

        r = rutas(fecha)
        horizonte = int(cargar_costos()["parametros_pedido"]["horizonte_dias"])
        # Calendario de feriados del horizonte incluido: se publica por adelantado.
        datos = leer_bandeja(fecha, r.bandeja, dias_calendario_futuro=horizonte)
        # Solo se conserva la historia que necesitan las features del pronóstico.
        datos = recortar_ventas(datos, dias_historia_necesarios(horizonte))
        return str(guardar_datos(datos, r.raw))

    @task
    def validar(fecha: str, ruta_raw: str) -> str:
        from inventario.almacen import cargar_datos, guardar_datos
        from inventario.orquestacion import rutas
        from inventario.validacion import validar_datos

        # Lanza DatosInvalidosError y detiene el pipeline si hay datos corruptos.
        datos = validar_datos(cargar_datos(ruta_raw))
        return str(guardar_datos(datos, rutas(fecha).validado))

    fecha = resolver_fecha()
    validado = validar(fecha, ingestar(fecha))

    disparar = TriggerDagRunOperator(
        task_id="disparar_preparacion_features",
        trigger_dag_id="02_preparacion_y_features",
        trigger_run_id=f"desde_01__{FECHA_XCOM}",
        conf={"ds": FECHA_XCOM},
        reset_dag_run=True,
        wait_for_completion=False,
    )
    validado >> disparar


ingesta_y_validacion()
