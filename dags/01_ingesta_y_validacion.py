"""DAG 01 - Ingesta y validación de la bandeja de entrada.

Lee de bandeja_origen/ las entregas disponibles a la fecha lógica (sin data leakage: una
entrega posterior, aunque traiga datos de días anteriores, no se usa), guarda la foto en
data/raw/<ds>/, ejecuta los controles de calidad y deja los datos aprobados en
data/processed/<ds>/validado/. Al terminar dispara el DAG 02 con la misma fecha.

Situaciones esperadas que NO fallan el run:
- ventas del día aún no entregadas (entrega tardía): el run se omite (skipped);
- archivo inválido: se pone en cuarentena (se excluye y se registra en el log y en meta.json);
- fuente secundaria ausente (petróleo, transacciones, feriados, tiendas): se continúa;
- correcciones: prevalece la última entrega.

Programación: todos los días a las 06:00 (America/Guayaquil) con intervalos de datos: el
run de las 06:00 del día D procesa ds = D-1. Para fechas pasadas se usa un backfill:

    airflow backfill create --dag-id 01_ingesta_y_validacion \
        --from-date 2017-07-25 --to-date 2017-08-09

Reprocesamiento con correcciones posteriores (no sobrescribe lo publicado de ds):

    airflow dags trigger 01_ingesta_y_validacion \
        -c '{"ds": "2017-07-20", "conocimiento": "2017-08-15"}'
"""

from __future__ import annotations

import logging
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
log = logging.getLogger(__name__)


def _xcom(clave: str) -> str:
    return "{{ ti.xcom_pull(task_ids='resolver_fecha', key='" + clave + "') }}"


@dag(
    dag_id="01_ingesta_y_validacion",
    description="Ingesta de bandeja_origen/ por fecha de entrega y controles de calidad",
    schedule=CronDataIntervalTimetable("0 6 * * *", timezone=ZONA),
    start_date=pendulum.datetime(2017, 1, 1, tz=ZONA),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "inventario", "retries": 2, "retry_delay": timedelta(minutes=5)},
    params={
        "ds": Param(None, type=["null", "string"], description="Fecha lógica YYYY-MM-DD (opcional)"),
        "conocimiento": Param(
            None, type=["null", "string"],
            description="Reprocesar con las entregas conocidas hasta esta fecha (opcional)",
        ),
    },
    tags=["inventario", "ingesta"],
)
def ingesta_y_validacion():
    @task(multiple_outputs=True)
    def resolver_fecha() -> dict[str, str]:
        from inventario.orquestacion import resolver_conocimiento, resolver_fecha as resolver, rutas

        contexto = get_current_context()
        r = rutas(resolver(contexto), conocimiento=resolver_conocimiento(contexto))
        return {"ds": r.ds, "conocimiento": r.conocimiento or "", "clave": r.clave}

    @task
    def ingestar(fecha: str, conocimiento: str) -> str:
        from airflow.sdk.exceptions import AirflowSkipException

        from inventario.almacen import guardar_datos, recortar_ventas
        from inventario.config import cargar_costos
        from inventario.ingesta import DatosNoDisponiblesError, exigir_ventas_del_dia, leer_bandeja
        from inventario.orquestacion import rutas
        from inventario.predecir import dias_historia_necesarios

        r = rutas(fecha, conocimiento=conocimiento or None)
        horizonte = int(cargar_costos()["parametros_pedido"]["horizonte_dias"])
        try:
            datos = exigir_ventas_del_dia(leer_bandeja(fecha, r.bandeja, fecha_conocimiento=conocimiento or None))
        except DatosNoDisponiblesError as error:
            # Sin ventas del día no se emite pedido; validar y el disparo del 02 se omiten.
            raise AirflowSkipException(str(error)) from error

        for archivo, motivo in datos.cuarentena:
            log.warning("Entrega en cuarentena (no se usa): %s -> %s", archivo, motivo)
        corregidas = {f: n for f, n in datos.correcciones.items() if n}
        if corregidas:
            log.info("Filas reemplazadas por entregas posteriores (correcciones): %s", corregidas)

        # Solo se conserva la historia que necesitan las features del pronóstico.
        datos = recortar_ventas(datos, dias_historia_necesarios(horizonte))
        return str(guardar_datos(datos, r.raw))

    @task
    def validar(fecha: str, conocimiento: str, ruta_raw: str) -> str:
        from inventario.almacen import cargar_datos, guardar_datos
        from inventario.orquestacion import rutas
        from inventario.validacion import validar_datos

        # Control final sobre los datos ya combinados: si falla, es un error real y detiene el run.
        datos = validar_datos(cargar_datos(ruta_raw))
        return str(guardar_datos(datos, rutas(fecha, conocimiento=conocimiento or None).validado))

    fecha = resolver_fecha()
    validado = validar(fecha["ds"], fecha["conocimiento"], ingestar(fecha["ds"], fecha["conocimiento"]))

    disparar = TriggerDagRunOperator(
        task_id="disparar_preparacion_features",
        trigger_dag_id="02_preparacion_y_features",
        trigger_run_id="desde_01__" + _xcom("clave"),
        conf={"ds": _xcom("ds"), "conocimiento": _xcom("conocimiento")},
        reset_dag_run=True,
        wait_for_completion=False,
    )
    validado >> disparar


ingesta_y_validacion()
