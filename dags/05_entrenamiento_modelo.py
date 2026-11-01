"""DAG 05 - Entrenamiento y promoción del modelo.

Corre todos los días a las 05:00 (antes del DAG 01) con la misma fecha lógica que el resto
(ds = último día con ventas completas), pero solo entrena cuando hace falta:
- si todavía no hay un modelo vigente a ds (ninguno promovido con datos hasta ds), o
- si ds cae en `entrenamiento.dia_reentrenamiento` de params.yaml (domingo por defecto).
Los demás días la tarea `hay_que_entrenar` corta el flujo (short-circuit) sin costo. Si
las ventas de ds todavía no llegaron, el entrenamiento se omite, igual que el DAG 01.

El entrenamiento usa solo lo entregado en la bandeja hasta ds, compara Baseline y ML, y el
mejor reemplaza al vigente únicamente si reduce el costo de inventario al menos
`promocion.mejora_minima` en la misma ventana de validación (DECISIONES.md, sección 9).
Cada candidato queda registrado en MLflow con la decisión y su motivo.
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
    dag_id="05_entrenamiento_modelo",
    description="Reentrenamiento semanal y criterio de reemplazo del modelo vigente",
    schedule=CronDataIntervalTimetable("0 5 * * *", timezone=ZONA),
    start_date=pendulum.datetime(2017, 1, 1, tz=ZONA),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "inventario", "retries": 1, "retry_delay": timedelta(minutes=5)},
    params={"ds": Param(None, type=["null", "string"], description="Fecha lógica YYYY-MM-DD (opcional)")},
    tags=["inventario", "modelo"],
)
def entrenamiento_modelo():
    @task
    def resolver_fecha() -> str:
        from inventario.orquestacion import resolver_fecha as resolver

        return resolver(get_current_context())

    @task.short_circuit
    def hay_que_entrenar(fecha: str) -> bool:
        from inventario.config import cargar_params
        from inventario.entrenar import debe_entrenar, version_vigente_a_fecha

        vigente = version_vigente_a_fecha(fecha)
        entrenar, motivo = debe_entrenar(fecha, cargar_params()["entrenamiento"]["dia_reentrenamiento"], vigente is not None)
        log.info("%s: %s (vigente: %s)", "Se entrena" if entrenar else "No se entrena", motivo, vigente or "ninguno")
        return entrenar

    # Pool de 1 slot compartido con DAG 03.asegurar_modelo: cada entrenamiento usa ~1.6 GiB y
    # todos los núcleos; varios en paralelo (p. ej. un backfill) agotan la memoria de WSL.
    @task(pool="entrenamiento")
    def entrenar_y_decidir(fecha: str) -> dict:
        from airflow.sdk.exceptions import AirflowSkipException

        from inventario.entrenar import ejecutar_entrenamiento
        from inventario.ingesta import DatosNoDisponiblesError
        from inventario.orquestacion import rutas

        try:
            # Como el resto del flujo: sin las ventas del día no se entrena (run omitido).
            resultado = ejecutar_entrenamiento(fecha, rutas(fecha).bandeja, exigir_datos_del_dia=True)
        except DatosNoDisponiblesError as error:
            raise AirflowSkipException(str(error)) from error

        resumen = {
            "fecha": fecha,
            "candidato": resultado.ganador,
            "version_candidata": resultado.version,
            "costo_candidato": resultado.metricas[resultado.ganador]["costo_total_inventario"],
            "version_vigente": resultado.version_vigente,
            "costo_vigente": resultado.costo_vigente,
            "promovido": resultado.promovido,
            "motivo": resultado.motivo,
        }
        log.info("Resultado del entrenamiento: %s", resumen)
        return resumen

    fecha = resolver_fecha()
    hay_que_entrenar(fecha) >> entrenar_y_decidir(fecha)


entrenamiento_modelo()
