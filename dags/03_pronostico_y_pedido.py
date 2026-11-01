"""DAG 03 - Pronóstico y pedido sugerido.

1. `asegurar_modelo`: toma la versión con alias @Champions; si todavía no existe (primer
   arranque de un clon nuevo), entrena una con lo entregado hasta ds. Así el pronóstico no
   depende de ningún paso manual.
2. `pronosticar_y_generar_pedido`: carga ESA versión, pronostica los próximos días
   (`parametros_pedido.horizonte_dias` de config/costos.yaml) a partir de
   data/processed/<ds>/features.parquet y guarda data/predictions/pedido_<ds>.csv, con la
   columna `modelo_version` (se sobrescribe por fecha: idempotente).

Lo dispara el DAG 02 con conf={"ds": ...}. Un solo run a la vez, para que dos backfills
no entrenen en paralelo el modelo inicial. El servidor MLflow se toma de
MLFLOW_TRACKING_URI (por defecto http://mlflow:5000).
"""

from __future__ import annotations

import logging
import sys
from datetime import timedelta
from pathlib import Path

import pendulum
from airflow.sdk import Param, dag, get_current_context, task

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ZONA = "America/Guayaquil"
log = logging.getLogger(__name__)


@dag(
    dag_id="03_pronostico_y_pedido",
    description="Pronóstico con el modelo @Champions y pedido sugerido",
    schedule=None,
    start_date=pendulum.datetime(2017, 1, 1, tz=ZONA),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "inventario", "retries": 2, "retry_delay": timedelta(minutes=5)},
    params={
        "ds": Param(None, type=["null", "string"], description="Fecha lógica YYYY-MM-DD"),
        "conocimiento": Param(None, type=["null", "string"], description="Fecha de conocimiento (reproceso)"),
    },
    tags=["inventario", "prediccion"],
)
def pronostico_y_pedido():
    @task(multiple_outputs=True)
    def resolver_fecha() -> dict[str, str]:
        from inventario.orquestacion import resolver_conocimiento, resolver_fecha as resolver, rutas

        contexto = get_current_context()
        r = rutas(resolver(contexto), conocimiento=resolver_conocimiento(contexto))
        return {"ds": r.ds, "conocimiento": r.conocimiento or "", "clave": r.clave}

    # Puede entrenar: comparte el pool de 1 slot con DAG 05.entrenar_y_decidir (memoria).
    @task(pool="entrenamiento")
    def asegurar_modelo(fecha: str) -> str:
        from inventario.entrenar import asegurar_modelo_vigente
        from inventario.orquestacion import rutas

        version, entrenado = asegurar_modelo_vigente(fecha, rutas(fecha).bandeja)
        if entrenado:
            log.info("No había modelo vigente: se entrenó la versión %s con datos hasta %s", version, fecha)
        return version

    @task
    def pronosticar_y_generar_pedido(fecha: str, conocimiento: str, version: str) -> str:
        import pandas as pd

        from inventario.almacen import cargar_tabla
        from inventario.orquestacion import rutas
        from inventario.predecir import (
            cargar_modelo_champions,
            generar_pedido,
            guardar_pedido,
            pronosticar_desde_features,
        )

        r = rutas(fecha, conocimiento=conocimiento or None)
        corte = pd.Timestamp(fecha)
        features = cargar_tabla(r.features)
        modelo = cargar_modelo_champions(version=version)
        pronostico = pronosticar_desde_features(modelo, features, corte)
        # Un reproceso escribe pedido_<ds>__conocimiento_<fecha>.csv: el publicado no cambia.
        pedido = generar_pedido(pronostico, corte, modelo_version=version)
        return str(guardar_pedido(pedido, corte, ruta=r.pedido))

    fecha = resolver_fecha()
    pronosticar_y_generar_pedido(fecha["ds"], fecha["conocimiento"], asegurar_modelo(fecha["ds"]))


pronostico_y_pedido()
