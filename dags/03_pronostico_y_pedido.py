"""DAG 03 - Pronóstico y pedido sugerido.

1. `asegurar_modelo`: toma la versión con alias @Champions; si todavía no existe (primer
   arranque de un clon nuevo), entrena una con lo entregado hasta ds. Así el pronóstico no
   depende de ningún paso manual.
2. `exportar_modelo`: copia esa versión a models/<nombre>/v<versión>/ (rastreado por DVC).
3. `pronosticar_y_generar_pedido`: carga ESA copia, pronostica los próximos días
   (`parametros_pedido.horizonte_dias` de config/costos.yaml) a partir de
   data/processed/<ds>/features.parquet, guarda el pronóstico diario por cuantil
   (data/predictions/pronostico_<ds>.csv: p10, p50, objetivo y p90) y el pedido
   (pedido_<ds>.csv), ambos con `modelo_version` (se sobrescriben por fecha: idempotente),
   y su manifiesto pedido_<ds>.json (commit del código, modelo y huellas).
4. `versionar`: `dvc add` de datos, pedidos y modelos, y `dvc push` si el remoto está
   montado (DECISIONES.md, sección 10).

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
    def exportar_modelo(version: str) -> str:
        from inventario.versionado import exportar_modelo as exportar

        # models/<nombre>/v<versión>/: copia rastreada por DVC; no se re-descarga si ya existe.
        return str(exportar(version))

    @task
    def pronosticar_y_generar_pedido(fecha: str, conocimiento: str, version: str, ruta_modelo: str) -> str:
        import pandas as pd

        from inventario.almacen import cargar_tabla
        from inventario.entrenar import MODELO_REGISTRADO_DEFAULT
        from inventario.orquestacion import rutas
        from inventario.predecir import (
            cargar_modelo_local,
            generar_pedido,
            guardar_pedido,
            guardar_pronostico,
            pronosticar_con_cuantiles,
        )
        from inventario.versionado import guardar_manifiesto, manifiesto_pedido

        r = rutas(fecha, conocimiento=conocimiento or None)
        corte = pd.Timestamp(fecha)
        features = cargar_tabla(r.features)
        # Se usa la copia exportada (la que versiona DVC), no la del servidor MLflow.
        modelo = cargar_modelo_local(ruta_modelo)
        pronostico = pronosticar_con_cuantiles(modelo, features, corte)
        # Un reproceso escribe pedido_<ds>__conocimiento_<fecha>.csv: el publicado no cambia.
        guardar_pronostico(pronostico, r.pronostico, modelo_version=version)
        pedido = generar_pedido(pronostico, corte, modelo_version=version)
        guardar_pedido(pedido, corte, ruta=r.pedido)
        guardar_manifiesto(manifiesto_pedido(r, MODELO_REGISTRADO_DEFAULT, version, ruta_modelo), r.manifiesto)
        return str(r.pedido)

    # DVC usa un bloqueo por repositorio: un solo versionado a la vez entre los DAGs 03, 04 y 05.
    @task(pool="versionado")
    def versionar(_pedido: str) -> dict:
        from dataclasses import asdict

        from inventario.versionado import versionar_con_dvc

        resultado = versionar_con_dvc()
        log.info("Versionado con DVC: %s", resultado)
        return asdict(resultado)

    fecha = resolver_fecha()
    version = asegurar_modelo(fecha["ds"])
    pedido = pronosticar_y_generar_pedido(fecha["ds"], fecha["conocimiento"], version, exportar_modelo(version))
    versionar(pedido)


pronostico_y_pedido()
