"""Ejecución de un periodo completo con backfills, en el orden correcto y de a uno.

Para un rango de fechas lógicas `ds` (inclusive), lanza y espera:

1. DAG 05 (entrenamiento): el modelo vigente de cada día queda decidido antes de pronosticar.
2. DAG 01 (ingesta y validación), que dispara el 02 (features) y este el 03 (pronóstico,
   pedido y versionado): se espera a que terminen los tres.
3. DAG 04 (seguimiento): evalúa los pedidos ya publicados.

Cada backfill corre con `--max-active-runs 1` (memoria, sección 5 de DECISIONES.md). Las
fechas lógicas de los DAGs programados procesan `ds` = fecha del intervalo, y el límite
superior de `airflow backfill create` no incluye su propio día: para llegar a `hasta` se
pide `hasta + 1`.

Se ejecuta dentro del contenedor del scheduler (scripts/ejecutar_periodo.py).
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from datetime import date, timedelta
from typing import Callable, Iterable

log = logging.getLogger(__name__)

ETAPAS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("05_entrenamiento_modelo", ("05_entrenamiento_modelo",)),
    ("01_ingesta_y_validacion", ("01_ingesta_y_validacion", "02_preparacion_y_features", "03_pronostico_y_pedido")),
    ("04_seguimiento_desempeno", ("04_seguimiento_desempeno",)),
)
ESTADOS_ACTIVOS = {"queued", "running"}
REPROCESO = {"ninguno": "none", "fallidos": "failed", "todos": "completed"}

Ejecutor = Callable[..., subprocess.CompletedProcess]


def rango_backfill(desde: str, hasta: str) -> tuple[str, str]:
    """(--from-date, --to-date) para procesar los `ds` de `desde` a `hasta`, inclusive."""
    inicio, fin = date.fromisoformat(desde), date.fromisoformat(hasta)
    if fin < inicio:
        raise ValueError(f"hasta ({hasta}) es anterior a desde ({desde})")
    return inicio.isoformat(), (fin + timedelta(days=1)).isoformat()


def _airflow(argumentos: list[str], ejecutar: Ejecutor) -> str:
    resultado = ejecutar(["airflow", *argumentos], capture_output=True, text=True)
    if resultado.returncode != 0:
        raise RuntimeError(f"airflow {' '.join(argumentos)} falló: {(resultado.stderr or resultado.stdout).strip()[-800:]}")
    return resultado.stdout


def runs_activos(dag_ids: Iterable[str], ejecutar: Ejecutor = subprocess.run) -> int:
    """Cantidad de runs en cola o en ejecución de los DAGs indicados."""
    total = 0
    for dag_id in dag_ids:
        salida = _airflow(["dags", "list-runs", dag_id, "-o", "json"], ejecutar)
        inicio = salida.find("[")  # la CLI puede imprimir avisos antes del JSON
        runs = json.loads(salida[inicio:]) if inicio >= 0 else []
        total += sum(1 for r in runs if r.get("state") in ESTADOS_ACTIVOS)
    return total


def esperar_inactividad(
    dag_ids: Iterable[str],
    ejecutar: Ejecutor = subprocess.run,
    intervalo: float = 15.0,
    confirmaciones: int = 3,
    limite: float = 6 * 3600,
    dormir: Callable[[float], None] = time.sleep,
) -> None:
    """Espera a que no haya runs activos en `confirmaciones` consultas seguidas.

    Varias confirmaciones porque entre que un DAG termina y dispara al siguiente hay un
    instante sin runs activos.
    """
    dag_ids = tuple(dag_ids)
    seguidas, esperado = 0, 0.0
    while seguidas < confirmaciones:
        if esperado > limite:
            raise TimeoutError(f"Siguen runs activos en {dag_ids} después de {limite:.0f} s")
        seguidas = seguidas + 1 if runs_activos(dag_ids, ejecutar) == 0 else 0
        if seguidas < confirmaciones:
            dormir(intervalo)
            esperado += intervalo


def ejecutar_periodo(
    desde: str,
    hasta: str,
    reproceso: str = "fallidos",
    ejecutar: Ejecutor = subprocess.run,
    **kwargs_espera,
) -> list[str]:
    """Lanza los backfills de las tres etapas en orden y espera cada una. Devuelve los DAGs
    lanzados. `reproceso`: qué hacer con runs que ya existían (ninguno, fallidos, todos)."""
    if reproceso not in REPROCESO:
        raise ValueError(f"reproceso debe ser uno de {sorted(REPROCESO)}")
    inicio, fin = rango_backfill(desde, hasta)
    lanzados = []
    for dag_id, a_esperar in ETAPAS:
        log.info("Backfill de %s: ds %s a %s", dag_id, desde, hasta)
        _airflow(
            ["backfill", "create", "--dag-id", dag_id, "--from-date", inicio, "--to-date", fin,
             "--max-active-runs", "1", "--reprocess-behavior", REPROCESO[reproceso]],
            ejecutar,
        )
        lanzados.append(dag_id)
        esperar_inactividad(a_esperar, ejecutar, **kwargs_espera)
    return lanzados
