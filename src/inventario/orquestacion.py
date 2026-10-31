"""Utilidades para los DAGs: fecha lógica de ejecución y rutas de cada etapa.

No importa pandas ni scikit-learn para que el parseo de los DAGs sea rápido.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

BASE_DEFAULT = Path(__file__).resolve().parents[2]


def _a_iso(valor: Any) -> str:
    if isinstance(valor, datetime):
        return valor.date().isoformat()
    if isinstance(valor, date):
        return valor.isoformat()
    texto = str(valor).strip()
    return date.fromisoformat(texto[:10]).isoformat()


def resolver_fecha(contexto: Mapping[str, Any]) -> str:
    """Fecha lógica (YYYY-MM-DD) de una ejecución de Airflow.

    Prioridad:
    1. `ds` en la configuración del run o en los params (DAGs disparados por otro DAG
       o ejecuciones manuales con fecha explícita).
    2. `ds` del contexto (ejecuciones programadas).
    3. Fecha de `run_after` o `logical_date` del DagRun (Airflow 3 permite runs manuales
       sin fecha lógica).
    """
    dag_run = contexto.get("dag_run")
    conf = getattr(dag_run, "conf", None) or {}
    params = contexto.get("params") or {}
    for candidato in (conf.get("ds"), params.get("ds"), contexto.get("ds")):
        if candidato:
            return _a_iso(candidato)
    for atributo in ("logical_date", "run_after"):
        valor = getattr(dag_run, atributo, None)
        if valor:
            return _a_iso(valor)
    raise ValueError("No se pudo determinar la fecha lógica de la ejecución")


@dataclass(frozen=True)
class Rutas:
    """Ubicación de los artefactos de cada etapa para una fecha lógica `ds`."""

    base: Path
    ds: str

    @property
    def bandeja(self) -> Path:
        return Path(os.environ.get("INVENTARIO_BANDEJA", self.base / "bandeja_origen"))

    @property
    def raw(self) -> Path:
        return self.base / "data" / "raw" / self.ds

    @property
    def validado(self) -> Path:
        return self.base / "data" / "processed" / self.ds / "validado"

    @property
    def features(self) -> Path:
        return self.base / "data" / "processed" / self.ds / "features.parquet"

    @property
    def predicciones(self) -> Path:
        return self.base / "data" / "predictions"

    @property
    def pedido(self) -> Path:
        return self.predicciones / f"pedido_{self.ds}.csv"

    @property
    def seguimiento(self) -> Path:
        return self.predicciones / f"seguimiento_{self.ds}.csv"


def rutas(ds: str, base: str | Path | None = None) -> Rutas:
    """Rutas para `ds`. La base se toma de INVENTARIO_DIR o, por defecto, de la raíz del repo."""
    return Rutas(Path(base or os.environ.get("INVENTARIO_DIR", BASE_DEFAULT)), _a_iso(ds))
