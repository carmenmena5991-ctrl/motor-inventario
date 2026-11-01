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


def resolver_conocimiento(contexto: Mapping[str, Any]) -> str | None:
    """Fecha de conocimiento opcional (conf o params `conocimiento`), para reprocesar una
    fecha pasada con las entregas conocidas hasta una fecha posterior (correcciones).
    None = uso normal: solo lo entregado hasta la fecha lógica."""
    dag_run = contexto.get("dag_run")
    conf = getattr(dag_run, "conf", None) or {}
    params = contexto.get("params") or {}
    for candidato in (conf.get("conocimiento"), params.get("conocimiento")):
        if candidato and str(candidato).strip().lower() not in {"none", "null"}:
            return _a_iso(candidato)
    return None


@dataclass(frozen=True)
class Rutas:
    """Ubicación de los artefactos de cada etapa para una fecha lógica `ds`.

    En un reprocesamiento (`conocimiento` posterior a `ds`), las salidas usan la clave
    `<ds>__conocimiento_<fecha>` para no sobrescribir el resultado publicado de `ds`.
    """

    base: Path
    ds: str
    conocimiento: str | None = None

    @property
    def clave(self) -> str:
        if self.conocimiento is None:
            return self.ds
        return f"{self.ds}__conocimiento_{self.conocimiento}"

    @property
    def bandeja(self) -> Path:
        return Path(os.environ.get("INVENTARIO_BANDEJA", self.base / "bandeja_origen"))

    @property
    def raw(self) -> Path:
        return self.base / "data" / "raw" / self.clave

    @property
    def validado(self) -> Path:
        return self.base / "data" / "processed" / self.clave / "validado"

    @property
    def features(self) -> Path:
        return self.base / "data" / "processed" / self.clave / "features.parquet"

    @property
    def predicciones(self) -> Path:
        return self.base / "data" / "predictions"

    @property
    def pedido(self) -> Path:
        return self.predicciones / f"pedido_{self.clave}.csv"

    @property
    def manifiesto(self) -> Path:
        """Qué produjo el pedido: código, modelo y datos con sus huellas (versionado.py)."""
        return self.predicciones / f"pedido_{self.clave}.json"

    @property
    def seguimiento(self) -> Path:
        return self.predicciones / f"seguimiento_{self.ds}.csv"


def rutas(ds: str, base: str | Path | None = None, conocimiento: str | None = None) -> Rutas:
    """Rutas para `ds`. La base se toma de INVENTARIO_DIR o, por defecto, de la raíz del repo."""
    ds = _a_iso(ds)
    conocimiento = _a_iso(conocimiento) if conocimiento else None
    if conocimiento is not None and conocimiento < ds:
        raise ValueError("conocimiento no puede ser anterior a ds")
    if conocimiento == ds:
        conocimiento = None
    return Rutas(Path(base or os.environ.get("INVENTARIO_DIR", BASE_DEFAULT)), ds, conocimiento)
