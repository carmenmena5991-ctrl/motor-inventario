"""Carga de la configuración versionada: costos de negocio (config/costos.yaml) y
parámetros de entrenamiento y promoción (params.yaml)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

RUTA_COSTOS_DEFAULT = Path(__file__).resolve().parents[2] / "config" / "costos.yaml"
RUTA_PARAMS_DEFAULT = Path(__file__).resolve().parents[2] / "params.yaml"


def cargar_params(ruta: str | Path | None = None) -> dict[str, Any]:
    """Lee params.yaml (argumento > variable de entorno INVENTARIO_PARAMS > raíz del repo)."""
    ruta = Path(ruta or os.environ.get("INVENTARIO_PARAMS", RUTA_PARAMS_DEFAULT))
    with open(ruta, encoding="utf-8") as f:
        params = yaml.safe_load(f)

    for seccion, claves in {
        "entrenamiento": ("dias_historia", "dia_reentrenamiento", "hiperparametros"),
        "promocion": ("mejora_minima",),
    }.items():
        faltantes = [c for c in claves if c not in (params.get(seccion) or {})]
        if faltantes:
            raise KeyError(f"Faltan claves en {ruta} -> {seccion}: {faltantes}")
    if not 0 <= int(params["entrenamiento"]["dia_reentrenamiento"]) <= 6:
        raise ValueError("entrenamiento.dia_reentrenamiento debe estar entre 0 (lunes) y 6 (domingo)")
    if not 0 <= float(params["promocion"]["mejora_minima"]) < 1:
        raise ValueError("promocion.mejora_minima debe estar en [0, 1)")
    cuantiles = params["entrenamiento"].get("cuantiles_intervalo") or []
    if not isinstance(cuantiles, list) or not all(0 < float(q) < 1 for q in cuantiles):
        raise ValueError("entrenamiento.cuantiles_intervalo debe ser una lista de valores en (0, 1)")
    return params


def cargar_costos(ruta: str | Path | None = None) -> dict[str, Any]:
    """Lee config/costos.yaml.

    Prioridad de la ruta: argumento > variable de entorno INVENTARIO_COSTOS > ruta por defecto.
    """
    ruta = Path(ruta or os.environ.get("INVENTARIO_COSTOS", RUTA_COSTOS_DEFAULT))
    with open(ruta, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for seccion, claves in {
        "costos": ("quiebre_stock", "merma"),
        "parametros_pedido": ("horizonte_dias", "cuantil_objetivo", "factor_seguridad_default"),
        "evaluacion": ("dias_historicos_comparar",),
    }.items():
        faltantes = [c for c in claves if c not in config.get(seccion, {})]
        if faltantes:
            raise KeyError(f"Faltan claves en {ruta} -> {seccion}: {faltantes}")
    return config
