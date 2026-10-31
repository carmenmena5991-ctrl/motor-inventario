"""Métricas técnicas (RMSE, MAE) y de negocio (costo total de inventario)."""

from __future__ import annotations

from typing import Any

import numpy as np


def _arrays(y_real, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_real.shape != y_pred.shape:
        raise ValueError(f"Formas distintas: {y_real.shape} vs {y_pred.shape}")
    if y_real.size == 0:
        raise ValueError("No hay observaciones para evaluar")
    return y_real, y_pred


def rmse(y_real, y_pred) -> float:
    y_real, y_pred = _arrays(y_real, y_pred)
    return float(np.sqrt(np.mean((y_real - y_pred) ** 2)))


def mae(y_real, y_pred) -> float:
    y_real, y_pred = _arrays(y_real, y_pred)
    return float(np.mean(np.abs(y_real - y_pred)))


def costo_total_inventario(y_real, y_pred, costo_quiebre: float, costo_merma: float) -> float:
    """Σ costo_quiebre · faltante + costo_merma · sobrante.

    faltante = max(real - pedido, 0)  -> ventas perdidas por quiebre de stock
    sobrante = max(pedido - real, 0)  -> exceso de inventario / merma
    """
    y_real, y_pred = _arrays(y_real, y_pred)
    faltante = np.clip(y_real - y_pred, 0, None)
    sobrante = np.clip(y_pred - y_real, 0, None)
    return float(np.sum(costo_quiebre * faltante + costo_merma * sobrante))


def evaluar(y_real, y_pred, config: dict[str, Any]) -> dict[str, float]:
    """Calcula todas las métricas usando los costos de config/costos.yaml."""
    costos = config["costos"]
    return {
        "rmse": rmse(y_real, y_pred),
        "mae": mae(y_real, y_pred),
        "costo_total_inventario": costo_total_inventario(
            y_real, y_pred, costos["quiebre_stock"], costos["merma"]
        ),
    }
