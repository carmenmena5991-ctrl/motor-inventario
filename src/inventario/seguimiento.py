"""Seguimiento del desempeño: compara pedidos ya publicados con las ventas reales.

Un pedido emitido en la fecha E cubre (E, E + horizonte]. Se puede evaluar cuando la
bandeja ya trae ventas reales hasta E + horizonte, así que en la fecha de corte C se
evalúan los pedidos emitidos en los `dias_historicos_comparar` días que terminan en
C - horizonte.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from inventario.config import cargar_costos
from inventario.evaluacion import costo_total_inventario, mae, rmse
from inventario.features import CLAVES
from inventario.ingesta import BANDEJA_DEFAULT, exigir_ventas_del_dia, leer_bandeja, normalizar_fecha
from inventario.validacion import validar_ventas


def fechas_emision_evaluables(fecha_corte: pd.Timestamp, horizonte: int, dias: int) -> list[pd.Timestamp]:
    """Emisiones cuyo horizonte completo ya es observable a `fecha_corte` (más reciente primero)."""
    ultima = fecha_corte - pd.Timedelta(days=horizonte)
    return [ultima - pd.Timedelta(days=i) for i in range(dias)]


def evaluar_pedido(
    pedido: pd.DataFrame,
    ventas: pd.DataFrame,
    fecha_emision: pd.Timestamp,
    horizonte: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Métricas de un pedido frente a las ventas reales de su horizonte.

    - RMSE y MAE: pronóstico del horizonte vs venta real (calidad del modelo).
    - Costo real de inventario: pedido sugerido vs venta real (impacto en el negocio).
    """
    inicio = fecha_emision + pd.Timedelta(days=1)
    fin = fecha_emision + pd.Timedelta(days=horizonte)
    if ventas["date"].max() < fin:
        raise ValueError(f"Aún no hay ventas reales hasta {fin.date()} para evaluar el pedido de {fecha_emision.date()}")

    reales = (
        ventas.loc[(ventas["date"] >= inicio) & (ventas["date"] <= fin)]
        .groupby(CLAVES, as_index=False)["sales"]
        .sum()
        .rename(columns={"sales": "venta_real"})
    )
    comparacion = pedido.merge(reales, on=CLAVES, how="left")
    # Una serie sin registros de venta en el horizonte vendió 0 unidades.
    comparacion["venta_real"] = comparacion["venta_real"].fillna(0.0)

    real = comparacion["venta_real"].to_numpy(dtype=float)
    pronostico = comparacion["pronostico_horizonte"].to_numpy(dtype=float)
    pedido_sugerido = comparacion["pedido_sugerido"].to_numpy(dtype=float)
    costos = config["costos"]

    return {
        "fecha_emision": fecha_emision.date().isoformat(),
        "series": len(comparacion),
        "venta_real_total": float(real.sum()),
        "pedido_total": float(pedido_sugerido.sum()),
        "unidades_quiebre": float((real - pedido_sugerido).clip(min=0).sum()),
        "unidades_merma": float((pedido_sugerido - real).clip(min=0).sum()),
        "rmse": rmse(real, pronostico),
        "mae": mae(real, pronostico),
        "costo_real_inventario": costo_total_inventario(
            real, pedido_sugerido, costos["quiebre_stock"], costos["merma"]
        ),
    }


def ejecutar_seguimiento(
    fecha_corte,
    bandeja: str | Path = BANDEJA_DEFAULT,
    directorio_predicciones: str | Path = Path("data/predictions"),
    ruta_costos: str | Path | None = None,
    ruta_salida: str | Path | None = None,
) -> pd.DataFrame:
    """Evalúa los pedidos publicados con horizonte ya observado y guarda el resumen.

    Devuelve un DataFrame (vacío si aún no hay pedidos evaluables). Si hay resultados, se
    escriben en `ruta_salida` (por defecto seguimiento_<fecha>.csv), sobrescribiendo.
    """
    config = cargar_costos(ruta_costos)
    horizonte = int(config["parametros_pedido"]["horizonte_dias"])
    dias = int(config["evaluacion"]["dias_historicos_comparar"])
    corte = normalizar_fecha(fecha_corte)
    directorio_predicciones = Path(directorio_predicciones)

    # Sin ventas nuevas del día no hay nada nuevo que evaluar (DatosNoDisponiblesError).
    ventas = validar_ventas(exigir_ventas_del_dia(leer_bandeja(corte, bandeja)).ventas)

    filas = []
    for emision in fechas_emision_evaluables(corte, horizonte, dias):
        ruta_pedido = directorio_predicciones / f"pedido_{emision.date()}.csv"
        if ruta_pedido.exists():
            pedido = pd.read_csv(ruta_pedido)
            filas.append(evaluar_pedido(pedido, ventas, emision, horizonte, config))

    resultado = pd.DataFrame(filas)
    if not resultado.empty:
        resultado = resultado.sort_values("fecha_emision").reset_index(drop=True)
        resultado.insert(0, "fecha_corte", corte.date().isoformat())
        ruta_salida = Path(ruta_salida or directorio_predicciones / f"seguimiento_{corte.date()}.csv")
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        resultado.to_csv(ruta_salida, index=False)
    return resultado
