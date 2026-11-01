"""Compara el costo real de inventario de distintos cuantiles objetivo en varios periodos.

Para cada corte mensual C:
  1. Entrena un modelo por cuantil con los `--dias-historia` días hasta C (misma
     separación de validación que inventario.entrenar).
  2. Emite los pedidos de C+2 a C+15 (14 días, como el DAG 03) y los evalúa contra las
     ventas reales con inventario.seguimiento.evaluar_pedido (como el DAG 04 en C+22).
No usa MLflow ni modifica @Champions. Es el análisis de DECISIONES.md, sección 7.

Uso:
    python scripts/comparar_cuantiles.py
    python scripts/comparar_cuantiles.py --desde 2016-09-24 --hasta 2017-07-24 \
        --cuantiles 0.5 0.6 0.65 0.71 0.8 0.9
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import pandas as pd  # noqa: E402

from inventario.almacen import recortar_ventas  # noqa: E402
from inventario.config import cargar_costos  # noqa: E402
from inventario.entrenar import (  # noqa: E402
    BaselinePromedioMovil,
    columnas_modelo,
    construir_modelo_ml,
    dividir_temporal,
)
from inventario.features import construir_features  # noqa: E402
from inventario.ingesta import DatosBandeja, leer_bandeja  # noqa: E402
from inventario.predecir import (  # noqa: E402
    generar_pedido,
    preparar_features_prediccion,
    pronosticar_desde_features,
)
from inventario.seguimiento import evaluar_pedido, fechas_emision_evaluables  # noqa: E402


def cortes_mensuales(desde: str, hasta: str) -> list[pd.Timestamp]:
    cortes, actual = [], pd.Timestamp(desde)
    while actual <= pd.Timestamp(hasta):
        cortes.append(actual)
        actual += pd.DateOffset(months=1)
    return cortes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--desde", default="2016-09-24", help="Primer corte de entrenamiento")
    parser.add_argument("--hasta", default="2017-07-24", help="Último corte de entrenamiento")
    parser.add_argument("--cuantiles", type=float, nargs="+", default=[0.50, 0.60, 0.65, 0.71, 0.80, 0.90])
    parser.add_argument("--dias-historia", type=int, default=365)
    parser.add_argument("--bandeja", type=Path, default=RAIZ / "bandeja_origen")
    parser.add_argument("--salida", type=Path, default=RAIZ / "data" / "processed" / "comparacion_cuantiles")
    args = parser.parse_args()

    inicio = time.time()
    config = cargar_costos()
    h = int(config["parametros_pedido"]["horizonte_dias"])
    dias = int(config["evaluacion"]["dias_historicos_comparar"])
    cortes = cortes_mensuales(args.desde, args.hasta)
    ultima_evaluacion = cortes[-1] + pd.Timedelta(days=h + dias + 1)
    completo = leer_bandeja(ultima_evaluacion, args.bandeja)

    def a_fecha(fecha: pd.Timestamp, dias_futuro: int = 0) -> DatosBandeja:
        """Lo que la bandeja tenía disponible en `fecha` (sin leakage)."""
        v, p, f = completo.ventas, completo.petroleo, completo.feriados
        return DatosBandeja(
            v[v["date"] <= fecha], p[p["date"] <= fecha],
            f[f["date"] <= fecha + pd.Timedelta(days=dias_futuro)], completo.tiendas, fecha,
        )

    nombres = [f"p{round(q * 100)}" for q in args.cuantiles] + ["baseline"]
    detalle = []
    for corte in cortes:
        t0 = time.time()
        datos = recortar_ventas(a_fecha(corte), args.dias_historia)
        features = construir_features(datos.ventas, datos.petroleo, datos.feriados, datos.tiendas, horizonte=h)
        entrenamiento, _ = dividir_temporal(features, dias)
        X, y = entrenamiento[columnas_modelo()], entrenamiento["sales"].to_numpy(dtype=float)
        del features, entrenamiento

        modelos = {f"p{round(q * 100)}": construir_modelo_ml(q).fit(X, y) for q in args.cuantiles}
        modelos["baseline"] = BaselinePromedioMovil(
            factor_seguridad=float(config["parametros_pedido"]["factor_seguridad_default"])
        ).fit(X, y)

        evaluacion = corte + pd.Timedelta(days=h + dias + 1)
        for emision in sorted(fechas_emision_evaluables(evaluacion, h, dias)):
            feats = preparar_features_prediccion(a_fecha(emision, h), h)
            for nombre, modelo in modelos.items():
                pedido = generar_pedido(pronosticar_desde_features(modelo, feats, emision), emision)
                metricas = evaluar_pedido(pedido, completo.ventas, emision, h, config)
                detalle.append({"corte": corte.date(), "modelo": nombre, **metricas})
        print(f"{corte.date()} listo en {time.time() - t0:.0f} s", file=sys.stderr, flush=True)

    detalle = pd.DataFrame(detalle)
    por_periodo = (
        detalle.groupby(["corte", "modelo"])
        .agg(costo=("costo_real_inventario", "sum"), quiebre=("unidades_quiebre", "sum"),
             merma=("unidades_merma", "sum"), venta=("venta_real_total", "sum"), pedido=("pedido_total", "sum"))
        .reset_index()
    )
    args.salida.mkdir(parents=True, exist_ok=True)
    detalle.to_csv(args.salida / "detalle.csv", index=False)
    por_periodo.to_csv(args.salida / "por_periodo.csv", index=False)

    referencia = nombres[-2] if "p90" not in nombres else "p90"
    costo = por_periodo.pivot(index="corte", columns="modelo", values="costo")[nombres]
    rango = costo.rank(axis=1)
    totales = por_periodo.groupby("modelo")[["costo", "quiebre", "merma", "venta", "pedido"]].sum().loc[nombres]
    resumen = pd.DataFrame({
        "costo_total_M": totales["costo"] / 1e6,
        f"vs_{referencia}": totales["costo"] / totales.loc[referencia, "costo"] - 1,
        "periodos_ganados": (rango == 1).sum().reindex(nombres),
        "rango_medio": rango.mean().reindex(nombres),
        "peor_vs_mejor_del_periodo": (costo.div(costo.min(axis=1), axis=0) - 1).max().reindex(nombres),
        "venta_cubierta": 1 - totales["quiebre"] / totales["venta"],
        "pedido/venta": totales["pedido"] / totales["venta"],
    })
    resumen.to_csv(args.salida / "resumen.csv")

    pd.set_option("display.width", 220)
    print("== Costo real por periodo (millones)")
    print((costo / 1e6).to_string(float_format=lambda v: f"{v:,.2f}"))
    print(f"\n== Costo relativo a {referencia}")
    print(costo.div(costo[referencia], axis=0).sub(1).to_string(float_format=lambda v: f"{v:+.1%}"))
    print(f"\n== Resumen de los {len(cortes)} periodos")
    print(resumen.to_string(float_format=lambda v: f"{v:,.3f}"))
    print(f"\nResultados en {args.salida} ({time.time() - inicio:.0f} s)", file=sys.stderr)


if __name__ == "__main__":
    main()
