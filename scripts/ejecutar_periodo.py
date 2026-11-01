"""Ejecuta el flujo completo sobre un periodo de fechas (ds), sin más pasos manuales.

Corre dentro del contenedor del scheduler. Desde la raíz del repo:
    make periodo DESDE=2017-07-01 HASTA=2017-08-15
    # equivale a:
    docker compose exec airflow-scheduler \
        python /opt/airflow/scripts/ejecutar_periodo.py --desde 2017-07-01 --hasta 2017-08-15

Orden: DAG 05 (entrenamiento) -> DAG 01 (que dispara 02 y 03) -> DAG 04 (seguimiento),
esperando que termine cada etapa. Ver inventario.periodo.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from inventario.periodo import REPROCESO, ejecutar_periodo  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--desde", required=True, help="Primera fecha lógica ds (YYYY-MM-DD)")
    parser.add_argument("--hasta", required=True, help="Última fecha lógica ds, inclusive (YYYY-MM-DD)")
    parser.add_argument(
        "--reproceso", choices=sorted(REPROCESO), default="fallidos",
        help="Runs que ya existían: ninguno, fallidos (por defecto) o todos",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    lanzados = ejecutar_periodo(args.desde, args.hasta, args.reproceso)
    print(f"Periodo {args.desde} a {args.hasta} terminado: {', '.join(lanzados)}")
    print("Pedidos y manifiestos en data/predictions/; seguimiento en data/predictions/seguimiento_<ds>.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
