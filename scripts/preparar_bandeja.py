"""Genera bandeja_origen/ (sistema de origen simulado) a partir de los CSV de Kaggle.

Crea la carga histórica inicial y un archivo por fuente y fecha de entrega, con las
frecuencias, retrasos e incidencias de config/bandeja.yaml (ver inventario.simulacion).
Es idempotente: con la misma configuración produce exactamente los mismos archivos.

Uso:
    python scripts/preparar_bandeja.py
    python scripts/preparar_bandeja.py --config config/bandeja.yaml \
        --origen data/raw_kaggle --destino bandeja_origen
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from inventario.simulacion import cargar_config_bandeja, generar_bandeja  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=RAIZ / "config" / "bandeja.yaml")
    parser.add_argument("--origen", type=Path, default=RAIZ / "data" / "raw_kaggle")
    parser.add_argument("--destino", type=Path, default=RAIZ / "bandeja_origen")
    parser.add_argument(
        "--incidencias", type=Path, default=RAIZ / "data" / "processed" / "bandeja_incidencias.csv",
        help="Dónde guardar el detalle de las incidencias simuladas",
    )
    args = parser.parse_args()

    config = cargar_config_bandeja(args.config)
    entregas, incidencias = generar_bandeja(args.origen, args.destino, config)

    print(f"Bandeja generada en {args.destino}")
    resumen = entregas.groupby("fuente").agg(
        archivos=("fecha_entrega", "count"), filas=("filas", "sum"),
        primera=("fecha_entrega", "min"), ultima=("fecha_entrega", "max"),
    )
    print(resumen.to_string())
    print("\nIncidencias simuladas:")
    print(incidencias.groupby(["tipo", "fuente"]).size().to_string() if len(incidencias) else "ninguna")

    args.incidencias.parent.mkdir(parents=True, exist_ok=True)
    incidencias.to_csv(args.incidencias, index=False)
    print(f"\nDetalle de incidencias: {args.incidencias}")


if __name__ == "__main__":
    main()
