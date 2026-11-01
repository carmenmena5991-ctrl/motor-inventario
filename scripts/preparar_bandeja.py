"""Genera bandeja_origen/ (sistema de origen simulado) a partir de los CSV de Kaggle.

Crea la carga histórica inicial y un archivo por fuente y fecha de entrega, con las
frecuencias, retrasos e incidencias de config/bandeja.yaml (ver inventario.simulacion).
Es idempotente: con la misma configuración produce exactamente los mismos archivos.

Docker Compose lo ejecuta en cada arranque (servicio `preparar-bandeja`), antes que
Airflow: la primera vez genera la bandeja y después solo la regenera si cambian los CSV
de Kaggle o config/bandeja.yaml (huella en bandeja_origen/.huella_bandeja.json).

Uso manual:
    python scripts/preparar_bandeja.py
    python scripts/preparar_bandeja.py --forzar
    python scripts/preparar_bandeja.py --config config/bandeja.yaml \
        --origen data/raw_kaggle --destino bandeja_origen
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from inventario.simulacion import cargar_config_bandeja, preparar_bandeja  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=RAIZ / "config" / "bandeja.yaml")
    parser.add_argument("--origen", type=Path, default=RAIZ / "data" / "raw_kaggle")
    parser.add_argument("--destino", type=Path, default=RAIZ / "bandeja_origen")
    parser.add_argument(
        "--incidencias", type=Path, default=RAIZ / "data" / "processed" / "bandeja_incidencias.csv",
        help="Dónde guardar el detalle de las incidencias simuladas",
    )
    parser.add_argument("--forzar", action="store_true", help="Regenerar aunque las entradas no hayan cambiado")
    args = parser.parse_args()

    config = cargar_config_bandeja(args.config)
    try:
        # Sin el detalle de incidencias (p. ej. tras borrar data/processed) se regenera todo.
        forzar = args.forzar or not args.incidencias.exists()
        resultado = preparar_bandeja(args.origen, args.destino, config, forzar=forzar)
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    periodo = f"carga histórica hasta {resultado.fin_carga_historica}, entregas hasta {resultado.fin_simulacion}"
    if not resultado.generada:
        print(f"Bandeja al día en {args.destino} ({periodo}): sin cambios en las entradas")
        return 0

    entregas, incidencias = resultado.entregas, resultado.incidencias
    print(f"Bandeja generada en {args.destino} ({periodo})")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
