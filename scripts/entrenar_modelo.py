"""Entrena Baseline y ML, los registra en MLflow y asigna @Champions al de menor costo.

Uso (desde Ubuntu, con el servidor MLflow de docker compose arriba):
    MLFLOW_TRACKING_URI=http://localhost:5000 \
        python scripts/entrenar_modelo.py --fecha-corte 2017-08-15 --dias-historia 365

Dentro de la red de Docker el servidor es http://mlflow:5000 (valor por defecto).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from inventario.entrenar import ejecutar_entrenamiento  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fecha-corte", required=True, help="Última fecha de datos usable (YYYY-MM-DD)")
    parser.add_argument("--bandeja", type=Path, default=RAIZ / "bandeja_origen")
    parser.add_argument("--dias-historia", type=int, default=None,
                        help="Días de historia a usar (por defecto, entrenamiento.dias_historia de params.yaml)")
    parser.add_argument("--experimento", default="motor_inventario")
    parser.add_argument("--nombre-modelo", default="motor_inventario")
    args = parser.parse_args()

    resultado = ejecutar_entrenamiento(
        args.fecha_corte,
        args.bandeja,
        dias_historia=args.dias_historia,
        experimento=args.experimento,
        nombre_modelo=args.nombre_modelo,
    )
    print(f"Candidato: {resultado.ganador} (versión {resultado.version}, run {resultado.run_id})")
    for modelo, metricas in resultado.metricas.items():
        print(f"  {modelo}: " + ", ".join(f"{k}={v:,.2f}" for k, v in metricas.items()))
    vigente = f"versión {resultado.version_vigente}" if resultado.version_vigente else "ninguno"
    costo = f", costo {resultado.costo_vigente:,.2f}" if resultado.costo_vigente is not None else ""
    print(f"Vigente al entrenar: {vigente}{costo}")
    print(f"Decisión: {'PROMOVIDO a @Champions' if resultado.promovido else 'no promovido'} ({resultado.motivo})")


if __name__ == "__main__":
    main()
