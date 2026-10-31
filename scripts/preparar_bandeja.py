"""Prepara bandeja_origen/ a partir del dataset descargado de Kaggle.

Copia las fuentes que usa el motor con los nombres que espera inventario.ingesta:

    train.csv            ventas            (recortadas con --hasta)
    oil.csv              petróleo          (recortado con --hasta)
    holidays_events.csv  feriados          (completo: el calendario se publica por adelantado)
    stores.csv           tiendas           (completo: no depende del tiempo)

Con --hasta se simula que la bandeja solo ha recibido datos hasta esa fecha, útil para
probar los DAGs como si se ejecutaran en el pasado. Es idempotente: cada ejecución
reemplaza los archivos de la bandeja.

Uso:
    python scripts/preparar_bandeja.py
    python scripts/preparar_bandeja.py --hasta 2017-06-30
    python scripts/preparar_bandeja.py --origen data/raw_kaggle --destino bandeja_origen
"""

from __future__ import annotations

import argparse
import os
import shutil
from datetime import date
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
CON_FECHA = ("train.csv", "oil.csv")
SIN_RECORTE = ("holidays_events.csv", "stores.csv")


def _escribir_atomico(df: pd.DataFrame, ruta: Path) -> None:
    temporal = ruta.with_name(ruta.name + ".tmp")
    df.to_csv(temporal, index=False)
    os.replace(temporal, ruta)


def preparar_bandeja(origen: Path, destino: Path, hasta: date | None = None) -> dict[str, int]:
    """Copia las fuentes de `origen` a `destino`. Devuelve las filas escritas por archivo."""
    faltantes = [n for n in (*CON_FECHA, *SIN_RECORTE) if not (origen / n).exists()]
    if faltantes:
        raise FileNotFoundError(f"Faltan archivos en {origen}: {faltantes}")

    destino.mkdir(parents=True, exist_ok=True)
    filas: dict[str, int] = {}

    for nombre in CON_FECHA:
        if hasta is None:
            shutil.copyfile(origen / nombre, destino / nombre)
            with open(destino / nombre, encoding="utf-8") as f:
                filas[nombre] = sum(1 for _ in f) - 1
        else:
            df = pd.read_csv(origen / nombre, parse_dates=["date"])
            df = df.loc[df["date"] <= pd.Timestamp(hasta)]
            df["date"] = df["date"].dt.strftime("%Y-%m-%d")
            _escribir_atomico(df, destino / nombre)
            filas[nombre] = len(df)

    for nombre in SIN_RECORTE:
        shutil.copyfile(origen / nombre, destino / nombre)
        with open(destino / nombre, encoding="utf-8") as f:
            filas[nombre] = sum(1 for _ in f) - 1

    return filas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--origen", type=Path, default=RAIZ / "data" / "raw_kaggle")
    parser.add_argument("--destino", type=Path, default=RAIZ / "bandeja_origen")
    parser.add_argument("--hasta", type=date.fromisoformat, default=None, help="Última fecha disponible (YYYY-MM-DD)")
    args = parser.parse_args()

    filas = preparar_bandeja(args.origen, args.destino, args.hasta)
    for nombre, n in filas.items():
        print(f"{args.destino / nombre}: {n} filas")


if __name__ == "__main__":
    main()
