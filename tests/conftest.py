"""Datos sintéticos con el esquema de la bandeja (Favorita) para las pruebas."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

FECHA_INICIO = "2017-01-01"
DIAS = 120
TIENDAS = [1, 2]
FAMILIAS = ["GROCERY I", "BEVERAGES"]


def generar_ventas(dias: int = DIAS, semilla: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    fechas = pd.date_range(FECHA_INICIO, periods=dias, freq="D")
    filas = []
    for tienda in TIENDAS:
        for i, familia in enumerate(FAMILIAS):
            base = 20 * tienda + 10 * i
            estacional = 5 * np.sin(2 * np.pi * fechas.day_of_week / 7)
            promo = rng.integers(0, 3, size=dias)
            ventas = np.clip(base + estacional + 3 * promo + rng.normal(0, 2, size=dias), 0, None)
            filas.append(
                pd.DataFrame(
                    {
                        "date": fechas,
                        "store_nbr": tienda,
                        "family": familia,
                        "sales": ventas.round(2),
                        "onpromotion": promo,
                    }
                )
            )
    df = pd.concat(filas, ignore_index=True)
    df.insert(0, "id", range(len(df)))
    return df


def generar_petroleo(dias: int = DIAS) -> pd.DataFrame:
    fechas = pd.date_range(FECHA_INICIO, periods=dias, freq="D")
    precio = 50 + np.arange(dias) * 0.1
    df = pd.DataFrame({"date": fechas, "dcoilwtico": precio})
    df.loc[df["date"].dt.day_of_week >= 5, "dcoilwtico"] = np.nan  # fines de semana sin cotización
    return df


def generar_feriados() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2017-01-01", "2017-02-27", "2017-03-10", "2017-04-14", "2017-04-29", "2017-05-02"]),
            "type": ["Holiday", "Holiday", "Holiday", "Holiday", "Work Day", "Holiday"],
            "locale": ["National", "National", "Local", "National", "National", "National"],
            "locale_name": ["Ecuador", "Ecuador", "Quito", "Ecuador", "Ecuador", "Ecuador"],
            "description": ["Primer dia del ano", "Carnaval", "Fundacion local", "Viernes Santo", "Recupero", "Futuro"],
            "transferred": [False, False, False, False, False, False],
        }
    )


def generar_tiendas() -> pd.DataFrame:
    return pd.DataFrame(
        {"store_nbr": TIENDAS, "city": ["Quito", "Guayaquil"], "state": ["Pichincha", "Guayas"], "type": ["D", "A"], "cluster": [13, 8]}
    )


@pytest.fixture
def bandeja(tmp_path: Path) -> Path:
    ruta = tmp_path / "bandeja_origen"
    ruta.mkdir()
    generar_ventas().to_csv(ruta / "train.csv", index=False)
    generar_petroleo().to_csv(ruta / "oil.csv", index=False)
    generar_feriados().to_csv(ruta / "holidays_events.csv", index=False)
    generar_tiendas().to_csv(ruta / "stores.csv", index=False)
    return ruta


@pytest.fixture
def config() -> dict:
    return {
        "costos": {"quiebre_stock": 5.0, "merma": 2.0},
        "parametros_pedido": {"horizonte_dias": 7, "cuantil_objetivo": 0.9, "factor_seguridad_default": 0.15},
        "evaluacion": {"dias_historicos_comparar": 14},
    }


@pytest.fixture
def ruta_costos(tmp_path: Path, config: dict) -> Path:
    ruta = tmp_path / "costos.yaml"
    ruta.write_text(yaml.safe_dump(config), encoding="utf-8")
    return ruta


@pytest.fixture
def mlflow_local(tmp_path: Path) -> str:
    """Servidor MLflow simulado con SQLite y artefactos en tmp_path (sin red)."""
    import mlflow

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment("pruebas", artifact_location=(tmp_path / "artefactos").as_uri())
    return uri
