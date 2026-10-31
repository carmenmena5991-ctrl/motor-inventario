import importlib.util
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from conftest import generar_feriados, generar_petroleo, generar_tiendas, generar_ventas
from inventario.ingesta import leer_bandeja

_spec = importlib.util.spec_from_file_location(
    "preparar_bandeja", Path(__file__).resolve().parents[1] / "scripts" / "preparar_bandeja.py"
)
preparar_bandeja = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(preparar_bandeja)


@pytest.fixture
def origen(tmp_path):
    ruta = tmp_path / "raw_kaggle"
    ruta.mkdir()
    generar_ventas().to_csv(ruta / "train.csv", index=False)
    generar_petroleo().to_csv(ruta / "oil.csv", index=False)
    generar_feriados().to_csv(ruta / "holidays_events.csv", index=False)
    generar_tiendas().to_csv(ruta / "stores.csv", index=False)
    return ruta


def test_copia_completa_es_legible_por_la_ingesta(origen, tmp_path):
    destino = tmp_path / "bandeja"
    filas = preparar_bandeja.preparar_bandeja(origen, destino)
    assert filas["train.csv"] == len(generar_ventas())
    datos = leer_bandeja("2017-04-30", destino)
    assert len(datos.ventas) == len(generar_ventas())


def test_recorte_hasta_fecha(origen, tmp_path):
    destino = tmp_path / "bandeja"
    preparar_bandeja.preparar_bandeja(origen, destino, hasta=date(2017, 3, 1))
    ventas = pd.read_csv(destino / "train.csv", parse_dates=["date"])
    assert ventas["date"].max() == pd.Timestamp("2017-03-01")
    # El calendario de feriados se copia completo.
    assert len(pd.read_csv(destino / "holidays_events.csv")) == len(generar_feriados())


def test_es_idempotente(origen, tmp_path):
    destino = tmp_path / "bandeja"
    preparar_bandeja.preparar_bandeja(origen, destino, hasta=date(2017, 3, 1))
    primera = (destino / "train.csv").read_bytes()
    preparar_bandeja.preparar_bandeja(origen, destino, hasta=date(2017, 3, 1))
    assert (destino / "train.csv").read_bytes() == primera
    assert sorted(p.name for p in destino.iterdir()) == ["holidays_events.csv", "oil.csv", "stores.csv", "train.csv"]


def test_falta_archivo(origen, tmp_path):
    (origen / "oil.csv").unlink()
    with pytest.raises(FileNotFoundError, match="oil.csv"):
        preparar_bandeja.preparar_bandeja(origen, tmp_path / "bandeja")
