from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from inventario.orquestacion import resolver_fecha, rutas


def test_prioriza_ds_de_la_configuracion_del_run():
    contexto = {
        "dag_run": SimpleNamespace(conf={"ds": "2017-03-15"}, logical_date=None, run_after=None),
        "params": {"ds": "2017-01-01"},
        "ds": "2017-02-01",
    }
    assert resolver_fecha(contexto) == "2017-03-15"


def test_usa_params_y_luego_ds_del_contexto():
    run = SimpleNamespace(conf={}, logical_date=None, run_after=None)
    assert resolver_fecha({"dag_run": run, "params": {"ds": "2017-01-01"}, "ds": "2017-02-01"}) == "2017-01-01"
    assert resolver_fecha({"dag_run": run, "params": {"ds": None}, "ds": "2017-02-01"}) == "2017-02-01"


def test_run_manual_sin_fecha_logica_usa_run_after():
    tz = timezone(timedelta(hours=-5))
    run = SimpleNamespace(conf={}, logical_date=None, run_after=datetime(2017, 3, 15, 23, 0, tzinfo=tz))
    assert resolver_fecha({"dag_run": run, "params": {}}) == "2017-03-15"


def test_sin_fecha_falla():
    with pytest.raises(ValueError):
        resolver_fecha({"dag_run": SimpleNamespace(conf={}, logical_date=None, run_after=None)})


def test_rutas_por_fecha(tmp_path):
    r = rutas("2017-03-15T00:00:00", base=tmp_path)
    assert r.ds == "2017-03-15"
    assert r.raw == tmp_path / "data" / "raw" / "2017-03-15"
    assert r.validado == tmp_path / "data" / "processed" / "2017-03-15" / "validado"
    assert r.features == tmp_path / "data" / "processed" / "2017-03-15" / "features.parquet"
    assert r.pedido == tmp_path / "data" / "predictions" / "pedido_2017-03-15.csv"
    assert r.seguimiento == tmp_path / "data" / "predictions" / "seguimiento_2017-03-15.csv"
    assert r.bandeja == tmp_path / "bandeja_origen"


def test_rutas_desde_variables_de_entorno(tmp_path, monkeypatch):
    monkeypatch.setenv("INVENTARIO_DIR", str(tmp_path))
    monkeypatch.setenv("INVENTARIO_BANDEJA", "/datos/bandeja")
    r = rutas("2017-03-15")
    assert r.base == tmp_path
    assert r.bandeja == Path("/datos/bandeja")
