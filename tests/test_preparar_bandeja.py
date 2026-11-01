"""Simulador del sistema de origen (inventario.simulacion) y su script."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from conftest import config_bandeja, generar_ventas
from inventario.ingesta import exigir_ventas_del_dia, leer_bandeja, DatosNoDisponiblesError
from inventario.simulacion import cargar_config_bandeja, generar_bandeja

INCIDENCIAS = {
    "entregas_tardias": {"fuentes": ["ventas", "transacciones", "petroleo"], "probabilidad": 0.15, "dias_extra": [1, 3]},
    "correcciones": {"fuentes": ["ventas"], "probabilidad": 0.3, "fraccion_filas": 0.5,
                     "factor_error": [0.5, 1.5], "dias_despues": [1, 5]},
    "entregas_corruptas": {"fuentes": ["ventas"], "probabilidad": 0.1, "dias_reenvio": 1},
}
CLAVES = ["date", "store_nbr", "family"]


def _ordenar(df):
    return df[CLAVES + ["sales"]].sort_values(CLAVES).reset_index(drop=True)


def test_un_archivo_por_fuente_y_fecha_de_entrega(origen_kaggle, tmp_path):
    destino = tmp_path / "bandeja"
    entregas, incidencias = generar_bandeja(origen_kaggle, destino, config_bandeja())
    assert incidencias.empty
    assert sorted(p.name for p in destino.iterdir()) == ["feriados", "petroleo", "tiendas", "transacciones", "ventas"]
    assert (destino / "ventas" / "ventas_2017-01-31.csv").exists()  # carga histórica
    dia = pd.read_csv(destino / "ventas" / "ventas_2017-02-15.csv")
    assert set(dia["date"]) == {"2017-02-15"}
    assert list(destino.glob("tiendas/*.csv")) == [destino / "tiendas" / "tiendas_2017-01-31.csv"]
    assert not list(destino.glob("petroleo/petroleo_2017-02-1[12].csv"))  # fin de semana


def test_sin_incidencias_reconstruye_los_datos_originales(bandeja):
    final = leer_bandeja("2017-04-30", bandeja)
    pd.testing.assert_frame_equal(_ordenar(final.ventas), _ordenar(generar_ventas().assign(date=lambda d: pd.to_datetime(d["date"]))))


def test_incidencias_y_version_final_igual_al_original(origen_kaggle, tmp_path):
    destino = tmp_path / "bandeja"
    _, incidencias = generar_bandeja(origen_kaggle, destino, config_bandeja(INCIDENCIAS))
    assert {"entrega_tardia", "correccion", "entrega_corrupta", "reenvio"} <= set(incidencias["tipo"])

    # Con todo lo entregado, la última versión de cada fila es el dato real de Kaggle.
    final = leer_bandeja("2017-04-30", destino)
    original = generar_ventas().assign(date=lambda d: pd.to_datetime(d["date"]))
    pd.testing.assert_frame_equal(_ordenar(final.ventas), _ordenar(original))
    assert final.correcciones["ventas"] > 0
    assert len(final.cuarentena) == (incidencias["tipo"] == "entrega_corrupta").sum()

    # El día antes de que llegue una corrección se ve el valor erróneo; desde su entrega, el real.
    for _, c in incidencias.query("tipo == 'correccion'").iterrows():
        def del_dia(d):
            return _ordenar(d.ventas.loc[d.ventas["date"] == c["fecha_datos"]])

        antes = leer_bandeja(c["fecha_datos"], destino, fecha_conocimiento=c["fecha_entrega"] - pd.Timedelta(days=1))
        despues = leer_bandeja(c["fecha_datos"], destino, fecha_conocimiento=c["fecha_entrega"])
        assert len(del_dia(antes)) == len(del_dia(despues)) > 0
        assert not del_dia(antes).equals(del_dia(despues))
        pd.testing.assert_frame_equal(del_dia(despues), del_dia(final))

    # El día de una entrega corrupta no hay ventas del día; tras el reenvío, sí.
    k = incidencias.query("tipo == 'entrega_corrupta'").iloc[0]
    with pytest.raises(DatosNoDisponiblesError):
        exigir_ventas_del_dia(leer_bandeja(k["fecha_datos"], destino))
    assert (leer_bandeja(k["fecha_datos"] + pd.Timedelta(days=1), destino).ventas["date"] == k["fecha_datos"]).any()


def test_es_idempotente(origen_kaggle, tmp_path):
    destino = tmp_path / "bandeja"
    generar_bandeja(origen_kaggle, destino, config_bandeja(INCIDENCIAS))
    primera = {p.relative_to(destino): p.read_bytes() for p in destino.rglob("*.csv")}
    (destino / "ventas" / "ventas_2099-01-01.csv").write_text("basura")  # restos de otra ejecución
    generar_bandeja(origen_kaggle, destino, config_bandeja(INCIDENCIAS))
    assert {p.relative_to(destino): p.read_bytes() for p in destino.rglob("*.csv")} == primera


def test_config_del_repositorio_es_valida():
    config = cargar_config_bandeja()
    assert set(config["fuentes"]) == {"ventas", "transacciones", "petroleo", "feriados", "tiendas"}
    assert pd.Timestamp(config["periodo"]["fin_carga_historica"]) < pd.Timestamp(config["periodo"]["fin_simulacion"])


def test_script(origen_kaggle, tmp_path):
    ruta_config = tmp_path / "bandeja.yaml"
    ruta_config.write_text(yaml.safe_dump(config_bandeja(INCIDENCIAS)), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "preparar_bandeja.py"
    salida = subprocess.run(
        [sys.executable, str(script), "--config", str(ruta_config), "--origen", str(origen_kaggle),
         "--destino", str(tmp_path / "bandeja"), "--incidencias", str(tmp_path / "incidencias.csv")],
        capture_output=True, text=True, check=True,
    )
    assert "Bandeja generada" in salida.stdout
    assert (tmp_path / "incidencias.csv").exists()
