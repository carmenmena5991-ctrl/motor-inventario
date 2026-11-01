"""Simulador del sistema de origen (inventario.simulacion) y su script."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from conftest import config_bandeja, generar_ventas
from inventario.ingesta import exigir_ventas_del_dia, leer_bandeja, DatosNoDisponiblesError
from inventario.simulacion import (
    ARCHIVO_HUELLA,
    cargar_config_bandeja,
    generar_bandeja,
    preparar_bandeja,
    resolver_periodo,
)

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
    fin = config["periodo"]["fin_simulacion"]
    assert fin == "auto" or pd.Timestamp(config["periodo"]["fin_carga_historica"]) < pd.Timestamp(fin)


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


# --- Preparación automática (servicio preparar-bandeja de Docker Compose) -------------


def _archivos(directorio: Path) -> dict[str, bytes]:
    return {p.relative_to(directorio).as_posix(): p.read_bytes() for p in sorted(directorio.rglob("*.csv"))}


def _config_auto(incidencias=None) -> dict:
    config = config_bandeja(incidencias)
    config["periodo"]["fin_simulacion"] = "auto"
    return config


def test_fin_simulacion_auto_es_la_ultima_fecha_de_ventas(origen_kaggle):
    _, fin = resolver_periodo(origen_kaggle, _config_auto())
    assert fin == generar_ventas()["date"].max()


def test_fin_auto_da_la_misma_bandeja_que_la_fecha_fija(origen_kaggle, tmp_path):
    generar_bandeja(origen_kaggle, tmp_path / "fija", config_bandeja(INCIDENCIAS))
    generar_bandeja(origen_kaggle, tmp_path / "auto", _config_auto(INCIDENCIAS))
    assert _archivos(tmp_path / "fija") == _archivos(tmp_path / "auto")


def test_fin_auto_cubre_un_periodo_posterior(origen_kaggle, tmp_path):
    """Si se reemplaza train.csv por uno más largo, la bandeja llega hasta su último día."""
    generar_ventas(dias=150).to_csv(origen_kaggle / "train.csv", index=False)
    generar_bandeja(origen_kaggle, tmp_path / "b", _config_auto())
    ultima = sorted((tmp_path / "b" / "ventas").glob("ventas_*.csv"))[-1].name
    assert ultima == f"ventas_{generar_ventas(dias=150)['date'].max().date()}.csv"


def test_fin_simulacion_no_posterior_a_la_carga_historica(origen_kaggle):
    config = config_bandeja()
    config["periodo"]["fin_simulacion"] = config["periodo"]["fin_carga_historica"]
    with pytest.raises(ValueError, match="posterior"):
        resolver_periodo(origen_kaggle, config)


def test_falta_un_csv_de_kaggle(origen_kaggle, tmp_path):
    (origen_kaggle / "oil.csv").unlink()
    with pytest.raises(FileNotFoundError, match="oil.csv"):
        preparar_bandeja(origen_kaggle, tmp_path / "b", config_bandeja())
    assert not (tmp_path / "b").exists()


def test_preparar_solo_regenera_si_cambian_las_entradas(origen_kaggle, tmp_path):
    destino = tmp_path / "b"
    config = config_bandeja(INCIDENCIAS)
    primera = preparar_bandeja(origen_kaggle, destino, config)
    assert primera.generada and (destino / ARCHIVO_HUELLA).exists()
    contenido = _archivos(destino)

    assert not preparar_bandeja(origen_kaggle, destino, config).generada  # nada cambió
    assert _archivos(destino) == contenido

    config["semilla"] = 99  # cambió la configuración
    assert preparar_bandeja(origen_kaggle, destino, config).generada

    generar_ventas(semilla=5).to_csv(origen_kaggle / "train.csv", index=False)  # cambió Kaggle
    assert preparar_bandeja(origen_kaggle, destino, config).generada
    assert preparar_bandeja(origen_kaggle, destino, config, forzar=True).generada


def test_bandeja_borrada_o_interrumpida_se_regenera(origen_kaggle, tmp_path):
    import shutil

    destino = tmp_path / "b"
    config = config_bandeja()
    preparar_bandeja(origen_kaggle, destino, config)
    shutil.rmtree(destino / "ventas")  # p. ej. alguien borró una fuente
    assert preparar_bandeja(origen_kaggle, destino, config).generada

    (destino / ARCHIVO_HUELLA).unlink()  # generación interrumpida: no llegó a escribir la marca
    assert preparar_bandeja(origen_kaggle, destino, config).generada


def test_la_marca_no_es_una_entrega(origen_kaggle, tmp_path):
    destino = tmp_path / "b"
    preparar_bandeja(origen_kaggle, destino, config_bandeja())
    con_marca = leer_bandeja("2017-04-20", destino)
    (destino / ARCHIVO_HUELLA).unlink()
    pd.testing.assert_frame_equal(leer_bandeja("2017-04-20", destino).ventas, con_marca.ventas)


def test_script_sin_cambios_y_sin_kaggle(origen_kaggle, tmp_path):
    ruta_config = tmp_path / "bandeja.yaml"
    ruta_config.write_text(yaml.safe_dump(config_bandeja()), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "preparar_bandeja.py"
    argumentos = [sys.executable, str(script), "--config", str(ruta_config), "--origen", str(origen_kaggle),
                  "--destino", str(tmp_path / "b"), "--incidencias", str(tmp_path / "incidencias.csv")]

    assert "Bandeja generada" in subprocess.run(argumentos, capture_output=True, text=True, check=True).stdout
    assert "sin cambios" in subprocess.run(argumentos, capture_output=True, text=True, check=True).stdout

    (origen_kaggle / "train.csv").unlink()
    fallo = subprocess.run(argumentos, capture_output=True, text=True)
    assert fallo.returncode == 1 and "train.csv" in fallo.stderr
