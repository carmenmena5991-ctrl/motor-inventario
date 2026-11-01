"""Huellas, manifiesto, exportación del modelo y versionado con DVC."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from inventario.versionado import (
    exportar_modelo,
    guardar_manifiesto,
    info_codigo,
    md5_archivo,
    md5_directorio,
    remoto_accesible,
    ruta_modelo_local,
    versionar_con_dvc,
)


def _git(base: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(base), "-c", "user.email=p@p", "-c", "user.name=p", *args],
        check=True, capture_output=True,
    )


def _dvc(base: Path, *args: str) -> None:
    subprocess.run([sys.executable, "-m", "dvc", *args], cwd=base, check=True, capture_output=True)


# --- huellas -----------------------------------------------------------------------


def test_md5_directorio_no_depende_del_orden_de_creacion(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for directorio, orden in ((a, ["x.txt", "sub/y.txt"]), (b, ["sub/y.txt", "x.txt"])):
        for nombre in orden:
            (directorio / nombre).parent.mkdir(parents=True, exist_ok=True)
            (directorio / nombre).write_text(nombre)
    assert md5_directorio(a) == md5_directorio(b)


def test_md5_directorio_cambia_con_el_contenido_y_con_el_nombre(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    (d / "x.txt").write_text("1")
    original = md5_directorio(d)
    (d / "x.txt").write_text("2")
    assert md5_directorio(d) != original
    (d / "x.txt").write_text("1")
    (d / "x.txt").rename(d / "z.txt")
    assert md5_directorio(d) != original


def test_manifiesto_estable_byte_a_byte(tmp_path):
    m = {"b": 1, "a": {"y": 2, "x": "ñ"}}
    r1 = guardar_manifiesto(m, tmp_path / "m.json")
    h1 = md5_archivo(r1)
    guardar_manifiesto(dict(reversed(list(m.items()))), tmp_path / "m.json")
    assert md5_archivo(r1) == h1


# --- código ------------------------------------------------------------------------


def test_info_codigo_sin_git(tmp_path):
    assert info_codigo(tmp_path) == {"commit": None, "modificado": None}


def test_info_codigo_detecta_cambios_sin_confirmar(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "m.py").write_text("x = 1\n")
    (tmp_path / "otro.txt").write_text("no es código del producto\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "inicial")

    info = info_codigo(tmp_path)
    assert len(info["commit"]) == 40 and info["modificado"] is False
    (tmp_path / "otro.txt").write_text("cambio fuera del código\n")
    assert info_codigo(tmp_path)["modificado"] is False
    (tmp_path / "src" / "m.py").write_text("x = 2\n")
    assert info_codigo(tmp_path)["modificado"] is True


# --- exportación del modelo ----------------------------------------------------------


@pytest.fixture
def modelo_registrado(bandeja, ruta_costos, mlflow_local):
    from inventario.entrenar import ejecutar_entrenamiento

    resultado = ejecutar_entrenamiento(
        "2017-04-20", bandeja, ruta_costos=ruta_costos, tracking_uri=mlflow_local,
        experimento="pruebas", nombre_modelo="modelo_prueba",
    )
    return resultado.version


def test_exportar_modelo_es_idempotente(modelo_registrado, mlflow_local, tmp_path):
    base = tmp_path / "proyecto"
    ruta = exportar_modelo(modelo_registrado, "modelo_prueba", base, mlflow_local)
    assert ruta == ruta_modelo_local("modelo_prueba", modelo_registrado, base)
    assert (ruta / "MLmodel").exists()
    huella, mtime = md5_directorio(ruta), (ruta / "MLmodel").stat().st_mtime_ns

    assert exportar_modelo(modelo_registrado, "modelo_prueba", base, mlflow_local) == ruta
    assert md5_directorio(ruta) == huella
    assert (ruta / "MLmodel").stat().st_mtime_ns == mtime  # no se volvió a descargar
    assert not ruta.with_name(ruta.name + ".tmp").exists()


def test_modelo_exportado_predice_sin_mlflow(modelo_registrado, mlflow_local, bandeja, tmp_path):
    import pandas as pd

    from inventario.ingesta import leer_bandeja
    from inventario.predecir import cargar_modelo_champions, cargar_modelo_local, pronosticar

    ruta = exportar_modelo(modelo_registrado, "modelo_prueba", tmp_path / "p", mlflow_local)
    datos = leer_bandeja("2017-04-20", bandeja)
    local = pronosticar(cargar_modelo_local(ruta), datos, 7)
    registro = pronosticar(cargar_modelo_champions("modelo_prueba", tracking_uri=mlflow_local), datos, 7)
    pd.testing.assert_frame_equal(local, registro)


# --- DVC -----------------------------------------------------------------------------

pytest.importorskip("dvc")


@pytest.fixture
def repo_dvc(tmp_path) -> Path:
    base = tmp_path / "repo"
    (base / "data" / "predictions").mkdir(parents=True)
    (base / "data" / "predictions" / "pedido_2017-04-20.csv").write_text("store_nbr,pedido\n1,4\n")
    _dvc(base, "init", "--no-scm", "-q")
    return base


def test_versionar_guarda_y_sube_al_remoto(repo_dvc, tmp_path):
    remoto = tmp_path / "remoto"
    remoto.mkdir()
    _dvc(repo_dvc, "remote", "add", "-d", "r", str(remoto))

    resultado = versionar_con_dvc(repo_dvc, rutas=("data/predictions", "models"))
    assert resultado.rutas == ("data/predictions",)  # models no existe: se omite
    assert resultado.subido
    puntero = repo_dvc / "data" / "predictions.dvc"
    assert puntero.exists()
    assert any(p.is_file() for p in remoto.rglob("*"))

    contenido = puntero.read_text()
    assert versionar_con_dvc(repo_dvc, rutas=("data/predictions",)).subido
    assert puntero.read_text() == contenido  # sin cambios en los datos, mismo puntero


def test_remoto_no_montado_no_falla_ni_crea_el_directorio(repo_dvc, tmp_path):
    ausente = tmp_path / "no_montado" / "remoto"
    _dvc(repo_dvc, "remote", "add", "-d", "r", str(ausente))

    accesible, detalle = remoto_accesible(repo_dvc)
    assert not accesible and "no existe" in detalle
    resultado = versionar_con_dvc(repo_dvc, rutas=("data/predictions",))
    assert not resultado.subido
    assert (repo_dvc / "data" / "predictions.dvc").exists()  # quedó en la caché local
    assert not ausente.exists()


def test_sin_remoto_por_defecto(repo_dvc):
    assert remoto_accesible(repo_dvc) == (False, "sin remoto por defecto")


def test_versionar_sin_rutas_existentes(repo_dvc):
    resultado = versionar_con_dvc(repo_dvc, rutas=("models",))
    assert resultado.rutas == () and not resultado.subido
