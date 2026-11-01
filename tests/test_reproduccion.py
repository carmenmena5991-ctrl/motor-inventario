"""Un pedido publicado se reproduce idéntico desde lo versionado, y se detecta cuando no."""

from __future__ import annotations

import pandas as pd
import pytest

from inventario.almacen import guardar_datos, guardar_tabla
from inventario.entrenar import ejecutar_entrenamiento
from inventario.ingesta import leer_bandeja
from inventario.orquestacion import rutas
from inventario.predecir import (
    cargar_modelo_local,
    generar_pedido,
    guardar_pedido,
    preparar_features_prediccion,
    pronosticar_desde_features,
)
from inventario.reproduccion import reproducir_pedido
from inventario.versionado import exportar_modelo, guardar_manifiesto, manifiesto_pedido

DS = "2017-04-20"


@pytest.fixture
def publicado(bandeja, ruta_costos, mlflow_local, tmp_path):
    """Lo que hacen los DAGs 01-03 para DS, sobre una base temporal."""
    base = tmp_path / "proyecto"
    r = rutas(DS, base=base)
    datos = leer_bandeja(DS, bandeja)
    guardar_datos(datos, r.raw)
    guardar_tabla(preparar_features_prediccion(datos, 7), r.features)

    version = ejecutar_entrenamiento(
        DS, bandeja, ruta_costos=ruta_costos, tracking_uri=mlflow_local,
        experimento="pruebas", nombre_modelo="modelo_prueba",
    ).version
    ruta_modelo = exportar_modelo(version, "modelo_prueba", base, mlflow_local)

    corte = pd.Timestamp(DS)
    pronostico = pronosticar_desde_features(cargar_modelo_local(ruta_modelo), pd.read_parquet(r.features), corte)
    guardar_pedido(generar_pedido(pronostico, corte, modelo_version=version), corte, ruta=r.pedido)
    guardar_manifiesto(manifiesto_pedido(r, "modelo_prueba", version, ruta_modelo), r.manifiesto)
    return r


def test_pedido_publicado_se_reproduce_identico(publicado):
    resultado = reproducir_pedido(DS, publicado.base)
    assert resultado.verificaciones == {"features": True, "datos_crudos": True, "modelo": True}
    assert resultado.identico
    assert resultado.filas == 4


def test_manifiesto_es_idempotente(publicado):
    antes = publicado.manifiesto.read_bytes()
    m = manifiesto_pedido(publicado, "modelo_prueba", reproducir_pedido(DS, publicado.base).modelo_version,
                          publicado.base / "models" / "modelo_prueba" / "v1")
    guardar_manifiesto(m, publicado.manifiesto)
    assert publicado.manifiesto.read_bytes() == antes


def test_pedido_alterado_no_es_identico(publicado):
    pedido = pd.read_csv(publicado.pedido)
    pedido.loc[0, "pedido_sugerido"] += 1
    pedido.to_csv(publicado.pedido, index=False)

    resultado = reproducir_pedido(DS, publicado.base)
    assert resultado.entradas_coinciden
    assert not resultado.identico


def test_features_distintas_se_detectan_y_no_se_regenera(publicado):
    features = pd.read_parquet(publicado.features)
    features.loc[features.index[0], "onpromotion"] = 99
    features.to_parquet(publicado.features, index=False)

    resultado = reproducir_pedido(DS, publicado.base)
    assert resultado.verificaciones["features"] is False
    assert resultado.md5_reproducido == ""
    assert not resultado.identico


def test_sin_manifiesto(tmp_path):
    with pytest.raises(FileNotFoundError):
        reproducir_pedido(DS, tmp_path)
