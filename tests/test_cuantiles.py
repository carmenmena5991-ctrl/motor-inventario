"""Pronóstico diario por cuantil: modelo, publicación, calibración y configuración."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from inventario.config import cargar_params
from inventario.entrenar import (
    BaselinePromedioMovil,
    ModeloCuantiles,
    cobertura_cuantiles,
    columnas_modelo,
    nombre_cuantil,
)
from inventario.features import construir_features
from inventario.ingesta import leer_bandeja
from inventario.predecir import (
    generar_pedido,
    guardar_pronostico,
    preparar_features_prediccion,
    pronosticar_con_cuantiles,
    pronosticar_desde_features,
)
from inventario.seguimiento import calibracion_pronostico

CORTE = pd.Timestamp("2017-04-20")


@pytest.fixture(scope="module")
def datos_entrenamiento(tmp_path_factory):
    from conftest import (
        generar_feriados,
        generar_petroleo,
        generar_tiendas,
        generar_ventas,
    )

    features = construir_features(generar_ventas(), generar_petroleo(), generar_feriados(), generar_tiendas(), horizonte=7)
    con_ventas = features.loc[features["sales"].notna()]
    return con_ventas[columnas_modelo()], con_ventas["sales"].to_numpy(dtype=float)


@pytest.fixture(scope="module")
def modelo(datos_entrenamiento):
    X, y = datos_entrenamiento
    return ModeloCuantiles(0.71, (0.1, 0.5, 0.9), {"max_iter": 30}).fit(X, y)


def test_nombre_cuantil():
    assert [nombre_cuantil(q) for q in (0.1, 0.5, 0.71, 0.9)] == ["p10", "p50", "p71", "p90"]


def test_un_modelo_por_cuantil_sin_repetir_el_objetivo(datos_entrenamiento):
    X, y = datos_entrenamiento
    m = ModeloCuantiles(0.9, (0.1, 0.5, 0.9), {"max_iter": 5}).fit(X, y)
    assert list(m.modelos_) == ["p10", "p50", "p90"]


def test_cuantiles_ordenados_no_negativos_y_predict_es_el_objetivo(modelo, datos_entrenamiento):
    X, _ = datos_entrenamiento
    cuantiles = modelo.predict_cuantiles(X)
    assert list(cuantiles.columns) == ["p10", "p50", "p71", "p90"]
    assert (np.diff(cuantiles.to_numpy(), axis=1) >= 0).all()  # nunca se cruzan
    assert (cuantiles.to_numpy() >= 0).all()
    np.testing.assert_array_equal(modelo.predict(X), cuantiles["p71"].to_numpy())


def test_intervalo_del_80_por_ciento_razonable_en_entrenamiento(modelo, datos_entrenamiento):
    X, y = datos_entrenamiento
    cobertura = cobertura_cuantiles(y, modelo.predict_cuantiles(X))
    assert cobertura["cobertura_p10"] < cobertura["cobertura_p50"] < cobertura["cobertura_p90"]
    assert 0.6 < cobertura["cobertura_p10_p90"] <= 1.0
    assert cobertura["ancho_medio_p10_p90"] > 0


def test_cobertura_con_valores_conocidos():
    y = np.array([1.0, 5.0, 9.0, 20.0])
    cuantiles = pd.DataFrame({"p10": [2.0] * 4, "p50": [6.0] * 4, "p90": [10.0] * 4})
    c = cobertura_cuantiles(y, cuantiles)
    assert c["cobertura_p10"] == 0.25 and c["cobertura_p50"] == 0.5 and c["cobertura_p90"] == 0.75
    assert c["cobertura_p10_p90"] == 0.5  # 5 y 9 quedan dentro de [2, 10]
    assert c["ancho_medio_p10_p90"] == 8.0


def test_pronostico_diario_y_pedido_coherentes(modelo, bandeja):
    features = preparar_features_prediccion(leer_bandeja(CORTE, bandeja), 7)
    pronostico = pronosticar_con_cuantiles(modelo, features, CORTE)
    assert {"p10", "p50", "p71", "p90", "pronostico"} <= set(pronostico.columns)
    assert pronostico["date"].min() == CORTE + pd.Timedelta(days=1)
    assert pronostico["date"].nunique() == 7
    pd.testing.assert_series_equal(pronostico["pronostico"], pronostico["p71"], check_names=False)
    # El pedido usa exactamente el cuantil objetivo del pronóstico diario.
    pedido = generar_pedido(pronostico, CORTE)
    suma = pronostico.groupby(["store_nbr", "family"])["p71"].sum().round(6)
    assert (pedido.set_index(["store_nbr", "family"])["pedido_sugerido"] == np.ceil(suma)).all()


def test_modelo_de_un_cuantil_solo_entrega_el_objetivo(bandeja):
    features = preparar_features_prediccion(leer_bandeja(CORTE, bandeja), 7)
    baseline = BaselinePromedioMovil().fit(pd.DataFrame({"media_movil_28": [0.0]}))
    pronostico = pronosticar_con_cuantiles(baseline, features, CORTE, cuantil_objetivo=0.71)
    assert [c for c in pronostico.columns if c.startswith("p") and c[1:].isdigit()] == ["p71"]
    pd.testing.assert_frame_equal(
        pronosticar_desde_features(baseline, features, CORTE),
        pronostico[["date", "store_nbr", "family", "pronostico"]],
    )


def test_guardar_pronostico_idempotente_y_sin_columna_interna(modelo, bandeja, tmp_path):
    features = preparar_features_prediccion(leer_bandeja(CORTE, bandeja), 7)
    pronostico = pronosticar_con_cuantiles(modelo, features, CORTE)
    ruta = guardar_pronostico(pronostico, tmp_path / "pronostico.csv", modelo_version="3")
    contenido = ruta.read_bytes()
    guardar_pronostico(pronostico, tmp_path / "pronostico.csv", modelo_version="3")
    assert ruta.read_bytes() == contenido
    leido = pd.read_csv(ruta)
    assert list(leido.columns) == ["date", "store_nbr", "family", "p10", "p50", "p71", "p90", "modelo_version"]


def test_modelo_registrado_en_mlflow_entrega_cuantiles(bandeja, ruta_costos, mlflow_local, tmp_path):
    from inventario.entrenar import ejecutar_entrenamiento
    from inventario.predecir import cargar_modelo_local
    from inventario.versionado import exportar_modelo

    version = ejecutar_entrenamiento(
        CORTE, bandeja, ruta_costos=ruta_costos, tracking_uri=mlflow_local,
        experimento="pruebas", nombre_modelo="modelo_prueba",
    ).version
    local = cargar_modelo_local(exportar_modelo(version, "modelo_prueba", tmp_path, mlflow_local))
    features = preparar_features_prediccion(leer_bandeja(CORTE, bandeja), 7)
    pronostico = pronosticar_con_cuantiles(local, features, CORTE)
    # Config de prueba: objetivo 0.9 (coincide con el p90 del intervalo).
    assert {"p10", "p50", "p90"} <= set(pronostico.columns)
    # El predict de MLflow (pyfunc) da el mismo cuantil objetivo que el pronóstico diario.
    pyfunc = np.clip(local.predict(features.loc[features["date"] > CORTE, columnas_modelo()]), 0, None)
    np.testing.assert_allclose(pronostico["pronostico"].to_numpy(), np.asarray(pyfunc, dtype=float))


def test_calibracion_del_seguimiento_con_dias_sin_venta():
    pronostico = pd.DataFrame({
        "date": ["2017-04-21", "2017-04-22", "2017-04-21", "2017-04-22"],
        "store_nbr": [1, 1, 2, 2], "family": ["A"] * 4,
        "p10": [1.0] * 4, "p50": [5.0] * 4, "p90": [9.0] * 4, "modelo_version": ["1"] * 4,
    })
    ventas = pd.DataFrame({  # la tienda 2 no registró venta el 22: vendió 0
        "date": pd.to_datetime(["2017-04-21", "2017-04-22", "2017-04-21"]),
        "store_nbr": [1, 1, 2], "family": ["A"] * 3, "sales": [4.0, 12.0, 6.0],
    })
    c = calibracion_pronostico(pronostico, ventas)
    assert c["cobertura_p10"] == 0.25    # solo el 0
    assert c["cobertura_p50"] == 0.5     # 4 y 0
    assert c["cobertura_p90"] == 0.75    # todo menos 12
    assert c["cobertura_p10_p90"] == 0.5  # 4 y 6


def test_params_cuantiles_invalidos(tmp_path):
    params = yaml.safe_load(open(cargar_params.__globals__["RUTA_PARAMS_DEFAULT"], encoding="utf-8"))
    assert cargar_params()["entrenamiento"]["cuantiles_intervalo"] == [0.10, 0.50, 0.90]
    params["entrenamiento"]["cuantiles_intervalo"] = [0.1, 1.5]
    ruta = tmp_path / "params.yaml"
    ruta.write_text(yaml.safe_dump(params), encoding="utf-8")
    with pytest.raises(ValueError, match="cuantiles_intervalo"):
        cargar_params(ruta)
