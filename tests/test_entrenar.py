import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

from inventario.entrenar import (
    ALIAS_CAMPEON,
    BaselinePromedioMovil,
    construir_modelo_ml,
    dividir_temporal,
    ejecutar_entrenamiento,
)


def test_baseline_aplica_factor_de_seguridad():
    X = pd.DataFrame({"media_movil_28": [10.0, np.nan, 0.0]})
    modelo = BaselinePromedioMovil(ventana=28, factor_seguridad=0.15).fit(X)
    np.testing.assert_allclose(modelo.predict(X), [11.5, 0.0, 0.0])


def test_modelo_ml_usa_perdida_cuantilica():
    regresor = construir_modelo_ml(0.9).named_steps["regresor"]
    assert regresor.loss == "quantile"
    assert regresor.quantile == 0.9


def test_division_temporal_sin_solape():
    df = pd.DataFrame({"date": pd.date_range("2017-01-01", periods=30), "sales": 1.0})
    entrenamiento, validacion = dividir_temporal(df, 14)
    assert entrenamiento["date"].max() < validacion["date"].min()
    assert validacion["date"].nunique() == 14


def test_division_con_historia_insuficiente():
    df = pd.DataFrame({"date": pd.date_range("2017-01-01", periods=10), "sales": 1.0})
    with pytest.raises(ValueError, match="insuficiente"):
        dividir_temporal(df, 14)


def test_entrenamiento_registra_y_asigna_champions(bandeja, ruta_costos, mlflow_local):
    resultado = ejecutar_entrenamiento(
        "2017-04-20",
        bandeja,
        ruta_costos=ruta_costos,
        tracking_uri=mlflow_local,
        experimento="pruebas",
        nombre_modelo="modelo_prueba",
    )

    assert set(resultado.metricas) == {"baseline_promedio_movil", "hist_gradient_boosting"}
    for metricas in resultado.metricas.values():
        assert {"rmse", "mae", "costo_total_inventario"} <= set(metricas)
    costos = {n: m["costo_total_inventario"] for n, m in resultado.metricas.items()}
    assert resultado.ganador == min(costos, key=costos.get)

    cliente = MlflowClient(tracking_uri=mlflow_local)
    version = cliente.get_model_version_by_alias("modelo_prueba", ALIAS_CAMPEON)
    assert str(version.version) == resultado.version
    assert version.tags["modelo"] == resultado.ganador

    run = cliente.get_run(resultado.run_id)
    assert run.data.tags["fecha_corte"] == "2017-04-20"
    assert run.data.params["cuantil_objetivo"] == "0.9"
