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
    assert resultado.promovido and resultado.motivo == "no hay modelo vigente"
    assert version.tags["decision"] == "promovido"


# --- Criterio de reemplazo del modelo vigente -------------------------------------------

from inventario.entrenar import asegurar_modelo_vigente, debe_entrenar, decidir_reemplazo, version_vigente  # noqa: E402


@pytest.mark.parametrize(
    "candidato, vigente, hay_vigente, esperado",
    [
        (100.0, None, False, True),   # no hay vigente
        (100.0, None, True, True),    # vigente no evaluable
        (97.0, 100.0, True, True),    # mejora 3 % >= 2 %
        (98.0, 100.0, True, True),    # mejora justo en el umbral
        (99.0, 100.0, True, False),   # mejora 1 % < 2 %
        (105.0, 100.0, True, False),  # empeora
    ],
)
def test_decidir_reemplazo(candidato, vigente, hay_vigente, esperado):
    promover, motivo = decidir_reemplazo(candidato, vigente, hay_vigente, mejora_minima=0.02)
    assert promover is esperado
    assert motivo


def test_debe_entrenar():
    domingo, lunes = "2017-07-09", "2017-07-10"
    assert debe_entrenar(lunes, 6, hay_vigente=False)[0]       # arranque: siempre
    assert debe_entrenar(domingo, 6, hay_vigente=True)[0]      # día de reentrenamiento
    assert not debe_entrenar(lunes, 6, hay_vigente=True)[0]


def _entrenar(bandeja, ruta_costos, mlflow_local, **kwargs):
    return ejecutar_entrenamiento(
        "2017-04-20", bandeja, ruta_costos=ruta_costos, tracking_uri=mlflow_local,
        experimento="pruebas", nombre_modelo="modelo_prueba", **kwargs,
    )


def test_candidato_que_no_mejora_no_reemplaza_al_vigente(bandeja, ruta_costos, mlflow_local):
    primero = _entrenar(bandeja, ruta_costos, mlflow_local)
    segundo = _entrenar(bandeja, ruta_costos, mlflow_local)  # mismos datos: mismo costo

    assert not segundo.promovido
    assert segundo.version_vigente == primero.version
    assert segundo.costo_vigente == pytest.approx(segundo.metricas[segundo.ganador]["costo_total_inventario"])
    cliente = MlflowClient(tracking_uri=mlflow_local)
    assert str(cliente.get_model_version_by_alias("modelo_prueba", ALIAS_CAMPEON).version) == primero.version
    assert cliente.get_model_version("modelo_prueba", segundo.version).tags["decision"] == "no_promovido"


def test_candidato_mejor_reemplaza_al_vigente(bandeja, ruta_costos, mlflow_local):
    import mlflow

    from inventario.entrenar import TIPOS_CONFIABLES_SKOPS

    # Vigente deliberadamente malo: pide 6 veces el promedio móvil (merma enorme).
    mlflow.set_experiment("pruebas")
    with mlflow.start_run():
        info = mlflow.sklearn.log_model(
            BaselinePromedioMovil(factor_seguridad=5.0).fit(pd.DataFrame({"media_movil_28": [0.0]})),
            name="modelo", skops_trusted_types=TIPOS_CONFIABLES_SKOPS,
        )
    malo = mlflow.register_model(info.model_uri, "modelo_prueba")
    cliente = MlflowClient(tracking_uri=mlflow_local)
    cliente.set_model_version_tag("modelo_prueba", malo.version, "decision", "promovido")
    cliente.set_model_version_tag("modelo_prueba", malo.version, "fecha_corte", "2017-04-01")
    cliente.set_registered_model_alias("modelo_prueba", ALIAS_CAMPEON, malo.version)

    bueno = _entrenar(bandeja, ruta_costos, mlflow_local)

    assert bueno.version_vigente == str(malo.version)
    assert bueno.metricas[bueno.ganador]["costo_total_inventario"] < 0.98 * bueno.costo_vigente
    assert bueno.promovido
    assert version_vigente("modelo_prueba", tracking_uri=mlflow_local) == bueno.version


def test_asegurar_modelo_vigente(bandeja, ruta_costos, mlflow_local):
    assert version_vigente("modelo_prueba", tracking_uri=mlflow_local) is None
    version, entrenado = asegurar_modelo_vigente(
        "2017-04-20", bandeja, nombre_modelo="modelo_prueba", tracking_uri=mlflow_local,
        ruta_costos=ruta_costos, experimento="pruebas",
    )
    assert entrenado and version_vigente("modelo_prueba", tracking_uri=mlflow_local) == version

    otra, entrenado = asegurar_modelo_vigente(
        "2017-04-20", bandeja, nombre_modelo="modelo_prueba", tracking_uri=mlflow_local,
        ruta_costos=ruta_costos, experimento="pruebas",
    )
    assert (otra, entrenado) == (version, False)


def test_modelo_vigente_por_fecha_y_alias_que_no_retrocede(bandeja, ruta_costos, mlflow_local):
    from inventario.entrenar import version_vigente_a_fecha

    kwargs = dict(nombre_modelo="modelo_prueba", tracking_uri=mlflow_local, ruta_costos=ruta_costos, experimento="pruebas")
    abril, _ = asegurar_modelo_vigente("2017-04-20", bandeja, **kwargs)
    # Backfill de una fecha anterior: el modelo de abril no existía aún -> se entrena otro.
    marzo, entrenado = asegurar_modelo_vigente("2017-03-20", bandeja, **kwargs)
    assert entrenado and marzo != abril

    assert version_vigente_a_fecha("2017-03-10", "modelo_prueba", mlflow_local) is None
    assert version_vigente_a_fecha("2017-03-25", "modelo_prueba", mlflow_local) == marzo
    assert version_vigente_a_fecha("2017-04-25", "modelo_prueba", mlflow_local) == abril
    # @Champions sigue en el modelo con datos más recientes.
    assert version_vigente("modelo_prueba", tracking_uri=mlflow_local) == abril


def test_entrenamiento_exige_datos_del_dia(bandeja, ruta_costos, mlflow_local):
    from inventario.ingesta import DatosNoDisponiblesError

    with pytest.raises(DatosNoDisponiblesError):
        ejecutar_entrenamiento("2017-05-10", bandeja, ruta_costos=ruta_costos, tracking_uri=mlflow_local,
                               experimento="pruebas", nombre_modelo="modelo_prueba", exigir_datos_del_dia=True)
