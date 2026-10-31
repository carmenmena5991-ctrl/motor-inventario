import pandas as pd
import pytest

from inventario.entrenar import BaselinePromedioMovil, ejecutar_entrenamiento
from inventario.ingesta import leer_bandeja
from inventario.predecir import (
    cargar_modelo_champions,
    crear_filas_futuras,
    ejecutar_prediccion,
    generar_pedido,
    guardar_pedido,
    pronosticar,
)

CORTE = pd.Timestamp("2017-04-20")


@pytest.fixture
def baseline():
    return BaselinePromedioMovil(ventana=28, factor_seguridad=0.15).fit(pd.DataFrame({"media_movil_28": [0.0]}))


def test_filas_futuras_cubren_el_horizonte():
    ventas = pd.DataFrame({"date": [CORTE] * 2, "store_nbr": [1, 2], "family": ["A", "A"], "sales": [1.0, 2.0], "onpromotion": [0, 0]})
    futuras = crear_filas_futuras(ventas, CORTE, 7)
    assert len(futuras) == 2 * 7
    assert futuras["date"].min() == CORTE + pd.Timedelta(days=1)
    assert futuras["date"].max() == CORTE + pd.Timedelta(days=7)
    assert futuras["sales"].isna().all()


def test_filas_futuras_con_promociones_planificadas():
    ventas = pd.DataFrame({"date": [CORTE], "store_nbr": [1], "family": ["A"], "sales": [1.0], "onpromotion": [0]})
    promos = pd.DataFrame({"date": [CORTE + pd.Timedelta(days=2)], "store_nbr": [1], "family": ["A"], "onpromotion": [4]})
    futuras = crear_filas_futuras(ventas, CORTE, 3, promos)
    assert futuras["onpromotion"].tolist() == [0, 4, 0]


def test_pronostico_solo_para_fechas_posteriores_al_corte(bandeja, baseline):
    datos = leer_bandeja(CORTE, bandeja, dias_calendario_futuro=7)
    pronostico = pronosticar(baseline, datos, horizonte=7)
    assert len(pronostico) == 2 * 2 * 7  # tiendas x familias x días
    assert pronostico["date"].min() > CORTE
    assert (pronostico["pronostico"] > 0).all()


def test_pedido_redondea_hacia_arriba():
    pronostico = pd.DataFrame(
        {
            "date": pd.date_range("2017-04-21", periods=3).tolist() * 1,
            "store_nbr": [1, 1, 1],
            "family": ["A", "A", "A"],
            "pronostico": [1.2, 1.3, 1.1],
        }
    )
    pedido = generar_pedido(pronostico, CORTE)
    assert pedido.loc[0, "pedido_sugerido"] == 4
    assert pedido.loc[0, "dias_horizonte"] == 3


def test_guardar_es_idempotente(tmp_path):
    pedido = pd.DataFrame({"store_nbr": [1], "family": ["A"], "pedido_sugerido": [4]})
    ruta1 = guardar_pedido(pedido, CORTE, tmp_path)
    ruta2 = guardar_pedido(pedido, CORTE, tmp_path)
    assert ruta1 == ruta2
    assert len(list(tmp_path.iterdir())) == 1
    pd.testing.assert_frame_equal(pd.read_csv(ruta2), pedido)


def test_flujo_completo_con_modelo_champions(bandeja, ruta_costos, mlflow_local, tmp_path):
    ejecutar_entrenamiento(
        CORTE, bandeja, ruta_costos=ruta_costos, tracking_uri=mlflow_local,
        experimento="pruebas", nombre_modelo="modelo_prueba",
    )
    modelo = cargar_modelo_champions("modelo_prueba", tracking_uri=mlflow_local)

    ruta = ejecutar_prediccion(
        CORTE, bandeja, ruta_costos=ruta_costos, directorio_salida=tmp_path / "pred", modelo=modelo
    )
    pedido = pd.read_csv(ruta)
    assert ruta.name == "pedido_2017-04-20.csv"
    assert len(pedido) == 4
    assert (pedido["pedido_sugerido"] >= 0).all()
    assert (pedido["dias_horizonte"] == 7).all()
