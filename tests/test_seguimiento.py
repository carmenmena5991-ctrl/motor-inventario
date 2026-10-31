import pandas as pd
import pytest

from inventario.seguimiento import ejecutar_seguimiento, evaluar_pedido, fechas_emision_evaluables

EMISION = pd.Timestamp("2017-04-10")


def _ventas(valores_por_dia):
    fechas = pd.date_range(EMISION + pd.Timedelta(days=1), periods=len(valores_por_dia))
    return pd.DataFrame({"date": fechas, "store_nbr": 1, "family": "A", "sales": valores_por_dia, "onpromotion": 0})


def _pedido(pronostico, pedido):
    return pd.DataFrame(
        {"fecha_emision": [EMISION.date()], "store_nbr": [1], "family": ["A"],
         "pronostico_horizonte": [pronostico], "dias_horizonte": [3], "pedido_sugerido": [pedido]}
    )


def test_fechas_evaluables():
    fechas = fechas_emision_evaluables(pd.Timestamp("2017-04-20"), 7, 3)
    assert fechas == [pd.Timestamp("2017-04-13"), pd.Timestamp("2017-04-12"), pd.Timestamp("2017-04-11")]


def test_costo_real_con_quiebre(config):
    # Vende 10 en el horizonte, se pidieron 7 -> faltan 3 -> 3 * 5 = 15.
    metricas = evaluar_pedido(_pedido(6.5, 7), _ventas([4, 3, 3]), EMISION, 3, config)
    assert metricas["venta_real_total"] == 10
    assert metricas["unidades_quiebre"] == 3
    assert metricas["unidades_merma"] == 0
    assert metricas["costo_real_inventario"] == pytest.approx(15)
    assert metricas["mae"] == pytest.approx(3.5)


def test_costo_real_con_merma(config):
    # Vende 10, se pidieron 14 -> sobran 4 -> 4 * 2 = 8.
    metricas = evaluar_pedido(_pedido(12, 14), _ventas([4, 3, 3]), EMISION, 3, config)
    assert metricas["unidades_merma"] == 4
    assert metricas["costo_real_inventario"] == pytest.approx(8)


def test_serie_sin_ventas_cuenta_como_cero(config):
    pedido = pd.concat([_pedido(6.5, 7), _pedido(2, 2).assign(family="B")])
    metricas = evaluar_pedido(pedido, _ventas([4, 3, 3]), EMISION, 3, config)
    assert metricas["series"] == 2
    assert metricas["unidades_merma"] == 2


def test_horizonte_incompleto_falla(config):
    with pytest.raises(ValueError, match="Aún no hay ventas"):
        evaluar_pedido(_pedido(6.5, 7), _ventas([4, 3]), EMISION, 3, config)


def test_ejecutar_seguimiento(bandeja, ruta_costos, tmp_path):
    predicciones = tmp_path / "predicciones"
    predicciones.mkdir()
    pd.DataFrame(
        {"fecha_emision": ["2017-04-13"] * 2, "store_nbr": [1, 2], "family": ["GROCERY I"] * 2,
         "pronostico_horizonte": [150.0, 300.0], "dias_horizonte": [7, 7], "pedido_sugerido": [150, 300]}
    ).to_csv(predicciones / "pedido_2017-04-13.csv", index=False)

    resultado = ejecutar_seguimiento("2017-04-20", bandeja, predicciones, ruta_costos)
    assert resultado["fecha_emision"].tolist() == ["2017-04-13"]
    assert resultado.loc[0, "costo_real_inventario"] > 0

    archivo = predicciones / "seguimiento_2017-04-20.csv"
    primera = archivo.read_text()
    ejecutar_seguimiento("2017-04-20", bandeja, predicciones, ruta_costos)
    assert archivo.read_text() == primera  # idempotente


def test_sin_pedidos_evaluables(bandeja, ruta_costos, tmp_path):
    predicciones = tmp_path / "predicciones"
    predicciones.mkdir()
    resultado = ejecutar_seguimiento("2017-04-20", bandeja, predicciones, ruta_costos)
    assert resultado.empty
    assert not list(predicciones.iterdir())


def test_sin_ventas_del_dia_no_evalua(bandeja, ruta_costos, tmp_path):
    from inventario.ingesta import DatosNoDisponiblesError

    with pytest.raises(DatosNoDisponiblesError):
        ejecutar_seguimiento("2017-05-10", bandeja, tmp_path, ruta_costos)
