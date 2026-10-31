import pytest

from inventario.evaluacion import costo_total_inventario, evaluar, mae, rmse


def test_metricas_tecnicas():
    assert rmse([1, 2, 3], [1, 2, 3]) == 0
    assert mae([0, 0], [1, 3]) == pytest.approx(2)
    assert rmse([0, 0], [3, 4]) == pytest.approx((12.5) ** 0.5)


def test_costo_quiebre_y_merma():
    # real 10, pedido 7 -> faltan 3 -> 3 * 5 = 15
    assert costo_total_inventario([10], [7], 5.0, 2.0) == pytest.approx(15)
    # real 10, pedido 14 -> sobran 4 -> 4 * 2 = 8
    assert costo_total_inventario([10], [14], 5.0, 2.0) == pytest.approx(8)
    assert costo_total_inventario([10, 10], [7, 14], 5.0, 2.0) == pytest.approx(23)


def test_evaluar_usa_config(config):
    metricas = evaluar([10, 10], [7, 14], config)
    assert set(metricas) == {"rmse", "mae", "costo_total_inventario"}
    assert metricas["costo_total_inventario"] == pytest.approx(23)


def test_formas_distintas():
    with pytest.raises(ValueError):
        mae([1, 2], [1])
