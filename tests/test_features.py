import numpy as np
import pandas as pd
import pytest

from inventario.features import construir_features, nombres_features
from conftest import generar_feriados, generar_petroleo, generar_tiendas, generar_ventas


@pytest.fixture
def features():
    return construir_features(generar_ventas(), generar_petroleo(), generar_feriados(), generar_tiendas())


def _fila(df, fecha, tienda=1, familia="GROCERY I"):
    return df.loc[(df["date"] == pd.Timestamp(fecha)) & (df["store_nbr"] == tienda) & (df["family"] == familia)].iloc[0]


def test_genera_todas_las_columnas(features):
    assert set(nombres_features()).issubset(features.columns)
    assert len(features) == len(generar_ventas())


def test_lags(features):
    ventas = generar_ventas()
    for k in (7, 14, 28):
        esperado = _fila(ventas, pd.Timestamp("2017-03-01") - pd.Timedelta(days=k))["sales"]
        assert _fila(features, "2017-03-01")[f"lag_{k}"] == pytest.approx(esperado)


def test_media_movil_desplazada_por_horizonte(features):
    ventas = generar_ventas()
    serie = ventas.loc[(ventas["store_nbr"] == 1) & (ventas["family"] == "GROCERY I")].set_index("date")["sales"]
    # Para 2017-03-01 con horizonte 7: media de los 7 días que terminan el 2017-02-22.
    esperado = serie.loc["2017-02-16":"2017-02-22"].mean()
    assert _fila(features, "2017-03-01")["media_movil_7"] == pytest.approx(esperado)


def test_sin_leakage_ventas_recientes_no_afectan(features):
    """Cambiar las ventas de los últimos 6 días no debe cambiar ninguna feature del último día."""
    ventas = generar_ventas()
    ultimo = ventas["date"].max()
    alteradas = ventas.copy()
    recientes = alteradas["date"] > ultimo - pd.Timedelta(days=7)
    alteradas.loc[recientes & (alteradas["date"] < ultimo), "sales"] = 10_000

    nuevas = construir_features(alteradas, generar_petroleo(), generar_feriados(), generar_tiendas())
    columnas = nombres_features()
    pd.testing.assert_frame_equal(
        features.loc[features["date"] == ultimo, columnas].reset_index(drop=True),
        nuevas.loc[nuevas["date"] == ultimo, columnas].reset_index(drop=True),
    )


def test_lags_menores_que_horizonte_fallan():
    with pytest.raises(ValueError, match="leakage"):
        construir_features(generar_ventas(), generar_petroleo(), generar_feriados(), lags=(1, 7), horizonte=7)


def test_feriados(features):
    assert _fila(features, "2017-02-27")["es_feriado_nacional"] == 1
    assert _fila(features, "2017-02-28")["es_feriado"] == 0
    # Feriado local de Quito: solo la tienda 1 está en Quito.
    assert _fila(features, "2017-03-10", tienda=1)["es_feriado_local"] == 1
    assert _fila(features, "2017-03-10", tienda=2)["es_feriado_local"] == 0
    # "Work Day" no es feriado.
    assert _fila(features, "2017-04-29")["es_feriado"] == 0


def test_promocion(features):
    assert ((features["onpromotion"] > 0).astype(int) == features["en_promocion"]).all()


def test_petroleo_rellena_fines_de_semana_sin_usar_futuro(features):
    # 2017-03-06 (lunes) - 7 días = 2017-02-27 (lunes) -> precio de ese día.
    petroleo = generar_petroleo().set_index("date")["dcoilwtico"]
    assert _fila(features, "2017-03-06")["petroleo"] == pytest.approx(petroleo["2017-02-27"])
    # 2017-03-05 (domingo) - 7 = domingo sin cotización -> último precio previo (viernes 2017-02-24).
    assert _fila(features, "2017-03-05")["petroleo"] == pytest.approx(petroleo["2017-02-24"])


def test_no_modifica_entradas_y_es_idempotente():
    ventas = generar_ventas()
    copia = ventas.copy()
    a = construir_features(ventas, generar_petroleo(), generar_feriados())
    b = construir_features(ventas, generar_petroleo(), generar_feriados())
    pd.testing.assert_frame_equal(ventas, copia)
    pd.testing.assert_frame_equal(a, b)
