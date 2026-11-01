from datetime import datetime, timezone

import pandas as pd
import pytest

from inventario.ingesta import leer_bandeja, normalizar_fecha


def test_recorta_ventas_y_petroleo_a_la_fecha_de_corte(bandeja):
    datos = leer_bandeja("2017-03-15", bandeja)
    corte = pd.Timestamp("2017-03-15")
    assert datos.ventas["date"].max() == corte
    assert datos.petroleo["date"].max() <= corte
    assert datos.feriados["date"].max() <= corte
    assert datos.fecha_corte == corte


def test_calendario_futuro_solo_afecta_feriados(bandeja):
    datos = leer_bandeja("2017-04-10", bandeja, dias_calendario_futuro=7)
    assert pd.Timestamp("2017-04-14") in set(datos.feriados["date"])
    assert datos.ventas["date"].max() == pd.Timestamp("2017-04-10")


def test_es_idempotente(bandeja):
    a = leer_bandeja("2017-03-15", bandeja)
    b = leer_bandeja("2017-03-15", bandeja)
    pd.testing.assert_frame_equal(a.ventas, b.ventas)
    pd.testing.assert_frame_equal(a.petroleo, b.petroleo)


def test_particion_reenviada_prevalece_la_ultima(bandeja):
    ventas = pd.read_csv(bandeja / "train.csv")
    correccion = ventas.iloc[[0]].copy()
    correccion["sales"] = 999.0
    correccion.to_csv(bandeja / "train_zz_correccion.csv", index=False)

    datos = leer_bandeja("2017-03-15", bandeja)
    fila = datos.ventas.loc[
        (datos.ventas["date"] == pd.Timestamp(ventas.loc[0, "date"]))
        & (datos.ventas["store_nbr"] == ventas.loc[0, "store_nbr"])
        & (datos.ventas["family"] == ventas.loc[0, "family"])
    ]
    assert len(fila) == 1
    assert fila["sales"].iloc[0] == 999.0


def test_bandeja_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        leer_bandeja("2017-03-15", tmp_path / "no_existe")


def test_fuente_obligatoria_faltante(bandeja):
    (bandeja / "oil.csv").unlink()
    with pytest.raises(FileNotFoundError, match="oil"):
        leer_bandeja("2017-03-15", bandeja)


def test_corte_anterior_a_toda_la_historia(bandeja):
    with pytest.raises(ValueError, match="no tiene ventas"):
        leer_bandeja("2016-01-01", bandeja)


def test_normaliza_fecha_de_airflow_con_zona_horaria():
    assert normalizar_fecha(datetime(2017, 3, 15, 0, 0, tzinfo=timezone.utc)) == pd.Timestamp("2017-03-15")
    assert normalizar_fecha("2017-03-15T10:30:00") == pd.Timestamp("2017-03-15")


def test_exigir_ventas_del_dia(bandeja):
    from inventario.ingesta import DatosNoDisponiblesError, exigir_ventas_del_dia

    datos = leer_bandeja("2017-03-15", bandeja)
    assert exigir_ventas_del_dia(datos) is datos

    # La bandeja sintética termina el 2017-04-30: pedir una fecha posterior no tiene datos del día.
    futuro = leer_bandeja("2017-05-10", bandeja)
    with pytest.raises(DatosNoDisponiblesError, match="ventas hasta 2017-04-30"):
        exigir_ventas_del_dia(futuro)
