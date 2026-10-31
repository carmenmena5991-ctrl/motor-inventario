from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from inventario.ingesta import leer_bandeja
from inventario.validacion import (
    DatosInvalidosError,
    validar_datos,
    validar_petroleo,
    validar_tiendas,
    validar_ventas,
)
from conftest import generar_petroleo, generar_tiendas, generar_ventas


def test_datos_validos_pasan_y_no_se_modifican(bandeja):
    datos = leer_bandeja("2017-03-15", bandeja)
    copia = datos.ventas.copy()
    assert validar_datos(datos) is datos
    pd.testing.assert_frame_equal(datos.ventas, copia)


def test_es_idempotente():
    ventas = generar_ventas()
    pd.testing.assert_frame_equal(validar_ventas(validar_ventas(ventas)), ventas)


def test_columna_requerida_faltante():
    with pytest.raises(DatosInvalidosError, match="sales"):
        validar_ventas(generar_ventas().drop(columns="sales"))


def test_ventas_negativas():
    ventas = generar_ventas()
    ventas.loc[5, "sales"] = -1
    with pytest.raises(DatosInvalidosError, match="negativos"):
        validar_ventas(ventas)


def test_ventas_no_numericas():
    ventas = generar_ventas()
    ventas["sales"] = ventas["sales"].astype(object)
    ventas.loc[3, "sales"] = "abc"
    with pytest.raises(DatosInvalidosError, match="no numéricos"):
        validar_ventas(ventas)


def test_ventas_duplicadas():
    ventas = generar_ventas()
    with pytest.raises(DatosInvalidosError, match="duplicadas"):
        validar_ventas(pd.concat([ventas, ventas.iloc[[0]]]))


def test_ventas_vacias():
    with pytest.raises(DatosInvalidosError, match="vacío"):
        validar_ventas(generar_ventas().iloc[0:0])


def test_fecha_no_parseable():
    ventas = generar_ventas()
    ventas.loc[0, "date"] = pd.NaT
    with pytest.raises(DatosInvalidosError, match="date"):
        validar_ventas(ventas)


def test_petroleo_negativo():
    petroleo = generar_petroleo()
    petroleo.loc[0, "dcoilwtico"] = -3.5
    with pytest.raises(DatosInvalidosError, match="negativos"):
        validar_petroleo(petroleo)


def test_petroleo_con_huecos_es_valido():
    petroleo = generar_petroleo()
    assert petroleo["dcoilwtico"].isna().any()
    validar_petroleo(petroleo)


def test_tiendas_duplicadas():
    tiendas = generar_tiendas()
    with pytest.raises(DatosInvalidosError, match="duplicado"):
        validar_tiendas(pd.concat([tiendas, tiendas]))


def test_detecta_leakage(bandeja):
    datos = leer_bandeja("2017-03-15", bandeja)
    futuras = datos.ventas.tail(1).assign(date=pd.Timestamp("2017-03-20"))
    con_futuro = replace(datos, ventas=pd.concat([datos.ventas, futuras], ignore_index=True))
    with pytest.raises(DatosInvalidosError, match="posteriores"):
        validar_datos(con_futuro)
