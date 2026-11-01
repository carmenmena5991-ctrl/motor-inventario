from dataclasses import replace

import pandas as pd
import pytest

from inventario.almacen import cargar_datos, cargar_tabla, guardar_datos, guardar_tabla, recortar_ventas
from inventario.entrenar import BaselinePromedioMovil
from inventario.ingesta import leer_bandeja
from inventario.predecir import (
    dias_historia_necesarios,
    preparar_features_prediccion,
    pronosticar,
    pronosticar_desde_features,
)
from inventario.validacion import validar_datos

CORTE = "2017-04-20"


def test_guardar_y_cargar_conserva_los_datos(bandeja, tmp_path):
    datos = leer_bandeja(CORTE, bandeja)
    cargados = cargar_datos(guardar_datos(datos, tmp_path / "etapa"))
    assert cargados.fecha_corte == datos.fecha_corte
    pd.testing.assert_frame_equal(cargados.ventas, datos.ventas, check_dtype=False)
    pd.testing.assert_frame_equal(cargados.tiendas, datos.tiendas, check_dtype=False)


def test_reescritura_reemplaza_salida_anterior(bandeja, tmp_path):
    datos = leer_bandeja(CORTE, bandeja)
    destino = tmp_path / "etapa"
    guardar_datos(datos, destino)
    guardar_datos(replace(datos, tiendas=None), destino)
    assert not (destino / "tiendas.parquet").exists()
    assert cargar_datos(destino).tiendas is None
    assert not destino.with_name("etapa.tmp").exists()


def test_cargar_sin_etapa_previa(tmp_path):
    with pytest.raises(FileNotFoundError, match="etapa anterior"):
        cargar_datos(tmp_path / "no_existe")
    with pytest.raises(FileNotFoundError, match="etapa anterior"):
        cargar_tabla(tmp_path / "no_existe.parquet")


def test_recortar_ventas(bandeja):
    datos = recortar_ventas(leer_bandeja(CORTE, bandeja), 10)
    assert datos.ventas["date"].min() == pd.Timestamp(CORTE) - pd.Timedelta(days=9)
    assert datos.ventas["date"].max() == pd.Timestamp(CORTE)


def test_guardar_tabla_idempotente(tmp_path):
    df = pd.DataFrame({"a": [1, 2]})
    ruta = guardar_tabla(df, tmp_path / "x" / "t.parquet")
    guardar_tabla(df, ruta)
    pd.testing.assert_frame_equal(cargar_tabla(ruta), df)
    assert [p.name for p in ruta.parent.iterdir()] == ["t.parquet"]


def test_cadena_de_dags_equivale_a_predecir_con_toda_la_historia(bandeja, tmp_path):
    """Simula 01 (ingesta + ventana + validación) -> 02 (features) -> 03 (pronóstico) y
    comprueba que recortar la historia no cambia el resultado."""
    horizonte = 7
    modelo = BaselinePromedioMovil(ventana=28).fit(pd.DataFrame({"media_movil_28": [0.0]}))

    # DAG 01
    datos = leer_bandeja(CORTE, bandeja)
    raw = guardar_datos(recortar_ventas(datos, dias_historia_necesarios(horizonte)), tmp_path / "raw")
    validado = guardar_datos(validar_datos(cargar_datos(raw)), tmp_path / "validado")
    # DAG 02
    ruta_features = guardar_tabla(
        preparar_features_prediccion(cargar_datos(validado), horizonte), tmp_path / "features.parquet"
    )
    # DAG 03
    encadenado = pronosticar_desde_features(modelo, cargar_tabla(ruta_features), pd.Timestamp(CORTE))

    directo = pronosticar(modelo, datos, horizonte)
    pd.testing.assert_frame_equal(encadenado, directo, check_dtype=False)


def test_guarda_metadatos_de_trazabilidad(bandeja, tmp_path):
    ruta = bandeja / "ventas" / "ventas_2017-04-10.csv"
    corrupta = pd.read_csv(ruta)
    corrupta.loc[0, "sales"] = -1
    corrupta.to_csv(ruta, index=False)

    datos = leer_bandeja(CORTE, bandeja)
    cargados = cargar_datos(guardar_datos(datos, tmp_path / "etapa"))
    assert cargados.entregas == datos.entregas
    assert cargados.cuarentena == datos.cuarentena == (
        ("ventas/ventas_2017-04-10.csv", datos.cuarentena[0][1]),
    )
    assert cargados.correcciones == datos.correcciones
    assert cargados.fecha_conocimiento == datos.fecha_conocimiento
    pd.testing.assert_frame_equal(cargados.transacciones, datos.transacciones, check_dtype=False)
