from datetime import datetime, timezone

import pandas as pd
import pytest

from inventario.ingesta import (
    DatosNoDisponiblesError,
    exigir_ventas_del_dia,
    leer_bandeja,
    listar_entregas,
    normalizar_fecha,
)


def _fila(ventas, fecha, tienda=1, familia="GROCERY I"):
    return ventas.loc[
        (ventas["date"] == pd.Timestamp(fecha)) & (ventas["store_nbr"] == tienda) & (ventas["family"] == familia)
    ]


def _entregar(bandeja, fuente, fecha, df):
    """Simula que el sistema de origen deja un archivo nuevo en la bandeja."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df.to_csv(bandeja / fuente / f"{fuente}_{fecha}.csv", index=False)


def test_estructura_de_la_bandeja(bandeja):
    entregas = listar_entregas(bandeja, "ventas", pd.Timestamp("2017-02-03"))
    assert [e[1].name for e in entregas] == [
        "ventas_2017-01-31.csv", "ventas_2017-02-01.csv", "ventas_2017-02-02.csv", "ventas_2017-02-03.csv",
    ]


def test_solo_usa_lo_entregado_hasta_la_fecha(bandeja):
    datos = leer_bandeja("2017-03-15", bandeja)
    corte = pd.Timestamp("2017-03-15")
    assert datos.ventas["date"].max() == corte
    # Transacciones llegan con 1 día de retraso: las del 15 aún no se conocen.
    assert datos.transacciones["date"].max() == corte - pd.Timedelta(days=1)
    # Petróleo: el precio del martes 14 se entrega el miércoles 15.
    assert datos.petroleo["date"].max() == pd.Timestamp("2017-03-14")
    # Y lo del sábado 11 y domingo 12 llega recién el lunes 13 (solo días hábiles).
    assert leer_bandeja("2017-03-12", bandeja).petroleo["date"].max() == pd.Timestamp("2017-03-09")
    assert all(e.split("_")[-1] <= "2017-03-15.csv" for e in datos.entregas)
    assert datos.fecha_corte == datos.fecha_conocimiento == corte


def test_feriados_entregados_por_adelantado(bandeja):
    # El calendario de abril llega el 1 de marzo: el 14-abr ya se conoce el 10-mar.
    assert pd.Timestamp("2017-04-14") in set(leer_bandeja("2017-03-10", bandeja).feriados["date"])
    # El de mayo llega el 1 de abril: el 2-may aún no se conoce el 31-mar.
    assert pd.Timestamp("2017-05-02") not in set(leer_bandeja("2017-03-31", bandeja).feriados["date"])


def test_es_idempotente_y_no_modifica_la_bandeja(bandeja):
    antes = sorted((p.name, p.stat().st_mtime_ns) for p in bandeja.rglob("*.csv"))
    a = leer_bandeja("2017-03-15", bandeja)
    b = leer_bandeja("2017-03-15", bandeja)
    pd.testing.assert_frame_equal(a.ventas, b.ventas)
    pd.testing.assert_frame_equal(a.petroleo, b.petroleo)
    assert sorted((p.name, p.stat().st_mtime_ns) for p in bandeja.rglob("*.csv")) == antes


def test_correccion_solo_se_ve_desde_su_entrega(bandeja):
    original = _fila(leer_bandeja("2017-03-10", bandeja).ventas, "2017-03-10")
    _entregar(bandeja, "ventas", "2017-03-12", original.assign(sales=999.0))

    # Al 11-mar todavía no había llegado la corrección.
    assert _fila(leer_bandeja("2017-03-11", bandeja).ventas, "2017-03-10")["sales"].iloc[0] == original["sales"].iloc[0]
    # Desde el 12-mar prevalece la entrega más reciente.
    datos = leer_bandeja("2017-03-12", bandeja)
    assert _fila(datos.ventas, "2017-03-10")["sales"].iloc[0] == 999.0
    assert datos.correcciones["ventas"] == 1


def test_reproceso_con_fecha_de_conocimiento_posterior(bandeja):
    original = _fila(leer_bandeja("2017-03-10", bandeja).ventas, "2017-03-10")
    _entregar(bandeja, "ventas", "2017-03-20", original.assign(sales=999.0))

    reproceso = leer_bandeja("2017-03-10", bandeja, fecha_conocimiento="2017-03-20")
    assert _fila(reproceso.ventas, "2017-03-10")["sales"].iloc[0] == 999.0
    # Incorpora la corrección, pero no datos posteriores al corte.
    assert reproceso.ventas["date"].max() == pd.Timestamp("2017-03-10")

    with pytest.raises(ValueError):
        leer_bandeja("2017-03-10", bandeja, fecha_conocimiento="2017-03-01")


def test_entrega_tardia(bandeja):
    del_dia = leer_bandeja("2017-03-10", bandeja).ventas.query("date == '2017-03-10'")
    (bandeja / "ventas" / "ventas_2017-03-10.csv").unlink()
    _entregar(bandeja, "ventas", "2017-03-12", pd.concat([
        del_dia, pd.read_csv(bandeja / "ventas" / "ventas_2017-03-12.csv")
    ]))

    with pytest.raises(DatosNoDisponiblesError, match="ventas hasta 2017-03-09"):
        exigir_ventas_del_dia(leer_bandeja("2017-03-10", bandeja))
    assert len(leer_bandeja("2017-03-12", bandeja).ventas.query("date == '2017-03-10'")) == len(del_dia)


def test_entrega_invalida_queda_en_cuarentena(bandeja):
    archivo = bandeja / "ventas" / "ventas_2017-03-10.csv"
    corrupta = pd.read_csv(archivo)
    corrupta.loc[0, "sales"] = -5
    corrupta.to_csv(archivo, index=False)

    datos = leer_bandeja("2017-03-10", bandeja)
    assert datos.cuarentena[0][0] == "ventas/ventas_2017-03-10.csv"
    assert "negativos" in datos.cuarentena[0][1]
    assert (datos.ventas["sales"] >= 0).all()
    # Sin las ventas del día, el run se omite; los días siguientes siguen funcionando.
    with pytest.raises(DatosNoDisponiblesError):
        exigir_ventas_del_dia(datos)
    exigir_ventas_del_dia(leer_bandeja("2017-03-11", bandeja))


def test_fuentes_secundarias_ausentes_no_fallan(bandeja):
    for fuente in ("petroleo", "transacciones", "tiendas"):
        for archivo in (bandeja / fuente).iterdir():
            archivo.unlink()
    datos = leer_bandeja("2017-03-15", bandeja)
    assert datos.petroleo.empty and datos.transacciones.empty
    assert datos.tiendas is None
    assert not datos.ventas.empty


def test_sin_ventas_entregadas(bandeja):
    with pytest.raises(DatosNoDisponiblesError):
        leer_bandeja("2017-01-15", bandeja)  # la carga histórica llega el 31-ene


def test_bandeja_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        leer_bandeja("2017-03-15", tmp_path / "no_existe")


def test_ignora_archivos_que_no_siguen_la_convencion(bandeja):
    (bandeja / "ventas" / "notas.csv").write_text("x\n1\n")
    (bandeja / "ventas" / "ventas_2017-03-10.csv.tmp").write_text("x\n1\n")
    leer_bandeja("2017-03-15", bandeja)


def test_normaliza_fecha_de_airflow_con_zona_horaria():
    assert normalizar_fecha(datetime(2017, 3, 15, 0, 0, tzinfo=timezone.utc)) == pd.Timestamp("2017-03-15")
    assert normalizar_fecha("2017-03-15T10:30:00") == pd.Timestamp("2017-03-15")


def test_exigir_ventas_del_dia(bandeja):
    datos = leer_bandeja("2017-03-15", bandeja)
    assert exigir_ventas_del_dia(datos) is datos

    # La bandeja sintética termina el 2017-04-30: pedir una fecha posterior no tiene datos del día.
    with pytest.raises(DatosNoDisponiblesError, match="ventas hasta 2017-04-30"):
        exigir_ventas_del_dia(leer_bandeja("2017-05-10", bandeja))
