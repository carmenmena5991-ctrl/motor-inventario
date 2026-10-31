"""Variables explicativas de series de tiempo por tienda (store_nbr) y familia (family).

Todas las variables que dependen de ventas o petróleo se desplazan al menos `horizonte`
días: para predecir el día t emitiendo en t - horizonte, solo se usa información hasta
t - horizonte. Así el mismo cálculo sirve para entrenar y para predecir sin leakage.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

LAGS = (7, 14, 28)
VENTANAS = (7, 14, 28)
CLAVES = ["store_nbr", "family"]
COLUMNAS_CATEGORICAS = ["store_nbr", "family"]
TIPOS_NO_FERIADO = {"Work Day"}


def nombres_features(lags: Sequence[int] = LAGS, ventanas: Sequence[int] = VENTANAS) -> list[str]:
    """Columnas numéricas que consume el modelo (las categóricas van aparte)."""
    return (
        [f"lag_{k}" for k in lags]
        + [f"media_movil_{w}" for w in ventanas]
        + [
            "onpromotion",
            "en_promocion",
            "petroleo",
            "es_feriado_nacional",
            "es_feriado_local",
            "es_feriado",
            "dia_semana",
            "dia_mes",
            "mes",
        ]
    )


def _desplazar(df: pd.DataFrame, columna_origen: str, columna_destino: str, dias: int) -> pd.DataFrame:
    """Mueve el valor observado en la fecha d a la fecha d + dias (merge por fecha, robusto a
    días faltantes en la serie)."""
    out = df[["date", *CLAVES, columna_origen]].copy()
    out["date"] = out["date"] + pd.Timedelta(days=dias)
    return out.rename(columns={columna_origen: columna_destino})


def _agregar_lags_y_medias(
    df: pd.DataFrame, lags: Sequence[int], ventanas: Sequence[int], horizonte: int
) -> pd.DataFrame:
    historia = df.loc[df["sales"].notna(), ["date", *CLAVES, "sales"]].sort_values([*CLAVES, "date"])

    for k in lags:
        df = df.merge(_desplazar(historia, "sales", f"lag_{k}", k), on=["date", *CLAVES], how="left")

    indexada = historia.set_index("date")
    for w in ventanas:
        # Media de las ventas en (d - w, d], publicada en d + horizonte.
        medias = (
            indexada.groupby(CLAVES)["sales"]
            .rolling(f"{w}D", min_periods=1)
            .mean()
            .rename("media")
            .reset_index()
        )
        df = df.merge(
            _desplazar(medias, "media", f"media_movil_{w}", horizonte), on=["date", *CLAVES], how="left"
        )
    return df


def _agregar_petroleo(df: pd.DataFrame, petroleo: pd.DataFrame, horizonte: int) -> pd.DataFrame:
    """Precio del petróleo conocido al emitir (fecha - horizonte), rellenando días sin cotización."""
    serie = (
        petroleo.dropna(subset=["date"])
        .drop_duplicates("date", keep="last")
        .set_index("date")["dcoilwtico"]
        .astype(float)
        .sort_index()
    )
    fechas_necesarias = (df["date"] - pd.Timedelta(days=horizonte)).drop_duplicates()
    if serie.empty:
        valores = pd.Series(float("nan"), index=fechas_necesarias)
    else:
        rango = pd.date_range(min(serie.index.min(), fechas_necesarias.min()), fechas_necesarias.max(), freq="D")
        # Solo hacia adelante (ffill): nunca usa precios futuros.
        valores = serie.reindex(rango.union(serie.index)).ffill().reindex(fechas_necesarias)
    mapa = pd.DataFrame({"date": valores.index + pd.Timedelta(days=horizonte), "petroleo": valores.to_numpy()})
    return df.merge(mapa, on="date", how="left")


def _agregar_feriados(
    df: pd.DataFrame, feriados: pd.DataFrame, tiendas: pd.DataFrame | None
) -> pd.DataFrame:
    transferido = feriados["transferred"].astype(str).str.lower().isin({"true", "1"})
    vigentes = feriados.loc[~transferido & ~feriados["type"].isin(TIPOS_NO_FERIADO)]

    nacionales = set(vigentes.loc[vigentes["locale"] == "National", "date"])
    df["es_feriado_nacional"] = df["date"].isin(nacionales).astype(int)

    df["es_feriado_local"] = 0
    if tiendas is not None and {"city", "state"}.issubset(tiendas.columns):
        locales = vigentes.loc[vigentes["locale"].isin(["Local", "Regional"]), ["date", "locale_name"]]
        ubicacion = df[["date", "store_nbr"]].merge(
            tiendas[["store_nbr", "city", "state"]], on="store_nbr", how="left"
        )
        pares = set(zip(locales["date"], locales["locale_name"]))
        es_local = [
            (d, c) in pares or (d, s) in pares
            for d, c, s in zip(ubicacion["date"], ubicacion["city"], ubicacion["state"])
        ]
        df["es_feriado_local"] = pd.Series(es_local, index=df.index).astype(int)

    df["es_feriado"] = (df["es_feriado_nacional"] | df["es_feriado_local"]).astype(int)
    return df


def construir_features(
    ventas: pd.DataFrame,
    petroleo: pd.DataFrame,
    feriados: pd.DataFrame,
    tiendas: pd.DataFrame | None = None,
    horizonte: int = 7,
    lags: Sequence[int] = LAGS,
    ventanas: Sequence[int] = VENTANAS,
) -> pd.DataFrame:
    """Devuelve `ventas` con las variables explicativas añadidas.

    Las filas con `sales` nulo (días futuros a predecir) también reciben features, calculadas
    solo con la historia disponible. No modifica los DataFrames de entrada.
    """
    if horizonte < 1:
        raise ValueError("horizonte debe ser >= 1")
    if min(lags) < horizonte:
        raise ValueError(f"Todos los lags deben ser >= horizonte ({horizonte}) para evitar leakage")

    df = ventas[["date", *CLAVES, "sales", "onpromotion"]].copy()
    df = _agregar_lags_y_medias(df, lags, ventanas, horizonte)

    df["onpromotion"] = df["onpromotion"].fillna(0)
    df["en_promocion"] = (df["onpromotion"] > 0).astype(int)

    df = _agregar_petroleo(df, petroleo, horizonte)
    df = _agregar_feriados(df, feriados, tiendas)

    df["dia_semana"] = df["date"].dt.day_of_week
    df["dia_mes"] = df["date"].dt.day
    df["mes"] = df["date"].dt.month

    # Todas las numéricas como float: el esquema de MLflow no admite nulos en enteros.
    numericas = nombres_features(lags, ventanas)
    df[numericas] = df[numericas].astype("float64")
    return df.sort_values(["date", *CLAVES]).reset_index(drop=True)
