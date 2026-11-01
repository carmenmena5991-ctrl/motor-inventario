"""Entrenamiento del Baseline y del modelo de ML, registro en MLflow y promoción.

1. Baseline y ML se entrenan y evalúan con el mismo esquema temporal: los últimos
   `evaluacion.dias_historicos_comparar` días quedan fuera del entrenamiento.
2. El de menor Costo Total de Inventario es el candidato y se registra en el Model Registry.
3. Criterio de reemplazo: el modelo vigente (@Champions) se evalúa en la misma ventana; el
   candidato recibe el alias solo si no hay vigente o si su costo es al menos
   `promocion.mejora_minima` (params.yaml) menor que el del vigente.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from inventario.config import cargar_costos, cargar_params
from inventario.evaluacion import evaluar
from inventario.features import COLUMNAS_CATEGORICAS, VENTANAS, construir_features, nombres_features
from inventario.ingesta import BANDEJA_DEFAULT, leer_bandeja
from inventario.validacion import validar_datos

TRACKING_URI_DEFAULT = "http://mlflow:5000"
EXPERIMENTO_DEFAULT = "motor_inventario"
MODELO_REGISTRADO_DEFAULT = "motor_inventario"
ALIAS_CAMPEON = "Champions"

HIPERPARAMETROS_DEFAULT = {
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
    "random_state": 42,
}


# MLflow 3 serializa los modelos sklearn con skops, que solo carga tipos declarados como
# confiables. Estos son los que usan nuestros dos candidatos; son seguros porque los
# artefactos los genera este mismo pipeline y viven en nuestro servidor MLflow. No añadir
# tipos sin revisarlos.
TIPOS_CONFIABLES_SKOPS = [
    "inventario.entrenar.BaselinePromedioMovil",
    "inventario.entrenar.ModeloCuantiles",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "sklearn.utils.validation.check_array",
    "functools.partial",
]


def columnas_modelo() -> list[str]:
    return COLUMNAS_CATEGORICAS + nombres_features()


def resolver_tracking_uri(tracking_uri: str | None = None) -> str:
    return tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", TRACKING_URI_DEFAULT)


class BaselinePromedioMovil(RegressorMixin, BaseEstimator):
    """Promedio móvil histórico por tienda y familia, inflado por el factor de seguridad.

    Usa la columna `media_movil_{ventana}` de features.py, que ya está desplazada por el
    horizonte (solo información disponible al emitir).
    """

    def __init__(self, ventana: int = max(VENTANAS), factor_seguridad: float = 0.15):
        self.ventana = ventana
        self.factor_seguridad = factor_seguridad

    def fit(self, X: pd.DataFrame, y=None):
        columna = f"media_movil_{self.ventana}"
        if columna not in X.columns:
            raise KeyError(f"Falta la columna {columna}")
        self.columna_ = columna
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        base = pd.to_numeric(X[self.columna_], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return np.clip(base * (1 + self.factor_seguridad), 0, None)


def construir_modelo_ml(cuantil: float, hiperparametros: dict[str, Any] | None = None) -> Pipeline:
    """HistGradientBoostingRegressor con pérdida cuantílica sobre el cuantil objetivo."""
    hp = {**HIPERPARAMETROS_DEFAULT, **(hiperparametros or {})}
    numericas = nombres_features()
    preprocesado = ColumnTransformer(
        [
            (
                "categoricas",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
                COLUMNAS_CATEGORICAS,
            ),
            ("numericas", "passthrough", numericas),
        ],
        remainder="drop",
    )
    es_categorica = [True] * len(COLUMNAS_CATEGORICAS) + [False] * len(numericas)
    regresor = HistGradientBoostingRegressor(
        loss="quantile", quantile=cuantil, categorical_features=es_categorica, **hp
    )
    return Pipeline([("preprocesado", preprocesado), ("regresor", regresor)])


def nombre_cuantil(cuantil: float) -> str:
    """0.71 -> "p71": nombre de la columna de un cuantil en el pronóstico."""
    return f"p{round(float(cuantil) * 100):d}"


class ModeloCuantiles(RegressorMixin, BaseEstimator):
    """Un HistGradientBoosting cuantílico por cuantil: el objetivo (el del pedido) y los
    del intervalo de incertidumbre (p. ej. p10, p50, p90).

    `predict` devuelve el cuantil objetivo, así que el costo, el criterio de reemplazo y el
    pedido se calculan igual que con un solo modelo. `predict_cuantiles` devuelve todos.

    Los modelos se entrenan por separado y sus predicciones pueden cruzarse (p. ej. p90 <
    p71 en alguna fila). Se ordenan por fila (*monotone rearrangement*): el conjunto de
    valores no cambia, solo qué cuantil recibe cada uno, y el error de cada cuantil no
    empeora. El cuantil objetivo que usa `predict` es el ya ordenado, el mismo del pronóstico.
    """

    def __init__(
        self,
        cuantil_objetivo: float = 0.71,
        cuantiles_intervalo: tuple[float, ...] = (),
        hiperparametros: dict[str, Any] | None = None,
    ):
        self.cuantil_objetivo = cuantil_objetivo
        self.cuantiles_intervalo = cuantiles_intervalo
        self.hiperparametros = hiperparametros

    def cuantiles(self) -> list[float]:
        return sorted({round(float(q), 4) for q in (self.cuantil_objetivo, *self.cuantiles_intervalo)})

    def fit(self, X: pd.DataFrame, y):
        self.modelos_ = {
            nombre_cuantil(q): construir_modelo_ml(q, self.hiperparametros).fit(X, y) for q in self.cuantiles()
        }
        return self

    def predict_cuantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        columnas = [nombre_cuantil(q) for q in self.cuantiles()]
        crudo = np.column_stack([self.modelos_[c].predict(X) for c in columnas])
        ordenado = np.sort(np.clip(crudo, 0, None), axis=1)
        return pd.DataFrame(ordenado, columns=columnas, index=X.index)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_cuantiles(X)[nombre_cuantil(self.cuantil_objetivo)].to_numpy()


def cobertura_cuantiles(y: np.ndarray, cuantiles: pd.DataFrame) -> dict[str, float]:
    """Calibración: fracción de valores reales por debajo de cada cuantil (un p90 bien
    calibrado deja ~90 % por debajo) y dentro del intervalo entre el menor y el mayor."""
    y = np.asarray(y, dtype=float)
    resultado = {f"cobertura_{c}": float(np.mean(y <= cuantiles[c].to_numpy())) for c in cuantiles.columns}
    if len(cuantiles.columns) >= 3:
        bajo, alto = cuantiles.columns[0], cuantiles.columns[-1]
        dentro = (y >= cuantiles[bajo].to_numpy()) & (y <= cuantiles[alto].to_numpy())
        resultado[f"cobertura_{bajo}_{alto}"] = float(np.mean(dentro))
        resultado[f"ancho_medio_{bajo}_{alto}"] = float(np.mean(cuantiles[alto] - cuantiles[bajo]))
    return resultado


def dividir_temporal(features: pd.DataFrame, dias_validacion: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Últimos `dias_validacion` días con ventas observadas -> validación; el resto -> entrenamiento."""
    con_objetivo = features.loc[features["sales"].notna()]
    fecha_max = con_objetivo["date"].max()
    limite = fecha_max - pd.Timedelta(days=dias_validacion)
    entrenamiento = con_objetivo.loc[con_objetivo["date"] <= limite]
    validacion = con_objetivo.loc[con_objetivo["date"] > limite]
    if entrenamiento.empty or validacion.empty:
        raise ValueError(
            f"Historia insuficiente para separar {dias_validacion} días de validación "
            f"({con_objetivo['date'].min().date()} a {fecha_max.date()})"
        )
    return entrenamiento, validacion


@dataclass
class ResultadoEntrenamiento:
    ganador: str
    version: str
    run_id: str
    metricas: dict[str, dict[str, float]] = field(default_factory=dict)
    promovido: bool = True
    motivo: str = ""
    version_vigente: str | None = None
    costo_vigente: float | None = None


def decidir_reemplazo(
    costo_candidato: float, costo_vigente: float | None, hay_vigente: bool, mejora_minima: float
) -> tuple[bool, str]:
    """Criterio explícito para reemplazar el modelo vigente. Devuelve (promover, motivo).

    - Sin modelo vigente: se promueve el candidato.
    - Vigente que no se pudo evaluar (p. ej. incompatible con las features actuales): se
      promueve el candidato, porque el vigente ya no es utilizable.
    - En otro caso, se promueve solo si el costo del candidato es al menos `mejora_minima`
      menor que el del vigente en la misma ventana de validación.
    """
    if not hay_vigente:
        return True, "no hay modelo vigente"
    if costo_vigente is None:
        return True, "el modelo vigente no se pudo evaluar con los datos actuales"
    if costo_vigente <= 0:
        return False, "el modelo vigente tiene costo 0 en la ventana de validación"
    mejora = 1 - costo_candidato / costo_vigente
    if mejora >= mejora_minima:
        return True, f"el candidato reduce el costo un {mejora:.1%} (mínimo exigido {mejora_minima:.1%})"
    return False, f"mejora de {mejora:.1%}, menor que el mínimo exigido de {mejora_minima:.1%}"


def debe_entrenar(fecha, dia_reentrenamiento: int, hay_vigente: bool) -> tuple[bool, str]:
    """El DAG 05 entrena si no hay modelo vigente o si `fecha` cae en el día de reentrenamiento."""
    if not hay_vigente:
        return True, "no hay modelo vigente"
    dia = pd.Timestamp(fecha).day_of_week
    if dia == int(dia_reentrenamiento):
        return True, "día de reentrenamiento"
    return False, f"no es día de reentrenamiento (día {dia}, se reentrena el día {dia_reentrenamiento})"


def version_vigente(
    nombre_modelo: str = MODELO_REGISTRADO_DEFAULT, alias: str = ALIAS_CAMPEON, tracking_uri: str | None = None
) -> str | None:
    """Versión registrada que tiene el alias, o None si todavía no existe."""
    cliente = MlflowClient(tracking_uri=resolver_tracking_uri(tracking_uri))
    try:
        return str(cliente.get_model_version_by_alias(nombre_modelo, alias).version)
    except MlflowException:
        return None


def version_vigente_a_fecha(
    fecha, nombre_modelo: str = MODELO_REGISTRADO_DEFAULT, tracking_uri: str | None = None
) -> str | None:
    """Modelo vigente en `fecha`: la última versión promovida entrenada con datos hasta esa
    fecha (etiqueta `fecha_corte` <= fecha). Si hay varias con la misma fecha, la más reciente.

    Así un pedido usa el modelo que existía al emitirlo, aunque los backfills de los DAGs 05
    y 01 se ejecuten en otro orden. No se consideran las versiones sin etiqueta `decision`
    (anteriores al criterio de reemplazo) ni sin etiqueta `cuantiles` (anteriores al
    pronóstico con intervalos de incertidumbre).
    """
    limite = str(pd.Timestamp(fecha).date())
    cliente = MlflowClient(tracking_uri=resolver_tracking_uri(tracking_uri))
    try:
        versiones = cliente.search_model_versions(f"name='{nombre_modelo}'")
    except MlflowException:
        return None
    candidatas = [
        (v.tags["fecha_corte"], int(v.version))
        for v in versiones
        if v.tags.get("decision") == "promovido"
        and "cuantiles" in v.tags
        and v.tags.get("fecha_corte", "9999") <= limite
    ]
    return str(max(candidatas)[1]) if candidatas else None


def _fecha_corte_de_version(cliente: MlflowClient, nombre_modelo: str, version: str) -> str | None:
    return cliente.get_model_version(nombre_modelo, version).tags.get("fecha_corte")


def _evaluar_version(
    nombre_modelo: str, version: str, X: pd.DataFrame, y: np.ndarray, config: dict[str, Any]
) -> float | None:
    """Costo de una versión registrada en la ventana de validación del candidato."""
    try:
        modelo = mlflow.pyfunc.load_model(f"models:/{nombre_modelo}/{version}")
        pred = np.clip(np.asarray(modelo.predict(X), dtype=float), 0, None)
    except Exception:  # noqa: BLE001 - un vigente inutilizable no debe impedir entrenar
        return None
    return evaluar(y, pred, config)["costo_total_inventario"]


def entrenar(
    features: pd.DataFrame,
    config: dict[str, Any] | None = None,
    tracking_uri: str | None = None,
    experimento: str = EXPERIMENTO_DEFAULT,
    nombre_modelo: str = MODELO_REGISTRADO_DEFAULT,
    hiperparametros: dict[str, Any] | None = None,
    alias: str = ALIAS_CAMPEON,
    etiquetas: dict[str, str] | None = None,
    mejora_minima: float = 0.0,
    cuantiles_intervalo: tuple[float, ...] = (),
) -> ResultadoEntrenamiento:
    """Entrena Baseline y ML, registra el mejor como candidato y decide si reemplaza al
    modelo vigente a la fecha de corte (`version_vigente_a_fecha` + `decidir_reemplazo`).

    Si se promueve, la versión queda etiquetada `decision=promovido` y recibe `@alias`,
    salvo que el alias ya apunte a un modelo entrenado con datos más recientes (el alias
    nunca retrocede). Re-ejecutar con los mismos datos registra otra versión, pero al no
    mejorar al vigente que dejó la ejecución anterior, no se promueve.
    """
    config = config or cargar_costos()
    fecha_corte = (etiquetas or {}).get("fecha_corte") or str(features.loc[features["sales"].notna(), "date"].max().date())
    etiquetas = {**(etiquetas or {}), "fecha_corte": fecha_corte}
    pedido = config["parametros_pedido"]
    dias_validacion = int(config["evaluacion"]["dias_historicos_comparar"])

    entrenamiento, validacion = dividir_temporal(features, dias_validacion)
    columnas = columnas_modelo()
    X_tr, y_tr = entrenamiento[columnas], entrenamiento["sales"].to_numpy(dtype=float)
    X_va, y_va = validacion[columnas], validacion["sales"].to_numpy(dtype=float)

    candidatos = {
        "baseline_promedio_movil": BaselinePromedioMovil(
            factor_seguridad=float(pedido["factor_seguridad_default"])
        ),
        # El cuantil objetivo (el del pedido) y los del intervalo de incertidumbre.
        "hist_gradient_boosting": ModeloCuantiles(
            float(pedido["cuantil_objetivo"]), tuple(float(q) for q in cuantiles_intervalo), hiperparametros
        ),
    }

    tracking_uri = resolver_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experimento)

    metricas: dict[str, dict[str, float]] = {}
    modelos_logueados: dict[str, Any] = {}
    run_ids: dict[str, str] = {}

    with mlflow.start_run(run_name="entrenamiento") as run_padre:
        mlflow.set_tags(etiquetas or {})
        mlflow.log_params(
            {
                "horizonte_dias": pedido["horizonte_dias"],
                "cuantil_objetivo": pedido["cuantil_objetivo"],
                "dias_validacion": dias_validacion,
                "costo_quiebre_stock": config["costos"]["quiebre_stock"],
                "costo_merma": config["costos"]["merma"],
                "filas_entrenamiento": len(X_tr),
                "filas_validacion": len(X_va),
            }
        )

        for nombre, modelo in candidatos.items():
            with mlflow.start_run(run_name=nombre, nested=True) as run_hijo:
                modelo.fit(X_tr, y_tr)
                pred = np.clip(modelo.predict(X_va), 0, None)
                metricas[nombre] = evaluar(y_va, pred, config)

                if isinstance(modelo, ModeloCuantiles):
                    objetivo = modelo.modelos_[nombre_cuantil(modelo.cuantil_objetivo)]
                    parametros = {
                        **objetivo.named_steps["regresor"].get_params(),
                        "cuantiles": ",".join(modelo.modelos_),
                    }
                    # Calibración en la ventana de validación (métricas aparte del costo).
                    mlflow.log_metrics(cobertura_cuantiles(y_va, modelo.predict_cuantiles(X_va)))
                else:
                    parametros = modelo.get_params(deep=False)
                mlflow.log_params({"modelo": nombre, **{k: v for k, v in parametros.items() if k != "categorical_features"}})
                mlflow.log_metrics(metricas[nombre])
                modelos_logueados[nombre] = mlflow.sklearn.log_model(
                    modelo,
                    name="modelo",
                    signature=infer_signature(X_va.head(100), pred[:100]),
                    skops_trusted_types=TIPOS_CONFIABLES_SKOPS,
                )
                run_ids[nombre] = run_hijo.info.run_id

        ganador = min(metricas, key=lambda n: metricas[n]["costo_total_inventario"])
        costo_candidato = metricas[ganador]["costo_total_inventario"]
        mlflow.set_tag("modelo_ganador", ganador)
        mlflow.log_metric("costo_total_inventario_ganador", costo_candidato)

        # Criterio de reemplazo: el vigente a la fecha de corte y el candidato, en la misma
        # ventana de validación (que ninguno de los dos usó para entrenar).
        vigente = version_vigente_a_fecha(fecha_corte, nombre_modelo, tracking_uri)
        costo_vigente = _evaluar_version(nombre_modelo, vigente, X_va, y_va, config) if vigente else None
        promovido, motivo = decidir_reemplazo(costo_candidato, costo_vigente, vigente is not None, mejora_minima)
        mlflow.log_param("mejora_minima", mejora_minima)
        if costo_vigente is not None:
            mlflow.log_metric("costo_total_inventario_vigente", costo_vigente)
        mlflow.set_tags({
            "decision": "promovido" if promovido else "no_promovido",
            "motivo": motivo,
            "version_vigente": vigente or "ninguna",
        })

    version = mlflow.register_model(modelos_logueados[ganador].model_uri, nombre_modelo)
    cliente = MlflowClient(tracking_uri=tracking_uri)
    modelo_ganador = candidatos[ganador]
    etiquetas_version = {
        "modelo": ganador,
        # Columnas del pronóstico que entrega esta versión (el baseline solo el objetivo).
        "cuantiles": ",".join(modelo_ganador.modelos_) if isinstance(modelo_ganador, ModeloCuantiles)
        else nombre_cuantil(float(pedido["cuantil_objetivo"])),
        "costo_total_inventario": str(costo_candidato),
        "decision": "promovido" if promovido else "no_promovido",
        "motivo": motivo,
        "version_vigente_al_entrenar": vigente or "ninguna",
        **{k: str(v) for k, v in (etiquetas or {}).items()},
    }
    for clave, valor in etiquetas_version.items():
        cliente.set_model_version_tag(nombre_modelo, version.version, clave, valor)
    if promovido:
        actual = version_vigente(nombre_modelo, alias, tracking_uri)
        corte_actual = _fecha_corte_de_version(cliente, nombre_modelo, actual) if actual else None
        if corte_actual is None or corte_actual <= fecha_corte:
            cliente.set_registered_model_alias(nombre_modelo, alias, version.version)

    return ResultadoEntrenamiento(
        ganador=ganador,
        version=str(version.version),
        run_id=run_padre.info.run_id,
        metricas=metricas,
        promovido=promovido,
        motivo=motivo,
        version_vigente=vigente,
        costo_vigente=costo_vigente,
    )


def ejecutar_entrenamiento(
    fecha_corte,
    bandeja: str | Path = BANDEJA_DEFAULT,
    ruta_costos: str | Path | None = None,
    dias_historia: int | None = None,
    fecha_conocimiento=None,
    ruta_params: str | Path | None = None,
    exigir_datos_del_dia: bool = False,
    **kwargs,
) -> ResultadoEntrenamiento:
    """Pipeline completo: ingesta -> validación -> features -> entrenamiento -> promoción.

    Solo usa lo entregado en la bandeja hasta `fecha_corte` (o hasta `fecha_conocimiento`,
    para entrenar con correcciones posteriores). `dias_historia`, los hiperparámetros y la
    mejora mínima para reemplazar al vigente salen de params.yaml si no se indican.
    """
    from inventario.almacen import recortar_ventas

    config = cargar_costos(ruta_costos)
    params = cargar_params(ruta_params)
    if dias_historia is None:
        dias_historia = params["entrenamiento"]["dias_historia"]
    kwargs.setdefault("hiperparametros", params["entrenamiento"]["hiperparametros"])
    kwargs.setdefault("mejora_minima", float(params["promocion"]["mejora_minima"]))
    kwargs.setdefault("cuantiles_intervalo", tuple(params["entrenamiento"].get("cuantiles_intervalo") or ()))

    datos = leer_bandeja(fecha_corte, bandeja, fecha_conocimiento=fecha_conocimiento)
    if exigir_datos_del_dia:
        from inventario.ingesta import exigir_ventas_del_dia

        exigir_ventas_del_dia(datos)  # DatosNoDisponiblesError si aún no llegaron
    if dias_historia:
        datos = recortar_ventas(datos, int(dias_historia))
    datos = validar_datos(datos)
    features = construir_features(
        datos.ventas,
        datos.petroleo,
        datos.feriados,
        datos.tiendas,
        horizonte=int(config["parametros_pedido"]["horizonte_dias"]),
    )
    etiquetas = {
        "fecha_corte": str(datos.fecha_corte.date()),
        "dias_historia": str(dias_historia or "todos"),
        **kwargs.pop("etiquetas", {}),
    }
    return entrenar(features, config, etiquetas=etiquetas, **kwargs)


def asegurar_modelo_vigente(
    fecha_corte,
    bandeja: str | Path = BANDEJA_DEFAULT,
    nombre_modelo: str = MODELO_REGISTRADO_DEFAULT,
    alias: str = ALIAS_CAMPEON,
    tracking_uri: str | None = None,
    **kwargs,
) -> tuple[str, bool]:
    """Modelo vigente a `fecha_corte`. Devuelve (versión, se_entrenó_ahora).

    Si no hay ningún modelo promovido con datos hasta esa fecha (primer arranque de un clon
    nuevo, o un backfill anterior a todos los entrenamientos), entrena uno con lo entregado
    hasta `fecha_corte`; así el pronóstico nunca depende de un paso manual.
    """
    version = version_vigente_a_fecha(fecha_corte, nombre_modelo, tracking_uri)
    if version is not None:
        return version, False
    resultado = ejecutar_entrenamiento(
        fecha_corte, bandeja, nombre_modelo=nombre_modelo, alias=alias, tracking_uri=tracking_uri,
        etiquetas={"origen": "arranque_automatico"}, **kwargs,
    )
    return resultado.version, True
