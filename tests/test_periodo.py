"""Orden de los backfills de un periodo y espera a que cada etapa termine."""

from __future__ import annotations

import json
import subprocess

import pytest

from inventario.periodo import ejecutar_periodo, esperar_inactividad, rango_backfill, runs_activos


def test_rango_backfill_incluye_el_ultimo_dia():
    assert rango_backfill("2017-07-01", "2017-08-15") == ("2017-07-01", "2017-08-16")
    assert rango_backfill("2017-07-31", "2017-07-31") == ("2017-07-31", "2017-08-01")
    with pytest.raises(ValueError):
        rango_backfill("2017-08-02", "2017-08-01")


class AirflowFalso:
    """Simula la CLI: cada `list-runs` devuelve el siguiente estado programado del DAG."""

    def __init__(self, estados: dict[str, list[list[str]]] | None = None):
        self.llamadas: list[list[str]] = []
        self.estados = estados or {}

    def __call__(self, comando, **_):
        self.llamadas.append(comando)
        if comando[1:3] == ["dags", "list-runs"]:
            pendientes = self.estados.get(comando[3], [])
            actuales = pendientes.pop(0) if pendientes else ["success"]
            salida = "aviso de la CLI\n" + json.dumps([{"state": s} for s in actuales])
            return subprocess.CompletedProcess(comando, 0, salida, "")
        return subprocess.CompletedProcess(comando, 0, "", "")


def test_runs_activos_ignora_avisos_antes_del_json():
    falso = AirflowFalso({"d": [["running", "queued", "success", "failed"]]})
    assert runs_activos(["d"], falso) == 2


def test_espera_varias_confirmaciones_seguidas():
    # El 01 termina y un instante después aparece el 02 en cola: no basta una consulta vacía.
    falso = AirflowFalso({"d": [["running"], ["success"], ["queued"], ["success"], ["success"], ["success"]]})
    pausas = []
    esperar_inactividad(["d"], falso, intervalo=1, confirmaciones=3, dormir=pausas.append)
    consultas = [c for c in falso.llamadas if c[1:3] == ["dags", "list-runs"]]
    assert len(consultas) == 6


def test_limite_de_espera():
    falso = AirflowFalso({"d": [["running"]] * 100})
    with pytest.raises(TimeoutError):
        esperar_inactividad(["d"], falso, intervalo=10, limite=30, dormir=lambda _: None)


def test_periodo_lanza_05_01_04_en_orden_y_de_a_uno():
    falso = AirflowFalso()
    lanzados = ejecutar_periodo("2017-07-01", "2017-07-10", "todos", falso, intervalo=0, dormir=lambda _: None)
    assert lanzados == ["05_entrenamiento_modelo", "01_ingesta_y_validacion", "04_seguimiento_desempeno"]

    backfills = [c for c in falso.llamadas if c[1:3] == ["backfill", "create"]]
    assert [c[c.index("--dag-id") + 1] for c in backfills] == lanzados
    for c in backfills:
        assert c[c.index("--from-date") + 1] == "2017-07-01"
        assert c[c.index("--to-date") + 1] == "2017-07-11"
        assert c[c.index("--max-active-runs") + 1] == "1"
        assert c[c.index("--reprocess-behavior") + 1] == "completed"

    # Tras lanzar el 01 se espera también a los DAGs que dispara (02 y 03).
    i01 = falso.llamadas.index(backfills[1])
    i04 = falso.llamadas.index(backfills[2])
    esperados = {c[3] for c in falso.llamadas[i01:i04] if c[1:3] == ["dags", "list-runs"]}
    assert esperados == {"01_ingesta_y_validacion", "02_preparacion_y_features", "03_pronostico_y_pedido"}


def test_falla_de_la_cli_se_informa():
    def falla(comando, **_):
        return subprocess.CompletedProcess(comando, 1, "", "DAG no encontrado")

    with pytest.raises(RuntimeError, match="DAG no encontrado"):
        ejecutar_periodo("2017-07-01", "2017-07-02", ejecutar=falla)


def test_reproceso_invalido():
    with pytest.raises(ValueError):
        ejecutar_periodo("2017-07-01", "2017-07-02", "algunos", AirflowFalso())
