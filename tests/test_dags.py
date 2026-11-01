"""Los DAGs se parsean sin errores y tienen la estructura esperada.

Requiere Airflow instalado; si no lo está, se omite. En CI corre en un job aparte con
`pytest tests/test_dags.py --noconftest`.
"""

from pathlib import Path

import pytest

pytest.importorskip("airflow")

try:  # Airflow >= 3.1
    from airflow.dag_processing.dagbag import DagBag
except ImportError:  # Airflow 3.0
    from airflow.models import DagBag

DAGS = Path(__file__).resolve().parents[1] / "dags"


@pytest.fixture(scope="module")
def dagbag():
    # Los ejemplos se desactivan con AIRFLOW__CORE__LOAD_EXAMPLES=false (el argumento
    # include_examples ya no existe en Airflow >= 3.3).
    return DagBag(dag_folder=str(DAGS))


def test_sin_errores_de_importacion(dagbag):
    assert dagbag.import_errors == {}


def test_existen_los_cuatro_dags(dagbag):
    assert set(dagbag.dags) == {
        "01_ingesta_y_validacion",
        "02_preparacion_y_features",
        "03_pronostico_y_pedido",
        "04_seguimiento_desempeno",
        "05_entrenamiento_modelo",
    }


@pytest.mark.parametrize(
    "dag_id, tareas",
    [
        ("01_ingesta_y_validacion", ["resolver_fecha", "ingestar", "validar", "disparar_preparacion_features"]),
        ("02_preparacion_y_features", ["resolver_fecha", "construir_features", "disparar_pronostico"]),
        ("03_pronostico_y_pedido", [
            "resolver_fecha", "asegurar_modelo", "exportar_modelo", "pronosticar_y_generar_pedido", "versionar",
        ]),
        ("05_entrenamiento_modelo", [
            "resolver_fecha", "hay_que_entrenar", "entrenar_y_decidir", "exportar_modelo", "versionar",
        ]),
        ("04_seguimiento_desempeno", ["resolver_fecha", "evaluar_pedidos_publicados", "versionar"]),
    ],
)
def test_tareas_en_orden(dagbag, dag_id, tareas):
    dag = dagbag.dags[dag_id]
    assert [t.task_id for t in dag.topological_sort()] == tareas


def test_encadenamiento(dagbag):
    disparo_01 = dagbag.dags["01_ingesta_y_validacion"].get_task("disparar_preparacion_features")
    disparo_02 = dagbag.dags["02_preparacion_y_features"].get_task("disparar_pronostico")
    assert disparo_01.trigger_dag_id == "02_preparacion_y_features"
    assert disparo_02.trigger_dag_id == "03_pronostico_y_pedido"
    assert disparo_01.reset_dag_run and disparo_02.reset_dag_run


def test_solo_01_y_04_estan_programados(dagbag):
    assert dagbag.dags["02_preparacion_y_features"].timetable.can_be_scheduled is False
    assert dagbag.dags["03_pronostico_y_pedido"].timetable.can_be_scheduled is False
    assert dagbag.dags["01_ingesta_y_validacion"].timetable.can_be_scheduled
    assert dagbag.dags["04_seguimiento_desempeno"].timetable.can_be_scheduled
    assert dagbag.dags["05_entrenamiento_modelo"].timetable.can_be_scheduled


# Claves del contexto de Airflow que una tarea TaskFlow no puede recibir como argumento
# explícito (Airflow 3 lanza "The key 'ds' in args is a part of kwargs and therefore reserved").
CLAVES_RESERVADAS = {
    "ds", "ds_nodash", "ts", "ts_nodash", "logical_date", "data_interval_start",
    "data_interval_end", "params", "ti", "task_instance", "dag_run", "dag", "run_id",
    "conf", "macros", "var", "conn", "inlets", "outlets", "task", "triggering_asset_events",
}


@pytest.mark.parametrize(
    "dag_id, task_id",
    [("05_entrenamiento_modelo", "entrenar_y_decidir"), ("03_pronostico_y_pedido", "asegurar_modelo")],
)
def test_tareas_que_entrenan_usan_el_pool_de_un_slot(dagbag, dag_id, task_id):
    assert dagbag.dags[dag_id].get_task(task_id).pool == "entrenamiento"


@pytest.mark.parametrize("dag_id", ["03_pronostico_y_pedido", "04_seguimiento_desempeno", "05_entrenamiento_modelo"])
def test_versionado_con_dvc_al_final_y_en_el_pool_de_un_slot(dagbag, dag_id):
    dag = dagbag.dags[dag_id]
    tarea = dag.get_task("versionar")
    assert tarea.pool == "versionado"
    assert not tarea.downstream_list  # es lo último: versiona lo que el DAG ya escribió


def test_pedido_usa_el_modelo_exportado(dagbag):
    dag = dagbag.dags["03_pronostico_y_pedido"]
    assert "exportar_modelo" in dag.get_task("pronosticar_y_generar_pedido").upstream_task_ids


def _compose() -> dict:
    import yaml

    return yaml.safe_load((DAGS.parent / "docker-compose.yaml").read_text(encoding="utf-8"))


def test_dags_activos_desde_el_primer_arranque():
    entorno = _compose()["x-airflow-comun"]["environment"]
    assert entorno["AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION"] == "false"


def test_airflow_arranca_despues_de_preparar_directorios_y_bandeja():
    servicios = _compose()["services"]
    assert servicios["airflow-init"]["depends_on"]["preparar-bandeja"]["condition"] == "service_completed_successfully"
    assert servicios["preparar-bandeja"]["depends_on"]["preparar-directorios"]["condition"] == "service_completed_successfully"
    # El flujo nunca escribe en la bandeja: solo el simulador la genera.
    montajes = servicios["preparar-bandeja"]["volumes"]
    assert "./bandeja_origen:/opt/airflow/bandeja_origen" in montajes
    assert "./bandeja_origen:/opt/airflow/bandeja_origen:ro" in _compose()["x-airflow-comun"]["volumes"]
    comando = " ".join(servicios["preparar-directorios"]["command"])
    for directorio in ("logs", "models", "bandeja_origen"):
        assert directorio in comando


@pytest.mark.parametrize("pool", ["entrenamiento", "versionado"])
def test_compose_crea_los_pools_de_un_slot(pool):
    import yaml

    compose = yaml.safe_load((DAGS.parent / "docker-compose.yaml").read_text(encoding="utf-8"))
    comando = " ".join(compose["services"]["airflow-init"]["command"])
    assert f"airflow pools set {pool} 1" in comando


def test_tareas_no_usan_claves_reservadas_del_contexto(dagbag):
    import inspect

    for dag in dagbag.dags.values():
        for tarea in dag.tasks:
            funcion = getattr(tarea, "python_callable", None)
            if funcion is None:
                continue
            conflictos = set(inspect.signature(funcion).parameters) & CLAVES_RESERVADAS
            assert not conflictos, f"{dag.dag_id}.{tarea.task_id} usa claves reservadas: {conflictos}"
