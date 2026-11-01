import pytest
import yaml

from inventario.config import cargar_costos, cargar_params


def test_configuracion_del_repositorio_es_valida():
    cargar_costos()
    params = cargar_params()
    assert 0 <= params["promocion"]["mejora_minima"] < 1
    assert 0 <= params["entrenamiento"]["dia_reentrenamiento"] <= 6


@pytest.mark.parametrize(
    "cambio, error",
    [
        ({"promocion": {}}, KeyError),
        ({"promocion": {"mejora_minima": 1.5}}, ValueError),
        ({"entrenamiento": {"dias_historia": 365, "dia_reentrenamiento": 9, "hiperparametros": {}}}, ValueError),
    ],
)
def test_params_invalidos(tmp_path, cambio, error):
    params = cargar_params()
    params.update(cambio)
    ruta = tmp_path / "params.yaml"
    ruta.write_text(yaml.safe_dump(params), encoding="utf-8")
    with pytest.raises(error):
        cargar_params(ruta)
