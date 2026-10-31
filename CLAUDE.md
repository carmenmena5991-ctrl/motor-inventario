## Estructura del Proyecto
- Código modular en `src/inventario/`
- DAGs de Airflow en `dags/`
- Pruebas unitarias en `tests/`
- Configuración parametrizada en `config/costos.yaml`

## Reglas de Control de Versiones y Datos
- NINGÚN archivo de datos (`.csv`, `.db`, etc.) ni modelos entrenados (`.joblib`) debe subir a Git.
- Todos los datos y artefactos de modelos deben rastrearse mediante DVC.
- Todo trabajo debe realizarse en ramas secundarias y Pull Requests hacia `main`.

## Ejecución y Entorno
- La infraestructura corre en Docker Compose (Airflow 3 + MLflow).
- La red compartida es `inventarioNet`.
- Todas las funciones deben ser idempotentes y contar con pruebas automatizadas (`pytest`).
