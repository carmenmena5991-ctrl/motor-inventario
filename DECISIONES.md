# Registro de Decisiones Técnicas y de Producto - Laboratorio 1

## 1. Manejo del Tiempo y Data Leakage
- **Filtrado por Fecha de Corte:** Las funciones de ingesta descartan cualquier registro posterior a la fecha lógica de ejecución (`logical_date` / `ds`), garantizando que solo se use la información disponible hasta el momento de emisión.
- **Horizonte de Rezago en Features:** Las variables explicativas (lags y medias móviles) se calculan con una ventana mínima de 7 días atrás para evitar fugas de información futura durante el entrenamiento e inferencia.

## 2. Idempotencia, Backfill y Manejo de Errores
- **Idempotencia:** La generación de pronósticos escribe los resultados en `data/predictions/pedido_<fecha>.csv` de forma determinista y sobreescribible.
- **Validación Estricta:** Si se detectan datos con ventas/precios negativos o esquemas alterados, `validacion.py` lanza una excepción `DatosInvalidosError` para detener el DAG sin contaminar los modelos.

## 3. Métrica de Negocio vs. Métrica Técnica
- Se contrapone el error técnico tradicional (MAE/RMSE) frente a la penalización económica por inventario:
  - Costo de Quiebre de Stock = $5.0 (Venta perdida)
  - Costo de Merma = $2.0 (Producto vencido)
- El modelo candidato es evaluado seleccionando el que minimice el Costo Total de Inventario.

## 4. Promoción y Registro de Modelo Campeón
- Se registran ejecuciones en MLflow clasificando modelos. El modelo óptimo se promueve dinámicamente asignándole el alias `@Champions`, permitiendo a los DAGs de Airflow consumirlo sin modificar código.

## 5. Infraestructura (Docker Compose)
- **MLflow 3.17.2 en servidor y cliente:** el cliente (entrenamiento y Airflow) usa la API de modelos de MLflow 3; un servidor 2.x no la soporta. Las versiones se fijan iguales en `docker/airflow/requirements.txt`.
- **Artefactos servidos por MLflow** (`--artifacts-destination` + `--serve-artifacts`): los modelos se suben y descargan a través del servidor (`mlflow-artifacts:/`), sin depender de rutas locales compartidas. Base de datos y artefactos viven en volúmenes de Docker (`mlflow_db`, `mlflow_artefactos`) y no en `models/`, que está rastreado por DVC.
- **`--allowed-hosts`:** MLflow 3 solo acepta `localhost` por defecto; se habilita `mlflow:5000` para las llamadas desde la red `inventarioNet`.
- **Airflow 3 con componentes separados** (`api-server`, `scheduler`, `dag-processor`) sobre Postgres y `LocalExecutor`. El comando `webserver` y `SequentialExecutor` ya no existen en Airflow 3.
- **Mismas versiones de librerías al entrenar y al predecir:** el modelo se serializa con `skops` y debe cargarse con la misma versión de scikit-learn con la que se entrenó.
- **Historia de entrenamiento acotada** (`--dias-historia`): con el dataset completo (3 M filas) y 6 GB de RAM se entrena con el último año para limitar memoria y tiempo.
