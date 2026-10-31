ARG AIRFLOW_VERSION=3.3.2
FROM apache/airflow:${AIRFLOW_VERSION}-python3.12

ARG AIRFLOW_VERSION
COPY requirements.txt /tmp/requirements-inventario.txt

# Se fija apache-airflow a la versión de la imagen para que pip no la cambie al resolver.
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /tmp/requirements-inventario.txt
