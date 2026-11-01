ARG AIRFLOW_VERSION=3.3.2
FROM apache/airflow:${AIRFLOW_VERSION}-python3.12

ARG AIRFLOW_VERSION
COPY requirements.txt /tmp/requirements-inventario.txt

# Se fija apache-airflow a la versión de la imagen para que pip no la cambie al resolver.
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /tmp/requirements-inventario.txt

# DVC sin SCM dentro del contenedor (ver dvc-global/config).
COPY dvc-global/config /opt/dvc-global/config
# /opt/airflow pertenece al usuario de la imagen y el .git montado al del host: sin
# safe.directory, git se niega a leer el commit del código para los manifiestos.
ENV DVC_GLOBAL_CONFIG_DIR=/opt/dvc-global \
    DVC_NO_ANALYTICS=1 \
    GIT_CONFIG_COUNT=1 \
    GIT_CONFIG_KEY_0=safe.directory \
    GIT_CONFIG_VALUE_0=/opt/airflow
