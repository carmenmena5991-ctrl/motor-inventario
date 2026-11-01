# Motor de Reposición de Inventario

Producto de ML que, de forma automática y reproducible, convierte las ventas que llegan cada día a la bandeja de un sistema de origen en un **pedido sugerido por tienda y familia de producto** para el centro de distribución. Datos: [Corporación Favorita (Kaggle)](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data).

En cada fecha de corte (`ds`) el producto entrega:

1. **Pronóstico** de los próximos 7 días por tienda y familia con un modelo de regresión cuantílica (cuantil 0.71, el ratio crítico de los costos).
2. **Pedido sugerido** (`data/predictions/pedido_<ds>.csv`) a partir de ese pronóstico y de los costos de `config/costos.yaml` (quiebre de stock 5, merma 2). Si cambian los costos, cambia la recomendación sin tocar el código.
3. **Seguimiento del desempeño** (`data/predictions/seguimiento_<ds>.csv`): error y costo real de los pedidos ya publicados, a medida que se conocen las ventas.

Cada pedido se puede rastrear hasta el código, los datos y el modelo que lo produjeron (Git + DVC + MLflow), y reproducirse byte a byte.

Las decisiones técnicas y sus alternativas están en [`DECISIONES.md`](DECISIONES.md).

---

## Arquitectura

```
data/raw_kaggle/ (CSV de Kaggle)
      │  preparar-bandeja (al levantar Docker Compose)
      ▼
bandeja_origen/<fuente>/<fuente>_<fecha de entrega>.csv     ← sistema de origen simulado (solo lectura para el flujo)
      │
      ├─ DAG 05  05:00  entrenamiento semanal y criterio de reemplazo   → MLflow (@Champions) + models/
      ├─ DAG 01  06:00  ingesta por fecha de entrega y validación        → data/raw/<ds>/, data/processed/<ds>/validado/
      │     └─ DAG 02     features (lags, medias móviles, feriados, petróleo) → data/processed/<ds>/features.parquet
      │          └─ DAG 03  pronóstico, pedido, manifiesto y versionado  → data/predictions/pedido_<ds>.{csv,json} + DVC
      └─ DAG 04  07:00  seguimiento: pedidos publicados vs. ventas reales → data/predictions/seguimiento_<ds>.csv + DVC
```

- **Infraestructura:** Docker Compose con Airflow 3.3 (api-server, scheduler y dag-processor sobre Postgres, `LocalExecutor`) y MLflow 3.17 (tracking y Model Registry) en la red `inventarioNet`.
- **Lógica en módulos** (`src/inventario/`) con pruebas; los DAGs solo orquestan.

## Requisitos

- Docker con Docker Compose v2. Reserve unos 6 GB de RAM para Docker o WSL: un entrenamiento usa unos 1.6 GiB.
- Los 5 CSV de Kaggle: `train.csv`, `stores.csv`, `holidays_events.csv`, `oil.csv` y `transactions.csv`.
- Python 3.12 o superior en el host, solo para correr las pruebas o reproducir pedidos fuera de Docker.

## Puesta en marcha desde cero

```bash
git clone https://github.com/carmenmena5991-ctrl/motor-inventario.git
cd motor-inventario

# 1. Datos de Kaggle (los 5 CSV) en data/raw_kaggle/
mkdir -p data/raw_kaggle
cp ~/Descargas/{train,stores,holidays_events,oil,transactions}.csv data/raw_kaggle/
#    o: kaggle competitions download -c store-sales-time-series-forecasting && unzip -d data/raw_kaggle ...

# 2. (Solo en Linux, si su UID no es 1000) para que Airflow pueda escribir en el repo:
echo "AIRFLOW_UID=$(id -u)" >> .env

# 3. Levantar todo
docker compose up -d --build        # o: make levantar
```

No hay más pasos manuales. Antes de Airflow corren dos servicios de un solo uso:
- **`preparar-directorios`** crea `logs/`, `models/` y `bandeja_origen/` con el UID de Airflow.
- **`preparar-bandeja`** genera la bandeja simulada desde `data/raw_kaggle/`, con su carga histórica hasta 2017-06-30 y una entrega diaria por fuente hasta la última fecha de `train.csv`. Incluye incidencias reales: entregas tardías, correcciones y archivos corruptos (`DECISIONES.md`, sección 8). Solo se regenera si cambian los CSV o `config/bandeja.yaml`. **Si falta un CSV, se detiene diciendo cuál**: `docker compose logs preparar-bandeja`.

Los DAGs quedan **activos**. Sus ejecuciones programadas de hoy terminan omitidas (*skipped*), porque la bandeja no tiene ventas del día actual.

| Servicio | URL |
|---|---|
| Airflow | http://localhost:8080 (sin login: uso local) |
| MLflow | http://localhost:5000 |

## Ejecutar el flujo sobre un periodo

```bash
make periodo DESDE=2017-07-01 HASTA=2017-08-15
```

Lanza los backfills en orden y espera a que termine cada etapa, siempre de a un run por vez:
1. DAG 05, que entrena los domingos o cuando no hay modelo vigente;
2. DAG 01, que dispara el 02 y este el 03;
3. DAG 04.

`DESDE` y `HASTA` son fechas lógicas `ds`, inclusive. Para el periodo completo de Kaggle tarda unos 33 minutos. Con `REPROCESO=todos` vuelve a ejecutar fechas que ya se habían corrido; por ser idempotente, da los mismos resultados.

Equivale a lanzar `airflow backfill create` sobre cada DAG. También se puede ejecutar un día suelto:

```bash
docker compose exec airflow-scheduler airflow dags trigger 01_ingesta_y_validacion -c '{"ds": "2017-08-10"}'
```

**Reprocesar una fecha con correcciones posteriores** no sobrescribe lo publicado. Se escribe `pedido_<ds>__conocimiento_<fecha>.csv` (`DECISIONES.md`, sección 8.3):

```bash
docker compose exec airflow-scheduler airflow dags trigger 01_ingesta_y_validacion \
    -c '{"ds": "2017-07-20", "conocimiento": "2017-08-15"}'
```

### Situaciones de operación que el flujo maneja sin fallar

| Situación | Qué hace |
|---|---|
| Las ventas del día llegan tarde | El run se omite: sin ventas del día no se emite pedido. El día siguiente usa lo que haya llegado |
| Archivo inválido (p. ej. ventas negativas) | Cuarentena: se excluye, queda en el log y en `meta.json`, y se usa el reenvío válido |
| Corrección de datos ya entregados | Prevalece la última entrega. Los pedidos publicados no cambian; el seguimiento y los reentrenamientos usan el dato corregido |
| Falta una fuente secundaria (petróleo, feriados, etc.) | Se continúa. El petróleo faltante se rellena con el último precio conocido |
| Datos inválidos después de combinar | `validar_datos` detiene el run antes de contaminar el resultado |

## Configuración

| Archivo | Qué controla |
|---|---|
| `config/costos.yaml` | Costos de quiebre y de merma, horizonte (7 días), cuantil objetivo y ventana de evaluación |
| `params.yaml` | Historia de entrenamiento (365 días), día de reentrenamiento (domingo), hiperparámetros y mejora mínima para reemplazar al modelo vigente (2 %) |
| `config/bandeja.yaml` | Simulación del sistema de origen: frecuencia y retraso de cada fuente, e incidencias |
| `.env` (opcional, no versionado) | `AIRFLOW_UID` y `DVC_REMOTO` (ver abajo) |

## Modelado

- **Baseline:** promedio móvil de 28 días por tienda y familia, con un margen de seguridad del 15 %.
- **Modelo de ML:** `HistGradientBoostingRegressor` con pérdida cuantílica (q = 0.71), usando lags de 7, 14 y 28 días, medias móviles, promociones, feriados, petróleo y calendario.
- **Validación temporal**, igual para ambos: los últimos 14 días quedan fuera del entrenamiento.
- **Métricas:** RMSE y MAE (técnicas) y **costo total de inventario** (negocio: quiebres × 5 + merma × 2). Gana el candidato de menor costo.
- **Criterio de reemplazo:** el candidato reemplaza al modelo vigente solo si, **en la misma ventana de validación**, cuesta al menos un 2 % menos (`DECISIONES.md`, sección 9). Cada pedido usa el modelo que estaba vigente en su fecha.

## Versionado de datos y modelos con DVC

Nada de datos ni de modelos va en Git. Al terminar, los DAGs 03, 04 y 05 ejecutan `dvc add` y `dvc push` sobre:

| Ruta | Contenido |
|---|---|
| `data/raw_kaggle/` | Datos de entrada (CSV de Kaggle) |
| `data/raw/<ds>/` | Foto de lo entregado en la bandeja hasta `ds` |
| `data/processed/<ds>/` | Datos validados y features |
| `data/predictions/` | Pedidos publicados, su **manifiesto** `.json` y los reportes de seguimiento |
| `models/<nombre>/v<N>/` | Cada versión del modelo que se usó o se entrenó, exportada desde MLflow |

El manifiesto de cada pedido registra:
- el commit del código;
- la versión y la huella MD5 del modelo;
- las huellas de los datos crudos, de las features y del propio pedido.

El pedido se genera con la copia exportada del modelo, así que lo que versiona DVC es exactamente lo que lo produjo (`DECISIONES.md`, sección 10).

**Remoto de DVC.** El repo trae configurado un remoto local de la autora. Para usar uno propio:

```bash
dvc remote add -d --local mi_remoto /ruta/a/mi/remoto      # o s3://..., gdrive://..., etc.
echo "DVC_REMOTO=/ruta/a/mi/remoto" >> .env                # para que los contenedores lo monten
docker compose up -d
```

Sin remoto montado, los datos quedan en la caché local de DVC (`.dvc/cache`) y la tarea `versionar` lo informa sin fallar.

Los punteros `.dvc` que actualizan los DAGs se confirman en Git **en una rama con Pull Request**. Airflow nunca escribe en Git.

## Reproducir un pronóstico publicado

```bash
git checkout <commit con los punteros .dvc de la publicación>
dvc pull                 # o dvc checkout, si los datos están en la caché local
python scripts/reproducir_pedido.py --clave 2017-07-31      # o: make reproducir CLAVE=2017-07-31
```

El script hace tres cosas:
1. comprueba que las features, los datos crudos y el modelo restaurados por DVC tengan las huellas del manifiesto;
2. vuelve a pronosticar con ese modelo y esas features;
3. compara byte a byte el pedido regenerado con el publicado.

No usa MLflow ni la bandeja. Para el entorno del host use las mismas versiones que la imagen de Airflow: `pip install -r docker/airflow/requirements.txt`. El modelo se serializa con `skops` y necesita la misma versión de scikit-learn.

### Evidencia: reproducción desde un clon nuevo

Publicación del periodo 2017-07-01 a 2017-08-15 (commit `6a3ce1a`, generado con el código de `39f15a7`), reproducida en una carpeta vacía:

```text
$ git clone <repo> prueba-reproduccion && cd prueba-reproduccion && git checkout 6a3ce1a
$ ls data/predictions models
ls: cannot access 'data/predictions': No such file or directory
ls: cannot access 'models': No such file or directory
$ dvc pull                                   # 3 s: 35M data/raw, 97M data/processed, 4.2M data/predictions, 23M models

$ python scripts/reproducir_pedido.py --clave 2017-07-31
Pedido 2017-07-31: modelo v5, código 39f15a7b8339aae8f09860571344d1da1c923ced
  features      coincide con el manifiesto
  datos_crudos  coincide con el manifiesto
  modelo        coincide con el manifiesto
  md5 publicado   06c2daad75b79f9dfecdffd56ff2bf00
  md5 reproducido 06c2daad75b79f9dfecdffd56ff2bf00  (1782 filas)
Resultado: IDÉNTICO
```

- **Los 43 pedidos del periodo se reproducen idénticos.** El 2017-07-01, el 2017-07-21, el 2017-07-31 y el 2017-08-15 se muestran en [`docs/EVIDENCIA.md`](docs/EVIDENCIA.md).
- **Un pedido alterado a propósito** (una cantidad cambiada a mano) da `Resultado: DISTINTO`.
- El pedido del 2017-07-31 tiene el mismo MD5 en dos publicaciones hechas con código distinto (`600d5ca` y `39f15a7`), porque los cambios entre ambas no afectan el cálculo del pedido.

## Evidencia de funcionamiento

Resumen de `make periodo DESDE=2017-07-01 HASTA=2017-08-15`. El detalle, con todas las tablas, está en [`docs/EVIDENCIA.md`](docs/EVIDENCIA.md).

**Ejecuciones en Airflow:** 0 tareas fallidas en 33 minutos.

| DAG | Runs exitosos | Notas |
|---|---:|---|
| 05 entrenamiento | 46 | 7 entrenamientos (domingos); los demás días, short-circuit |
| 01 ingesta y validación | 46 | 3 días omitidos por ventas tardías: 07-05, 07-28 y 08-07. 2 archivos en cuarentena |
| 02 features | 43 | |
| 03 pronóstico y pedido | 43 | 43 pedidos con manifiesto, versionados con DVC |
| 04 seguimiento | 46 | 36 pedidos evaluados con el horizonte completo |

**Comparación de modelos.** Costo total de inventario en la ventana de validación de cada entrenamiento:

| Fecha de corte | Baseline | HGB (candidato) | Vigente v5 | Decisión |
|---|---:|---:|---:|---|
| 2017-07-02 | 10,728,587 | 4,476,412 | 4,515,259 | no reemplaza (mejora 0.9 %) |
| 2017-07-09 | 10,292,861 | 4,616,634 | 4,644,324 | no reemplaza (0.6 %) |
| 2017-07-16 | 8,640,706 | 4,614,713 | 4,614,944 | no reemplaza (0.0 %) |
| 2017-07-23 | 8,691,291 | 4,541,512 | 4,570,200 | no reemplaza (0.6 %) |
| 2017-07-30 | 8,351,865 | 4,545,015 | 4,519,935 | no reemplaza (−0.6 %) |
| 2017-08-06 | 8,988,107 | 5,804,480 | 5,700,284 | no reemplaza (−1.8 %) |
| 2017-08-13 | 8,288,274 | 5,934,833 | 5,980,119 | no reemplaza (0.8 %) |

El modelo de ML reduce el costo del baseline entre un 28 % y un 58 % en cada corte. Ningún candidato mejoró al vigente (v5, entrenado al 2017-07-01) por el 2 % exigido, así que los 43 pedidos usan v5.

**Seguimiento del desempeño.** 36 pedidos con el horizonte completo, emitidos del 2017-07-01 al 2017-08-08:

| Venta real | Pedido | Cobertura | Quiebre | Merma | MAE medio | Costo real |
|---:|---:|---:|---:|---:|---:|---:|
| 212.9 M | 225.1 M | 105.7 % | 2.5 % de la venta | 8.2 % de la venta | 356.6 | 61.9 M |

- **11 pedidos cambiaron su venta real entre evaluaciones**, porque llegaron correcciones o ventas tardías después. El DAG 04 reevalúa con la última versión de los datos.
- **En agosto suben la merma y el MAE** (de 412 a 594): el seguimiento muestra que el modelo vigente se está quedando atrás, aunque en la validación de cada entrenamiento los candidatos no lo superaron. Ver `docs/EVIDENCIA.md`.

## Pruebas y CI

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pytest -q                       # o: make pruebas
```

GitHub Actions corre `pytest` en cada Pull Request hacia `main`, con Python 3.12 y 3.14, y valida los DAGs con Airflow 3. La rama `main` está protegida: todo cambio entra por Pull Request.

## Estructura del repositorio

```
src/inventario/      lógica del producto (ingesta, validación, features, entrenamiento, predicción,
                     seguimiento, simulación de la bandeja, versionado, reproducción, ejecución de periodos)
dags/                DAGs de Airflow (solo orquestan)
tests/               pruebas automatizadas (pytest)
config/              costos.yaml y bandeja.yaml
params.yaml          parámetros de entrenamiento y promoción
scripts/             preparar_bandeja, ejecutar_periodo, reproducir_pedido, entrenar_modelo, comparar_cuantiles
docker/airflow/      imagen de Airflow (dependencias del motor y DVC)
docker-compose.yaml  Airflow 3 + MLflow 3 + Postgres en la red inventarioNet
*.dvc, data/*.dvc    punteros DVC de datos y modelos
DECISIONES.md        decisiones técnicas y de producto, alternativas y justificación
CLAUDE.md            reglas del proyecto para el agente de programación
```
