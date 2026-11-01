# Registro de Decisiones Técnicas y de Producto - Laboratorio 1

## 1. Manejo del Tiempo y Data Leakage
- **Filtrado por Fecha de Corte:** Las funciones de ingesta descartan cualquier registro posterior a la fecha lógica de ejecución (`logical_date` / `ds`), garantizando que solo se use la información disponible hasta el momento de emisión.
- **Filtrado por Fecha de Entrega:** además del corte por fecha de los datos, solo se usan los archivos que ya se habían entregado en la bandeja en la fecha de emisión, para que las entregas tardías y las correcciones no se filtren hacia atrás (ver sección 8).
- **Horizonte de Rezago en Features:** Las variables explicativas (lags y medias móviles) se calculan con una ventana mínima de 7 días atrás para evitar fugas de información futura durante el entrenamiento e inferencia.

## 2. Idempotencia, Backfill y Manejo de Errores
- **Idempotencia:** La generación de pronósticos escribe los resultados en `data/predictions/pedido_<fecha>.csv` de forma determinista y sobreescribible.
- **Validación Estricta:** Si se detectan datos con ventas/precios negativos o esquemas alterados, `validacion.py` lanza una excepción `DatosInvalidosError` para detener el DAG sin contaminar los modelos.

## 3. Métrica de Negocio vs. Métrica Técnica
- Se contrapone el error técnico tradicional (MAE/RMSE) frente a la penalización económica por inventario:
  - Costo de Quiebre de Stock = $5.0 (Venta perdida)
  - Costo de Merma = $2.0 (Producto vencido)
- El modelo candidato es evaluado seleccionando el que minimice el Costo Total de Inventario.
- Estos costos determinan el cuantil objetivo del modelo (ver sección 7).

## 4. Promoción y Registro de Modelo Campeón
- Se registran ejecuciones en MLflow clasificando modelos. El modelo óptimo se promueve dinámicamente asignándole el alias `@Champions`, permitiendo a los DAGs de Airflow consumirlo sin modificar código.
- El criterio explícito de reemplazo y la elección del modelo de cada pedido están en la sección 9.

## 5. Infraestructura (Docker Compose)
- **MLflow 3.17.2 en servidor y cliente:** el cliente (entrenamiento y Airflow) usa la API de modelos de MLflow 3; un servidor 2.x no la soporta. Las versiones se fijan iguales en `docker/airflow/requirements.txt`.
- **Artefactos servidos por MLflow** (`--artifacts-destination` + `--serve-artifacts`): los modelos se suben y descargan a través del servidor (`mlflow-artifacts:/`), sin depender de rutas locales compartidas. Base de datos y artefactos viven en volúmenes de Docker (`mlflow_db`, `mlflow_artefactos`) ; además, cada versión que usa un pedido o que se entrena se exporta a `models/`, rastreado por DVC (sección 10).
- **`--allowed-hosts`:** MLflow 3 solo acepta `localhost` por defecto; se habilita `mlflow:5000` para las llamadas desde la red `inventarioNet`.
- **Airflow 3 con componentes separados** (`api-server`, `scheduler`, `dag-processor`) sobre Postgres y `LocalExecutor`. El comando `webserver` y `SequentialExecutor` ya no existen en Airflow 3.
- **Mismas versiones de librerías al entrenar y al predecir:** el modelo se serializa con `skops` y debe cargarse con la misma versión de scikit-learn con la que se entrenó.
- **Historia de entrenamiento acotada** (`--dias-historia`): con el dataset completo (3 M filas) y 6 GB de RAM se entrena con el último año para limitar memoria y tiempo.
- **Presupuesto de memoria (WSL ~6.6 GiB).** El primer backfill del DAG 05 (2017-07-02 a 07-17) agotó la memoria y tumbó WSL. Un `airflow backfill create` usa su propio `--max-active-runs` (10 por defecto), no el `max_active_runs=1` del DAG, y todos los runs entrenaban a la vez porque aún no había modelo vigente. Medido: **un entrenamiento usa ~1.6 GiB de pico y tarda ~40 s**; en paralelo se roban CPU y memoria y no terminan. Medidas:
  - **Pool `entrenamiento` de 1 slot** (lo crea `airflow-init` con `airflow pools set`, idempotente) para `05.entrenar_y_decidir` y `03.asegurar_modelo`: nunca corren dos entrenamientos a la vez, venga el run de un backfill, del calendario o de una ejecución manual.
  - **`mem_limit` por contenedor:** scheduler 3.5 GiB (con `LocalExecutor` las tareas corren dentro de él), api-server y dag-processor 1 GiB, MLflow 1 GiB. Si alguno se excede, Docker mata ese contenedor y no WSL entero. `AIRFLOW__CORE__PARALLELISM=4` en el scheduler.
  - **MLflow sin ejecutor de jobs** (`MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false`) y con `--workers 2`: en reposo ocupaba 2.2 GiB, casi todo de 7 procesos `huey_consumer` para jobs de GenAI/evaluación que este proyecto no usa. Queda en ~0.55 GiB.
  - Los backfills que entrenan se lanzan con `--max-active-runs 1`.

## 6. Operación de los DAGs
- **Sin datos del día, el run se omite:** si la bandeja todavía no tiene ventas de la fecha lógica, los DAGs 01 y 04 terminan con la tarea omitida (*skipped*) en lugar de procesar datos viejos como si fueran del día. Así 01 y 04 pueden estar activos aunque la bandeja contenga datos históricos. En producción esto significa que, si los datos llegan tarde, ese día no se emite pedido: conviene vigilar los runs omitidos.
- **Fechas pasadas por backfill:** `airflow backfill create --dag-id 01_ingesta_y_validacion --from-date AAAA-MM-DD --to-date AAAA-MM-DD`. El límite superior no incluye su propio día: para llegar al día X se indica X+1.
- **Evaluación honesta del seguimiento:** para medir el desempeño de un periodo pasado, el modelo `@Champions` debe estar entrenado solo con datos anteriores a ese periodo.
- **Remoto de DVC permanente** en `~/product-dev/dvc-storage/motor-inventario` (antes `/tmp/dvc-storage`, que se borra al reiniciar WSL).

## 7. Cuantil objetivo: de 0.90 a 0.71
**Decisión:** `parametros_pedido.cuantil_objetivo` pasa de `0.90` a `0.71` en `config/costos.yaml`.

### 7.1 Qué es el cuantil objetivo
El modelo de ML (`HistGradientBoostingRegressor` con pérdida cuantílica) no pronostica la venta promedio sino un cuantil de la demanda diaria: con 0.90 pronostica una cantidad que la venta real no supera el 90 % de los días. El pedido sugerido es la suma de ese pronóstico en los 7 días del horizonte. Un cuantil más alto reduce los quiebres de stock y aumenta la merma; uno más bajo hace lo contrario.

### 7.2 Justificación teórica (ratio crítico)
Con los costos de la sección 3, agregar una unidad más al pedido gana 5 si esa unidad se vende (evita un quiebre) y pierde 2 si no se vende (merma). Conviene agregarla mientras la probabilidad de venderla supere 2 / (5 + 2), es decir, hasta el cuantil:

> cuantil óptimo = costo_quiebre / (costo_quiebre + costo_merma) = 5 / (5 + 2) ≈ **0.71**

Es el resultado clásico del problema del *vendedor de periódicos* (newsvendor). Con 0.90 se piden unidades cuyo costo esperado supera su beneficio esperado.

### 7.3 Señal en producción (DAG 04)
El seguimiento al 2017-08-15 de los 14 pedidos emitidos del 26-jul al 8-ago (modelo p90 entrenado hasta el 24-jul) mostró que el pedido quedaba sistemáticamente por encima de la venta real: la merma (641 mil a 1,56 M unidades por pedido) superaba con creces a los quiebres (18 mil a 211 mil).

### 7.4 Evaluación empírica en 11 periodos
**Método** (`scripts/comparar_cuantiles.py`, sin MLflow): para cada corte mensual C entre el 24-sep-2016 y el 24-jul-2017, se entrenó un modelo por cuantil con los 365 días hasta C y se evaluaron los 14 pedidos emitidos de C+2 a C+15 contra las ventas reales, con las mismas funciones del motor (`generar_pedido`, `evaluar_pedido`). Ningún modelo vio las ventas con las que se le evalúa. Como control, el resultado p90 del último periodo coincide exactamente con el que calculó el DAG 04 en Airflow.

**Resumen (11 periodos, 154 pedidos por modelo):**

| Modelo | Costo total | vs p90 | Periodos ganados | Puesto medio | Peor periodo vs el mejor | Venta cubierta | Pedido / venta |
|---|---|---|---|---|---|---|---|
| p50 | 411 M | +7 % | 0 | 6,1 | +91 % | 92,2 % | 0,95 |
| p60 | 339 M | −12 % | 1 | 4,3 | +44 % | 94,4 % | 0,98 |
| p65 | 313 M | −19 % | 2 | 2,8 | +28 % | 95,4 % | 1,00 |
| **p71** | **296 M** | **−23 %** | **4** | **2,0** | **+12 %** | 96,4 % | 1,03 |
| p80 | 304 M | −21 % | 3 | 2,7 | +36 % | 97,8 % | 1,09 |
| p90 (anterior) | 384 M | — | 1 | 5,4 | +104 % | 98,9 % | 1,17 |
| Baseline (+15 %) | 356 M | −7 % | 0 | 4,7 | +61 % | 97,6 % | 1,11 |

*"Peor periodo vs el mejor": en el periodo que peor le fue, cuánto más costó ese modelo que el mejor modelo de ese periodo (mide el riesgo).*

**Costo de cada periodo relativo a p90:**

| Corte | p60 | p65 | p71 | p80 | Mejor del periodo |
|---|---|---|---|---|---|
| 2016-09-24 | −14 % | −28 % | −33 % | −30 % | p71 |
| 2016-10-24 | +29 % | +18 % | +10 % | +1 % | p90 |
| 2016-11-24 | −20 % | −24 % | −30 % | −22 % | p71 |
| 2016-12-24 | +4 % | −3 % | −11 % | −19 % | p80 |
| 2017-01-24 | −49 % | −51 % | −50 % | −33 % | p65 |
| 2017-02-24 | +27 % | +13 % | −1 % | −12 % | p80 |
| 2017-03-24 | −13 % | −17 % | −19 % | −18 % | p71 |
| 2017-04-24 | −3 % | −16 % | −20 % | −23 % | p80 |
| 2017-05-24 | −6 % | −17 % | −26 % | −26 % | p71 |
| 2017-06-24 | −40 % | −42 % | −39 % | −28 % | p65 |
| 2017-07-24 | −31 % | −31 % | −29 % | −21 % | p60 |

### 7.5 Conclusiones
- **p71 tiene el menor costo total** (−23 % frente a p90), el mejor puesto medio y el menor riesgo: en su peor periodo costó solo un 12 % más que la mejor opción. Coincide con el valor teórico de 7.2.
- **p90 es caro de forma sistemática:** solo ganó en 1 de 11 periodos y en su peor periodo duplicó el costo de la mejor opción.
- **Un solo periodo no basta para decidir:** con solo julio de 2017 ganaba p60, pero en otros meses p60 llega a costar un 29 % más que p90.
- **Hay periodos donde conviene un cuantil más alto** (oct-2016, dic-2016, feb-2017 y abr-2017 los ganan p80 o p90), probablemente cuando la demanda supera lo que el modelo esperaba. Un valor fijo de 0.71 cede algo en esos meses a cambio de ganar en el resto.
- **Compromiso aceptado:** el pedido pasa de cubrir el 117 % al 103 % de la venta real, y la venta cubierta baja del 98,9 % al 96,4 %.

### 7.6 Supuestos y cuándo revisar esta decisión
- La decisión depende de que los costos de `costos.yaml` (quiebre 5, merma 2) sean los reales. Si el negocio quiere un nivel de servicio mayor (por ejemplo, porque un cliente sin producto puede no volver), lo correcto es subir `quiebre_stock`, y el cuantil óptimo se recalcula con la fórmula de 7.2.
- Si cambian los costos o el horizonte, repetir el análisis con `python scripts/comparar_cuantiles.py` (unos 17 minutos para los 11 periodos).
- Mejora posible: pronosticar directamente el cuantil de la demanda semanal en lugar de sumar cuantiles diarios, que tiende a sobrestimar la semana.

### 7.7 Puesta en marcha
- `@Champions` reentrenado con datos hasta el 2017-08-15 y `cuantil_objetivo: 0.71` → versión 4 del modelo `motor_inventario` (gana HistGradientBoosting).
- En los 14 días de validación del entrenamiento, el costo de inventario bajó de 7,37 M (p90, versión 3) a 5,57 M (p71), un 24 % menos; el RMSE pasó de 442 a 312.
- El pedido sugerido del 2017-08-15, regenerado por la cadena 01 → 02 → 03 en Airflow, bajó de 6,32 M a 5,74 M unidades (−9 %).

## 8. Bandeja por entregas, disponibilidad de la información y correcciones
### 8.1 Sistema de origen simulado
`scripts/preparar_bandeja.py` (lógica en `inventario.simulacion`) convierte los CSV de Kaggle en una bandeja con **un archivo por fuente y por fecha de entrega**: `bandeja_origen/<fuente>/<fuente>_<AAAA-MM-DD>.csv`. La fecha del nombre es cuándo llegó el archivo, no la fecha de los datos que contiene. La configuración está versionada en `config/bandeja.yaml`:

| Fuente | Cómo llega | Justificación |
|---|---|---|
| ventas (`train.csv`) | Diario; las ventas de D en la entrega D | Fuente principal; sin ella no se emite pedido |
| transacciones (`transactions.csv`) | Diario, 1 día de retraso | Sistema de cajas que consolida al día siguiente |
| petróleo (`oil.csv`) | Días hábiles, 1 día de retraso | Precio de cierre publicado al día siguiente; sin mercado en fin de semana |
| feriados (`holidays_events.csv`) | Mensual: el día 1 llega el calendario del mes siguiente | El calendario oficial se publica por adelantado |
| tiendas (`stores.csv`) | Una vez, en la carga histórica | Catálogo casi estático |

- **Carga histórica inicial:** todo lo entregado hasta `fin_carga_historica` (2017-06-30) llega en un solo archivo por fuente con esa fecha. Del 2017-07-01 al 2017-08-15 hay entregas diarias.
- **Incidencias** (semilla fija, reproducibles): entregas tardías (5 % de los días de ventas, transacciones y petróleo, 1-3 días extra), correcciones (10 % de los días de ventas; 2 % de sus filas llegan con valor ×0,5-1,5 y el valor real llega 1-5 días después) y entregas corruptas (3 % de los archivos de ventas traen una venta negativa y se reenvían válidos al día siguiente). La última versión entregada de cada fila coincide con el dato de Kaggle.
- El detalle de cada incidencia simulada queda en `data/processed/bandeja_incidencias.csv`.

### 8.2 Reglas de lectura (`inventario.ingesta.leer_bandeja`)
1. **Por fecha de entrega:** solo se usan archivos entregados hasta la fecha de conocimiento (por defecto, la fecha lógica `ds`). Es la regla que garantiza el principio fundamental: un backfill al 1-ago no ve una corrección ni un dato tardío que llegó el 10-ago.
2. **Cuarentena por archivo:** cada archivo se valida por separado; si es inválido se excluye y queda registrado en el log del DAG 01 y en `meta.json` (`cuarentena`). El control final sobre los datos combinados (`validar_datos`) sigue deteniendo el run si algo inválido llegara a pasar.
3. **Correcciones:** si una fila llega varias veces, prevalece la entrega más reciente. `meta.json` registra cuántas filas fueron reemplazadas.
4. **Corte por fecha de datos:** ventas, transacciones y petróleo nunca pasan de `ds`; los feriados entregados pueden traer fechas futuras porque el calendario se conoce por adelantado.
5. **Fuentes secundarias opcionales:** si petróleo, transacciones, feriados o tiendas no han llegado, se continúa (el petróleo faltante se rellena con el último precio conocido). Si faltan las ventas del día, el run se omite.
6. **La bandeja es de solo lectura:** el código nunca la modifica y en Docker se monta con `:ro`.

`meta.json` de cada etapa guarda además la lista de archivos de la bandeja usados (`entregas`), lo que permite rastrear cada pedido hasta los datos que lo produjeron.

### 8.3 Reprocesamiento de datos corregidos
- **Los pedidos publicados no se reescriben:** reflejan lo que se sabía al emitirlos, y así se pueden reproducir.
- **Las correcciones se incorporan solas** en todo lo que se calcula después: los pedidos de los días siguientes, el seguimiento del DAG 04 (que en cada run reevalúa su ventana con la última versión de las ventas) y los reentrenamientos.
- **Reproceso explícito de una fecha pasada:** `airflow dags trigger 01_ingesta_y_validacion -c '{"ds": "2017-07-04", "conocimiento": "2017-07-06"}'` usa las entregas hasta `conocimiento` sin datos posteriores a `ds`. Las salidas se guardan con la clave `2017-07-04__conocimiento_2017-07-06` (por ejemplo, `pedido_2017-07-04__conocimiento_2017-07-06.csv`), así que el pedido publicado sigue intacto y se puede comparar con el reprocesado.

### 8.4 Alternativas descartadas
- **Filtrar por la fecha de los datos** (diseño anterior): un backfill usaba correcciones y entregas tardías que aún no existían al emitir, una fuga de información.
- **Detener el run ante un archivo inválido:** como cada run lee todas las entregas anteriores, un solo archivo corrupto habría hecho fallar todos los runs futuros. La cuarentena aísla el archivo y deja continuar.
- **Sobrescribir el pedido publicado al llegar una corrección:** se perdería la reproducibilidad de lo que realmente se entregó al negocio.
- **Esperar las ventas tardías con un sensor:** bloquearía el slot del DAG por un tiempo indefinido; se prefirió omitir el run y que el día siguiente use los datos que hayan llegado.

## 9. Entrenamiento y criterio de reemplazo del modelo vigente
### 9.1 Cuándo se entrena (DAG `05_entrenamiento_modelo`)
- Corre a diario a las 05:00 (antes del DAG 01) y entrena solo si **no hay modelo vigente a `ds`** o si `ds` es el **día de reentrenamiento** (`entrenamiento.dia_reentrenamiento` en `params.yaml`, domingo por defecto). Los demás días se corta con un *short-circuit*.
- Usa solo lo entregado en la bandeja hasta `ds` y los últimos `entrenamiento.dias_historia` días (365). Si las ventas de `ds` aún no llegaron, el entrenamiento se omite, como el resto del flujo.
- **Arranque sin pasos manuales:** el DAG 03 tiene la tarea `asegurar_modelo`; si no existe un modelo vigente a `ds` (clon nuevo, o backfill anterior a todo entrenamiento), entrena uno en ese momento con los datos hasta `ds`.
- **Alternativas descartadas:** reentrenar a diario (más costo y cambios de modelo por ruido sin beneficio claro en una semana de datos nuevos) o entrenar solo a mano (rompe la operación sin intervención manual).

### 9.2 Criterio explícito de reemplazo
1. Baseline y ML se entrenan y validan con el **mismo esquema temporal**: los últimos `evaluacion.dias_historicos_comparar` días (14) quedan fuera del entrenamiento. El de menor costo total de inventario es el **candidato**.
2. El **modelo vigente** se evalúa en **esa misma ventana**. Ninguno de los dos la usó para entrenar: el vigente se entrenó con datos anteriores.
3. El candidato **reemplaza al vigente solo si su costo es al menos `promocion.mejora_minima` (2 %) menor**. Si no hay vigente, o el vigente ya no se puede evaluar con las features actuales, el candidato se promueve.
4. Cada candidato se registra en MLflow, aunque no se promueva, con las etiquetas `decision` (`promovido` / `no_promovido`), `motivo`, `fecha_corte` y la versión vigente contra la que se comparó. Queda la auditoría de todas las decisiones.

**¿Por qué un umbral del 2 %?** Dos entrenamientos con datos casi iguales dan costos que difieren por ruido. Sin umbral, el modelo cambiaría cada semana sin una mejora real, y cada cambio dificulta comparar el seguimiento entre semanas. El 2 % es pequeño frente a las diferencias que importan: el cambio de cuantil de la sección 7 redujo el costo un 23 %.

### 9.3 Qué modelo usa cada pedido (selección por fecha)
- El DAG 03 no toma "el `@Champions` de ahora", sino el **modelo vigente a `ds`**: la última versión promovida con `fecha_corte <= ds`. Así un pedido usa el modelo que existía al emitirlo, aunque los backfills de los DAGs 05 y 01 se ejecuten en otro orden, y reprocesar una fecha da el mismo resultado.
- El pedido guarda la columna `modelo_version`, que lo liga a la versión exacta que lo produjo.
- `@Champions` apunta siempre al último modelo promovido y **nunca retrocede**: si un backfill entrena un modelo para una fecha antigua, el alias no se mueve hacia él.
- Las versiones registradas antes de este criterio (sin etiqueta `decision`) no se consideran vigentes.

## 10. Versionado con DVC y reproducción de pedidos publicados
### 10.1 Qué se versiona y quién lo hace
| Ruta | Contenido | Lo versiona |
|---|---|---|
| `data/raw_kaggle/` | CSV originales de Kaggle (entrada) | a mano, una vez |
| `data/raw/<clave>/` | foto de lo entregado en la bandeja hasta `ds` (DAG 01) | DAG 03 |
| `data/processed/<clave>/` | datos validados y `features.parquet` (DAGs 01 y 02) | DAG 03 |
| `data/predictions/` | `pedido_<clave>.csv`, su manifiesto `.json` y `seguimiento_<ds>.csv` | DAGs 03 y 04 |
| `models/<nombre>/v<versión>/` | copia de cada versión del Model Registry que se usa o se entrena | DAGs 03 y 05 |

- La tarea final `versionar` de los DAGs 03, 04 y 05 ejecuta `dvc add --no-relink` sobre esas rutas y `dvc push` al remoto. Corre en el pool `versionado` (1 slot), porque DVC bloquea el repositorio y dos operaciones a la vez fallarían.
- `--no-relink`: sin él, después de guardar en la caché DVC borra y recrea desde ella los archivos rastreados. Los DAGs 01 y 02 siguen escribiendo mientras otro DAG versiona, y un reproceso de esa misma fecha podría quedar pisado por la versión anterior.
- Cada fecha escribe en su propio directorio (`<clave>` = `ds`, o `<ds>__conocimiento_<fecha>` en un reproceso), así que una versión de DVC posterior **contiene todas las publicaciones anteriores sin cambios**: no hace falta un commit de DVC por pedido para poder recuperarlo.
- **El pedido se genera con la copia exportada del modelo** (`models/…`), no con la del servidor MLflow. Lo que DVC versiona es exactamente lo que produjo el pedido, y reproducirlo no depende de que el servidor ni sus volúmenes sigan existiendo.

### 10.2 Manifiesto de cada pedido
`data/predictions/pedido_<clave>.json` registra el commit del código (y si `src/`, `dags/`, `config/` o `params.yaml` tenían cambios sin confirmar), la versión del modelo y las huellas MD5 del modelo exportado, de los datos crudos, de las features y del propio pedido. No incluye la hora de generación: re-ejecutar la tarea produce el mismo archivo (idempotencia).

### 10.3 DVC dentro de los contenedores
- El repo se monta en `/opt/airflow` (punteros `.dvc`, `.dvc/` con la caché, `models/`) y `.git` **en solo lectura**, solo para leer el commit.
- DVC corre con `core.no_scm = true` (configuración global de la imagen, `docker/airflow/dvc-global/config`): **nunca modifica el índice ni el historial de git del host**.
- El remoto se monta en la misma ruta que tiene en `.dvc/config`, tomada de `DVC_REMOTO` en `.env`. Si no está montado (un clon nuevo, por ejemplo), `dvc push` crearía el directorio dentro del contenedor y los datos se perderían; por eso `versionar` lo verifica antes, deja los datos en la caché local y lo informa en el log sin fallar (situación esperada).

### 10.4 Commit de los punteros en git: paso humano
Los `.dvc` actualizados quedan como cambios en el árbol de trabajo. Se confirman en una rama con Pull Request, como el resto del trabajo (`CLAUDE.md`).
- **Alternativa descartada:** que Airflow haga commit y push a git. Escribiría en `main` sin revisión, o necesitaría credenciales de GitHub dentro de los contenedores.

### 10.5 Reproducir un pedido pasado
```bash
git checkout <commit con los punteros .dvc de esa publicación>
dvc checkout            # o dvc pull, si los datos no están en la caché local
python scripts/reproducir_pedido.py --clave 2017-07-31
```
`inventario.reproduccion` comprueba que las features, los datos crudos y el modelo coincidan con las huellas del manifiesto, vuelve a pronosticar y compara el pedido regenerado byte a byte con el publicado. No usa MLflow ni la bandeja.
