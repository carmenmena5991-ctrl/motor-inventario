# Evidencia de funcionamiento

Ejecución del periodo completo de la bandeja simulada, `ds` del 2017-07-01 al 2017-08-15, sobre la infraestructura de Docker Compose:

```bash
make periodo DESDE=2017-07-01 HASTA=2017-08-15 REPROCESO=todos
```

- Código: commit `39f15a7`. Los 43 manifiestos registran `"commit": "39f15a7…"` y `"modificado": false`.
- Datos publicados: commit `6a3ce1a`, con los punteros DVC.
- Duración: 32 min 50 s, de a un run por vez.

## 1. Ejecuciones en Airflow

```text
2026-11-01 04:23:40 Backfill de 05_entrenamiento_modelo: ds 2017-07-01 a 2017-08-15
2026-11-01 04:33:37 Backfill de 01_ingesta_y_validacion: ds 2017-07-01 a 2017-08-15
2026-11-01 04:48:43 Backfill de 04_seguimiento_desempeno: ds 2017-07-01 a 2017-08-15
Periodo 2017-07-01 a 2017-08-15 terminado: 05_entrenamiento_modelo, 01_ingesta_y_validacion, 04_seguimiento_desempeno
```

| DAG | Runs | Estado | Tareas fallidas |
|---|---:|---|---:|
| 05_entrenamiento_modelo | 46 | success | 0 |
| 01_ingesta_y_validacion | 46 | success | 0 |
| 02_preparacion_y_features | 43 | success | 0 |
| 03_pronostico_y_pedido | 43 | success | 0 |
| 04_seguimiento_desempeno | 46 | success | 0 |

### Situaciones de operación que se presentaron

| Situación | Fecha | Qué hizo el flujo |
|---|---|---|
| Ventas tardías | ds 2017-07-05 | `ingestar` omitido: *"La bandeja tiene ventas hasta 2017-07-04; aún no hay datos de 2017-07-05"*. No se emitió pedido; el 02 y el 03 no se dispararon |
| Ventas tardías | ds 2017-07-28 | Igual: *"…ventas hasta 2017-07-27; aún no hay datos de 2017-07-28"* |
| Ventas tardías | ds 2017-08-07 | Igual: *"…ventas hasta 2017-08-06; aún no hay datos de 2017-08-07"* |
| Archivo inválido | entrega `ventas_2017-07-05.csv` | Cuarentena: *"[ventas] 1 valores negativos en 'sales'"*. Se usó el reenvío válido |
| Archivo inválido | entrega `ventas_2017-08-07.csv` | Cuarentena, por el mismo motivo |
| Correcciones | todo el periodo | Prevalece la última entrega. Al 2017-08-15 se habían reemplazado 154 filas de ventas |
| Días sin reentrenamiento | 39 de 46 | `hay_que_entrenar` corta el DAG 05 (*short-circuit*) sin costo |

### Versionado (tarea `versionar`, DAG 03)

```text
Versionado con DVC: ResultadoVersionado(rutas=('data/raw', 'data/processed', 'data/predictions', 'models'),
  subido=True, detalle='remoto local_storage (/home/camena/product-dev/dvc-storage/motor-inventario)')
$ dvc status && dvc status -c
Data and pipelines are up to date.
Cache and remote 'local_storage' are in sync.
```

## 2. Comparación de modelos (MLflow)

Cada entrenamiento compara el baseline con HGB en la misma ventana temporal: los últimos 14 días quedan fuera del entrenamiento. El candidato de menor costo se compara después con el modelo vigente **en esa misma ventana**.

| Fecha de corte | Modelo | RMSE | MAE | Costo total de inventario |
|---|---|---:|---:|---:|
| 2017-07-02 | baseline_promedio_movil | 530.2 | 161.8 | 10,728,587 |
| 2017-07-02 | hist_gradient_boosting | 235.5 | 68.2 | 4,476,412 |
| 2017-07-09 | baseline_promedio_movil | 490.3 | 146.8 | 10,292,861 |
| 2017-07-09 | hist_gradient_boosting | 243.5 | 72.9 | 4,616,634 |
| 2017-07-16 | baseline_promedio_movil | 440.1 | 131.5 | 8,640,706 |
| 2017-07-16 | hist_gradient_boosting | 259.8 | 73.3 | 4,614,713 |
| 2017-07-23 | baseline_promedio_movil | 445.6 | 138.5 | 8,691,291 |
| 2017-07-23 | hist_gradient_boosting | 242.6 | 67.8 | 4,541,512 |
| 2017-07-30 | baseline_promedio_movil | 433.6 | 138.2 | 8,351,865 |
| 2017-07-30 | hist_gradient_boosting | 225.1 | 65.0 | 4,545,015 |
| 2017-08-06 | baseline_promedio_movil | 404.8 | 126.7 | 8,988,107 |
| 2017-08-06 | hist_gradient_boosting | 263.8 | 76.8 | 5,804,480 |
| 2017-08-13 | baseline_promedio_movil | 398.2 | 127.7 | 8,288,274 |
| 2017-08-13 | hist_gradient_boosting | 319.2 | 94.7 | 5,934,833 |

### Decisiones del Model Registry

| Versión | Fecha de corte | Costo del candidato | Costo del vigente (v5), misma ventana | Decisión | Motivo |
|---|---|---:|---:|---|---|
| v5 | 2017-07-01 | 4,839,658 | — | **promovido** | no hay modelo vigente |
| v9 | 2017-07-02 | 4,476,412 | 4,515,259 | no promovido | mejora de 0.9 %, menor que el mínimo exigido de 2.0 % |
| v10 | 2017-07-09 | 4,616,634 | 4,644,324 | no promovido | mejora de 0.6 % |
| v11 | 2017-07-16 | 4,614,713 | 4,614,944 | no promovido | mejora de 0.0 % |
| v12 | 2017-07-23 | 4,541,512 | 4,570,200 | no promovido | mejora de 0.6 % |
| v13 | 2017-07-30 | 4,545,015 | 4,519,935 | no promovido | mejora de −0.6 % |
| v14 | 2017-08-06 | 5,804,480 | 5,700,284 | no promovido | mejora de −1.8 % |
| v15 | 2017-08-13 | 5,934,833 | 5,980,119 | no promovido | mejora de 0.8 % |

- `@Champions` sigue en v5 y los 43 pedidos del periodo usan v5, según su manifiesto.
- Las versiones v6 a v8 son de una corrida anterior de las mismas fechas, con los mismos costos que v9 a v11. Volver a entrenar con los mismos datos registra otra versión, pero no la promueve (`DECISIONES.md`, sección 9).
- Todas las versiones están exportadas en `models/motor_inventario/v<N>/` y versionadas con DVC.

## 3. Seguimiento del desempeño (DAG 04)

Cada pedido, con su última evaluación, cuando ya se conocían las ventas de todo su horizonte de 7 días. Costo real = quiebre × 5 + merma × 2.

| Emisión | Venta real | Pedido | Quiebre (u) | Merma (u) | MAE | Costo real |
|---|---:|---:|---:|---:|---:|---:|
| 2017-07-01 | 6,402,595 | 6,706,857 | 91,279 | 395,541 | 272.9 | 1,247,474 |
| 2017-07-02 | 6,207,049 | 6,628,161 | 77,465 | 498,577 | 322.9 | 1,384,481 |
| 2017-07-03 | 6,086,065 | 6,472,164 | 99,162 | 485,262 | 327.6 | 1,466,336 |
| 2017-07-04 | 5,984,240 | 6,358,770 | 107,205 | 481,735 | 330.2 | 1,499,494 |
| 2017-07-06 | 5,818,118 | 6,266,468 | 103,366 | 551,716 | 367.3 | 1,620,259 |
| 2017-07-07 | 5,777,815 | 6,224,084 | 102,213 | 548,482 | 364.8 | 1,608,029 |
| 2017-07-08 | 5,748,545 | 6,226,443 | 95,020 | 572,918 | 374.5 | 1,620,933 |
| 2017-07-09 | 5,748,515 | 6,154,919 | 99,581 | 505,984 | 339.5 | 1,509,873 |
| 2017-07-10 | 5,762,682 | 6,214,751 | 91,910 | 543,979 | 356.5 | 1,547,509 |
| 2017-07-11 | 5,762,281 | 6,235,067 | 87,905 | 560,691 | 363.6 | 1,560,905 |
| 2017-07-12 | 5,781,460 | 6,229,822 | 91,831 | 540,193 | 354.3 | 1,539,540 |
| 2017-07-13 | 5,840,097 | 6,212,264 | 105,198 | 477,366 | 326.6 | 1,480,724 |
| 2017-07-14 | 5,857,025 | 6,190,763 | 110,550 | 444,288 | 311.0 | 1,441,324 |
| 2017-07-15 | 5,820,850 | 6,098,013 | 108,774 | 385,937 | 277.3 | 1,315,747 |
| 2017-07-16 | 5,744,335 | 6,066,283 | 97,620 | 419,568 | 289.9 | 1,327,234 |
| 2017-07-17 | 5,742,574 | 6,031,323 | 95,308 | 384,057 | 268.7 | 1,244,653 |
| 2017-07-18 | 5,726,022 | 5,980,561 | 93,173 | 347,712 | 247.1 | 1,161,289 |
| 2017-07-19 | 5,698,696 | 5,947,568 | 96,325 | 345,197 | 247.5 | 1,172,017 |
| 2017-07-20 | 5,670,258 | 5,951,499 | 95,460 | 376,701 | 264.6 | 1,230,704 |
| 2017-07-21 | 5,722,939 | 5,951,989 | 115,408 | 344,458 | 257.8 | 1,265,958 |
| 2017-07-22 | 5,822,348 | 5,960,882 | 157,932 | 296,466 | 254.7 | 1,382,595 |
| 2017-07-23 | 5,921,812 | 5,937,154 | 238,989 | 254,332 | 276.6 | 1,703,611 |
| 2017-07-24 | 5,991,104 | 5,987,952 | 263,169 | 260,017 | 293.4 | 1,835,881 |
| 2017-07-25 | 6,266,051 | 6,194,899 | 327,700 | 256,548 | 327.6 | 2,151,596 |
| 2017-07-26 | 6,490,109 | 6,364,190 | 385,568 | 259,649 | 361.8 | 2,447,140 |
| 2017-07-27 | 6,558,328 | 6,518,856 | 344,888 | 305,415 | 364.7 | 2,335,269 |
| 2017-07-29 | 6,484,387 | 5,964,455 | 607,698 | 87,766 | 390.1 | 3,214,020 |
| 2017-07-30 | 6,410,194 | 6,042,306 | 488,787 | 120,899 | 341.9 | 2,685,734 |
| 2017-07-31 | 6,321,802 | 6,834,485 | 111,253 | 623,936 | 412.2 | 1,804,136 |
| 2017-08-01 | 6,051,040 | 6,704,326 | 71,386 | 724,672 | 446.4 | 1,806,276 |
| 2017-08-02 | 5,820,468 | 6,585,049 | 53,411 | 817,992 | 488.7 | 1,903,041 |
| 2017-08-03 | 5,743,786 | 6,445,023 | 63,958 | 765,194 | 465.0 | 1,850,178 |
| 2017-08-04 | 5,742,384 | 6,460,449 | 72,473 | 790,537 | 484.0 | 1,943,438 |
| 2017-08-05 | 5,569,321 | 6,375,980 | 76,680 | 883,338 | 538.4 | 2,150,075 |
| 2017-08-06 | 5,385,402 | 6,314,665 | 64,621 | 993,884 | 593.6 | 2,310,875 |
| 2017-08-08 | 5,393,755 | 6,212,326 | 67,183 | 885,754 | 534.4 | 2,107,424 |
| **Total** | **212,874,453** | **225,050,766** | **5,360,449** | **17,536,762** | **356.6** | **61,875,770** |

El pedido cubrió el 105.7 % de la venta real. Los quiebres fueron el 2.5 % de la venta y la merma el 8.2 %.

**Correcciones incorporadas al seguimiento.** El DAG 04 reevalúa su ventana en cada run con la última versión de las ventas. En 11 pedidos la venta real cambió entre evaluaciones:
- en el 2017-07-08 y el 2017-07-09, unas 1,759 unidades por correcciones;
- en los emitidos del 2017-07-21 al 2017-07-23, unas 835,100 unidades, porque las ventas tardías del 2017-07-28 todavía no habían llegado cuando se evaluaron por primera vez.

Los pedidos publicados no cambian: reflejan lo que se sabía al emitirlos.

**Lectura del seguimiento.**
- Hasta mediados de julio el pedido sobrestima un 5 % a 8 %, y el costo lo domina la merma, como corresponde al cuantil 0.71.
- Entre el 23 y el 30 de julio la demanda alcanza o supera al pedido y pasan a dominar los quiebres.
- Desde el 31 de julio vuelve a dominar la merma, y el MAE crece de 412 a 594.

v5 está entrenado con datos hasta el 2017-07-01 y no fue reemplazado, porque en la ventana de validación de cada entrenamiento los candidatos no lo mejoraban un 2 %. El seguimiento real sí muestra deterioro en agosto. Esta señal no la usa hoy el criterio de reemplazo (posible mejora: considerar también el costo real del seguimiento).

## 4. Reproducción de pedidos publicados con DVC

En una carpeta vacía, a partir del commit de la publicación:

```text
$ git clone <repo> prueba-reproduccion && cd prueba-reproduccion && git checkout 6a3ce1a
6a3ce1a data: publicación del periodo 2017-07-01 a 2017-08-15
$ ls data/predictions models
ls: cannot access 'data/predictions': No such file or directory
ls: cannot access 'models': No such file or directory
$ dvc pull
real    0m3.027s
  35M   data/raw
  97M   data/processed
  4.2M  data/predictions
  23M   models

$ python scripts/reproducir_pedido.py --clave 2017-07-01
Pedido 2017-07-01: modelo v5, código 39f15a7b8339aae8f09860571344d1da1c923ced
  features      coincide con el manifiesto
  datos_crudos  coincide con el manifiesto
  modelo        coincide con el manifiesto
  md5 publicado   d5678744682032f6fe9dda57f8940ed1
  md5 reproducido d5678744682032f6fe9dda57f8940ed1  (1782 filas)
Resultado: IDÉNTICO

$ python scripts/reproducir_pedido.py --clave 2017-07-21
Pedido 2017-07-21: modelo v5, código 39f15a7b8339aae8f09860571344d1da1c923ced
  features      coincide con el manifiesto
  datos_crudos  coincide con el manifiesto
  modelo        coincide con el manifiesto
  md5 publicado   deef927ff595f148beac05daf803d21c
  md5 reproducido deef927ff595f148beac05daf803d21c  (1782 filas)
Resultado: IDÉNTICO

$ python scripts/reproducir_pedido.py --clave 2017-07-31
Pedido 2017-07-31: modelo v5, código 39f15a7b8339aae8f09860571344d1da1c923ced
  features      coincide con el manifiesto
  datos_crudos  coincide con el manifiesto
  modelo        coincide con el manifiesto
  md5 publicado   06c2daad75b79f9dfecdffd56ff2bf00
  md5 reproducido 06c2daad75b79f9dfecdffd56ff2bf00  (1782 filas)
Resultado: IDÉNTICO

$ python scripts/reproducir_pedido.py --clave 2017-08-15
Pedido 2017-08-15: modelo v5, código 39f15a7b8339aae8f09860571344d1da1c923ced
  features      coincide con el manifiesto
  datos_crudos  coincide con el manifiesto
  modelo        coincide con el manifiesto
  md5 publicado   64cc7091bcbc6760a0bb88c82c3ef136
  md5 reproducido 64cc7091bcbc6760a0bb88c82c3ef136  (1782 filas)
Resultado: IDÉNTICO

$ for c in <las 43 claves>; do python scripts/reproducir_pedido.py --clave $c; done
43 de 43 pedidos reproducidos idénticos
```

**Control negativo:** con una cantidad del pedido publicado del 2017-07-31 cambiada a mano, el script lo detecta.

```text
  md5 publicado   47372923d6210f591d70475490b856c1
  md5 reproducido 06c2daad75b79f9dfecdffd56ff2bf00  (1782 filas)
Resultado: DISTINTO
```

**Idempotencia:** al volver a ejecutar el DAG 03 del 2017-07-31, el pedido, el manifiesto y `data/predictions.dvc` quedan iguales byte a byte. Además, el pedido del 2017-07-31 tiene el mismo MD5 (`06c2daad…`) en dos publicaciones hechas con código distinto (`600d5ca` y `39f15a7`).

## 5. Capturas de la interfaz

Las capturas de la UI de Airflow (*Grid* de los DAGs) y de MLflow (comparación de runs y Model Registry) van en `docs/img/`.
