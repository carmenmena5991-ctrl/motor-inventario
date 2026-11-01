# Evidencia de funcionamiento

Ejecución del periodo completo de la bandeja simulada, `ds` del 2017-07-01 al 2017-08-15, con el pronóstico diario por cuantiles (`DECISIONES.md`, sección 11). Corrió en el orden de `make periodo`: DAG 05, luego 01 → 02 → 03, luego 04, con backfills de a un run por vez.

- Código: commit `bd89c6a`. Los 43 manifiestos registran `"commit": "bd89c6a…"` y `"modificado": false`.
- Datos publicados: commit `b6561db`, con los punteros DVC.
- Duración: unos 44 minutos.

| Etapa | Inicio (UTC) |
|---|---|
| DAG 05 | 16:11 |
| DAG 01 → 02 → 03 | 16:27 |
| DAG 04 | 16:44 |
| Fin | 16:55 |

## 1. Ejecuciones en Airflow

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
| Sin modelo vigente | ds 2017-07-01 | El DAG 05 entrena y promueve el primer modelo (v24) |
| Días sin reentrenamiento | 38 de 46 | `hay_que_entrenar` corta el DAG 05 (*short-circuit*) sin costo |

### Versionado (tarea `versionar`)

```text
$ dvc status && dvc status -c
Data and pipelines are up to date.
Cache and remote 'local_storage' are in sync.
```

## 2. Comparación de modelos (MLflow)

Cada entrenamiento compara el baseline con HGB en la misma ventana temporal: los últimos 14 días quedan fuera del entrenamiento. El candidato de menor costo se compara después con el vigente **en esa misma ventana**. El costo se calcula con el cuantil objetivo, p71.

| Fecha de corte | Baseline: RMSE / MAE / costo | HGB: RMSE / MAE / costo | Vigente v24, misma ventana | Decisión |
|---|---|---|---:|---|
| 2017-07-01 | 507.8 / 154.8 / 9,597,618 | 284.4 / 77.5 / 4,810,618 | — | **promovido** (v24): no había vigente |
| 2017-07-02 | 530.2 / 161.8 / 10,728,587 | 234.9 / 67.9 / 4,450,921 | 4,480,800 | v25 no promovido (mejora 0.7 %) |
| 2017-07-09 | 490.3 / 146.8 / 10,292,861 | 242.8 / 72.4 / 4,580,239 | 4,600,756 | v26 no promovido (0.4 %) |
| 2017-07-16 | 440.1 / 131.5 / 8,640,706 | 259.6 / 73.1 / 4,592,762 | 4,575,644 | v27 no promovido (−0.4 %) |
| 2017-07-23 | 445.6 / 138.5 / 8,691,291 | 242.3 / 67.1 / 4,498,332 | 4,533,972 | v28 no promovido (0.8 %) |
| 2017-07-30 | 433.6 / 138.2 / 8,351,865 | 224.6 / 64.6 / 4,516,918 | 4,488,191 | v29 no promovido (−0.6 %) |
| 2017-08-06 | 404.8 / 126.7 / 8,988,107 | 263.5 / 76.5 / 5,772,074 | 5,676,427 | v30 no promovido (−1.7 %) |
| 2017-08-13 | 398.2 / 127.7 / 8,288,274 | 318.6 / 94.3 / 5,902,127 | 5,959,426 | v31 no promovido (1.0 %) |

- El modelo de ML reduce el costo del baseline entre un 29 % y un 59 % en cada corte.
- Ningún candidato mejoró al vigente por el 2 % exigido, así que `@Champions` sigue en v24 y los 43 pedidos usan v24.
- Las 8 versiones tienen la etiqueta `cuantiles: p10,p50,p71,p90` y están exportadas en `models/motor_inventario/v<N>/`, versionadas con DVC.

### Calibración de los cuantiles en la validación

Fracción de la venta real por debajo de cada cuantil. Un cuantil bien calibrado deja por debajo su propio porcentaje.

| Fecha de corte | p10 | p50 | p71 | p90 | Dentro de p10–p90 | Ancho medio p10–p90 |
|---|---:|---:|---:|---:|---:|---:|
| 2017-07-01 | 14.7 % | 59.8 % | 77.8 % | 92.7 % | 92.3 % | 540 |
| 2017-07-02 | 14.5 % | 57.5 % | 76.7 % | 92.1 % | 91.8 % | 550 |
| 2017-07-09 | 14.5 % | 58.2 % | 77.6 % | 92.3 % | 92.0 % | 576 |
| 2017-07-16 | 14.6 % | 57.9 % | 77.4 % | 92.3 % | 91.9 % | 552 |
| 2017-07-23 | 14.9 % | 56.1 % | 75.4 % | 92.0 % | 91.7 % | 527 |
| 2017-07-30 | 15.0 % | 52.6 % | 72.2 % | 89.7 % | 89.6 % | 515 |
| 2017-08-06 | 14.8 % | 52.9 % | 71.8 % | 89.2 % | 89.0 % | 555 |
| 2017-08-13 | 14.8 % | 59.7 % | 76.6 % | 91.2 % | 90.9 % | 579 |

### Versiones invalidadas

v16 a v23 se entrenaron con un `params.yaml` desactualizado dentro del contenedor (montaje de archivo suelto, `DECISIONES.md` 10.3) y quedaron con un solo cuantil. En el registro llevan `decision: invalidado`, la decisión original en `decision_original` y el motivo. No se borraron, para que quede la auditoría.

## 3. Pronóstico diario publicado

`data/predictions/pronostico_2017-07-31.csv`: 12,474 filas (7 días × 1,782 series).

```text
      date  store_nbr     family  p10      p50      p71       p90  modelo_version
2017-08-01          1 AUTOMOTIVE  0.0 5.572068 7.631156 11.516568              24
2017-08-01          1  BABY CARE  0.0 0.000005 0.000072  0.002478              24
2017-08-01          1     BEAUTY  0.0 4.263150 5.367530  8.777248              24
```

## 4. Seguimiento del desempeño (DAG 04)

Cada pedido, con su última evaluación, cuando ya se conocían las ventas de todo su horizonte de 7 días. Costo real = quiebre × 5 + merma × 2.

Las columnas pXX dicen qué fracción de las ventas reales diarias quedó por debajo de cada cuantil del pronóstico publicado.

| Emisión | Venta real | Pedido | Quiebre (u) | Merma (u) | MAE | Costo real | p10 | p50 | p71 | p90 | p10–p90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2017-07-01 | 6,402,595 | 6,705,255 | 90,592 | 393,251 | 271.2 | 1,239,460 | 14.2 % | 49.3 % | 70.9 % | 89.1 % | 88.9 % |
| 2017-07-02 | 6,207,049 | 6,627,645 | 76,727 | 497,323 | 321.8 | 1,378,280 | 14.3 % | 50.8 % | 72.6 % | 90.1 % | 89.9 % |
| 2017-07-03 | 6,086,065 | 6,471,536 | 98,207 | 483,679 | 326.2 | 1,458,395 | 14.5 % | 50.2 % | 72.2 % | 89.9 % | 89.7 % |
| 2017-07-04 | 5,984,240 | 6,358,129 | 106,357 | 480,246 | 328.8 | 1,492,275 | 14.6 % | 50.4 % | 72.4 % | 90.3 % | 90.1 % |
| 2017-07-06 | 5,818,118 | 6,265,575 | 102,263 | 549,720 | 365.5 | 1,610,754 | 14.7 % | 52.6 % | 74.2 % | 91.1 % | 90.9 % |
| 2017-07-07 | 5,777,815 | 6,223,501 | 101,496 | 547,181 | 363.7 | 1,601,841 | 14.5 % | 53.1 % | 74.0 % | 91.1 % | 90.9 % |
| 2017-07-08 | 5,748,545 | 6,225,105 | 94,276 | 570,836 | 372.9 | 1,613,053 | 14.6 % | 54.0 % | 74.2 % | 91.1 % | 90.9 % |
| 2017-07-09 | 5,748,515 | 6,155,314 | 98,560 | 505,359 | 338.5 | 1,503,517 | 14.7 % | 53.1 % | 73.3 % | 91.0 % | 90.7 % |
| 2017-07-10 | 5,762,682 | 6,212,821 | 91,763 | 541,902 | 355.2 | 1,542,620 | 14.6 % | 54.9 % | 74.3 % | 91.2 % | 91.0 % |
| 2017-07-11 | 5,762,281 | 6,233,270 | 87,297 | 558,286 | 361.9 | 1,553,059 | 14.7 % | 56.0 % | 74.7 % | 91.3 % | 91.1 % |
| 2017-07-12 | 5,781,460 | 6,229,123 | 90,964 | 538,627 | 352.9 | 1,532,073 | 14.9 % | 56.5 % | 74.7 % | 91.1 % | 91.0 % |
| 2017-07-13 | 5,840,097 | 6,211,366 | 104,367 | 475,636 | 325.1 | 1,473,105 | 14.8 % | 54.9 % | 73.1 % | 90.5 % | 90.3 % |
| 2017-07-14 | 5,857,025 | 6,189,393 | 109,838 | 442,206 | 309.4 | 1,433,603 | 14.9 % | 54.7 % | 72.8 % | 90.3 % | 90.1 % |
| 2017-07-15 | 5,820,850 | 6,097,765 | 108,288 | 385,203 | 276.6 | 1,311,843 | 15.0 % | 53.5 % | 72.2 % | 90.0 % | 89.8 % |
| 2017-07-16 | 5,744,335 | 6,066,425 | 97,274 | 419,364 | 289.6 | 1,325,096 | 15.0 % | 54.0 % | 72.0 % | 89.8 % | 89.6 % |
| 2017-07-17 | 5,742,574 | 6,032,605 | 94,441 | 384,472 | 268.4 | 1,241,152 | 15.1 % | 52.7 % | 71.0 % | 89.2 % | 89.0 % |
| 2017-07-18 | 5,726,022 | 5,981,643 | 92,289 | 347,910 | 246.7 | 1,157,267 | 15.0 % | 51.5 % | 70.1 % | 88.9 % | 88.7 % |
| 2017-07-19 | 5,698,696 | 5,948,705 | 95,517 | 345,526 | 247.2 | 1,168,637 | 15.0 % | 51.0 % | 69.7 % | 88.7 % | 88.4 % |
| 2017-07-20 | 5,670,258 | 5,952,329 | 94,130 | 376,201 | 263.6 | 1,223,053 | 15.0 % | 51.6 % | 70.7 % | 89.0 % | 88.8 % |
| 2017-07-21 | 5,722,939 | 5,952,725 | 113,763 | 343,549 | 256.3 | 1,255,912 | 15.0 % | 50.5 % | 69.8 % | 88.4 % | 88.1 % |
| 2017-07-22 | 5,822,348 | 5,961,610 | 156,144 | 295,406 | 253.1 | 1,371,532 | 15.1 % | 49.4 % | 68.8 % | 87.4 % | 87.1 % |
| 2017-07-23 | 5,921,812 | 5,938,062 | 236,833 | 253,083 | 274.7 | 1,690,330 | 15.1 % | 48.6 % | 68.3 % | 86.9 % | 86.6 % |
| 2017-07-24 | 5,991,104 | 5,988,635 | 261,333 | 258,864 | 291.7 | 1,824,392 | 15.1 % | 49.5 % | 69.0 % | 87.3 % | 87.0 % |
| 2017-07-25 | 6,266,051 | 6,194,152 | 327,281 | 255,383 | 326.7 | 2,147,172 | 15.0 % | 49.1 % | 68.6 % | 86.8 % | 86.5 % |
| 2017-07-26 | 6,490,109 | 6,362,126 | 385,708 | 257,724 | 360.8 | 2,443,987 | 14.8 % | 47.8 % | 67.5 % | 86.1 % | 85.8 % |
| 2017-07-27 | 6,558,328 | 6,516,556 | 344,737 | 302,965 | 363.2 | 2,329,616 | 14.7 % | 49.0 % | 68.2 % | 86.3 % | 86.1 % |
| 2017-07-29 | 6,484,387 | 5,962,717 | 608,338 | 86,668 | 389.8 | 3,215,024 | 14.6 % | 48.7 % | 68.5 % | 87.9 % | 87.6 % |
| 2017-07-30 | 6,410,194 | 6,039,965 | 490,101 | 119,872 | 342.1 | 2,690,247 | 14.6 % | 49.9 % | 69.6 % | 88.4 % | 88.2 % |
| 2017-07-31 | 6,321,802 | 6,832,428 | 111,218 | 621,845 | 411.1 | 1,799,782 | 14.7 % | 54.3 % | 72.7 % | 89.4 % | 89.2 % |
| 2017-08-01 | 6,051,040 | 6,703,905 | 70,655 | 723,520 | 445.3 | 1,800,316 | 14.8 % | 55.7 % | 73.9 % | 90.1 % | 89.9 % |
| 2017-08-02 | 5,820,468 | 6,586,264 | 52,849 | 818,645 | 488.7 | 1,901,537 | 14.8 % | 57.4 % | 75.3 % | 91.0 % | 90.7 % |
| 2017-08-03 | 5,743,786 | 6,446,958 | 63,047 | 766,218 | 465.0 | 1,847,670 | 14.9 % | 55.9 % | 73.8 % | 90.2 % | 89.8 % |
| 2017-08-04 | 5,742,384 | 6,463,000 | 71,759 | 792,375 | 484.6 | 1,943,547 | 14.7 % | 56.0 % | 74.2 % | 90.3 % | 90.0 % |
| 2017-08-05 | 5,569,321 | 6,377,858 | 76,190 | 884,726 | 538.9 | 2,150,400 | 14.8 % | 57.6 % | 75.1 % | 90.4 % | 90.1 % |
| 2017-08-06 | 5,385,402 | 6,315,176 | 64,218 | 993,992 | 593.5 | 2,309,076 | 14.9 % | 59.6 % | 76.2 % | 90.9 % | 90.4 % |
| 2017-08-08 | 5,393,755 | 6,212,415 | 66,502 | 885,162 | 533.7 | 2,102,835 | 14.9 % | 57.6 % | 74.4 % | 89.8 % | 89.4 % |
| **Total / media** | **212,874,453** | **225,042,057** | **5,335,321** | **17,502,925** | **355.7** | **61,682,458** | **14.8 %** | **52.8 %** | **72.0 %** | **89.5 %** | **89.2 %** |

- El pedido cubrió el 105.7 % de la venta real. Los quiebres fueron el 2.5 % de la venta y la merma el 8.2 %.
- **Calibración real.** El p71, que decide el pedido, dejó por debajo el 72.0 % de las ventas diarias, frente al 71 % nominal. El p50 dejó el 52.8 % y el p90 el 89.5 %. Por pedido, el p71 osciló entre el 67.5 % y el 76.2 %, el p90 entre el 86.1 % y el 91.3 %, y el p50 entre el 47.8 % y el 59.6 %. El p50 es el más sensible a las semanas en que la demanda se desvía.
- **El p10 deja un 14.8 %** en lugar del 10 %, y por eso el intervalo p10–p90 cubre el 89.2 % y no el 80 %. La causa son las series que no venden (familias poco comunes en algunas tiendas): el p10 es 0 y la venta real es 0, así que quedan "por debajo" y "dentro" a la vez. El intervalo es conservador en la cola baja, no en la alta.
- **Correcciones incorporadas.** El DAG 04 reevalúa su ventana en cada run con la última versión de las ventas, y 11 pedidos cambiaron su venta real entre evaluaciones. Por ejemplo, en los emitidos del 2017-07-21 al 2017-07-23 la primera evaluación se hizo cuando aún no habían llegado las ventas tardías del 2017-07-28. Los pedidos publicados no cambian.

**Lectura del seguimiento.**
- Hasta mediados de julio el pedido sobrestima un 5 % a 8 %, y el costo lo domina la merma, como corresponde al p71.
- Entre el 23 y el 30 de julio la demanda alcanza o supera al pedido y pasan a dominar los quiebres. En esas semanas el p50 baja del 50 % y el p90 al 86 %: la demanda real se fue a la parte alta del pronóstico.
- Desde el 31 de julio vuelve a dominar la merma, el MAE crece de 411 a 594 y el p50 sube por encima del 55 %.

v24 está entrenado con datos hasta el 2017-07-01 y no fue reemplazado, porque en la validación de cada entrenamiento los candidatos no lo mejoraban un 2 %. El seguimiento sí muestra que en agosto sobrestima. Esta señal no la usa hoy el criterio de reemplazo (posible mejora: considerar también el costo real del seguimiento).

## 5. Reproducción con DVC

En una carpeta vacía, a partir del commit de la publicación:

```text
$ git clone <repo> prueba-reproduccion && cd prueba-reproduccion && git checkout b6561db
b6561db data: publicación con pronóstico por cuantiles, 2017-07-01 a 2017-08-15
$ dvc pull                                    # 5 s
  35M   data/raw
  97M   data/processed
  52M   data/predictions
  148M  models

$ python scripts/reproducir_pedido.py --clave 2017-07-01
Pedido 2017-07-01: modelo v24, código bd89c6ae9a17281a6441fe4ba480ea8df3a8048f
  features      coincide con el manifiesto
  datos_crudos  coincide con el manifiesto
  modelo        coincide con el manifiesto
  md5 publicado   cd9ac84567b3d414e3c15f815c72321a
  md5 reproducido cd9ac84567b3d414e3c15f815c72321a  (1782 filas)
  pronóstico diario: publicado 7ff709078d15606a2186bc6d5c22ae0d, reproducido 7ff709078d15606a2186bc6d5c22ae0d
Resultado: IDÉNTICO

$ python scripts/reproducir_pedido.py --clave 2017-07-31
Pedido 2017-07-31: modelo v24, código bd89c6ae9a17281a6441fe4ba480ea8df3a8048f
  features      coincide con el manifiesto
  datos_crudos  coincide con el manifiesto
  modelo        coincide con el manifiesto
  md5 publicado   f3b2c98518e1dff261ce73a709fbd5f9
  md5 reproducido f3b2c98518e1dff261ce73a709fbd5f9  (1782 filas)
  pronóstico diario: publicado cf7d8685d182f07dfa8ab3fb3b1333f3, reproducido cf7d8685d182f07dfa8ab3fb3b1333f3
Resultado: IDÉNTICO

$ python scripts/reproducir_pedido.py --clave 2017-08-15
Pedido 2017-08-15: modelo v24, código bd89c6ae9a17281a6441fe4ba480ea8df3a8048f
  features      coincide con el manifiesto
  datos_crudos  coincide con el manifiesto
  modelo        coincide con el manifiesto
  md5 publicado   df98f632cbac5631317f6a212cc3bbba
  md5 reproducido df98f632cbac5631317f6a212cc3bbba  (1782 filas)
  pronóstico diario: publicado 523939235dc92cb3d8fd260b763db9b4, reproducido 523939235dc92cb3d8fd260b763db9b4
Resultado: IDÉNTICO

$ for c in <las 43 claves>; do python scripts/reproducir_pedido.py --clave $c; done
43 de 43 pedidos y pronósticos reproducidos idénticos
```

**Control negativo:** con el p90 de una fila del pronóstico publicado del 2017-07-31 cambiado a mano, el pedido sigue igual y el script detecta el cambio.

```text
  md5 reproducido f3b2c98518e1dff261ce73a709fbd5f9  (1782 filas)
  pronóstico diario: publicado c7a8a91c744e167d0957b7d81cc79216, reproducido cf7d8685d182f07dfa8ab3fb3b1333f3
Resultado: DISTINTO
```

**Publicación anterior.** En el mismo clon, `git checkout 6a3ce1a && dvc checkout` restaura la publicación previa al pronóstico por cuantiles: modelo v5, código `39f15a7`, sin pronóstico diario. Se sigue reproduciendo idéntica:

```text
Pedido 2017-07-31: modelo v5, código 39f15a7b8339aae8f09860571344d1da1c923ced
  md5 publicado   06c2daad75b79f9dfecdffd56ff2bf00
  md5 reproducido 06c2daad75b79f9dfecdffd56ff2bf00  (1782 filas)
Resultado: IDÉNTICO
```

**Idempotencia:** volver a ejecutar el DAG 03 de una fecha deja el pedido, el pronóstico, el manifiesto y los punteros `.dvc` iguales byte a byte. Lo verifica la prueba `test_manifiesto_es_idempotente`, y en Airflow se comprobó con el 2017-07-31 en la publicación anterior.

## 6. Capturas de la interfaz

Las capturas de la UI de Airflow (*Grid* de los DAGs) y de MLflow (comparación de runs y Model Registry) van en `docs/img/`.
