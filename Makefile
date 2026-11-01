# Atajos para operar el producto. Ver README.md.
#   make levantar                                     # construye y levanta todo
#   make periodo DESDE=2017-07-01 HASTA=2017-08-15    # flujo completo sobre un periodo
#   make reproducir CLAVE=2017-07-31                  # reproduce un pedido publicado
#   make pruebas

DESDE ?=
HASTA ?=
REPROCESO ?= fallidos
CLAVE ?=
PYTHON ?= python

.PHONY: levantar detener periodo reproducir pruebas

levantar:
	docker compose up -d --build

detener:
	docker compose down

periodo:
	@test -n "$(DESDE)" -a -n "$(HASTA)" || (echo "Uso: make periodo DESDE=AAAA-MM-DD HASTA=AAAA-MM-DD [REPROCESO=ninguno|fallidos|todos]"; exit 2)
	docker compose exec airflow-scheduler python /opt/airflow/scripts/ejecutar_periodo.py \
		--desde $(DESDE) --hasta $(HASTA) --reproceso $(REPROCESO)

reproducir:
	@test -n "$(CLAVE)" || (echo "Uso: make reproducir CLAVE=AAAA-MM-DD"; exit 2)
	$(PYTHON) scripts/reproducir_pedido.py --clave $(CLAVE)

pruebas:
	$(PYTHON) -m pytest -q
