COMPOSE_FILES := -f docker-compose.yml
ifneq ($(wildcard docker-compose.override.yml),)
COMPOSE_FILES += -f docker-compose.override.yml
endif
COMPOSE := docker compose $(COMPOSE_FILES)

.PHONY: start stop restart logs status ingest reindex smoke query link-knowledge test clean reset-data

start:
	@test -f .env || cp .env.example .env
	$(COMPOSE) up -d --build

stop:
	$(COMPOSE) down

restart:
	$(COMPOSE) restart product-memory

logs:
	$(COMPOSE) logs -f product-memory

status:
	$(COMPOSE) exec product-memory product-memory status

ingest:
	$(COMPOSE) exec product-memory product-memory ingest-once

reindex:
	$(COMPOSE) exec product-memory product-memory reindex

smoke:
	$(COMPOSE) exec product-memory product-memory smoke-test --url http://127.0.0.1:8080/mcp

query:
	@test -n "$(q)" || (echo "Usage: make query q='What did we decide about payment retries?'" >&2; exit 2)
	$(COMPOSE) exec product-memory product-memory query --url http://127.0.0.1:8080/mcp $(args) "$(q)"

link-knowledge:
	@test -f .env || cp .env.example .env
	python scripts/link_knowledge.py

test:
	python -m pytest

clean:
	$(COMPOSE) down --remove-orphans

reset-data:
	$(COMPOSE) down -v
