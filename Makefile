COMPOSE_FILES := -f docker-compose.yml
ifneq ($(wildcard docker-compose.knowledge-links.yml),)
COMPOSE_FILES += -f docker-compose.knowledge-links.yml
endif
COMPOSE := docker compose $(COMPOSE_FILES)

.PHONY: start stop restart logs status ingest reindex smoke link-knowledge test clean

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
	$(COMPOSE) exec product-memory product-memory smoke-test

link-knowledge:
	@test -f .env || cp .env.example .env
	python scripts/link_knowledge.py

test:
	python -m pytest

clean:
	$(COMPOSE) down -v
