PYTHON ?= python3

COMPOSE_FILES := -f docker-compose.yml
ifneq ($(wildcard docker-compose.override.yml),)
COMPOSE_FILES += -f docker-compose.override.yml
endif

COMPOSE := docker compose $(COMPOSE_FILES)

.PHONY: start stop restart logs status skipped ingest reindex rebuild smoke query eval link-knowledge test clean reset-data

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

# Files that hold no indexable text, and why each one was left out.
skipped:
	$(COMPOSE) exec product-memory product-memory skipped

ingest:
	$(COMPOSE) exec product-memory product-memory ingest-once

# Rebuilds embeddings from stored content.
reindex:
	$(COMPOSE) exec product-memory product-memory reindex

# Re-reads every file from disk. Use after an extraction change, such as new OCR support.
rebuild:
	$(COMPOSE) exec product-memory product-memory rebuild

smoke:
	$(COMPOSE) exec product-memory product-memory smoke-test --url http://127.0.0.1:8080/mcp

query:
	@test -n "$(q)" || (echo "Usage: make query q='What did we decide about payment retries?'" >&2; exit 2)
	$(COMPOSE) exec product-memory product-memory query --url http://127.0.0.1:8080/mcp $(args) "$(q)"

# Scores retrieval against eval/questions.yaml, which stays out of version control.
eval:
	@test -f eval/questions.yaml || (echo "Create eval/questions.yaml from eval/questions.example.yaml" >&2; exit 2)
	$(COMPOSE) exec product-memory product-memory eval $(args)

link-knowledge:
	@test -f .env || cp .env.example .env
	$(PYTHON) scripts/link_knowledge.py

test:
	$(PYTHON) -m pytest

clean:
	$(COMPOSE) down --remove-orphans

reset-data:
	$(COMPOSE) down -v