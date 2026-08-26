PYTHON ?= python3

COMPOSE_FILES := -f docker-compose.yml
ifneq ($(wildcard docker-compose.override.yml),)
COMPOSE_FILES += -f docker-compose.override.yml
endif

COMPOSE := docker compose $(COMPOSE_FILES)

.PHONY: start stop restart logs status skipped warmup observability observability-stop grafana prometheus mlflow metrics ingest reindex rebuild smoke query eval generate-eval compare-embeddings link-knowledge test clean reset-data

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

# The only command that downloads models. Prints the revisions to pin afterwards.
warmup:
	$(COMPOSE) exec product-memory product-memory warmup

# Prometheus, Loki, Promtail and Grafana. Logs must be JSON for Loki to index their fields.
observability:
	@grep -q '^GRAFANA_ADMIN_PASSWORD=.\+' .env || \
		(echo "Set GRAFANA_ADMIN_PASSWORD in .env first." >&2; exit 2)
	@grep -q '^LOG_FORMAT=json' .env || \
		echo "Note: LOG_FORMAT is not json, so Loki will index whole lines rather than fields."
	$(COMPOSE) --profile observability up -d
	@echo "Grafana    http://localhost:$${GRAFANA_PORT:-2601}"
	@echo "Prometheus http://localhost:$${PROMETHEUS_PORT:-2602}"
	@echo "MLflow     http://localhost:$${MLFLOW_PORT:-2604}"

observability-stop:
	$(COMPOSE) --profile observability stop grafana prometheus loki promtail mlflow

grafana:
	open http://localhost:$${GRAFANA_PORT:-2601}

prometheus:
	open http://localhost:$${PROMETHEUS_PORT:-2602}

mlflow:
	open http://localhost:$${MLFLOW_PORT:-2604}

# Raw scrape output, useful when a dashboard panel is empty and you need to know why.
metrics:
	curl -fsS http://localhost:$${MCP_PORT:-2600}/metrics

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
	@test -f eval/questions.yaml || (echo "Create eval/questions.yaml from eval/questions.example.yaml, or run 'make generate-eval'" >&2; exit 2)
	$(COMPOSE) exec product-memory product-memory eval $(args)

# Builds a question set from your own indexed documents. Review it, then rename it to
# eval/questions.yaml. Both names are gitignored.
generate-eval:
	$(COMPOSE) exec -T product-memory product-memory generate-eval $(args) > eval/questions.generated.yaml
	@echo "Wrote eval/questions.generated.yaml"

# Judges another embedding model against the current one without re-embedding the index.
# Usage: make compare-embeddings model=intfloat/multilingual-e5-large
compare-embeddings:
	$(COMPOSE) exec product-memory product-memory compare-embeddings --model "$(model)" $(args)

link-knowledge:
	@test -f .env || cp .env.example .env
	$(PYTHON) scripts/link_knowledge.py

test:
	$(PYTHON) -m pytest

clean:
	$(COMPOSE) down --remove-orphans

reset-data:
	$(COMPOSE) down -v