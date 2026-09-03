PYTHON ?= python3

COMPOSE_FILES := -f docker-compose.yml
ifneq ($(wildcard docker-compose.override.yml),)
COMPOSE_FILES += -f docker-compose.override.yml
endif

COMPOSE := docker compose $(COMPOSE_FILES)

.PHONY: start stop restart logs status skipped failures warmup observability observability-stop grafana prometheus mlflow metrics ingest reindex rebuild smoke query ask chat chat-stop chat-logs chat-check eval generate-eval compare-embeddings link-knowledge branding test backup check-private clean reset-data

start: branding
	@test -f .env || cp .env.example .env
	$(COMPOSE) up -d --build

stop:
	$(COMPOSE) down

# Restarts the process, not the container. Environment is fixed when a container is created, so
# after editing .env use `make start` instead, which recreates it.
restart:
	$(COMPOSE) restart product-memory

logs:
	$(COMPOSE) logs -f product-memory

status:
	$(COMPOSE) exec product-memory product-memory status

# Files that hold no indexable text, and why each one was left out.
skipped:
	$(COMPOSE) exec product-memory product-memory skipped
# Files that could not be read at all, and the error for each.
failures:
	$(COMPOSE) exec product-memory product-memory failures
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

# Open WebUI runs with the core stack, talking to this service's grounded endpoint over the compose
# network. Ollama itself runs on the host: Docker on macOS cannot reach the GPU, and a CPU-bound
# model is unusable here. This target checks the model behind the UI and prints where to find it.
chat: branding
	@grep -q '^CHAT_ENABLED=true' .env || \
		(echo "Set CHAT_ENABLED=true in .env first." >&2; exit 2)
	$(COMPOSE) up -d product-memory open-webui
	$(MAKE) chat-check
	@echo "Open WebUI http://localhost:$${OPEN_WEBUI_PORT:-2605}"

chat-stop:
	$(COMPOSE) stop open-webui

chat-logs:
	$(COMPOSE) logs -f open-webui

# Is the local model server up, and does it hold the models .env asks for?
chat-check:
	$(COMPOSE) exec product-memory product-memory chat-check

# One grounded answer, without a browser.
ask:
	@test -n "$(q)" || (echo "Usage: make ask q='What did we decide about payment retries?'" >&2; exit 2)
	$(COMPOSE) exec product-memory product-memory ask $(args) "$(q)"

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

# Renders the BRAND_* settings into the stylesheet Open WebUI serves. Recreate the container
# afterwards to pick it up; `make start` does both.
branding:
	@test -f .env || cp .env.example .env
	$(PYTHON) scripts/branding.py

test:
	$(PYTHON) -m pytest

check-private:
	@test -f .private-terms || { echo "no .private-terms yet — copy .private-terms.example and fill it in"; exit 0; }
	@found=$$(git grep -inIE -f .private-terms -- . || true); \
	if [ -n "$$found" ]; then \
		echo "$$found"; \
		echo ""; \
		echo "private terms found in tracked files — do not commit"; \
		exit 1; \
	fi; \
	echo "tracked files carry none of the private terms"

backup:
	@mkdir -p backups
	@stamp=$$(date +%Y%m%d-%H%M%S); \
	dir=backups/product-memory-$$stamp; mkdir -p $$dir; \
	$(COMPOSE) exec -T db sh -c 'pg_dump -Fc -U "$$POSTGRES_USER" "$$POSTGRES_DB"' > $$dir/database.dump; \
	cp eval/questions*.yaml $$dir/ 2>/dev/null || true; \
	tar -czf $$dir.tar.gz -C backups product-memory-$$stamp && rm -rf $$dir; \
	echo "wrote $$dir.tar.gz"

clean:
	$(COMPOSE) down --remove-orphans

reset-data:
	$(COMPOSE) down -v