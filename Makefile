.PHONY: start stop restart logs status ingest reindex smoke test clean

start:
	@test -f .env || cp .env.example .env
	docker compose up -d --build

stop:
	docker compose down

restart:
	docker compose restart product-memory

logs:
	docker compose logs -f product-memory

status:
	docker compose exec product-memory product-memory status

ingest:
	docker compose exec product-memory product-memory ingest-once

reindex:
	docker compose exec product-memory product-memory reindex

smoke:
	docker compose exec product-memory product-memory smoke-test

test:
	python -m pytest

clean:
	docker compose down -v
