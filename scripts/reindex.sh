#!/usr/bin/env sh
set -eu
docker compose exec product-memory product-memory reindex
