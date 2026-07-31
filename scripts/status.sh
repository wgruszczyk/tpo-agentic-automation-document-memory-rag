#!/usr/bin/env sh
set -eu
curl -fsS http://localhost:${MCP_PORT:-8080}/health
printf '\n'
