from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_postgres_uses_stable_persistent_volume() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))

    assert "postgres_data:/var/lib/postgresql/data" in compose["services"]["db"]["volumes"]
    assert (
        compose["volumes"]["postgres_data"]["name"]
        == "${COMPOSE_PROJECT_NAME:-tpo-agentic-automation-document-memory-rag}_postgres_data"
    )


def test_container_healthcheck_survives_a_rebuild() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))

    probe = " ".join(compose["services"]["product-memory"]["healthcheck"]["test"])

    assert "/health/live" in probe


def test_clean_preserves_volumes_and_reset_data_is_explicit() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    clean_section = makefile.split("\nclean:\n", maxsplit=1)[1].split("\nreset-data:\n", maxsplit=1)[0]

    assert "down -v" not in clean_section
    assert "$(COMPOSE) down --remove-orphans" in clean_section
    assert "\nreset-data:\n\t$(COMPOSE) down -v" in makefile


def test_make_query_requires_question_and_calls_cli_query() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    query_section = makefile.split("\nquery:\n", maxsplit=1)[1].split("\nlink-knowledge:\n", maxsplit=1)[0]

    assert "Usage: make query q=" in query_section
    assert "product-memory query --url http://127.0.0.1:8080/mcp" in query_section
    assert '$(args) "$(q)"' in query_section


def test_make_smoke_uses_container_internal_mcp_url() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    smoke_section = makefile.split("\nsmoke:\n", maxsplit=1)[1].split("\nquery:\n", maxsplit=1)[0]

    assert "product-memory smoke-test --url http://127.0.0.1:8080/mcp" in smoke_section


def test_the_chat_ui_is_optional_pinned_and_bound_to_this_machine() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    webui = compose["services"]["open-webui"]

    assert webui["profiles"] == ["chat"]
    assert ":main" not in webui["image"] and ":" in webui["image"]
    assert webui["ports"] == ["127.0.0.1:${OPEN_WEBUI_PORT:-2605}:8080"]


def test_the_chat_ui_reaches_only_this_service_and_nothing_outside() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    environment = compose["services"]["open-webui"]["environment"]

    assert environment["OPENAI_API_BASE_URLS"] == "http://product-memory:8080/v1"
    # A raw model looks the same in the UI and answers from its training instead of the index.
    assert environment["ENABLE_OLLAMA_API"] == "false"
    assert environment["ENABLE_WEB_SEARCH"] == "false"
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["ANONYMIZED_TELEMETRY"] == "false"


def test_the_service_can_reach_a_model_running_on_the_host() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))

    assert "host.docker.internal:host-gateway" in compose["services"]["product-memory"]["extra_hosts"]


def test_make_chat_refuses_until_conversation_is_switched_on() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    chat_section = makefile.split("\nchat:\n", maxsplit=1)[1].split("\nchat-stop:\n", maxsplit=1)[0]

    assert "CHAT_ENABLED=true" in chat_section
    assert "chat-check" in chat_section
    assert "--profile chat up -d" in chat_section
