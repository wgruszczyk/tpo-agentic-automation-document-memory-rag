from __future__ import annotations

import pytest

from product_memory.generation.factory import ensure_local_url, is_local_host


@pytest.mark.parametrize(
    "host",
    [
        "localhost",
        "127.0.0.1",
        "127.5.5.5",
        "::1",
        "host.docker.internal",
        "gateway.docker.internal",
        "workstation.local",
        "192.168.1.40",
        "10.0.0.7",
        "172.16.4.2",
        "169.254.10.1",
    ],
)
def test_local_hosts_are_accepted(host: str) -> None:
    assert is_local_host(host)


@pytest.mark.parametrize(
    "host",
    [
        "api.openai.com",
        "example.com",
        "8.8.8.8",
        "93.184.216.34",
        "2606:4700:4700::1111",
        "",
    ],
)
def test_remote_hosts_are_refused(host: str) -> None:
    assert not is_local_host(host)


def test_private_documents_never_leave_the_machine() -> None:
    with pytest.raises(ValueError, match="not local"):
        ensure_local_url("https://api.openai.com/v1", setting="OLLAMA_BASE_URL")


@pytest.mark.parametrize("url", ["ftp://localhost:11434", "localhost:11434", "not a url"])
def test_a_url_that_is_not_http_is_refused(url: str) -> None:
    with pytest.raises(ValueError, match="http"):
        ensure_local_url(url, setting="OLLAMA_BASE_URL")


def test_a_local_url_is_normalised() -> None:
    assert (
        ensure_local_url(" http://host.docker.internal:11434/ ", setting="OLLAMA_BASE_URL")
        == "http://host.docker.internal:11434"
    )
