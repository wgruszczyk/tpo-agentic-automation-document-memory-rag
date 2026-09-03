from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

from product_memory.generation.base import ChatProvider
from product_memory.generation.ollama_client import OllamaChatProvider
from product_memory.settings import Settings

# Names that only ever resolve to this machine or to the host running its containers. Hostnames
# are matched rather than resolved: DNS at start-up is one more thing that can be wrong or
# poisoned, and a name that has to be looked up is by definition not obviously local.
LOCAL_HOSTNAMES = frozenset(
    {"localhost", "host.docker.internal", "gateway.docker.internal", "host.lima.internal"}
)


def is_local_host(host: str) -> bool:
    host = host.strip().strip("[]").lower()
    if not host:
        return False
    if host in LOCAL_HOSTNAMES or host.endswith(".local") or host.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_private or address.is_link_local


def ensure_local_url(url: str, *, setting: str) -> str:
    """Refuse an inference endpoint that is not on this machine or this network.

    The whole point of this service is that private documents are read by something the owner
    controls. A base URL pointing anywhere else would ship the corpus off the machine one answer
    at a time, and would do it silently, so it is a start-up failure rather than a warning.
    """
    parts = urlsplit(url.strip())
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError(f"{setting} must be an http(s) URL with a host, got {url!r}")
    if not is_local_host(parts.hostname):
        raise ValueError(
            f"{setting} points at {parts.hostname!r}, which is not local. This service only talks "
            "to inference running on this machine or this private network."
        )
    return url.strip().rstrip("/")


def create_chat_provider(settings: Settings) -> ChatProvider:
    if settings.chat_provider == "ollama":
        ensure_local_url(settings.ollama_base_url, setting="OLLAMA_BASE_URL")
        return OllamaChatProvider(settings)
    raise ValueError(f"Unsupported chat provider: {settings.chat_provider}")
