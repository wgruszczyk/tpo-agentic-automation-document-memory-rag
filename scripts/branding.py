"""Renders branding/custom.css from .env, for the stylesheet Open WebUI already serves.

Open WebUI has no branding environment variables worth using. WEBUI_NAME works but appends
"(Open WebUI)", and CUSTOM_NAME polls their servers, which this project does not do. What it does
have is an empty /static/custom.css loaded by every page. That is the whole hook, so this fills it.

The logo is inlined as a data URI rather than mounted, so exactly one file has to be mapped into
the container and none of the image's own assets are hidden behind a volume.
"""

from __future__ import annotations

import base64
import os
import re
import sys
from pathlib import Path

ENV_FILE = ".env"
TEMPLATE = "branding/custom.css.template"
OUTPUT = "branding/custom.css"

# Values land in a stylesheet served to the browser, so they are matched, not escaped. A stray
# brace would otherwise end the rule and let the rest of the value become markup of its own.
COLOR = re.compile(r"^(?:#[0-9a-fA-F]{3,8}|[a-zA-Z]+)$")
FONT = re.compile(r"^[\w\s,'\"-]+$")

DEFAULTS = {
    "BRAND_ACCENT": "#0069b4",
    "BRAND_SURFACE": "#101418",
    "BRAND_FONT": "Inter, system-ui, sans-serif",
}

MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

# The two images Open WebUI's own markup points at: the header mark and the loading splash.
LOGO_TARGETS = ('img[src="/static/favicon.png"]', 'img[src="/static/splash.png"]')


def main(repo_root: Path | None = None) -> int:
    repo_root = repo_root or Path(__file__).resolve().parents[1]
    config = _read_env_file(repo_root / ENV_FILE)

    try:
        values = {
            key: _checked(key, config.get(key, "").strip() or default)
            for key, default in DEFAULTS.items()
        }
        logo_rule = _logo_rule(config.get("BRAND_LOGO", "").strip(), repo_root)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    css = (repo_root / TEMPLATE).read_text(encoding="utf-8")
    for key, value in values.items():
        css = css.replace(f"__{key}__", value)
    css = css.replace("__BRAND_LOGO_RULE__", logo_rule)

    output = repo_root / OUTPUT
    output.write_text(css, encoding="utf-8")
    print(f"Wrote {OUTPUT} ({'with' if logo_rule else 'no'} logo). Run 'make start' to apply it.")
    return 0


def _checked(key: str, value: str) -> str:
    pattern = FONT if key == "BRAND_FONT" else COLOR
    if not pattern.fullmatch(value):
        raise ValueError(f"{key}={value!r} is not a value this can safely put in a stylesheet")
    return value


def _logo_rule(raw_path: str, repo_root: Path) -> str:
    if not raw_path:
        return ""

    path = Path(os.path.expandvars(raw_path)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    if not path.is_file():
        raise ValueError(f"BRAND_LOGO points at {path}, which does not exist")

    mime = MIME_TYPES.get(path.suffix.lower())
    if mime is None:
        raise ValueError(f"BRAND_LOGO must be one of {', '.join(sorted(MIME_TYPES))}, not {path.suffix}")

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    selectors = ",\n".join(LOGO_TARGETS)
    return f"{selectors} {{\n  content: url(data:{mime};base64,{encoded});\n}}"


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = _strip_quotes(value.strip())
    return values


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
