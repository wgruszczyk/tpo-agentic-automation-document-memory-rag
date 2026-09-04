"""Renders branding/custom.css and branding/loader.js from .env, for the hooks Open WebUI serves.

Open WebUI has no branding environment variables worth using. WEBUI_NAME works but appends
"(Open WebUI)", and CUSTOM_NAME polls their servers, which this project does not do. What it does
have is an empty /static/custom.css and an empty /static/loader.js, both loaded by every page.
Those are the whole hook, so this fills them.

The logo is inlined as a data URI rather than mounted, so only generated files have to be mapped
into the container and none of the image's own assets are hidden behind a volume. The tab icon is
a <link> that no stylesheet can reach, which is what the script is for.
"""

from __future__ import annotations

import base64
import os
import re
import sys
from pathlib import Path

ENV_FILE = ".env"
CSS_TEMPLATE = "branding/custom.css.template"
CSS_OUTPUT = "branding/custom.css"
JS_TEMPLATE = "branding/loader.js.template"
JS_OUTPUT = "branding/loader.js"

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
    ".ico": "image/x-icon",
}

# The two images Open WebUI's own markup points at: the header mark and the loading splash.
LOGO_TARGETS = ('img[src="/static/favicon.png"]', 'img[src="/static/splash.png"]')


def main(repo_root: Path | None = None) -> int:
    repo_root = repo_root or Path(__file__).resolve().parents[1]
    config = _read_env_file(repo_root / ENV_FILE)
    logo = config.get("BRAND_LOGO", "").strip()

    try:
        values = {
            key: _checked(key, config.get(key, "").strip() or default)
            for key, default in DEFAULTS.items()
        }
        logo_rule = _logo_rule(logo, repo_root)
        favicon = _data_uri(config.get("BRAND_FAVICON", "").strip() or logo, "BRAND_FAVICON", repo_root)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    css = (repo_root / CSS_TEMPLATE).read_text(encoding="utf-8")
    for key, value in values.items():
        css = css.replace(f"__{key}__", value)
    css = css.replace("__BRAND_LOGO_RULE__", logo_rule)
    (repo_root / CSS_OUTPUT).write_text(css, encoding="utf-8")

    (repo_root / JS_OUTPUT).write_text(_loader(favicon, repo_root), encoding="utf-8")

    print(
        f"Wrote {CSS_OUTPUT} ({'with' if logo_rule else 'no'} logo) and "
        f"{JS_OUTPUT} ({'with' if favicon else 'no'} favicon). Run 'make start' to apply them."
    )
    return 0


def _loader(favicon: tuple[str, str] | None, repo_root: Path) -> str:
    # Written either way: Docker creates a directory where a missing bind-mounted file should be.
    if favicon is None:
        return "// No BRAND_FAVICON or BRAND_LOGO set, so Open WebUI's own tab icon is left alone.\n"

    mime, data = favicon
    script = (repo_root / JS_TEMPLATE).read_text(encoding="utf-8")
    return script.replace("__BRAND_FAVICON_MIME__", mime).replace(
        "__BRAND_FAVICON_DATA__", f"data:{mime};base64,{data}"
    )


def _checked(key: str, value: str) -> str:
    pattern = FONT if key == "BRAND_FONT" else COLOR
    if not pattern.fullmatch(value):
        raise ValueError(f"{key}={value!r} is not a value this can safely put in a stylesheet")
    return value


def _logo_rule(raw_path: str, repo_root: Path) -> str:
    inlined = _data_uri(raw_path, "BRAND_LOGO", repo_root)
    if inlined is None:
        return ""

    mime, data = inlined
    selectors = ",\n".join(LOGO_TARGETS)
    return f"{selectors} {{\n  content: url(data:{mime};base64,{data});\n}}"


def _data_uri(raw_path: str, key: str, repo_root: Path) -> tuple[str, str] | None:
    if not raw_path:
        return None

    path = Path(os.path.expandvars(raw_path)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    if not path.is_file():
        raise ValueError(f"{key} points at {path}, which does not exist")

    mime = MIME_TYPES.get(path.suffix.lower())
    if mime is None:
        raise ValueError(f"{key} must be one of {', '.join(sorted(MIME_TYPES))}, not {path.suffix}")

    return mime, base64.b64encode(path.read_bytes()).decode("ascii")


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
