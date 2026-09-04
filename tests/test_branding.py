import base64
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import branding  # noqa: E402

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
SVG = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1"/>'


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "branding").mkdir()
    shutil.copy(ROOT / branding.CSS_TEMPLATE, tmp_path / branding.CSS_TEMPLATE)
    shutil.copy(ROOT / branding.JS_TEMPLATE, tmp_path / branding.JS_TEMPLATE)
    return tmp_path


def _write_env(repo: Path, body: str) -> None:
    (repo / ".env").write_text(body, encoding="utf-8")


def test_settings_reach_the_stylesheet(repo: Path) -> None:
    _write_env(repo, "BRAND_ACCENT=#ff0000\nBRAND_SURFACE=white\nBRAND_FONT=Georgia, serif\n")

    assert branding.main(repo) == 0

    css = (repo / branding.CSS_OUTPUT).read_text(encoding="utf-8")
    assert "--brand-accent: #ff0000;" in css
    assert "--brand-font: Georgia, serif;" in css


def test_an_absent_setting_falls_back_to_the_documented_default(repo: Path) -> None:
    _write_env(repo, "BRAND_ACCENT=\n")

    assert branding.main(repo) == 0

    css = (repo / branding.CSS_OUTPUT).read_text(encoding="utf-8")
    assert f"--brand-accent: {branding.DEFAULTS['BRAND_ACCENT']};" in css


def test_a_logo_is_inlined_rather_than_mounted(repo: Path) -> None:
    (repo / "branding" / "logo.png").write_bytes(PNG)
    _write_env(repo, "BRAND_LOGO=branding/logo.png\n")

    assert branding.main(repo) == 0

    css = (repo / branding.CSS_OUTPUT).read_text(encoding="utf-8")
    assert f"content: url(data:image/png;base64,{base64.b64encode(PNG).decode()});" in css
    assert 'img[src="/static/favicon.png"]' in css


def test_no_logo_leaves_the_stock_one_alone(repo: Path) -> None:
    _write_env(repo, "BRAND_LOGO=\n")

    assert branding.main(repo) == 0

    assert "content: url(" not in (repo / branding.CSS_OUTPUT).read_text(encoding="utf-8")


def test_the_favicon_defaults_to_the_logo(repo: Path) -> None:
    (repo / "branding" / "logo.png").write_bytes(PNG)
    _write_env(repo, "BRAND_LOGO=branding/logo.png\nBRAND_FAVICON=\n")

    assert branding.main(repo) == 0

    script = (repo / branding.JS_OUTPUT).read_text(encoding="utf-8")
    assert f"data:image/png;base64,{base64.b64encode(PNG).decode()}" in script
    assert 'link[rel~="icon"]' in script
    # The app appends an icon link of its own after hydration, and the last one declared wins.
    assert "MutationObserver" in script


def test_the_favicon_can_differ_from_the_logo(repo: Path) -> None:
    (repo / "branding" / "logo.png").write_bytes(PNG)
    (repo / "branding" / "favicon.svg").write_bytes(SVG)
    _write_env(repo, "BRAND_LOGO=branding/logo.png\nBRAND_FAVICON=branding/favicon.svg\n")

    assert branding.main(repo) == 0

    script = (repo / branding.JS_OUTPUT).read_text(encoding="utf-8")
    assert f"data:image/svg+xml;base64,{base64.b64encode(SVG).decode()}" in script


def test_the_loader_is_written_even_with_nothing_to_put_in_it(repo: Path) -> None:
    _write_env(repo, "BRAND_LOGO=\n")

    assert branding.main(repo) == 0

    # A missing file would become a directory when Docker mounts it.
    assert (repo / branding.JS_OUTPUT).exists()
    assert "link[rel" not in (repo / branding.JS_OUTPUT).read_text(encoding="utf-8")


def test_a_missing_logo_is_reported_instead_of_silently_skipped(repo: Path) -> None:
    _write_env(repo, "BRAND_LOGO=branding/nope.png\n")

    assert branding.main(repo) == 1
    assert not (repo / branding.CSS_OUTPUT).exists()


@pytest.mark.parametrize(
    "line",
    [
        "BRAND_ACCENT=red} body{display:none",
        "BRAND_SURFACE=#fff;} html{content:'x'",
        "BRAND_FONT=serif} * {background:url(http://example.com)",
    ],
)
def test_a_value_cannot_break_out_of_its_rule(repo: Path, line: str) -> None:
    _write_env(repo, f"{line}\n")

    assert branding.main(repo) == 1
    assert not (repo / branding.CSS_OUTPUT).exists()
