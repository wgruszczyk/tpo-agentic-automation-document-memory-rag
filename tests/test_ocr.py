import pytesseract
import pytest
from PIL import Image

from product_memory.ingestion.ocr import OcrEngine
from product_memory.settings import Settings


def engine(**overrides) -> OcrEngine:
    return OcrEngine(Settings(_env_file=None, **overrides))


def _ocr_data(words: list[str], confidences: list[float], lines: list[int] | None = None) -> dict:
    count = len(words)
    return {
        "text": words,
        "conf": confidences,
        "block_num": [0] * count,
        "par_num": [0] * count,
        "line_num": lines or [0] * count,
    }


def _install_tesseract(monkeypatch: pytest.MonkeyPatch, data: dict) -> None:
    monkeypatch.setattr(pytesseract, "get_tesseract_version", lambda *_a, **_k: "5.3.0")
    monkeypatch.setattr(pytesseract, "get_languages", lambda *_a, **_k: ["eng"])
    monkeypatch.setattr(pytesseract, "image_to_data", lambda *_a, **_k: data)


def test_ocr_is_enabled_by_default() -> None:
    assert Settings(_env_file=None).enable_ocr is True


def test_disabled_flag_short_circuits_before_probing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args, **_kwargs):
        raise AssertionError("Tesseract must not be probed when OCR is disabled")

    monkeypatch.setattr(pytesseract, "get_tesseract_version", fail)

    assert engine(enable_ocr=False).available() is False


def test_missing_tesseract_degrades_to_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pytesseract,
        "get_tesseract_version",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("tesseract not installed")),
    )
    ocr = engine()

    assert ocr.available() is False
    assert ocr.image_to_text(Image.new("RGB", (50, 50))) == ""


def test_unavailable_language_packs_are_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pytesseract, "get_tesseract_version", lambda *_a, **_k: "5.3.0")
    monkeypatch.setattr(pytesseract, "get_languages", lambda *_a, **_k: ["eng", "deu"])
    ocr = engine(ocr_languages="eng+pol")

    assert ocr.available() is True
    assert ocr._languages == "eng"


def test_falls_back_to_english_when_no_requested_pack_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pytesseract, "get_tesseract_version", lambda *_a, **_k: "5.3.0")
    monkeypatch.setattr(pytesseract, "get_languages", lambda *_a, **_k: ["eng"])
    ocr = engine(ocr_languages="pol")

    assert ocr.available() is True
    assert ocr._languages == "eng"


def test_output_shorter_than_threshold_is_discarded(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_tesseract(monkeypatch, _ocr_data(["a", "b", "c", "d", "e", "f"], [95] * 6))

    assert engine(ocr_min_characters=12).image_to_text(Image.new("RGB", (50, 50))) == ""


def test_readable_output_is_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_tesseract(
        monkeypatch,
        _ocr_data(
            ["Invoice", "total", "", "1200", "PLN", "net", "30"],
            [96, 95, -1, 92, 90, 88, 91],
            lines=[0, 0, 0, 1, 1, 1, 1],
        ),
    )

    assert engine().image_to_text(Image.new("RGB", (50, 50))) == "Invoice total\n1200 PLN net 30"


def test_tiny_images_are_not_worth_reading() -> None:
    ocr = engine(ocr_min_image_pixels=10_000)

    assert ocr.should_read(Image.new("RGB", (20, 20))) is False
    assert ocr.should_read(Image.new("RGB", (200, 200))) is True


def test_language_list_accepts_plus_and_comma_separators() -> None:
    assert Settings(_env_file=None, ocr_languages="eng+pol").ocr_language_list == ["eng", "pol"]
    assert Settings(_env_file=None, ocr_languages="eng, pol").ocr_language_list == ["eng", "pol"]


def test_a_photograph_is_skipped_because_no_word_is_read_confidently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Tesseract still returns tokens for stock photography, but scores them very low.
    _install_tesseract(monkeypatch, _ocr_data(["WN", "a", "~~", "TF", "ih", "Ae"], [34, 12, 8, 21, 30, 19]))

    assert engine().image_to_text(Image.new("RGB", (400, 300))) == ""


def test_a_logo_is_skipped_because_it_carries_too_few_words(monkeypatch: pytest.MonkeyPatch) -> None:
    # A brand mark reads cleanly, so only its length gives it away as decoration.
    _install_tesseract(monkeypatch, _ocr_data(["ACME", "INTERNATIONAL"], [96, 94]))

    assert engine().image_to_text(Image.new("RGB", (400, 300))) == ""


def test_a_diagram_is_read_because_it_carries_confident_words(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_tesseract(
        monkeypatch,
        _ocr_data(
            ["Pilot", "1", "Turkey", "Poland", "Switzerland", "Belgium", "rollout", "2026"],
            [92, 81, 95, 93, 90, 88, 86, 94],
        ),
    )

    assert engine().image_to_text(Image.new("RGB", (1147, 546))) == (
        "Pilot 1 Turkey Poland Switzerland Belgium rollout 2026"
    )
