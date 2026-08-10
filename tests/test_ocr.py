import pytesseract
import pytest
from PIL import Image

from product_memory.ingestion.ocr import OcrEngine
from product_memory.settings import Settings


def engine(**overrides) -> OcrEngine:
    return OcrEngine(Settings(_env_file=None, **overrides))


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
    monkeypatch.setattr(pytesseract, "get_tesseract_version", lambda *_a, **_k: "5.3.0")
    monkeypatch.setattr(pytesseract, "get_languages", lambda *_a, **_k: ["eng"])
    monkeypatch.setattr(pytesseract, "image_to_string", lambda *_a, **_k: "  a\n\n ")

    assert engine(ocr_min_characters=12).image_to_text(Image.new("RGB", (50, 50))) == ""


def test_readable_output_is_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pytesseract, "get_tesseract_version", lambda *_a, **_k: "5.3.0")
    monkeypatch.setattr(pytesseract, "get_languages", lambda *_a, **_k: ["eng"])
    monkeypatch.setattr(pytesseract, "image_to_string", lambda *_a, **_k: "Invoice total\n\n  \n1200 PLN  \n")

    assert engine().image_to_text(Image.new("RGB", (50, 50))) == "Invoice total\n1200 PLN"


def test_tiny_images_are_not_worth_reading() -> None:
    ocr = engine(ocr_min_image_pixels=10_000)

    assert ocr.should_read(Image.new("RGB", (20, 20))) is False
    assert ocr.should_read(Image.new("RGB", (200, 200))) is True


def test_language_list_accepts_plus_and_comma_separators() -> None:
    assert Settings(_env_file=None, ocr_languages="eng+pol").ocr_language_list == ["eng", "pol"]
    assert Settings(_env_file=None, ocr_languages="eng, pol").ocr_language_list == ["eng", "pol"]
