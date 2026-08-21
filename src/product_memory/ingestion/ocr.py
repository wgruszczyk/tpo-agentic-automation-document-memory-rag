from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from product_memory.settings import Settings

if TYPE_CHECKING:
    from PIL.Image import Image

LOGGER = logging.getLogger(__name__)
FALLBACK_LANGUAGE = "eng"


def _confidence(value: object) -> float:
    """Tesseract reports confidence as a string, and -1 for entries it did not score."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


class OcrEngine:
    """Tesseract-backed OCR that degrades to a no-op when disabled or unavailable."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._probed = False
        self._languages = ""

    @property
    def enabled(self) -> bool:
        return self.settings.enable_ocr

    def available(self) -> bool:
        if not self.enabled:
            return False
        if not self._probed:
            self._probed = True
            self._languages = self._resolve_languages()
        return bool(self._languages)

    def _resolve_languages(self) -> str:
        try:
            import pytesseract

            pytesseract.get_tesseract_version()
            installed = set(pytesseract.get_languages(config=""))
        except Exception as error:
            LOGGER.warning("OCR is enabled but Tesseract is unavailable, image text is skipped: %s", error)
            return ""

        requested = self.settings.ocr_language_list
        usable = [language for language in requested if language in installed]
        missing = [language for language in requested if language not in installed]
        if missing:
            LOGGER.warning("Tesseract language packs are missing and will be ignored: %s", ", ".join(missing))
        if not usable:
            if FALLBACK_LANGUAGE not in installed:
                LOGGER.warning("No usable Tesseract language pack found, image text is skipped")
                return ""
            usable = [FALLBACK_LANGUAGE]
        return "+".join(usable)

    def should_read(self, image: Image) -> bool:
        width, height = image.size
        return width * height >= self.settings.ocr_min_image_pixels

    def image_to_text(self, image: Image) -> str:
        """Return an image's text, or nothing when the image carries no information.

        Decks are full of logos, stock photography and decorative shapes. Tesseract reads
        them anyway, so a brand mark becomes a document about that brand and a photograph
        becomes a handful of nonsense tokens. Both crowd out real content in search results.
        Confidence separates the two: photographs score far below a diagram's labels, and a
        logo scores well but yields only a word or two, so we require enough confident words
        before treating an image as worth indexing.
        """
        if not self.available():
            return ""

        import pytesseract

        try:
            data = pytesseract.image_to_data(
                image,
                lang=self._languages,
                timeout=self.settings.ocr_timeout_seconds,
                output_type=pytesseract.Output.DICT,
            )
        except Exception:
            LOGGER.warning("OCR failed for an image, continuing without its text", exc_info=True)
            return ""

        lines: dict[tuple[int, int, int], list[str]] = {}
        confident_words = 0
        for index, raw_word in enumerate(data.get("text", [])):
            word = raw_word.strip()
            if not word or _confidence(data["conf"][index]) < self.settings.ocr_min_word_confidence:
                continue
            confident_words += 1
            key = (data["block_num"][index], data["par_num"][index], data["line_num"][index])
            lines.setdefault(key, []).append(word)

        if confident_words < self.settings.ocr_min_words:
            return ""

        text = "\n".join(" ".join(words) for _, words in sorted(lines.items())).strip()
        return text if len(text) >= self.settings.ocr_min_characters else ""
