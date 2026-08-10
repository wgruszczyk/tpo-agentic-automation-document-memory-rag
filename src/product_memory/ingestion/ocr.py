from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from product_memory.settings import Settings

if TYPE_CHECKING:
    from PIL.Image import Image

LOGGER = logging.getLogger(__name__)
FALLBACK_LANGUAGE = "eng"


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
        if not self.available():
            return ""

        import pytesseract

        try:
            text = pytesseract.image_to_string(
                image,
                lang=self._languages,
                timeout=self.settings.ocr_timeout_seconds,
            )
        except Exception:
            LOGGER.warning("OCR failed for an image, continuing without its text", exc_info=True)
            return ""

        text = "\n".join(line.rstrip() for line in text.splitlines() if line.strip()).strip()
        return text if len(text) >= self.settings.ocr_min_characters else ""
