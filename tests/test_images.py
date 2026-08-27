from __future__ import annotations

import io
from datetime import UTC, datetime

from PIL import Image

from product_memory.ingestion.extractors import _image_payload, _OcrCollector
from product_memory.mcp_server import _download_name
from product_memory.models import ChunkResult, ImageRef
from product_memory.retrieval.service import _IMAGE_MARKER, Retriever
from product_memory.settings import Settings


def _png(colour: str = "red", size: tuple[int, int] = (20, 10)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeOcr:
    def __init__(self, text: str):
        self.settings = Settings(_env_file=None)
        self._text = text

    def available(self) -> bool:
        return True

    def should_read(self, _image) -> bool:
        return True

    def image_to_text(self, _image) -> str:
        return self._text


def _collector(text: str, **overrides) -> _OcrCollector:
    return _OcrCollector(FakeOcr(text), keep_images=True, max_bytes=1_000_000, **overrides)


def test_a_picture_is_kept_when_its_text_is_worth_indexing() -> None:
    collector = _collector("Checkout error 500")
    collector.add_bytes("screenshot.png", _png())

    assert len(collector.images) == 1
    image = collector.images[0]
    assert image.label == "screenshot.png"
    assert image.media_type == "image/png"
    assert (image.width, image.height) == (20, 10)
    assert image.text == "Checkout error 500"
    assert image.data == _png()


def test_a_picture_with_no_readable_text_is_not_kept() -> None:
    # Nothing could ever match it, so storing the bytes would only cost space.
    collector = _collector("")
    collector.add_bytes("logo.png", _png())

    assert collector.images == []


def test_a_picture_larger_than_the_limit_is_read_but_not_kept() -> None:
    collector = _OcrCollector(FakeOcr("readable"), keep_images=True, max_bytes=10)
    collector.add_bytes("huge.png", _png())

    assert collector.images_read == 1
    assert collector.images == []


def test_the_original_bytes_are_preferred_over_a_re_encode() -> None:
    original = _png()
    image = Image.open(io.BytesIO(original))

    payload, media_type = _image_payload(image, original)

    assert payload == original
    assert media_type == "image/png"


def test_an_image_with_no_original_bytes_is_re_encoded_as_png() -> None:
    # PDF pages hand back a decoded picture rather than the bytes that were embedded.
    image = Image.new("RGB", (4, 4), "blue")

    payload, media_type = _image_payload(image, None)

    assert media_type == "image/png"
    assert Image.open(io.BytesIO(payload)).size == (4, 4)


def test_the_marker_the_extractor_writes_is_the_one_retrieval_looks_for() -> None:
    collector = _collector("Order total 970")
    collector.add_bytes("slide 4", _png())

    assert _IMAGE_MARKER.findall(collector.compose("body text")) == ["slide 4"]


def _chunk(content: str, source_path: str, score: float, chunk_id: str) -> ChunkResult:
    return ChunkResult(
        id=chunk_id,
        document_id="doc-1",
        document_title="Deck",
        source_path=source_path,
        chunk_index=0,
        content=content,
        start_char=0,
        end_char=len(content),
        effective_at=datetime(2026, 1, 1, tzinfo=UTC),
        semantic_score=0.0,
        lexical_score=0.0,
        recency_score=0.0,
        score=score,
    )


def _service(found: dict[tuple[str, str], list[ImageRef]]) -> Retriever:
    service = Retriever.__new__(Retriever)
    service.settings = Settings(_env_file=None)
    service._load_images = lambda _keys: found  # type: ignore[method-assign]
    return service


def _ref(identifier: str, label: str, path: str) -> ImageRef:
    return ImageRef(
        id=identifier,
        source_path=path,
        label=label,
        media_type="image/png",
        width=1,
        height=1,
        byte_size=1,
        url=f"http://localhost:2600/images/{identifier}",
    )


def test_a_chunk_carries_the_pictures_its_text_was_read_from() -> None:
    chunk = _chunk("[Image text: slide 4]\nOrder total", "deck.pptx", 0.9, "chunk-a")
    service = _service({("deck.pptx", "slide 4"): [_ref("img-1", "slide 4", "deck.pptx")]})

    images = service._attach_images([chunk])

    assert [image.id for image in chunk.images] == ["img-1"]
    assert [image.id for image in images] == ["img-1"]
    assert images[0].score == 0.9


def test_pictures_come_back_in_the_order_their_text_ranked() -> None:
    strong = _chunk("[Image text: slide 9]", "deck.pptx", 0.9, "chunk-a")
    weak = _chunk("[Image text: slide 2]", "deck.pptx", 0.1, "chunk-b")
    service = _service(
        {
            ("deck.pptx", "slide 9"): [_ref("img-9", "slide 9", "deck.pptx")],
            ("deck.pptx", "slide 2"): [_ref("img-2", "slide 2", "deck.pptx")],
        }
    )

    images = service._attach_images([strong, weak])

    assert [image.id for image in images] == ["img-9", "img-2"]


def test_the_same_picture_matched_twice_is_returned_once() -> None:
    first = _chunk("[Image text: slide 4]", "deck.pptx", 0.9, "chunk-a")
    second = _chunk("[Image text: slide 4]", "deck.pptx", 0.5, "chunk-b")
    service = _service({("deck.pptx", "slide 4"): [_ref("img-1", "slide 4", "deck.pptx")]})

    images = service._attach_images([first, second])

    assert [image.id for image in images] == ["img-1"]
    # The better placed chunk decides the score a caller sees.
    assert images[0].score == 0.9


def test_text_with_no_picture_marker_asks_the_database_for_nothing() -> None:
    def _explode(_keys):
        raise AssertionError("should not query for images")

    service = Retriever.__new__(Retriever)
    service.settings = Settings(_env_file=None)
    service._load_images = _explode  # type: ignore[method-assign]

    assert service._attach_images([_chunk("just words", "notes.md", 0.5, "chunk-a")]) == []


def test_one_slides_pictures_are_attached_once_each() -> None:
    # Every picture on a slide is labelled with that slide, so the chunk carries the same label
    # once per picture and each lookup answers with the whole set.
    chunk = _chunk("[Image text: slide 5]\na\n[Image text: slide 5]\nb", "deck.pptx", 0.8, "c")
    service = _service(
        {
            ("deck.pptx", "slide 5"): [
                _ref("img-1", "slide 5", "deck.pptx"),
                _ref("img-2", "slide 5", "deck.pptx"),
            ]
        }
    )

    images = service._attach_images([chunk])

    assert [image.id for image in chunk.images] == ["img-1", "img-2"]
    assert [image.id for image in images] == ["img-1", "img-2"]


def test_a_download_keeps_a_usable_filename() -> None:
    assert _download_name("slide 4", "image/png") == "slide-4.png"
    assert _download_name("screenshot.png", "image/png") == "screenshot.png"
    assert _download_name("../../etc/passwd", "image/jpeg") == "etc-passwd.jpeg"
