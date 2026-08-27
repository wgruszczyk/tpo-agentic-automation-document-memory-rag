from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from product_memory.ingestion.extractors import RECORDING_EXTENSIONS, _extract_recording
from product_memory.ingestion.parser import EmptyDocumentError
from product_memory.ingestion.transcription import (
    Transcriber,
    UnsupportedLanguageError,
    _timestamp,
)
from product_memory.settings import Settings


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


class FakeModel:
    def __init__(self, segments, language="en", probability=0.99):
        self._segments = segments
        self._language = language
        self._probability = probability

    def transcribe(self, _audio, **_kwargs):
        info = SimpleNamespace(
            language=self._language, language_probability=self._probability, duration=61.0
        )
        return iter(self._segments), info


def _transcriber(model: FakeModel, **overrides) -> Transcriber:
    transcriber = Transcriber(_settings(**overrides))
    transcriber._model = model  # noqa: SLF001
    transcriber._extract_audio = lambda _path, _destination: None  # type: ignore[method-assign]
    return transcriber


def test_speech_is_returned_as_timestamped_lines() -> None:
    model = FakeModel(
        [
            SimpleNamespace(start=0.0, text=" We agreed the terms."),
            SimpleNamespace(start=754.0, text="  Rollout starts in March. "),
        ]
    )

    content, metadata = _transcriber(model).transcribe(Path("meeting.mp4"))

    assert content == "[00:00:00] We agreed the terms.\n[00:12:34] Rollout starts in March."
    assert metadata["spoken_language"] == "en"
    assert metadata["duration_seconds"] == 61.0
    assert metadata["source_format"] == "recording"


def test_a_recording_in_another_language_is_refused_rather_than_guessed() -> None:
    model = FakeModel([SimpleNamespace(start=0.0, text="cokolwiek")], language="pl")

    with pytest.raises(UnsupportedLanguageError, match="pl"):
        _transcriber(model).transcribe(Path("spotkanie.mp4"))


def test_an_accepted_extra_language_is_transcribed() -> None:
    model = FakeModel([SimpleNamespace(start=0.0, text="wir haben")], language="de")

    content, metadata = _transcriber(model, transcription_languages="en,de").transcribe(
        Path("besprechung.mp4")
    )

    assert content == "[00:00:00] wir haben"
    assert metadata["spoken_language"] == "de"


def test_segments_that_are_only_whitespace_are_dropped() -> None:
    model = FakeModel(
        [SimpleNamespace(start=0.0, text="   "), SimpleNamespace(start=5.0, text="real speech")]
    )

    content, _ = _transcriber(model).transcribe(Path("meeting.mp4"))

    assert content == "[00:00:05] real speech"


def test_timestamps_pass_an_hour() -> None:
    assert _timestamp(0) == "00:00:00"
    assert _timestamp(3661) == "01:01:01"


def test_every_recording_extension_is_indexable() -> None:
    # A format the scanner accepts but the extractor does not route would fall through to being
    # read as text, which for a video means binary noise in the index.
    assert RECORDING_EXTENSIONS <= _settings().extensions


class RefusingTranscriber:
    def __init__(self, error: Exception):
        self._error = error

    def available(self) -> bool:
        return True

    def transcribe(self, _path):
        raise self._error


def test_an_unsupported_language_is_skipped_not_failed() -> None:
    # Skips are an expected part of a corpus; failures mean something needs attention.
    with pytest.raises(EmptyDocumentError, match="only en is indexed"):
        _extract_recording(
            Path("spotkanie.mp4"),
            RefusingTranscriber(UnsupportedLanguageError("spoken language is pl, and only en is indexed")),
        )


def test_a_file_with_no_audio_track_is_reported_as_unreadable() -> None:
    from product_memory.ingestion.extractors import UnreadableDocumentError

    with pytest.raises(UnreadableDocumentError, match="no readable audio track"):
        _extract_recording(
            Path("broken.mp4"), RefusingTranscriber(subprocess.SubprocessError("bad input"))
        )


def test_transcription_is_skipped_when_the_engine_is_unavailable() -> None:
    with pytest.raises(EmptyDocumentError, match="Transcription is unavailable"):
        _extract_recording(Path("meeting.mp4"), None)


class CachingParser:
    def __init__(self, cached: set[str]):
        self._cached = cached

    def has_cached_extraction(self, _path: Path, relative_path: str) -> bool:
        return relative_path in self._cached


def _service(cached: set[str]):
    from product_memory.ingestion.service import IngestionService

    return IngestionService(
        _settings(),
        db=None,  # type: ignore[arg-type]
        provider=None,  # type: ignore[arg-type]
        parser=CachingParser(cached),  # type: ignore[arg-type]
        chunker=None,  # type: ignore[arg-type]
    )


def test_only_a_recording_that_still_needs_reading_counts_against_the_scan_budget() -> None:
    service = _service(cached={"already.mp4"})

    # A transcript already in the cache costs nothing to reuse, so it must not consume the budget
    # that exists to stop one scan disappearing into hours of audio.
    assert service._is_unread_recording(Path("/k/new.mp4"), "new.mp4") is True  # noqa: SLF001
    assert service._is_unread_recording(Path("/k/already.mp4"), "already.mp4") is False  # noqa: SLF001
    assert service._is_unread_recording(Path("/k/notes.pdf"), "notes.pdf") is False  # noqa: SLF001
