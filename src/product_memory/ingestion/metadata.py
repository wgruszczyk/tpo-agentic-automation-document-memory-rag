from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dateutil import parser as date_parser

_WEBVTT_HEADER = re.compile(r"^\ufeff?WEBVTT\b", re.IGNORECASE)
_VTT_VOICE = re.compile(r"<v\s+([^>]+)>", re.IGNORECASE)
_NOTE_FIELD = re.compile(r'^NOTE\s+([a-z_ -]+)\s*:\s*"?([^"\n]+)"?\s*$', re.IGNORECASE | re.MULTILINE)
_LABELED_TITLE = re.compile(
    r"^\s*(?:meeting\s+title|title|topic|subject)\s*[:=-]\s*(?P<value>.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_LABELED_DATE = re.compile(
    r"^\s*(?:meeting\s+date|date|started(?:\s+at)?|start\s+time|recorded\s+on)\s*[:=-]\s*(?P<value>.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_VTT_CUE = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2}\.\d{3})\s+-->\s+(?P<end>\d{2}:\d{2}:\d{2}\.\d{3})"
)
_MONTH_NAME = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    re.IGNORECASE,
)
_ISO_DATE = re.compile(r"\b20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}\b")
_NUMERIC_DATE = re.compile(r"\b(?P<a>\d{1,2})[/.](?P<b>\d{1,2})[/.](?P<year>20\d{2})\b")


def infer_document_metadata(path: Path, content: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    is_webvtt = bool(_WEBVTT_HEADER.search(content))
    preamble = _metadata_preamble(content)

    if is_webvtt:
        metadata["source_type"] = "meeting_transcript"
        metadata["transcript_format"] = "webvtt"

    title = _extract_labeled_value(_LABELED_TITLE, preamble)
    if title:
        metadata["title"] = title

    effective_at = _extract_labeled_date(preamble)
    if effective_at:
        metadata["effective_at"] = effective_at.isoformat()

    note_fields = _extract_note_fields(content)
    if note_fields.get("language"):
        metadata["language"] = note_fields["language"]

    duration_seconds = _extract_duration_seconds(note_fields.get("duration"), content)
    if duration_seconds is not None:
        metadata["duration_seconds"] = duration_seconds

    speakers = _extract_vtt_speakers(content)
    if speakers:
        metadata["source_type"] = "meeting_transcript"
        metadata["speakers"] = speakers
        metadata["speaker_count"] = len(speakers)

    if "teams" in path.name.lower() and metadata.get("source_type") == "meeting_transcript":
        metadata["source_app"] = "microsoft_teams"

    return metadata


def _extract_labeled_value(pattern: re.Pattern[str], content: str) -> str | None:
    for match in pattern.finditer(content):
        value = match.group("value").strip().strip('"')
        if value:
            return value
    return None


def _metadata_preamble(content: str) -> str:
    first_cue = _VTT_CUE.search(content)
    if first_cue:
        return content[: first_cue.start()]
    return "\n".join(content.splitlines()[:40])


def _extract_labeled_date(content: str) -> datetime | None:
    for match in _LABELED_DATE.finditer(content):
        value = match.group("value").strip()
        if _looks_like_date(value):
            parsed = date_parser.parse(value, fuzzy=True)
            return parsed.replace(tzinfo=parsed.tzinfo or UTC).astimezone(UTC)
    return None


def _looks_like_date(value: str) -> bool:
    if _MONTH_NAME.search(value) or _ISO_DATE.search(value):
        return True

    match = _NUMERIC_DATE.search(value)
    if not match:
        return False

    first = int(match.group("a"))
    second = int(match.group("b"))
    return first > 12 or second > 12


def _extract_note_fields(content: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for match in _NOTE_FIELD.finditer(content):
        key = match.group(1).strip().lower().replace(" ", "_").replace("-", "_")
        fields[key] = match.group(2).strip()
    return fields


def _extract_duration_seconds(note_duration: str | None, content: str) -> float | None:
    if note_duration:
        try:
            return _timestamp_to_seconds(note_duration)
        except ValueError:
            return None

    last_end = None
    for match in _VTT_CUE.finditer(content):
        try:
            last_end = _timestamp_to_seconds(match.group("end"))
        except ValueError:
            continue
    return last_end


def _timestamp_to_seconds(value: str) -> float:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"Unsupported timestamp: {value}")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def _extract_vtt_speakers(content: str) -> list[str]:
    speakers: list[str] = []
    seen: set[str] = set()

    for match in _VTT_VOICE.finditer(content):
        speaker = match.group(1).strip()
        if not speaker or speaker in seen:
            continue
        seen.add(speaker)
        speakers.append(speaker)

    return speakers
