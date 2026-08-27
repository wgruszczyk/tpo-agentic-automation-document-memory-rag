from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from product_memory.model_cache import prepare_model_load
from product_memory.settings import Settings

LOGGER = logging.getLogger(__name__)

# faster-whisper publishes converted weights under this account.
_MODEL_REPO = "Systran/faster-whisper-{model}"
_SAMPLE_RATE = "16000"
# A wav holding no samples is just its header, which is how a window past the end arrives.
_WAV_HEADER_BYTES = 1024


class UnsupportedLanguageError(ValueError):
    """Raised when a recording is not in a language this index is meant to hold."""


class NoAudioError(ValueError):
    """Raised when a file carries no audio track, so there is no speech to index."""


def _timestamp(seconds: float) -> str:
    minutes, second = divmod(int(seconds), 60)
    hour, minute = divmod(minutes, 60)
    return f"{hour:02d}:{minute:02d}:{second:02d}"


class Transcriber:
    """Whisper-backed speech to text that degrades to a no-op when disabled or unavailable."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model: Any = None

    @property
    def enabled(self) -> bool:
        return self.settings.enable_transcription

    @property
    def repository(self) -> str:
        return _MODEL_REPO.format(model=self.settings.transcription_model)

    def available(self) -> bool:
        if not self.enabled:
            return False
        try:
            subprocess.run(
                ["ffmpeg", "-version"], capture_output=True, check=True, timeout=30
            )
        except (OSError, subprocess.SubprocessError) as error:
            LOGGER.warning("Transcription is enabled but ffmpeg is unavailable: %s", error)
            return False
        return True

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        cached = prepare_model_load(
            self.settings.hf_home, self.repository, self.settings.allow_model_download
        )
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self.settings.transcription_model,
            device="cpu",
            compute_type="int8",
            cpu_threads=self.settings.transcription_threads,
            download_root=str(self.settings.hf_home),
            local_files_only=cached,
        )
        return self._model

    def _duration_seconds(self, path: Path) -> float:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        try:
            return float(result.stdout.strip())
        except ValueError:
            return 0.0

    def _has_audio(self, path: Path) -> bool:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
             "stream=index", "-of", "csv=p=0", str(path)],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        return bool(result.stdout.strip())

    def _extract_audio(self, path: Path, destination: Path, start: float, length: float) -> None:
        result = subprocess.run(
            # Mono 16 kHz PCM is what the model wants; anything else is decoded again internally.
            # Seeking before -i lets ffmpeg skip to the window instead of decoding up to it.
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
             "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(path),
             "-vn", "-ac", "1", "-ar", _SAMPLE_RATE, "-f", "wav", str(destination)],
            capture_output=True,
            text=True,
            timeout=self.settings.transcription_timeout_seconds,
        )
        if result.returncode != 0:
            detail = (result.stderr or "").strip().splitlines()
            raise subprocess.SubprocessError(
                f"ffmpeg could not read audio from {path.name}: "
                f"{detail[-1] if detail else f'exit status {result.returncode}'}"
            )

    def transcribe(self, path: Path) -> tuple[str, dict[str, Any]]:
        """Return a recording's speech as timestamped lines, with what was heard about it.

        Raises UnsupportedLanguageError when the speech is not in an accepted language: a
        recording transcribed by the wrong language model produces confident nonsense, which is
        worse in an index than an honest gap.

        The audio is read one window at a time. Decoding a long meeting whole costs hundreds of
        megabytes before the model sees a single word, and the recordings worth transcribing are
        exactly the long ones.
        """
        if not self._has_audio(path):
            raise NoAudioError(f"{path.name} has no audio track")

        model = self._load_model()
        duration = self._duration_seconds(path)
        window = float(self.settings.transcription_window_seconds)
        language = ""
        probability = 0.0
        lines: list[str] = []
        start = 0.0

        while start < max(duration, 1.0):
            with tempfile.TemporaryDirectory() as directory:
                audio = Path(directory) / "audio.wav"
                self._extract_audio(path, audio, start, window)
                if not audio.exists() or audio.stat().st_size <= _WAV_HEADER_BYTES:
                    break
                segments, info = model.transcribe(
                    str(audio),
                    beam_size=self.settings.transcription_beam_size,
                    vad_filter=True,
                )

                if not language:
                    language = (info.language or "").lower()
                    probability = float(info.language_probability or 0.0)
                    accepted = self.settings.transcription_language_list
                    if language not in accepted:
                        raise UnsupportedLanguageError(
                            f"spoken language is {language or 'undetermined'} "
                            f"({probability:.0%} confident), and only "
                            f"{', '.join(accepted)} is indexed"
                        )

                lines.extend(
                    f"[{_timestamp(start + segment.start)}] {text}"
                    for segment in segments
                    if (text := segment.text.strip())
                )
            start += window

        metadata = {
            "source_format": "recording",
            "spoken_language": language,
            "language_probability": round(probability, 3),
            "duration_seconds": round(duration, 1),
            "transcription_model": self.settings.transcription_model,
        }
        return "\n".join(lines), metadata
