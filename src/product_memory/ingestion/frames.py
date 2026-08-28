from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from product_memory.settings import Settings

LOGGER = logging.getLogger(__name__)

# ffmpeg's metadata filter prints one of these per frame it let through.
_PTS_TIME = re.compile(r"pts_time:([0-9.]+)")


@dataclass(slots=True)
class VideoFrame:
    offset_seconds: float
    data: bytes


class FrameSampler:
    """Takes a picture of a recording whenever what is on screen changes.

    A meeting is mostly one still image, so sampling on a clock would store the same slide
    hundreds of times and still miss the moment it changed. Scene detection asks ffmpeg for the
    frames that differ from what came before, and a maximum interval covers a screen that is
    shared unchanged for a long stretch.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def enabled(self) -> bool:
        return self.settings.enable_video_frames

    def sample(self, path: Path) -> list[VideoFrame]:
        if not self.enabled:
            return []
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            times = folder / "times.txt"
            selector = (
                f"gt(scene\\,{self.settings.frame_scene_threshold})"
                f"+gte(t-prev_selected_t\\,{self.settings.frame_max_interval_seconds})"
            )
            result = subprocess.run(
                [
                    "ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path),
                    "-vf",
                    f"select={selector},metadata=print:file={times},"
                    f"scale={self.settings.frame_width}:-2",
                    "-vsync", "vfr", "-q:v", "4",
                    "-frames:v", str(self.settings.frame_max_per_recording),
                    str(folder / "%05d.jpg"),
                ],
                capture_output=True,
                text=True,
                timeout=self.settings.frame_timeout_seconds,
            )
            if result.returncode != 0 or not times.exists():
                LOGGER.warning("Could not read frames from %s", path.name)
                return []
            offsets = [float(value) for value in _PTS_TIME.findall(times.read_text())]
            # ffmpeg writes one timestamp per frame it emitted, in the same order as the files.
            return [
                VideoFrame(offset_seconds=offset, data=image.read_bytes())
                for offset, image in zip(offsets, sorted(folder.glob("*.jpg")), strict=False)
            ]
