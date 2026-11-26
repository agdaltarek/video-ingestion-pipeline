"""
Input handling: format detection, metadata extraction via ffprobe,
and error handling for corrupt or truncated files.

Design note: ffprobe is used as the authoritative metadata source (not OpenCV)
because it reads container and stream headers directly via libavformat, giving
accurate codec parameters, fractional frame rates, and embedded tags even for
formats OpenCV cannot open at all.
"""

import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("pipeline.ingest")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class VideoMetadata:
    """All metadata extracted from a single video file."""
    source_path: str
    container_format: str       # e.g. "mov,mp4,m4a,3gp,3g2,mj2"
    duration_s: float
    size_bytes: int
    video_codec: str            # e.g. "h264", "hevc", "mjpeg"
    width: int
    height: int
    fps: float
    total_frames: Optional[int]  # absent for some containers / streams
    audio_codec: Optional[str]
    bit_rate_kbps: Optional[float]
    tags: dict = field(default_factory=dict)  # GPS, camera ID, creation time, etc.


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class IngestError(Exception):
    """Raised when a video file cannot be ingested (corrupt, unsupported, etc.)."""
    pass


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def probe_video(path: Path) -> VideoMetadata:
    """Extract metadata from a video file using ffprobe.

    Args:
        path: Path to the video file.

    Returns:
        Populated VideoMetadata dataclass.

    Raises:
        FileNotFoundError: File does not exist on disk.
        IngestError: File is corrupt, truncated, not a video, or ffprobe failed.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    logger.debug("Probing: %s", path.name)

    cmd = [
        "ffprobe",
        "-v", "quiet",           # suppress banner / progress
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True, #return output as a string
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        raise IngestError(f"ffprobe timed out reading '{path.name}'")
    except FileNotFoundError:
        raise IngestError(
            "ffprobe executable not found — "
            "activate the conda environment: conda activate video-ingest"
        )

    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise IngestError(
            f"ffprobe could not read '{path.name}': {stderr or 'unknown error'}"
        )

    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise IngestError(
            f"ffprobe returned malformed JSON for '{path.name}': {exc}"
        )

    return _parse_ffprobe_output(path, info)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_ffprobe_output(path: Path, info: dict) -> VideoMetadata:
    """Build a VideoMetadata instance from a parsed ffprobe JSON dict."""
    fmt = info.get("format", {})
    streams = info.get("streams", [])

    video_stream = next(
        (s for s in streams if s.get("codec_type") == "video"), None
    )
    audio_stream = next(
        (s for s in streams if s.get("codec_type") == "audio"), None
    )

    if video_stream is None:
        raise IngestError(f"No video stream found in '{path.name}'")

    # Fractional fps strings like "30000/1001" are common in broadcast footage.
    fps = _parse_fraction(
        video_stream.get("avg_frame_rate")
        or video_stream.get("r_frame_rate", "0/1")
    )

    nb_frames = video_stream.get("nb_frames", "")
    total_frames = int(nb_frames) if nb_frames.isdigit() else None

    bit_rate = fmt.get("bit_rate")
    bit_rate_kbps = float(bit_rate) / 1000.0 if bit_rate else None

    # Merge container-level and stream-level tags (GPS, camera ID, etc.)
    tags: dict = {}
    tags.update(fmt.get("tags", {}))
    tags.update(video_stream.get("tags", {}))

    meta = VideoMetadata(
        source_path=str(path.resolve()),
        container_format=fmt.get("format_name", "unknown"),
        duration_s=float(fmt.get("duration", 0.0)),
        size_bytes=int(fmt.get("size", 0)),
        video_codec=video_stream.get("codec_name", "unknown"),
        width=int(video_stream.get("width", 0)),
        height=int(video_stream.get("height", 0)),
        fps=fps,
        total_frames=total_frames,
        audio_codec=audio_stream.get("codec_name") if audio_stream else None,
        bit_rate_kbps=bit_rate_kbps,
        tags=tags,
    )

    logger.info(
        "Probed '%s': %s  %dx%d  %.2f fps  %.1fs",
        path.name,
        meta.video_codec,
        meta.width,
        meta.height,
        meta.fps,
        meta.duration_s,
    )
    return meta


def _parse_fraction(fraction_str: str) -> float:
    """Parse an ffprobe fraction string like '30000/1001' or '25' into a float.

    Args:
        fraction_str: Numerator/denominator string, or a plain numeric string.

    Returns:
        Float value, or 0.0 if parsing fails.
    """
    try:
        if "/" in fraction_str:
            num, den = fraction_str.split("/")
            return float(num) / float(den) if float(den) != 0.0 else 0.0
        return float(fraction_str)
    except (ValueError, ZeroDivisionError):
        return 0.0