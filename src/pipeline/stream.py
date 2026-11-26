"""
Streaming input handler: RTSP, HLS, and RTMP sources.

Design note:
    OpenCV's VideoCapture accepts streaming URLs directly via its FFmpeg
    backend — the same backend that handles file-based sources. This means
    our existing frame extraction and pipeline code works unchanged on
    streaming sources; only the source URL format differs.

    Supported URL formats:
        RTSP:  rtsp://username:password@192.168.1.100:554/stream
        HLS:   http://example.com/stream.m3u8  or  /local/path/stream.m3u8
        RTMP:  rtmp://server/live/streamkey

    Known limitations:
        - OpenCV buffers several frames internally for RTSP, introducing
          ~100-500ms latency. For near-real-time analysis, GStreamer or
          a direct FFmpeg subprocess pipe would be more appropriate.
        - RTMP requires an active streaming server (e.g. nginx-rtmp).
          There is no lightweight local test equivalent unlike HLS.
        - The pipeline has a basic retry mechanism (max_retries). A 
          production system would need exponential backoff and reconnect 
          logic.
        - Scene detection on live streams requires a sliding-window
          approach since the full video is not available upfront.

    Sports deployment note:
        For a stadium with multiple IP cameras, each camera would be a
        separate RTSP source. The pipeline would run one instance per
        camera, with results aggregated by a separate coordination layer.
        This is a natural extension of the current single-source design.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np

logger = logging.getLogger("pipeline.stream")


# ---------------------------------------------------------------------------
# Protocol detection
# ---------------------------------------------------------------------------

STREAMING_PREFIXES = ("rtsp://", "rtsps://", "rtmp://", "rtmps://", "http://", "https://")


def is_streaming_url(source: str) -> bool:
    """Return True if the source string looks like a streaming URL.

    Also accepts local .m3u8 paths as HLS streams.

    Args:
        source: Source string — URL or file path.

    Returns:
        True if source is a streaming URL or local HLS playlist.
    """
    source_lower = source.lower()
    if any(source_lower.startswith(prefix) for prefix in STREAMING_PREFIXES):
        return True
    if source_lower.endswith(".m3u8"):
        return True
    return False


def detect_protocol(source: str) -> str:
    """Identify the streaming protocol from a source URL.

    Args:
        source: Streaming URL or local HLS path.

    Returns:
        Protocol string: "rtsp", "hls", "rtmp", or "unknown".
    """
    source_lower = source.lower()
    if source_lower.startswith("rtsp://") or source_lower.startswith("rtsps://"):
        return "rtsp"
    if source_lower.startswith("rtmp://") or source_lower.startswith("rtmps://"):
        return "rtmp"
    if source_lower.endswith(".m3u8") or "m3u8" in source_lower:
        return "hls"
    if source_lower.startswith("http://") or source_lower.startswith("https://"):
        return "hls"
    return "unknown"


# ---------------------------------------------------------------------------
# Stream reader
# ---------------------------------------------------------------------------

@dataclass
class StreamConfig:
    """Configuration for a streaming source."""
    source: str
    protocol: str
    max_frames: Optional[int] = None      # None = read until stream ends
    every_n_frames: int = 5
    connection_timeout_s: float = 10.0
    max_retries: int = 3


def open_stream(source: str, timeout_s: float = 10.0) -> cv2.VideoCapture:
    """Open a streaming source via OpenCV/FFmpeg.

    Args:
        source:    Streaming URL or local HLS path.
        timeout_s: Seconds to wait for the stream to open.

    Returns:
        Opened cv2.VideoCapture object.

    Raises:
        ConnectionError: Stream could not be opened within timeout.
    """
    logger.info("Opening stream: %s", source)

    # For RTSP: set OpenCV to use TCP transport (more reliable than UDP
    # in most network environments; avoids packet loss artefacts).
    if source.lower().startswith("rtsp"):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"H264"))
    else:
        cap = cv2.VideoCapture(source)

    # Poll until the stream opens or timeout expires.
    deadline = time.time() + timeout_s
    while not cap.isOpened():
        if time.time() > deadline:
            cap.release()
            raise ConnectionError(
                f"Could not open stream within {timeout_s}s: {source}"
            )
        time.sleep(0.5)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(
        "Stream opened: %dx%d @ %.2f fps",
        width, height, fps,
    )
    return cap


def iter_stream_frames(
    source: str,
    every_n_frames: int = 5,
    max_frames: Optional[int] = None,
    timeout_s: float = 10.0,
    max_retries: int = 3,
) -> Generator[tuple[int, float, np.ndarray], None, None]:
    """Yield (frame_index, timestamp_s, frame_bgr) from a streaming source.

    Yields every N-th frame to match the file-based extraction behaviour.
    Automatically retries on connection failure up to max_retries times.

    Args:
        source:         Streaming URL or local HLS path.
        every_n_frames: Yield every N-th frame (1 = every frame).
        max_frames:     Stop after yielding this many frames (None = no limit).
        timeout_s:      Connection timeout in seconds.
        max_retries:    Number of reconnection attempts on failure.

    Yields:
        Tuples of (frame_index, timestamp_s, frame_bgr).
    """
    protocol = detect_protocol(source)
    logger.info(
        "Stream reader: protocol=%s every_n=%d max_frames=%s | %s",
        protocol, every_n_frames, max_frames, source,
    )

    attempt = 0
    while attempt <= max_retries:
        try:
            cap = open_stream(source, timeout_s)
        except ConnectionError as exc:
            attempt += 1
            if attempt > max_retries:
                raise
            logger.warning(
                "Stream connection failed (attempt %d/%d): %s — retrying...",
                attempt, max_retries, exc,
            )
            time.sleep(2.0)
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_index = 0
        yielded = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.info(
                        "Stream ended at frame %d (yielded %d)",
                        frame_index, yielded,
                    )
                    break

                if frame_index % every_n_frames == 0:
                    timestamp_s = round(frame_index / fps, 4)
                    yield frame_index, timestamp_s, frame
                    yielded += 1

                    if max_frames is not None and yielded >= max_frames:
                        logger.info(
                            "Reached max_frames=%d — stopping stream reader",
                            max_frames,
                        )
                        break

                frame_index += 1

        finally:
            cap.release()

        break 

    logger.info("Stream reader finished.")