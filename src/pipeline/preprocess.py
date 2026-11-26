"""
Pre-processing pipeline: frame extraction, transforms.

Design note: OpenCV's VideoCapture is used for frame decoding here because
it provides per-frame random access and is sufficient for file-based sources.
For streaming sources, we will switch to a subprocess-based FFmpeg
pipe to avoid OpenCV's RTSP buffering limitations.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import cv2
import numpy as np

logger = logging.getLogger("pipeline.preprocess")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FrameRecord:
    """Represents a single extracted frame and its associated metadata."""
    source_path: str
    frame_index: int          # original frame index in the source video
    timestamp_s: float        # position in the source video (seconds)
    output_path: str          # path where the frame was written to disk
    width: int
    height: int
    transformations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames(
    source_path: Path,
    output_dir: Path,
    mode: str = "interval",
    every_n_frames: int = 5,
    target_fps: float = 2.0,
) -> list[FrameRecord]:
    """Extract frames from a video file and write them to disk.

    Output structure:
        output_dir / <video_stem> / frames / frame_XXXXXX.jpg

    Args:
        source_path:    Path to the source video file.
        output_dir:     Root directory for all pipeline output.
        mode:           "interval" — every N-th frame.
                        "fps"      — resample to a target frame rate.
        every_n_frames: Frame step when mode = "interval".
        target_fps:     Target frame rate when mode = "fps".

    Returns:
        List of FrameRecord objects, one per extracted frame.

    Raises:
        FileNotFoundError: Source video does not exist.
        ValueError:         Unsupported mode, or video cannot be opened.
    """
    if not source_path.is_file():
        raise FileNotFoundError(f"Video not found: {source_path}")

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video: {source_path.name}")

    source_fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Determine which frame indices to keep.
    if mode == "interval":
        step = max(1, every_n_frames)
        logger.info(
            "Extraction mode: every %d frame(s) from '%s'",
            step, source_path.name,
        )
    elif mode == "fps":
        step = max(1, round(source_fps / target_fps))
        logger.info(
            "Extraction mode: target %.2f fps → step %d (source %.2f fps) from '%s'",
            target_fps, step, source_fps, source_path.name,
        )
    else:
        raise ValueError(f"Unknown extraction mode: '{mode}'. Use 'interval' or 'fps'.")

    # Prepare output directory.
    frame_dir = output_dir / source_path.stem / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    records: list[FrameRecord] = []
    frame_index = 0
    written = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % step == 0:
            timestamp_s = frame_index / source_fps
            out_path = frame_dir / f"frame_{frame_index:06d}.jpg"

            cv2.imwrite(str(out_path), frame)

            h, w = frame.shape[:2]
            records.append(
                FrameRecord(
                    source_path=str(source_path.resolve()),
                    frame_index=frame_index,
                    timestamp_s=round(timestamp_s, 4),
                    output_path=str(out_path.resolve()),
                    width=w,
                    height=h,
                )
            )
            written += 1

        frame_index += 1

    cap.release()

    logger.info(
        "Extracted %d frames from '%s' (%d total frames scanned)",
        written, source_path.name, frame_index,
    )
    return records


# ---------------------------------------------------------------------------
# Frame generator (memory-efficient alternative — used by benchmarking later)
# ---------------------------------------------------------------------------

def iter_frames(
    source_path: Path,
    mode: str = "interval",
    every_n_frames: int = 5,
    target_fps: float = 2.0,
) -> Generator[tuple[int, float, np.ndarray], None, None]:
    """Yield (frame_index, timestamp_s, frame_bgr) without writing to disk.

    Useful for in-memory pipelines and benchmarking.
    """
    if not source_path.is_file():
        raise FileNotFoundError(f"Video not found: {source_path}")

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open: {source_path.name}")

    source_fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if mode == "interval":
        step = max(1, every_n_frames)
    elif mode == "fps":
        step = max(1, round(source_fps / target_fps))
    else:
        raise ValueError(f"Unknown mode: '{mode}'")

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % step == 0:
            yield frame_index, round(frame_index / source_fps, 4), frame
        frame_index += 1

    cap.release()


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def resize_frame(
    frame: np.ndarray,
    target_width: int,
    target_height: int,
    keep_aspect_ratio: bool = True,
) -> np.ndarray:
    """Resize a frame to the target dimensions.

    When keep_aspect_ratio is True the frame is scaled to fit inside the target
    box and the remaining area is padded with black pixels (letterbox / pillarbox).
    This preserves spatial proportions, which matters downstream because object
    detectors are sensitive to aspect-ratio distortion.

    Args:
        frame:             Input BGR frame (H x W x 3 or H x W for grayscale).
        target_width:      Desired output width in pixels.
        target_height:     Desired output height in pixels.
        keep_aspect_ratio: If True, letterbox/pillarbox; if False, stretch.

    Returns:
        Resized (and optionally padded) frame as a numpy array.
    """
    h, w = frame.shape[:2]

    if not keep_aspect_ratio:
        return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    # Compute scale so the image fits inside target box without cropping.
    scale = min(target_width / w, target_height / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Determine number of channels for the padded canvas.
    channels = 1 if resized.ndim == 2 else resized.shape[2]
    if channels == 1:
        canvas = np.zeros((target_height, target_width), dtype=np.uint8)
    else:
        canvas = np.zeros((target_height, target_width, channels), dtype=np.uint8)

    # Centre the resized image on the canvas.
    pad_top = (target_height - new_h) // 2
    pad_left = (target_width - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    return canvas


def convert_colour_space(frame: np.ndarray, mode: str) -> np.ndarray:
    """Convert a BGR frame to the requested colour space.

    OpenCV reads video frames as BGR by default. Most CV models (YOLO, etc.)
    expect RGB. Grayscale halves memory/bandwidth for pipelines that don't
    need colour.

    Args:
        frame: Input BGR frame (H x W x 3).
        mode:  "rgb"       — convert BGR → RGB (channel order swap only).
               "grayscale" — convert BGR → single-channel grayscale.
               "bgr"       — no-op; return frame unchanged.

    Returns:
        Converted frame as a numpy array.

    Raises:
        ValueError: Unknown colour space mode.
    """
    if mode == "bgr":
        return frame
    if mode == "rgb":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if mode == "grayscale":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unknown colour space mode: '{mode}'. Use 'rgb', 'grayscale', or 'bgr'.")


def apply_transforms(
    frame: np.ndarray,
    resize_enabled: bool = True,
    target_width: int = 640,
    target_height: int = 360,
    keep_aspect_ratio: bool = True,
    colour_mode: str = "rgb",
    colour_enabled: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Apply the configured transform chain to a single frame.

    Transforms are applied in a fixed order: resize first, then colour
    conversion. Resizing before colour conversion is more efficient because
    the spatial reduction shrinks the number of pixels the colour maths
    needs to touch.

    Args:
        frame:             Input BGR frame.
        resize_enabled:    Whether to apply resizing.
        target_width:      Target width (used if resize_enabled).
        target_height:     Target height (used if resize_enabled).
        keep_aspect_ratio: Letterbox/pillarbox vs stretch.
        colour_mode:       Target colour space.
        colour_enabled:    Whether to apply colour conversion.

    Returns:
        Tuple of (transformed frame, list of transform description strings).
    """
    applied: list[str] = []

    if resize_enabled:
        frame = resize_frame(frame, target_width, target_height, keep_aspect_ratio)
        applied.append(f"resize:{target_width}x{target_height}:keep_aspect={keep_aspect_ratio}")

    if colour_enabled:
        frame = convert_colour_space(frame, colour_mode)
        applied.append(f"colour_space:{colour_mode}")

    return frame, applied