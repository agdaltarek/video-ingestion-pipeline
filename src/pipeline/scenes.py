"""
Scene / shot-boundary detection.

Implementation: custom histogram difference with adaptive thresholding + (optional) PySceneDetect

Algorithm overview:
    1. For each frame, compute a normalised per-channel colour histogram.
    2. Compare consecutive histograms using chi-squared distance.
    3. A shot boundary is declared when the distance exceeds a threshold.
    4. In adaptive mode, the threshold is set relative to a rolling mean of
       recent distances — this handles gradual lighting changes (e.g. clouds
       passing over an outdoor pitch) that would fool a fixed threshold.
    5. A minimum scene length guard filters false positives from fast camera
       pans, which produce a large histogram spike mid-shot.

Why chi-squared distance?
    Chi-squared is more sensitive to differences in low-frequency (dark) bins
    than L1/L2 norms, making it better suited to detecting cuts between shots
    that share similar overall brightness but differ in content distribution.
    This is a standard choice in the shot-boundary detection literature
    (Lienhart, 1999; Smeaton et al., 2010).

Limitations:
    - Fast pans produce false positives even with the min_scene_length guard.
    - Dissolve / fade transitions are missed — chi-squared distance rises
      gradually rather than spiking, staying below the threshold throughout.
    - Threshold calibration is content-dependent; learned detectors
      (e.g. TransNet v2) are more robust across broadcast styles.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("pipeline.scenes")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SceneBoundary:
    """A detected shot boundary."""
    frame_index: int       # first frame of the NEW shot
    timestamp_s: float
    distance_score: float  # chi-squared distance that triggered detection


@dataclass
class Scene:
    """A single shot/scene segment."""
    scene_id: int
    start_frame: int
    end_frame: int          # inclusive
    start_time_s: float
    end_time_s: float
    frame_count: int
    keyframe_index: int     # first frame of the scene — best representative


# ---------------------------------------------------------------------------
# Histogram helpers
# ---------------------------------------------------------------------------

def _compute_histogram(frame: np.ndarray, bins: int = 64) -> np.ndarray:
    """Compute a normalised per-channel histogram for a BGR frame.

    Args:
        frame: BGR frame (H x W x 3).
        bins:  Number of bins per channel.

    Returns:
        Concatenated normalised histogram (bins * 3,).
    """
    hists = []
    for ch in range(3):
        hist = cv2.calcHist(
            [frame], [ch], None, [bins], [0, 256]
        ).flatten()
        hist /= (hist.sum() + 1e-7)  # normalise; 1e-7 : small number to avoid dividing by zero
        hists.append(hist)
    return np.concatenate(hists)


def _chi_squared_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """Compute chi-squared distance between two normalised histograms.

    Chi-squared distance: sum((h1 - h2)^2 / (h1 + h2 + eps))

    Args:
        h1, h2: Normalised histogram vectors of equal length.

    Returns:
        Scalar distance value (higher = more different).
    """
    eps = 1e-10
    return float(np.sum((h1 - h2) ** 2 / (h1 + h2 + eps)))


# ---------------------------------------------------------------------------
# Core detector
# ---------------------------------------------------------------------------

def detect_scenes(
    source_path: Path,
    bins: int = 64,
    threshold: float = 0.4,
    adaptive: bool = True,
    adaptive_window: int = 10,
    min_scene_length: int = 15,
    min_absolute_threshold: float = 0.05,
    source_fps: Optional[float] = None,
) -> tuple[list[SceneBoundary], list[Scene]]:
    """Detect shot boundaries in a video using histogram difference.

    Args:
        source_path:       Path to the video file.
        bins:              Histogram bins per channel.
        threshold:         Chi-squared distance threshold for boundary detection.
                           In adaptive mode this becomes a multiplier on the
                           rolling mean (e.g. 0.4 → boundary if distance is
                           40% above the rolling mean baseline).
        adaptive:          Whether to use adaptive (rolling-mean) thresholding.
        adaptive_window:   Rolling window size in frames for adaptive mode.
        min_scene_length:  Minimum frames between consecutive boundaries.
        source_fps:        Source frame rate (auto-detected if None).

    Returns:
        Tuple of (list of SceneBoundary, list of Scene).
    """
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {source_path.name}")

    fps = source_fps or cap.get(cv2.CAP_PROP_FPS) or 25.0

    logger.info(
        "Scene detection: method=histogram bins=%d threshold=%.2f "
        "adaptive=%s min_scene=%d frames min_abs=%.3f | '%s'",
        bins, threshold, adaptive, min_scene_length,
        min_absolute_threshold, source_path.name,
    )

    distances: list[float] = []
    boundaries: list[SceneBoundary] = []

    prev_hist: Optional[np.ndarray] = None
    frame_index = 0
    last_boundary_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hist = _compute_histogram(frame, bins)

        if prev_hist is not None:
            dist = _chi_squared_distance(prev_hist, hist)
            distances.append(dist)

            # Determine effective threshold.
            if adaptive and len(distances) >= adaptive_window:
                window = distances[-adaptive_window:]
                rolling_mean = float(np.mean(window))
                effective_threshold = rolling_mean * (1.0 + threshold)
            else:
                effective_threshold = threshold

            # Apply absolute floor — prevents the adaptive threshold from
            # collapsing to near-zero on static or low-motion content,
            # which would cause false positives on minor pixel-level changes.
            effective_threshold = max(effective_threshold, min_absolute_threshold)

            gap = frame_index - last_boundary_frame
            if dist > effective_threshold and gap >= min_scene_length:
                boundaries.append(SceneBoundary(
                    frame_index=frame_index,
                    timestamp_s=round(frame_index / fps, 4),
                    distance_score=round(dist, 6),
                ))
                last_boundary_frame = frame_index
                logger.debug(
                    "Boundary at frame %d (%.2fs) dist=%.4f threshold=%.4f",
                    frame_index, frame_index / fps, dist, effective_threshold,
                )

        prev_hist = hist
        frame_index += 1

    cap.release()
    total_frames = frame_index

    # Build Scene objects from boundary list.
    scenes = _boundaries_to_scenes(boundaries, total_frames, fps)

    logger.info(
        "Detected %d scene(s) from %d boundaries in '%s'",
        len(scenes), len(boundaries), source_path.name,
    )
    return boundaries, scenes


# ---------------------------------------------------------------------------
# Scene builder
# ---------------------------------------------------------------------------

def _boundaries_to_scenes(
    boundaries: list[SceneBoundary],
    total_frames: int,
    fps: float,
) -> list[Scene]:
    """Convert a list of shot boundaries into Scene objects.

    Args:
        boundaries:    Detected shot boundaries (sorted by frame_index).
        total_frames:  Total number of frames in the video.
        fps:           Frame rate (for timestamp calculation).

    Returns:
        List of Scene objects covering the full video.
    """
    # Build cut points: start of video + each boundary + end of video.
    cut_points = [0] + [b.frame_index for b in boundaries] + [total_frames]

    scenes: list[Scene] = []
    for scene_id, (start, end) in enumerate(zip(cut_points, cut_points[1:])):
        scenes.append(Scene(
            scene_id=scene_id,
            start_frame=start,
            end_frame=end - 1,
            start_time_s=round(start / fps, 4),
            end_time_s=round((end - 1) / fps, 4),
            frame_count=end - start,
            keyframe_index=start,
        ))

    return scenes

# ---------------------------------------------------------------------------
# PySceneDetect
# ---------------------------------------------------------------------------

def _detect_scenes_pyscenedetect(
    source_path: Path,
    threshold: float = 27.0,
    min_scene_length: int = 15,
    source_fps: Optional[float] = None,
) -> tuple[list[SceneBoundary], list[Scene]]:
    """Detect shot boundaries using PySceneDetect's ContentDetector.

    This is a fallback / validation path. It uses the same ContentDetector
    algorithm (HSV colour difference) as PySceneDetect's default, wraps the
    results into our own SceneBoundary / Scene data models, and returns the
    same structure as the custom histogram path so the rest of the pipeline
    is agnostic to which method was used.

    Note: PySceneDetect's threshold scale is different from our chi-squared
    scale. A value of 27.0 is PySceneDetect's recommended default for
    ContentDetector on broadcast footage.

    Args:
        source_path:       Path to the video file.
        threshold:         ContentDetector threshold (PySceneDetect scale).
        min_scene_length:  Minimum scene length in frames.
        source_fps:        Source frame rate (auto-detected if None).

    Returns:
        Tuple of (list of SceneBoundary, list of Scene).
    """
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector
    except ImportError:
        raise ImportError(
            "PySceneDetect is not installed. "
            "Run: pip install scenedetect[opencv]"
        )

    logger.info(
        "Scene detection: method=pyscenedetect threshold=%.1f "
        "min_scene=%d | '%s'",
        threshold, min_scene_length, source_path.name,
    )

    video = open_video(str(source_path))
    fps = source_fps or video.frame_rate or 25.0

    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(
            threshold=threshold,
            min_scene_len=min_scene_length,
        )
    )
    scene_manager.detect_scenes(video)

    raw_scenes = scene_manager.get_scene_list()

    # Convert PySceneDetect FrameTimecode pairs → our data models.
    boundaries: list[SceneBoundary] = []
    scenes: list[Scene] = []

    # If no cuts detected, PySceneDetect returns an empty list.
    # Fall back to a single scene covering the full video.
    if not raw_scenes:
        total_frames = int(video.duration.frame_num)
        scenes.append(Scene(
            scene_id=0,
            start_frame=0,
            end_frame=total_frames - 1,
            start_time_s=0.0,
            end_time_s=round((total_frames - 1) / fps, 4),
            frame_count=total_frames,
            keyframe_index=0,
        ))
        logger.info(
            "No cuts detected — returning single scene covering full video '%s'",
            source_path.name,
        )
        return boundaries, scenes

    for scene_id, (start_tc, end_tc) in enumerate(raw_scenes):
        start_frame = start_tc.get_frames()
        end_frame = end_tc.get_frames() - 1

        # Every scene start after the first is a boundary.
        if scene_id > 0:
            boundaries.append(SceneBoundary(
                frame_index=start_frame,
                timestamp_s=round(start_frame / fps, 4),
                distance_score=0.0,  # PySceneDetect doesn't expose raw distance
            ))

        scenes.append(Scene(
            scene_id=scene_id,
            start_frame=start_frame,
            end_frame=end_frame,
            start_time_s=round(start_frame / fps, 4),
            end_time_s=round(end_frame / fps, 4),
            frame_count=end_frame - start_frame + 1,
            keyframe_index=start_frame,
        ))

    logger.info(
        "Detected %d scene(s) from %d boundaries in '%s'",
        len(scenes), len(boundaries), source_path.name,
    )
    return boundaries, scenes


# ---------------------------------------------------------------------------
# Public dispatcher — routes to custom or PySceneDetect based on config
# ---------------------------------------------------------------------------

def detect_scenes_from_config(
    source_path: Path,
    scene_cfg: dict,
    source_fps: Optional[float] = None,
) -> tuple[list[SceneBoundary], list[Scene]]:
    """Dispatch scene detection to the configured method.

    Args:
        source_path: Path to the video file.
        scene_cfg:   The 'scenes' section of config.yaml as a dict.
        source_fps:  Source frame rate (auto-detected if None).

    Returns:
        Tuple of (list of SceneBoundary, list of Scene).
    """
    method = scene_cfg.get("method", "histogram")

    if method == "histogram":
        hist_cfg = scene_cfg.get("histogram", {})
        return detect_scenes(
            source_path=source_path,
            bins=int(hist_cfg.get("bins", 64)),
            threshold=float(hist_cfg.get("threshold", 0.4)),
            adaptive=bool(hist_cfg.get("adaptive", True)),
            adaptive_window=int(hist_cfg.get("adaptive_window", 10)),
            min_scene_length=int(hist_cfg.get("min_scene_length", 15)),
            min_absolute_threshold=float(hist_cfg.get("min_absolute_threshold", 0.05)),
            source_fps=source_fps,
        )

    if method == "pyscenedetect":
        psd_cfg = scene_cfg.get("pyscenedetect", {})
        return _detect_scenes_pyscenedetect(
            source_path=source_path,
            threshold=float(psd_cfg.get("threshold", 27.0)),
            min_scene_length=int(psd_cfg.get("min_scene_length", 15)),
            source_fps=source_fps,
        )

    raise ValueError(
        f"Unknown scene detection method: '{method}'. "
        "Use 'histogram' or 'pyscenedetect'."
    )