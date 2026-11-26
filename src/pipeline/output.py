"""
Structured output: writes processed frames to a directory hierarchy
and produces a per-frame JSON manifest.

Output structure:
    output_dir/
    └── <video_stem>/
        ├── frames/
        │   ├── scene_00/
        │   │   ├── frame_000000.jpg
        │   │   └── frame_000005.jpg
        │   └── scene_01/
        │       └── frame_000025.jpg
        └── manifest.json

Design note: organising frames by scene (rather than a flat directory) means
a downstream CV service can process one scene at a time, reset trackers at
scene boundaries, and skip unwanted segment types — all without reading the
manifest first. The manifest provides the full metadata for every frame.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from pipeline.preprocess import apply_transforms, iter_frames
from pipeline.quality import QualityReport, assess_quality
from pipeline.scenes import Scene

logger = logging.getLogger("pipeline.output")


# ---------------------------------------------------------------------------
# Data model — one record per written frame
# ---------------------------------------------------------------------------

@dataclass
class FrameManifestEntry:
    """Manifest record for a single extracted and processed frame."""
    source_video: str
    frame_index: int
    timestamp_s: float
    scene_id: int
    scene_start_frame: int
    scene_end_frame: int
    output_path: str          # relative to output_dir for portability
    width: int
    height: int
    transformations: list[str]
    blur_score: float
    brightness_mean: float
    quality_passed: bool
    quality_flags: list[str]


# ---------------------------------------------------------------------------
# Main pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    source_path: Path,
    scenes: list[Scene],
    output_dir: Path,
    # Extraction
    mode: str = "interval",
    every_n_frames: int = 5,
    target_fps: float = 2.0,
    # Transforms
    resize_enabled: bool = True,
    target_width: int = 640,
    target_height: int = 360,
    keep_aspect_ratio: bool = True,
    colour_mode: str = "rgb",
    colour_enabled: bool = True,
    # Quality
    blur_threshold: float = 100.0,
    brightness_low: float = 30.0,
    brightness_high: float = 225.0,
    flag_only: bool = True,
    # Output
    output_format: str = "jpg",
    jpg_quality: int = 95,
    save_keyframes_only: bool = False,
    write_manifest: bool = True,
    manifest_filename: str = "manifest.json",
) -> list[FrameManifestEntry]:
    """Run the full pipeline: extract → transform → quality → write → manifest.

    This is the single authoritative write path. Frames are never written
    to disk anywhere else in the pipeline.

    Args:
        source_path:          Path to the source video.
        scenes:               Scene list from detect_scenes / detect_scenes_from_config.
        output_dir:           Root output directory.
        mode:                 Frame extraction mode ("interval" or "fps").
        every_n_frames:       Step size for interval mode.
        target_fps:           Target fps for fps mode.
        resize_enabled:       Whether to resize frames.
        target_width:         Target width after resize.
        target_height:        Target height after resize.
        keep_aspect_ratio:    Letterbox if True, stretch if False.
        colour_mode:          "rgb", "grayscale", or "bgr".
        colour_enabled:       Whether to apply colour conversion.
        blur_threshold:       Laplacian variance threshold.
        brightness_low:       Lower brightness bound.
        brightness_high:      Upper brightness bound.
        flag_only:            Keep flagged frames (True) or discard (False).
        output_format:        Image format ("jpg" or "png").
        jpg_quality:          JPEG quality (0-100).
        save_keyframes_only:  Only write the first frame of each scene.
        write_manifest:       Whether to write manifest.json.
        manifest_filename:    Name for the manifest file.

    Returns:
        List of FrameManifestEntry, one per written frame.
    """
    video_stem = source_path.stem
    video_output_dir = output_dir / video_stem
    video_output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-build a fast frame_index → scene lookup.
    frame_to_scene: dict[int, Scene] = {}
    for scene in scenes:
        for fi in range(scene.start_frame, scene.end_frame + 1):
            frame_to_scene[fi] = scene

    # Determine which frame indices to extract.
    cap_probe = cv2.VideoCapture(str(source_path))
    source_fps = cap_probe.get(cv2.CAP_PROP_FPS) or 25.0
    cap_probe.release()

    if mode == "interval":
        step = max(1, every_n_frames)
    else:
        step = max(1, round(source_fps / target_fps))

    encode_params = []
    ext = output_format.lower()
    if ext == "jpg":
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpg_quality]

    manifest: list[FrameManifestEntry] = []
    written = 0
    discarded = 0

    logger.info(
        "Running full pipeline on '%s' → %s",
        source_path.name, video_output_dir,
    )

    for frame_index, timestamp_s, frame_bgr in iter_frames(
        source_path, mode=mode, every_n_frames=every_n_frames, target_fps=target_fps
    ):
        scene = frame_to_scene.get(frame_index)
        if scene is None:
            # Frame beyond last scene boundary — assign to last scene.
            scene = scenes[-1]

        # Skip non-keyframes if save_keyframes_only is set.
        if save_keyframes_only and frame_index != scene.keyframe_index:
            continue

        # Quality assessment (on raw BGR before transforms for consistency).
        report: QualityReport = assess_quality(
            frame_bgr,
            blur_threshold=blur_threshold,
            brightness_low=brightness_low,
            brightness_high=brightness_high,
        )

        if not report.passed and not flag_only:
            discarded += 1
            logger.debug(
                "Discarded frame %d (quality flags: %s)",
                frame_index, report.flags,
            )
            continue

        # Apply transforms.
        transformed, applied_transforms = apply_transforms(
            frame_bgr,
            resize_enabled=resize_enabled,
            target_width=target_width,
            target_height=target_height,
            keep_aspect_ratio=keep_aspect_ratio,
            colour_mode=colour_mode,
            colour_enabled=colour_enabled,
        )

        # Write frame to structured directory.
        scene_dir = video_output_dir / "frames" / f"scene_{scene.scene_id:02d}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        out_path = scene_dir / f"frame_{frame_index:06d}.{ext}"

        # OpenCV expects BGR for writing — convert back if we switched to RGB.
        write_frame = transformed
        if colour_mode == "rgb" and colour_enabled:
            write_frame = cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(out_path), write_frame, encode_params)

        h, w = transformed.shape[:2]

        manifest.append(FrameManifestEntry(
            source_video=str(source_path.resolve()),
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            scene_id=scene.scene_id,
            scene_start_frame=scene.start_frame,
            scene_end_frame=scene.end_frame,
            output_path=str(out_path.relative_to(output_dir)),
            width=w,
            height=h,
            transformations=applied_transforms,
            blur_score=report.blur_score,
            brightness_mean=report.brightness_mean,
            quality_passed=report.passed,
            quality_flags=report.flags,
        ))
        written += 1

    # Write manifest.
    if write_manifest:
        manifest_path = video_output_dir / manifest_filename
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(
                [asdict(entry) for entry in manifest],
                f,
                indent=2,
            )
        logger.info("Manifest written: %s (%d entries)", manifest_path, len(manifest))

    logger.info(
        "Pipeline complete: %d frames written, %d discarded | '%s'",
        written, discarded, source_path.name,
    )
    return manifest