"""
CV integration: YOLOv8 object detection on pipeline output frames.

Design notes:
    - We use YOLOv8n (nano) as the default model.
    - Frames fed to YOLO must be in RGB format. Our pipeline outputs RGB, 
      so no additional conversion is needed here.
    - Confidence threshold is configurable via config.yaml. 0.25 is
      YOLO's default.
    - The model weights (~6MB for yolov8n) are downloaded automatically
      on first run and cached in the ultralytics cache directory.

Sports-domain note:
    YOLOv8 is trained on COCO (80 classes). For production use, a 
    fine-tuned model on sports footage would significantly improve 
    detection of partially occluded players, referees, and sport-specific 
    equipment.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

# NOTE: torch MUST be imported before cv2 on Windows. The conda-forge OpenCV
# build and the pip torch wheel ship overlapping runtime DLLs; if cv2 loads
# first it claims an incompatible version and torch's shm.dll fails to load
# (WinError 127). Importing torch first establishes the correct DLL load order.
import torch  # noqa: F401  (imported for DLL load order, used transitively)

import cv2
import numpy as np

logger = logging.getLogger("pipeline.inference")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """A single object detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]    # [x1, y1, x2, y2] in pixels


@dataclass
class FrameInferenceResult:
    """All detections for a single frame."""
    frame_path: str
    frame_index: int
    timestamp_s: float
    scene_id: int
    detections: list[Detection] = field(default_factory=list)
    inference_time_ms: float = 0.0

    @property
    def person_count(self) -> int:
        """Number of 'person' detections in this frame."""
        return sum(1 for d in self.detections if d.class_name == "person")


# ---------------------------------------------------------------------------
# Inference runner
# ---------------------------------------------------------------------------

def run_inference(
    manifest_entries: list[dict],
    model_name: str = "yolov8n.pt",
    confidence: float = 0.25,
    device: str = "cpu",
    max_frames: int = 5,
    save_annotated: bool = True,
    output_dir: Path = Path("output"),
) -> list[FrameInferenceResult]:
    """Run YOLOv8 inference on frames from the pipeline manifest.

    Args:
        manifest_entries:  List of manifest entry dicts (from manifest.json).
        model_name:        YOLOv8 model weights file (e.g. 'yolov8n.pt').
        confidence:        Minimum detection confidence threshold.
        device:            Inference device ('cpu' or '0' for GPU).
        max_frames:        Maximum number of frames to process (demo cap).
        save_annotated:    Whether to save annotated frames to disk.
        output_dir:        Root output directory for annotated frames.

    Returns:
        List of FrameInferenceResult, one per processed frame.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics is not installed. Run: pip install ultralytics"
        )

    import time

    logger.info(
        "Loading YOLO model: %s | device=%s | confidence=%.2f",
        model_name, device, confidence,
    )
    model = YOLO(model_name)

    # Limit to max_frames for demo mode.
    entries_to_process = manifest_entries[:max_frames]
    logger.info(
        "Running inference on %d frame(s) (of %d total in manifest)",
        len(entries_to_process), len(manifest_entries),
    )

    results: list[FrameInferenceResult] = []

    for entry in entries_to_process:
        frame_path = Path(entry["output_path"])

        # output_path in manifest is relative to output_dir.
        abs_path = output_dir / frame_path
        if not abs_path.is_file():
            logger.warning("Frame not found, skipping: %s", abs_path)
            continue

        # Load frame — YOLO accepts BGR numpy arrays directly.
        frame_bgr = cv2.imread(str(abs_path))
        if frame_bgr is None:
            logger.warning("Could not read frame: %s", abs_path)
            continue

        # Run inference.
        t0 = time.perf_counter()
        yolo_results = model(
            frame_bgr,
            conf=confidence,
            device=device,
            verbose=False,
        )
        inference_ms = (time.perf_counter() - t0) * 1000.0

        # Parse detections.
        detections: list[Detection] = []
        for result in yolo_results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls[0])
                detections.append(Detection(
                    class_id=class_id,
                    class_name=model.names[class_id],
                    confidence=round(float(box.conf[0]), 4),
                    bbox_xyxy=[round(float(x), 2) for x in box.xyxy[0].tolist()],
                ))

        frame_result = FrameInferenceResult(
            frame_path=str(frame_path),
            frame_index=entry["frame_index"],
            timestamp_s=entry["timestamp_s"],
            scene_id=entry["scene_id"],
            detections=detections,
            inference_time_ms=round(inference_ms, 2),
        )
        results.append(frame_result)

        logger.info(
            "Frame %06d | %d detection(s) | %d person(s) | %.1fms",
            entry["frame_index"],
            len(detections),
            frame_result.person_count,
            inference_ms,
        )

        # Save annotated frame.
        if save_annotated:
            annotated_dir = output_dir / Path(entry["output_path"]).parts[0] / "annotated"
            annotated_dir.mkdir(parents=True, exist_ok=True)
            annotated_path = annotated_dir / f"annotated_{Path(entry['output_path']).name}"

            annotated_frame = yolo_results[0].plot()  # BGR with boxes drawn
            cv2.imwrite(str(annotated_path), annotated_frame)

    # Summary
    total_detections = sum(len(r.detections) for r in results)
    total_persons = sum(r.person_count for r in results)
    avg_ms = sum(r.inference_time_ms for r in results) / max(len(results), 1)

    logger.info(
        "Inference complete: %d frames | %d total detections | "
        "%d persons | avg %.1fms/frame",
        len(results), total_detections, total_persons, avg_ms,
    )

    return results