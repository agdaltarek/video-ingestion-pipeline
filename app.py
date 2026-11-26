"""
Interactive demo interface for the Video Ingestion Pipeline.

Usage:
    python app.py
    Open http://localhost:7860 in your browser.
"""

import os
import json
import tempfile
import shutil
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch  # must import before cv2 on Windows
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline.logging_setup import configure_logging
from pipeline.ingest import probe_video, IngestError
from pipeline.scenes import detect_scenes_from_config
from pipeline.output import run_pipeline
from pipeline.inference import run_inference

logger = configure_logging("INFO")

# ---------------------------------------------------------------------------
# Output directory for the app
# ---------------------------------------------------------------------------

APP_OUTPUT_DIR = Path("output") / "app_demo"
APP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Core pipeline function
# ---------------------------------------------------------------------------

def run_pipeline_ui(
    video_file,
    extraction_mode,
    every_n_frames,
    target_fps,
    target_width,
    target_height,
    keep_aspect_ratio,
    colour_mode,
    blur_threshold,
    brightness_low,
    brightness_high,
    flag_only,
    scene_method,
    scene_threshold,
    scene_min_length,
    run_inference_flag,
    yolo_confidence,
    yolo_max_frames,
    progress=gr.Progress(),
):
    """Run the full pipeline and return results for the Gradio UI."""

    if video_file is None:
        return (
            "No video uploaded.", "", "", None, None, None, None, None
        )

    try:
        source_path = Path(video_file)
        progress(0.05, desc="Probing video metadata...")

        # --- Ingest ---
        try:
            meta = probe_video(source_path)
        except (FileNotFoundError, IngestError) as exc:
            return (str(exc), "", "", None, None, None, None, None)

        metadata_text = (
            f"Filename     : {source_path.name}\n"
            f"Container    : {meta.container_format}\n"
            f"Codec        : {meta.video_codec}\n"
            f"Resolution   : {meta.width}x{meta.height}\n"
            f"Frame rate   : {meta.fps:.2f} fps\n"
            f"Duration     : {meta.duration_s:.1f}s\n"
            f"Bit rate     : {f'{meta.bit_rate_kbps:.1f} kbps' if meta.bit_rate_kbps else 'n/a'}\n"
            f"Audio codec  : {meta.audio_codec or 'none'}\n"
            f"Tags         : {meta.tags or '{}'}"
        )

        progress(0.15, desc="Detecting scenes...")

        # --- Scene detection ---
        scene_cfg = {
            "method": scene_method,
            "histogram": {
                "bins": 64,
                "threshold": scene_threshold,
                "adaptive": True,
                "adaptive_window": 10,
                "min_scene_length": scene_min_length,
                "min_absolute_threshold": 0.05,
            },
            "pyscenedetect": {
                "threshold": 27.0,
                "min_scene_length": scene_min_length,
            },
        }

        _, scenes = detect_scenes_from_config(
            source_path, scene_cfg, source_fps=meta.fps
        )

        scenes_text = f"Scenes detected: {len(scenes)}\n\n"
        for s in scenes:
            scenes_text += (
                f"  Scene {s.scene_id:02d} | "
                f"frames {s.start_frame}–{s.end_frame} | "
                f"{s.start_time_s:.2f}s – {s.end_time_s:.2f}s | "
                f"{s.frame_count} frames\n"
            )

        progress(0.30, desc="Extracting and processing frames...")

        # --- Full pipeline ---
        video_output_dir = APP_OUTPUT_DIR / source_path.stem
        if video_output_dir.exists():
            shutil.rmtree(video_output_dir)

        manifest = run_pipeline(
            source_path=source_path,
            scenes=scenes,
            output_dir=APP_OUTPUT_DIR,
            mode=extraction_mode,
            every_n_frames=int(every_n_frames),
            target_fps=float(target_fps),
            resize_enabled=True,
            target_width=int(target_width),
            target_height=int(target_height),
            keep_aspect_ratio=keep_aspect_ratio,
            colour_mode=colour_mode,
            colour_enabled=True,
            blur_threshold=float(blur_threshold),
            brightness_low=float(brightness_low),
            brightness_high=float(brightness_high),
            flag_only=flag_only,
            output_format="jpg",
            jpg_quality=95,
            write_manifest=True,
            manifest_filename="manifest.json",
        )

        progress(0.60, desc="Generating quality charts...")

        # --- Quality charts ---
        blur_scores  = [e.blur_score for e in manifest]
        brightness   = [e.brightness_mean for e in manifest]
        passed       = [e.quality_passed for e in manifest]
        n_passed     = sum(passed)
        n_flagged    = len(passed) - n_passed

        quality_summary = (
            f"Total frames extracted : {len(manifest)}\n"
            f"Frames passed QA       : {n_passed}\n"
            f"Frames flagged QA      : {n_flagged}\n"
            f"Blur  — min: {min(blur_scores):.1f}  "
            f"max: {max(blur_scores):.1f}  "
            f"avg: {sum(blur_scores)/len(blur_scores):.1f}\n"
            f"Brightness — min: {min(brightness):.1f}  "
            f"max: {max(brightness):.1f}  "
            f"avg: {sum(brightness)/len(brightness):.1f}"
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
        colours = ["green" if p else "red" for p in passed]
        ax1.bar(range(len(blur_scores)), blur_scores, color=colours, alpha=0.7)
        ax1.axhline(y=float(blur_threshold), color="red",
                    linestyle="--", label=f"threshold ({blur_threshold})")
        ax1.set_title("Laplacian Blur Score per Frame")
        ax1.set_xlabel("Frame (extracted index)")
        ax1.set_ylabel("Blur score (higher = sharper)")
        ax1.legend()

        ax2.bar(range(len(brightness)), brightness,
                color="steelblue", alpha=0.7)
        ax2.axhline(y=float(brightness_low), color="orange",
                    linestyle="--", label=f"dark ({brightness_low})")
        ax2.axhline(y=float(brightness_high), color="red",
                    linestyle="--", label=f"overexposed ({brightness_high})")
        ax2.set_title("Mean Brightness per Frame")
        ax2.set_xlabel("Frame (extracted index)")
        ax2.set_ylabel("Mean pixel intensity (0–255)")
        ax2.legend()
        plt.tight_layout()

        progress(0.70, desc="Loading sample frames...")

        # --- Sample frames gallery ---
        sample_frames = []
        for entry in manifest[:12]:
            img_path = APP_OUTPUT_DIR / entry.output_path
            if img_path.is_file():
                sample_frames.append(str(img_path))

        progress(0.80, desc="Running YOLO inference...")

        # --- Inference ---
        annotated_images = []
        inference_text   = "Inference not run."

        if run_inference_flag and len(manifest) > 0:
            manifest_path = (
                APP_OUTPUT_DIR / source_path.stem / "manifest.json"
            )
            with manifest_path.open() as f:
                manifest_dicts = json.load(f)

            inference_results = run_inference(
                manifest_entries=manifest_dicts,
                model_name="yolov8n.pt",
                confidence=float(yolo_confidence),
                device="cpu",
                max_frames=int(yolo_max_frames),
                save_annotated=True,
                output_dir=APP_OUTPUT_DIR,
            )

            annotated_dir = (
                APP_OUTPUT_DIR / source_path.stem / "annotated"
            )
            if annotated_dir.exists():
                annotated_images = [
                    str(p) for p in sorted(annotated_dir.glob("*.jpg"))
                ]

            total_det  = sum(len(r.detections) for r in inference_results)
            total_pers = sum(r.person_count for r in inference_results)
            avg_ms     = sum(r.inference_time_ms for r in inference_results) / max(len(inference_results), 1)

            inference_text = (
                f"Frames processed : {len(inference_results)}\n"
                f"Total detections : {total_det}\n"
                f"Persons detected : {total_pers}\n"
                f"Avg infer time   : {avg_ms:.1f}ms/frame\n\n"
            )
            for r in inference_results:
                classes = ", ".join(
                    f"{d.class_name}({d.confidence:.2f})"
                    for d in r.detections
                ) or "none"
                inference_text += (
                    f"  Frame {r.frame_index:06d} | "
                    f"t={r.timestamp_s:.2f}s | "
                    f"{len(r.detections)} detections | "
                    f"{classes}\n"
                )

        progress(1.0, desc="Done.")

        # --- Flagged frames ---
        flagged_frames = []
        flagged_by_type = {"blurry": 0, "too_dark": 0, "overexposed": 0}

        for entry in manifest:
            if not entry.quality_passed:
                img_path = APP_OUTPUT_DIR / entry.output_path
                if img_path.is_file():
                    flagged_frames.append(str(img_path))
                for flag in entry.quality_flags:
                    if flag in flagged_by_type:
                        flagged_by_type[flag] += 1

        flagged_summary_text = (
            f"Total flagged frames : {len(flagged_frames)}\n"
            f"  Blurry             : {flagged_by_type['blurry']}\n"
            f"  Too dark           : {flagged_by_type['too_dark']}\n"
            f"  Overexposed        : {flagged_by_type['overexposed']}\n"
        )

        # --- Scene dropdown choices ---
        scene_choices = [
            f"Scene {s.scene_id:02d} | "
            f"{s.start_time_s:.2f}s – {s.end_time_s:.2f}s | "
            f"{s.frame_count} frames"
            for s in scenes
        ]

        # Serialise manifest for state storage
        manifest_json = json.dumps([
            {
                "scene_id"      : e.scene_id,
                "frame_index"   : e.frame_index,
                "timestamp_s"   : e.timestamp_s,
                "output_path"   : e.output_path,
                "quality_passed": e.quality_passed,
                "quality_flags" : e.quality_flags,
                "blur_score"    : e.blur_score,
                "brightness_mean": e.brightness_mean,
            }
            for e in manifest
        ])

        return (
            metadata_text,
            scenes_text,
            quality_summary,
            fig,
            sample_frames,
            flagged_summary_text,
            flagged_frames,
            annotated_images,
            inference_text,
            gr.Dropdown(choices=scene_choices, value=scene_choices[0] if scene_choices else None),
            manifest_json,
            None,
        )

    except Exception as exc:
        import traceback
        return (
            f"Pipeline error: {exc}\n\n{traceback.format_exc()}",
            "", "", None, None, "", None, None, "",
            gr.Dropdown(choices=[]), None, None,
        )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def load_scene_frames(scene_label, manifest_data_json):
    """Load frames for a selected scene from the manifest."""
    if not scene_label or not manifest_data_json:
        return [], ""

    try:
        manifest_data = json.loads(manifest_data_json)
        scene_id = int(scene_label.split("Scene ")[1].split(" ")[0])

        scene_frames = [
            e for e in manifest_data if e["scene_id"] == scene_id
        ]

        if not scene_frames:
            return [], "No frames found for this scene."

        frame_paths = []
        for entry in scene_frames:
            img_path = APP_OUTPUT_DIR / entry["output_path"]
            if img_path.is_file():
                frame_paths.append(str(img_path))

        first = scene_frames[0]
        last  = scene_frames[-1]
        info  = (
            f"Scene {scene_id:02d} | "
            f"{first['timestamp_s']:.2f}s – {last['timestamp_s']:.2f}s | "
            f"{len(scene_frames)} frames extracted | "
            f"Quality: {sum(1 for e in scene_frames if e['quality_passed'])} passed, "
            f"{sum(1 for e in scene_frames if not e['quality_passed'])} flagged"
        )
        return frame_paths, info

    except Exception as exc:
        return [], f"Error loading scene: {exc}"


with gr.Blocks(title="Video Ingestion Pipeline") as demo:

    gr.Markdown("""
    # Video Ingestion Pipeline
    Upload a sports video and configure the pipeline parameters.
    The pipeline will extract frames, detect scenes, assess quality,
    and optionally run YOLOv8 object detection.
    """)

    manifest_state = gr.State(value=None)

    with gr.Row():

        # --- Left column: inputs ---
        with gr.Column(scale=1):
            gr.Markdown("## Input")
            video_input = gr.File(
                label="Upload Video (MP4, MKV, AVI)",
                file_types=[".mp4", ".mkv", ".avi", ".webm", ".ogv"],
            )

            gr.Markdown("## Extraction")
            extraction_mode = gr.Radio(
                choices=["interval", "fps"],
                value="interval",
                label="Extraction mode",
            )
            every_n = gr.Slider(
                1, 30, value=5, step=1,
                label="Extract every N-th frame (used when mode = interval)",
            )
            target_fps = gr.Slider(
                0.5, 25.0, value=2.0, step=0.5,
                label="Target FPS (used when mode = fps)",
            )

            gr.Markdown("## Transforms")
            with gr.Row():
                width  = gr.Slider(
                    320, 1920, value=640, step=32, label="Target width"
                )
                height = gr.Slider(
                    180, 1080, value=360, step=32, label="Target height"
                )
            keep_ar = gr.Checkbox(
                value=True, label="Keep aspect ratio (letterbox)"
            )
            colour = gr.Dropdown(
                ["rgb", "grayscale", "bgr"],
                value="rgb",
                label="Colour space",
            )

            gr.Markdown("## Quality Thresholds")
            blur_thresh = gr.Slider(
                0, 500, value=100, step=5,
                label="Blur threshold (Laplacian variance)",
            )
            with gr.Row():
                bright_low  = gr.Slider(
                    0, 100, value=30, step=5,
                    label="Brightness low (dark)",
                )
                bright_high = gr.Slider(
                    150, 255, value=225, step=5,
                    label="Brightness high (overexposed)",
                )
            flag_only = gr.Checkbox(
                value=True,
                label="Flag only (keep flagged frames, don't discard)",
            )

            gr.Markdown("## Scene Detection")
            scene_method = gr.Dropdown(
                ["histogram", "pyscenedetect"],
                value="histogram",
                label="Detection method",
            )
            scene_thresh = gr.Slider(
                0.1, 2.0, value=0.4, step=0.05,
                label="Scene threshold",
            )
            scene_min = gr.Slider(
                5, 60, value=15, step=5,
                label="Minimum scene length (frames)",
            )

            gr.Markdown("## YOLO Inference")
            run_inf = gr.Checkbox(
                value=True, label="Run YOLO inference"
            )
            yolo_conf = gr.Slider(
                0.1, 0.9, value=0.25, step=0.05,
                label="Confidence threshold",
            )
            yolo_frames = gr.Slider(
                1, 20, value=5, step=1,
                label="Number of frames to run inference on",
            )

            run_btn = gr.Button(
                "▶  Run Pipeline", variant="primary", size="lg"
            )

        # --- Right column: outputs ---
        with gr.Column(scale=2):
            gr.Markdown("## Results")

            with gr.Tab("Metadata"):
                metadata_out = gr.Textbox(
                    label="Video Metadata",
                    lines=10,
                    interactive=False,
                )

            with gr.Tab("Scenes"):
                scenes_out = gr.Textbox(
                    label="Detected Scenes",
                    lines=15,
                    interactive=False,
                )

            with gr.Tab("Quality"):
                quality_out = gr.Textbox(
                    label="Quality Summary",
                    lines=6,
                    interactive=False,
                )
                quality_chart = gr.Plot(label="Quality Charts")

            with gr.Tab("Frames"):
                frames_gallery = gr.Gallery(
                    label="Sample Extracted Frames (first 12)",
                    columns=4,
                    height=400,
                )

            with gr.Tab("Scenes Viewer"):
                scene_selector = gr.Dropdown(
                    choices=[],
                    label="Select a scene to inspect",
                    interactive=True,
                )
                scene_info = gr.Textbox(
                    label="Scene Info",
                    lines=3,
                    interactive=False,
                )
                scene_gallery = gr.Gallery(
                    label="Frames in selected scene",
                    columns=4,
                    height=400,
                )

            with gr.Tab("Flagged Frames"):
                flagged_summary = gr.Textbox(
                    label="Flagged Frame Summary",
                    lines=4,
                    interactive=False,
                )
                flagged_gallery = gr.Gallery(
                    label="Flagged Frames (blur / dark / overexposed)",
                    columns=4,
                    height=400,
                )

            with gr.Tab("YOLO Detections"):
                inference_out = gr.Textbox(
                    label="Inference Results",
                    lines=10,
                    interactive=False,
                )
                annotated_gallery = gr.Gallery(
                    label="Annotated Frames",
                    columns=3,
                    height=400,
                )

    run_btn.click(
        fn=run_pipeline_ui,
        inputs=[
            video_input,
            extraction_mode,
            every_n,
            target_fps,
            width, height, keep_ar, colour,
            blur_thresh, bright_low, bright_high, flag_only,
            scene_method, scene_thresh, scene_min,
            run_inf, yolo_conf, yolo_frames,
        ],
        outputs=[
            metadata_out,
            scenes_out,
            quality_out,
            quality_chart,
            frames_gallery,
            flagged_summary,
            flagged_gallery,
            annotated_gallery,
            inference_out,
            scene_selector,
            manifest_state,
            gr.Textbox(visible=False),
        ],
    )

    scene_selector.change(
        fn=load_scene_frames,
        inputs=[scene_selector, manifest_state],
        outputs=[scene_gallery, scene_info],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )