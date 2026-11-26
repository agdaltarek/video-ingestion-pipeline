"""
Performance benchmark for the video ingestion pipeline.

Measures throughput, memory usage, and CPU utilisation across
multiple resolutions and durations.

Usage:
    python benchmarks/benchmark.py

Output:
    benchmarks/benchmark_results.json  — raw results
    benchmarks/benchmark_report.txt   — human-readable summary

Requirements:
    pip install psutil
"""

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import psutil

# Make sure the pipeline package is importable.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline.ingest import probe_video
from pipeline.scenes import detect_scenes_from_config
from pipeline.output import run_pipeline
from pipeline.logging_setup import configure_logging

# Suppress pipeline logs during benchmarking for clean output.
import logging
logging.getLogger("pipeline").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_DIR    = Path(__file__).parent.parent / "output" / "benchmark"
RESULTS_FILE  = Path(__file__).parent / "benchmark_results.json"
REPORT_FILE   = Path(__file__).parent / "benchmark_report.txt"

# Benchmark matrix: (label, width, height, fps, duration_s)
VIDEO_CONFIGS = [
    ("480p_10s",  854,  480, 25.0, 10.0),
    ("480p_30s",  854,  480, 25.0, 30.0),
    ("1080p_10s", 1920, 1080, 25.0, 10.0),
    ("1080p_30s", 1920, 1080, 25.0, 30.0),
]

SCENE_CFG = {
    "method": "histogram",
    "histogram": {
        "bins": 64,
        "threshold": 0.4,
        "adaptive": True,
        "adaptive_window": 10,
        "min_scene_length": 15,
        "min_absolute_threshold": 0.05,
    },
}

PIPELINE_KWARGS = dict(
    mode="interval",
    every_n_frames=5,
    resize_enabled=True,
    target_width=640,
    target_height=360,
    keep_aspect_ratio=True,
    colour_mode="rgb",
    colour_enabled=True,
    blur_threshold=100.0,
    brightness_low=30.0,
    brightness_high=225.0,
    flag_only=True,
    output_format="jpg",
    jpg_quality=95,
    write_manifest=True,
    manifest_filename="manifest.json",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_synthetic_video(path: Path, width: int, height: int,
                          fps: float, duration_s: float) -> None:
    """Write a synthetic video to disk for benchmarking."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    total_frames = int(fps * duration_s)
    rng = np.random.default_rng(seed=42)
    positions = rng.integers(50, min(width, height) - 50, size=(5, 2)).astype(float)
    velocities = rng.uniform(-3, 3, size=(5, 2))
    colours = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    for _ in range(total_frames):
        frame = np.full((height, width, 3), (34, 139, 34), dtype=np.uint8)
        for i, (pos, col) in enumerate(zip(positions, colours)):
            positions[i] += velocities[i]
            for axis, limit in enumerate([width, height]):
                if positions[i][axis] < 20 or positions[i][axis] > limit - 20:
                    velocities[i][axis] *= -1
            cv2.circle(frame, (int(pos[0]), int(pos[1])), 15, col, -1)
        writer.write(frame)
    writer.release()


def measure_pipeline(video_path: Path) -> dict:
    """Run the full pipeline on one video and return performance metrics."""
    process = psutil.Process(os.getpid())

    # Baseline memory before run.
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    # CPU measurement — sample over the run.
    process.cpu_percent(interval=None)  # first call initialises

    t_start = time.perf_counter()

    # --- Ingest ---
    meta = probe_video(video_path)

    # --- Scene detection ---
    _, scenes = detect_scenes_from_config(
        video_path, SCENE_CFG, source_fps=meta.fps
    )

    # --- Full pipeline ---
    video_output = OUTPUT_DIR / video_path.stem
    manifest = run_pipeline(
        source_path=video_path,
        scenes=scenes,
        output_dir=OUTPUT_DIR,
        **PIPELINE_KWARGS,
    )

    t_end = time.perf_counter()
    elapsed_s = t_end - t_start

    # Memory and CPU after run.
    mem_after  = process.memory_info().rss / 1024 / 1024
    cpu_pct    = process.cpu_percent(interval=None)

    total_frames = int(meta.fps * meta.duration_s)
    extracted    = len(manifest)
    throughput   = total_frames / elapsed_s  # frames/second scanned

    return {
        "video"           : video_path.name,
        "resolution"      : f"{meta.width}x{meta.height}",
        "duration_s"      : round(meta.duration_s, 1),
        "total_frames"    : total_frames,
        "extracted_frames": extracted,
        "elapsed_s"       : round(elapsed_s, 2),
        "throughput_fps"  : round(throughput, 1),
        "speedup_vs_rt"   : round(throughput / meta.fps, 2),
        "mem_before_mb"   : round(mem_before, 1),
        "mem_after_mb"    : round(mem_after, 1),
        "mem_delta_mb"    : round(mem_after - mem_before, 1),
        "cpu_pct"         : round(cpu_pct, 1),
        "scenes_detected" : len(scenes),
    }


def format_report(results: list[dict]) -> str:
    """Format benchmark results as a human-readable report."""
    lines = [
        "=" * 70,
        "  VIDEO INGESTION PIPELINE — PERFORMANCE BENCHMARK REPORT",
        "=" * 70,
        "",
        f"  Platform    : {sys.platform}",
        f"  CPU cores   : {psutil.cpu_count(logical=True)} logical",
        f"  RAM total   : {psutil.virtual_memory().total / 1024**3:.1f} GB",
        "",
        "-" * 70,
        f"  {'Video':<18} {'Res':>12} {'Dur':>6} {'Frames':>7} "
        f"{'Time(s)':>8} {'FPS':>7} {'Speed':>7} {'RAM+MB':>7}",
        "-" * 70,
    ]

    for r in results:
        lines.append(
            f"  {r['video']:<18} {r['resolution']:>12} {r['duration_s']:>5.0f}s "
            f"{r['total_frames']:>7} {r['elapsed_s']:>8.1f} "
            f"{r['throughput_fps']:>7.1f} {r['speedup_vs_rt']:>6.1f}x "
            f"{r['mem_delta_mb']:>7.1f}"
        )

    lines += [
        "-" * 70,
        "",
        "  Columns:",
        "    Res     = source resolution",
        "    Dur     = video duration in seconds",
        "    Frames  = total frames in source video",
        "    Time(s) = wall-clock processing time",
        "    FPS     = frames scanned per second",
        "    Speed   = processing speed vs real-time (>1.0 = faster than RT)",
        "    RAM+MB  = peak memory increase during processing",
        "",
        "  Notes:",
        "    - Benchmarks run on synthetic video (CPU-only, no GPU)",
        "    - Every 5th frame extracted and transformed to 640x360 RGB",
        "    - Scene detection runs on all frames; frame write on extracted only",
        "    - First run may be slower due to filesystem cache cold start",
        "=" * 70,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(__file__).parent / "tmp_videos"
    tmp_dir.mkdir(exist_ok=True)

    print("Video Ingestion Pipeline — Performance Benchmark")
    print("=" * 50)

    results = []

    for label, width, height, fps, duration_s in VIDEO_CONFIGS:
        video_path = tmp_dir / f"{label}.mp4"

        # Generate synthetic video if not already present.
        if not video_path.is_file():
            print(f"  Generating {label} ({width}x{height}, {duration_s}s)...",
                  end=" ", flush=True)
            make_synthetic_video(video_path, width, height, fps, duration_s)
            print("done")

        print(f"  Benchmarking {label}...", end=" ", flush=True)
        result = measure_pipeline(video_path)
        results.append(result)
        print(
            f"done — {result['elapsed_s']:.1f}s | "
            f"{result['throughput_fps']:.0f} fps | "
            f"{result['speedup_vs_rt']:.1f}x real-time"
        )

    # Save raw results.
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Raw results saved: {RESULTS_FILE}")

    # Save and print report.
    report = format_report(results)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Report saved: {REPORT_FILE}")
    print()
    print(report)


if __name__ == "__main__":
    main()