# Video Ingestion Pipeline

A production-quality video ingestion and pre-processing pipeline for
sports computer vision applications. Accepts raw team-sports video in
multiple container formats and streaming protocols, and produces
structured, analysis-ready frames and per-frame metadata suitable for
direct consumption by a CV inference service (YOLOv8).

---

## Features

- **Multi-format input:** MP4/H.264, MKV/H.265, AVI/MJPEG
- **Streaming input:** HLS, RTSP, RTMP via OpenCV/FFmpeg
- **Metadata extraction:** codec, fps, resolution, duration, tags via ffprobe
- **Frame extraction:** configurable interval or target FPS
- **Transforms:** letterbox resize + colour-space conversion (BGR→RGB/grayscale)
- **Quality assessment:** Laplacian blur detection + brightness heuristics
- **Scene detection:** custom histogram difference + adaptive thresholding
  (PySceneDetect fallback available)
- **Structured output:** frames organised by scene + JSON manifest
- **CV integration:** YOLOv8n object detection on extracted frames
- **Performance:** 9×real-time at 480p, 2.1×real-time at 1080p (CPU-only)

---

## Project Structure

```text
video-ingestion/
├── README.md
├── pyproject.toml
├── config.yaml                  # all tunable parameters
├── environment.yml              # conda environment spec
├── generate_test_videos.py      # generates synthetic test videos
├── app.py                       # interactive testing app
├── src/pipeline/
│   ├── __init__.py
│   ├── cli.py                   # command-line entry point
│   ├── config.py                # YAML config loader
│   ├── logging_setup.py         # structured logging
│   ├── ingest.py                # metadata extraction (ffprobe)
│   ├── preprocess.py            # frame extraction + transforms
│   ├── quality.py               # blur + brightness heuristics
│   ├── scenes.py                # scene/shot-boundary detection
│   ├── output.py                # structured output + JSON manifest
│   ├── stream.py                # streaming input (HLS/RTSP/RTMP)
│   └── inference.py             # YOLOv8 integration
├── tests/
│   ├── test_preprocess.py
│   ├── test_quality.py
│   ├── test_integration.py
│   └── test_config_logging.py
├── benchmarks/
│   ├── benchmark.py             # throughput benchmarking
│   ├── benchmark_results.json
│   └── benchmark_report.txt
└── docs/
    ├── design_document.md       # technical design document
    └── limitations_and_roadmap.md  # known limitations + roadmap
```

---

## Setup (Windows + conda)

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/video-ingestion.git
cd video-ingestion
```

### 2. Create the conda environment

```bash
conda create -n video-ingest -c conda-forge \
    python=3.11 ffmpeg opencv numpy pyyaml \
    pytest pytest-cov matplotlib -y
conda activate video-ingest
```

### 3. Install PyTorch (CPU) and remaining pip packages

> **Windows note:** install order matters. conda-forge packages must be
> present before pip packages to avoid a cv2/torch DLL conflict.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics scenedetect[opencv] psutil gradio
```

### 4. Install the pipeline package (editable)

```bash
pip install -e .
```

### 5. Verify the installation

```bash
video-ingest --version
python -c "import cv2, torch; from ultralytics import YOLO; print('All imports OK')"
```

---

## Usage

### Process a video file

```bash
video-ingest --config config.yaml --input data/your_video.mp4
```

### Process a streaming source (HLS)

```bash
video-ingest --config config.yaml --stream http://example.com/stream.m3u8
```

### Run in diagnostic mode

To see detailed per-frame decisions (scene boundary distances, quality
scores, and threshold comparisons) run with debug logging enabled:

```bash
video-ingest --config config.yaml --input data/your_video.mp4 --log-level DEBUG
```

---

## Configuration

All parameters are in `config.yaml`. Key sections:

```yaml
extraction:
  mode: interval          # "interval" = every N-th frame
                          # "fps"      = resample to target frame rate
  every_n_frames: 5       # used when mode = interval
  target_fps: 2.0         # used when mode = fps (e.g. 2.0 = 2 frames/sec)
  output_dir: output

transforms:
  resize:
    enabled: true
    width: 640
    height: 360
    keep_aspect_ratio: true
  colour_space:
    enabled: true
    mode: rgb             # "rgb", "grayscale", or "bgr"

quality:
  blur:
    threshold: 100.0      # Laplacian variance floor
  brightness:
    low_threshold: 30.0
    high_threshold: 225.0
  flag_only: true         # keep flagged frames (true) or discard (false)

scenes:
  method: histogram       # "histogram" or "pyscenedetect"

inference:
  enabled: true
  model: yolov8n.pt
  confidence: 0.25
  max_frames: 5           # frames to run inference on
```

---

## Test Data

The pipeline ships with a synthetic video generator for
dependency-free testing:

```bash
python generate_test_videos.py
```

This creates `test_480p_10s.mp4`, `test_720p_30s.mp4`,
`test_1080p_15s.mp4`, and `test_corrupt.mp4` in `data/`.

For real footage testing, source a Creative Commons licensed clip
and save it as `data/sports_footage.mp4`.

---

## Running Tests

No video file needed. Each test generates its own synthetic data internally 
and deletes it afterwards

```bash
pytest tests/ -v --cov=pipeline --cov-report=term-missing
```

Expected: 51 tests passing, ~63% coverage.

---

## Running the Benchmark

```bash
python benchmarks/benchmark.py
```

Results are saved to `benchmarks/benchmark_results.json` and
`benchmarks/benchmark_report.txt`.

---

## Running the Interactive Demo App

```bash
python app.py
```

Opens automatically at `http://127.0.0.1:7860`.

Features:
- Upload any MP4, MKV, or AVI file
- Adjust all pipeline parameters via sliders and dropdowns
- View extracted frames grouped by detected scene
- Inspect flagged (low quality) frames separately
- See YOLO detections with annotated bounding boxes
- View blur and brightness quality charts

---

## Documentation

| File | Description |
|------|-------------|
| `docs/design_document.md` | Technical design document |
| `docs/limitations_and_roadmap.md` | Known limitations, failure modes, and roadmap |
| `benchmarks/benchmark_report.txt` | Performance benchmark |

---

## Known Windows Notes

- Import torch before cv2 in any module using both (DLL load order)
- Set `KMP_DUPLICATE_LIB_OK=TRUE` before importing matplotlib + torch
  together (Intel OpenMP duplicate runtime)
- Both are handled automatically by the pipeline code
