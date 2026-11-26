# Technical Design Document
## Video Ingestion Pipeline for Sports Computer Vision Applications

**Version:** 1.0  

---

## 1. Introduction

This document describes the architectural decisions, design trade-offs, and
technical rationale underpinning the video ingestion pipeline. 
The pipeline accepts raw team-sports video in multiple container formats and 
streaming protocols, and produces structured, analysis-ready frames and 
per-frame metadata suitable for direct consumption by a computer vision 
inference service.

The document is organised as follows: Section 2 presents the system
architecture and data flow. Section 3 justifies the technology stack.
Section 4 details key design decisions at the component level. Section 5
discusses scalability. Section 6 addresses deployment considerations.

---

## 2. System Architecture

### 2.1 Architecture Overview

The pipeline is structured as a six-stage process, where each stage consumes 
the output of the previous and produces a well-defined output structure. 
Stages are implemented as independent Python modules with clean function 
signatures, enabling unit testing in isolation and future replacement of 
individual components without modifying the rest of the pipeline.


```text
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   INPUT          │     │   SCENE          │     │   FRAME          │
│   HANDLING       │────▶│   DETECTION      │────▶│   EXTRACTION     │
│                  │     │                  │     │                  │
│   ingest.py      │     │   scenes.py      │     │   preprocess.py  │
│   stream.py      │     │                  │     │                  │
└──────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                           │
                                                           ▼
┌──────────────────┐                            ┌──────────────────┐
│   OUTPUT         │                            │   TRANSFORM      │
│   MANIFEST       │◀───────────────────────────│   + QUALITY      │
│                  │                            │                  │
│   output.py      │                            │   preprocess.py  │
│                  │                            │   quality.py     │
└────────┬─────────┘                            └──────────────────┘
         │
         ▼
┌──────────────────┐
│   CV INFERENCE   │
│                  │
│   inference.py   │
└──────────────────┘
```

### 2.2 Data Flow Diagram

```text
Raw Video File / Stream URL
│
▼
┌─────────────────────────────────────────────────┐
│ INGESTION                                        │
│  ffprobe → VideoMetadata dataclass               │
│  (codec, fps, resolution, duration, tags)        │
└─────────────────────┬───────────────────────────┘
│ VideoMetadata
▼
┌─────────────────────────────────────────────────┐
│ SCENE DETECTION                                  │
│  Per-frame histogram → chi-squared distance      │
│  Adaptive threshold + absolute floor             │
│  → list[SceneBoundary], list[Scene]              │
└─────────────────────┬───────────────────────────┘
│ Scene list
▼
┌─────────────────────────────────────────────────┐
│ EXTRACTION + TRANSFORM + QUALITY (single pass)  │
│  iter_frames() generator → frame_bgr (numpy)    │
│  resize_frame() → letterbox to 640×360           │
│  convert_colour_space() → RGB                   │
│  assess_quality() → QualityReport               │
└─────────────────────┬───────────────────────────┘
│ transformed frame + report
▼
┌─────────────────────────────────────────────────┐
│ OUTPUT                                           │
│  output/<video>/<scene_id>/frame_XXXXXX.jpg      │
│  output/<video>/manifest.json                    │
└─────────────────────┬───────────────────────────┘
│ manifest entries
▼
┌─────────────────────────────────────────────────┐
│ CV INFERENCE (optional)                          │
│  YOLOv8n → Detection list per frame              │
│  output/<video>/annotated/annotated_*.jpg        │
│  output/<video>/inference_results.json           │
└─────────────────────────────────────────────────┘
```

### 2.3 Single-Pass Design

Transformation, quality assessment, and disk write occur in a **single pass** 
over the video (implemented in `output.run_pipeline()`). Why the option of having 
a multi-pass algorithm is rejected :

1. **Memory efficiency.** A single pass means only one frame is held in
   memory at any time. The generator-based `iter_frames()` function yields
   one frame at a time rather than loading the entire video into memory.
   Benchmark results confirm peak memory delta of under 10MB regardless of
   video duration or resolution.

2. **Disk Input/Output efficiency.** Separate passes would require writing raw frames to
   disk, reading them back for transforms, writing again, reading again for
   quality assessment. A single pass writes each frame exactly once.

3. **Auditability.** Each frame's transform chain and quality report are
   recorded atomically in the manifest at write time, ensuring the manifest
   always reflects what was actually written. Even if the pipeline crashes
   mid-run, all frames processed up to that point will have correct and
   complete manifest entries.
  

---

## 3. Technology Stack Justification

### 3.1 Language: Python 3.11

Python was selected primarily for its ecosystem compatibility with our
chosen infrastructure. Its large community also makes debugging and
problem resolution significantly faster in practice. The primary
trade-off relative to other alternatives is runtime performance. 
However, the benchmark results confirm that we were able to meet 
the throughput requirement on commodity hardware without requiring 
GPU acceleration or compiled extensions.


As an alternative, **Rust** would guarantee better performance for
long-running pipeline processes, but at significant cost to ecosystem
compatibility.


### 3.2 Video Decoding: FFmpeg + OpenCV

Two distinct tools are used :

**ffprobe (FFmpeg)** is used exclusively for metadata extraction.
ffprobe reads container and stream headers directly via libavformat,
providing accurate fractional frame rates, embedded GPS tags, camera 
identifiers, and colour space parameters that OpenCV's VideoCapture 
does not expose.

**OpenCV (cv2.VideoCapture)** is used for frame decoding. The trade-off 
is that OpenCV's RTSP implementation buffers several frames internally, 
introducing some latency which is not ideal for near-real-time analysis. 
If you need real-time processing (e.g., for AI inference), use FFmpeg with 
-fflags nobuffer or GStreamer instead of plain cv2.VideoCapture.

Alternative considered: **PyAV** provides Python bindings to
libavcodec directly, offering lower-level access to codec parameters
and packet-level decoding.

### 3.3 Scene Detection: Custom Histogram Difference

For greater flexibility and technical depth, a custom scene detection
method was implemented based on chi-squared histogram difference with
adaptive thresholding. PySceneDetect is also integrated as a ready-to-use
alternative, selectable via the configuration file.

**Rationale for custom implementation:**

Chi-squared distance between consecutive per-channel colour
histograms is a well-established shot-boundary detection method. 
The chi-squared metric is preferred over L1 or L2 histogram 
distance because it is more sensitive to differences in 
low-frequency (dark) bins, making it better suited to detecting
cuts between shots that share similar overall brightness but 
differ in content distribution.

**Adaptive thresholding:**

A fixed threshold is insufficient because of potential changes
in scene conditions over the broadcast period (fog, cloud cover,
nightfall...). Adaptive thresholding sets the detection threshold
relative to a rolling mean of recent inter-frame distances,
automatically compensating as lighting conditions evolve.

**Absolute floor parameter:**

During development, testing revealed that on near-static content 
represented by our synthetic videos, the adaptive threshold can 
collapse to near-zero (observed values of 0.001–0.003), causing 
the detector to trigger on very low variations. The diagnosis 
confirmed the rolling baseline itself was the cause.

The fix — a `min_absolute_threshold` parameter (default 0.05) that
floors the effective threshold regardless of the rolling baseline. 
On real broadcast footage, genuine cut distances of 0.5–2.9 are well 
above this floor, so it has no effect in production use.

**PySceneDetect :**

PySceneDetect's ContentDetector (HSV colour difference) is retained
as a configurable alternative (`method: pyscenedetect` in
config.yaml). Both methods return identical `SceneBoundary` and
`Scene` data structures, making the rest of the pipeline indifferent
to which method was used.


### 3.4 Quality Assessment: Laplacian Variance + Mean Brightness

**Blur detection** uses the variance of the Laplacian operator
applied to a grayscale conversion of the frame. The Laplacian is a
second-order derivative operator that responds strongly to edges.
Sharp frames have high edge density and thus high Laplacian variance,
while blurry frames have suppressed edges and low variance. This is 
a standard no-reference sharpness estimator that runs in approximately 
2ms per 640×360 frame on the test machine.

Validation on football broadcast footage confirmed sharp frames score 
150–685 (mean ~230) and motion-blurred frames during fast camera pans 
score 75–99.

Alternative considered: **BRISQUE** (Blind/Referenceless Image
Spatial Quality Evaluator) is a more sophisticated no-reference
image quality metric that correlates with human perceptual quality
judgements. Unlike our approach, BRISQUE doesn't use a fixed threshold,
which makes it more reliable for multi-purpose problems. Though it must
be stated that it runs approximately 10× slower than Laplacian variance.
For its self-calibration capacity, BRISQUE would be recommended
for deployments covering diverse venue types and camera conditions.

**Brightness assessment** uses mean pixel intensity on the grayscale
frame. This is a simple but effective first-pass filter for
underexposed frames (arena lights not yet on, tunnel shots) and
overexposed frames (direct sunlight on outdoor pitches). On the
football broadcast test clip, all frames fell in the 63–119 range, well 
within the 30–225 thresholds.

### 3.5 YOLOv8n for object detection

YOLOv8n (nano) was selected as the object detection method. It is the
smallest model in the YOLOv8 family, running at 40–55ms per frame on
CPU (12-core Ryzen 5 3600, no GPU).

The pipeline feeds frames to YOLO in RGB format. the colour space 
conversion step converts BGR (OpenCV native) to RGB, which is what
YOLO's preprocessing expects. Eliminating the conversion inside the
inference call avoids redundant processing.

At production level, a model trained on a custom specialized dataset
would be recommended. It would be a time consuming but necessary step.
YOLOv8m or YOLOv8x would improve detection accuracy at the cost of
~3–5× inference time.

### 3.6 Configuration file

All tunable parameters are exposed via `config.yaml` with CLI
overrides via argparse. Parameters are never hard-coded in module
source files. This allows configurability and enables reproducible 
experiments by committing config files alongside results.

---

## 4. Key Design Decisions

### 4.1 Output Directory Hierarchy

Frames are organised by scene rather than written to a flat directory:
output/<video_stem>/frames/scene_00/frame_000000.jpg
output/<video_stem>/frames/scene_01/frame_000025.jpg

This structure allows a downstream CV service to process one scene at
a time, reset object trackers at scene boundaries, and skip unwanted
segment types (crowd cutaways, replays) by directory.

### 4.2 Manifest Design

The JSON manifest records the complete provenance of every frame:
source video path, frame index, timestamp, scene assignment,
post-transform dimensions, the full transform chain applied, and
quality scores.

### 4.3 Streaming Architecture

The streaming path (`stream.py`) uses `cv2.VideoCapture` with
streaming URLs, which delegates to FFmpeg's network demuxers. This
means the same FFmpeg codec stack handles both file and stream
decoding. The streaming path differs from the file path in two 
respects:

1. **No scene detection.** Scene detection requires scanning the
entire video to establish a rolling baseline, which is impossible
for a live stream of unknown duration. The streaming path performs
quality assessment and transforms frame-by-frame without scene
segmentation. 
Scene detection on live streams would require a sliding-window
approach, which processes frames as they arrive and only keeps
the last N frame distances in memory, removing the need to
read the full video upfront.

2. **Retry logic.** Unlike file-based input where the source is always
available on disk, a live stream can drop unexpectedly due to network
instability. The streaming path implements automatic reconnection with
a configurable number of retries, making it more resilient to brief
network failures.

### 4.4 Dependency Management

The pipeline uses a strict install order on Windows: conda-forge
packages (OpenCV, FFmpeg, NumPy) are installed first, then PyTorch
from the official CPU wheel index, then ultralytics. This order is
required because the conda-forge OpenCV build and the pip PyTorch
wheel ship overlapping Windows runtime DLLs (notably the Intel
OpenMP runtime libiomp5md.dll). If OpenCV loads first it claims an
incompatible DLL version and PyTorch's shm.dll fails to initialise
(WinError 127). Importing torch before cv2 in any module that uses
both resolves the load-order conflict. This constraint is documented
in `environment.yml` and `README.md`.

---

## 5. Performance Benchmark Results

Benchmarks were run on a Ryzen 5 3600 (12 logical cores, 15.9GB RAM,
no GPU) on Windows 11. All measurements use synthetic test videos
with seeded randomness for reproducibility.

| Resolution | Duration | Frames | Time (s) | Throughput | Speed vs RT | RAM Δ |
|------------|----------|--------|----------|------------|-------------|-------|
| 854×480    | 10s      | 250    | 1.1s     | 224 fps    | 9.0×        | +6MB  |
| 854×480    | 30s      | 750    | 3.0s     | 250 fps    | 10.0×       | ~0MB  |
| 1920×1080  | 10s      | 250    | 4.8s     | 52 fps     | 2.1×        | +1MB  |
| 1920×1080  | 30s      | 750    | 14.5s    | 52 fps     | 2.1×        | ~0MB  |

**Key observations:**

- No memory leaks or performance degradation over longer videos.
- 1080p throughput is ~4× lower than 480p, consistent with the 4×
  pixel count increase (2,073,600 vs 409,920 pixels per frame).
- A 10-minute 1080p video would process in approximately 4.8 minutes
  on this hardware — within the 5-minute processing target.
- Peak memory increase is under 10MB regardless of resolution or
  duration, confirming the generator-based single-pass design
  eliminates in-memory video buffering.
- The pipeline is single-threaded and CPU-bound. One core is
  saturated during scene detection (histogram computation on all
  frames) and during frame writes (JPEG encoding). Parallelising
  scene detection and extraction across CPU cores would reduce
  1080p processing time by an estimated 3–4×.

**Validation on broadcast footage:**

Informal testing on a broadcast-quality MP4 (H.264, 1920×1080,
25fps, 46 seconds, 4 Mbps) produced blur scores in the range 75–685 
(mean 229), confirming the Laplacian threshold of 100.0 is appropriately 
calibrated for professional football broadcast content. Scene detection 
identified 8 boundaries with chi-squared distances of 0.06–2.92, with 
strong cuts (dist > 1.0) corresponding to genuine camera switches and 
weaker signals (dist 0.06–0.17) corresponding to fast pans.

---

## 6. Scalability Considerations

### 6.1 Vertical Scaling

The single-threaded pipeline can be accelerated on the existing
hardware through two approaches:

**GPU-accelerated decoding.** FFmpeg supports NVDEC/CUVID hardware
decoding on NVIDIA GPUs. Replacing `cv2.VideoCapture` with an FFmpeg
subprocess pipe reading NVDEC-decoded frames would reduce decoding
time by approximately 5–8× for H.264 content on a modern GPU.

**Parallel frame processing.** The extraction, transform, and quality
stages are embarrassingly parallel.
Python's `multiprocessing.Pool` could process N frames simultaneously
using N CPU cores, reducing total pipeline time.

### 6.2 Horizontal Scaling

For a multi-camera deployment, you could run one pipeline instance per 
camera. Each instance is stateless with respect to other cameras. 
An orchestration layer (e.g. Kubernetes Jobs) could then schedule and 
monitor instances without modifying the pipeline code.

Ensuring frames from different cameras are synchronised to the same game 
clock would require a shared time reference (GPS clock or NTP 
synchronisation) and a separate alignment layer consuming the per-camera 
manifests.

### 6.3 Cloud-Native Deployment

For a cloud deployment, the structured output format (frames +
JSON manifest per video) maps naturally to object storage (S3,
GCS). The manifest's relative `output_path` field ensures portability
across storage backends. A production deployment would replace local
`cv2.imwrite()` calls with direct S3 `put_object` calls, with the
manifest written as a single JSON document to the same bucket.

The stateless, file-in / files-out design of the pipeline means it
can be containerised (Dockerfile provided) and deployed as a
serverless function (AWS Lambda, Google Cloud Run) triggered by
S3 object creation events — enabling automatic processing of newly
uploaded footage without a persistent server.

---


