# Limitations & Roadmap
## Video Ingestion Pipeline

**Version:** 1.0  

---

## 1. Overview

This document identifies the principal limitations of the implemented
pipeline, examines potential failure modes in a production team-sports
environment, and proposes research-grounded improvements.

---

## 2. Limitations of the Implemented System

### 2.1 Scene Detection Does Not Classify Shot Type

The histogram-based detector correctly identifies temporal boundaries
between shots. It does not classify what type of shot follows the cut. 
In the example of a football match footage, wide tactical views, 
close-ups of individual players, crowd cutaways, replay segments, are 
all segmented scenes but not distinguished. A pipeline that wants to 
process only wide tactical shots must either inspect every segment
manually or apply a separate classifier.

This is a fundamental limitation of unsupervised shot-boundary
detection. Resolving it requires a shot classification model consuming 
the segmented output of this pipeline. The suggestion is to train a CNN 
model to recognise broadcast shot types. It is worth noting that this 
approach carries a significant limitation when applied at the scene level: 
occasionally, frames from one shot type will wrongfully get mixed into a 
scene of a different type due to an undetected cut. Since the classifier 
labels the entire scene based on the keyframe alone, and the keyframe only 
represents the first frame, the majority of frames in that scene could be 
assigned the wrong label.

### 2.2 Quality Metrics Require Content-Specific Calibration

The Laplacian variance threshold (100.0) was calibrated on a specific
broadcast clip. Different venue types (indoor arenas, outdoor pitches
under different conditions) will produce different baseline sharpness
distributions. A threshold appropriate for one broadcast clip could
flag the majority of frames from another recording as blurry, or pass
genuinely blurry frames from a high-bitrate but motion-heavy action
sequence.

A possible solution is a **learned no-reference quality metric**
such as BRISQUE or NIQE, which model the statistical regularities of 
natural images and produce quality scores without requiring a reference 
image or manually tuned threshold. These metrics generalise better across 
content types at the cost of higher computational overhead.

### 2.3 Streaming Path Lacks Scene Detection

The live streaming path captures and processes frames individually
without scene detection. This is due to the current scene detection
implementation which requires reading the full video to establish
a rolling baseline. 

A sliding-window scene detector by maintaining a rolling histogram
baseline over the last N seconds of a live stream and detecting spikes
in real time, would resolve this.

### 2.4 Single-Threaded Performance at 1080p

The pipeline processes a 1080p video at 2.1× real-time speed on a
12-core CPU, meaning one core is saturated while eleven sit idle.

GPU-accelerated processing would reduce the per-frame compute budget 
significantly.

### 2.5 No Temporal Alignment Across Cameras

Each pipeline instance processes one video source independently.
Without a shared time reference, frames from two cameras at the same
nominal timestamp may not correspond to the same moment in the match,
due to encoding latency, or minor frame-rate differences
accumulating over time.

Temporal alignment is essential for multi-view player tracking (which
requires matching detections across cameras) and for 3D pose
reconstruction (which requires geometrically consistent multi-view
frames). Implementing alignment would require either hardware
synchronisation (GPS-locked timecodes, common in broadcast
production) or software synchronisation using audio cross-correlation
or visual event detection (e.g. whistle flash, ball kick) as anchor
points.

---

## 3. Production Failure Modes in Team-Sports Environments

### 3.1 Fast Camera Pans Generate False Scene Boundaries

When a camera operator whip-pans to follow a fast break or a long
ball, the entire frame content changes within 2–4 frames. The
histogram difference spikes sharply, often above the scene detection
threshold, producing a false boundary mid-shot. The `min_scene_length` 
parameter reduces consecutive false detections by requiring a minimum 
gap between boundaries. A fast pan lasting only 2 frames however still 
produces a single spike sufficient to trigger one false detection.

In a 90-minute football match with an active camera operator, false pan
boundaries could fragment what should be one single continuous tactical
shot into multiple segments, degrading the tracker's continuity. 
A system distinguishing between the gradual histogram change of a slow 
pan and the abrupt spike of a genuine cut would reduce this failure mode. 
Optical flow magnitude averaged across the frame is a computationally 
inexpensive pan discriminator:
a genuine cut produces a near-zero optical flow field (the new shot
begins with a static camera), while a pan produces a large, spatially
coherent flow field.

### 3.2 Broadcast Overlays Affect Quality Scores

Broadcast graphics (substitution animations, replay indicators...)
introduce high-frequency edges that elevate Laplacian variance. 
A frame partially covered by a graphic will score as sharper than it
actually is, potentially masking genuine motion blur in the background.

A graphics detection and masking step, using a pre-trained
segmentation model for example, would produce more reliable quality 
scores on broadcast content.

### 3.3 Variable Frame Rate Footage

Some mobile phone recordings and screen-capture tools produce
variable frame rate (VFR) video. The pipeline assumes constant frame
rate when computing timestamps.
On VFR footage this produces systematically incorrect timestamps,
an error that compounds over the duration of the recording and could
cause downstream events (e.g. goal detections) to be timestamped
incorrectly by several seconds.

FFmpeg can detect and report VFR via the `avg_frame_rate` vs
`r_frame_rate` discrepancy in ffprobe output. The ingestion module
could flag VFR sources and switch to container-embedded presentation
timestamps (PTS) for accurate per-frame timing.

---

## 4. Proposed Improvements

### 4.1 Shot Classification Model

As discussed in Section 2.1, a logical improvement of the current 
pipeline is a shot classifier consuming the segmented output. Training a 
classifier on a dataset of labelled broadcast shots (wide tactical,
close-up player, crowd, replay) is recommended. Scene-level
classification based on keyframes alone carries significant reliability
risks, as discussed in Section 2.1. A frame-level classifier, although
more resource intensive, would produce more accurate and trustworthy
results. The objective is adding a `shot_type` field to the manifest,
enabling the ability to filter by shot type without manual inspection.

### 4.2 Learned Quality Assessment

Replacing Laplacian variance with a learned no-reference metric
(BRISQUE or a domain-specific model trained on sports footage) would
eliminate manual threshold calibration and generalise better across
venue types.

### 4.3 TransNet v2 for Scene Detection

TransNet v2 (Souček & Lokoč, 2020) is a learned shot boundary
detector trained on a large corpus of broadcast footage. It
significantly outperforms histogram-based methods on dissolve and
fade transitions, which the current chi-squared detector cannot
detect (gradual transitions do not produce histogram spikes).
TransNet v2 runs at approximately 30fps on CPU for 48×27 downscaled
frames, making it computationally feasible as a replacement for the
custom histogram method.

### 4.4 Multi-Camera Temporal Alignment

For a full sports intelligence platform, temporal alignment across
cameras is a prerequisite for multi-view analysis. A practical
alignment approach for post-match footage is audio cross-correlation:
the broadcast commentary audio track is present on all camera feeds
and provides a common signal for alignment to within ~10ms. For live
streaming scenarios, hardware PTP (Precision Time Protocol) or SMPTE
timecodes embedded in the transport stream provide frame-accurate
synchronisation.

---

## 5. Conclusion

The implemented pipeline provides a solid foundation for sports video
ingestion: it handles multiple container formats, streaming protocols,
and provides configurable extraction, transformation, quality
assessment, and scene segmentation with a functional CV integration.
The principal limitations are well-understood problems with established
solutions. Each proposed improvement is a self-contained extension
that can be integrated without modifying the existing pipeline
architecture.

---

## References

Souček, T., & Lokoč, J. (2020). TransNet V2: An effective deep
network architecture for fast shot transition detection. *arXiv
preprint arXiv:2008.04838*.