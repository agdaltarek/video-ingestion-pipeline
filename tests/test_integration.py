"""
Integration test: full pipeline on a synthetic test video.

This test exercises the complete path:
    ingest → scene detection → extract → transform → quality → write → manifest

It verifies the pipeline produces the expected number of frames,
writes all files to disk, and generates a valid manifest.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import cv2

from pipeline.ingest import probe_video
from pipeline.scenes import detect_scenes_from_config
from pipeline.output import run_pipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    """Create a short synthetic MP4 for integration testing."""
    tmp = tmp_path_factory.mktemp("videos")
    path = tmp / "integration_test.mp4"

    width, height, fps, duration_s = 320, 240, 25.0, 4.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    total_frames = int(fps * duration_s)

    rng = np.random.default_rng(seed=99)
    for i in range(total_frames):
        frame = np.full((height, width, 3), (34, 139, 34), dtype=np.uint8)
        x = int(rng.uniform(20, width - 20))
        y = int(rng.uniform(20, height - 20))
        cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)
        writer.write(frame)
    writer.release()

    return path


@pytest.fixture(scope="module")
def pipeline_output(synthetic_video, tmp_path_factory):
    """Run the full pipeline once and return (manifest, output_dir)."""
    output_dir = tmp_path_factory.mktemp("output")

    scene_cfg = {
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

    meta = probe_video(synthetic_video)
    _, scenes = detect_scenes_from_config(
        synthetic_video, scene_cfg, source_fps=meta.fps
    )

    manifest = run_pipeline(
        source_path=synthetic_video,
        scenes=scenes,
        output_dir=output_dir,
        mode="interval",
        every_n_frames=5,
        resize_enabled=True,
        target_width=160,
        target_height=120,
        keep_aspect_ratio=True,
        colour_mode="rgb",
        colour_enabled=True,
        blur_threshold=5.0,
        brightness_low=10.0,
        brightness_high=245.0,
        flag_only=True,
        output_format="jpg",
        jpg_quality=90,
        write_manifest=True,
        manifest_filename="manifest.json",
    )

    return manifest, output_dir, synthetic_video


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestFullPipeline:

    def test_probe_succeeds(self, synthetic_video):
        """ffprobe can read the synthetic video without error."""
        meta = probe_video(synthetic_video)
        assert meta.width == 320
        assert meta.height == 240
        assert abs(meta.fps - 25.0) < 0.1
        assert meta.duration_s > 3.5

    def test_correct_frame_count(self, pipeline_output):
        """Pipeline extracts expected number of frames (every 5th of 100)."""
        manifest, _, _ = pipeline_output
        # 4s × 25fps = 100 frames total; every 5th = 20 frames.
        assert len(manifest) == 20

    def test_output_dimensions(self, pipeline_output):
        """All extracted frames have the target dimensions."""
        manifest, _, _ = pipeline_output
        for entry in manifest:
            assert entry.width == 160
            assert entry.height == 120

    def test_frames_written_to_disk(self, pipeline_output):
        """Every frame in the manifest exists as a file on disk."""
        manifest, output_dir, _ = pipeline_output
        for entry in manifest:
            path = output_dir / entry.output_path
            assert path.is_file(), f"Missing frame: {path}"

    def test_manifest_json_written(self, pipeline_output):
        """manifest.json is written and contains the correct number of entries."""
        manifest, output_dir, video_path = pipeline_output
        manifest_path = output_dir / video_path.stem / "manifest.json"
        assert manifest_path.is_file()

        with manifest_path.open() as f:
            data = json.load(f)
        assert len(data) == len(manifest)

    def test_manifest_fields_present(self, pipeline_output):
        """Every manifest entry contains all required fields."""
        manifest, output_dir, video_path = pipeline_output
        manifest_path = output_dir / video_path.stem / "manifest.json"
        with manifest_path.open() as f:
            data = json.load(f)

        required_fields = {
            "source_video", "frame_index", "timestamp_s", "scene_id",
            "output_path", "width", "height", "transformations",
            "blur_score", "brightness_mean", "quality_passed", "quality_flags",
        }
        for entry in data:
            assert required_fields.issubset(entry.keys())

    def test_transforms_recorded(self, pipeline_output):
        """Transformations list is non-empty for every frame."""
        manifest, _, _ = pipeline_output
        for entry in manifest:
            assert len(entry.transformations) > 0

    def test_timestamps_monotonically_increasing(self, pipeline_output):
        """Frame timestamps increase monotonically through the manifest."""
        manifest, _, _ = pipeline_output
        timestamps = [e.timestamp_s for e in manifest]
        assert timestamps == sorted(timestamps)

    def test_corrupt_file_raises_ingest_error(self, tmp_path):
        """Corrupt file raises IngestError, not an unhandled exception."""
        from pipeline.ingest import IngestError
        corrupt = tmp_path / "corrupt.mp4"
        corrupt.write_bytes(b"\x00" * 512)
        with pytest.raises(IngestError):
            probe_video(corrupt)

    def test_missing_file_raises_file_not_found(self, tmp_path):
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            probe_video(tmp_path / "does_not_exist.mp4")


class TestPySceneDetectPath:

    def test_pyscenedetect_returns_at_least_one_scene(self, synthetic_video):
        """PySceneDetect fallback path returns at least one scene."""
        from pipeline.scenes import detect_scenes_from_config
        from pipeline.ingest import probe_video

        meta = probe_video(synthetic_video)
        scene_cfg = {
            "method": "pyscenedetect",
            "pyscenedetect": {
                "threshold": 27.0,
                "min_scene_length": 15,
            },
        }
        _, scenes = detect_scenes_from_config(
            synthetic_video, scene_cfg, source_fps=meta.fps
        )
        assert len(scenes) >= 1

    def test_invalid_scene_method_raises(self, synthetic_video):
        """Unknown scene detection method raises ValueError."""
        from pipeline.scenes import detect_scenes_from_config

        with pytest.raises(ValueError, match="Unknown scene detection method"):
            detect_scenes_from_config(
                synthetic_video,
                {"method": "invalid_method"},
            )


class TestStreamingPath:

    def test_hls_stream_yields_frames(self, synthetic_video, tmp_path):
        """Local HLS stream yields frames via iter_stream_frames."""
        from pipeline.stream import iter_stream_frames, is_streaming_url, detect_protocol

        # Generate HLS segments from the synthetic video
        hls_dir = tmp_path / "hls"
        hls_dir.mkdir()
        playlist = hls_dir / "stream.m3u8"

        import subprocess
        result = subprocess.run([
            "ffmpeg", "-i", str(synthetic_video),
            "-c:v", "libx264",   # re-encode to H.264 for HLS compatibility
            "-preset", "fast",
            "-start_number", "0",
            "-hls_time", "2",
            "-hls_list_size", "0",
            "-f", "hls",
            str(playlist)
        ], capture_output=True)

        assert playlist.is_file(), "HLS playlist was not created"

        # Read frames from the local HLS stream
        frames = list(iter_stream_frames(
            source=str(playlist),
            every_n_frames=5,
            max_frames=5,
        ))

        assert len(frames) == 5
        for frame_index, timestamp_s, frame_bgr in frames:
            assert frame_bgr is not None
            assert frame_bgr.ndim == 3
            assert timestamp_s >= 0.0

    def test_is_streaming_url_detects_m3u8(self):
        """is_streaming_url returns True for .m3u8 paths."""
        from pipeline.stream import is_streaming_url
        assert is_streaming_url("output/hls_test/stream.m3u8") is True
        assert is_streaming_url("data/video.mp4") is False

    def test_detect_protocol_hls(self):
        """detect_protocol correctly identifies HLS sources."""
        from pipeline.stream import detect_protocol
        assert detect_protocol("stream.m3u8") == "hls"
        assert detect_protocol("http://example.com/stream.m3u8") == "hls"

    def test_detect_protocol_rtsp(self):
        """detect_protocol correctly identifies RTSP sources."""
        from pipeline.stream import detect_protocol
        assert detect_protocol("rtsp://192.168.1.1:554/stream") == "rtsp"

    def test_detect_protocol_rtmp(self):
        """detect_protocol correctly identifies RTMP sources."""
        from pipeline.stream import detect_protocol
        assert detect_protocol("rtmp://server/live/key") == "rtmp"