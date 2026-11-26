"""Unit tests for frame quality assessment heuristics."""

import numpy as np
import pytest
import cv2

from pipeline.quality import assess_quality, QualityReport


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sharp_frame():
    """A frame with strong edges — should score high on blur metric."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # Checkerboard pattern — maximum edge density.
    for i in range(0, 100, 10):
        for j in range(0, 100, 10):
            if (i // 10 + j // 10) % 2 == 0:
                frame[i:i+10, j:j+10] = 255
    return frame


@pytest.fixture
def blurry_frame(sharp_frame):
    """A heavily blurred version of the sharp frame."""
    return cv2.GaussianBlur(sharp_frame, (31, 31), 0)


@pytest.fixture
def dark_frame():
    """A very dark frame — should trigger too_dark flag."""
    return np.full((100, 100, 3), 10, dtype=np.uint8)


@pytest.fixture
def bright_frame():
    """An overexposed frame — should trigger overexposed flag."""
    return np.full((100, 100, 3), 250, dtype=np.uint8)


@pytest.fixture
def normal_frame():
    """A normal frame — should pass all quality checks."""
    return np.full((100, 100, 3), 128, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Blur detection
# ---------------------------------------------------------------------------

class TestBlurDetection:

    def test_sharp_frame_scores_higher_than_blurry(self, sharp_frame, blurry_frame):
        """Sharp frame must have higher Laplacian variance than blurry."""
        sharp_report = assess_quality(sharp_frame, blur_threshold=100.0)
        blurry_report = assess_quality(blurry_frame, blur_threshold=100.0)
        assert sharp_report.blur_score > blurry_report.blur_score

    def test_blurry_frame_flagged(self, blurry_frame):
        """Blurry frame is flagged when score is below threshold."""
        report = assess_quality(blurry_frame, blur_threshold=500.0)
        assert report.is_blurry is True
        assert "blurry" in report.flags

    def test_sharp_frame_not_flagged(self, sharp_frame):
        """Sharp frame is not flagged when score exceeds threshold."""
        report = assess_quality(sharp_frame, blur_threshold=10.0)
        assert report.is_blurry is False

    def test_blur_score_is_non_negative(self, normal_frame):
        """Blur score is always non-negative."""
        report = assess_quality(normal_frame)
        assert report.blur_score >= 0.0


# ---------------------------------------------------------------------------
# Brightness detection
# ---------------------------------------------------------------------------

class TestBrightnessDetection:

    def test_dark_frame_flagged(self, dark_frame):
        """Dark frame triggers too_dark flag."""
        report = assess_quality(dark_frame, brightness_low=30.0)
        assert report.is_too_dark is True
        assert "too_dark" in report.flags

    def test_bright_frame_flagged(self, bright_frame):
        """Overexposed frame triggers overexposed flag."""
        report = assess_quality(bright_frame, brightness_high=225.0)
        assert report.is_overexposed is True
        assert "overexposed" in report.flags

    def test_normal_frame_passes_brightness(self, normal_frame):
        """Normal brightness frame passes both brightness checks."""
        report = assess_quality(
            normal_frame,
            brightness_low=30.0,
            brightness_high=225.0,
        )
        assert report.is_too_dark is False
        assert report.is_overexposed is False


# ---------------------------------------------------------------------------
# QualityReport properties
# ---------------------------------------------------------------------------

class TestQualityReport:

    def test_passed_when_all_clear(self, sharp_frame):
        """passed property is True when no flags are set."""
        report = assess_quality(
            sharp_frame,
            blur_threshold=10.0,
            brightness_low=30.0,
            brightness_high=225.0,
        )
        assert report.passed is True
        assert report.flags == []

    def test_failed_when_blurry(self, blurry_frame):
        """passed property is False when frame is flagged."""
        report = assess_quality(blurry_frame, blur_threshold=500.0)
        assert report.passed is False

    def test_multiple_flags(self):
        """Frame can carry multiple quality flags simultaneously."""
        # Nearly black frame with heavy blur.
        dark_blurry = np.full((100, 100, 3), 5, dtype=np.uint8)
        dark_blurry = cv2.GaussianBlur(dark_blurry, (31, 31), 0)
        report = assess_quality(
            dark_blurry,
            blur_threshold=500.0,
            brightness_low=30.0,
        )
        assert report.is_too_dark is True
        assert len(report.flags) >= 1

    def test_grayscale_input(self):
        """assess_quality handles grayscale (2D) input without error."""
        gray = np.full((100, 100), 128, dtype=np.uint8)
        report = assess_quality(gray)
        assert isinstance(report, QualityReport)


# ---------------------------------------------------------------------------
# Additional coverage — filter_by_quality
# ---------------------------------------------------------------------------

class TestFilterByQuality:

    def test_flag_only_keeps_all_frames(self, sharp_frame, blurry_frame):
        """flag_only=True keeps all frames regardless of quality."""
        from pipeline.quality import filter_by_quality
        frames = [(sharp_frame, "sharp.jpg"), (blurry_frame, "blurry.jpg")]
        results, failed = filter_by_quality(
            frames,
            blur_threshold=500.0,
            flag_only=True,
        )
        assert len(results) == 2
        assert failed == 1

    def test_discard_mode_removes_failed_frames(self, blurry_frame, sharp_frame):
        """flag_only=False discards frames that fail quality checks."""
        from pipeline.quality import filter_by_quality
        frames = [(blurry_frame, "blurry.jpg"), (sharp_frame, "sharp.jpg")]
        results, failed = filter_by_quality(
            frames,
            blur_threshold=500.0,
            flag_only=False,
        )
        assert len(results) == 1
        assert failed == 1
        assert results[0][1] == "sharp.jpg"