"""Unit tests for frame extraction and transform functions."""

import numpy as np
import pytest

from pipeline.preprocess import resize_frame, convert_colour_space, apply_transforms


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bgr_frame():
    """A simple 100x200 BGR test frame (green)."""
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[:, :] = (34, 139, 34)  # BGR green
    return frame


@pytest.fixture
def gray_frame():
    """A simple 100x200 grayscale test frame."""
    return np.full((100, 200), 128, dtype=np.uint8)


# ---------------------------------------------------------------------------
# resize_frame
# ---------------------------------------------------------------------------

class TestResizeFrame:

    def test_exact_resize_no_aspect(self, bgr_frame):
        """Resize without aspect ratio preservation stretches to exact dims."""
        result = resize_frame(bgr_frame, 64, 64, keep_aspect_ratio=False)
        assert result.shape == (64, 64, 3)

    def test_letterbox_preserves_aspect(self, bgr_frame):
        """Letterbox resize keeps aspect ratio and pads to target dims."""
        # Input is 200x100 (2:1 ratio). Target is 200x200 (1:1).
        # Expected: image scaled to 200x100, padded with 50px top and bottom.
        result = resize_frame(bgr_frame, 200, 200, keep_aspect_ratio=True)
        assert result.shape == (200, 200, 3)

    def test_letterbox_padding_is_black(self, bgr_frame):
        """Padding pixels introduced by letterboxing are black (zero)."""
        result = resize_frame(bgr_frame, 200, 200, keep_aspect_ratio=True)
        # Top 50 rows should be black padding.
        assert np.all(result[:50, :] == 0)

    def test_resize_grayscale(self, gray_frame):
        """Grayscale frame resizes without error."""
        result = resize_frame(gray_frame, 50, 50, keep_aspect_ratio=False)
        assert result.shape == (50, 50)

    def test_resize_preserves_dtype(self, bgr_frame):
        """Output dtype matches input dtype."""
        result = resize_frame(bgr_frame, 64, 64, keep_aspect_ratio=False)
        assert result.dtype == np.uint8

    def test_resize_step1_larger(self, bgr_frame):
        """Upscaling to a larger resolution works without error."""
        result = resize_frame(bgr_frame, 400, 300, keep_aspect_ratio=False)
        assert result.shape == (300, 400, 3)


# ---------------------------------------------------------------------------
# convert_colour_space
# ---------------------------------------------------------------------------

class TestConvertColourSpace:

    def test_bgr_to_rgb_swaps_channels(self, bgr_frame):
        """BGR->RGB conversion swaps first and last channels."""
        result = convert_colour_space(bgr_frame, "rgb")
        # Original B channel (index 0) should now be R (index 2).
        assert result.shape == bgr_frame.shape
        np.testing.assert_array_equal(result[:, :, 0], bgr_frame[:, :, 2])
        np.testing.assert_array_equal(result[:, :, 2], bgr_frame[:, :, 0])

    def test_bgr_noop(self, bgr_frame):
        """BGR mode returns frame unchanged."""
        result = convert_colour_space(bgr_frame, "bgr")
        np.testing.assert_array_equal(result, bgr_frame)

    def test_grayscale_reduces_channels(self, bgr_frame):
        """Grayscale conversion produces a 2D array."""
        result = convert_colour_space(bgr_frame, "grayscale")
        assert result.ndim == 2
        assert result.shape == (100, 200)

    def test_invalid_mode_raises(self, bgr_frame):
        """Unknown colour mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown colour space mode"):
            convert_colour_space(bgr_frame, "xyz")


# ---------------------------------------------------------------------------
# apply_transforms
# ---------------------------------------------------------------------------

class TestApplyTransforms:

    def test_both_transforms_applied(self, bgr_frame):
        """Resize and colour conversion both applied in correct order."""
        result, applied = apply_transforms(
            bgr_frame,
            resize_enabled=True,
            target_width=64,
            target_height=64,
            keep_aspect_ratio=False,
            colour_mode="rgb",
            colour_enabled=True,
        )
        assert result.shape == (64, 64, 3)
        assert any("resize" in t for t in applied)
        assert any("colour_space" in t for t in applied)

    def test_resize_only(self, bgr_frame):
        """Only resize applied when colour disabled."""
        result, applied = apply_transforms(
            bgr_frame,
            resize_enabled=True,
            target_width=64,
            target_height=64,
            colour_enabled=False,
        )
        assert result.shape == (64, 64, 3)
        assert len(applied) == 1
        assert "resize" in applied[0]

    def test_colour_only(self, bgr_frame):
        """Only colour conversion applied when resize disabled."""
        result, applied = apply_transforms(
            bgr_frame,
            resize_enabled=False,
            colour_mode="grayscale",
            colour_enabled=True,
        )
        assert result.ndim == 2
        assert len(applied) == 1
        assert "colour_space" in applied[0]

    def test_no_transforms(self, bgr_frame):
        """No transforms returns original frame and empty list."""
        result, applied = apply_transforms(
            bgr_frame,
            resize_enabled=False,
            colour_enabled=False,
        )
        np.testing.assert_array_equal(result, bgr_frame)
        assert applied == []