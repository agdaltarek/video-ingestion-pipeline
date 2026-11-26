"""
Frame quality assessment heuristics.

Two heuristics are implemented:

1. Blur detection via Laplacian variance.
   The Laplacian operator is a second-order derivative that responds strongly
   to edges. A sharp frame has many strong edges → high variance. A blurry
   frame has few edges → low variance. This is a standard, fast no-reference
   sharpness estimator.

   Sports-domain note: fast camera pans produce motion blur across the whole
   frame, collapsing Laplacian variance sharply. This is a useful signal for
   flagging frames captured during camera movement (exactly the frames where
   player detection accuracy drops most).

2. Brightness / exposure assessment via mean pixel intensity.
   A simple but effective first-pass filter for frames that are too dark
   or overexposed. Both conditions degrade CV model performance significantly.
"""

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger("pipeline.quality")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class QualityReport:
    """Quality assessment results for a single frame."""
    blur_score: float          # Laplacian variance — higher = sharper
    brightness_mean: float     # Mean pixel intensity (0–255)
    is_blurry: bool
    is_too_dark: bool
    is_overexposed: bool

    @property
    def passed(self) -> bool:
        """True if the frame passed all quality checks."""
        return not (self.is_blurry or self.is_too_dark or self.is_overexposed)

    @property
    def flags(self) -> list[str]:
        """List of human-readable quality failure reasons."""
        reasons = []
        if self.is_blurry:
            reasons.append("blurry")
        if self.is_too_dark:
            reasons.append("too_dark")
        if self.is_overexposed:
            reasons.append("overexposed")
        return reasons


# ---------------------------------------------------------------------------
# Core assessment function
# ---------------------------------------------------------------------------

def assess_quality(
    frame: np.ndarray,
    blur_threshold: float = 100.0,
    brightness_low: float = 30.0,
    brightness_high: float = 225.0,
) -> QualityReport:
    """Assess the quality of a single frame.

    The input frame may be BGR or RGB — brightness uses all channels equally
    and blur operates on a grayscale conversion, so channel order does not
    affect the result.

    Args:
        frame:              Input frame as a numpy array (H x W x 3) or
                            (H x W) for grayscale.
        blur_threshold:     Laplacian variance below this value → blurry.
        brightness_low:     Mean pixel value below this → too dark.
        brightness_high:    Mean pixel value above this → overexposed.

    Returns:
        QualityReport with scores and boolean flags.
    """
    # Convert to grayscale for blur detection if needed.
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    # --- Blur: Laplacian variance ---
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # --- Brightness: mean pixel intensity ---
    brightness_mean = float(np.mean(gray))

    return QualityReport(
        blur_score=round(blur_score, 4),
        brightness_mean=round(brightness_mean, 4),
        is_blurry=blur_score < blur_threshold,
        is_too_dark=brightness_mean < brightness_low,
        is_overexposed=brightness_mean > brightness_high,
    )


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def filter_by_quality(
    frames_and_paths: list[tuple[np.ndarray, str]],
    blur_threshold: float = 100.0,
    brightness_low: float = 30.0,
    brightness_high: float = 225.0,
    flag_only: bool = True,
) -> tuple[list[tuple[np.ndarray, str, QualityReport]], int]:
    """Assess quality for a batch of frames and optionally filter poor ones.

    Args:
        frames_and_paths:   List of (frame, path_string) tuples.
        blur_threshold:     Blur threshold (passed to assess_quality).
        brightness_low:     Darkness threshold.
        brightness_high:    Overexposure threshold.
        flag_only:          If True, keep all frames but attach QualityReport.
                            If False, discard frames that fail any check.

    Returns:
        Tuple of:
          - list of (frame, path, QualityReport)
          - number of frames that failed at least one quality check
    """
    results = []
    failed = 0

    for frame, path in frames_and_paths:
        report = assess_quality(frame, blur_threshold, brightness_low, brightness_high)
        if not report.passed:
            failed += 1
            if not flag_only:
                logger.debug("Discarding frame (quality): %s | flags=%s", path, report.flags)
                continue
            else:
                logger.debug("Flagging frame (quality): %s | flags=%s", path, report.flags)
        results.append((frame, path, report))

    return results, failed