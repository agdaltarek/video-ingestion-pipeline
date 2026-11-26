"""
One-off script to generate synthetic test videos for pipeline development.
Produces three valid videos of varying resolution/duration, plus one corrupt file.
Run once from the project root: python data/generate_test_videos.py
"""

import cv2
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "data"


def make_video(path: Path, width: int, height: int, fps: float, duration_s: float):
    """Write a synthetic sports-like video: moving coloured blobs on a green pitch."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    total_frames = int(fps * duration_s)

    rng = np.random.default_rng(seed=42)  # seeded — reproducible

    # Simulate 5 "players" as moving circles
    positions = rng.integers(50, min(width, height) - 50, size=(5, 2)).astype(float)
    velocities = rng.uniform(-3, 3, size=(5, 2))
    colours = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

    for _ in range(total_frames):
        frame = np.full((height, width, 3), (34, 139, 34), dtype=np.uint8)  # green pitch

        for i, (pos, col) in enumerate(zip(positions, colours)):
            positions[i] += velocities[i]
            # Bounce off walls
            for axis, limit in enumerate([width, height]):
                if positions[i][axis] < 20 or positions[i][axis] > limit - 20:
                    velocities[i][axis] *= -1
            cv2.circle(frame, (int(pos[0]), int(pos[1])), 15, col, -1)

        writer.write(frame)

    writer.release()
    print(f"  Created: {path.name}  ({width}x{height}, {fps}fps, {duration_s}s)")


def make_corrupt(path: Path):
    """Write a file that looks like an MP4 but has truncated/garbage content."""
    with open(path, "wb") as f:
        # Valid MP4 ftyp box header, then pure garbage — ffprobe will reject it
        f.write(bytes.fromhex("0000001c6674797069736f6d"))  # ftyp header
        f.write(b"\x00" * 512)  # truncated / zeroed body
    print(f"  Created: {path.name}  (deliberately corrupt)")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating test videos...")

    # Video 1 — 480p, 25fps, 10 seconds (low-res short clip)
    make_video(OUTPUT_DIR / "test_480p_10s.mp4", 854, 480, 25.0, 10.0)

    # Video 2 — 720p, 30fps, 30 seconds (medium-res, longer)
    make_video(OUTPUT_DIR / "test_720p_30s.mp4", 1280, 720, 30.0, 30.0)

    # Video 3 — 1080p, 25fps, 15 seconds (high-res, different fps)
    make_video(OUTPUT_DIR / "test_1080p_15s.mp4", 1920, 1080, 25.0, 15.0)

    # Corrupt file — for error handling tests
    make_corrupt(OUTPUT_DIR / "test_corrupt.mp4")

    print("Done. Files written to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()