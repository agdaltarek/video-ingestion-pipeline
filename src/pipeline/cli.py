"""Command-line entry point for the video ingestion pipeline."""

import argparse
from pathlib import Path

from pipeline import __version__
from pipeline.config import load_config
from pipeline.logging_setup import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-ingest",
        description="Video ingestion and pre-processing pipeline for sports CV.",
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input", help="Path to a video file.")
    parser.add_argument("--stream", help="Streaming URL or local HLS .m3u8 path.")
    parser.add_argument("--log-level", default=None)
    parser.add_argument(
        "--version", action="version", version=f"video-ingestion {__version__}"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    log_level = args.log_level or config.get("logging", {}).get("level", "INFO")
    logger = configure_logging(log_level)
    logger.info("Pipeline initialised (v%s)", __version__)

    if not args.input and not args.stream:
        logger.info("No --input or --stream provided.")
        return

    from pipeline.ingest import probe_video, IngestError
    from pipeline.scenes import detect_scenes_from_config
    from pipeline.output import run_pipeline
    from pipeline.inference import run_inference

    # --- Config sections ---
    ext_cfg    = config.get("extraction", {})
    tr_cfg     = config.get("transforms", {})
    q_cfg      = config.get("quality", {})
    scene_cfg  = config.get("scenes", {})
    out_cfg    = config.get("output", {})
    inf_cfg    = config.get("inference", {})
    stream_cfg = config.get("stream", {})

    resize_cfg = tr_cfg.get("resize", {})
    colour_cfg = tr_cfg.get("colour_space", {})
    blur_cfg   = q_cfg.get("blur", {})
    bright_cfg = q_cfg.get("brightness", {})
    output_dir = Path(ext_cfg.get("output_dir", "output"))

    # -----------------------------------------------------------------------
    # File-based path
    # -----------------------------------------------------------------------
    if args.input:
        try:
            meta = probe_video(Path(args.input))
        except (FileNotFoundError, IngestError) as exc:
            logger.error("Ingestion failed: %s", exc)
            return

        boundaries, scenes = detect_scenes_from_config(
            source_path=Path(args.input),
            scene_cfg=scene_cfg,
            source_fps=meta.fps,
        )
        logger.info("Scenes detected: %d", len(scenes))

        manifest = run_pipeline(
            source_path=Path(args.input),
            scenes=scenes,
            output_dir=output_dir,
            mode=ext_cfg.get("mode", "interval"),
            every_n_frames=int(ext_cfg.get("every_n_frames", 5)),
            target_fps=float(ext_cfg.get("target_fps", 2.0)),
            resize_enabled=resize_cfg.get("enabled", True),
            target_width=int(resize_cfg.get("width", 640)),
            target_height=int(resize_cfg.get("height", 360)),
            keep_aspect_ratio=bool(resize_cfg.get("keep_aspect_ratio", True)),
            colour_mode=colour_cfg.get("mode", "rgb"),
            colour_enabled=colour_cfg.get("enabled", True),
            blur_threshold=float(blur_cfg.get("threshold", 100.0)),
            brightness_low=float(bright_cfg.get("low_threshold", 30.0)),
            brightness_high=float(bright_cfg.get("high_threshold", 225.0)),
            flag_only=bool(q_cfg.get("flag_only", True)),
            output_format=out_cfg.get("format", "jpg"),
            jpg_quality=int(out_cfg.get("jpg_quality", 95)),
            save_keyframes_only=bool(out_cfg.get("save_keyframes_only", False)),
            write_manifest=bool(out_cfg.get("write_manifest", True)),
            manifest_filename=out_cfg.get("manifest_filename", "manifest.json"),
        )
        logger.info("Total frames in manifest: %d", len(manifest))

        if inf_cfg.get("enabled", False):
            import json
            from dataclasses import asdict
            manifest_path = output_dir / Path(args.input).stem / out_cfg.get(
                "manifest_filename", "manifest.json"
            )
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest_dicts = json.load(f)
            inference_results = run_inference(
                manifest_entries=manifest_dicts,
                model_name=inf_cfg.get("model", "yolov8n.pt"),
                confidence=float(inf_cfg.get("confidence", 0.25)),
                device=inf_cfg.get("device", "cpu"),
                max_frames=int(inf_cfg.get("max_frames", 5)),
                save_annotated=bool(inf_cfg.get("save_annotated", True)),
                output_dir=output_dir,
            )
            inference_out = output_dir / Path(args.input).stem / "inference_results.json"
            import json
            from dataclasses import asdict
            with inference_out.open("w", encoding="utf-8") as f:
                json.dump([asdict(r) for r in inference_results], f, indent=2)
            logger.info("Inference results saved: %s", inference_out)

    # -----------------------------------------------------------------------
    # Streaming path
    # -----------------------------------------------------------------------
    elif args.stream:
        from pipeline.stream import iter_stream_frames, detect_protocol
        from pipeline.preprocess import apply_transforms
        from pipeline.quality import assess_quality
        import cv2 as _cv2

        source = args.stream
        protocol = detect_protocol(source)
        logger.info("Streaming mode: protocol=%s source=%s", protocol, source)

        every_n  = int(stream_cfg.get("every_n_frames", 5))
        max_frm  = stream_cfg.get("max_frames", 50)
        max_frm  = int(max_frm) if max_frm is not None else None

        stream_output_dir = output_dir / "stream_output" / "frames"
        stream_output_dir.mkdir(parents=True, exist_ok=True)

        written = 0
        passed  = 0
        flagged = 0

        for frame_index, timestamp_s, frame_bgr in iter_stream_frames(
            source=source,
            every_n_frames=every_n,
            max_frames=max_frm,
            timeout_s=float(stream_cfg.get("connection_timeout_s", 10.0)),
            max_retries=int(stream_cfg.get("max_retries", 3)),
        ):
            # Quality check
            report = assess_quality(
                frame_bgr,
                blur_threshold=float(blur_cfg.get("threshold", 100.0)),
                brightness_low=float(bright_cfg.get("low_threshold", 30.0)),
                brightness_high=float(bright_cfg.get("high_threshold", 225.0)),
            )

            # Transform
            transformed, applied = apply_transforms(
                frame_bgr,
                resize_enabled=resize_cfg.get("enabled", True),
                target_width=int(resize_cfg.get("width", 640)),
                target_height=int(resize_cfg.get("height", 360)),
                keep_aspect_ratio=bool(resize_cfg.get("keep_aspect_ratio", True)),
                colour_mode=colour_cfg.get("mode", "rgb"),
                colour_enabled=colour_cfg.get("enabled", True),
            )

            # Write frame
            out_path = stream_output_dir / f"frame_{frame_index:06d}.jpg"
            write_frame = transformed
            if colour_cfg.get("mode", "rgb") == "rgb" and colour_cfg.get("enabled", True):
                write_frame = _cv2.cvtColor(transformed, _cv2.COLOR_RGB2BGR)
            _cv2.imwrite(str(out_path), write_frame)
            written += 1

            status = "PASS" if report.passed else f"FLAG {report.flags}"
            logger.info(
                "Stream frame %06d | t=%.2fs | blur=%.1f | brightness=%.1f | %s",
                frame_index, timestamp_s,
                report.blur_score, report.brightness_mean, status,
            )

            if report.passed:
                passed += 1
            else:
                flagged += 1

        logger.info(
            "Stream capture complete: %d frames written | %d passed | %d flagged",
            written, passed, flagged,
        )
        logger.info("Frames saved to: %s", stream_output_dir.resolve())


if __name__ == "__main__":
    main()