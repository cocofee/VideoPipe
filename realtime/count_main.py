"""Run a single-camera independent counting service alongside FinishReview."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import uuid
import hashlib

from .count_reconciliation import ClockMapping, ReconciliationScope
from .count_runtime import CountRuntime, make_replay
from .count_rescan import RecordingEvidenceProvider
from .independent_counting import CountRules, FrameStamp


def load_config(path):
    path = Path(path).absolute()
    config = json.loads(path.read_text(encoding="utf-8"))
    rules_data = dict(config["rules"])
    rules_data["line"] = tuple(tuple(point) for point in rules_data["line"])
    rules_data["allowed_classes"] = tuple(rules_data["allowed_classes"])
    rules_data["roi"] = tuple(tuple(point) for point in rules_data.get("roi", ()))
    rules = CountRules(**rules_data)
    scope = ReconciliationScope(**{name: getattr(rules, name) for name in (
        "session_id", "race_id", "stage_id", "timing_position", "source_id")}, **config.get("reconciliation", {}))
    clock = ClockMapping(**config["clock"])
    if clock.error_ms < 0 or abs(clock.drift_ppm) > 10000:
        raise ValueError("invalid clock error or drift")
    if scope.group_ids:
        raise ValueError("anonymous counting requires an overall comparable group scope")
    for key in ("model_path", "chip_journal", "output_dir", "video_timeline"):
        if key not in config:
            continue
        value = Path(config[key])
        config[key] = str(value if value.is_absolute() else path.parent / value)
    if not Path(config["model_path"]).is_file():
        raise ValueError("model_path must name an existing local model")
    source = config["source"]
    if isinstance(source, str) and not source.startswith(("rtsp://", "http://", "https://")):
        source_path = Path(source)
        source_path = source_path if source_path.is_absolute() else path.parent / source_path
        if not source_path.is_file():
            raise ValueError("video source does not exist")
        config["source"] = str(source_path)
        if "recording_start_ms" not in config:
            raise ValueError("file input requires recording_start_ms in the camera clock domain")
    if config.get("end_position_ms", float("inf")) <= config.get("start_position_ms", 0):
        raise ValueError("end_position_ms must follow start_position_ms")
    return config, rules, scope, clock


def run(config, rules, scope, clock, *, max_frames=None):
    import cv2
    import numpy as np
    from ultralytics import YOLO
    from .detector import Detector
    from .stream_reader import StreamReader

    model_path = config["model_path"]
    fingerprint = hashlib.sha256()
    with Path(model_path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            fingerprint.update(chunk)
    fingerprint.update(Path(__file__).with_name("count_bytetrack.yaml").read_text(encoding="utf-8").encode())
    fingerprint.update(json.dumps({"imgsz": config.get("imgsz", 640),
                                   "replay_imgsz": config.get("replay_imgsz", 1280),
                                   "confidence": config.get("confidence", 0.06)}, sort_keys=True).encode())
    is_file = isinstance(config["source"], str) and Path(config["source"]).is_file()
    provider = None
    if is_file:
        provider = RecordingEvidenceProvider(source_file=config["source"], recording_start_ms=config["recording_start_ms"])
    elif config.get("video_timeline"):
        provider = RecordingEvidenceProvider(timeline=config["video_timeline"], camera_index=config.get("camera_index", 1),
                                              race_id=rules.race_id, timing_error_ms=config.get("recording_timing_error_ms", 0))
    # A separate replay model is loaded lazily on the first anomaly.
    replay = make_replay(model_path, rules, imgsz=config.get("replay_imgsz", 1280),
                         confidence=config.get("confidence", 0.06),
                         timeout_seconds=config.get("replay_timeout_seconds", 20))
    runtime = CountRuntime(rules, scope, clock, config["output_dir"], config["chip_journal"], replay,
                           buffer_bytes=config.get("buffer_bytes", 256 * 1024 * 1024),
                           retention_ms=config.get("buffer_retention_ms", 15000), recording_provider=provider,
                           pipeline_fingerprint=fingerprint.hexdigest())
    detector = None
    reader = None
    capture = None
    frames_processed = 0
    try:
        detector = Detector(model_path, model=YOLO(model_path), ocr=None)
        detector.process_imgsz = config.get("imgsz", 640)
        detector.detection_conf = config.get("confidence", 0.06)
        # Pay first-inference initialization before starting the live reader.
        detector._model.predict(np.zeros((detector.process_imgsz, detector.process_imgsz, 3), dtype=np.uint8),
                                imgsz=detector.process_imgsz, conf=detector.detection_conf, verbose=False)
        detector.configure_counting(rules, runtime.on_event, runtime.on_diagnostics, count_only=True)
        detector.enable_roi_filter = False
        detector.adaptive_frame_skip = False
        generation = uuid.uuid4().hex
        if is_file:
            generation = hashlib.sha256(f"{Path(config['source']).absolute()}:{config['recording_start_ms']}".encode()).hexdigest()[:24]
        start = None
        last_sequence, last_generation = 0, ""
        if is_file:
            capture = cv2.VideoCapture(config["source"])
            if not capture.isOpened():
                raise ValueError("cannot open input video")
            fps = capture.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                raise ValueError("file input requires a valid frame rate")
            capture.set(cv2.CAP_PROP_POS_MSEC, config.get("start_position_ms", 0))
            previous_pts = None
        else:
            reader = StreamReader(config["source"])
            reader.set_on_frame_packet(runtime.buffer.append)
            if not reader.start():
                raise ValueError("cannot open camera source")
        while max_frames is None or frames_processed < max_frames:
            if is_file:
                ok, frame = capture.read()
                if not ok:
                    break
                frames_processed += 1
                raw_pts = capture.get(cv2.CAP_PROP_POS_MSEC)
                if raw_pts < 0 or previous_pts is not None and raw_pts <= previous_pts:
                    raise ValueError("video decoder does not provide monotonic source timestamps")
                previous_pts = raw_pts
                media_ms = round(raw_pts)
                if media_ms > config.get("end_position_ms", float("inf")):
                    break
                sequence = round(capture.get(cv2.CAP_PROP_POS_FRAMES))
                stamp = FrameStamp(sequence, generation, config["recording_start_ms"] + media_ms,
                                   int(time.time() * 1000), "recording_metadata", media_ms)
                runtime.buffer.append(frame, stamp)
            else:
                frame, stamp = reader.get_frame_packet_after(last_sequence, last_generation)
                if frame is None:
                    runtime.runtime_quality = ["camera_disconnected"] if not reader.is_connected else []
                    runtime.tick()
                    time.sleep(0.005)
                    continue
                frames_processed += 1
                last_sequence, last_generation = stamp.sequence, stamp.generation
            if start is None:
                start = stamp.observed_at_ms
            try:
                detector.process_frame(frame, timestamp=(stamp.observed_at_ms - start) / 1000, frame_stamp=stamp)
            except Exception as error:
                runtime.add_quality({"source_id": rules.source_id, "reason": "inference_failed",
                                     "start_ms": stamp.observed_at_ms, "end_ms": stamp.observed_at_ms})
                raise RuntimeError("independent counting frame failed") from error
            runtime.runtime_quality = [] if Path(config["chip_journal"]).exists() else ["chip_journal_waiting"]
            runtime.tick()
        runtime.tick(force=True)
        # Offline validation drains the bounded worker; live shutdown is also
        # cooperative and retains unresolved tasks when evidence is insufficient.
        while runtime.rescans.running or runtime.rescans.pending:
            runtime.tick(force=True)
            time.sleep(0.01)
    except KeyboardInterrupt:
        runtime.runtime_quality = ["stopped"]
    finally:
        if reader is not None:
            reader.stop()
        if capture is not None:
            capture.release()
        if detector is not None:
            detector.stop()
        runtime.close()
    return frames_processed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-config", action="store_true")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args(argv)
    try:
        config, rules, scope, clock = load_config(args.config)
        if args.validate_config:
            print("Configuration valid; clock calibrated:", clock.calibrated)
            return 0
        count = run(config, rules, scope, clock, max_frames=args.max_frames)
        print(f"Processed {count} frames; report: {Path(config['output_dir']) / 'count_reconciliation.json'}")
        return 0
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as error:
        parser.exit(2, f"Count service error: {type(error).__name__}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
