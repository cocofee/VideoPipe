#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import hashlib
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class RegressionMetrics:
    """Aggregate detector layers without writing race or event data."""

    frames_processed: int = 0
    raw_bike_detections: int = 0
    raw_bib_detections: int = 0
    validated_track_observations: int = 0
    synthetic_track_observations: int = 0
    crossing_events: int = 0
    rejected_candidates: int = 0
    _inference_ms: List[float] = field(default_factory=list)
    _postprocess_ms: List[float] = field(default_factory=list)
    _total_ms: List[float] = field(default_factory=list)

    def observe_frame(
        self,
        frame_metrics: Dict[str, Any],
        crossing_events: Iterable[Any],
        rejected_candidates: Iterable[Any] = (),
    ) -> None:
        self.frames_processed += 1
        self.raw_bike_detections += int(frame_metrics.get("raw_bikes", 0) or 0)
        self.raw_bib_detections += int(frame_metrics.get("raw_bibs", 0) or 0)
        self.validated_track_observations += int(frame_metrics.get("validated_tracks", 0) or 0)
        self.synthetic_track_observations += int(frame_metrics.get("synthetic_tracks", 0) or 0)
        self.crossing_events += sum(1 for _ in crossing_events)
        self.rejected_candidates += sum(1 for _ in rejected_candidates)
        self._inference_ms.append(float(frame_metrics.get("inference_ms", 0.0) or 0.0))
        self._postprocess_ms.append(float(frame_metrics.get("postprocess_ms", 0.0) or 0.0))
        self._total_ms.append(float(frame_metrics.get("total_ms", 0.0) or 0.0))

    @staticmethod
    def _latency_summary(samples: List[float]) -> Dict[str, float]:
        if not samples:
            return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}

        ordered = sorted(samples)

        def percentile(value: float) -> float:
            index = max(0, math.ceil(value * len(ordered)) - 1)
            return float(ordered[index])

        return {
            "avg": round(sum(ordered) / len(ordered), 3),
            "p50": round(percentile(0.50), 3),
            "p95": round(percentile(0.95), 3),
            "max": round(float(ordered[-1]), 3),
        }

    def to_report(self) -> Dict[str, Any]:
        return {
            "frames_processed": self.frames_processed,
            "counts": {
                "raw_bike_detections": self.raw_bike_detections,
                "raw_bib_detections": self.raw_bib_detections,
                "validated_track_observations": self.validated_track_observations,
                "synthetic_track_observations": self.synthetic_track_observations,
                "crossing_events": self.crossing_events,
                "rejected_candidates": self.rejected_candidates,
            },
            "latency_ms": {
                "inference": self._latency_summary(self._inference_ms),
                "postprocess": self._latency_summary(self._postprocess_ms),
                "total": self._latency_summary(self._total_ms),
            },
        }


def validate_report_output_path(path: Path) -> Path:
    """Reject report destinations inside formal RaceData directories."""

    resolved = path.expanduser().resolve()
    if any(part.casefold() == "racedata" for part in resolved.parts):
        raise ValueError(f"Regression reports must not be written inside RaceData: {resolved}")
    return resolved


def video_timestamp_seconds(position_ms: float, frame_index: int, fps: float) -> float:
    """Return deterministic media time for offline detector processing."""

    if position_ms > 0.0:
        return float(position_ms) / 1000.0
    if fps > 0.0:
        return float(frame_index) / float(fps)
    return float(frame_index)


def build_event_record(event: Any, state: Any, frame_index: int, media_time: float) -> Dict[str, Any]:
    """Build a deterministic event record with its offline video position."""

    return {
        "event_id": event.event_id,
        "track_id": event.track_id,
        "cross_time": event.cross_time,
        "cross_realtime": event.cross_realtime,
        "frame_index": int(frame_index),
        "media_time": float(media_time),
        "bib_number": event.bib_number,
        "bib_status": str(event.bib_status),
        "has_bib_box": bool(getattr(state, "has_bib_box", False)) if state else False,
        "synthetic_kind": getattr(state, "synthetic_kind", None) if state else None,
        "bbox": list(event.bbox),
        "position": list(event.position),
    }


def build_rejected_candidate_record(
    candidate: Dict[str, Any],
    rejection_id: int,
    frame_index: int,
    media_time: float,
) -> Dict[str, Any]:
    """Build a reviewable record without allocating an official event ID."""

    return {
        "rejection_id": int(rejection_id),
        "track_id": int(candidate.get("track_id", -1)),
        "reason": str(candidate.get("reason") or "unknown"),
        "frame_index": int(frame_index),
        "media_time": float(media_time),
        "has_bib_box": bool(candidate.get("has_bib_box", False)),
        "bbox": list(candidate.get("bbox") or []),
        "position": list(candidate.get("position") or []),
    }


def save_event_evidence(
    event: Any,
    record: Dict[str, Any],
    evidence_dir: Path,
    state: Any = None,
) -> Dict[str, Path]:
    """Persist the exact frame and crops captured by the crossing event."""

    import cv2

    output_dir = validate_report_output_path(Path(evidence_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"event_{int(record['event_id']):03d}_track_{int(record['track_id'])}"
        f"_frame_{int(record['frame_index']):06d}"
    )
    evidence: Dict[str, Path] = {}
    for name, image in (
        ("frame", getattr(event, "frame", None)),
        ("athlete", getattr(event, "crop", None)),
        ("bib", getattr(event, "bib_crop", None)),
    ):
        if image is None or not hasattr(image, "size") or image.size == 0:
            continue
        path = output_dir / f"{stem}_{name}.jpg"
        if cv2.imwrite(str(path), image):
            evidence[name] = path

    if state is not None:
        for prefix, cache in (
            ("bib_candidate", getattr(state, "bib_crops_cache", [])),
            ("fallback_candidate", getattr(state, "fallback_bib_crops_cache", [])),
        ):
            for index, candidate in enumerate(cache[:4], start=1):
                if len(candidate) < 2:
                    continue
                image = candidate[1]
                if image is None or not hasattr(image, "size") or image.size == 0:
                    continue
                key = f"{prefix}_{index:02d}"
                path = output_dir / f"{stem}_{key}.jpg"
                if cv2.imwrite(str(path), image):
                    evidence[key] = path
    return evidence


def save_rejected_candidate_evidence(
    candidate: Dict[str, Any],
    record: Dict[str, Any],
    evidence_dir: Path,
) -> Dict[str, Path]:
    """Persist rejected evidence separately from official crossing events."""

    import cv2

    output_dir = validate_report_output_path(Path(evidence_dir)) / "rejected"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"rejected_{int(record['rejection_id']):03d}_track_{int(record['track_id'])}"
        f"_frame_{int(record['frame_index']):06d}"
    )
    evidence: Dict[str, Path] = {}
    for name, image in (
        ("frame", candidate.get("frame")),
        ("athlete", candidate.get("crop")),
        ("bib", candidate.get("bib_crop")),
    ):
        if image is None or not hasattr(image, "size") or image.size == 0:
            continue
        path = output_dir / f"{stem}_{name}.jpg"
        if cv2.imwrite(str(path), image):
            evidence[name] = path
    return evidence


def result_signature(report: Dict[str, Any]) -> str:
    """Hash deterministic detection results while excluding runtime-only fields."""

    event_keys = (
        "event_id",
        "track_id",
        "frame_index",
        "media_time",
        "bib_number",
        "bib_status",
        "has_bib_box",
        "synthetic_kind",
        "bbox",
        "position",
    )
    payload = {
        "frames_processed": int(report.get("frames_processed", 0) or 0),
        "counts": report.get("counts", {}),
        "events": [
            {key: event.get(key) for key in event_keys if key in event}
            for event in report.get("events", [])
        ],
        "rejected_candidates": [
            {
                key: candidate.get(key)
                for key in (
                    "rejection_id",
                    "track_id",
                    "reason",
                    "frame_index",
                    "media_time",
                    "has_bib_box",
                    "bbox",
                    "position",
                )
                if key in candidate
            }
            for candidate in report.get("rejected_candidates", [])
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

def _force_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _extract_line_and_roi(config: Dict[str, Any], source_id: int = 0) -> Tuple[Optional[Dict[str, int]], Optional[list]]:
    line_cfg = None
    try:
        finish_lines = config.get("finish_lines", {})
        if isinstance(finish_lines, dict):
            line_cfg = finish_lines.get(str(source_id))
        if not line_cfg and source_id == 0:
            line_cfg = config.get("finish_line")
    except Exception:
        line_cfg = None

    roi_points = None
    try:
        rois = config.get("rois", {})
        if isinstance(rois, dict):
            roi_points = rois.get(str(source_id))
        if not roi_points and source_id == 0:
            roi_points = config.get("roi_points")
    except Exception:
        roi_points = None

    if not isinstance(line_cfg, dict):
        line_cfg = None
    if not isinstance(roi_points, list):
        roi_points = None
    return line_cfg, roi_points


def main() -> int:
    # Windows 控制台经常有编码问题；工具输出尽量使用 ASCII，避免乱码。
    try:
        logging.basicConfig(level=logging.WARNING)
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="VideoPipe regression report (offline video, no parameter changes)")
    parser.add_argument("--video", required=True, help="视频路径")
    parser.add_argument("--model", required=True, help="模型路径（.pt 或 .engine）")
    parser.add_argument("--validator-model", type=str, default="", help="可选：无号码事件的自行车二次校验模型")
    parser.add_argument("--frames", type=int, default=1800, help="处理帧数（默认 1800 帧 ≈ 60 秒@30fps）")
    parser.add_argument("--start-frame", type=int, default=0, help="起始帧（默认 0）")
    parser.add_argument("--config", type=str, default="", help="可选：config.json 或 config_preset_*.json（用于复用终点线/ROI）")
    parser.add_argument("--out", type=str, default="", help="输出报告 json 路径（默认放到当前目录）")
    parser.add_argument("--evidence-dir", type=str, default="", help="可选：保存事件原始帧、运动员裁剪和号码牌裁剪")
    parser.add_argument("--verbose", action="store_true", help="打印检测器 INFO 日志")
    args = parser.parse_args()

    import cv2
    import numpy as np
    from ultralytics import YOLO

    from realtime.detector import Detector, LOCAL_VIDEO_EVENT_SETTLE_SECONDS
    if not args.verbose:
        try:
            logging.getLogger().setLevel(logging.ERROR)
            logging.getLogger("VideoPipe").setLevel(logging.ERROR)
        except Exception:
            pass

    video_path = Path(args.video)
    model_path = Path(args.model)
    if not video_path.exists():
        print(f"[FAIL] 视频不存在: {video_path}")
        return 2
    if not model_path.exists():
        print(f"[FAIL] 模型不存在: {model_path}")
        return 2

    validator_model_path = Path(args.validator_model) if args.validator_model else None
    if validator_model_path is not None and not validator_model_path.exists():
        print(f"[FAIL] 二次校验模型不存在: {validator_model_path}")
        return 2

    config = None
    if args.config:
        cfg_path = Path(args.config)
        if cfg_path.exists():
            config = _load_json(cfg_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[FAIL] 无法打开视频: {video_path}")
        return 2

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    yolo = YOLO(str(model_path))
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    yolo.predict(source=dummy, verbose=False)

    athlete_validator = None
    if validator_model_path is not None:
        athlete_validator = YOLO(str(validator_model_path))
        athlete_validator.predict(
            source=np.zeros((320, 320, 3), dtype=np.uint8),
            imgsz=320,
            conf=0.20,
            verbose=False,
            device="cpu",
        )

    detector = Detector(
        model_path=str(model_path),
        source_id=0,
        model=yolo,
        ocr=None,
        athlete_validator=athlete_validator,
        event_settle_seconds=LOCAL_VIDEO_EVENT_SETTLE_SECONDS,
    )
    detector.realtime_ocr_enabled = False

    line_cfg = None
    roi_points = None
    crossing_direction = None
    if config:
        line_cfg, roi_points = _extract_line_and_roi(config, source_id=0)
        crossing_direction = config.get("crossing_direction")

    if line_cfg and all(k in line_cfg for k in ("x1", "y1", "x2", "y2")):
        detector.set_finish_line((int(line_cfg["x1"]), int(line_cfg["y1"])), (int(line_cfg["x2"]), int(line_cfg["y2"])))
    else:
        # 默认：用一条水平线（仅用于离线回归；现场以你手动画线为准）
        line_y = int(height * 0.60)
        detector.set_finish_line((0, line_y), (width, line_y))

    if roi_points and len(roi_points) >= 3:
        detector.set_roi_polygon(roi_points)
    else:
        detector.set_roi_polygon(None)

    if isinstance(crossing_direction, str) or crossing_direction is None:
        detector.set_crossing_direction(crossing_direction)

    frames_target = int(max(1, args.frames))
    frames_done = 0
    athletes_sum = 0
    athletes_max = 0
    bibs_sum = 0
    events = []
    rejected_candidates = []
    regression_metrics = RegressionMetrics()
    evidence_dir = Path(args.evidence_dir) if args.evidence_dir else None

    t0 = time.time()
    while frames_done < frames_target:
        ok, frame = cap.read()
        if not ok:
            break

        video_timestamp = video_timestamp_seconds(
            cap.get(cv2.CAP_PROP_POS_MSEC),
            frame_index=int(args.start_frame) + frames_done,
            fps=fps,
        )
        crossing_events, athletes, bibs = detector.process_frame(frame, timestamp=video_timestamp)
        frame_rejections = list(detector.last_rejected_candidates)
        regression_metrics.observe_frame(
            detector.last_frame_metrics,
            crossing_events,
            rejected_candidates=frame_rejections,
        )

        athletes_sum += len(athletes)
        athletes_max = max(athletes_max, len(athletes))
        bibs_sum += len(bibs)

        if crossing_events:
            with detector._lock:
                for ev in crossing_events:
                    state = detector._track_states.get(ev.track_id)
                    record = build_event_record(
                        ev,
                        state,
                        frame_index=int(args.start_frame) + frames_done,
                        media_time=video_timestamp,
                    )
                    if evidence_dir is not None:
                        saved = save_event_evidence(ev, record, evidence_dir, state=state)
                        record["evidence"] = {name: str(path) for name, path in saved.items()}
                    events.append(record)

        for candidate in frame_rejections:
            record = build_rejected_candidate_record(
                candidate,
                rejection_id=len(rejected_candidates) + 1,
                frame_index=int(args.start_frame) + frames_done,
                media_time=video_timestamp,
            )
            if evidence_dir is not None:
                saved = save_rejected_candidate_evidence(candidate, record, evidence_dir)
                record["evidence"] = {name: str(path) for name, path in saved.items()}
            rejected_candidates.append(record)

        frames_done += 1

    elapsed = time.time() - t0
    proc_fps = (frames_done / elapsed) if elapsed > 1e-6 else 0.0
    metric_report = regression_metrics.to_report()

    report: Dict[str, Any] = {
        "video": str(video_path),
        "model": str(model_path),
        "start_frame": int(args.start_frame),
        "frames_requested": int(frames_target),
        "frames_processed": int(frames_done),
        "video_info": {
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total_frames,
        },
        "performance": {
            "elapsed_sec": float(round(elapsed, 3)),
            "processing_fps": float(round(proc_fps, 2)),
        },
        "counts": {
            **metric_report["counts"],
            "events": len(events),
            "events_unique_tracks": len({e["track_id"] for e in events}),
            "rejected_candidates": len(rejected_candidates),
            "athletes_avg_per_frame": float(round(athletes_sum / max(1, frames_done), 3)),
            "athletes_max_per_frame": int(athletes_max),
            "bibs_avg_per_frame": float(round(bibs_sum / max(1, frames_done), 3)),
        },
        "latency_ms": metric_report["latency_ms"],
        "events": events,
        "rejected_candidates": rejected_candidates,
    }
    report["result_signature"] = result_signature(report)

    try:
        out_path = validate_report_output_path(
            Path(args.out) if args.out else Path.cwd() / f"regression_report_{model_path.stem}.json"
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[FAIL] 写报告失败: {e}")
        return 2

    print("Regression report generated")
    print(f"- out: {out_path}")
    print(f"- frames: {frames_done}/{frames_target}")
    print(f"- processing_fps: {proc_fps:.1f}")
    print(f"- events: {report['counts']['events']} (unique_tracks: {report['counts']['events_unique_tracks']})")
    print(f"- rejected_candidates: {report['counts']['rejected_candidates']}")
    print(f"- athletes_avg_per_frame: {report['counts']['athletes_avg_per_frame']} (max: {athletes_max})")
    print(f"- bibs_avg_per_frame: {report['counts']['bibs_avg_per_frame']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
