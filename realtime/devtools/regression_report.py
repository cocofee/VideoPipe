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
    _inference_ms: List[float] = field(default_factory=list)
    _postprocess_ms: List[float] = field(default_factory=list)
    _total_ms: List[float] = field(default_factory=list)

    def observe_frame(self, frame_metrics: Dict[str, Any], crossing_events: Iterable[Any]) -> None:
        self.frames_processed += 1
        self.raw_bike_detections += int(frame_metrics.get("raw_bikes", 0) or 0)
        self.raw_bib_detections += int(frame_metrics.get("raw_bibs", 0) or 0)
        self.validated_track_observations += int(frame_metrics.get("validated_tracks", 0) or 0)
        self.synthetic_track_observations += int(frame_metrics.get("synthetic_tracks", 0) or 0)
        self.crossing_events += sum(1 for _ in crossing_events)
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


def result_signature(report: Dict[str, Any]) -> str:
    """Hash deterministic detection results while excluding runtime-only fields."""

    event_keys = (
        "event_id",
        "track_id",
        "cross_time",
        "bib_number",
        "bib_status",
        "has_bib_box",
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
    parser.add_argument("--frames", type=int, default=1800, help="处理帧数（默认 1800 帧 ≈ 60 秒@30fps）")
    parser.add_argument("--start-frame", type=int, default=0, help="起始帧（默认 0）")
    parser.add_argument("--config", type=str, default="", help="可选：config.json 或 config_preset_*.json（用于复用终点线/ROI）")
    parser.add_argument("--out", type=str, default="", help="输出报告 json 路径（默认放到当前目录）")
    parser.add_argument("--verbose", action="store_true", help="打印检测器 INFO 日志")
    args = parser.parse_args()

    import cv2
    import numpy as np
    from ultralytics import YOLO

    from realtime.detector import Detector
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

    detector = Detector(model_path=str(model_path), source_id=0, model=yolo, ocr=None)
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
    regression_metrics = RegressionMetrics()

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
        regression_metrics.observe_frame(detector.last_frame_metrics, crossing_events)

        athletes_sum += len(athletes)
        athletes_max = max(athletes_max, len(athletes))
        bibs_sum += len(bibs)

        if crossing_events:
            with detector._lock:
                for ev in crossing_events:
                    state = detector._track_states.get(ev.track_id)
                    events.append(
                        {
                            "event_id": ev.event_id,
                            "track_id": ev.track_id,
                            "cross_time": ev.cross_time,
                            "cross_realtime": ev.cross_realtime,
                            "bib_number": ev.bib_number,
                            "bib_status": str(ev.bib_status),
                            "has_bib_box": bool(getattr(state, "has_bib_box", False)) if state else False,
                            "bbox": list(ev.bbox),
                            "position": list(ev.position),
                        }
                    )

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
            "athletes_avg_per_frame": float(round(athletes_sum / max(1, frames_done), 3)),
            "athletes_max_per_frame": int(athletes_max),
            "bibs_avg_per_frame": float(round(bibs_sum / max(1, frames_done), 3)),
        },
        "latency_ms": metric_report["latency_ms"],
        "events": events,
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
    print(f"- athletes_avg_per_frame: {report['counts']['athletes_avg_per_frame']} (max: {athletes_max})")
    print(f"- bibs_avg_per_frame: {report['counts']['bibs_avg_per_frame']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
