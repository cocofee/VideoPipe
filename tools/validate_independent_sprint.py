"""Validate frozen video windows through the production independent outlet.

The counter never reads chip data. Optional expected_count must come from
visual annotation, and is applied only after inference has completed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import cv2


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from realtime.detector import Detector  # noqa: E402
from realtime.independent_counting import CountRules, FrameStamp  # noqa: E402
from realtime.count_storage import atomic_report  # noqa: E402


def validate(manifest, output, *, imgsz=1280):
    from ultralytics import YOLO
    model_path = ROOT / manifest.get("model", "yolo26n.pt")
    detector = Detector(str(model_path), model=YOLO(str(model_path)), ocr=None)
    detector.process_imgsz = imgsz
    detector.detection_conf = manifest.get("confidence", 0.06)
    detector.adaptive_frame_skip = False
    events, diagnostics, notices = [], [], []
    capture = cv2.VideoCapture(manifest["video_path"])
    if not capture.isOpened():
        raise ValueError("fixture recording cannot be opened")
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rules = CountRules("sprint-validation", "visual-truth", "finish", "finish", "camera-1",
                      tuple(tuple(point) for point in manifest.get("line", ((0.518 * width, 0), (0.518 * width, height)))),
                      "pos_to_neg", tuple(manifest.get("allowed_classes", [0])),
                      anchor=manifest.get("anchor", "center"), side_observations=manifest.get("side_observations", 2))
    detector.configure_counting(rules, events.append,
                                lambda stamp, raw, linked, decisions, quality: (diagnostics.append({"frame": asdict(stamp), "raw": len(raw), "linked": len(linked), "decisions": decisions}), notices.extend(quality)),
                                count_only=True)
    capture.set(cv2.CAP_PROP_POS_MSEC, manifest["start_ms"])
    started, latencies, frames = time.monotonic(), [], 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            media_ms = round(capture.get(cv2.CAP_PROP_POS_MSEC))
            if media_ms > manifest["end_ms"]:
                break
            sequence = round(capture.get(cv2.CAP_PROP_POS_FRAMES))
            stamp = FrameStamp(sequence, "frozen-file", media_ms, int(time.time() * 1000), "source_pts", media_ms)
            before = time.monotonic()
            detector.process_frame(frame, media_ms / 1000, frame_stamp=stamp)
            latencies.append((time.monotonic() - before) * 1000)
            frames += 1
    finally:
        capture.release()
        detector.stop()
    group_events = [event for event in events if manifest.get("group_start_ms", manifest["start_ms"]) <= event.estimated_cross_time_ms <= manifest.get("group_end_ms", manifest["end_ms"])]
    admitted = [event for event in group_events if event.eligibility == "admitted"]
    result = {"fixture": manifest, "imgsz": imgsz, "frames": frames, "elapsed_seconds": time.monotonic() - started,
              "tracker_config": (ROOT / "realtime/count_bytetrack.yaml").read_text(encoding="utf-8"),
              "video_count": len(admitted), "pending_count": sum(event.eligibility == "pending" for event in group_events),
              "excluded_count": sum(event.eligibility == "excluded" for event in group_events),
              "quality_reasons": dict(Counter(notice["reason"] for notice in notices)),
              "max_processing_ms": max(latencies, default=0),
              "events": [event.payload() for event in events], "frame_metrics": diagnostics,
              "expected_count": manifest.get("expected_count"),
              "visible_lower_bound": manifest.get("visible_lower_bound"),
              "meets_visible_lower_bound": len(admitted) >= manifest["visible_lower_bound"] if "visible_lower_bound" in manifest else None,
              "passed": len(admitted) == manifest["expected_count"] if "expected_count" in manifest else None}
    if result["meets_visible_lower_bound"] is False:
        result["passed"] = False
    atomic_report(output, result)
    print(json.dumps({key: result[key] for key in ("frames", "elapsed_seconds", "video_count", "pending_count", "quality_reasons", "expected_count", "passed")}, ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    args = parser.parse_args()
    result = validate(json.loads(args.manifest.read_text(encoding="utf-8")), args.output, imgsz=args.imgsz)
    return 1 if result["passed"] is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
