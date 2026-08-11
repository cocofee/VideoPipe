#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import hashlib
import json
import logging
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPORT_METRIC_KEYS = (
    "frames_processed",
    "processing_fps",
    "raw_tracks",
    "stable_participants",
    "track_fragments_merged",
    "identity_ambiguities",
    "crossing_events",
    "duplicate_passages",
    "rejected_candidates",
    "evidence_complete",
    "ocr_recognized",
    "ocr_conflicts",
    "ocr_unrecognized",
    "queue_depth_max",
    "dropped_frames",
)


def _enum_text(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "").strip().upper()


def _has_image(value: Any) -> bool:
    return value is not None and int(getattr(value, "size", 0) or 0) > 0


def validate_competition_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a manifest that references local inputs only through env vars."""

    if not isinstance(manifest, dict):
        raise ValueError("Competition manifest must be a JSON object")
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported competition manifest schema_version")

    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Competition manifest cases must be a non-empty list")

    seen_case_ids = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each competition manifest case must be an object")
        case_id = str(case.get("case_id") or "").strip()
        if not case_id or case_id in seen_case_ids:
            raise ValueError("Competition manifest case_id values must be unique")
        seen_case_ids.add(case_id)

        for key in case:
            if key.endswith("_path"):
                raise ValueError(
                    f"Manifest paths must use environment variables, not {key}"
                )
        for key in ("video_path_env", "model_path_env"):
            if not str(case.get(key) or "").strip():
                raise ValueError(f"Competition manifest requires {key}")

        profile = case.get("event_profile")
        if not isinstance(profile, dict):
            raise ValueError(f"Case {case_id} requires an event_profile object")
        try:
            from realtime.event_profile import build_event_profile

            build_event_profile(
                name=profile.get("name", case_id),
                pipeline=profile.get("pipeline", "athlete_bib"),
                required_equipment=profile.get("required_equipment"),
                crossing_mode=profile.get("crossing_mode", "finish_once"),
                bib_regions=tuple(profile.get("bib_regions", ("torso", "back"))),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid event_profile for case {case_id}: {exc}") from exc

        for key in ("known_same_participant_tracks", "known_distinct_tracks"):
            groups = case.get(key, [])
            if not isinstance(groups, list):
                raise ValueError(f"Case {case_id} field {key} must be a list")
            for group in groups:
                if (
                    not isinstance(group, list)
                    or len(group) < 2
                    or any(not isinstance(track_id, int) or track_id < 0 for track_id in group)
                ):
                    raise ValueError(f"Case {case_id} field {key} contains an invalid track group")

        expected = case.get("expected_in_clip_crossings")
        if expected is not None and (not isinstance(expected, int) or expected < 0):
            raise ValueError(f"Case {case_id} expected_in_clip_crossings must be a non-negative integer or null")

    return manifest


def load_competition_manifest(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as handle:
        manifest = json.load(handle)
    return validate_competition_manifest(manifest)


def validate_case_report(case: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
    """Check known identity invariants without changing event data."""

    issues: List[str] = []
    track_to_participant: Dict[int, str] = {}
    for participant_id, track_ids in (report.get("participant_tracks") or {}).items():
        for track_id in track_ids or []:
            try:
                track_to_participant[int(track_id)] = str(participant_id)
            except (TypeError, ValueError):
                continue

    for group in case.get("known_same_participant_tracks", []):
        resolved = {track_to_participant.get(int(track_id)) for track_id in group}
        if None in resolved or len(resolved) != 1:
            issues.append(f"known_same_participant_tracks unresolved or split: {group}")

    for group in case.get("known_distinct_tracks", []):
        resolved = {track_to_participant.get(int(track_id)) for track_id in group}
        if None in resolved or len(resolved) != len(group):
            issues.append(f"known_distinct_tracks merged or unresolved: {group}")

    expected_crossings = case.get("expected_in_clip_crossings")
    if expected_crossings is not None and int(report.get("crossing_events", 0) or 0) != expected_crossings:
        issues.append(
            f"crossing count {report.get('crossing_events', 0)} != expected {expected_crossings}"
        )

    if case.get("require_evidence") and int(report.get("evidence_complete", 0) or 0) < int(report.get("crossing_events", 0) or 0):
        issues.append("required event evidence is incomplete")

    baseline_signature = case.get("baseline_signature")
    if baseline_signature and report.get("result_signature") != baseline_signature:
        issues.append("result signature differs from the manifest baseline")

    return {"passed": not issues, "issues": issues}


def aggregate_case_reports(case_reports: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    cases = []
    validation_issues = []
    for case_report in case_reports:
        case = case_report.get("manifest_case") or {}
        validation = validate_case_report(case, case_report)
        case_id = case.get("case_id") or case_report.get("case_id")
        cases.append(
            {
                "case_id": case_id,
                "report": case_report,
                "validation": validation,
            }
        )
        validation_issues.extend(
            f"{case_id}: {issue}" for issue in validation["issues"]
        )

    return {
        "schema_version": 1,
        "cases": cases,
        "validation": {
            "passed": not validation_issues,
            "issues": validation_issues,
        },
    }


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
    _raw_track_ids: set[int] = field(default_factory=set, repr=False)
    _participant_ids: set[str] = field(default_factory=set, repr=False)
    _raw_tracks_max: int = 0
    _participants_max: int = 0
    _track_fragments_merged: int = 0
    _identity_ambiguities: int = 0
    _duplicate_passages: int = 0
    _evidence_complete: int = 0
    _ocr_recognized: int = 0
    _ocr_conflicts: int = 0
    _ocr_unrecognized: int = 0
    _queue_depth_max: int = 0
    _dropped_frames: int = 0
    _passage_keys: set[Tuple[str, int]] = field(default_factory=set, repr=False)

    def observe_frame(
        self,
        frame_metrics: Dict[str, Any],
        crossing_events: Iterable[Any],
        rejected_candidates: Iterable[Any] = (),
        participant_ids: Iterable[Any] = (),
        raw_track_ids: Iterable[Any] = (),
    ) -> None:
        self.frames_processed += 1
        self.raw_bike_detections += int(frame_metrics.get("raw_bikes", 0) or 0)
        self.raw_bib_detections += int(frame_metrics.get("raw_bibs", 0) or 0)
        self.validated_track_observations += int(frame_metrics.get("validated_tracks", 0) or 0)
        self.synthetic_track_observations += int(frame_metrics.get("synthetic_tracks", 0) or 0)
        frame_events = list(crossing_events)
        self.crossing_events += len(frame_events)
        self.rejected_candidates += sum(1 for _ in rejected_candidates)
        frame_raw_track_count = int(frame_metrics.get("raw_tracks", 0) or 0)
        self._raw_tracks_max = max(self._raw_tracks_max, frame_raw_track_count)
        for track_id in raw_track_ids or ():
            try:
                track_id = int(track_id)
            except (TypeError, ValueError):
                continue
            if track_id >= 0:
                self._raw_track_ids.add(track_id)
        frame_participant_count = int(frame_metrics.get("participants", 0) or 0)
        self._participants_max = max(self._participants_max, frame_participant_count)
        for participant_id in participant_ids or ():
            participant_id = str(participant_id or "").strip()
            if participant_id:
                self._participant_ids.add(participant_id)
        self._track_fragments_merged += int(frame_metrics.get("track_fragments_merged", 0) or 0)
        self._identity_ambiguities += int(frame_metrics.get("identity_ambiguities", 0) or 0)
        self._queue_depth_max = max(
            self._queue_depth_max,
            int(frame_metrics.get("queue_depth", 0) or 0),
        )
        self._dropped_frames = max(
            self._dropped_frames,
            int(
                frame_metrics.get(
                    "dropped_frames",
                    frame_metrics.get("dropped_frame_count", 0),
                )
                or 0
            ),
        )
        for event in frame_events:
            participant_id = str(getattr(event, "participant_id", "") or "").strip()
            if participant_id:
                try:
                    passage_index = int(getattr(event, "passage_index", 1) or 1)
                except (TypeError, ValueError):
                    passage_index = 1
                passage_key = (participant_id, passage_index)
                if passage_key in self._passage_keys:
                    self._duplicate_passages += 1
                self._passage_keys.add(passage_key)

            status = _enum_text(getattr(event, "bib_status", ""))
            participant_ocr_status = _enum_text(getattr(event, "participant_ocr_status", ""))
            if "CONFLICT" in status or "CONFLICT" in participant_ocr_status:
                self._ocr_conflicts += 1
            elif status in {"RECOGNIZED", "CONFIRMED", "DONE"}:
                self._ocr_recognized += 1
            else:
                self._ocr_unrecognized += 1

            if _has_image(getattr(event, "frame", None)) and _has_image(getattr(event, "crop", None)):
                self._evidence_complete += 1
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

    def to_report(self, processing_fps: float = 0.0) -> Dict[str, Any]:
        raw_tracks = len(self._raw_track_ids) or self._raw_tracks_max
        stable_participants = len(self._participant_ids) or self._participants_max
        return {
            "frames_processed": self.frames_processed,
            "processing_fps": float(processing_fps),
            "raw_tracks": raw_tracks,
            "stable_participants": stable_participants,
            "track_fragments_merged": self._track_fragments_merged,
            "identity_ambiguities": self._identity_ambiguities,
            "crossing_events": self.crossing_events,
            "duplicate_passages": self._duplicate_passages,
            "rejected_candidates": self.rejected_candidates,
            "evidence_complete": self._evidence_complete,
            "ocr_recognized": self._ocr_recognized,
            "ocr_conflicts": self._ocr_conflicts,
            "ocr_unrecognized": self._ocr_unrecognized,
            "queue_depth_max": self._queue_depth_max,
            "dropped_frames": self._dropped_frames,
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


def write_json_report(path: Path, report: Dict[str, Any]) -> Path:
    output_path = validate_report_output_path(Path(path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    temp_path.replace(output_path)
    return output_path


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
        "participant_id": str(getattr(event, "participant_id", "") or ""),
        "raw_track_ids": list(getattr(event, "raw_track_ids", ()) or ()),
        "passage_index": int(getattr(event, "passage_index", 1) or 1),
        "sport_profile": str(getattr(event, "sport_profile", "") or ""),
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
        "participant_id",
        "raw_track_ids",
        "passage_index",
        "sport_profile",
        "frame_index",
        "media_time",
        "bib_number",
        "bib_status",
        "has_bib_box",
        "synthetic_kind",
        "bbox",
        "position",
    )
    rejected_records = report.get("rejected_candidate_records")
    if rejected_records is None:
        legacy_records = report.get("rejected_candidates", [])
        rejected_records = legacy_records if isinstance(legacy_records, list) else []

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
            for candidate in rejected_records
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
        roi_enabled = config.get("roi_enabled", {})
        source_roi_enabled = None
        if isinstance(roi_enabled, dict):
            source_roi_enabled = roi_enabled.get(str(source_id), roi_enabled.get(source_id))
        rois = config.get("rois", {})
        if isinstance(rois, dict):
            roi_points = rois.get(str(source_id))
        if not roi_points and source_id == 0:
            roi_points = config.get("roi_points")
        if source_roi_enabled is False:
            roi_points = None
    except Exception:
        roi_points = None

    if not isinstance(line_cfg, dict):
        line_cfg = None
    if not isinstance(roi_points, list):
        roi_points = None
    return line_cfg, roi_points


def _parse_frame_rate(value: Any) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        try:
            denominator_value = float(denominator)
            if denominator_value == 0.0:
                return 0.0
            return float(numerator) / denominator_value
        except (TypeError, ValueError):
            return 0.0
    try:
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def _probe_video_with_ffprobe(video_path: Path) -> Dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe is not available")

    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames:format=duration",
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    payload = json.loads(completed.stdout or "{}")
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError("ffprobe did not return a video stream")

    stream = streams[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    nominal_fps = _parse_frame_rate(stream.get("avg_frame_rate")) or _parse_frame_rate(
        stream.get("r_frame_rate")
    )
    duration = float((payload.get("format") or {}).get("duration") or 0.0)
    total_frames = int(stream.get("nb_read_frames") or stream.get("nb_frames") or 0)
    effective_fps = total_frames / duration if total_frames > 0 and duration > 0.0 else 0.0
    fps = effective_fps or nominal_fps
    if total_frames <= 0 and fps > 0.0 and duration > 0.0:
        total_frames = int(round(fps * duration))

    if width <= 0 or height <= 0 or fps <= 0.0:
        raise RuntimeError("ffprobe returned incomplete video metadata")
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "nominal_fps": nominal_fps,
        "total_frames": total_frames,
        "duration": duration,
    }


def _read_exact(stream: Any, size: int) -> bytes:
    chunks = []
    remaining = int(size)
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


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
    parser.add_argument(
        "--sport-profile",
        choices=("cycling", "speed_skating"),
        default="cycling",
        help="赛事检测配置（默认 cycling）",
    )
    parser.add_argument("--event-profile-json", type=str, default="", help="可选：EventProfile JSON 对象")
    parser.add_argument("--out", type=str, default="", help="输出报告 json 路径（默认放到当前目录）")
    parser.add_argument("--evidence-dir", type=str, default="", help="可选：保存事件原始帧、运动员裁剪和号码牌裁剪")
    parser.add_argument("--verbose", action="store_true", help="打印检测器 INFO 日志")
    args = parser.parse_args()

    import cv2
    import numpy as np
    from ultralytics import YOLO

    from realtime.detector import Detector, LOCAL_VIDEO_EVENT_SETTLE_SECONDS
    from realtime.event_profile import build_event_profile, build_sport_event_profile
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
    ffmpeg_process = None
    use_ffmpeg = False
    if cap.isOpened():
        if args.start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
        can_read, _ = cap.read()
        if can_read:
            cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
        else:
            cap.release()
            use_ffmpeg = True
    else:
        use_ffmpeg = True

    if use_ffmpeg:
        try:
            video_info = _probe_video_with_ffprobe(video_path)
        except Exception as exc:
            print(f"[FAIL] 无法读取视频信息: {exc}")
            return 2
        width = int(video_info["width"])
        height = int(video_info["height"])
        fps = float(video_info["fps"])
        nominal_fps = float(video_info["nominal_fps"])
        total_frames = int(video_info["total_frames"])
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            print("[FAIL] OpenCV 无法解码视频，且未找到 ffmpeg")
            return 2
        start_seconds = max(0.0, float(args.start_frame) / fps)
        ffmpeg_process = subprocess.Popen(
            [
                ffmpeg,
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                "-ss",
                f"{start_seconds:.6f}",
                "-an",
                "-sn",
                "-dn",
                "-fps_mode",
                "passthrough",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=max(1, width * height * 3 * 2),
        )
    else:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        nominal_fps = fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

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

    event_profile = build_sport_event_profile(args.sport_profile)
    if args.event_profile_json:
        try:
            profile_payload = json.loads(args.event_profile_json)
            event_profile = build_event_profile(
                name=profile_payload.get("name", "current-event"),
                pipeline=profile_payload.get("pipeline", "athlete_bib"),
                required_equipment=profile_payload.get("required_equipment"),
                crossing_mode=profile_payload.get("crossing_mode", "finish_once"),
                bib_regions=tuple(profile_payload.get("bib_regions", ("torso", "back"))),
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            print(f"[FAIL] invalid event profile: {exc}")
            return 2

    detector = Detector(
        model_path=str(model_path),
        source_id=0,
        model=yolo,
        ocr=None,
        athlete_validator=athlete_validator,
        event_settle_seconds=LOCAL_VIDEO_EVENT_SETTLE_SECONDS,
        event_profile=event_profile,
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
    participant_tracks: Dict[str, set[int]] = {}
    regression_metrics = RegressionMetrics()
    evidence_dir = Path(args.evidence_dir) if args.evidence_dir else None
    first_video_timestamp = None
    last_video_timestamp = None

    t0 = time.time()
    while frames_done < frames_target:
        if use_ffmpeg:
            raw_frame = _read_exact(ffmpeg_process.stdout, width * height * 3)
            if len(raw_frame) != width * height * 3:
                break
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
            video_timestamp = (int(args.start_frame) + frames_done) / max(fps, 1e-6)
        else:
            ok, frame = cap.read()
            if not ok:
                break
            video_timestamp = video_timestamp_seconds(
                cap.get(cv2.CAP_PROP_POS_MSEC),
                frame_index=int(args.start_frame) + frames_done,
                fps=fps,
            )
        if first_video_timestamp is None:
            first_video_timestamp = float(video_timestamp)
        last_video_timestamp = float(video_timestamp)
        crossing_events, athletes, bibs = detector.process_frame(frame, timestamp=video_timestamp)
        frame_rejections = list(detector.last_rejected_candidates)
        frame_participant_ids = []
        frame_raw_track_ids = []
        for athlete in athletes:
            participant_id = str(athlete.get("participant_id") or "").strip()
            if participant_id:
                frame_participant_ids.append(participant_id)
                participant_tracks.setdefault(participant_id, set())
            raw_ids = athlete.get("raw_track_ids") or ([athlete.get("track_id")] if athlete.get("track_id") is not None else [])
            for raw_track_id in raw_ids:
                try:
                    raw_track_id = int(raw_track_id)
                except (TypeError, ValueError):
                    continue
                if raw_track_id < 0:
                    continue
                frame_raw_track_ids.append(raw_track_id)
                if participant_id:
                    participant_tracks[participant_id].add(raw_track_id)
        regression_metrics.observe_frame(
            detector.last_frame_metrics,
            crossing_events,
            rejected_candidates=frame_rejections,
            participant_ids=frame_participant_ids,
            raw_track_ids=frame_raw_track_ids,
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

    if use_ffmpeg and ffmpeg_process is not None:
        if ffmpeg_process.stdout is not None:
            ffmpeg_process.stdout.close()
        if ffmpeg_process.poll() is None:
            ffmpeg_process.terminate()
        try:
            ffmpeg_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            ffmpeg_process.kill()
            ffmpeg_process.wait(timeout=5)
    else:
        cap.release()

    elapsed = time.time() - t0
    proc_fps = (frames_done / elapsed) if elapsed > 1e-6 else 0.0
    observed_video_fps = 0.0
    if (
        frames_done > 1
        and first_video_timestamp is not None
        and last_video_timestamp is not None
        and last_video_timestamp > first_video_timestamp
    ):
        observed_video_fps = (frames_done - 1) / (last_video_timestamp - first_video_timestamp)
    metric_report = regression_metrics.to_report(processing_fps=proc_fps)

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
            "nominal_fps": nominal_fps,
            "observed_fps": float(round(observed_video_fps, 3)),
            "total_frames": total_frames,
            "decoded_frames": frames_done,
        },
        "performance": {
            "elapsed_sec": float(round(elapsed, 3)),
            "processing_fps": float(round(proc_fps, 2)),
        },
        "processing_fps": float(round(proc_fps, 2)),
        "raw_tracks": metric_report["raw_tracks"],
        "stable_participants": metric_report["stable_participants"],
        "track_fragments_merged": metric_report["track_fragments_merged"],
        "identity_ambiguities": metric_report["identity_ambiguities"],
        "crossing_events": metric_report["crossing_events"],
        "duplicate_passages": metric_report["duplicate_passages"],
        "rejected_candidates": metric_report["rejected_candidates"],
        "evidence_complete": metric_report["evidence_complete"],
        "ocr_recognized": metric_report["ocr_recognized"],
        "ocr_conflicts": metric_report["ocr_conflicts"],
        "ocr_unrecognized": metric_report["ocr_unrecognized"],
        "queue_depth_max": metric_report["queue_depth_max"],
        "dropped_frames": metric_report["dropped_frames"],
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
        "rejected_candidate_records": rejected_candidates,
        "participant_tracks": {
            participant_id: sorted(track_ids)
            for participant_id, track_ids in sorted(participant_tracks.items())
        },
        "event_profile": (
            {
                "name": event_profile.name,
                "pipeline": event_profile.pipeline,
                "required_equipment": event_profile.required_equipment,
                "crossing_mode": event_profile.crossing_mode,
                "bib_regions": list(event_profile.bib_regions),
            }
            if event_profile is not None
            else None
        ),
    }
    report["result_signature"] = result_signature(report)

    try:
        out_path = write_json_report(
            Path(args.out) if args.out else Path.cwd() / f"regression_report_{model_path.stem}.json",
            report,
        )
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
