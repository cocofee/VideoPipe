"""Standard-library scorer for cycling crossing events."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union


Event = Mapping[str, Any]
Match = Tuple[int, int]


def load_events(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load an event list from a JSON file.

    The accepted top-level forms are ``[{...}, ...]`` and
    ``{"events": [{...}, ...]}``.  A fresh list of dictionaries is returned
    so callers can safely inspect or annotate it.
    """

    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)

    if isinstance(payload, Mapping):
        payload = payload.get("events")
    if not isinstance(payload, list):
        raise ValueError("event JSON must be an array or an object with an 'events' array")

    events: List[Dict[str, Any]] = []
    for index, event in enumerate(payload):
        if not isinstance(event, Mapping):
            raise ValueError(f"event at index {index} must be a JSON object")
        events.append(dict(event))
    return events


def _athlete_id(event: Event) -> Optional[str]:
    """Return the normalized identity used for matching."""

    value = event.get("athlete_id")
    if value is not None and str(value).strip():
        return str(value).strip()
    return None


def _timestamp_ms(event: Event) -> float:
    value = event.get("timestamp_ms")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("each event requires numeric timestamp_ms")
    try:
        timestamp = float(value)
    except OverflowError as exc:
        raise ValueError("each event requires finite timestamp_ms") from exc
    if not math.isfinite(timestamp):
        raise ValueError("each event requires finite timestamp_ms")
    return timestamp


def _validate_events(events: Sequence[Event], label: str) -> None:
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise ValueError(f"{label} event at index {index} must be an object")
        if _athlete_id(event) is None:
            raise ValueError(
                f"{label} event at index {index} requires non-empty athlete_id"
            )
        try:
            _timestamp_ms(event)
        except ValueError as exc:
            raise ValueError(f"{label} event at index {index}: {exc}") from exc


def match_events(
    ground_truth: Sequence[Event],
    predictions: Sequence[Event],
    tolerance_ms: float,
) -> List[Match]:
    """Return one-to-one ``(ground_truth_index, prediction_index)`` matches.

    Events are matched in timestamp order independently for each athlete.  A
    monotonic two-pointer walk gives the maximum-cardinality matching for a
    fixed timestamp tolerance while keeping ties deterministic.
    """

    if tolerance_ms < 0:
        raise ValueError("tolerance_ms must be non-negative")

    _validate_events(ground_truth, "ground truth")
    _validate_events(predictions, "prediction")

    gt_by_athlete: Dict[str, List[Tuple[float, int]]] = {}
    pred_by_athlete: Dict[str, List[Tuple[float, int]]] = {}
    for index, event in enumerate(ground_truth):
        athlete = _athlete_id(event)
        if athlete is not None:
            gt_by_athlete.setdefault(athlete, []).append((_timestamp_ms(event), index))
    for index, event in enumerate(predictions):
        athlete = _athlete_id(event)
        if athlete is not None:
            pred_by_athlete.setdefault(athlete, []).append((_timestamp_ms(event), index))

    matches: List[Match] = []
    for athlete in sorted(set(gt_by_athlete) & set(pred_by_athlete)):
        gt_items = sorted(gt_by_athlete[athlete])
        pred_items = sorted(pred_by_athlete[athlete])
        gt_pos = pred_pos = 0
        while gt_pos < len(gt_items) and pred_pos < len(pred_items):
            gt_time, gt_index = gt_items[gt_pos]
            pred_time, pred_index = pred_items[pred_pos]
            delta = pred_time - gt_time
            if abs(delta) <= tolerance_ms:
                matches.append((gt_index, pred_index))
                gt_pos += 1
                pred_pos += 1
            elif pred_time < gt_time - tolerance_ms:
                pred_pos += 1
            else:
                gt_pos += 1

    return sorted(matches)


def _ocr_value(event: Event) -> Optional[str]:
    for key in ("ocr", "ocr_text", "bib", "bib_number"):
        value = event.get(key)
        if isinstance(value, Mapping):
            value = value.get("text")
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _evidence_complete(event: Event) -> bool:
    evidence = event.get("evidence")
    if not isinstance(evidence, Mapping):
        return False

    def is_path(value: Any) -> bool:
        return isinstance(value, str) and bool(value.strip())

    frames = evidence.get("frame_paths")
    if isinstance(frames, (list, tuple)):
        has_frame = bool(frames) and all(is_path(item) for item in frames)
    else:
        has_frame = is_path(evidence.get("frame_path"))
    return has_frame and is_path(evidence.get("clip_path"))


def score(
    ground_truth: Sequence[Event],
    predictions: Sequence[Event],
    tolerance_ms: float = 1000,
) -> Dict[str, float]:
    """Compute baseline event quality rates.

    OCR and evidence are evaluated only for one-to-one athlete matches, so an
    OCR error cannot turn an observed athlete into a missed event.
    """

    matches = match_events(ground_truth, predictions, tolerance_ms)
    matched_gt = {gt_index for gt_index, _ in matches}
    matched_pred = {pred_index for _, pred_index in matches}

    duplicate_count = 0
    false_count = 0
    for pred_index, prediction in enumerate(predictions):
        if pred_index in matched_pred:
            continue
        athlete = _athlete_id(prediction)
        is_duplicate = False
        if athlete is not None:
            pred_time = _timestamp_ms(prediction)
            is_duplicate = any(
                _athlete_id(annotation) == athlete
                and abs(_timestamp_ms(annotation) - pred_time) <= tolerance_ms
                for annotation in ground_truth
            )
        if is_duplicate:
            duplicate_count += 1
        else:
            false_count += 1

    labelled_pairs = []
    for gt_index, pred_index in matches:
        expected_ocr = _ocr_value(ground_truth[gt_index])
        if expected_ocr is not None:
            labelled_pairs.append((expected_ocr, predictions[pred_index]))

    confirmed_count = 0
    for expected_ocr, prediction in labelled_pairs:
        observed_ocr = _ocr_value(prediction)
        explicitly_rejected = prediction.get("ocr_confirmed") is False
        if (
            observed_ocr is not None
            and observed_ocr == expected_ocr
            and not explicitly_rejected
        ):
            confirmed_count += 1

    complete_count = sum(
        1 for _, pred_index in matches if _evidence_complete(predictions[pred_index])
    )
    prediction_count = len(predictions)
    gt_count = len(ground_truth)
    matched_count = len(matches)
    labelled_count = len(labelled_pairs)

    return {
        "recall": len(matched_gt) / gt_count if gt_count else 0.0,
        "false_event_rate": false_count / prediction_count if prediction_count else 0.0,
        "duplicate_rate": duplicate_count / prediction_count if prediction_count else 0.0,
        "ocr_confirmed_rate": confirmed_count / labelled_count if labelled_count else 0.0,
        "evidence_complete_rate": complete_count / matched_count if matched_count else 0.0,
    }


__all__ = ["load_events", "match_events", "score"]
