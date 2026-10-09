"""Observed crossings, independent of OCR, chip counts and legacy heuristics."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class FrameStamp:
    sequence: int
    generation: str
    observed_at_ms: int
    received_at_ms: int
    time_source: str = "receiver_clock"
    media_position_ms: int | None = None


@dataclass(frozen=True)
class CountRules:
    session_id: str
    race_id: str
    stage_id: str
    timing_position: str
    source_id: str
    line: tuple[tuple[float, float], tuple[float, float]]
    direction: str
    allowed_classes: tuple[int, ...]
    roi: tuple[tuple[float, float], ...] = ()
    version: str = "1"
    deadband_px: float = 3.0
    side_observations: int = 2
    max_gap_ms: int = 400
    max_frame_gap: int = 5
    track_timeout_ms: int = 2000
    duplicate_identity_ms: int = 500
    anchor: str = "bottom_center"

    def __post_init__(self):
        if not all((self.session_id, self.race_id, self.stage_id,
                    self.timing_position, self.source_id, self.allowed_classes)):
            raise ValueError("count scope and allowed_classes must be explicit")
        if self.direction not in {"pos_to_neg", "neg_to_pos"}:
            raise ValueError("direction must be pos_to_neg or neg_to_pos")
        if math.dist(*self.line) == 0:
            raise ValueError("count line must have positive length")
        if self.side_observations < 1 or self.max_gap_ms < 1 or self.max_frame_gap < 1:
            raise ValueError("observation limits must be positive")
        if self.anchor not in {"bottom_center", "center"}:
            raise ValueError("unsupported crossing anchor")
        if self.deadband_px < 0 or self.track_timeout_ms < self.max_gap_ms or self.duplicate_identity_ms < 0:
            raise ValueError("invalid hysteresis or track limits")


@dataclass(frozen=True)
class Observation:
    track_id: int | None
    bbox: tuple[float, float, float, float]
    class_id: int
    confidence: float
    detection_ref: str
    observed: bool = True
    quality: tuple[str, ...] = ()
    identity: str = ""
    identity_reliable: bool = False
    excluded_identity_reason: str = ""


@dataclass(frozen=True)
class CountEvent:
    event_uid: str
    revision: int
    is_active: bool
    session_id: str
    race_id: str
    stage_id: str
    timing_position: str
    source_id: str
    rule_version: str
    track_id: int
    generation: str
    direction: str
    eligibility: str
    reasons: tuple[str, ...]
    quality: tuple[str, ...]
    cross_time_start_ms: int
    cross_time_end_ms: int
    estimated_cross_time_ms: int
    before: dict
    after: dict
    processed_at_ms: int
    published_at_ms: int
    time_source: str
    identity: str = ""
    identity_reliable: bool = False
    duplicate_of: str = ""
    group_id: str = ""

    def payload(self):
        return asdict(self)


def inside_polygon(point, polygon):
    if not polygon:
        return True
    x, y = point
    inside = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        cross = (x - a[0]) * (b[1] - a[1]) - (y - a[1]) * (b[0] - a[0])
        if abs(cross) < 1e-6 and min(a[0], b[0]) <= x <= max(a[0], b[0]) and min(a[1], b[1]) <= y <= max(a[1], b[1]):
            return True
        if (a[1] > y) != (b[1] > y) and x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]:
            inside = not inside
    return inside


@dataclass
class _Track:
    stable_side: int = 0
    candidate_side: int = 0
    candidate_count: int = 0
    before: dict | None = None
    first_opposite: dict | None = None
    last_ms: int = 0
    passages: int = 0
    quality: set[str] = field(default_factory=set)


class IndependentCounter:
    def __init__(self, rules: CountRules):
        self.rules = rules
        self._tracks = {}
        self._identities = {}
        self._generation = None
        self._last_stamp = None
        self.last_diagnostics = ()
        self.last_quality = ()
        self.processed_until_ms = None

    def geometry(self, bbox):
        x1, y1, x2, y2 = bbox
        point = ((x1 + x2) / 2, y2 if self.rules.anchor == "bottom_center" else (y1 + y2) / 2)
        a, b = self.rules.line
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        distance = (dx * (point[1] - a[1]) - dy * (point[0] - a[0])) / length
        projection = ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length**2
        return point, distance, projection

    def process(self, stamp: FrameStamp, observations, *, raw_detections=()):
        diagnostics, quality, events = [], [], []

        def notice(reason, start=None, end=None):
            quality.append({"reason": reason, "start_ms": stamp.observed_at_ms if start is None else start,
                            "end_ms": stamp.observed_at_ms if end is None else end,
                            "source_id": self.rules.source_id})

        if self._generation != stamp.generation:
            if self._generation is not None:
                notice("stream_reconnected", self._last_stamp.observed_at_ms, stamp.observed_at_ms)
            self._tracks.clear()
            self._generation = stamp.generation
            self._last_stamp = None
        if self._last_stamp:
            if stamp.sequence <= self._last_stamp.sequence:
                return ()
            if stamp.observed_at_ms <= self._last_stamp.observed_at_ms:
                notice("clock_regressed", stamp.observed_at_ms, self._last_stamp.observed_at_ms)
                self._tracks.clear()
            if stamp.observed_at_ms - self._last_stamp.observed_at_ms > self.rules.max_gap_ms or stamp.sequence - self._last_stamp.sequence > self.rules.max_frame_gap:
                notice("sampling_gap", self._last_stamp.observed_at_ms, stamp.observed_at_ms)
        for key, track in tuple(self._tracks.items()):
            if stamp.observed_at_ms - track.last_ms > self.rules.track_timeout_ms:
                if track.before and (track.first_opposite or abs(track.before["signed_distance"]) < 80):
                    notice("broken_track", track.before["observed_at_ms"], track.last_ms)
                del self._tracks[key]
        linked_refs = set()
        for obs in observations:
            point, distance, projection = self.geometry(obs.bbox)
            reason = ""
            if obs.class_id not in self.rules.allowed_classes:
                reason = "class_not_allowed"
            elif not obs.observed:
                reason = "prediction_only"
            elif obs.track_id is None or obs.track_id < 0:
                reason = "untracked_detection"
            linked_refs.add(obs.detection_ref)
            diag = {"detection_ref": obs.detection_ref, "track_id": obs.track_id,
                    "bbox": obs.bbox, "class_id": obs.class_id, "confidence": obs.confidence,
                    "anchor": point, "signed_distance": distance, "reason": reason,
                    "frame": asdict(stamp)}
            diagnostics.append(diag)
            if reason:
                if reason != "class_not_allowed" and abs(distance) < 80:
                    notice(reason)
                continue
            track = self._tracks.setdefault(obs.track_id, _Track())
            track.last_ms = stamp.observed_at_ms
            track.quality.update(obs.quality)
            if track.before:
                previous_bbox = track.before["bbox"]
                old_area = (previous_bbox[2] - previous_bbox[0]) * (previous_bbox[3] - previous_bbox[1])
                new_area = (obs.bbox[2] - obs.bbox[0]) * (obs.bbox[3] - obs.bbox[1])
                if old_area > 0 and new_area > old_area * 1.8:
                    track.quality.add("suspected_merged_box")
                    notice("suspected_merged_box")
            for flag in obs.quality:
                notice(flag)
            side = 1 if distance > self.rules.deadband_px else -1 if distance < -self.rules.deadband_px else 0
            current = {**asdict(stamp), "anchor": point, "signed_distance": distance,
                       "detection_ref": obs.detection_ref, "bbox": obs.bbox,
                       "projection": projection, "inside_lane": inside_polygon(point, self.rules.roi)}
            if side == 0:
                continue
            if side == track.stable_side:
                track.before = current
                track.first_opposite = None
                track.candidate_count = 0
                continue
            if track.candidate_side != side or track.candidate_count == 0:
                track.candidate_side = side
                track.candidate_count = 1
                track.first_opposite = current
            else:
                track.candidate_count += 1
            if track.candidate_count < self.rules.side_observations:
                continue
            if track.stable_side and track.before:
                before, after = track.before, track.first_opposite
                direction = "pos_to_neg" if track.stable_side > 0 else "neg_to_pos"
                flags = set(track.quality)
                reasons = []
                eligibility = "admitted"
                gap = after["observed_at_ms"] - before["observed_at_ms"]
                if gap > self.rules.max_gap_ms or after["sequence"] - before["sequence"] > self.rules.max_frame_gap:
                    flags.add("observation_insufficient")
                    eligibility = "pending"
                if direction != self.rules.direction:
                    eligibility, reasons = "excluded", ["reverse_direction"]
                fraction = abs(before["signed_distance"]) / (abs(before["signed_distance"]) + abs(after["signed_distance"]))
                crossing_point = tuple(a + fraction * (b - a) for a, b in zip(before["anchor"], after["anchor"]))
                a, b = self.rules.line
                projection_at_crossing = before["projection"] + fraction * (after["projection"] - before["projection"])
                if not 0 <= projection_at_crossing <= 1 or not inside_polygon(crossing_point, self.rules.roi):
                    eligibility, reasons = "excluded", ["outside_course"]
                if obs.identity_reliable and obs.excluded_identity_reason:
                    eligibility, reasons = "excluded", [obs.excluded_identity_reason]
                estimated = round(before["observed_at_ms"] + fraction * gap)
                duplicate_of = ""
                prior = self._identities.get(obs.identity) if obs.identity_reliable else None
                if prior and prior[3] != obs.track_id and 0 <= estimated - prior[0] <= self.rules.duplicate_identity_ms and direction == prior[2]:
                    duplicate_of = prior[1]
                    eligibility, reasons = "excluded", ["reliable_identity_duplicate"]
                track.passages += 1
                key = f"{self.rules.session_id}:{self.rules.source_id}:{stamp.generation}:{obs.track_id}:{track.passages}:{before['sequence']}"
                uid = hashlib.sha256(key.encode()).hexdigest()[:32]
                now = int(time.time() * 1000)
                event = CountEvent(uid, 1, True, self.rules.session_id, self.rules.race_id,
                                   self.rules.stage_id, self.rules.timing_position, self.rules.source_id,
                                   self.rules.version, obs.track_id, stamp.generation, direction,
                                   eligibility, tuple(reasons), tuple(sorted(flags)), before["observed_at_ms"],
                                   after["observed_at_ms"], estimated, before, after, now, now,
                                   stamp.time_source, obs.identity, obs.identity_reliable, duplicate_of)
                events.append(event)
                if obs.identity_reliable and eligibility == "admitted":
                    self._identities[obs.identity] = (estimated, uid, direction, obs.track_id)
            elif side == (-1 if self.rules.direction == "pos_to_neg" else 1):
                notice("first_observed_after_line")
            track.stable_side = side
            track.before = current
            track.first_opposite = None
            track.candidate_count = 0
            track.quality.clear()
        for raw in raw_detections:
            if raw.class_id in self.rules.allowed_classes and raw.detection_ref not in linked_refs:
                diagnostics.append({**asdict(raw), "reason": "untracked_detection", "frame": asdict(stamp)})
                if abs(self.geometry(raw.bbox)[1]) < 80:
                    notice("untracked_detection")
        cutoff = stamp.observed_at_ms - self.rules.duplicate_identity_ms
        self._identities = {key: value for key, value in self._identities.items() if value[0] >= cutoff}
        self._last_stamp = stamp
        self.processed_until_ms = stamp.observed_at_ms
        self.last_diagnostics, self.last_quality = tuple(diagnostics), tuple(quality)
        return tuple(events)
