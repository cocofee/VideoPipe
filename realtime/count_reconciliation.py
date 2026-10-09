"""Revision-aware, single-position group reconciliation in chip clock time."""

from __future__ import annotations

import bisect
import hashlib
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ClockMapping:
    offset_ms: int = 0
    anchor_ms: int = 0
    drift_ppm: float = 0.0
    error_ms: int = 0
    version: str = "1"
    calibrated: bool = False

    def map(self, value):
        return round(value + self.offset_ms + (value - self.anchor_ms) * self.drift_ppm / 1_000_000)


@dataclass(frozen=True)
class ReconciliationScope:
    session_id: str
    race_id: str
    stage_id: str
    timing_position: str
    source_id: str
    chip_source: str = "cyclerace"
    group_ids: tuple[str, ...] = ()
    group_gap_ms: int = 1000
    max_group_ms: int = 10000
    context_ms: int = 700
    late_budget_ms: int = 2000

    def __post_init__(self):
        if not all((self.session_id, self.race_id, self.stage_id, self.timing_position, self.source_id)):
            raise ValueError("reconciliation scope must be explicit")
        if self.group_gap_ms <= 0 or self.max_group_ms < self.group_gap_ms or self.late_budget_ms < 0 or self.context_ms < 0:
            raise ValueError("invalid group or lateness limits")


class GroupReconciler:
    """Replace old contributions before indexing a revision's new contribution."""

    def __init__(self, scope: ReconciliationScope, clock: ClockMapping):
        self.scope, self.clock = scope, clock
        self.latest = {"video": {}, "chip": {}}
        self._index = []
        self._contributions = {}
        self._quality = {}
        self.video_progress_ms = None
        self.video_start_ms = None
        self.chip_watermark_ms = None
        self.coverage_gaps = []
        self.version = 0

    def _key(self, stream, event):
        return (event.get("race_id", ""), event.get("stage_id", ""),
                event["event_uid"] if stream == "video" else event["event_id"])

    def ingest(self, stream, event):
        if stream not in self.latest:
            raise ValueError("stream must be video or chip")
        event = dict(event)
        key = self._key(stream, event)
        old = self.latest[stream].get(key)
        revision = event.get("revision", 1)
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise ValueError("revision must be a positive integer")
        if old and revision < old.get("revision", 1):
            return False
        if old and revision == old.get("revision", 1):
            if event != old:
                raise ValueError("conflicting event revision")
            return False
        contribution_key = (stream, key)
        prior = self._contributions.pop(contribution_key, None)
        if prior is not None:
            self._index.remove(prior)
        self.latest[stream][key] = event
        contribution = self._contribution(stream, key, event)
        if contribution is not None:
            bisect.insort(self._index, contribution)
            self._contributions[contribution_key] = contribution
        self.version += 1
        return True

    def _contribution(self, stream, key, event):
        scope = self.scope
        if event.get("race_id") != scope.race_id or event.get("stage_id") != scope.stage_id:
            return None
        if not event.get("is_active", True):
            return None
        if stream == "video":
            if event.get("session_id") != scope.session_id or event.get("source_id") != scope.source_id or event.get("timing_position") != scope.timing_position:
                return None
            # Unknown group membership cannot be inferred from a chip match.
            if scope.group_ids and event.get("group_id") and event.get("group_id") not in scope.group_ids:
                return None
            stamp = self.clock.map(event["estimated_cross_time_ms"])
        else:
            # Protocol v1 has no timing_position; this subscription binds its source.
            if event.get("source", "cyclerace") != scope.chip_source:
                return None
            if event.get("timing_position", scope.timing_position) != scope.timing_position:
                return None
            if scope.group_ids and event.get("group_id") not in scope.group_ids:
                return None
            stamp = event.get("passage_timestamp_ms")
            if stamp is None:
                return None
        return (int(stamp), stream, key)

    def update_clock(self, clock):
        if clock == self.clock:
            return
        self.clock = clock
        self._index.clear()
        self._contributions.clear()
        for stream, records in self.latest.items():
            for key, event in records.items():
                contribution = self._contribution(stream, key, event)
                if contribution is not None:
                    self._contributions[(stream, key)] = contribution
                    self._index.append(contribution)
        self._index.sort()
        self.version += 1

    def add_quality(self, notice):
        if notice.get("source_id") != self.scope.source_id:
            return
        if notice["end_ms"] - notice["start_ms"] > self.scope.max_group_ms:
            for start in range(notice["start_ms"], notice["end_ms"], self.scope.max_group_ms):
                self.add_quality({**notice, "start_ms": start, "end_ms": min(notice["end_ms"], start + self.scope.max_group_ms)})
            return
        key = (notice["reason"], notice["start_ms"], notice["end_ms"])
        if self._quality.get(key) == notice:
            return
        self._quality[key] = dict(notice)
        if notice["reason"] in {"sampling_gap", "stream_reconnected", "clock_regressed", "inference_failed"}:
            self.coverage_gaps.append((notice["start_ms"], notice["end_ms"]))
        self.version += 1

    def progress(self, video_ms, *, start_ms=None, chip_watermark_ms=None):
        self.video_progress_ms = video_ms if self.video_progress_ms is None else max(self.video_progress_ms, video_ms)
        if start_ms is not None:
            self.video_start_ms = start_ms if self.video_start_ms is None else min(self.video_start_ms, start_ms)
        if chip_watermark_ms is not None:
            self.chip_watermark_ms = chip_watermark_ms

    @property
    def cutoff_ms(self):
        if self.video_progress_ms is None:
            return None
        video = self.clock.map(self.video_progress_ms) - self.clock.error_ms
        return min(video, self.chip_watermark_ms) if self.chip_watermark_ms is not None else video - self.scope.late_budget_ms

    def clear_replayed_quality(self, start_ms, end_ms, reasons):
        reasons = set(reasons)
        self._quality = {key: value for key, value in self._quality.items()
                         if not (value["reason"] in reasons and start_ms <= value["start_ms"] and value["end_ms"] <= end_ms)}
        self.coverage_gaps = [(a, b) for a, b in self.coverage_gaps if not (start_ms <= a and b <= end_ms)]
        self.version += 1

    def groups(self):
        cutoff = self.cutoff_ms
        if cutoff is None or self.video_start_ms is None:
            return []
        start = self.clock.map(self.video_start_ms) - self.clock.error_ms
        items = [row for row in self._index if start <= row[0] <= self.clock.map(self.video_progress_ms)]
        for key, notice in self._quality.items():
            a, b = self.clock.map(notice["start_ms"]), self.clock.map(notice["end_ms"])
            if b >= start:
                items.append((max(start, a), "quality", key))
        items.sort(key=lambda row: (row[0], row[1], str(row[2])))
        batches = []
        for item in items:
            item_end = self.clock.map(self._quality[item[2]]["end_ms"]) if item[1] == "quality" else item[0]
            if not batches or item[0] - batches[-1][-1][0] > self.scope.group_gap_ms or item_end - batches[-1][0][0] > self.scope.max_group_ms:
                batches.append([])
            batches[-1].append(item)
        result = []
        for batch in batches:
            a, b = batch[0][0], batch[-1][0]
            video, chips, flags, members = [], [], set(), []
            for _, stream, key in batch:
                if stream == "quality":
                    flags.add(self._quality[key]["reason"])
                    b = max(b, self.clock.map(self._quality[key]["end_ms"]))
                else:
                    record = self.latest[stream][key]
                    (video if stream == "video" else chips).append(record)
                    members.append((stream, key, record.get("revision", 1)))
            if not self.clock.calibrated:
                flags.add("clock_uncertain")
            if self._missing_chip_time_count():
                flags.add("chip_timestamp_missing")
            if any(not row.get("group_id") for row in video) and self.scope.group_ids:
                flags.add("group_scope_uncertain")
            for row in video:
                flags.update(row.get("quality", ()))
            admitted = sum(row.get("eligibility") == "admitted" and (not self.scope.group_ids or row.get("group_id") in self.scope.group_ids) for row in video)
            pending = sum(row.get("eligibility") == "pending" or (bool(self.scope.group_ids) and not row.get("group_id") and row.get("eligibility") != "excluded") for row in video)
            excluded = sum(row.get("eligibility") == "excluded" for row in video)
            closed = b + self.scope.group_gap_ms + self.clock.error_ms <= cutoff
            if pending:
                flags.add("eligibility_pending")
            if admitted != len(chips):
                flags.add("count_difference")
            signature = hashlib.sha256(repr((self.scope, self.clock, a, b, members, sorted(flags))).encode()).hexdigest()[:24]
            # Excluded-only passages remain auditable without opening a review task.
            result.append({"window_id": signature, "start_ms": a, "end_ms": b,
                           "evidence_start_ms": max(start, a - self.scope.context_ms),
                           "evidence_end_ms": b + self.scope.context_ms,
                           "chip_count": len(chips), "video_count": admitted,
                           "pending_count": pending, "excluded_count": excluded,
                           "difference": admitted - len(chips), "reasons": sorted(flags),
                           "status": "waiting" if not closed else "anomaly" if flags else "matched",
                           "provisional": self.chip_watermark_ms is None,
                           "cutoff_ms": cutoff, "clock_version": self.clock.version,
                           "video_event_uids": [row["event_uid"] for row in video],
                           "chip_event_ids": [row["event_id"] for row in chips],
                           "unmatched_reliable_identities": self._missing_identities(video, chips)})
        return result

    @staticmethod
    def _missing_identities(video, chips):
        if any(not row.get("athlete_id") for row in chips):
            return []
        chip_ids = {row.get("athlete_id") for row in chips if row.get("athlete_id")}
        return [row["identity"] for row in video if row.get("eligibility") == "admitted"
                and row.get("identity_reliable") and row.get("identity") and row["identity"] not in chip_ids]

    def report(self):
        groups = self.groups()
        return {"schema_version": 1, "scope": asdict(self.scope), "clock": asdict(self.clock),
                "version": self.version, "cutoff_ms": self.cutoff_ms,
                "video_progress_ms": self.video_progress_ms, "chip_watermark_ms": self.chip_watermark_ms,
                "groups": groups,
                "chip_records_without_absolute_time": self._missing_chip_time_count()}

    def _missing_chip_time_count(self):
        scope = self.scope
        return sum(row.get("is_active", True) and row.get("passage_timestamp_ms") is None
                   and row.get("race_id") == scope.race_id and row.get("stage_id") == scope.stage_id
                   and row.get("source", "cyclerace") == scope.chip_source
                   and row.get("timing_position", scope.timing_position) == scope.timing_position
                   and (not scope.group_ids or row.get("group_id") in scope.group_ids)
                   for row in self.latest["chip"].values())
