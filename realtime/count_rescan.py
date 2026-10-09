"""Bounded original-frame evidence and a single asynchronous replay worker."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
import threading
import math
from pathlib import Path
import time
import uuid

import cv2

from .count_storage import JsonlTail
from .independent_counting import FrameStamp

from .count_observations import overlap


@dataclass(frozen=True)
class ReplayResult:
    events: tuple = ()
    quality: tuple = ()
    evidence: dict = field(default_factory=dict)
    error: str = ""


class EvidenceUnavailableError(RuntimeError):
    """Original recording cannot cover the entire requested short window."""


@dataclass(frozen=True)
class RecordingEvidenceRequest:
    provider: object
    start_ms: int
    end_ms: int

    def load(self, max_frames, max_bytes):
        return self.provider.load(self.start_ms, self.end_ms, max_frames, max_bytes)


class RecordingEvidenceProvider:
    """Replay original files using declared media time, without loading models."""

    def __init__(self, *, source_file=None, recording_start_ms=None, timeline=None,
                 camera_index=1, race_id="", timing_error_ms=0):
        self.source_file = Path(source_file) if source_file else None
        self.recording_start_ms = recording_start_ms
        self.timeline = JsonlTail(timeline) if timeline else None
        self.camera_index, self.race_id = camera_index, race_id
        self.timing_error_ms = timing_error_ms
        self._segments = {}

    def request(self, start_ms, end_ms):
        return RecordingEvidenceRequest(self, start_ms, end_ms)

    def load(self, start_ms, end_ms, max_frames, max_bytes):
        if self.source_file:
            segments = [(self.recording_start_ms, self.source_file, None)]
        else:
            for record in self.timeline.read():
                uid = record.get("segment_id")
                if record.get("record_type") == "segment_started":
                    self._segments[uid] = dict(record)
                elif record.get("record_type") == "segment_ended" and uid in self._segments:
                    self._segments[uid].update(record)
            segments = []
            for row in self._segments.values():
                if row.get("camera_index") != self.camera_index or row.get("race_id", "") != self.race_id:
                    continue
                # An uncalibrated recording locator cannot repair timing evidence.
                if row.get("media_started_at_ms") is None or row.get("timing_error_ms", 2000) > self.timing_error_ms:
                    continue
                path = Path(row["video_path"])
                if not path.is_absolute():
                    path = self.timeline.path.parent / path
                segments.append((row["media_started_at_ms"], path, row.get("media_duration_ms")))
        collected, total_bytes = [], 0
        generation = "recording-" + uuid.uuid4().hex
        for media_start, path, duration in sorted(segments, key=lambda row: row[0]):
            if media_start is None or duration is not None and (media_start + duration < start_ms or media_start > end_ms):
                continue
            capture = cv2.VideoCapture(str(path))
            try:
                if not capture.isOpened():
                    continue
                fps = capture.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    continue
                frame_index = max(0, math.floor((start_ms - media_start) * fps / 1000) - 1)
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                previous_pts = None
                while len(collected) <= max_frames:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    raw_pts = capture.get(cv2.CAP_PROP_POS_MSEC)
                    if raw_pts < 0 or previous_pts is not None and raw_pts <= previous_pts:
                        raise EvidenceUnavailableError("recording has no monotonic source timestamps")
                    previous_pts = raw_pts
                    media_position_ms = round(raw_pts)
                    stamp_ms = media_start + media_position_ms
                    frame_index += 1
                    if collected and stamp_ms <= collected[-1][0].observed_at_ms:
                        continue
                    if stamp_ms < start_ms - 1000 / fps:
                        continue
                    if total_bytes + frame.nbytes > max_bytes:
                        raise EvidenceUnavailableError("rescan memory budget cannot cover the requested original frames")
                    collected.append((FrameStamp(len(collected) + 1, generation, stamp_ms,
                                                 int(time.time() * 1000), "recording_metadata", media_position_ms), frame))
                    total_bytes += frame.nbytes
                    if stamp_ms >= end_ms:
                        break
            finally:
                capture.release()
            if collected and collected[-1][0].observed_at_ms >= end_ms:
                break
        if not collected or collected[0][0].observed_at_ms > start_ms or collected[-1][0].observed_at_ms < end_ms or len(collected) > max_frames:
            raise EvidenceUnavailableError("original recording does not cover this short window")
        if any(b.observed_at_ms - a.observed_at_ms > 200 for (a, _), (b, _) in zip(collected, collected[1:])):
            raise EvidenceUnavailableError("recording window contains a gap")
        return tuple(collected)


class FrameEvidenceBuffer:
    def __init__(self, max_bytes=256 * 1024 * 1024, retention_ms=15000):
        if max_bytes <= 0 or retention_ms <= 0:
            raise ValueError("evidence budgets must be positive")
        self.max_bytes, self.retention_ms = max_bytes, retention_ms
        self.frames = deque()
        self.bytes = 0
        self.evicted_frames = 0
        self._lock = threading.Lock()

    def append(self, frame, stamp):
        with self._lock:
            copy = frame.copy()
            self.frames.append((stamp, copy))
            self.bytes += copy.nbytes
            while self.frames and (self.bytes > self.max_bytes or stamp.observed_at_ms - self.frames[0][0].observed_at_ms > self.retention_ms):
                _, dropped = self.frames.popleft()
                self.bytes -= dropped.nbytes
                self.evicted_frames += 1

    def window(self, start_ms, end_ms):
        with self._lock:
            if not self.frames or self.frames[0][0].observed_at_ms > start_ms or self.frames[-1][0].observed_at_ms < end_ms:
                return None
            all_frames = list(self.frames)
            indices = [index for index, (stamp, _) in enumerate(all_frames) if start_ms <= stamp.observed_at_ms <= end_ms]
            if not indices:
                return None
            frames = all_frames[max(0, indices[0] - 1):min(len(all_frames), indices[-1] + 2)]
        if not frames or len({stamp.generation for stamp, _ in frames}) != 1:
            return None
        if any(b.sequence != a.sequence + 1 for (a, _), (b, _) in zip(frames, frames[1:])):
            return None
        # Immutable copies are owned by the buffer and never modified by replay.
        return tuple(frames)

    def stats(self):
        with self._lock:
            return {"bytes": self.bytes, "max_bytes": self.max_bytes, "frames": len(self.frames),
                    "evicted_frames": self.evicted_frames}


class RescanQueue:
    def __init__(self, replay, *, max_pending=8, max_frames=900, max_bytes=256 * 1024 * 1024):
        self.replay = replay
        self.max_pending, self.max_frames = max_pending, max_frames
        self.max_bytes = max_bytes
        self._running_bytes = 0
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="count-replay")
        self.pending = deque()
        self.running = None
        self.completed = {}

    def submit(self, window, frames):
        uid = window["window_id"]
        if uid in self.completed or any(task[0]["window_id"] == uid for task in self.pending) or (self.running and self.running[0]["window_id"] == uid):
            return False
        deferred = isinstance(frames, RecordingEvidenceRequest)
        if frames is None or not deferred and len(frames) > self.max_frames:
            self.completed[uid] = {"status": "unresolved", "reason": "evidence_unavailable" if frames is None else "rescan_frame_budget"}
            return False
        if len(self.pending) >= self.max_pending:
            # Leave the task retryable when capacity becomes available.
            return False
        active = [task[0] for task in self.pending]
        if self.running:
            active.append(self.running[0])
        if any(window["evidence_start_ms"] <= task["evidence_end_ms"] and task["evidence_start_ms"] <= window["evidence_end_ms"] for task in active):
            return False
        pending_bytes = sum(self.max_bytes if isinstance(saved, RecordingEvidenceRequest) else sum(frame.nbytes for _, frame in saved) for _, saved in self.pending)
        requested_bytes = self.max_bytes if deferred else sum(frame.nbytes for _, frame in frames)
        if requested_bytes > self.max_bytes:
            self.completed[uid] = {"status": "unresolved", "reason": "rescan_memory_budget"}
            return False
        if requested_bytes + pending_bytes + self._running_bytes > self.max_bytes:
            return False
        self.pending.append((dict(window), frames))
        self._start_next()
        return True

    def _start_next(self):
        if self.running is None and self.pending:
            window, frames = self.pending.popleft()
            self._running_bytes = self.max_bytes if isinstance(frames, RecordingEvidenceRequest) else sum(frame.nbytes for _, frame in frames)
            self.running = (window, self.executor.submit(self._execute, window, frames))

    def _execute(self, window, frames):
        if isinstance(frames, RecordingEvidenceRequest):
            frames = frames.load(self.max_frames, self.max_bytes)
        return self.replay(window, frames)

    def poll(self):
        results = []
        if self.running and self.running[1].done():
            window, future = self.running
            try:
                result = future.result()
                self.completed[window["window_id"]] = {"status": "replayed"}
                results.append((window, result, None))
            except Exception as error:
                self.completed[window["window_id"]] = {"status": "unresolved", "reason": type(error).__name__}
                results.append((window, None, type(error).__name__))
            self.running = None
            self._running_bytes = 0
        self._start_next()
        return results

    def close(self):
        self.pending.clear()
        self.executor.shutdown(wait=True, cancel_futures=True)


def replay_revisions(originals, replayed, *, tolerance_ms=250):
    """Retain original UIDs. Missing replay detections cannot prove exclusion."""
    revisions, used, ambiguous = [], set(), False
    for fresh in replayed:
        matches = []
        for index, old in enumerate(originals):
            if index in used or old["direction"] != fresh.direction:
                continue
            intersects = fresh.cross_time_start_ms <= old["cross_time_end_ms"] + tolerance_ms and old["cross_time_start_ms"] <= fresh.cross_time_end_ms + tolerance_ms
            if intersects and overlap(old["after"]["bbox"], fresh.after["bbox"]) >= 0.3:
                matches.append((overlap(old["after"]["bbox"], fresh.after["bbox"]), index, old))
        matches.sort(reverse=True, key=lambda row: row[0])
        if len(matches) > 1 and matches[0][0] - matches[1][0] < 0.1:
            ambiguous = True
            continue
        if matches:
            _, index, old = matches[0]
            used.add(index)
            # Retain trusted identity decisions across anonymous replay.
            eligibility = old["eligibility"] if old.get("identity_reliable") else fresh.eligibility
            revised = replace(fresh, event_uid=old["event_uid"], revision=old["revision"] + 1,
                              identity=old.get("identity", ""), identity_reliable=old.get("identity_reliable", False),
                              group_id=old.get("group_id", ""), eligibility=eligibility,
                              duplicate_of=old.get("duplicate_of", "") if old.get("identity_reliable") else fresh.duplicate_of,
                              reasons=tuple(old.get("reasons", ())) if old.get("identity_reliable") else fresh.reasons)
            revisions.append(revised.payload())
        elif not any(abs(old["estimated_cross_time_ms"] - fresh.estimated_cross_time_ms) <= tolerance_ms
                     and overlap(old["after"]["bbox"], fresh.after["bbox"]) > 0.1 for old in originals):
            revisions.append(fresh.payload())
        else:
            ambiguous = True
    recovered_all = len(used) == len(originals) and not ambiguous
    return revisions, recovered_all
