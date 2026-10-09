"""Live independent count -> revision reconciliation -> bounded replay loop."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import time
import uuid

import cv2

from .count_reconciliation import ClockMapping, GroupReconciler, ReconciliationScope
from .count_rescan import FrameEvidenceBuffer, ReplayResult, RescanQueue, replay_revisions
from .count_storage import CountJournal, JsonlTail, atomic_report
from .independent_counting import CountRules


class CountRuntime:
    def __init__(self, rules: CountRules, scope: ReconciliationScope, clock: ClockMapping,
                 output_dir, chip_journal, replay, *, buffer_bytes=256 * 1024 * 1024,
                 retention_ms=15000, recording_provider=None, pipeline_fingerprint=""):
        self.rules, self.scope, self.output_dir = rules, scope, Path(output_dir)
        if scope.group_ids:
            raise ValueError("first release compares the whole visible scope; anonymous riders cannot be assigned to a selected group")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock_handle = (self.output_dir / "count_service.lock").open("a+b")
        self._lock_output()
        contract_path = self.output_dir / "count_pipeline.json"
        contract = json.loads(json.dumps({"rules": asdict(rules), "scope": asdict(scope), "pipeline_fingerprint": pipeline_fingerprint}))
        if contract_path.exists() and json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            self._lock_handle.close()
            raise ValueError("output directory belongs to different rules, session or model; use a separate output directory")
        atomic_report(contract_path, contract)
        self.reconciler = GroupReconciler(scope, clock)
        self.events = CountJournal(self.output_dir / "camera_passages.jsonl")
        self.diagnostics = CountJournal(self.output_dir / "count_observations.jsonl")
        self.lifecycle = CountJournal(self.output_dir / "count_lifecycle.jsonl")
        self._observation_events = {}
        saved = JsonlTail(self.events.path)
        while records := saved.read():
            for event in records:
                self.reconciler.ingest("video", event)
                self._observation_events[self._observation_key(event)] = event["event_uid"]
        saved = JsonlTail(self.lifecycle.path)
        while records := saved.read():
            for record in records:
                if record.get("session_id") != rules.session_id:
                    continue
                if record["record_type"] == "quality":
                    self.reconciler.add_quality(record["notice"])
                elif record["record_type"] == "replayed_quality":
                    self.reconciler.clear_replayed_quality(record["start_ms"], record["end_ms"], record["reasons"])
                elif record["record_type"] == "progress":
                    self.reconciler.progress(record["video_ms"], start_ms=record["start_ms"])
        self.chips = JsonlTail(chip_journal)
        self.camera_revisions = JsonlTail(self.output_dir / "camera_passage_revisions.jsonl")
        self.buffer = FrameEvidenceBuffer(buffer_bytes, retention_ms)
        self.recording_provider = recording_provider
        def replay_with_evidence(window, frames):
            evidence = self._save_evidence(window, frames)
            try:
                result = replay(window, frames)
                if isinstance(result, ReplayResult):
                    return replace(result, evidence=evidence)
                return ReplayResult(tuple(result), evidence=evidence)
            except Exception as error:
                return ReplayResult(evidence=evidence, error=type(error).__name__)
        self.rescans = RescanQueue(replay_with_evidence, max_bytes=buffer_bytes)
        self._start_ms = self.reconciler.video_start_ms
        self._last_publish = 0
        self.runtime_quality = []
        self._evidence = {}
        self._rescan_history = []
        self._closing = False
        report_path = self.output_dir / "count_reconciliation.json"
        if report_path.exists():
            old = json.loads(report_path.read_text(encoding="utf-8"))
            if old.get("scope") == asdict(scope) or old.get("scope") == json.loads(json.dumps(asdict(scope))):
                self.rescans.completed.update(old.get("rescan_completed", {}))
                self._evidence.update(old.get("evidence", {}))

    def _lock_output(self):
        if os.name == "nt":
            import msvcrt
            self._lock_handle.seek(0)
            self._lock_handle.write(b"0")
            self._lock_handle.flush()
            self._lock_handle.seek(0)
            try:
                msvcrt.locking(self._lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                self._lock_handle.close()
                raise RuntimeError("another count service owns this output directory") from error
        else:
            import fcntl
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def on_event(self, event):
        payload = event.payload()
        key = (payload["race_id"], payload["stage_id"], payload["event_uid"])
        if key in self.reconciler.latest["video"] or self._observation_key(payload) in self._observation_events:
            return
        self.events.append([payload])
        self.reconciler.ingest("video", payload)
        self._observation_events[self._observation_key(payload)] = payload["event_uid"]

    @staticmethod
    def _observation_key(event):
        return (event["session_id"], event["source_id"], event["before"]["detection_ref"], event["after"]["detection_ref"])

    def on_diagnostics(self, stamp, raw, linked, decisions, quality):
        self.diagnostics.append([{"frame": asdict(stamp), "raw_detections": [asdict(row) for row in raw],
                                  "linked_observations": [asdict(row) for row in linked], "decisions": decisions}])
        for notice in quality:
            self.add_quality(notice)
        if self._start_ms is None:
            self._start_ms = stamp.observed_at_ms
        self.reconciler.progress(stamp.observed_at_ms, start_ms=self._start_ms)

    def add_quality(self, notice):
        self.lifecycle.append([{"record_type": "quality", "session_id": self.rules.session_id, "notice": notice}])
        self.reconciler.add_quality(notice)

    def tick(self, *, force=False):
        for record in self.chips.read():
            if record.get("record_type") == "watermark":
                # Only explicit event-time completeness, never receipt/heartbeat.
                if all(record.get(name) == getattr(self.scope, name) for name in ("race_id", "stage_id", "timing_position")):
                    self.reconciler.chip_watermark_ms = record["completed_until_ms"]
            elif record.get("message_type", "passage") == "passage":
                self.reconciler.ingest("chip", record)
        for update in self.camera_revisions.read():
            self.apply_identity_revision(update)
        for window, replayed, failure in self.rescans.poll():
            if replayed is not None:
                self._evidence[window["window_id"]] = replayed.evidence
                failure = failure or replayed.error
            if failure:
                self.rescans.completed[window["window_id"]] = {"status": "unresolved", "reason": failure}
                self._rescan_history.append({"window_id": window["window_id"], "result": failure})
                continue
            originals = [event for event in self.reconciler.latest["video"].values()
                         if event.get("session_id") == self.rules.session_id and event.get("source_id") == self.rules.source_id
                         and window["camera_group_start_ms"] <= event["estimated_cross_time_ms"] <= window["camera_group_end_ms"]
                         and event.get("is_active", True)]
            revisions, recovered = replay_revisions(originals, replayed.events)
            self.events.append(revisions)
            for event in revisions:
                self.reconciler.ingest("video", event)
                self._observation_events[self._observation_key(event)] = event["event_uid"]
            if recovered:
                a, b = window["camera_evidence_start_ms"], window["camera_evidence_end_ms"]
                replay_quality = {flag for row in replayed.events for flag in row.quality}
                replay_quality.update(notice["reason"] for notice in replayed.quality)
                cleared = [reason for reason in window["reasons"] if reason not in replay_quality and reason in {
                    "sampling_gap", "untracked_detection", "first_observed_after_line",
                    "broken_track", "observation_insufficient", "association_ambiguous",
                    "overlapping_detections", "suspected_merged_box"}]
                self.lifecycle.append([{"record_type": "replayed_quality", "session_id": self.rules.session_id,
                                       "start_ms": a, "end_ms": b, "reasons": cleared}])
                self.reconciler.clear_replayed_quality(a, b, cleared)
            for notice in replayed.quality:
                self.add_quality(notice)
            # Mark the post-revision windows examined, preventing rescan loops.
            for current in self.reconciler.groups():
                if current["start_ms"] <= window["end_ms"] and current["end_ms"] >= window["start_ms"]:
                    state = "unresolved" if current["reasons"] else "resolved"
                    self.rescans.completed[current["window_id"]] = {"status": state, "reason": "remaining_evidence" if state == "unresolved" else ""}
                    if window["window_id"] in self._evidence:
                        self._evidence[current["window_id"]] = self._evidence[window["window_id"]]
            self._rescan_history.append({"window_id": window["window_id"], "revisions": len(revisions), "recovered_all": recovered})
        now = time.monotonic()
        if not force and now - self._last_publish < 0.5:
            return
        self._last_publish = now
        clock = self.reconciler.clock

        def unmap(value):
            return round((value - clock.offset_ms + clock.anchor_ms * clock.drift_ppm / 1_000_000) / (1 + clock.drift_ppm / 1_000_000))

        for window in self.reconciler.groups():
            if self._closing or window["status"] != "anomaly" or window["window_id"] in self.rescans.completed:
                continue
            a, b = unmap(window["evidence_start_ms"]), unmap(window["evidence_end_ms"])
            # The post-group context must be received before requesting evidence.
            frames = self.buffer.window(a, b)
            if frames is None and self.recording_provider is not None:
                frames = self.recording_provider.request(a, b)
            task = {**window, "camera_evidence_start_ms": a, "camera_evidence_end_ms": b,
                    "camera_group_start_ms": unmap(window["start_ms"]) - clock.error_ms,
                    "camera_group_end_ms": unmap(window["end_ms"]) + clock.error_ms}
            self.rescans.submit(task, frames)
        self.publish()

    def apply_identity_revision(self, update):
        """Accept explicit trusted identity decisions; absence never excludes."""
        key = (self.rules.race_id, self.rules.stage_id, update["event_uid"])
        old = self.reconciler.latest["video"].get(key)
        if old is None or old.get("session_id") != self.rules.session_id or old.get("source_id") != self.rules.source_id:
            raise ValueError("camera identity revision does not refer to this session")
        allowed = {"event_uid", "revision", "identity", "identity_reliable", "eligibility", "reasons", "duplicate_of", "group_id", "is_active"}
        if set(update) - allowed:
            raise ValueError("identity revisions cannot alter raw crossing observations or scope")
        revised = {**old, **update}
        if not isinstance(revised["revision"], int) or isinstance(revised["revision"], bool) or revised["revision"] < 1:
            raise ValueError("revision must be a positive integer")
        if not isinstance(revised["reasons"], (list, tuple)) or any(not isinstance(reason, str) for reason in revised["reasons"]):
            raise ValueError("reasons must be a list of explicit reason strings")
        if not isinstance(revised["identity"], str) or not isinstance(revised["identity_reliable"], bool) or not isinstance(revised["is_active"], bool):
            raise ValueError("invalid trusted identity revision types")
        if revised["eligibility"] not in {"admitted", "excluded", "pending"}:
            raise ValueError("invalid revised eligibility")
        if (revised["eligibility"] == "excluded" or not revised["is_active"]) and (not revised["identity_reliable"] or not revised["identity"] or not revised["reasons"]):
            raise ValueError("identity exclusion requires trusted identity and an explicit reason")
        if revised["eligibility"] == "excluded" and any(reason in {"no_chip", "unreadable_bib", "slow_motion", "unknown_identity"} for reason in revised["reasons"]):
            raise ValueError("missing chip, unreadable bib and slow motion cannot exclude a crossing")
        if revised["revision"] <= old["revision"]:
            self.reconciler.ingest("video", revised)
            return
        # Persist the accepted decision before updating live contributions.
        self.events.append([revised])
        self.reconciler.ingest("video", revised)

    def _save_evidence(self, window, frames):
        directory = self.output_dir / "count_evidence" / window["window_id"]
        directory.mkdir(parents=True, exist_ok=True)
        keys = []
        for index in sorted({0, len(frames) // 2, len(frames) - 1}):
            path = directory / f"frame_{frames[index][0].sequence}.jpg"
            ok, data = cv2.imencode(".jpg", frames[index][1], [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise RuntimeError("cannot encode evidence keyframe")
            data.tofile(str(path))
            keys.append(str(path.absolute()))
        clip = directory / "clip.mp4"
        span = frames[-1][0].observed_at_ms - frames[0][0].observed_at_ms
        fps = min(120, max(1, (len(frames) - 1) * 1000 / max(1, span)))
        height, width = frames[0][1].shape[:2]
        writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        clip_path = ""
        try:
            if writer.isOpened():
                for _, frame in frames:
                    writer.write(frame)
                clip_path = str(clip.absolute())
        finally:
            writer.release()
        atomic_report(directory / "frames.json", [asdict(stamp) for stamp, _ in frames])
        return {"keyframes": keys, "clip": clip_path, "frame_metadata": str((directory / "frames.json").absolute())}

    def publish(self):
        report = self.reconciler.report()
        for window in report["groups"]:
            attempt = self.rescans.completed.get(window["window_id"])
            if window["status"] == "anomaly":
                window["status"] = "unresolved" if attempt else "rescanning"
                window["rescan"] = attempt or {"status": "queued"}
            window["evidence"] = self._evidence.get(window["window_id"], {})
        report.update({"updated_at_ms": int(time.time() * 1000), "buffer": self.buffer.stats(),
                       "rescan_pending": len(self.rescans.pending), "rescan_running": bool(self.rescans.running),
                       "rescan_completed": self.rescans.completed, "rescan_history": self._rescan_history[-50:],
                       "runtime_quality": self.runtime_quality, "evidence": self._evidence})
        atomic_report(self.output_dir / "count_reconciliation.json", report)
        if self.reconciler.video_progress_ms is not None:
            self.lifecycle.append([{"record_type": "progress", "session_id": self.rules.session_id,
                                    "video_ms": self.reconciler.video_progress_ms, "start_ms": self._start_ms}])

    def close(self):
        self._closing = True
        self.rescans.close()
        self.tick(force=True)
        self.events.close()
        self.diagnostics.close()
        self.lifecycle.close()
        self._lock_handle.close()


def make_replay(model_path, rules, *, imgsz=1280, confidence=0.06, timeout_seconds=20):
    """Lazy, separate model/tracker: replay cannot mutate the live tracker."""
    detector = None

    def replay(window, frames):
        nonlocal detector
        from .detector import Detector
        if detector is None:
            from ultralytics import YOLO
            detector = Detector(model_path, model=YOLO(model_path), ocr=None)
        if getattr(detector._model, "predictor", None) is not None:
            for tracker in getattr(detector._model.predictor, "trackers", ()):
                tracker.reset()
        collected = []
        quality = []
        detector.configure_counting(rules, collected.append,
                                    lambda stamp, raw, linked, decisions, notices: quality.extend(notices), count_only=True)
        detector.process_imgsz = imgsz
        detector.detection_conf = confidence
        # Replay original pixels at higher input resolution without frame skips.
        detector.enable_roi_filter = False
        detector.adaptive_frame_skip = False
        generation = "replay-" + uuid.uuid4().hex
        started = time.monotonic()
        for stamp, frame in frames:
            if time.monotonic() - started > timeout_seconds:
                raise TimeoutError("rescan exceeded its compute budget")
            detector.process_frame(frame, timestamp=(stamp.observed_at_ms - frames[0][0].observed_at_ms) / 1000,
                                   frame_stamp=replace(stamp, generation=generation))
        # Caller supplies chip-domain boundaries; the selection uses original
        # camera bounds to preserve the configured mapping, including drift.
        camera_start = window.get("camera_group_start_ms", frames[0][0].observed_at_ms)
        camera_end = window.get("camera_group_end_ms", frames[-1][0].observed_at_ms)
        relevant_quality = tuple(notice for notice in quality if camera_start <= notice["end_ms"] and notice["start_ms"] <= camera_end)
        return ReplayResult(tuple(event for event in collected if camera_start <= event.estimated_cross_time_ms <= camera_end), relevant_quality)

    return replay
