from dataclasses import replace
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from realtime.count_reconciliation import ClockMapping, ReconciliationScope
from realtime.count_rescan import EvidenceUnavailableError, FrameEvidenceBuffer, RecordingEvidenceProvider, RescanQueue, replay_revisions
from realtime.count_runtime import CountRuntime
from realtime.count_storage import CountJournal, JsonlTail
from realtime.detector import Detector
from realtime.independent_counting import CountRules, FrameStamp, IndependentCounter, Observation


def rules():
    return CountRules("s", "r", "st", "finish", "cam", ((50, 0), (50, 100)), "pos_to_neg", (0,), side_observations=1)


def crossings():
    counter = IndependentCounter(rules())
    counter.process(FrameStamp(1, "g", 1000, 9000), [Observation(1, (20, 10, 40, 30), 0, 0.9, "d1")])
    return counter.process(FrameStamp(2, "g", 1040, 10000), [Observation(1, (60, 10, 80, 30), 0, 0.9, "d2")])[0]


def test_tail_handles_partial_record_and_journal_recovers_torn_suffix(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"revision":1}\n{"revision":')
    tail = JsonlTail(path)
    assert tail.read() == [{"revision": 1}]
    assert tail.read() == []
    with path.open("ab") as handle:
        handle.write(b'2}\n')
    assert tail.read() == [{"revision": 2}]
    with path.open("ab") as handle:
        handle.write(b'{"torn":')
    journal = CountJournal(path)
    journal.append([{"revision": 3}])
    journal.close()
    assert JsonlTail(path).read() == [{"revision": 1}, {"revision": 2}, {"revision": 3}]
    assert path.with_name(path.name + ".torn-tail").exists()


def test_evidence_buffer_bounds_memory_and_reports_missing_frames():
    buffer = FrameEvidenceBuffer(max_bytes=500, retention_ms=10000)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    buffer.append(frame, FrameStamp(1, "g", 1000, 1000))
    buffer.append(frame, FrameStamp(2, "g", 1040, 1040))
    assert buffer.stats()["bytes"] <= 500
    assert buffer.stats()["evicted_frames"] == 1
    assert buffer.window(1000, 1040) is None
    buffer = FrameEvidenceBuffer(max_bytes=5000)
    for sequence in (1, 3, 4):
        buffer.append(frame, FrameStamp(sequence, "g", 1000 + sequence * 40, 9999))
    assert buffer.window(1040, 1160) is None


def test_replay_revises_original_uid_but_does_not_delete_unrecovered_rider():
    old = crossings().payload()
    fresh = replace(crossings(), event_uid="new", quality=())
    revised, complete = replay_revisions([old], [fresh])
    assert complete
    assert revised[0]["event_uid"] == old["event_uid"]
    assert revised[0]["revision"] == 2
    revised, complete = replay_revisions([old], [])
    assert not complete and revised == []


def test_queue_is_asynchronous_deduplicates_and_preserves_failure():
    barrier = threading.Event()
    def replay(window, frames):
        barrier.wait(1)
        raise TimeoutError()
    queue = RescanQueue(replay)
    window = {"window_id": "w", "evidence_start_ms": 1000, "evidence_end_ms": 2000}
    frames = [(FrameStamp(1, "g", 1000, 1000), np.zeros((10, 10, 3), dtype=np.uint8))]
    try:
        assert queue.submit(window, frames)
        assert not queue.submit(window, frames)
        assert not queue.submit({**window, "window_id": "other"}, frames)
        assert queue.poll() == []
        barrier.set()
        for _ in range(100):
            result = queue.poll()
            if result:
                break
            time.sleep(0.005)
        assert result[0][2] == "TimeoutError"
        assert queue.completed["w"]["status"] == "unresolved"
    finally:
        barrier.set()
        queue.close()


def test_runtime_replays_equal_count_quality_and_recovers_revision_on_restart(tmp_path):
    scope = ReconciliationScope("s", "r", "st", "finish", "cam", group_gap_ms=100, context_ms=100, late_budget_ms=100)
    chip_path = tmp_path / "chips.jsonl"
    chip_path.write_text(json.dumps({"event_id": "c", "race_id": "r", "stage_id": "st", "revision": 1,
                                     "passage_timestamp_ms": 1020, "is_active": True}) + "\n", encoding="utf-8")
    fresh = crossings()
    old = replace(fresh, quality=("observation_insufficient",))
    runtime = CountRuntime(rules(), scope, ClockMapping(calibrated=True), tmp_path / "out", chip_path, lambda window, frames: [fresh])
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    try:
        for sequence in range(1, 31):
            stamp = FrameStamp(sequence, "g", 900 + sequence * 20, 9999)
            runtime.buffer.append(frame, stamp)
        runtime.on_event(old)
        runtime.reconciler.progress(1500, start_ms=920)
        runtime._start_ms = 920
        runtime.tick(force=True)
        for _ in range(200):
            runtime.tick(force=True)
            if runtime._rescan_history:
                break
            time.sleep(0.005)
        report = json.loads((tmp_path / "out" / "count_reconciliation.json").read_text(encoding="utf-8"))
        group, = report["groups"]
        assert group["status"] == "matched"
        assert group["video_count"] == group["chip_count"] == 1
        assert group["evidence"]["keyframes"]
        assert runtime.reconciler.latest["video"][("r", "st", fresh.event_uid)]["revision"] == 2
    finally:
        runtime.close()
    recovered = CountRuntime(rules(), scope, ClockMapping(calibrated=True), tmp_path / "out", chip_path, lambda window, frames: [])
    try:
        recovered.tick(force=True)
        assert recovered.reconciler.latest["video"][("r", "st", fresh.event_uid)]["revision"] == 2
        assert recovered.reconciler.groups()[0]["difference"] == 0
    finally:
        recovered.close()


def test_runtime_keeps_missing_evidence_unresolved(tmp_path):
    scope = ReconciliationScope("s", "r", "st", "finish", "cam", group_gap_ms=100, context_ms=100, late_budget_ms=100)
    runtime = CountRuntime(rules(), scope, ClockMapping(calibrated=True), tmp_path / "out", tmp_path / "chips", lambda window, frames: [])
    try:
        runtime.on_event(crossings())
        runtime.reconciler.progress(1500, start_ms=900)
        runtime._start_ms = 900
        runtime.tick(force=True)
        report = json.loads((tmp_path / "out" / "count_reconciliation.json").read_text(encoding="utf-8"))
        assert report["groups"][0]["status"] == "unresolved"
        assert report["groups"][0]["rescan"]["reason"] == "evidence_unavailable"
    finally:
        runtime.close()


class _Boxes:
    def __init__(self, x, tracked=False):
        self.xyxy = np.array([[x - 10, 10, x + 10, 30]], dtype=float)
        self.cls, self.conf = np.array([0]), np.array([0.9])
        self.id = np.array([1]) if tracked else None

    def __iter__(self):
        yield SimpleNamespace(xyxy=self.xyxy, cls=self.cls, conf=self.conf)

    def __len__(self):
        return 1


class _Model:
    names = {0: "person"}
    def __init__(self):
        self.callbacks = {"on_predict_postprocess_end": []}
        self.x = 30
    def track(self, frame, **kwargs):
        result = SimpleNamespace(boxes=_Boxes(self.x), names=self.names)
        for callback in self.callbacks["on_predict_postprocess_end"]:
            callback(SimpleNamespace(results=[result]))
        result.boxes = _Boxes(self.x, tracked=True)
        return [result]


def test_detector_count_outlet_precedes_legacy_filter_and_synchronous_ocr():
    model = _Model()
    detector = Detector("fake", model=model, ocr=None)
    events, audits = [], []
    detector.configure_counting(rules(), events.append, lambda *args: audits.append(args), count_only=True)
    detector._run_sync_ocr_for_track = lambda *args: pytest.fail("counting must not call OCR")
    try:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detector.process_frame(frame, 0.0, frame_stamp=FrameStamp(1, "g", 1000, 1000))
        model.x = 70
        detector.process_frame(frame, 0.04, frame_stamp=FrameStamp(2, "g", 1040, 1040))
        assert len(events) == 1 and events[0].eligibility == "admitted"
        assert len(audits[0][1]) == 1
        assert audits[0][1][0].track_id is None
        assert audits[0][2][0].track_id == 1
    finally:
        detector.stop()


def test_original_recording_replay_covers_evicted_window_and_enforces_memory_budget(tmp_path):
    import cv2
    path = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (64, 48))
    assert writer.isOpened()
    for index in range(50):
        writer.write(np.full((48, 64, 3), index, dtype=np.uint8))
    writer.release()
    provider = RecordingEvidenceProvider(source_file=path, recording_start_ms=10000)
    frames = provider.request(10420, 11000).load(100, 2 * 1024 * 1024)
    assert frames[0][0].observed_at_ms <= 10420
    assert frames[-1][0].observed_at_ms >= 11000
    assert len({stamp.generation for stamp, _ in frames}) == 1
    with pytest.raises(EvidenceUnavailableError, match="memory"):
        provider.load(10420, 11000, 100, 100)
    with pytest.raises(EvidenceUnavailableError, match="cover"):
        provider.load(10420, 15000, 100, 2 * 1024 * 1024)


def test_identity_revisions_require_trusted_evidence_and_restore_after_restart(tmp_path):
    scope = ReconciliationScope("s", "r", "st", "finish", "cam")
    runtime = CountRuntime(rules(), scope, ClockMapping(calibrated=True), tmp_path / "out", tmp_path / "chips", lambda *args: [])
    event = crossings()
    try:
        runtime.on_event(event)
        with pytest.raises(ValueError, match="trusted"):
            runtime.apply_identity_revision({"event_uid": event.event_uid, "revision": 2, "eligibility": "excluded", "reasons": ["staff"]})
        with pytest.raises(ValueError, match="cannot exclude"):
            runtime.apply_identity_revision({"event_uid": event.event_uid, "revision": 2, "identity": "rider-1", "identity_reliable": True, "eligibility": "excluded", "reasons": ["no_chip"]})
        with pytest.raises(ValueError, match="raw crossing"):
            runtime.apply_identity_revision({"event_uid": event.event_uid, "revision": 2, "estimated_cross_time_ms": 1})
        with pytest.raises(ValueError, match="integer"):
            runtime.apply_identity_revision({"event_uid": event.event_uid, "revision": True})
        runtime.apply_identity_revision({"event_uid": event.event_uid, "revision": 2, "identity": "official-1", "identity_reliable": True, "eligibility": "excluded", "reasons": ["staff"]})
    finally:
        runtime.close()
    restored = CountRuntime(rules(), scope, ClockMapping(calibrated=True), tmp_path / "out", tmp_path / "chips", lambda *args: [])
    try:
        current = restored.reconciler.latest["video"][("r", "st", event.event_uid)]
        assert current["revision"] == 2 and current["eligibility"] == "excluded"
        restored.on_event(event)
        assert len(JsonlTail(restored.events.path).read()) == 2
    finally:
        restored.close()
    with pytest.raises(ValueError, match="different rules"):
        CountRuntime(replace(rules(), version="2"), scope, ClockMapping(), tmp_path / "out", tmp_path / "chips", lambda *args: [])
