import json
from types import SimpleNamespace

import numpy as np
import pytest

from realtime.frame_envelope import FrameEnvelope
from realtime.field_issue_log import FieldIssueLog
from realtime.main_window import MainWindow


def test_issue_marker_records_exact_processing_context(tmp_path):
    log = FieldIssueLog(tmp_path / "issues.jsonl")
    marker = log.record(
        session_id="race-2026-08-06",
        source_id=0,
        category="duplicate_athlete",
        frame_index=1200,
        capture_time_ms=40_000.0,
        participant_ids=("P17",),
        event_ids=(8,),
        metrics={"queue_depth": 2, "processing_time_ms": 14.2},
        note="same athlete appeared twice",
    )

    assert marker.issue_id
    assert marker.capture_time_ms == 40_000.0
    assert marker.category == "duplicate_athlete"

    records = [
        json.loads(line)
        for line in (tmp_path / "issues.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    assert records[0]["participant_ids"] == ["P17"]
    assert records[0]["event_ids"] == [8]
    assert records[0]["queue_depth"] == 2
    assert records[0]["processing_time_ms"] == 14.2
    assert records[0]["operator_note"] == "same athlete appeared twice"


def test_issue_marker_does_not_create_or_modify_crossing_events(tmp_path):
    database_path = tmp_path / "events.db"
    database_path.write_bytes(b"existing-event-database")
    before = database_path.read_bytes()

    FieldIssueLog(tmp_path / "issues.jsonl").record(
        session_id="race",
        source_id=0,
        category="missed_athlete",
        frame_index=300,
        capture_time_ms=10_000.0,
    )

    assert database_path.read_bytes() == before


def test_issue_marker_rejects_unknown_category(tmp_path):
    with pytest.raises(ValueError, match="Unsupported field issue category"):
        FieldIssueLog(tmp_path / "issues.jsonl").record(
            session_id="race",
            source_id=0,
            category="not-a-real-category",
            frame_index=1,
            capture_time_ms=1.0,
        )


def test_field_issue_context_uses_resolved_software_commit(tmp_path):
    frame = np.zeros((8, 12, 3), dtype=np.uint8)
    envelope = FrameEnvelope(
        original_frame=frame,
        frame_index=42,
        capture_time_ms=1_700_000_000_000.0,
        arrival_time_ms=1_700_000_000_005.0,
        segment_id=3,
        processing_time_ms=7.5,
    )
    thread = SimpleNamespace(
        last_processed_envelope=envelope,
        last_frame_metrics={"processing_time_ms": 7.5},
    )
    window = SimpleNamespace(
        video_threads={0: thread},
        field_issue_log=FieldIssueLog(tmp_path / "issues.jsonl"),
        readers={},
        _latest_frame_observations={0: ([], [])},
        model_path=None,
        output_dir=tmp_path / "race-session",
        config={},
        _software_commit="abc123",
    )

    context = MainWindow._capture_field_issue_context(window)

    assert context is not None
    assert context["software_commit"] == "abc123"
    assert context["frame_index"] == 42
    assert not np.shares_memory(context["frame"], frame)


def test_field_issue_persistence_does_not_emit_after_window_cleanup(tmp_path):
    emitted = []
    window = SimpleNamespace(
        _field_issue_closing=True,
        field_issue_saved_signal=SimpleNamespace(emit=emitted.append),
    )
    context = {
        "issue_log": FieldIssueLog(tmp_path / "issues.jsonl"),
        "session_id": "race-session",
        "source_id": 0,
        "category": "missed_athlete",
        "frame_index": 42,
        "capture_time_ms": 1_700_000_000_000.0,
        "segment_id": 3,
        "frame": np.zeros((8, 12, 3), dtype=np.uint8),
        "metrics": {},
        "participant_ids": (),
        "raw_track_ids": (),
        "event_ids": (),
        "software_commit": "abc123",
        "model_identity": None,
        "event_profile": None,
        "note": "",
        "screenshot_relative": tmp_path / "issue.jpg",
        "screenshot_path": tmp_path / "issue.jpg",
    }

    MainWindow._persist_field_issue(window, context)

    assert (tmp_path / "issues.jsonl").exists()
    assert emitted == []
