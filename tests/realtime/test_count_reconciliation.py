from dataclasses import replace

import pytest

from realtime.count_reconciliation import ClockMapping, GroupReconciler, ReconciliationScope


def reconciler(**changes):
    scope = ReconciliationScope("s", "r", "st", "finish", "cam", group_gap_ms=400, late_budget_ms=500)
    engine = GroupReconciler(replace(scope, **changes), ClockMapping(calibrated=True))
    engine.progress(15000, start_ms=10000)
    return engine


def video(uid="v1", stamp=11000, **changes):
    row = {"event_uid": uid, "revision": 1, "is_active": True, "session_id": "s", "race_id": "r",
           "stage_id": "st", "timing_position": "finish", "source_id": "cam", "group_id": "",
           "estimated_cross_time_ms": stamp, "eligibility": "admitted", "quality": []}
    row.update(changes)
    return row


def chip(uid="c1", stamp=11000, **changes):
    row = {"event_id": uid, "revision": 1, "is_active": True, "race_id": "r", "stage_id": "st",
           "source": "cyclerace", "group_id": "g1", "passage_timestamp_ms": stamp}
    row.update(changes)
    return row


def test_group_sprint_reconciles_six_overlapping_times_not_per_rider_identity():
    engine = reconciler()
    for index in range(6):
        engine.ingest("video", video(f"v{index}", 11000 + index * 25))
        engine.ingest("chip", chip(f"c{index}", 11010 + index * 25))
    window, = engine.groups()
    assert window["video_count"] == window["chip_count"] == 6
    assert window["status"] == "matched"
    assert window["provisional"] is True


def test_late_duplicate_correction_and_retraction_remove_old_contributions():
    engine = reconciler()
    engine.ingest("video", video())
    assert engine.groups()[0]["difference"] == 1
    engine.ingest("chip", chip())
    assert engine.groups()[0]["difference"] == 0
    assert not engine.ingest("chip", chip())
    engine.ingest("chip", chip(revision=2, passage_timestamp_ms=13000))
    assert [(row["video_count"], row["chip_count"]) for row in engine.groups()] == [(1, 0), (0, 1)]
    engine.ingest("chip", chip(revision=3, is_active=False))
    assert [(row["video_count"], row["chip_count"]) for row in engine.groups()] == [(1, 0)]
    assert not engine.ingest("chip", chip(revision=2))


def test_scope_filter_runs_after_highest_revision():
    engine = reconciler(group_ids=("g1",))
    engine.ingest("video", video(group_id="g1"))
    engine.ingest("chip", chip())
    engine.ingest("chip", chip(revision=2, group_id="g2"))
    assert engine.groups()[0]["chip_count"] == 0
    engine.ingest("video", video(revision=2, timing_position="start"))
    assert engine.groups() == []


def test_anonymous_group_is_preserved_as_pending_in_restricted_scope():
    engine = reconciler(group_ids=("g1",))
    engine.ingest("video", video())
    window, = engine.groups()
    assert window["pending_count"] == 1
    assert "group_scope_uncertain" in window["reasons"]


def test_equal_counts_with_quality_evidence_still_require_rescan():
    engine = reconciler()
    engine.ingest("video", video(quality=["suspected_merged_box"]))
    engine.ingest("chip", chip())
    window, = engine.groups()
    assert window["difference"] == 0
    assert window["status"] == "anomaly"
    assert "suspected_merged_box" in window["reasons"]


def test_clock_update_moves_video_contribution_and_explicit_watermark_closes_window():
    engine = reconciler()
    engine.ingest("video", video(stamp=11000))
    engine.ingest("chip", chip(stamp=13000))
    assert len(engine.groups()) == 2
    engine.update_clock(ClockMapping(offset_ms=2000, version="2", calibrated=True))
    window, = engine.groups()
    assert window["difference"] == 0
    assert window["clock_version"] == "2"
    engine.progress(15000, chip_watermark_ms=16000)
    assert engine.groups()[0]["provisional"] is False


def test_clock_uncertainty_and_progress_block_false_confirmation():
    engine = reconciler()
    engine.ingest("video", video(stamp=14600))
    engine.ingest("chip", chip(stamp=14600))
    assert engine.groups()[0]["status"] == "waiting"
    engine.update_clock(ClockMapping(calibrated=False))
    assert "clock_uncertain" in engine.groups()[0]["reasons"]


def test_conflicting_revision_fails_and_missing_chip_timestamp_is_reported():
    engine = reconciler()
    engine.ingest("chip", chip())
    with pytest.raises(ValueError, match="conflicting"):
        engine.ingest("chip", chip(passage_timestamp_ms=12000))
    engine.ingest("chip", chip("c2", passage_timestamp_ms=None))
    assert engine.report()["chip_records_without_absolute_time"] == 1


def test_long_continuous_bunches_have_bounded_windows_and_keep_every_contribution():
    engine = reconciler(max_group_ms=1000)
    for index in range(20):
        engine.ingest("video", video(f"v{index}", 10000 + index * 200))
    groups = engine.groups()
    assert sum(row["video_count"] for row in groups) == 20
    assert all(row["end_ms"] - row["start_ms"] <= 1000 for row in groups)


def test_chip_missing_identity_cannot_name_the_missing_person():
    engine = reconciler()
    engine.ingest("video", video(identity="rider-1", identity_reliable=True))
    engine.ingest("chip", chip())
    assert engine.groups()[0]["unmatched_reliable_identities"] == []


def test_quality_only_windows_are_distinct_and_long_outage_is_split():
    engine = reconciler(max_group_ms=1000)
    engine.add_quality({"source_id": "cam", "reason": "sampling_gap", "start_ms": 10000, "end_ms": 13000})
    groups = engine.groups()
    assert len(groups) == 3
    assert len({row["window_id"] for row in groups}) == 3
    assert all(row["end_ms"] - row["start_ms"] <= 1000 for row in groups)


def test_missing_absolute_chip_time_blocks_match_until_revision_repairs_it():
    engine = reconciler()
    engine.ingest("video", video())
    engine.ingest("chip", chip())
    engine.ingest("chip", chip("unknown", passage_timestamp_ms=None))
    assert "chip_timestamp_missing" in engine.groups()[0]["reasons"]
    engine.ingest("chip", chip("unknown", revision=2, is_active=False))
    assert engine.groups()[0]["status"] == "matched"


def test_restart_or_clock_regression_cannot_erase_processed_history():
    engine = reconciler()
    engine.ingest("video", video())
    engine.ingest("chip", chip())
    before = engine.cutoff_ms
    engine.progress(10000, start_ms=10000)
    assert engine.cutoff_ms == before
    assert engine.groups()[0]["status"] == "matched"
