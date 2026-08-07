import builtins
import sys
from collections import deque
from types import SimpleNamespace

from realtime.main_window import MainWindow
from realtime.participant_identity import IdentityConfig, ParticipantIdentityManager
from realtime.participant_models import ParticipantObservation


def _observation(track_id, time_ms, bbox=(100, 100, 200, 300)):
    return ParticipantObservation(
        source_id=0,
        segment_id=1,
        frame_index=int(time_ms),
        capture_time_ms=float(time_ms),
        raw_track_id=track_id,
        participant_bbox=bbox,
        person_bbox=bbox,
        equipment_bbox=None,
        bib_bboxes=(),
        confidence=0.9,
    )


def test_expired_participant_prunes_identity_state_and_raw_track_indexes():
    manager = ParticipantIdentityManager(
        IdentityConfig(
            max_match_gap_ms=100.0,
            participant_retention_ms=500.0,
            raw_track_retention_ms=200.0,
        )
    )
    first = manager.resolve(_observation(405, 0.0))

    second = manager.resolve(_observation(405, 600.0))
    metrics = manager.runtime_metrics()

    assert second.participant_id != first.participant_id
    assert metrics["active_participants"] == 1
    assert metrics["appearance_summaries"] == 1
    assert metrics["raw_track_mappings"] == 1


def test_ambiguous_identity_uses_a_shorter_retention_window():
    manager = ParticipantIdentityManager(
        IdentityConfig(
            participant_retention_ms=1_000.0,
            ambiguity_retention_ms=100.0,
        )
    )
    manager.resolve(_observation(101, 0.0, bbox=(100, 100, 200, 300)))
    manager.resolve(_observation(202, 0.0, bbox=(180, 100, 280, 300)))
    ambiguous = manager.resolve(
        _observation(303, 400.0, bbox=(140, 100, 240, 300))
    )
    assert ambiguous.participant.identity_status == "AMBIGUOUS"

    manager.prune_expired(600.0)
    metrics = manager.runtime_metrics()

    assert metrics["active_participants"] == 2
    assert metrics["ambiguity_records"] == 0


def test_identity_resolution_does_not_import_a_heavy_reid_runtime(monkeypatch):
    forbidden = {"torchreid", "fastreid"}
    imported = []
    real_import = builtins.__import__

    def tracking_import(name, *args, **kwargs):
        imported.append(name.split(".", 1)[0])
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", tracking_import)
    before = set(sys.modules)

    ParticipantIdentityManager().resolve(_observation(405, 0.0))

    assert forbidden.isdisjoint(imported)
    assert forbidden.isdisjoint(set(sys.modules) - before)


def test_live_monitor_reports_identity_and_queue_metrics_without_extra_refreshes():
    thread = SimpleNamespace(
        last_frame_metrics={
            "participants": 2,
            "track_fragments_merged": 1,
            "identity_ambiguities": 1,
            "queue_depth": 3,
            "dropped_frames": 4,
        }
    )
    window = SimpleNamespace(
        _live_monitor_enabled=True,
        _live_monitor_samples={},
        _live_monitor_window_seconds=30.0,
        _live_monitor_min_fps=12.0,
        video_threads={0: thread},
        _fps_0=20.0,
    )

    MainWindow._update_live_monitor_sample(window, 0, 100.0, [{"bbox": (0, 0, 1, 1)}], [])
    evaluation = MainWindow._evaluate_live_monitor_source(window, 0, 100.0)

    assert isinstance(window._live_monitor_samples[0], deque)
    assert evaluation["participant_count"] == 2
    assert evaluation["fragment_merges"] == 1
    assert evaluation["identity_ambiguities"] == 1
    assert evaluation["queue_depth_max"] == 3
    assert evaluation["dropped_frames"] == 4
    assert "身份=2" in evaluation["detail"]
