import pytest

from realtime.crossing_lifecycle import CrossingLifecycle
from realtime.crossing_lifecycle import CrossingState
from realtime.participant_models import CrossingCandidate


def crossing(participant_id, raw_track_id=1, time_ms=1_000):
    return CrossingCandidate(
        session_id="test-session",
        source_id=0,
        participant_id=participant_id,
        raw_track_id=raw_track_id,
        crossing_time_ms=time_ms,
    )


def test_track_split_during_one_crossing_emits_one_event():
    lifecycle = CrossingLifecycle(mode="finish_once")
    assert lifecycle.admit(crossing(participant_id="P4", raw_track_id=405)) is True
    assert lifecycle.admit(crossing(participant_id="P4", raw_track_id=605)) is False


def test_adjacent_participants_each_emit_an_event():
    lifecycle = CrossingLifecycle(mode="finish_once")
    assert lifecycle.admit(crossing(participant_id="P18", raw_track_id=5534)) is True
    assert lifecycle.admit(crossing(participant_id="P19", raw_track_id=6543)) is True


def test_multi_lap_reuses_identity_but_allows_new_passage():
    lifecycle = CrossingLifecycle(mode="multi_lap", min_lap_interval_ms=30_000)
    assert lifecycle.admit(crossing(participant_id="P4", time_ms=1_000)) is True
    assert lifecycle.admit(crossing(participant_id="P4", time_ms=31_500)) is True


def test_invalid_mode_fails_explicitly():
    with pytest.raises(ValueError, match="crossing lifecycle mode"):
        CrossingLifecycle(mode="unknown")


def test_snapshot_preserves_passage_and_raw_track_audit():
    lifecycle = CrossingLifecycle(mode="finish_once")
    first = crossing(participant_id="P4", raw_track_id=405)
    second = crossing(participant_id="P4", raw_track_id=605, time_ms=1_100)

    assert lifecycle.admit(first) is True
    assert lifecycle.admit(second) is False

    snapshot = lifecycle.snapshot(first)
    assert snapshot is not None
    assert snapshot.state is CrossingState.EXPIRED
    assert snapshot.passage_index == 1
    assert snapshot.raw_track_ids == (405, 605)
    assert snapshot.state_history == (
        CrossingState.APPROACHING,
        CrossingState.IN_GATE,
        CrossingState.CROSSED,
        CrossingState.COOLDOWN,
        CrossingState.EXPIRED,
    )
