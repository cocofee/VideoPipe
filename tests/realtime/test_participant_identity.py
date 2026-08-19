import pytest

from realtime.participant_identity import (
    IdentityConfig,
    ParticipantIdentityManager,
)
from realtime.participant_models import ParticipantObservation


def observation(
    track_id,
    time_ms,
    bbox=(100, 100, 200, 300),
    *,
    source_id=0,
    segment_id=1,
    bib_bboxes=(),
):
    return ParticipantObservation(
        source_id=source_id,
        segment_id=segment_id,
        frame_index=int(time_ms),
        capture_time_ms=time_ms,
        raw_track_id=track_id,
        participant_bbox=bbox,
        person_bbox=bbox,
        equipment_bbox=None,
        bib_bboxes=bib_bboxes,
        confidence=0.9,
    )


def test_new_raw_track_creates_participant():
    manager = ParticipantIdentityManager(config=IdentityConfig())

    result = manager.resolve(observation(track_id=405, time_ms=1000))

    assert result.created is True
    assert result.merged_raw_track is False
    assert result.participant.participant_id == "P000001"
    assert result.participant.raw_track_ids == {405}


def test_new_participants_receive_sequential_ids():
    manager = ParticipantIdentityManager(config=IdentityConfig())

    first = manager.resolve(observation(track_id=101, time_ms=1000))
    second = manager.resolve(
        observation(track_id=202, time_ms=1000, bbox=(400, 100, 500, 300))
    )

    assert first.participant.participant_id == "P000001"
    assert second.participant.participant_id == "P000002"


def test_same_raw_track_returns_existing_participant():
    manager = ParticipantIdentityManager(config=IdentityConfig())
    first = manager.resolve(observation(track_id=405, time_ms=1000))

    second = manager.resolve(
        observation(track_id=405, time_ms=1100, bbox=(105, 105, 205, 305))
    )

    assert second.participant.participant_id == first.participant.participant_id
    assert second.created is False
    assert second.merged_raw_track is False


def test_same_raw_track_reuse_after_large_reverse_jump_creates_participant():
    manager = ParticipantIdentityManager(config=IdentityConfig())
    first = manager.resolve(
        observation(track_id=405, time_ms=1000, bbox=(1000, 100, 1100, 300))
    )
    manager.resolve(
        observation(track_id=405, time_ms=1100, bbox=(800, 100, 900, 300))
    )

    reused = manager.resolve(
        observation(track_id=405, time_ms=1800, bbox=(1200, 100, 1300, 300))
    )

    assert reused.participant.participant_id != first.participant.participant_id
    assert reused.created is True
    assert reused.merged_raw_track is False
    assert "raw_track_discontinuity" in reused.reasons


def test_same_raw_track_reuse_after_scaled_reverse_jump_creates_participant():
    manager = ParticipantIdentityManager(config=IdentityConfig())
    first = manager.resolve(
        observation(track_id=900016, time_ms=1000, bbox=(1271, 513, 1550, 874))
    )
    manager.resolve(
        observation(track_id=900016, time_ms=1040, bbox=(1106, 525, 1379, 868))
    )

    reused = manager.resolve(
        observation(track_id=900016, time_ms=1998, bbox=(1645, 365, 1879, 746))
    )

    assert reused.participant.participant_id != first.participant.participant_id
    assert reused.created is True
    assert "raw_track_discontinuity" in reused.reasons


def test_same_raw_track_reuse_on_same_frame_after_hard_jump_creates_participant():
    manager = ParticipantIdentityManager(config=IdentityConfig())
    first = manager.resolve(
        observation(track_id=900012, time_ms=1000, bbox=(100, 500, 300, 900))
    )
    merged = manager.resolve(
        observation(track_id=9298, time_ms=1100, bbox=(80, 500, 280, 900))
    )

    reused = manager.resolve(
        observation(track_id=900012, time_ms=1100, bbox=(2100, 300, 2350, 700))
    )

    assert merged.participant.participant_id == first.participant.participant_id
    assert reused.participant.participant_id != first.participant.participant_id
    assert reused.created is True
    assert "raw_track_discontinuity" in reused.reasons


def test_same_raw_track_large_forward_motion_keeps_participant():
    manager = ParticipantIdentityManager(config=IdentityConfig())
    first = manager.resolve(
        observation(track_id=405, time_ms=1000, bbox=(1600, 100, 1800, 500))
    )
    manager.resolve(
        observation(track_id=405, time_ms=1100, bbox=(1400, 100, 1600, 500))
    )

    continued = manager.resolve(
        observation(track_id=405, time_ms=1800, bbox=(0, 100, 200, 500))
    )

    assert continued.participant.participant_id == first.participant.participant_id
    assert continued.created is False


@pytest.mark.parametrize(
    ("source_id", "segment_id"),
    [(1, 1), (0, 2)],
)
def test_same_raw_track_in_another_source_or_segment_creates_participant(
    source_id, segment_id
):
    manager = ParticipantIdentityManager(config=IdentityConfig())
    first = manager.resolve(observation(track_id=405, time_ms=1000))

    separate = manager.resolve(
        observation(
            track_id=405,
            time_ms=1400,
            source_id=source_id,
            segment_id=segment_id,
        )
    )

    assert separate.participant.participant_id != first.participant.participant_id
    assert separate.created is True


def test_fragmented_track_reuses_recent_participant():
    manager = ParticipantIdentityManager(config=IdentityConfig())
    first = manager.resolve(
        observation(track_id=405, time_ms=1000, bbox=(700, 300, 850, 900))
    )

    second = manager.resolve(
        observation(track_id=605, time_ms=1400, bbox=(705, 320, 846, 902))
    )

    assert second.participant.participant_id == first.participant.participant_id
    assert second.participant.raw_track_ids == {405, 605}
    assert second.created is False
    assert second.merged_raw_track is True


def test_fragment_moving_against_established_direction_does_not_merge():
    manager = ParticipantIdentityManager(config=IdentityConfig())
    first = manager.resolve(
        observation(track_id=405, time_ms=1000, bbox=(100, 100, 200, 300))
    )
    manager.resolve(
        observation(track_id=405, time_ms=1100, bbox=(120, 100, 220, 300))
    )

    reversed_fragment = manager.resolve(
        observation(track_id=605, time_ms=1200, bbox=(100, 100, 200, 300))
    )

    assert reversed_fragment.participant.participant_id != first.participant.participant_id
    assert reversed_fragment.created is True


def test_simultaneous_adjacent_riders_never_merge():
    manager = ParticipantIdentityManager(config=IdentityConfig())
    left = manager.resolve(
        observation(track_id=6543, time_ms=1000, bbox=(900, 300, 1050, 900))
    )

    right = manager.resolve(
        observation(track_id=5534, time_ms=1000, bbox=(1040, 300, 1190, 900))
    )

    assert left.participant.participant_id != right.participant.participant_id
    assert right.created is True


def test_participant_already_used_in_frame_is_not_reused_by_exact_track_mapping():
    manager = ParticipantIdentityManager(config=IdentityConfig())
    first = manager.resolve(
        observation(track_id=405, time_ms=1000, bbox=(100, 100, 200, 300))
    )

    collision = manager.resolve(
        observation(track_id=405, time_ms=1040, bbox=(400, 100, 500, 300)),
        excluded_participant_ids={first.participant.participant_id},
    )

    assert collision.participant.participant_id != first.participant.participant_id
    assert collision.created is True


def test_participant_already_used_in_frame_is_not_reused_by_fragment_match():
    manager = ParticipantIdentityManager(config=IdentityConfig())
    first = manager.resolve(
        observation(track_id=405, time_ms=1000, bbox=(100, 100, 200, 300))
    )

    collision = manager.resolve(
        observation(track_id=605, time_ms=1040, bbox=(105, 100, 205, 300)),
        excluded_participant_ids={first.participant.participant_id},
    )

    assert collision.participant.participant_id != first.participant.participant_id
    assert collision.created is True


def test_similar_bib_layout_only_supports_a_geometrically_valid_match():
    manager = ParticipantIdentityManager(config=IdentityConfig())
    first = manager.resolve(
        observation(
            track_id=405,
            time_ms=1000,
            bib_bboxes=((130, 140, 160, 170),),
        )
    )

    fragment = manager.resolve(
        observation(
            track_id=605,
            time_ms=1200,
            bbox=(105, 100, 205, 300),
            bib_bboxes=((135, 140, 165, 170),),
        )
    )

    assert fragment.participant.participant_id == first.participant.participant_id
    assert any("bib_layout_similarity" in reason for reason in fragment.reasons)


def test_similar_bib_layout_cannot_bypass_geometric_match_gates():
    manager = ParticipantIdentityManager(config=IdentityConfig())
    first = manager.resolve(
        observation(
            track_id=405,
            time_ms=1000,
            bib_bboxes=((130, 140, 160, 170),),
        )
    )

    impossible_match = manager.resolve(
        observation(
            track_id=605,
            time_ms=1200,
            bbox=(1000, 100, 1100, 300),
            bib_bboxes=((1030, 140, 1060, 170),),
        )
    )

    assert impossible_match.participant.participant_id != first.participant.participant_id
    assert impossible_match.created is True


def test_ambiguous_match_creates_participant_and_preserves_reasons():
    manager = ParticipantIdentityManager(config=IdentityConfig())
    left = manager.resolve(
        observation(track_id=101, time_ms=1000, bbox=(100, 100, 200, 300))
    )
    right = manager.resolve(
        observation(track_id=202, time_ms=1000, bbox=(180, 100, 280, 300))
    )

    ambiguous = manager.resolve(
        observation(track_id=303, time_ms=1400, bbox=(140, 100, 240, 300))
    )

    assert ambiguous.participant.participant_id not in {
        left.participant.participant_id,
        right.participant.participant_id,
    }
    assert ambiguous.participant.identity_status == "AMBIGUOUS"
    assert ambiguous.created is True
    assert ambiguous.reasons
    assert any("ambiguous" in reason for reason in ambiguous.reasons)
