from dataclasses import FrozenInstanceError

import pytest

from realtime.participant_models import (
    BibEvidence,
    CrossingCandidate,
    ParticipantIdentity,
    ParticipantObservation,
)


def _minimal_observation() -> ParticipantObservation:
    return ParticipantObservation(
        source_id=0,
        segment_id=1,
        frame_index=1,
        capture_time_ms=33.0,
        raw_track_id=None,
        participant_bbox=(100, 100, 200, 300),
        person_bbox=(100, 100, 200, 300),
        equipment_bbox=None,
        bib_bboxes=(),
        confidence=0.8,
    )


def test_participant_observation_keeps_physical_evidence_separate():
    observation = ParticipantObservation(
        source_id=0,
        segment_id=1,
        frame_index=120,
        capture_time_ms=4_000.0,
        raw_track_id=405,
        participant_bbox=(700, 300, 850, 900),
        person_bbox=(720, 300, 835, 780),
        equipment_bbox=(700, 600, 850, 900),
        bib_bboxes=((750, 430, 805, 490),),
        confidence=0.91,
    )

    assert observation.raw_track_id == 405
    assert observation.person_bbox != observation.equipment_bbox
    assert observation.bib_bboxes == ((750, 430, 805, 490),)


def test_observation_does_not_require_a_bib_or_tracker_id():
    observation = ParticipantObservation(
        source_id=0,
        segment_id=1,
        frame_index=121,
        capture_time_ms=4_033.0,
        raw_track_id=None,
        participant_bbox=(700, 300, 850, 900),
        person_bbox=None,
        equipment_bbox=None,
        bib_bboxes=(),
        confidence=0.42,
    )

    assert observation.raw_track_id is None
    assert observation.bib_bboxes == ()


def test_observation_is_immutable():
    observation = _minimal_observation()

    with pytest.raises(FrozenInstanceError):
        observation.frame_index = 999


def test_bib_evidence_keeps_source_position_and_optional_text():
    evidence = BibEvidence(
        bbox=(750, 430, 805, 490),
        frame_index=120,
        capture_time_ms=4_000.0,
        confidence=0.88,
        crop_quality=0.76,
    )

    assert evidence.text_candidate is None
    assert evidence.frame_index == 120


def test_participant_identity_collects_raw_tracks_and_bib_evidence():
    evidence = BibEvidence(
        bbox=(750, 430, 805, 490),
        frame_index=120,
        capture_time_ms=4_000.0,
        confidence=0.88,
        crop_quality=0.76,
        text_candidate="165",
    )
    identity = ParticipantIdentity(
        participant_id="P000004",
        event_profile_name="current-event",
        raw_track_ids={405, 605},
        bib_evidence=[evidence],
    )

    assert identity.raw_track_ids == {405, 605}
    assert identity.bib_evidence[0].text_candidate == "165"
    assert identity.identity_status == "ACTIVE"


def test_crossing_candidate_uses_stable_participant_identity():
    candidate = CrossingCandidate(
        session_id="race-2026-08-06",
        source_id=0,
        participant_id="P000004",
        raw_track_id=605,
        crossing_time_ms=4_200.0,
    )

    assert candidate.participant_id == "P000004"
    assert candidate.raw_track_id == 605
