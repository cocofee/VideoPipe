from realtime.ocr_manager import ParticipantOcrState
from realtime.ocr_manager import OCRManager


class _Database:
    def get_config(self, key, default=None):
        return default


def test_ocr_votes_from_fragmented_tracks_attach_to_one_participant():
    state = ParticipantOcrState(participant_id="P4")
    state.add_vote(raw_track_id=405, text="165", confidence=0.88)
    state.add_vote(raw_track_id=605, text="165", confidence=0.91)

    assert state.best_candidate == "165"
    assert state.raw_track_ids == {405, 605}


def test_ocr_conflict_does_not_create_second_participant():
    state = ParticipantOcrState(participant_id="P4")
    state.add_vote(raw_track_id=405, text="165", confidence=0.82)
    state.add_vote(raw_track_id=605, text="185", confidence=0.81)

    assert state.status == "CONFLICT"
    assert state.participant_id == "P4"


def test_ocr_manager_normalizes_and_aggregates_raw_tracks_by_participant():
    manager = OCRManager(_Database())

    manager._add_participant_ocr_vote("P4", 405, " 165 ", 0.88)
    manager._add_participant_ocr_vote("P4", 605, "165", 0.91)
    manager._add_participant_ocr_vote("P4", 605, "", 0.99)

    state = manager.get_participant_ocr_state("P4")
    assert state.best_candidate == "165"
    assert state.raw_track_ids == {405, 605}
    assert state.votes == [("165", 0.88), ("165", 0.91)]


def test_participant_ocr_exposes_confidence_for_the_selected_candidate():
    state = ParticipantOcrState(participant_id="P4")

    state.add_vote(raw_track_id=405, text="165", confidence=0.95)
    state.add_vote(raw_track_id=605, text="165", confidence=0.95)
    state.add_vote(raw_track_id=705, text="185", confidence=0.80)

    assert state.best_candidate == "165"
    assert state.best_confidence == 0.95
