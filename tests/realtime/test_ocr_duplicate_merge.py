from realtime.database import Database
from realtime.ocr_manager import OCRManager


def _event(event_id, cross_time, **overrides):
    event = {
        "event_id": event_id,
        "track_id": event_id,
        "cross_time": cross_time,
        "cross_time_str": f"00:00:{cross_time:06.3f}",
        "cross_realtime": "2026-08-04 12:00:00.000",
        "bib_number": None,
        "bib_confidence": 0.0,
        "bib_status": "unrecognized",
        "detection_confidence": 0.9,
        "source_id": 0,
        "position_x": 100,
        "position_y": 200,
        "bbox": [80, 100, 140, 220],
        "ocr_state": "PENDING",
    }
    event.update(overrides)
    return event


def test_ocr_duplicate_merge_keeps_one_event_and_moves_evidence(tmp_path):
    db = Database(str(tmp_path / "event.db"))
    try:
        db.insert_event(_event(1, 10.0, screenshot_full="first.jpg"), enable_dedup=False)
        db.insert_event(_event(2, 11.0, screenshot_bib="second-bib.jpg"), enable_dedup=False)
        db.insert_evidence(
            {
                "event_id": 2,
                "source_id": 0,
                "screenshot_bib": "second-bib.jpg",
                "confidence": 0.95,
            }
        )

        merged = db.merge_ocr_duplicate_event(
            duplicate_event_id=2,
            target_event_id=1,
            bib="66",
            conf=0.95,
            ocr_state="DONE",
            evidence_dir="event-2",
            notes="Source: LOCAL. Merged to 1",
            use_ocr_conn=True,
        )

        assert merged is True
        target = db.get_event(1)
        duplicate = db.get_event(2)
        assert target["bib_number"] == "66"
        assert target["bib_confidence"] == 0.95
        assert target["screenshot_full"] == "first.jpg"
        assert target["screenshot_bib"] == "second-bib.jpg"
        assert duplicate["is_void"] == 1
        assert "MERGED_TO:1" in duplicate["notes"]
        assert [event["event_id"] for event in db.get_all_events()] == [1]
        assert {evidence["event_id"] for evidence in db.get_event_evidences(1)} == {1}
    finally:
        db.close()


class _MergeDatabase:
    def __init__(self, merge_result=True):
        self.merge_result = merge_result
        self.merge_calls = []
        self.update_calls = []

    def merge_ocr_duplicate_event(self, **kwargs):
        self.merge_calls.append(kwargs)
        return self.merge_result

    def update_event_ocr_result(self, *args, **kwargs):
        self.update_calls.append((args, kwargs))
        return True


def test_ocr_manager_uses_real_merge_for_confirmed_duplicate():
    db = _MergeDatabase()
    manager = OCRManager(db)

    saved = manager._persist_ocr_resolution(
        event_id=2,
        bib="66",
        conf=0.95,
        status="DONE",
        evidence_dir="event-2",
        notes="Source: LOCAL. Merged to 1",
        merged_to=1,
    )

    assert saved is True
    assert db.merge_calls[0]["duplicate_event_id"] == 2
    assert db.merge_calls[0]["target_event_id"] == 1
    assert db.update_calls == []


def test_ocr_manager_falls_back_to_normal_update_when_merge_fails():
    db = _MergeDatabase(merge_result=False)
    manager = OCRManager(db)

    saved = manager._persist_ocr_resolution(
        event_id=2,
        bib="66",
        conf=0.95,
        status="DONE",
        evidence_dir="event-2",
        notes="Source: LOCAL. Merged to 1",
        merged_to=1,
    )

    assert saved is True
    assert len(db.merge_calls) == 1
    assert len(db.update_calls) == 1
