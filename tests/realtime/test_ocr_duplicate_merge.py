import json
from pathlib import Path

import cv2
import numpy as np

from realtime.database import Database
from realtime.detector import BibStatus, CrossingEvent
from realtime.event_recorder import EventRecorder
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

    def get_config(self, key, default=None):
        return default

    def get_event_by_bib_and_time(self, *args, **kwargs):
        return None


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


def test_pending_ocr_clears_a_previous_automatic_bib(tmp_path):
    db = Database(str(tmp_path / "event.db"))
    try:
        db.insert_event(
            _event(1, 10.0, bib_number="4", bib_confidence=0.95, ocr_state="DONE"),
            enable_dedup=False,
        )
        manager = OCRManager(db)

        manager._persist_ocr_resolution(
            event_id=1,
            bib="2",
            conf=0.30,
            status="PENDING",
            evidence_dir="event-1",
            notes="conflict",
        )

        event = db.get_event(1)
        assert event["bib_number"] is None
        assert event["bib_confidence"] == 0.0
        assert event["ocr_state"] == "PENDING"
    finally:
        db.close()


def test_pending_ocr_preserves_a_manually_corrected_bib(tmp_path):
    db = Database(str(tmp_path / "event.db"))
    try:
        db.insert_event(_event(1, 10.0), enable_dedup=False)
        db.update_event(1, {"bib_number": "33"})
        manager = OCRManager(db)

        manager._persist_ocr_resolution(
            event_id=1,
            bib="4",
            conf=0.30,
            status="PENDING",
            evidence_dir="event-1",
            notes="conflict",
        )

        event = db.get_event(1)
        assert event["bib_number"] == "33"
        assert event["manual_corrected"] == 1
    finally:
        db.close()


def test_cycling_ocr_defaults_to_short_numeric_bibs():
    manager = OCRManager(_MergeDatabase())

    assert manager.only_numeric is True
    assert manager._normalize_bib_text("23") == "23"
    assert manager._normalize_bib_text("5") == "5"


class _ScaleSensitiveOcr:
    def __call__(self, image):
        if image.shape[1] >= 96:
            return [[None, "23", 0.95]], 0.0
        return [], 0.0


class _TightRecoveryOcr:
    def __init__(self, detected_text, detected_conf, tight_text, tight_conf):
        self.detected_text = detected_text
        self.detected_conf = detected_conf
        self.tight_text = tight_text
        self.tight_conf = tight_conf
        self.tight_shapes = []

    def __call__(self, image):
        return [[None, self.detected_text, self.detected_conf]], 0.0

    def recognize_only(self, image):
        self.tight_shapes.append(image.shape[:2])
        return [[None, self.tight_text, self.tight_conf]], 0.0


def test_multi_scale_ocr_recovers_multi_digit_bib_from_tight_crop(tmp_path):
    db = _MergeDatabase()
    ocr = _TightRecoveryOcr("4", 0.30, "33", 0.94)
    manager = OCRManager(db, ocr_engine=ocr)
    manager.only_numeric = True
    event_dir = tmp_path / "000001"
    _write_consensus_event(event_dir, [20, 80, 120])

    manager._process_event(event_dir)

    result = json.loads((event_dir / "result.json").read_text(encoding="utf-8"))
    assert result["bib"] == "33"
    assert result["status"] == "DONE"
    assert ocr.tight_shapes


def test_tight_crop_single_digit_disagreement_cannot_replace_single_digit(tmp_path):
    db = _MergeDatabase()
    ocr = _TightRecoveryOcr("3", 0.42, "5", 0.90)
    manager = OCRManager(db, ocr_engine=ocr)
    manager.only_numeric = True
    event_dir = tmp_path / "000001"
    _write_consensus_event(event_dir, [20, 80, 120])

    manager._process_event(event_dir)

    result = json.loads((event_dir / "result.json").read_text(encoding="utf-8"))
    assert result["bib"] == "3"
    assert result["status"] == "DONE"


def test_event_processing_upscales_small_detected_bib_before_fallback(tmp_path):
    db = _MergeDatabase()
    manager = OCRManager(db, ocr_engine=_ScaleSensitiveOcr())
    manager.only_numeric = True
    event_dir = tmp_path / "000001"
    event_dir.mkdir()
    (event_dir / "meta.json").write_text(
        json.dumps(
            {
                "event_id": 1,
                "cross_time_unix": 10.0,
                "bbox_athlete": [0, 0, 80, 120],
                "bbox_bib": [0, 0, 32, 27],
            }
        ),
        encoding="utf-8",
    )
    cv2.imwrite(str(event_dir / "bib.jpg"), np.zeros((27, 32, 3), dtype=np.uint8))

    manager._process_event(event_dir)

    result = json.loads((event_dir / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "DONE"
    assert result["bib"] == "23"
    assert result["confidence"] == 0.95
    assert result["image_source"] == "bib"


def _write_consensus_event(event_dir, pixel_values):
    event_dir.mkdir()
    candidate_names = []
    for index, pixel_value in enumerate(pixel_values[1:], start=1):
        name = f"bib_candidate_{index:02d}.png"
        cv2.imwrite(str(event_dir / name), np.full((40, 60, 3), pixel_value, dtype=np.uint8))
        candidate_names.append(name)
    cv2.imwrite(str(event_dir / "bib.jpg"), np.full((40, 60, 3), pixel_values[0], dtype=np.uint8))
    (event_dir / "meta.json").write_text(
        json.dumps(
            {
                "event_id": 1,
                "cross_time_unix": 10.0,
                "bbox_athlete": [0, 0, 100, 160],
                "bbox_bib": [20, 40, 80, 80],
                "bib_evidence_kind": "detected",
                "paths": {
                    "bib": "bib.jpg",
                    "bib_candidates": candidate_names,
                },
            }
        ),
        encoding="utf-8",
    )


def test_multi_frame_consensus_beats_single_high_confidence_misread(tmp_path):
    db = _MergeDatabase()
    manager = OCRManager(db)
    manager.only_numeric = True
    manager.ocr = object()
    event_dir = tmp_path / "000001"
    _write_consensus_event(event_dir, [20, 80, 120])

    def fake_multi_scale(img):
        value = int(round(float(img.mean())))
        if value < 50:
            return "47", 0.95, "fake"
        return "23", 0.72 if value < 100 else 0.82, "fake"

    manager._run_multi_scale_ocr = fake_multi_scale

    manager._process_event(event_dir)

    result = json.loads((event_dir / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "DONE"
    assert result["bib"] == "23"
    assert result["image_source"] == "multi_frame"


def test_conflicting_multi_frame_results_stay_pending(tmp_path):
    db = _MergeDatabase()
    manager = OCRManager(db)
    manager.only_numeric = True
    manager.ocr = object()
    event_dir = tmp_path / "000001"
    _write_consensus_event(event_dir, [20, 80, 120, 180])

    outputs = {
        20: ("47", 0.95, "fake"),
        80: ("2", 0.90, "fake"),
        120: ("7", 0.85, "fake"),
        180: ("3", 0.80, "fake"),
    }
    manager._run_multi_scale_ocr = lambda img: outputs[min(outputs, key=lambda value: abs(value - int(round(float(img.mean())))))]

    manager._process_event(event_dir)

    result = json.loads((event_dir / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "PENDING"
    assert result["error"] == "INCONSISTENT_MULTI_FRAME"
    assert db.merge_calls == []
    assert db.update_calls[-1][0][1] == ""


def test_fallback_only_evidence_never_auto_confirms_or_updates_formal_bib(tmp_path):
    db = _MergeDatabase()
    manager = OCRManager(db)
    manager.only_numeric = True
    manager.ocr = object()
    event_dir = tmp_path / "000001"
    event_dir.mkdir()
    cv2.imwrite(str(event_dir / "bib.jpg"), np.zeros((80, 100, 3), dtype=np.uint8))
    (event_dir / "meta.json").write_text(
        json.dumps(
            {
                "event_id": 1,
                "cross_time_unix": 10.0,
                "bbox_athlete": [0, 0, 100, 160],
                "bbox_bib": [20, 30, 80, 110],
                "bib_evidence_kind": "fallback",
                "paths": {
                    "bib": "bib.jpg",
                    "bib_candidates": [],
                },
            }
        ),
        encoding="utf-8",
    )
    manager._run_multi_scale_ocr = lambda img: ("4", 0.95, "fake")

    manager._process_event(event_dir)

    result = json.loads((event_dir / "result.json").read_text(encoding="utf-8"))
    assert result["bib"] == "4"
    assert result["status"] == "PENDING"
    assert result["error"] == "FALLBACK_ONLY_EVIDENCE"
    assert db.merge_calls == []
    assert db.update_calls[-1][0][1] == ""


def test_new_detected_event_requires_more_than_one_frame_for_auto_confirmation(tmp_path):
    db = _MergeDatabase()
    manager = OCRManager(db)
    manager.only_numeric = True
    manager.ocr = object()
    event_dir = tmp_path / "000001"
    event_dir.mkdir()
    cv2.imwrite(str(event_dir / "bib.jpg"), np.zeros((40, 60, 3), dtype=np.uint8))
    cv2.imwrite(str(event_dir / "bib_candidate_01.jpg"), np.zeros((40, 60, 3), dtype=np.uint8))
    (event_dir / "meta.json").write_text(
        json.dumps(
            {
                "event_id": 1,
                "cross_time_unix": 10.0,
                "bbox_athlete": [0, 0, 100, 160],
                "bbox_bib": [20, 40, 80, 80],
                "bib_evidence_kind": "detected",
                "paths": {
                    "bib": "bib.jpg",
                    "bib_candidates": ["bib_candidate_01.jpg"],
                },
            }
        ),
        encoding="utf-8",
    )
    manager._run_multi_scale_ocr = lambda img: ("23", 0.95, "fake")

    manager._process_event(event_dir)

    result = json.loads((event_dir / "result.json").read_text(encoding="utf-8"))
    assert result["bib"] == "23"
    assert result["status"] == "PENDING"
    assert result["error"] == "INSUFFICIENT_MULTI_FRAME_EVIDENCE"
    assert db.update_calls[-1][0][1] == ""


def test_event_recorder_saves_ranked_bib_candidates(tmp_path):
    db = Database(str(tmp_path / "event.db"))
    try:
        recorder = EventRecorder(str(tmp_path), db)
        event = CrossingEvent(
            event_id=1,
            rank=1,
            track_id=7,
            cross_time=10.0,
            cross_time_str="00:00:10.000",
            cross_realtime="2026-08-04 12:00:10.000",
            bib_number=None,
            bib_confidence=0.0,
            bib_status=BibStatus.UNRECOGNIZED,
            detection_confidence=0.9,
            position=(50, 100),
            bbox=(20, 30, 80, 110),
            bib_crop=np.full((40, 60, 3), 20, dtype=np.uint8),
            bib_evidence_kind="detected",
            bib_candidates=[
                (0.9, np.full((40, 60, 3), 40, dtype=np.uint8), [20, 40, 80, 80]),
                (0.8, np.full((38, 58, 3), 80, dtype=np.uint8), [21, 41, 79, 79]),
            ],
        )

        event_dir = Path(recorder._save_evidence_to_disk(event))
        meta = json.loads((event_dir / "meta.json").read_text(encoding="utf-8"))

        assert (event_dir / "bib_candidate_01.jpg").exists()
        assert (event_dir / "bib_candidate_02.jpg").exists()
        assert meta["paths"]["bib_candidates"] == [
            "bib_candidate_01.jpg",
            "bib_candidate_02.jpg",
        ]
        assert meta["bib_evidence_kind"] == "detected"
    finally:
        db.close()
