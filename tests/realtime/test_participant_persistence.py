import json
import sqlite3

from realtime.database import Database
from realtime.detector import BibStatus, CrossingEvent
from realtime.event_recorder import EventRecorder


def _create_legacy_database(path):
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE crossing_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER UNIQUE,
                rank INTEGER,
                track_id INTEGER,
                cross_time REAL,
                cross_time_str TEXT,
                cross_realtime TEXT,
                finish_time TEXT,
                bib_number TEXT,
                original_bib TEXT,
                bib_confidence REAL,
                bib_status TEXT,
                detection_confidence REAL,
                position_x INTEGER,
                position_y INTEGER,
                bbox TEXT,
                screenshot_full TEXT,
                screenshot_clean TEXT,
                screenshot_crop TEXT,
                screenshot_bib TEXT,
                source_id INTEGER DEFAULT 0,
                is_test INTEGER DEFAULT 0,
                is_void INTEGER DEFAULT 0,
                manual_corrected INTEGER DEFAULT 0,
                is_ai_correction INTEGER DEFAULT 0,
                evidence_dir TEXT,
                ocr_state TEXT DEFAULT 'PENDING',
                created_at TEXT,
                modified_at TEXT,
                notes TEXT
            );
            INSERT INTO crossing_events (
                event_id, rank, track_id, cross_time, cross_time_str,
                cross_realtime, bib_number, original_bib, bib_confidence,
                bib_status, detection_confidence, position_x, position_y,
                bbox, source_id, is_void, ocr_state
            ) VALUES (1, 1, 405, 10.0, '00:00:10.000',
                      '2026-08-06 12:00:10.000', NULL, NULL, 0.0,
                      'unrecognized', 0.9, 100, 200, '[1, 2, 3, 4]',
                      0, 0, 'PENDING');
            """
        )
        connection.commit()
    finally:
        connection.close()


def test_existing_crossing_database_adds_nullable_identity_columns(tmp_path):
    database_path = tmp_path / "legacy.db"
    _create_legacy_database(database_path)

    database = Database(str(database_path))
    try:
        columns = {
            row[1]: row
            for row in database._get_conn().execute("PRAGMA table_info(crossing_events)")
        }

        assert "participant_id" in columns
        assert "raw_track_ids_json" in columns
        assert "sport_profile" in columns
        assert "passage_index" in columns
    finally:
        database.close()


def test_existing_event_rows_remain_readable_after_identity_migration(tmp_path):
    database_path = tmp_path / "legacy.db"
    _create_legacy_database(database_path)

    database = Database(str(database_path))
    try:
        record = database.get_event(1)

        assert record["event_id"] == 1
        assert record["track_id"] == 405
        assert record["participant_id"] is None
        assert record["raw_track_ids"] == [405]
        assert record["sport_profile"] == "cycling"
        assert record["passage_index"] == 1

        database._get_conn().execute(
            "UPDATE crossing_events SET raw_track_ids_json = ? WHERE event_id = 1",
            ("not-json",),
        )
        database._get_conn().commit()
        assert database.get_event(1)["raw_track_ids"] == [405]
    finally:
        database.close()


def test_new_event_persists_participant_identity_and_raw_track_audit(tmp_path):
    database = Database(str(tmp_path / "events.db"))
    try:
        database.insert_event(
            {
                "event_id": 1,
                "rank": 1,
                "track_id": 605,
                "cross_time": 10.0,
                "cross_time_str": "00:00:10.000",
                "cross_realtime": "2026-08-06 12:00:10.000",
                "bib_number": None,
                "bib_status": "unrecognized",
                "bib_confidence": 0.0,
                "detection_confidence": 0.9,
                "position_x": 100,
                "position_y": 200,
                "bbox": [1, 2, 3, 4],
                "participant_id": "P000004",
                "raw_track_ids": [405, 605],
                "sport_profile": "cycling-mountain",
                "passage_index": 1,
            },
            enable_dedup=False,
        )

        record = database.get_event(1)

        assert record["participant_id"] == "P000004"
        assert json.loads(record["raw_track_ids_json"]) == [405, 605]
        assert record["raw_track_ids"] == [405, 605]
        assert record["sport_profile"] == "cycling-mountain"
        assert record["passage_index"] == 1
    finally:
        database.close()


def test_event_recorder_persists_identity_fields_into_crossing_event(tmp_path):
    database = Database(str(tmp_path / "events.db"))
    try:
        recorder = EventRecorder(str(tmp_path), database)
        recorder._save_event(
            CrossingEvent(
                event_id=1,
                rank=1,
                track_id=605,
                cross_time=10.0,
                cross_time_str="00:00:10.000",
                cross_realtime="2026-08-06 12:00:10.000",
                bib_number=None,
                bib_confidence=0.0,
                bib_status=BibStatus.UNRECOGNIZED,
                detection_confidence=0.9,
                position=(100, 200),
                bbox=(1, 2, 3, 4),
                participant_id="P000004",
                raw_track_ids=(405, 605),
                sport_profile="cycling-mountain",
                passage_index=1,
            )
        )

        record = database.get_event(1)

        assert record["participant_id"] == "P000004"
        assert record["raw_track_ids"] == [405, 605]
        assert record["sport_profile"] == "cycling-mountain"
        assert record["passage_index"] == 1
    finally:
        database.close()
