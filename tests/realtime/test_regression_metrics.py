from types import SimpleNamespace
from pathlib import Path
from io import BytesIO

import numpy as np
import pytest

from realtime.devtools import regression_report


def test_parse_frame_rate_supports_ffprobe_fraction_values():
    assert regression_report._parse_frame_rate("50/1") == 50.0
    assert regression_report._parse_frame_rate("30000/1001") == pytest.approx(29.97003)
    assert regression_report._parse_frame_rate("0/0") == 0.0
    assert regression_report._parse_frame_rate(25) == 25.0


def test_read_exact_returns_partial_data_only_at_eof():
    assert regression_report._read_exact(BytesIO(b"abcdef"), 4) == b"abcd"
    assert regression_report._read_exact(BytesIO(b"abc"), 5) == b"abc"


def test_extract_line_and_roi_respects_disabled_roi_state():
    line, roi = regression_report._extract_line_and_roi(
        {
            "finish_line": {"x1": 1, "y1": 2, "x2": 3, "y2": 4},
            "roi_points": [[0, 0], [10, 0], [10, 10]],
            "roi_enabled": {"0": False},
        }
    )

    assert line == {"x1": 1, "y1": 2, "x2": 3, "y2": 4}
    assert roi is None


def test_regression_cli_exposes_speed_skating_profile():
    source = Path(regression_report.__file__).read_text(encoding="utf-8")

    assert '"--sport-profile"' in source
    assert '("cycling", "speed_skating")' in source
    assert "build_sport_event_profile(args.sport_profile)" in source
    assert '"-fps_mode"' in source
    assert '"passthrough"' in source
    assert '"nominal_fps"' in source
    assert '"observed_fps"' in source


def test_regression_metrics_reports_detection_layers_and_latency():
    assert hasattr(regression_report, "RegressionMetrics")

    metrics = regression_report.RegressionMetrics()
    metrics.observe_frame(
        {
            "raw_bikes": 3,
            "raw_bibs": 4,
            "validated_tracks": 2,
            "synthetic_tracks": 1,
            "inference_ms": 12.0,
            "postprocess_ms": 8.0,
            "total_ms": 20.0,
        },
        crossing_events=[object(), object()],
        rejected_candidates=[{"reason": "no_bicycle"}],
    )

    report = metrics.to_report()

    assert report["counts"] == {
        "raw_bike_detections": 3,
        "raw_bib_detections": 4,
        "validated_track_observations": 2,
        "synthetic_track_observations": 1,
        "crossing_events": 2,
        "rejected_candidates": 1,
    }
    assert report["latency_ms"]["inference"]["p95"] == 12.0
    assert report["latency_ms"]["postprocess"]["p95"] == 8.0
    assert report["latency_ms"]["total"]["p95"] == 20.0


def test_regression_metrics_reports_identity_ocr_evidence_and_runtime_contract():
    metrics = regression_report.RegressionMetrics()
    event = SimpleNamespace(
        participant_id="P4",
        passage_index=1,
        bib_status="recognized",
        frame=np.ones((4, 4, 3), dtype=np.uint8),
        crop=np.ones((2, 2, 3), dtype=np.uint8),
        bib_crop=np.ones((1, 1, 3), dtype=np.uint8),
    )
    metrics.observe_frame(
        {
            "raw_bikes": 3,
            "raw_bibs": 2,
            "validated_tracks": 2,
            "raw_tracks": 2,
            "participants": 1,
            "track_fragments_merged": 1,
            "identity_ambiguities": 0,
            "queue_depth": 4,
            "dropped_frame_count": 2,
            "inference_ms": 5.0,
            "postprocess_ms": 3.0,
            "total_ms": 8.0,
        },
        crossing_events=[event],
        participant_ids=["P4"],
        raw_track_ids=[405, 605],
    )

    report = metrics.to_report(processing_fps=12.5)

    assert report["frames_processed"] == 1
    assert report["processing_fps"] == 12.5
    assert report["raw_tracks"] == 2
    assert report["stable_participants"] == 1
    assert report["track_fragments_merged"] == 1
    assert report["identity_ambiguities"] == 0
    assert report["crossing_events"] == 1
    assert report["duplicate_passages"] == 0
    assert report["rejected_candidates"] == 0
    assert report["evidence_complete"] == 1
    assert report["ocr_recognized"] == 1
    assert report["ocr_conflicts"] == 0
    assert report["ocr_unrecognized"] == 0
    assert report["queue_depth_max"] == 4
    assert report["dropped_frames"] == 2


def test_competition_manifest_uses_environment_paths_and_validates_identity_cases():
    manifest_path = Path(__file__).parent / "fixtures" / "competition_regression_manifest.json"
    manifest = regression_report.load_competition_manifest(manifest_path)

    assert manifest["schema_version"] == 1
    assert len(manifest["cases"]) == 1
    case = manifest["cases"][0]
    assert case["video_path_env"]
    assert case["model_path_env"]
    assert "video_path" not in case
    assert case["event_profile"]["pipeline"] == "athlete_bib"
    assert case["known_same_participant_tracks"] == [[405, 605]]


def test_competition_manifest_accepts_windows_utf8_bom(tmp_path):
    manifest = {
        "schema_version": 1,
        "cases": [
            {
                "case_id": "bom",
                "video_path_env": "VIDEO",
                "model_path_env": "MODEL",
                "event_profile": {"name": "cycling"},
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(
        b"\xef\xbb\xbf" + __import__("json").dumps(manifest).encode("utf-8")
    )

    assert regression_report.load_competition_manifest(manifest_path)["schema_version"] == 1


def test_competition_manifest_rejects_absolute_video_paths():
    with pytest.raises(ValueError, match="environment"):
        regression_report.validate_competition_manifest(
            {
                "schema_version": 1,
                "cases": [
                    {
                        "case_id": "bad",
                        "video_path": "C:/private/race.mp4",
                        "model_path_env": "MODEL",
                        "event_profile": {"name": "cycling"},
                    }
                ],
            }
        )


def test_case_report_validation_checks_known_participant_groups():
    case = {
        "case_id": "cycling",
        "video_path_env": "VIDEO",
        "model_path_env": "MODEL",
        "event_profile": {"name": "cycling"},
        "known_same_participant_tracks": [[405, 605]],
        "known_distinct_tracks": [[5534, 6543]],
    }
    report = {
        "participant_tracks": {
            "P4": [405, 605],
            "P18": [5534],
            "P19": [6543],
        },
        "crossing_events": 2,
        "evidence_complete": 2,
        "result_signature": "sig",
    }

    validation = regression_report.validate_case_report(case, report)

    assert validation["passed"] is True
    assert validation["issues"] == []


def test_competition_runner_resolves_environment_paths_without_absolute_manifest_paths(tmp_path, monkeypatch):
    from tools.run_competition_regression import resolve_case_paths

    video_path = tmp_path / "race.mp4"
    model_path = tmp_path / "model.pt"
    video_path.write_bytes(b"video")
    model_path.write_bytes(b"model")
    monkeypatch.setenv("VIDEO_INPUT", str(video_path))
    monkeypatch.setenv("MODEL_INPUT", str(model_path))

    paths = resolve_case_paths(
        {
            "video_path_env": "VIDEO_INPUT",
            "model_path_env": "MODEL_INPUT",
        }
    )

    assert paths["video"] == video_path.resolve()
    assert paths["model"] == model_path.resolve()


def test_competition_runner_writes_aggregate_report_with_injected_case_runner(tmp_path):
    from tools.run_competition_regression import run_manifest

    manifest_path = Path(__file__).parent / "fixtures" / "competition_regression_manifest.json"
    video_path = tmp_path / "race.mp4"
    model_path = tmp_path / "model.pt"
    video_path.write_bytes(b"video")
    model_path.write_bytes(b"model")

    def fake_runner(case, paths, report_path):
        assert paths["video"] == video_path.resolve()
        assert paths["model"] == model_path.resolve()
        return {
            "frames_processed": 1,
            "processing_fps": 10.0,
            "raw_tracks": 2,
            "stable_participants": 1,
            "track_fragments_merged": 1,
            "identity_ambiguities": 0,
            "crossing_events": 1,
            "duplicate_passages": 0,
            "rejected_candidates": 0,
            "evidence_complete": 1,
            "ocr_recognized": 1,
            "ocr_conflicts": 0,
            "ocr_unrecognized": 0,
            "queue_depth_max": 0,
            "dropped_frames": 0,
            "participant_tracks": {"P4": [405, 605]},
            "result_signature": "sig",
        }

    manifest = regression_report.load_competition_manifest(manifest_path)
    manifest["cases"][0]["expected_in_clip_crossings"] = 1
    manifest["cases"][0]["known_distinct_tracks"] = []
    manifest["cases"][0]["video_path_env"] = "VIDEO_INPUT"
    manifest["cases"][0]["model_path_env"] = "MODEL_INPUT"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(__import__("json").dumps(manifest), encoding="utf-8")
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("VIDEO_INPUT", str(video_path))
    monkeypatch.setenv("MODEL_INPUT", str(model_path))
    try:
        aggregate = run_manifest(manifest_path, tmp_path / "out", runner=fake_runner)
    finally:
        monkeypatch.undo()

    assert aggregate["validation"]["passed"] is True
    assert (tmp_path / "out" / "aggregate_report.json").exists()


def test_report_output_path_rejects_race_data(tmp_path):
    protected_output = tmp_path / "RaceData" / "regression.json"

    with pytest.raises(ValueError, match="RaceData"):
        regression_report.validate_report_output_path(protected_output)

    safe_output = tmp_path / "reports" / "regression.json"
    assert regression_report.validate_report_output_path(safe_output) == safe_output.resolve()


def test_write_report_creates_parent_directory_atomically(tmp_path):
    output_path = tmp_path / "nested" / "regression.json"

    written = regression_report.write_json_report(output_path, {"frames_processed": 1})

    assert written == output_path.resolve()
    assert output_path.exists()


def test_result_signature_ignores_runtime_only_fields():
    first = {
        "frames_processed": 10,
        "performance": {"elapsed_sec": 1.0, "processing_fps": 10.0},
        "counts": {"crossing_events": 1},
        "events": [
            {
                "event_id": 1,
                "track_id": 7,
                "cross_time": 0.25,
                "cross_realtime": 1000.0,
                "frame_index": 15,
                "media_time": 0.5,
                "bib_number": "123",
                "bib_status": "CONFIRMED",
                "bbox": [1, 2, 3, 4],
                "position": [2, 4],
            }
        ],
    }
    second = {
        **first,
        "performance": {"elapsed_sec": 2.0, "processing_fps": 5.0},
        "events": [
            {
                **first["events"][0],
                "cross_time": 10.25,
                "cross_realtime": 2000.0,
            }
        ],
    }

    assert regression_report.result_signature(first) == regression_report.result_signature(second)


def test_result_signature_uses_rejected_records_without_overloading_count_metric():
    first = {
        "frames_processed": 1,
        "counts": {"rejected_candidates": 1},
        "rejected_candidates": 1,
        "rejected_candidate_records": [
            {"rejection_id": 1, "track_id": 7, "reason": "no_bicycle"}
        ],
    }
    second = {
        **first,
        "rejected_candidate_records": [
            {"rejection_id": 1, "track_id": 7, "reason": "background"}
        ],
    }

    assert regression_report.result_signature(first) != regression_report.result_signature(second)


def test_video_timestamp_uses_media_time_with_frame_index_fallback():
    assert regression_report.video_timestamp_seconds(1500.0, frame_index=3, fps=2.0) == 1.5
    assert regression_report.video_timestamp_seconds(0.0, frame_index=3, fps=2.0) == 1.5


def test_build_event_record_keeps_video_position_for_visual_review():
    event = SimpleNamespace(
        event_id=4,
        track_id=17,
        cross_time=1234.5,
        cross_realtime="2026-08-04 12:00:00.000",
        bib_number=None,
        bib_status="UNRECOGNIZED",
        bbox=(10, 20, 30, 40),
        position=(20, 40),
    )
    state = SimpleNamespace(has_bib_box=True, synthetic_kind="bib_split")

    record = regression_report.build_event_record(
        event,
        state,
        frame_index=90,
        media_time=3.0,
    )

    assert record["frame_index"] == 90
    assert record["media_time"] == 3.0
    assert record["synthetic_kind"] == "bib_split"
    assert record["has_bib_box"] is True


def test_save_event_evidence_uses_captured_event_images(tmp_path):
    event = SimpleNamespace(
        frame=np.full((24, 32, 3), 20, dtype=np.uint8),
        crop=np.full((12, 10, 3), 40, dtype=np.uint8),
        bib_crop=np.full((6, 8, 3), 60, dtype=np.uint8),
    )
    record = {"event_id": 2, "track_id": 17, "frame_index": 90}

    evidence = regression_report.save_event_evidence(event, record, tmp_path / "evidence")

    assert set(evidence) == {"frame", "athlete", "bib"}
    for path in evidence.values():
        assert path.exists()


def test_save_event_evidence_keeps_ranked_bib_candidates(tmp_path):
    event = SimpleNamespace(frame=None, crop=None, bib_crop=None)
    candidate = np.full((18, 24, 3), 80, dtype=np.uint8)
    state = SimpleNamespace(
        bib_crops_cache=[(0.8, candidate, None, [1, 2, 3, 4])],
        fallback_bib_crops_cache=[(0.6, candidate, None, [5, 6, 7, 8])],
    )
    record = {"event_id": 3, "track_id": 21, "frame_index": 120}

    evidence = regression_report.save_event_evidence(
        event,
        record,
        tmp_path / "evidence",
        state=state,
    )

    assert evidence["bib_candidate_01"].exists()
    assert evidence["fallback_candidate_01"].exists()


def test_build_rejected_candidate_record_keeps_audit_fields():
    candidate = {
        "track_id": 31,
        "reason": "no_bicycle",
        "bbox": [10, 20, 30, 40],
        "position": [20, 40],
        "has_bib_box": True,
    }

    record = regression_report.build_rejected_candidate_record(
        candidate,
        rejection_id=4,
        frame_index=90,
        media_time=3.0,
    )

    assert record == {
        "rejection_id": 4,
        "track_id": 31,
        "reason": "no_bicycle",
        "frame_index": 90,
        "media_time": 3.0,
        "has_bib_box": True,
        "bbox": [10, 20, 30, 40],
        "position": [20, 40],
    }


def test_save_rejected_candidate_evidence_is_separate_from_events(tmp_path):
    candidate = {
        "frame": np.full((24, 32, 3), 20, dtype=np.uint8),
        "crop": np.full((12, 10, 3), 40, dtype=np.uint8),
        "bib_crop": np.full((6, 8, 3), 60, dtype=np.uint8),
    }
    record = {"rejection_id": 2, "track_id": 31, "frame_index": 90}

    evidence = regression_report.save_rejected_candidate_evidence(
        candidate,
        record,
        tmp_path / "evidence",
    )

    assert set(evidence) == {"frame", "athlete", "bib"}
    assert all(path.parent.name == "rejected" for path in evidence.values())
