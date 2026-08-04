import pytest

from realtime.devtools import regression_report


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
    )

    report = metrics.to_report()

    assert report["counts"] == {
        "raw_bike_detections": 3,
        "raw_bib_detections": 4,
        "validated_track_observations": 2,
        "synthetic_track_observations": 1,
        "crossing_events": 2,
    }
    assert report["latency_ms"]["inference"]["p95"] == 12.0
    assert report["latency_ms"]["postprocess"]["p95"] == 8.0
    assert report["latency_ms"]["total"]["p95"] == 20.0


def test_report_output_path_rejects_race_data(tmp_path):
    protected_output = tmp_path / "RaceData" / "regression.json"

    with pytest.raises(ValueError, match="RaceData"):
        regression_report.validate_report_output_path(protected_output)

    safe_output = tmp_path / "reports" / "regression.json"
    assert regression_report.validate_report_output_path(safe_output) == safe_output.resolve()


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
        "events": [{**first["events"][0], "cross_realtime": 2000.0}],
    }

    assert regression_report.result_signature(first) == regression_report.result_signature(second)


def test_video_timestamp_uses_media_time_with_frame_index_fallback():
    assert regression_report.video_timestamp_seconds(1500.0, frame_index=3, fps=2.0) == 1.5
    assert regression_report.video_timestamp_seconds(0.0, frame_index=3, fps=2.0) == 1.5
