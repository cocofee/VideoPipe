import numpy as np

from realtime.detector import Detector


IDENTITY_METRIC_KEYS = (
    "raw_tracks",
    "participants",
    "track_fragments_merged",
    "identity_ambiguities",
)


class _Boxes:
    def __init__(self):
        self.cls = np.array([0, 1], dtype=np.float32)
        self.conf = np.array([0.9, 0.8], dtype=np.float32)
        self.xyxy = np.array(
            [[220, 180, 380, 560], [270, 260, 330, 320]],
            dtype=np.float32,
        )
        self.id = np.array([11, 12], dtype=np.float32)

    def __len__(self):
        return len(self.cls)


class _Result:
    boxes = _Boxes()
    names = {0: "bike", 1: "BIB"}


class _Model:
    names = {0: "bike", 1: "BIB"}

    def track(self, frame, **kwargs):
        return [_Result()]


def test_detector_exposes_raw_validated_synthetic_and_latency_metrics():
    detector = Detector(model_path="fake.pt", model=_Model(), ocr=None)
    detector.enable_static_background_filter = False
    detector.enable_finish_segment_filter = False

    try:
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        _, athletes, _ = detector.process_frame(frame, timestamp=0.0)

        assert hasattr(detector, "last_frame_metrics")
        metrics = detector.last_frame_metrics
        assert metrics["raw_bikes"] == 1
        assert metrics["raw_bibs"] == 1
        assert metrics["validated_tracks"] == len(athletes)
        assert all(
            {"participant_id", "raw_track_ids", "identity_status"}
            <= athlete.keys()
            for athlete in athletes
        )
        for key in IDENTITY_METRIC_KEYS:
            assert isinstance(metrics[key], int)
        expected_raw_tracks = {
            athlete["track_id"]
            for athlete in athletes
            if athlete.get("track_id") is not None and athlete["track_id"] >= 0
        }
        assert metrics["raw_tracks"] == len(expected_raw_tracks)
        assert metrics["participants"] == len(athletes)
        assert metrics["track_fragments_merged"] == 0
        assert metrics["identity_ambiguities"] == 0
        assert metrics["synthetic_tracks"] == sum(
            1
            for athlete in athletes
            if athlete.get("split_from_track") is not None or athlete.get("bib_driven")
        )
        assert metrics["inference_ms"] >= 0.0
        assert metrics["postprocess_ms"] >= 0.0
        assert metrics["total_ms"] >= metrics["inference_ms"]
    finally:
        detector.stop()


def test_detector_metrics_include_identity_fields_when_model_is_missing():
    detector = Detector(model_path="fake.pt", model=None, ocr=None)
    try:
        detector.process_frame(np.zeros((64, 64, 3), dtype=np.uint8), timestamp=0.0)

        metrics = detector.last_frame_metrics
        for key in IDENTITY_METRIC_KEYS:
            assert isinstance(metrics[key], int)
            assert metrics[key] == 0
    finally:
        detector.stop()


def test_detector_metrics_include_identity_fields_on_adaptive_skip():
    detector = Detector(model_path="fake.pt", model=_Model(), ocr=None)
    detector.adaptive_frame_skip = True
    detector.frame_skip = 2
    try:
        detector.process_frame(np.zeros((640, 640, 3), dtype=np.uint8), timestamp=0.0)

        metrics = detector.last_frame_metrics
        for key in IDENTITY_METRIC_KEYS:
            assert isinstance(metrics[key], int)
            assert metrics[key] == 0
    finally:
        detector.stop()
