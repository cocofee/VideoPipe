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


class _RoiRejectedBoxes:
    def __init__(self):
        self.cls = np.zeros(4, dtype=np.float32)
        self.conf = np.full(4, 0.9, dtype=np.float32)
        self.xyxy = np.array(
            [
                [200, 200, 260, 360],
                [300, 200, 360, 360],
                [400, 200, 460, 360],
                [500, 200, 560, 360],
            ],
            dtype=np.float32,
        )
        self.id = np.arange(1, 5, dtype=np.float32)

    def __len__(self):
        return len(self.cls)


class _RoiRejectedResult:
    boxes = _RoiRejectedBoxes()
    names = {0: "person"}


class _RoiRejectedModel:
    names = {0: "person"}

    def track(self, frame, **kwargs):
        return [_RoiRejectedResult()]


class _OffFinishSegmentBoxes:
    def __init__(self):
        self.cls = np.array([0], dtype=np.float32)
        self.conf = np.array([0.9], dtype=np.float32)
        self.xyxy = np.array([[500, 500, 700, 800]], dtype=np.float32)
        self.id = np.array([1], dtype=np.float32)

    def __len__(self):
        return len(self.cls)


class _OffFinishSegmentResult:
    boxes = _OffFinishSegmentBoxes()
    names = {0: "person"}


class _OffFinishSegmentModel:
    names = {0: "person"}

    def track(self, frame, **kwargs):
        return [_OffFinishSegmentResult()]


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
        assert metrics["roi_auto_disabled"] is False
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
        assert metrics["roi_auto_disabled"] is False
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
        assert metrics["roi_auto_disabled"] is False
    finally:
        detector.stop()


def test_detector_reports_when_roi_overfilter_protection_disables_roi():
    detector = Detector(model_path="fake.pt", model=_RoiRejectedModel(), ocr=None)
    detector.enable_finish_segment_filter = False
    detector.enable_static_background_filter = False
    detector.set_roi_polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    detector._roi_overfilter_streak = 4

    try:
        _, athletes, _ = detector.process_frame(
            np.zeros((640, 640, 3), dtype=np.uint8),
            timestamp=0.0,
        )

        assert athletes == []
        assert detector.enable_roi_filter is False
        assert detector.last_frame_metrics["roi_auto_disabled"] is True
    finally:
        detector.stop()


def test_model_class_mapping_clears_default_bib_for_coco_names():
    detector = Detector(model_path="fake.pt", model=None, ocr=None)

    try:
        assert detector._map_model_class_ids(
            {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle"}
        ) is True
        assert detector.person_class_ids == {0}
        assert detector.bib_class_ids == set()
        assert detector.bike_model_mode is False
    finally:
        detector.stop()


def test_static_filter_cooldown_does_not_rearm_while_active():
    detector = Detector(model_path="fake.pt", model=_RoiRejectedModel(), ocr=None)
    detector.enable_finish_segment_filter = False
    detector.set_roi_polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    detector._static_filter_cooldown_frames = 10
    detector._under_detect_streak = 9

    try:
        detector.process_frame(
            np.zeros((640, 640, 3), dtype=np.uint8),
            timestamp=0.0,
        )

        assert detector._static_filter_cooldown_frames == 9
        assert detector._under_detect_streak == 0
    finally:
        detector.stop()


def test_finish_segment_filter_does_not_hide_detected_athletes():
    detector = Detector(
        model_path="fake.pt",
        model=_OffFinishSegmentModel(),
        ocr=None,
        sport_profile="speed_skating",
    )
    detector.enable_static_background_filter = False
    detector.set_finish_line((1280, 1439), (1355, 800))

    try:
        _, athletes, _ = detector.process_frame(
            np.zeros((1440, 2560, 3), dtype=np.uint8),
            timestamp=0.0,
        )

        assert len(athletes) == 1
        assert athletes[0]["track_id"] == 1
    finally:
        detector.stop()
