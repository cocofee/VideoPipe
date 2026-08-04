import numpy as np
from types import SimpleNamespace

from realtime.detector import (
    Detector,
    TrackState,
    resolve_athlete_validator_model,
    resolve_performance_profile,
)


def _athlete(track_id, bbox):
    x1, y1, x2, y2 = bbox
    return {
        "bbox": list(bbox),
        "conf": 0.9,
        "track_id": track_id,
        "center_x": (x1 + x2) // 2,
        "center_y": (y1 + y2) // 2,
        "bottom_y": y2,
    }


def _bib(bbox, conf=0.9):
    x1, y1, x2, y2 = bbox
    return {
        "bbox": list(bbox),
        "conf": conf,
        "center_x": (x1 + x2) // 2,
        "center_y": (y1 + y2) // 2,
    }


class _ValidatorModel:
    def __init__(self, class_names):
        self.class_names = class_names
        self.calls = 0

    def predict(self, source, **kwargs):
        self.calls += 1
        names = {0: "person", 1: "bicycle"}
        classes = [1 if name == "bicycle" else 0 for name in self.class_names]
        boxes = SimpleNamespace(cls=np.asarray(classes, dtype=np.float32))
        return [SimpleNamespace(boxes=boxes, names=names)]


class _EmptyTrackingModel:
    names = {0: "bike", 1: "BIB"}

    def track(self, frame, **kwargs):
        return []


class _Boxes:
    def __init__(self):
        self.cls = np.asarray([0, 1], dtype=np.float32)
        self.conf = np.asarray([0.95, 0.90], dtype=np.float32)
        self.xyxy = np.asarray(
            [[600, 350, 1000, 1000], [720, 550, 840, 630]],
            dtype=np.float32,
        )
        self.id = np.asarray([11, 12], dtype=np.float32)

    def __len__(self):
        return len(self.cls)


class _OriginalFrameTrackingModel:
    names = {0: "bike", 1: "BIB"}

    def __init__(self):
        self.source = None
        self.kwargs = None

    def track(self, frame, **kwargs):
        self.source = frame
        self.kwargs = kwargs
        return [SimpleNamespace(boxes=_Boxes(), names=self.names)]


def _pending_crossing_state(frame, track_id):
    state = TrackState(prev_x=100, prev_y=200)
    state.crossed = True
    state.crossed_time = 0.0
    state.pending_event_data = {
        "cross_time": 1_700_000_000.0 + track_id,
        "bbox": [100, 120, 180, 280],
        "conf": 0.9,
        "position": (140, 280),
        "frame": frame,
    }
    state.best_athlete_crop = frame[120:280, 100:180].copy()
    return state


def test_bib_center_outside_athlete_is_not_assigned():
    detector = Detector(model_path="fake.pt", model=None, ocr=None)
    athletes = [_athlete(1, [100, 100, 220, 320])]
    bibs = [_bib([218, 170, 238, 200])]

    assignments, unmatched = detector._assign_bibs_to_athletes(athletes, bibs)

    assert assignments == {}
    assert unmatched == {0}


def test_each_bib_is_assigned_to_only_one_best_athlete():
    detector = Detector(model_path="fake.pt", model=None, ocr=None)
    athletes = [
        _athlete(11, [100, 100, 260, 360]),
        _athlete(22, [180, 100, 340, 360]),
    ]
    bibs = [_bib([210, 190, 250, 235])]

    assignments, unmatched = detector._assign_bibs_to_athletes(athletes, bibs)

    assigned_indices = [bib_index for indices in assignments.values() for bib_index in indices]
    assert assigned_indices == [0]
    assert list(assignments) == [1]
    assert unmatched == set()


def test_unmatched_bib_does_not_create_a_synthetic_athlete():
    detector = Detector(model_path="fake.pt", model=None, ocr=None)
    frame = np.zeros((640, 640, 3), dtype=np.uint8)

    athletes = detector._ensure_athletes_from_bibs(
        athletes=[],
        bibs=[_bib([280, 300, 340, 340])],
        frame_original=frame,
        current_time=1.0,
    )

    assert athletes == []


def test_wide_box_is_not_split_without_multiple_bib_assignments():
    detector = Detector(model_path="fake.pt", model=None, ocr=None)
    detector.enable_gate_guard = False
    athlete = _athlete(7, [120, 180, 460, 500])

    result = detector._rescue_split_wide_boxes(
        athletes=[athlete],
        bibs=[],
        tid_bib_centers={},
        frame_shape=(640, 640),
        current_time=1.0,
    )

    assert result == [athlete]


def test_only_evidence_backed_tracks_can_emit_crossing_events():
    detector = Detector(model_path="fake.pt", model=None, ocr=None)

    assert detector._is_event_eligible_athlete({"synthetic_kind": "bib_only"}, observation_count=10) is False
    assert detector._is_event_eligible_athlete({"synthetic_kind": "wide_split"}, observation_count=10) is False
    assert detector._is_event_eligible_athlete({"synthetic_kind": "bib_split"}, observation_count=2) is False
    assert detector._is_event_eligible_athlete({"synthetic_kind": "bib_split"}, observation_count=3) is True
    assert detector._is_event_eligible_athlete({}, observation_count=1) is True


def test_nested_unknown_crossing_boxes_are_event_duplicates():
    detector = Detector(model_path="fake.pt", model=None, ocr=None)
    first = [813, 594, 947, 813]
    duplicate = [808, 477, 953, 816]
    adjacent_rider = [705, 543, 886, 821]

    assert detector._is_duplicate_unknown_event_bbox(duplicate, first, time_diff=0.0) is True
    assert detector._is_duplicate_unknown_event_bbox(adjacent_rider, first, time_diff=0.2) is False


def test_overlapping_boxes_with_distinct_bibs_are_not_deduplicated():
    detector = Detector(model_path="fake.pt", model=None, ocr=None)
    athletes = [
        _athlete(11, [477, 404, 742, 882]),
        _athlete(22, [457, 390, 719, 876]),
    ]
    bib_centers = {
        11: [(540, 600, [520, 580, 560, 620])],
        22: [(650, 600, [630, 580, 670, 620])],
    }

    result = detector._iou_dedup_athletes(athletes, tid_bib_centers=bib_centers)

    assert [athlete["track_id"] for athlete in result] == [11, 22]


def test_unknown_event_is_rejected_when_validator_finds_no_bicycle():
    validator = _ValidatorModel(["person"])
    detector = Detector(model_path="fake.pt", model=None, ocr=None, athlete_validator=validator)
    state = TrackState(prev_x=0, prev_y=0)
    crop = np.zeros((160, 100, 3), dtype=np.uint8)

    assert detector._passes_athlete_event_validation(state, crop) is False
    assert validator.calls == 1


def test_bib_evidence_skips_secondary_athlete_validation():
    validator = _ValidatorModel(["person"])
    detector = Detector(model_path="fake.pt", model=None, ocr=None, athlete_validator=validator)
    state = TrackState(prev_x=0, prev_y=0, has_bib_box=True)
    crop = np.zeros((160, 100, 3), dtype=np.uint8)

    assert detector._passes_athlete_event_validation(state, crop) is True
    assert validator.calls == 0


def test_missing_secondary_validator_fails_open():
    detector = Detector(model_path="fake.pt", model=None, ocr=None, athlete_validator=None)
    state = TrackState(prev_x=0, prev_y=0)
    crop = np.zeros((160, 100, 3), dtype=np.uint8)

    assert detector._passes_athlete_event_validation(state, crop) is True


def test_rejected_candidate_does_not_consume_event_id():
    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    detector = Detector(
        model_path="fake.pt",
        model=_EmptyTrackingModel(),
        ocr=None,
        athlete_validator=_ValidatorModel(["person"]),
    )
    detector.enable_static_background_filter = False
    detector.enable_finish_segment_filter = False
    detector._track_states[101] = _pending_crossing_state(frame, 101)

    rejected_events, _, _ = detector.process_frame(frame, timestamp=1.0)

    assert rejected_events == []
    assert detector._event_id == 0

    detector._athlete_validator = _ValidatorModel(["bicycle"])
    detector._track_states[202] = _pending_crossing_state(frame, 202)

    accepted_events, _, _ = detector.process_frame(frame, timestamp=2.0)

    assert [event.event_id for event in accepted_events] == [1]


def test_auto_performance_profile_uses_laptop_settings_without_cuda():
    profile = resolve_performance_profile("auto", cuda_available=False)

    assert profile["name"] == "laptop"
    assert profile["process_imgsz"] == 640
    assert profile["adaptive_frame_skip"] is True


def test_laptop_profile_keeps_ocr_crop_on_original_frame():
    model = _OriginalFrameTrackingModel()
    detector = Detector(
        model_path="fake.pt",
        model=model,
        ocr=None,
        performance_profile="laptop",
    )
    detector.enable_static_background_filter = False
    detector.enable_finish_segment_filter = False
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[545:635, 715:845] = (17, 83, 201)

    detector.process_frame(frame, timestamp=0.0)

    assert model.source is frame
    assert model.kwargs["imgsz"] == 640
    state = detector._track_states[11]
    _, crop, frame_ref, bbox = state.bib_crops_cache[0]
    assert bbox == [720, 550, 840, 630]
    assert frame_ref is frame
    assert np.array_equal(crop, frame[545:635, 715:845])


def test_resolve_athlete_validator_prefers_config_then_default(tmp_path):
    configured = tmp_path / "configured.pt"
    configured.write_bytes(b"configured")
    fallback_root = tmp_path / "fallback"
    fallback_root.mkdir()
    fallback = fallback_root / "yolov8s.pt"
    fallback.write_bytes(b"fallback")

    assert resolve_athlete_validator_model(str(configured), [fallback_root]) == configured.resolve()
    assert resolve_athlete_validator_model("", [fallback_root]) == fallback.resolve()


def test_small_high_contrast_bib_crop_is_penalized():
    detector = Detector(model_path="fake.pt", model=None, ocr=None)
    checkerboard = (np.indices((23, 28)).sum(axis=0) % 2 * 255).astype(np.uint8)
    crop = np.repeat(checkerboard[:, :, None], 3, axis=2)

    assert detector._calculate_image_quality(crop) < 0.55


def test_detected_bib_candidates_take_priority_over_fallback_crops():
    detector = Detector(model_path="fake.pt", model=None, ocr=None)
    real_crop = np.zeros((30, 40, 3), dtype=np.uint8)
    fallback_crop = np.zeros((160, 100, 3), dtype=np.uint8)
    state = TrackState(prev_x=0, prev_y=0)
    state.bib_crops_cache = [(0.50, real_crop, None, [1, 2, 3, 4])]
    state.fallback_bib_crops_cache = [(0.95, fallback_crop, None, [5, 6, 7, 8])]

    candidates = detector._get_ocr_candidates(state)

    assert candidates == state.bib_crops_cache
