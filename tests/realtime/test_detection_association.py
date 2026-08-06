import numpy as np
import pytest
from datetime import datetime
from types import SimpleNamespace

from realtime.detector import (
    Detector,
    LOCAL_VIDEO_EVENT_SETTLE_SECONDS,
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


class _SequencedValidatorModel:
    def __init__(self, predictions):
        self.predictions = list(predictions)
        self.calls = 0

    def predict(self, source, **kwargs):
        prediction = self.predictions[min(self.calls, len(self.predictions) - 1)]
        self.calls += 1
        names = {0: "person", 1: "bicycle"}
        classes = [1 if name == "bicycle" else 0 for name in prediction]
        boxes = SimpleNamespace(cls=np.asarray(classes, dtype=np.float32))
        return [SimpleNamespace(boxes=boxes, names=names)]


class _FailingValidatorModel:
    def predict(self, source, **kwargs):
        raise RuntimeError("validator unavailable")


class _SpatialValidatorModel:
    def __init__(self, bicycle_bbox):
        self.bicycle_bbox = bicycle_bbox

    def predict(self, source, **kwargs):
        boxes = SimpleNamespace(
            cls=np.asarray([1], dtype=np.float32),
            conf=np.asarray([0.9], dtype=np.float32),
            xyxy=np.asarray([self.bicycle_bbox], dtype=np.float32),
        )
        return [SimpleNamespace(boxes=boxes, names={0: "person", 1: "bicycle"})]


class _SequencedSpatialValidatorModel:
    def __init__(self, bicycle_bboxes):
        self.bicycle_bboxes = list(bicycle_bboxes)
        self.calls = 0

    def predict(self, source, **kwargs):
        bicycle_bbox = self.bicycle_bboxes[min(self.calls, len(self.bicycle_bboxes) - 1)]
        self.calls += 1
        boxes = SimpleNamespace(
            cls=np.asarray([1], dtype=np.float32),
            conf=np.asarray([0.9], dtype=np.float32),
            xyxy=np.asarray([bicycle_bbox], dtype=np.float32),
        )
        return [SimpleNamespace(boxes=boxes, names={0: "person", 1: "bicycle"})]


class _CycleAmbiguityValidatorModel:
    def __init__(self):
        self.calls = 0

    def predict(self, source, **kwargs):
        self.calls += 1
        boxes = SimpleNamespace(
            cls=np.asarray([1, 2], dtype=np.float32),
            conf=np.asarray([0.9, 0.8], dtype=np.float32),
            xyxy=np.asarray(
                [
                    [30, 40, 70, 95],
                    [28, 38, 72, 96],
                ],
                dtype=np.float32,
            ),
        )
        return [
            SimpleNamespace(
                boxes=boxes,
                names={0: "person", 1: "bicycle", 2: "motorcycle"},
            )
        ]


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


class _SimpleBoxes:
    def __init__(self, cls, conf, xyxy, ids=None):
        self.cls = np.asarray(cls, dtype=np.float32)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.xyxy = np.asarray(xyxy, dtype=np.float32)
        self.id = None if ids is None else np.asarray(ids, dtype=np.float32)

    def __len__(self):
        return len(self.cls)


class _RawBibCaptureTrackingModel:
    names = {0: "bike", 1: "BIB"}

    def __init__(self):
        self.postprocess_callbacks = []

    def add_callback(self, event, callback):
        if event == "on_predict_postprocess_end":
            self.postprocess_callbacks.append(callback)

    def track(self, frame, **kwargs):
        raw_result = SimpleNamespace(
            boxes=_SimpleBoxes(
                cls=[0, 1],
                conf=[0.95, 0.90],
                xyxy=[[80, 100, 220, 300], [120, 140, 160, 170]],
            ),
            names=self.names,
        )
        predictor = SimpleNamespace(results=[raw_result])
        for callback in self.postprocess_callbacks:
            callback(predictor)

        tracked_result = SimpleNamespace(
            boxes=_SimpleBoxes(
                cls=[0],
                conf=[0.95],
                xyxy=[[80, 100, 220, 300]],
                ids=[11],
            ),
            names=self.names,
        )
        return [tracked_result]


class _LowConfidenceUntrackedModel:
    names = {0: "bike", 1: "BIB"}

    def __init__(self, *, track_id=None):
        self.track_id = track_id

    def track(self, frame, **kwargs):
        return [
            SimpleNamespace(
                boxes=_SimpleBoxes(
                    cls=[0],
                    conf=[0.15],
                    xyxy=[[600, 350, 1000, 1000]],
                    ids=None if self.track_id is None else [self.track_id],
                ),
                names=self.names,
            )
        ]


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


def test_low_confidence_untracked_candidate_is_not_emitted_as_athlete():
    detector = Detector(
        model_path="fake.pt",
        model=_LowConfidenceUntrackedModel(),
        ocr=None,
    )
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    _, athletes, _ = detector.process_frame(frame, timestamp=1.0)

    assert athletes == []


def test_low_confidence_tracked_candidate_remains_available_for_recall():
    detector = Detector(
        model_path="fake.pt",
        model=_LowConfidenceUntrackedModel(track_id=11),
        ocr=None,
    )
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    _, athletes, _ = detector.process_frame(frame, timestamp=1.0)

    assert [athlete["track_id"] for athlete in athletes] == [11]


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


def test_detected_bib_evidence_prevents_unknown_nested_event_deduplication():
    detector = Detector(model_path="fake.pt", model=None, ocr=None)
    previous = [687, 289, 948, 626]
    current = [721, 232, 992, 619]

    assert detector._is_duplicate_unknown_event_bbox(
        current,
        previous,
        time_diff=1.19,
        current_bib_evidence_kind="detected",
        previous_bib_evidence_kind="fallback",
    ) is False


def test_slow_nested_unknown_track_is_deduplicated_without_merging_real_riders():
    detector = Detector(model_path="fake.pt", model=None, ocr=None)
    first = [894, 483, 973, 610]
    slow_duplicate = [873, 484, 980, 618]
    hevc_duplicate = [908, 489, 975, 613]
    hevc_first = [887, 483, 968, 610]
    first_real_rider = [805, 501, 895, 645]
    second_real_rider = [824, 508, 942, 680]

    assert detector._is_duplicate_unknown_event_bbox(slow_duplicate, first, time_diff=0.87) is True
    assert detector._is_duplicate_unknown_event_bbox(hevc_duplicate, hevc_first, time_diff=0.67) is True
    assert detector._is_duplicate_unknown_event_bbox(second_real_rider, first_real_rider, time_diff=0.0) is False


def test_scale_changed_unknown_track_from_long_video_is_deduplicated():
    detector = Detector(model_path="fake.pt", model=None, ocr=None)
    first_event_bbox = [809, 594, 948, 820]
    fragmented_track_bbox = [821, 603, 932, 840]
    adjacent_rider_bbox = [777, 626, 892, 820]

    assert detector._is_duplicate_unknown_event_bbox(
        fragmented_track_bbox,
        first_event_bbox,
        time_diff=0.4,
        current_bib_evidence_kind="detected",
        previous_bib_evidence_kind="detected",
    ) is True
    assert detector._is_duplicate_unknown_event_bbox(
        adjacent_rider_bbox,
        first_event_bbox,
        time_diff=0.4,
        current_bib_evidence_kind="detected",
        previous_bib_evidence_kind="detected",
    ) is False


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


def test_ambiguous_bib_is_not_assigned_to_an_arbitrary_athlete():
    detector = Detector(model_path="fake.pt", model=None, ocr=None)
    athletes = [
        _athlete(11, [0, 0, 100, 200]),
        _athlete(22, [10, 0, 110, 200]),
    ]
    bibs = [{
        "bbox": [45, 60, 65, 90],
        "center_x": 55,
        "center_y": 75,
        "conf": 0.9,
    }]

    assignments, unmatched = detector._assign_bibs_to_athletes(athletes, bibs)

    assert assignments == {}
    assert unmatched == {0}


def test_clear_bib_owner_is_still_assigned():
    detector = Detector(model_path="fake.pt", model=None, ocr=None)
    athletes = [
        _athlete(11, [0, 0, 100, 200]),
        _athlete(22, [120, 0, 220, 200]),
    ]
    bibs = [{
        "bbox": [35, 60, 55, 90],
        "center_x": 45,
        "center_y": 75,
        "conf": 0.9,
    }]

    assignments, unmatched = detector._assign_bibs_to_athletes(athletes, bibs)

    assert assignments == {0: [0]}
    assert unmatched == set()


def test_unknown_event_is_rejected_when_validator_finds_no_bicycle():
    validator = _ValidatorModel(["person"])
    detector = Detector(model_path="fake.pt", model=None, ocr=None, athlete_validator=validator)
    state = TrackState(prev_x=0, prev_y=0)
    crop = np.zeros((160, 100, 3), dtype=np.uint8)

    assert detector._passes_athlete_event_validation(state, crop) is False
    assert validator.calls == 1


def test_default_sport_profile_is_cycling():
    detector = Detector(model_path="fake.pt", model=None, ocr=None)

    assert detector.sport_profile == "cycling"


def test_running_profile_is_not_silently_treated_as_cycling():
    with pytest.raises(ValueError, match="running"):
        Detector(model_path="fake.pt", model=None, ocr=None, sport_profile="running")


def test_bib_evidence_requires_secondary_athlete_validation_in_cycling_profile():
    validator = _ValidatorModel(["person"])
    detector = Detector(model_path="fake.pt", model=None, ocr=None, athlete_validator=validator)
    state = TrackState(prev_x=0, prev_y=0, has_bib_box=True)
    crop = np.zeros((160, 100, 3), dtype=np.uint8)

    assert detector._passes_athlete_event_validation(state, crop, [crop]) is False
    assert validator.calls > 0


def test_bibbed_cyclist_requires_bicycle_in_athlete_and_context_evidence():
    validator = _SequencedValidatorModel([["bicycle"], ["bicycle"]])
    detector = Detector(model_path="fake.pt", model=None, ocr=None, athlete_validator=validator)
    state = TrackState(prev_x=0, prev_y=0, has_bib_box=True)
    crop = np.zeros((160, 100, 3), dtype=np.uint8)
    context = np.zeros((220, 160, 3), dtype=np.uint8)

    assert detector._passes_athlete_event_validation(state, crop, [context]) is True


def test_centered_tight_crop_bicycle_still_needs_context_confirmation():
    validator = _SequencedValidatorModel([["bicycle"], ["person"]])
    detector = Detector(model_path="fake.pt", model=None, ocr=None, athlete_validator=validator)
    state = TrackState(prev_x=0, prev_y=0, has_bib_box=True)
    crop = np.zeros((160, 100, 3), dtype=np.uint8)
    context = np.zeros((220, 160, 3), dtype=np.uint8)

    assert detector._passes_athlete_event_validation(state, crop, [context]) is False
    assert validator.calls == 2


def test_overlapping_bicycle_and_motorcycle_evidence_handles_distant_cyclist():
    validator = _CycleAmbiguityValidatorModel()
    detector = Detector(model_path="fake.pt", model=None, ocr=None, athlete_validator=validator)
    state = TrackState(prev_x=0, prev_y=0, has_bib_box=False)
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    context = np.zeros((100, 100, 3), dtype=np.uint8)

    assert detector._passes_athlete_event_validation(state, crop, [context]) is True
    assert validator.calls == 1


def test_context_only_bicycle_evidence_is_not_enough():
    validator = _SequencedValidatorModel([["person"], ["bicycle"]])
    detector = Detector(model_path="fake.pt", model=None, ocr=None, athlete_validator=validator)
    state = TrackState(prev_x=0, prev_y=0, has_bib_box=True)
    crop = np.zeros((160, 100, 3), dtype=np.uint8)
    context = np.zeros((220, 160, 3), dtype=np.uint8)

    assert detector._passes_athlete_event_validation(state, crop, [context]) is False


def test_edge_bicycle_requires_centered_context_evidence():
    validator = _SequencedSpatialValidatorModel([
        [0, 40, 30, 95],
        [0, 40, 30, 95],
    ])
    detector = Detector(model_path="fake.pt", model=None, ocr=None, athlete_validator=validator)
    state = TrackState(prev_x=0, prev_y=0, has_bib_box=True)
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    context = np.zeros((100, 100, 3), dtype=np.uint8)

    assert detector._passes_athlete_event_validation(state, crop, [context]) is False


def test_edge_bicycle_passes_with_centered_context_evidence():
    validator = _SequencedSpatialValidatorModel([
        [0, 40, 30, 95],
        [25, 40, 75, 95],
    ])
    detector = Detector(model_path="fake.pt", model=None, ocr=None, athlete_validator=validator)
    state = TrackState(prev_x=0, prev_y=0, has_bib_box=True)
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    context = np.zeros((100, 100, 3), dtype=np.uint8)

    assert detector._passes_athlete_event_validation(state, crop, [context]) is True


def test_frame_edge_candidate_can_use_significant_edge_bicycle_evidence():
    validator = _SpatialValidatorModel([0, 40, 30, 95])
    detector = Detector(model_path="fake.pt", model=None, ocr=None, athlete_validator=validator)
    state = TrackState(prev_x=0, prev_y=0, has_bib_box=True)
    crop = np.zeros((100, 100, 3), dtype=np.uint8)

    assert detector._passes_athlete_event_validation(
        state,
        crop,
        [crop],
        allow_edge_bicycle=True,
    ) is True


def test_missing_secondary_validator_fails_open():
    detector = Detector(model_path="fake.pt", model=None, ocr=None, athlete_validator=None)
    state = TrackState(prev_x=0, prev_y=0)
    crop = np.zeros((160, 100, 3), dtype=np.uint8)

    assert detector._passes_athlete_event_validation(state, crop) is True


def test_failed_secondary_validator_fails_open():
    detector = Detector(
        model_path="fake.pt",
        model=None,
        ocr=None,
        athlete_validator=_FailingValidatorModel(),
    )
    state = TrackState(prev_x=0, prev_y=0, has_bib_box=True)
    crop = np.zeros((160, 100, 3), dtype=np.uint8)

    assert detector._passes_athlete_event_validation(state, crop, [crop]) is True


def test_tiny_bicycle_box_does_not_validate_a_cycling_event():
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    detector = Detector(
        model_path="fake.pt",
        model=None,
        ocr=None,
        athlete_validator=_SpatialValidatorModel([80, 40, 90, 70]),
    )

    assert detector._verify_bicycle_in_crop(crop) is False


def test_substantial_bicycle_box_validates_a_cycling_event():
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    detector = Detector(
        model_path="fake.pt",
        model=None,
        ocr=None,
        athlete_validator=_SpatialValidatorModel([20, 40, 80, 95]),
    )

    assert detector._verify_bicycle_in_crop(crop) is True


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
    assert len(detector.last_rejected_candidates) == 1
    assert detector.last_rejected_candidates[0]["reason"] == "no_bicycle"
    assert detector.last_rejected_candidates[0]["track_id"] == 101

    detector._athlete_validator = _ValidatorModel(["bicycle"])
    detector._track_states[202] = _pending_crossing_state(frame, 202)

    accepted_events, _, _ = detector.process_frame(frame, timestamp=2.0)

    assert [event.event_id for event in accepted_events] == [1]


def test_rejected_bib_candidate_does_not_consume_event_id():
    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    detector = Detector(
        model_path="fake.pt",
        model=_EmptyTrackingModel(),
        ocr=None,
        athlete_validator=_ValidatorModel(["person"]),
    )
    detector.enable_static_background_filter = False
    detector.enable_finish_segment_filter = False
    state = _pending_crossing_state(frame, 101)
    state.has_bib_box = True
    detector._track_states[101] = state

    rejected_events, _, _ = detector.process_frame(frame, timestamp=1.0)

    assert rejected_events == []
    assert detector._event_id == 0


def test_auto_performance_profile_preserves_small_target_resolution_without_cuda():
    profile = resolve_performance_profile("auto", cuda_available=False)

    assert profile["name"] == "laptop"
    assert profile["process_imgsz"] == 896
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
    assert model.kwargs["imgsz"] == 896
    state = detector._track_states[11]
    _, crop, frame_ref, bbox = state.bib_crops_cache[0]
    assert bbox == [720, 550, 840, 630]
    assert frame_ref is frame
    assert np.array_equal(crop, frame[545:635, 715:845])


def _crossing_detector_for_timestamp_test():
    detector = Detector(
        model_path="fake.pt",
        model=_OriginalFrameTrackingModel(),
        ocr=None,
        athlete_validator=_ValidatorModel(["bicycle"]),
        event_settle_seconds=0.0,
    )
    detector.enable_static_background_filter = False
    detector.enable_finish_segment_filter = False
    state = TrackState(prev_x=800, prev_y=900)
    state.start_pos = (800, 800)
    state.start_time = 0.0
    state.best_bib = "7"
    state.best_bib_conf = 0.9
    state.has_bib_box = True
    detector._track_states[11] = state
    return detector


def test_crossing_event_uses_supplied_capture_timestamp_and_wall_realtime(monkeypatch):
    wall_time = 1_900_000_000.0
    capture_time = 123.456
    monkeypatch.setattr("realtime.detector.time.time", lambda: wall_time)
    detector = _crossing_detector_for_timestamp_test()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    events, _, _ = detector.process_frame(frame, timestamp=capture_time)

    assert len(events) == 1
    assert events[0].cross_time == capture_time
    assert events[0].cross_realtime == datetime.fromtimestamp(wall_time).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def test_crossing_event_without_timestamp_keeps_wall_clock_behavior(monkeypatch):
    wall_time = 1_900_000_000.0
    monkeypatch.setattr("realtime.detector.time.time", lambda: wall_time)
    detector = _crossing_detector_for_timestamp_test()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    events, _, _ = detector.process_frame(frame)

    assert len(events) == 1
    assert events[0].cross_time == wall_time
    assert events[0].cross_realtime == datetime.fromtimestamp(wall_time).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def test_crossing_event_keeps_zero_capture_timestamp(monkeypatch):
    wall_time = 1_900_000_000.0
    monkeypatch.setattr("realtime.detector.time.time", lambda: wall_time)
    detector = _crossing_detector_for_timestamp_test()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    events, _, _ = detector.process_frame(frame, timestamp=0.0)

    assert len(events) == 1
    assert events[0].cross_time == 0.0


def test_raw_bib_detection_is_kept_when_tracker_returns_only_athlete():
    model = _RawBibCaptureTrackingModel()
    detector = Detector(model_path="fake.pt", model=model, ocr=None)
    detector.enable_static_background_filter = False
    detector.enable_finish_segment_filter = False
    detector.adaptive_frame_skip = False
    detector.set_finish_line((0, 310), (320, 310))
    frame = np.zeros((320, 320, 3), dtype=np.uint8)

    detector.process_frame(frame, timestamp=0.0)

    state = detector._track_states[11]
    assert state.has_bib_box is True
    assert state.bib_crops_cache[0][3] == [120, 140, 160, 170]


def test_event_evidence_prefers_detected_bib_over_torso_fallback():
    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    detected_crop = np.full((24, 28, 3), 180, dtype=np.uint8)
    fallback_crop = np.full((160, 100, 3), 80, dtype=np.uint8)
    detector = Detector(
        model_path="fake.pt",
        model=_EmptyTrackingModel(),
        ocr=None,
        athlete_validator=_ValidatorModel(["bicycle"]),
    )
    detector.enable_static_background_filter = False
    detector.enable_finish_segment_filter = False
    state = _pending_crossing_state(frame, 101)
    state.has_bib_box = True
    state.bib_crops_cache = [(0.50, detected_crop, None, [120, 140, 148, 164])]
    state.fallback_bib_crops_cache = [(0.95, fallback_crop, frame, [100, 120, 200, 280])]
    detector._track_states[101] = state

    events, _, _ = detector.process_frame(frame, timestamp=2.0)

    assert len(events) == 1
    assert events[0].bib_bbox == [120, 140, 148, 164]
    assert np.array_equal(events[0].bib_crop, detected_crop)


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


def test_small_detected_bib_candidate_keeps_fallback_as_secondary_evidence():
    detector = Detector(model_path="fake.pt", model=None, ocr=None)
    real_crop = np.zeros((24, 28, 3), dtype=np.uint8)
    fallback_crop = np.zeros((160, 100, 3), dtype=np.uint8)
    state = TrackState(prev_x=0, prev_y=0)
    state.bib_crops_cache = [(0.50, real_crop, None, [1, 2, 3, 4])]
    state.fallback_bib_crops_cache = [(0.95, fallback_crop, None, [5, 6, 7, 8])]

    candidates = detector._get_ocr_candidates(state)

    assert candidates == [state.bib_crops_cache[0], state.fallback_bib_crops_cache[0]]


def test_usable_detected_bib_candidates_keep_priority_without_fallback_noise():
    detector = Detector(model_path="fake.pt", model=None, ocr=None)
    real_crop = np.zeros((48, 64, 3), dtype=np.uint8)
    fallback_crop = np.zeros((160, 100, 3), dtype=np.uint8)
    state = TrackState(prev_x=0, prev_y=0)
    state.bib_crops_cache = [(0.75, real_crop, None, [1, 2, 3, 4])]
    state.fallback_bib_crops_cache = [(0.95, fallback_crop, None, [5, 6, 7, 8])]

    candidates = detector._get_ocr_candidates(state)

    assert candidates == state.bib_crops_cache


def test_implausibly_wide_bib_sequence_falls_back_to_athlete_evidence():
    detector = Detector(model_path="fake.pt", model=None, ocr=None)
    false_crop = np.zeros((26, 69, 3), dtype=np.uint8)
    fallback_crop = np.zeros((160, 100, 3), dtype=np.uint8)
    state = TrackState(prev_x=0, prev_y=0)
    state.bib_crops_cache = [
        (0.77, false_crop, None, [0, 0, 59, 16]),
        (0.68, false_crop, None, [0, 0, 41, 16]),
        (0.67, false_crop, None, [0, 0, 39, 17]),
        (0.66, false_crop, None, [0, 0, 38, 16]),
    ]
    state.fallback_bib_crops_cache = [
        (0.60, fallback_crop, None, [10, 20, 110, 180]),
    ]

    candidates = detector._get_ocr_candidates(state)

    assert candidates == state.fallback_bib_crops_cache


def test_event_keeps_ranked_detected_bib_candidates_for_batch_consensus():
    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    detector = Detector(
        model_path="fake.pt",
        model=_EmptyTrackingModel(),
        ocr=None,
        athlete_validator=_ValidatorModel(["bicycle"]),
    )
    detector.enable_static_background_filter = False
    detector.enable_finish_segment_filter = False
    state = _pending_crossing_state(frame, 202)
    state.has_bib_box = True
    state.bib_crops_cache = [
        (0.90, np.full((48, 64, 3), 30, dtype=np.uint8), frame, [120, 140, 184, 188]),
        (0.80, np.full((44, 60, 3), 60, dtype=np.uint8), frame, [122, 142, 182, 186]),
        (0.70, np.full((40, 56, 3), 90, dtype=np.uint8), frame, [124, 144, 180, 184]),
    ]
    detector._track_states[202] = state

    events, _, _ = detector.process_frame(frame, timestamp=2.0)

    assert len(events) == 1
    assert events[0].bib_evidence_kind == "detected"
    assert [int(candidate[1][0, 0, 0]) for candidate in events[0].bib_candidates] == [30, 60, 90]


def test_event_marks_implausible_detected_bib_history_as_fallback_evidence():
    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    detector = Detector(
        model_path="fake.pt",
        model=_EmptyTrackingModel(),
        ocr=None,
        athlete_validator=_ValidatorModel(["bicycle"]),
    )
    detector.enable_static_background_filter = False
    detector.enable_finish_segment_filter = False
    state = _pending_crossing_state(frame, 203)
    state.has_bib_box = True
    false_crop = np.zeros((26, 69, 3), dtype=np.uint8)
    state.bib_crops_cache = [
        (0.77, false_crop, frame, [0, 0, 59, 16]),
        (0.68, false_crop, frame, [0, 0, 41, 16]),
        (0.67, false_crop, frame, [0, 0, 39, 17]),
    ]
    state.fallback_bib_crops_cache = [
        (0.60, np.zeros((80, 100, 3), dtype=np.uint8), frame, [100, 90, 200, 170]),
    ]
    detector._track_states[203] = state

    events, _, _ = detector.process_frame(frame, timestamp=2.0)

    assert len(events) == 1
    assert events[0].bib_evidence_kind == "fallback"
    assert events[0].bib_candidates == []


def test_local_video_mode_uses_a_bounded_settle_delay():
    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    detector = Detector(
        model_path="fake.pt",
        model=_EmptyTrackingModel(),
        ocr=None,
        athlete_validator=_ValidatorModel(["bicycle"]),
        event_settle_seconds=LOCAL_VIDEO_EVENT_SETTLE_SECONDS,
    )
    detector.enable_static_background_filter = False
    detector.enable_finish_segment_filter = False
    state = _pending_crossing_state(frame, 204)
    state.crossed_time = 2.0
    detector._track_states[204] = state

    early_events, _, _ = detector.process_frame(frame, timestamp=2.84)
    events, _, _ = detector.process_frame(frame, timestamp=2.86)

    assert early_events == []
    assert len(events) == 1
