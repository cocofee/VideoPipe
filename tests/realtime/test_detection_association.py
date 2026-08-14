import cv2
import numpy as np
import pytest
from datetime import datetime
from types import SimpleNamespace

from realtime.detector import (
    BibEvidenceCandidate,
    Detector,
    LOCAL_VIDEO_EVENT_SETTLE_SECONDS,
    TrackState,
    resolve_athlete_validator_model,
    resolve_performance_profile,
)
from realtime.event_profile import build_event_profile


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


class _VerticalFinishCrossingModel:
    names = {0: "person"}

    def __init__(self):
        self._centers = iter((400, 350, 300))

    def track(self, frame, **kwargs):
        center_x = next(self._centers)
        boxes = _SimpleBoxes(
            cls=[0],
            conf=[0.95],
            xyxy=[[center_x - 50, 300, center_x + 50, 500]],
            ids=[11],
        )
        return [SimpleNamespace(boxes=boxes, names=self.names)]


class _UpperEndpointFinishCrossingModel:
    names = {0: "person"}

    def __init__(self):
        self._centers = iter((400, 350, 300))

    def track(self, frame, **kwargs):
        center_x = next(self._centers)
        boxes = _SimpleBoxes(
            cls=[0],
            conf=[0.95],
            xyxy=[[center_x - 50, 10, center_x + 50, 90]],
            ids=[11],
        )
        return [SimpleNamespace(boxes=boxes, names=self.names)]


class _ApproachingUpperEndpointFinishCrossingModel:
    names = {0: "person"}

    def __init__(self):
        self._positions = iter(
            (
                (520, 20),
                (460, 20),
                (400, 20),
                (350, 45),
                (300, 50),
            )
        )

    def track(self, frame, **kwargs):
        center_x, bottom_y = next(self._positions)
        boxes = _SimpleBoxes(
            cls=[0],
            conf=[0.95],
            xyxy=[[center_x - 50, bottom_y - 80, center_x + 50, bottom_y]],
            ids=[11],
        )
        return [SimpleNamespace(boxes=boxes, names=self.names)]


class _BackgroundBibTrackingModel:
    names = {0: "bike", 1: "BIB"}

    def track(self, frame, **kwargs):
        boxes = _SimpleBoxes(
            cls=[0, 1],
            conf=[0.95, 0.93],
            xyxy=[[715, 484, 858, 812], [790, 511, 836, 565]],
            ids=[9, 99],
        )
        return [SimpleNamespace(boxes=boxes, names=self.names)]


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


class _RawPersonCaptureTrackingModel:
    names = {0: "person"}

    def __init__(self):
        self.postprocess_callbacks = []

    def add_callback(self, event, callback):
        if event == "on_predict_postprocess_end":
            self.postprocess_callbacks.append(callback)

    def track(self, frame, **kwargs):
        raw_result = SimpleNamespace(
            boxes=_SimpleBoxes(
                cls=[0, 0, 0],
                conf=[0.95, 0.90, 0.88],
                xyxy=[
                    [10, 140, 190, 300],
                    [200, 140, 280, 300],
                    [200, 20, 260, 100],
                ],
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
                xyxy=[[40, 140, 120, 300]],
                ids=[11],
            ),
            names=self.names,
        )
        return [tracked_result]


class _RawEdgeEntryTrackingModel:
    names = {0: "person"}

    def __init__(self):
        self.frame_index = 0
        self.postprocess_callbacks = []

    def add_callback(self, event, callback):
        if event == "on_predict_postprocess_end":
            self.postprocess_callbacks.append(callback)

    def track(self, frame, **kwargs):
        self.frame_index += 1
        if self.frame_index == 1:
            raw_xyxy = [[608, 220, 639, 430]]
            tracked_result = SimpleNamespace(
                boxes=_SimpleBoxes(cls=[], conf=[], xyxy=[]),
                names=self.names,
            )
        else:
            raw_xyxy = [[575, 230, 639, 440]]
            tracked_result = SimpleNamespace(
                boxes=_SimpleBoxes(
                    cls=[0],
                    conf=[0.91],
                    xyxy=[[592, 230, 632, 440]],
                    ids=[117],
                ),
                names=self.names,
            )

        raw_result = SimpleNamespace(
            boxes=_SimpleBoxes(cls=[0], conf=[0.91], xyxy=raw_xyxy),
            names=self.names,
        )
        predictor = SimpleNamespace(results=[raw_result])
        for callback in self.postprocess_callbacks:
            callback(predictor)
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


def test_distinct_participants_are_not_deduplicated_by_nested_unknown_boxes():
    detector = Detector(model_path="fake.pt", model=None, ocr=None)
    first = [1062, 452, 1378, 865]
    second = [1120, 446, 1401, 852]

    assert detector._is_duplicate_unknown_event_bbox(
        second,
        first,
        time_diff=0.84,
        current_participant_id="P000110",
        previous_participant_id="P000111",
    ) is False
    assert detector._is_duplicate_unknown_event_bbox(
        second,
        first,
        time_diff=0.84,
        current_participant_id="P000111",
        previous_participant_id="P000111",
    ) is True


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


def test_background_bib_above_rider_torso_is_not_assigned():
    detector = Detector(model_path="fake.pt", model=None, ocr=None)
    athletes = [_athlete(9, [715, 484, 858, 812])]
    bibs = [_bib([790, 511, 836, 565], conf=0.93)]

    assignments, unmatched = detector._assign_bibs_to_athletes(athletes, bibs)

    assert assignments == {}
    assert unmatched == {0}


def test_background_bib_never_enters_detected_candidate_cache():
    detector = Detector(model_path="fake.pt", model=_BackgroundBibTrackingModel(), ocr=None)
    detector.enable_static_background_filter = False
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    detector.process_frame(frame, timestamp=1.0)

    assert detector._last_bib_assign_stats == {"bibs": 1, "assigned": 0, "rejected": 1}
    assert detector._track_states[9].bib_crops_cache == []


def test_lower_torso_bib_is_assigned_to_unique_rider():
    detector = Detector(model_path="fake.pt", model=None, ocr=None)
    athletes = [_athlete(1, [100, 100, 260, 420])]
    bibs = [_bib([155, 280, 210, 335], conf=0.93)]

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


def test_detector_exposes_stable_participant_id_for_track_fragment():
    detector = Detector(model_path="fake.pt", model=None, ocr=None)
    try:
        first = detector.resolve_participant(
            _athlete(405, (700, 300, 850, 900)), timestamp=1.0
        )
        second = detector.resolve_participant(
            _athlete(605, (705, 320, 846, 902)), timestamp=1.4
        )

        assert second["participant_id"] == first["participant_id"]
        assert second["raw_track_ids"] == [405, 605]
        assert second["identity_status"] == "ACTIVE"
    finally:
        detector.stop()


def test_detector_keeps_raw_track_id_for_audit():
    detector = Detector(model_path="fake.pt", model=None, ocr=None)
    try:
        resolved = detector.resolve_participant(
            _athlete(405, (700, 300, 850, 900)), timestamp=1.0
        )

        assert resolved["track_id"] == 405
        assert resolved["raw_track_ids"] == [405]
        assert resolved["participant_id"]
        assert resolved["identity_status"] == "ACTIVE"
    finally:
        detector.stop()


def test_detector_reset_starts_a_new_participant_identity_session():
    detector = Detector(model_path="fake.pt", model=None, ocr=None)
    try:
        first, first_resolution = detector._resolve_participant_with_resolution(
            _athlete(405, (700, 300, 850, 900)), timestamp=1.0
        )

        detector.reset()

        second, second_resolution = detector._resolve_participant_with_resolution(
            _athlete(405, (700, 300, 850, 900)), timestamp=1.0
        )

        assert first["participant_id"] == "P000001"
        assert second["participant_id"] == "P000001"
        assert first_resolution.created is True
        assert second_resolution.created is True
    finally:
        detector.stop()


def test_running_profile_uses_common_pipeline_without_bicycle_validation():
    validator = _ValidatorModel(["person"])
    detector = Detector(
        model_path="fake.pt",
        model=None,
        ocr=None,
        athlete_validator=validator,
        sport_profile="running",
    )
    crop = np.zeros((160, 100, 3), dtype=np.uint8)

    assert detector.sport_profile == "running"
    assert detector.event_profile.pipeline == "athlete_bib"
    assert detector.event_profile.required_equipment is None
    assert detector._passes_athlete_event_validation(
        TrackState(prev_x=0, prev_y=0),
        crop,
    ) is True
    assert validator.calls == 0


def test_explicit_event_profile_overrides_legacy_sport_name():
    validator = _ValidatorModel(["person"])
    profile = build_event_profile(name="current-skating-event")
    detector = Detector(
        model_path="fake.pt",
        model=None,
        ocr=None,
        athlete_validator=validator,
        sport_profile="cycling",
        event_profile=profile,
    )
    crop = np.zeros((160, 100, 3), dtype=np.uint8)

    assert detector.event_profile is profile
    assert detector.sport_profile == "current-skating-event"
    assert detector._passes_athlete_event_validation(
        TrackState(prev_x=0, prev_y=0),
        crop,
    ) is True
    assert validator.calls == 0


def test_speed_skating_profile_keeps_the_full_finish_segment_available():
    profile = build_event_profile(
        name="speed_skating",
        bib_regions=("helmet", "left_thigh", "right_thigh"),
    )
    detector = Detector(
        model_path="fake.pt",
        model=None,
        ocr=None,
        event_profile=profile,
    )

    assert detector.finish_segment_margin_px == 60.0
    assert detector.finish_segment_end_shrink_px == 0.0


def test_blank_legacy_sport_profile_preserves_cycling_validation():
    validator = _ValidatorModel(["person"])
    detector = Detector(
        model_path="fake.pt",
        model=None,
        ocr=None,
        athlete_validator=validator,
        sport_profile="   ",
    )
    crop = np.zeros((160, 100, 3), dtype=np.uint8)

    assert detector.sport_profile == "cycling"
    assert detector.event_profile.required_equipment == "bicycle"
    assert detector._passes_athlete_event_validation(
        TrackState(prev_x=0, prev_y=0),
        crop,
    ) is False
    assert validator.calls == 1


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


def test_crossing_event_keeps_all_athlete_boxes_from_source_frame():
    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    detector = Detector(
        model_path="fake.pt",
        model=_EmptyTrackingModel(),
        ocr=None,
        athlete_validator=_ValidatorModel(["bicycle"]),
    )
    detector.enable_static_background_filter = False
    detector.enable_finish_segment_filter = False
    state = _pending_crossing_state(frame, 101)
    state.pending_event_data["frame_athletes"] = (
        {"bbox": (100, 120, 180, 280), "track_id": 101, "conf": 0.9},
        {"bbox": (190, 100, 260, 270), "track_id": 202, "conf": 0.8},
    )
    detector._track_states[101] = state

    events, _, _ = detector.process_frame(frame, timestamp=1.0)

    assert len(events) == 1
    assert events[0].frame_athletes == state.pending_event_data["frame_athletes"]


def test_crossing_lifecycle_admits_by_participant_before_allocating_event_id():
    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    detector = Detector(
        model_path="fake.pt",
        model=_EmptyTrackingModel(),
        ocr=None,
        athlete_validator=_ValidatorModel(["bicycle"]),
    )
    detector.enable_static_background_filter = False
    detector.enable_finish_segment_filter = False

    first_state = _pending_crossing_state(frame, 101)
    first_state.pending_event_data["participant_id"] = "P4"
    first_state.pending_event_data["raw_track_ids"] = (55, 101)
    detector._track_states[101] = first_state
    first_events, _, _ = detector.process_frame(frame, timestamp=1.0)

    second_state = _pending_crossing_state(frame, 202)
    second_state.pending_event_data.update(
        {
            "participant_id": "P4",
            "raw_track_ids": (202,),
            "bbox": [400, 120, 480, 280],
        }
    )
    second_state.best_athlete_crop = frame[120:280, 400:480].copy()
    detector._track_states[202] = second_state
    second_events, _, _ = detector.process_frame(frame, timestamp=2.0)

    assert [event.event_id for event in first_events] == [1]
    assert first_events[0].participant_id == "P4"
    assert first_events[0].raw_track_ids == (55, 101)
    assert second_events == []
    assert detector._event_id == 1
    snapshot = detector._crossing_lifecycle.snapshot_for_key(
        (detector._crossing_session_id, detector.source_id, "P4")
    )
    assert snapshot.raw_track_ids == (101, 202)


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
    assert profile["process_imgsz"] == 832
    assert profile["adaptive_frame_skip"] is True


def test_yolo_only_detector_keeps_ocr_pipeline_disabled():
    detector = Detector(
        model_path="fake.pt",
        model=None,
        ocr=object(),
        realtime_ocr=True,
        ocr_pipeline_enabled=False,
    )

    detector.set_ocr(object())

    assert detector._ocr is None
    assert detector.realtime_ocr_enabled is False
    assert detector._ocr_running is False
    assert detector._get_ocr_candidates(TrackState(prev_x=0, prev_y=0)) == []


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
    frame[545:635, 715:845] = (255, 255, 255)
    cv2.putText(
        frame,
        "206",
        (730, 605),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )

    detector.process_frame(frame, timestamp=0.0)

    assert model.source is frame
    assert model.kwargs["imgsz"] == 832
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


def test_raw_near_track_person_is_kept_when_tracker_drops_new_athlete():
    model = _RawPersonCaptureTrackingModel()
    detector = Detector(model_path="fake.pt", model=model, ocr=None)
    detector.enable_static_background_filter = False
    detector.enable_finish_segment_filter = False
    detector.adaptive_frame_skip = False
    frame = np.zeros((320, 320, 3), dtype=np.uint8)

    _, athletes, _ = detector.process_frame(frame, timestamp=0.0)

    assert len(athletes) == 2
    assert sorted(athlete["bbox"] for athlete in athletes) == [
        [40, 140, 120, 300],
        [200, 140, 280, 300],
    ]
    assert {athlete["track_id"] for athlete in athletes} == {11, 900000}


def test_edge_entry_athlete_keeps_raw_box_when_tracker_clips_it_too_narrow():
    detector = Detector(
        model_path="fake.pt",
        model=_RawEdgeEntryTrackingModel(),
        ocr=None,
        sport_profile="speed_skating",
    )
    detector.enable_static_background_filter = False
    detector.enable_finish_segment_filter = False
    detector.adaptive_frame_skip = False
    frame = np.zeros((640, 640, 3), dtype=np.uint8)

    _, first_athletes, _ = detector.process_frame(frame, timestamp=0.0)
    _, second_athletes, _ = detector.process_frame(frame, timestamp=0.04)

    assert [athlete["bbox"] for athlete in first_athletes] == [[608, 220, 639, 430]]
    assert [athlete["bbox"] for athlete in second_athletes] == [[575, 230, 639, 440]]
    assert first_athletes[0]["track_id"] == second_athletes[0]["track_id"] == 900000


def test_edge_aspect_exception_does_not_accept_a_tall_side_pillar():
    detector = Detector(model_path="fake.pt", model=None, ocr=None)

    assert detector._passes_athlete_aspect_filter(
        [616, 230, 639, 355],
        0.68,
        (640, 640),
    ) is True
    assert detector._passes_athlete_aspect_filter(
        [610, 100, 639, 500],
        0.95,
        (640, 640),
    ) is False


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


def test_speed_skating_profile_accepts_helmet_bib_association():
    athlete = {"bbox": [100, 100, 300, 500], "track_id": 7}
    helmet_bib = {
        "bbox": [175, 108, 225, 132],
        "center_x": 200,
        "center_y": 120,
        "conf": 0.9,
    }
    speed_skating = Detector(
        model_path="fake.pt",
        model=None,
        ocr=None,
        sport_profile="speed_skating",
    )
    cycling = Detector(model_path="fake.pt", model=None, ocr=None, sport_profile="cycling")

    speed_assignments, _ = speed_skating._assign_bibs_to_athletes([athlete], [helmet_bib])
    cycling_assignments, _ = cycling._assign_bibs_to_athletes([athlete], [helmet_bib])

    assert speed_assignments == {0: [0]}
    assert cycling_assignments == {}


def test_speed_skating_fallbacks_cover_helmet_and_both_thighs():
    detector = Detector(
        model_path="fake.pt",
        model=None,
        ocr=None,
        sport_profile="speed_skating",
    )
    frame = np.zeros((600, 800, 3), dtype=np.uint8)

    candidates = detector._extract_bib_fallbacks_from_athlete(
        frame,
        [100, 100, 500, 500],
    )
    sources = {source for source, _, _ in candidates}

    assert {"helmet_left", "helmet_right", "left_thigh", "right_thigh"} <= sources
    assert all(crop.size > 0 for _, crop, _ in candidates)


def test_speed_skating_profile_rejects_spectator_sized_boxes_only():
    speed_skating = Detector(
        model_path="fake.pt",
        model=None,
        ocr=None,
        sport_profile="speed_skating",
    )
    cycling = Detector(model_path="fake.pt", model=None, ocr=None, sport_profile="cycling")
    frame_shape = (1440, 2560)
    small_spectator = [1218, 451, 1266, 496]
    finish_skater = [1086, 601, 1345, 954]

    assert speed_skating._passes_profile_athlete_geometry(small_spectator, frame_shape) is False
    assert speed_skating._passes_profile_athlete_geometry(finish_skater, frame_shape) is True
    assert cycling._passes_profile_athlete_geometry(small_spectator, frame_shape) is True


def test_speed_skating_unknown_athlete_crosses_vertical_finish_line():
    detector = Detector(
        model_path="fake.pt",
        model=_VerticalFinishCrossingModel(),
        ocr=None,
        event_settle_seconds=0.0,
        sport_profile="speed_skating",
    )
    detector.enable_static_background_filter = False
    detector.enable_finish_segment_filter = False
    detector.adaptive_frame_skip = False
    detector.set_finish_line((320, 100), (320, 600))
    frame = np.zeros((640, 640, 3), dtype=np.uint8)

    try:
        events = []
        for timestamp in (0.0, 0.1, 0.2):
            frame_events, _, _ = detector.process_frame(frame, timestamp=timestamp)
            events.extend(frame_events)

        assert len(events) == 1
        assert events[0].track_id == 11
    finally:
        detector.stop()


def test_speed_skating_finish_segment_margin_accepts_upper_endpoint_crossing():
    detector = Detector(
        model_path="fake.pt",
        model=_UpperEndpointFinishCrossingModel(),
        ocr=None,
        event_settle_seconds=0.0,
        sport_profile="speed_skating",
    )
    detector.enable_static_background_filter = False
    detector.adaptive_frame_skip = False
    detector.set_finish_line((320, 100), (320, 600))
    frame = np.zeros((640, 640, 3), dtype=np.uint8)

    try:
        events = []
        for timestamp in (0.0, 0.1, 0.2):
            frame_events, _, _ = detector.process_frame(frame, timestamp=timestamp)
            events.extend(frame_events)

        assert len(events) == 1
        assert events[0].track_id == 11
    finally:
        detector.stop()


def test_speed_skating_keeps_motion_history_before_finish_segment_entry():
    detector = Detector(
        model_path="fake.pt",
        model=_ApproachingUpperEndpointFinishCrossingModel(),
        ocr=None,
        event_settle_seconds=0.0,
        sport_profile="speed_skating",
    )
    detector.enable_static_background_filter = False
    detector.adaptive_frame_skip = False
    detector.set_finish_line((320, 100), (320, 600))
    frame = np.zeros((640, 640, 3), dtype=np.uint8)

    try:
        events = []
        for timestamp in (0.0, 0.1, 0.2, 0.3, 0.4):
            frame_events, _, _ = detector.process_frame(frame, timestamp=timestamp)
            events.extend(frame_events)

        assert len(events) == 1
        assert events[0].track_id == 11
    finally:
        detector.stop()


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


def test_candidate_cache_keeps_only_best_crop_per_source_frame():
    detector = Detector(model_path="fake.pt", model=None, ocr=None)
    state = TrackState(prev_x=0, prev_y=0)
    first = BibEvidenceCandidate(
        quality=0.50,
        crop=np.full((40, 60, 3), 10, dtype=np.uint8),
        frame=None,
        frame_index=12,
        capture_time_ms=400.0,
        athlete_bbox=(100, 100, 260, 420),
        bib_bbox=(150, 260, 210, 310),
        source="detected",
        owner_validated=True,
    )
    better = BibEvidenceCandidate(
        quality=0.80,
        crop=np.full((40, 60, 3), 20, dtype=np.uint8),
        frame=None,
        frame_index=12,
        capture_time_ms=400.0,
        athlete_bbox=(100, 100, 260, 420),
        bib_bbox=(150, 260, 210, 310),
        source="detected",
        owner_validated=True,
    )

    detector._cache_ocr_candidate(state, first, detected_bib=True)
    detector._cache_ocr_candidate(state, better, detected_bib=True)

    assert len(state.bib_crops_cache) == 1
    assert state.bib_crops_cache[0].quality == 0.80
    assert int(state.bib_crops_cache[0].crop[0, 0, 0]) == 20


def test_saturated_low_text_region_is_rejected_as_non_bib():
    detector = Detector(model_path="fake.pt", model=None, ocr=None)
    flag_crop = np.full((68, 74, 3), (30, 220, 30), dtype=np.uint8)
    cv2.rectangle(flag_crop, (0, 0), (73, 67), (20, 80, 20), 2)

    assert detector._has_strong_non_bib_signature(flag_crop) is True


def test_colored_bib_with_visible_digits_is_not_rejected():
    detector = Detector(model_path="fake.pt", model=None, ocr=None)
    bib_crop = np.full((68, 74, 3), (30, 220, 30), dtype=np.uint8)
    cv2.putText(
        bib_crop,
        "206",
        (2, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.15,
        (10, 10, 10),
        3,
        cv2.LINE_AA,
    )

    assert detector._has_strong_non_bib_signature(bib_crop) is False


def _owned_candidate(quality, rel_x, rel_y, frame_index, value):
    athlete_bbox = (100, 100, 300, 500)
    bib_width = 50
    bib_height = 40
    center_x = athlete_bbox[0] + rel_x * (athlete_bbox[2] - athlete_bbox[0])
    center_y = athlete_bbox[1] + rel_y * (athlete_bbox[3] - athlete_bbox[1])
    bib_bbox = (
        int(center_x - bib_width / 2),
        int(center_y - bib_height / 2),
        int(center_x + bib_width / 2),
        int(center_y + bib_height / 2),
    )
    return BibEvidenceCandidate(
        quality=quality,
        crop=np.full((48, 64, 3), value, dtype=np.uint8),
        frame=None,
        frame_index=frame_index,
        capture_time_ms=float(frame_index) * 10.0,
        athlete_bbox=athlete_bbox,
        bib_bbox=bib_bbox,
        source="detected",
        owner_validated=True,
    )


def test_spatially_inconsistent_detected_history_falls_back_to_torso():
    detector = Detector(model_path="fake.pt", model=None, ocr=None)
    fallback = (0.60, np.zeros((120, 90, 3), dtype=np.uint8), None, [120, 180, 210, 300])
    state = TrackState(prev_x=0, prev_y=0)
    state.bib_crops_cache = [
        _owned_candidate(0.80, 0.68, 0.46, 10, 30),
        _owned_candidate(0.75, 0.43, 0.40, 20, 60),
    ]
    state.fallback_bib_crops_cache = [fallback]

    assert detector._get_plausible_detected_bib_candidates(state) == []
    assert detector._get_ocr_candidates(state) == [fallback]


def test_spatially_consistent_detected_history_keeps_best_pair():
    detector = Detector(model_path="fake.pt", model=None, ocr=None)
    consistent_first = _owned_candidate(0.80, 0.48, 0.46, 10, 30)
    consistent_second = _owned_candidate(0.75, 0.53, 0.50, 20, 60)
    outlier = _owned_candidate(0.90, 0.78, 0.30, 30, 90)
    state = TrackState(prev_x=0, prev_y=0)
    state.bib_crops_cache = [outlier, consistent_first, consistent_second]

    candidates = detector._get_plausible_detected_bib_candidates(state)

    assert candidates == [consistent_first, consistent_second]


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


def test_event_keeps_two_ranked_detected_bib_candidates_for_frame_consensus():
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
    assert [int(candidate[1][0, 0, 0]) for candidate in events[0].bib_candidates] == [30, 60]


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


def test_event_fallback_is_recropped_from_the_crossing_athlete():
    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    stale_crop = np.full((96, 80, 3), 251, dtype=np.uint8)
    detector = Detector(
        model_path="fake.pt",
        model=_EmptyTrackingModel(),
        ocr=None,
        sport_profile="speed_skating",
    )
    detector.enable_static_background_filter = False
    detector.enable_finish_segment_filter = False
    state = _pending_crossing_state(frame, 205)
    state.fallback_bib_crops_cache = [
        BibEvidenceCandidate(
            quality=0.99,
            crop=stale_crop,
            frame=np.full_like(frame, 251),
            frame_index=10,
            capture_time_ms=100.0,
            athlete_bbox=(210, 100, 300, 290),
            bib_bbox=(220, 130, 290, 226),
            source="fallback:left_thigh",
            owner_validated=False,
        )
    ]
    detector._track_states[205] = state

    events, _, _ = detector.process_frame(frame, timestamp=2.0)

    assert len(events) == 1
    event = events[0]
    assert event.bib_evidence_kind == "fallback"
    assert detector._bbox_contains_bbox(event.bbox, event.bib_bbox)
    x1, y1, x2, y2 = event.bib_bbox
    assert np.array_equal(event.bib_crop, frame[y1:y2, x1:x2])
    assert not np.array_equal(event.bib_crop, stale_crop)


def test_sync_crossing_ocr_uses_current_event_fallback():
    class RecordingOCR:
        def __init__(self):
            self.minimum_values = []

        def __call__(self, image):
            self.minimum_values.append(int(image.min()))
            return [([0, 0, 1, 1], "137", 0.99)], 0.0

    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    ocr = RecordingOCR()
    detector = Detector(
        model_path="fake.pt",
        model=None,
        ocr=ocr,
        realtime_ocr=True,
        sport_profile="speed_skating",
    )
    state = _pending_crossing_state(frame, 207)
    state.fallback_bib_crops_cache = [
        BibEvidenceCandidate(
            quality=0.99,
            crop=np.full((96, 80, 3), 251, dtype=np.uint8),
            frame=np.full_like(frame, 251),
            frame_index=10,
            capture_time_ms=100.0,
            athlete_bbox=(210, 100, 300, 290),
            bib_bbox=(220, 130, 290, 226),
            source="fallback:left_thigh",
            owner_validated=False,
        )
    ]

    try:
        detector._try_sync_crossing_ocr(
            207,
            state,
            current_time=2.0,
            event_data=state.pending_event_data,
        )
    finally:
        detector.stop()

    assert ocr.minimum_values and ocr.minimum_values[0] < 100
    assert state.best_bib == "137"
    assert detector._bbox_contains_bbox(
        state.pending_event_data["bbox"],
        state.best_bib_bbox,
    )


def test_event_rejects_detected_candidate_without_valid_owner_geometry():
    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    detector = Detector(
        model_path="fake.pt",
        model=_EmptyTrackingModel(),
        ocr=None,
        sport_profile="speed_skating",
    )
    detector.enable_static_background_filter = False
    detector.enable_finish_segment_filter = False
    state = _pending_crossing_state(frame, 206)
    state.has_bib_box = True
    state.bib_crops_cache = [
        BibEvidenceCandidate(
            quality=0.95,
            crop=np.full((48, 64, 3), 200, dtype=np.uint8),
            frame=frame,
            frame_index=20,
            capture_time_ms=200.0,
            athlete_bbox=(100, 120, 180, 280),
            bib_bbox=(220, 130, 284, 178),
            source="detected",
            owner_validated=True,
        )
    ]
    detector._track_states[206] = state

    events, _, _ = detector.process_frame(frame, timestamp=2.0)

    assert len(events) == 1
    assert events[0].bib_evidence_kind == "fallback"
    assert events[0].bib_candidates == []
    assert detector._bbox_contains_bbox(events[0].bbox, events[0].bib_bbox)


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
