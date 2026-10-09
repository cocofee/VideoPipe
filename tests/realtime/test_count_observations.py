from types import SimpleNamespace

import numpy as np
from ultralytics.engine.results import Results

from realtime.count_observations import ObservedBoxCollector
from realtime.independent_counting import FrameStamp


def test_class_filter_keeps_raw_audit_and_tracker_detection_index_beats_geometry():
    original_callback = lambda predictor: None
    model = SimpleNamespace(callbacks={"on_predict_postprocess_end": [original_callback]})
    collector = ObservedBoxCollector(model, allowed_classes=(1,))
    collector.begin(FrameStamp(1, "g", 1000, 1000))
    raw = np.array([[10, 10, 40, 40, 0.9, 0], [10, 10, 40, 40, 0.8, 1], [20, 10, 50, 40, 0.7, 1]])
    result = Results(np.zeros((100, 100, 3), dtype=np.uint8), path="test", names={0: "person", 1: "bicycle"}, boxes=raw)
    tracker = SimpleNamespace(update=lambda *args: np.array([[10, 10, 40, 40, 7, 0.7, 1, 1], [20, 10, 50, 40, 8, 0.8, 1, 0]]))
    original_update = tracker.update
    try:
        predictor = SimpleNamespace(results=[result], trackers=[tracker])
        collector.capture(predictor)
        assert len(collector.raw) == 3
        assert len(predictor.results[0].boxes) == 2
        assert list(predictor.results[0].boxes.cls) == [1, 1]
        tracker.update()
        linked = collector.link([(1, 0.7, (10, 10, 40, 40), 7), (1, 0.8, (20, 10, 50, 40), 8)])
        assert linked[0].detection_ref == "g:1:det:2"
        assert linked[1].detection_ref == "g:1:det:1"
        assert linked[0].bbox == (20, 10, 50, 40)
    finally:
        collector.close()
    assert tracker.update is original_update
    assert model.callbacks["on_predict_postprocess_end"] == [original_callback]


def test_backend_without_raw_callback_cannot_claim_observed_crossing():
    collector = ObservedBoxCollector(SimpleNamespace())
    collector.begin(FrameStamp(1, "g", 1000, 1000))
    observation, = collector.link([(1, 0.9, (10, 10, 40, 40), 1)])
    assert not observation.observed
    assert observation.quality == ("raw_audit_unavailable",)
