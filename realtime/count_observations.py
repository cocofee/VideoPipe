"""Capture detections before Ultralytics association, then link observed boxes."""

from __future__ import annotations


try:
    from .independent_counting import Observation
except ImportError:
    from independent_counting import Observation


def overlap(a, b):
    area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - area
    return area / union if union > 0 else 0


class ObservedBoxCollector:
    def __init__(self, model, *, allowed_classes=None):
        self.raw = []
        self.stamp = None
        self.model = model
        self.allowed_classes = set(allowed_classes) if allowed_classes is not None else None
        self._tracker_updates = []
        self._selected_by_tracker = {}
        self._association_refs = {}
        self.available = hasattr(model, "callbacks")
        if self.available:
            # Track registration appends or replaces its own callback. Taking
            # this snapshot first also works if tracking was already registered.
            model.callbacks["on_predict_postprocess_end"].insert(0, self.capture)

    def capture(self, predictor):
        self.raw = []
        self._association_refs = {}
        for result_index, result in enumerate(predictor.results):
            boxes = result.boxes
            if boxes is None:
                continue
            if boxes.id is not None:
                # A reordered callback cannot silently masquerade as raw audit.
                self.available = False
                continue
            frame_raw = []
            for index, box in enumerate(boxes):
                bbox = tuple(float(x) for x in box.xyxy[0].tolist())
                ref = f"{self.stamp.generation}:{self.stamp.sequence}:det:{index}"
                observation = Observation(None, bbox, int(box.cls[0]), float(box.conf[0]), ref)
                self.raw.append(observation)
                frame_raw.append(observation)
            selected = frame_raw
            if self.allowed_classes is not None:
                keep = [index for index, value in enumerate(boxes.cls) if int(value) in self.allowed_classes]
                selected = [frame_raw[index] for index in keep]
                if len(keep) != len(boxes):
                    predictor.results[result_index] = result[keep]
            trackers = getattr(predictor, "trackers", ())
            if trackers:
                tracker_index = result_index if len(trackers) > 1 else 0
                tracker = trackers[tracker_index]
                self._selected_by_tracker[id(tracker)] = selected
                if not any(item[0] is tracker for item in self._tracker_updates):
                    original_update = tracker.update
                    def audited_update(*args, _tracker=tracker, _update=original_update, **kwargs):
                        tracks = _update(*args, **kwargs)
                        detections = self._selected_by_tracker.get(id(_tracker), ())
                        for track in tracks:
                            # Axis-aligned Ultralytics ByteTrack returns
                            # xyxy, track_id, score, cls, original_detection_idx.
                            if len(track) == 8:
                                index = int(track[-1])
                                if 0 <= index < len(detections):
                                    self._association_refs[int(track[4])] = detections[index]
                        return tracks
                    tracker.update = audited_update
                    self._tracker_updates.append((tracker, original_update))

    def begin(self, stamp):
        self.stamp = stamp
        self.raw = []

    def link(self, boxes):
        if not self.available:
            # Test adapters and unsupported backends must declare missing audit.
            return [Observation(tid, tuple(bbox), cls, confidence,
                                f"{self.stamp.generation}:{self.stamp.sequence}:track:{index}",
                                observed=False, quality=("raw_audit_unavailable",))
                    for index, (cls, confidence, bbox, tid) in enumerate(boxes)]
        pairs = []
        for ti, (cls, _, bbox, _) in enumerate(boxes):
            for ri, raw in enumerate(self.raw):
                if cls == raw.class_id:
                    score = overlap(bbox, raw.bbox)
                    if score >= 0.3:
                        pairs.append((score, ti, ri))
        assignments, used = {}, set()
        for score, ti, ri in sorted(pairs, reverse=True):
            if ti not in assignments and ri not in used:
                assignments[ti] = ri
                used.add(ri)
        linked = []
        for index, (cls, confidence, bbox, tid) in enumerate(boxes):
            exact = self._association_refs.get(tid)
            ri = self.raw.index(exact) if exact is not None else assignments.get(index)
            if ri is None:
                linked.append(Observation(tid, tuple(bbox), cls, confidence,
                                          f"{self.stamp.generation}:{self.stamp.sequence}:prediction:{index}", False))
                continue
            raw = self.raw[ri]
            flags = set()
            if any(other != ri and overlap(raw.bbox, item.bbox) > 0.5
                   for other, item in enumerate(self.raw) if item.class_id == raw.class_id):
                flags.add("overlapping_detections")
            scores = sorted((score for score, ti, _ in pairs if ti == index), reverse=True)
            if len(scores) > 1 and scores[0] - scores[1] < 0.05:
                flags.add("association_ambiguous")
            linked.append(Observation(tid, raw.bbox, raw.class_id, raw.confidence,
                                      raw.detection_ref, quality=tuple(sorted(flags))))
        return linked

    def close(self):
        for tracker, update in self._tracker_updates:
            tracker.update = update
        self._tracker_updates.clear()
        if hasattr(self.model, "callbacks"):
            callbacks = self.model.callbacks["on_predict_postprocess_end"]
            if self.capture in callbacks:
                callbacks.remove(self.capture)
