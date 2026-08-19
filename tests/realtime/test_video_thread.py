import numpy as np

from realtime.frame_envelope import FrameEnvelope
from realtime.main_window import MainWindow, PreviewThread, VideoThread
from realtime.stream_reader import StreamStatus


class _Reader:
    def __init__(self):
        self.after_calls = []
        self.legacy_calls = 0

    def set_on_status_change(self, callback):
        self.status_callback = callback

    def get_frame_after(self, timestamp):
        self.after_calls.append(timestamp)
        return object(), 1.0

    def get_frame_with_time(self):
        self.legacy_calls += 1
        return object(), 1.0


class _Detector:
    def set_on_bib_update(self, callback):
        self.bib_callback = callback

    def process_frame(self, frame):
        self.thread._running = False
        return [], [], []


def test_video_thread_requests_only_frames_newer_than_last_timestamp():
    reader = _Reader()
    detector = _Detector()
    thread = VideoThread(reader, detector, ui_skip=99)
    detector.thread = thread

    thread.run()

    assert reader.after_calls == [0.0]
    assert reader.legacy_calls == 0


class _EndedReader:
    status = StreamStatus.ENDED

    def set_on_status_change(self, callback):
        self.status_callback = callback

    def get_frame_after(self, timestamp):
        return None, timestamp


def test_video_thread_exits_when_local_video_reaches_eof():
    reader = _EndedReader()
    detector = _Detector()
    thread = VideoThread(reader, detector, ui_skip=99)

    thread.start()
    try:
        assert thread.wait(500) is True
    finally:
        if thread.isRunning():
            thread.stop()


class _EnvelopeReader:
    status = StreamStatus.CONNECTED

    def __init__(self, envelope):
        self.envelope = envelope
        self.calls = 0
        self.legacy_calls = 0

    def set_on_status_change(self, callback):
        self.status_callback = callback

    def get_frame_envelope(self):
        self.calls += 1
        return self.envelope

    def get_frame_after(self, timestamp):
        self.legacy_calls += 1
        self.thread._running = False
        return None, timestamp


class _TimestampDetector:
    def __init__(self):
        self.received_frame = None
        self.received_timestamp = None

    def set_on_bib_update(self, callback):
        self.bib_callback = callback

    def process_frame(self, frame, timestamp=None):
        self.received_frame = frame
        self.received_timestamp = timestamp
        self.thread._running = False
        return [], [], []


class _ObservedDetector(_TimestampDetector):
    def process_frame(self, frame, timestamp=None):
        self.received_frame = frame
        self.received_timestamp = timestamp
        self.thread._running = False
        return [], [{"bbox": [1, 2, 3, 4], "track_id": 17}], [{"bbox": [2, 3, 4, 5]}]


class _RoiRecoveryDetector(_TimestampDetector):
    def process_frame(self, frame, timestamp=None):
        self.last_frame_metrics = {"roi_auto_disabled": True}
        return super().process_frame(frame, timestamp=timestamp)


def test_video_thread_passes_original_frame_and_capture_timestamp_to_detector():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    envelope = FrameEnvelope(
        original_frame=frame,
        frame_index=12,
        capture_time_ms=1_700_000_000_000.0,
        arrival_time_ms=1_700_000_000_005.0,
        segment_id=3,
    )
    reader = _EnvelopeReader(envelope)
    detector = _TimestampDetector()
    thread = VideoThread(reader, detector, ui_skip=99)
    reader.thread = thread
    detector.thread = thread

    thread.run()

    assert reader.calls == 1
    assert reader.legacy_calls == 0
    assert detector.received_frame is frame
    assert detector.received_frame.shape == (1080, 1920, 3)
    assert detector.received_timestamp == 1_700_000_000.0
    assert thread.last_processed_envelope.processing_time_ms is not None
    assert thread.last_processed_envelope.original_frame is frame
    assert thread.last_frame_metrics["capture_time_ms"] == envelope.capture_time_ms
    assert thread.last_frame_metrics["capture_latency_ms"] == 5.0
    assert thread.last_frame_metrics["queue_latency_ms"] >= 0.0
    assert thread.last_frame_metrics["processing_time_ms"] >= 0.0


def test_video_thread_publishes_detections_with_the_frame_that_produced_them():
    frame = np.full((8, 12, 3), 23, dtype=np.uint8)
    envelope = FrameEnvelope(
        original_frame=frame,
        frame_index=21,
        capture_time_ms=3_000.0,
        arrival_time_ms=3_005.0,
        segment_id=1,
    )
    reader = _EnvelopeReader(envelope)
    detector = _ObservedDetector()
    thread = VideoThread(reader, detector, source_id=2, ui_skip=1)
    reader.thread = thread
    detector.thread = thread
    emitted = []
    thread.frame_ready.connect(
        lambda shown_frame, athletes, bibs, source_id: emitted.append(
            (shown_frame, athletes, bibs, source_id)
        )
    )

    thread.run()

    assert len(emitted) == 1
    shown_frame, athletes, bibs, source_id = emitted[0]
    assert shown_frame is frame
    assert athletes == [{"bbox": [1, 2, 3, 4], "track_id": 17}]
    assert bibs == [{"bbox": [2, 3, 4, 5]}]
    assert source_id == 2


def test_video_thread_limits_pending_ui_frames_to_one():
    thread = VideoThread(_Reader(), _Detector())

    assert thread._reserve_ui_publish_slot() is True
    assert thread._reserve_ui_publish_slot() is False

    thread.mark_ui_consumed()

    assert thread._reserve_ui_publish_slot() is True


def test_video_thread_forwards_roi_auto_disable_to_ui():
    frame = np.zeros((8, 12, 3), dtype=np.uint8)
    envelope = FrameEnvelope(
        original_frame=frame,
        frame_index=13,
        capture_time_ms=2_000.0,
        arrival_time_ms=2_005.0,
        segment_id=0,
    )
    reader = _EnvelopeReader(envelope)
    detector = _RoiRecoveryDetector()
    thread = VideoThread(reader, detector, source_id=4, ui_skip=99)
    reader.thread = thread
    detector.thread = thread
    emitted = []
    thread.roi_auto_disabled.connect(emitted.append)

    thread.run()

    assert emitted == [4]


class _Signal:
    def __init__(self):
        self.disconnect_calls = 0

    def disconnect(self, callback):
        self.disconnect_calls += 1


class _CheckBox:
    def __init__(self):
        self.checked = True
        self.blocked = []

    def blockSignals(self, blocked):
        self.blocked.append(blocked)

    def setChecked(self, checked):
        self.checked = checked


class _VideoLabel:
    def __init__(self):
        self.show_roi = True
        self.roi_changed = _Signal()

    def set_show_roi(self, enabled):
        self.show_roi = enabled


class _RoiDetector:
    def __init__(self):
        self.points = "unchanged"

    def set_roi_polygon(self, points):
        self.points = points


class _StatusBar:
    def __init__(self):
        self.message = ""

    def showMessage(self, message):
        self.message = message


class _MainWindowState:
    def __init__(self):
        self.roi_enabled = {0: True}
        self.config = {"roi_enabled": {0: True}}
        self.roi_checkboxes = {0: _CheckBox()}
        self.video_labels = {0: _VideoLabel()}
        self.detectors = {0: _RoiDetector()}
        self.saved = 0
        self._status_bar = _StatusBar()

    def _save_config(self):
        self.saved += 1

    def _on_roi_changed(self, points):
        return None

    def statusBar(self):
        return self._status_bar


def test_main_window_persists_roi_auto_disable_state():
    window = _MainWindowState()

    MainWindow._on_roi_auto_disabled(window, 0)

    assert window.roi_enabled == {0: False}
    assert window.config["roi_enabled"] == {0: False}
    assert window.roi_checkboxes[0].checked is False
    assert window.roi_checkboxes[0].blocked == [True, False]
    assert window.video_labels[0].show_roi is False
    assert window.video_labels[0].roi_changed.disconnect_calls == 1
    assert window.detectors[0].points is None
    assert window.saved == 1
    assert "已自动关闭" in window._status_bar.message


def test_video_thread_combines_identity_and_reader_metrics():
    frame = np.zeros((8, 12, 3), dtype=np.uint8)
    envelope = FrameEnvelope(
        original_frame=frame,
        frame_index=7,
        capture_time_ms=1_700_000_000_000.0,
        arrival_time_ms=1_700_000_000_002.0,
        segment_id=2,
    )
    reader = _EnvelopeReader(envelope)
    reader.get_info = lambda: {
        "queue_depth": 3,
        "actual_fps": 50.0,
        "dropped_frame_count": 4,
        "consumer_skipped_frame_count": 6,
        "discarded_frame_count": 10,
        "drop_rate": 0.2,
    }
    detector = _TimestampDetector()
    detector.last_frame_metrics = {
        "participants": 2,
        "track_fragments_merged": 1,
        "identity_ambiguities": 1,
    }
    thread = VideoThread(reader, detector, ui_skip=99)
    reader.thread = thread
    detector.thread = thread

    thread.run()

    assert thread.last_frame_metrics["participants"] == 2
    assert thread.last_frame_metrics["track_fragments_merged"] == 1
    assert thread.last_frame_metrics["identity_ambiguities"] == 1
    assert thread.last_frame_metrics["queue_depth"] == 3
    assert thread.last_frame_metrics["capture_fps"] == 50.0
    assert thread.last_frame_metrics["inference_fps"] > 0.0
    assert thread.last_frame_metrics["dropped_frames"] == 4
    assert thread.last_frame_metrics["consumer_skipped_frames"] == 6
    assert thread.last_frame_metrics["discarded_frames"] == 10
    assert thread.last_frame_metrics["drop_rate"] == 0.2


class _LatestEnvelopeReader(_EnvelopeReader):
    def __init__(self, envelope):
        super().__init__(envelope)
        self.fifo_calls = 0

    def get_latest_frame_envelope(self):
        self.calls += 1
        return self.envelope

    def get_frame_envelope(self):
        self.fifo_calls += 1
        return self.envelope


def test_video_thread_prefers_latest_frame_for_inference():
    frame = np.zeros((8, 12, 3), dtype=np.uint8)
    envelope = FrameEnvelope(
        original_frame=frame,
        frame_index=9,
        capture_time_ms=2_000.0,
        arrival_time_ms=2_005.0,
        segment_id=0,
    )
    reader = _LatestEnvelopeReader(envelope)
    detector = _TimestampDetector()
    thread = VideoThread(reader, detector, ui_skip=99)
    detector.thread = thread

    thread.run()

    assert reader.calls == 1
    assert reader.fifo_calls == 0
    assert detector.received_frame is frame


def test_video_thread_preserves_fifo_order_for_local_video():
    frame = np.zeros((8, 12, 3), dtype=np.uint8)
    envelope = FrameEnvelope(
        original_frame=frame,
        frame_index=9,
        capture_time_ms=2_000.0,
        arrival_time_ms=2_005.0,
        segment_id=0,
    )
    reader = _LatestEnvelopeReader(envelope)
    reader.source = "race.mkv"
    detector = _TimestampDetector()
    thread = VideoThread(reader, detector, ui_skip=99)
    detector.thread = thread

    thread.run()

    assert reader.calls == 0
    assert reader.fifo_calls == 1
    assert detector.received_frame is frame


class _PreviewReader:
    status = StreamStatus.CONNECTED

    def __init__(self):
        self.calls = []
        self.thread = None

    def get_frame_after(self, timestamp):
        self.calls.append(timestamp)
        self.thread._running = False
        return np.zeros((4, 6, 3), dtype=np.uint8), 12.5


def test_preview_thread_reads_latest_frame_without_detector():
    reader = _PreviewReader()
    thread = PreviewThread(reader, source_id=3, target_fps=30.0)
    reader.thread = thread
    emitted = []
    thread.frame_ready.connect(lambda frame, source_id: emitted.append((frame.shape, source_id)))

    thread.run()

    assert reader.calls == [0.0]
    assert emitted == [((4, 6, 3), 3)]
