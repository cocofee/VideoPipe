import numpy as np

from realtime.frame_envelope import FrameEnvelope
from realtime.main_window import VideoThread
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
