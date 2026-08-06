import cv2
import numpy as np

from realtime.stream_reader import StreamReader, StreamStatus


class _CopyCountingFrame:
    def __init__(self):
        self.copy_count = 0

    def copy(self):
        self.copy_count += 1
        return object()


def test_get_frame_after_skips_copy_until_timestamp_advances():
    reader = StreamReader(source="unused")
    source_frame = _CopyCountingFrame()
    reader._frame = source_frame
    reader._frame_time = 10.0

    stale_frame, stale_timestamp = reader.get_frame_after(10.0)

    assert stale_frame is None
    assert stale_timestamp == 10.0
    assert source_frame.copy_count == 0

    new_frame, new_timestamp = reader.get_frame_after(9.0)

    assert new_frame is not None
    assert new_timestamp == 10.0
    assert source_frame.copy_count == 1


class _FiniteVideoCapture:
    def __init__(self, frames=None, *, fps=0.0, media_time_ms=None, on_exhausted=None):
        self.frames = list(frames or ["first", "second"])
        self.fps = fps
        self.media_time_ms = media_time_ms
        self.on_exhausted = on_exhausted
        self.position = 0
        self.opened = True
        self.released = False
        self.set_calls = []

    def isOpened(self):
        return self.opened

    def read(self):
        if self.position >= len(self.frames):
            if self.on_exhausted is not None:
                self.on_exhausted()
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame

    def get(self, prop):
        values = {
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            cv2.CAP_PROP_FPS: self.fps,
            cv2.CAP_PROP_FOURCC: 0,
            cv2.CAP_PROP_POS_MSEC: (
                self.media_time_ms
                if self.media_time_ms is not None
                else max(0, self.position - 1) * 40.0
            ),
        }
        return values.get(prop, 0)

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.position = int(value)
        return True

    def release(self):
        self.opened = False
        self.released = True


def test_local_video_plays_first_frame_once_and_stops_at_eof(monkeypatch):
    capture = _FiniteVideoCapture()
    monkeypatch.setattr("realtime.stream_reader.cv2.VideoCapture", lambda source: capture)
    monkeypatch.setattr("realtime.stream_reader.time.sleep", lambda seconds: None)
    reader = StreamReader("race.mp4")
    frames = []
    reader.set_on_frame(frames.append)

    assert reader._connect() is True

    reconnect_calls = []

    def unexpected_reconnect():
        reconnect_calls.append(True)
        reader._running = False
        return False

    reader._connect = unexpected_reconnect
    reader._running = True
    reader._read_loop()

    assert frames == ["first", "second"]
    assert reconnect_calls == []
    assert reader.status == StreamStatus.ENDED
    assert capture.released is True


def test_video_file_source_detection_excludes_cameras_and_rtsp():
    assert StreamReader.is_video_file_source("race.mp4") is True
    assert StreamReader.is_video_file_source("rtsp://camera/live") is False
    assert StreamReader.is_video_file_source(0) is False


def test_bounded_queue_drops_oldest_frame_and_reports_metrics(monkeypatch):
    capture = _FiniteVideoCapture(["first", "second", "third"])
    monkeypatch.setattr("realtime.stream_reader.cv2.VideoCapture", lambda source: capture)
    monkeypatch.setattr("realtime.stream_reader.time.sleep", lambda seconds: None)
    reader = StreamReader("race.mp4", queue_size=2, overflow_policy="drop_oldest")

    assert reader._connect() is True

    reader._running = True
    reader._read_loop()

    assert reader.queue_capacity == 2
    assert reader.buffer_size == 1
    assert reader.overflow_policy == "drop_oldest"
    assert reader.dropped_frame_count == 1
    assert reader.queue_depth == 2

    envelopes = [reader.get_frame_envelope(), reader.get_frame_envelope()]

    assert [envelope.original_frame for envelope in envelopes] == ["second", "third"]
    assert [envelope.frame_index for envelope in envelopes] == [1, 2]
    assert [envelope.capture_time_ms for envelope in envelopes] == [40.0, 80.0]
    assert all(envelope.arrival_time_ms > 0 for envelope in envelopes)
    assert reader.queue_depth == 0
    assert reader.get_info()["dropped_frame_count"] == 1


class _LiveCapture:
    def __init__(self, frames, on_exhausted=None):
        self.frames = list(frames)
        self.on_exhausted = on_exhausted
        self.opened = True

    def isOpened(self):
        return self.opened

    def read(self):
        if self.frames:
            return True, self.frames.pop(0)
        if self.on_exhausted is not None:
            self.on_exhausted()
        return False, None

    def release(self):
        self.opened = False


def test_realtime_reconnect_increments_segment_id(monkeypatch):
    reader = StreamReader(0, queue_size=4)
    first_capture = _LiveCapture(["segment-zero"])
    second_capture = _LiveCapture(["segment-one"], on_exhausted=lambda: setattr(reader, "_running", False))
    reader._cap = first_capture
    reconnect_calls = []

    def reconnect():
        reconnect_calls.append(True)
        reader._cap = second_capture
        return True

    monkeypatch.setattr(reader, "_connect", reconnect)
    monkeypatch.setattr("realtime.stream_reader.time.sleep", lambda seconds: None)
    reader._running = True

    reader._read_loop()

    envelopes = [reader.get_frame_envelope(), reader.get_frame_envelope()]
    assert reconnect_calls == [True]
    assert [envelope.original_frame for envelope in envelopes] == ["segment-zero", "segment-one"]
    assert [envelope.segment_id for envelope in envelopes] == [0, 1]
    assert [envelope.frame_index for envelope in envelopes] == [0, 1]


def test_queue_size_is_separate_from_capture_buffer_size(monkeypatch):
    capture = _FiniteVideoCapture(["probe"])
    monkeypatch.setattr("realtime.stream_reader.cv2.VideoCapture", lambda source: capture)
    reader = StreamReader("race.mp4", buffer_size=3, queue_size=5)

    assert reader._connect() is True

    assert reader.buffer_size == 3
    assert reader.queue_capacity == 5
    assert (cv2.CAP_PROP_BUFFERSIZE, 3) in capture.set_calls
    assert StreamReader("unused").queue_capacity == 8


def test_live_timestamps_and_legacy_frame_time_use_wall_clock(monkeypatch):
    wall_time = 1_700_000_000.25
    frame = np.zeros((8, 12, 3), dtype=np.uint8)
    reader = StreamReader(0, queue_size=2)
    capture = _LiveCapture([frame], on_exhausted=lambda: setattr(reader, "_running", False))
    reader._cap = capture
    monkeypatch.setattr("realtime.stream_reader.time.time", lambda: wall_time)
    monkeypatch.setattr("realtime.stream_reader.time.monotonic", lambda: 42.0)
    reader._running = True

    reader._read_loop()

    envelope = reader.get_frame_envelope()
    legacy_frame, legacy_timestamp = reader.get_frame_with_time()
    assert envelope.original_frame is frame
    assert envelope.capture_time_ms == wall_time * 1000.0
    assert envelope.arrival_time_ms == wall_time * 1000.0
    assert legacy_frame.shape == frame.shape
    assert legacy_timestamp == wall_time
    assert legacy_timestamp != 42.0


def test_local_video_invalid_media_time_falls_back_to_frame_index_and_fps(monkeypatch):
    capture = _FiniteVideoCapture(
        [np.zeros((4, 4, 3), dtype=np.uint8), np.ones((4, 4, 3), dtype=np.uint8)],
        fps=25.0,
        media_time_ms=-1.0,
    )
    monkeypatch.setattr("realtime.stream_reader.cv2.VideoCapture", lambda source: capture)
    monkeypatch.setattr("realtime.stream_reader.time.sleep", lambda seconds: None)
    reader = StreamReader("race.mp4", queue_size=2)

    assert reader._connect() is True
    reader._running = True
    reader._read_loop()

    envelopes = [reader.get_frame_envelope(), reader.get_frame_envelope()]
    assert [envelope.capture_time_ms for envelope in envelopes] == [0.0, 40.0]


def test_reconnect_after_initial_probe_starts_a_new_segment(monkeypatch):
    reader = StreamReader(0, queue_size=2)
    first_capture = _FiniteVideoCapture([np.zeros((2, 2, 3), dtype=np.uint8)])
    second_capture = _FiniteVideoCapture(
        [np.ones((2, 2, 3), dtype=np.uint8), np.full((2, 2, 3), 2, dtype=np.uint8)],
        on_exhausted=lambda: setattr(reader, "_running", False),
    )
    captures = iter([first_capture, second_capture])
    monkeypatch.setattr("realtime.stream_reader.cv2.VideoCapture", lambda *args: next(captures))
    monkeypatch.setattr("realtime.stream_reader.time.sleep", lambda seconds: None)

    assert reader._connect() is True
    reader._running = True
    reader._read_loop()

    envelope = reader.get_frame_envelope()
    assert envelope.frame_index == 0
    assert envelope.segment_id == 1
