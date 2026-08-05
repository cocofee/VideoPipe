import cv2

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
    def __init__(self):
        self.frames = ["first", "second"]
        self.position = 0
        self.opened = True
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame

    def get(self, prop):
        values = {
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            cv2.CAP_PROP_FPS: 0.0,
            cv2.CAP_PROP_FOURCC: 0,
        }
        return values.get(prop, 0)

    def set(self, prop, value):
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
