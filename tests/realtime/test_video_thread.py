from realtime.main_window import VideoThread


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
