from realtime.independent_counting import FrameStamp
from realtime.main_window import VideoThread


class _Reader:
    def __init__(self):
        self.calls = []

    def get_frame_packet_after(self, sequence, generation):
        self.calls.append((sequence, generation))
        index = len(self.calls)
        return object(), FrameStamp(index, "camera", 1000 + index * 40, 9000)


class _Detector:
    def __init__(self):
        self.frames = []

    def process_frame(self, frame, timestamp=None, *, frame_stamp=None):
        self.frames.append((timestamp, frame_stamp))
        if len(self.frames) == 2:
            self.thread._running = False
        return [], [], []


def test_video_thread_forwards_source_time_and_provenance():
    reader, detector = _Reader(), _Detector()
    thread = VideoThread(reader, detector, skip_frames=1)
    detector.thread = thread
    thread.run()
    assert reader.calls == [(0, ""), (1, "camera")]
    assert [timestamp for timestamp, _ in detector.frames] == [0.0, 0.04]
    assert detector.frames[-1][1].sequence == 2
