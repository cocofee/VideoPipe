from realtime.stream_reader import StreamReader
from realtime.independent_counting import FrameStamp


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


def test_packet_reader_preserves_metadata_and_accepts_reconnect_sequence_reset():
    reader = StreamReader("unused")
    frame = _CopyCountingFrame()
    reader._frame = frame
    reader._frame_stamp = FrameStamp(10, "old", 1000, 1001)
    assert reader.get_frame_packet_after(10, "old")[0] is None
    reader._frame_stamp = FrameStamp(1, "new", 1040, 1041)
    result, stamp = reader.get_frame_packet_after(10, "old")
    assert result is not None and stamp.generation == "new" and stamp.sequence == 1
    assert stamp.observed_at_ms == 1040 and frame.copy_count == 1


def test_stream_log_label_omits_credentials_and_query():
    reader = StreamReader("rtsp://user:secret@example.invalid/live?token=secret")
    assert reader._source_label == "rtsp://example.invalid/live"
