from realtime.stream_reader import StreamReader


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
