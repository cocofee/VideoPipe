from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from realtime.frame_envelope import FrameEnvelope


def test_frame_envelope_is_immutable_and_preserves_original_frame():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    envelope = FrameEnvelope(
        original_frame=frame,
        frame_index=7,
        capture_time_ms=240.0,
        arrival_time_ms=250.0,
        segment_id=2,
    )

    assert envelope.original_frame is frame
    assert envelope.original_frame.shape == (1080, 1920, 3)

    with pytest.raises(FrozenInstanceError):
        envelope.segment_id = 3


def test_with_processing_time_returns_updated_envelope_without_mutating_original():
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    envelope = FrameEnvelope(
        original_frame=frame,
        frame_index=1,
        capture_time_ms=40.0,
        arrival_time_ms=45.0,
        segment_id=0,
    )

    processed = envelope.with_processing_time(12.5)

    assert envelope.processing_time_ms is None
    assert processed.processing_time_ms == 12.5
    assert processed.original_frame is frame
    assert processed.frame_index == envelope.frame_index
    assert processed.capture_time_ms == envelope.capture_time_ms
    assert processed.arrival_time_ms == envelope.arrival_time_ms
    assert processed.segment_id == envelope.segment_id
