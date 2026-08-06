"""Frame identity and timing metadata for realtime processing."""

from dataclasses import dataclass, replace
from typing import Any, Optional


@dataclass(frozen=True)
class FrameEnvelope:
    """Immutable metadata wrapper that keeps the source frame untouched."""

    original_frame: Any
    frame_index: int
    capture_time_ms: float
    arrival_time_ms: float
    segment_id: int
    processing_time_ms: Optional[float] = None

    def with_processing_time(self, processing_time_ms: float) -> "FrameEnvelope":
        """Return a new envelope completed at the processing boundary."""
        return replace(self, processing_time_ms=float(processing_time_ms))
