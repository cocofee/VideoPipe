"""Core participant observation and identity contracts."""

from dataclasses import dataclass, field
from typing import Optional


BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class BibEvidence:
    """Bib evidence captured from one source frame."""

    bbox: BBox
    frame_index: int
    capture_time_ms: float
    confidence: float
    crop_quality: float
    text_candidate: Optional[str] = None


@dataclass(frozen=True)
class ParticipantObservation:
    """One immutable frame-level observation of a physical participant."""

    source_id: int
    segment_id: int
    frame_index: int
    capture_time_ms: float
    raw_track_id: Optional[int]
    participant_bbox: BBox
    person_bbox: Optional[BBox]
    equipment_bbox: Optional[BBox]
    bib_bboxes: tuple[BBox, ...]
    confidence: float


@dataclass
class ParticipantIdentity:
    """Stable participant state assembled from one or more observations."""

    participant_id: str
    event_profile_name: str
    raw_track_ids: set[int] = field(default_factory=set)
    first_seen_ms: float = 0.0
    last_seen_ms: float = 0.0
    last_bbox: Optional[BBox] = None
    identity_status: str = "ACTIVE"
    bib_evidence: list[BibEvidence] = field(default_factory=list)


@dataclass(frozen=True)
class CrossingCandidate:
    """A participant-level candidate awaiting crossing lifecycle admission."""

    session_id: str
    source_id: int
    participant_id: str
    raw_track_id: Optional[int]
    crossing_time_ms: float
