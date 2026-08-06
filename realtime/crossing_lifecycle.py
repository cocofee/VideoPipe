"""Participant-keyed crossing admission lifecycle."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple

if __package__:
    from .participant_models import CrossingCandidate
else:
    from participant_models import CrossingCandidate


LifecycleKey = Tuple[str, int, str]


class CrossingState(str, Enum):
    """Auditable phases for one participant crossing lifecycle."""

    APPROACHING = "APPROACHING"
    IN_GATE = "IN_GATE"
    CROSSED = "CROSSED"
    COOLDOWN = "COOLDOWN"
    EXPIRED = "EXPIRED"


@dataclass(frozen=True)
class CrossingLifecycleSnapshot:
    """Immutable audit view of one participant's crossing state."""

    key: LifecycleKey
    state: CrossingState
    passage_index: int
    raw_track_ids: Tuple[int, ...]
    first_crossing_time_ms: float
    last_crossing_time_ms: float
    state_history: Tuple[CrossingState, ...]


@dataclass
class _LifecycleEntry:
    key: LifecycleKey
    state: CrossingState = CrossingState.APPROACHING
    passage_index: int = 0
    raw_track_ids: set[int] = field(default_factory=set)
    first_crossing_time_ms: float = 0.0
    last_crossing_time_ms: float = 0.0
    state_history: list[CrossingState] = field(
        default_factory=lambda: [CrossingState.APPROACHING]
    )


class CrossingLifecycle:
    """Admit crossings by stable participant identity instead of raw track ID."""

    VALID_MODES = frozenset({"finish_once", "multi_lap"})

    def __init__(self, mode: str = "finish_once", min_lap_interval_ms: float = 0.0):
        normalized_mode = str(mode or "").strip().lower()
        if normalized_mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported crossing lifecycle mode: {mode}")
        if float(min_lap_interval_ms) < 0.0:
            raise ValueError("min_lap_interval_ms must be non-negative")

        self.mode = normalized_mode
        self.min_lap_interval_ms = float(min_lap_interval_ms)
        self._entries: Dict[LifecycleKey, _LifecycleEntry] = {}

    @staticmethod
    def key_for(candidate: CrossingCandidate) -> LifecycleKey:
        return (
            str(candidate.session_id),
            int(candidate.source_id),
            str(candidate.participant_id),
        )

    @staticmethod
    def _transition(entry: _LifecycleEntry, state: CrossingState) -> None:
        entry.state = state
        entry.state_history.append(state)

    def _start_passage(
        self, entry: _LifecycleEntry, crossing_time_ms: float
    ) -> None:
        if entry.passage_index:
            self._transition(entry, CrossingState.APPROACHING)
        self._transition(entry, CrossingState.IN_GATE)
        self._transition(entry, CrossingState.CROSSED)
        entry.passage_index += 1
        if entry.passage_index == 1:
            entry.first_crossing_time_ms = crossing_time_ms
        entry.last_crossing_time_ms = crossing_time_ms
        self._transition(entry, CrossingState.COOLDOWN)

    def admit(self, candidate: CrossingCandidate) -> bool:
        """Return whether the candidate may allocate a new crossing event."""

        key = self.key_for(candidate)
        crossing_time_ms = float(candidate.crossing_time_ms)
        entry = self._entries.get(key)
        if entry is None:
            entry = _LifecycleEntry(key=key)
            self._entries[key] = entry
        elif self.mode == "finish_once":
            self._add_raw_track_evidence(entry, candidate.raw_track_id)
            if entry.state is not CrossingState.EXPIRED:
                self._transition(entry, CrossingState.EXPIRED)
            return False
        elif crossing_time_ms - entry.last_crossing_time_ms < self.min_lap_interval_ms:
            self._add_raw_track_evidence(entry, candidate.raw_track_id)
            return False
        else:
            self._transition(entry, CrossingState.EXPIRED)

        self._add_raw_track_evidence(entry, candidate.raw_track_id)
        self._start_passage(entry, crossing_time_ms)
        return True

    @staticmethod
    def _add_raw_track_evidence(
        entry: _LifecycleEntry, raw_track_id: Optional[int]
    ) -> None:
        if raw_track_id is not None:
            entry.raw_track_ids.add(int(raw_track_id))

    def snapshot(self, candidate: CrossingCandidate) -> Optional[CrossingLifecycleSnapshot]:
        """Return an immutable audit snapshot for the candidate's identity key."""

        return self.snapshot_for_key(self.key_for(candidate))

    def snapshot_for_key(
        self, key: LifecycleKey
    ) -> Optional[CrossingLifecycleSnapshot]:
        entry = self._entries.get(key)
        if entry is None:
            return None
        return CrossingLifecycleSnapshot(
            key=entry.key,
            state=entry.state,
            passage_index=entry.passage_index,
            raw_track_ids=tuple(sorted(entry.raw_track_ids)),
            first_crossing_time_ms=entry.first_crossing_time_ms,
            last_crossing_time_ms=entry.last_crossing_time_ms,
            state_history=tuple(entry.state_history),
        )
