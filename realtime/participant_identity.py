"""Conservative participant identity resolution for raw track observations."""

from dataclasses import dataclass
from math import hypot
from typing import Optional

if __package__:
    from .participant_models import BBox, ParticipantIdentity, ParticipantObservation
else:
    from participant_models import BBox, ParticipantIdentity, ParticipantObservation


@dataclass(frozen=True)
class IdentityConfig:
    """Thresholds for lightweight raw-track fragment matching."""

    event_profile_name: str = "current-event"
    max_match_gap_ms: float = 1_000.0
    min_iou: float = 0.25
    min_scale_ratio: float = 0.60
    max_center_distance_ratio: float = 0.75
    min_match_score: float = 0.45
    ambiguity_margin: float = 0.05
    simultaneous_window_ms: float = 0.0
    min_direction_motion_ratio: float = 0.05
    min_direction_cosine: float = 0.0
    max_exact_track_jump_ratio: float = 1.0
    hard_exact_track_jump_ratio: float = 4.0
    max_exact_track_reverse_cosine: float = -0.25
    bib_layout_weight: float = 0.03
    participant_retention_ms: float = 30_000.0
    raw_track_retention_ms: float = 5_000.0
    ambiguity_retention_ms: float = 5_000.0


@dataclass(frozen=True)
class IdentityResolution:
    participant: ParticipantIdentity
    created: bool
    merged_raw_track: bool
    match_score: float
    reasons: tuple[str, ...]

    @property
    def participant_id(self) -> str:
        return self.participant.participant_id


@dataclass
class _ParticipantState:
    source_id: int
    segment_id: int
    last_center: tuple[float, float]
    velocity: tuple[float, float] = (0.0, 0.0)
    bib_layout_signature: Optional[tuple[float, float, float, float]] = None


@dataclass(frozen=True)
class _MatchCandidate:
    participant: ParticipantIdentity
    score: float
    reasons: tuple[str, ...]


class ParticipantIdentityManager:
    """Map tracker fragments to stable participants without a ReID model."""

    def __init__(self, config: Optional[IdentityConfig] = None):
        self.config = config or IdentityConfig()
        self._next_participant_number = 1
        self._participants: dict[str, ParticipantIdentity] = {}
        self._states: dict[str, _ParticipantState] = {}
        self._raw_track_map: dict[tuple[int, int, int], str] = {}
        self._raw_track_last_seen_ms: dict[tuple[int, int, int], float] = {}
        self._fragment_merge_count = 0
        self._ambiguity_count = 0

    def resolve(
        self,
        observation: ParticipantObservation,
        excluded_participant_ids: Optional[set[str]] = None,
    ) -> IdentityResolution:
        """Resolve one observation to an existing or newly created participant."""

        self.prune_expired(observation.capture_time_ms)
        excluded_participant_ids = set(excluded_participant_ids or ())
        raw_track_key = self._raw_track_key(observation)
        if raw_track_key is not None:
            participant_id = self._raw_track_map.get(raw_track_key)
            if participant_id is not None and participant_id not in excluded_participant_ids:
                participant = self._participants.get(participant_id)
                if participant is None:
                    self._raw_track_map.pop(raw_track_key, None)
                    self._raw_track_last_seen_ms.pop(raw_track_key, None)
                else:
                    discontinuity_reasons = self._exact_track_discontinuity_reasons(
                        participant,
                        observation,
                    )
                    if discontinuity_reasons:
                        return self._create_participant(
                            observation,
                            identity_status="ACTIVE",
                            reasons=discontinuity_reasons,
                        )
                    self._update_participant(participant, observation)
                    return IdentityResolution(
                        participant=participant,
                        created=False,
                        merged_raw_track=False,
                        match_score=1.0,
                        reasons=("exact_raw_track",),
                    )

        candidates = sorted(
            self._match_candidates(observation, excluded_participant_ids),
            key=lambda candidate: candidate.score,
            reverse=True,
        )
        if candidates:
            best = candidates[0]
            if (
                len(candidates) > 1
                and best.score - candidates[1].score <= self.config.ambiguity_margin
            ):
                reasons = (
                    "ambiguous_match",
                    *(
                        f"candidate={candidate.participant.participant_id},"
                        f"score={candidate.score:.3f}"
                        for candidate in candidates[:2]
                    ),
                )
                return self._create_participant(
                    observation,
                    identity_status="AMBIGUOUS",
                    reasons=reasons,
                )

            participant = best.participant
            self._update_participant(participant, observation)
            if raw_track_key is not None:
                self._fragment_merge_count += 1
            return IdentityResolution(
                participant=participant,
                created=False,
                merged_raw_track=raw_track_key is not None,
                match_score=best.score,
                reasons=best.reasons,
            )

        return self._create_participant(
            observation,
            identity_status="ACTIVE",
            reasons=("no_matching_participant",),
        )

    def _exact_track_discontinuity_reasons(
        self,
        participant: ParticipantIdentity,
        observation: ParticipantObservation,
    ) -> tuple[str, ...]:
        """Detect tracker-ID reuse after an athlete has left the frame."""
        if participant.last_bbox is None:
            return ()
        state = self._states.get(participant.participant_id)
        if state is None:
            return ()

        elapsed_ms = observation.capture_time_ms - participant.last_seen_ms
        if elapsed_ms < 0.0:
            return ()

        observed_center = _center(observation.participant_bbox)
        jump_ratio = _center_distance_ratio(
            state.last_center,
            observed_center,
            participant.last_bbox,
            observation.participant_bbox,
        )
        if jump_ratio < self.config.max_exact_track_jump_ratio:
            return ()

        direction_cosine = None
        if elapsed_ms > 0.0:
            direction_cosine = _direction_cosine(
                state,
                observed_center,
                elapsed_ms,
                participant.last_bbox,
                observation.participant_bbox,
                self.config.min_direction_motion_ratio,
            )
        hard_jump = jump_ratio >= self.config.hard_exact_track_jump_ratio
        reversed_jump = (
            direction_cosine is not None
            and direction_cosine <= self.config.max_exact_track_reverse_cosine
        )
        if not (hard_jump or reversed_jump):
            return ()

        reasons = [
            "raw_track_discontinuity",
            f"elapsed_ms={elapsed_ms:.1f}",
            f"center_jump_ratio={jump_ratio:.3f}",
        ]
        if direction_cosine is not None:
            reasons.append(f"direction_cosine={direction_cosine:.3f}")
        return tuple(reasons)

    def _match_candidates(
        self,
        observation: ParticipantObservation,
        excluded_participant_ids: Optional[set[str]] = None,
    ) -> list[_MatchCandidate]:
        excluded_participant_ids = set(excluded_participant_ids or ())
        candidates = []
        for participant_id, participant in self._participants.items():
            if participant_id in excluded_participant_ids:
                continue
            state = self._states[participant_id]
            if (
                state.source_id != observation.source_id
                or state.segment_id != observation.segment_id
                or participant.last_bbox is None
            ):
                continue

            elapsed_ms = observation.capture_time_ms - participant.last_seen_ms
            if elapsed_ms <= self.config.simultaneous_window_ms:
                continue
            if elapsed_ms > self.config.max_match_gap_ms:
                continue

            scale_ratio = _scale_ratio(participant.last_bbox, observation.participant_bbox)
            if scale_ratio < self.config.min_scale_ratio:
                continue

            predicted_center = (
                state.last_center[0] + state.velocity[0] * elapsed_ms,
                state.last_center[1] + state.velocity[1] * elapsed_ms,
            )
            observed_center = _center(observation.participant_bbox)
            distance_ratio = _center_distance_ratio(
                predicted_center,
                observed_center,
                participant.last_bbox,
                observation.participant_bbox,
            )
            iou = _iou(participant.last_bbox, observation.participant_bbox)
            if (
                iou < self.config.min_iou
                and distance_ratio > self.config.max_center_distance_ratio
            ):
                continue

            direction_cosine = _direction_cosine(
                state,
                observed_center,
                elapsed_ms,
                participant.last_bbox,
                observation.participant_bbox,
                self.config.min_direction_motion_ratio,
            )
            if (
                direction_cosine is not None
                and direction_cosine < self.config.min_direction_cosine
            ):
                continue

            proximity = max(
                0.0,
                1.0 - distance_ratio / self.config.max_center_distance_ratio,
            )
            score = 0.50 * iou + 0.25 * scale_ratio + 0.25 * proximity
            reasons = [
                f"elapsed_ms={elapsed_ms:.1f}",
                f"iou={iou:.3f}",
                f"scale_ratio={scale_ratio:.3f}",
                f"predicted_center_distance_ratio={distance_ratio:.3f}",
            ]
            if direction_cosine is not None:
                reasons.append(f"direction_cosine={direction_cosine:.3f}")

            bib_layout_signature = _bib_layout_signature(
                observation.participant_bbox,
                observation.bib_bboxes,
            )
            if state.bib_layout_signature is not None and bib_layout_signature is not None:
                bib_similarity = _bib_layout_similarity(
                    state.bib_layout_signature,
                    bib_layout_signature,
                )
                score = min(
                    1.0,
                    score + self.config.bib_layout_weight * bib_similarity,
                )
                reasons.append(f"bib_layout_similarity={bib_similarity:.3f}")
            if score < self.config.min_match_score:
                continue

            candidates.append(
                _MatchCandidate(
                    participant=participant,
                    score=score,
                    reasons=tuple(reasons),
                )
            )
        return candidates

    def _create_participant(
        self,
        observation: ParticipantObservation,
        identity_status: str,
        reasons: tuple[str, ...],
    ) -> IdentityResolution:
        participant_id = f"P{self._next_participant_number:06d}"
        self._next_participant_number += 1
        raw_track_ids = (
            {observation.raw_track_id}
            if observation.raw_track_id is not None
            else set()
        )
        participant = ParticipantIdentity(
            participant_id=participant_id,
            event_profile_name=self.config.event_profile_name,
            raw_track_ids=raw_track_ids,
            first_seen_ms=observation.capture_time_ms,
            last_seen_ms=observation.capture_time_ms,
            last_bbox=observation.participant_bbox,
            identity_status=identity_status,
        )
        self._participants[participant_id] = participant
        self._states[participant_id] = _ParticipantState(
            source_id=observation.source_id,
            segment_id=observation.segment_id,
            last_center=_center(observation.participant_bbox),
            bib_layout_signature=_bib_layout_signature(
                observation.participant_bbox,
                observation.bib_bboxes,
            ),
        )
        raw_track_key = self._raw_track_key(observation)
        if raw_track_key is not None:
            self._raw_track_map[raw_track_key] = participant_id
            self._raw_track_last_seen_ms[raw_track_key] = observation.capture_time_ms
        if identity_status == "AMBIGUOUS":
            self._ambiguity_count += 1
        return IdentityResolution(
            participant=participant,
            created=True,
            merged_raw_track=False,
            match_score=0.0,
            reasons=reasons,
        )

    def _update_participant(
        self,
        participant: ParticipantIdentity,
        observation: ParticipantObservation,
    ) -> None:
        state = self._states[participant.participant_id]
        observed_center = _center(observation.participant_bbox)
        elapsed_ms = observation.capture_time_ms - participant.last_seen_ms
        if elapsed_ms > 0:
            state.velocity = (
                (observed_center[0] - state.last_center[0]) / elapsed_ms,
                (observed_center[1] - state.last_center[1]) / elapsed_ms,
            )
        state.last_center = observed_center
        bib_layout_signature = _bib_layout_signature(
            observation.participant_bbox,
            observation.bib_bboxes,
        )
        if bib_layout_signature is not None:
            state.bib_layout_signature = bib_layout_signature
        participant.last_seen_ms = observation.capture_time_ms
        participant.last_bbox = observation.participant_bbox
        if observation.raw_track_id is not None:
            participant.raw_track_ids.add(observation.raw_track_id)
            raw_track_key = self._raw_track_key(observation)
            if raw_track_key is not None:
                self._raw_track_map[raw_track_key] = participant.participant_id
                self._raw_track_last_seen_ms[raw_track_key] = observation.capture_time_ms

    def prune_expired(self, current_time_ms: float) -> int:
        """Prune stale identity indexes while preserving persisted race evidence."""

        current_time_ms = float(current_time_ms)
        raw_track_retention_ms = max(0.0, float(self.config.raw_track_retention_ms))
        stale_raw_tracks = [
            key
            for key, last_seen_ms in self._raw_track_last_seen_ms.items()
            if current_time_ms - last_seen_ms > raw_track_retention_ms
        ]
        for key in stale_raw_tracks:
            self._raw_track_map.pop(key, None)
            self._raw_track_last_seen_ms.pop(key, None)

        stale_participants = []
        for participant_id, participant in self._participants.items():
            retention_ms = (
                self.config.ambiguity_retention_ms
                if participant.identity_status == "AMBIGUOUS"
                else self.config.participant_retention_ms
            )
            if current_time_ms - participant.last_seen_ms > max(0.0, float(retention_ms)):
                stale_participants.append(participant_id)

        if not stale_participants:
            return 0

        stale_ids = set(stale_participants)
        for participant_id in stale_participants:
            self._participants.pop(participant_id, None)
            self._states.pop(participant_id, None)
        for key, participant_id in list(self._raw_track_map.items()):
            if participant_id in stale_ids:
                self._raw_track_map.pop(key, None)
                self._raw_track_last_seen_ms.pop(key, None)
        return len(stale_participants)

    def runtime_metrics(self) -> dict[str, int]:
        """Return lightweight identity-state counters for periodic monitoring."""

        return {
            "active_participants": len(self._participants),
            "raw_track_mappings": len(self._raw_track_map),
            "appearance_summaries": len(self._states),
            "ambiguity_records": sum(
                participant.identity_status == "AMBIGUOUS"
                for participant in self._participants.values()
            ),
            "track_fragments_merged": self._fragment_merge_count,
            "identity_ambiguities": self._ambiguity_count,
        }

    @staticmethod
    def _raw_track_key(
        observation: ParticipantObservation,
    ) -> Optional[tuple[int, int, int]]:
        if observation.raw_track_id is None:
            return None
        return (
            observation.source_id,
            observation.segment_id,
            observation.raw_track_id,
        )


def _center(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _area(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def _scale_ratio(first: BBox, second: BBox) -> float:
    first_area = _area(first)
    second_area = _area(second)
    larger_area = max(first_area, second_area)
    if larger_area == 0:
        return 0.0
    return min(first_area, second_area) / larger_area


def _iou(first: BBox, second: BBox) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = float(max(0, right - left) * max(0, bottom - top))
    union = _area(first) + _area(second) - intersection
    if union == 0:
        return 0.0
    return intersection / union


def _center_distance_ratio(
    first_center: tuple[float, float],
    second_center: tuple[float, float],
    first_bbox: BBox,
    second_bbox: BBox,
) -> float:
    distance = hypot(
        first_center[0] - second_center[0],
        first_center[1] - second_center[1],
    )
    first_diagonal = hypot(first_bbox[2] - first_bbox[0], first_bbox[3] - first_bbox[1])
    second_diagonal = hypot(
        second_bbox[2] - second_bbox[0], second_bbox[3] - second_bbox[1]
    )
    normalizer = max(first_diagonal, second_diagonal, 1.0)
    return distance / normalizer


def _direction_cosine(
    state: _ParticipantState,
    observed_center: tuple[float, float],
    elapsed_ms: float,
    first_bbox: BBox,
    second_bbox: BBox,
    min_motion_ratio: float,
) -> Optional[float]:
    expected_motion = (
        state.velocity[0] * elapsed_ms,
        state.velocity[1] * elapsed_ms,
    )
    observed_motion = (
        observed_center[0] - state.last_center[0],
        observed_center[1] - state.last_center[1],
    )
    expected_magnitude = hypot(*expected_motion)
    observed_magnitude = hypot(*observed_motion)
    first_diagonal = hypot(first_bbox[2] - first_bbox[0], first_bbox[3] - first_bbox[1])
    second_diagonal = hypot(
        second_bbox[2] - second_bbox[0], second_bbox[3] - second_bbox[1]
    )
    normalizer = max(first_diagonal, second_diagonal, 1.0)
    if (
        expected_magnitude / normalizer < min_motion_ratio
        or observed_magnitude / normalizer < min_motion_ratio
    ):
        return None
    dot_product = (
        expected_motion[0] * observed_motion[0]
        + expected_motion[1] * observed_motion[1]
    )
    return dot_product / (expected_magnitude * observed_magnitude)


def _bib_layout_signature(
    participant_bbox: BBox,
    bib_bboxes: tuple[BBox, ...],
) -> Optional[tuple[float, float, float, float]]:
    if not bib_bboxes:
        return None
    participant_width = participant_bbox[2] - participant_bbox[0]
    participant_height = participant_bbox[3] - participant_bbox[1]
    if participant_width <= 0 or participant_height <= 0:
        return None

    normalized = []
    for bib_bbox in bib_bboxes:
        bib_center = _center(bib_bbox)
        normalized.append(
            (
                (bib_center[0] - participant_bbox[0]) / participant_width,
                (bib_center[1] - participant_bbox[1]) / participant_height,
                (bib_bbox[2] - bib_bbox[0]) / participant_width,
                (bib_bbox[3] - bib_bbox[1]) / participant_height,
            )
        )
    count = len(normalized)
    return tuple(
        sum(signature[index] for signature in normalized) / count
        for index in range(4)
    )


def _bib_layout_similarity(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    distance = hypot(
        hypot(first[0] - second[0], first[1] - second[1]),
        hypot(first[2] - second[2], first[3] - second[3]),
    )
    return max(0.0, 1.0 - distance)
