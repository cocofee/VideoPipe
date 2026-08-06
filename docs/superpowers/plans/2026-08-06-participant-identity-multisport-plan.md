# Participant Identity and Multi-Sport Evolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立一套持续用于不同比赛的“运动员 + 号码布 + 稳定身份 + 过线事件”公共系统；每次有什么比赛就运行和验证什么比赛，并把新问题继续反馈到同一个算法中。

**Architecture:** 保留 `StreamReader -> Detector -> EventRecorder/Database -> OCRManager -> MainWindow` 主链路，在检测框和过线事件之间新增稳定的 `participant_id` 身份层。所有常规比赛先进入同一条 `athlete_bib` 管线：检测运动员、维持身份、判断过线、保存原图证据、异步识别号码布。`EventProfile` 只描述本场比赛的器材证据、号码布区域、方向、密度和圈次规则；它不是另一套算法。原始 `track_id` 只表示跟踪器片段，一个 `participant_id` 可以包含多个原始轨迹。

**Tech Stack:** Python 3.12, PyQt5, OpenCV, Ultralytics YOLO/ByteTrack, PaddleOCR, SQLite, JSON regression reports, pytest.

---

## 1. Confirmed Scope

### Current execution scope

- The application is not developed in a fixed sport order.
- Whichever competition is currently available becomes the next real validation and optimization input.
- Cycling, skating, running, pentathlon, and similar events use the same athlete/bib identity pipeline unless real evidence proves a different detector representation is required.
- Sport names are event metadata. They do not select separate copies of the detector, tracker, database, OCR pipeline, or UI.
- The primary target is participant identity stability, not OCR coverage.
- One physical athlete may produce many frame boxes and several raw tracker IDs, but must map to one stable `participant_id`.
- A stable participant may have one crossing in `finish_once` mode or multiple passage records in `multi_lap` mode.
- Missing or failed OCR must preserve the participant, crossing event, and evidence.
- Available videos are automated regression inputs. Operators do not review every video after every algorithm change.
- CycleRace remains the only official timing/result authority. VideoPipe remains independent auxiliary evidence.

### Continuous competition-driven roadmap

1. Implement the common participant identity and event lifecycle once.
2. Use the next available competition, regardless of sport name, as the current real-world validation input.
3. Create a small `EventProfile` for that competition from its actual camera angle, bib placement, optional equipment, direction, density, and lap rules.
4. Run the common pipeline, record missed athletes, duplicate identities, false participants, bib association errors, OCR conflicts, and performance metrics.
5. Fix the common algorithm first; add event-specific behavior only when the evidence shows it cannot be expressed as configuration.
6. Add the confirmed failure case to the unified regression set so an improvement for one competition cannot silently damage another.

### Explicitly excluded from this plan

- A speculative marathon-only dense-crowd implementation before an actual competition demonstrates that the common athlete/bib pipeline is insufficient.
- Live communication between the CycleRace and VideoPipe laptops.
- Automatic writes into CycleRace timing/result tables.
- Multi-camera or cross-camera identity fusion.
- Face recognition as an athlete identity source.
- A new heavyweight ReID dependency running on every frame.
- Separate copies of the detector, tracker, database, OCR pipeline, or UI for each non-dense sport.
- Hardcoded sport branches that duplicate behavior already represented by `EventProfile` configuration.

## 2. First-Principles Model

The design starts from observable facts, not sport names, UI rows, tracker IDs, or existing heuristics.

### Principle 1: The athlete is the physical entity

A real athlete exists independently of detector boxes, tracker IDs, OCR output, and database rows. The durable system identity is `participant_id`.

```text
one physical athlete
  != one frame box
  != one raw track_id
  != one OCR value
  != one crossing row
```

### Principle 2: A frame box is only an observation

Each frame provides incomplete and noisy evidence. One athlete may have zero, one, or several candidate boxes in a frame. The system must normalize those candidates into participant observations before identity and event logic.

```python
@dataclass(frozen=True)
class ParticipantObservation:
    source_id: int
    segment_id: int
    frame_index: int
    capture_time_ms: float
    raw_track_id: int | None
    participant_bbox: tuple[int, int, int, int]
    person_bbox: tuple[int, int, int, int] | None
    equipment_bbox: tuple[int, int, int, int] | None
    bib_bboxes: tuple[tuple[int, int, int, int], ...]
    confidence: float
```

`participant_bbox` is the normalized object used for tracking/display. Human, bicycle/equipment, and bib evidence remain separate so training and association do not depend on inconsistent box meanings across competitions.

### Principle 3: Identity is a temporal inference

`participant_id` is inferred from a sequence of observations using time, position, motion, overlap, scale, simultaneous visibility, optional appearance, and bib evidence. Raw tracker IDs are inputs to this inference, never permanent identity.

### Principle 4: Crossing is a trajectory event

A crossing is not “a box touched the line in one frame.” It is a validated transition of one stable participant trajectory from one side of the configured line/gate to the other, with direction and lifecycle state.

### Principle 5: The bib is evidence, not existence

A bib crop and OCR text are optional evidence attached to `participant_id`. Missing, blurred, occluded, or conflicting bib results cannot erase an athlete, create another athlete, or allocate another crossing event.

### Principle 6: Sport is context, not a separate pipeline

Cycling, skating, running, pentathlon, and other ordinary finish-line events all begin with the same `athlete_bib` pipeline. An `EventProfile` supplies only observable differences such as equipment evidence, likely bib regions, movement direction, density, and lap rules.

### Principle 7: Invisible information cannot be recovered

If the camera never records usable pixels for an athlete or bib, software cannot reconstruct them reliably. Camera placement, focus, shutter/exposure, resolution, finish-line geometry, and source-video retention are part of the recognition system, not external details.

### Principle 8: Realtime resources have a strict priority

```text
frame acquisition
  -> original evidence preservation
  -> athlete observation
  -> stable identity
  -> crossing event
  -> asynchronous bib detection/OCR
  -> UI rendering
```

A lower-priority stage may be delayed or degraded under load, but it cannot discard a higher-priority result.

### Failure classification

Every field problem must be assigned to one primary layer before code changes:

```text
OBSERVATION   missed athlete, duplicate same-frame boxes, background/gate box
IDENTITY      one athlete split into several identities, two athletes merged
PASSAGE       missed crossing, duplicate crossing, invalid recross/lap
BIB           wrong owner, unreadable crop, OCR conflict, wrong confirmation
EVIDENCE      missing original frame/crop/metadata
PERFORMANCE   queue growth, dropped frames, UI stall, reconnect failure
```

A repair changes one primary layer first and reruns all downstream regression gates. This prevents accumulating unrelated threshold patches inside `detector.py`.


## 3. Field-Driven Improvement Loop

This is the normal operating and development cycle for every competition:

```text
Before the event
  -> prepare camera, current verified software, event profile, storage, and time reference
During the event
  -> run VideoPipe, preserve all athlete evidence, and mark visible problems with exact time
After the event
  -> collect full source video, issue markers, database, evidence, logs, and exact software commit
Analysis
  -> reproduce each marked problem from video and write a failing regression test
Repair
  -> change the common athlete/identity/event/OCR pipeline with the smallest justified fix
Regression
  -> run all retained competition cases, not only the latest sport
Next event
  -> deploy the newly verified build and continue the same loop
```

### Before the event

- Confirm the camera position, finish line, direction, resolution, FPS, focus, exposure, and available storage.
- Confirm that a complete original source video will be retained by the camera, recorder, or an explicitly verified recording path.
- Record the exact Git commit, model path/hash, Python environment, `EventProfile`, database/session directory, and camera source.
- Start with a fresh event/session so previous events cannot create duplicate evidence or misleading UI counts.
- Run a short preflight containing at least one real person crossing the configured line.

### During the event

- VideoPipe runs as auxiliary evidence; CycleRace remains the official timing authority.
- OCR failure never removes the athlete observation.
- The operator records an exact problem time and category when observing a miss, duplicate athlete, false athlete, incorrect bib, missing bib, freeze, or reconnect problem.
- Field operation does not require reviewing every athlete or changing several detector parameters during the race.
- The complete source video and runtime evidence remain available for post-event reproduction.

### After the event

- Preserve the source video, issue markers, `EventProfile`, database copy, evidence directory, logs, and software/model identity together as one analysis package.
- Reproduce marked issues first; use unmarked video portions for broader automated regression and performance measurement.
- Every confirmed defect becomes a focused test or a machine-readable regression assertion before implementation changes.
- A fix is rejected when it improves the latest event but causes a retained historical case to miss an athlete, merge two athletes, create a duplicate, lose evidence, or materially reduce laptop performance.

## 4. Current Verified Baseline

Baseline commit: `3db92317817fd8d7798d9b045b18e38fc6fcb855`

- [x] Immutable `FrameEnvelope` carries original frame and frame/timing metadata.
- [x] `StreamReader` uses a bounded FIFO and exposes queue/drop metrics.
- [x] `VideoThread` separates capture/media time from wall-clock processing time.
- [x] Low-confidence detections without a valid tracker ID are rejected below `0.20`.
- [x] OCR failure does not prevent an athlete event from being retained.
- [x] The known `track 405` / `track 605` scale-change duplicate is suppressed.
- [x] Adjacent end-of-video riders remain separate in the current regression.
- [x] Current automated verification passes `89` tests.
- [x] The 3510-frame cycling video processes at about `75 FPS` on the current machine.

The baseline does **not** yet provide a general participant identity manager. Its event deduplication still contains detector-specific geometric rules and raw tracker IDs remain too influential.

## 5. Runtime Invariants

1. Every accepted athlete observation has exactly one stable `participant_id`.
2. Multiple raw `track_id` values may be attached to the same `participant_id`.
3. Two athletes visible at the same time cannot be merged unless the detections are proven duplicate/nested observations of the same target.
4. OCR equality alone cannot merge identities; OCR conflict alone cannot split an identity.
5. An unrecognized athlete is still a complete participant observation.
6. Event IDs are allocated only after participant admission, identity resolution, and crossing validation pass.
7. Original-resolution evidence is preserved before OCR and UI work.
8. `finish_once` emits at most one finish event per participant.
9. `multi_lap` reuses the participant identity but emits a new passage after the configured lap reset condition.
10. Ambiguous identity matching preserves both raw evidence sets and exposes the ambiguity in regression metrics.

## 6. Target Data Flow

```text
Original frame
  -> detector inference frame
  -> same-frame athlete candidates
  -> duplicate-box fusion / adjacent-athlete guard
  -> raw tracker observations
  -> ParticipantIdentityManager
       raw track 405 ----\
                         -> participant P000004
       raw track 605 ----/
  -> CrossingLifecycle(participant_id)
  -> persisted video observation and original-frame evidence
  -> asynchronous OCR votes attached to participant_id
```

Target identity model:

```python
@dataclass
class ParticipantIdentity:
    participant_id: str
    event_profile_name: str
    raw_track_ids: set[int] = field(default_factory=set)
    first_seen_ms: float = 0.0
    last_seen_ms: float = 0.0
    last_bbox: tuple[int, int, int, int] | None = None
    identity_status: str = "ACTIVE"
    bib_evidence: list[BibEvidence] = field(default_factory=list)
```

Required identity states:

```text
ACTIVE
OCCLUDED
RECENTLY_LOST
CROSSED
EXPIRED
AMBIGUOUS
```

## 7. Acceptance Cases

| Case | Expected result |
|---|---|
| Long-video raw tracks `405` and `605` | One `participant_id`, one finish event |
| Adjacent riders `A18` and `A19` | Two `participant_id` values, two finish events |
| Athlete temporarily occluded and assigned a new raw track ID | Original `participant_id` reused |
| Same athlete has no readable bib | Participant and event remain valid |
| Same athlete receives conflicting OCR candidates | One participant with `CONFLICT`, not two participants |
| Two athletes wear similar colors | They remain separate when simultaneously visible or spatially inconsistent |
| Staff, pedestrian, gate, sign, or road structure | No accepted cycling participant event |
| Valid multi-lap passage | Same participant identity, new passage event |

Statistical thresholds such as recall percentages will be set only after the regression set contains enough representative material. Until then, every known hard case must remain stable and every run must report raw metrics.

---

### Task 0: Lock the First-Principles Domain Contract

**Files:**
- Create: `realtime/participant_models.py`
- Create: `tests/realtime/test_participant_models.py`

- [x] **Step 1: Write failing immutable-domain tests**

```python
from dataclasses import FrozenInstanceError

import pytest

from realtime.participant_models import (
    BibEvidence,
    CrossingCandidate,
    ParticipantIdentity,
    ParticipantObservation,
)


def minimal_observation():
    return ParticipantObservation(
        source_id=0,
        segment_id=1,
        frame_index=1,
        capture_time_ms=33.0,
        raw_track_id=None,
        participant_bbox=(100, 100, 200, 300),
        person_bbox=(100, 100, 200, 300),
        equipment_bbox=None,
        bib_bboxes=(),
        confidence=0.8,
    )


def test_participant_observation_keeps_physical_evidence_separate():
    observation = ParticipantObservation(
        source_id=0,
        segment_id=1,
        frame_index=120,
        capture_time_ms=4_000.0,
        raw_track_id=405,
        participant_bbox=(700, 300, 850, 900),
        person_bbox=(720, 300, 835, 780),
        equipment_bbox=(700, 600, 850, 900),
        bib_bboxes=((750, 430, 805, 490),),
        confidence=0.91,
    )
    assert observation.raw_track_id == 405
    assert observation.person_bbox != observation.equipment_bbox
    assert observation.bib_bboxes == ((750, 430, 805, 490),)


def test_observation_does_not_require_a_bib_or_tracker_id():
    observation = ParticipantObservation(
        source_id=0,
        segment_id=1,
        frame_index=121,
        capture_time_ms=4_033.0,
        raw_track_id=None,
        participant_bbox=(700, 300, 850, 900),
        person_bbox=None,
        equipment_bbox=None,
        bib_bboxes=(),
        confidence=0.42,
    )
    assert observation.raw_track_id is None
    assert observation.bib_bboxes == ()


def test_observation_is_immutable():
    observation = minimal_observation()
    with pytest.raises(FrozenInstanceError):
        observation.frame_index = 999
```

- [x] **Step 2: Run tests and verify RED**

```powershell
python -m pytest tests/realtime/test_participant_models.py -q --basetemp .pytest_tmp_participant_models
```

Expected: failure because `realtime.participant_models` does not exist.

- [x] **Step 3: Implement the domain types**

```python
from dataclasses import dataclass, field


BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class BibEvidence:
    bbox: BBox
    frame_index: int
    capture_time_ms: float
    confidence: float
    crop_quality: float
    text_candidate: str | None = None


@dataclass(frozen=True)
class ParticipantObservation:
    source_id: int
    segment_id: int
    frame_index: int
    capture_time_ms: float
    raw_track_id: int | None
    participant_bbox: BBox
    person_bbox: BBox | None
    equipment_bbox: BBox | None
    bib_bboxes: tuple[BBox, ...]
    confidence: float


@dataclass
class ParticipantIdentity:
    participant_id: str
    event_profile_name: str
    raw_track_ids: set[int] = field(default_factory=set)
    first_seen_ms: float = 0.0
    last_seen_ms: float = 0.0
    last_bbox: BBox | None = None
    identity_status: str = "ACTIVE"
    bib_evidence: list[BibEvidence] = field(default_factory=list)


@dataclass(frozen=True)
class CrossingCandidate:
    session_id: str
    source_id: int
    participant_id: str
    raw_track_id: int | None
    crossing_time_ms: float
```

- [x] **Step 4: Run domain tests**

```powershell
python -m pytest tests/realtime/test_participant_models.py -q --basetemp .pytest_tmp_participant_models
```

Expected: domain tests pass without loading YOLO, PaddleOCR, Qt, a database, or a camera.


### Task 1: Establish the Shared Event Profile Boundary

**Files:**
- Create: `realtime/event_profile.py`
- Modify: `realtime/detector.py`
- Create: `tests/realtime/test_event_profile.py`
- Modify: `tests/realtime/test_detection_association.py`

- [x] **Step 1: Write failing profile-contract tests**

```python
import pytest

from realtime.event_profile import EventProfile, build_event_profile


def test_default_event_uses_the_common_athlete_bib_pipeline():
    profile = build_event_profile(name="current-event")
    assert profile.name == "current-event"
    assert profile.pipeline == "athlete_bib"
    assert profile.required_equipment is None


def test_event_profile_accepts_real_competition_differences_as_configuration():
    profile = build_event_profile(
        name="current-cycling-event",
        required_equipment="bicycle",
        bib_regions=("torso", "back", "handlebar", "frame"),
        crossing_mode="multi_lap",
    )
    assert profile.pipeline == "athlete_bib"
    assert profile.required_equipment == "bicycle"
    assert profile.crossing_mode == "multi_lap"


def test_unknown_pipeline_fails_instead_of_silently_switching_algorithms():
    with pytest.raises(ValueError, match="pipeline"):
        build_event_profile(name="dense-test", pipeline="unknown")
```

- [x] **Step 2: Run the tests and verify RED**

```powershell
python -m pytest tests/realtime/test_event_profile.py -q --basetemp .pytest_tmp_event_profile
```

Expected: failure because `realtime.event_profile` does not exist.

- [x] **Step 3: Add the minimal profile model**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class EventProfile:
    name: str
    required_equipment: str | None = None
    pipeline: str = "athlete_bib"
    crossing_mode: str = "finish_once"
    bib_regions: tuple[str, ...] = ("torso", "back")


def build_event_profile(
    name: str,
    *,
    pipeline: str = "athlete_bib",
    crossing_mode: str = "finish_once",
    required_equipment: str | None = None,
    bib_regions: tuple[str, ...] = ("torso", "back"),
) -> EventProfile:
    if pipeline != "athlete_bib":
        raise ValueError(f"Unsupported event pipeline: {pipeline}")
    return EventProfile(
        name=str(name or "current-event").strip(),
        pipeline=pipeline,
        crossing_mode=crossing_mode,
        required_equipment=required_equipment,
        bib_regions=tuple(bib_regions),
    )
```

- [x] **Step 4: Replace inline sport branching in `Detector`**

```python
self.event_profile = event_profile or build_event_profile(name=sport_profile)
self.sport_profile = self.event_profile.name
```

Add an optional `event_profile: EventProfile | None = None` constructor argument while retaining the existing `sport_profile` string as a compatibility name for saved settings and older callers.

- [x] **Step 5: Run focused and full tests**

```powershell
python -m pytest tests/realtime/test_event_profile.py tests/realtime/test_detection_association.py -q --basetemp .pytest_tmp_event_profile_full
```

Expected: arbitrary current competitions use the same `athlete_bib` pipeline, while actual differences are explicit configuration values and unsupported pipelines fail clearly.

### Task 2: Implement Stable Participant Identity Resolution

**Files:**
- Create: `realtime/participant_identity.py`
- Create: `tests/realtime/test_participant_identity.py`

- [x] **Step 1: Write failing identity lifecycle tests**

```python
from realtime.participant_identity import (
    IdentityConfig,
    ParticipantIdentityManager,
)
from realtime.participant_models import ParticipantObservation


def observation(track_id, time_ms, bbox=(100, 100, 200, 300)):
    return ParticipantObservation(
        source_id=0,
        segment_id=1,
        frame_index=int(time_ms),
        capture_time_ms=time_ms,
        raw_track_id=track_id,
        participant_bbox=bbox,
        person_bbox=bbox,
        equipment_bbox=None,
        bib_bboxes=(),
        confidence=0.9,
    )


def test_new_raw_track_creates_participant():
    manager = ParticipantIdentityManager(config=IdentityConfig())
    result = manager.resolve(observation(track_id=405, time_ms=1000))
    assert result.created is True
    assert result.participant.raw_track_ids == {405}


def test_fragmented_track_reuses_recent_participant():
    manager = ParticipantIdentityManager(config=IdentityConfig())
    first = manager.resolve(
        observation(track_id=405, time_ms=1000, bbox=(700, 300, 850, 900))
    )
    second = manager.resolve(
        observation(track_id=605, time_ms=1400, bbox=(705, 320, 846, 902))
    )
    assert second.participant.participant_id == first.participant.participant_id
    assert second.participant.raw_track_ids == {405, 605}


def test_simultaneous_adjacent_riders_never_merge():
    manager = ParticipantIdentityManager(config=IdentityConfig())
    left = manager.resolve(
        observation(track_id=6543, time_ms=1000, bbox=(900, 300, 1050, 900))
    )
    right = manager.resolve(
        observation(track_id=5534, time_ms=1000, bbox=(1040, 300, 1190, 900))
    )
    assert left.participant.participant_id != right.participant.participant_id
```

- [x] **Step 2: Run the tests and verify RED**

```powershell
python -m pytest tests/realtime/test_participant_identity.py -q --basetemp .pytest_tmp_participant_identity
```

- [x] **Step 3: Implement explicit identity output and manager state**

```python
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
```

- [x] **Step 4: Implement conservative track-fragment matching**

`ParticipantIdentityManager.resolve()` consumes `ParticipantObservation` values and maintains the velocity/history needed for matching. The first version uses existing lightweight evidence only:

- exact raw track mapping;
- elapsed time;
- predicted center from recent velocity;
- bbox containment/IoU and scale ratio;
- crossing direction consistency;
- simultaneous-visibility guard;
- bib candidates as supporting evidence only.

It must not load a new neural ReID model. Ambiguous candidates create a separate identity with `identity_status=AMBIGUOUS` and preserve diagnostic reasons.

- [x] **Step 5: Run identity tests**

```powershell
python -m pytest tests/realtime/test_participant_identity.py -q --basetemp .pytest_tmp_participant_identity
```

### Task 3: Integrate Participant Identity into Detector Output

**Files:**
- Modify: `realtime/detector.py`
- Modify: `tests/realtime/test_detection_association.py`
- Modify: `tests/realtime/test_detector_metrics.py`

- [x] **Step 1: Add failing detector integration tests**

```python
def test_detector_exposes_stable_participant_id_for_track_fragment():
    detector = Detector(model_path="fake.pt", model=None, ocr=None)
    first = detector.resolve_participant(
        _athlete(405, (700, 300, 850, 900)), timestamp=1.0
    )
    second = detector.resolve_participant(
        _athlete(605, (705, 320, 846, 902)), timestamp=1.4
    )
    assert second["participant_id"] == first["participant_id"]


def test_detector_keeps_raw_track_id_for_audit():
    detector = Detector(model_path="fake.pt", model=None, ocr=None)
    resolved = detector.resolve_participant(
        _athlete(405, (700, 300, 850, 900)), timestamp=1.0
    )
    assert resolved["track_id"] == 405
    assert resolved["raw_track_ids"] == [405]
```

- [x] **Step 2: Run focused tests and verify RED**

```powershell
python -m pytest tests/realtime/test_detection_association.py -k "participant_id" -q --basetemp .pytest_tmp_detector_identity
```

- [x] **Step 3: Resolve identity after same-frame candidate filtering**

The integration order in `process_frame()` must be:

```text
model boxes
-> person/bib class mapping
-> ROI/background filtering
-> same-frame duplicate handling
-> raw tracker observations
-> ParticipantIdentityManager.resolve()
-> crossing lifecycle
```

Every returned athlete dict must include:

```python
{
    "track_id": raw_track_id,
    "participant_id": participant_id,
    "raw_track_ids": sorted(identity.raw_track_ids),
    "identity_status": identity.identity_status,
}
```

- [x] **Step 4: Add identity metrics**

`Detector.last_frame_metrics` must include:

```python
{
    "raw_tracks": int,
    "participants": int,
    "track_fragments_merged": int,
    "identity_ambiguities": int,
}
```

- [x] **Step 5: Run detector regression tests**

```powershell
python -m pytest tests/realtime/test_detection_association.py tests/realtime/test_detector_metrics.py -q --basetemp .pytest_tmp_detector_identity_full
```

### Task 4: Key Crossing Lifecycle by Participant, Not Raw Track

**Files:**
- Create: `realtime/crossing_lifecycle.py`
- Modify: `realtime/detector.py`
- Create: `tests/realtime/test_crossing_lifecycle.py`
- Modify: `tests/realtime/test_detection_association.py`

- [x] **Step 1: Write failing event-identity tests**

```python
from realtime.crossing_lifecycle import CrossingLifecycle
from realtime.participant_models import CrossingCandidate


def crossing(participant_id, raw_track_id=1, time_ms=1_000):
    return CrossingCandidate(
        session_id="test-session",
        source_id=0,
        participant_id=participant_id,
        raw_track_id=raw_track_id,
        crossing_time_ms=time_ms,
    )


def test_track_split_during_one_crossing_emits_one_event():
    lifecycle = CrossingLifecycle(mode="finish_once")
    assert lifecycle.admit(crossing(participant_id="P4", raw_track_id=405)) is True
    assert lifecycle.admit(crossing(participant_id="P4", raw_track_id=605)) is False


def test_adjacent_participants_each_emit_an_event():
    lifecycle = CrossingLifecycle(mode="finish_once")
    assert lifecycle.admit(crossing(participant_id="P18", raw_track_id=5534)) is True
    assert lifecycle.admit(crossing(participant_id="P19", raw_track_id=6543)) is True


def test_multi_lap_reuses_identity_but_allows_new_passage():
    lifecycle = CrossingLifecycle(mode="multi_lap", min_lap_interval_ms=30_000)
    assert lifecycle.admit(crossing(participant_id="P4", time_ms=1_000)) is True
    assert lifecycle.admit(crossing(participant_id="P4", time_ms=31_500)) is True
```

- [x] **Step 2: Run the tests and verify RED**

```powershell
python -m pytest tests/realtime/test_crossing_lifecycle.py -q --basetemp .pytest_tmp_crossing_lifecycle
```

- [x] **Step 3: Implement lifecycle states**

Reuse `CrossingCandidate` from `realtime.participant_models`; do not create a second crossing-candidate representation.

```text
APPROACHING -> IN_GATE -> CROSSED -> COOLDOWN -> EXPIRED
```

The lifecycle key is `(session_id, source_id, participant_id)`. Raw track IDs remain attached evidence and cannot independently allocate event IDs.

- [x] **Step 4: Allocate event ID only after lifecycle admission**

`CrossingEvent` gains compatibility-safe fields:

```python
participant_id: str = ""
raw_track_ids: tuple[int, ...] = ()
passage_index: int = 1
sport_profile: str = "cycling"
```

Existing `track_id` remains the raw track active at the event frame for compatibility.

- [x] **Step 5: Run event tests**

```powershell
python -m pytest tests/realtime/test_crossing_lifecycle.py tests/realtime/test_detection_association.py -q --basetemp .pytest_tmp_crossing_lifecycle_full
```

### Task 5: Aggregate Evidence and OCR by Participant

**Files:**
- Modify: `realtime/detector.py`
- Modify: `realtime/ocr_manager.py`
- Modify: `realtime/event_recorder.py`
- Modify: `tests/realtime/test_ocr_duplicate_merge.py`
- Create: `tests/realtime/test_participant_ocr.py`

- [x] **Step 1: Write failing OCR identity tests**

```python
from realtime.ocr_manager import ParticipantOcrState


def test_ocr_votes_from_fragmented_tracks_attach_to_one_participant():
    state = ParticipantOcrState(participant_id="P4")
    state.add_vote(raw_track_id=405, text="165", confidence=0.88)
    state.add_vote(raw_track_id=605, text="165", confidence=0.91)
    assert state.best_candidate == "165"
    assert state.raw_track_ids == {405, 605}


def test_ocr_conflict_does_not_create_second_participant():
    state = ParticipantOcrState(participant_id="P4")
    state.add_vote(raw_track_id=405, text="165", confidence=0.82)
    state.add_vote(raw_track_id=605, text="185", confidence=0.81)
    assert state.status == "CONFLICT"
    assert state.participant_id == "P4"
```

- [x] **Step 2: Run tests and verify RED**

```powershell
python -m pytest tests/realtime/test_ocr_duplicate_merge.py tests/realtime/test_participant_ocr.py -q --basetemp .pytest_tmp_participant_ocr
```

- [x] **Step 3: Re-key OCR aggregation**

OCR work items and vote histories must carry both `participant_id` and `raw_track_id`. Caches and final consensus use `participant_id`; the raw track remains audit metadata.

The participant-level state added to `realtime/ocr_manager.py` must expose this minimum contract:

```python
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class ParticipantOcrState:
    participant_id: str
    raw_track_ids: set[int] = field(default_factory=set)
    votes: list[tuple[str, float]] = field(default_factory=list)
    best_candidate: str | None = None
    status: str = "PENDING"

    def add_vote(self, raw_track_id: int, text: str, confidence: float) -> None:
        normalized = str(text or "").strip().upper()
        if not normalized:
            return
        self.raw_track_ids.add(int(raw_track_id))
        self.votes.append((normalized, float(confidence)))
        scores = defaultdict(float)
        for candidate, score in self.votes:
            scores[candidate] += score
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        self.best_candidate = ranked[0][0]
        if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < 0.25:
            self.status = "CONFLICT"
        else:
            self.status = "RECOGNIZED"
```

The final implementation must call the existing OCR candidate-normalization and roster-validation helpers before adding a vote. It recomputes consensus and sets `CONFLICT` when competing plausible values cannot be resolved.

- [x] **Step 4: Preserve original-resolution evidence**

Inference may use a reduced image, but athlete and bib crops must continue to use `FrameEnvelope.original_frame`. OCR queue overload may delay or drop optional OCR attempts, but cannot remove the participant observation or required evidence.

- [x] **Step 5: Run OCR and evidence tests**

```powershell
python -m pytest tests/realtime/test_ocr_duplicate_merge.py tests/realtime/test_participant_ocr.py tests/realtime/test_detection_association.py -q --basetemp .pytest_tmp_participant_ocr_full
```

### Task 6: Persist Stable Identity without Breaking Existing Databases

**Files:**
- Modify: `realtime/database.py`
- Modify: `realtime/event_recorder.py`
- Create: `tests/realtime/test_participant_persistence.py`

- [x] **Step 1: Write failing migration and persistence tests**

```python
def test_existing_crossing_database_adds_nullable_identity_columns(tmp_path):
    db = create_legacy_database(tmp_path)
    Database(db)
    columns = table_columns(db, "crossing_events")
    assert "participant_id" in columns
    assert "raw_track_ids_json" in columns
    assert "sport_profile" in columns
    assert "passage_index" in columns


def test_existing_event_rows_remain_readable_after_migration(tmp_path):
    db = create_legacy_database_with_event(tmp_path)
    record = Database(db).get_event(1)
    assert record["event_id"] == 1
```

- [x] **Step 2: Run tests and verify RED**

```powershell
python -m pytest tests/realtime/test_participant_persistence.py -q --basetemp .pytest_tmp_participant_db
```

- [x] **Step 3: Add additive columns only**

```sql
ALTER TABLE crossing_events ADD COLUMN participant_id TEXT;
ALTER TABLE crossing_events ADD COLUMN raw_track_ids_json TEXT;
ALTER TABLE crossing_events ADD COLUMN sport_profile TEXT DEFAULT 'cycling';
ALTER TABLE crossing_events ADD COLUMN passage_index INTEGER DEFAULT 1;
```

Migration code must inspect `PRAGMA table_info(crossing_events)` before each addition. It must not rewrite or renumber existing event rows.

- [x] **Step 4: Persist raw track audit data as JSON**

Use `json.dumps(sorted(set(raw_track_ids)))` and parse invalid/empty legacy values as a one-element list containing the compatibility `track_id` when available.

- [x] **Step 5: Run database tests**

```powershell
python -m pytest tests/realtime/test_participant_persistence.py -q --basetemp .pytest_tmp_participant_db
```

### Task 7: Convert All Competition Videos into Unified Batch Regression

**Files:**
- Create: `tests/realtime/fixtures/competition_regression_manifest.json`
- Modify: `realtime/devtools/regression_report.py`
- Modify: `tests/realtime/test_regression_metrics.py`
- Create: `tools/run_competition_regression.py`

- [x] **Step 1: Define a manifest that references external videos**

```json
{
  "schema_version": 1,
  "cases": [
    {
      "case_id": "long_mountain_finish",
      "video_path_env": "VIDEOPIPE_LONG_CYCLING_VIDEO",
      "event_profile": {
        "name": "cycling-mountain-finish",
        "pipeline": "athlete_bib",
        "required_equipment": "bicycle",
        "crossing_mode": "finish_once",
        "bib_regions": ["torso", "back", "handlebar", "frame"]
      },
      "known_same_participant_tracks": [[405, 605]],
      "known_distinct_tracks": [[5534, 6543]],
      "expected_in_clip_crossings": 17
    }
  ]
}
```

The manifest stores no video, athlete personal data, or absolute operator path in Git. Paths are supplied through environment variables or a local ignored override file. New competitions append cases to this same manifest regardless of sport name.

- [x] **Step 2: Add failing report-schema tests**

The report must contain:

```text
frames_processed
processing_fps
raw_tracks
stable_participants
track_fragments_merged
identity_ambiguities
crossing_events
duplicate_passages
rejected_candidates
evidence_complete
ocr_recognized
ocr_conflicts
ocr_unrecognized
queue_depth_max
dropped_frames
```

- [x] **Step 3: Run report tests and verify RED**

```powershell
python -m pytest tests/realtime/test_regression_metrics.py -q --basetemp .pytest_tmp_identity_report
```

- [x] **Step 4: Implement deterministic batch execution**

```powershell
python tools/run_competition_regression.py --manifest tests/realtime/fixtures/competition_regression_manifest.json --output C:\Users\Administrator\Documents\video_validation\latest-competition-regression
```

One command processes every retained configured case from all available competitions and writes one aggregate JSON report. Manual inspection is required only when a hard invariant changes, a new field marker is reproduced, or an ambiguity is newly introduced.

- [x] **Step 5: Encode the confirmed identity cases**

The automated comparison must fail when:

- `405` and `605` resolve to different participants;
- `5534` and `6543` resolve to the same participant;
- the crossing count differs from the stored known case without an explicitly reviewed baseline update;
- required evidence is missing;
- result signatures change without a report explaining which identities changed.

### Task 8: Add Field Problem Markers and Reproducible Analysis Packages

**Files:**
- Create: `realtime/field_issue_log.py`
- Modify: `realtime/main_window.py`
- Create: `tests/realtime/test_field_issue_log.py`
- Create: `tools/export_field_issue_package.py`
- Create: `tests/realtime/test_field_issue_export.py`
- Create: `docs/field_issue_workflow.md`

- [ ] **Step 1: Write failing marker tests**

```python
from realtime.field_issue_log import FieldIssueLog


def test_issue_marker_records_exact_processing_context(tmp_path):
    log = FieldIssueLog(tmp_path / "issues.jsonl")
    marker = log.record(
        session_id="race-2026-08-06",
        source_id=0,
        category="duplicate_athlete",
        frame_index=1200,
        capture_time_ms=40_000.0,
        participant_ids=("P17",),
        event_ids=(8,),
        metrics={"queue_depth": 2, "processing_time_ms": 14.2},
        note="same athlete appeared twice",
    )
    assert marker.issue_id
    assert marker.capture_time_ms == 40_000.0
    assert marker.category == "duplicate_athlete"


def test_issue_marker_does_not_create_or_modify_crossing_events(tmp_path):
    database_path = tmp_path / "events.db"
    database_path.write_bytes(b"existing-event-database")
    before = database_path.read_bytes()
    FieldIssueLog(tmp_path / "issues.jsonl").record(
        session_id="race",
        source_id=0,
        category="missed_athlete",
        frame_index=300,
        capture_time_ms=10_000.0,
    )
    assert database_path.read_bytes() == before
```

Required categories:

```text
missed_athlete
duplicate_athlete
false_athlete
wrong_bib
missing_bib
ui_freeze
camera_reconnect
other
```

- [ ] **Step 2: Run marker tests and verify RED**

```powershell
python -m pytest tests/realtime/test_field_issue_log.py -q --basetemp .pytest_tmp_field_issue_log
```

- [ ] **Step 3: Implement append-only issue records**

Each JSONL record must contain:

```text
issue_id
created_at
session_id
source_id
category
frame_index
capture_time_ms
segment_id
participant_ids
raw_track_ids
event_ids
screenshot_path
queue_depth
dropped_frames
processing_time_ms
software_commit
model_identity
event_profile
operator_note
```

Writes use a process lock, flush, and `os.fsync()`. A marker is diagnostic metadata only; it cannot alter participant, crossing, OCR, or CycleRace data.

- [ ] **Step 4: Add a non-blocking UI marker command**

Add one flag/bookmark icon with a tooltip. The command captures the latest `FrameEnvelope` context, saves one original-resolution screenshot asynchronously, and appends the marker without blocking the video thread. Category selection and an optional short note may be entered after the timestamp has already been secured.

- [ ] **Step 5: Write failing analysis-package tests**

```python
import json

from tools.export_field_issue_package import export_field_issue_package


def test_export_contains_reproduction_inputs(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "issues.jsonl").write_text(
        json.dumps({"issue_id": "I1", "capture_time_ms": 10_000.0}) + "\n",
        encoding="utf-8",
    )
    (session_dir / "session.json").write_text(
        json.dumps({"software_commit": "test"}),
        encoding="utf-8",
    )
    source_video = tmp_path / "race.mp4"
    source_video.write_bytes(b"video-placeholder")
    package = export_field_issue_package(
        session_dir=session_dir,
        source_video=source_video,
        output_dir=tmp_path / "package",
    )
    assert (package / "session.json").exists()
    assert (package / "issues.jsonl").exists()
    assert (package / "checksums.json").exists()
    assert (package / "evidence").exists()
```

- [ ] **Step 6: Implement deterministic package export**

```powershell
python tools/export_field_issue_package.py --session C:\RaceData\current-session --video D:\recordings\current-race.mp4 --output C:\Users\Administrator\Documents\video_validation\current-race-issues
```

The package records the source video path, size, duration, and checksum. It does not copy a large source video unless `--copy-video` is explicitly supplied. It includes the issue log, session metadata, event profile, database copy, relevant evidence, logs, software commit, model identity, and checksums.

- [ ] **Step 7: Verify the complete field workflow**

```powershell
python -m pytest tests/realtime/test_field_issue_log.py tests/realtime/test_field_issue_export.py -q --basetemp .pytest_tmp_field_issue_full
```

Expected: issue markers are durable, preserve exact video position, do not change race observations, and can be exported with the information needed to reproduce a field defect from the retained source video.


### Task 9: Verify Laptop Cost and Full Regression

**Files:**
- Modify: `realtime/participant_identity.py`
- Modify: `realtime/main_window.py`
- Create: `tests/realtime/test_identity_performance.py`
- Modify: `tests/realtime/test_video_thread.py`

- [ ] **Step 1: Add bounded-state tests**

Verify that expired participants, raw-track mappings, appearance summaries, and ambiguity records are pruned after configured time windows.

- [ ] **Step 2: Add a no-heavy-ReID regression**

The common athlete identity path must run with geometry, trajectory, existing OpenCV operations, and optional cached evidence. No new GPU model is loaded by default.

- [ ] **Step 3: Surface identity metrics without increasing UI refresh pressure**

Expose participant count, fragment merge count, ambiguity count, queue depth, and dropped frames through existing periodic monitoring. Do not repaint the UI for every detector observation.

- [ ] **Step 4: Run complete verification**

```powershell
python -m pytest tests/realtime -q --basetemp .pytest_tmp_participant_identity_full
python -m compileall -q realtime tests/realtime tools
git diff --check
```

- [ ] **Step 5: Run the configured cross-competition regression batch**

```powershell
python tools/run_competition_regression.py --manifest tests/realtime/fixtures/competition_regression_manifest.json --output C:\Users\Administrator\Documents\video_validation\identity-performance
```

The report compares against the `3db9231` baseline. New missed crossings, adjacent-athlete merges, duplicate passages, missing evidence, unbounded queues, or material performance regressions block publication.

- [ ] **Step 6: Inspect repository scope**

```powershell
git status --short
git diff --stat
git diff --check
```

Do not stage videos, model weights, databases, evidence images, logs, pytest directories, or unrelated dirty-worktree files. Commit and push only after explicit user authorization.

---

## 8. Competition-by-Competition Operation Rule

The product has one continuously improving athlete/bib pipeline. Development priority follows the actual competition calendar rather than a fixed sport roadmap.

### Before any competition

- Prepare and test the camera, recording path, current verified software build, storage, event session, finish line, and minimal `EventProfile`.
- Use the common athlete/bib pipeline by default.
- Configure only differences already known from the actual event, such as bicycle evidence, bib regions, direction, density, or lap mode.
- Do not fork the application or duplicate the processing pipeline for a sport name.

### During and after any competition

- Run the current verified build and preserve the complete source video.
- Mark visible problems with exact time/category instead of trying to diagnose them during the race.
- Export the post-event analysis package and reproduce the marked times from video.
- Convert every confirmed issue into a focused test and unified regression case.
- Fix the common pipeline first and rerun all retained competition cases before the next release.

### Marathon

Marathon does not receive speculative special development now. If an actual marathon occurs, run and record it through the common athlete/bib workflow first. Only confirmed dense-crowd failures justify a separate head/visible-body detection extension, which must still feed the same participant identity, event, evidence, and OCR contracts.

## 9. Plan Completion Definition

This plan is complete only when:

- known fragmented tracks resolve to one stable participant;
- adjacent real athletes remain separate;
- crossing uniqueness is keyed by participant identity;
- OCR votes aggregate by participant without controlling event existence;
- existing databases migrate additively and retain prior events;
- one command runs all configured historical competition cases and produces identity/performance metrics;
- the operator can mark a field problem with an exact video position without changing event data;
- a reproducible post-event analysis package can be exported with software, model, profile, video, database, evidence, issue, and log identity;
- the full realtime test suite, compilation, and diff checks pass;
- the same common pipeline can be configured for the current competition without a separate application branch;
- new field fixes do not regress retained cases from earlier competitions;
- no duplicated sport pipeline, CycleRace database coupling, or live two-PC transport has been introduced.
