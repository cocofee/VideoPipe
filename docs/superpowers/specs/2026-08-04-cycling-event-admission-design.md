# Cycling Event Admission Design

## Scope

The first implementation targets bicycle races only. The shared pipeline must retain an explicit sport profile boundary so a future running profile can reuse video input, finish-line detection, evidence storage, OCR scheduling, and event persistence without inheriting bicycle-only validation rules.

This change does not retrain models, modify formal RaceData, or implement marathon timing.

## Confirmed Problems

- The deployed detector treats a detected bib as sufficient athlete evidence. In the full 8,979-frame marathon negative video, this produced 24 false cycling events; 21 carried a detected bib box.
- Bicycle validation currently runs only for events without bib evidence. Runners with bibs therefore bypass the validator.
- Event deduplication primarily covers sub-second tracker jitter. A rider who remains near the finish line and receives a new track ID several seconds later can emit repeated events.
- The same mountain video produced 26 events in the source encoding and 27 after HEVC re-encoding, showing that tracker fragmentation changes event counts.
- OCR quality depends on the evidence selected before OCR. Event validation and ownership must finish before OCR is allowed to affect an official event.

## Sport Profiles

The detector exposes a `sport_profile` value. The initial supported value is `cycling`; `running` is reserved for a later implementation.

Shared behavior:

- video and camera input;
- ROI and finish-line geometry;
- evidence capture from original frames;
- event numbering and persistence;
- OCR task scheduling and result review;
- performance profiles for GPU and laptop CPU use.

Cycling-only behavior:

- a candidate event must have bicycle evidence when the optional validator is available;
- bib detections are evidence for OCR and athlete association, not proof that the subject is a cyclist;
- a candidate explicitly rejected by the bicycle validator must not consume an official event ID;
- validator failures or unavailable models fail open and preserve the event, avoiding silent loss caused by infrastructure failure;
- rejected candidates retain evidence and a reason for audit, but are not official crossing events.

Future running behavior:

- no bicycle validator;
- runner/person validation and chest/waist bib geometry;
- separate tracker and dense-pack deduplication settings;
- separate model paths, regression videos, and acceptance thresholds.

Cycling and running training data and model selection must remain separate. A combined model is not part of this phase.

## Cycling Event Admission

At the final event boundary, the detector performs validation in this order:

1. Confirm the candidate is eligible, inside the configured finish segment, and not gate-like static structure.
2. Finish bib-to-rider ownership using the existing containment-based association.
3. Build bicycle-validation evidence from the original frame: the stored athlete crop and one context crop expanded around the candidate bounding box.
4. Run the lightweight generic bicycle validator only for the pending crossing candidate, not on every frame.
5. Accept when bicycle evidence is found. Reject only when the validator returns an explicit negative result from valid evidence. Preserve the event when the validator is absent or raises an error.
6. Perform duplicate suppression before allocating an event ID.
7. Allocate the event ID, save evidence, and submit OCR only after admission succeeds.

The validator remains event-only so a laptop CPU does not pay the secondary-model cost on every frame.

## Track Handoff Deduplication

Fixed cooldowns alone are unsafe because closely spaced riders may legitimately cross within the same second. The detector therefore keeps a short-lived finish-line occupant record for each accepted event:

- last bounding box near the finish line;
- last observation time;
- original event time and track ID;
- whether the occupant is still continuously visible near the line.

When ByteTrack assigns a new ID, a candidate is treated as the same occupant only when it remains continuously associated with the previous event region using strict spatial overlap and center/size consistency. The occupant expires after a short absence, allowing the next rider in the same lane to generate a new event.

This addresses a rider stopping or being assisted near the line without using a GPU-heavy ReID network and without applying a broad multi-second cooldown to the whole finish area.

## OCR Boundary

- OCR never determines whether a candidate is a cyclist.
- Detected bib crops and fallback torso crops remain separate evidence classes.
- Crops are always taken from the original-resolution frame.
- OCR starts only after the crossing candidate passes cycling admission.
- Low-resolution or uncertain OCR results remain reviewable and do not change event identity automatically.

## Configuration

The initial default remains `sport_profile: cycling` for backward compatibility. Bicycle validation remains optional at startup, but when a validator model is loaded it applies to both bib and non-bib cycling candidates.

No new dependency is required. The implementation uses the existing Ultralytics validator and OpenCV/numpy geometry.

## Verification

Automated tests must prove:

- a cycling candidate with a bib is rejected when the validator explicitly finds no bicycle;
- a valid bibbed cyclist is accepted;
- a missing or failed validator fails open;
- a rejected candidate does not consume an event ID;
- a continuously visible occupant with a new track ID is suppressed;
- two adjacent riders crossing close together are not merged;
- the original-frame OCR crop behavior remains unchanged;
- `sport_profile=running` is rejected as unsupported in this phase rather than silently using cycling rules.

Video regression acceptance for this phase:

- marathon negative video: zero official cycling events, while rejected evidence remains available;
- road clips: retain the currently verified two and five rider events;
- city finish video: retain its valid rider events;
- mountain source and HEVC variants: remove repeated events and converge toward the same unique-rider count;
- processing remains suitable for event-only validation on laptop CPU profiles.

## Deferred Work

- clean, leakage-free cycling dataset reconstruction and model retraining;
- a separately designed running profile;
- dense peloton ground-truth annotation;
- OCR model comparison after high-quality, correctly owned bib crops are available.
