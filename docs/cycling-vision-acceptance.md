# Cycling Vision Acceptance Contract

This document defines the external event format and the baseline metrics used
to evaluate cycling crossing-event detection.  The video system is an
auxiliary evidence source: an observation must remain present even when OCR
is missing or wrong.

## Input files

`ground_truth.json` and `predictions.json` are UTF-8 JSON files.  The top-level
value is either an array of event objects or an object with an `events` array.
Each event uses milliseconds from the same video/race clock:

```json
{
  "event_id": "gt-001",
  "athlete_id": "rider-17",
  "timestamp_ms": 123456,
  "ocr": "101",
  "evidence": {
    "frame_path": "frames/123456.jpg",
    "clip_path": "clips/rider-17.mp4"
  }
}
```

Fields:

- `athlete_id` (required for a match): stable identity assigned by the
  annotation/tracking layer.  It is compared as a string; OCR text is not an
  identity key.
- `timestamp_ms` (required): event time in integer or floating-point
  milliseconds.
- `ocr` (optional): the OCR candidate text.  `ocr_text`, `bib`, and
  `bib_number` are accepted aliases by the reference scorer.
- `ocr_confirmed` (optional prediction field): an explicit confirmation flag.
  `false` rejects a candidate; `true` is accepted only when the non-empty
  prediction OCR text equals the ground-truth OCR text.  When omitted, equal
  non-empty OCR text is treated as confirmed by the reference scorer.
- `evidence` (required for a complete prediction): an object containing
  non-empty `frame_path` and `clip_path`.  `frame_paths` (a non-empty array)
  may be used instead of `frame_path` when several stills are retained.
  Additional evidence fields are allowed.
- `event_id` is optional metadata and is not used for matching.

Unknown fields must be preserved by producers.  Empty strings and `null` are
treated as missing values.

## Event matching and duplicate rules

The default timestamp tolerance is **1,000 ms** (`score(...,
tolerance_ms=1000)`).  A ground-truth event and prediction match when their
normalized `athlete_id` values are equal and the absolute timestamp delta is
at most the tolerance.  Matching is one-to-one and time ordered within each
athlete, so one prediction cannot satisfy two annotations.

After matching, an unmatched prediction is a **duplicate** when it is still
within tolerance of any ground-truth event for the same athlete (including an
event already consumed by another prediction).  Other unmatched predictions
are false events.  This separation makes duplicate generation visible rather
than hiding it in false positives.

OCR disagreement never removes an athlete/time match.  A missing or incorrect
OCR value is an OCR failure, not an athlete miss.  An observed athlete with no
OCR must remain in `predictions.json` with its `athlete_id`, timestamp, and
evidence.

## Metrics

The reference scorer reports values in the range `[0, 1]` (rates, not
percentages).  Empty denominators produce `0.0`.

- **Athlete recall** = matched ground-truth events / total ground-truth
  events.
- **False-event rate** = false (non-duplicate) unmatched predictions / total
  predictions.
- **Duplicate rate** = duplicate unmatched predictions / total predictions.
- **OCR candidate rate** = matched events with a non-empty prediction OCR /
  matched events whose ground truth has OCR.  This is a diagnostic rate and is
  not currently returned by `score_events.score`.
- **OCR confirmed rate** = matched events whose prediction OCR equals the
  ground-truth OCR (and is therefore confirmed) / matched events whose ground
  truth has OCR.  OCR mismatches leave athlete recall unchanged.
- **Evidence completeness** = matched predictions containing complete evidence
  / matched predictions.  Complete evidence requires a still (`frame_path` or
  non-empty `frame_paths`) and a clip (`clip_path`).
- **Track fragmentation** = `max(0, observed_track_segments - 1)` per athlete,
  or the aggregate sum divided by annotated athletes.  A segment is a
  contiguous run of observations with the same `track_id`; report both the
  aggregate and per-athlete distribution when available.
- **Processed FPS** = processed video frames / elapsed processing seconds.
  Record the frame count and wall-clock interval used for reproducibility.
- **p95 latency** = 95th percentile of per-frame processing latency in
  milliseconds, using a clearly stated percentile interpolation method.
- **Max backlog** = maximum queued-but-not-yet-processed frame count sampled
  during the run.  A growing backlog is a realtime failure even if eventual
  recall is high.

For release acceptance, report the metric values together with the input file
hashes, tolerance, video clock definition, and whether OCR/evidence fields
were absent or intentionally unlabelled.

## Promotion Contract

The comparison unit is one fixed clip/annotation set processed by both
profiles with identical camera input, class map, thresholds, timestamp
tolerance, and output layout.

| Gate | Required result |
| --- | --- |
| Athlete recall | Candidate is greater than or equal to baseline |
| False-event rate | Candidate is less than or equal to baseline |
| Duplicate rate | Candidate is less than or equal to baseline |
| Evidence completeness | Exactly `1.0` for matched observations |
| Throughput | Processed FPS is at least source FPS for a live run |
| Backlog | Bounded with no sustained upward trend |
| OCR | Queue remains bounded; failure never removes an observation/evidence |
| Reconnect/endurance | 30-minute run recovers from one controlled interruption without unrecovered evidence loss |

All gates must pass. A detector improvement does not compensate for lower
athlete recall, more duplicate events, incomplete evidence, or an unbounded
OCR/frame queue.

## Profile Identity

The executable profiles are:

- `baseline`: YOLOv8 + SORT + synchronous OCR.
- `candidate`: YOLO11 + ByteTrack + asynchronous OCR.

`--detector`, `--tracker`, `--ocr`, and `--runtime-dir` may override profile
defaults independently for diagnosis. An override run is not the named
profile and must be recorded with its complete effective configuration.

The legacy invocation without `--profile` remains YOLOv8 detection-only. It
exists for compatibility and is not the formal baseline comparison profile.

## Decision Status

On 2026-08-07 the field promotion decision is `NOT ELIGIBLE`. The scoring
implementation and component-level synthetic tests exist, but no selected
camera, model artifacts, annotated race video, complete runtime, real A/B
metrics, or reconnect/endurance result is available. These missing inputs are
`BLOCKED / NOT TESTED`, not passes.
