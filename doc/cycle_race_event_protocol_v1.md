# CycleRace -> VideoPipe PassageEvent Protocol v1

## Purpose

This contract carries official chip-passage observations from CycleRace to VideoPipe.
It is transport-neutral: the first implementation accepts an in-process or test payload,
while a TCP/HTTP/WebSocket adapter can be added without changing the event model.

CycleRace remains the official timing and result authority. VideoPipe stores and reviews
the event, but does not directly overwrite official results.

## JSON payload

```json
{
  "schema_version": 1,
  "message_type": "passage",
  "event_id": "race-2026-final-000001",
  "race_id": "race-2026",
  "stage_id": "final",
  "group_id": "men-open",
  "sequence": 1,
  "chip_id": "chip-23",
  "bib": "23",
  "passage_time_ms": 123456,
  "lap": 1,
  "source": "cyclerace",
  "emitted_at_ms": 0,
  "revision": 1
}
```

## Field rules

| Field | Required | Rule |
|---|---:|---|
| `schema_version` | yes | Must be `1` |
| `message_type` | yes | Must be `passage` |
| `event_id` | yes | Stable idempotency key; retries reuse it |
| `race_id` | yes | Race identifier |
| `stage_id` | yes | Stage or event identifier |
| `group_id` | yes | Start group/category identifier |
| `sequence` | yes | Positive CycleRace passage order |
| `chip_id` | one of chip/bib | Chip identifier when available |
| `bib` | one of chip/bib | Bib number when already mapped |
| `passage_time_ms` | yes | Non-negative CycleRace race time in milliseconds |
| `lap` | yes | Zero for non-lap races, otherwise current lap |
| `source` | yes | Normally `cyclerace` |
| `emitted_at_ms` | no | Sender diagnostic timestamp; never used as official race time |
| `revision` | no | Positive passage correction version; omitted legacy payloads mean `1` |

## Delivery semantics

1. CycleRace may retry the same message after a timeout.
2. VideoPipe deduplicates by `event_id` and `revision`.
3. A higher revision for the same `event_id` is an update (for example, a dual-chip
   correction to `passage_time_ms`) and is appended to the journal.
4. Repeating the same `event_id` and revision with identical content is a duplicate;
   repeating a revision with different content is a conflict.
5. A newly accepted event is appended to the local JSONL journal before the review callback runs.
6. A restart reloads the journal and keeps the latest revision for each event id.
7. A syntactically incomplete final JSONL line may be truncated during startup recovery;
   malformed complete lines and any middle-of-file corruption stop recovery with an explicit error.

## HTTP transport

The first LAN adapter exposes:

```text
POST /api/v1/passage-events
Content-Type: application/json
```

Responses are JSON. A new event returns HTTP `201` and `status=accepted`; a retry of an
already persisted `event_id` returns HTTP `200` and `status=duplicate`; malformed events return
HTTP `400` and `status=rejected`. CycleRace can retry until it receives either `accepted` or
`duplicate`.

The initial adapter is intended for a dedicated race LAN. Authentication, source allow-listing
and TLS must be added before exposing the endpoint outside that isolated network.
