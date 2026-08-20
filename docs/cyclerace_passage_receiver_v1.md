# CycleRace passage receiver v1

VideoPipe exposes the following endpoint while a race directory is active:

```text
POST /api/v1/passage-events
```

The default listener is `0.0.0.0:18765`. Override it with:

```text
--passage-host 192.168.1.20 --passage-port 18765
```

Use `--disable-passage-receiver` when the listener must remain off.

## Persistence and idempotency

Each race stores the original CycleRace messages in:

```text
<race directory>/cyclerace_passage_events.jsonl
```

The journal is appended and flushed before VideoPipe acknowledges the request.
On restart, VideoPipe restores the latest revision for every `event_id` and
repairs only an incomplete final JSON record. A complete malformed record is
treated as journal corruption and is not silently ignored.

Request results:

```text
201 accepted   new event or a newer revision
200 duplicate  identical or stale revision
400 rejected   malformed JSON or invalid protocol fields
409 rejected   same event_id and revision with different content
503 retry      event persisted but UI notification failed; sender should retry
```

## Product boundary

CycleRace remains the only official timing and result authority. The receiver
does not insert chip passages into VideoPipe's `crossing_events` table and does
not alter official results. It keeps the raw passage stream separate, then
notifies the VideoPipe operator workspace so later work can map chip time to
video evidence without changing detection, OCR, or evidence semantics.

The main window shows the listener state and the number of unique passage
events received for the active race. The latest bib/chip, group, lap, and
revision are available in the status tooltip.
