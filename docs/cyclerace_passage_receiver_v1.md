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

## Video timeline lookup

`passage_time_ms` retains CycleRace's official milliseconds since local
midnight. When the competition has a valid same-day `stage.date`, CycleRace
also sends the optional `passage_timestamp_ms` field as Unix epoch milliseconds
for that Beijing passage time (`UTC+08:00`). VideoPipe uses the absolute field
for video lookup when present and falls back to `passage_time_ms` for legacy
senders. The review table always renders absolute passage timestamps in fixed
Beijing time, regardless of the VideoPipe computer's local timezone setting.
VideoPipe records each RTSP recording segment in:

```text
<race directory>/video_timeline.jsonl
```

The segment journal remains separate from `crossing_events`. It records the
VideoPipe system-clock start and end of each camera file, including automatic
FFmpeg restart segments. When a segment ends, VideoPipe also probes its frame
count and FPS with OpenCV and stores the verified media duration and derived
media start time. The review window maps a passage to every camera segment
whose media range contains the adjusted timestamp, then opens the video a few
seconds before the estimated position.

Older timeline records without media duration remain openable, but the review
window labels their range as unverified. A timestamp that only falls inside the
FFmpeg process interval but outside the verified media interval is reported as
outside the media range and is not presented as a located frame.

Two-computer clocks must be calibrated explicitly. VideoPipe applies:

```text
VideoPipe timestamp = preferred passage timestamp + passage_clock_offset_ms
```

The default offset is `0`. The review window always displays the active offset
and the configured segment timing uncertainty. This is an approximate video
navigation aid: FFmpeg process start is not guaranteed to equal the first
captured frame, and neither the offset nor the located frame changes official
CycleRace timing or results.
