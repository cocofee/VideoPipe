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

## External sparse clip import

Third-party high-speed clips can be added to the same timeline with an explicit
JSON sidecar. VideoPipe does not infer capture time from file modification time
and does not modify or copy the source media.

```json
{
  "schema_version": 1,
  "race_id": "cyclerace-race-id",
  "clips": [
    {
      "video_path": "clips/finish-0001.mkv",
      "source_id": "high_speed_01",
      "camera_index": 3,
      "capture_timestamp": "2026-08-10T10:35:04.250+08:00",
      "timestamp_anchor": "start",
      "timing_error_ms": 20
    }
  ]
}
```

`race_id` is required and must match the single CycleRace race identity already
present in the active race's `cyclerace_passage_events.jsonl`. Import is blocked
until at least one official passage has been received, and a sidecar for a
different race is rejected before the timeline changes. Imported high-speed
segments persist that identity in `video_timeline.jsonl`; passage review filters
them by the event's `race_id` and will not fall back to a time-only match.

`capture_timestamp` must include the Beijing `+08:00` offset.
`timestamp_anchor` is `start` when the timestamp identifies the first media
frame and `end` when it identifies the end of the clip. `timing_error_ms` is the
measured or conservatively estimated capture-clock uncertainty. Relative video
paths are resolved from the sidecar directory.

In the packaged application, stop recording and use **Data Management > Import
high-speed camera clips** for the active race. Media probing runs in a cancellable
background task so the Qt interface remains responsive. Before writing the
verified clips, VideoPipe checks that the race directory, passage store, timeline
store, and `race_id` are still the same; results from an old race are discarded.
The import uses the in-process timeline so the review window sees new segments
immediately.

Source checkouts can also run the offline command while VideoPipe is closed:

```text
python -m realtime.external_clip_import <sidecar.json> <race directory>/video_timeline.jsonl
```

The importer validates every sidecar entry and probes every media duration
before changing the timeline. Re-importing identical metadata is idempotent,
source and camera identifiers must remain one-to-one, and conflicting metadata
for an already imported path is rejected. Sparse gaps between verified clips
remain `recording_gap`, which represents expected high-speed-camera behavior
rather than a recording failure. A passage just outside an external clip but
inside its declared `timing_error_ms` is presented as an uncertain boundary
candidate that a judge may open; it is not reported as a confirmed location.
