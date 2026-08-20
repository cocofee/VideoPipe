# Video Timeline Index v1

## Goal

Locate camera evidence around a CycleRace passage without using network arrival time or
deriving capture time from `frame_index / fps`.

## Frame timestamps

Each `vp_frame_meta` carries:

| Field | Meaning |
|---|---|
| `source_pts_us` | Source presentation timestamp in microseconds, or `-1` when unavailable |
| `source_session` | Timestamp timeline generation; increments after reconnect/restart |
| `capture_monotonic_us` | Local steady-clock time captured immediately after acquisition |
| `capture_wall_time_ms` | Local system-clock time captured at the same point |

OpenCV/GStreamer live sources currently preserve `source_pts_us=-1` unless the backend
exposes a reliable source timeline. File inputs use `CAP_PROP_POS_MSEC`. The optional FFmpeg
input converts decoded `best_effort_timestamp` using the stream time base.

`vp_meta::create_time` remains a pipeline object creation time and must not be treated as
camera exposure time.

## Sparse persistent index

`vp_video_frame_index_writer` writes a JSONL checkpoint when a new recording segment starts
and then at a configurable interval, defaulting to 250 ms. Each checkpoint records:

- camera and channel identifiers;
- source session and optional source PTS;
- local monotonic and wall-clock capture times;
- pipeline frame index;
- video file path and frame number within that file.

The writer uses `capture_monotonic_us` to decide when the interval has elapsed, so an NTP or
manual wall-clock correction cannot pause checkpoint creation. A new `source_session` always
forces its first checkpoint. The persisted wall clock is still used for cross-process lookup
and must be interpreted together with calibration error.

Writer-generated `entry_id` values include camera, channel, source session, pipeline frame,
video path, and file-local frame. This prevents two channels or a restarted recording segment
from being discarded as the same checkpoint when their local counters happen to match.

The persistent store is keyed by `camera_id + channel_index`, so two cameras may both use
channel `0` without mixing results. Restart recovery rebuilds the in-memory search index from
the JSONL journal.

The indexed recorder keeps segment checkpoints in memory until the `.partial.mp4` has been
closed, verified as readable, and renamed to its final `.mp4` name. It then appends the
checkpoint batch; a failed append remains queued for retry at the next segment boundary.

## CycleRace mapping

A CycleRace passage time is a race-clock value, not a computer wall-clock timestamp. Video
lookup requires a calibration point:

```text
video_target_wall_ms = video_reference_wall_ms
                     + (passage_time_ms - race_reference_time_ms)
                     * (1 + drift_ppm / 1,000,000)
```

The locator returns the nearest checkpoint and all checkpoints inside the requested window.
The default UI target is one second before and one second after the mapped passage time. The
calibration's expected error expands that window rather than pretending the clocks are exact.

## Recording integration boundary

Continuous recording must call `vp_video_frame_index_writer::append_frame` only after the
frame has been handed to the active video segment. The index must therefore reference a
real file and file-local frame number. A later continuous recorder may rotate files, but it
must not change this indexing contract. OpenCV's `VideoWriter::write` has no success return
value, so deployment validation must also check that the resulting MP4 is playable.

`vp_indexed_file_des_node` implements this boundary for one camera/channel. It rotates the
MP4 segment when the configured duration expires, the source session changes, or video
dimensions/FPS change. It checks free disk space before opening each segment and reports
recording, low-disk, open, write, and index failures through a status hook.

For dual cameras, create two recorder nodes with distinct `camera_id` values and attach each
to its matching channel. Both nodes may share one `vp_video_frame_index_store`; the store is
mutex-protected and its lookup key keeps camera and channel timelines isolated.

`cycle_race_dual_camera_record_sample` accepts two RTSP URLs and one recording directory and
shows this wiring without adding OCR or detection to the critical recording path.
