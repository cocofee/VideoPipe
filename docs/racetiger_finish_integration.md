# RaceTiger finish feed

The finish console can read RaceTiger as a read-only timing source. Configure
the active race's `config.json` with the API base URL and token (do not commit
real credentials):

```json
{
  "timing_provider": "racetiger",
  "racetiger_base_url": "https://rqs.racetigertiming.com",
  "racetiger_pc": "<event-pc>",
  "racetiger_rid": "<event-rid>",
  "racetiger_token": "<local-secret>",
  "racetiger_poll_interval_seconds": 2.0
}
```

The adapter sends `POST` requests to `Dif/info`, `Dif/bio`, `Dif/score`, and
`Dif/split`. `pc`, `rid`, `token`, and `page=1` are query parameters; the
token remains local configuration and is never logged. Only finish-status
records and `FINISH` timing-point rows are imported. `PassTime` is converted
from Beijing local time to the absolute timestamp used by the existing video
timeline.

RaceTiger observations are stored in `racetiger_passage_events.jsonl`. Manual
observations for people visible in the video without a matching RaceTiger chip
record are stored separately in `video_supplements.jsonl` through the
`视频补录` button. Neither file changes official CycleRace or RaceTiger results.
