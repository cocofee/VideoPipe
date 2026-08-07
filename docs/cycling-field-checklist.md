# Cycling Field Validation Checklist

Status on 2026-08-07: `BLOCKED / NOT TESTED`. No camera, model artifacts,
annotated race clips, or Paddle runtime are available in this checkout. This
checklist is the execution record to use before any candidate is promoted.

## Provisional Camera Purchase

The first trial should use a fixed bullet camera with a motorized varifocal
lens. Do not use a PTZ camera for the finish-line view: camera motion and
autofocus changes make repeatable detection and timing validation harder.

| Priority | JD-listed candidate | Intended placement | Verified from public JD listing | Still required before payment |
| --- | --- | --- | --- | --- |
| 1 | Dahua `DH-IPC-HFW4843F1-ZYL-PV-AS` | Normal finish-line placement, approximately 5-15 m from the capture zone | 8 MP, PoE, 2.7-12 mm motorized varifocal | H.264 main stream, RTSP URL, 3840x2160 frame rate, manual shutter, focus lock, WDR controls, exact sensor size, firmware support |
| 2 | Dahua `DH-IPC-HFW4843K-ZAS-I4-TGD` | Long or narrow view where 12 mm is not sufficient | 8 MP, PoE, 8-32 mm varifocal, public JD listing | Same protocol/exposure confirmation as priority 1; confirm minimum focus distance at the actual mounting distance |
| 3 | Hikvision `DS-2CD2686FWDA2-IZS` | Alternative supplier path | 8 MP, PoE, 2.7-13.5 mm varifocal | Listing is an order item and explicitly requires seller confirmation; obtain the exact datasheet and delivery date |

Public listings are acquisition leads, not acceptance evidence. The seller
must provide the exact model datasheet or device configuration screenshots
showing all of the following:

- `3840x2160` at least `25 fps`; `30 fps` is preferred.
- H.264 on the main stream and an RTSP URL usable without a vendor cloud.
- Manual electronic shutter including `1/1000 s` or faster.
- Ability to disable slow shutter and automatic day/night exposure changes.
- Motorized zoom with focus that can be locked after installation.
- PoE or PoE+ requirement and maximum power draw.
- Configurable WDR, gain, bitrate, GOP/I-frame interval, and constant frame
  rate behavior.
- Outdoor rating and operating temperature suitable for the race location.

The discontinued Dahua `IPC-HFW5842H-ZHE` is not the recommended purchase,
but its official specification is the capability benchmark: 8 MP 1/1.8-inch
sensor, 3840x2160 at 25/30 fps, 2.7-12 mm motorized lens, H.264, RTSP, and
manual shutter up to 1/100000 s. Reject a current seller candidate that cannot
meet the required subset above.

Source snapshot, checked 2026-08-07:

- JD normal-distance listing: `https://item.jd.com/10094327463060.html`
- JD long-lens listing: `https://item.jd.com/10123848723423.html`
- JD Hikvision listing: `https://item.jd.com/10267484770737.html`
- Dahua benchmark specification: `https://www.dahuasecurity.com/ceen/products/All-Products/Discontinued-Products/Network-Cameras/IPC-HFW5842H-ZHE=S3`

Do not record a price until the exact SKU, invoice type, stock, warranty, and
delivery date have been confirmed on the purchase day.

## Supporting Equipment

- [ ] Managed gigabit PoE/PoE+ switch or injector with at least 30% power
  reserve above the camera's documented maximum.
- [ ] Outdoor Cat6 cable, waterproof junction box, strain relief, and spare
  pre-terminated cable.
- [ ] Rigid pole/wall mount. No hand-held tripod for the acceptance run.
- [ ] UPS for camera, switch, and the VideoPipe laptop.
- [ ] Independent local recording path so the original RTSP stream survives a
  detector or OCR failure.
- [ ] Tape measure, focus chart or numbered rider target, and a rain-safe lens
  hood that does not enter the field of view.

## Initial Placement

The following values are starting points, not acceptance results:

- [ ] Mount 3-4 m high and 8-15 m from the center of the capture zone when the
  2.7-12 mm candidate is used.
- [ ] Start with a 20-35 degree downward angle. Avoid a nearly head-on view
  where riders continuously occlude one another.
- [ ] Frame a 6-10 m wide crossing zone and keep riders large enough that bib
  crops remain useful. Record the final focal length and measured distance.
- [ ] Keep the finish line away from the extreme image edge and reserve entry
  and exit space for tracking.
- [ ] Tighten and mark every mount adjustment after framing is approved.

## Camera Configuration

- [ ] Main stream: `3840x2160`, 25/30 fps, H.264, constant frame rate.
- [ ] Start shutter at `1/1000 s`; test `1/1500 s` or `1/2000 s` in bright
  conditions when wheel/rider blur remains visible.
- [ ] Disable automatic slow shutter. Limit gain before accepting excessive
  motion blur or noise.
- [ ] Start with moderate WDR for backlight; do not leave WDR at a high value
  without checking rider edges and bib contrast.
- [ ] Use motor zoom and autofocus only during setup, then lock focus if the
  camera supports it.
- [ ] Set GOP/I-frame interval to approximately one second and record bitrate,
  rate-control mode, and actual received FPS.
- [ ] Configure RTSP over TCP as the baseline. Test UDP separately and retain
  TCP when UDP increases loss or evidence gaps.
- [ ] Confirm the received stream is H.264 rather than silently falling back to
  H.265 or a low-resolution substream.

## Required Recordings

Keep camera position, focal length, and stream settings fixed across both
software profiles.

| Scenario | Minimum capture | Acceptance record |
| --- | --- | --- |
| Daylight | 10 minutes, normal and fast riders | Rider count, blur, received FPS, dropped frames |
| Backlight | 10 minutes with riders crossing the bright background | Recall, false boxes, bib contrast, WDR setting |
| Dusk | 10 minutes before and after illuminator/day-night transition | Exposure changes, motion blur, focus stability |
| Dense pack | At least 10 group crossings | Misses, ID switches, duplicate events, evidence completeness |
| Rain/wet road | 10 minutes where safe | Reflections, droplets, focus drift, false events |
| Reconnect/endurance | 30 minutes with one controlled cable/stream interruption | Reconnect time, frame-index continuity, evidence loss, backlog recovery |

## Runtime Layout And Profiles

Keep baseline and candidate artifacts in separate directories. Never overwrite
or delete the baseline engine during a candidate test.

```text
C:\cycling-runtime\baseline\
  yolov8s.engine
  labels.txt
  ocr\ch_PP-OCRv3_det_infer\
  ocr\ch_ppocr_mobile_v2.0_cls_infer\
  ocr\ch_PP-OCRv3_rec_infer\
  ocr\ppocr_keys_v1.txt

C:\cycling-runtime\candidate\
  yolo11s.onnx
  labels.txt
  ocr\...
```

Inspect the effective selection before loading a model:

```powershell
cycling_upgrade_sample --profile=baseline --runtime-dir=C:\cycling-runtime\baseline --print-config
cycling_upgrade_sample --profile=candidate --runtime-dir=C:\cycling-runtime\candidate --print-config
```

Run the same clip and retain separate output directories:

```powershell
cycling_upgrade_sample C:\clips\field.mp4 --profile=baseline --runtime-dir=C:\cycling-runtime\baseline
cycling_upgrade_sample C:\clips\field.mp4 --profile=candidate --runtime-dir=C:\cycling-runtime\candidate
```

Component overrides are allowed for fault isolation, for example:

```powershell
cycling_upgrade_sample C:\clips\field.mp4 --profile=candidate --tracker=sort --ocr=none --runtime-dir=C:\cycling-runtime\candidate
```

The executable still requires the corresponding compile-time capabilities:
YOLOv8 requires TensorRT, YOLO11 requires `VP_WITH_YOLO11`, synchronous OCR
requires Paddle, and asynchronous OCR requires both Paddle and
`VP_WITH_ASYNC_OCR`.

## Promotion And Rollback

- [ ] Score baseline and candidate against the same annotations using the same
  timestamp tolerance and class map.
- [ ] Candidate athlete recall is not lower than baseline.
- [ ] Candidate false-event and duplicate rates do not increase.
- [ ] Evidence completeness is exactly `1.0` for all matched observations.
- [ ] Processed FPS keeps up with source FPS and backlog has no sustained
  upward trend.
- [ ] OCR queue remains bounded; OCR failure does not remove an observation or
  its evidence.
- [ ] The 30-minute reconnect/endurance run completes without unrecovered
  stream loss or missing athlete evidence.
- [ ] Model, dataset, clip, configuration, and executable hashes are recorded.

Rollback is a command/configuration change to `--profile=baseline` and the
baseline runtime directory. Do not rebuild in the incident window, delete the
candidate, or modify CycleRace. CycleRace remains the only official timing and
result authority.

## Current Execution Record

| Gate | Status on 2026-08-07 | Reason |
| --- | --- | --- |
| Exact camera SKU purchased and inspected | `BLOCKED / NOT TESTED` | No camera is available |
| H.264 RTSP/manual shutter/fixed focus | `BLOCKED / NOT TESTED` | JD listings do not prove these settings |
| Day/backlight/dusk/dense/rain recordings | `BLOCKED / NOT TESTED` | No field recordings |
| Baseline/candidate real-video A/B | `BLOCKED / NOT TESTED` | Models and annotated clips are absent |
| 30-minute reconnect/endurance | `BLOCKED / NOT TESTED` | No RTSP camera/runtime |
| Final promotion decision | `NOT ELIGIBLE` | Required gates above are incomplete |
