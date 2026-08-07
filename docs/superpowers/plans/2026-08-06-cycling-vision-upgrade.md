# VideoPipe Cycling Vision Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 提升 VideoPipe 在骑行冲线场景中的运动员召回、Track 稳定性、号码 OCR 可用率和证据完整性，同时保持 YOLOv8/SORT/旧运行时可回退。

**Architecture:** 先建立固定数据集和指标基线，再以兼容接口增加输入配置、YOLO11 ONNX 后端、ByteTrack 适配和异步 OCR。现有 YOLOv8、SORT 和 PaddleOCR 2.6 实现保持默认路径；新实现通过 CMake 选项和运行时配置启用，所有阶段以真实视频 A/B 结果作为晋级门槛。

**Tech Stack:** C++17, CMake/CTest, OpenCV DNN, GStreamer/RTSP, TensorRT, PaddleOCR C++ inference, CUDA, JSON/CSV evaluation with Python standard library.

---

## Scope And Gates

- 官方计时仍由 CycleRace 负责；VideoPipe 只保存辅助视频证据和候选 OCR，不写官方结果表。
- 不在同一阶段同时升级摄像头、YOLO、TensorRT 和 OCR 运行时。
- 旧路径必须继续可编译、可运行、可回滚。
- 真实数据放在工作区外，通过 `VP_CYCLING_DATASET` 指向，不提交视频、号码布图片、日志或正式比赛数据。
- 每一阶段结束时记录：输入 FPS、端到端 FPS、队列长度、运动员召回、误检事件、重复事件、Track 断裂、OCR 确认率和证据文件数。

## File Map

### Create

- `docs/cycling-vision-acceptance.md`：数据集格式、指标定义和晋级门槛。
- `docs/cycling-runtime-matrix.md`：GPU、驱动、CUDA、TensorRT、OpenCV、GStreamer、PaddleOCR 版本矩阵。
- `tests/CMakeLists.txt`：CTest 目标和编译选项。
- `tests/test_stream_profile.cpp`：RTSP/app source 元数据和编码配置测试。
- `tests/test_yolo11_postprocess.cpp`：YOLO11 输出布局、框解码和 NMS 测试。
- `tests/test_bytetrack.cpp`：遮挡、低置信检测和 ID 稳定性测试。
- `tests/test_ocr_dispatch.cpp`：OCR 队列、背压和 OCR 失败保留证据测试。
- `tools/cycling_eval/score_events.py`：只依赖 Python 标准库的离线指标计算器。
- `tools/cycling_eval/test_score_events.py`：指标计算器单元测试。
- `nodes/infers/vp_yolo11_detector_node.h` / `.cpp`：YOLO11 ONNX/OpenCV DNN 适配器。
- `nodes/track/vp_bytetrack_node.h` / `.cpp`：基于现有 Track 抽象和 Hungarian 实现的两阶段关联器。
- `nodes/infers/vp_async_ocr_node.h` / `.cpp`：有界 OCR worker 队列和结果回写。
- `objects/vp_ocr_job.h` / `.cpp`：OCR 任务快照和证据路径。
- `objects/vp_ocr_consensus.h` / `.cpp`：按 Track 聚合多帧 OCR 候选。

### Modify

- `CMakeLists.txt`：增加 `VP_BUILD_TESTS`、`VP_WITH_YOLO11`、`VP_WITH_ASYNC_OCR`，默认保持关闭。
- `nodes/vp_rtsp_src_node.h` / `.cpp`：增加 H.264/H.265、TCP/UDP、延迟和流配置构造参数；H.264/TCP 保持默认兼容行为。
- `nodes/vp_app_src_node.h` / `.cpp`：允许调用方传入真实 FPS，保留默认 `1` 以兼容旧调用方。
- `nodes/vp_primary_infer_node.cpp`：保留现有 blob 流程，并为 YOLO11 适配器提供 letterbox 元数据。
- `nodes/track/vp_track_node.h` / `.cpp`：补充 Track 生命周期统计和可选 tracker 名称。
- `nodes/infers/vp_ppocr_text_detector_node.h` / `.cpp`：保留旧同步实现，增加与新异步 OCR 结果结构的转换边界。
- `samples/CMakeLists.txt`：增加 `cycling_upgrade_sample`，支持旧后端和新后端切换。

## Task 1: Establish Baseline And Test Harness

**Files:**
- Create: `docs/cycling-vision-acceptance.md`
- Create: `tools/cycling_eval/score_events.py`
- Create: `tools/cycling_eval/test_score_events.py`
- Create: `tests/CMakeLists.txt`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Define the external dataset and event formats**

`ground_truth.json` and `predictions.json` use one record per crossing event:

```json
{
  "event_id": "camera0-athlete-001",
  "channel": 0,
  "timestamp_ms": 123456,
  "track_id": 7,
  "bib": "33",
  "evidence_image": "relative/path.jpg",
  "evidence_video": "relative/path.mp4"
}
```

The acceptance document must define matching tolerance, duplicate-event rules, OCR conflict rules, and these metrics: athlete recall, false-event rate, duplicate rate, Track fragmentation, OCR candidate rate, OCR confirmed rate, evidence completeness, processed FPS, p95 latency, and maximum queue backlog.

- [ ] **Step 2: Implement deterministic event scoring**

`score_events.py` must expose:

```python
def load_events(path): ...
def match_events(ground_truth, predictions, tolerance_ms): ...
def score(ground_truth, predictions, tolerance_ms=1000): ...
```

The returned dictionary must contain `recall`, `false_event_rate`, `duplicate_rate`, `ocr_confirmed_rate`, and `evidence_complete_rate`. OCR mismatch must not convert a matched athlete event into a missed athlete event.

- [ ] **Step 3: Add standard-library tests**

`test_score_events.py` must cover: exact match, timestamp tolerance, one prediction matching only one ground-truth event, OCR mismatch with preserved athlete match, missing evidence, and duplicate predictions.

Run:

```powershell
python -m unittest discover -s tools/cycling_eval -p "test_*.py" -v
```

Expected: all scoring tests pass without importing OpenCV, CUDA, TensorRT, or PaddleOCR.

- [ ] **Step 4: Add CTest support without changing default builds**

Add to the root `CMakeLists.txt`:

```cmake
option(VP_BUILD_TESTS "Build VideoPipe tests" OFF)
if(VP_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
```

`tests/CMakeLists.txt` must define one executable per test source, link only the required VideoPipe target, and register each executable with `add_test(NAME ... COMMAND ...)`.

- [ ] **Step 5: Run the baseline build**

```powershell
cmake -S . -B build-baseline -DVP_WITH_CUDA=OFF -DVP_WITH_TRT=OFF -DVP_WITH_PADDLE=OFF -DVP_BUILD_TESTS=ON
cmake --build build-baseline --config Debug
ctest --test-dir build-baseline -C Debug --output-on-failure
```

Expected: the existing CPU build and the new deterministic tests pass. If the local machine lacks GStreamer development packages, record the exact configure error in `docs/cycling-runtime-matrix.md` and do not change dependencies in this task.

## Task 2: Make Camera And Stream Metadata Correct

**Files:**
- Modify: `nodes/vp_rtsp_src_node.h` / `.cpp`
- Modify: `nodes/vp_app_src_node.h` / `.cpp`
- Create: `tests/test_stream_profile.cpp`

- [ ] **Step 1: Add explicit stream configuration types**

Add English enums and a value object in `vp_rtsp_src_node.h`:

```cpp
enum class vp_rtsp_codec { H264, H265 };
enum class vp_rtsp_transport { TCP, UDP };

struct vp_rtsp_profile {
    vp_rtsp_codec codec = vp_rtsp_codec::H264;
    vp_rtsp_transport transport = vp_rtsp_transport::TCP;
    int latency_ms = 200;
};
```

Keep the existing constructor signature and delegate it to a new overload so old samples compile unchanged.

- [ ] **Step 2: Centralize GStreamer pipeline construction**

Implement a private or free helper with deterministic output:

```cpp
std::string build_rtsp_gst_pipeline(const std::string& url,
                                    const vp_rtsp_profile& profile,
                                    const std::string& decoder);
```

The default output must remain H.264 depay/parser, TCP transport, and the configured decoder. H.265 support must select `rtph265depay ! h265parse`; no automatic codec guessing is allowed.

- [ ] **Step 3: Preserve real FPS for app sources**

Add an overload:

```cpp
vp_app_src_node(std::string node_name, int channel_index, int source_fps);
```

Store `source_fps`, reject non-positive values, and set `original_fps = source_fps` on the first frame. Keep the existing two-argument constructor delegating to `source_fps=1`.

- [ ] **Step 4: Test stream profiles and metadata**

`test_stream_profile.cpp` must assert H.264/TCP and H.265/UDP pipeline selection, reject invalid FPS, and verify that an app source with `source_fps=60` reports 60 rather than 1.

Run:

```powershell
cmake --build build-baseline --config Debug --target test_stream_profile
ctest --test-dir build-baseline -C Debug -R stream_profile --output-on-failure
```

- [ ] **Step 5: Hardware acceptance gate**

Before model work, capture four external clips with the selected camera: daylight, backlight, dusk, and dense group crossing. Use H.264 RTSP, no frame skipping, fixed focus, and save the original stream metadata. Promote the camera only if the stream is stable for 30 minutes and no athlete evidence is lost during reconnect tests.

## Task 3: Add A/B Detector Path For YOLO11

**Files:**
- Create: `nodes/infers/vp_yolo11_detector_node.h` / `.cpp`
- Create: `tests/test_yolo11_postprocess.cpp`
- Modify: `CMakeLists.txt`
- Modify: `samples/CMakeLists.txt`
- Create: `samples/cycling_upgrade_sample.cpp`

- [ ] **Step 1: Define a detector interface that returns existing targets**

The new node must inherit `vp_primary_infer_node` and expose:

```cpp
vp_yolo11_detector_node(std::string node_name,
                        std::string model_path,
                        std::string labels_path,
                        int input_width = 640,
                        int input_height = 640,
                        float confidence_threshold = 0.25f,
                        float nms_threshold = 0.45f);
```

It must write `vp_frame_target` objects and must not alter existing target IDs or metadata fields.

- [ ] **Step 2: Normalize output layouts before decoding**

Support the two exported layouts used by the validation engines: `[1, channels, candidates]` and `[1, candidates, channels]`. Normalize to rows of `[x, y, w, h, class_scores...]`, convert letterbox coordinates back to the original frame, apply class-aware NMS, clamp boxes, and reject zero-area boxes.

- [ ] **Step 3: Add a synthetic postprocess test**

`test_yolo11_postprocess.cpp` must feed one known output row and assert the decoded box, class ID, confidence, letterbox reversal, and suppression of an overlapping lower-score box. It must test both tensor layouts.

- [ ] **Step 4: Keep the old path as the default**

Add:

```cmake
option(VP_WITH_YOLO11 "Build YOLO11 adapter" OFF)
```

The sample must select the detector with a command-line argument, while the default remains the existing YOLOv8 TensorRT path.

- [ ] **Step 5: Train and export comparable models outside the repository**

Train `YOLOv8s` and `YOLO11s` on the same cycling dataset, use the same image size and class map, and export separate ONNX/engine artifacts under `VP_CYCLING_MODEL_DIR`. Do not commit weights or engines.

- [ ] **Step 6: Run A/B detector validation**

For each model, record detector recall, false boxes, p95 inference time, and output FPS. Do not promote YOLO11 unless it has no athlete-recall regression and its end-to-end queue remains bounded at the camera FPS.

## Task 4: Replace SORT Only After Measuring ID Failures

**Files:**
- Create: `nodes/track/vp_bytetrack_node.h` / `.cpp`
- Create: `tests/test_bytetrack.cpp`
- Modify: `nodes/track/vp_track_node.h` / `.cpp`

- [ ] **Step 1: Add tracking statistics to the base node**

Expose counters for `created_tracks`, `deleted_tracks`, `matched_detections`, `unmatched_detections`, and `track_fragments`. Keep existing `track_id` and `tracks` output unchanged.

- [ ] **Step 2: Implement two-stage IoU association**

Reuse `nodes/track/sort/Hungarian.h`. Associate high-confidence detections first, then associate unmatched tracks with low-confidence detections. Confirm a track only after `min_hits`; remove it after `max_age`; never reuse an active ID.

- [ ] **Step 3: Test occlusion and low-confidence recovery**

The test sequence must contain: a continuous target, a two-frame miss, two adjacent targets, and a low-confidence recovery. Assert stable ID for the recovered target and no merge of adjacent targets.

- [ ] **Step 4: Run tracker-only comparison**

Run the same detector outputs through SORT and ByteTrack. Promote only if ID switches and duplicate crossing events decrease without lowering athlete recall.

## Task 5: Make OCR Asynchronous And Evidence-Preserving

**Files:**
- Create: `objects/vp_ocr_job.h` / `.cpp`
- Create: `objects/vp_ocr_consensus.h` / `.cpp`
- Create: `nodes/infers/vp_async_ocr_node.h` / `.cpp`
- Create: `tests/test_ocr_dispatch.cpp`
- Modify: `nodes/infers/vp_ppocr_text_detector_node.h` / `.cpp`

- [ ] **Step 1: Define immutable OCR jobs**

`vp_ocr_job` must contain channel, frame index, track ID, timestamp/FPS, copied crop, source box, and evidence image path. A failed OCR job must still leave the source target and evidence intact.

- [ ] **Step 2: Implement a bounded worker queue**

Use a mutex, condition variable, and fixed capacity. When full, discard the oldest unprocessed OCR job, increment `dropped_jobs`, and never discard the frame or athlete observation. Worker shutdown must signal, drain or cancel explicitly, and join before destruction.

- [ ] **Step 3: Restrict OCR to useful crops**

Only enqueue crops attached to a valid track and inside the configured crop-size range. Use tight crop first, then one padded fallback. The OCR result is candidate metadata, not an event-confirmation requirement.

- [ ] **Step 4: Add multi-frame consensus**

`vp_ocr_consensus` must group candidates by `(channel, track_id)`, normalize numeric strings, count confidence-weighted votes, and return `PENDING` on unresolved conflicts. A one-frame result cannot finalize an athlete bib.

- [ ] **Step 5: Test queue backpressure and failure behavior**

`test_ocr_dispatch.cpp` must assert bounded queue size, deterministic drop count, worker shutdown, OCR exception handling, retained athlete target, retained evidence path, and `PENDING` status for conflicting candidates.

- [ ] **Step 6: Compare old and new OCR paths**

Run both synchronous OCR and async OCR on the same clips. Promotion requires equal or higher athlete evidence count, no OCR-induced frame backlog, and no increase in duplicate events.

## Task 6: Runtime Upgrade Pilot

**Files:**
- Create: `docs/cycling-runtime-matrix.md`
- Modify: `CMakeLists.txt`
- Modify: `third_party/paddle_ocr/README.md` only after the new runtime is measured

- [ ] **Step 1: Record the current baseline**

Document the repository baseline: OpenCV 4.6 minimum, GStreamer 1.14.5, CUDA 11.1, TensorRT 8.5, and PaddleOCR C++ code derived from release 2.6. Record the actual installed versions from the target deployment machine, not only README values.

- [ ] **Step 2: Build an isolated pilot environment**

Create a separate build directory and environment profile for the YOLO11 engine. Keep the current runtime untouched. Record GPU model, driver, CUDA, TensorRT, OpenCV, GStreamer, and Paddle Inference versions in the matrix.

- [ ] **Step 3: Upgrade TensorRT/CUDA only as a matched pair**

Build the YOLO11 ONNX engine in the pilot environment, run `trtexec` or the project detector sample, and compare output boxes against the ONNX reference. Do not update the old YOLOv8 builder in place.

- [ ] **Step 4: Evaluate PaddleOCR 3.x separately**

Wrap the new OCR runtime behind the same `vp_ocr_job -> vp_ocr_consensus` interface. Keep the old PaddleOCR implementation as the fallback until recognition and latency are measured on the same crop set.

- [ ] **Step 5: Defer OpenCV/GStreamer upgrade unless a measured failure exists**

Only upgrade them when the matrix contains a reproducible RTSP, hardware decode, color conversion, or ABI failure. Re-run stream tests, recorder tests, and the 30-minute reconnect test after every runtime change.

## Task 7: Field Validation And Rollout

**Files:**
- Modify: `docs/cycling-vision-acceptance.md`
- Modify: `docs/cycling-runtime-matrix.md`
- Create: `docs/cycling-field-checklist.md`

- [ ] **Step 1: Validate camera placement**

For the selected 4K PoE varifocal camera, record daytime, backlight, dusk, dense-pack, and rain/wet-road clips. Confirm H.264 RTSP, stable FPS, manual shutter, fixed focus, and no frame skipping.

- [ ] **Step 2: Validate two deployment profiles**

Run `baseline` with YOLOv8 + SORT + synchronous OCR and `candidate` with YOLO11 + ByteTrack + async OCR. Keep camera, clip, thresholds, class map, and output directory layout identical.

- [ ] **Step 3: Apply promotion gates**

Promote a candidate only when all conditions hold: athlete recall is not lower than baseline, false events and duplicate events do not increase, evidence completeness is 100% for observed athletes, OCR backlog stays bounded, and end-to-end processing keeps up with the source stream.

- [ ] **Step 4: Roll back by configuration**

The production profile must select detector, tracker, OCR backend, and runtime directory independently. A failed candidate must be disabled by configuration without rebuilding or deleting the baseline engine.

- [ ] **Step 5: Publish the final field report**

Record camera SKU/lens, mounting distance, resolution/FPS, exposure settings, GPU/runtime versions, model hashes, dataset hash, all acceptance metrics, known misses, and the selected rollback profile. Do not include private race data or credentials.

## Verification Commands

Run after each code task:

```powershell
python -m unittest discover -s tools/cycling_eval -p "test_*.py" -v
cmake -S . -B build-baseline -DVP_WITH_CUDA=OFF -DVP_WITH_TRT=OFF -DVP_WITH_PADDLE=OFF -DVP_BUILD_TESTS=ON
cmake --build build-baseline --config Debug
ctest --test-dir build-baseline -C Debug --output-on-failure
git diff --check
```

Run on a target deployment machine before promotion:

```powershell
cmake -S . -B build-cycling -DVP_WITH_CUDA=ON -DVP_WITH_TRT=ON -DVP_WITH_PADDLE=ON -DVP_BUILD_TESTS=ON
cmake --build build-cycling --config Release
ctest --test-dir build-cycling -C Release --output-on-failure
```

The second command set is a deployment validation, not a substitute for real-video acceptance. If CUDA/TensorRT/Paddle dependencies are unavailable, keep the baseline build result separate from the feature result and record the missing dependency in the runtime matrix.

## Self-Review Checklist

- [ ] Every proposed change has an exact file boundary.
- [ ] YOLOv8/SORT/synchronous OCR remain rollback paths.
- [ ] OCR failure never removes an athlete observation or evidence file.
- [ ] Camera and runtime changes are gated by real video, not only compilation.
- [ ] No formal CycleRace result tables are written by this plan.
- [ ] No race videos, model weights, TensorRT engines, keys, or tokens are committed.
