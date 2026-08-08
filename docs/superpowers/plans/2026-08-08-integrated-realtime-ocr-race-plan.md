# 集显实时 YOLO + OCR 比赛交付计划

> **执行约束：** 本计划按测试先行、逐阶段门控实施。任何阶段未达到性能门槛，都必须停止叠加功能并使用上一阶段已验证包。未经明确授权，不提交、不推送、不安装新依赖。

**目标：** 在普通 Windows 集显/CPU 笔记本上保持流畅摄像头预览和可靠的 YOLO 过线事件，同时让号码牌 OCR 在独立后台进程中异步回填；OCR 失败、超时或崩溃不得影响画面、事件和证据保存。

**交付期限：** 2026-08-08 完成代码和开发机验证，2026-08-09 完成目标集显电脑实测与双包冻结，2026-08-10 比赛只运行已经通过实机门控的包。

**架构：** 保留当前已经进行的采集/预览/YOLO 解耦，取消所有批量、波次、巡检和逐帧 OCR。过线事件先落库并保存证据，再由主进程把一个事件任务非阻塞地交给低优先级 OCR 子进程。子进程只加载 `TextRecognition` 和 `en_PP-OCRv5_mobile_rec`，每个事件最多识别两张已筛选号码牌候选；主进程接收结果并更新数据库。子进程队列容量固定为 1，所有图片通过证据文件路径传递，不在进程间复制完整视频帧。

**技术栈：** Python、PyQt5、Ultralytics YOLO、Paddle `TextRecognition`、OpenCV、SQLite、`multiprocessing`、pytest、PyInstaller。

---

## 1. 已验证问题与决策

### 1.1 已验证性能事实

- RTX 4070 上 YOLO 单帧推理约为 `12-30 ms`，YOLO 本身不是当前卡死主因。
- 点击批量 OCR 后，界面帧率实测从 `13.7 FPS` 逐步跌到 `0.9 FPS`，并出现 `AppHangB1`。
- 当前完整 PaddleOCR 单张原图约 `3.739 s`，放大图最坏约 `37.252 s`。
- `TextRecognition + en_PP-OCRv5_mobile_rec` 单图约 `0.14-0.21 s`，四张候选约 `0.56-0.65 s/事件`。
- 轻量 OCR 留在主进程时，YOLO 吞吐下降约 `37.3%`；放入低优先级独立进程并限制两个逻辑核心后，YOLO 只下降约 `5.6%`。
- CPU YOLO 使用四个逻辑核心、OCR 独占两个逻辑核心时，实测 `16.89 -> 17.08 FPS`；较弱的后四个逻辑核心运行 YOLO 时仍约 `10.51 FPS`。

### 1.2 产品决策

1. 比赛版删除批量 OCR，不保留“临时可用”的隐藏入口。
2. 删除波次 OCR 和定时巡检补号，不允许历史任务突然占满 CPU。
3. 删除检测器内部连续逐帧 OCR 和过线同步 OCR，过线主链路只负责检测、事件和证据。
4. OCR 只在事件保存后执行，只使用轻量识别模型，不加载 `PP-OCRv5_server_det`。
5. OCR 是辅助号码候选，不是保留运动员观察的条件，也不是正式比赛结果来源。
6. 第一版比赛候选包继续使用 `best.pt`，不在截止日前临时引入 OpenVINO。
7. 同时交付 OCR 包和 YOLO-only 备用包；任何性能异常都可以立即切换备用包。

### 1.3 明确不做

- 不做网页版。
- 不接入云端大模型/VLM OCR。
- 不在比赛前更换 YOLO 训练模型或改变事件判定语义。
- 不自动写入 CycleRace 的正式计时或成绩数据。
- 不为了 OCR 成功而阻塞事件保存、等待识别结果或丢弃无号码运动员。
- 不在没有目标电脑数据时宣称“集显已通过”。

---

## 2. 最终运行链路

```text
Camera / RTSP / local video
          |
          v
StreamReader: capture latest frame, bounded buffer
          |
          +--------------------> Preview/UI: always display newest frame
          |
          v
YOLO worker: newest frame only, stale frames dropped
          |
          v
CrossingEvent -> EventRecorder -> SQLite + evidence images
                                  |
                                  | non-blocking event notification
                                  v
                     OCR supervisor in main process
                                  |
                     job queue, maxsize=1, path only
                                  v
                  low-priority OCR child process
                  TextRecognition/mobile_rec only
                                  |
                     result queue, maxsize=1
                                  v
                   main process updates event OCR
```

### 2.1 主链路不变量

- 摄像头读取、预览、YOLO、事件保存不等待 OCR。
- 推理消费最新帧，陈旧视频帧直接丢弃，不允许延迟队列持续增长。
- 每个过线事件先保存，即使号码为空或 OCR 不可用也必须存在。
- OCR 子进程不持有 Qt 对象、YOLO 模型或主进程 SQLite 连接。
- SQLite 更新只在主进程执行，避免子进程数据库连接与事件写入竞争。
- OCR 结果只能补充 `bib_number`、置信度和审核状态，不能删除事件。

### 2.2 OCR 任务语义

- 子进程工作队列容量为 1，只允许一个等待任务；当前执行任务不计入队列容量。
- 队列满时，事件仍以 `PENDING` 保存，提交动作立即返回，绝不阻塞 UI 或 YOLO。
- OCR 完成后，监督器按事件顺序尝试调度下一个尚未在本次运行中尝试的 `PENDING` 事件；这是一条事件驱动的串行链路，不是批量扫描或定时巡检。
- 每个事件在一次程序运行中最多自动尝试一次。失败、超时、无有效候选都保持 `PENDING`，留给人工核对，避免无限重试占用 CPU。
- 子进程异常退出时，当前事件恢复为 `PENDING`；监督器最多自动重启一次。再次异常则本次运行禁用 OCR，YOLO 和事件保存继续工作。

---

## 3. 性能与验收门槛

### 3.1 开发机门槛

| 指标 | YOLO-only | YOLO + OCR |
| --- | ---: | ---: |
| 预览显示 | `>= 25 FPS` | `>= 25 FPS` |
| YOLO 吞吐 | 记录基线 | 不低于基线的 `90%` |
| 持续画面停顿 | `< 200 ms` | `< 200 ms` |
| 视频最新帧队列 | `<= 1` | `<= 1` |
| OCR 子进程任务队列 | 不适用 | `<= 1` |
| 事件保存 | 不丢失 | 不丢失 |

### 3.2 目标集显电脑硬门槛

- 预览在连续 20 分钟测试中保持 `>= 25 FPS`，不得出现持续超过 `200 ms` 的卡顿。
- YOLO 平均吞吐 `>= 10 FPS`；目标值为 `>= 15 FPS`。
- 开启 OCR 后，YOLO 吞吐下降不超过 `10%`。
- 连续制造至少 30 个过线事件，事件数和证据目录数与 YOLO-only 基线一致。
- OCR 处理时 UI 可拖动、切换事件和停止运行，不能出现“未响应”或 `AppHangB1`。
- OCR 子进程内存异常、超时或强制终止后，主程序继续检测并保存事件。
- 号码识别正确率单独统计，不以错误号码覆盖人工已修正结果。

### 3.3 失败标准和决策门

1. **预览低于 25 FPS，但 YOLO 达标：** 先修复显示/解码/UI刷新，禁止继续优化 OCR。
2. **YOLO-only 低于 10 FPS：** 进入 OpenVINO 决策门；当前 `best.pt` 包只能作为开发诊断包，不能宣称集显可用。
3. **YOLO-only 达标，开启 OCR 后下降超过 10%：** 检查进程优先级、核心亲和性、Paddle线程数和候选数量；未解决前比赛使用 YOLO-only 包。
4. **OCR 正确率不够，但画面和事件稳定：** 保留 OCR 为 `PENDING/needs_review` 辅助信息，比赛不允许因此回退到完整 PaddleOCR。
5. **任何包出现卡死、事件丢失或证据缺失：** 立即判定该包不合格，不以平均 FPS 掩盖功能故障。

---

## 4. 实施任务

### Task 0: 冻结当前解耦基线

**文件：**
- Verify: `realtime/stream_reader.py`
- Verify: `realtime/main_window.py`
- Verify: `tests/realtime/test_stream_reader.py`
- Verify: `tests/realtime/test_video_thread.py`

- [ ] **Step 1: 记录当前工作树和现有改动**

不得清理当前未提交修改和测试/打包产物。先执行：

```powershell
git status --short --branch
git diff -- realtime/stream_reader.py realtime/main_window.py tests/realtime/test_stream_reader.py tests/realtime/test_video_thread.py
```

- [ ] **Step 2: 验证最新帧语义**

补齐或确认测试覆盖：预览可持续读取最新帧；YOLO 不重复消费同一帧；慢推理时旧帧被替换而不是累积；原始采集时间和帧号继续传入事件链路。

- [ ] **Step 3: 建立 YOLO-only 性能基线**

```powershell
python -m pytest tests/realtime/test_stream_reader.py tests/realtime/test_video_thread.py tests/realtime/test_detection_association.py -q --basetemp .pytest_tmp_race_yolo_baseline
python -m realtime.main --model .\best.pt --source .\test.mp4 --output .\RaceData-YOLO-Baseline --auto-start --yolo-only
```

记录 10 分钟平均/最低预览 FPS、YOLO FPS、丢帧数、最大处理延迟、CPU、内存、事件数和证据数。未形成这份基线，不进入 OCR 改动。

### Task 1: 删除所有高风险 OCR 入口

**文件：**
- Modify: `realtime/main_window.py`
- Modify: `realtime/ocr_manager.py`
- Modify: `realtime/detector.py`
- Delete: `tests/realtime/test_wave_ocr_scheduling.py`
- Modify: `tests/realtime/test_ocr_duplicate_merge.py`
- Modify: `tests/realtime/test_detection_association.py`

- [ ] **Step 1: 先写失败测试**

测试应证明：

- UI 不再创建“开始批处理 OCR”按钮；
- `OCRManager` 不再暴露 `start_batch()`；
- `MainWindow` 不再创建波次队列、波次定时器和自动巡检定时器；
- `Detector.process_frame()` 在 OCR 模式下也不会提交逐帧 OCR；
- 过线事件不会执行同步 OCR。

- [ ] **Step 2: 验证 RED**

```powershell
python -m pytest tests/realtime/test_ocr_duplicate_merge.py tests/realtime/test_detection_association.py tests/realtime/test_wave_ocr_scheduling.py -q --basetemp .pytest_tmp_race_remove_ocr_paths
```

预期：测试针对当前批量、波次、巡检、逐帧和同步 OCR 行为失败。

- [ ] **Step 3: 最小删除旧链路**

删除：

- `batch_ocr_btn`、`_start_batch_ocr()` 及手工批处理状态；
- `OCRManager.start_batch()` 及目录全量扫描入口；
- `_wave_*` 配置、队列、定时器和调度函数；
- `_run_auto_ocr_patrol()` 及其定时器、重试状态；
- `Detector` 的逐帧 OCR 队列提交、过线短窗同步 OCR 和 VLM 兜底调用。

保留事件证据保存、OCR文本规范化、名单/号码段校验、结果回填和人工修正保护。

- [ ] **Step 4: 验证 GREEN**

```powershell
python -m pytest tests/realtime/test_ocr_duplicate_merge.py tests/realtime/test_detection_association.py tests/realtime/test_main_startup.py -q --basetemp .pytest_tmp_race_removed_ocr_paths_green
```

### Task 2: 建立可审计的号码牌候选结构

**文件：**
- Modify: `realtime/detector.py`
- Modify: `realtime/event_recorder.py`
- Modify: `tests/realtime/test_detection_association.py`
- Modify: `tests/realtime/test_ocr_duplicate_merge.py`
- Modify: `tests/realtime/test_regression_metrics.py`

- [ ] **Step 1: 写候选关联失败测试**

用不同帧、不同运动员和躯干误框构造测试，证明当前四元组不足以判断候选是否属于过线运动员。

- [ ] **Step 2: 增加候选数据结构**

把 `(quality, crop, frame_ref, bib_bbox)` 替换为明确字段的数据类，至少包含：

```python
@dataclass
class BibOcrCandidate:
    quality: float
    crop: np.ndarray
    frame_index: int
    capture_time_ms: float
    athlete_bbox: tuple[int, int, int, int]
    bib_bbox: tuple[int, int, int, int]
    source: str  # detected | fallback
```

候选必须在产生时记录同一帧的运动员框，不得在过线时把历史号码框与当前运动员框重新拼接。

- [ ] **Step 3: 实现保守候选筛选**

- 检测到的号码牌候选优先于躯干兜底。
- 候选必须与同帧运动员框满足包含关系、相对位置和面积比例约束。
- 过小候选不再直接消失：可以作为第二候选保留，但不得仅因清晰度高排在可信检测框之前。
- 躯干兜底最多保留一张，并明确标记 `fallback`。
- 每个事件最多保存两张 OCR 候选；不再保存四张用于多尺度批处理。

- [ ] **Step 4: 扩展证据元数据**

`event_recorder.py` 在事件元数据中保存每张候选的帧号、采集时间、运动员框、号码框、来源和质量分。旧事件目录没有这些字段时仍可展示，但不自动做比赛实时 OCR。

- [ ] **Step 5: 验证**

```powershell
python -m pytest tests/realtime/test_detection_association.py tests/realtime/test_ocr_duplicate_merge.py tests/realtime/test_regression_metrics.py -q --basetemp .pytest_tmp_race_candidate_metadata
```

重点人工检查已知误例：`205 -> 206`、`125` 小框被跳过、`145 -> 45`、`66 -> GG`，以及事件 8、11 的躯干/错人候选。目标不是立即保证全识别正确，而是保证 OCR 输入确实属于正确运动员并可追溯。

### Task 3: 只初始化轻量识别模型

**文件：**
- Modify: `realtime/main_window.py`
- Modify: `realtime/detector.py`
- Modify: `tests/realtime/test_ocr_init.py`

- [ ] **Step 1: 改写 OCR 初始化测试并验证 RED**

测试应断言只构造：

```python
TextRecognition(
    model_name="en_PP-OCRv5_mobile_rec",
    model_dir=...,
    cpu_threads=1,
)
```

同时断言 `PaddleOCR` 和 `PP-OCRv5_server_det` 不被导入、不被构造。

```powershell
python -m pytest tests/realtime/test_ocr_init.py -q --basetemp .pytest_tmp_race_mobile_rec_red
```

- [ ] **Step 2: 移除完整 OCR 初始化**

从 `OCRInitThread` 移除 `PaddleOCR`，并把识别模型的创建职责迁入后续 OCR 子进程。主进程只维护 `starting/ready/busy/failed/disabled` 状态，不持有 Paddle 推理对象。

- [ ] **Step 3: 保留兼容适配边界**

识别结果解析继续通过小型适配函数完成，接受 Paddle 版本可能返回的对象/字典差异；不得退回完整检测模型兜底。

- [ ] **Step 4: 验证 GREEN**

```powershell
python -m pytest tests/realtime/test_ocr_init.py -q --basetemp .pytest_tmp_race_mobile_rec_green
```

### Task 4: OCR 独立进程和资源隔离

**文件：**
- Create: `realtime/ocr_worker.py`
- Modify: `realtime/main.py`
- Modify: `realtime/main_window.py`
- Modify: `realtime/ocr_manager.py`
- Create: `tests/realtime/test_ocr_worker.py`
- Modify: `tests/realtime/test_main_startup.py`

- [ ] **Step 1: 写纯协议测试**

先测试可序列化任务和结果，不启动真实 Paddle：

```python
OcrJob(event_id, candidate_paths, numeric_only, bib_ranges)
OcrResult(event_id, text, confidence, status, elapsed_ms, error)
```

任务只携带事件 ID、最多两个图片路径和号码规则，不携带 `np.ndarray`、Qt 对象、YOLO 对象或数据库连接。

- [ ] **Step 2: 写进程生命周期失败测试**

覆盖：

- 队列 `maxsize=1`；
- 队列满时 `submit()` 在短时间内返回 `False`；
- 子进程只初始化一次识别模型；
- 停止时发送哨兵并有限时等待；
- 工作超时、异常退出和无响应都不会阻塞主进程；
- 最多自动重启一次，第二次失败进入 `disabled`；
- Windows `spawn` 和 PyInstaller 入口不会递归创建窗口。

- [ ] **Step 3: 实现进程入口**

`realtime/main.py` 在导入/创建 Qt 窗口前调用 `multiprocessing.freeze_support()`。`ocr_worker.py` 使用 `multiprocessing.get_context("spawn")`，在子进程内部导入 Paddle 并创建 `TextRecognition`。

- [ ] **Step 4: 限制 CPU 资源**

子进程默认：

- `cpu_threads=1`，只有目标机实测证明不影响 YOLO 时才允许 `2`；
- Windows 进程优先级设为 `BELOW_NORMAL_PRIORITY_CLASS`；
- 优先绑定到预留的 1-2 个逻辑核心；若系统不支持或核心数不足，记录警告并继续使用低优先级，不让启动失败。

不新增第三方进程管理依赖，优先使用 Python 标准库和 Windows API。

- [ ] **Step 5: 主进程监督和结果回填**

`OCRManager` 改为轻量监督器：事件保存后非阻塞提交；用 Qt 定时器或现有状态刷新周期轮询小型结果队列；主进程执行号码规范化、名单校验和数据库更新。人工已修正事件不得被异步结果覆盖。

- [ ] **Step 6: 验证**

```powershell
python -m pytest tests/realtime/test_ocr_worker.py tests/realtime/test_ocr_init.py tests/realtime/test_main_startup.py tests/realtime/test_ocr_duplicate_merge.py -q --basetemp .pytest_tmp_race_ocr_process
python -m compileall -q realtime tests/realtime
```

### Task 5: 每事件一次、最多两张的轻量识别

**文件：**
- Modify: `realtime/ocr_worker.py`
- Modify: `realtime/ocr_manager.py`
- Modify: `tests/realtime/test_ocr_worker.py`
- Modify: `tests/realtime/test_ocr_duplicate_merge.py`

- [ ] **Step 1: 写有界工作量测试**

断言：一个事件最多调用识别器两次；不会执行文本检测、多尺度循环、多角度循环、颜色通道循环或 VLM；第一张得到高可信合法号码时不再识别第二张。

- [ ] **Step 2: 实现最小预处理**

只允许确定性的轻量步骤，例如保持比例放大、灰度/对比度增强二选一。禁止为一张图生成大量变体。任何候选总耗时超过配置超时时，返回超时结果。

- [ ] **Step 3: 结果规则**

- 合法且高置信度：`RECOGNIZED`；
- 有数字但冲突或置信度不足：`NEEDS_REVIEW`；
- 无合法结果、超时或异常：保持 `PENDING`；
- 永远不把纯字母 `GG` 等结果写成号码；
- 两个候选冲突时不得用简单最高分静默覆盖，应标记人工审核。

- [ ] **Step 4: 验证耗时上限**

用固定 30 个事件证据样本记录 P50/P95/最大耗时。开发机目标为单事件 P95 `< 0.7 s`，不得出现数秒到数十秒的完整 OCR 调用。

### Task 6: UI 状态和失败降级

**文件：**
- Modify: `realtime/main_window.py`
- Modify: `tests/realtime/test_main_startup.py`
- Create: `tests/realtime/test_ocr_ui_state.py`

- [ ] **Step 1: 写状态转换测试**

覆盖 `starting -> ready -> busy -> ready`、`timeout -> ready`、`crashed -> restarting -> ready` 和第二次崩溃后的 `disabled`。

- [ ] **Step 2: 精简 UI**

- 保留一个只读 OCR 状态指示：未启动、就绪、识别中、失败/已停用。
- 删除“实时 OCR”复选框和批量按钮，避免用户重新开启旧串行链路。
- 事件列表允许显示 `PENDING`、`NEEDS_REVIEW`、`RECOGNIZED`。
- OCR 失败提示不得使用阻塞式消息框打断比赛操作。

- [ ] **Step 3: 停止和退出**

停止视频时不强制杀死仍可正常退出的 OCR 进程；关闭应用时先停止接收新任务，再发送哨兵，有限时等待后才终止子进程。退出过程不得访问已销毁 Qt 对象。

- [ ] **Step 4: 无界面自动化验证**

```powershell
$env:QT_QPA_PLATFORM='offscreen'
python -m pytest tests/realtime/test_main_startup.py tests/realtime/test_ocr_ui_state.py -q --basetemp .pytest_tmp_race_ocr_ui
Remove-Item Env:QT_QPA_PLATFORM
```

### Task 7: OCR 包瘦身和双包构建

**文件：**
- Modify: `packaging/VideoPipeRealtimeOCR.spec`
- Modify: `packaging/VideoPipeRealtimeYOLO.spec`
- Modify: `packaging/build_realtime.ps1`
- Modify: `packaging/README.md`
- Create: `tests/realtime/test_packaging_contract.py`

- [ ] **Step 1: 写打包契约测试**

断言 OCR spec：

- 只包含 `en_PP-OCRv5_mobile_rec`；
- 不包含 `PP-OCRv5_server_det`；
- 不把 CUDA/TensorRT/OpenVINO 作为本阶段依赖；
- YOLO-only 包不包含 Paddle/PaddleOCR/PaddleX；
- 启动器传入正确模型、视频源和模式参数。

- [ ] **Step 2: 收窄 OCR 收集范围**

避免对 Paddle 生态无差别 `collect_all()`。根据实际导入追踪只收集运行 `TextRecognition` 必需模块、二进制、元数据和 `7.64 MB` 移动识别模型。打包后检查启动，不能只以体积变小作为成功标准。

- [ ] **Step 3: 生成两个候选包**

```powershell
.\packaging\build_realtime.ps1 -Variant YOLO -ModelPath .\best.pt -SourcePath .\test.mp4
.\packaging\build_realtime.ps1 -Variant OCR -ModelPath .\best.pt -SourcePath .\test.mp4 -OcrModelsPath "$HOME\.paddlex\official_models"
```

产物：

- `VideoPipeRealtimeYOLO`: 比赛稳定备用包；
- `VideoPipeRealtimeOCR`: 独立轻量 OCR 候选包。

- [ ] **Step 4: 干净目录冒烟测试**

分别从构建目录外启动两个包，确认模型、tracker 配置、OCR 模型和输出目录都从运行目录正确解析；连续运行至少 10 分钟并正常关闭。

### Task 8: 开发机回归和故障注入

**文件：**
- Test only: `tests/realtime/**`
- Generate only: `artifacts/race-readiness-20260808/**`

- [ ] **Step 1: 自动测试**

```powershell
python -m pytest tests/realtime/test_stream_reader.py tests/realtime/test_video_thread.py tests/realtime/test_detection_association.py tests/realtime/test_ocr_init.py tests/realtime/test_ocr_worker.py tests/realtime/test_ocr_duplicate_merge.py tests/realtime/test_main_startup.py tests/realtime/test_ocr_ui_state.py tests/realtime/test_packaging_contract.py -q --basetemp .pytest_tmp_race_integrated
python -m compileall -q realtime tests/realtime
git diff --check
```

- [ ] **Step 2: 三组运行对比**

使用同一视频、同一模型、同一 `imgsz` 和同一输出配置，分别运行：

1. YOLO-only；
2. YOLO + 独立 OCR；
3. YOLO + 独立 OCR，并连续制造 OCR 队列满、识别超时和子进程崩溃。

输出 CSV/JSON 指标：预览 FPS、YOLO FPS、P50/P95帧耗时、最大停顿、捕获丢帧、推理跳帧、事件数、证据数、OCR队列深度、OCR耗时、OCR成功/审核/PENDING数量、CPU和内存。

- [ ] **Step 3: 正确性对照**

YOLO + OCR 与 YOLO-only 的运动员事件数、事件时间和证据必须一致。OCR 只能增加号码候选信息，不能改变是否生成事件。

### Task 9: 2026-08-09 目标集显电脑实测

**执行位置：** 实际比赛笔记本，不是开发台式机。

- [ ] **Step 1: 记录硬件和运行环境**

记录 CPU 型号、物理/逻辑核心数、内存、集显型号、Windows 版本、摄像头/采集卡型号、输入分辨率、输入 FPS、视频编码、存储盘类型和电源模式。

- [ ] **Step 2: 先测 YOLO-only**

使用实际摄像头连续运行 20 分钟，完成静态、单人、多人并发和快速过线。若 YOLO 平均低于 `10 FPS` 或预览低于 `25 FPS`，停止 OCR 测试，直接进入 Task 10 决策。

- [ ] **Step 3: 再测 OCR 包**

使用相同场景运行 20 分钟，至少制造 30 个事件和一组密集过线。对比 YOLO-only，确认吞吐下降不超过 `10%`、无卡死、无事件/证据损失。

- [ ] **Step 4: 摄像头链路检查**

实测 USB/RTSP 摄像头而不是只测本地视频。记录实际解码 CPU 占用；若 RTSP 解码明显成为瓶颈，再单独评估硬件解码，不能把本地视频结果当作摄像头结论。

- [ ] **Step 5: 冻结比赛包**

只冻结通过全部硬门槛的包，记录 Git SHA、模型 SHA256、包 SHA256、配置、启动命令和目标机指标。未通过的 OCR 包不得在比赛现场继续试错。

### Task 10: OpenVINO 条件分支

**触发条件：** 仅当目标机 YOLO-only 平均吞吐低于 `10 FPS`，且确认瓶颈是 YOLO CPU 推理而非摄像头解码或 UI。

- [ ] **Step 1: 保留当前可运行包**

先归档 `best.pt` 双包和全部实测数据。OpenVINO 是新的候选分支，不覆盖已验证产物。

- [ ] **Step 2: 明确新增依赖授权**

安装 OpenVINO、导出模型或修改 spec 前必须单独说明原因并获得授权。比赛前不同时更换 OCR、YOLO模型、推理后端和摄像头配置。

- [ ] **Step 3: 建立等价性门槛**

OpenVINO 候选必须与 `best.pt` 在同一视频上比较 Athlete/BIB 检测数、置信度、过线事件、号码牌小框保留率和速度。速度提高但号码牌召回明显下降时不得采用。

- [ ] **Step 4: 最晚停止条件**

若 2026-08-09 无法完成实机等价性测试，2026-08-10 比赛使用已经验证的 YOLO-only 包，不在现场首次启用 OpenVINO。

---

## 5. 比赛日运行与回退

### 5.1 启动前

- 使用 2026-08-09 冻结的包、模型和配置，不临时更新代码。
- 确认摄像头 30 FPS、固定曝光/快门、号码牌清晰度、磁盘剩余空间和输出目录可写。
- 先录制短视频并回放检查真实号码牌像素尺寸，不能只看预览“感觉清楚”。
- 同时保留源视频录制；VideoPipe 只作为辅助视频证据系统。

### 5.2 运行选择

- OCR 包通过集显门槛：优先运行 OCR 包。
- OCR 包未通过或现场出现一次明显卡死：立即停止并切换 YOLO-only 包。
- YOLO-only 仍卡：降低非关键 UI 刷新和 YOLO输入尺寸前先保留源视频；不得为了帧率关闭事件证据。

### 5.3 回退后数据原则

- 已经保存的事件和证据不删除、不覆盖。
- OCR 未完成事件保持 `PENDING`，赛后人工查看图片。
- 不因 OCR 失败生成 DNS/DNF、名次或正式计时结果。
- CycleRace 继续作为唯一正式计时和成绩权威。

---

## 6. 风险清单

| 风险 | 影响 | 控制措施 |
| --- | --- | --- |
| Paddle 子进程仍抢占全部 CPU | YOLO下降、画面卡顿 | 低优先级、1线程、核心亲和性、队列1、实机10%门槛 |
| Windows/PyInstaller 递归 spawn | 启动闪退或重复窗口 | `freeze_support()`、独立顶层 worker、打包冒烟测试 |
| OCR 候选属于错人/错帧 | 错号码回填 | 候选保存同帧运动员框、帧号和时间；最多两张；冲突转人工审核 |
| 密集过线产生 OCR 积压 | 内存上涨、延迟扩大 | 图片先落盘、子队列1、事件驱动串行调度、每事件每次运行只尝试一次 |
| OCR 子进程崩溃 | OCR中断 | 当前事件恢复PENDING、最多重启一次、主链路继续 |
| 打包仍包含完整 Paddle 生态 | 包体过大、加载慢 | 打包契约测试、收窄收集、检查模型和依赖清单 |
| 本地视频通过但摄像头失败 | 比赛现场不可用 | 2026-08-09 必须使用实际摄像头连续实测 |
| OpenVINO临时切换导致召回下降 | 漏运动员/号码牌 | 只在YOLO<10 FPS触发，做同视频等价性门控，未完成则不用 |

---

## 7. 完成定义

本计划只有在以下条件全部满足时才算完成：

- 旧批量、波次、巡检、逐帧和同步 OCR 路径已删除，相关测试已更新。
- 主进程不加载完整 PaddleOCR，也不持有轻量识别模型。
- OCR 独立进程队列容量 1、每事件最多两张、可超时、可恢复、可禁用。
- YOLO + OCR 与 YOLO-only 的事件和证据结果一致。
- 两个 PyInstaller 包都能在干净目录启动和正常退出。
- 目标集显电脑使用实际摄像头完成 20 分钟实测。
- 预览、YOLO吞吐、OCR影响、事件完整性和故障降级达到硬门槛。
- 比赛包的代码、模型、配置、校验值和回退包已冻结记录。

未达到目标机门槛时，完成结论只能是“YOLO-only 比赛包可用”或“当前目标机不可用”，不能以开发机 4070 结果替代集显验收。

未经明确授权，本计划本身不执行提交、推送、依赖安装或发布。
