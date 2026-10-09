# 独立过线计数与增量对账

本轮已实现单路计数服务、芯片修订对账、异步原帧补扫及 FinishReview 未决短窗入口。服务复用现有 Ultralytics，不加载 OCR，不用芯片数量调整检测或过滤。**密集集团冲刺准确率尚未通过完整实片验收，当前交付供开发验证，不能视作现场验收完成。**

## 启动

需要 Python 3.10+ 以及现有 NumPy、OpenCV、Ultralytics 环境；UI 与测试还需 PyQt5、pytest。计数服务不加载 OCR。本轮没有安装新依赖。远端基线尚没有 `realtime` 包，因此本提交同时带入运行所需的既有基础模块，没有上传本地旧提交中的训练数据或虚拟环境。

在 `VideoPipe` 目录执行：

```powershell
python -m realtime.count_main --config realtime/counting.example.json --validate-config
python -m realtime.count_main --config path/to/counting.json
```

先复制示例并填写现场配置。示例不会修改旧 UI 或生产配置；旧“号码识别”计数不是本服务的计数出口。需要同时启动 FinishReview 的既有芯片接收，将其实际 `cyclerace_passage_events.jsonl` 路径填入 `chip_journal`。点击 FinishReview 的“计数对账”，选择本服务输出目录；默认只列补扫后未决的窗口，支持关键帧和短片。

`--validate-config` 检查配置结构、路径和声明，不连接摄像头，不证明现场规则已正确。`--max-frames N` 供有限帧开发检查。

## 必须明确的现场规则

- `session_id/race_id/stage_id/timing_position/source_id` 必须与现场范围一致。协议 v1 芯片记录没有计时位置，必须让输入日志只对应这一位置；不能混入其他位置或把多机位人数相加。首版比较整个可见范围，禁止仅按芯片组别筛选匿名视频目标。
- `line`、`roi` 均为原图像素坐标。示例的 1920×1080 竖线仅是占位，需实测现场赛道。朝下的竖线，左侧为正，右侧为负，左向右通过用 `pos_to_neg`。
- 类别由本地模型的 `names` 确认并固定。示例 COCO 模型的 `1` 是 bicycle，不能同时计 person 和 bicycle，也不能把旧 bib 类别编号直接套用。启动时检查类别存在，但现场适用性仍需验证。
- `anchor` 选择检测框中心或底边中心，与芯片物理触发位置做校准。`side_observations=1` 适用于只有少量可见帧的高速片段，仍需两侧真实观测；提高此值可减少抖动，亦可能漏掉短轨。
- ROI、方向或可靠身份可排除无效通过；无号码、没有芯片匹配、慢速都不能单独作为排除依据。无可靠身份的同向工作人员仍可能计入并留下差异。

原始检测在跟踪关联前保留；计数专用路径先保存全部检测，再仅把允许类别交给 ByteTrack，避免人框与车框互换 ID。跟踪输出通过实际关联的检测索引引用原框。出口先于旧号码过滤、密集截断、几何去重和同步 OCR。过线事件保存真实侧别、前后观测时间区间、估计过线时刻、帧序号和排除原因。预测框、首次在线后出现及断轨只产生质量证据。

## 时间、迟到与修订

映射方向为：

```text
chip_time = camera_time + offset_ms
          + (camera_time - anchor_ms) * drift_ppm / 1_000_000
```

直播帧使用接收时钟，不能把它声称为曝光时间；网络延迟、检测锚点与芯片位置差都需计入校准与 `error_ms`。文件输入必须声明 `recording_start_ms`（相机时钟域），再加解码器的实际 PTS，禁止用帧号除名义 FPS 代替。可用 `start_position_ms/end_position_ms` 限定文件片段。

`calibrated=false` 时保留时间未校准异常。输出截止时刻取实际**处理完成**的相机进度，扣除误差，并受芯片完整性约束。没有明确水位时使用实测 `late_budget_ms`，结果始终标为暂定；心跳、最近事件和接收序号不代表数据已收齐。可追加如下明确水位记录，前提是发送方确实保证这一时间前的事件完整：

```json
{"record_type":"watermark","race_id":"race","stage_id":"stage","timing_position":"finish","completed_until_ms":1790000000000}
```

同一芯片 `event_id` 取最高 `revision`，先撤回旧贡献再插入新贡献。重发不累计，迟到、取消、改时间/组别均更新已有集团；同一 revision 的冲突会报错。缺绝对时间的有效芯片记录会阻止相关范围显示正常一致。校时改变可更新 `GroupReconciler.update_clock`，或使用相同输出目录按新校时配置重启；会重建所有时间贡献。

## 自动补扫与人工入口

数量不同，或数量相同但有重叠框、合框、断轨、未建轨、采样缺口等质量证据，都会触发补扫。集团按事件间隔形成，持续人流和长断流拆成有界窗口。补扫单线程、任务和内存有预算；使用独立模型与 tracker，不修改主计数轨迹。先保存证据，再异步高分辨率重放原帧。

原帧内存不足时，文件输入可从原文件读取；直播可配置 `video_timeline`、`camera_index`、`recording_timing_error_ms` 接 FinishReview 原录像。只接受有 `media_started_at_ms` 且时间误差在预算内的段；通常需段已封口，未校准录像不能冒充完整证据。跨段缺口、重连、超预算或超时均保留未决。

补扫匹配原事件时保留 UID 并增加 revision。补扫没有找到某人**不能删除原通过**；多候选关联不明确时不猜。可靠身份决定在匿名补扫后保留。质量问题没有解除，就算数量一样也继续列为未决。当前人工入口提供查看证据；可信身份修订可以通过以下追加日志或 API 接入，尚未提供 UI 中直接修改人数的操作。

将完整一行追加到输出目录的 `camera_passage_revisions.jsonl`：

```json
{"event_uid":"existing-camera-event-uid","revision":2,"identity":"official-1","identity_reliable":true,"eligibility":"excluded","reasons":["staff"]}
```

仅允许修改身份、资格、组别、重复关联与有效状态，不允许改原始观测或范围。排除/撤销必须有可靠非空身份与明确理由；`no_chip/unreadable_bib/slow_motion/unknown_identity` 不接受为排除理由。

## 输出与资源

| 文件 | 内容 |
| --- | --- |
| `camera_passages.jsonl` | 相机通过事件及所有修订 |
| `count_observations.jsonl` | 原始检测、关联观测、逐框决定 |
| `count_lifecycle.jsonl` | 质量证据、处理进度、补扫解除记录 |
| `count_reconciliation.json` | 原子更新的集团状态、截止时刻、暂定标志与证据路径 |
| `count_evidence/` | 关键帧、短片、原帧元数据 |
| `count_pipeline.json` | 固定规则、范围、模型和推理参数指纹 |

输出目录单写者加锁，可从日志恢复并防止同片重复累计。改模型、输入尺寸、类别、规则或会话时使用新目录；校时允许更新。写入日志尾部中断时保留 `.torn-tail` 并恢复完整行。

1920×1080 BGR 原帧约 6.22 MB，25 FPS 约 155.5 MB/s。默认 256 MiB 只够约 1.7 秒；示例 1.5 GiB 约够 10 秒，实际还受 15 秒保留时间限制。补扫也有独立同额内存预算，总占用还包含两套模型、解码器和主进程。长集团、迟到预算和上下文超过缓冲覆盖时依赖原录像，否则留未决。报告显示帧淘汰数、队列状态；不应仅凭模型平均 FPS 判断直播吞吐。逐帧诊断与证据日志会持续增长，需要现场容量规划。

## 已执行验证与仍缺的验收

本提交在干净检出上通过 40 项计数/帧时间相关测试，并复现实片及重启闭环结果。详细记录见 [验证摘要](../validation/README.md)。`validation/fixtures` 的视频路径是占位符，执行前需填写本地原录像和模型路径。

```powershell
# VideoPipe
python -m pytest tests/realtime -q -p no:cacheprovider --basetemp=.pytest_counting_verify
python -m realtime.count_main --config realtime/counting.example.json --validate-config
# VideoPipe: independent visual truth, no chip input
python tools/validate_independent_sprint.py --manifest validation/fixtures/independent_sprint_clear_fixture.json --output path/to/clear_result.json --imgsz 1280
python tools/validate_independent_sprint.py --manifest validation/fixtures/independent_sprint_fixture.json --output path/to/dense_result.json --imgsz 1280
# FinishReview
python -m pytest tests/realtime/test_count_review.py tests/realtime/test_review_window.py tests/realtime/test_passage_receiver.py tests/realtime/test_packaging_contract.py -q --basetemp=.pytest_counting_verify
```

清晰集团实片冻结视觉总数 3，得到 3，33 帧通过；六条重叠轨迹的计数、双向异常、迟到/撤销/修订、同圈重复与新圈通过、等数质量补扫、原录像预算、恢复等已有自动测试。

密集集团 50.0–51.5 秒片段当前得到 3；视觉证据至少 6，验证明确失败（命令返回 1）。有针对性的低阈值/宽匹配对照得到 8，但出现更多合框及反向轨迹，且未冻结完整逐人真值，因此没有把实验参数提升为默认，更不能把“接近芯片”当验收。

`validation/integration_summary.json` 是原录像＋**合成芯片**端到端检查，不是真实漏芯片准确率评估。它验证缓冲淘汰后的原录像补扫、保留原 UID/修订，以及 3 对 3 仍因质量证据留未决并有关键帧/短片。

后续必须以冻结的代表性集团逐人真值验证检测与关联，包含远侧并排、前后遮挡、逆向、多圈、工作人员；需在目标机按实际播放速度验证帧间隔、补扫积压和人工窗口量。完整遮挡没有可恢复图像，同向且不可识别的工作人员缺少身份依据，仍只能保留未决。当前首版整体现场验收未完成。
