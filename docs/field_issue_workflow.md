# 现场问题标记与复现包

现场问题标记只用于 VideoPipe 赛后诊断。它不会创建、修改或删除过线记录，也不会写入 CycleRace 数据库。

## 现场标记

1. 先选择或创建当前赛事目录并开始视频处理。
2. 发现漏检、重复、误检、号码错误、界面卡顿或相机重连等问题时，点击“标记问题”。
3. 系统会立即锁定最新已处理帧的原始画面和时间上下文，再显示问题类型与备注输入框。
4. 截图和 `issues.jsonl` 在后台写入赛事目录；状态栏会显示保存结果。

支持的问题类型：

- `missed_athlete`
- `duplicate_athlete`
- `false_athlete`
- `wrong_bib`
- `missing_bib`
- `ui_freeze`
- `camera_reconnect`
- `other`

每条记录包含机位、帧号、采集时间、重连分段、关联参赛者/轨迹/事件、队列与丢帧指标、软件 commit、模型标识、赛事配置和操作员备注。没有号码或截图写入失败时，时间标记仍可保留。

## 赛后导出

保留原始比赛视频和完整赛事目录，然后执行：

```powershell
python tools/export_field_issue_package.py `
  --session "C:\RaceData\current-session" `
  --video "D:\recordings\current-race.mp4" `
  --output "C:\Users\Administrator\Documents\video_validation\current-race-issues"
```

导出包包含：

- `issues.jsonl`
- `session.json`
- `source_video.json`
- `checksums.json`
- `database/`
- `evidence/`
- `logs/`

默认不复制大型源视频，只记录绝对路径、大小、时长和 SHA-256。确需复制时显式增加 `--copy-video`。
导出目标必须是不存在或为空的目录，工具不会覆盖已有复现包。

## 安全边界

- 导出前保留原始赛事目录，不在现场清理证据。
- 会话元数据、问题备注和日志中的常见凭据字段会被替换为 `[REDACTED]`。
- 复现包用于离线分析和编写失败回归测试，不用于自动修正官方成绩。
- CycleRace 始终是正式计时与成绩的唯一权威。
