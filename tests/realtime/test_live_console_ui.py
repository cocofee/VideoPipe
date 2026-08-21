import json
import os
import threading
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtGui import QColor, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QLabel

import realtime.main_window as main_window
from realtime.event_list_widget import EventListWidget
from realtime.live_event_review import LiveEventReview
from realtime.main_window import (
    ExternalClipProbeThread,
    MainWindow,
    NewRaceDialog,
    VLMConfigDialog,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _event(event_id, bib, status="recognized", cross_time=1.0, is_void=0, screenshot=""):
    return {
        "event_id": event_id,
        "bib_number": bib,
        "bib_status": status,
        "cross_time": cross_time,
        "cross_time_str": f"00:00:{cross_time:06.3f}",
        "source_id": 0,
        "is_void": is_void,
        "screenshot_full": screenshot,
        "bbox": "[]",
    }


def test_live_event_review_uses_saved_evidence_and_emits_actions(qapp, tmp_path):
    image_path = tmp_path / "evidence.jpg"
    pixmap = QPixmap(320, 180)
    pixmap.fill(QColor("#335577"))
    assert pixmap.save(str(image_path))

    review = LiveEventReview(output_dir=tmp_path)
    review.resize(520, 360)
    review.show()
    review.set_event(_event(7, None, status="unrecognized", screenshot=str(image_path)))
    qapp.processEvents()

    assert len(review.current_evidences) == 1
    assert review.current_evidences[0]["label"] == "全景"
    assert not review.image_label.pixmap().isNull()
    assert review._thumbnail_buttons[0].size().width() == 72
    assert review._thumbnail_buttons[0].size().height() == 52

    bib_changes = []
    next_requests = []
    void_changes = []
    review.bib_changed.connect(lambda event_id, bib: bib_changes.append((event_id, bib)))
    review.next_requested.connect(next_requests.append)
    review.void_changed.connect(lambda event_id, is_void: void_changes.append((event_id, is_void)))

    review.bib_input.setText("A0123")
    review._save_and_next()
    review._toggle_void()

    assert bib_changes == [(7, "A0123")]
    assert next_requests == [7]
    assert void_changes == [(7, True)]
    review.close()


def test_pending_ocr_candidate_is_visible_but_not_formal(qapp, tmp_path):
    event_dir = tmp_path / "evidence_photos" / "000007"
    event_dir.mkdir(parents=True)
    (event_dir / "result.json").write_text(
        json.dumps(
            {
                "event_id": 7,
                "bib": "014",
                "confidence": 0.99,
                "status": "PENDING",
                "error": "FALLBACK_ONLY_EVIDENCE",
                "source": "event_vlm:OpenAIVLMAssistant",
            }
        ),
        encoding="utf-8",
    )
    event = _event(7, None, status="unrecognized")
    event["evidence_dir"] = str(event_dir)
    event["ocr_state"] = "PENDING"

    widget = EventListWidget(None)
    widget._on_all_events_fetched([event])
    qapp.processEvents()

    assert widget.table.item(0, 1).text() == "014"
    assert widget.table.item(0, 5).text() == "待核对"
    assert widget._get_event_at_row(0)["bib_number"] is None
    assert widget._get_event_at_row(0)["_ocr_candidate"] == "014"

    review = LiveEventReview(output_dir=tmp_path)
    review.set_event(event)
    qapp.processEvents()

    assert review.bib_input.text() == "014"
    assert "OCR候选 014" in review.feedback_label.text()
    widget.close()
    review.close()


def test_event_queue_keeps_selection_when_new_event_arrives(qapp):
    widget = EventListWidget(None)
    widget.show()
    widget._on_all_events_fetched([
        _event(1, "101", cross_time=1.0),
        _event(2, None, status="unrecognized", cross_time=2.0),
    ])
    qapp.processEvents()

    assert widget.table.isColumnHidden(2)
    assert widget.table.isColumnHidden(4)
    assert widget.table.isColumnHidden(7)
    assert widget.table.horizontalHeaderItem(0).text() == "视频序号"
    assert widget.table.verticalHeader().defaultSectionSize() == 44
    assert widget.filter_combo.isHidden()
    assert widget.filter_all_btn.text() == "全部 2"
    assert widget.filter_review_btn.text() == "需处理 1"
    assert widget.filter_all_btn.isChecked()

    unrecognized_row = next(
        row
        for row in range(widget.table.rowCount())
        if widget._get_event_at_row(row)["event_id"] == 2
    )
    assert widget.table.item(unrecognized_row, 1).text() == "--"

    selected = []
    widget.event_selected.connect(lambda event: selected.append(event["event_id"]))
    assert widget.select_event_by_id(1, emit_signal=True)
    widget.add_event(_event(3, "303", cross_time=3.0))

    assert widget._selected_event_id() == 1
    assert selected == [1]
    assert widget.new_events_btn.isVisible()
    assert widget.new_events_btn.text() == "新增 1 条"
    assert widget.filter_all_btn.text() == "全部 3"
    assert widget.filter_review_btn.text() == "需处理 1"

    widget._jump_to_latest_unseen()
    assert widget._selected_event_id() == 3
    assert selected == [1, 3]
    assert widget.new_events_btn.isHidden()

    widget._set_filter_value("需处理")
    assert widget.filter_combo.currentText() == "需处理"
    assert widget.filter_review_btn.isChecked()
    widget.close()


def test_main_window_uses_live_console_structure(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(MainWindow, "_prompt_race_selection", lambda self: None)
    monkeypatch.setattr(MainWindow, "_init_ocr_runtime", lambda self: None)

    window = MainWindow({
        "source": "unused.mp4",
        "model_path": str(tmp_path / "unused.pt"),
        "output_dir": str(tmp_path),
        "yolo_only_mode": True,
        "live_monitor_enabled": False,
    })
    window.show()
    qapp.processEvents()

    assert window.windowTitle() == "VideoPipe 视频取证工作台"
    assert not hasattr(window, "rank_label")
    assert not hasattr(window, "time_label")
    assert not hasattr(window, "index_label")
    assert not window.model_path_input.isVisible()
    assert window.log_output.isHidden()
    assert not window.event_list.isHidden()
    assert not window.live_event_review.isHidden()
    assert window.start_btn.text() == "开始采集"
    assert window.objectName() == "video_evidence_main_window"
    assert window.splitter.handleWidth() == 5
    assert window.right_splitter.handleWidth() == 5
    assert window.conn_status_label.isHidden()
    assert window.stats_status_label.isHidden()
    assert window.session_count_label.text() == "本轮新增 0"
    assert [window.sport_profile_combo.itemData(i) for i in range(window.sport_profile_combo.count())] == [
        "cycling",
        "running",
        "speed_skating",
    ]
    assert window.sport_profile_combo.currentData() == "cycling"
    assert not hasattr(window, "numeric_only_action")
    assert not hasattr(window, "gate_guard_action")
    settings_text = [
        action.text()
        for action in window.settings_btn.menu().actions()
    ]
    options_menu = next(
        action.menu()
        for action in window.settings_btn.menu().actions()
        if action.text() == "识别选项"
    )
    settings_text.extend(action.text() for action in options_menu.actions())
    assert "号码段设置..." not in settings_text
    assert "仅限纯数字" not in settings_text
    assert "龙门安全模式" not in settings_text
    assert window.ocr_status_label.text() == "OCR: 已关闭"
    assert not window.ocr_status_label.isEnabled()

    data_menu = next(
        action.menu()
        for action in window.menuBar().actions()
        if action.text().startswith("数据管理")
    )
    data_actions = [action.text() for action in data_menu.actions()]
    assert "打开已有赛事..." in data_actions
    assert "录像回放..." in data_actions
    assert "导入高速摄像片段..." in data_actions

    visible_text = [label.text() for label in window.findChildren(QLabel)]
    assert not any(text.startswith("RANK ") for text in visible_text)
    assert not any(text.startswith("TIME ") for text in visible_text)

    window.event_list._on_all_events_fetched([_event(11, None, status="unrecognized")])
    qapp.processEvents()
    assert window.live_event_review.current_event["event_id"] == 11

    window._initialized = True
    window._running = True
    window._set_capture_state(True)
    assert not window.sport_profile_combo.isEnabled()
    assert window.capture_detail_label.text() == "等待机位连接"
    window.cam_fps_labels[0].setText("15 FPS")
    window._stop()
    assert not window.event_list.isHidden()
    assert not window.live_event_review.isHidden()
    assert window.capture_state_label.text() == "采集已停止"
    assert window.conn_status_indicator.text() == "● 主相机待机"
    assert window.cam_fps_labels[0].text() == "-- FPS"
    assert window.live_monitor_label.text() == "巡检: 待机"
    assert window.sport_profile_combo.isEnabled()
    window.close()


def test_external_clip_import_uses_active_passage_race_id(tmp_path, monkeypatch):
    sidecar = tmp_path / "clips.json"
    sidecar.write_text("{}", encoding="utf-8")
    timeline_store = object()
    begin_calls = []

    class _Window:
        _import_external_clips = MainWindow._import_external_clips

        def __init__(self):
            self._race_ready = True
            self.recording_manager = None
            self.video_timeline_store = timeline_store
            self.passage_event_store = SimpleNamespace(
                events=lambda: (SimpleNamespace(race_id="race-1"),)
            )
            self.passage_review_dialog = None
            self.output_dir = tmp_path

        def _begin_external_clip_import(self, path, race_id):
            begin_calls.append((path, race_id))

    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(sidecar), "JSON sidecar (*.json)"),
    )
    _Window()._import_external_clips()

    assert begin_calls == [(str(sidecar), "race-1")]


def test_external_clip_probe_runs_outside_the_ui_thread(qapp, monkeypatch):
    ui_thread_id = threading.get_ident()
    worker_thread_ids = []
    completed = []

    def _load(sidecar_path, **kwargs):
        worker_thread_ids.append(threading.get_ident())
        kwargs["progress_callback"](1, 1, sidecar_path)
        return ("verified-clip",)

    monkeypatch.setattr(main_window, "load_external_clip_sidecar", _load)
    thread = ExternalClipProbeThread("clips.json", "race-1")
    thread.completed.connect(completed.append)

    thread.start()
    assert thread.wait(2_000)
    qapp.processEvents()

    assert worker_thread_ids and worker_thread_ids[0] != ui_thread_id
    assert completed == [("verified-clip",)]
    thread.deleteLater()


def test_external_clip_result_is_discarded_after_race_switch(tmp_path, monkeypatch):
    import_calls = []
    warnings = []
    old_timeline = object()
    old_passage = object()

    class _Window:
        _on_external_clip_import_verified = MainWindow._on_external_clip_import_verified

        def __init__(self):
            self._race_ready = True
            self.output_dir = tmp_path / "race-2"
            self.video_timeline_store = object()
            self.passage_event_store = object()
            self._external_clip_import_context = {
                "race_dir": (tmp_path / "race-1").absolute(),
                "race_id": "race-1",
                "timeline_store": old_timeline,
                "passage_store": old_passage,
            }

    monkeypatch.setattr(
        main_window,
        "import_verified_external_clips",
        lambda *args, **kwargs: import_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        main_window.QMessageBox,
        "warning",
        lambda *args: warnings.append(args[1:]),
    )

    _Window()._on_external_clip_import_verified(())

    assert import_calls == []
    assert any("赛事已切换" in str(item) for item in warnings)


def test_external_clip_result_is_discarded_after_late_cancel(tmp_path, monkeypatch):
    import_calls = []
    messages = []

    class _StatusBar:
        def showMessage(self, message, timeout=0):
            messages.append((message, timeout))

    class _Window:
        _on_external_clip_import_verified = MainWindow._on_external_clip_import_verified
        _on_external_clip_import_cancelled = MainWindow._on_external_clip_import_cancelled

        def __init__(self):
            self._external_clip_import_context = {"race_id": "race-1"}
            self._external_clip_import_thread = SimpleNamespace(
                isInterruptionRequested=lambda: True
            )
            self._status_bar = _StatusBar()

        def statusBar(self):
            return self._status_bar

    monkeypatch.setattr(
        main_window,
        "import_verified_external_clips",
        lambda *args, **kwargs: import_calls.append((args, kwargs)),
    )

    _Window()._on_external_clip_import_verified((object(),))

    assert import_calls == []
    assert any("已取消" in str(item) for item in messages)


def test_external_clip_result_is_applied_to_unchanged_race(tmp_path, monkeypatch):
    import_calls = []
    messages = []
    timeline_store = object()
    passage_store = SimpleNamespace(
        events=lambda: (SimpleNamespace(race_id="race-1"),)
    )

    class _StatusBar:
        def showMessage(self, message, timeout=0):
            messages.append((message, timeout))

    class _Window:
        _on_external_clip_import_verified = MainWindow._on_external_clip_import_verified

        def __init__(self):
            self._race_ready = True
            self.output_dir = tmp_path
            self.video_timeline_store = timeline_store
            self.passage_event_store = passage_store
            self.passage_review_dialog = None
            self._external_clip_import_context = {
                "race_dir": tmp_path.absolute(),
                "race_id": "race-1",
                "timeline_store": timeline_store,
                "passage_store": passage_store,
            }
            self._status_bar = _StatusBar()

        def statusBar(self):
            return self._status_bar

    monkeypatch.setattr(
        main_window,
        "import_verified_external_clips",
        lambda store, clips, *, expected_race_id: (
            import_calls.append((store, clips, expected_race_id))
            or SimpleNamespace(
                created_count=2,
                repaired_count=1,
                duplicate_count=3,
            )
        ),
    )
    monkeypatch.setattr(
        main_window.QMessageBox,
        "information",
        lambda *args: messages.append(args[1:]),
    )

    clips = (object(),)
    _Window()._on_external_clip_import_verified(clips)

    assert import_calls == [(timeline_store, clips, "race-1")]
    assert any("已导入 2 个片段" in str(item) for item in messages)


def test_vlm_config_dialog_restores_openai_option(qapp):
    dialog = VLMConfigDialog({"vlm_config": {"model_type": "qwen"}})

    assert dialog.model_type.findText("openai") >= 0
    dialog.model_type.setCurrentText("openai")
    qapp.processEvents()

    assert dialog.endpoint_label.text() == "模型名称:"
    assert "gpt-4.1-mini" in dialog.endpoint_id.placeholderText()
    assert not dialog.base_url_label.isHidden()
    dialog.base_url.setText("https://proxy.example/v1")
    assert dialog.get_result()["base_url"] == "https://proxy.example/v1"
    assert dialog.get_result()["model_type"] == "openai"
    dialog.close()


def test_startup_race_dialog_can_browse_existing_race(qapp, tmp_path, monkeypatch):
    race_root = tmp_path / "RaceData"
    race_dir = race_root / "20260810_旧赛事"
    race_dir.mkdir(parents=True)
    (race_dir / "timing.db").touch()
    monkeypatch.setattr(
        QFileDialog,
        "getExistingDirectory",
        lambda *args, **kwargs: str(race_dir),
    )

    dialog = NewRaceDialog(base_path=str(race_root), allow_existing=True)
    dialog._open_existing()

    assert dialog.result() == QDialog.Accepted
    assert dialog.selected_race_dir == race_dir.absolute()
    assert dialog.result_name == race_dir.name
    assert dialog.result_path == str(race_root)


def test_race_config_updates_sport_profile_and_combo(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(MainWindow, "_prompt_race_selection", lambda self: None)
    monkeypatch.setattr(MainWindow, "_init_ocr_runtime", lambda self: None)
    race_dir = tmp_path / "race"
    race_dir.mkdir()
    (race_dir / "config.json").write_text(
        json.dumps(
            {
                "sport_profile": "triathlon_run",
                "gate_guard_enabled": False,
                "model_path": "D:/old-computer/best.pt",
            }
        ),
        encoding="utf-8",
    )
    current_model = str(tmp_path / "unused.pt")
    window = MainWindow({
        "source": "unused.mp4",
        "model_path": current_model,
        "output_dir": str(tmp_path),
        "yolo_only_mode": True,
        "live_monitor_enabled": False,
    })

    window._apply_race_config(race_dir)

    assert window.sport_profile == "running"
    assert window.config["sport_profile"] == "running"
    assert window.config["gate_guard_enabled"] is True
    assert window.sport_profile_combo.currentData() == "running"
    assert window._gate_guard_enabled is True
    assert window.model_path == current_model
    assert window.config["model_path"] == current_model
    window.close()


def test_explicit_runtime_source_and_profile_survive_race_config(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(MainWindow, "_prompt_race_selection", lambda self: None)
    monkeypatch.setattr(MainWindow, "_init_ocr_runtime", lambda self: None)
    race_dir = tmp_path / "race"
    race_dir.mkdir()
    (race_dir / "config.json").write_text(
        json.dumps(
            {
                "source": "old-video.mp4",
                "sources": ["old-video.mp4"],
                "sport_profile": "cycling",
                "model_path": "D:/old-computer/yolov8s.pt",
            }
        ),
        encoding="utf-8",
    )
    runtime_source = "rtsp://camera.example/live"
    current_model = str(tmp_path / "yolo11-best.pt")
    window = MainWindow(
        {
            "source": runtime_source,
            "sources": [runtime_source],
            "model_path": current_model,
            "sport_profile": "speed_skating",
            "output_dir": str(tmp_path),
            "yolo_only_mode": True,
            "live_monitor_enabled": False,
        },
        runtime_source_override=runtime_source,
        runtime_sport_profile_override="speed_skating",
    )

    window._apply_race_config(race_dir)

    assert window.sources == [runtime_source]
    assert window.config["source"] == runtime_source
    assert window.config["sources"] == [runtime_source]
    assert window.sport_profile == "speed_skating"
    assert window.config["sport_profile"] == "speed_skating"
    assert window.model_path == current_model
    assert window.config["model_path"] == current_model
    window.close()


def test_race_config_refreshes_passage_video_timing_fields(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(MainWindow, "_prompt_race_selection", lambda self: None)
    monkeypatch.setattr(MainWindow, "_init_ocr_runtime", lambda self: None)
    configured_race = tmp_path / "configured"
    configured_race.mkdir()
    (configured_race / "config.json").write_text(
        json.dumps(
            {
                "passage_clock_offset_ms": 850,
                "passage_video_preroll_ms": 4_500,
                "video_timeline_timing_error_ms": 1_250,
            }
        ),
        encoding="utf-8",
    )
    default_race = tmp_path / "default"
    default_race.mkdir()
    (default_race / "config.json").write_text("{}", encoding="utf-8")
    window = MainWindow(
        {
            "source": "unused.mp4",
            "output_dir": str(tmp_path),
            "yolo_only_mode": True,
            "live_monitor_enabled": False,
        }
    )

    window._apply_race_config(configured_race)
    assert window._passage_clock_offset_ms == 850
    assert window._passage_video_preroll_ms == 4_500
    assert window._video_timeline_timing_error_ms == 1_250

    window._apply_race_config(default_race)
    assert window._passage_clock_offset_ms == 0
    assert window._passage_video_preroll_ms == 3_000
    assert window._video_timeline_timing_error_ms == 2_000
    window.close()


def test_recording_timeline_warning_remains_visible_after_poll(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(MainWindow, "_prompt_race_selection", lambda self: None)
    monkeypatch.setattr(MainWindow, "_init_ocr_runtime", lambda self: None)
    window = MainWindow(
        {
            "source": "unused.mp4",
            "output_dir": str(tmp_path),
            "yolo_only_mode": True,
            "live_monitor_enabled": False,
        }
    )

    class _Manager:
        is_recording = True
        elapsed_seconds = 4.0
        total_size_bytes = 1024

        def __init__(self):
            self.warning = "机位 1 录像已开始，但时间线写入失败: disk error"

        def check_error(self):
            return None

        def consume_recovery_notice(self):
            return None

        def consume_timeline_warning(self):
            warning = self.warning
            self.warning = None
            return warning

    manager = _Manager()
    window.recording_manager = manager

    window._poll_recording_status()
    assert window.recording_status_label.text() == "录像: 时间线异常"
    assert "disk error" in window.recording_status_label.toolTip()

    window._poll_recording_status()
    assert window.recording_status_label.text() == "录像: 时间线异常"
    window.recording_manager = None
    window.close()


def test_passage_video_can_open_while_capture_is_running(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(MainWindow, "_prompt_race_selection", lambda self: None)
    monkeypatch.setattr(MainWindow, "_init_ocr_runtime", lambda self: None)
    opened = []

    class _PlaybackDialog:
        def __init__(self, video_path, parent, **kwargs):
            opened.append((video_path, parent, kwargs))

        def exec_(self):
            return 0

    monkeypatch.setattr(main_window, "VideoPlaybackDialog", _PlaybackDialog)
    window = MainWindow(
        {
            "source": "unused.mp4",
            "output_dir": str(tmp_path),
            "yolo_only_mode": True,
            "live_monitor_enabled": False,
        }
    )
    video_path = tmp_path / "camera_01.mkv"
    video_path.write_bytes(b"video")
    event = SimpleNamespace(bib="23", chip_id="chip-23")
    location = SimpleNamespace(
        status="located",
        video_path=video_path,
        playback_position_ms=2_500,
        passage_position_ms=5_500,
        timing_error_ms=1_000,
        segment=SimpleNamespace(camera_index=1),
    )
    window._running = True

    window._open_passage_location(event, location)

    assert opened[0][0] == video_path
    assert opened[0][2]["initial_position_ms"] == 2_500
    assert opened[0][2]["target_position_ms"] == 5_500
    assert "机位 1" in opened[0][2]["context_text"]
    assert opened[0][2]["autoplay"] is False
    window._running = False
    window.close()
