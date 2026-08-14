import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtGui import QColor, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel

from realtime.event_list_widget import EventListWidget
from realtime.live_event_review import LiveEventReview
from realtime.main_window import MainWindow, VLMConfigDialog


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


def test_race_config_updates_sport_profile_and_combo(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(MainWindow, "_prompt_race_selection", lambda self: None)
    monkeypatch.setattr(MainWindow, "_init_ocr_runtime", lambda self: None)
    race_dir = tmp_path / "race"
    race_dir.mkdir()
    (race_dir / "config.json").write_text(
        '{"sport_profile": "triathlon_run", "gate_guard_enabled": false}',
        encoding="utf-8",
    )
    window = MainWindow({
        "source": "unused.mp4",
        "model_path": str(tmp_path / "unused.pt"),
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
    window.close()
