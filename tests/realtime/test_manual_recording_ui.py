from pathlib import Path

import realtime.main_window as main_window
from realtime.main_window import MainWindow


class _Button:
    def __init__(self):
        self.text = ""
        self.object_name = ""
        self.enabled = False

    def setText(self, value):
        self.text = value

    def setObjectName(self, value):
        self.object_name = value

    def setEnabled(self, value):
        self.enabled = bool(value)

    def style(self):
        return self

    def setStyle(self, _style):
        pass


class _Label:
    def __init__(self):
        self.text = ""
        self.style = ""

    def setText(self, value):
        self.text = value

    def setStyleSheet(self, value):
        self.style = value


class _Timer:
    def __init__(self):
        self.active = False

    def start(self):
        self.active = True

    def stop(self):
        self.active = False


class _StatusBar:
    def __init__(self):
        self.message = ""

    def showMessage(self, value):
        self.message = value


class _Manager:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.is_recording = False
        self.elapsed_seconds = 65
        self.total_size_bytes = 12 * 1024 * 1024
        self.stopped = False

    def start(self):
        self.is_recording = True
        return (self.output_dir / "camera_01_20260808_120000.mkv",)

    def stop(self):
        self.is_recording = False
        self.stopped = True
        return (self.output_dir / "camera_01_20260808_120000.mkv",)

    def check_error(self):
        return None


class _Harness:
    _format_recording_duration = staticmethod(MainWindow._format_recording_duration)
    _refresh_recording_ui = MainWindow._refresh_recording_ui
    _start_manual_recording = MainWindow._start_manual_recording
    _stop_manual_recording = MainWindow._stop_manual_recording

    def __init__(self, tmp_path):
        self._race_ready = True
        self._running = True
        self.sources = ["rtsp://camera/live"]
        self.output_dir = tmp_path
        self.recording_manager = None
        self.record_btn = _Button()
        self.recording_status_label = _Label()
        self._recording_poll_timer = _Timer()
        self._status_bar = _StatusBar()

    def statusBar(self):
        return self._status_bar


def test_manual_recording_ui_starts_and_stops_manager(monkeypatch, tmp_path):
    created = []

    def _manager_factory(sources, output_dir):
        assert sources == ["rtsp://camera/live"]
        manager = _Manager(output_dir)
        created.append(manager)
        return manager

    monkeypatch.setattr(main_window, "ManualRecordingManager", _manager_factory)
    window = _Harness(tmp_path)

    window._start_manual_recording()

    assert window.recording_manager is created[0]
    assert created[0].output_dir == tmp_path / "videos"
    assert window._recording_poll_timer.active is True
    assert window.record_btn.text == "停止录像"
    assert window.record_btn.object_name == "recording_btn"
    assert window.recording_status_label.text == "录像: 00:01:05 | 12.0 MB"

    paths = window._stop_manual_recording()

    assert created[0].stopped is True
    assert window.recording_manager is None
    assert window._recording_poll_timer.active is False
    assert window.record_btn.text == "开始录像"
    assert paths == (tmp_path / "videos" / "camera_01_20260808_120000.mkv",)


def test_manual_recording_requires_running_detection(monkeypatch, tmp_path):
    warnings = []
    monkeypatch.setattr(
        main_window.QMessageBox,
        "warning",
        lambda _parent, title, message: warnings.append((title, message)),
    )
    window = _Harness(tmp_path)
    window._running = False

    window._start_manual_recording()

    assert window.recording_manager is None
    assert warnings == [("提示", "请先启动AI检测，再开始录像。")]
