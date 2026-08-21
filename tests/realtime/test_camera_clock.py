import os
from datetime import datetime
from types import SimpleNamespace
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QApplication, QDialog, QPushButton

import realtime.main_window as main_window
from realtime.camera_clock import (
    camera_clock_check_required,
    check_hikvision_camera_clock,
    parse_hikvision_time,
)
from realtime.main_window import (
    CameraClockPreflightThread,
    CameraConfigDialog,
    ConnectionTester,
    MainWindow,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _time_xml(value: str, mode: str = "NTP") -> bytes:
    return (
        '<Time xmlns="http://www.hikvision.com/ver20/XMLSchema">'
        f"<timeMode>{mode}</timeMode><localTime>{value}</localTime>"
        "</Time>"
    ).encode("utf-8")


def test_camera_clock_check_uses_read_only_isapi_and_decodes_credentials():
    camera_time = datetime.fromisoformat("2026-08-21T12:00:00+08:00")
    calls = []

    def request_get(url, *, auth, timeout):
        calls.append((url, auth.username, auth.password, timeout))
        return SimpleNamespace(
            status_code=200,
            content=_time_xml("2026-08-21T12:00:00+08:00"),
        )

    result = check_hikvision_camera_clock(
        "rtsp://admin:p%40ss@192.0.2.5:554/Streaming/Channels/101",
        request_get=request_get,
        clock=lambda: camera_time.timestamp(),
    )

    assert result.ok
    assert result.skew_ms == 0
    assert calls == [("http://192.0.2.5/ISAPI/System/time", "admin", "p@ss", 3.0)]


def test_camera_clock_check_uses_configured_https_management_port():
    camera_time = datetime.fromisoformat("2026-08-21T12:00:00+08:00")
    calls = []

    result = check_hikvision_camera_clock(
        "rtsp://admin:secret@192.0.2.5:554/Streaming/Channels/101",
        management_url="https://192.0.2.5:8443",
        request_get=lambda url, **kwargs: (
            calls.append((url, kwargs["auth"].username, kwargs["auth"].password))
            or SimpleNamespace(
                status_code=200,
                content=_time_xml("2026-08-21T12:00:00+08:00"),
            )
        ),
        clock=lambda: camera_time.timestamp(),
    )

    assert result.ok
    assert calls == [("https://192.0.2.5:8443/ISAPI/System/time", "admin", "secret")]


def test_camera_clock_rejects_management_host_mismatch_before_request():
    result = check_hikvision_camera_clock(
        "rtsp://admin:secret@192.0.2.5:554/Streaming/Channels/101",
        management_url="https://192.0.2.6:8443",
        request_get=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("mismatched management host must not receive credentials")
        ),
    )

    assert result.ok is False
    assert "管理地址主机必须与 RTSP 相机主机一致" in result.message
    assert "secret" not in result.message


def test_camera_clock_requirement_includes_advanced_credentialed_rtsp():
    assert camera_clock_check_required(
        "rtsp://admin:secret@192.0.2.5:8554/custom/live"
    )
    assert not camera_clock_check_required("rtsp://localhost:8554/test")
    assert not camera_clock_check_required("rtsp://[")


def test_camera_clock_check_rejects_1970_osd_time_without_leaking_password():
    computer_time = datetime.fromisoformat("2026-08-21T12:00:00+08:00")
    result = check_hikvision_camera_clock(
        "rtsp://admin:secret@192.0.2.5:554/Streaming/Channels/101",
        request_get=lambda *args, **kwargs: SimpleNamespace(
            status_code=200,
            content=_time_xml("1970-01-02T02:20:00+08:00", mode="manual"),
        ),
        clock=lambda: computer_time.timestamp(),
    )

    assert result.ok is False
    assert "1970-01-02" in result.message
    assert "请先校时" in result.message
    assert "secret" not in result.message


def test_hikvision_time_requires_timezone_offset():
    with pytest.raises(ValueError, match="timezone offset"):
        parse_hikvision_time(_time_xml("2026-08-21T12:00:00"))


def test_connection_tester_fails_when_video_works_but_camera_clock_is_wrong(
    qapp,
    monkeypatch,
):
    class _Capture:
        def isOpened(self):
            return True

        def set(self, *_args):
            pass

        def read(self):
            return True, object()

        def release(self):
            pass

    monkeypatch.setattr(main_window.cv2, "VideoCapture", lambda *_args: _Capture())
    monkeypatch.setattr(
        main_window,
        "check_hikvision_camera_clock",
        lambda _source, **_kwargs: SimpleNamespace(
            ok=False,
            message="相机时间异常：1970-01-02",
        ),
    )
    results = []
    tester = ConnectionTester(
        "rtsp://admin:secret@192.0.2.5/Streaming/Channels/101",
        verify_hikvision_clock=True,
    )
    tester.finished.connect(lambda success, message: results.append((success, message)))

    tester.run()

    assert results == [(False, "画面读取正常，但相机时间核验失败：相机时间异常：1970-01-02")]


def test_standard_hikvision_configuration_requires_successful_clock_check(
    qapp,
    monkeypatch,
):
    warnings = []
    monkeypatch.setattr(
        main_window.QMessageBox,
        "warning",
        lambda _parent, title, message: warnings.append((title, message)),
    )
    dialog = CameraConfigDialog("")
    dialog.radio_rtsp.setChecked(True)
    dialog.hik_ip.setText("192.0.2.5")
    dialog.hik_user.setText("admin")
    dialog.hik_pwd.setText("secret")
    assert dialog.status_label.wordWrap()

    dialog._on_save()

    assert dialog.result() == QDialog.Rejected
    assert warnings == [("相机时间未核验", "请先点击“测试连接”，确认画面和相机时间均正常。")]

    dialog._clock_check_passed = True
    dialog._on_save()

    assert dialog.result() == QDialog.Accepted
    assert dialog.result_source.startswith("rtsp://admin:secret@192.0.2.5:554/")
    assert dialog.result_management_url == "http://192.0.2.5:80"
    dialog.close()


def test_camera_dialog_decodes_credentials_and_preserves_advanced_url(qapp):
    source = "rtsp://admin:p%40ss@192.0.2.5:8554/custom/live"
    dialog = CameraConfigDialog(
        source,
        current_management_url="https://192.0.2.5:8443",
    )

    assert dialog.advance_rtsp_check.isChecked()
    assert dialog.hik_user.text() == "admin"
    assert dialog.hik_pwd.text() == "p@ss"
    assert dialog.management_scheme_combo.currentData() == "https"
    assert dialog.management_port_spin.value() == 8443
    assert dialog._requires_camera_clock_check()

    dialog._clock_check_passed = True
    dialog._on_save()

    assert dialog.result() == QDialog.Accepted
    assert dialog.result_source == source
    assert dialog.result_management_url == "https://192.0.2.5:8443"
    dialog.close()


def test_camera_dialog_does_not_double_encode_standard_credentials(qapp):
    dialog = CameraConfigDialog(
        "rtsp://admin:p%40ss@192.0.2.5:554/Streaming/Channels/101"
    )

    assert not dialog.advance_rtsp_check.isChecked()
    assert dialog.hik_pwd.text() == "p@ss"
    dialog._clock_check_passed = True
    dialog._on_save()

    assert "p%40ss" in dialog.result_source
    assert "%2540" not in dialog.result_source
    dialog.close()


class _StatusBar:
    def __init__(self):
        self.message = ""

    def showMessage(self, message):
        self.message = message


class _StartHarness(QObject):
    _start = MainWindow._start
    _camera_clock_targets = MainWindow._camera_clock_targets
    _on_camera_clock_preflight_finished = (
        MainWindow._on_camera_clock_preflight_finished
    )

    def __init__(self):
        super().__init__()
        self._race_ready = True
        self._running = False
        self.sources = [
            "rtsp://admin:secret@192.0.2.5:554/Streaming/Channels/101"
        ]
        self.camera_clock_management_urls = ["https://192.0.2.5:8443"]
        self._camera_clock_preflight_thread = None
        self._camera_clock_verified_signature = None
        self.start_btn = QPushButton()
        self._status_bar = _StatusBar()
        self.capture_starts = 0

    def statusBar(self):
        return self._status_bar

    def _prompt_race_selection(self):
        raise AssertionError("race is already ready")

    def _start_capture(self):
        self.capture_starts += 1


def _wait_for_preflight(qapp, harness):
    deadline = time.time() + 2.0
    while harness._camera_clock_preflight_thread is not None and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()


def test_start_blocks_capture_when_camera_clock_preflight_fails(
    qapp,
    monkeypatch,
):
    failures = []
    monkeypatch.setattr(
        main_window,
        "check_hikvision_camera_clock",
        lambda _source, **_kwargs: SimpleNamespace(
            ok=False,
            message="相机时间异常：1970-01-02",
        ),
    )
    monkeypatch.setattr(
        main_window.QMessageBox,
        "critical",
        lambda _parent, title, message: failures.append((title, message)),
    )
    harness = _StartHarness()

    harness._start()
    _wait_for_preflight(qapp, harness)

    assert harness.capture_starts == 0
    assert harness._camera_clock_verified_signature is None
    assert failures == [("无法开始采集", "机位 1：相机时间异常：1970-01-02")]


def test_start_continues_only_after_all_camera_clocks_pass(qapp, monkeypatch):
    calls = []
    monkeypatch.setattr(
        main_window,
        "check_hikvision_camera_clock",
        lambda source, **kwargs: (
            calls.append((source, kwargs["management_url"]))
            or SimpleNamespace(ok=True, message="相机时间已核验")
        ),
    )
    harness = _StartHarness()

    harness._start()
    _wait_for_preflight(qapp, harness)

    assert harness.capture_starts == 1
    assert harness._camera_clock_verified_signature == harness._camera_clock_targets()
    assert calls == [
        (
            "rtsp://admin:secret@192.0.2.5:554/Streaming/Channels/101",
            "https://192.0.2.5:8443",
        )
    ]
