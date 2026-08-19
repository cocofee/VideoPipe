import realtime.main_window as main_window
import realtime.detector as detector_module
import realtime.vlm_utils as vlm_utils
from realtime.main_window import MainWindow


class _StartupHarness:
    start_when_race_ready = MainWindow.start_when_race_ready

    def __init__(self, race_ready):
        self._race_ready = race_ready
        self._running = False
        self.starts = 0

    def _start(self):
        self.starts += 1

    def isVisible(self):
        return True


def test_auto_start_runs_once_after_race_is_ready():
    harness = _StartupHarness(race_ready=True)

    harness.start_when_race_ready()
    harness._running = True
    harness.start_when_race_ready()

    assert harness.starts == 1


def test_auto_start_waits_without_opening_race_dialog_again(monkeypatch):
    scheduled = []

    class _Timer:
        @staticmethod
        def singleShot(delay_ms, callback):
            scheduled.append((delay_ms, callback))

    monkeypatch.setattr(main_window, "QTimer", _Timer)
    harness = _StartupHarness(race_ready=False)

    harness.start_when_race_ready()

    assert harness.starts == 0
    assert scheduled == [(100, harness.start_when_race_ready)]


def test_releasing_current_race_stops_workers_and_closes_database():
    class _Stoppable:
        def __init__(self):
            self.stopped = 0

        def stop(self):
            self.stopped += 1

    class _Database:
        def __init__(self):
            self.closed = 0

        def close(self):
            self.closed += 1

    class _Timer:
        def __init__(self):
            self.active = True
            self.stopped = 0

        def isActive(self):
            return self.active

        def stop(self):
            self.active = False
            self.stopped += 1

    class _Harness:
        _release_current_race = MainWindow._release_current_race

        def __init__(self):
            self.recording_manager = None
            self.preview_threads = {0: _Stoppable()}
            self.video_threads = {0: _Stoppable()}
            self.readers = {0: _Stoppable()}
            self.detectors = {0: object()}
            self.recorder = _Stoppable()
            self.ocr_manager = _Stoppable()
            self.database = _Database()
            self._ocr_poll_timer = _Timer()
            self.shared_model = object()
            self.shared_athlete_validator = object()
            self._athlete_validator_checked = True
            self.shared_vlm = object()
            self.field_issue_log = object()
            self._initialized = True
            self._race_ready = True
            self._session_event_count = 5
            self._latest_frame_observations = {0: ([], [])}
            self._live_monitor_samples = {0: object()}
            self._live_monitor_last_warn_ts = {0: 1.0}
            self._yolo_only_mode = False
            self._ocr_runtime_state = "ready"

        def _refresh_ocr_runtime_ui(self):
            pass

        def _stop_manual_recording(self, *, show_message):
            raise AssertionError("manual recorder should not be stopped when absent")

    harness = _Harness()
    preview = harness.preview_threads[0]
    video = harness.video_threads[0]
    reader = harness.readers[0]
    recorder = harness.recorder
    ocr_manager = harness.ocr_manager
    database = harness.database
    timer = harness._ocr_poll_timer

    harness._release_current_race()

    assert preview.stopped == 1
    assert video.stopped == 1
    assert reader.stopped == 1
    assert recorder.stopped == 1
    assert ocr_manager.stopped == 1
    assert database.closed == 1
    assert timer.stopped == 1
    assert harness.preview_threads == {}
    assert harness.video_threads == {}
    assert harness.readers == {}
    assert harness.detectors == {}
    assert harness.recorder is None
    assert harness.ocr_manager is None
    assert harness.database is None
    assert harness.shared_model is None
    assert harness._race_ready is False
    assert harness._initialized is False
    assert harness._ocr_runtime_state == "idle"


def test_releasing_race_keeps_database_open_when_recorder_is_still_writing():
    class _Recorder:
        def stop(self):
            return False

    class _Database:
        def __init__(self):
            self.closed = 0

        def close(self):
            self.closed += 1

    class _Timer:
        def isActive(self):
            return False

    class _Harness:
        _release_current_race = MainWindow._release_current_race

        def __init__(self):
            self.recording_manager = None
            self.preview_threads = {}
            self.video_threads = {}
            self.readers = {}
            self.detectors = {}
            self.recorder = _Recorder()
            self.ocr_manager = None
            self.database = _Database()
            self._ocr_poll_timer = _Timer()

    harness = _Harness()

    try:
        harness._release_current_race()
    except RuntimeError as exc:
        assert "仍在写入" in str(exc)
    else:
        raise AssertionError("race release should fail while the recorder is still writing")

    assert harness.database.closed == 0
    assert harness.recorder is not None


def test_vlm_settings_keep_cloud_ocr_out_of_detector_frame_path(monkeypatch):
    class _Vlm:
        def __init__(self, api_key, model):
            self.api_key = api_key
            self.model = model

    class _Detector:
        def __init__(self):
            self.disabled = 0
            self.enabled = 0

        def disable_vlm(self):
            self.disabled += 1

        def enable_vlm(self, **kwargs):
            self.enabled += 1

    class _OcrManager:
        vlm = None
        vlm_mode = "fallback"
        vlm_max_calls_per_minute = 0

    class _Harness:
        _apply_vlm_settings = MainWindow._apply_vlm_settings

        def __init__(self):
            self._yolo_only_mode = False
            self.config = {
                "vlm_config": {
                    "enabled": True,
                    "model_type": "qwen",
                    "api_key": "test-key",
                    "endpoint_id": "qwen3.5-ocr",
                    "max_calls_per_minute": 24,
                    "ocr_mode": "fallback",
                }
            }
            self.detectors = {0: _Detector()}
            self.ocr_manager = _OcrManager()
            self.shared_vlm = None

    monkeypatch.setattr(detector_module, "QwenVLMAssistant", _Vlm)
    harness = _Harness()

    harness._apply_vlm_settings()

    assert harness.ocr_manager.vlm is harness.shared_vlm
    assert harness.ocr_manager.vlm_max_calls_per_minute == 24
    assert harness.detectors[0].disabled == 1
    assert harness.detectors[0].enabled == 0


def test_openai_vlm_settings_use_openai_assistant(monkeypatch):
    class _Vlm:
        def __init__(self, api_key, model, base_url=None):
            self.api_key = api_key
            self.model = model
            self.base_url = base_url

    class _Detector:
        def __init__(self):
            self.disabled = 0

        def disable_vlm(self):
            self.disabled += 1

    class _OcrManager:
        vlm = None
        vlm_mode = "fallback"
        vlm_max_calls_per_minute = 0

    class _Harness:
        _apply_vlm_settings = MainWindow._apply_vlm_settings

        def __init__(self):
            self._yolo_only_mode = False
            self.config = {
                "vlm_config": {
                    "enabled": True,
                    "model_type": "openai",
                    "api_key": "fake-openai-key",
                    "endpoint_id": "gpt-4.1-mini",
                    "base_url": "https://proxy.example/v1",
                    "max_calls_per_minute": 12,
                    "ocr_mode": "fallback",
                }
            }
            self.detectors = {0: _Detector()}
            self.ocr_manager = _OcrManager()
            self.shared_vlm = None

    monkeypatch.setattr(vlm_utils, "OpenAIVLMAssistant", _Vlm)
    harness = _Harness()

    harness._apply_vlm_settings()

    assert isinstance(harness.shared_vlm, _Vlm)
    assert harness.shared_vlm.api_key == "fake-openai-key"
    assert harness.shared_vlm.model == "gpt-4.1-mini"
    assert harness.shared_vlm.base_url == "https://proxy.example/v1"
    assert harness.ocr_manager.vlm is harness.shared_vlm
    assert harness.ocr_manager.vlm_max_calls_per_minute == 12
    assert harness.detectors[0].disabled == 1


def test_pending_events_are_requeued_after_vlm_configuration():
    class _OcrManager:
        def __init__(self):
            self.calls = []

        def enqueue_live_event(self, **kwargs):
            self.calls.append(kwargs)
            return kwargs["event_id"] == 2

    class _Database:
        def get_all_events(self):
            return [
                {"event_id": 1, "bib_number": "12", "ocr_state": "DONE", "evidence_dir": "e1"},
                {"event_id": 2, "bib_number": "", "ocr_state": "PENDING", "evidence_dir": "e2"},
                {"event_id": 3, "bib_number": "UNKNOWN", "ocr_state": "FAIL", "evidence_dir": "e3", "manual_corrected": 1},
            ]

    class _Harness:
        _enqueue_pending_ocr_events = MainWindow._enqueue_pending_ocr_events

        def __init__(self):
            self._yolo_only_mode = False
            self.ocr_manager = _OcrManager()
            self.database = _Database()

    harness = _Harness()

    assert harness._enqueue_pending_ocr_events() == 1
    assert [call["event_id"] for call in harness.ocr_manager.calls] == [2]
