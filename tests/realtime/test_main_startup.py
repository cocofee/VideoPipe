import realtime.main_window as main_window
import realtime.detector as detector_module
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
