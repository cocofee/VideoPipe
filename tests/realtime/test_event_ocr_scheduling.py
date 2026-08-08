from realtime.main_window import MainWindow
from realtime.ocr_manager import OCRManager


class _Database:
    def get_event(self, event_id):
        return {
            "event_id": event_id,
            "bib_number": "UNKNOWN",
            "ocr_state": "PENDING",
            "evidence_dir": "evidence",
            "cross_time": 123.0,
        }


class _OcrManager:
    runtime_state = "ready"

    def __init__(self):
        self.calls = []
        self.polls = 0

    def enqueue_live_event(self, **kwargs):
        self.calls.append(kwargs)
        return True

    def poll_process_results(self):
        self.polls += 1
        return []


class _Harness:
    _on_event_saved_callback = MainWindow._on_event_saved_callback
    _poll_ocr_runtime = MainWindow._poll_ocr_runtime

    def __init__(self, yolo_only=False):
        self._yolo_only_mode = yolo_only
        self.database = _Database()
        self.ocr_manager = _OcrManager()
        self._ocr_runtime_state = "idle"

    def _refresh_ocr_runtime_ui(self):
        pass


def test_race_ui_has_no_batch_wave_patrol_or_inline_ocr_entry_points():
    assert not hasattr(MainWindow, "_start_batch_ocr")
    assert not hasattr(MainWindow, "_run_wave_ocr_scheduler")
    assert not hasattr(MainWindow, "_run_auto_ocr_patrol")
    assert not hasattr(MainWindow, "_on_realtime_ocr_changed")
    assert not hasattr(OCRManager, "start_batch")


def test_saved_event_is_submitted_once_to_async_ocr_manager():
    harness = _Harness()

    harness._on_event_saved_callback(7)

    assert harness.ocr_manager.calls == [
        {
            "event_id": 7,
            "evidence_dir": "evidence",
            "cross_time": 123.0,
        }
    ]


def test_yolo_only_saved_event_never_enters_ocr():
    harness = _Harness(yolo_only=True)

    harness._on_event_saved_callback(7)

    assert harness.ocr_manager.calls == []


def test_poll_updates_window_runtime_state_from_manager():
    harness = _Harness()

    harness._poll_ocr_runtime()

    assert harness.ocr_manager.polls == 1
    assert harness._ocr_runtime_state == "ready"
