import realtime.main_window as main_window
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
