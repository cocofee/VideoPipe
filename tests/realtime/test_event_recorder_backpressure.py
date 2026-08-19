import queue
import threading
import time
from types import SimpleNamespace

from realtime.event_recorder import EventRecorder


def _recorder_harness():
    recorder = EventRecorder.__new__(EventRecorder)
    recorder._queue = queue.Queue(maxsize=3)
    recorder._snapshot_queue = queue.Queue(maxsize=3)
    recorder._running = True
    recorder._accepting_events = True
    recorder._lifecycle_lock = threading.RLock()
    recorder._stop_lock = threading.Lock()
    recorder._restart_blocked = False
    recorder._saved_count = 0
    recorder._dropped_critical_count = 0
    recorder._thread = None
    recorder._snapshot_thread = None
    recorder.camera_configs = {}
    recorder.source_count = 1
    return recorder


def test_stop_drains_pending_events_before_worker_exit():
    recorder = _recorder_harness()
    saved = []

    def save_event(event):
        time.sleep(0.01)
        saved.append(event.track_id)

    recorder._save_event = save_event
    recorder._queue.put(SimpleNamespace(track_id=1))
    recorder._queue.put(SimpleNamespace(track_id=2))
    recorder._thread = threading.Thread(target=recorder._process_loop, daemon=True)
    recorder._snapshot_thread = threading.Thread(target=recorder._snapshot_loop, daemon=True)
    recorder._thread.start()
    recorder._snapshot_thread.start()

    recorder.stop()

    assert saved == [1, 2]
    assert recorder._queue.empty()
    assert recorder._queue.unfinished_tasks == 0
    assert recorder._thread is None
    assert recorder._snapshot_thread is None
    assert recorder._running is False
    assert recorder._accepting_events is False


def test_save_failure_still_completes_queue_task_and_allows_stop():
    recorder = _recorder_harness()
    recorder._save_event = lambda _event: (_ for _ in ()).throw(RuntimeError("write failed"))
    recorder._queue.put(SimpleNamespace(track_id=3))
    recorder._thread = threading.Thread(target=recorder._process_loop, daemon=True)
    recorder._snapshot_thread = threading.Thread(target=recorder._snapshot_loop, daemon=True)
    recorder._thread.start()
    recorder._snapshot_thread.start()

    recorder.stop()

    assert recorder._queue.unfinished_tasks == 0
    assert recorder._thread is None


def test_full_queue_compaction_preserves_unfinished_task_count():
    recorder = _recorder_harness()
    recorder._queue.put(SimpleNamespace(track_id=1, is_update_only=True))
    recorder._queue.put(SimpleNamespace(track_id=2, is_info_update=True))
    recorder._queue.put(SimpleNamespace(track_id=3))

    recorder._enqueue_critical_event(SimpleNamespace(track_id=4, bib_number=None))

    assert recorder._queue.qsize() == 2
    assert recorder._queue.unfinished_tasks == 2
    assert [item.track_id for item in list(recorder._queue.queue)] == [3, 4]


def test_stop_waits_for_an_inflight_record_submission():
    recorder = _recorder_harness()
    recorder._running = True
    entered = threading.Event()
    release = threading.Event()
    stop_finished = threading.Event()

    def enqueue(_event):
        entered.set()
        assert release.wait(timeout=1.0)

    recorder._enqueue_critical_event = enqueue
    producer = threading.Thread(
        target=recorder.record,
        args=(SimpleNamespace(track_id=5, bib_number=None),),
    )
    stopper = threading.Thread(
        target=lambda: (recorder.stop(), stop_finished.set()),
    )

    producer.start()
    assert entered.wait(timeout=1.0)
    stopper.start()
    assert stop_finished.wait(timeout=0.05) is False

    release.set()
    producer.join(timeout=1.0)
    stopper.join(timeout=1.0)

    assert stop_finished.is_set()
    assert recorder._accepting_events is False


def test_stop_timeout_blocks_restart_until_thread_exits(monkeypatch):
    recorder = _recorder_harness()

    class _Thread:
        def __init__(self):
            self.alive = True

        def join(self, timeout=None):
            pass

        def is_alive(self):
            return self.alive

    worker = _Thread()
    recorder._thread = worker
    recorder._snapshot_thread = None
    monkeypatch.setattr(recorder, "_drain_queue", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(recorder, "_put_stop_sentinel", lambda *_args, **_kwargs: None)

    assert recorder.stop() is False
    assert recorder.restart_blocked is True

    try:
        recorder.start()
    except RuntimeError as exc:
        assert "禁止重新启动" in str(exc)
    else:
        raise AssertionError("restart should be blocked while the old worker is alive")

    worker.alive = False
    assert recorder.stop() is True
    assert recorder.restart_blocked is False
