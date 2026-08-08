import json
import queue
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

import realtime.ocr_worker as ocr_worker
from realtime.ocr_worker import (
    OcrJob,
    OcrProcessController,
    OcrResult,
    collect_candidate_paths,
    parse_recognition_predictions,
    recognize_job,
)
from realtime.ocr_manager import OCRManager


def _write_image(path: Path, value: int = 0) -> None:
    image = np.full((24, 40, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_collect_candidate_paths_prefers_detected_bib_and_limits_work(tmp_path):
    event_dir = tmp_path / "000001"
    event_dir.mkdir()
    for index, name in enumerate(
        ("bib.jpg", "bib_candidate_01.jpg", "bib_candidate_02.jpg", "athlete.jpg")
    ):
        _write_image(event_dir / name, index)
    (event_dir / "meta.json").write_text(
        json.dumps(
            {
                "paths": {
                    "bib": "bib.jpg",
                    "bib_candidates": [
                        "bib_candidate_01.jpg",
                        "bib_candidate_02.jpg",
                    ],
                    "athlete": "athlete.jpg",
                },
                "bib_candidate_metadata": [
                    {
                        "path": "bib_candidate_01.jpg",
                        "frame_index": 10,
                        "athlete_bbox": [0, 0, 100, 200],
                        "bib_bbox": [20, 60, 60, 100],
                        "source": "detected",
                        "owner_validated": True,
                    },
                    {
                        "path": "bib_candidate_02.jpg",
                        "frame_index": 11,
                        "athlete_bbox": [0, 0, 100, 200],
                        "bib_bbox": [21, 61, 61, 101],
                        "source": "detected",
                        "owner_validated": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    candidates = collect_candidate_paths(event_dir, max_candidates=2)

    assert candidates == [event_dir / "bib_candidate_01.jpg", event_dir / "bib_candidate_02.jpg"]


def test_parse_recognition_predictions_accepts_paddle_v3_payloads():
    predictions = [
        {"res": {"rec_text": "125", "rec_score": 0.93}},
        {"res": {"rec_texts": ["", "126"], "rec_scores": [0.1, 0.71]}},
    ]

    assert parse_recognition_predictions(predictions) == [
        ("125", 0.93),
        ("126", 0.71),
    ]


def test_recognize_job_requires_two_independent_matching_candidates(tmp_path):
    first = tmp_path / "bib.jpg"
    second = tmp_path / "bib_candidate_01.jpg"
    _write_image(first, 1)
    _write_image(second, 2)

    class _Recognizer:
        def __init__(self):
            self.calls = 0

        def predict(self, image, batch_size=1):
            self.calls += 1
            return [{"res": {"rec_text": "125", "rec_score": 0.94}}]

    recognizer = _Recognizer()
    result = recognize_job(
        recognizer,
        OcrJob(event_id=7, event_dir=str(tmp_path), candidate_paths=(str(first), str(second))),
        high_confidence=0.85,
    )

    assert result.event_id == 7
    assert result.text == "125"
    assert result.confidence == 0.94
    assert result.status == "DONE"
    assert result.attempted_candidates == 2
    assert recognizer.calls == 2


def test_recognize_job_keeps_single_high_confidence_result_pending(tmp_path):
    candidate = tmp_path / "bib_candidate_01.jpg"
    _write_image(candidate, 1)

    class _Recognizer:
        def predict(self, image, batch_size=1):
            return [{"res": {"rec_text": "145", "rec_score": 0.98}}]

    result = recognize_job(
        _Recognizer(),
        OcrJob(event_id=9, event_dir=str(tmp_path), candidate_paths=(str(candidate),)),
        high_confidence=0.85,
    )

    assert result.text == "145"
    assert result.confidence == 0.98
    assert result.status == "PENDING"
    assert result.error == "INSUFFICIENT_MULTI_FRAME_EVIDENCE"


def test_recognize_job_ignores_duplicate_image_files_as_independent_evidence(tmp_path):
    first = tmp_path / "bib.jpg"
    duplicate = tmp_path / "bib_candidate_01.jpg"
    _write_image(first, 7)
    duplicate.write_bytes(first.read_bytes())

    class _Recognizer:
        def __init__(self):
            self.calls = 0

        def predict(self, image, batch_size=1):
            self.calls += 1
            return [{"res": {"rec_text": "206", "rec_score": 0.99}}]

    recognizer = _Recognizer()
    result = recognize_job(
        recognizer,
        OcrJob(event_id=4, event_dir=str(tmp_path), candidate_paths=(str(first), str(duplicate))),
        high_confidence=0.85,
    )

    assert result.status == "PENDING"
    assert result.error == "INSUFFICIENT_MULTI_FRAME_EVIDENCE"
    assert result.attempted_candidates == 1
    assert recognizer.calls == 1


def test_create_recognizer_uses_offline_modelscope_stub(monkeypatch, tmp_path):
    calls = {}

    class _TextRecognition:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    fake_paddleocr = types.ModuleType("paddleocr")
    fake_paddleocr.TextRecognition = _TextRecognition
    monkeypatch.setitem(sys.modules, "paddleocr", fake_paddleocr)
    monkeypatch.delitem(sys.modules, "modelscope", raising=False)
    monkeypatch.setattr(
        ocr_worker.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name="modelscope")),
    )
    model_dir = tmp_path / "en_PP-OCRv5_mobile_rec"
    model_dir.mkdir()
    monkeypatch.setattr(
        ocr_worker,
        "find_recognition_model_dir",
        lambda models_root=None: model_dir,
    )

    ocr_worker._create_recognizer(cpu_threads=1, models_root=str(tmp_path))

    assert calls["model_name"] == "en_PP-OCRv5_mobile_rec"
    assert calls["model_dir"] == str(model_dir)
    assert calls["cpu_threads"] == 1
    assert ocr_worker.os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] == "True"
    with pytest.raises(RuntimeError, match="bundle the OCR model"):
        sys.modules["modelscope"].snapshot_download("unused")


def test_controller_submit_returns_immediately_when_queue_is_full():
    job_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)
    controller = OcrProcessController(
        cpu_threads=1,
        job_queue=job_queue,
        result_queue=result_queue,
        process_factory=lambda *args, **kwargs: None,
    )
    first = OcrJob(event_id=1, event_dir="one", candidate_paths=("one.jpg",))
    second = OcrJob(event_id=2, event_dir="two", candidate_paths=("two.jpg",))

    assert controller.submit(first) is True
    assert controller.submit(second) is False
    assert job_queue.qsize() == 1


class _ProcessController:
    disabled = False

    def __init__(self):
        self.is_alive = True
        self.started = 0
        self.submitted = []
        self.results = []
        self.stopped = 0
        self.restarted = 0

    def start(self):
        self.started += 1
        return True

    def submit(self, job):
        self.submitted.append(job)
        return True

    def poll(self, max_results=8):
        results = list(self.results[:max_results])
        del self.results[:max_results]
        return results

    def stop(self, timeout=2.0):
        self.stopped += 1

    def restart(self):
        self.restarted += 1
        self.is_alive = True
        return True


class _Database:
    def __init__(self, event):
        self.event = event
        self.updates = []

    def get_config(self, key, default=None):
        return default

    def get_event(self, event_id):
        return dict(self.event) if int(event_id) == int(self.event["event_id"]) else None

    def update_event_ocr_result(self, *args, **kwargs):
        self.updates.append((args, kwargs))
        return True


def test_ocr_manager_submits_saved_event_to_process_runtime(tmp_path):
    event_dir = tmp_path / "000007"
    event_dir.mkdir()
    _write_image(event_dir / "bib.jpg", 7)
    (event_dir / "meta.json").write_text(
        json.dumps({"event_id": 7, "paths": {"bib": "bib.jpg"}}),
        encoding="utf-8",
    )
    database = _Database(
        {"event_id": 7, "evidence_dir": str(event_dir), "manual_corrected": 0}
    )
    controller = _ProcessController()
    manager = OCRManager(database, process_controller=controller)

    assert manager.start_process_runtime() is True
    controller.results.append(OcrResult(event_id=-1, status="READY"))
    manager.poll_process_results()
    assert manager.runtime_state == "ready"
    assert manager.enqueue_live_event(7, evidence_dir=str(event_dir)) is True
    assert controller.submitted[0].event_id == 7
    assert controller.submitted[0].candidate_paths == (str((event_dir / "bib.jpg").resolve()),)


def test_ocr_manager_applies_result_in_main_process_without_overwriting_manual_result(tmp_path):
    event_dir = tmp_path / "000007"
    event_dir.mkdir()
    (event_dir / "meta.json").write_text(
        json.dumps({"event_id": 7, "paths": {}}),
        encoding="utf-8",
    )
    event = {"event_id": 7, "evidence_dir": str(event_dir), "manual_corrected": 0}
    database = _Database(event)
    controller = _ProcessController()
    manager = OCRManager(database, process_controller=controller)
    completed = []
    manager.on_event_done = lambda event_id, result: completed.append((event_id, result))
    controller.results.append(
        OcrResult(
            event_id=7,
            text="125",
            confidence=0.93,
            status="DONE",
            source="bib.jpg",
            elapsed_ms=180.0,
            attempted_candidates=1,
        )
    )

    manager.poll_process_results()

    assert database.updates[0][0][:4] == (7, "125", 0.93, "DONE")
    assert json.loads((event_dir / "result.json").read_text(encoding="utf-8"))["bib"] == "125"
    assert completed[0][0] == 7

    event["manual_corrected"] = 1
    controller.results.append(OcrResult(event_id=7, text="126", confidence=0.99, status="DONE"))
    manager.poll_process_results()
    assert len(database.updates) == 1


def test_ocr_manager_restarts_once_then_disables_failed_process():
    database = _Database({"event_id": 7, "evidence_dir": "missing", "manual_corrected": 0})
    controller = _ProcessController()
    manager = OCRManager(database, process_controller=controller)
    manager.runtime_state = "ready"
    controller.is_alive = False

    manager.poll_process_results()

    assert controller.restarted == 1
    assert manager.runtime_state == "loading"

    controller.is_alive = False
    manager.poll_process_results()

    assert controller.restarted == 1
    assert manager.runtime_state == "disabled"
