import sys
from types import ModuleType
from pathlib import Path

import numpy as np

from realtime.detector import PaddleOcrAdapter
from realtime import ocr_worker


def test_recognition_worker_uses_mobile_model_without_server_detector(monkeypatch, tmp_path):
    captured_recognition_kwargs = {}

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            raise AssertionError("Full PaddleOCR must not be constructed")

    class FakeTextRecognition:
        def __init__(self, **kwargs):
            captured_recognition_kwargs.update(kwargs)

    paddleocr_module = ModuleType("paddleocr")
    paddleocr_module.PaddleOCR = FakePaddleOCR
    paddleocr_module.TextRecognition = FakeTextRecognition
    paddleocr_module.__version__ = "3.3.2"

    monkeypatch.setitem(sys.modules, "paddleocr", paddleocr_module)
    recognition_model = tmp_path / "en_PP-OCRv5_mobile_rec"
    recognition_model.mkdir()
    monkeypatch.setattr(
        ocr_worker,
        "find_recognition_model_dir",
        lambda models_root=None: recognition_model.resolve(),
    )

    recognizer = ocr_worker._create_recognizer(cpu_threads=2, models_root=str(tmp_path))

    assert isinstance(recognizer, FakeTextRecognition)
    assert captured_recognition_kwargs["enable_mkldnn"] is False
    assert captured_recognition_kwargs["cpu_threads"] == 2
    assert captured_recognition_kwargs["device"] == "cpu"
    assert captured_recognition_kwargs["model_name"] == "en_PP-OCRv5_mobile_rec"
    assert captured_recognition_kwargs["model_dir"] == str(recognition_model.resolve())


def test_paddle_adapter_uses_recognition_only_when_detection_returns_empty():
    class EmptyDetectionOcr:
        def __init__(self):
            self.calls = []

        def ocr(self, image, det, cls):
            self.calls.append((det, cls))
            if det:
                return [[]]
            return [[["A1234", 0.91]]]

    ocr = EmptyDetectionOcr()
    adapter = PaddleOcrAdapter(ocr)

    lines, _ = adapter(np.zeros((32, 40, 3), dtype=np.uint8))

    assert lines == [[None, "A1234", 0.91]]
    assert ocr.calls == [(True, True), (False, False)]


def test_paddle_v3_adapter_uses_public_recognizer_when_predict_returns_empty():
    class EmptyDetectionPipeline:
        def __init__(self):
            self.calls = 0

        def predict(self, image, **kwargs):
            self.calls += 1
            return [{"rec_texts": [""], "rec_scores": [0.0]}]

    class RecognitionOnly:
        def __init__(self):
            self.calls = 0

        def predict(self, image, batch_size):
            self.calls += 1
            return [{"res": {"rec_text": "A1234", "rec_score": 0.91}}]

    ocr = EmptyDetectionPipeline()
    recognizer = RecognitionOnly()
    adapter = PaddleOcrAdapter(ocr, recognizer=recognizer)

    lines, _ = adapter(np.zeros((32, 40, 3), dtype=np.uint8))

    assert lines == [[None, "A1234", 0.91]]
    assert ocr.calls == 1
    assert recognizer.calls == 1


def test_paddle_v3_adapter_skips_recognizer_when_predict_finds_text():
    class DetectionPipeline:
        def predict(self, image, **kwargs):
            return [{"rec_texts": ["A1234"], "rec_scores": [0.93]}]

    class RecognitionOnly:
        def __init__(self):
            self.calls = 0

        def predict(self, image, batch_size):
            self.calls += 1
            return []

    recognizer = RecognitionOnly()
    adapter = PaddleOcrAdapter(DetectionPipeline(), recognizer=recognizer)

    lines, _ = adapter(np.zeros((32, 40, 3), dtype=np.uint8))

    assert lines == [[None, "A1234", 0.93]]
    assert recognizer.calls == 0


def test_paddle_v3_adapter_can_run_recognition_only_explicitly():
    class DetectionPipeline:
        def predict(self, image, **kwargs):
            return [{"rec_texts": ["4"], "rec_scores": [0.3]}]

    class RecognitionOnly:
        def __init__(self):
            self.calls = 0

        def predict(self, image, batch_size):
            self.calls += 1
            return [{"res": {"rec_text": "33", "rec_score": 0.94}}]

    recognizer = RecognitionOnly()
    adapter = PaddleOcrAdapter(DetectionPipeline(), recognizer=recognizer)

    lines, _ = adapter.recognize_only(np.zeros((32, 40, 3), dtype=np.uint8))

    assert lines == [[None, "33", 0.94]]
    assert recognizer.calls == 1
