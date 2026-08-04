import sys
from types import ModuleType

import numpy as np

from realtime.detector import PaddleOcrAdapter
from realtime.main_window import OCRInitThread


def test_paddle_ocr_init_disables_mkldnn(monkeypatch):
    captured_ocr_kwargs = {}
    captured_recognition_kwargs = {}

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            captured_ocr_kwargs.update(kwargs)

    class FakeTextRecognition:
        def __init__(self, **kwargs):
            captured_recognition_kwargs.update(kwargs)

    paddleocr_module = ModuleType("paddleocr")
    paddleocr_module.PaddleOCR = FakePaddleOCR
    paddleocr_module.TextRecognition = FakeTextRecognition
    paddleocr_module.__version__ = "3.3.2"

    paddle_module = ModuleType("paddle")
    paddle_module.__version__ = "3.3.0"
    paddle_module.is_compiled_with_cuda = lambda: False

    paddlex_module = ModuleType("paddlex")
    paddlex_module.__version__ = "3.3.13"

    monkeypatch.setitem(sys.modules, "paddleocr", paddleocr_module)
    monkeypatch.setitem(sys.modules, "paddle", paddle_module)
    monkeypatch.setitem(sys.modules, "paddlex", paddlex_module)

    thread = OCRInitThread("paddleocr")
    thread.run()

    assert captured_ocr_kwargs["enable_mkldnn"] is False
    assert captured_recognition_kwargs["enable_mkldnn"] is False
    assert captured_recognition_kwargs["model_name"] == "en_PP-OCRv5_mobile_rec"


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
