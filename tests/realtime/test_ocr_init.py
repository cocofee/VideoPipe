import sys
from types import ModuleType

from realtime.main_window import OCRInitThread


def test_paddle_ocr_init_disables_mkldnn(monkeypatch):
    captured_kwargs = {}

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    paddleocr_module = ModuleType("paddleocr")
    paddleocr_module.PaddleOCR = FakePaddleOCR
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

    assert captured_kwargs["enable_mkldnn"] is False
