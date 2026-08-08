from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_ocr_package_contains_only_mobile_recognition_model():
    spec = (ROOT / "packaging" / "VideoPipeRealtimeOCR.spec").read_text(encoding="utf-8")
    worker = (ROOT / "realtime" / "ocr_worker.py").read_text(encoding="utf-8")

    assert "en_PP-OCRv5_mobile_rec" in spec
    assert "PP-OCRv5_server_det" not in spec
    assert '("paddle", "paddleocr", "paddlex", "modelscope")' not in spec
    assert '"modelscope"' in spec
    assert 'sys.modules["modelscope"]' in worker
    assert "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK" in worker


def test_yolo_package_excludes_all_ocr_runtimes():
    spec = (ROOT / "packaging" / "VideoPipeRealtimeYOLO.spec").read_text(encoding="utf-8")

    for package in ("paddle", "paddleocr", "paddlex", "rapidocr_onnxruntime"):
        assert f'"{package}"' in spec


def test_build_script_validates_inputs_before_replacing_package():
    script = (ROOT / "packaging" / "build_realtime.ps1").read_text(encoding="utf-8")

    build_index = script.index("python -m PyInstaller")
    assert script.index("$ResolvedModel =") < build_index
    assert script.index("$ResolvedSource =") < build_index
    assert "--ocr-cpu-threads 1" in script


def test_main_window_never_constructs_full_paddleocr():
    source = (ROOT / "realtime" / "main_window.py").read_text(encoding="utf-8")

    assert "from paddleocr import PaddleOCR" not in source
    assert "PP-OCRv5_server_det" not in source
