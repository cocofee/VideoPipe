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


def test_realtime_packages_bundle_ffmpeg_for_manual_recording():
    recorder = (ROOT / "realtime" / "stream_recorder.py").read_text(encoding="utf-8")
    assert "resource_dir()" in recorder

    for name in ("VideoPipeRealtimeOCR.spec", "VideoPipeRealtimeYOLO.spec"):
        spec = (ROOT / "packaging" / name).read_text(encoding="utf-8")
        assert "VIDEOPIPE_FFMPEG" in spec
        assert 'binaries.append((str(ffmpeg_path), "."))' in spec


def test_build_script_validates_inputs_before_replacing_package():
    script = (ROOT / "packaging" / "build_realtime.ps1").read_text(encoding="utf-8")

    build_index = script.index("python -m PyInstaller")
    assert script.index("$ResolvedModel =") < build_index
    assert script.index("$ResolvedSource =") < build_index
    assert script.index("$ResolvedFfmpeg =") < build_index
    assert "$env:VIDEOPIPE_FFMPEG = $ResolvedFfmpeg" in script
    assert "--ocr-cpu-threads 1" in script
    assert '--sport-profile $SportProfile' in script
    assert 'Start-$Variant-$SportProfile.cmd' in script


def test_build_requires_a_clean_distribution_directory():
    build_script = (ROOT / "packaging" / "build_realtime.ps1").read_text(encoding="utf-8")
    clean_check = (ROOT / "packaging" / "assert_clean_distribution.ps1").read_text(
        encoding="utf-8"
    )

    assert 'assert_clean_distribution.ps1") -AppDir $AppDir' in build_script
    for runtime_path in ("RaceData", "logs", "config.json", "global_config.json"):
        assert f'"{runtime_path}"' in clean_check
    assert "Distribution contains runtime state" in clean_check


def test_main_window_never_constructs_full_paddleocr():
    source = (ROOT / "realtime" / "main_window.py").read_text(encoding="utf-8")

    assert "from paddleocr import PaddleOCR" not in source
    assert "PP-OCRv5_server_det" not in source
