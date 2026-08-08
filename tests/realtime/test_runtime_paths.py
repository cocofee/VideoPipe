import os
from pathlib import Path

from realtime.runtime_paths import (
    application_dir,
    find_model,
    find_ocr_model_dirs,
    find_recognition_model_dir,
    resolve_output_dir,
    resolve_runtime_path,
    resolve_source,
)


def test_application_dir_uses_executable_parent_when_frozen(tmp_path):
    executable = tmp_path / "VideoPipeRealtime.exe"

    assert application_dir(frozen=True, executable=str(executable)) == tmp_path.resolve()


def test_relative_runtime_paths_are_based_on_application_dir(tmp_path):
    assert resolve_runtime_path("best.pt", base_dir=tmp_path) == (tmp_path / "best.pt").resolve()
    assert resolve_output_dir("RaceData", base_dir=tmp_path) == (tmp_path / "RaceData").resolve()


def test_resolve_source_only_rebases_local_files(tmp_path):
    assert resolve_source("test.mp4", base_dir=tmp_path) == str((tmp_path / "test.mp4").resolve())
    assert resolve_source("0", base_dir=tmp_path) == "0"
    assert resolve_source("rtsp://camera/live", base_dir=tmp_path) == "rtsp://camera/live"


def test_find_model_prefers_file_next_to_executable(tmp_path):
    packaged_model = tmp_path / "best.pt"
    packaged_model.write_bytes(b"packaged")
    trained_model = tmp_path / "runs" / "detect" / "train" / "best.engine"
    trained_model.parent.mkdir(parents=True)
    trained_model.write_bytes(b"trained")
    os.utime(trained_model, (packaged_model.stat().st_mtime + 10, packaged_model.stat().st_mtime + 10))

    assert find_model(base_dir=tmp_path, project_root=tmp_path, cwd=tmp_path) == packaged_model.resolve()


def test_find_model_uses_newest_training_output(tmp_path):
    older = tmp_path / "runs" / "detect" / "old" / "best.pt"
    newer = tmp_path / "runs" / "detect" / "new" / "best.engine"
    older.parent.mkdir(parents=True)
    newer.parent.mkdir(parents=True)
    older.write_bytes(b"old")
    newer.write_bytes(b"new")
    os.utime(older, (100, 100))
    os.utime(newer, (200, 200))

    assert find_model(base_dir=tmp_path, project_root=tmp_path, cwd=tmp_path) == newer.resolve()


def test_find_ocr_model_dirs_requires_complete_offline_pair(tmp_path):
    detection = tmp_path / "PP-OCRv5_server_det"
    recognition = tmp_path / "en_PP-OCRv5_mobile_rec"
    detection.mkdir()

    assert find_ocr_model_dirs(models_root=tmp_path) != (
        detection.resolve(),
        recognition.resolve(),
    )

    recognition.mkdir()
    assert find_ocr_model_dirs(models_root=tmp_path) == (
        detection.resolve(),
        recognition.resolve(),
    )


def test_find_recognition_model_does_not_require_server_detection(tmp_path):
    recognition = tmp_path / "en_PP-OCRv5_mobile_rec"
    recognition.mkdir()

    assert find_recognition_model_dir(models_root=tmp_path) == recognition.resolve()
