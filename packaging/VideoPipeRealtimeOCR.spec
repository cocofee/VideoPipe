# -*- mode: python ; coding: utf-8 -*-

import os
import shutil
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, copy_metadata


ROOT = Path(SPECPATH).resolve().parent
datas = [
    (str(ROOT / "realtime" / "custom_bytetrack.yaml"), "realtime"),
]
binaries = []
hiddenimports = []

ffmpeg_value = os.environ.get("VIDEOPIPE_FFMPEG") or shutil.which("ffmpeg")
ffmpeg_path = Path(ffmpeg_value).expanduser().resolve() if ffmpeg_value else None
if ffmpeg_path is None or not ffmpeg_path.is_file():
    raise SystemExit("Required FFmpeg executable is missing; set VIDEOPIPE_FFMPEG")
binaries.append((str(ffmpeg_path), "."))

datas += collect_data_files("ultralytics", includes=["cfg/trackers/*.yaml"])
for package in ("paddle", "paddleocr", "paddlex"):
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

for package in ("imagesize", "pyclipper", "pypdfium2", "bidi", "shapely"):
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

for distribution in (
    "imagesize",
    "opencv-contrib-python",
    "pyclipper",
    "pypdfium2",
    "python-bidi",
    "shapely",
):
    datas += copy_metadata(distribution)

models_root = Path(
    os.environ.get("VIDEOPIPE_OCR_MODELS", Path.home() / ".paddlex" / "official_models")
).expanduser().resolve()
model_name = "en_PP-OCRv5_mobile_rec"
model_dir = models_root / model_name
if not model_dir.is_dir():
    raise SystemExit(f"Required offline OCR model is missing: {model_dir}")
datas.append((str(model_dir), f"ocr_models/{model_name}"))

a = Analysis(
    [str(ROOT / "realtime" / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["datasets", "modelscope", "openvino", "rapidocr_onnxruntime", "tensorrt"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VideoPipeRealtimeOCR",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="VideoPipeRealtimeOCR",
)
