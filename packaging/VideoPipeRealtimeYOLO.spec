# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files


ROOT = Path(SPECPATH).resolve().parent
datas = [
    (str(ROOT / "realtime" / "custom_bytetrack.yaml"), "realtime"),
]
datas += collect_data_files("ultralytics", includes=["cfg/trackers/*.yaml"])

a = Analysis(
    [str(ROOT / "realtime" / "main.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "onnxruntime",
        "openvino",
        "paddle",
        "paddleocr",
        "paddlex",
        "rapidocr_onnxruntime",
        "tensorrt",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VideoPipeRealtimeYOLO",
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
    name="VideoPipeRealtimeYOLO",
)
