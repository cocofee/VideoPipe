"""Runtime path helpers shared by source and packaged executions."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


def application_dir(
    *,
    frozen: Optional[bool] = None,
    executable: Optional[str] = None,
    module_file: Optional[str] = None,
) -> Path:
    """Return the directory that owns runtime files such as models and RaceData."""
    if frozen is None:
        frozen = bool(getattr(sys, "frozen", False))
    if frozen:
        return Path(executable or sys.executable).expanduser().resolve().parent
    return Path(module_file or __file__).expanduser().resolve().parent.parent


def resource_dir(relative: str = "") -> Path:
    """Resolve a bundled PyInstaller resource, with a source-tree fallback."""
    bundle_root = Path(getattr(sys, "_MEIPASS", application_dir())).resolve()
    return (bundle_root / relative).resolve() if relative else bundle_root


def resolve_runtime_path(value: str | os.PathLike[str], *, base_dir: Optional[Path] = None) -> Path:
    """Resolve a CLI path relative to the executable/project directory."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return ((base_dir or application_dir()) / path).resolve()


def resolve_output_dir(value: str | os.PathLike[str], *, base_dir: Optional[Path] = None) -> Path:
    return resolve_runtime_path(value, base_dir=base_dir)


def resolve_source(value: str, *, base_dir: Optional[Path] = None) -> str:
    """Resolve local video files while preserving cameras and network streams."""
    normalized = str(value).strip()
    if not normalized or normalized.isdigit() or "://" in normalized:
        return normalized
    return str(resolve_runtime_path(normalized, base_dir=base_dir))


def find_model(
    *,
    base_dir: Optional[Path] = None,
    project_root: Optional[Path] = None,
    cwd: Optional[Path] = None,
) -> Optional[Path]:
    """Find a packaged model first, then the newest training output."""
    runtime_root = (base_dir or application_dir()).resolve()

    for name in ("best.engine", "best.pt"):
        candidate = runtime_root / name
        if candidate.is_file():
            return candidate.resolve()

    roots = []
    for root in (
        runtime_root / "runs" / "detect",
        (project_root or runtime_root) / "runs" / "detect",
        (cwd or Path.cwd()) / "runs" / "detect",
    ):
        resolved = root.resolve()
        if resolved not in roots:
            roots.append(resolved)

    candidates = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in ("**/best.engine", "**/best.pt"):
            candidates.extend(path for path in root.glob(pattern) if path.is_file())

    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0].resolve()


def find_ocr_model_dirs(*, models_root: Optional[Path] = None) -> tuple[Optional[Path], Optional[Path]]:
    """Locate offline PaddleOCR detection and recognition models."""
    configured_root = os.environ.get("VIDEOPIPE_OCR_MODELS")
    roots = []
    for root in (
        models_root,
        Path(configured_root).expanduser() if configured_root else None,
        resource_dir("ocr_models"),
        Path.home() / ".paddlex" / "official_models",
    ):
        if root is None:
            continue
        resolved = root.resolve()
        if resolved not in roots:
            roots.append(resolved)

    for root in roots:
        detection = root / "PP-OCRv5_server_det"
        recognition = root / "en_PP-OCRv5_mobile_rec"
        if detection.is_dir() and recognition.is_dir():
            return detection, recognition
    return None, find_recognition_model_dir(models_root=models_root)


def find_recognition_model_dir(*, models_root: Optional[Path] = None) -> Optional[Path]:
    """Locate the mobile recognizer without requiring a text-detection model."""
    configured_root = os.environ.get("VIDEOPIPE_OCR_MODELS")
    roots = []
    for root in (
        models_root,
        Path(configured_root).expanduser() if configured_root else None,
        resource_dir("ocr_models"),
        Path.home() / ".paddlex" / "official_models",
    ):
        if root is None:
            continue
        resolved = root.resolve()
        if resolved not in roots:
            roots.append(resolved)

    for root in roots:
        recognition = root / "en_PP-OCRv5_mobile_rec"
        if recognition.is_dir():
            return recognition.resolve()
    return None
