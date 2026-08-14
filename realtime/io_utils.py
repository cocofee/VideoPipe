import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np


def read_image_unicode(path: Path) -> Optional[np.ndarray]:
    try:
        path = Path(path).absolute()
        if not path.exists():
            return None
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def write_image_unicode(path: Path, image: np.ndarray, quality: int = 90, encode_ext: Optional[str] = None) -> bool:
    if image is None or (hasattr(image, 'size') and image.size == 0):
        return False
    try:
        path = Path(path).absolute()
        path.parent.mkdir(parents=True, exist_ok=True)
        ext = (encode_ext or path.suffix or ".jpg").lower()
        if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
            ext = ".jpg"
        encode_params = []
        if ext in [".jpg", ".jpeg"]:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
        success, encoded = cv2.imencode(ext, image, encode_params)
        if not success:
            return False
        encoded.tofile(str(path))
        return path.exists()
    except Exception:
        return False


def read_pending_ocr_candidate(
    event: Dict[str, Any],
    output_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Read a non-formal OCR candidate saved beside the event evidence."""
    if not event or event.get("bib_number") or int(event.get("manual_corrected") or 0):
        return None

    evidence_dir = event.get("evidence_dir")
    if evidence_dir:
        event_dir = Path(str(evidence_dir)).expanduser()
        if not event_dir.is_absolute() and output_dir:
            event_dir = Path(output_dir) / event_dir
    else:
        event_id = event.get("event_id")
        if not output_dir or not event_id:
            return None
        event_dir = Path(output_dir) / "evidence_photos" / f"{int(event_id):06d}"

    result_path = event_dir / "result.json"
    if not result_path.is_file():
        return None

    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(result, dict) or str(result.get("status") or "").upper() == "DONE":
        return None

    bib = str(result.get("bib") or "").strip().upper()
    if not re.fullmatch(r"(?:[A-Z]{1,3})?\d{1,6}", bib):
        return None
    try:
        confidence = max(0.0, min(1.0, float(result.get("confidence") or 0.0)))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "bib": bib,
        "confidence": confidence,
        "error": str(result.get("error") or ""),
        "source": str(result.get("source") or ""),
    }
