from pathlib import Path
from typing import Optional

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
