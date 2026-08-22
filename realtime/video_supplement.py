"""Manual video observations that have no matching RaceTiger chip record."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional


BEIJING_TZ = timezone(timedelta(hours=8))
logger = logging.getLogger("VideoPipe.VideoSupplement")


def parse_beijing_datetime(value: str) -> Optional[int]:
    text = str(value or "").strip().replace("T", " ")
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=BEIJING_TZ)
    return int(parsed.timestamp() * 1000.0)


@dataclass(frozen=True, slots=True)
class VideoSupplement:
    supplement_id: str
    bib: str
    observed_at_ms: int
    camera_label: str = ""
    video_path: str = ""
    frame_index: int = -1
    note: str = ""
    created_at_ms: int = 0

    def __post_init__(self) -> None:
        if not self.supplement_id.strip():
            raise ValueError("supplement_id is required")
        if not self.bib.strip():
            raise ValueError("bib is required")
        if self.observed_at_ms < 0:
            raise ValueError("observed_at_ms must be non-negative")
        if self.frame_index < -1:
            raise ValueError("frame_index must be -1 or non-negative")

    def to_payload(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "VideoSupplement":
        return cls(
            supplement_id=str(payload.get("supplement_id", "")),
            bib=str(payload.get("bib", "")),
            observed_at_ms=int(payload.get("observed_at_ms", -1)),
            camera_label=str(payload.get("camera_label", "")),
            video_path=str(payload.get("video_path", "")),
            frame_index=int(payload.get("frame_index", -1)),
            note=str(payload.get("note", "")),
            created_at_ms=int(payload.get("created_at_ms", 0)),
        )


class VideoSupplementStore:
    """Append-only JSONL store kept separate from official timing data."""

    def __init__(self, journal_path: str | Path):
        self.journal_path = Path(journal_path).expanduser().absolute()
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._items: list[VideoSupplement] = []
        self._load()

    def _load(self) -> None:
        if not self.journal_path.exists():
            return
        content = self.journal_path.read_bytes()
        for line_number, raw_line in enumerate(content.splitlines(), start=1):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("record must be a JSON object")
                item = VideoSupplement.from_payload(payload)
            except (AttributeError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
                logger.warning(
                    "Skipping invalid video supplement journal line %s: %s",
                    line_number,
                    error,
                )
                continue
            self._items.append(item)

    def append(
        self,
        *,
        bib: str,
        observed_at_ms: int,
        camera_label: str = "",
        video_path: str = "",
        frame_index: int = -1,
        note: str = "",
    ) -> VideoSupplement:
        item = VideoSupplement(
            supplement_id=f"video:{uuid.uuid4().hex}",
            bib=str(bib).strip(),
            observed_at_ms=int(observed_at_ms),
            camera_label=str(camera_label).strip(),
            video_path=str(video_path).strip(),
            frame_index=int(frame_index),
            note=str(note).strip(),
            created_at_ms=int(time.time() * 1000.0),
        )
        record = json.dumps(item.to_payload(), ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock:
            with self.journal_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(record)
                handle.flush()
                os.fsync(handle.fileno())
            self._items.append(item)
        return item

    def items(self) -> tuple[VideoSupplement, ...]:
        with self._lock:
            return tuple(self._items)

    def __len__(self) -> int:
        return len(self._items)


__all__ = ["VideoSupplement", "VideoSupplementStore", "parse_beijing_datetime"]
