"""Durable operator markers for post-event video reproduction."""

from __future__ import annotations

import json
import os
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


FIELD_ISSUE_CATEGORIES = frozenset(
    {
        "missed_athlete",
        "duplicate_athlete",
        "false_athlete",
        "wrong_bib",
        "missing_bib",
        "ui_freeze",
        "camera_reconnect",
        "other",
    }
)


@dataclass(frozen=True)
class FieldIssueMarker:
    """One immutable diagnostic marker; it does not represent a race event."""

    issue_id: str
    created_at: str
    session_id: str
    source_id: int
    category: str
    frame_index: int
    capture_time_ms: float
    segment_id: Optional[int] = None
    participant_ids: tuple[str, ...] = field(default_factory=tuple)
    raw_track_ids: tuple[int, ...] = field(default_factory=tuple)
    event_ids: tuple[int, ...] = field(default_factory=tuple)
    screenshot_path: Optional[str] = None
    queue_depth: Optional[int] = None
    dropped_frames: Optional[int] = None
    processing_time_ms: Optional[float] = None
    software_commit: Optional[str] = None
    model_identity: Optional[str] = None
    event_profile: Optional[Any] = None
    operator_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("participant_ids", "raw_track_ids", "event_ids"):
            value[key] = list(value[key])
        return value


class FieldIssueLog:
    """Append-only JSONL issue log with process/thread-safe writes."""

    _thread_lock = threading.Lock()

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        session_id: str,
        source_id: int,
        category: str,
        frame_index: int,
        capture_time_ms: float,
        segment_id: Optional[int] = None,
        participant_ids: Iterable[str] = (),
        raw_track_ids: Iterable[int] = (),
        event_ids: Iterable[int] = (),
        screenshot_path: Optional[str] = None,
        metrics: Optional[Mapping[str, Any]] = None,
        software_commit: Optional[str] = None,
        model_identity: Optional[str] = None,
        event_profile: Optional[Any] = None,
        note: str = "",
        operator_note: Optional[str] = None,
    ) -> FieldIssueMarker:
        normalized_category = str(category or "").strip()
        if normalized_category not in FIELD_ISSUE_CATEGORIES:
            raise ValueError(f"Unsupported field issue category: {normalized_category}")

        metrics = dict(metrics or {})
        marker = FieldIssueMarker(
            issue_id=f"I-{uuid.uuid4().hex}",
            created_at=datetime.now(timezone.utc).isoformat(),
            session_id=str(session_id or "").strip(),
            source_id=int(source_id),
            category=normalized_category,
            frame_index=int(frame_index),
            capture_time_ms=float(capture_time_ms),
            segment_id=None if segment_id is None else int(segment_id),
            participant_ids=tuple(str(value) for value in participant_ids),
            raw_track_ids=tuple(int(value) for value in raw_track_ids),
            event_ids=tuple(int(value) for value in event_ids),
            screenshot_path=None if screenshot_path is None else str(screenshot_path),
            queue_depth=_optional_int(metrics.get("queue_depth")),
            dropped_frames=_optional_int(metrics.get("dropped_frames", metrics.get("dropped_frame_count"))),
            processing_time_ms=_optional_float(metrics.get("processing_time_ms")),
            software_commit=None if software_commit is None else str(software_commit),
            model_identity=None if model_identity is None else str(model_identity),
            event_profile=_jsonable(event_profile),
            operator_note=str(operator_note if operator_note is not None else note or ""),
        )
        payload = json.dumps(marker.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self._thread_lock:
            with _process_file_lock(self.path.with_suffix(self.path.suffix + ".lock")):
                with self.path.open("ab") as stream:
                    stream.write((payload + "\n").encode("utf-8"))
                    stream.flush()
                    os.fsync(stream.fileno())
        return marker


def _optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


@contextmanager
def _process_file_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
