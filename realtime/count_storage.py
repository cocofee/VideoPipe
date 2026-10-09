"""Single-writer counting journals and incremental readers of chip revisions."""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading


class JsonlTail:
    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self._identity = None

    def read(self, limit=10000):
        if not self.path.exists():
            return []
        stat = self.path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity != self._identity or stat.st_size < self.offset:
            self.offset = 0
            self._identity = identity
        records = []
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            for _ in range(limit):
                start = handle.tell()
                line = handle.readline(4 * 1024 * 1024 + 1)
                if len(line) > 4 * 1024 * 1024:
                    raise ValueError("journal record exceeds 4 MiB")
                if not line or not line.endswith(b"\n"):
                    handle.seek(start)
                    break
                if line.strip():
                    records.append(json.loads(line))
                self.offset = handle.tell()
        return records


class CountJournal:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # Keep a torn suffix for diagnosis before restoring this service's file.
        if self.path.exists():
            with self.path.open("rb") as handle:
                size = handle.seek(0, os.SEEK_END)
                handle.seek(max(0, size - 4 * 1024 * 1024))
                data = handle.read()
            if data and not data.endswith(b"\n"):
                end = data.rfind(b"\n") + 1
                if end == 0 and size > len(data):
                    raise ValueError("torn journal suffix exceeds recovery limit")
                backup = self.path.with_name(self.path.name + ".torn-tail")
                with backup.open("ab") as handle:
                    handle.write(data[end:] + b"\n")
                with self.path.open("r+b") as handle:
                    handle.truncate(size - len(data) + end)
        self._handle = self.path.open("ab")

    def append(self, records):
        payload = b"".join((json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8") for record in records)
        if not payload:
            return
        with self._lock:
            start = self._handle.tell()
            try:
                self._handle.write(payload)
                self._handle.flush()
                os.fsync(self._handle.fileno())
            except Exception:
                self._handle.seek(start)
                self._handle.truncate()
                raise

    def close(self):
        self._handle.close()


def atomic_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
