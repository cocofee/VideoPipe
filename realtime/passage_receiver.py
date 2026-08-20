"""Durable CycleRace passage-event receiver for VideoPipe."""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass
from enum import Enum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping, Optional
from urllib.parse import urlsplit


logger = logging.getLogger("VideoPipe.PassageReceiver")

SCHEMA_VERSION = 1
MESSAGE_TYPE = "passage"
ACK_MESSAGE_TYPE = "passage_ack"
DEFAULT_PATH = "/api/v1/passage-events"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 18765
MAX_BODY_BYTES = 1024 * 1024

_MISSING = object()


class PassageEventError(ValueError):
    """Raised when a passage payload does not satisfy protocol v1."""


class PassageEventConflictError(RuntimeError):
    """Raised when one revision is reused with different content."""


class PassageEventDeliveryError(RuntimeError):
    """Raised when the accepted-event callback cannot be notified."""


class PassageJournalError(RuntimeError):
    """Raised when the durable JSONL journal is invalid or unavailable."""


class PassageIngestResult(str, Enum):
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"


def _string_field(
    payload: Mapping[str, Any],
    name: str,
    *,
    default: object = _MISSING,
) -> str:
    if name not in payload:
        if default is _MISSING:
            raise PassageEventError(f"{name} is required")
        return str(default)
    value = payload[name]
    if not isinstance(value, str):
        raise PassageEventError(f"{name} must be a string")
    return value


def _integer_field(
    payload: Mapping[str, Any],
    name: str,
    *,
    default: object = _MISSING,
) -> int:
    if name not in payload:
        if default is _MISSING:
            raise PassageEventError(f"{name} is required")
        return int(default)
    value = payload[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise PassageEventError(f"{name} must be an integer")
    return value


@dataclass(frozen=True, slots=True)
class PassageEvent:
    """Stable CycleRace -> VideoPipe protocol v1 payload."""

    event_id: str
    race_id: str
    stage_id: str
    group_id: str
    sequence: int
    chip_id: str = ""
    bib: str = ""
    passage_time_ms: int = 0
    lap: int = 0
    source: str = "cyclerace"
    emitted_at_ms: int = 0
    revision: int = 1
    schema_version: int = SCHEMA_VERSION
    message_type: str = MESSAGE_TYPE

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise PassageEventError("unsupported passage event schema_version")
        if self.message_type != MESSAGE_TYPE:
            raise PassageEventError("message_type must be passage")
        for name in ("event_id", "race_id", "stage_id", "group_id"):
            if not str(getattr(self, name)).strip():
                raise PassageEventError(f"{name} is required")
        if self.sequence <= 0:
            raise PassageEventError("sequence must be positive")
        if not self.chip_id.strip() and not self.bib.strip():
            raise PassageEventError("chip_id or bib is required")
        if self.passage_time_ms < 0:
            raise PassageEventError("passage_time_ms must be non-negative")
        if self.lap < 0:
            raise PassageEventError("lap must be non-negative")
        if not self.source.strip():
            raise PassageEventError("source is required")
        if self.emitted_at_ms < 0:
            raise PassageEventError("emitted_at_ms must be non-negative")
        if self.revision <= 0:
            raise PassageEventError("revision must be positive")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PassageEvent":
        if not isinstance(payload, Mapping):
            raise PassageEventError("passage event must be a JSON object")
        return cls(
            schema_version=_integer_field(payload, "schema_version"),
            message_type=_string_field(payload, "message_type"),
            event_id=_string_field(payload, "event_id"),
            race_id=_string_field(payload, "race_id"),
            stage_id=_string_field(payload, "stage_id"),
            group_id=_string_field(payload, "group_id"),
            sequence=_integer_field(payload, "sequence"),
            chip_id=_string_field(payload, "chip_id", default=""),
            bib=_string_field(payload, "bib", default=""),
            passage_time_ms=_integer_field(payload, "passage_time_ms"),
            lap=_integer_field(payload, "lap"),
            source=_string_field(payload, "source"),
            emitted_at_ms=_integer_field(payload, "emitted_at_ms", default=0),
            revision=_integer_field(payload, "revision", default=1),
        )

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        return {
            "schema_version": payload["schema_version"],
            "message_type": payload["message_type"],
            "event_id": payload["event_id"],
            "race_id": payload["race_id"],
            "stage_id": payload["stage_id"],
            "group_id": payload["group_id"],
            "sequence": payload["sequence"],
            "chip_id": payload["chip_id"],
            "bib": payload["bib"],
            "passage_time_ms": payload["passage_time_ms"],
            "lap": payload["lap"],
            "source": payload["source"],
            "emitted_at_ms": payload["emitted_at_ms"],
            "revision": payload["revision"],
        }


def _looks_like_incomplete_json(value: str) -> bool:
    in_string = False
    escaped = False
    nesting = 0
    for character in value:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            nesting += 1
        elif character in "]}":
            if nesting == 0:
                return False
            nesting -= 1
    return in_string or nesting > 0


class PassageEventStore:
    """Append-only JSONL store retaining the latest revision per event id."""

    def __init__(self, journal_path: str | Path):
        self.journal_path = Path(journal_path).expanduser().absolute()
        if not str(self.journal_path):
            raise ValueError("passage event journal path is required")
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._events: dict[str, PassageEvent] = {}
        self._event_order: list[str] = []
        self._recovered_incomplete_tail = False
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.journal_path.exists():
            return
        try:
            content = self.journal_path.read_bytes()
        except OSError as error:
            raise PassageJournalError(
                f"failed to read passage event journal: {self.journal_path}"
            ) from error

        offset = 0
        lines = content.splitlines(keepends=True)
        for line_number, raw_line in enumerate(lines, start=1):
            terminated = raw_line.endswith(b"\n") or raw_line.endswith(b"\r")
            stripped = raw_line.rstrip(b"\r\n")
            if not stripped:
                offset += len(raw_line)
                continue
            try:
                text = stripped.decode("utf-8")
                payload = json.loads(text)
                event = PassageEvent.from_payload(payload)
            except (UnicodeDecodeError, json.JSONDecodeError, PassageEventError) as error:
                is_tail = line_number == len(lines) and not terminated
                if is_tail:
                    try:
                        candidate = stripped.decode("utf-8")
                    except UnicodeDecodeError:
                        candidate = ""
                    if candidate and _looks_like_incomplete_json(candidate):
                        self._truncate(offset)
                        self._recovered_incomplete_tail = True
                        return
                raise PassageJournalError(
                    f"invalid passage event journal line {line_number}: {error}"
                ) from error
            self._merge_loaded(event, line_number)
            offset += len(raw_line)

    def _merge_loaded(self, event: PassageEvent, line_number: int) -> None:
        current = self._events.get(event.event_id)
        if current is None:
            self._event_order.append(event.event_id)
            self._events[event.event_id] = event
            return
        if event.revision < current.revision:
            return
        if event.revision == current.revision:
            if event != current:
                raise PassageJournalError(
                    "conflicting passage event revision in journal "
                    f"line {line_number}: {event.event_id}"
                )
            return
        self._events[event.event_id] = event

    def _truncate(self, size: int) -> None:
        try:
            with self.journal_path.open("r+b") as journal:
                journal.truncate(size)
                journal.flush()
                os.fsync(journal.fileno())
        except OSError as error:
            raise PassageJournalError(
                f"failed to recover passage event journal: {self.journal_path}"
            ) from error

    def append(self, event: PassageEvent) -> PassageIngestResult:
        if not isinstance(event, PassageEvent):
            raise TypeError("event must be a PassageEvent")
        with self._lock:
            current = self._events.get(event.event_id)
            if current is not None:
                if event.revision < current.revision:
                    return PassageIngestResult.DUPLICATE
                if event.revision == current.revision:
                    if event != current:
                        raise PassageEventConflictError(
                            "passage event revision was reused with different content: "
                            f"{event.event_id}"
                        )
                    return PassageIngestResult.DUPLICATE

            record = (
                json.dumps(
                    event.to_payload(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
            original_size = (
                self.journal_path.stat().st_size
                if self.journal_path.exists()
                else 0
            )
            separator = b""
            if original_size:
                try:
                    with self.journal_path.open("rb") as journal:
                        journal.seek(-1, os.SEEK_END)
                        if journal.read(1) not in {b"\n", b"\r"}:
                            separator = b"\n"
                except OSError as error:
                    raise PassageJournalError(
                        f"failed to inspect passage event journal: {self.journal_path}"
                    ) from error
            try:
                with self.journal_path.open("ab") as journal:
                    journal.write(separator)
                    journal.write(record)
                    journal.flush()
                    os.fsync(journal.fileno())
            except OSError as error:
                try:
                    self._truncate(original_size)
                except PassageJournalError:
                    logger.exception("Failed to roll back passage event journal")
                raise PassageJournalError(
                    f"failed to append passage event journal: {self.journal_path}"
                ) from error

            if current is None:
                self._event_order.append(event.event_id)
            self._events[event.event_id] = event
            return PassageIngestResult.ACCEPTED

    def get(self, event_id: str) -> Optional[PassageEvent]:
        with self._lock:
            return self._events.get(str(event_id))

    def events(self) -> tuple[PassageEvent, ...]:
        with self._lock:
            return tuple(self._events[event_id] for event_id in self._event_order)

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    @property
    def recovered_incomplete_tail(self) -> bool:
        with self._lock:
            return self._recovered_incomplete_tail


class PassageEventIngestor:
    """Transport-neutral ingestion with durable-before-callback ordering."""

    def __init__(
        self,
        store: PassageEventStore,
        on_accepted: Optional[Callable[[PassageEvent], None]] = None,
    ):
        self.store = store
        self._on_accepted = on_accepted
        self._delivery_lock = threading.RLock()
        self._delivered_revisions: dict[str, int] = {}

    def ingest_payload(self, payload: Mapping[str, Any]) -> PassageIngestResult:
        return self.ingest(PassageEvent.from_payload(payload))

    def ingest(self, event: PassageEvent) -> PassageIngestResult:
        result = self.store.append(event)
        current = self.store.get(event.event_id)
        if current is None or current.revision != event.revision:
            return PassageIngestResult.DUPLICATE

        with self._delivery_lock:
            delivered_revision = self._delivered_revisions.get(event.event_id, 0)
            if self._on_accepted is not None and delivered_revision < event.revision:
                try:
                    self._on_accepted(current)
                except Exception as error:
                    raise PassageEventDeliveryError(str(error)) from error
                self._delivered_revisions[event.event_id] = event.revision
        return result


class _PassageHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def _handler_type(
    ingestor: PassageEventIngestor,
    request_path: str,
    max_body_bytes: int,
):
    class PassageRequestHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            if urlsplit(self.path).path != request_path:
                self._send_json(404, "rejected", "endpoint not found")
                return
            length_header = self.headers.get("Content-Length")
            try:
                content_length = int(length_header) if length_header is not None else -1
            except ValueError:
                content_length = -1
            if content_length < 0:
                self._send_json(400, "rejected", "valid Content-Length is required")
                return
            if content_length > max_body_bytes:
                self._send_json(413, "rejected", "passage event payload is too large")
                return
            try:
                raw_body = self.rfile.read(content_length)
                payload = json.loads(raw_body.decode("utf-8"))
                result = ingestor.ingest_payload(payload)
                status = 201 if result is PassageIngestResult.ACCEPTED else 200
                self._send_json(status, result.value)
            except PassageEventConflictError as error:
                self._send_json(409, "rejected", str(error))
            except (UnicodeDecodeError, json.JSONDecodeError, PassageEventError) as error:
                self._send_json(400, "rejected", str(error))
            except PassageEventDeliveryError as error:
                self._send_json(503, "retry", str(error))
            except Exception as error:
                logger.exception("Passage event request failed")
                self._send_json(500, "error", str(error))

        def _send_json(self, status: int, result: str, error: str = "") -> None:
            body: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "message_type": ACK_MESSAGE_TYPE,
                "status": result,
            }
            if error:
                body["error"] = error
            encoded = json.dumps(
                body,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(encoded)
            self.close_connection = True

        def log_message(self, format: str, *args: Any) -> None:
            logger.debug("CycleRace HTTP: " + format, *args)

    return PassageRequestHandler


class PassageEventReceiver:
    """Background HTTP receiver scoped to one VideoPipe race directory."""

    def __init__(
        self,
        host: str,
        port: int,
        store: PassageEventStore,
        *,
        request_path: str = DEFAULT_PATH,
        max_body_bytes: int = MAX_BODY_BYTES,
        on_accepted: Optional[Callable[[PassageEvent], None]] = None,
    ):
        host = str(host).strip()
        port = int(port)
        request_path = str(request_path).strip()
        if not host:
            raise ValueError("passage receiver host is required")
        if port < 0 or port > 65535:
            raise ValueError("passage receiver port is out of range")
        if not request_path.startswith("/"):
            raise ValueError("passage receiver path must start with /")
        if int(max_body_bytes) <= 0:
            raise ValueError("max_body_bytes must be positive")
        self.host = host
        self.port = port
        self.request_path = request_path
        self.max_body_bytes = int(max_body_bytes)
        self.store = store
        self.ingestor = PassageEventIngestor(store, on_accepted=on_accepted)
        self._lock = threading.RLock()
        self._server: Optional[_PassageHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            handler = _handler_type(
                self.ingestor,
                self.request_path,
                self.max_body_bytes,
            )
            server = _PassageHTTPServer((self.host, self.port), handler)
            thread = threading.Thread(
                target=server.serve_forever,
                kwargs={"poll_interval": 0.1},
                name="CycleRacePassageReceiver",
                daemon=True,
            )
            self._server = server
            self._thread = thread
            thread.start()
            logger.info(
                "CycleRace passage receiver listening on %s:%s%s",
                self.host,
                self.listen_port,
                self.request_path,
            )

    def stop(self) -> None:
        with self._lock:
            server = self._server
            thread = self._thread
            self._server = None
            self._thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=5.0)
            if thread.is_alive():
                logger.error("CycleRace passage receiver did not stop within 5 seconds")

    @property
    def listen_port(self) -> int:
        with self._lock:
            if self._server is not None:
                return int(self._server.server_address[1])
            return self.port

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()


__all__ = [
    "ACK_MESSAGE_TYPE",
    "DEFAULT_HOST",
    "DEFAULT_PATH",
    "DEFAULT_PORT",
    "MESSAGE_TYPE",
    "PassageEvent",
    "PassageEventConflictError",
    "PassageEventDeliveryError",
    "PassageEventError",
    "PassageEventIngestor",
    "PassageEventReceiver",
    "PassageEventStore",
    "PassageIngestResult",
    "PassageJournalError",
    "SCHEMA_VERSION",
]
