"""Export a reproducible field-issue package without copying large video by default."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urlsplit, urlunsplit


_SENSITIVE_KEY_PARTS = ("api_key", "apikey", "token", "password", "passwd", "secret", "authorization")
_SENSITIVE_VALUE_PATTERN = re.compile(
    r"(?i)(\b(?:api[_-]?key|token|password|passwd|secret|authorization)\b\s*[:=]\s*)([^\s,;]+)"
)


def export_field_issue_package(
    *,
    session_dir: str | Path,
    source_video: str | Path,
    output_dir: str | Path,
    copy_video: bool = False,
) -> Path:
    session_dir = Path(session_dir).resolve()
    source_video = Path(source_video).resolve()
    output_dir = Path(output_dir).resolve()
    if not session_dir.is_dir():
        raise FileNotFoundError(f"Session directory does not exist: {session_dir}")
    if not source_video.is_file():
        raise FileNotFoundError(f"Source video does not exist: {source_video}")
    if output_dir == session_dir or session_dir in output_dir.parents:
        raise ValueError("Output directory must be outside the session directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evidence").mkdir(exist_ok=True)
    (output_dir / "database").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    _copy_redacted_text(session_dir / "issues.jsonl", output_dir / "issues.jsonl")
    _copy_session_metadata(session_dir, output_dir)
    _copy_database(session_dir, output_dir / "database")
    _copy_evidence(session_dir, output_dir / "evidence")
    _copy_logs(session_dir, output_dir / "logs")

    source_metadata = {
        "path": str(source_video),
        "size_bytes": source_video.stat().st_size,
        "sha256": _sha256(source_video),
        "duration_seconds": _video_duration_seconds(source_video),
        "copied": bool(copy_video),
        "package_relative_path": "source_video.mp4" if copy_video else None,
    }
    _write_json(output_dir / "source_video.json", source_metadata)
    if copy_video:
        shutil.copy2(source_video, output_dir / "source_video.mp4")

    checksums = _build_checksums(output_dir)
    _write_json(output_dir / "checksums.json", {"files": checksums})
    return output_dir


def _copy_session_metadata(session_dir: Path, output_dir: Path) -> None:
    source = session_dir / "session.json"
    if source.exists():
        metadata = _load_json(source)
        _write_json(output_dir / "session.json", _redact_value(metadata))
        return

    config = _load_json(session_dir / "config.json")
    metadata = {
        "session_dir": str(session_dir),
        "software_commit": _git_commit(session_dir),
        "model_identity": config.get("model_path"),
        "event_profile": config.get("event_profile", config.get("sport_profile")),
        "config": config,
    }
    _write_json(output_dir / "session.json", _redact_value(metadata))


def _copy_database(session_dir: Path, output_dir: Path) -> None:
    for name in ("timing.db", "events.db", "race.db"):
        source = session_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / name)


def _copy_evidence(session_dir: Path, output_dir: Path) -> None:
    for name in ("evidence", "evidence_photos"):
        source = session_dir / name
        if source.is_dir():
            shutil.copytree(source, output_dir, dirs_exist_ok=True)
    screenshot_dir = session_dir / "field_issues" / "screenshots"
    if screenshot_dir.is_dir():
        shutil.copytree(screenshot_dir, output_dir / "screenshots", dirs_exist_ok=True)


def _copy_logs(session_dir: Path, output_dir: Path) -> None:
    logs_dir = session_dir / "logs"
    if logs_dir.is_dir():
        for source in sorted(logs_dir.rglob("*")):
            if not source.is_file():
                continue
            destination = output_dir / source.relative_to(logs_dir)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(_redact_text(source.read_text(encoding="utf-8", errors="replace")), encoding="utf-8")
    for source in sorted(session_dir.glob("*.log")):
        (output_dir / source.name).write_text(
            _redact_text(source.read_text(encoding="utf-8", errors="replace")),
            encoding="utf-8",
        )


def _copy_redacted_text(source: Path, destination: Path) -> None:
    if not source.exists():
        destination.write_text("", encoding="utf-8")
        return
    destination.write_text(
        _redact_text(source.read_text(encoding="utf-8", errors="replace")),
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _redact_value(value: Any, key: Optional[str] = None) -> Any:
    if key and any(part in key.lower() for part in _SENSITIVE_KEY_PARTS):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(item_key): _redact_value(item, str(item_key)) for item_key, item in value.items()}
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    return value


def _redact_text(value: str) -> str:
    value = _SENSITIVE_VALUE_PATTERN.sub(r"\1[REDACTED]", value)
    try:
        parsed = urlsplit(value)
        if parsed.scheme and parsed.netloc and (parsed.username or parsed.password):
            host = parsed.hostname or ""
            if parsed.port:
                host = f"{host}:{parsed.port}"
            value = urlunsplit((parsed.scheme, host, parsed.path, parsed.query, parsed.fragment))
    except ValueError:
        pass
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _video_duration_seconds(path: Path) -> Optional[float]:
    try:
        import cv2

        capture = cv2.VideoCapture(str(path))
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
            if fps > 0.0 and frames >= 0.0:
                return frames / fps
        finally:
            capture.release()
    except Exception:
        return None
    return None


def _git_commit(path: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _build_checksums(output_dir: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "checksums.json":
            continue
        relative = path.relative_to(output_dir).as_posix()
        files[relative] = _sha256(path)
    return files


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--copy-video", action="store_true")
    args = parser.parse_args(argv)
    package = export_field_issue_package(
        session_dir=args.session,
        source_video=args.video,
        output_dir=args.output,
        copy_video=args.copy_video,
    )
    print(package)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
