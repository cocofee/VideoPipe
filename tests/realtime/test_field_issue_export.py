import json

import pytest

from tools.export_field_issue_package import export_field_issue_package


def test_export_contains_reproduction_inputs(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "issues.jsonl").write_text(
        json.dumps({"issue_id": "I1", "capture_time_ms": 10_000.0}) + "\n",
        encoding="utf-8",
    )
    (session_dir / "session.json").write_text(
        json.dumps({"software_commit": "test"}),
        encoding="utf-8",
    )
    (session_dir / "timing.db").write_bytes(b"database")
    (session_dir / "evidence_photos").mkdir()
    (session_dir / "evidence_photos" / "event-1.jpg").write_bytes(b"evidence")
    (session_dir / "field_issues" / "screenshots").mkdir(parents=True)
    (session_dir / "field_issues" / "screenshots" / "issue-1.jpg").write_bytes(
        b"issue-screenshot"
    )
    source_video = tmp_path / "race.mp4"
    source_video.write_bytes(b"video-placeholder")

    package = export_field_issue_package(
        session_dir=session_dir,
        source_video=source_video,
        output_dir=tmp_path / "package",
    )

    assert (package / "session.json").exists()
    assert (package / "issues.jsonl").exists()
    assert (package / "checksums.json").exists()
    assert (package / "evidence" / "event-1.jpg").exists()
    assert (package / "evidence" / "screenshots" / "issue-1.jpg").exists()
    assert (package / "database" / "timing.db").read_bytes() == b"database"
    assert not (package / "source_video.mp4").exists()

    source_meta = json.loads((package / "source_video.json").read_text(encoding="utf-8"))
    assert source_meta["path"] == str(source_video.resolve())
    assert source_meta["size_bytes"] == len(b"video-placeholder")
    assert source_meta["sha256"]

    checksums = json.loads((package / "checksums.json").read_text(encoding="utf-8"))
    assert checksums["files"]["session.json"]
    assert checksums["files"]["evidence/event-1.jpg"]


def test_export_can_copy_video_only_when_explicitly_requested(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    source_video = tmp_path / "race.mp4"
    source_video.write_bytes(b"video-placeholder")

    package = export_field_issue_package(
        session_dir=session_dir,
        source_video=source_video,
        output_dir=tmp_path / "package",
        copy_video=True,
    )

    assert (package / "source_video.mp4").read_bytes() == b"video-placeholder"


def test_export_redacts_credentials_from_session_metadata(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "session.json").write_text(
        json.dumps(
            {
                "api_key": "secret-api-key",
                "source": "rtsp://camera-user:camera-password@192.0.2.10/live",
                "nested": {"token": "secret-token"},
            }
        ),
        encoding="utf-8",
    )
    source_video = tmp_path / "race.mp4"
    source_video.write_bytes(b"video-placeholder")

    package = export_field_issue_package(
        session_dir=session_dir,
        source_video=source_video,
        output_dir=tmp_path / "package",
    )

    exported = (package / "session.json").read_text(encoding="utf-8")
    assert "secret-api-key" not in exported
    assert "camera-password" not in exported
    assert "secret-token" not in exported
    assert "[REDACTED]" in exported


def test_export_redacts_credentials_from_issue_log_and_runtime_logs(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "issues.jsonl").write_text(
        json.dumps(
            {
                "issue_id": "I1",
                "operator_note": "authorization=secret-issue-token",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (session_dir / "runtime.log").write_text(
        "api_key=secret-log-key\n",
        encoding="utf-8",
    )
    source_video = tmp_path / "race.mp4"
    source_video.write_bytes(b"video-placeholder")

    package = export_field_issue_package(
        session_dir=session_dir,
        source_video=source_video,
        output_dir=tmp_path / "package",
    )

    issue_log = (package / "issues.jsonl").read_text(encoding="utf-8")
    runtime_log = (package / "logs" / "runtime.log").read_text(encoding="utf-8")
    assert "secret-issue-token" not in issue_log
    assert "secret-log-key" not in runtime_log
    assert "[REDACTED]" in issue_log
    assert "[REDACTED]" in runtime_log


def test_export_rejects_a_nonempty_output_directory(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    source_video = tmp_path / "race.mp4"
    source_video.write_bytes(b"video-placeholder")
    output_dir = tmp_path / "package"
    output_dir.mkdir()
    (output_dir / "stale.txt").write_text("old package", encoding="utf-8")

    with pytest.raises(FileExistsError, match="must be empty"):
        export_field_issue_package(
            session_dir=session_dir,
            source_video=source_video,
            output_dir=output_dir,
        )

    assert (output_dir / "stale.txt").read_text(encoding="utf-8") == "old package"
