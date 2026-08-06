#!/usr/bin/env python

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from realtime.devtools.regression_report import (  # noqa: E402
    aggregate_case_reports,
    load_competition_manifest,
    validate_report_output_path,
)


CaseRunner = Callable[[Dict[str, Any], Dict[str, Path], Path], Dict[str, Any]]


def _resolve_env_path(
    case: Dict[str, Any],
    key: str,
    environ: Mapping[str, str],
    *,
    required: bool,
) -> Optional[Path]:
    env_name = str(case.get(key) or "").strip()
    if not env_name:
        if required:
            raise ValueError(f"Manifest case requires {key}")
        return None
    raw_path = str(environ.get(env_name) or "").strip()
    if not raw_path:
        raise ValueError(f"Environment variable {env_name} is not set")
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"Environment variable {env_name} points to a missing path: {path}")
    return path


def resolve_case_paths(
    case: Dict[str, Any],
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, Path]:
    environment = os.environ if environ is None else environ
    paths = {
        "video": _resolve_env_path(case, "video_path_env", environment, required=True),
        "model": _resolve_env_path(case, "model_path_env", environment, required=True),
    }
    for name, key in (
        ("validator_model", "validator_model_path_env"),
        ("config", "config_path_env"),
    ):
        path = _resolve_env_path(case, key, environment, required=False)
        if path is not None:
            paths[name] = path
    return paths


def _run_case_subprocess(
    case: Dict[str, Any],
    paths: Dict[str, Path],
    report_path: Path,
) -> Dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "realtime.devtools.regression_report",
        "--video",
        str(paths["video"]),
        "--model",
        str(paths["model"]),
        "--out",
        str(report_path),
        "--event-profile-json",
        json.dumps(case["event_profile"], ensure_ascii=False),
    ]
    if "validator_model" in paths:
        command.extend(("--validator-model", str(paths["validator_model"])))
    if "config" in paths:
        command.extend(("--config", str(paths["config"])))
    if case.get("frames") is not None:
        command.extend(("--frames", str(int(case["frames"]))))
    if case.get("start_frame") is not None:
        command.extend(("--start-frame", str(int(case["start_frame"]))))
    if case.get("require_evidence"):
        command.extend(("--evidence-dir", str(report_path.parent / f"{case['case_id']}_evidence")))

    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "unknown error").strip()
        raise RuntimeError(f"Regression case {case['case_id']} failed: {detail}")
    with open(report_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def run_manifest(
    manifest_path: Path,
    output_dir: Path,
    *,
    environ: Optional[Mapping[str, str]] = None,
    runner: Optional[CaseRunner] = None,
) -> Dict[str, Any]:
    manifest = load_competition_manifest(Path(manifest_path))
    aggregate_path = validate_report_output_path(Path(output_dir) / "aggregate_report.json")
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    case_runner = runner or _run_case_subprocess

    case_reports = []
    for case in manifest["cases"]:
        paths = resolve_case_paths(case, environ=environ)
        report_path = aggregate_path.parent / f"{case['case_id']}.json"
        report = dict(case_runner(case, paths, report_path))
        report["case_id"] = case["case_id"]
        report["manifest_case"] = case
        case_reports.append(report)

    aggregate = aggregate_case_reports(case_reports)
    temp_path = aggregate_path.with_suffix(".json.tmp")
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, ensure_ascii=False, indent=2)
    temp_path.replace(aggregate_path)
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all configured VideoPipe competition regressions")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    try:
        aggregate = run_manifest(args.manifest, args.output)
    except Exception as exc:
        print(f"[FAIL] {exc}")
        return 2

    print(f"Aggregate report: {(args.output / 'aggregate_report.json').resolve()}")
    print(f"Cases: {len(aggregate['cases'])}")
    print(f"Validation passed: {aggregate['validation']['passed']}")
    for issue in aggregate["validation"]["issues"]:
        print(f"- {issue}")
    return 0 if aggregate["validation"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
