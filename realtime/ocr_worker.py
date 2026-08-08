"""Bounded recognition-only OCR process for saved crossing events."""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import json
import logging
import multiprocessing
import os
import queue
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import cv2
import numpy as np

from .io_utils import read_image_unicode
from .runtime_paths import find_recognition_model_dir


logger = logging.getLogger(__name__)

_STOP = "__VIDEOPIPE_OCR_STOP__"


@dataclass(frozen=True)
class OcrJob:
    event_id: int
    event_dir: str
    candidate_paths: tuple[str, ...]


@dataclass(frozen=True)
class OcrResult:
    event_id: int
    text: str = ""
    confidence: float = 0.0
    status: str = "PENDING"
    source: str = ""
    error: str = ""
    elapsed_ms: float = 0.0
    attempted_candidates: int = 0


def collect_candidate_paths(event_dir: Path, max_candidates: int = 2) -> list[Path]:
    """Return at most two saved bib crops, never full-frame or athlete images."""
    event_dir = Path(event_dir)
    limit = max(1, min(2, int(max_candidates)))
    meta_path = event_dir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        meta = {}

    paths = meta.get("paths") if isinstance(meta, dict) else {}
    paths = paths if isinstance(paths, dict) else {}
    configured: list[str] = []
    candidate_metadata = meta.get("bib_candidate_metadata") if isinstance(meta, dict) else None
    if isinstance(candidate_metadata, list):
        seen_frames = set()
        for item in candidate_metadata:
            if not isinstance(item, dict) or str(item.get("source") or "").lower() != "detected":
                continue
            if item.get("owner_validated") is not True:
                continue
            frame_index = item.get("frame_index")
            try:
                frame_index = int(frame_index)
            except (TypeError, ValueError):
                continue
            if frame_index < 0 or frame_index in seen_frames:
                continue
            athlete_bbox = item.get("athlete_bbox")
            bib_bbox = item.get("bib_bbox")
            if not isinstance(athlete_bbox, list) or len(athlete_bbox) != 4:
                continue
            if not isinstance(bib_bbox, list) or len(bib_bbox) != 4:
                continue
            path_value = item.get("path")
            if not path_value:
                continue
            seen_frames.add(frame_index)
            configured.append(str(path_value))

    # Legacy evidence has no same-frame ownership metadata. It may be inspected,
    # but a single path can never satisfy the automatic confirmation gate.
    if not configured:
        bib_path = paths.get("bib")
        if bib_path:
            configured.append(str(bib_path))
        elif (event_dir / "bib.jpg").is_file():
            configured.append("bib.jpg")

    selected: list[Path] = []
    seen: set[Path] = set()
    for value in configured:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = event_dir / candidate
        candidate = candidate.resolve()
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def parse_recognition_predictions(result: Any) -> list[tuple[str, float]]:
    """Normalize PaddleOCR TextRecognition result shapes."""
    parsed: list[tuple[str, float]] = []
    if result is None:
        return parsed
    if isinstance(result, dict) or not isinstance(result, Iterable):
        result = [result]

    for item in result:
        payload = item
        if not hasattr(payload, "get"):
            payload = getattr(item, "json", None)
            if callable(payload):
                payload = payload()
        if not hasattr(payload, "get"):
            continue

        nested = payload.get("res")
        data = nested if hasattr(nested, "get") else payload
        texts = data.get("rec_texts")
        scores = data.get("rec_scores")
        if texts is not None and scores is not None:
            for text, score in zip(texts, scores):
                normalized = str(text or "").strip()
                if normalized:
                    parsed.append((normalized, float(score or 0.0)))

        text = str(data.get("rec_text") or "").strip()
        if text:
            parsed.append((text, float(data.get("rec_score") or 0.0)))
    return parsed


def _prepare_candidate(image: np.ndarray) -> np.ndarray:
    """Apply one bounded resize so tiny bib crops reach recognizer scale."""
    if image is None or image.size == 0:
        return image
    height, width = image.shape[:2]
    shortest = max(1, min(height, width))
    if shortest >= 48:
        return image
    scale = min(3.0, 48.0 / shortest)
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)


def _best_prediction(predictions: Sequence[tuple[str, float]]) -> tuple[str, float]:
    valid = [
        (str(text or "").strip(), float(confidence or 0.0))
        for text, confidence in predictions
        if str(text or "").strip() and any(ch.isdigit() for ch in str(text))
    ]
    if not valid:
        return "", 0.0
    return max(valid, key=lambda item: item[1])


def recognize_job(
    recognizer: Any,
    job: OcrJob,
    *,
    high_confidence: float = 0.85,
) -> OcrResult:
    """Recognize at most two candidates with one inference per image."""
    started = time.perf_counter()
    observations: list[tuple[str, float, str]] = []
    attempted = 0
    seen_images: set[bytes] = set()

    for path_value in tuple(job.candidate_paths)[:2]:
        candidate_path = Path(path_value)
        image = read_image_unicode(candidate_path)
        if image is None or image.size == 0:
            continue
        image_fingerprint = hashlib.sha256(
            str(image.shape).encode("ascii") + image.dtype.str.encode("ascii") + np.ascontiguousarray(image).tobytes()
        ).digest()
        if image_fingerprint in seen_images:
            continue
        seen_images.add(image_fingerprint)
        attempted += 1
        prepared = _prepare_candidate(image)
        try:
            predictions = recognizer.predict(prepared, batch_size=1)
        except TypeError:
            predictions = recognizer.predict(prepared)
        text, confidence = _best_prediction(parse_recognition_predictions(predictions))
        if text:
            observations.append((text, confidence, candidate_path.name))

    if not observations:
        return OcrResult(
            event_id=int(job.event_id),
            status="PENDING",
            error="OCR_FAILED" if attempted else "NO_VALID_CANDIDATE",
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            attempted_candidates=attempted,
        )

    grouped: dict[str, list[tuple[float, str]]] = {}
    for text, confidence, source in observations:
        grouped.setdefault(text, []).append((confidence, source))
    ranked = sorted(
        grouped.items(),
        key=lambda item: (len(item[1]), max(value[0] for value in item[1])),
        reverse=True,
    )
    best_text, best_values = ranked[0]
    conflict = len(ranked) > 1
    supporting = [value for value in best_values if value[0] >= float(high_confidence)]
    has_independent_consensus = len(supporting) >= 2 and not conflict
    if has_independent_consensus:
        best_confidence = min(value[0] for value in supporting)
        best_source = "+".join(value[1] for value in supporting[:2])
        status = "DONE"
        error = ""
    else:
        best_confidence, best_source = max(best_values, key=lambda item: item[0])
        status = "PENDING"
        if conflict:
            error = "OCR_CONFLICT"
        elif attempted < 2 or len(best_values) < 2:
            error = "INSUFFICIENT_MULTI_FRAME_EVIDENCE"
        else:
            error = "LOW_CONF"
    return OcrResult(
        event_id=int(job.event_id),
        text=best_text,
        confidence=best_confidence,
        status=status,
        source=best_source,
        error=error,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        attempted_candidates=attempted,
    )


def _set_worker_limits(cpu_threads: int) -> None:
    threads = max(1, min(2, int(cpu_threads)))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(threads)
    if os.name != "nt":
        return
    try:
        kernel32 = ctypes.windll.kernel32
        process = kernel32.GetCurrentProcess()
        kernel32.SetPriorityClass(process, 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS
        logical_cpus = os.cpu_count() or 1
        selected = min(threads, logical_cpus)
        first_cpu = max(0, logical_cpus - selected)
        affinity_mask = sum(1 << index for index in range(first_cpu, logical_cpus))
        if affinity_mask:
            kernel32.SetProcessAffinityMask(process, affinity_mask)
    except Exception as exc:
        logger.warning("Unable to apply OCR worker CPU limits: %s", exc)


def _ensure_modelscope_importable() -> None:
    """Provide the import PaddleX requires without bundling ModelScope."""
    try:
        importlib.import_module("modelscope")
        return
    except ModuleNotFoundError as exc:
        if exc.name != "modelscope":
            raise

    module = types.ModuleType("modelscope")

    def snapshot_download(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError(
            "ModelScope downloads are disabled; bundle the OCR model for offline use"
        )

    module.snapshot_download = snapshot_download
    sys.modules["modelscope"] = module


def _create_recognizer(cpu_threads: int, models_root: Optional[str]) -> Any:
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    _ensure_modelscope_importable()
    from paddleocr import TextRecognition

    recognition_dir = find_recognition_model_dir(
        models_root=Path(models_root).resolve() if models_root else None
    )
    kwargs = {
        "model_name": "en_PP-OCRv5_mobile_rec",
        "device": "cpu",
        "cpu_threads": max(1, min(2, int(cpu_threads))),
        "enable_mkldnn": False,
    }
    if recognition_dir:
        kwargs["model_dir"] = str(recognition_dir)
    return TextRecognition(**kwargs)


def _offer_result(result_queue: Any, result: OcrResult) -> None:
    try:
        result_queue.put(result, timeout=0.5)
    except queue.Full:
        logger.error("OCR result queue is full; result for event %s was dropped", result.event_id)


def _worker_main(
    job_queue: Any,
    result_queue: Any,
    cpu_threads: int,
    models_root: Optional[str],
) -> None:
    _set_worker_limits(cpu_threads)
    try:
        recognizer = _create_recognizer(cpu_threads, models_root)
    except Exception as exc:
        _offer_result(
            result_queue,
            OcrResult(event_id=-1, status="FAILED", error=f"OCR_INIT_FAILED: {exc}"),
        )
        return

    _offer_result(result_queue, OcrResult(event_id=-1, status="READY"))
    while True:
        item = job_queue.get()
        if item == _STOP:
            return
        if not isinstance(item, OcrJob):
            continue
        try:
            result = recognize_job(recognizer, item)
        except Exception as exc:
            result = OcrResult(
                event_id=int(item.event_id),
                status="PENDING",
                error=f"OCR_WORKER_ERROR: {exc}",
            )
        _offer_result(result_queue, result)


class OcrProcessController:
    """Own one low-priority OCR child and a single-slot heavy-work queue."""

    def __init__(
        self,
        *,
        cpu_threads: int = 1,
        models_root: Optional[str] = None,
        job_queue: Any = None,
        result_queue: Any = None,
        process_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.cpu_threads = max(1, min(2, int(cpu_threads)))
        self.models_root = str(models_root) if models_root else None
        self._context = multiprocessing.get_context("spawn")
        self.job_queue = job_queue or self._context.Queue(maxsize=1)
        self.result_queue = result_queue or self._context.Queue(maxsize=8)
        self._process_factory = process_factory or self._context.Process
        self._process = None
        self._disabled = False

    @property
    def disabled(self) -> bool:
        return self._disabled

    @property
    def is_alive(self) -> bool:
        return bool(self._process is not None and self._process.is_alive())

    def start(self) -> bool:
        if self._disabled:
            return False
        if self.is_alive:
            return True
        if self._process is not None:
            try:
                self._process.close()
            except (AttributeError, ValueError):
                pass
            self._process = None
        self._process = self._process_factory(
            target=_worker_main,
            args=(self.job_queue, self.result_queue, self.cpu_threads, self.models_root),
            name="VideoPipe-OCR",
            daemon=True,
        )
        if self._process is None:
            return False
        self._process.start()
        return True

    def restart(self) -> bool:
        """Recreate queues so a stale sentinel or job cannot kill the new child."""
        self.stop()
        self.job_queue = self._context.Queue(maxsize=1)
        self.result_queue = self._context.Queue(maxsize=8)
        return self.start()

    def submit(self, job: OcrJob) -> bool:
        if self._disabled:
            return False
        try:
            self.job_queue.put_nowait(job)
        except queue.Full:
            return False
        return True

    def poll(self, max_results: int = 8) -> list[OcrResult]:
        results: list[OcrResult] = []
        for _ in range(max(1, int(max_results))):
            try:
                item = self.result_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, OcrResult):
                results.append(item)
        return results

    def disable(self) -> None:
        self._disabled = True
        self.stop()

    def stop(self, timeout: float = 2.0) -> None:
        process = self._process
        if process is None:
            return
        if process.is_alive():
            try:
                self.job_queue.put_nowait(_STOP)
            except queue.Full:
                try:
                    self.job_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.job_queue.put_nowait(_STOP)
                except queue.Full:
                    pass
            process.join(timeout=max(0.0, float(timeout)))
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
        try:
            process.close()
        except (AttributeError, ValueError):
            pass
        self._process = None
