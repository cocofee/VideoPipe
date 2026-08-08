import os
import json
import time
import threading
import queue
import logging
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

from .detector import PaddleOcrAdapter, BibStatus
from .database import Database
from .io_utils import read_image_unicode
from .ocr_worker import OcrJob, OcrProcessController, OcrResult, collect_candidate_paths

logger = logging.getLogger(__name__)


_EVENT_VLM_PROMPT = """You are reading a cycling race bib number.
The first {candidate_count} image(s) are owner-validated bib crops from the same athlete.
The last image, when present, is the athlete crop for context only.

Rules:
1. Read only the large race number attached to the athlete.
2. Ignore jersey text, sponsor text, logos, clocks, and people in the background.
3. Use all bib crops together; do not guess a hidden leading or trailing digit.
4. Allowed format: {allowed_format}.
5. If the number is not readable, return UNREADABLE.
{roster_hint}

Return exactly three lines:
line 1: bib number or UNREADABLE
line 2: confidence from 0.0 to 1.0
line 3: a short reason
"""


@dataclass
class ParticipantOcrState:
    """Participant-keyed OCR evidence while retaining raw-track audit data."""

    participant_id: str
    raw_track_ids: set[int] = field(default_factory=set)
    votes: list[tuple[str, float]] = field(default_factory=list)
    best_candidate: Optional[str] = None
    best_confidence: float = 0.0
    status: str = "PENDING"

    def add_vote(self, raw_track_id: int, text: str, confidence: float) -> None:
        normalized = str(text or "").strip().upper()
        if not normalized:
            return

        self.raw_track_ids.add(int(raw_track_id))
        self.votes.append((normalized, float(confidence)))
        scores = defaultdict(float)
        confidences = defaultdict(float)
        for candidate, score in self.votes:
            scores[candidate] += score
            confidences[candidate] = max(confidences[candidate], score)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        self.best_candidate = ranked[0][0]
        self.best_confidence = confidences[self.best_candidate]
        if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < 0.25:
            self.status = "CONFLICT"
        else:
            self.status = "RECOGNIZED"


@dataclass(frozen=True)
class EventVlmResult:
    event_id: int
    text: str = ""
    confidence: float = 0.0
    note: str = ""
    error: str = ""
    source: str = ""
    candidate_count: int = 0

class OCRManager:
    """
    OCR 管理器：负责从磁盘加载证据、执行批处理识别、断点续传及归并逻辑。
    """
    def __init__(
        self,
        database: Database,
        ocr_engine: Optional[PaddleOcrAdapter] = None,
        vlm_engine: Optional[Any] = None,
        process_controller: Optional[OcrProcessController] = None,
    ):
        self.db = database
        self.ocr = ocr_engine
        self.vlm = vlm_engine
        # VLM OCR 策略：
        # - fallback: 仅在本地 OCR 失败/很低置信度时兜底（默认，最省钱/最稳）
        # - first: 优先走 VLM（不行再本地）
        # - only: 只要有 VLM 就只走 VLM（失败才退回本地，避免全失败）
        self.vlm_mode = "fallback"

        # OCR 号码规则（从 DB 同步）
        # - 自行车：常见 1~500（可能 1~3 位数字），需要允许短纯数字
        # - 马拉松：常见 A0001-F9999（字母+4位数字），短数字通常是误识别，应过滤
        self.only_numeric = False
        self._bib_ranges = []
        self._refresh_rules_from_db()
        
        # 任务队列
        self.task_queue = queue.PriorityQueue()
        self._running = False
        self._worker_thread = None
        self._queue_lock = threading.Lock()
        self._queued_event_dirs = set()
        self.process_controller = process_controller
        self.runtime_state = "idle"
        self._pending_process_jobs = deque(maxlen=64)
        self._queued_process_event_ids: set[int] = set()
        self._process_restart_count = 0
        self._vlm_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="videopipe-event-vlm",
        )
        self._vlm_futures: Dict[Future, dict] = {}
        self._vlm_pending_event_ids: set[int] = set()
        self._vlm_call_times = deque(maxlen=1000)
        self._vlm_lock = threading.Lock()
        self.vlm_max_calls_per_minute = 30
        self.vlm_auto_confirm_threshold = 0.72
        self.vlm_agreement_threshold = 0.58
        self.vlm_single_frame_threshold = 0.84
        self.vlm_conflict_threshold = 0.88
        self.vlm_max_pending_jobs = 64
        # Participant-keyed consensus is additive; event/track APIs remain intact.
        self._participant_ocr_states: Dict[str, ParticipantOcrState] = {}
        self._participant_ocr_lock = threading.Lock()
        
        # 统计
        self.stats = {
            "total": 0,
            "pending": 0,
            "done": 0,
            "resolved_done": 0,
            "pending_result": 0,
            "processed": 0,
            "failed": 0,
            "merged": 0,
            "vlm_used": 0,
            "vlm_pending": 0,
        }
        
        self.merge_window_seconds = 60.0
        self.conf_threshold = 0.4
        self.bbox_expand_ratio = 0.1
        self.vlm_min_size = 14
        self.throttle_live_window_seconds = 3.0
        # CPU OCR 节流参数：OCR 在 CPU 上运行（~5-10s/次），不影响 GPU 上的 YOLO
        # 仅需少量 sleep 让出 CPU 给 UI 线程和流读取线程
        self.throttle_sleep_idle_seconds = 0.1    # 空闲：事件间隔短，尽快跑完
        self.throttle_sleep_live_seconds = 0.2    # 实时：稍让一下CPU给流处理
        self.throttle_pause_every = 10            # 每10个事件额外暂停一次
        self.throttle_pause_idle_seconds = 0.5    # 空闲额外暂停
        self.throttle_pause_live_seconds = 1.0    # 实时额外暂停
        
        # 回调
        self.on_progress = None # (current, total, stats)
        self.on_event_done = None # (event_id, result_dict)

    def _normalize_bib_text(self, text: Any) -> str:
        """将 OCR 文本规范化为可用号码；不合规则返回空串。"""
        if text is None:
            return ""
        raw = str(text).strip().upper()
        if not raw:
            return ""

        cleaned = ''.join(ch for ch in raw if ch.isalnum())
        if len(cleaned) > 10:
            return ""

        # 号码必须包含数字，避免把背景词/单字母误当号码
        if not any(ch.isdigit() for ch in cleaned):
            return ""

        import re

        # 自行车“仅限纯数字”模式：允许 1~6 位数字（支持 1~500 这种短号）
        if self.only_numeric:
            if not cleaned.isdigit():
                return ""
            if not re.match(r'^[0-9]{1,6}$', cleaned):
                return ""
            return cleaned

        # 非纯数字模式：可选字母前缀 + 3~6 位数字（避免把 1~2 位噪声当号码）
        # 例：A0559, 012849
        if not re.match(r'^[A-Z]{0,3}[0-9]{3,6}$', cleaned):
            return ""

        # 号码范围过滤（关键）：如果配置了 bib_ranges，只接受范围内的号码
        # 这能过滤掉 "183378"（计时器）、"AS477"（广告牌）等背景垃圾文字
        if self._bib_ranges:
            if self._is_in_bib_ranges(cleaned):
                return cleaned
            # OCR 经常丢失字母前缀（如识别出 "2851" 而非 "A2851"）
            # 尝试自动补全：如果所有规则共享同一个单字母前缀，补上再检查
            if cleaned[0].isdigit():
                prefixes = set(r['prefix'] for r in self._bib_ranges if r.get('type') == 'range' and r.get('prefix'))
                if len(prefixes) == 1:
                    guess_prefix = prefixes.pop()
                    candidate = guess_prefix + cleaned
                    if self._is_in_bib_ranges(candidate):
                        return candidate
                else:
                    # 多个前缀：逐个尝试
                    for p in sorted(prefixes):
                        candidate = p + cleaned
                        if self._is_in_bib_ranges(candidate):
                            return candidate
            return ""

        return cleaned

    def _refresh_rules_from_db(self):
        """从数据库读取号码规则开关（容错、低成本）。"""
        try:
            self.only_numeric = str(self.db.get_config("numeric_only", "1")) == "1"
        except Exception:
            pass
        # 读取号码范围（关键：过滤背景垃圾文字）
        try:
            ranges_str = str(self.db.get_config("bib_ranges", "") or "")
            self._parse_bib_ranges(ranges_str)
        except Exception:
            self._bib_ranges = []

    def _parse_bib_ranges(self, range_str: str):
        """解析号码范围字符串，如 'A0001-A9999, B001-B500, A0001-B9999'

        支持三种格式：
        1. 同前缀范围：A0001-A9999 → prefix=A, 1~9999
        2. 跨前缀范围：A0001-B9999 → 展开为 A0001-A9999 + B0001-B9999
        3. 精确匹配：A1234 → 只匹配这一个号码
        """
        self._bib_ranges = []
        if not range_str:
            return
        import re
        parts = [p.strip().upper() for p in range_str.split(',') if p.strip()]
        for part in parts:
            if '-' in part:
                try:
                    start_str, end_str = part.split('-', 1)
                    start_match = re.match(r'^([A-Z]*)([0-9]+)$', start_str.strip())
                    end_match = re.match(r'^([A-Z]*)([0-9]+)$', end_str.strip())
                    if start_match and end_match:
                        s_prefix = start_match.group(1)
                        e_prefix = end_match.group(1)
                        start_num = int(start_match.group(2))
                        end_num = int(end_match.group(2))
                        if s_prefix == e_prefix:
                            # 同前缀：A0001-A9999
                            self._bib_ranges.append({
                                'type': 'range', 'prefix': s_prefix,
                                'start': min(start_num, end_num),
                                'end': max(start_num, end_num)
                            })
                        elif len(s_prefix) == 1 and len(e_prefix) == 1:
                            # 跨前缀：A0001-B9999 → 展开为每个字母一条规则
                            for ch_code in range(ord(s_prefix), ord(e_prefix) + 1):
                                p = chr(ch_code)
                                r_start = start_num if p == s_prefix else 1
                                r_end = end_num if p == e_prefix else 9999
                                self._bib_ranges.append({
                                    'type': 'range', 'prefix': p,
                                    'start': r_start, 'end': r_end
                                })
                        else:
                            logger.warning(f"[OCRManager] 无法解析号码范围: {part}")
                except Exception:
                    continue
            else:
                self._bib_ranges.append({'type': 'exact', 'value': part})
        if self._bib_ranges:
            for r in self._bib_ranges:
                if r['type'] == 'range':
                    logger.info(f"[OCRManager] 号码范围: {r['prefix']}{r['start']:04d} ~ {r['prefix']}{r['end']:04d}")
                else:
                    logger.info(f"[OCRManager] 精确匹配: {r['value']}")
            logger.info(f"[OCRManager] 共加载 {len(self._bib_ranges)} 条号码段规则")
        else:
            logger.warning(f"[OCRManager] 号码范围字符串 '{range_str}' 解析后为空，不会进行范围过滤！")

    def _is_in_bib_ranges(self, bib: str) -> bool:
        """检查号码是否在合法范围内"""
        if not self._bib_ranges:
            return True  # 未配置范围时，不过滤
        import re
        bib = str(bib).strip().upper()
        m = re.match(r'^([A-Z]*)([0-9]+)$', bib)
        if not m:
            return False
        bib_prefix = m.group(1)
        bib_num = int(m.group(2))
        for rule in self._bib_ranges:
            if rule['type'] == 'range':
                if bib_prefix == rule['prefix'] and rule['start'] <= bib_num <= rule['end']:
                    return True
            elif rule['type'] == 'exact':
                if bib == rule['value']:
                    return True
        return False

    def get_participant_ocr_state(self, participant_id: Optional[str], create: bool = True) -> Optional[ParticipantOcrState]:
        """Return participant OCR state without making identity decisions."""
        key = str(participant_id or "").strip()
        if not key:
            return None
        with self._participant_ocr_lock:
            state = self._participant_ocr_states.get(key)
            if state is None and create:
                state = ParticipantOcrState(participant_id=key)
                self._participant_ocr_states[key] = state
            return state

    def _add_participant_ocr_vote(
        self,
        participant_id: Optional[str],
        raw_track_id: Optional[int],
        text: Any,
        confidence: float,
    ) -> Optional[ParticipantOcrState]:
        """Normalize and roster-validate a candidate before participant aggregation."""
        if not participant_id or raw_track_id is None:
            return None
        normalized = self._normalize_bib_text(text)
        if not normalized or not self._is_in_bib_ranges(normalized):
            return None
        state = self.get_participant_ocr_state(participant_id)
        if state is None:
            return None
        with self._participant_ocr_lock:
            state.add_vote(raw_track_id=int(raw_track_id), text=normalized, confidence=float(confidence or 0.0))
        return state

    def set_ocr(self, ocr_engine: PaddleOcrAdapter, vlm_engine: Optional[Any] = None):
        """动态设置识别引擎"""
        self.ocr = ocr_engine
        if vlm_engine:
            self.vlm = vlm_engine

    def start_process_runtime(
        self,
        *,
        cpu_threads: int = 1,
        models_root: Optional[str] = None,
    ) -> bool:
        """Start the recognition-only child without importing Paddle in this process."""
        if self.process_controller is None:
            self.process_controller = OcrProcessController(
                cpu_threads=cpu_threads,
                models_root=models_root,
            )
        if self.process_controller.disabled:
            self.runtime_state = "disabled"
            return False
        started = bool(self.process_controller.start())
        self.runtime_state = "loading" if started else "failed"
        return started

    def _dispatch_pending_process_job(self) -> None:
        if self.runtime_state != "ready" or self.process_controller is None:
            return
        while self._pending_process_jobs:
            job = self._pending_process_jobs[0]
            if not self.process_controller.submit(job):
                return
            self._pending_process_jobs.popleft()
            return

    def poll_process_results(self) -> List[OcrResult]:
        """Drain small child-process messages and apply event results in this process."""
        results = self.process_controller.poll() if self.process_controller is not None else []
        for result in results:
            if int(result.event_id) < 0:
                normalized_state = str(result.status or "").strip().lower()
                if normalized_state == "ready":
                    self.runtime_state = "ready"
                    self._dispatch_pending_process_job()
                elif normalized_state == "failed":
                    self.runtime_state = "failed"
                    logger.error("[OCRManager] OCR process initialization failed: %s", result.error)
                continue

            self._queued_process_event_ids.discard(int(result.event_id))
            if self.stats.get("pending", 0) > 0:
                self.stats["pending"] -= 1
            self._apply_process_result(result)
            self._dispatch_pending_process_job()
        self._poll_event_vlm_results()
        if (
            self.process_controller is not None
            and self.runtime_state in {"loading", "ready", "failed"}
            and not self.process_controller.is_alive
        ):
            self._recover_process_runtime()
        return results

    def _collect_owned_vlm_paths(
        self,
        event_dir: Path,
        meta: dict,
    ) -> tuple[tuple[Path, ...], Optional[Path]]:
        metadata = meta.get("bib_candidate_metadata") if isinstance(meta, dict) else None
        if not isinstance(metadata, list):
            return (), None

        candidates = []
        seen_frames = set()
        for item in metadata:
            if not isinstance(item, dict):
                continue
            if str(item.get("source") or "").strip().lower() != "detected":
                continue
            if item.get("owner_validated") is not True:
                continue
            try:
                frame_index = int(item.get("frame_index"))
            except (TypeError, ValueError):
                continue
            if frame_index < 0 or frame_index in seen_frames:
                continue
            path_value = item.get("path")
            if not path_value:
                continue
            candidate_path = Path(str(path_value))
            if not candidate_path.is_absolute():
                candidate_path = event_dir / candidate_path
            if not candidate_path.is_file():
                continue
            seen_frames.add(frame_index)
            candidates.append(candidate_path.resolve())
            if len(candidates) >= 2:
                break

        if not candidates:
            return (), None

        paths = meta.get("paths") if isinstance(meta, dict) else None
        paths = paths if isinstance(paths, dict) else {}
        athlete_value = paths.get("athlete") or "athlete.jpg"
        athlete_path = Path(str(athlete_value))
        if not athlete_path.is_absolute():
            athlete_path = event_dir / athlete_path
        if athlete_path.is_file():
            athlete_path = athlete_path.resolve()
        else:
            athlete_path = None
        return tuple(candidates), athlete_path

    def _get_roster_bibs(self) -> list[str]:
        get_all = getattr(self.db, "get_all_athletes", None)
        if not callable(get_all):
            return []
        try:
            athletes = get_all() or []
        except Exception:
            return []
        bibs = []
        for athlete in athletes:
            if not isinstance(athlete, dict):
                continue
            bib = str(athlete.get("bib_number") or "").strip().upper()
            if bib and bib not in bibs:
                bibs.append(bib)
            if len(bibs) >= 200:
                break
        return bibs

    def _build_event_vlm_prompt(self, candidate_count: int, roster_bibs: list[str]) -> str:
        allowed_format = (
            "1 to 6 digits"
            if self.only_numeric
            else "an optional letter prefix and 3 to 6 digits"
        )
        roster_hint = ""
        if roster_bibs:
            roster_hint = (
                "Valid roster bibs (reference only; never invent a match): "
                + ", ".join(roster_bibs)
            )
        return _EVENT_VLM_PROMPT.format(
            candidate_count=max(1, int(candidate_count)),
            allowed_format=allowed_format,
            roster_hint=roster_hint,
        )

    @staticmethod
    def _prepare_vlm_crop(image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return image
        height, width = image.shape[:2]
        shortest = max(1, min(height, width))
        if shortest >= 160:
            return image
        scale = min(6.0, 160.0 / shortest)
        target_width = max(1, int(round(width * scale)))
        target_height = max(1, int(round(height * scale)))
        return cv2.resize(
            image,
            (target_width, target_height),
            interpolation=cv2.INTER_CUBIC,
        )

    def _run_event_vlm_job(
        self,
        *,
        event_id: int,
        candidate_paths: tuple[Path, ...],
        athlete_path: Optional[Path],
        roster_bibs: list[str],
        vlm_engine: Any,
    ) -> EventVlmResult:
        images = []
        for candidate_path in candidate_paths:
            image = read_image_unicode(candidate_path)
            if image is None or image.size == 0:
                continue
            images.append(self._prepare_vlm_crop(image))
        candidate_count = len(images)
        if candidate_count == 0:
            return EventVlmResult(
                event_id=event_id,
                error="VLM_NO_VALID_CANDIDATE",
            )

        if athlete_path is not None:
            athlete_image = read_image_unicode(athlete_path)
            if athlete_image is not None and athlete_image.size:
                images.append(athlete_image)

        prompt = self._build_event_vlm_prompt(candidate_count, roster_bibs)
        try:
            text, confidence, note = vlm_engine.ocr_bib(
                images,
                athlete_list=roster_bibs,
                only_numeric=self.only_numeric,
                prompt=prompt,
            )
        except Exception as exc:
            logger.warning("[OCRManager] Event VLM failed for %s: %s", event_id, exc)
            return EventVlmResult(
                event_id=event_id,
                error="VLM_REQUEST_FAILED",
                source=type(vlm_engine).__name__,
                candidate_count=candidate_count,
            )
        return EventVlmResult(
            event_id=event_id,
            text=str(text or "").strip(),
            confidence=max(0.0, min(1.0, float(confidence or 0.0))),
            note=str(note or "").strip(),
            error="" if text else "VLM_UNREADABLE",
            source=type(vlm_engine).__name__,
            candidate_count=candidate_count,
        )

    def _schedule_event_vlm(
        self,
        *,
        event: dict,
        event_dir: Path,
        meta: dict,
        local_text: str,
        local_confidence: float,
        local_error: str,
    ) -> bool:
        vlm_engine = self.vlm
        if vlm_engine is None:
            return False
        event_id = int(event.get("event_id") or meta.get("event_id") or 0)
        if event_id <= 0 or event_id in self._vlm_pending_event_ids:
            return False
        candidate_paths, athlete_path = self._collect_owned_vlm_paths(event_dir, meta)
        if not candidate_paths:
            return False

        now = time.monotonic()
        with self._vlm_lock:
            while self._vlm_call_times and now - self._vlm_call_times[0] >= 60.0:
                self._vlm_call_times.popleft()
            if len(self._vlm_call_times) >= max(1, int(self.vlm_max_calls_per_minute)):
                logger.warning(
                    "[OCRManager] Event VLM rate limit reached; event %s remains PENDING",
                    event_id,
                )
                return False
            if len(self._vlm_futures) >= max(1, int(self.vlm_max_pending_jobs)):
                logger.warning(
                    "[OCRManager] Event VLM queue full; event %s remains PENDING",
                    event_id,
                )
                return False
            self._vlm_call_times.append(now)

        context = {
            "event_id": event_id,
            "event_dir": event_dir,
            "meta": dict(meta),
            "local_text": str(local_text or "").strip(),
            "local_confidence": float(local_confidence or 0.0),
            "local_error": str(local_error or ""),
        }
        future = self._vlm_executor.submit(
            self._run_event_vlm_job,
            event_id=event_id,
            candidate_paths=candidate_paths,
            athlete_path=athlete_path,
            roster_bibs=self._get_roster_bibs(),
            vlm_engine=vlm_engine,
        )
        self._vlm_futures[future] = context
        self._vlm_pending_event_ids.add(event_id)
        self.stats["vlm_pending"] = int(self.stats.get("vlm_pending", 0)) + 1
        return True

    def _poll_event_vlm_results(self) -> None:
        completed = [future for future in self._vlm_futures if future.done()]
        for future in completed:
            context = self._vlm_futures.pop(future)
            event_id = int(context["event_id"])
            self._vlm_pending_event_ids.discard(event_id)
            if self.stats.get("vlm_pending", 0) > 0:
                self.stats["vlm_pending"] -= 1
            try:
                result = future.result()
            except Exception as exc:
                logger.warning(
                    "[OCRManager] Event VLM worker failed for %s: %s",
                    event_id,
                    exc,
                )
                result = EventVlmResult(
                    event_id=event_id,
                    error="VLM_WORKER_FAILED",
                )
            self._apply_event_vlm_result(context, result)

    def _apply_event_vlm_result(self, context: dict, result: EventVlmResult) -> None:
        event_id = int(context["event_id"])
        event = self.db.get_event(event_id)
        if not event or int(event.get("manual_corrected") or 0) != 0:
            return
        event_dir = Path(context["event_dir"])
        if not event_dir.is_dir():
            return
        meta = dict(context["meta"])
        meta["event_id"] = event_id

        normalized = self._normalize_bib_text(result.text)
        local_text = self._normalize_bib_text(context.get("local_text"))
        local_confidence = float(context.get("local_confidence") or 0.0)
        confidence = float(result.confidence or 0.0)
        agrees_with_local = bool(normalized and local_text and normalized == local_text)
        strong_local_conflict = bool(
            normalized
            and local_text
            and normalized != local_text
            and len(local_text) >= 2
            and local_confidence >= 0.75
        )

        required_confidence = float(self.vlm_auto_confirm_threshold)
        if agrees_with_local:
            required_confidence = float(self.vlm_agreement_threshold)
        elif strong_local_conflict:
            required_confidence = float(self.vlm_conflict_threshold)
        if int(result.candidate_count or 0) < 2:
            required_confidence = max(
                required_confidence,
                float(self.vlm_single_frame_threshold),
            )

        participant_state = None
        if normalized:
            participant_state = self._add_participant_ocr_vote(
                event.get("participant_id"),
                event.get("track_id"),
                normalized,
                confidence,
            )
        participant_conflict = bool(
            participant_state and participant_state.status == "CONFLICT"
        )
        accepted = bool(
            normalized
            and not result.error
            and confidence >= required_confidence
            and not participant_conflict
        )
        status = "DONE" if accepted else "PENDING"
        if accepted:
            error = ""
        elif participant_conflict:
            error = "PARTICIPANT_OCR_CONFLICT"
        elif result.error:
            error = result.error
        elif normalized:
            error = "VLM_LOW_CONF"
        else:
            error = "VLM_UNREADABLE"

        output_text = normalized or local_text or str(result.text or "").strip()
        output_confidence = confidence if normalized else local_confidence
        source = f"event_vlm:{result.source or 'unknown'}"
        result_data = self._save_result(
            event_dir,
            meta,
            output_text,
            output_confidence,
            status,
            error=error,
            source=source,
            image_source="owned_bib_candidates+athlete",
            participant_id=event.get("participant_id"),
            raw_track_id=event.get("track_id"),
            participant_status=(participant_state.status if participant_state else "PENDING"),
        )
        notes = f"Source: {source}"
        if result.note:
            notes += f"; note={result.note[:120]}"
        self._persist_ocr_resolution(
            event_id=event_id,
            bib=normalized,
            conf=output_confidence,
            status=status,
            evidence_dir=str(event_dir),
            notes=notes,
        )
        self.stats["vlm_used"] = int(self.stats.get("vlm_used", 0)) + 1
        if status == "DONE":
            self.stats["done"] = int(self.stats.get("done", 0)) + 1
            self.stats["resolved_done"] = int(self.stats.get("resolved_done", 0)) + 1
        else:
            self.stats["pending_result"] = int(self.stats.get("pending_result", 0)) + 1
        if self.on_event_done:
            self.on_event_done(event_id, result_data)

    def _recover_process_runtime(self) -> bool:
        if self.process_controller is None:
            self.runtime_state = "disabled"
            return False
        if self._process_restart_count >= 1:
            logger.error("[OCRManager] OCR process failed again; disabling OCR for this run")
            disable = getattr(self.process_controller, "disable", None)
            if callable(disable):
                disable()
            else:
                self.process_controller.stop(timeout=1.0)
            self.runtime_state = "disabled"
            return False

        self._process_restart_count += 1
        logger.warning("[OCRManager] OCR process exited; restarting once")
        restart = getattr(self.process_controller, "restart", None)
        restarted = bool(restart()) if callable(restart) else False
        self.runtime_state = "loading" if restarted else "disabled"
        return restarted

    def _apply_process_result(self, result: OcrResult) -> None:
        event_id = int(result.event_id)
        event = self.db.get_event(event_id)
        if not event:
            return
        if int(event.get("manual_corrected") or 0) != 0:
            logger.info("[OCRManager] Skip async OCR for manually corrected event %s", event_id)
            return

        event_dir = Path(str(event.get("evidence_dir") or ""))
        if not event_dir.is_dir():
            return
        meta_path = event_dir / "meta.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            meta = {"event_id": event_id}
        meta["event_id"] = event_id

        normalized = self._normalize_bib_text(result.text)
        accepted = (
            str(result.status or "").upper() == "DONE"
            and bool(normalized)
            and float(result.confidence or 0.0) >= float(self.conf_threshold)
        )
        status = "DONE" if accepted else "PENDING"
        error = "" if accepted else (result.error or "OCR_FAILED")
        result_data = self._save_result(
            event_dir,
            meta,
            normalized or str(result.text or "").strip(),
            float(result.confidence or 0.0),
            status,
            error=error,
            source=f"mobile_rec:{result.source}" if result.source else "mobile_rec",
            image_source=result.source,
            participant_id=event.get("participant_id"),
            raw_track_id=event.get("track_id"),
        )
        self._persist_ocr_resolution(
            event_id=event_id,
            bib=normalized,
            conf=float(result.confidence or 0.0),
            status=status,
            evidence_dir=str(event_dir),
            notes=(
                f"Source: mobile_rec:{result.source}; elapsed_ms={result.elapsed_ms:.1f}; "
                f"attempts={result.attempted_candidates}"
            ),
        )
        vlm_scheduled = False
        if status != "DONE":
            vlm_scheduled = self._schedule_event_vlm(
                event=event,
                event_dir=event_dir,
                meta=meta,
                local_text=normalized or str(result.text or "").strip(),
                local_confidence=float(result.confidence or 0.0),
                local_error=error,
            )
        if status == "DONE":
            self.stats["done"] = int(self.stats.get("done", 0)) + 1
            self.stats["resolved_done"] = int(self.stats.get("resolved_done", 0)) + 1
        elif not vlm_scheduled:
            self.stats["pending_result"] = int(self.stats.get("pending_result", 0)) + 1
        if self.on_event_done:
            self.on_event_done(event_id, result_data)

    def _persist_ocr_resolution(
        self,
        event_id: int,
        bib: Optional[str],
        conf: float,
        status: str,
        evidence_dir: str,
        notes: str,
        merged_to: Optional[int] = None,
    ) -> bool:
        normalized_status = str(status or "").strip().upper()
        formal_bib = bib if normalized_status == "DONE" else ""

        if merged_to is not None and formal_bib:
            merge_method = getattr(self.db, 'merge_ocr_duplicate_event', None)
            if merge_method is not None:
                merged = merge_method(
                    duplicate_event_id=int(event_id),
                    target_event_id=int(merged_to),
                    bib=formal_bib,
                    conf=float(conf or 0.0),
                    ocr_state=status,
                    evidence_dir=evidence_dir,
                    notes=notes,
                    use_ocr_conn=True,
                )
                if merged:
                    return True

        return self.db.update_event_ocr_result(
            event_id,
            formal_bib,
            conf,
            status,
            evidence_dir=evidence_dir,
            notes=notes,
            use_ocr_conn=True,
        )

    def _calc_progress_current(self) -> int:
        """计算已处理进度（成功 + 待补 + 失败）。"""
        return (
            int(self.stats.get("done", 0))
            + int(self.stats.get("pending_result", 0))
            + int(self.stats.get("failed", 0))
        )

    def _sync_processed_stat(self) -> int:
        """同步 processed 字段，保持 UI 统计一致。"""
        current = self._calc_progress_current()
        self.stats["processed"] = current
        return current

    def enqueue_live_event(
        self,
        event_id: int,
        evidence_dir: Optional[str] = None,
        cross_time: Optional[float] = None,
        participant_id: Optional[str] = None,
        raw_track_id: Optional[int] = None,
    ) -> bool:
        """实时自动补号入口：将单个事件加入 OCR 队列（若可处理）。"""
        if event_id is None:
            return False

        if self.process_controller is not None:
            return self._enqueue_process_event(
                event_id=event_id,
                evidence_dir=evidence_dir,
                participant_id=participant_id,
                raw_track_id=raw_track_id,
            )

        if not self.ocr:
            logger.debug(f"[OCRManager] 跳过自动OCR: OCR引擎未就绪 (event_id={event_id})")
            return False

        resolved_event = None
        if not evidence_dir or not str(evidence_dir).strip():
            try:
                resolved_event = self.db.get_event(int(event_id))
                if resolved_event:
                    evidence_dir = resolved_event.get("evidence_dir")
                    if cross_time is None:
                        cross_time = resolved_event.get("cross_time")
                    if participant_id is None:
                        participant_id = resolved_event.get("participant_id")
                    if raw_track_id is None:
                        raw_track_id = resolved_event.get("track_id")
            except Exception as e:
                logger.debug(f"[OCRManager] 读取事件失败，无法自动入队: event_id={event_id}, err={e}")
                return False

        if not evidence_dir:
            return False

        event_dir = Path(str(evidence_dir))
        if not event_dir.exists() or not event_dir.is_dir():
            return False

        result_path = event_dir / "result.json"
        if result_path.exists():
            try:
                data = json.loads(result_path.read_text(encoding="utf-8"))
                if str(data.get("status", "")).upper() in {"DONE", "MANUAL"}:
                    return False
            except Exception:
                pass

        priority = float(cross_time) if cross_time is not None else time.time()
        event_dir_str = str(event_dir)

        with self._queue_lock:
            if event_dir_str in self._queued_event_dirs:
                return False
            self._queued_event_dirs.add(event_dir_str)

        try:
            raw_track_id = int(raw_track_id) if raw_track_id is not None else None
        except (TypeError, ValueError):
            raw_track_id = None
        self.task_queue.put((priority, event_dir_str, str(participant_id or ""), raw_track_id))
        self.stats["total"] = int(self.stats.get("total", 0)) + 1
        self.stats["pending"] = int(self.stats.get("pending", 0)) + 1

        if not self._running:
            self._running = True
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()

        return True

    def _enqueue_process_event(
        self,
        *,
        event_id: int,
        evidence_dir: Optional[str],
        participant_id: Optional[str],
        raw_track_id: Optional[int],
    ) -> bool:
        """Queue paths only; never wait for Paddle or copy image arrays."""
        try:
            event_id = int(event_id)
        except (TypeError, ValueError):
            return False
        if event_id in self._queued_process_event_ids or event_id in self._vlm_pending_event_ids:
            return False

        event = self.db.get_event(event_id)
        if not event:
            return False
        if int(event.get("manual_corrected") or 0) != 0:
            return False
        bib = str(event.get("bib_number") or "").strip().upper()
        ocr_state = str(event.get("ocr_state") or "PENDING").strip().upper()
        if bib and bib != "UNKNOWN" and ocr_state == "DONE":
            return False

        event_dir = Path(str(evidence_dir or event.get("evidence_dir") or ""))
        if not event_dir.is_dir():
            return False
        candidate_paths = collect_candidate_paths(event_dir, max_candidates=2)
        if not candidate_paths:
            return False

        job = OcrJob(
            event_id=event_id,
            event_dir=str(event_dir.resolve()),
            candidate_paths=tuple(str(path) for path in candidate_paths),
        )
        accepted = False
        if self.runtime_state == "ready":
            accepted = bool(self.process_controller.submit(job))
        if not accepted:
            if len(self._pending_process_jobs) >= self._pending_process_jobs.maxlen:
                logger.warning("[OCRManager] Pending OCR ids are full; event %s remains PENDING", event_id)
                return False
            self._pending_process_jobs.append(job)

        self._queued_process_event_ids.add(event_id)
        self.stats["total"] = int(self.stats.get("total", 0)) + 1
        self.stats["pending"] = int(self.stats.get("pending", 0)) + 1
        return True

    def stop(self):
        """停止处理"""
        if self.process_controller is not None:
            self.process_controller.stop(timeout=2.0)
            self.runtime_state = "disabled"
            self._pending_process_jobs.clear()
            self._queued_process_event_ids.clear()
        self._vlm_pending_event_ids.clear()
        self._vlm_futures.clear()
        self._vlm_executor.shutdown(wait=False, cancel_futures=True)
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)

    def _worker_loop(self):
        """后台处理循环 (优化版：减少UI刷新频率)"""
        batch_count = 0  # 批次计数器，用于控制UI刷新频率

        while self._running and not self.task_queue.empty():
            try:
                task_item = self.task_queue.get(timeout=1.0)
                if len(task_item) >= 4:
                    _, event_dir_str, participant_id, raw_track_id = task_item
                else:
                    _, event_dir_str = task_item
                    participant_id, raw_track_id = "", None
                try:
                    # 规则在 start_batch / enqueue_live_event 时已同步
                    # 仅实时模式下每50个事件刷新一次（用户可能在UI修改规则）
                    if batch_count > 0 and batch_count % 50 == 0:
                        self._refresh_rules_from_db()
                    self._process_event(
                        Path(event_dir_str),
                        participant_id=participant_id,
                        raw_track_id=raw_track_id,
                    )
                finally:
                    with self._queue_lock:
                        self._queued_event_dirs.discard(event_dir_str)
                    if self.stats.get("pending", 0) > 0:
                        self.stats["pending"] -= 1
                batch_count += 1

                # 进度回调（每处理5个事件才刷新一次UI，减少UI压力）
                if self.on_progress:
                    current = self._sync_processed_stat()
                    # 只在以下情况刷新UI：每5个事件、最后一个事件、或队列为空
                    if batch_count % 5 == 0 or self.task_queue.empty() or current >= self.stats["total"]:
                        self.on_progress(current, self.stats["total"], self.stats)

                # 【关键】节流：让出CPU给主线程，保证画面流畅
                is_live = self._is_race_live()
                base_sleep = self.throttle_sleep_live_seconds if is_live else self.throttle_sleep_idle_seconds
                extra_sleep = 0.0
                if self.throttle_pause_every > 0 and (batch_count % self.throttle_pause_every) == 0:
                    extra_sleep = self.throttle_pause_live_seconds if is_live else self.throttle_pause_idle_seconds
                time.sleep(base_sleep + extra_sleep)

            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"处理循环异常: {e}")

        self._running = False

        # 最终刷新一次UI
        if self.on_progress:
            current = self._sync_processed_stat()
            self.on_progress(current, self.stats["total"], self.stats)

        logger.info("OCR 批处理工作完成")

    def _is_race_live(self) -> bool:
        try:
            latest = self.db.get_latest_cross_time()
        except Exception:
            return False
        if latest is None:
            return False
        now = time.time()
        if latest > 1e12:
            latest = latest / 1000.0
        return (now - float(latest)) <= float(self.throttle_live_window_seconds)

    def _enhance_image(self, img):
        """
        图像增强：用于提升 OCR 识别率 (不改变原图)
        包含：CLAHE 对比度增强、锐化
        """
        if img is None:
            return None
            
        try:
            # 1. 转换为灰度图进行增强
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 2. CLAHE (限制对比度自适应直方图均衡化)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # 3. 锐化 (使用非锐化掩模)
            blurred = cv2.GaussianBlur(enhanced, (0, 0), 3)
            sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
            
            # 4. 转回 BGR (PaddleOCR 期望 3 通道)
            return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            logger.warning(f"图像增强失败: {e}")
            return img

    def _get_vlm_prompt(self):
        """获取针对自行车比赛优化的 VLM 提示词"""
        return (
            "这是一张自行车比赛的过线截图。请仔细观察画面中运动员身上的号码布（通常在背部、腰部或车架上）。"
            "只输出你看到的纯数字号码，不要包含任何其他文字或描述。"
            "如果看不清，请尝试根据轮廓推断，但要确保它是号码布上的数字。"
        )

    def _load_image(self, path: Path) -> Optional[np.ndarray]:
        try:
            return read_image_unicode(path)
        except Exception:
            return None

    def _normalize_bbox(self, bbox: List[float], width: int, height: int, expand_ratio: float) -> Optional[Tuple[int, int, int, int]]:
        if not bbox or len(bbox) != 4:
            return None
        x1, y1, x2, y2 = [int(v) for v in bbox]
        if x2 <= x1 or y2 <= y1:
            return None
        if expand_ratio > 0:
            dx = int((x2 - x1) * expand_ratio)
            dy = int((y2 - y1) * expand_ratio)
            x1 -= dx
            y1 -= dy
            x2 += dx
            y2 += dy
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _crop_with_bbox(self, img: np.ndarray, bbox: List[float], expand_ratio: float) -> Optional[np.ndarray]:
        height, width = img.shape[:2]
        norm = self._normalize_bbox(bbox, width, height, expand_ratio)
        if not norm:
            return None
        x1, y1, x2, y2 = norm
        return img[y1:y2, x1:x2]

    def _run_local_ocr(self, img: np.ndarray) -> Tuple[str, float]:
        """执行本地OCR识别（带异常处理和图像验证）"""
        # 图像有效性检查
        if img is None:
            return "", 0.0
        if not isinstance(img, np.ndarray):
            return "", 0.0
        if img.size == 0:
            return "", 0.0
        if len(img.shape) < 2:
            return "", 0.0
        h, w = img.shape[:2]
        if h < 5 or w < 5:  # 图像太小，OCR无法处理
            return "", 0.0
        if h > 4096 or w > 4096:  # 图像太大，可能导致内存问题
            try:
                scale = min(4096 / h, 4096 / w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            except:
                return "", 0.0

        try:
            results, _ = self.ocr(img)
            best_bib = ""
            best_conf = 0.0
            if results:
                for _, text, conf in results:
                    norm_text = self._normalize_bib_text(text)
                    if not norm_text:
                        continue
                    if conf > best_conf:
                        best_bib = norm_text
                        best_conf = conf
            return best_bib, best_conf
        except Exception as e:
            logger.warning(f"[OCRManager] OCR调用失败: {e}")
            return "", 0.0

    def _assess_quality(self, meta: dict) -> Tuple[str, str]:
        """
        质量预筛：根据 bbox 尺寸和位置关系，判断 OCR 是否有意义。

        返回: (quality_grade, reason)
          - "good":     号码牌够大且在运动员框内，值得 OCR
          - "marginal": 勉强可试，但降低 OCR 调用次数
          - "poor":     号码牌太小或位置错误，直接标记为需人工
        """
        bbox_athlete = meta.get("bbox_athlete")
        bbox_bib = meta.get("bbox_bib")

        if not bbox_bib or len(bbox_bib) != 4:
            return "poor", "无号码牌检测框"

        bx1, by1, bx2, by2 = [int(v) for v in bbox_bib]
        bib_w = bx2 - bx1
        bib_h = by2 - by1

        if bib_w < 15 or bib_h < 15:
            return "poor", f"号码牌框太小({bib_w}x{bib_h}px)"

        # 检查 bib 是否在 athlete 框内
        bib_in_athlete = False
        if bbox_athlete and len(bbox_athlete) == 4:
            ax1, ay1, ax2, ay2 = [int(v) for v in bbox_athlete]
            if bx2 > ax1 and bx1 < ax2 and by2 > ay1 and by1 < ay2:
                bib_in_athlete = True

        if bib_w < 30:
            if not bib_in_athlete:
                return "poor", f"号码牌框极小({bib_w}x{bib_h}px)且不在运动员框内"
            return "poor", f"号码牌框极小({bib_w}x{bib_h}px)"

        if bib_w < 50:
            if not bib_in_athlete:
                return "marginal", f"号码牌框较小({bib_w}x{bib_h}px)且不在运动员框内"
            return "marginal", f"号码牌框较小({bib_w}x{bib_h}px)"

        if not bib_in_athlete:
            return "marginal", f"号码牌框尺寸尚可({bib_w}x{bib_h}px)但不在运动员框内"

        return "good", f"号码牌框正常({bib_w}x{bib_h}px)"

    def _process_event(
        self,
        event_dir: Path,
        participant_id: Optional[str] = None,
        raw_track_id: Optional[int] = None,
    ):
        """处理单个事件（CPU模式：OCR在CPU运行，GPU 100% 留给YOLO实时检测）

        CPU模式策略（每事件最多3次OCR，平衡速度与准确率）：
        1. bib.jpg 原图 → 图很小，CPU秒出
        2. athlete.jpg 放大3x → 核心策略，CPU约5-10s
        3. athlete.jpg 放大2x → 仅在3x未命中时尝试
        """
        meta_path = event_dir / "meta.json"
        bib_path = event_dir / "bib.jpg"
        athlete_path = event_dir / "athlete.jpg"
        full_path = event_dir / "full.jpg"

        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)

            event_id = meta["event_id"]
            participant_id = str(participant_id or meta.get("participant_id") or "").strip()
            if raw_track_id is None:
                raw_track_id = meta.get("track_id")
            try:
                raw_track_id = int(raw_track_id) if raw_track_id is not None else None
            except (TypeError, ValueError):
                raw_track_id = None

            if not self.ocr:
                logger.error("未设置 OCR 引擎")
                return

            quality_grade, quality_reason = self._assess_quality(meta)
            best_bib, best_conf, best_source, best_image = "", 0.0, "LOCAL", ""
            multi_frame_required = False
            multi_frame_consistent = False
            evidence_kind = str(meta.get("bib_evidence_kind") or "").strip().lower()
            fallback_only_evidence = evidence_kind == "fallback"
            insufficient_multi_frame_evidence = False

            def _update_best(bib, conf, source, image):
                nonlocal best_bib, best_conf, best_source, best_image
                if bib and conf > best_conf:
                    best_bib, best_conf, best_source, best_image = bib, conf, source, image

            candidate_paths = []
            meta_paths = meta.get("paths") if isinstance(meta, dict) else None
            if isinstance(meta_paths, dict):
                configured_candidates = meta_paths.get("bib_candidates")
                if isinstance(configured_candidates, list):
                    for relative_path in configured_candidates[:4]:
                        candidate_path = event_dir / str(relative_path)
                        if candidate_path.exists():
                            candidate_paths.append(candidate_path)

            # New events use distinct frames for consensus. Old events keep the bib.jpg path.
            multi_frame_required = len(candidate_paths) >= 2
            insufficient_multi_frame_evidence = evidence_kind == "detected" and len(candidate_paths) < 2
            if quality_grade in ("good", "marginal") and multi_frame_required:
                frame_results = []
                for candidate_path in candidate_paths:
                    candidate_img = self._load_image(candidate_path)
                    if candidate_img is None:
                        continue
                    candidate_bib, candidate_conf, candidate_source = self._run_multi_scale_ocr(candidate_img)
                    tight_bib, tight_conf, tight_source = self._run_tight_numeric_recognition(candidate_img)
                    if tight_bib:
                        normal_is_multidigit = bool(candidate_bib and len(candidate_bib) >= 2)
                        if not normal_is_multidigit or tight_bib == candidate_bib:
                            if tight_bib != candidate_bib or tight_conf > candidate_conf:
                                candidate_bib = tight_bib
                                candidate_conf = tight_conf
                                candidate_source = tight_source
                    if candidate_bib:
                        frame_results.append(
                            (candidate_bib, float(candidate_conf), candidate_source, candidate_path.name)
                        )

                grouped = {}
                for candidate_bib, candidate_conf, candidate_source, candidate_name in frame_results:
                    grouped.setdefault(candidate_bib, []).append(
                        (candidate_conf, candidate_source, candidate_name)
                    )

                ranked_groups = sorted(
                    grouped.items(),
                    key=lambda item: (len(item[1]), sum(value[0] for value in item[1])),
                    reverse=True,
                )
                if ranked_groups:
                    selected_bib, selected_values = ranked_groups[0]
                    runner_up_count = len(ranked_groups[1][1]) if len(ranked_groups) > 1 else 0
                    multi_frame_consistent = len(selected_values) >= 2 and len(selected_values) > runner_up_count
                    best_bib = selected_bib
                    best_conf = float(sum(value[0] for value in selected_values) / len(selected_values))
                    best_source = "multi_frame_consensus" if multi_frame_consistent else "multi_frame_conflict"
                    best_image = "multi_frame"
            elif quality_grade in ("good", "marginal") and bib_path.exists():
                bib_img = self._load_image(bib_path)
                if bib_img is not None:
                    bib_bib, bib_conf, bib_source = self._run_multi_scale_ocr(bib_img)
                    _update_best(bib_bib, bib_conf, bib_source, "bib")

            # 高置信直接跳过后续
            if best_bib and best_conf >= 0.8 and (not multi_frame_required or multi_frame_consistent):
                pass
            elif not multi_frame_required:
                # ── 步骤2：athlete.jpg 放大3x（核心策略，CPU约5-10s） ──
                if athlete_path.exists():
                    athlete_img = self._load_image(athlete_path)
                    if athlete_img is not None:
                        h, w = athlete_img.shape[:2]

                        scale3 = 3
                        if max(h, w) * scale3 > 2000:
                            scale3 = max(1, 2000 // max(h, w))
                        if scale3 >= 2:
                            up3 = cv2.resize(athlete_img, (w * scale3, h * scale3),
                                             interpolation=cv2.INTER_CUBIC)
                            bib3, conf3 = self._run_local_ocr(up3)
                            _update_best(bib3, conf3, f"upscale_{scale3}x", "athlete")

                        # ── 步骤3：athlete.jpg 放大2x（仅在3x未命中时，不同尺度可能触发不同检测框） ──
                        if best_conf < 0.5 and max(h, w) * 2 <= 2000:
                            up2 = cv2.resize(athlete_img, (w * 2, h * 2),
                                             interpolation=cv2.INTER_CUBIC)
                            bib2, conf2 = self._run_local_ocr(up2)
                            _update_best(bib2, conf2, "upscale_2x", "athlete")

            # ── 步骤3：归并 & 保存 ──
            error = ""

            # 3. 归并逻辑 (统一为一个运动员)
            participant_state = None
            if best_bib and participant_id and raw_track_id is not None:
                participant_state = self._add_participant_ocr_vote(
                    participant_id,
                    raw_track_id,
                    best_bib,
                    best_conf,
                )
                if participant_state and participant_state.best_candidate:
                    best_bib = participant_state.best_candidate
                    best_conf = participant_state.best_confidence
            participant_conflict = bool(
                participant_state and participant_state.status == "CONFLICT"
            )

            merged_to = None
            can_auto_confirm = (
                (not multi_frame_required or multi_frame_consistent)
                and not fallback_only_evidence
                and not insufficient_multi_frame_evidence
                and not participant_conflict
            )
            if best_bib and best_conf >= self.conf_threshold and can_auto_confirm:
                cross_time = meta.get("cross_time_unix")
                try:
                    cross_time = float(cross_time)
                except (TypeError, ValueError):
                    cross_time = None

                nearby_event = None
                if cross_time is not None and not participant_id:
                    if cross_time > 1e12:
                        cross_time = cross_time / 1000.0
                    nearby_event = self.db.get_event_by_bib_and_time(
                        best_bib,
                        cross_time,
                        window_seconds=self.merge_window_seconds
                    )
                
                if nearby_event and nearby_event["event_id"] != event_id:
                    merged_to = nearby_event["event_id"]
                    self.stats["merged"] += 1
                    logger.info(f"事件 {event_id} 归并到 {merged_to} (Bib: {best_bib})")

            status = "DONE" if best_bib and best_conf >= self.conf_threshold and can_auto_confirm else "PENDING"
            if status != "DONE":
                if fallback_only_evidence:
                    error = "FALLBACK_ONLY_EVIDENCE"
                elif insufficient_multi_frame_evidence:
                    error = "INSUFFICIENT_MULTI_FRAME_EVIDENCE"
                elif multi_frame_required and best_bib and not multi_frame_consistent:
                    error = "INCONSISTENT_MULTI_FRAME"
                elif participant_conflict:
                    error = "PARTICIPANT_OCR_CONFLICT"
                elif best_bib:
                    error = "LOW_CONF"
                elif quality_grade == "poor":
                    error = f"QUALITY_POOR: {quality_reason}"
                else:
                    error = "OCR_FAILED"

            source = best_source
            if best_image:
                source = f"{best_source}:{best_image}"

            result_data = self._save_result(
                event_dir,
                meta,
                best_bib,
                best_conf,
                status,
                error=error,
                merged_to=merged_to,
                source=source,
                image_source=best_image,
                crop_applied=False,
                participant_id=participant_id,
                raw_track_id=raw_track_id,
                participant_status=(participant_state.status if participant_state else "PENDING"),
            )
            
            self._persist_ocr_resolution(
                event_id=event_id,
                bib=best_bib,
                conf=best_conf,
                status=status,
                evidence_dir=str(event_dir),
                notes=f"Source: {source}. Merged to {merged_to}" if merged_to else f"Source: {source}",
                merged_to=merged_to,
            )
            
            if status == "DONE":
                self.stats["done"] += 1
                self.stats["resolved_done"] = int(self.stats.get("resolved_done", 0)) + 1
            else:
                self.stats["pending_result"] = int(self.stats.get("pending_result", 0)) + 1
            if self.on_event_done:
                self.on_event_done(event_id, result_data)

        except Exception as e:
            logger.error(f"处理事件 {event_dir.name} 失败: {e}")
            self.stats["failed"] += 1

    def _enhance_image_multi(self, img: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """
        多策略图像增强
        返回: [(策略名称, 增强后的图像), ...]
        """
        if img is None or img.size == 0:
            return []

        candidates = []

        try:
            # 策略0: 原图
            candidates.append(("original", img.copy()))

            # 策略1: CLAHE增强 (现有方法)
            clahe_img = self._enhance_image(img)
            if clahe_img is not None:
                candidates.append(("clahe", clahe_img))

            # 转换为灰度图供后续处理
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

            # 策略2: 自适应二值化
            try:
                binary = cv2.adaptiveThreshold(
                    gray, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    blockSize=15,
                    C=5
                )
                candidates.append(("binary", cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)))
            except Exception as e:
                logger.debug(f"二值化失败: {e}")

            # 策略3: Otsu二值化
            try:
                _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                candidates.append(("otsu", cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)))
            except Exception as e:
                logger.debug(f"Otsu二值化失败: {e}")

            # 策略4: 形态学增强（加粗文字，填补断裂）
            try:
                # 先二值化
                _, binary_for_morph = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # 膨胀操作（加粗文字）
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                dilated = cv2.dilate(binary_for_morph, kernel, iterations=1)
                candidates.append(("morph", cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)))
            except Exception as e:
                logger.debug(f"形态学增强失败: {e}")

            # 策略5: 反色（白底黑字 → 黑底白字）
            # PaddleOCR对黑底白字识别效果更好
            try:
                inverted = cv2.bitwise_not(gray)
                candidates.append(("inverted", cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)))
            except Exception as e:
                logger.debug(f"反色失败: {e}")

            # 策略6: 去噪 + 锐化
            try:
                # 高斯去噪
                denoised = cv2.GaussianBlur(gray, (3, 3), 0)
                # 锐化
                kernel_sharp = np.array([[-1, -1, -1],
                                          [-1,  9, -1],
                                          [-1, -1, -1]])
                sharpened = cv2.filter2D(denoised, -1, kernel_sharp)
                candidates.append(("sharp", cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)))
            except Exception as e:
                logger.debug(f"锐化失败: {e}")

            # 策略7: 对比度拉伸
            try:
                # 直方图均衡化
                equalized = cv2.equalizeHist(gray)
                candidates.append(("equalized", cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)))
            except Exception as e:
                logger.debug(f"直方图均衡化失败: {e}")

        except Exception as e:
            logger.error(f"图像增强异常: {e}")

        return candidates

    def _run_tight_numeric_recognition(self, img: np.ndarray) -> Tuple[str, float, str]:
        """Recover multi-digit bicycle bibs hidden by padding around a small detected crop."""
        if not self.only_numeric or img is None or img.size == 0:
            return "", 0.0, "tight_unavailable"
        if self.ocr is None or not hasattr(self.ocr, "recognize_only"):
            return "", 0.0, "tight_unavailable"

        height, width = img.shape[:2]
        min_side = min(height, width)
        if min_side < 24:
            return "", 0.0, "tight_too_small"

        border = min(5, max(2, int(round(min_side * 0.12))))
        if height <= border * 2 + 8 or width <= border * 2 + 8:
            return "", 0.0, "tight_too_small"

        tight = img[border:height - border, border:width - border]
        tight = cv2.resize(tight, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

        try:
            results, _ = self.ocr.recognize_only(tight)
        except Exception as e:
            logger.debug(f"Tight recognition failed: {e}")
            return "", 0.0, "tight_failed"

        best_bib = ""
        best_conf = 0.0
        for _, text, conf in results or []:
            normalized = self._normalize_bib_text(text)
            # This path may restore missing adjacent digits, but must never
            # replace one valid single-digit bib with another single digit.
            if not normalized.isdigit() or len(normalized) < 2:
                continue
            if float(conf) > best_conf:
                best_bib = normalized
                best_conf = float(conf)

        if not best_bib:
            return "", 0.0, "tight_no_multidigit"
        return best_bib, best_conf, "tight_recognition"

    def _run_multi_scale_ocr(self, img: np.ndarray) -> Tuple[str, float, str]:
        """
        多尺度 + 多策略 OCR (优化版：减少调用次数，提升响应速度)
        返回: (最佳号码, 最佳置信度, 来源策略)

        优化策略：
        1. 原图OCR如果置信度>=0.6，直接返回（快速路径）
        2. 只在原图失败时才尝试增强策略
        3. 增强策略精简为3种最有效的（CLAHE、二值化、放大）
        """
        if img is None or img.size == 0:
            return "", 0.0, "invalid"

        results = []
        h, w = img.shape[:2]

        # 1. 原图OCR（快速路径）
        bib, conf = self._run_local_ocr(img)
        results.append((bib, conf, "original"))

        # ✅ 快速路径：如果原图识别置信度足够高，直接返回
        if bib and conf >= 0.6:
            logger.debug(f"[OCR快速路径] 原图识别成功: {bib} (conf={conf:.2f})")
            return bib, conf, "original"

        # 速度优化：如果配置了号码范围且原图没产出有效结果，
        # 增强策略大概率也只会产出范围外的垃圾，最多只尝试1种增强
        has_range_filter = bool(self._bib_ranges)
        original_empty = not bib

        # 2. 图像放大（对 athlete.jpg 等小图至关重要）
        # 号码文字在小图中可能只有20-40px，PaddleOCR检测需要更大尺寸
        # 对 athlete.jpg (~100-400px) 放大到最大边~800px
        # 安全上限：放大后不超过2000px，且原图最大边不超过600px才放大（避免对全帧图放大）
        if max(h, w) < 600:
            try:
                scale = max(2, min(4, 800 // max(h, w, 1)))  # 放大到最大边~800px
                if max(h, w) * scale > 2000:
                    scale = max(1, 2000 // max(h, w))
                upscaled = cv2.resize(img, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
                bib, conf = self._run_local_ocr(upscaled)
                results.append((bib, conf, f"upscale_{scale}x"))

                # 放大后识别成功，直接返回
                if bib and conf >= 0.5:
                    logger.debug(f"[OCR放大] 放大{scale}x识别成功: {bib} (conf={conf:.2f})")
                    return bib, conf, f"upscale_{scale}x"
            except Exception as e:
                logger.debug(f"图像放大失败: {e}")

            # 范围过滤模式下，放大也失败了就不再浪费时间
            if has_range_filter and not any(r[0] for r in results):
                return "", 0.0, "range_filtered"

        # 3. 精简的增强策略
        # 只有在前面都失败时才执行，避免不必要的计算
        if not any(r[0] and r[1] >= 0.4 for r in results):
            try:
                # 策略A: CLAHE增强（对低对比度图像有效）
                clahe_img = self._enhance_image(img)
                if clahe_img is not None:
                    bib, conf = self._run_local_ocr(clahe_img)
                    results.append((bib, conf, "enhance_clahe"))
                    if bib and conf >= 0.5:
                        return bib, conf, "enhance_clahe"
            except Exception as e:
                logger.debug(f"CLAHE增强失败: {e}")

            # 范围过滤模式下，CLAHE也失败了就不再继续Otsu（节省1次OCR调用）
            if has_range_filter:
                pass  # 跳过Otsu
            else:
                try:
                    # 策略B: Otsu二值化（对复杂背景有效）
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
                    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    otsu_bgr = cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)
                    bib, conf = self._run_local_ocr(otsu_bgr)
                    results.append((bib, conf, "enhance_otsu"))
                except Exception as e:
                    logger.debug(f"Otsu二值化失败: {e}")

        # 4. 选择置信度最高的结果
        valid_results = [(b, c, s) for b, c, s in results if b and c > 0]
        if not valid_results:
            return "", 0.0, "all_failed"

        # 排序：置信度从高到低
        valid_results.sort(key=lambda x: x[1], reverse=True)
        best_bib, best_conf, best_source = valid_results[0]

        logger.debug(f"[OCR多尺度] 共{len(results)}种策略，最佳结果: {best_bib} (conf={best_conf:.2f}, source={best_source})")

        return best_bib, best_conf, best_source

    def _vote_ocr_results(self, results: List[Tuple[str, float, str]]) -> Tuple[str, float]:
        """
        OCR结果投票
        输入: [(号码, 置信度, 来源), ...]
        输出: (最终号码, 综合置信度)
        """
        if not results:
            return "", 0.0

        # 过滤掉空结果
        valid_results = [(b, c, s) for b, c, s in results if b]
        if not valid_results:
            return "", 0.0

        # 1. 数字部分投票
        digit_votes = {}  # {数字部分: 累计置信度}
        for bib, conf, source in valid_results:
            digits = ''.join(filter(str.isdigit, bib))
            if not digits:
                continue
            digit_votes[digits] = digit_votes.get(digits, 0.0) + conf

        if not digit_votes:
            # 没有数字，直接返回置信度最高的
            valid_results.sort(key=lambda x: x[1], reverse=True)
            return valid_results[0][0], valid_results[0][1]

        # 找到得票最高的数字部分
        best_digits = max(digit_votes, key=digit_votes.get)

        # 2. 字母前缀投票（针对AXXXX格式）
        prefix_votes = {}  # {字母前缀: 累计置信度}
        for bib, conf, source in valid_results:
            current_digits = ''.join(filter(str.isdigit, bib))
            if current_digits == best_digits:
                prefix = ''.join(filter(str.isalpha, bib)).upper()
                prefix_votes[prefix] = prefix_votes.get(prefix, 0.0) + conf

        best_prefix = max(prefix_votes, key=prefix_votes.get) if prefix_votes else ""

        # 3. 组合最终结果
        final_bib = best_prefix + best_digits

        # 4. 计算综合置信度
        total_conf = digit_votes[best_digits] + prefix_votes.get(best_prefix, 0.0)
        final_conf = min(1.0, total_conf / len(valid_results))

        logger.info(f"[OCR投票] 数字部分: {best_digits} (票数={digit_votes[best_digits]:.2f}), "
                   f"前缀: {best_prefix} (票数={prefix_votes.get(best_prefix, 0):.2f}), "
                   f"最终: {final_bib} (综合置信度={final_conf:.2f})")

        return final_bib, final_conf

    def _process_event_with_enhanced_ocr(self, meta, candidates):
        """
        使用增强OCR处理事件 (优化版：快速路径 + 减少调用次数)
        """
        best_bib = ""
        best_conf = 0.0
        best_source = "LOCAL"
        best_image = ""
        best_crop_applied = False
        error = ""
        used_vlm = False

        # ✅ 优化：按优先级排序候选图片（bib > athlete > full）
        # bib.jpg 通常是最清晰的号码截图，优先处理
        priority_order = {"bib": 0, "bib_fallback": 1, "athlete": 2, "full": 3}
        candidates.sort(key=lambda c: priority_order.get(c["name"], 99))

        for candidate in candidates:
            img = self._load_image(candidate["path"])
            if img is None:
                if not error:
                    error = f"Invalid image: {candidate['name']}"
                continue

            crop_applied = False
            if candidate["use_bbox"]:
                cand_bbox = candidate.get("bbox")
                if not cand_bbox:
                    if not error:
                        error = f"Missing bbox for {candidate['name']}"
                    continue
                cropped = self._crop_with_bbox(img, cand_bbox, self.bbox_expand_ratio)
                if cropped is None:
                    if not error:
                        error = f"Invalid bbox for {candidate['name']}"
                    continue
                img = cropped
                crop_applied = True

            # 检查是否需要VLM（默认只对 bib 图启用，避免把整人图/全图送云端）
            img_height, img_width = img.shape[:2]
            too_small_for_vlm = min(img_height, img_width) < self.vlm_min_size
            allow_vlm = candidate["name"] in ["bib", "bib_fallback"]

            vlm_mode = str(getattr(self, "vlm_mode", "fallback") or "fallback").lower()
            if vlm_mode not in {"fallback", "first", "only"}:
                vlm_mode = "fallback"

            local_bib = ""
            local_conf = 0.0
            ocr_source = "LOCAL"

            def _try_vlm() -> Tuple[str, float]:
                nonlocal used_vlm
                if (not self.vlm) or too_small_for_vlm or (not allow_vlm):
                    return "", 0.0

                athlete_list = None
                if hasattr(self.db, "get_all_athlete_bibs"):
                    athlete_list = self.db.get_all_athlete_bibs()

                try:
                    vlm_bib, vlm_conf, _ = self.vlm.ocr_bib(
                        img,
                        athlete_list=athlete_list,
                        only_numeric=bool(self.only_numeric),
                        prompt=None,
                    )
                except Exception as e:
                    logger.debug(f"[OCRManager] VLM OCR调用失败: {e}")
                    return "", 0.0

                vlm_norm = self._normalize_bib_text(vlm_bib)
                if not vlm_norm:
                    return "", 0.0

                if not used_vlm:
                    self.stats["vlm_used"] += 1
                    used_vlm = True
                return vlm_norm, float(vlm_conf or 0.0)

            # 1) VLM 优先 / 仅 VLM（仅对 bib 图生效）
            if vlm_mode in {"first", "only"}:
                v_bib, v_conf = _try_vlm()
                if v_bib:
                    local_bib = v_bib
                    local_conf = v_conf
                    ocr_source = "VLM"

            # 2) 本地 OCR（除非用户选择“只用大模型”且大模型已成功）
            if (vlm_mode != "only") or (not local_bib):
                p_bib, p_conf, p_source = self._run_multi_scale_ocr(img)
                if p_bib and p_conf >= local_conf:
                    local_bib = p_bib
                    local_conf = p_conf
                    ocr_source = p_source

            # ✅ 快速路径：高置信度结果直接返回，不再尝试其他候选图片
            if local_bib and local_conf >= 0.6:
                logger.debug(f"事件 {meta['event_id']} 快速路径命中: {local_bib} (conf={local_conf:.2f}, source={candidate['name']})")
                return local_bib, local_conf, ocr_source, candidate["name"], crop_applied, ""

            # 3) 仅兜底模式：本地失败/很低置信度时才调用 VLM
            if vlm_mode == "fallback" and (not local_bib or local_conf < 0.3):
                v_bib, v_conf = _try_vlm()
                if v_bib and v_conf > local_conf:
                    local_bib = v_bib
                    local_conf = v_conf
                    ocr_source = "VLM"
                    logger.info(f"事件 {meta['event_id']} VLM识别成功: {v_bib} (conf={v_conf:.2f})")

            # 更新最佳结果
            if local_bib and local_conf >= self.conf_threshold:
                best_bib = local_bib
                best_conf = local_conf
                best_source = ocr_source
                best_image = candidate["name"]
                best_crop_applied = crop_applied
                error = ""
                break  # 找到高置信度结果，提前退出

            if local_bib and local_conf > best_conf:
                best_bib = local_bib
                best_conf = local_conf
                best_source = ocr_source
                best_image = candidate["name"]
                best_crop_applied = crop_applied

        return best_bib, best_conf, best_source, best_image, best_crop_applied, error

    def _save_result(
        self,
        event_dir: Path,
        meta: dict,
        bib: Optional[str],
        conf: float,
        status: str,
        error: str = "",
        merged_to: Optional[int] = None,
        source: str = "LOCAL",
        image_source: str = "",
        crop_applied: bool = False,
        participant_id: Optional[str] = None,
        raw_track_id: Optional[int] = None,
        participant_status: str = "PENDING",
    ):
        """保存 result.json (原子操作)"""
        result = {
            "event_id": meta["event_id"],
            "participant_id": str(participant_id or meta.get("participant_id") or ""),
            "raw_track_id": raw_track_id if raw_track_id is not None else meta.get("track_id"),
            "participant_ocr_status": participant_status,
            "bib": bib,
            "confidence": float(conf),
            "status": status,
            "error": error,
            "merged_to": merged_to,
            "source": source,
            "image_source": image_source,
            "crop_applied": bool(crop_applied),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        }
        
        target_path = event_dir / "result.json"
        tmp_path = target_path.with_suffix(".json.tmp")
        
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
        if target_path.exists():
            target_path.unlink()
        tmp_path.rename(target_path)
        
        return result
