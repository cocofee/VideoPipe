"""
增强版实时检测模块

改进点：
1. 多维度龙门/背景过滤（速度+宽高比+位置+尺寸）
2. 智能并排过线识别（号码牌驱动+动态阈值）
3. 号码牌驱动检测补全（防漏检）
4. VLM 辅助判定（豆包 Doubao-Seed-1.8）
5. 增强 OCR（多尺度+多角度+多通道+投票）
"""

import cv2
import numpy as np
import os
from ultralytics import YOLO
import threading
import queue
import time
import difflib
import requests
import base64
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Dict, Any, Tuple, List, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from concurrent.futures import ThreadPoolExecutor

if __package__:
    from .crossing_lifecycle import CrossingLifecycle
    from .event_profile import EventProfile, build_sport_event_profile
    from .participant_identity import IdentityConfig, ParticipantIdentityManager
    from .participant_models import CrossingCandidate, ParticipantObservation
    from .vlm_utils import OpenAIVLMAssistant, VLMRateLimitError as SharedVLMRateLimitError
else:
    from crossing_lifecycle import CrossingLifecycle
    from event_profile import EventProfile, build_sport_event_profile
    from participant_identity import IdentityConfig, ParticipantIdentityManager
    from participant_models import CrossingCandidate, ParticipantObservation
    from vlm_utils import OpenAIVLMAssistant, VLMRateLimitError as SharedVLMRateLimitError

try:
    from ensemble_boxes import weighted_boxes_fusion
    HAS_WBF = True
except ImportError:
    HAS_WBF = False

try:
    from .logger import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Detector")


LOCAL_VIDEO_EVENT_SETTLE_SECONDS = 0.85


def resolve_athlete_validator_model(configured_path: str, search_roots: List[Path]) -> Optional[Path]:
    """Resolve an optional generic bicycle detector without requiring it."""
    roots = [Path(root).expanduser() for root in search_roots]
    candidates: List[Path] = []
    if configured_path:
        configured = Path(configured_path).expanduser()
        if configured.is_absolute():
            candidates.append(configured)
        else:
            candidates.extend(root / configured for root in roots)
            candidates.append(configured)
    candidates.extend(root / "yolov8s.pt" for root in roots)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def resolve_athlete_detector_model(configured_path: str, search_roots: List[Path]) -> Optional[Path]:
    """Resolve the optional generic YOLO11 person detector."""
    roots = [Path(root).expanduser() for root in search_roots]
    candidates: List[Path] = []
    if configured_path:
        configured = Path(configured_path).expanduser()
        if configured.is_absolute():
            candidates.append(configured)
        else:
            candidates.extend(root / configured for root in roots)
            candidates.append(configured)
    candidates.extend(root / "yolo11s.pt" for root in roots)
    candidates.extend(root / "yolo11n.pt" for root in roots)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def resolve_performance_profile(
    requested: str = "auto",
    *,
    cuda_available: Optional[bool] = None,
) -> Dict[str, Any]:
    """Resolve detection cost without changing source-frame evidence resolution."""
    profiles = {
        "accuracy": {
            "name": "accuracy",
            "process_imgsz": 896,
            "adaptive_frame_skip": False,
        },
        "gpu": {
            "name": "gpu",
            "process_imgsz": 896,
            "adaptive_frame_skip": True,
        },
        "balanced": {
            "name": "balanced",
            "process_imgsz": 832,
            "adaptive_frame_skip": True,
        },
        "laptop": {
            "name": "laptop",
            "process_imgsz": 832,
            "adaptive_frame_skip": True,
        },
    }
    aliases = {
        "cpu": "laptop",
        "low_power": "laptop",
        "low-power": "laptop",
    }
    name = aliases.get(str(requested or "auto").strip().lower(), str(requested or "auto").strip().lower())

    if name == "auto":
        if cuda_available is None:
            try:
                import torch

                cuda_available = bool(torch.cuda.is_available())
            except Exception:
                cuda_available = False
        name = "gpu" if cuda_available else "laptop"

    if name not in profiles:
        logger.warning(f"未知性能档 {requested!r}，回退到自动档")
        return resolve_performance_profile("auto", cuda_available=cuda_available)
    return dict(profiles[name])


# ============================================================================
# 数据类定义
# ============================================================================

class PaddleOcrAdapter:
    def __init__(self, ocr, recognizer=None):
        self.ocr = ocr
        self.recognizer = recognizer

    @staticmethod
    def _parse_paddle_predictions(result):
        lines = []
        if result is None:
            return lines

        for item in result:
            payload = item
            if not hasattr(payload, "get"):
                payload = getattr(item, "json", None)
                if callable(payload):
                    payload = payload()

            if hasattr(payload, "get"):
                nested = payload.get("res")
                data = nested if hasattr(nested, "get") else payload
                texts = data.get("rec_texts")
                scores = data.get("rec_scores")
                if texts is not None and scores is not None:
                    for text, conf in zip(texts, scores):
                        normalized_text = str(text or "").strip()
                        if normalized_text:
                            lines.append([None, normalized_text, float(conf)])

                text = str(data.get("rec_text") or "").strip()
                if text:
                    lines.append([None, text, float(data.get("rec_score") or 0.0)])
                continue

            if isinstance(item, list):
                for sub in item:
                    if sub and len(sub) >= 2 and isinstance(sub[1], (list, tuple)) and len(sub[1]) >= 2:
                        text = str(sub[1][0] or "").strip()
                        if text:
                            lines.append([sub[0], text, float(sub[1][1])])
        return lines

    def recognize_only(self, img):
        """Run whole-image text recognition without the text detector."""
        if self.recognizer is not None:
            try:
                result = self.recognizer.predict(img, batch_size=1)
                return self._parse_paddle_predictions(result), 0.0
            except Exception as e:
                logger.debug(f"PaddleOCR explicit recognition-only failed: {e}")

        if hasattr(self.ocr, "ocr"):
            try:
                result = self.ocr.ocr(img, det=False, cls=False)
                if result and result[0] and result[0][0]:
                    text = result[0][0][0]
                    conf = result[0][0][1]
                    if text:
                        return [[None, text, float(conf)]], 0.0
            except Exception as e:
                logger.debug(f"PaddleOCR legacy recognition-only failed: {e}")

        return [], 0.0

    def __call__(self, img):
        lines = []

        modern_predict_attempted = False
        if self.recognizer is not None and hasattr(self.ocr, "predict"):
            modern_predict_attempted = True
            try:
                result = self.ocr.predict(
                    img,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    text_det_thresh=0.1,
                    text_det_box_thresh=0.2,
                    text_det_unclip_ratio=1.5,
                    text_det_limit_side_len=960,
                    text_det_limit_type="max",
                    text_rec_score_thresh=0.0,
                )
                lines = self._parse_paddle_predictions(result)
                if lines:
                    return lines, 0.0
            except Exception as e:
                logger.debug(f"PaddleOCR predict failed, trying recognition-only: {e}")

            try:
                lines, _ = self.recognize_only(img)
                if lines:
                    return lines, 0.0
            except Exception as e:
                logger.debug(f"PaddleOCR recognition-only failed: {e}")

        # ✅ 改进：优先使用 det=True 模式，先定位文字区域再识别
        # 原因：号码牌剪裁图通常包含运动员身体、车架等干扰
        if not modern_predict_attempted and hasattr(self.ocr, "ocr"):
            detection_failed = False
            try:
                # 方案A: 完整检测+识别模式 (更准确，但稍慢)
                result = self.ocr.ocr(img, det=True, cls=True)
                if result and result[0]:
                    for line in result[0]:
                        if line and len(line) >= 2:
                            bbox = line[0]
                            text_info = line[1]
                            if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                                text, conf = text_info[0], text_info[1]
                                lines.append([bbox, text, float(conf)])
                    if lines:
                        return lines, 0.0
            except Exception as e:
                detection_failed = True
                logger.debug(f"PaddleOCR detection mode failed, trying recognition-only: {e}")

                # 方案B: 仅识别模式（快速回退）
                try:
                    result = self.ocr.ocr(img, det=False, cls=False)
                    if result and result[0] and result[0][0]:
                        text = result[0][0][0]
                        conf = result[0][0][1]
                        if text:
                            return [[None, text, float(conf)]], 0.0
                except Exception as e2:
                    logger.debug(f"PaddleOCR recognition-only also failed: {e2}")

            if not lines and not detection_failed:
                try:
                    result = self.ocr.ocr(img, det=False, cls=False)
                    if result and result[0] and result[0][0]:
                        text = result[0][0][0]
                        conf = result[0][0][1]
                        if text:
                            return [[None, text, float(conf)]], 0.0
                except Exception as e2:
                    logger.debug(f"PaddleOCR recognition-only also failed: {e2}")

        # 如果 ocr 方法不可用，尝试 predict 方法（部分 OCR 引擎）
        if not modern_predict_attempted and hasattr(self.ocr, "predict"):
            try:
                result = self.ocr.predict(
                    img,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    text_det_thresh=0.1,
                    text_det_box_thresh=0.2,
                    text_det_unclip_ratio=1.5,
                    text_det_limit_side_len=960,
                    text_det_limit_type="max",
                    text_rec_score_thresh=0.0
                )
                if result:
                    for item in result:
                        if isinstance(item, dict):
                            texts = item.get("rec_texts") or []
                            scores = item.get("rec_scores") or []
                            for text, conf in zip(texts, scores):
                                lines.append([None, text, float(conf)])
                        elif isinstance(item, list):
                            for sub in item:
                                if sub and len(sub) >= 2 and isinstance(sub[1], (list, tuple)) and len(sub[1]) >= 2:
                                    text = sub[1][0]
                                    conf = sub[1][1]
                                    lines.append([None, text, float(conf)])
                    if lines:
                        return lines, 0.0
            except Exception as e:
                logger.error(f"PaddleOCR predict error: {e}")

        # RapidOCR 常见入口：直接 __call__(img)
        # 返回通常是 (results, elapsed)，其中 results 形如：
        # [[bbox, text, conf], ...]
        if callable(self.ocr):
            try:
                result = self.ocr(img)
                items = result[0] if isinstance(result, tuple) else result
                if items:
                    for item in items:
                        if not item:
                            continue
                        # 标准 RapidOCR: [bbox, text, conf]
                        if isinstance(item, (list, tuple)) and len(item) >= 3:
                            bbox = item[0]
                            text = item[1]
                            conf = item[2]
                            if text is not None:
                                lines.append([bbox, str(text), float(conf)])
                            continue
                        # 兼容字典结构
                        if isinstance(item, dict):
                            text = item.get("text") or item.get("rec_text")
                            conf = item.get("score") or item.get("conf") or item.get("rec_score") or 0.0
                            bbox = item.get("box") or item.get("bbox")
                            if text is not None:
                                lines.append([bbox, str(text), float(conf)])
                    if lines:
                        return lines, 0.0
            except Exception as e:
                logger.debug(f"OCR callable mode failed: {e}")

        return [], 0.0

class BibStatus(Enum):
    """号码识别状态"""
    UNRECOGNIZED = "unrecognized"
    NEEDS_REVIEW = "needs_review"
    RECOGNIZED = "recognized"


@dataclass(eq=False)
class BibEvidenceCandidate:
    """One bib crop with the same-frame athlete association used to create it."""

    quality: float
    crop: np.ndarray
    frame: Optional[np.ndarray]
    frame_index: int
    capture_time_ms: float
    athlete_bbox: Tuple[int, int, int, int]
    bib_bbox: Tuple[int, int, int, int]
    source: str
    owner_validated: bool = False

    def _legacy_tuple(self):
        return self.quality, self.crop, self.frame, list(self.bib_bbox)

    def __iter__(self):
        return iter(self._legacy_tuple())

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return self._legacy_tuple()[index]

    def detached_copy(self) -> "BibEvidenceCandidate":
        return BibEvidenceCandidate(
            quality=float(self.quality),
            crop=self.crop.copy(),
            frame=None,
            frame_index=int(self.frame_index),
            capture_time_ms=float(self.capture_time_ms),
            athlete_bbox=tuple(int(value) for value in self.athlete_bbox),
            bib_bbox=tuple(int(value) for value in self.bib_bbox),
            source=str(self.source),
            owner_validated=bool(self.owner_validated),
        )


class VLMRateLimitError(Exception):
    """VLM API 频率限制或配额耗尽错误"""
    def __init__(self, message, status_code=429, is_quota_exceeded=False):
        super().__init__(message)
        self.status_code = status_code
        self.is_quota_exceeded = is_quota_exceeded


@dataclass
class CrossingEvent:
    """过线事件"""
    event_id: int
    rank: int
    track_id: int
    cross_time: float
    cross_time_str: str
    cross_realtime: str
    bib_number: Optional[str]
    bib_confidence: Optional[float]
    bib_status: BibStatus
    detection_confidence: float
    position: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    source_id: int = 0
    is_capture: bool = False
    frame: Optional[np.ndarray] = None
    crop: Optional[np.ndarray] = None
    bib_crop: Optional[np.ndarray] = None
    bib_bbox: Optional[List[int]] = None # 新增：号码牌框
    bib_evidence_kind: str = ""
    bib_candidates: List[BibEvidenceCandidate] = field(default_factory=list)
    quality_score: float = 0.0 # 新增：质量分
    is_update_only: bool = False # 新增：标记为仅更新截图数据
    is_info_update: bool = False # 新增：标记为仅更新文本/状态信息
    participant_id: str = ""
    raw_track_ids: Tuple[int, ...] = ()
    passage_index: int = 1
    sport_profile: str = "cycling"
    frame_athletes: Tuple[Dict[str, Any], ...] = ()


@dataclass
class TrackState:
    """跟踪状态（增强版）"""
    prev_x: int
    prev_y: int
    crossed: bool = False
    crossed_time: float = 0.0
    cross_signal_count: int = 0
    cross_signal_last_time: float = 0.0
    event_emitted: bool = False
    best_bib: Optional[str] = None
    best_bib_conf: float = 0.0
    best_bib_crop: Optional[np.ndarray] = None
    best_bib_frame: Optional[np.ndarray] = None
    best_bib_quality: float = 0.0
    best_bib_bbox: List[int] = field(default_factory=list)
    best_athlete_crop: Optional[np.ndarray] = None
    last_seen_time: float = 0.0
    last_ocr_time: float = 0.0
    pending_event_data: Dict[str, Any] = field(default_factory=dict)
    participant_id: str = ""
    participant_ocr_status: str = "PENDING"
    
    # OCR 投票与历史
    bib_candidates: Dict[str, int] = field(default_factory=dict)
    bib_confidences: Dict[str, float] = field(default_factory=dict)
    bib_history: List[str] = field(default_factory=list) # 最近几次识别到的号码
    stable_count: int = 0 # 当前号码稳定出现的次数
    last_bib_rel_pos: Optional[Tuple[float, float]] = None # 号码牌相对于运动员框的相对位置 (rx, ry)
    
    # 【新增】高质量截图缓存
    bib_crops_cache: List[Any] = field(default_factory=list)
    fallback_bib_crops_cache: List[Any] = field(default_factory=list)
    max_cache_size: int = 8
    
    # 运动活性检查
    start_pos: Tuple[int, int] = field(default_factory=lambda: (0, 0))
    start_time: float = 0.0
    max_displacement: float = 0.0
    is_static: bool = False
    trigger_snapshot: bool = False
    pre_enter_time: float = 0.0
    last_positions: List[Tuple[int, int]] = field(default_factory=list)
    
    # 【新增】增强版背景检测与 VLM 状态
    position_history: List[Tuple[float, int, int]] = field(default_factory=list)  # (time, x, y)
    avg_speed: float = 0.0
    merged_into: Optional[int] = None # 如果发生了 ID 跳变，记录合并到的目标 ID
    static_confidence: float = 0.0
    vlm_verified: bool = False
    vlm_is_background: bool = False
    vlm_ocr_pending: bool = False # 是否正在等待 VLM 的 OCR 结果
    last_vlm_bib: Optional[str] = None
    last_vlm_conf: float = 0.0
    last_vlm_time: float = 0.0

    # Evidence from bib detector (not OCR).
    has_bib_box: bool = False
    start_line_dist: Optional[float] = None
    observation_count: int = 0
    synthetic_kind: str = ""


# ============================================================================
# VLM 提示词模板
# ============================================================================

PROMPT_BACKGROUND_CHECK = """你是一个体育赛事图像分析助手。请判断图中主体的类别。

【类别定义】
- ATHLETE: 正在运动中、准备通过终点的参赛选手（跑步、骑行）。特征：身体前倾、发力感、戴头盔/号码布、有明显的行进动态。
- GATE: 赛道设施。特征：横跨赛道、桁架结构、固定不动、挂有计时设备或横幅。
- BYSTANDER: 非参赛人员。特征：站立不动、坐着、背对赛道、无号码布、手持相机、姿态放松。
- OTHER: 无法判断或其他物体（如车辆、流浪小动物、被风吹动的旗帜）。

【判定要点】
1. 动态 vs 静态：运动员必须有明显的行进趋势。
2. 装备：运动员通常佩戴头盔、紧身运动服或挂有号码布。
3. 位置：运动员通常在赛道正中间，而观众在两侧。

【输出格式】
第一行：类别（只写一个词：ATHLETE/GATE/BYSTANDER/OTHER）
第二行：置信度（0.0-1.0之间的小数）
第三行：简短理由（15字以内）"""


PROMPT_COUNT_ATHLETES = """你是一个体育赛事计数助手。请数一下图中【正在通过终点/赛道的运动员】有几人。

【计数规则】
1. 只数正在运动中的参赛选手（穿号码布的）
2. 不要数观众、工作人员、摄影师
3. 如果有人被遮挡但能看到部分身体，也算1人
4. 如果完全看不到人，回答0

【输出格式】
第一行：数字（0/1/2/3/4/5...）
第二行：简短说明每个人的位置

示例输出：
3
左1人穿红衣，中间2人并排"""


# ✅ 通用赛事 OCR 提示词
PROMPT_BIB_OCR_ENHANCED = """你是专业的体育比赛号码识别专家。请仔细识别图中运动员身上的参赛号码。

【本场比赛号码特征】
- 格式规则: 【1-6位数字】，或【1-3个大写字母前缀 + 1-6位数字】
- 示例: 5, 23, 1056, A123, AB7
- 外观特征: 白底黑字或深色底白字，粗体数字，字高通常15-30像素
- 位置特征: 背部、前胸、头盔、大腿或车座下方，可能部分被身体或器材遮挡

【严格识别规则】
1. 字符限制: 只允许识别【大写字母A-Z】和【数字0-9】
2. 严禁识别: "安宁"、"马拉松"、"经校"、"2024"等赛事名称汉字
3. 严禁识别: 赞助商名称、广告文字
4. 如果只看到部分号码(如"A12__"，后两位模糊)，可以推断但必须标注为低置信度
5. 如果完全无法识别，输出 UNREADABLE

【常见错误校正】
- 字母O和数字0易混淆: 如果是开头字母，通常是O；如果在数字中，通常是0
- 数字1和字母I易混淆: 号码中通常是数字1
- 数字5和字母S易混淆: 号码中通常是数字5
- 数字8和字母B易混淆: 号码中通常是数字8

【输出格式】(严格按照以下3行输出)
第一行: 号码 (如 A123、23 或 UNREADABLE)
第二行: 置信度 (0.0-1.0之间的小数)
第三行: 说明 (10字内，如"清晰可见"或"部分遮挡推断")

示例输出1:
A123
0.95
清晰可见

示例输出2:
2350
0.85
部分遮挡推断

示例输出3:
UNREADABLE
0.0
严重模糊无法识别
"""


# ✅ 新版带名单的OCR提示词
def build_ocr_prompt_with_list_enhanced(athlete_list: List[str], only_numeric: bool = False) -> str:
    """
    构建带选手名单的增强版OCR提示词
    """
    if not athlete_list:
        return PROMPT_BIB_OCR_ENHANCED

    # 取前100名选手作为参考（太多会超过token限制）
    sample_bibs = athlete_list[:100] if len(athlete_list) > 100 else athlete_list

    # 按数字排序，方便查找
    try:
        sample_bibs_sorted = sorted(sample_bibs, key=lambda x: (
            ''.join(filter(str.isalpha, x)),  # 先按字母排序
            int(''.join(filter(str.isdigit, x)) or '0')  # 再按数字排序
        ))
    except:
        sample_bibs_sorted = sample_bibs

    # 分组显示（每行10个号码）
    bib_lines = []
    for i in range(0, len(sample_bibs_sorted), 10):
        chunk = sample_bibs_sorted[i:i+10]
        bib_lines.append(", ".join(chunk))
    bib_str = "\n".join(bib_lines)

    # 识别规则
    if only_numeric:
        rule_text = "只允许识别【数字0-9】。严禁识别任何字母、汉字、标点。"
        # 自行车常见 1~500：可能只有 1~3 位数字
        format_text = "格式: 纯数字，允许1-6位，如 7, 12, 125, 2350"
        example = "125"
    else:
        rule_text = "只允许识别【大写字母A-Z】和【数字0-9】。严禁识别汉字、标点。"
        format_text = "格式: 1-6位数字，或1-3个大写字母前缀加1-6位数字"
        example = "A123"

    prompt = f"""你是专业的自行车比赛号码识别专家。请仔细识别图中运动员身上的参赛号码。

【本场比赛参考名单】(共{len(athlete_list)}名选手，显示前{len(sample_bibs)}名)
{bib_str}

【识别优先级】
1. 优先匹配上述名单中的号码
2. 如果看到的号码与名单中某个号码很接近（如只差1个数字），倾向于匹配名单号码
3. 如果完全不在名单中，但符合格式规则，也可以输出

【号码特征】
- {format_text}
- 外观: 白底黑字或深色底白字，粗体数字，字高15-30像素
- 位置: 背部腰际、前胸或车架横杆，可能部分遮挡

【严格规则】
1. {rule_text}
2. 严禁识别赛事名称("安宁"、"马拉松"等汉字)
3. 严禁识别赞助商名称、广告文字
4. 如果部分模糊，可推断但标注低置信度

【常见错误校正】
- O(字母)与0(数字): 开头是字母O，中间是数字0
- I(字母)与1(数字): 号码中通常是数字1
- S(字母)与5(数字): 号码中通常是数字5

【输出格式】(严格3行)
第一行: 号码 (如 {example})
第二行: 置信度 (0.0-1.0)
第三行: 在名单中? (YES/NO/SIMILAR) + 简短说明

示例输出:
{example}
0.92
YES 清晰可见
"""

    return prompt


# ============================================================================
# VLM 结果解析函数
# ============================================================================

def parse_background_result(response: str) -> Tuple[str, float, str]:
    """解析 VLM 背景判定结果"""
    lines = response.strip().split('\n')
    
    category = "OTHER"
    confidence = 0.5
    reason = ""
    
    if len(lines) >= 1:
        cat = lines[0].strip().upper()
        if cat in ["ATHLETE", "GATE", "BYSTANDER", "OTHER"]:
            category = cat
    
    if len(lines) >= 2:
        try:
            # 尝试提取数字
            conf_str = lines[1].strip()
            match = re.search(r'\d+\.?\d*', conf_str)
            if match:
                confidence = float(match.group())
                if confidence > 1.0: confidence /= 100.0
                confidence = max(0.0, min(1.0, confidence))
        except:
            confidence = 0.5
    
    if len(lines) >= 3:
        reason = lines[2].strip()[:20]
    
    return category, confidence, reason


def parse_count_result(response: str) -> Tuple[int, str]:
    """解析人数计数结果"""
    lines = response.strip().split('\n')
    
    count = 1
    description = ""
    
    if len(lines) >= 1:
        match = re.search(r'\d+', lines[0])
        if match:
            count = int(match.group())
            count = max(0, min(10, count))
    
    if len(lines) >= 2:
        description = lines[1].strip()
    
    return count, description


def parse_ocr_result(response: str) -> Tuple[Optional[str], float, str]:
    """解析 VLM OCR 结果"""
    lines = response.strip().split('\n')
    
    bib_number = None
    confidence = 0.0
    note = ""
    
    if len(lines) >= 1:
        raw = lines[0].strip().upper()
        
        # 0. 基础排除
        if raw in ["UNREADABLE", "NO_BIB", "UNCLEAR", "无法识别", "NONE", "NULL", "EMPTY"]:
            return None, 0.0, raw
            
        # 1. 深度清洗：移除干扰词
        # 移除如 "号码：", "BIB:", "BIB NUMBER:", "运动员：" 等
        junk_patterns = [
            r'^号码[：:]\s*', r'^BIB[：:]\s*', r'^NUMBER[：:]\s*', 
            r'^参赛号[：:]\s*', r'^BIB NUMBER[：:]\s*', r'^选手[：:]\s*'
        ]
        clean_raw = raw
        for p in junk_patterns:
            clean_raw = re.sub(p, '', clean_raw, flags=re.IGNORECASE)
            
        # 2. 移除非字母数字字符（保留 A-Z, 0-9）
        # 这一步非常重要，它能把 "号码：A0621" 变成 "A0621"
        clean = ''.join(c for c in clean_raw if c.isalnum() and ord(c) < 128)
        
        # 3. 提取核心号码 (正则表达式匹配 A1234 或 1234 这种模式)
        # Cycling events may use a one-digit bib.
        match = re.search(r'[A-Z]?\d{1,6}', clean)
        if match:
            extracted = match.group()
            if 1 <= len(extracted) <= 8:
                bib_number = extracted
    
    if len(lines) >= 2:
        try:
            # 尝试提取数字
            conf_str = lines[1].strip()
            match = re.search(r'\d+\.?\d*', conf_str)
            if match:
                confidence = float(match.group())
                # 如果模型返回的是 95.5 这种百分比，归一化到 0-1
                if confidence > 1.0: confidence /= 100.0
                confidence = max(0.0, min(1.0, confidence))
        except:
            confidence = 0.5 if bib_number else 0.0
    
    if len(lines) >= 3:
        note = lines[2].strip()
    
    return bib_number, confidence, note


# ============================================================================
# VLM 辅助类
# ============================================================================

class DoubaoVLMAssistant:
    """豆包 VLM 辅助识别"""
    
    def __init__(self, api_key: str, endpoint_id: str = None):
        self.api_key = api_key
        # 优先使用用户提供的 API 格式
        self.api_url = "https://ark.cn-beijing.volces.com/api/v3/responses"
        self.model = endpoint_id or "doubao-seed-1-8-251228"
        self.timeout = 10
    
    def _call_api(self, images: Union[np.ndarray, List[np.ndarray]], prompt: str, max_tokens: int = 100) -> str:
        """调用 API (支持单图或多图)"""
        if isinstance(images, np.ndarray):
            images = [images]
            
        content = []
        for img in images:
            if img is None or img.size == 0: continue
            _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            content.append({
                "type": "input_image", 
                "image_url": f"data:image/jpeg;base64,{img_base64}"
            })
            
        if not content:
            return ""
            
        content.append({
            "type": "input_text", 
            "text": prompt
        })
        
        try:
            payload = {
                "model": self.model,
                "input": [
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            }
            
            logger.debug(f"[VLM-Doubao] 发送请求: model={self.model}, url={self.api_url}")
            
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                res_json = response.json()
                try:
                    # 1. 优先解析 v3/responses 的列表形式 output 结构
                    if 'output' in res_json and isinstance(res_json['output'], list):
                        for item in res_json['output']:
                            if item.get('type') == 'message' and 'content' in item:
                                for content_item in item['content']:
                                    if content_item.get('type') == 'output_text':
                                        return content_item.get('text', '')
                                    elif isinstance(content_item, str):
                                        return content_item
                    
                    # 2. 兼容旧版 output 字典结构
                    if 'output' in res_json and isinstance(res_json['output'], dict):
                        choices = res_json['output'].get('choices', [])
                        if choices:
                            return choices[0]['message']['content']
                            
                    # 3. 尝试从 standard chat completions 结构解析
                    if 'choices' in res_json:
                        return res_json['choices'][0]['message']['content']
                        
                    # 4. 尝试从直接的 message 结构解析
                    if 'message' in res_json:
                        return res_json['message']['content']
                        
                    return str(res_json)
                except Exception as e:
                    logger.warning(f"[VLM-Doubao] 结果解析失败: {e}, 原始结果: {res_json}")
                    return str(res_json)
            elif response.status_code == 429:
                is_quota = "SetLimitExceeded" in response.text or "quota" in response.text.lower()
                raise VLMRateLimitError(f"Doubao API Rate Limit: {response.text}", status_code=429, is_quota_exceeded=is_quota)
            else:
                logger.error(f"[VLM-Doubao] API 错误: {response.status_code} - {response.text}")
                # 如果 404，可能是 endpoint 没对上，或者 model ID 没对上
                if response.status_code == 404:
                    if "responses" in self.api_url:
                        self.api_url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
                        logger.info("[VLM-Doubao] 404 错误，尝试切换到 chat/completions 接口")
                        return self._call_api_legacy(images, prompt, max_tokens)
                    else:
                        logger.warning("[VLM-Doubao] 接口 404，请检查 API Key 和 Endpoint ID 是否正确。")
                return ""
                
        except VLMRateLimitError:
            raise
        except Exception as e:
            logger.error(f"[VLM-Doubao] 请求异常: {e}")
            return ""

    def _call_api_legacy(self, image: np.ndarray, prompt: str, max_tokens: int = 100) -> str:
        """传统的 OpenAI 兼容格式调用"""
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                            {"type": "text", "text": prompt}
                        ]
                    }],
                    "max_tokens": max_tokens
                },
                timeout=self.timeout
            )
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            return ""
        except:
            return ""
    
    def check_background(self, crop: np.ndarray) -> Tuple[bool, float, str]:
        """判断是否为运动员（非背景）"""
        result = self._call_api(crop, PROMPT_BACKGROUND_CHECK, max_tokens=50)
        category, conf, reason = parse_background_result(result)
        is_athlete = (category == "ATHLETE")
        return is_athlete, conf, reason
    
    def count_athletes(self, region: np.ndarray) -> Tuple[int, str]:
        """计数运动员"""
        result = self._call_api(region, PROMPT_COUNT_ATHLETES, max_tokens=80)
        return parse_count_result(result)
    
    def ocr_bib(self, bib_crop: np.ndarray, athlete_list: List[str] = None,
                only_numeric: bool = False, prompt: str = None) -> Tuple[Optional[str], float, str]:
        """
        OCR 号码牌识别（增强版）
        """
        if not prompt:
            if athlete_list:
                # ✅ 使用增强版带名单提示词
                prompt = build_ocr_prompt_with_list_enhanced(athlete_list, only_numeric)
            else:
                # ✅ 使用增强版基础提示词
                prompt = PROMPT_BIB_OCR_ENHANCED

        result = self._call_api(bib_crop, prompt, max_tokens=120)
        return parse_ocr_result(result)

    def compare_persons(self, crops: List[np.ndarray]) -> Tuple[str, float, str]:
        """比对多张图片是否为同一个人"""
        result = self._call_api(crops, PROMPT_PERSON_MATCH, max_tokens=80)
        lines = result.strip().split('\n')
        
        conclusion = "UNCERTAIN"
        confidence = 0.0
        reason = ""
        
        if len(lines) >= 1:
            raw = lines[0].strip().upper()
            if "SAME" in raw: conclusion = "SAME"
            elif "DIFFERENT" in raw: conclusion = "DIFFERENT"
            
        if len(lines) >= 2:
            try:
                # 寻找数字
                match = re.search(r'\d+\.?\d*', lines[1])
                if match:
                    confidence = float(match.group())
                    if confidence > 1.0: confidence /= 100.0
            except:
                confidence = 0.5
                
        if len(lines) >= 3:
            reason = lines[2].strip()
            
        return conclusion, confidence, reason


class QwenVLMAssistant:
    """通义千问 (Qwen-VL) 辅助识别"""
    
    def __init__(self, api_key: str, model: str = "qwen-vl-max"):
        self.api_key = api_key
        self.api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-conversation/generation"
        self.model = model
        self.timeout = 30 # 增加超时时间到 30s
    
    def _call_api(self, images: Union[np.ndarray, List[np.ndarray]], prompt: str, max_tokens: int = 100) -> str:
        """调用 DashScope API (支持单图或多图)"""
        if isinstance(images, np.ndarray):
            images = [images]
            
        content = []
        for img in images:
            if img is None or img.size == 0: continue
            _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            content.append({"image": f"data:image/jpeg;base64,{img_base64}"})
            
        if not content:
            return ""
            
        content.append({"text": prompt})
        
        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "input": {
                        "messages": [{
                            "role": "user",
                            "content": content
                        }]
                    },
                    "parameters": {
                        "max_tokens": max_tokens,
                        "temperature": 0.1
                    }
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                res_data = response.json()
                try:
                    # 鲁棒性：多层级检查
                    if 'output' in res_data and 'choices' in res_data['output']:
                        choice = res_data['output']['choices'][0]
                        if 'message' in choice and 'content' in choice['message']:
                            content = choice['message']['content']
                            if isinstance(content, list) and len(content) > 0 and 'text' in content[0]:
                                return content[0]['text']
                            elif isinstance(content, str):
                                return content
                    return str(res_data)
                except Exception as e:
                    logger.warning(f"[VLM-Qwen] 结果解析异常: {e}")
                    return str(res_data)
            elif response.status_code == 429:
                raise VLMRateLimitError(f"Qwen API Rate Limit: {response.text}", status_code=429)
            else:
                logger.error(f"[VLM-Qwen] API 错误: {response.status_code} {response.text}")
                return ""
                
        except VLMRateLimitError:
            raise
        except Exception as e:
            logger.error(f"[VLM-Qwen] 请求异常: {e}")
            return ""

    def check_background(self, crop: np.ndarray) -> Tuple[bool, float, str]:
        """判断是否为运动员（非背景）"""
        result = self._call_api(crop, PROMPT_BACKGROUND_CHECK, max_tokens=50)
        category, conf, reason = parse_background_result(result)
        is_athlete = (category == "ATHLETE")
        return is_athlete, conf, reason
    
    def count_athletes(self, region: np.ndarray) -> Tuple[int, str]:
        """计数运动员"""
        result = self._call_api(region, PROMPT_COUNT_ATHLETES, max_tokens=80)
        return parse_count_result(result)
    
    def ocr_bib(self, bib_crop: np.ndarray, athlete_list: List[str] = None,
                only_numeric: bool = False, prompt: str = None) -> Tuple[Optional[str], float, str]:
        """OCR 号码牌（增强版）"""
        if not prompt:
            if athlete_list:
                prompt = build_ocr_prompt_with_list_enhanced(athlete_list, only_numeric)
            else:
                prompt = PROMPT_BIB_OCR_ENHANCED

        result = self._call_api(bib_crop, prompt, max_tokens=120)
        return parse_ocr_result(result)


class AsyncVLMPipeline:
    """异步 VLM 处理管道"""
    
    def __init__(self, vlm_assistant: Any, max_workers: int = 4):
        self.vlm = vlm_assistant
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # 结果缓存
        self.background_cache: Set[int] = set()
        self.athlete_cache: Set[int] = set()
        self.ocr_cache: Dict[int, Tuple[str, float]] = {}
        self.match_cache: Dict[str, Tuple[str, float, str]] = {}  # key: "tid1-tid2"
        self.pending_ocr: Set[int] = set() # 正在进行的 OCR 任务
        
        # 速率限制
        self.call_times = deque(maxlen=60)
        self.max_calls_per_minute = 30
        self.cooldown_until = 0.0
        self.is_quota_exhausted = False
        
        self._lock = threading.Lock()
    
    def _can_call(self, is_priority: bool = False) -> bool:
        """检查是否可以调用 API"""
        now = time.time()
        with self._lock:
            # 1. 检查冷却时间
            if now < self.cooldown_until:
                return False
            
            # 2. 如果配额耗尽，彻底停止调用 (除非是极高优先级的调试)
            if self.is_quota_exhausted:
                return False
                
            # 3. 速率限制检查
            while self.call_times and now - self.call_times[0] > 60:
                self.call_times.popleft()
            
            # 非优先级任务限制更严
            limit = self.max_calls_per_minute if is_priority else (self.max_calls_per_minute // 2)
            return len(self.call_times) < limit
    
    def _handle_error(self, e: Exception):
        """处理 VLM 错误并设置冷却"""
        now = time.time()
        with self._lock:
            if isinstance(e, (VLMRateLimitError, SharedVLMRateLimitError)):
                if e.is_quota_exceeded:
                    self.is_quota_exhausted = True
                    self.cooldown_until = now + 300 # 5分钟冷却
                    logger.error("="*50)
                    logger.error("VLM 配额已耗尽或触发安全体验模式限制！")
                    logger.error("请访问火山引擎控制台 -> 模型中心 -> 调整“安全体验模式”或增加配额。")
                    logger.error("系统将暂停非关键 VLM 任务 5 分钟，仅保留核心 OCR 尝试。")
                    logger.error("="*50)
                else:
                    self.cooldown_until = now + 30 # 30秒冷却
                    logger.warning(f"[VLM] 触发频率限制，进入 30s 冷却期。")
            else:
                # 其他异常，小额冷却
                self.cooldown_until = now + 5
    
    def _record_call(self):
        """记录 API 调用"""
        with self._lock:
            self.call_times.append(time.time())
    
    def submit_background_check(self, track_id: int, crop: np.ndarray):
        """提交背景判定任务 (低优先级)"""
        if track_id in self.background_cache or track_id in self.athlete_cache:
            return
        if not self._can_call(is_priority=False):
            return
        
        self._record_call()
        future = self.executor.submit(self.vlm.check_background, crop)
        future.add_done_callback(lambda f: self._on_background_result(track_id, f))
    
    def _on_background_result(self, track_id: int, future):
        """背景判定结果回调"""
        try:
            is_athlete, conf, reason = future.result()
            with self._lock:
                if is_athlete:
                    self.athlete_cache.add(track_id)
                else:
                    self.background_cache.add(track_id)
                    logger.info(f"[VLM] ID {track_id} 判定为背景: {reason}")
        except Exception as e:
            self._handle_error(e)
            logger.error(f"[VLM] 背景判定失败: {e}")
    
    def submit_ocr(self, track_id: int, bib_crop: np.ndarray, athlete_list: List[str] = None, only_numeric: bool = False):
        """提交 OCR 任务 (高优先级)"""
        with self._lock:
            if track_id in self.ocr_cache or track_id in self.pending_ocr:
                return
        
        if not self._can_call(is_priority=True):
            return
        
        with self._lock:
            self.pending_ocr.add(track_id)
            
        self._record_call()
        future = self.executor.submit(self.vlm.ocr_bib, bib_crop, athlete_list, only_numeric)
        future.add_done_callback(lambda f: self._on_ocr_result(track_id, f))
    
    def is_ocr_pending(self, track_id: int) -> bool:
        """检查 OCR 是否正在进行"""
        with self._lock:
            return track_id in self.pending_ocr
    
    def submit_person_match(self, tid1: int, crop1: np.ndarray, tid2: int, crop2: np.ndarray):
        """提交双人对比任务"""
        key = f"{min(tid1, tid2)}-{max(tid1, tid2)}"
        if key in self.match_cache:
            return
        if not self._can_call():
            return
            
        self._record_call()
        future = self.executor.submit(self.vlm.compare_persons, [crop1, crop2])
        future.add_done_callback(lambda f: self._on_match_result(key, f))
        
    def _on_match_result(self, key: str, future):
        """对比结果回调"""
        try:
            conclusion, conf, reason = future.result()
            with self._lock:
                self.match_cache[key] = (conclusion, conf, reason)
                logger.info(f"[VLM-Match] {key} -> {conclusion} (conf={conf:.2f}, {reason})")
        except Exception as e:
            self._handle_error(e)
            logger.error(f"[VLM-Match] 失败: {e}")

    def get_match_result(self, tid1: int, tid2: int) -> Optional[Tuple[str, float, str]]:
        key = f"{min(tid1, tid2)}-{max(tid1, tid2)}"
        with self._lock:
            return self.match_cache.get(key)
    
    def get_status(self) -> dict:
        """获取当前管道状态"""
        with self._lock:
            now = time.time()
            return {
                'pending_ocr': len(self.pending_ocr),
                'cooldown': max(0, int(self.cooldown_until - now)),
                'quota_exhausted': self.is_quota_exhausted,
                'calls_per_minute': len(self.call_times)
            }
    
    def _on_ocr_result(self, track_id: int, future):
        """OCR 结果回调"""
        try:
            text, conf, note = future.result()
            with self._lock:
                if track_id in self.pending_ocr:
                    self.pending_ocr.remove(track_id)
                if text and conf > 0.3:
                    self.ocr_cache[track_id] = (text, conf)
            
            if text:
                logger.info(f"[VLM-OCR] ID {track_id} -> {text} (conf={conf:.2f}, note={note})")
        except Exception as e:
            self._handle_error(e)
            with self._lock:
                if track_id in self.pending_ocr:
                    self.pending_ocr.remove(track_id)
            logger.error(f"[VLM-OCR] ID {track_id} 失败: {e}")
    
    def is_known_background(self, track_id: int) -> bool:
        with self._lock:
            return track_id in self.background_cache
    
    def get_cached_bib(self, track_id: int) -> Optional[Tuple[str, float]]:
        with self._lock:
            return self.ocr_cache.get(track_id)
    
    def shutdown(self):
        """关闭线程池"""
        self.executor.shutdown(wait=False)


# ============================================================================
# 图像增强与 OCR 工具函数
# ============================================================================

# 常见 OCR 错误字符映射
COMMON_OCR_ERRORS = {
    '0': ['O', 'D', 'Q', 'o'],
    '1': ['I', 'L', 'J', 'i', 'l'],
    '2': ['Z', 'z'],
    '4': ['A'],
    '5': ['S', 's'],
    '6': ['G', 'g'],
    '8': ['B', 'b'],
    'A': ['4'],
    'B': ['8'],
    'S': ['5'],
    'Z': ['2'],
    'O': ['0'],
    'I': ['1'],
    'G': ['6'],
    'D': ['0'],
    'Q': ['0']
}

def fuzzy_match_bib(bib: str, athlete_list: set, threshold: float = 0.7) -> Optional[str]:
    """
    更智能的号码模糊匹配
    1. 处理常见 OCR 错误 (1/I, 0/O, 4/A)
    2. 使用 difflib 进行相似度匹配
    """
    if not bib or not athlete_list:
        return None
    
    # 如果已经精准匹配，直接返回
    if bib in athlete_list:
        return bib
        
    # 1. 尝试常见错误替换后的匹配
    for i, char in enumerate(bib):
        if char in COMMON_OCR_ERRORS:
            for replacement in COMMON_OCR_ERRORS[char]:
                candidate = bib[:i] + replacement + bib[i+1:]
                if candidate in athlete_list:
                    return candidate
                    
    # 2. 使用 difflib 寻找最接近的匹配
    matches = difflib.get_close_matches(bib, list(athlete_list), n=1, cutoff=threshold)
    if matches:
        return matches[0]
        
    return None


# ============================================================================
# 几何计算工具函数
# ============================================================================

def point_side_of_line(px: int, py: int, x1: int, y1: int, x2: int, y2: int) -> float:
    """计算点相对于线的位置"""
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def point_to_line_dist(px: int, py: int, x1: int, y1: int, x2: int, y2: int) -> float:
    """计算点到直线的垂直距离"""
    dx = x2 - x1
    dy = y2 - y1
    line_len = np.sqrt(dx * dx + dy * dy)
    if line_len == 0:
        return np.sqrt((px - x1)**2 + (py - y1)**2)
    return abs((x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)) / line_len


def point_line_segment_t(px: int, py: int, x1: int, y1: int, x2: int, y2: int) -> float:
    """
    Return the projection parameter t of point P onto segment A(x1,y1)->B(x2,y2),
    where P_proj = A + t*(B-A). For segment range, t in [0,1].
    """
    dx = x2 - x1
    dy = y2 - y1
    denom = (dx * dx + dy * dy)
    if denom == 0:
        return 0.0
    return ((px - x1) * dx + (py - y1) * dy) / denom


def bbox_iou(a: List[int], b: List[int]) -> float:
    """计算两个边界框的 IoU"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    aw, ah = max(0, ax2 - ax1), max(0, ay2 - ay1)
    bw, bh = max(0, bx2 - bx1), max(0, by2 - by1)
    union = aw * ah + bw * bh - inter
    return (inter / union) if union > 0 else 0.0


def bbox_overlap_over_smaller(a: List[int], b: List[int]) -> float:
    """Return intersection area divided by the smaller box area."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    smaller = min(area_a, area_b)
    return (inter / smaller) if smaller > 0 else 0.0


def read_image_unicode(path: Path) -> Optional[np.ndarray]:
    """支持 Unicode 路径的图像读取"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def write_image_unicode(path: Path, image: np.ndarray) -> bool:
    """支持 Unicode 路径的图像写入"""
    try:
        ext = path.suffix if path.suffix else ".jpg"
        success, encoded = cv2.imencode(ext, image)
        if not success:
            return False
        encoded.tofile(str(path))
        return True
    except Exception:
        return False


# ============================================================================
# 增强版检测器
# ============================================================================

class Detector:
    """
    增强版实时检测器
    
    改进点：
    1. 多维度龙门/背景过滤
    2. 智能并排过线识别
    3. 号码牌驱动检测补全
    4. VLM 辅助判定
    5. 增强 OCR
    """

    def __init__(self, model_path: str, allow_recross: bool = False, source_id: int = 0,
                 model: Optional[YOLO] = None, ocr: Optional[Any] = None, ocr_engine: str = "rapidocr",
                 only_numeric: bool = False, realtime_ocr: bool = False,
                 gate_guard_enabled: bool = True, athlete_validator: Optional[Any] = None,
                 athlete_detector: Optional[Any] = None,
                 performance_profile: str = "auto", sport_profile: str = "cycling",
                 event_settle_seconds: Optional[float] = None,
                 event_profile: Optional[EventProfile] = None,
                 ocr_pipeline_enabled: bool = True):
        self.model_path = model_path
        self.source_id = source_id
        self._model = model
        self.ocr_pipeline_enabled = bool(ocr_pipeline_enabled)
        self._ocr = ocr if self.ocr_pipeline_enabled else None
        self.ocr_engine = ocr_engine
        self.only_numeric = only_numeric # 新增：是否仅允许纯数字号码
        self.realtime_ocr_enabled = bool(realtime_ocr and self.ocr_pipeline_enabled)
        self.enable_gate_guard = bool(gate_guard_enabled) # 龙门安全模式（抑制龙门/拱门误检）
        self._athlete_validator = athlete_validator
        self._athlete_detector = athlete_detector
        self._athlete_detector_failed = False
        normalized_sport_profile = (
            str(sport_profile or "cycling").strip().lower() or "cycling"
        )
        if event_profile is None:
            event_profile = build_sport_event_profile(normalized_sport_profile)
        self.event_profile = event_profile
        self.sport_profile = self.event_profile.name
        self._participant_identity_manager = ParticipantIdentityManager(
            config=IdentityConfig(event_profile_name=self.event_profile.name)
        )
        self._crossing_session_id = f"{self.event_profile.name}:{self.source_id}"
        self._crossing_lifecycle = CrossingLifecycle(
            mode=self.event_profile.crossing_mode,
            min_lap_interval_ms=(
                30_000.0 if self.event_profile.crossing_mode == "multi_lap" else 0.0
            ),
        )
        self._identity_segment_id = 0
        self.event_settle_seconds = (
            None if event_settle_seconds is None else max(0.0, float(event_settle_seconds))
        )
        self.athlete_validator_imgsz = 320
        self.athlete_validator_conf = 0.20
        self.athlete_validator_min_bicycle_area_ratio = 0.05
        self.athlete_validator_bicycle_center_x_min = 0.25
        self.athlete_validator_bicycle_center_x_max = 0.75
        self.athlete_detector_imgsz = 896
        self.athlete_detector_conf = 0.20
        self.athlete_detector_trackside_bottom_ratio = 0.48
        self.speed_skating_primary_bottom_ratio = 0.45
        resolved_profile = resolve_performance_profile(performance_profile)
        self.performance_profile = str(resolved_profile["name"])
        self._merged_pairs = set() # 用于减少重复合并日志

        self.allow_recross = allow_recross

        # 终点线配置
        self._line_pt1 = (0, 950)
        self._line_pt2 = (1920, 950)
        self._roi_polygon: Optional[np.ndarray] = None
        self.timing_enabled = True
        self.enable_roi_filter = False
        self.person_class_ids: Set[int] = {0}
        self.bib_class_ids: Set[int] = {1}
        self.bike_model_mode = False
        # 静态背景过滤：用于屏蔽观众/龙门结构等“几乎不动”的目标，减少无关框与误过线
        # 误判为静态后，只要出现明显位移也会自动恢复，不会长期吞掉真正运动员
        self.enable_static_background_filter = True
        self.enable_semantic_merge = False            # 极限抓人模式：关闭深度语义合并，避免并排吞人
        self.adaptive_frame_skip = bool(resolved_profile["adaptive_frame_skip"])

        # ===== 检测参数（优化版）=====
        self.detection_conf = 0.08  # 马拉松连续人流：继续提高召回，优先不漏人
        self.iou_threshold = 0.65 if source_id == 0 else 0.60  # 马拉松并排人流：提高NMS阈值，减少互相压框
        self.ocr_conf_threshold = 0.55  # ✅ 优化: 拒绝低质量OCR
        self.bib_assignment_min_score_margin = 0.05
        speed_skating_bibs = {
            "helmet",
            "left_thigh",
            "right_thigh",
        }.intersection(self.event_profile.bib_regions)
        self.bib_assignment_min_rel_y = 0.02 if speed_skating_bibs else 0.25
        self.bib_assignment_max_rel_y = 0.98 if speed_skating_bibs else 0.90
        self.bib_assignment_min_overlap = 0.75 if speed_skating_bibs else 0.90
        self.bib_preferred_rel_y_ranges = (
            ((0.02, 0.30), (0.36, 0.98))
            if speed_skating_bibs
            else ((0.28, 0.90),)
        )
        self.profile_min_athlete_area_ratio = 0.0025 if speed_skating_bibs else 0.0
        self.profile_min_athlete_height_ratio = 0.07 if speed_skating_bibs else 0.0
        self.untracked_detection_conf = 0.20
        self.ocr_detected_bib_min_width = 40
        self.ocr_detected_bib_min_height = 32
        self.ocr_detected_bib_min_quality = 0.55
        self.bib_candidate_max_aspect = 2.20
        self.bib_cache_max_median_aspect = 2.05
        self.bib_candidate_max_rel_dx = 0.18
        self.bib_candidate_max_rel_dy = 0.14
        self.bib_nontext_saturation_min = 125.0
        self.bib_nontext_gray_std_max = 58.0
        self.bib_nontext_edge_density_max = 0.18
        self.crossing_direction: Optional[str] = None
        self.cross_confirm_frames = 1          # 过线最少确认帧数（低FPS下避免漏判）
        self.cross_signal_window = 0.45        # 连续过线信号合并窗口（秒）
        self.cross_strong_dist_ratio = 0.30    # 强信号距离比例：越小越“确信过线”
        self.enable_crowd_adaptive = True

        # ===== 背景过滤参数（新增）=====
        self.static_speed_threshold = 1.2      # 优先召回：降低静态误杀，避免把慢速/远处运动员判成背景
        self.static_judge_delay = 1.6          # 拉长观察窗口，减少低FPS时的误判静态
        self.background_aspect_min = 0.25      # 宽高比下限
        self.background_aspect_max = 6.0       # 宽高比上限
        self.background_area_ratio_max = 0.15  # 面积比上限

        # ===== 并排过线参数（优化）=====
        self.parallel_center_min = 24          # 收紧并排误并，减少一框多人
        self.parallel_center_ratio = 0.18      # 更保守中心距比例
        self.parallel_time_window = 0.15
        
        # ===== 去重参数 =====
        self.dedup_window = 2.5                # 优先不漏人：缩短去重时间窗，避免相邻选手被误并
        self.center_dup_threshold = 15 if source_id == 0 else 20 # 进一步放宽距离阈值
        self.iou_dup_threshold = 0.65 if source_id == 0 else 0.60 # 降低 IoU 阈值以捕捉更多重复
        
        # 同帧去重
        self.same_frame_iou_threshold = 0.72
        self.same_frame_center_ratio = 0.16
        self.same_frame_size_diff = 0.35
        self.merge_iou_threshold = 0.94
        self.merge_center_ratio = 0.03
        self.merge_center_min = 5
        self.merge_size_diff = 0.10
        self.enable_wbf_merge = False          # 线上优先保一人一框，默认关闭 WBF 合框

        # 体型过滤
        self.aspect_min = 0.22
        self.aspect_max = 5.0
        self.min_bh = 8
        self.min_bw = 6
        # ROI 边缘收紧：>0 表示“离 ROI 边缘太近也算无效”（用于剔除边缘噪声）
        self.roi_edge_margin = 0
        # ROI 外侧容忍：允许目标中心点略微在 ROI 外（像素距离），用于解决 ROI 画得过紧导致“有人跑过但没框住”
        # 注意：这个值越大，越容易把 ROI 外的背景也当作人；默认 25px 属于保守放宽
        self.roi_outside_tolerance = 25

        # Finish-line segment gating (filter spectators / side area targets).
        # Only active when timing_enabled + finish line is configured.
        self.enable_finish_segment_filter = True
        self.finish_segment_margin_px = 60.0
        # Speed-skating finishers can cross just beyond a manually drawn line
        # endpoint, so keep the endpoint margin instead of cancelling it out.
        self.finish_segment_end_shrink_px = 0.0 if speed_skating_bibs else 140.0

        # ===== 性能优化参数 (关键：解决 0.8 FPS 问题) =====
        self.frame_skip = 1                    # 默认不跳帧
        self.last_process_time = 0
        self.last_frame_metrics: Dict[str, Any] = {
            "raw_bikes": 0,
            "raw_bibs": 0,
            "validated_tracks": 0,
            "synthetic_tracks": 0,
            "raw_tracks": 0,
            "participants": 0,
            "track_fragments_merged": 0,
            "identity_ambiguities": 0,
            "roi_auto_disabled": False,
            "inference_ms": 0.0,
            "postprocess_ms": 0.0,
            "total_ms": 0.0,
        }
        self.last_rejected_candidates: List[Dict[str, Any]] = []
        self.max_athletes_per_frame = 24       # 马拉松场景提高上限，避免人群被截断
        self.ocr_queue_limit = 20              # OCR 队列限制，超过则丢弃
        self.process_imgsz = int(resolved_profile["process_imgsz"])
        logger.info(
            f"[Detector-{self.source_id}] 性能档: {self.performance_profile}, "
            f"imgsz={self.process_imgsz}, adaptive_skip={self.adaptive_frame_skip}"
        )

        # 抓人优先：提高每帧允许处理的运动员上限，减少“人太多被截断导致漏框”
        # 同时显式设置 OCR 队列上限（历史上这里容易被注释串行影响）
        self.max_athletes_per_frame = 40
        self.ocr_queue_limit = 20

        self._base_parallel_center_min = self.parallel_center_min
        self._base_parallel_center_ratio = self.parallel_center_ratio
        self._base_parallel_time_window = self.parallel_time_window
        self._base_same_frame_iou_threshold = self.same_frame_iou_threshold
        self._base_iou_dup_threshold = self.iou_dup_threshold
        self._base_center_dup_threshold = self.center_dup_threshold
        self._base_max_athletes_per_frame = self.max_athletes_per_frame
        self._base_cross_confirm_frames = self.cross_confirm_frames

        self._crowd_level = "sparse"
        self._crowd_score = 0.0
        self._crowd_score_window: deque = deque(maxlen=30)
        self._crowd_pending_level: Optional[str] = None
        self._crowd_pending_count = 0
        self._crowd_hold_frames = 15
        self._crowd_enter_medium = 0.35
        self._crowd_exit_medium = 0.25
        self._crowd_enter_dense = 0.62
        self._crowd_exit_dense = 0.52
        self._last_crowd_stats: Dict[str, Any] = {"level": self._crowd_level, "score": 0.0}

        # 状态
        self._track_states: Dict[int, TrackState] = {}
        self._participant_ocr_states: Dict[str, Any] = {}
        self._finished_bibs: Dict[str, Any] = {} # 记录 bib/digits 去重信息（时间、前缀、bbox）
        self._recent_events: List[Dict[str, Any]] = []
        self._event_id = 0
        self._start_time = None
        self._frame_count = 0
        self._ocr_attempts = 0
        self._class_map_logged = False
        self._fallback_class_logged = False
        self._roi_overfilter_streak = 0
        self._empty_athlete_streak = 0
        self._under_detect_streak = 0
        self._static_filter_cooldown_frames = 0
        self._noid_track_pool: Dict[int, Tuple[int, int, float]] = {}
        self._noid_next_track_id = 900000
        self._split_track_pool: Dict[int, Tuple[int, int, float]] = {}
        self._split_next_track_id = 970000
        self._athlete_detector_track_pool: Dict[int, Dict[str, Any]] = {}
        self._athlete_detector_next_track_id = 920000
        self._forced_bib_rescue_last: Dict[Tuple[int, int], float] = {}
        self._wide_split_last: Dict[Any, float] = {}

        # 回调
        self._on_crossing: Optional[Callable[[CrossingEvent], None]] = None
        self._on_bib_update: Optional[Callable[[int, str, float, BibStatus, int], None]] = None

        # 线程锁
        self._lock = threading.RLock()
        self._raw_model_boxes: List[Tuple[int, float, List[float]]] = []
        self._raw_detection_capture_registered = False
        self._register_raw_detection_capture()
        self._initialize_athlete_detector()

        # 选手名单
        self._athlete_list: set = set()
        self._athlete_summary: Dict[str, Any] = {}
        self._bib_ranges: List[Dict[str, Any]] = []
        
        # 清理配置
        self._cleanup_interval = 60.0
        self._stale_threshold = 300.0
        self._last_cleanup_time = 0.0

        # OCR 异步队列
        self._ocr_queue = queue.LifoQueue(maxsize=50) # 降低队列深度，减少内存占用
        self._ocr_running = False
        self._ocr_thread = None
        
        # ===== VLM 辅助（新增）=====
        self._vlm_enabled = False
        self._vlm_assistant: Optional[Union[DoubaoVLMAssistant, QwenVLMAssistant, OpenAIVLMAssistant]] = None
        self._vlm_pipeline: Optional[AsyncVLMPipeline] = None
        self._vlm_athlete_list: List[str] = []
        
        # 号码牌驱动检测
        self.bib_driven_detection = True
        
        # ===== 号码校验规则 (针对 AXXXX 格式优化) =====
        self.bib_regex = r'^[A-Z]*[0-9]{1,6}$'
        self.bib_regex_strict = False
        self.bib_prefix_required = False

        if self._ocr is not None:
            self.start_ocr_worker()

    def _register_raw_detection_capture(self) -> None:
        """Keep raw detections before tracking drops fast, small BIB boxes."""
        add_callback = getattr(self._model, "add_callback", None)
        if not callable(add_callback):
            return
        try:
            add_callback("on_predict_postprocess_end", self._capture_raw_model_boxes)
            self._raw_detection_capture_registered = True
        except Exception as exc:
            logger.debug(f"[Detector-{self.source_id}] Raw detection capture unavailable: {exc}")

    def _initialize_athlete_detector(self) -> None:
        """Load a sibling generic YOLO11 model for speed-skating person recovery."""
        if self.sport_profile != "speed_skating" or self._athlete_detector is not None:
            return

        names = getattr(self._model, "names", None)
        if (not names) and hasattr(self._model, "model"):
            names = getattr(self._model.model, "names", None)
        if isinstance(names, dict):
            normalized_names = {
                int(index): str(label).strip().lower()
                for index, label in names.items()
                if str(index).lstrip("-").isdigit()
            }
        elif isinstance(names, (list, tuple)):
            normalized_names = {index: str(label).strip().lower() for index, label in enumerate(names)}
        else:
            normalized_names = {}
        if (
            len(normalized_names) >= 20
            and normalized_names.get(0) == "person"
            and normalized_names.get(1) == "bicycle"
        ):
            logger.info(f"[Detector-{self.source_id}] Primary COCO YOLO model already provides person detection")
            return

        model_root = Path(self.model_path).expanduser().resolve().parent
        detector_path = resolve_athlete_detector_model("", [model_root])
        if detector_path is None:
            logger.warning(
                f"[Detector-{self.source_id}] YOLO11 person model not found; using the primary model only"
            )
            return

        try:
            self._athlete_detector = YOLO(str(detector_path))
            logger.info(
                f"[Detector-{self.source_id}] Speed-skating person detector ready: {detector_path.name}"
            )
        except Exception as exc:
            self._athlete_detector = None
            self._athlete_detector_failed = True
            logger.warning(
                f"[Detector-{self.source_id}] Person detector unavailable; using the custom model only: {exc}"
            )

    def _capture_raw_model_boxes(self, predictor: Any) -> None:
        captured: List[Tuple[int, float, List[float]]] = []
        try:
            for result in getattr(predictor, "results", []) or []:
                boxes = getattr(result, "boxes", None)
                if boxes is None:
                    continue
                for index in range(len(boxes)):
                    cls_value = boxes.cls[index]
                    conf_value = boxes.conf[index]
                    xyxy_value = boxes.xyxy[index]
                    cls_id = int(cls_value.item()) if hasattr(cls_value, "item") else int(cls_value)
                    confidence = float(conf_value.item()) if hasattr(conf_value, "item") else float(conf_value)
                    xyxy = xyxy_value.tolist() if hasattr(xyxy_value, "tolist") else list(xyxy_value)
                    captured.append((cls_id, confidence, [float(value) for value in xyxy]))
        except Exception as exc:
            logger.debug(f"[Detector-{self.source_id}] Raw detection capture failed: {exc}")
            captured = []
        self._raw_model_boxes = captured

    # =========================================================================
    # VLM 配置
    # =========================================================================
    
    def enable_vlm(self, api_key: str, endpoint_id: str = None, max_calls_per_minute: int = 60,
                   model_type: str = "doubao", base_url: str = None):
        """启用 VLM 辅助"""
        if model_type == "doubao":
            self._vlm_assistant = DoubaoVLMAssistant(api_key, endpoint_id)
        elif model_type == "qwen":
            self._vlm_assistant = QwenVLMAssistant(api_key, endpoint_id or "qwen-vl-max")
        elif model_type == "openai":
            self._vlm_assistant = OpenAIVLMAssistant(
                api_key, endpoint_id or "gpt-4.1-mini", base_url=base_url
            )
        else:
            logger.error(f"[Detector-{self.source_id}] 不支持的 VLM 类型: {model_type}")
            return

        self._vlm_pipeline = AsyncVLMPipeline(self._vlm_assistant)
        self._vlm_pipeline.max_calls_per_minute = max_calls_per_minute
        self._vlm_enabled = True
        logger.info(f"[Detector-{self.source_id}] VLM 辅助已启用 ({model_type}, 限速: {max_calls_per_minute}/min)")
    
    def disable_vlm(self):
        """禁用 VLM 辅助"""
        if self._vlm_pipeline:
            self._vlm_pipeline.shutdown()
        self._vlm_enabled = False
        self._vlm_assistant = None
        self._vlm_pipeline = None
        logger.info(f"[Detector-{self.source_id}] VLM 辅助已禁用")

    # =========================================================================
    # 背景检测（增强版）
    # =========================================================================
    
    def _is_likely_background(self, state: TrackState, bbox: List[int],
                               current_time: float, frame_hw: Tuple[int, int]) -> Tuple[bool, str]:
        """
        多维度背景判定（龙门、拱门、观众等）
        """
        # 如果 VLM 已经确认是背景，直接返回
        if state.vlm_is_background:
            return True, "vlm_confirmed"

        h, w = frame_hw
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        track_duration = current_time - state.start_time
        
        if bw <= 0 or bh <= 0:
            return False, ""

        # 热修：优先避免误杀运动员（先保证刷新与不漏人）
        if state.best_bib and state.best_bib_conf >= 0.65:
            return False, ""
        # 注意：不再以 bib_crops_cache 作为放行条件，避免龙门因兜底裁剪被长期放行
        if track_duration < max(1.0, self.static_judge_delay):
            return False, ""
        
        # === 维度1：宽高比异常 ===
        aspect = bh / bw
        # 自行车运动员通常 aspect 在 0.5 - 2.5 之间。
        # 如果非常细长（如电线杆）或非常宽（如横梁），可能是背景。
        if aspect < 0.2 or aspect > 8.0:
            return True, f"aspect={aspect:.2f}"
        
        # === 维度2：尺寸异常（大面积+宽扁 = 龙门）===
        area_ratio = (bw * bh) / (w * h)
        if area_ratio > self.background_area_ratio_max and aspect < 0.8:
            return True, f"large_wide={area_ratio:.2%}"
        
        # === 维度3：位置固定性（核心）===
        if track_duration > self.static_judge_delay:
            # 更新位置历史
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            state.position_history.append((current_time, cx, cy))
            # 只保留最近2秒
            state.position_history = [(t, x, y) for t, x, y in state.position_history
                                       if current_time - t <= 2.0]
            
            if len(state.position_history) >= 5:
                positions = state.position_history
                total_dist = sum(
                    np.sqrt((positions[i][1] - positions[i-1][1])**2 +
                            (positions[i][2] - positions[i-1][2])**2)
                    for i in range(1, len(positions))
                )
                time_span = positions[-1][0] - positions[0][0]
                state.avg_speed = total_dist / time_span if time_span > 0 else 0
                
                # 速度太慢 → 累积静止置信度
                # 优化：如果是极低速（可能是检测抖动），增加权重
                if state.avg_speed < 0.35:
                    state.static_confidence += 0.08
                elif state.avg_speed < self.static_speed_threshold:
                    state.static_confidence += 0.04
                elif state.avg_speed > 15.0:
                    state.static_confidence = max(0, state.static_confidence - 0.35)
                
                if state.static_confidence > 0.92:
                    return True, f"static(speed={state.avg_speed:.1f}px/s)"
        
        # === 维度4：垂直支柱判定 (侧边固定高瘦物体) ===
        # 如果在画面两侧，且高度很高，且基本不动
        if (x1 < w * 0.12 or x2 > w * 0.88) and bh > h * 0.42 and state.avg_speed < 2.0:
            state.static_confidence += 0.05
            if state.static_confidence > 0.90:
                return True, "side_pillar"

        # === 维度5：位于画面顶部的宽物体 ===
        center_y = (y1 + y2) // 2
        if center_y < h * 0.22 and bw > w * 0.34:
            state.static_confidence += 0.05
            if state.static_confidence > 0.90:
                return True, "top_wide"
        
        # === 维度6：大面积且相对静止 (典型如广告牌、龙门架) ===
        if (area_ratio > 0.28 and state.static_confidence > 0.92 and track_duration > 3.5 and state.avg_speed < 0.5 and center_y < h * 0.45):
            return True, f"large_static(area={area_ratio:.1%})"
        
        return False, ""

    # =========================================================================
    # 运动员去重逻辑
    # =========================================================================

    def _iou_dedup_athletes(self, athletes: List[dict], tid_bib_centers: Dict[int, List]) -> List[dict]:
        """基础 IoU 去重逻辑 (作为 WBF 失败时的 Fallback)"""
        deduped = []
        skip_idx = set()
        for i in range(len(athletes)):
            if i in skip_idx: continue
            curr = athletes[i]
            for j in range(i + 1, len(athletes)):
                if j in skip_idx: continue
                other = athletes[j]
                iou = bbox_iou(curr['bbox'], other['bbox'])
                if iou > 0.88:
                    cx_curr = curr.get('center_x', (curr['bbox'][0] + curr['bbox'][2]) * 0.5)
                    cy_curr = curr.get('center_y', (curr['bbox'][1] + curr['bbox'][3]) * 0.5)
                    cx_other = other.get('center_x', (other['bbox'][0] + other['bbox'][2]) * 0.5)
                    cy_other = other.get('center_y', (other['bbox'][1] + other['bbox'][3]) * 0.5)
                    dx = abs(cx_curr - cx_other)
                    dy = abs(cy_curr - cy_other)
                    min_w = max(1.0, min(curr['bbox'][2] - curr['bbox'][0], other['bbox'][2] - other['bbox'][0]))
                    min_h = max(1.0, min(curr['bbox'][3] - curr['bbox'][1], other['bbox'][3] - other['bbox'][1]))

                    if dx > max(10.0, min_w * 0.18) or dy > max(12.0, min_h * 0.18):
                        continue

                    skip_idx.add(j)
                    tid_j = other['track_id']
                    tid_curr = curr['track_id']
                    if tid_j != -1 and tid_j in tid_bib_centers:
                        bibs_to_move = tid_bib_centers.pop(tid_j)
                        if tid_curr != -1:
                            tid_bib_centers.setdefault(tid_curr, []).extend(bibs_to_move)
            deduped.append(curr)
        return deduped

    def _is_duplicate_unknown_event_bbox(
        self,
        current_bbox: List[int],
        previous_bbox: List[int],
        time_diff: float,
        current_bib_evidence_kind: str = "",
        previous_bib_evidence_kind: str = "",
        current_participant_id: str = "",
        previous_participant_id: str = "",
    ) -> bool:
        """Match nested detector boxes only at the final event boundary."""
        current_participant_id = str(current_participant_id or "").strip()
        previous_participant_id = str(previous_participant_id or "").strip()
        if (
            current_participant_id
            and previous_participant_id
            and current_participant_id != previous_participant_id
        ):
            return False

        has_detected_bib_evidence = "detected" in {
            str(current_bib_evidence_kind or "").strip().lower(),
            str(previous_bib_evidence_kind or "").strip().lower(),
        }
        if time_diff < 0.0 or time_diff >= 1.20:
            return False

        cx1, cy1, cx2, cy2 = [float(value) for value in current_bbox]
        px1, py1, px2, py2 = [float(value) for value in previous_bbox]
        current_w = max(1.0, cx2 - cx1)
        current_h = max(1.0, cy2 - cy1)
        previous_w = max(1.0, px2 - px1)
        previous_h = max(1.0, py2 - py1)
        min_w = min(current_w, previous_w)
        min_h = min(current_h, previous_h)

        intersection_w = max(0.0, min(cx2, px2) - max(cx1, px1))
        intersection_h = max(0.0, min(cy2, py2) - max(cy1, py1))
        intersection = intersection_w * intersection_h
        containment = intersection / max(1.0, min(current_w * current_h, previous_w * previous_h))
        center_dx = abs((cx1 + cx2 - px1 - px2) * 0.5)
        bottom_diff = abs(cy2 - py2)
        width_ratio = max(current_w, previous_w) / min_w

        base_match = (
            containment >= 0.85
            and width_ratio <= 1.40
            and center_dx <= max(16.0, min_w * 0.16)
            and bottom_diff <= max(12.0, min_h * 0.08)
        )
        # A fragmented track can change scale between adjacent crossing frames.
        scale_changed_match = (
            time_diff < 0.65
            and containment >= 0.90
            and width_ratio <= 1.40
            and center_dx <= max(16.0, min_w * 0.16)
            and bottom_diff <= max(20.0, min_h * 0.10)
        )
        if has_detected_bib_evidence:
            return scale_changed_match
        return base_match or scale_changed_match

    # =========================================================================
    # 并排过线判定
    # =========================================================================
    
    def _should_merge_athletes(self, athlete_a: dict, athlete_b: dict,
                               tid_bib_centers: Dict[int, List]) -> bool:
        """智能合并判定 (增加 VLM 异步辅助)"""
        tid1, tid2 = athlete_a.get('track_id', -1), athlete_b.get('track_id', -1)
        b1, b2 = athlete_a['bbox'], athlete_b['bbox']
        
        # 【新增】状态层面的合并记录检查
        with self._lock:
            state1 = self._track_states.get(tid1)
            state2 = self._track_states.get(tid2)
            if state1 and state1.merged_into == tid2: return True
            if state2 and state2.merged_into == tid1: return True
        
        iou = bbox_iou(b1, b2)
        dx = abs(athlete_a['center_x'] - athlete_b['center_x'])
        dy = abs(athlete_a['center_y'] - athlete_b['center_y'])

        # 马拉松并排防误合并：仅在“极高重叠 + 极近中心”时才允许走后续合并
        # 避免 2~4 人并排经过时被深度合并吞成 1 人
        wa = max(1.0, float(b1[2] - b1[0]))
        wb = max(1.0, float(b2[2] - b2[0]))
        ha = max(1.0, float(b1[3] - b1[1]))
        hb = max(1.0, float(b2[3] - b2[1]))
        min_w = min(wa, wb)
        min_h = min(ha, hb)
        if iou < 0.90:
            return False
        if dx > max(8.0, min_w * 0.10) or dy > max(8.0, min_h * 0.10):
            return False
        
        # 1. 号码牌驱动判定 (同步)
        bibs_a = tid_bib_centers.get(tid1, [])
        bibs_b = tid_bib_centers.get(tid2, [])
        
        if bibs_a and bibs_b:
            for ba in bibs_a:
                for bb in bibs_b:
                    bib_dist = np.sqrt((ba[0] - bb[0])**2 + (ba[1] - bb[1])**2)
                    # 更保守：仅在极近 + 高重叠时才允许触发合并
                    if bib_dist < 12 and iou > 0.93:
                        return True
        
        # 2. VLM 异步判定结果检查 (异步)
        if self._vlm_enabled and self._vlm_pipeline and tid1 != -1 and tid2 != -1:
            match_res = self._vlm_pipeline.get_match_result(tid1, tid2)
            if match_res:
                conclusion, conf, reason = match_res
                if conclusion == "SAME" and conf > 0.8:
                    logger.info(f"[Detector-{self.source_id}] VLM 确认 ID {tid1} 和 ID {tid2} 是同一个人: {reason}")
                    return True
            else:
                # 如果目前没有结果，但两者“值得怀疑”，则提交任务
                # 条件：中等重叠，或者距离极近但无号码
                if 0.1 < iou < 0.6 or (iou == 0 and dx < 80 and dy < 80):
                    # 提交对比任务
                    h, w = self._last_frame_shape if hasattr(self, '_last_frame_shape') else (1080, 1920)
                    frame = self._last_frame if hasattr(self, '_last_frame') else None
                    if frame is not None:
                        p_pad = 10
                        crop1 = frame[max(0, int(b1[1]-p_pad)):min(h, int(b1[3]+p_pad)), 
                                      max(0, int(b1[0]-p_pad)):min(w, int(b1[2]+p_pad))].copy()
                        crop2 = frame[max(0, int(b2[1]-p_pad)):min(h, int(b2[3]+p_pad)), 
                                      max(0, int(b2[0]-p_pad)):min(w, int(b2[2]+p_pad))].copy()
                        self._vlm_pipeline.submit_person_match(tid1, crop1, tid2, crop2)
        
        # 2.5 历史号码匹配判定 (增强：必须前缀一致)
        if tid1 != -1 and tid2 != -1:
            with self._lock:
                state1 = self._track_states.get(tid1)
                state2 = self._track_states.get(tid2)
                bib1 = state1.best_bib if state1 else None
                bib2 = state2.best_bib if state2 else None
                
                if bib1 and bib2:
                    # 如果两个 ID 都有号码且号码不同，绝对不合并
                    if bib1 != bib2:
                        # 模糊匹配检查：如果数字部分相同，可能还是同一个人（OCR 误读字母）
                        curr_digits = "".join(filter(str.isdigit, bib1))
                        other_digits = "".join(filter(str.isdigit, bib2))
                        if curr_digits != other_digits:
                            logger.info(f"[Detector] ID {tid1}({bib1}) 与 ID {tid2}({bib2}) 号码不同，禁止合并")
                            return False
                    
                    # 获取前缀和数字
                    m1 = re.match(r'^([A-Z]*)([0-9]+)$', bib1)
                    m2 = re.match(r'^([A-Z]*)([0-9]+)$', bib2)
                    
                    if m1 and m2:
                        p1, d1 = m1.groups()
                        p2, d2 = m2.groups()
                        
                        # 如果数字相同但前缀不同 (如 A3269 vs B3269)，严禁合并
                        if d1 == d2 and p1 != p2:
                            logger.warning(f"[Detector] ID {tid1}({bib1}) 与 ID {tid2}({bib2}) 前缀冲突，严禁合并")
                            return False
                            
                        # 如果前缀和数字都相同，允许合并
                        if d1 == d2 and p1 == p2:
                            logger.info(f"[Detector] 号码完全一致，合并 ID {tid2} -> {tid1} ({bib1})")
                            return True
                    elif bib1 == bib2:
                        return True
        
        # 3. 增加号码牌物理距离判定
        if bibs_a and bibs_b:
            # 计算两组号码牌之间的最小距离
            min_dist = 9999
            for ba in bibs_a:
                for bb in bibs_b:
                    d = np.sqrt((ba[0] - bb[0])**2 + (ba[1] - bb[1])**2)
                    if d < min_dist: min_dist = d
            
            if min_dist > 35: # 号码牌距离超过 35 像素，判定为不同人
                return False

        # 4. 基础物理空间判定
        avg_width = (b1[2] - b1[0] + b2[2] - b2[0]) / 2
        parallel_threshold = max(self.parallel_center_min, int(avg_width * self.parallel_center_ratio))
        min_w = max(1.0, min(float(b1[2] - b1[0]), float(b2[2] - b2[0])))
        min_h = max(1.0, min(float(b1[3] - b1[1]), float(b2[3] - b2[1])))
        
        if dy > 50 and iou < 0.85:
            return False
            
        # P0防漏：无号码/无VLM强证据时，几何合并条件再收紧，避免并排多人被误并
        strict_iou_thr = max(0.93, self.same_frame_iou_threshold + 0.08)
        if iou >= strict_iou_thr:
            if dx >= max(8.0, min_w * 0.16, parallel_threshold * 0.7):
                return False
            if dy >= max(8.0, min_h * 0.16):
                return False
            return True
        
        return False

    # =========================================================================
    # 号码牌驱动检测补全
    # =========================================================================
    
    def _assign_bibs_to_athletes(self, athletes: List[dict], bibs: List[dict]) -> Tuple[Dict[int, List[int]], Set[int]]:
        """Assign each bib detection to at most one physically plausible athlete."""
        assignments: Dict[int, List[int]] = {}
        unmatched = set(range(len(bibs)))

        for bib_idx, bib in enumerate(bibs):
            bib_bbox = bib.get('bbox') or []
            if len(bib_bbox) != 4:
                continue

            bx1, by1, bx2, by2 = [float(v) for v in bib_bbox]
            if bx2 <= bx1 or by2 <= by1:
                continue
            bib_w = bx2 - bx1
            bib_h = by2 - by1
            bib_area = bib_w * bib_h
            bib_aspect = bib_w / bib_h
            if bib_aspect < 0.35 or bib_aspect > self.bib_candidate_max_aspect:
                continue
            bib_cx = float(bib.get('center_x', (bx1 + bx2) * 0.5))
            bib_cy = float(bib.get('center_y', (by1 + by2) * 0.5))
            bib_conf = float(bib.get('conf', 0.0) or 0.0)

            best_athlete_idx = None
            best_score = -1.0
            second_best_score = -1.0
            for athlete_idx, athlete in enumerate(athletes):
                athlete_bbox = athlete.get('bbox') or []
                if len(athlete_bbox) != 4:
                    continue

                ax1, ay1, ax2, ay2 = [float(v) for v in athlete_bbox]
                aw = max(1.0, ax2 - ax1)
                ah = max(1.0, ay2 - ay1)

                if not (ax1 <= bib_cx <= ax2 and ay1 <= bib_cy <= ay2):
                    continue

                ix1 = max(ax1, bx1)
                iy1 = max(ay1, by1)
                ix2 = min(ax2, bx2)
                iy2 = min(ay2, by2)
                overlap_ratio = (max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)) / bib_area
                if overlap_ratio < self.bib_assignment_min_overlap:
                    continue

                rel_x = (bib_cx - ax1) / aw
                rel_y = (bib_cy - ay1) / ah
                if not (0.08 <= rel_x <= 0.92):
                    continue
                if not (self.bib_assignment_min_rel_y <= rel_y <= self.bib_assignment_max_rel_y):
                    continue

                bib_area_ratio = bib_area / (aw * ah)
                bib_width_ratio = bib_w / aw
                bib_height_ratio = bib_h / ah
                if not (0.002 <= bib_area_ratio <= 0.16):
                    continue
                if not (0.08 <= bib_width_ratio <= 0.80):
                    continue
                if not (0.035 <= bib_height_ratio <= 0.34):
                    continue

                center_score = max(0.0, 1.0 - abs(rel_x - 0.5) * 2.0)
                vertical_distance = min(
                    0.0
                    if lower <= rel_y <= upper
                    else min(abs(rel_y - lower), abs(rel_y - upper))
                    for lower, upper in self.bib_preferred_rel_y_ranges
                )
                vertical_score = max(0.0, 1.0 - vertical_distance * 4.0)

                athlete_cx = float(athlete.get('center_x', (ax1 + ax2) * 0.5))
                athlete_cy = float(athlete.get('center_y', (ay1 + ay2) * 0.5))
                normalized_dist = np.hypot(bib_cx - athlete_cx, bib_cy - athlete_cy) / max(aw, ah)
                distance_score = float(np.exp(-normalized_dist * 3.0))

                history_score = 0.5
                tid = athlete.get('track_id', -1)
                with self._lock:
                    state = self._track_states.get(tid)
                    if state and state.last_bib_rel_pos:
                        last_rx, last_ry = state.last_bib_rel_pos
                        history_score = float(np.exp(-np.hypot(rel_x - last_rx, rel_y - last_ry) * 5.0))

                score = (
                    overlap_ratio * 0.35
                    + center_score * 0.20
                    + vertical_score * 0.15
                    + distance_score * 0.15
                    + history_score * 0.10
                    + bib_conf * 0.05
                )
                if score > best_score:
                    second_best_score = best_score
                    best_score = score
                    best_athlete_idx = athlete_idx
                elif score > second_best_score:
                    second_best_score = score

            score_margin = best_score - second_best_score
            if (
                best_athlete_idx is not None
                and (
                    second_best_score < 0.0
                    or score_margin >= self.bib_assignment_min_score_margin
                )
            ):
                assignments.setdefault(best_athlete_idx, []).append(bib_idx)
                unmatched.discard(bib_idx)

        return assignments, unmatched

    def _allocate_athlete_detector_track_id(
        self,
        bbox: List[int],
        current_time: float,
        frame_shape: Tuple[int, int],
        used_track_ids: Set[int],
    ) -> int:
        """Associate generic-person detections with a short velocity-aware track."""
        height, width = int(frame_shape[0]), int(frame_shape[1])
        x1, y1, x2, y2 = [float(value) for value in bbox]
        center_x = (x1 + x2) * 0.5
        center_y = (y1 + y2) * 0.5
        box_width = max(1.0, x2 - x1)
        box_height = max(1.0, y2 - y1)

        stale_track_ids = [
            track_id
            for track_id, state in self._athlete_detector_track_pool.items()
            if current_time - float(state.get("time", 0.0)) > 0.75
        ]
        for track_id in stale_track_ids:
            self._athlete_detector_track_pool.pop(track_id, None)

        best_track_id = None
        best_score = float("inf")
        base_match_distance = max(90.0, 0.11 * max(width, height))
        for track_id, state in self._athlete_detector_track_pool.items():
            if track_id in used_track_ids:
                continue
            elapsed = current_time - float(state.get("time", 0.0))
            if elapsed < 0.0 or elapsed > 0.75:
                continue

            previous_width = max(1.0, float(state.get("width", box_width)))
            previous_height = max(1.0, float(state.get("height", box_height)))
            width_ratio = max(previous_width, box_width) / min(previous_width, box_width)
            height_ratio = max(previous_height, box_height) / min(previous_height, box_height)
            if width_ratio > 2.8 or height_ratio > 2.8:
                continue

            predicted_x = float(state.get("center_x", center_x)) + float(state.get("vx", 0.0)) * elapsed
            predicted_y = float(state.get("center_y", center_y)) + float(state.get("vy", 0.0)) * elapsed
            distance = float(np.hypot(center_x - predicted_x, center_y - predicted_y))
            size_penalty = (abs(width_ratio - 1.0) + abs(height_ratio - 1.0)) * 24.0
            score = distance + size_penalty
            max_distance = base_match_distance + min(80.0, elapsed * 180.0)
            if distance <= max_distance and score < best_score:
                best_track_id = int(track_id)
                best_score = score

        if best_track_id is None:
            best_track_id = int(self._athlete_detector_next_track_id)
            self._athlete_detector_next_track_id += 1
            if self._athlete_detector_next_track_id >= 969000:
                self._athlete_detector_next_track_id = 920000
            velocity_x = 0.0
            velocity_y = 0.0
        else:
            previous = self._athlete_detector_track_pool[best_track_id]
            elapsed = current_time - float(previous.get("time", current_time))
            velocity_x = float(previous.get("vx", 0.0))
            velocity_y = float(previous.get("vy", 0.0))
            if elapsed > 1e-3:
                observed_vx = (center_x - float(previous.get("center_x", center_x))) / elapsed
                observed_vy = (center_y - float(previous.get("center_y", center_y))) / elapsed
                velocity_x = velocity_x * 0.45 + observed_vx * 0.55
                velocity_y = velocity_y * 0.45 + observed_vy * 0.55

        self._athlete_detector_track_pool[best_track_id] = {
            "center_x": center_x,
            "center_y": center_y,
            "width": box_width,
            "height": box_height,
            "time": float(current_time),
            "vx": velocity_x,
            "vy": velocity_y,
        }
        used_track_ids.add(best_track_id)
        return best_track_id

    def _detect_speed_skating_athletes(
        self,
        infer_frame: np.ndarray,
        infer_offset_x: int,
        infer_offset_y: int,
        frame_shape: Tuple[int, int],
        current_time: float,
        existing_athletes: List[dict],
    ) -> Tuple[List[dict], float]:
        """Recover track-side skaters with a generic YOLO11 person detector."""
        if (
            self.sport_profile != "speed_skating"
            or self._athlete_detector is None
            or self._athlete_detector_failed
        ):
            return [], 0.0

        started = time.time()
        try:
            results = self._athlete_detector.predict(
                source=infer_frame,
                classes=[0],
                conf=self.athlete_detector_conf,
                iou=0.60,
                imgsz=self.athlete_detector_imgsz,
                verbose=False,
            )
        except Exception as exc:
            self._athlete_detector_failed = True
            logger.warning(
                f"[Detector-{self.source_id}] Person detector failed; disabling fallback: {exc}"
            )
            return [], (time.time() - started) * 1000.0

        height, width = int(frame_shape[0]), int(frame_shape[1])
        recovered: List[dict] = []
        used_track_ids: Set[int] = set()
        min_bottom = height * self.athlete_detector_trackside_bottom_ratio
        min_height = max(float(self.min_bh), height * 0.08)

        for result in results or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for index in range(len(boxes)):
                confidence_value = boxes.conf[index]
                confidence = (
                    float(confidence_value.item())
                    if hasattr(confidence_value, "item")
                    else float(confidence_value)
                )
                xyxy_value = boxes.xyxy[index]
                xyxy = xyxy_value.tolist() if hasattr(xyxy_value, "tolist") else list(xyxy_value)
                x1 = int(round(float(xyxy[0]) + infer_offset_x))
                y1 = int(round(float(xyxy[1]) + infer_offset_y))
                x2 = int(round(float(xyxy[2]) + infer_offset_x))
                y2 = int(round(float(xyxy[3]) + infer_offset_y))
                bbox = [
                    max(0, min(width, x1)),
                    max(0, min(height, y1)),
                    max(0, min(width, x2)),
                    max(0, min(height, y2)),
                ]
                box_width = bbox[2] - bbox[0]
                box_height = bbox[3] - bbox[1]
                if box_width <= 0 or box_height < min_height or bbox[3] < min_bottom:
                    continue
                if (box_width * box_height) > width * height * 0.30:
                    continue
                if not self._passes_athlete_aspect_filter(bbox, confidence, (height, width)):
                    continue

                duplicates_existing = any(
                    bbox_iou(bbox, athlete.get("bbox", [])) >= 0.35
                    or bbox_overlap_over_smaller(bbox, athlete.get("bbox", [])) >= 0.78
                    for athlete in existing_athletes + recovered
                    if athlete.get("bbox")
                )
                if duplicates_existing:
                    continue

                track_id = self._allocate_athlete_detector_track_id(
                    bbox,
                    current_time,
                    (height, width),
                    used_track_ids,
                )
                recovered.append({
                    "bbox": bbox,
                    "conf": confidence,
                    "track_id": track_id,
                    "bib_text": "",
                    "center_x": (bbox[0] + bbox[2]) // 2,
                    "center_y": (bbox[1] + bbox[3]) // 2,
                    "bottom_y": bbox[3],
                    "athlete_detector": "yolo11_person",
                })

        return recovered, (time.time() - started) * 1000.0

    def _passes_speed_skating_trackside_filter(
        self,
        bbox: List[int],
        frame_shape: Tuple[int, int],
    ) -> bool:
        """Reject person boxes that end above the track-side spectator barrier."""
        if self.sport_profile != "speed_skating":
            return True
        pt1 = getattr(self, "_line_pt1", (0, 0))
        pt2 = getattr(self, "_line_pt2", (0, 0))
        if abs(int(pt2[0]) - int(pt1[0])) < abs(int(pt2[1]) - int(pt1[1])):
            # A vertical finish line can legitimately be crossed in the upper
            # part of the image; keep the existing endpoint semantics there.
            return True
        height = max(1, int(frame_shape[0]))
        return int(bbox[3]) >= int(height * self.speed_skating_primary_bottom_ratio)

    @staticmethod
    def _is_event_eligible_athlete(athlete: dict, observation_count: int) -> bool:
        synthetic_kind = str(athlete.get('synthetic_kind') or '')
        if not synthetic_kind:
            return True
        if synthetic_kind == 'bib_split':
            return int(observation_count) >= 3
        return False

    def _ensure_athletes_from_bibs(self, athletes: List[dict], bibs: List[dict],
                                    frame_original: np.ndarray, current_time: float) -> List[dict]:
        """号码牌驱动检测：有号码牌的地方一定有运动员 (针对自行车优化)"""
        if not self.bib_driven_detection:
            return athletes
        
        height, width = frame_original.shape[:2]
        if not bibs:
            return athletes

        # 先把每个运动员关联到号码牌，再做“一个框多人”拆分
        assigned_candidates, _ = self._assign_bibs_to_athletes(athletes, bibs)
        matched_bib_indices = set()
        athlete_bib_map: Dict[int, List[int]] = {}
        for athlete_idx, candidate_indices in assigned_candidates.items():

            # 去掉同一号码牌的重复检测框，避免把“一个人”拆成多人
            unique_indices: List[int] = []
            for bib_idx in sorted(candidate_indices, key=lambda idx: bibs[idx].get('conf', 0.0), reverse=True):
                bx1, by1, bx2, by2 = bibs[bib_idx]['bbox']
                bcx = bibs[bib_idx].get('center_x', (bx1 + bx2) * 0.5)
                bcy = bibs[bib_idx].get('center_y', (by1 + by2) * 0.5)
                bib_w = max(1, bx2 - bx1)
                bib_h = max(1, by2 - by1)

                is_dup = False
                for kept_idx in unique_indices:
                    kx1, ky1, kx2, ky2 = bibs[kept_idx]['bbox']
                    kcx = bibs[kept_idx].get('center_x', (kx1 + kx2) * 0.5)
                    kcy = bibs[kept_idx].get('center_y', (ky1 + ky2) * 0.5)
                    if bbox_iou([bx1, by1, bx2, by2], [kx1, ky1, kx2, ky2]) > 0.45:
                        is_dup = True
                        break
                    if abs(bcx - kcx) < max(10.0, min(bib_w, max(1, kx2 - kx1)) * 0.55) and \
                       abs(bcy - kcy) < max(8.0, min(bib_h, max(1, ky2 - ky1)) * 0.55):
                        is_dup = True
                        break

                if not is_dup:
                    unique_indices.append(bib_idx)

            if unique_indices:
                athlete_bib_map[athlete_idx] = unique_indices
                matched_bib_indices.update(unique_indices)

        # P0：同一大框内出现多个独立号码牌时，拆成多个人框，减少“一框多人”和漏人
        split_athletes: List[dict] = []
        for athlete_idx, athlete in enumerate(athletes):
            bib_indices = athlete_bib_map.get(athlete_idx, [])
            if len(bib_indices) < 2:
                split_athletes.append(athlete)
                continue

            ax1, ay1, ax2, ay2 = athlete['bbox']
            ath_w = max(1, ax2 - ax1)
            ath_h = max(1, ay2 - ay1)

            # 马拉松并排场景：放宽触发，优先把单大框拆成多人框
            if ath_w < max(64, int(width * 0.09)) or ay2 < int(height * 0.30):
                split_athletes.append(athlete)
                continue

            bib_indices = sorted(bib_indices, key=lambda idx: bibs[idx].get('center_x', 0))
            # 放宽：号码牌置信度略低也参与拆分，优先防漏人
            bib_indices = [idx for idx in bib_indices if float(bibs[idx].get('conf', 0.0)) >= 0.35]
            if len(bib_indices) < 2:
                split_athletes.append(athlete)
                continue

            # 放宽：降低最小横向间距，适应贴身并排跑者
            distinct_indices: List[int] = []
            min_gap = max(12, int(ath_w * 0.12))
            for idx in bib_indices:
                cx = float(bibs[idx].get('center_x', 0.0))
                if all(abs(cx - float(bibs[k].get('center_x', 0.0))) >= min_gap for k in distinct_indices):
                    distinct_indices.append(idx)
            bib_indices = distinct_indices
            if len(bib_indices) < 2:
                split_athletes.append(athlete)
                continue

            if bib_indices:
                bib_span = bibs[bib_indices[-1]].get('center_x', 0) - bibs[bib_indices[0]].get('center_x', 0)
            else:
                bib_span = 0
            if bib_span < max(18, int(ath_w * 0.20)):
                split_athletes.append(athlete)
                continue

            generated: List[dict] = []
            base_tid = int(athlete.get('track_id', -1))
            for bib_idx in bib_indices:
                bib = bibs[bib_idx]
                bx1, by1, bx2, by2 = bib['bbox']
                bib_w = max(1, bx2 - bx1)
                cx = bib.get('center_x', (bx1 + bx2) // 2)
                cy = bib.get('center_y', (by1 + by2) // 2)

                split_w = int(max(bib_w * 3.2, ath_w / max(2.0, float(len(bib_indices)))))
                split_w = max(24, min(split_w, int(ath_w * 0.78)))
                sx1 = max(0, cx - split_w // 2)
                sx2 = min(width, cx + split_w // 2)

                y_pad_top = int(max(0.0, ath_h * 0.06))
                y_pad_bottom = int(max(0.0, ath_h * 0.03))
                sy1 = max(0, ay1 - y_pad_top)
                sy2 = min(height, ay2 + y_pad_bottom)

                limit_l = max(0, ax1 - int(ath_w * 0.12))
                limit_r = min(width, ax2 + int(ath_w * 0.12))
                sx1 = max(sx1, limit_l)
                sx2 = min(sx2, limit_r)

                if sx2 - sx1 < max(18, int(bib_w * 2.0)):
                    continue

                new_bbox = [int(sx1), int(sy1), int(sx2), int(sy2)]
                duplicate = False
                for old_ath in split_athletes:
                    if bbox_iou(new_bbox, old_ath['bbox']) > 0.72:
                        duplicate = True
                        break
                if not duplicate:
                    for old_ath in generated:
                        if bbox_iou(new_bbox, old_ath['bbox']) > 0.55:
                            duplicate = True
                            break
                if duplicate:
                    continue

                split_tid = self._allocate_split_track_id(cx, (new_bbox[1] + new_bbox[3]) // 2, current_time, (height, width))
                new_conf = max(float(athlete.get('conf', 0.0)) * 0.90, float(bib.get('conf', 0.0)) * 0.75)
                generated.append({
                    'bbox': new_bbox,
                    'conf': new_conf,
                    'track_id': split_tid,
                    'center_x': (new_bbox[0] + new_bbox[2]) // 2,
                    'center_y': (new_bbox[1] + new_bbox[3]) // 2,
                    'bottom_y': new_bbox[3],
                'bib_driven': True,
                'split_from_track': base_tid,
                'synthetic_kind': 'bib_split',
            })

            if len(generated) >= 2:
                split_athletes.extend(generated)
                logger.info(f"[Detector-{self.source_id}] 大框拆分: ID {base_tid} -> {len(generated)}人")
            else:
                split_athletes.append(athlete)

        return split_athletes

    def _rescue_athletes_from_bibs_forced(self, athletes: List[dict], bibs: List[dict],
                                          frame_shape: Tuple[int, int], current_time: float) -> List[dict]:
        """强制补人兜底：当出现“只有号码牌框、缺少人框”时，补出可过线目标。"""
        if not bibs:
            return athletes

        h, w = int(frame_shape[0]), int(frame_shape[1])
        rescued = list(athletes)
        added = 0
        max_add = 5

        # 清理过期补人记忆，避免字典无限增长
        stale_keys = [k for k, ts in self._forced_bib_rescue_last.items() if current_time - float(ts) > 3.0]
        for k in stale_keys:
            self._forced_bib_rescue_last.pop(k, None)

        for bib in sorted(bibs, key=lambda x: float(x.get('conf', 0.0)), reverse=True):
            if added >= max_add:
                break

            bx1, by1, bx2, by2 = [int(v) for v in bib.get('bbox', [0, 0, 0, 0])]
            bw = max(1, bx2 - bx1)
            bh = max(1, by2 - by1)
            if bw < 18 or bh < 12:
                continue

            bib_conf = float(bib.get('conf', 0.0))
            if bib_conf < 0.35:
                continue

            cx = int(bib.get('center_x', (bx1 + bx2) // 2))
            cy = int(bib.get('center_y', (by1 + by2) // 2))

            # 放宽区域：马拉松并排接近时，允许更早补人
            if cy < int(h * 0.35):
                continue

            # 降温：适度缩短冷却，优先不漏人
            cell_key = (int(cx // 24), int(cy // 20))
            last_rescue_ts = float(self._forced_bib_rescue_last.get(cell_key, -999.0))
            if current_time - last_rescue_ts < 0.35:
                continue

            est_h = int(max(42, bh * 4.2))
            est_w = int(max(24, bw * 3.2))
            ax1 = max(0, cx - est_w // 2)
            ax2 = min(w, cx + est_w // 2)
            ay1 = max(0, cy - int(est_h * 0.72))
            ay2 = min(h, cy + int(est_h * 0.28))
            new_bbox = [int(ax1), int(ay1), int(ax2), int(ay2)]

            if new_bbox[2] - new_bbox[0] < 20 or new_bbox[3] - new_bbox[1] < 28:
                continue

            if self.enable_gate_guard and self._is_gate_like_bbox(new_bbox, (h, w)):
                continue

            duplicate = False
            for athlete in rescued:
                old_bbox = athlete.get('bbox', [])
                if bbox_iou(new_bbox, old_bbox) > 0.18:
                    duplicate = True
                    break
                try:
                    ocx = int((old_bbox[0] + old_bbox[2]) // 2)
                    ocy = int((old_bbox[1] + old_bbox[3]) // 2)
                    if abs(ocx - cx) <= 18 and abs(ocy - cy) <= 30:
                        duplicate = True
                        break
                except Exception:
                    pass
            if duplicate:
                continue

            tid = self._allocate_split_track_id(cx, ay2, current_time, (h, w))
            rescued.append({
                'bbox': new_bbox,
                'conf': max(0.18, bib_conf * 0.72),
                'track_id': tid,
                'center_x': (new_bbox[0] + new_bbox[2]) // 2,
                'center_y': (new_bbox[1] + new_bbox[3]) // 2,
                'bottom_y': new_bbox[3],
                'bib_driven': True,
                'bib_driven_forced': True,
            })
            self._forced_bib_rescue_last[cell_key] = float(current_time)
            added += 1

        if added > 0:
            logger.info(f"[Detector-{self.source_id}] 号码牌强制补人: +{added}")
        return rescued

    def _allocate_split_track_id(self, cx: int, cy: int, current_time: float,
                                 frame_shape: Tuple[int, int]) -> int:
        """为宽框拆分生成稳定 track_id（短时复用，避免每帧新建 ID）。"""
        h, w = int(frame_shape[0]), int(frame_shape[1])
        stale = []
        best_tid = None
        best_dist = 1e9
        max_dist = max(34.0, 0.045 * max(w, h))

        for tid, (px, py, ts) in self._split_track_pool.items():
            if current_time - ts > 1.4:
                stale.append(tid)
                continue
            dist = np.hypot(float(cx - px), float(cy - py))
            if dist < best_dist:
                best_dist = dist
                best_tid = tid

        for tid in stale:
            self._split_track_pool.pop(tid, None)

        if best_tid is not None and best_dist <= max_dist:
            tid = int(best_tid)
        else:
            tid = int(self._split_next_track_id)
            self._split_next_track_id += 1
            if self._split_next_track_id >= 999500:
                self._split_next_track_id = 970000

        self._split_track_pool[tid] = (int(cx), int(cy), float(current_time))
        return tid

    def _rescue_split_wide_boxes(self, athletes: List[dict], bibs: List[dict],
                                 tid_bib_centers: Dict[int, List],
                                 frame_shape: Tuple[int, int], current_time: float) -> List[dict]:
        """漏人抢救：当出现异常宽框时，按宽度拆分为多人目标（过线前生效）。"""
        if not athletes:
            return athletes

        h, w = int(frame_shape[0]), int(frame_shape[1])
        result: List[dict] = []
        split_total = 0
        max_rescue_add = 5

        # 马拉松并排场景：仅在极高人数时才禁用拆分，避免“一框吞多人”
        if len(athletes) >= 10:
            return athletes

        for idx, athlete in enumerate(athletes):
            bbox = athlete.get('bbox')
            if not bbox or len(bbox) != 4:
                result.append(athlete)
                continue

            if athlete.get('split_from_track') is not None:
                result.append(athlete)
                continue

            x1, y1, x2, y2 = [int(v) for v in bbox]
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            aspect_w = bw / max(1.0, float(bh))

            # 马拉松并排场景：放宽“宽框”判定，避免4人并排只算1人
            if bw < max(72, int(w * 0.07)):
                result.append(athlete)
                continue
            if bh < max(52, int(h * 0.07)):
                result.append(athlete)
                continue
            # 阈值 0.60 比较激进，回调到 0.72 保持防漏同时降低噪声拆分
            if aspect_w < 0.72:
                result.append(athlete)
                continue
            if y2 < int(h * 0.38):
                result.append(athlete)
                continue
            if self.enable_gate_guard and self._is_gate_like_bbox([x1, y1, x2, y2], (h, w)):
                result.append(athlete)
                continue

            # 若周围已有大量独立人框，说明不是“单框吞多人”，不做拆分
            neighbors = 0
            for j, other in enumerate(athletes):
                if j == idx:
                    continue
                if bbox_iou(athlete['bbox'], other.get('bbox', [])) > 0.10:
                    neighbors += 1
            if neighbors >= 4:
                result.append(athlete)
                continue

            # 同一宽框拆分冷却：避免相邻帧对同一目标重复拆分，导致“框越拆越多”
            key_tid = int(athlete.get('track_id', -1))
            if key_tid != -1:
                split_key = ("tid", key_tid)
            else:
                split_key = ("cell", int(((x1 + x2) * 0.5) // 28), int((y2) // 24), int(bw // 30), int(bh // 30))
            last_split_ts = float(self._wide_split_last.get(split_key, -999.0))
            if current_time - last_split_ts < 0.55:
                result.append(athlete)
                continue

            tid = int(athlete.get('track_id', -1))
            bib_refs = tid_bib_centers.get(tid, []) if tid != -1 else []
            inside_bibs = [b for b in bib_refs if x1 <= int(b[0]) <= x2 and y1 <= int(b[1]) <= y2]

            # 马拉松优先召回：
            # 1) 有多个号码牌直接拆分
            # 2) 仅1个号码牌但框明显过宽，也允许拆2人
            # 3) 无号码牌时启用几何估算兜底
            if len(inside_bibs) < 2:
                result.append(athlete)
                continue
            split_n = min(5, len(inside_bibs))

            if split_n < 2:
                result.append(athlete)
                continue

            seg_w = bw / float(split_n)
            generated = []
            used_ids = set()
            for part_idx in range(split_n):
                pad_x = seg_w * 0.08
                sx1 = int(max(0, x1 + part_idx * seg_w - pad_x))
                sx2 = int(min(w, x1 + (part_idx + 1) * seg_w + pad_x))
                sy1 = int(max(0, y1 - bh * 0.02))
                sy2 = int(min(h, y2 + bh * 0.02))

                if sx2 - sx1 < max(20, int(w * 0.02)):
                    continue

                cx = (sx1 + sx2) // 2
                split_tid = self._allocate_split_track_id(cx, sy2, current_time, (h, w))
                if split_tid in used_ids:
                    split_tid = int(self._split_next_track_id)
                    self._split_next_track_id += 1
                    if self._split_next_track_id >= 999500:
                        self._split_next_track_id = 970000
                    self._split_track_pool[split_tid] = (int(cx), int(sy2), float(current_time))
                used_ids.add(split_tid)

                generated.append({
                    'bbox': [sx1, sy1, sx2, sy2],
                    'conf': max(0.18, float(athlete.get('conf', 0.0)) * 0.82),
                    'track_id': split_tid,
                    'center_x': cx,
                    'center_y': (sy1 + sy2) // 2,
                    'bottom_y': sy2,
                    'bib_text': athlete.get('bib_text', ''),
                    'split_from_track': tid,
                    'synthetic_kind': 'bib_split',
                })

            if len(generated) < 2:
                result.append(athlete)
                continue

            # 安全阈值：单帧最多补3个，避免极端场景误拆造成事件虚增
            remain_budget = max_rescue_add - split_total
            if remain_budget <= 0:
                result.append(athlete)
                continue
            if len(generated) > remain_budget:
                generated = generated[:remain_budget]
            if len(generated) < 2:
                result.append(athlete)
                continue

            # 把原 track 的号码牌关联迁移到新拆分目标
            old_refs = tid_bib_centers.pop(tid, []) if tid != -1 else []
            for bib_ref in old_refs:
                bcx, bcy, bb = bib_ref
                best_target = min(
                    generated,
                    key=lambda item: abs(float(item['center_x']) - float(bcx)) + abs(float(item['center_y']) - float(bcy)) * 0.25
                )
                tid_bib_centers.setdefault(best_target['track_id'], []).append((bcx, bcy, bb))

            with self._lock:
                for item in generated:
                    stid = int(item['track_id'])
                    cx = int(item['center_x'])
                    by = int(item['bottom_y'])
                    if stid not in self._track_states:
                        self._track_states[stid] = TrackState(
                            prev_x=cx,
                            prev_y=by,
                            last_seen_time=current_time,
                            start_pos=(cx, by),
                            start_time=current_time
                        )
                    else:
                        state = self._track_states[stid]
                        state.last_seen_time = current_time
                        state.prev_x = cx
                        state.prev_y = by
                        state.is_static = False
                        state.static_confidence = max(0.0, state.static_confidence - 0.25)

            result.extend(generated)
            split_total += len(generated)
            self._wide_split_last[split_key] = float(current_time)

        if split_total > 0:
            logger.info(f"[Detector-{self.source_id}] 宽框抢救拆分新增 {split_total} 个目标")

        return result

    # =========================================================================
    # OCR 相关
    # =========================================================================
    
    def set_ocr(self, ocr_instance):
        """设置并初始化 OCR 引擎"""
        if not self.ocr_pipeline_enabled:
            self._ocr = None
            return
        self._ocr = ocr_instance
        self.ensure_ocr()

    def ensure_ocr(self):
        """确保 OCR 引擎已初始化"""
        if not self.ocr_pipeline_enabled:
            return
        if self._ocr is not None and not self._ocr_running:
            self.start_ocr_worker()

    def start_ocr_worker(self):
        """启动 OCR 后台线程"""
        with self._lock:
            if self._ocr_running and self._ocr_thread and self._ocr_thread.is_alive():
                return
            
            if self._ocr is None:
                logger.warning(f"[Detector-{self.source_id}] 无法启动 OCR 线程：OCR 引擎为 None")
                return

            self._ocr_running = True
            self._ocr_thread = threading.Thread(
                target=self._ocr_worker,
                name=f"OCR-Worker-{self.source_id}",
                daemon=True
            )
            self._ocr_thread.start()
            logger.info(f"[Detector-{self.source_id}] OCR 后台线程已启动")

    def _preprocess_for_ocr(self, img: np.ndarray) -> np.ndarray:
        """OCR 图像预处理：增强对比度、多尺度缩放，显著提升识别率"""
        if img is None or img.size == 0:
            return img
            
        h, w = img.shape[:2]
        
        # 1. 尺度归一化：PaddleOCR 对 80-150 像素高度的文字识别效果最好
        # 针对自行车号码牌通常较小的特点，强制放大到 120 像素高度
        target_h = 120
        if h < target_h:
            scale = target_h / h
            img = cv2.resize(img, (int(w * scale), target_h), interpolation=cv2.INTER_CUBIC)
        elif h > 300: 
            scale = 240 / h
            img = cv2.resize(img, (int(w * scale), 240), interpolation=cv2.INTER_AREA)
        
        # 2. 补边 (Padding)：增加白色边缘，模拟真实纸质号码牌背景，提高模型检测率
        pad = 20
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        
        # 3. 自适应对比度增强 (CLAHE)
        try:
            # 转换为 LAB 空间进行亮度通道增强，保持颜色不变（如果 OCR 需要颜色信息）
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        except Exception as e:
            logger.debug(f"[Detector] CLAHE 预处理失败: {e}")
            
        return img

    def _record_participant_ocr_vote(
        self,
        participant_id: Optional[str],
        raw_track_id: int,
        text: str,
        confidence: float,
    ):
        """Aggregate validated OCR evidence without changing participant identity."""
        key = str(participant_id or "").strip()
        if not key or not text:
            return None
        if __package__:
            from .ocr_manager import ParticipantOcrState
        else:
            from ocr_manager import ParticipantOcrState

        state = self._participant_ocr_states.get(key)
        if state is None:
            state = ParticipantOcrState(participant_id=key)
            self._participant_ocr_states[key] = state
        state.add_vote(
            raw_track_id=raw_track_id,
            text=text,
            confidence=confidence,
        )
        return state

    def _ocr_worker(self):
        """后台 OCR 线程"""
        logger.info(f"[Detector-{self.source_id}] OCR 后台线程进入循环")
        while self._ocr_running:
            task = None
            try:
                try:
                    task = self._ocr_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                    
                if task is None:
                    continue
                
                if len(task) >= 7:
                    (
                        bib_crop,
                        track_id,
                        athlete_bbox,
                        bib_bbox,
                        frame_original,
                        crop_quality,
                        participant_id,
                    ) = task
                else:
                    bib_crop, track_id, athlete_bbox, bib_bbox, frame_original, crop_quality = task
                    participant_id = ""
                
                # 计算当前检测器时间戳
                current_time = time.time() - (self._start_time if self._start_time else time.time())
                
                # 再次确认 track_id 还在活跃状态
                with self._lock:
                    if track_id not in self._track_states:
                        continue

                rel_pos = None
                if athlete_bbox and bib_bbox:
                    ax1, ay1, ax2, ay2 = athlete_bbox
                    bx1, by1, bx2, by2 = bib_bbox
                    aw = max(1, ax2 - ax1)
                    ah = max(1, ay2 - ay1)
                    rel_pos = ((bx1 + bx2) * 0.5 - ax1) / aw, ((by1 + by2) * 0.5 - ay1) / ah
                
                # 相对位置约束：放宽限制以应对自行车运动中的前倾姿势
                if rel_pos:
                    rel_x, rel_y = rel_pos
                    # 自行车场景：号码牌可能在运动员框的边缘甚至稍微靠下（车头位置）
                    # rel_y 范围放宽到 1.1，rel_x 稍微超出左右边界也允许
                    if not (-0.1 <= rel_x <= 1.1) or not (0.0 <= rel_y <= 1.1):
                        logger.debug(f"[OCR-Worker-{self.source_id}] ID {track_id} 号码牌位置超出合理范围 (rel_x:{rel_x:.2f}, rel_y:{rel_y:.2f})，跳过")
                        continue

                # 图像预处理：放大和补边，显著提高小图识别率
                ocr_input = self._preprocess_for_ocr(bib_crop)

                # 直接识别
                lines, _ = self._ocr(ocr_input)
                text, conf = None, 0.0
                if lines:
                    # 遍历所有识别行，选置信度最高的有效结果
                    # （号码牌裁剪图里可能有多行文字：号码、赞助商名、赛事名等）
                    for _, raw_text, raw_conf in lines:
                        if self.only_numeric:
                            clean_text = ''.join(c for c in raw_text if c.isdigit())
                        else:
                            clean_text = ''.join(c for c in raw_text if c.isalnum())

                        if not clean_text or len(clean_text) > 10:
                            continue
                        # 必须包含数字，或者长度至少为2（防止识别出单个无意义字母）
                        if not (any(c.isdigit() for c in clean_text) or len(clean_text) >= 2):
                            continue
                        if raw_conf > conf:
                            text, conf = clean_text, raw_conf

                    if text:
                        logger.debug(f"[OCR-Worker-{self.source_id}] ID {track_id} 识别成功: {text} (conf:{conf:.2f}, quality:{crop_quality:.2f})")
                
                if not text:
                    logger.debug(f"[OCR-Worker-{self.source_id}] ID {track_id} 识别未发现号码")
                
                # VLM 辅助 (如果 OCR 失败、置信度太低 或 不符合名单/正则规则)
                if self._vlm_enabled and self._vlm_pipeline:
                    should_trigger_vlm = False
                    reason = ""
                    
                    if not text:
                        should_trigger_vlm = True
                        reason = "OCR_FAILED"
                    else:
                        corrected_text, val_score = self._validate_bib(text)
                        if conf < 0.6 or val_score < 0.5:
                            should_trigger_vlm = True
                            reason = f"LOW_CONF({conf:.2f}/{val_score:.2f})"
                        elif hasattr(self, '_athlete_list') and self._athlete_list and corrected_text not in self._athlete_list:
                            # 号码虽然符合正则，但不在名单里，强制 VLM 确认
                            should_trigger_vlm = True
                            reason = "NOT_IN_LIST"
                    
                    if should_trigger_vlm:
                        with self._lock:
                            state = self._track_states.get(track_id)
                            if state:
                                # 增加 VLM 触发约束：积压限制 + 冷却时间 + 状态检查
                                vlm_qsize = getattr(self._vlm_pipeline, 'qsize', lambda: 0)() if callable(getattr(self._vlm_pipeline, 'qsize', None)) else 0
                                vlm_interval = 3.0 # worker 触发间隔稍长
                                if vlm_qsize < 10 and (current_time - state.last_vlm_time > vlm_interval) and not state.vlm_ocr_pending:
                                    state.vlm_ocr_pending = True
                                    state.last_vlm_time = current_time
                                    # 提交异步 VLM 任务
                                    self._vlm_pipeline.submit_ocr(
                                        track_id, 
                                        bib_crop, 
                                        athlete_list=list(self._athlete_list) if self._athlete_list else None,
                                        only_numeric=self.only_numeric
                                    )
                                    logger.info(f"[OCR-Worker-{self.source_id}] ID {track_id} 触发 VLM 补强: {reason}")
                
                if text:
                    corrected_text, score = self._validate_bib(text)
                    final_conf = conf * score
                    
                    with self._lock:
                        if track_id in self._track_states:
                            state = self._track_states[track_id]
                            
                            pos_weight = 1.0
                            if state.last_bib_rel_pos and rel_pos:
                                dist = np.sqrt((rel_pos[0] - state.last_bib_rel_pos[0])**2 + 
                                               (rel_pos[1] - state.last_bib_rel_pos[1])**2)
                                if dist > 0.6:
                                    pos_weight = 0.25
                                    logger.debug(f"[Detector] ID {track_id} 号码位置偏移过大 ({dist:.2f})")
                                elif dist > 0.4:
                                    pos_weight = 0.5
                                    logger.debug(f"[Detector] ID {track_id} 号码位置偏移偏大 ({dist:.2f})")

                            # 1. 记录号码历史 (用于一致性校验)
                            state.bib_history.append(corrected_text)
                            if len(state.bib_history) > 10:
                                state.bib_history.pop(0)
                            
                            final_conf = final_conf * pos_weight
                            
                            # 2. 冲突检测：如果短时间内识别到了两个截然不同的名单内号码，发出警告
                            if len(state.bib_history) >= 3:
                                recent_list_bibs = [b for b in state.bib_history[-5:] if b in self._athlete_list]
                                if self._athlete_list and len(set(recent_list_bibs)) > 1:
                                    logger.warning(f"[Detector] ID {track_id} 出现号码冲突: {set(recent_list_bibs)}，强制 VLM 仲裁")
                                    if self._vlm_enabled and self._vlm_pipeline:
                                        self._vlm_pipeline.submit_ocr(
                                            track_id, 
                                            bib_crop, 
                                            athlete_list=list(self._athlete_list),
                                            only_numeric=self.only_numeric
                                        )

                            # 3. 投票机制
                            # 最低门槛：置信度太低的结果不投票，防止格式不对的号码靠次数堆积赢过真号码
                            # 例：号码段 A0001-A2000，OCR出 "12345"，validation score=0.1，
                            #     final_conf = 0.9×0.1 = 0.09 → 不进投票池
                            if final_conf < 0.15:
                                logger.debug(f"[Detector] ID {track_id} 号码 {corrected_text} 置信度过低 ({final_conf:.2f})，跳过投票")
                                continue
                            participant_ocr_state = self._record_participant_ocr_vote(
                                participant_id,
                                track_id,
                                corrected_text,
                                final_conf,
                            )
                            if participant_ocr_state is not None:
                                state.participant_id = participant_ocr_state.participant_id
                                state.participant_ocr_status = participant_ocr_state.status
                                if participant_ocr_state.status == "CONFLICT":
                                    logger.warning(
                                        f"[Detector] participant {participant_ocr_state.participant_id} "
                                        f"OCR conflict across raw tracks {sorted(participant_ocr_state.raw_track_ids)}"
                                    )
                                    continue
                                if participant_ocr_state.best_candidate:
                                    corrected_text = participant_ocr_state.best_candidate
                                    final_conf = participant_ocr_state.best_confidence
                            state.bib_candidates[corrected_text] = state.bib_candidates.get(corrected_text, 0) + 1
                            state.bib_confidences[corrected_text] = max(
                                state.bib_confidences.get(corrected_text, 0.0), final_conf
                            )
                            
                            # 选择最佳候选
                            best_candidate = None
                            max_score = -1.0
                            
                            # 统计名单内号码的总票数，用于检测冲突
                            list_bib_counts = {cand: count for cand, count in state.bib_candidates.items() if cand in self._athlete_list}
                            total_list_votes = sum(list_bib_counts.values())
                            
                            for cand, count in state.bib_candidates.items():
                                cand_conf = state.bib_confidences[cand]
                                # 名单奖励：如果在名单里，给予巨大权重
                                in_list = cand in self._athlete_list
                                list_bonus = 30.0 if in_list else 0.0  # 单摄像头增加名单奖励
                                # 号段奖励：如果在号段里，给予一定权重
                                range_bonus = 5.0 if self._is_in_bib_ranges(cand) else 0.0
                                
                                # 核心改进：引入“占有率”权重
                                share_bonus = 0.0
                                if in_list and total_list_votes > 2: # 降低生效门槛
                                    share = count / total_list_votes
                                    if share > 0.6: share_bonus = 15.0 # 绝对多数奖励增加
                                    elif share < 0.4: share_bonus = -20.0 # 少数派惩罚加重
                                
                                current_score = (count * 2.5) + (cand_conf * 3.0) + list_bonus + range_bonus + share_bonus
                                if current_score > max_score:
                                    max_score = current_score
                                    best_candidate = cand
                            
                            is_updated = False
                            # 只有当新号码比旧号码明显更好，或者旧号码不在名单内而新号码在名单内时，才更新
                            should_update = False
                            if best_candidate:
                                if state.best_bib is None:
                                    should_update = True
                                elif best_candidate != state.best_bib:
                                    old_in_list = state.best_bib in self._athlete_list
                                    new_in_list = best_candidate in self._athlete_list
                                    
                                    if new_in_list and not old_in_list:
                                        should_update = True # 名单内号码优先
                                    elif new_in_list == old_in_list:
                                        # 如果都在名单内，必须满足：
                                        old_count = state.bib_candidates.get(state.best_bib, 0)
                                        new_count = state.bib_candidates.get(best_candidate, 0)
                                        old_conf = state.bib_confidences.get(state.best_bib, 0.0)
                                        
                                        # 1. 新号码的票数优势 (单摄像头下略微降低门槛，提高反应速度)
                                        if new_count > old_count * 1.5:
                                            should_update = True
                                        # 2. 或者旧号码只出现过 1-2 次，而新号码已经连续出现
                                        elif old_count <= 2 and new_count >= 3:
                                            should_update = True
                                        # 3. 或者置信度有质的飞跃
                                        elif old_conf < 0.3 and final_conf > 0.7:
                                            should_update = True
                                        
                                        # 稳定度抵抗：旧号码已稳定 6 次以上时，新号码需要更强证据
                                        # 但不能锁死：倍率从 3 降到 2，且置信度飞跃可以穿透
                                        if state.stable_count >= 6 and new_count < old_count * 2:
                                            # 例外：旧号码本身很模糊、新号码很清晰 → 说明早期是误识别，放行
                                            if not (old_conf < 0.3 and final_conf > 0.7):
                                                should_update = False
                                    elif final_conf > state.best_bib_conf + 0.4:
                                        should_update = True

                            if should_update:
                                # 【新增】全局号码冲突检查：处理 ID 跳变或号码抢夺
                                with self._lock:
                                    conflict_tid = None
                                    for other_tid, other_state in self._track_states.items():
                                        if other_tid != track_id and other_state.best_bib == best_candidate:
                                            conflict_tid = other_tid
                                            break
                                    
                                    if conflict_tid is not None:
                                        other_state = self._track_states[conflict_tid]
                                        time_since_seen = time.time() - other_state.last_seen_time
                                        if time_since_seen > 1.5: # 对方消失超过 1.5 秒
                                            logger.info(f"[Detector] 检测到 ID 跳变: Bib {best_candidate} 从 ID {conflict_tid} 跳到 {track_id}")
                                            # 继承关键状态
                                            if other_state.event_emitted:
                                                state.event_emitted = True
                                            # 标记合并
                                            other_state.merged_into = track_id
                                        else:
                                            # 对方还活着，可能是重叠抢号。在这种情况下，我们仍然允许更新，但记录冲突
                                            logger.debug(f"[Detector] 号码竞争: ID {track_id} 试图从活跃 ID {conflict_tid} 处获取号码 {best_candidate}")

                                if best_candidate == state.best_bib:
                                    state.stable_count += 1
                                else:
                                    # 只有当旧号码不稳定或者新号码足够强时，才重置稳定计数
                                    if state.best_bib is not None:
                                        logger.info(f"[Detector] Track {track_id} 号码跳变: {state.best_bib} -> {best_candidate}")
                                    state.stable_count = 1
                                
                                state.best_bib = best_candidate
                                state.best_bib_conf = state.bib_confidences[best_candidate]
                                state.best_bib_crop = bib_crop.copy()
                                if frame_original is not None:
                                    # 只有质量显著提升或首次获得时才拷贝全景图，减少 CPU/内存压力
                                    if state.best_bib_frame is None or crop_quality > state.best_bib_quality + 0.1:
                                        state.best_bib_frame = frame_original.copy()
                                state.best_bib_quality = crop_quality
                                is_updated = True
                            else:
                                # 如果不更新号码，但识别到了当前最佳号码，且图片质量更高，则更新图片
                                if best_candidate == state.best_bib:
                                    state.stable_count += 1
                                    if crop_quality > state.best_bib_quality:
                                        state.best_bib_crop = bib_crop.copy()
                                        if frame_original is not None:
                                            # 只有质量显著提升才拷贝全景图
                                            if crop_quality > state.best_bib_quality + 0.1:
                                                state.best_bib_frame = frame_original.copy()
                                        state.best_bib_quality = crop_quality
                                        state.best_bib_conf = max(state.best_bib_conf, final_conf)
                                        is_updated = True # 标记更新，触发补录以刷新图片
                                        logger.debug(f"[Detector] ID {track_id} 号码不变，但更新了更高质量的图片 ({crop_quality:.2f})")
                                
                                # 裁剪运动员
                                if frame_original is not None:
                                    h, w = frame_original.shape[:2]
                                    px1, py1, px2, py2 = athlete_bbox
                                    p_pad = 20
                                    state.best_athlete_crop = frame_original[
                                        max(0, py1-p_pad):min(h, py2+p_pad),
                                        max(0, px1-p_pad):min(w, px2+p_pad)
                                    ].copy()
                                
                            backfill_target_tid = None
                            if is_updated:
                                backfill_target_tid = self._try_backfill_recent_unknown_event(
                                    track_id, athlete_bbox, state, current_time
                                )

                            # 修复过线后补录逻辑 (无论 should_update 是否为真，只要 is_updated 为真且 event 已发，就补录)
                            if state.event_emitted and is_updated:
                                # 同步更新最近事件列表中的号码，用于后续去重
                                with self._lock:
                                    for event_rec in self._recent_events:
                                        if event_rec.get('tid') == track_id:
                                            event_rec['bib'] = state.best_bib
                                            break

                                if self._on_bib_update:
                                    # 自行车场景优化：如果号码稳定出现 3 次以上且置信度达标，直接设为 RECOGNIZED，无需人工确认
                                    is_stable = state.stable_count >= 3
                                    is_high_conf = state.best_bib_conf >= self.ocr_conf_threshold
                                    
                                    if is_stable and is_high_conf:
                                        new_status = BibStatus.RECOGNIZED
                                    elif is_high_conf:
                                        new_status = BibStatus.RECOGNIZED # 只要置信度够高，直接通过
                                    else:
                                        new_status = BibStatus.NEEDS_REVIEW
                                        
                                    self._on_bib_update(
                                        track_id, 
                                        state.best_bib, 
                                        state.best_bib_conf, 
                                        new_status, 
                                        self.source_id,
                                        bib_crop=state.best_bib_crop,
                                        athlete_crop=state.best_athlete_crop,
                                        full_frame=state.best_bib_frame
                                    )
                                    logger.info(f"[Detector-{self.source_id}] 号码状态更新: ID {track_id} -> {state.best_bib} ({new_status.value})")

                            # 补号桥接：当前轨迹识别出号码后，尝试回填到“刚过线且 UNKNOWN”的最近事件
                            if backfill_target_tid is not None and self._on_bib_update:
                                bridge_status = (BibStatus.RECOGNIZED if state.best_bib_conf >= self.ocr_conf_threshold
                                                 else BibStatus.NEEDS_REVIEW)
                                self._on_bib_update(
                                    backfill_target_tid,
                                    state.best_bib,
                                    state.best_bib_conf,
                                    bridge_status,
                                    self.source_id,
                                    bib_crop=state.best_bib_crop,
                                    athlete_crop=state.best_athlete_crop,
                                    full_frame=state.best_bib_frame
                                )
                                logger.info(
                                    f"[Detector-{self.source_id}] 过线补号桥接: unknown_tid={backfill_target_tid} "
                                    f"<= bib={state.best_bib} (from tid={track_id}, conf={state.best_bib_conf:.2f})"
                                )
                
            except Exception as e:
                logger.error(f"[Detector-{self.source_id}] OCR 线程内部异常: {e}")
                time.sleep(0.1)
            finally:
                if task is not None:
                    self._ocr_queue.task_done()

    def _split_bib_prefix_digits(self, bib: Optional[str]) -> Tuple[str, str]:
        if not bib:
            return "", ""
        bib = str(bib).strip().upper()
        m = re.match(r'^([A-Z]*)([0-9]+)$', bib)
        if m:
            return m.group(1), m.group(2)
        digits = "".join(filter(str.isdigit, bib))
        prefix = re.sub(r'[0-9]', '', bib)
        return prefix, digits

    def _validate_bib(self, bib: str) -> Tuple[str, float]:
        """校验并修正 OCR 识别结果"""
        if not bib:
            return bib, 0.0

        bib = str(bib).strip().upper()
        
        # 0. 常见环境干扰词黑名单 (龙门、赞助商等)
        # 解决用户提到的 "GEONES", "DEONES" 等问题
        blacklist = {
            "GEONES", "DEONES", "XEROX", "CHAMPION", "FINISH", "START", 
            "SPONSOR", "TIMING", "SPORTS", "EVENT", "BICYCLE", "RACING",
            "CHINA", "BEIJING", "SHANGHAI", "GUANGZHOU", "SHENZHEN"
        }
        
        # 模糊匹配黑名单 (处理 OCR 识别偏差，如 GEONE5)
        for black_word in blacklist:
            if black_word in bib or difflib.SequenceMatcher(None, bib, black_word).ratio() > 0.8:
                logger.debug(f"[Detector] 号码 {bib} 命中环境黑名单({black_word})，已过滤")
                return bib, 0.0
            
        # 长度过长的非数字串通常是广告语
        if len(bib) > 8 and not any(c.isdigit() for c in bib):
            return bib, 0.0

        # 0. 纯数字校验（严格模式）
        if self.only_numeric:
            if not bib.isdigit():
                # 如果包含字母，尝试提取数字部分
                numeric_part = ''.join(c for c in bib if c.isdigit())
                if numeric_part and len(numeric_part) >= 1:
                    logger.debug(f"[Detector] 号码 {bib} 包含字母，提取数字部分: {numeric_part}")
                    bib = numeric_part
                else:
                    logger.debug(f"[Detector] 号码 {bib} 包含字母且无数字，在纯数字模式下无效")
                    return bib, 0.0
            
        # 0. 精准名单匹配（最高优先级）
        if hasattr(self, '_athlete_list') and bib in self._athlete_list:
            return bib, 2.0 # 极高置信度

        # 1. 基础校验逻辑
        # 校验正则规则 (针对 AXXXX 格式)
        if hasattr(self, 'bib_regex') and self.bib_regex:
            if re.match(self.bib_regex, bib):
                score = 1.3
                if hasattr(self, '_athlete_list') and bib not in self._athlete_list:
                    score = 0.9 
                return bib, score
            elif not self.only_numeric: # 仅在非纯数字模式下尝试 AXXXX 补全
                # 允许 [A-Z] + 1-5位数字的组合
                if re.match(r'^[A-Z][0-9]{1,5}$', bib):
                    if len(bib) >= 4:
                        return bib, 1.4
                    else:
                        if hasattr(self, '_athlete_list'):
                            matches = [a for a in self._athlete_list if a.startswith(bib)]
                            if len(matches) == 1:
                                return matches[0], 1.2
                        return bib, 0.7 
                elif not any(c.isalpha() for c in bib) and len(bib) >= 3: 
                    # 纯数字，且长度至少为3，在非纯数字模式下也是合理的
                    # 尝试看名单里是否有加上前缀后的匹配
                    if hasattr(self, '_athlete_list'):
                        prefixes = ['A', 'B', 'C']
                        possible = [f"{p}{bib}" for p in prefixes]
                        found = [p for p in possible if p in self._athlete_list]
                        if len(found) == 1:
                            return found[0], 1.1
                    return bib, 1.0 # 纯数字在非严格模式下给基础分 1.0
                
                if self.bib_regex_strict:
                    return bib, 0.1

        # 2. 号段校验
        if self._is_in_bib_ranges(bib):
            return bib, 1.2

        # 3. 基础数字校验
        if (not hasattr(self, '_athlete_summary') or not self._athlete_summary) and not self._bib_ranges:
            if re.match(r'^[A-Z]*[0-9]{1,6}$', bib):
                return bib, 0.8
            return bib, 0.4

        # 4. 允许字符校验
        summary = getattr(self, '_athlete_summary', {})
        allowed_chars = summary.get('allowed_chars', set())
        
        for rule in self._bib_ranges:
            if rule['type'] == 'range':
                for char in rule['prefix']:
                    allowed_chars.add(char)
            elif rule['type'] == 'exact':
                for char in rule['value']:
                    if char.isalpha():
                        allowed_chars.add(char)

        for char in bib:
            if char.isalpha() and char not in allowed_chars:
                return bib, 0.1

        # 5. 模糊名单匹配与易错字符修正（核心强化）
        if hasattr(self, '_athlete_list') and self._athlete_list:
            # 使用更强大的模糊匹配函数
            threshold = 0.88 if len(bib) <= 3 else 0.92
            fuzzy_bib = fuzzy_match_bib(bib, self._athlete_list, threshold=threshold)
            
            if fuzzy_bib:
                # 如果模糊匹配成功，且原始号码和模糊号码很像，给高分
                # 特别是处理 A1019 vs A0475 这种可能的误报
                if fuzzy_bib == bib:
                    return bib, 2.0
                else:
                    logger.info(f"[Detector] 号码 {bib} 模糊匹配到名单: {fuzzy_bib}")
                    return fuzzy_bib, 1.6 # 模糊匹配给较高分，但略低于精准匹配

        # 号码段已配置但走到这里 = 格式明显不匹配（如比赛用A0001，识别出12345）
        # 给极低分，防止靠票数堆积赢过真正的号码
        if self._bib_ranges:
            return bib, 0.1
        return bib, 0.3

    def _is_in_bib_ranges(self, bib: str) -> bool:
        """检查号码是否在定义的合法号段内"""
        if not self._bib_ranges:
            return True

        clean_bib = bib.strip().upper()
        match = re.match(r'^([A-Z]*)([0-9]+)$', clean_bib)
        if not match:
            for rule in self._bib_ranges:
                if rule['type'] == 'exact' and clean_bib == rule['value']:
                    return True
            return False
            
        prefix = match.group(1)
        try:
            num = int(match.group(2))
        except ValueError:
            return False

        for r in self._bib_ranges:
            if r['type'] == 'range':
                if r['prefix'] == prefix:
                    if r['start'] <= num <= r['end']:
                        return True
            elif r['type'] == 'exact':
                if clean_bib == r['value']:
                    return True
        return False

    def _map_model_class_ids(self, names: Any) -> bool:
        """根据模型类别名动态映射 person/bib 类别索引。"""
        normalized: Dict[int, str] = {}
        if isinstance(names, dict):
            items = names.items()
        elif isinstance(names, (list, tuple)):
            items = enumerate(names)
        else:
            return False

        person_keywords = ("person", "athlete", "rider", "cyclist", "runner", "human", "运动员", "行人")
        bike_keywords = ("bike", "bicycle", "cycler", "cycling", "自行车", "骑行")
        bib_keywords = ("bib", "number", "plate", "tag", "号码", "号牌")

        explicit_person_ids: Set[int] = set()
        bike_ids: Set[int] = set()
        bib_ids: Set[int] = set()

        for key, value in items:
            try:
                idx = int(key)
            except Exception:
                continue
            label = str(value).strip().lower()
            normalized[idx] = label
            if any(keyword in label for keyword in person_keywords):
                explicit_person_ids.add(idx)
            if any(keyword in label for keyword in bike_keywords):
                bike_ids.add(idx)
            if any(keyword in label for keyword in bib_keywords):
                bib_ids.add(idx)

        if not normalized:
            return False

        person_ids: Set[int] = set(explicit_person_ids)

        # 兜底 1：bike-only 模型（如 {0:'bike',1:'BIB'}）
        if not person_ids and bike_ids:
            person_ids = set(bike_ids)

        # 兜底 2：如果仍没识别出 person 类别，则把“非 bib 类”当 person
        if not person_ids:
            person_ids = set(normalized.keys()) - bib_ids
            if not person_ids:
                person_ids = {min(normalized.keys())}

        # 标记 bike-only 检测模式，用于更强的几何与背景过滤
        self.bike_model_mode = bool(person_ids and all(
            any(keyword in normalized.get(pid, "") for keyword in bike_keywords)
            for pid in person_ids
        ))

        # bike-only 模型下关闭 bib-driven 虚拟人框，避免龙门文字触发虚拟目标
        if self.bike_model_mode:
            self.bib_driven_detection = False

        self.person_class_ids = set(person_ids)
        self.bib_class_ids = set(bib_ids)
        return True

    # =========================================================================
    # 配置方法
    # =========================================================================
    
    def load_model(self) -> bool:
        """加载模型"""
        try:
            import torch
            device = '0' if torch.cuda.is_available() else 'cpu'
            logger.info(f"[Detector-{self.source_id}] 使用设备: {device}")

            if self._model is None:
                logger.info(f"[Detector-{self.source_id}] 加载YOLO模型: {self.model_path}")
                self._model = YOLO(self.model_path)
                model_suffix = Path(self.model_path).suffix.lower()
                if model_suffix in [".pt", ".pth"]:
                    self._model.to('cuda' if torch.cuda.is_available() else 'cpu')

            # 动态类别映射：避免模型类别索引变化导致“一个人都不刷新”
            try:
                names = {}
                if hasattr(self._model, 'model') and hasattr(self._model.model, 'names'):
                    names = self._model.model.names or {}
                elif hasattr(self._model, 'names'):
                    names = self._model.names or {}

                if self._map_model_class_ids(names):
                    logger.info(f"[Detector-{self.source_id}] 类别映射: person={sorted(self.person_class_ids)}, bib={sorted(self.bib_class_ids)}")
                else:
                    logger.warning(f"[Detector-{self.source_id}] 未读到有效类别名，使用默认类索引 person={sorted(self.person_class_ids)}, bib={sorted(self.bib_class_ids)}")
            except Exception as e:
                logger.warning(f"[Detector-{self.source_id}] 类别映射解析失败，使用默认类索引: {e}")

            if self._ocr is not None:
                self.start_ocr_worker()

            logger.info(f"[Detector-{self.source_id}] 检测器准备就绪")
            return True
        except Exception as e:
            logger.error(f"[Detector-{self.source_id}] 模型加载失败: {e}")
            return False

    def set_athlete_list(self, summary: Dict[str, Any]):
        """设置选手名单"""
        with self._lock:
            self._athlete_summary = summary
            self._athlete_list = summary.get('bibs', set())
            self._vlm_athlete_list = list(self._athlete_list)[:50]
            self.set_bib_ranges(summary.get('bib_ranges_str', ''))
            logger.info(f"[Detector] 已加载 {len(self._athlete_list)} 名选手名单")

    def set_bib_ranges(self, range_str: str):
        """设置合法号码段，支持跨前缀范围如 A0001-D9999"""
        with self._lock:
            self._bib_ranges = []
            if not range_str:
                return

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
                                # 跨前缀：A0001-D9999 → A + B + C + D
                                for ch_code in range(ord(s_prefix), ord(e_prefix) + 1):
                                    p = chr(ch_code)
                                    r_start = start_num if p == s_prefix else 1
                                    r_end = end_num if p == e_prefix else 9999
                                    self._bib_ranges.append({
                                        'type': 'range', 'prefix': p,
                                        'start': r_start, 'end': r_end
                                    })
                            else:
                                logger.warning(f"[Detector] 无法解析号码范围: {part}")
                    except:
                        continue
                else:
                    self._bib_ranges.append({'type': 'exact', 'value': part})

            if self._bib_ranges:
                for r in self._bib_ranges:
                    if r['type'] == 'range':
                        logger.info(f"[Detector] 号码范围: {r['prefix']}{r['start']:04d} ~ {r['prefix']}{r['end']:04d}")
                logger.info(f"[Detector] 共设置 {len(self._bib_ranges)} 条号码段规则")

    def set_finish_line(self, pt1: Tuple[int, int], pt2: Tuple[int, int]):
        """设置终点线"""
        self._line_pt1 = pt1
        self._line_pt2 = pt2

    def set_on_crossing(self, callback: Callable[[CrossingEvent], None]):
        """设置过线事件回调"""
        self._on_crossing = callback

    def set_on_bib_update(self, callback: Callable[[int, str, float, BibStatus, int, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]], None]):
        """设置号码更新回调 (用于过线后补录)"""
        self._on_bib_update = callback

    def set_roi_polygon(self, points: Optional[List[Tuple[int, int]]]):
        """设置检测区域"""
        if not points or len(points) < 3:
            self._roi_polygon = None
            self.enable_roi_filter = False
        else:
            self._roi_polygon = np.array(points, dtype=np.int32)
            self.enable_roi_filter = True

    def set_gate_guard_enabled(self, enabled: bool):
        """设置龙门安全模式开关"""
        self.enable_gate_guard = bool(enabled)

    def _bicycle_evidence_in_crop(
        self,
        crop: Optional[np.ndarray],
    ) -> Optional[Tuple[bool, bool, bool]]:
        """Return bicycle presence, ownership, and overlapping cycle ambiguity."""
        if self._athlete_validator is None or crop is None or crop.size == 0:
            return None

        try:
            results = self._athlete_validator.predict(
                source=crop,
                imgsz=self.athlete_validator_imgsz,
                conf=self.athlete_validator_conf,
                iou=0.60,
                verbose=False,
                device="cpu",
            )
            if not results:
                return False, False, False

            result = results[0]
            boxes = getattr(result, "boxes", None)
            class_values = getattr(boxes, "cls", None) if boxes is not None else None
            if class_values is None:
                return False, False, False
            class_ids = class_values.tolist() if hasattr(class_values, "tolist") else list(class_values)
            xyxy_values = getattr(boxes, "xyxy", None)
            if xyxy_values is not None:
                xyxy_values = (
                    xyxy_values.tolist()
                    if hasattr(xyxy_values, "tolist")
                    else list(xyxy_values)
                )
            names = getattr(result, "names", {}) or {}
            class_names = []
            for class_id in class_ids:
                index = int(class_id)
                name = names.get(index, "") if isinstance(names, dict) else names[index]
                class_names.append(str(name).strip().lower())
            if xyxy_values is None:
                has_bicycle = "bicycle" in class_names
                has_motorcycle = "motorcycle" in class_names
                return has_bicycle, has_bicycle, has_bicycle and has_motorcycle

            crop_height, crop_width = crop.shape[:2]
            crop_area = max(1, crop_width * crop_height)
            has_bicycle = False
            has_centered_bicycle = False
            bicycle_boxes: List[List[float]] = []
            motorcycle_boxes: List[List[float]] = []
            for box_index, class_id in enumerate(class_ids):
                index = int(class_id)
                name = names.get(index, "") if isinstance(names, dict) else names[index]
                normalized_name = str(name).strip().lower()
                if normalized_name not in {"bicycle", "motorcycle"}:
                    continue
                if box_index >= len(xyxy_values) or len(xyxy_values[box_index]) < 4:
                    continue
                x1, y1, x2, y2 = [float(value) for value in xyxy_values[box_index][:4]]
                clipped_x1 = max(0.0, x1)
                clipped_y1 = max(0.0, y1)
                clipped_x2 = min(float(crop_width), x2)
                clipped_y2 = min(float(crop_height), y2)
                clipped_width = max(0.0, clipped_x2 - clipped_x1)
                clipped_height = max(0.0, clipped_y2 - clipped_y1)
                area_ratio = (clipped_width * clipped_height) / crop_area
                if area_ratio < self.athlete_validator_min_bicycle_area_ratio:
                    continue

                clipped_box = [clipped_x1, clipped_y1, clipped_x2, clipped_y2]
                if normalized_name == "motorcycle":
                    motorcycle_boxes.append(clipped_box)
                    continue

                has_bicycle = True
                bicycle_boxes.append(clipped_box)
                center_x_ratio = ((clipped_x1 + clipped_x2) * 0.5) / max(1.0, float(crop_width))
                if (
                    self.athlete_validator_bicycle_center_x_min
                    <= center_x_ratio
                    <= self.athlete_validator_bicycle_center_x_max
                ):
                    has_centered_bicycle = True

            has_cycle_ambiguity = has_centered_bicycle and any(
                bbox_iou(bicycle_box, motorcycle_box) >= 0.50
                for bicycle_box in bicycle_boxes
                for motorcycle_box in motorcycle_boxes
            )
            return has_bicycle, has_centered_bicycle, has_cycle_ambiguity
        except Exception as exc:
            logger.warning(f"[Detector-{self.source_id}] 运动员二次校验失败，保留事件: {exc}")
            return None

    def _verify_bicycle_in_crop(self, crop: Optional[np.ndarray]) -> Optional[bool]:
        """Use the optional event-only validator to confirm bicycle presence."""
        evidence = self._bicycle_evidence_in_crop(crop)
        return evidence[0] if evidence is not None else None

    @staticmethod
    def _build_athlete_validation_context_crops(
        frame: Optional[np.ndarray],
        bbox: List[int],
    ) -> List[np.ndarray]:
        """Build wider event-only bicycle evidence from the original frame."""
        if frame is None or frame.size == 0 or not bbox or len(bbox) != 4:
            return []

        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [int(value) for value in bbox]
        box_width = max(1, x2 - x1)
        box_height = max(1, y2 - y1)
        crops: List[np.ndarray] = []
        for padding_ratio in (0.25, 0.50):
            pad_x = int(round(box_width * padding_ratio))
            pad_y = int(round(box_height * padding_ratio))
            crop = frame[
                max(0, y1 - pad_y):min(height, y2 + pad_y),
                max(0, x1 - pad_x):min(width, x2 + pad_x),
            ].copy()
            if crop.size > 0:
                crops.append(crop)
        return crops

    def _passes_athlete_event_validation(
        self,
        state: TrackState,
        crop: Optional[np.ndarray],
        context_crops: Optional[List[np.ndarray]] = None,
        allow_edge_bicycle: bool = False,
    ) -> bool:
        """Apply the event profile's optional equipment validation."""
        if self.event_profile.required_equipment is None:
            return True

        athlete_evidence = self._bicycle_evidence_in_crop(crop)
        if athlete_evidence is None:
            return True
        has_bicycle, has_centered_bicycle, has_cycle_ambiguity = athlete_evidence
        if not has_bicycle:
            return False
        if has_cycle_ambiguity:
            return True

        valid_context_crops = [
            context_crop
            for context_crop in (context_crops or [])
            if context_crop is not None and context_crop.size > 0
        ]
        if not valid_context_crops:
            return True

        for context_crop in valid_context_crops:
            context_evidence = self._bicycle_evidence_in_crop(context_crop)
            if context_evidence is None:
                return True
            context_has_bicycle, context_has_centered_bicycle, _ = context_evidence
            if has_centered_bicycle or allow_edge_bicycle:
                if context_has_bicycle:
                    return True
            elif context_has_centered_bicycle:
                return True
        return False

    def _passes_profile_athlete_geometry(
        self,
        bbox: List[int],
        frame_shape: Tuple[int, int],
    ) -> bool:
        """Reject distant spectator-sized boxes for profiles with a fixed finish camera."""
        if self.profile_min_athlete_area_ratio <= 0.0:
            return True
        if bbox is None or len(bbox) != 4:
            return False
        height, width = int(frame_shape[0]), int(frame_shape[1])
        x1, y1, x2, y2 = [int(value) for value in bbox]
        box_width = max(0, x2 - x1)
        box_height = max(0, y2 - y1)
        area_ratio = (box_width * box_height) / max(1, width * height)
        height_ratio = box_height / max(1, height)
        return (
            area_ratio >= self.profile_min_athlete_area_ratio
            and height_ratio >= self.profile_min_athlete_height_ratio
        )

    def _passes_athlete_aspect_filter(
        self,
        bbox: List[float],
        confidence: float,
        frame_shape: Tuple[int, int],
    ) -> bool:
        """Allow a tightly bounded exception for athletes entering at a frame edge."""
        if bbox is None or len(bbox) != 4:
            return False
        height, width = int(frame_shape[0]), int(frame_shape[1])
        x1, y1, x2, y2 = [float(value) for value in bbox]
        box_width = x2 - x1
        box_height = y2 - y1
        if box_width <= 0 or box_height <= 0:
            return False

        aspect_ratio = box_height / box_width
        if self.aspect_min <= aspect_ratio <= self.aspect_max:
            return True

        touches_edge = x1 <= 1.0 or x2 >= width - 1.0
        return (
            aspect_ratio > self.aspect_max
            and aspect_ratio <= 7.0
            and touches_edge
            and float(confidence) >= 0.50
            and y1 >= height * 0.30
            and y2 >= height * 0.52
            and box_height < height * 0.42
            and box_width >= max(float(self.min_bw), width * 0.02)
        )

    def set_crossing_direction(self, direction: Optional[str] = None):
        """设置过线方向"""
        self.crossing_direction = direction

    def _is_gate_like_bbox(self, bbox: List[int], frame_shape: Tuple[int, int]) -> bool:
        """判断是否为龙门/拱门类大结构框（用于误检抑制）。"""
        if not bbox or len(bbox) < 4:
            return False

        height, width = int(frame_shape[0]), int(frame_shape[1])
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        area_ratio = (bw * bh) / max(1, width * height)
        aspect_ratio = bh / max(1, bw)

        if y1 < int(height * 0.20) and bh > int(height * 0.40) and bw > int(width * 0.10) and area_ratio > 0.05:
            return True
        if y1 < int(height * 0.30) and bw > int(width * 0.18) and bh > int(height * 0.32) and aspect_ratio < 2.8:
            return True
        return False

    def reset(self, start_event_id: int = 0):
        """重置状态"""
        with self._lock:
            self._track_states.clear()
            self._participant_ocr_states.clear()
            self._finished_bibs.clear()
            self._recent_events.clear()
            self._event_id = start_event_id
            self._start_time = None
            self._split_track_pool.clear()
            self._forced_bib_rescue_last.clear()
            self._wide_split_last.clear()
            self._participant_identity_manager = ParticipantIdentityManager(
                config=IdentityConfig(event_profile_name=self.event_profile.name)
            )
            self._crossing_lifecycle = CrossingLifecycle(
                mode=self.event_profile.crossing_mode,
                min_lap_interval_ms=(
                    30_000.0 if self.event_profile.crossing_mode == "multi_lap" else 0.0
                ),
            )
            self._identity_segment_id = 0

    def stop(self):
        """停止检测器"""
        self._ocr_running = False
        if hasattr(self, '_ocr_thread') and self._ocr_thread and self._ocr_thread.is_alive():
            self._ocr_thread.join(timeout=2.0)
        if self._vlm_pipeline:
            self._vlm_pipeline.shutdown()
        logger.info(f"[Detector-{self.source_id}] 检测器已停止")

    def _cleanup_stale_tracks(self, current_time: float) -> int:
        """
        清理超时的跟踪状态，防止内存泄漏
        """
        # 检查是否需要清理
        if current_time - self._last_cleanup_time < self._cleanup_interval:
            return 0

        self._last_cleanup_time = current_time

        with self._lock:
            # 找出过期的track_id
            stale_ids = []
            for tid, state in self._track_states.items():
                time_since_seen = current_time - state.last_seen_time
                
                # 动态清理阈值：虚拟 ID 或已发送事件的目标保留更久，防止重复触发
                threshold = self._stale_threshold
                if tid < 0: # 虚拟 ID
                    threshold *= 5
                elif state.event_emitted: # 已发送过事件
                    threshold *= 3
                    
                if time_since_seen > threshold:
                    stale_ids.append(tid)

            # 清理前释放帧缓存内存
            for tid in stale_ids:
                state = self._track_states[tid]
                # 释放帧缓存
                if hasattr(state, 'best_bib_frame') and state.best_bib_frame is not None:
                    state.best_bib_frame = None
                if hasattr(state, 'best_athlete_crop') and state.best_athlete_crop is not None:
                    state.best_athlete_crop = None
                # 清理bib_crops_cache中的帧引用
                if hasattr(state, 'bib_crops_cache'):
                    state.bib_crops_cache.clear()
                if hasattr(state, 'fallback_bib_crops_cache'):
                    state.fallback_bib_crops_cache.clear()

            # 清理
            for tid in stale_ids:
                del self._track_states[tid]

            if stale_ids:
                logger.info(f"[Detector-{self.source_id}] 清理 {len(stale_ids)} 个过期跟踪对象，当前剩余 {len(self._track_states)} 个")

            return len(stale_ids)

    # =========================================================================
    # 核心处理方法
    # =========================================================================

    def get_vlm_status(self) -> Optional[dict]:
        """获取 VLM 工作状态"""
        if self._vlm_enabled and self._vlm_pipeline:
            return self._vlm_pipeline.get_status()
        return None

    def get_ocr_queue_size(self) -> int:
        """获取本地 OCR 队列长度"""
        return self._ocr_queue.qsize() if hasattr(self, '_ocr_queue') else 0

    def get_crowd_status(self) -> Dict[str, Any]:
        """获取拥挤度状态（用于 UI 显示）。"""
        return dict(self._last_crowd_stats)

    def _compute_crowd_score(self, athletes: List[dict], height: int, width: int) -> float:
        """计算当前帧拥挤度分数（0~1）。"""
        if not athletes:
            return 0.0

        count = len(athletes)
        if count == 1:
            return 0.06

        pair_count = 0
        overlap_sum = 0.0
        close_pairs = 0
        tiny_stable_pairs = 0

        for i in range(count):
            a = athletes[i]
            ax = a.get('center_x', (a['bbox'][0] + a['bbox'][2]) * 0.5)
            ay = a.get('center_y', (a['bbox'][1] + a['bbox'][3]) * 0.5)
            aw = max(1.0, a['bbox'][2] - a['bbox'][0])
            ah = max(1.0, a['bbox'][3] - a['bbox'][1])
            diag = max(1.0, np.sqrt(aw * aw + ah * ah))

            for j in range(i + 1, count):
                b = athletes[j]
                bx = b.get('center_x', (b['bbox'][0] + b['bbox'][2]) * 0.5)
                by = b.get('center_y', (b['bbox'][1] + b['bbox'][3]) * 0.5)
                bw = max(1.0, b['bbox'][2] - b['bbox'][0])
                bh = max(1.0, b['bbox'][3] - b['bbox'][1])
                bdiag = max(1.0, np.sqrt(bw * bw + bh * bh))

                pair_count += 1
                iou = bbox_iou(a['bbox'], b['bbox'])
                overlap_sum += iou

                dx = abs(ax - bx)
                dy = abs(ay - by)
                norm_dx = dx / max(1.0, min(diag, bdiag))
                if norm_dx < 0.95 and dy < max(26.0, min(ah, bh) * 0.35):
                    close_pairs += 1
                if iou > 0.22 and dx < max(14.0, min(aw, bw) * 0.18):
                    tiny_stable_pairs += 1

        mean_overlap = overlap_sum / max(1, pair_count)
        overlap_score = min(1.0, mean_overlap / 0.30)
        close_score = close_pairs / max(1, pair_count)
        tiny_pair_score = tiny_stable_pairs / max(1, pair_count)
        density_score = min(1.0, max(0, count - 1) / 6.0)

        score = (
            overlap_score * 0.33 +
            close_score * 0.32 +
            tiny_pair_score * 0.20 +
            density_score * 0.15
        )
        return float(max(0.0, min(1.0, score)))

    def _infer_crowd_target_level(self, score: float) -> str:
        current = self._crowd_level
        if current == "dense":
            if score <= self._crowd_exit_dense:
                return "medium" if score >= self._crowd_enter_medium else "sparse"
            return "dense"

        if current == "medium":
            if score >= self._crowd_enter_dense:
                return "dense"
            if score <= self._crowd_exit_medium:
                return "sparse"
            return "medium"

        if score >= self._crowd_enter_dense:
            return "dense"
        if score >= self._crowd_enter_medium:
            return "medium"
        return "sparse"

    def _apply_crowd_level_params(self, level: str) -> None:
        if level == "dense":
            self.parallel_center_min = max(8, int(round(self._base_parallel_center_min * 0.82)))
            self.parallel_center_ratio = max(0.12, self._base_parallel_center_ratio * 0.82)
            self.parallel_time_window = max(self._base_parallel_time_window, 0.25)
            self.same_frame_iou_threshold = max(0.82, self._base_same_frame_iou_threshold + 0.12)
            self.iou_dup_threshold = min(0.88, self._base_iou_dup_threshold + 0.10)
            self.center_dup_threshold = max(8, int(round(self._base_center_dup_threshold * 0.82)))
            self.max_athletes_per_frame = max(self._base_max_athletes_per_frame, 22)
            self.cross_confirm_frames = max(1, int(self._base_cross_confirm_frames))
            return

        if level == "medium":
            self.parallel_center_min = max(10, int(round(self._base_parallel_center_min * 0.90)))
            self.parallel_center_ratio = max(0.14, self._base_parallel_center_ratio * 0.90)
            self.parallel_time_window = max(self._base_parallel_time_window, 0.20)
            self.same_frame_iou_threshold = max(0.78, self._base_same_frame_iou_threshold + 0.06)
            self.iou_dup_threshold = min(0.82, self._base_iou_dup_threshold + 0.05)
            self.center_dup_threshold = max(10, int(round(self._base_center_dup_threshold * 0.90)))
            self.max_athletes_per_frame = max(self._base_max_athletes_per_frame, 18)
            self.cross_confirm_frames = max(1, int(self._base_cross_confirm_frames))
            return

        self.parallel_center_min = self._base_parallel_center_min
        self.parallel_center_ratio = self._base_parallel_center_ratio
        self.parallel_time_window = self._base_parallel_time_window
        self.same_frame_iou_threshold = self._base_same_frame_iou_threshold
        self.iou_dup_threshold = self._base_iou_dup_threshold
        self.center_dup_threshold = self._base_center_dup_threshold
        self.max_athletes_per_frame = self._base_max_athletes_per_frame
        self.cross_confirm_frames = self._base_cross_confirm_frames

    def _apply_crowd_adaptive_policy(self, athletes: List[dict], height: int, width: int, current_time: float) -> None:
        if not self.enable_crowd_adaptive:
            return

        score = self._compute_crowd_score(athletes, height, width)
        self._crowd_score_window.append(score)
        smooth_score = float(sum(self._crowd_score_window) / max(1, len(self._crowd_score_window)))

        target = self._infer_crowd_target_level(smooth_score)
        if target != self._crowd_level:
            if self._crowd_pending_level != target:
                self._crowd_pending_level = target
                self._crowd_pending_count = 1
            else:
                self._crowd_pending_count += 1

            if self._crowd_pending_count >= self._crowd_hold_frames:
                prev = self._crowd_level
                self._crowd_level = target
                self._crowd_pending_level = None
                self._crowd_pending_count = 0
                logger.info(
                    f"[Detector-{self.source_id}] 拥挤等级切换: {prev} -> {target} "
                    f"(score={smooth_score:.2f})"
                )
                self._apply_crowd_level_params(self._crowd_level)
        else:
            self._crowd_pending_level = None
            self._crowd_pending_count = 0
            self._apply_crowd_level_params(self._crowd_level)

        self._crowd_score = smooth_score
        self._last_crowd_stats = {
            "level": self._crowd_level,
            "score": round(self._crowd_score, 3),
            "raw_score": round(score, 3),
            "athletes": len(athletes),
            "time": round(current_time, 3),
        }

    def _update_crossing_vote(self, state: TrackState, current_time: float,
                              has_cross_signal: bool, is_strong_signal: bool = False) -> bool:
        """过线信号投票：连续信号确认，减少抖动导致的重复/误触发。"""
        if state.crossed:
            return False

        # 无信号时：只在超过窗口后清空，给相邻帧一点容错
        if not has_cross_signal:
            if state.cross_signal_count > 0 and (current_time - state.cross_signal_last_time) > self.cross_signal_window:
                state.cross_signal_count = 0
            return False

        # 有信号：在时间窗口内累计，否则重新计数
        if (current_time - state.cross_signal_last_time) > self.cross_signal_window:
            state.cross_signal_count = 1
        else:
            state.cross_signal_count += 1
        state.cross_signal_last_time = current_time

        required_votes = 1 if is_strong_signal else max(1, int(self.cross_confirm_frames))
        if state.cross_signal_count >= required_votes:
            state.cross_signal_count = 0
            return True
        return False

    def _try_sync_crossing_ocr(
        self,
        track_id: int,
        state: TrackState,
        current_time: float,
        event_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """过线短窗同步 OCR（轻量兜底，避免事件长期 Bib None）。"""
        if not getattr(self, 'realtime_ocr_enabled', True):
            return

        candidates = (
            self._get_crossing_ocr_candidates(state, event_data)
            if event_data
            else self._get_ocr_candidates(state)
        )
        if self._ocr is None or not candidates:
            return

        if state.best_bib and state.best_bib_conf >= max(self.ocr_conf_threshold, 0.75):
            return

        attempts = getattr(state, '_sync_cross_ocr_attempts', 0)
        if attempts >= 2:
            return

        last_sync_time = getattr(state, '_last_sync_cross_ocr_time', 0.0)
        min_interval = 0.35 if attempts > 0 else 0.20
        if current_time - last_sync_time < min_interval:
            return

        qsize = self.get_ocr_queue_size()
        if qsize > max(12, self.ocr_queue_limit):
            return

        # 同步OCR只取1个最佳候选，避免阻塞帧处理（每次OCR约100-300ms）
        max_candidates = 1
        setattr(state, '_last_sync_cross_ocr_time', current_time)
        setattr(state, '_sync_cross_ocr_attempts', attempts + 1)

        for quality, bib_crop, frame_ref, bib_bbox in candidates[:max_candidates]:
            if bib_crop is None or bib_crop.size == 0:
                continue

            try:
                ocr_input = self._preprocess_for_ocr(bib_crop)
                lines, _ = self._ocr(ocr_input)
            except Exception as e:
                logger.debug(f"[Detector-{self.source_id}] ID {track_id} 同步 OCR 失败: {e}")
                continue

            if not lines:
                continue

            best_text = None
            best_conf = 0.0
            for _, raw_text, raw_conf in lines:
                if self.only_numeric:
                    clean_text = ''.join(c for c in str(raw_text) if c.isdigit())
                else:
                    clean_text = ''.join(c for c in str(raw_text) if c.isalnum())

                if not clean_text:
                    continue

                validated_text, val_score = self._validate_bib(clean_text)
                final_conf = float(raw_conf) * max(0.1, float(val_score))

                has_athlete_list = hasattr(self, '_athlete_list') and bool(self._athlete_list)
                if has_athlete_list and validated_text not in self._athlete_list and final_conf < 0.80:
                    continue

                if final_conf > best_conf:
                    best_conf = final_conf
                    best_text = validated_text

            if not best_text:
                continue

            min_accept_conf = max(0.48, state.best_bib_conf + 0.05)
            if best_conf < min_accept_conf:
                continue

            state.best_bib = best_text
            state.best_bib_conf = best_conf
            state.best_bib_crop = bib_crop.copy()
            state.best_bib_bbox = bib_bbox
            state.best_bib_quality = quality
            if frame_ref is not None:
                state.best_bib_frame = frame_ref

            logger.info(f"[Detector-{self.source_id}] ID {track_id} 过线同步 OCR 命中: {best_text} (conf={best_conf:.2f})")
            break

    def _try_backfill_recent_unknown_event(self, track_id: int, athlete_bbox: Optional[List[int]],
                                           state: TrackState, current_time: float) -> Optional[int]:
        """短时间补号：将新识别号码回填到刚过线的 UNKNOWN 事件（严格约束，避免串号）。"""
        if not state.best_bib:
            return None

        bib_text = str(state.best_bib).strip().upper()
        if not bib_text or not any(ch.isdigit() for ch in bib_text):
            return None

        if state.best_bib_conf < max(self.ocr_conf_threshold, 0.72):
            return None

        if athlete_bbox is None or len(athlete_bbox) != 4:
            return None

        ax1, ay1, ax2, ay2 = [int(v) for v in athlete_bbox]
        acx = (ax1 + ax2) * 0.5
        acy = (ay1 + ay2) * 0.5

        with self._lock:
            candidates: List[Tuple[float, int, int]] = []
            for idx in range(len(self._recent_events) - 1, -1, -1):
                rec = self._recent_events[idx]
                rec_bib = str(rec.get('bib') or '').strip().upper()
                if rec_bib and rec_bib != 'UNKNOWN':
                    continue

                rec_tid = rec.get('tid')
                if rec_tid is None:
                    continue
                try:
                    rec_tid_i = int(rec_tid)
                except Exception:
                    continue
                if rec_tid_i == int(track_id):
                    continue

                rec_time = float(rec.get('time') or 0.0)
                dt = current_time - rec_time
                if dt < 0.0 or dt > 1.8:
                    continue

                rec_bbox = rec.get('bbox')
                if not rec_bbox or len(rec_bbox) != 4:
                    continue
                rx1, ry1, rx2, ry2 = [int(v) for v in rec_bbox]
                rcx = (rx1 + rx2) * 0.5
                rcy = (ry1 + ry2) * 0.5

                iou = bbox_iou((ax1, ay1, ax2, ay2), (rx1, ry1, rx2, ry2))
                dx = abs(acx - rcx)
                dy = abs(acy - rcy)

                if not (iou >= 0.25 or (dx <= 70 and dy <= 120)):
                    continue

                score = iou * 1.8 + max(0.0, 1.0 - dx / 120.0) * 0.4 + max(0.0, 1.0 - dy / 160.0) * 0.2 - dt * 0.15
                candidates.append((score, idx, rec_tid_i))

            if not candidates:
                return None

            candidates.sort(key=lambda item: item[0], reverse=True)
            if len(candidates) > 1:
                if (candidates[0][0] - candidates[1][0]) < 0.30:
                    return None

            _, best_idx, best_tid = candidates[0]
            self._recent_events[best_idx]['bib'] = state.best_bib
            return best_tid

    def _calculate_image_quality(self, image: np.ndarray) -> float:
        """
        计算图像质量评分 (字符像素规模 + 面积 + 清晰度)
        """
        if image is None or image.size == 0:
            return 0.0

        h, w = image.shape[:2]
        area = h * w
        width_score = min(1.0, w / 64.0)
        height_score = min(1.0, h / 48.0)
        resolution_score = (width_score + height_score) * 0.5
        area_score = min(1.0, area / 3072.0)

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        except Exception:
            laplacian_var = 0.0

        sharpness_score = min(1.0, laplacian_var / 700.0)
        return resolution_score * 0.55 + area_score * 0.25 + sharpness_score * 0.20

    def _has_strong_non_bib_signature(self, crop: np.ndarray) -> bool:
        """Reject only obvious solid-color regions that lack text-like structure."""
        if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0:
            return True
        height, width = crop.shape[:2]
        if height < 12 or width < 12:
            return True

        y_pad = max(1, height // 6)
        x_pad = max(1, width // 8)
        central = crop[y_pad:height - y_pad, x_pad:width - x_pad]
        if central.size == 0:
            central = crop

        try:
            gray = cv2.cvtColor(central, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(central, cv2.COLOR_BGR2HSV)
            edges = cv2.Canny(gray, 60, 150)
        except Exception:
            return False

        gray_std = float(gray.std())
        mean_saturation = float(hsv[..., 1].mean())
        edge_density = float(np.count_nonzero(edges)) / max(1, edges.size)
        return bool(
            mean_saturation >= self.bib_nontext_saturation_min
            and gray_std <= self.bib_nontext_gray_std_max
            and edge_density <= self.bib_nontext_edge_density_max
        )

    @staticmethod
    def _candidate_relative_bib_position(candidate: Any) -> Optional[Tuple[float, float]]:
        if not isinstance(candidate, BibEvidenceCandidate) or not candidate.owner_validated:
            return None
        if len(candidate.athlete_bbox) != 4 or len(candidate.bib_bbox) != 4:
            return None
        ax1, ay1, ax2, ay2 = [float(value) for value in candidate.athlete_bbox]
        bx1, by1, bx2, by2 = [float(value) for value in candidate.bib_bbox]
        athlete_width = ax2 - ax1
        athlete_height = ay2 - ay1
        if athlete_width <= 0.0 or athlete_height <= 0.0:
            return None
        return (
            ((bx1 + bx2) * 0.5 - ax1) / athlete_width,
            ((by1 + by2) * 0.5 - ay1) / athlete_height,
        )

    @staticmethod
    def _bbox_contains_bbox(
        outer_bbox: Any,
        inner_bbox: Any,
        tolerance: float = 0.0,
    ) -> bool:
        if outer_bbox is None or inner_bbox is None:
            return False
        try:
            outer_values = list(outer_bbox)
            inner_values = list(inner_bbox)
            if len(outer_values) != 4 or len(inner_values) != 4:
                return False
            ox1, oy1, ox2, oy2 = [float(value) for value in outer_values]
            ix1, iy1, ix2, iy2 = [float(value) for value in inner_values]
        except (TypeError, ValueError):
            return False
        if ox2 <= ox1 or oy2 <= oy1 or ix2 <= ix1 or iy2 <= iy1:
            return False
        margin = max(0.0, float(tolerance))
        return (
            ox1 - margin <= ix1
            and oy1 - margin <= iy1
            and ix2 <= ox2 + margin
            and iy2 <= oy2 + margin
        )

    def _candidate_has_valid_owner(self, candidate: Any) -> bool:
        if not isinstance(candidate, BibEvidenceCandidate):
            return True
        return bool(
            candidate.owner_validated
            and self._bbox_contains_bbox(candidate.athlete_bbox, candidate.bib_bbox)
        )

    def _get_plausible_detected_bib_candidates(
        self,
        state: TrackState,
    ) -> List[Any]:
        candidates = []
        aspects = []
        for candidate in state.bib_crops_cache:
            if len(candidate) < 4:
                continue
            if not self._candidate_has_valid_owner(candidate):
                continue
            _, crop, _, bbox = candidate
            if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0:
                continue
            if not bbox or len(bbox) != 4:
                continue
            if self._has_strong_non_bib_signature(crop):
                continue
            width = max(1.0, float(bbox[2]) - float(bbox[0]))
            height = max(1.0, float(bbox[3]) - float(bbox[1]))
            aspect = width / height
            aspects.append(aspect)
            candidates.append((candidate, aspect))

        if len(aspects) >= 3 and float(np.median(aspects)) > self.bib_cache_max_median_aspect:
            return []

        filtered = [
            candidate
            for candidate, aspect in candidates
            if aspect <= self.bib_candidate_max_aspect
        ]
        positioned = [
            (index, candidate, self._candidate_relative_bib_position(candidate))
            for index, candidate in enumerate(filtered)
        ]
        positioned = [item for item in positioned if item[2] is not None]
        if len(positioned) < 2:
            return filtered

        best_indices = set()
        best_rank = (0, 0.0)
        for anchor_index, _, anchor_position in positioned:
            cluster_indices = set()
            cluster_quality = 0.0
            for candidate_index, candidate, position in positioned:
                if (
                    abs(position[0] - anchor_position[0]) <= self.bib_candidate_max_rel_dx
                    and abs(position[1] - anchor_position[1]) <= self.bib_candidate_max_rel_dy
                ):
                    cluster_indices.add(candidate_index)
                    cluster_quality += float(candidate[0])
            rank = (len(cluster_indices), cluster_quality)
            if rank > best_rank:
                best_rank = rank
                best_indices = cluster_indices

        if best_rank[0] < 2:
            return [
                candidate
                for candidate in filtered
                if self._candidate_relative_bib_position(candidate) is None
            ]
        return [candidate for index, candidate in enumerate(filtered) if index in best_indices]

    def _get_ocr_candidates(self, state: TrackState) -> List[Any]:
        """Prefer detector-backed bib crops, falling back to continuously refreshed torso crops."""
        if not self.ocr_pipeline_enabled:
            return []
        detected_candidates = self._get_plausible_detected_bib_candidates(state)
        if detected_candidates:
            best_detected = detected_candidates[0]
            quality, crop = best_detected[0], best_detected[1]
            crop_height, crop_width = crop.shape[:2] if crop is not None and crop.size > 0 else (0, 0)
            detected_is_usable = (
                crop_width >= self.ocr_detected_bib_min_width
                and crop_height >= self.ocr_detected_bib_min_height
                and quality >= self.ocr_detected_bib_min_quality
            )
            if detected_is_usable or not state.fallback_bib_crops_cache:
                return detected_candidates
            return [best_detected, state.fallback_bib_crops_cache[0]]
        return state.fallback_bib_crops_cache

    def _get_crossing_detected_bib_candidates(
        self,
        state: TrackState,
        event_data: Dict[str, Any],
    ) -> List[Any]:
        event_bbox = event_data.get("bbox") or []
        candidates = []
        for candidate in self._get_plausible_detected_bib_candidates(state):
            if isinstance(candidate, BibEvidenceCandidate):
                if not self._candidate_has_valid_owner(candidate):
                    continue
            else:
                candidate_bbox = candidate[3] if len(candidate) >= 4 else None
                if not self._bbox_contains_bbox(event_bbox, candidate_bbox, tolerance=5.0):
                    continue
            candidates.append(candidate)
        return candidates

    def _get_current_event_fallback_candidates(
        self,
        event_data: Dict[str, Any],
    ) -> List[BibEvidenceCandidate]:
        frame = event_data.get("frame")
        athlete_bbox = event_data.get("bbox") or []
        if frame is None or not self._bbox_contains_bbox(athlete_bbox, athlete_bbox):
            return []

        candidates = []
        for source, crop, bib_bbox in self._extract_bib_fallbacks_from_athlete(
            frame,
            athlete_bbox,
        ):
            if not self._bbox_contains_bbox(athlete_bbox, bib_bbox):
                continue
            candidates.append(
                BibEvidenceCandidate(
                    quality=float(self._calculate_image_quality(crop)),
                    crop=crop,
                    frame=frame,
                    frame_index=int(event_data.get("frame_index", -1)),
                    capture_time_ms=float(event_data.get("cross_time", 0.0) or 0.0) * 1000.0,
                    athlete_bbox=tuple(int(value) for value in athlete_bbox),
                    bib_bbox=tuple(int(value) for value in bib_bbox),
                    source=f"fallback:{source}",
                    owner_validated=True,
                )
            )
        candidates.sort(key=lambda item: item.quality, reverse=True)
        return candidates

    def _get_crossing_ocr_candidates(
        self,
        state: TrackState,
        event_data: Dict[str, Any],
        detected_candidates: Optional[List[Any]] = None,
    ) -> List[Any]:
        if not self.ocr_pipeline_enabled:
            return []
        if detected_candidates is None:
            detected_candidates = self._get_crossing_detected_bib_candidates(state, event_data)
        fallback_candidates = self._get_current_event_fallback_candidates(event_data)
        if not detected_candidates:
            return fallback_candidates

        best_detected = detected_candidates[0]
        quality, crop = best_detected[0], best_detected[1]
        crop_height, crop_width = crop.shape[:2] if crop is not None and crop.size > 0 else (0, 0)
        detected_is_usable = (
            crop_width >= self.ocr_detected_bib_min_width
            and crop_height >= self.ocr_detected_bib_min_height
            and quality >= self.ocr_detected_bib_min_quality
        )
        if detected_is_usable or not fallback_candidates:
            return detected_candidates
        return [best_detected, fallback_candidates[0]]

    def _cache_ocr_candidate(
        self,
        state: TrackState,
        candidate: Any,
        *,
        detected_bib: bool,
    ) -> None:
        if not self.ocr_pipeline_enabled:
            return
        cache = state.bib_crops_cache if detected_bib else state.fallback_bib_crops_cache
        if isinstance(candidate, BibEvidenceCandidate) and candidate.frame_index >= 0:
            for index, existing in enumerate(cache):
                if not isinstance(existing, BibEvidenceCandidate):
                    continue
                if existing.source != candidate.source or existing.frame_index != candidate.frame_index:
                    continue
                if candidate.quality > existing.quality:
                    cache[index] = candidate
                cache.sort(key=lambda item: item[0], reverse=True)
                return
        cache.append(candidate)
        cache.sort(key=lambda item: item[0], reverse=True)
        if len(cache) > state.max_cache_size:
            del cache[state.max_cache_size:]

    def _extract_bib_fallbacks_from_athlete(
        self,
        frame: np.ndarray,
        athlete_bbox: List[int],
    ) -> List[Tuple[str, np.ndarray, List[int]]]:
        """Extract profile-specific OCR regions when the detector has no bib box."""
        if frame is None or athlete_bbox is None or len(athlete_bbox) != 4:
            return []

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(value) for value in athlete_bbox]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        if bw < 20 or bh < 24:
            return []

        region_specs = {
            "torso": (("torso", 0.20, 0.18, 0.80, 0.74),),
            "back": (("back", 0.16, 0.12, 0.84, 0.68),),
            "rear_saddle": (
                ("rear_saddle_left", 0.02, 0.34, 0.58, 0.88),
                ("rear_saddle_right", 0.42, 0.34, 0.98, 0.88),
            ),
            "helmet": (
                ("helmet_left", 0.02, 0.00, 0.58, 0.34),
                ("helmet_right", 0.42, 0.00, 0.98, 0.34),
            ),
            "left_thigh": (("left_thigh", 0.00, 0.34, 0.62, 0.94),),
            "right_thigh": (("right_thigh", 0.38, 0.34, 1.00, 0.94),),
        }

        results: List[Tuple[str, np.ndarray, List[int]]] = []
        seen_boxes = set()
        for region in self.event_profile.bib_regions:
            for source, rx1, ry1, rx2, ry2 in region_specs.get(region, ()):
                fx1 = max(0, x1 + int(round(bw * rx1)))
                fy1 = max(0, y1 + int(round(bh * ry1)))
                fx2 = min(w, x1 + int(round(bw * rx2)))
                fy2 = min(h, y1 + int(round(bh * ry2)))
                bbox = (fx1, fy1, fx2, fy2)
                if bbox in seen_boxes or fx2 - fx1 < 14 or fy2 - fy1 < 14:
                    continue
                crop = frame[fy1:fy2, fx1:fx2].copy()
                if crop.size == 0:
                    continue
                seen_boxes.add(bbox)
                results.append((source, crop, list(bbox)))
        return results

    def _extract_bib_fallback_from_athlete(self, frame: np.ndarray, athlete_bbox: List[int]) -> Tuple[Optional[np.ndarray], Optional[List[int]]]:
        """当未检出 bib 框时，从运动员框提取躯干区域作为 OCR 兜底输入。"""
        if frame is None or athlete_bbox is None or len(athlete_bbox) != 4:
            return None, None

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in athlete_bbox]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        if bw < 20 or bh < 24:
            return None, None

        # 取躯干区域：自行车选手前倾时号码牌在框的上部（约15%~35%），
        # 上边界从 32% 提到 18%，避免兜底裁剪刚好错过前倾骑手的号码牌
        fx1 = max(0, x1 + int(bw * 0.20))
        fx2 = min(w, x1 + int(bw * 0.80))
        fy1 = max(0, y1 + int(bh * 0.18))
        fy2 = min(h, y1 + int(bh * 0.74))

        if fx2 - fx1 < 14 or fy2 - fy1 < 14:
            return None, None

        crop = frame[fy1:fy2, fx1:fx2].copy()
        if crop.size == 0:
            return None, None
        return crop, [fx1, fy1, fx2, fy2]

    @staticmethod
    def _identity_bbox(value: Any) -> Optional[Tuple[int, int, int, int]]:
        if value is None:
            return None
        try:
            values = list(value)
        except (TypeError, ValueError):
            return None
        if len(values) != 4:
            return None
        try:
            return tuple(int(round(float(item))) for item in values)
        except (TypeError, ValueError):
            return None

    def _resolve_participant_with_resolution(
        self,
        athlete: dict,
        timestamp: float,
        *,
        bib_bboxes=(),
        excluded_participant_ids: Optional[Set[str]] = None,
    ) -> Tuple[dict, Any]:
        participant_bbox = self._identity_bbox(athlete.get("bbox"))
        if participant_bbox is None:
            raise ValueError("athlete bbox must contain four numeric values")

        raw_track_id = athlete.get("track_id")
        try:
            raw_track_id = int(raw_track_id) if raw_track_id is not None else None
        except (TypeError, ValueError):
            raw_track_id = None
        if raw_track_id is not None and raw_track_id < 0:
            raw_track_id = None

        try:
            segment_id = int(athlete.get("segment_id", self._identity_segment_id))
        except (TypeError, ValueError):
            segment_id = self._identity_segment_id
        try:
            source_id = int(athlete.get("source_id", self.source_id))
        except (TypeError, ValueError):
            source_id = self.source_id

        normalized_bib_bboxes = []
        for bib_bbox in bib_bboxes or ():
            normalized = self._identity_bbox(bib_bbox)
            if normalized is not None:
                normalized_bib_bboxes.append(normalized)

        try:
            capture_time_ms = float(timestamp) * 1000.0
        except (TypeError, ValueError):
            capture_time_ms = time.time() * 1000.0

        observation = ParticipantObservation(
            source_id=source_id,
            segment_id=segment_id,
            frame_index=int(self._frame_count),
            capture_time_ms=capture_time_ms,
            raw_track_id=raw_track_id,
            participant_bbox=participant_bbox,
            person_bbox=self._identity_bbox(athlete.get("person_bbox")) or participant_bbox,
            equipment_bbox=self._identity_bbox(athlete.get("equipment_bbox")),
            bib_bboxes=tuple(normalized_bib_bboxes),
            confidence=float(athlete.get("conf", 0.0) or 0.0),
        )
        resolution = self._participant_identity_manager.resolve(
            observation,
            excluded_participant_ids=excluded_participant_ids,
        )
        resolved = dict(athlete)
        resolved["participant_id"] = resolution.participant.participant_id
        resolved["raw_track_ids"] = sorted(resolution.participant.raw_track_ids)
        resolved["identity_status"] = resolution.participant.identity_status
        return resolved, resolution

    def _inherit_fragment_motion_state(
        self,
        track_id: int,
        state: TrackState,
        participant_id: str,
        current_time: float,
    ) -> None:
        """Carry crossing motion across a confirmed raw-track fragment switch."""
        state.participant_id = str(participant_id or "")
        if not state.participant_id:
            return

        max_gap_seconds = max(
            0.0,
            float(self._participant_identity_manager.config.max_match_gap_ms) / 1000.0,
        )
        candidates = []
        for previous_track_id, previous_state in self._track_states.items():
            if previous_track_id == track_id:
                continue
            if previous_state.participant_id != state.participant_id:
                continue
            if previous_state.crossed or previous_state.event_emitted:
                continue
            gap_seconds = float(current_time - previous_state.last_seen_time)
            if 0.0 <= gap_seconds <= max_gap_seconds:
                candidates.append((previous_state.last_seen_time, previous_state))

        if not candidates:
            return

        _, previous_state = max(candidates, key=lambda item: item[0])
        state.prev_x = previous_state.prev_x
        state.prev_y = previous_state.prev_y
        state.start_pos = previous_state.start_pos
        state.start_time = previous_state.start_time
        state.max_displacement = previous_state.max_displacement
        state.last_positions = list(previous_state.last_positions)
        state.position_history = list(previous_state.position_history)
        state.avg_speed = previous_state.avg_speed
        state.start_line_dist = previous_state.start_line_dist
        state.observation_count += previous_state.observation_count
        state.cross_signal_count = previous_state.cross_signal_count
        state.cross_signal_last_time = previous_state.cross_signal_last_time
        if hasattr(previous_state, "_near_line_hits"):
            setattr(state, "_near_line_hits", getattr(previous_state, "_near_line_hits"))

    def resolve_participant(
        self,
        athlete: dict,
        timestamp: float,
        *,
        bib_bboxes=(),
    ) -> dict:
        """Resolve one raw athlete observation to stable participant metadata."""
        resolved, _ = self._resolve_participant_with_resolution(
            athlete,
            timestamp,
            bib_bboxes=bib_bboxes,
        )
        return resolved

    def process_frame(self, frame: np.ndarray, timestamp: float = None) -> Tuple[List[CrossingEvent], List[dict], List[dict]]:
        """
        处理一帧（增强版）
        """
        self.last_rejected_candidates = []
        if self._model is None:
            self.last_frame_metrics = {
                "raw_bikes": 0,
                "raw_bibs": 0,
                "validated_tracks": 0,
                "synthetic_tracks": 0,
                "raw_tracks": 0,
                "participants": 0,
                "track_fragments_merged": 0,
                "identity_ambiguities": 0,
                "roi_auto_disabled": False,
                "inference_ms": 0.0,
                "postprocess_ms": 0.0,
                "total_ms": 0.0,
            }
            return [], [], []

        _frame_t0 = time.time()
        _infer_ms = 0.0
        self._frame_count += 1

        if self._start_time is None:
            self._start_time = time.time()

        timestamp_provided = timestamp is not None
        if timestamp is None:
            current_time = time.time() - self._start_time
        else:
            current_time = timestamp

        # === 性能优化 1: 跳帧逻辑 ===
        # 如果检测到处理速度极慢（FPS < 2），自动开启跳帧模式
        # 注意：过线检测需要连贯性，所以不能跳得太狠
        if self.adaptive_frame_skip and self._frame_count % 50 == 0:
            # 每 50 帧评估一次处理能力
            elapsed = time.time() - (self._last_perf_check if hasattr(self, '_last_perf_check') else self._start_time)
            fps = 50 / max(0.001, elapsed)
            self._last_perf_check = time.time()
            if fps < 2.0:
                self.frame_skip = 3 # 开启 1 隔 2 跳帧
            elif fps < 5.0:
                self.frame_skip = 2 # 开启 1 隔 1 跳帧
            else:
                self.frame_skip = 1
            
            if self.frame_skip > 1:
                logger.warning(f"[Detector-{self.source_id}] 检测到处理性能瓶颈 (FPS: {fps:.1f})，自动开启 {self.frame_skip} 倍跳帧模式")

        if self.adaptive_frame_skip and self._frame_count % self.frame_skip != 0:
            # 跳帧时，直接返回，不更新 TrackState 的时间，由清理逻辑自行处理超时
            total_ms = (time.time() - _frame_t0) * 1000
            self.last_frame_metrics = {
                "raw_bikes": 0,
                "raw_bibs": 0,
                "validated_tracks": 0,
                "synthetic_tracks": 0,
                "raw_tracks": 0,
                "participants": 0,
                "track_fragments_merged": 0,
                "identity_ambiguities": 0,
                "roi_auto_disabled": False,
                "inference_ms": 0.0,
                "postprocess_ms": total_ms,
                "total_ms": total_ms,
            }
            return [], [], []

        self._cleanup_stale_tracks(current_time)

        frame_original = frame # 移除 .copy()，改用引用
        height, width = frame.shape[:2]
        crossing_events = []

        # 1. YOLO 推理
        # 1.1 ROI 推理裁剪（性能关键）
        # 只在“设置了 ROI 且启用了 ROI 过滤”时裁剪推理输入，减少无关区域计算，提高帧率；坐标在收集结果时再加回偏移
        infer_frame = frame
        infer_offset_x = 0
        infer_offset_y = 0
        if self.enable_roi_filter and (self._roi_polygon is not None):
            try:
                xs = self._roi_polygon[:, 0]
                ys = self._roi_polygon[:, 1]
                rx1 = int(max(0, float(np.min(xs))))
                ry1 = int(max(0, float(np.min(ys))))
                rx2 = int(min(width, float(np.max(xs))))
                ry2 = int(min(height, float(np.max(ys))))

                edge = int(getattr(self, "roi_edge_margin", 0) or 0)
                rw = max(1, rx2 - rx1)
                rh = max(1, ry2 - ry1)
                mx = int(rw * 0.15)
                my = int(rh * 0.15)

                x1 = max(0, rx1 - edge - mx)
                y1 = max(0, ry1 - edge - my)
                x2 = min(width, rx2 + edge + mx)
                y2 = min(height, ry2 + edge + my)

                if (x2 - x1) >= 64 and (y2 - y1) >= 64:
                    infer_frame = frame[y1:y2, x1:x2]
                    infer_offset_x = x1
                    infer_offset_y = y1
            except Exception:
                infer_frame = frame
                infer_offset_x = 0
                infer_offset_y = 0

        try:
            names = getattr(self._model, "names", None) or {}
            if (not names) and hasattr(self._model, "model"):
                names = getattr(self._model.model, "names", None) or {}
            self._map_model_class_ids(names)
        except Exception:
            pass

        track_kwargs = {
            "persist": True,
            "verbose": False,
            "tracker": os.path.join(os.path.dirname(__file__), "custom_bytetrack.yaml"),
            "conf": self.detection_conf,
            "iou": self.iou_threshold,
            "augment": False,
            "imgsz": self.process_imgsz,  # 使用配置的尺寸
        }
        tracked_class_ids = sorted(self.person_class_ids | self.bib_class_ids)
        if tracked_class_ids:
            track_kwargs["classes"] = tracked_class_ids
        self._raw_model_boxes = []
        try:
            _t0 = time.time()
            results = self._model.track(infer_frame, **track_kwargs)
            _infer_ms = (time.time() - _t0) * 1000
            # 每100帧打印一次推理耗时，帮助定位性能瓶颈
            if self._frame_count % 100 == 1:
                logger.info(f"[Detector-{self.source_id}] YOLO推理耗时: {_infer_ms:.0f}ms (imgsz={self.process_imgsz})")
        except AssertionError as e:
            # 兼容静态 shape 的 TensorRT/Engine 模型：若 imgsz 不匹配，自动回退并重试一次。
            err = str(e)
            if "input size" in err and "max model size" in err:
                fallback_size = 640
                size_match = re.search(r'max model size[^\[]*\[(?:\d+\s*,\s*){2}(\d+)\s*,\s*(\d+)\]', err)
                if size_match:
                    try:
                        fallback_size = int(min(int(size_match.group(1)), int(size_match.group(2))))
                    except Exception:
                        fallback_size = 640

                if fallback_size != self.process_imgsz:
                    logger.warning(
                        f"[Detector-{self.source_id}] 模型输入尺寸不匹配，自动回退 imgsz: "
                        f"{self.process_imgsz} -> {fallback_size}"
                    )
                    self.process_imgsz = fallback_size

                track_kwargs["imgsz"] = self.process_imgsz
                results = self._model.track(infer_frame, **track_kwargs)
                _infer_ms = (time.time() - _t0) * 1000
            else:
                raise

        # 2. 收集结果
        athletes = []
        bibs = []
        
        all_boxes = []
        for result in results:
            if result.boxes is None:
                continue

            # 每帧动态类别兜底映射：防止模型类别索引与预设不一致
            try:
                names = getattr(result, 'names', None) or getattr(self._model, 'names', None) or {}
                if (not names) and hasattr(self._model, 'model'):
                    names = getattr(self._model.model, 'names', None) or {}
                mapped = self._map_model_class_ids(names)
                if mapped and not self._class_map_logged:
                    logger.info(
                        f"[Detector-{self.source_id}] 帧内类别映射: person={sorted(self.person_class_ids)}, "
                        f"bib={sorted(self.bib_class_ids)}, bike_mode={self.bike_model_mode}"
                    )
                    self._class_map_logged = True
            except Exception:
                pass

            boxes_len = len(result.boxes)
            for i in range(boxes_len):
                cls_i = int(result.boxes.cls[i].item()) if hasattr(result.boxes.cls[i], 'item') else int(result.boxes.cls[i])
                conf_i = float(result.boxes.conf[i].item()) if hasattr(result.boxes.conf[i], 'item') else float(result.boxes.conf[i])
                xyxy_i = result.boxes.xyxy[i].tolist() if hasattr(result.boxes.xyxy[i], 'tolist') else result.boxes.xyxy[i]
                if not isinstance(xyxy_i, list):
                    try:
                        xyxy_i = xyxy_i.tolist()
                    except Exception:
                        xyxy_i = list(xyxy_i)
                if infer_offset_x or infer_offset_y:
                    xyxy_i = [
                        float(xyxy_i[0]) + infer_offset_x,
                        float(xyxy_i[1]) + infer_offset_y,
                        float(xyxy_i[2]) + infer_offset_x,
                        float(xyxy_i[3]) + infer_offset_y,
                    ]
                track_i = None
                if result.boxes.id is not None:
                    box_id = result.boxes.id[i]
                    track_i = int(box_id.item()) if hasattr(box_id, 'item') else int(box_id)
                all_boxes.append((cls_i, conf_i, xyxy_i, track_i))

        if self._raw_model_boxes:
            tracked_person_boxes = [
                (list(xyxy), float(conf_i))
                for cls_i, conf_i, xyxy, _ in all_boxes
                if int(cls_i) in self.person_class_ids
            ]
            all_boxes = [box for box in all_boxes if int(box[0]) not in self.bib_class_ids]
            for cls_i, conf_i, xyxy_i in self._raw_model_boxes:
                raw_xyxy = list(xyxy_i)
                if infer_offset_x or infer_offset_y:
                    raw_xyxy = [
                        raw_xyxy[0] + infer_offset_x,
                        raw_xyxy[1] + infer_offset_y,
                        raw_xyxy[2] + infer_offset_x,
                        raw_xyxy[3] + infer_offset_y,
                    ]
                cls_i = int(cls_i)
                conf_i = float(conf_i)
                if cls_i in self.bib_class_ids:
                    all_boxes.append((cls_i, conf_i, raw_xyxy, None))
                    continue

                # ByteTrack can omit a newly visible fast athlete even when the
                # detector produced a strong person box. Rescue only near-track
                # boxes and keep tracked/raw duplicates out of the result.
                if cls_i not in self.person_class_ids:
                    continue
                if conf_i < self.untracked_detection_conf:
                    continue
                if raw_xyxy[3] <= int(height * 0.45):
                    continue
                if any(
                    self._passes_athlete_aspect_filter(
                        existing_bbox,
                        existing_conf,
                        (height, width),
                    )
                    and (
                        bbox_iou(raw_xyxy, existing_bbox) >= 0.45
                        or (
                            abs(conf_i - existing_conf) <= 0.02
                            and bbox_overlap_over_smaller(raw_xyxy, existing_bbox) >= 0.85
                        )
                    )
                    for existing_bbox, existing_conf in tracked_person_boxes
                ):
                    continue
                all_boxes.append((cls_i, conf_i, raw_xyxy, None))
                tracked_person_boxes.append((raw_xyxy, conf_i))
        
        present_class_ids = {int(cls) for cls, _, _, _ in all_boxes}
        runtime_person_class_ids = set(self.person_class_ids)

        # 帧内兜底：若当前帧没有任何“person类”命中，则把非 bib 类临时视为 person，避免整帧漏人
        if not self._class_map_logged and present_class_ids and present_class_ids.isdisjoint(runtime_person_class_ids):
            fallback_person_ids = set(present_class_ids) - set(self.bib_class_ids)
            if fallback_person_ids:
                runtime_person_class_ids = fallback_person_ids
                if not self._fallback_class_logged:
                    logger.warning(
                        f"[Detector-{self.source_id}] 帧内person类为空，启用临时类兜底: "
                        f"person={sorted(runtime_person_class_ids)}, bib={sorted(self.bib_class_ids)}"
                    )
                    self._fallback_class_logged = True

        raw_bike_detections = sum(
            1 for cls, _, _, _ in all_boxes if int(cls) in runtime_person_class_ids
        )
        raw_bib_detections = sum(
            1 for cls, _, _, _ in all_boxes if int(cls) in self.bib_class_ids
        )

        # === 性能优化 2: 目标数量限制 ===
        # 如果画面中目标太多（比如由于背景干扰），只处理置信度最高的前 N 个
        if len(all_boxes) > self.max_athletes_per_frame * 2:
            near_boxes = []
            far_boxes = []
            bib_boxes = []
            other_boxes = []
            near_threshold = int(height * 0.45)
            for cls, conf, xyxy, box_id in all_boxes:
                cls_i = int(cls)
                y2 = int(xyxy[3])
                if cls_i in runtime_person_class_ids:
                    if y2 > near_threshold:
                        near_boxes.append((cls, conf, xyxy, box_id))
                    else:
                        far_boxes.append((cls, conf, xyxy, box_id))
                elif cls_i in self.bib_class_ids:
                    bib_boxes.append((cls, conf, xyxy, box_id))
                else:
                    other_boxes.append((cls, conf, xyxy, box_id))
            far_boxes.sort(key=lambda x: x[1], reverse=True)
            bib_boxes.sort(key=lambda x: x[1], reverse=True)
            all_boxes = near_boxes + far_boxes[:self.max_athletes_per_frame] + bib_boxes[:self.max_athletes_per_frame * 2] + other_boxes

        raw_person_candidates = 0
        roi_rejected_person = 0
        roi_auto_disabled = False

        for cls, conf, xyxy, box_id in all_boxes:
            cls = int(cls)
            conf = float(conf)
            x1, y1, x2, y2 = map(int, xyxy)

            if cls in runtime_person_class_ids:
                raw_person_candidates += 1
            
            # ROI 过滤
            if self.enable_roi_filter and self._roi_polygon is not None:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                is_inside = cv2.pointPolygonTest(self._roi_polygon, (float(cx), float(cy)), True)
                outside_tol = float(getattr(self, "roi_outside_tolerance", 0) or 0)
                if is_inside < -outside_tol or is_inside < self.roi_edge_margin:
                    if cls in runtime_person_class_ids:
                        roi_rejected_person += 1
                    continue
            
            if cls in runtime_person_class_ids:  # Athlete
                bw, bh = x2 - x1, y2 - y1
                aspect_ratio = bh / bw if bw > 0 else 0
                area_ratio = (bw * bh) / max(1, width * height)
                center_y = (y1 + y2) * 0.5
                cx = (x1 + x2) * 0.5

                # 龙门结构硬过滤：靠边的超高竖框通常是门柱/立柱，不应作为运动员
                if (cx < width * 0.12 or cx > width * 0.88) and bh > height * 0.42 and bw < width * 0.16:
                    continue
                
                # 体型过滤
                if not self._passes_athlete_aspect_filter(
                    [x1, y1, x2, y2],
                    conf,
                    (height, width),
                ):
                    continue
                if not self._passes_speed_skating_trackside_filter(
                    [x1, y1, x2, y2],
                    (height, width),
                ):
                    continue
                if not self._passes_profile_athlete_geometry(
                    [x1, y1, x2, y2],
                    (height, width),
                ):
                    continue
                if bh > int(height * 0.95) or bw > int(width * 0.45):
                    continue
                if (bw * bh) > int(width * height * 0.35):
                    continue
                if bh < self.min_bh or bw < self.min_bw:
                    continue

                # bike-only 模型下的强约束：优先防龙门/拱门误框
                if self.enable_gate_guard and self.bike_model_mode:
                    if bw > int(width * 0.40) and bh < int(height * 0.18):
                        continue
                    if area_ratio > 0.18 and center_y < height * 0.45 and aspect_ratio < 0.86:
                        continue
                    if center_y < height * 0.26 and bw > int(width * 0.30):
                        continue
                    if area_ratio > 0.22 and center_y < height * 0.62:
                        continue

                # 龙门/横幅抑制：极大且偏上的宽框基本不是运动员
                if self.enable_gate_guard:
                    if self._is_gate_like_bbox([x1, y1, x2, y2], (height, width)):
                        continue

                    if area_ratio > 0.14 and center_y < height * 0.60 and aspect_ratio < 0.90:
                        continue
                    if bw > int(width * 0.38) and bh < int(height * 0.20):
                        continue

                    # 龙门硬过滤（P0）：顶部锚定的大型竖向结构，优先按背景处理
                    # 目标：抑制“把红色拱门当运动员”的误检，不影响正常人体框
                    if y1 < int(height * 0.20) and bh > int(height * 0.40) and bw > int(width * 0.10) and area_ratio > 0.05:
                        continue
                    if y1 < int(height * 0.30) and bw > int(width * 0.18) and bh > int(height * 0.32) and aspect_ratio < 2.8:
                        continue
                    
                track_id = -1
                parsed_track_id = None
                if box_id is not None:
                    try:
                        parsed_track_id = int(box_id)
                    except Exception:
                        parsed_track_id = None

                if parsed_track_id is not None and parsed_track_id > 0:
                    track_id = parsed_track_id
                else:
                    if conf < self.untracked_detection_conf:
                        continue
                    cx_i = int((x1 + x2) * 0.5)
                    cy_i = int((y1 + y2) * 0.5)
                    best_key = None
                    best_dist = 1e9
                    stale_keys = []
                    for key, (px, py, last_t) in self._noid_track_pool.items():
                        if current_time - last_t > 1.2:
                            stale_keys.append(key)
                            continue
                        dist = np.hypot(cx_i - px, cy_i - py)
                        if dist < best_dist:
                            best_dist = dist
                            best_key = key
                    for stale_key in stale_keys:
                        self._noid_track_pool.pop(stale_key, None)

                    if best_key is not None and best_dist <= max(40.0, 0.06 * max(width, height)):
                        track_id = best_key
                    else:
                        track_id = self._noid_next_track_id
                        self._noid_next_track_id += 1

                    self._noid_track_pool[track_id] = (cx_i, cy_i, current_time)
                
                # === 性能优化 3: 目标上限过滤 ===
                # 策略：靠近终点的人 (bottom_y > 0.45*H) 绝不丢弃，远处的人按 N 限制
                if y2 < int(height * 0.45) and len(athletes) >= self.max_athletes_per_frame:
                    continue
                
                bib_text = ""
                with self._lock:
                    if track_id in self._track_states:
                        st = self._track_states[track_id]
                        if st.best_bib:
                            bib_text = f" {st.best_bib}"
                
                athletes.append({
                    'bbox': [x1, y1, x2, y2],
                    'conf': conf,
                    'track_id': track_id,
                    'bib_text': bib_text,
                    'center_x': (x1 + x2) // 2,
                    'center_y': (y1 + y2) // 2,
                    'bottom_y': y2
                })
            
            elif cls in self.bib_class_ids:  # BIB
                bw_bib = x2 - x1
                bh_bib = y2 - y1
                if bw_bib < 18 or bh_bib < 12:
                    continue
                if bh_bib > bw_bib * 1.5:
                    continue
                bibs.append({
                    'bbox': [x1, y1, x2, y2],
                    'conf': conf,
                    'center_x': (x1 + x2) // 2,
                    'center_y': (y1 + y2) // 2
                })

        recovered_athletes, athlete_detector_ms = self._detect_speed_skating_athletes(
            infer_frame,
            infer_offset_x,
            infer_offset_y,
            (height, width),
            current_time,
            athletes,
        )
        if recovered_athletes:
            athletes.extend(recovered_athletes)
            raw_bike_detections += len(recovered_athletes)
            raw_person_candidates += len(recovered_athletes)
        _infer_ms += athlete_detector_ms

        # 3. 号码牌驱动补全
        if bibs:
            athletes = self._ensure_athletes_from_bibs(athletes, bibs, frame_original, current_time)
            # P0 抓漏兜底（马拉松档）：
            # 1) 当前完全没人框；或
            # 2) 号码牌明显多于人框（允许到中等人数），优先补回并排漏人

        self._apply_crowd_adaptive_policy(athletes, height, width, current_time)

        # 3.5 OCR 兜底：若未检出 bib 框，仍为每个运动员缓存一个躯干候选
        # 注意：即使关闭“实时 OCR”，也要保留这份轻量缓存，供“过线短窗同步 OCR”使用。
        if self.ocr_pipeline_enabled and athletes:
            with self._lock:
                for athlete in athletes:
                    tid = athlete.get('track_id', -1)
                    if tid == -1:
                        continue

                    if tid not in self._track_states:
                        self._track_states[tid] = TrackState(
                            prev_x=athlete['center_x'],
                            prev_y=athlete['bottom_y'],
                            last_seen_time=current_time,
                            start_pos=(athlete['center_x'], athlete['bottom_y']),
                            start_time=current_time
                        )

                    state = self._track_states[tid]
                    state.last_seen_time = current_time

                    if state.has_bib_box:
                        continue

                    # bike-only 模型下，避免给龙门/拱门等异常宽框生成 OCR 兜底缓存
                    bw = max(1, athlete['bbox'][2] - athlete['bbox'][0])
                    bh = max(1, athlete['bbox'][3] - athlete['bbox'][1])
                    area_ratio = (bw * bh) / max(1, width * height)
                    center_y = athlete.get('center_y', (athlete['bbox'][1] + athlete['bbox'][3]) // 2)
                    if self.bike_model_mode:
                        if bw > int(width * 0.34) and bh < int(height * 0.24):
                            continue
                        if area_ratio > 0.14 and center_y < height * 0.56 and bh < int(bw * 1.00):
                            continue
                        if center_y < height * 0.36 and bw > int(width * 0.24):
                            continue

                    fallback_candidates = self._extract_bib_fallbacks_from_athlete(
                        frame_original,
                        athlete['bbox'],
                    )
                    if not fallback_candidates:
                        continue

                    frame_ref = frame_original if athlete['bbox'][3] > int(height * 0.45) else None
                    for source, fallback_crop, fallback_bbox in fallback_candidates:
                        quality = self._calculate_image_quality(fallback_crop)
                        self._cache_ocr_candidate(
                            state,
                            BibEvidenceCandidate(
                                quality=float(quality),
                                crop=fallback_crop,
                                frame=frame_ref,
                                frame_index=int(self._frame_count),
                                capture_time_ms=float(current_time) * 1000.0,
                                athlete_bbox=tuple(int(value) for value in athlete['bbox']),
                                bib_bbox=tuple(int(value) for value in fallback_bbox),
                                source=f"fallback:{source}",
                                owner_validated=False,
                            ),
                            detected_bib=False,
                        )

        # 4. 初始化 TrackState 并进行背景检测
        for athlete in athletes:
            track_id = athlete['track_id']
            if track_id == -1:
                continue
            with self._lock:
                if track_id not in self._track_states:
                    self._track_states[track_id] = TrackState(
                        prev_x=athlete['center_x'],
                        prev_y=athlete['bottom_y'],
                        last_seen_time=current_time,
                        start_pos=(athlete['center_x'], athlete['bottom_y']),
                        start_time=current_time
                    )
                state = self._track_states[track_id]
                state.last_seen_time = current_time
                state.observation_count += 1
                state.synthetic_kind = str(athlete.get('synthetic_kind') or '')
                state.participant_id = str(athlete.get('participant_id') or state.participant_id or '')
                
                # === 性能优化 4: 背景过滤增强 ===
                # 在处理前进行背景判定
                static_filter_active = self.enable_static_background_filter and self._static_filter_cooldown_frames <= 0
                if static_filter_active and not state.is_static:
                    is_bg, reason = self._is_likely_background(
                        state, athlete['bbox'], current_time, (height, width)
                    )
                    if is_bg:
                        state.is_static = True
                        logger.info(f"[Detector-{self.source_id}] ID {track_id} 判定为背景: {reason}")

                if state.is_static:
                    # 热修：允许误判为静态的真实运动员在出现明显位移后自动恢复
                    if not state.vlm_is_background:
                        cx, cy = athlete['center_x'], athlete['bottom_y']
                        sx, sy = state.start_pos if state.start_pos else (cx, cy)
                        move_from_start = np.sqrt((cx - sx) ** 2 + (cy - sy) ** 2)
                        reactivate_dist = max(40.0, height * 0.04)
                        if move_from_start >= reactivate_dist:
                            state.is_static = False
                            state.static_confidence = max(0.0, state.static_confidence - 0.6)
                            state.position_history = [(current_time, cx, cy)]
                        else:
                            continue
                    else:
                        continue

                if state.is_static:
                    continue
 
                # --- 实时同步 VLM OCR 结果 ---
                if self._vlm_enabled and self._vlm_pipeline:
                    vlm_res = self._vlm_pipeline.get_cached_bib(track_id)
                    if vlm_res:
                        vlm_text, vlm_conf = vlm_res
                        
                        # 核心修复：VLM 结果也必须经过 _validate_bib 校验和过滤（处理纯数字模式）
                        vlm_text, val_score = self._validate_bib(vlm_text)
                        vlm_conf = vlm_conf * val_score
                        
                        if vlm_conf > state.best_bib_conf and vlm_conf > 0.4:
                            state.best_bib = vlm_text
                            state.best_bib_conf = vlm_conf
                            logger.info(f"[Detector-{self.source_id}] ID {track_id} 实时应用 VLM 校验结果: {vlm_text} (conf={vlm_conf:.2f})")
                            
                            # 如果已经过线且发射过事件，需要触发补录回调更新 UI 和数据库
                            if state.event_emitted and self._on_bib_update:
                                new_status = (BibStatus.RECOGNIZED if state.best_bib_conf >= self.ocr_conf_threshold
                                              else BibStatus.NEEDS_REVIEW)
                                self._on_bib_update(track_id, state.best_bib, state.best_bib_conf, new_status, self.source_id)
                        
                        # 收到结果，重置 pending 状态
                        state.vlm_ocr_pending = False
                
                # VLM 辅助背景检测
                if self._vlm_enabled and self._vlm_pipeline:
                    if self._vlm_pipeline.is_known_background(track_id):
                        state.is_static = True
                        state.vlm_is_background = True
                    elif not state.vlm_verified:
                        # 触发条件优化：
                        # 1. 面积较大 (降低到 3%)
                        # 2. 或者启发式算法判定为静止 (static_confidence > 0.3)
                        # 3. 且 track 已经存在了一段时间
                        area_ratio = ((athlete['bbox'][2] - athlete['bbox'][0]) *
                                      (athlete['bbox'][3] - athlete['bbox'][1])) / max(1, width * height)
                        track_duration = current_time - state.start_time
                        
                        if (area_ratio > 0.03 or state.static_confidence > 0.3) and track_duration > 0.5:
                            # 提交背景检查，传入运动员 crop
                            ax1, ay1, ax2, ay2 = athlete['bbox']
                            p_pad = 15
                            a_crop = frame_original[max(0, ay1-p_pad):min(height, ay2+p_pad),
                                                    max(0, ax1-p_pad):min(width, ax2+p_pad)].copy()
                            self._vlm_pipeline.submit_background_check(track_id, a_crop)
                            state.vlm_verified = True # 标记为已提交校验，防止重复提交

        # 5. 关联 BIB 与运动员，并按需进行 OCR
        tid_bib_centers: Dict[int, List] = {}
        bib_assignments, unmatched_bib_indices = self._assign_bibs_to_athletes(athletes, bibs)
        self._last_bib_assign_stats = {
            "bibs": len(bibs),
            "assigned": len(bibs) - len(unmatched_bib_indices),
            "rejected": len(unmatched_bib_indices),
        }

        for athlete_idx, bib_indices in bib_assignments.items():
            matched_athlete = athletes[athlete_idx]
            tid = matched_athlete.get('track_id', -1)
            if tid == -1:
                continue

            ax1, ay1, ax2, ay2 = matched_athlete['bbox']
            for bib_idx in bib_indices:
                bib = bibs[bib_idx]
                bib_cx = bib['center_x']
                bib_cy = bib['center_y']
                tid_bib_centers.setdefault(tid, []).append((bib_cx, bib_cy, bib['bbox']))

                rel_x = (bib_cx - ax1) / max(1, ax2 - ax1)
                rel_y = (bib_cy - ay1) / max(1, ay2 - ay1)
                with self._lock:
                    state = self._track_states.get(tid)
                    if not state:
                        continue

                    if not self.ocr_pipeline_enabled:
                        state.last_bib_rel_pos = (rel_x, rel_y)
                        state.has_bib_box = True
                        continue
                    bx1, by1, bx2, by2 = bib['bbox']
                    b_pad = 5
                    b_crop = frame_original[
                        max(0, by1 - b_pad):min(height, by2 + b_pad),
                        max(0, bx1 - b_pad):min(width, bx2 + b_pad),
                    ].copy()
                    if b_crop.size == 0:
                        continue

                    quality = self._calculate_image_quality(b_crop)
                    if self._has_strong_non_bib_signature(b_crop):
                        logger.debug(
                            "[Detector-%s] Reject non-bib crop for track %s at frame %s",
                            self.source_id,
                            tid,
                            self._frame_count,
                        )
                        continue
                    state.last_bib_rel_pos = (rel_x, rel_y)
                    state.has_bib_box = True
                    frame_ref = frame_original if (by2 > int(height * 0.45) or quality >= 0.6) else None
                    self._cache_ocr_candidate(
                        state,
                        BibEvidenceCandidate(
                            quality=float(quality),
                            crop=b_crop,
                            frame=frame_ref,
                            frame_index=int(self._frame_count),
                            capture_time_ms=float(current_time) * 1000.0,
                            athlete_bbox=tuple(int(value) for value in matched_athlete['bbox']),
                            bib_bbox=tuple(int(value) for value in bib['bbox']),
                            source="detected",
                            owner_validated=True,
                        ),
                        detected_bib=True,
                    )
                    state.has_bib_box = True
        if len(athletes) > 1:
            if HAS_WBF and self.enable_wbf_merge:
                # 第一阶段：WBF 物理框融合 (仅当 HAS_WBF 为 True)
                # 归一化坐标
                h_norm, w_norm = height, width
                norm_boxes = []
                norm_scores = []
                norm_labels = []
                
                for i, ath in enumerate(athletes):
                    b = ath['bbox']
                    nb = [
                        max(0.0, min(1.0, b[0] / w_norm)),
                        max(0.0, min(1.0, b[1] / h_norm)),
                        max(0.0, min(1.0, b[2] / w_norm)),
                        max(0.0, min(1.0, b[3] / h_norm))
                    ]
                    norm_boxes.append(nb)
                    norm_scores.append(ath['conf'])
                    norm_labels.append(0)
                
                iou_thr = 0.55
                skip_box_thr = 0.01
                
                try:
                    boxes_out, scores_out, labels_out = weighted_boxes_fusion(
                        [norm_boxes], 
                        [norm_scores], 
                        [norm_labels], 
                        weights=None, 
                        iou_thr=iou_thr, 
                        skip_box_thr=skip_box_thr,
                        conf_type='avg'
                    )
                except Exception as e:
                    logger.error(f"[Detector-{self.source_id}] WBF 运行失败: {e}，回退到原始列表")
                    boxes_out = []
                
                if len(boxes_out) > 0:
                    merged_athletes = []
                    processed_indices = set()
                    
                    for i in range(len(boxes_out)):
                        wb = boxes_out[i]
                        w_box = [
                            int(wb[0] * w_norm),
                            int(wb[1] * h_norm),
                            int(wb[2] * w_norm),
                            int(wb[3] * h_norm)
                        ]
                        
                        candidates = []
                        for idx, ath in enumerate(athletes):
                            if idx in processed_indices: continue
                            iou = bbox_iou(w_box, ath['bbox'])
                            if iou > iou_thr:
                                candidates.append((idx, ath))
                        
                        if not candidates:
                            continue
                            
                        candidates.sort(key=lambda x: (1 if x[1]['track_id'] != -1 else 0, x[1]['conf']), reverse=True)
                        main_idx, main_ath = candidates[0]
                        processed_indices.add(main_idx)
                        
                        final_ath = main_ath.copy()
                        final_ath['bbox'] = w_box
                        final_ath['conf'] = float(scores_out[i])
                        final_ath['center_x'] = (w_box[0] + w_box[2]) // 2
                        final_ath['center_y'] = (w_box[1] + w_box[3]) // 2
                        final_ath['bottom_y'] = w_box[3]
                        
                        for c_idx, c_ath in candidates[1:]:
                            processed_indices.add(c_idx)
                            c_tid = c_ath['track_id']
                            main_tid = final_ath['track_id']
                            
                            if c_tid != -1 and c_tid != main_tid:
                                if c_tid in tid_bib_centers:
                                    bibs_to_move = tid_bib_centers.pop(c_tid)
                                    if main_tid != -1:
                                        tid_bib_centers.setdefault(main_tid, []).extend(bibs_to_move)
                        
                        merged_athletes.append(final_ath)
                    athletes = merged_athletes
                else:
                    athletes = self._iou_dedup_athletes(athletes, tid_bib_centers)
            else:
                athletes = self._iou_dedup_athletes(athletes, tid_bib_centers)

            # 第二阶段：基于语义（号码、VLM、历史）的深度合并
            if self.enable_semantic_merge and len(athletes) > 1:
                deep_merged = []
                skip_deep = set()
                for i in range(len(athletes)):
                    if i in skip_deep: continue
                    
                    curr_ath = athletes[i]
                    for j in range(i + 1, len(athletes)):
                        if j in skip_deep: continue
                        
                        other_ath = athletes[j]
                        tid_i = curr_ath['track_id']
                        tid_j = other_ath['track_id']
                        
                        # 如果 ID 相同或者已经记录过合并，跳过
                        if tid_i == tid_j: continue
                        
                        if self._should_merge_athletes(curr_ath, other_ath, tid_bib_centers):
                            skip_deep.add(j)
                            
                            # 仅在第一次合并时记录日志，避免日志爆炸
                            merge_key = tuple(sorted([tid_i, tid_j]))
                            if merge_key not in self._merged_pairs:
                                logger.info(f"[Detector-{self.source_id}] 深度语义合并: ID {tid_j} -> ID {tid_i}")
                                self._merged_pairs.add(merge_key)
                            
                            if tid_j in tid_bib_centers:
                                bibs_to_move = tid_bib_centers.pop(tid_j)
                                if tid_i != -1:
                                    tid_bib_centers.setdefault(tid_i, []).extend(bibs_to_move)
                            
                            # 不再使用并集大框，避免把两人误并成一框
                            if other_ath.get('conf', 0.0) > curr_ath.get('conf', 0.0):
                                curr_ath['bbox'] = list(other_ath['bbox'])
                                curr_ath['conf'] = other_ath['conf']
                                curr_ath['center_x'] = other_ath.get('center_x', (curr_ath['bbox'][0] + curr_ath['bbox'][2]) // 2)
                                curr_ath['center_y'] = other_ath.get('center_y', (curr_ath['bbox'][1] + curr_ath['bbox'][3]) // 2)
                                curr_ath['bottom_y'] = other_ath.get('bottom_y', curr_ath['bbox'][3])
                    
                    deep_merged.append(curr_ath)
                athletes = deep_merged

        # 5.1.5 漏人抢救：对“单个宽框吞并多人”的场景做过线前拆分
        athletes = self._rescue_split_wide_boxes(
            athletes=athletes,
            bibs=bibs,
            tid_bib_centers=tid_bib_centers,
            frame_shape=(height, width),
            current_time=current_time,
        )

        athletes = [a for a in athletes if not a.get('is_static')]

        # Resolve identity only after same-frame deduplication and wide-box splitting.
        identity_resolutions = {}
        resolved_athletes = []
        assigned_participant_ids: Set[str] = set()
        for athlete in athletes:
            raw_track_id = athlete.get("track_id")
            bib_boxes = tuple(
                bib_entry[2]
                for bib_entry in tid_bib_centers.get(raw_track_id, ())
                if len(bib_entry) >= 3
            )
            resolved, resolution = self._resolve_participant_with_resolution(
                athlete,
                current_time,
                bib_bboxes=bib_boxes,
                excluded_participant_ids=assigned_participant_ids,
            )
            resolved_athletes.append(resolved)
            identity_resolutions[id(resolved)] = resolution
            assigned_participant_ids.add(resolution.participant.participant_id)
            if raw_track_id is not None and raw_track_id in self._track_states:
                state = self._track_states[raw_track_id]
                if resolution.merged_raw_track:
                    self._inherit_fragment_motion_state(
                        raw_track_id,
                        state,
                        resolution.participant.participant_id,
                        current_time,
                    )
                else:
                    state.participant_id = resolution.participant.participant_id
        athletes = resolved_athletes

        # 5.2 提交 OCR 任务 (异步双链路：基于高质量缓存)
        if self.realtime_ocr_enabled:
            # 除了“已检出 bib 框”的目标，也纳入“近终点且有兜底裁剪缓存”的目标，
            # 以减少模型偶发漏检 bib 框时的 UNKNOWN。
            ocr_candidate_tids = set(tid_bib_centers.keys())
            if self._ocr is not None:
                with self._lock:
                    for athlete in athletes:
                        tid = athlete.get('track_id', -1)
                        if tid == -1:
                            continue
                        state = self._track_states.get(tid)
                        if not state or not self._get_ocr_candidates(state):
                            continue
                        # 仅近终点才启用兜底异步 OCR，避免远端无效消耗
                        if athlete['bbox'][3] > int(height * 0.45):
                            ocr_candidate_tids.add(tid)

            for tid in ocr_candidate_tids:
                # 找到对应的运动员
                athlete = next((a for a in athletes if a['track_id'] == tid), None)
                if not athlete: continue
                
                with self._lock:
                    state = self._track_states.get(tid)
                    if state and not state.is_static:
                        # 1. 区域与状态判断
                        bottom_y = athlete['bbox'][3]
                        is_near = bottom_y > (height * 0.45)
                        is_very_near = bottom_y > (height * 0.7)
                        
                        # 2. 动态 OCR 触发策略
                        # 如果已经有极其确定的名单号码，停止 OCR
                        if state.best_bib in self._athlete_list and state.best_bib_conf > 0.98:
                            continue
                            
                        # 3. 决定是否从缓存中提交任务
                        qsize = self._ocr_queue.qsize()
                        # 如果过线了，或者在近端且长时间没做 OCR
                        should_batch_ocr = False
                        if state.crossed and not state.event_emitted:
                            # 刚过线，还没发事件，全力识别
                            should_batch_ocr = (current_time - state.last_ocr_time > 0.25)
                        elif is_very_near:
                            # 极近端，每 0.5s 识别一次
                            should_batch_ocr = (current_time - state.last_ocr_time > 0.5)
                        elif is_near:
                            # 近端，每 1.5s 识别一次
                            should_batch_ocr = (current_time - state.last_ocr_time > 1.5)
                        else:
                            # 远端，每 3.0s 识别一次
                            should_batch_ocr = (current_time - state.last_ocr_time > 3.0)
                            
                        candidates = (
                            self._get_crossing_ocr_candidates(state, state.pending_event_data)
                            if state.crossed and state.pending_event_data
                            else self._get_ocr_candidates(state)
                        )
                        if should_batch_ocr and candidates:
                            # 从缓存中选出 Top-3 高质量截图提交 (如果队列不拥堵，选 3 张；否则 1 张)
                            num_to_submit = 2 if qsize < 6 else 1
                            if state.crossed:
                                num_to_submit = 1 if qsize >= 2 else 2
                            
                            best_crops = candidates[:num_to_submit]
                            logger.debug(f"[Detector-{self.source_id}] ID {tid} 提交 {len(best_crops)} 个 OCR 任务 (缓存大小: {len(candidates)})")
                            for quality, b_crop, f_ref, b_bbox in best_crops:
                                try:
                                    # 提交 OCR (注意：不再 copy 整个 frame，增加 quality 分数)
                                    self._ocr_attempts += 1
                                    self._ocr_queue.put_nowait(
                                        (
                                            b_crop,
                                            tid,
                                            athlete['bbox'],
                                            b_bbox,
                                            f_ref,
                                            quality,
                                            str(athlete.get('participant_id') or state.participant_id or ''),
                                        )
                                    )
                                except queue.Full:
                                    break
                                
                            state.last_ocr_time = current_time
                            
                        # 4. 队列维护 (强制减负)
                        if qsize > 30:
                            logger.warning(f"[Detector-{self.source_id}] OCR 队列严重积压 ({qsize})，丢弃 50% 旧任务")
                            try:
                                for _ in range(int(qsize * 0.5)):
                                    self._ocr_queue.get_nowait()
                                    self._ocr_queue.task_done()
                            except: pass

                        # 5. VLM 逻辑
                        if self._vlm_enabled and self._vlm_pipeline:
                            if (is_very_near or state.crossed) and state.best_bib_conf < 0.85 and qsize < 15:
                                vlm_interval = 2.0
                                if current_time - state.last_vlm_time > vlm_interval and not state.vlm_ocr_pending:
                                    if candidates:
                                        _, best_vlm_crop, _, _ = candidates[0]
                                        self._vlm_pipeline.submit_ocr(
                                            tid, best_vlm_crop, self._vlm_athlete_list,
                                            only_numeric=self.only_numeric
                                        )
                                        state.last_vlm_time = current_time
                                        state.vlm_ocr_pending = True

        # 6. 过线检测逻辑
        pt1, pt2 = self._line_pt1, self._line_pt2
        if self.timing_enabled and pt1 is not None and pt2 is not None:
            for athlete in athletes:
                tid = athlete['track_id']
                if tid == -1 or tid < 0: continue
                
                with self._lock:
                    state = self._track_states.get(tid)
                    if not state or state.is_static: continue
                    
                    curr_x, curr_y = athlete['center_x'], athlete['bottom_y']
                    prev_x, prev_y = state.prev_x, state.prev_y

                    if not self._is_event_eligible_athlete(athlete, state.observation_count):
                        state.prev_x, state.prev_y = curr_x, curr_y
                        self._update_crossing_vote(state, current_time, has_cross_signal=False)
                        continue

                    if self.enable_gate_guard and self._is_gate_like_bbox(athlete['bbox'], (height, width)):
                        self._update_crossing_vote(state, current_time, has_cross_signal=False)
                        continue
                    
                    # 更新位置
                    state.prev_x, state.prev_y = curr_x, curr_y

                    # 针对 bike 类别模型：要求最小运动量后才允许过线，抑制龙门误触发
                    # 但在低刷新/近终点时放宽，避免“有框不刷新”
                    if len(state.position_history) >= 2:
                        try:
                            first = state.position_history[0]
                            last = state.position_history[-1]
                            move_dist = np.sqrt((last[1] - first[1]) ** 2 + (last[2] - first[2]) ** 2)
                            min_move = 6.0 if self.bike_model_mode else 4.0
                            if move_dist < min_move and not state.best_bib:
                                curr_line_dist_fast = point_to_line_dist(curr_x, curr_y, pt1[0], pt1[1], pt2[0], pt2[1])
                                prev_line_dist_fast = point_to_line_dist(prev_x, prev_y, pt1[0], pt1[1], pt2[0], pt2[1])
                                near_line_fast = (
                                    curr_line_dist_fast <= max(18.0, height * 0.018) and
                                    prev_line_dist_fast <= max(24.0, height * 0.022)
                                )
                                tiny_motion = (abs(curr_y - prev_y) >= 1 or abs(curr_x - prev_x) >= 2)
                                if not (near_line_fast and tiny_motion):
                                    continue
                        except Exception:
                            pass

                    # 计算相对于终点线的位置
                    d_curr = point_side_of_line(curr_x, curr_y, pt1[0], pt1[1],
                                               pt2[0], pt2[1])
                    d_prev = point_side_of_line(prev_x, prev_y, pt1[0], pt1[1],
                                               pt2[0], pt2[1])

                    line_dist_curr = point_to_line_dist(curr_x, curr_y, pt1[0], pt1[1], pt2[0], pt2[1])
                    line_dist_prev = point_to_line_dist(prev_x, prev_y, pt1[0], pt1[1], pt2[0], pt2[1])
                    near_line_dist = max(24.0, height * 0.020)
                    if state.start_line_dist is None:
                        state.start_line_dist = float(line_dist_curr)

                    # Preserve approach motion before applying the finish-segment
                    # event gate. Fast athletes may enter the endpoint margin only
                    # on the crossing frame, but still need prior motion evidence.
                    recent_speed = 0.0
                    recent_normal_motion = 0.0
                    recent_tangent_motion = 0.0
                    line_normal_dominant = False
                    try:
                        hist = state.position_history
                        hist.append((current_time, int(curr_x), int(curr_y)))
                        window_sec = 0.6
                        hist = [(t, x, y) for (t, x, y) in hist if (current_time - t) <= window_sec]
                        state.position_history = hist
                        if len(hist) >= 3:
                            total_dist = 0.0
                            for i in range(1, len(hist)):
                                total_dist += float(np.hypot(hist[i][1] - hist[i-1][1], hist[i][2] - hist[i-1][2]))
                            time_span = float(hist[-1][0] - hist[0][0])
                            if time_span > 1e-4:
                                recent_speed = total_dist / time_span
                            line_dx = float(pt2[0] - pt1[0])
                            line_dy = float(pt2[1] - pt1[1])
                            line_length = float(np.hypot(line_dx, line_dy))
                            if line_length > 1.0:
                                tangent_x = line_dx / line_length
                                tangent_y = line_dy / line_length
                                normal_x = -tangent_y
                                normal_y = tangent_x
                                tangent_positions = [x * tangent_x + y * tangent_y for _, x, y in hist]
                                normal_positions = [x * normal_x + y * normal_y for _, x, y in hist]
                                recent_tangent_motion = float(max(tangent_positions) - min(tangent_positions))
                                recent_normal_motion = float(max(normal_positions) - min(normal_positions))
                                min_normal_motion = max(14.0, height * 0.010)
                                line_normal_dominant = recent_normal_motion >= max(
                                    min_normal_motion,
                                    recent_tangent_motion * 0.70,
                                )
                    except Exception:
                        pass

                    # 终点线“线段范围”过滤：只允许投影落在 pt1-pt2 线段范围内的目标参与过线判定
                    seg_ok_curr = True
                    seg_ok_prev = True
                    if getattr(self, "enable_finish_segment_filter", False):
                        try:
                            seg_len = float(np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]))
                            if seg_len > 1.0:
                                margin_px = float(getattr(self, "finish_segment_margin_px", 0.0) or 0.0)
                                margin_t = margin_px / seg_len
                                shrink_px = float(getattr(self, "finish_segment_end_shrink_px", 0.0) or 0.0)
                                shrink_t = max(0.0, min(0.49, shrink_px / seg_len))
                                t_min = 0.0 + shrink_t - margin_t
                                t_max = 1.0 - shrink_t + margin_t
                                t_curr = point_line_segment_t(curr_x, curr_y, pt1[0], pt1[1], pt2[0], pt2[1])
                                t_prev = point_line_segment_t(prev_x, prev_y, pt1[0], pt1[1], pt2[0], pt2[1])
                                seg_ok_curr = (t_min <= t_curr <= t_max)
                                seg_ok_prev = (t_min <= t_prev <= t_max)
                                if not (seg_ok_curr or seg_ok_prev):
                                    self._update_crossing_vote(state, current_time, has_cross_signal=False)
                                    continue
                        except Exception:
                            pass

                    # 过线保护（关键）：没号码(UNKNOWN)时，必须先有“明显位移”，否则观众/工作人员的抖动会被误当成过线
                    # 说明：运动员从远处跑来，位移会很明显；观众基本不动（只有微抖），因此不会触发
                    move_from_start = None
                    track_age = float(current_time - state.start_time)

                    if not state.best_bib:
                        try:
                            sx, sy = state.start_pos if state.start_pos else (curr_x, curr_y)
                            move_from_start = float(np.hypot(curr_x - sx, curr_y - sy))
                            min_cross_move = max(18.0, height * 0.012)  # 约 15~20 像素量级（保守）
                            if track_age >= 0.35 and move_from_start < min_cross_move:
                                self._update_crossing_vote(state, current_time, has_cross_signal=False)
                                continue
                        except Exception:
                            pass

                    # 抓人优先：近线兜底触发，解决“有框但不刷新”
                    force_near_cross = False
                    if not state.crossed:
                        near_hits = int(getattr(state, '_near_line_hits', 0))
                        allow_force_near = True
                        # UNKNOWN 的“近线兜底”必须更严格：需要“最近一段时间有明显运动证据”，避免路人/观众抖动造成虚高
                        if not state.best_bib and not getattr(state, "has_bib_box", False):
                            try:
                                min_speed = 22.0
                                allow_force_near = line_normal_dominant and (recent_speed >= min_speed)

                                # 进一步拦截：起始就贴着终点线的目标（典型路人/观众）禁止兜底触发
                                start_dist = float(state.start_line_dist or 0.0)
                                min_start_dist = max(38.0, height * 0.028)
                                if start_dist < min_start_dist and track_age >= 0.20 and not line_normal_dominant:
                                    allow_force_near = False
                            except Exception:
                                allow_force_near = False

                        if (not seg_ok_curr) or (not allow_force_near):
                            near_hits = 0
                        elif line_dist_curr <= near_line_dist * 1.35:
                            if abs(curr_y - prev_y) >= 1 or abs(curr_x - prev_x) >= 1:
                                near_hits += 1
                            else:
                                near_hits = max(0, near_hits - 1)
                            if near_hits >= 2 and (current_time - state.start_time) >= 0.18:
                                force_near_cross = True
                        else:
                            near_hits = 0
                        setattr(state, '_near_line_hits', near_hits)

                    # 近线补偿：避免因抖动/跳帧导致 d_curr*d_prev 未变号而漏判
                    if (not state.crossed and
                        line_dist_curr <= near_line_dist * 1.6 and
                        line_dist_prev <= near_line_dist * 1.8 and
                        (abs(curr_y - prev_y) >= 1 or abs(curr_x - prev_x) >= 2)):
                        move_towards_line = line_dist_curr <= line_dist_prev * 1.90
                        if move_towards_line:
                            d_curr = 1e-6
                            d_prev = -1e-6

                    # 判定过线 (符号改变)
                    cross_signal = ((d_curr * d_prev < 0) or force_near_cross) and (not state.crossed)
                    if cross_signal:
                        # 检查过线方向 (如果设置了方向)
                        if force_near_cross:
                            is_correct_direction = True
                        else:
                            is_correct_direction = True
                            if self.crossing_direction == 'pos_to_neg':
                                is_correct_direction = (d_prev > 0 and d_curr < 0)
                            elif self.crossing_direction == 'neg_to_pos':
                                is_correct_direction = (d_prev < 0 and d_curr > 0)

                        if is_correct_direction:
                            # UNKNOWN 防路人：没有号码牌框证据时，要求运动方向以终点线法向为主
                            if not state.best_bib and not getattr(state, "has_bib_box", False):
                                try:
                                    min_speed = 22.0

                                    # 起始就贴着终点线的目标（典型路人/观众）更严格：没有明显运动证据则直接拦截
                                    start_dist = float(state.start_line_dist or 0.0)
                                    min_start_dist = max(38.0, height * 0.028)
                                    if start_dist < min_start_dist and track_age >= 0.20 and not line_normal_dominant:
                                        self._update_crossing_vote(state, current_time, has_cross_signal=False)
                                        continue

                                    if track_age >= 0.18 and (not line_normal_dominant or recent_speed < min_speed):
                                        self._update_crossing_vote(state, current_time, has_cross_signal=False)
                                        continue
                                except Exception:
                                    pass

                            strong_signal = (
                                line_dist_curr <= near_line_dist * self.cross_strong_dist_ratio and
                                line_dist_prev <= near_line_dist * self.cross_strong_dist_ratio
                            )
                            confirmed = self._update_crossing_vote(
                                state, current_time,
                                has_cross_signal=True,
                                is_strong_signal=strong_signal
                            )

                            if confirmed:
                                state.crossed = True
                                state.crossed_time = current_time
                                frame_athletes = []
                                for frame_athlete in athletes:
                                    frame_bbox = frame_athlete.get('bbox')
                                    if not frame_bbox or len(frame_bbox) != 4:
                                        continue
                                    frame_athletes.append({
                                        'bbox': tuple(int(value) for value in frame_bbox),
                                        'track_id': int(frame_athlete.get('track_id', -1)),
                                        'conf': float(frame_athlete.get('conf', 0.0)),
                                    })
                                state.pending_event_data = {
                                    'cross_time': current_time if timestamp_provided else time.time(),
                                    'cross_realtime': time.time(),
                                    'frame_index': int(self._frame_count),
                                    'bbox': athlete['bbox'],
                                    'conf': athlete['conf'],
                                    'position': (curr_x, curr_y),
                                    'frame': frame_original,
                                    'participant_id': athlete.get('participant_id', ''),
                                    'raw_track_ids': tuple(athlete.get('raw_track_ids') or ()),
                                    'frame_athletes': tuple(frame_athletes),
                                }
                                logger.info(f"[Detector-{self.source_id}] ID {tid} 触发过线判定 (方向: {self.crossing_direction or 'any'})")
                            else:
                                logger.debug(
                                    f"[Detector-{self.source_id}] ID {tid} 过线信号待确认 "
                                    f"({state.cross_signal_count}/{max(1, int(self.cross_confirm_frames))})"
                                )
                        else:
                            # 方向不符时清理投票，避免错误累计
                            self._update_crossing_vote(state, current_time, has_cross_signal=False)
                            logger.debug(f"[Detector-{self.source_id}] ID {tid} 跨越终点线但方向不符 (忽略)")
                    else:
                        self._update_crossing_vote(state, current_time, has_cross_signal=False)
        else:
            # 如果未启用计时或未配置线，仅更新轨迹位置
            for athlete in athletes:
                tid = athlete['track_id']
                if tid == -1 or tid < 0: continue
                with self._lock:
                    state = self._track_states.get(tid)
                    if state:
                        state.prev_x, state.prev_y = athlete['center_x'], athlete['bottom_y']

        # 7. 事件生成
        with self._lock:
            for tid, state in self._track_states.items():
                if state.crossed and not state.event_emitted:
                    # 7.0 同步 VLM 结果
                    if self._vlm_enabled and self._vlm_pipeline:
                        vlm_res = self._vlm_pipeline.get_cached_bib(tid)
                        if vlm_res:
                            vlm_text, vlm_conf = vlm_res
                            
                            # 核心修复：VLM 结果也必须经过 _validate_bib 校验和过滤（处理纯数字模式）
                            vlm_text, val_score = self._validate_bib(vlm_text)
                            vlm_conf = vlm_conf * val_score
                            
                            if vlm_conf > state.best_bib_conf and vlm_conf > 0.4:
                                state.best_bib = vlm_text
                                state.best_bib_conf = vlm_conf
                                logger.info(f"[Detector-{self.source_id}] ID {tid} 应用 VLM 结果: {vlm_text} (conf={vlm_conf:.2f})")
                            
                            # 收到结果，重置 pending 状态
                            state.vlm_ocr_pending = False

                    # 等待 OCR 窗口
                    # 如果启用了 VLM 且目前没号码，或者正在等待 VLM 结果，则等待更久
                    time_passed = current_time - state.crossed_time
                    candidates = self._get_ocr_candidates(state)
                    if self.realtime_ocr_enabled and (not state.best_bib) and candidates and time_passed >= 0.20:
                        self._try_sync_crossing_ocr(
                            tid,
                            state,
                            current_time,
                            state.pending_event_data,
                        )
                    
                    if not self.ocr_pipeline_enabled:
                        wait_window = 0.0
                    elif self.event_settle_seconds is not None:
                        wait_window = self.event_settle_seconds
                    elif state.best_bib:
                        wait_window = 0.18
                    elif candidates:
                        wait_window = 1.05
                    else:
                        wait_window = 0.55
                    if self._vlm_enabled and self._vlm_pipeline:
                        is_vlm_thinking = self._vlm_pipeline.is_ocr_pending(tid) or state.vlm_ocr_pending
                        has_roster = hasattr(self, '_athlete_list') and bool(self._athlete_list)
                        high_conf_roster_hit = has_roster and state.best_bib in self._athlete_list and state.best_bib_conf > 1.15

                        # 仅在短窗内给 VLM 一次机会，避免过线瞬间卡顿
                        if not high_conf_roster_hit:
                            if is_vlm_thinking and time_passed < 1.2:
                                continue
                            if time_passed < wait_window:
                                continue
                    else:
                        if time_passed < wait_window:
                            continue
                        
                    # 检查状态：如果该 ID 已经合并到其他 ID，则不发事件
                    with self._lock:
                        if state.merged_into is not None:
                            logger.info(f"[Detector-{self.source_id}] ID {tid} 已合并到 {state.merged_into}，取消重复事件")
                            state.event_emitted = True
                            continue

                    # 检查全局去重
                    data = state.pending_event_data
                    curr_x = data['position'][0]
                    curr_bbox = data['bbox']
                    curr_bib = state.best_bib
                    curr_participant_id = str(
                        data.get('participant_id') or state.participant_id or ''
                    ).strip()
                    detected_candidates = self._get_crossing_detected_bib_candidates(state, data)
                    crossing_ocr_candidates = self._get_crossing_ocr_candidates(
                        state,
                        data,
                        detected_candidates,
                    )
                    bib_evidence_kind_for_event = "detected" if detected_candidates else "fallback"

                    if self.enable_gate_guard and self._is_gate_like_bbox(curr_bbox, (height, width)):
                        state.crossed = False
                        state.pending_event_data = None
                        state.cross_signal_count = 0
                        continue
                    
                    is_dup = False
                    dup_reason = ""
                    
                    # 0. 全局号码去重：必须“时间近 + 空间近”才成立，避免漏人
                    if curr_bib:
                        with self._lock:
                            # P0防漏：全局号码去重窗口收紧，避免不同选手被误杀
                            dedup_time_window = min(1.0, max(0.55, self.dedup_window * 0.40))

                            entry = self._finished_bibs.get(curr_bib)
                            if entry:
                                if isinstance(entry, dict):
                                    last_time = float(entry.get("time", 0.0))
                                    last_bbox = entry.get("bbox")
                                else:
                                    last_time = float(entry)
                                    last_bbox = None

                                if current_time - last_time < dedup_time_window:
                                    if (last_bbox is None) or (bbox_iou(curr_bbox, last_bbox) > 0.55):
                                        is_dup = True
                                        dup_reason = f"same_bib_global:{curr_bib}"
                                        logger.info(f"[Detector-{self.source_id}] ID {tid} 号码 {curr_bib} 在短窗且空间高度接近，判定去重")

                            if not is_dup:
                                curr_prefix, curr_digits = self._split_bib_prefix_digits(curr_bib)
                                if curr_digits:
                                    key = f"DIGITS_{curr_digits}"
                                    entry = self._finished_bibs.get(key)
                                    if isinstance(entry, dict):
                                        last_time = float(entry.get("time", 0.0))
                                        prefixes = entry.get("prefixes", set())
                                        last_bbox = entry.get("bbox")
                                    else:
                                        last_time = float(entry or 0.0)
                                        prefixes = set()
                                        last_bbox = None
                                    if entry:
                                        same_prefix_group = (curr_prefix == "" or curr_prefix in prefixes or "" in prefixes)
                                        if same_prefix_group and current_time - last_time < dedup_time_window:
                                            if (last_bbox is None) or (bbox_iou(curr_bbox, last_bbox) > 0.55):
                                                is_dup = True
                                                dup_reason = f"same_digits_global:{curr_digits}"
                                                logger.info(f"[Detector-{self.source_id}] ID {tid} 数字段 {curr_digits} 在短窗且空间高度接近，判定去重")
                    
                    if not is_dup:
                        for old in self._recent_events:
                            # 0. 超时过滤
                            time_diff = current_time - old['time']
                            if time_diff > self.dedup_window * 1.5:
                                continue
                            giou = bbox_iou(curr_bbox, old['bbox'])
                            old_tid = int(old.get('tid', -1))
                            is_virtual_tid = int(tid) >= 900000
                            is_old_virtual = old_tid >= 900000

                            # 0.5 虚拟补框ID防重：仅在极短时间极高重叠时拦截，避免误吞不同人
                            if (not curr_bib) and (is_virtual_tid or is_old_virtual):
                                cbx1, cby1, cbx2, cby2 = curr_bbox
                                obx1, oby1, obx2, oby2 = old['bbox']
                                cdx = abs(((cbx1 + cbx2) * 0.5) - ((obx1 + obx2) * 0.5))
                                cdy = abs(((cby1 + cby2) * 0.5) - ((oby1 + oby2) * 0.5))
                                if time_diff < 0.60 and (giou > 0.72 or (giou > 0.42 and cdx <= 26 and cdy <= 58)):
                                    is_dup = True
                                    dup_reason = f"virtual_overlap:{giou:.2f}"
                                    logger.info(f"[Detector-{self.source_id}] ID {tid} 虚拟ID短时高重叠，判定去重")
                                    break
                            
                            # 1. 号码去重：仅在时空接近时生效，避免把不同人误并
                            if curr_bib and old.get('bib'):
                                # 完全一致
                                if curr_bib == old.get('bib'):
                                    if time_diff < min(0.95, max(0.55, self.dedup_window * 0.38)) and giou > 0.55:
                                        is_dup = True
                                        dup_reason = f"same_bib_recent:{curr_bib}"
                                        logger.info(f"[Detector-{self.source_id}] ID {tid} 因号码 {curr_bib} 且时空高度接近被去重")
                                        break
                                
                                curr_prefix, curr_digits = self._split_bib_prefix_digits(curr_bib)
                                old_prefix, old_digits = self._split_bib_prefix_digits(old.get('bib'))
                                if curr_digits and old_digits and len(curr_digits) >= 3 and curr_digits == old_digits:
                                    if curr_prefix == "" or old_prefix == "" or curr_prefix == old_prefix:
                                        if time_diff < min(0.95, max(0.55, self.dedup_window * 0.38)) and giou > 0.55:
                                            is_dup = True
                                            dup_reason = f"same_digits_recent:{curr_digits}"
                                            logger.info(f"[Detector-{self.source_id}] ID {tid} 因数字段一致且时空高度接近被去重")
                                            break
                            
                            # 2. 空间去重 (IoU)：抓人优先，进一步收紧触发条件
                            if time_diff < 0.22 and giou > 0.96:
                                is_dup = True
                                dup_reason = f"space_overlap:{giou:.2f}"
                                logger.info(f"[Detector-{self.source_id}] ID {tid} 因空间位置重叠 (IoU={giou:.2f}) 被去重")
                                break
                            
                            # 3. ID 继承去重
                            if state.merged_into is not None:
                                is_dup = True
                                dup_reason = f"merged_into:{state.merged_into}"
                                logger.info(f"[Detector-{self.source_id}] ID {tid} 已被记录或合并，去重")
                                break
                            if old.get('tid') == tid and time_diff < 1.0 and giou > 0.80:
                                is_dup = True
                                dup_reason = "same_tid_recent"
                                logger.info(f"[Detector-{self.source_id}] ID {tid} 短时同轨迹强重叠，判定去重")
                                break

                            # 4. 极近时间+空间去重 (如果都没有号码，且时间极近)
                            if not curr_bib and not old.get('bib'):
                                if self._is_duplicate_unknown_event_bbox(
                                    curr_bbox,
                                    old['bbox'],
                                    time_diff,
                                    current_bib_evidence_kind=bib_evidence_kind_for_event,
                                    previous_bib_evidence_kind=old.get('bib_evidence_kind', ''),
                                    current_participant_id=curr_participant_id,
                                    previous_participant_id=old.get('participant_id', ''),
                                ):
                                    is_dup = True
                                    dup_reason = "unknown_nested_bbox"
                                    logger.info(f"[Detector-{self.source_id}] ID {tid} UNKNOWN嵌套框重复过线，抑制输出")
                                    break
                                # P0防漏：UNKNOWN 场景不再仅凭邻近时空直接吞事件，避免“有框不刷新”
                                # 仅在“同 tid 且极高重叠”时保守去重
                                if old.get('tid') == tid and time_diff < 0.45 and giou > 0.90:
                                    is_dup = True
                                    dup_reason = "unknown_same_tid"
                                    logger.info(f"[Detector-{self.source_id}] ID {tid} UNKNOWN同轨迹高重叠，判定去重")
                                    break
                    
                    # 4.5 UNKNOWN 额外抑制：摆动/跟踪ID切换导致的重复过线（非常保守）
                    # 只在“时间极近 + 框大小几乎一致 + 框重叠较高 + 中心几乎不动”时才抑制
                    if (not is_dup) and (not curr_bib) and self._recent_events:
                        try:
                            old = self._recent_events[-1]
                            if (not old.get('bib')) and ('time' in old) and ('bbox' in old):
                                time_diff2 = float(current_time) - float(old.get('time') or 0.0)
                                old_bbox = old.get('bbox')
                                if old_bbox and len(old_bbox) == 4 and time_diff2 < 0.40:
                                    giou2 = bbox_iou(curr_bbox, old_bbox)
                                    cbx1, cby1, cbx2, cby2 = [float(v) for v in curr_bbox]
                                    obx1, oby1, obx2, oby2 = [float(v) for v in old_bbox]
                                    cw = max(1.0, cbx2 - cbx1)
                                    ch = max(1.0, cby2 - cby1)
                                    ow = max(1.0, obx2 - obx1)
                                    oh = max(1.0, oby2 - oby1)

                                    rw = cw / ow
                                    rh = ch / oh
                                    size_close = (0.88 <= rw <= 1.12) and (0.88 <= rh <= 1.12)

                                    ccx = (cbx1 + cbx2) * 0.5
                                    ccy = (cby1 + cby2) * 0.5
                                    ocx = (obx1 + obx2) * 0.5
                                    ocy = (oby1 + oby2) * 0.5
                                    dx2 = abs(ccx - ocx)
                                    dy2 = abs(ccy - ocy)
                                    dx_ok2 = dx2 <= max(12.0, cw * 0.06)
                                    dy_ok2 = dy2 <= max(28.0, ch * 0.10)

                                    old_participant_id = str(old.get('participant_id') or '').strip()
                                    participants_are_compatible = not (
                                        curr_participant_id
                                        and old_participant_id
                                        and curr_participant_id != old_participant_id
                                    )
                                    if (
                                        participants_are_compatible
                                        and size_close
                                        and dx_ok2
                                        and dy_ok2
                                        and giou2 >= 0.70
                                    ):
                                        is_dup = True
                                        dup_reason = "unknown_reid_jitter"
                                        logger.info(f"[Detector-{self.source_id}] ID {tid} UNKNOWN疑似摆动/ID切换重复过线，抑制输出")
                        except Exception:
                            pass

                    if is_dup and not self.allow_recross:
                        # P0防漏：UNKNOWN被去重时不直接判死，允许后续再次触发
                        if not curr_bib and not (dup_reason.startswith("same_tid") or dup_reason in {"unknown_same_tid", "unknown_reid_jitter", "unknown_nested_bbox"}):
                            state.crossed = False
                            state.pending_event_data = None
                            state.cross_signal_count = 0
                            state.cross_signal_last_time = 0.0
                            logger.info(f"[Detector-{self.source_id}] ID {tid} UNKNOWN疑似误去重({dup_reason})，重置等待下次触发")
                            continue
                        state.event_emitted = True
                        continue
                    
                    # 确定号码状态 (针对自行车赛优化：高置信度或名单匹配自动通过)
                    if state.best_bib:
                        is_in_list = hasattr(self, '_athlete_list') and state.best_bib in self._athlete_list
                        # 如果在名单内且置信度达标，或者置信度极高，直接标记为 RECOGNIZED，无需人工确认
                        if (is_in_list and state.best_bib_conf >= self.ocr_conf_threshold) or state.best_bib_conf >= 0.8:
                            bib_status = BibStatus.RECOGNIZED
                        else:
                            bib_status = BibStatus.NEEDS_REVIEW
                    else:
                        bib_status = BibStatus.UNRECOGNIZED
                        
                    # 裁剪运动员
                    crop = state.best_athlete_crop
                    if crop is None:
                        bx1, by1, bx2, by2 = data['bbox']
                        p_pad = 20
                        crop = data['frame'][max(0, by1-p_pad):min(height, by2+p_pad),
                                           max(0, bx1-p_pad):min(width, bx2+p_pad)].copy()

                    validation_frame = data.get('frame')
                    validation_bbox = data.get('bbox') or []
                    validation_context_crops = self._build_athlete_validation_context_crops(
                        validation_frame,
                        validation_bbox,
                    )
                    allow_edge_bicycle = False
                    if validation_frame is not None and len(validation_bbox) == 4:
                        frame_width = validation_frame.shape[1]
                        allow_edge_bicycle = (
                            int(validation_bbox[0]) <= 1
                            or int(validation_bbox[2]) >= frame_width - 1
                        )
                    if (
                        validation_frame is not None
                        and not self._passes_profile_athlete_geometry(
                            validation_bbox,
                            validation_frame.shape[:2],
                        )
                    ):
                        state.vlm_is_background = True
                        state.event_emitted = True
                        self.last_rejected_candidates.append({
                            "track_id": int(tid),
                            "reason": "profile_geometry",
                            "bbox": list(data['bbox']),
                            "position": list(data['position']),
                            "has_bib_box": bool(state.has_bib_box),
                            "frame": data.get('frame'),
                            "crop": crop,
                            "bib_crop": state.best_bib_crop,
                        })
                        continue
                    if not self._passes_athlete_event_validation(
                        state,
                        crop,
                        validation_context_crops,
                        allow_edge_bicycle=allow_edge_bicycle,
                    ):
                        state.vlm_is_background = True
                        state.event_emitted = True
                        rejected_bib_crop = state.best_bib_crop
                        if rejected_bib_crop is None:
                            rejected_candidates = self._get_ocr_candidates(state)
                            if rejected_candidates:
                                rejected_bib_crop = rejected_candidates[0][1]
                        self.last_rejected_candidates.append({
                            "track_id": int(tid),
                            "reason": "no_bicycle",
                            "bbox": list(data['bbox']),
                            "position": list(data['position']),
                            "has_bib_box": bool(state.has_bib_box),
                            "frame": data.get('frame'),
                            "crop": crop,
                            "bib_crop": rejected_bib_crop,
                        })
                        logger.info(f"[Detector-{self.source_id}] ID {tid} 未检测到自行车，取消非运动员事件")
                        continue

                    # === 选择“最清晰”的 bib 证据截图（不影响检测，只影响保存的 bib.jpg） ===
                    bib_crop_for_save = None
                    bib_bbox_for_save = []
                    bib_quality_for_save = 0.0

                    bib_candidates_for_event = []
                    seen_candidate_frames = set()
                    for candidate in detected_candidates:
                        q, candidate_crop, _, candidate_bbox = candidate
                        if candidate_crop is None or not isinstance(candidate_crop, np.ndarray) or candidate_crop.size == 0:
                            continue
                        if isinstance(candidate, BibEvidenceCandidate):
                            frame_key = (candidate.source, candidate.frame_index)
                            if frame_key in seen_candidate_frames:
                                continue
                            seen_candidate_frames.add(frame_key)
                            bib_candidates_for_event.append(candidate.detached_copy())
                        else:
                            if not self._bbox_contains_bbox(
                                data['bbox'],
                                candidate_bbox,
                                tolerance=5.0,
                            ):
                                continue
                            bib_candidates_for_event.append(
                                BibEvidenceCandidate(
                                    quality=float(q),
                                    crop=candidate_crop.copy(),
                                    frame=None,
                                    frame_index=-1,
                                    capture_time_ms=float(data.get('cross_time', current_time)) * 1000.0,
                                    athlete_bbox=tuple(int(value) for value in data['bbox']),
                                    bib_bbox=tuple(int(value) for value in (candidate_bbox or [])),
                                    source="detected",
                                    owner_validated=True,
                                )
                            )
                        if len(bib_candidates_for_event) >= 2:
                            break
                    candidates = crossing_ocr_candidates
                    if candidates:
                        try:
                            def _first_valid(cache_items, require_frame: bool):
                                for q, c, frame_ref, bb in cache_items:
                                    if c is None or (not isinstance(c, np.ndarray)) or c.size == 0:
                                        continue
                                    ch, cw = c.shape[:2]
                                    if ch < 14 or cw < 14:
                                        continue
                                    if require_frame and frame_ref is None:
                                        continue
                                    return float(q), c, bb
                                return None

                            # 优先选“接近过线/质量较高时刻”的缓存（frame_ref 不为 None）
                            best_near = _first_valid(candidates, require_frame=True)
                            best_any = _first_valid(candidates, require_frame=False)
                            detected_pick = _first_valid(detected_candidates, require_frame=False)
                            best_pick = detected_pick or best_near or best_any
                            if detected_pick is not None:
                                bib_crop_for_save = None

                            if best_pick:
                                pick_q, pick_crop, pick_bbox = best_pick

                                replace = False
                                if bib_crop_for_save is None or (not isinstance(bib_crop_for_save, np.ndarray)) or bib_crop_for_save.size == 0:
                                    replace = True
                                else:
                                    bh, bw = bib_crop_for_save.shape[:2]
                                    ph, pw = pick_crop.shape[:2]
                                    # 1) 当前 bib 图太小/太糊时，用更好的替换
                                    if bh < 18 or bw < 18:
                                        replace = True
                                    # 2) 缓存质量明显更好时替换（避免"人图清晰但 bib.jpg 糊"的情况）
                                    if pick_q > (bib_quality_for_save + 0.08):
                                        replace = True

                                if replace:
                                    bib_crop_for_save = pick_crop.copy()
                                    bib_bbox_for_save = pick_bbox
                                    bib_quality_for_save = pick_q
                        except Exception:
                            pass

                    # 最后兜底：如果实在没有 bib 小框，就从运动员框里截“胸腹区域”当作 OCR 输入证据
                    if (bib_crop_for_save is None or (not isinstance(bib_crop_for_save, np.ndarray)) or bib_crop_for_save.size == 0) and data.get("frame") is not None:
                        try:
                            fb_crop, fb_bbox = self._extract_bib_fallback_from_athlete(data["frame"], data["bbox"])
                            if fb_crop is not None and fb_crop.size > 0:
                                bib_crop_for_save = fb_crop
                                bib_bbox_for_save = fb_bbox or []
                                bib_quality_for_save = float(self._calculate_image_quality(fb_crop))
                        except Exception:
                            pass
                    
                    cross_time_value = data.get('cross_time')
                    cross_unix = float(time.time() if cross_time_value is None else cross_time_value)
                    cross_realtime_value = data.get('cross_realtime')
                    cross_realtime_unix = float(
                        cross_unix if cross_realtime_value is None else cross_realtime_value
                    )
                    cross_dt = datetime.fromtimestamp(cross_unix)
                    cross_realtime_dt = datetime.fromtimestamp(cross_realtime_unix)

                    participant_id = curr_participant_id
                    event_raw_track_ids = tuple(
                        sorted(
                            {
                                int(raw_track_id)
                                for raw_track_id in (data.get('raw_track_ids') or ())
                                if raw_track_id is not None and int(raw_track_id) >= 0
                            }
                        )
                    )

                    event = CrossingEvent(
                        event_id=0,
                        rank=0,
                        track_id=tid,
                        cross_time=cross_unix,
                        cross_time_str=cross_dt.strftime("%H:%M:%S.%f")[:-3],
                        cross_realtime=cross_realtime_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                        bib_number=state.best_bib,
                        bib_confidence=state.best_bib_conf,
                        bib_status=bib_status,
                        detection_confidence=data['conf'],
                        position=data['position'],
                        bbox=data['bbox'],
                        source_id=self.source_id,
                        frame=data['frame'],
                        crop=crop,
                        bib_crop=bib_crop_for_save,
                        bib_bbox=bib_bbox_for_save,
                        bib_evidence_kind=bib_evidence_kind_for_event,
                        bib_candidates=bib_candidates_for_event,
                        quality_score=bib_quality_for_save,
                        participant_id=participant_id,
                        raw_track_ids=event_raw_track_ids,
                        sport_profile=self.sport_profile,
                        frame_athletes=tuple(data.get('frame_athletes') or ()),
                    )

                    # P0：抑制补人虚拟ID的同帧重复事件（不影响真实ID）
                    if int(tid) >= 900000:
                        suppress_virtual_dup = False
                        for prev_event in crossing_events:
                            prev_tid = int(getattr(prev_event, 'track_id', -1))
                            prev_bbox = getattr(prev_event, 'bbox', None)
                            prev_pos = getattr(prev_event, 'position', None)
                            if not prev_bbox or not prev_pos:
                                continue
                            dxv = abs(float(event.position[0]) - float(prev_pos[0]))
                            dyv = abs(float(event.position[1]) - float(prev_pos[1]))
                            iouv = bbox_iou(list(event.bbox), list(prev_bbox))

                            if prev_tid < 900000:
                                if iouv > 0.40 or (dxv <= 28 and dyv <= 60 and iouv > 0.20):
                                    suppress_virtual_dup = True
                            else:
                                if iouv > 0.72 and dxv <= 20 and dyv <= 44:
                                    suppress_virtual_dup = True

                            if suppress_virtual_dup:
                                logger.info(f"[Detector-{self.source_id}] ID {tid} 虚拟补人同帧近邻重复，抑制事件输出")
                                state.event_emitted = True
                                break

                        if suppress_virtual_dup:
                            continue

                    if participant_id:
                        candidate = CrossingCandidate(
                            session_id=self._crossing_session_id,
                            source_id=self.source_id,
                            participant_id=participant_id,
                            raw_track_id=tid,
                            crossing_time_ms=cross_unix * 1000.0,
                        )
                        if not self._crossing_lifecycle.admit(candidate):
                            state.event_emitted = True
                            logger.info(
                                f"[Detector-{self.source_id}] participant {participant_id} crossing rejected by lifecycle"
                            )
                            continue
                        lifecycle_snapshot = self._crossing_lifecycle.snapshot(candidate)
                        if lifecycle_snapshot is not None:
                            event.passage_index = lifecycle_snapshot.passage_index
                            event.raw_track_ids = tuple(
                                sorted(
                                    set(lifecycle_snapshot.raw_track_ids).union(
                                        event_raw_track_ids
                                    )
                                )
                            )

                    event.event_id = self._event_id + 1
                    self._event_id = event.event_id
                    crossing_events.append(event)
                    state.event_emitted = True
                    
                    # 记录到最近事件
                    self._recent_events.append({
                        'time': current_time,
                        'bbox': data['bbox'],
                        'tid': tid,
                        'bib': state.best_bib,
                        'bib_evidence_kind': bib_evidence_kind_for_event,
                        'participant_id': participant_id,
                        'raw_track_ids': list(event_raw_track_ids),
                    })
                    
                    # 记录到已完成名单 (全局去重，记录时间戳)
                    if state.best_bib:
                        with self._lock:
                            # 记录最新时空位置，供短窗去重判定使用
                            self._finished_bibs[state.best_bib] = {
                                "time": current_time,
                                "bbox": list(data['bbox'])
                            }
                            
                            prefix, digits = self._split_bib_prefix_digits(state.best_bib)
                            if digits:
                                key = f"DIGITS_{digits}"
                                entry = self._finished_bibs.get(key)
                                if entry is None:
                                    self._finished_bibs[key] = {
                                        "time": current_time,
                                        "prefixes": {prefix},
                                        "bbox": list(data['bbox'])
                                    }
                                elif isinstance(entry, dict):
                                    entry["time"] = current_time
                                    entry["bbox"] = list(data['bbox'])
                                    entry.setdefault("prefixes", set())
                                    entry["prefixes"].add(prefix)
                                else:
                                    self._finished_bibs[key] = {
                                        "time": float(entry),
                                        "prefixes": {prefix},
                                        "bbox": list(data['bbox'])
                                    }
                    
                    # 限制大小，防止内存泄漏
                    if len(self._recent_events) > 100:
                        self._recent_events = self._recent_events[-100:]
                    
                    # 触发回调
                    if self._on_crossing:
                        try:
                            self._on_crossing(event)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")

        # 8. 最终运动员列表过滤与信息附加
        final_athletes = []
        for athlete in athletes:
            if athlete.get('is_static'):
                continue
            
            tid = athlete['track_id']
            with self._lock:
                state = self._track_states.get(tid)
                if state and state.is_static:
                    continue
                if state and state.best_bib:
                    athlete['bib_text'] = f" [{state.best_bib}]({state.best_bib_conf:.2f})"
                else:
                    athlete['bib_text'] = ""
            final_athletes.append(athlete)

        if self.enable_roi_filter and self._roi_polygon is not None:
            over_reject = roi_rejected_person >= max(1, int(raw_person_candidates * 0.65))
            if raw_person_candidates >= 2 and over_reject and len(final_athletes) <= 1:
                self._roi_overfilter_streak += 1
            else:
                self._roi_overfilter_streak = max(0, self._roi_overfilter_streak - 1)

        if self._roi_overfilter_streak >= 5:
            self.enable_roi_filter = False
            self._roi_overfilter_streak = 0
            roi_auto_disabled = True
            logger.warning(f"[Detector-{self.source_id}] ROI疑似过度过滤(多人候选被裁掉)，已自动关闭ROI过滤")

        if raw_person_candidates > 0 and len(final_athletes) == 0:
            self._empty_athlete_streak += 1
        else:
            self._empty_athlete_streak = 0

        # 过筛保护：候选很多但最终显示很少时，说明过滤过强，临时降级静态过滤
        if self._static_filter_cooldown_frames > 0:
            self._static_filter_cooldown_frames -= 1
            self._under_detect_streak = 0
        else:
            if raw_person_candidates >= 4 and len(final_athletes) <= 1:
                self._under_detect_streak += 1
            else:
                self._under_detect_streak = 0

            if self._under_detect_streak >= 10:
                self._static_filter_cooldown_frames = 120
                self._under_detect_streak = 0
                logger.warning(f"[Detector-{self.source_id}] 检测人数异常偏低，临时放宽静态过滤120帧")

        if self._empty_athlete_streak >= 8 and athletes:
            rescue_athletes = []
            for athlete in athletes:
                x1, y1, x2, y2 = athlete['bbox']
                bw = max(1, x2 - x1)
                bh = max(1, y2 - y1)
                area_ratio = (bw * bh) / max(1, width * height)
                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5

                if (cx < width * 0.10 or cx > width * 0.90) and bh > height * 0.45 and bw < width * 0.18:
                    continue
                if bw > width * 0.40 and bh < height * 0.22:
                    continue
                if area_ratio > 0.22 and cy < height * 0.72:
                    continue

                rescue_athletes.append(athlete)

            if rescue_athletes:
                with self._lock:
                    for athlete in rescue_athletes:
                        tid = athlete.get('track_id', -1)
                        state = self._track_states.get(tid)
                        if state and state.is_static and not state.vlm_is_background:
                            state.is_static = False
                            state.static_confidence = max(0.0, state.static_confidence - 0.4)
                final_athletes = rescue_athletes[:self.max_athletes_per_frame]
                self._empty_athlete_streak = 0
                logger.warning(f"[Detector-{self.source_id}] 启用空帧自恢复，恢复显示 {len(final_athletes)} 个目标")

        total_ms = (time.time() - _frame_t0) * 1000
        synthetic_tracks = sum(
            1
            for athlete in final_athletes
            if athlete.get('split_from_track') is not None or athlete.get('bib_driven')
        )
        raw_track_ids = {
            int(athlete["track_id"])
            for athlete in final_athletes
            if athlete.get("track_id") is not None
            and int(athlete.get("track_id", -1)) >= 0
        }
        participant_ids = {
            athlete.get("participant_id")
            for athlete in final_athletes
            if athlete.get("participant_id")
        }
        track_fragments_merged = sum(
            1
            for athlete in final_athletes
            if identity_resolutions.get(id(athlete)) is not None
            and identity_resolutions[id(athlete)].merged_raw_track
        )
        identity_ambiguities = sum(
            1
            for athlete in final_athletes
            if athlete.get("identity_status") == "AMBIGUOUS"
        )
        self.last_frame_metrics = {
            "raw_bikes": int(raw_bike_detections),
            "raw_bibs": int(raw_bib_detections),
            "validated_tracks": len(final_athletes),
            "synthetic_tracks": synthetic_tracks,
            "raw_tracks": len(raw_track_ids),
            "participants": len(participant_ids),
            "track_fragments_merged": int(track_fragments_merged),
            "identity_ambiguities": int(identity_ambiguities),
            "roi_auto_disabled": bool(roi_auto_disabled),
            "inference_ms": float(_infer_ms),
            "postprocess_ms": float(max(0.0, total_ms - _infer_ms)),
            "total_ms": float(total_ms),
        }

        # 每100帧输出一次总耗时，帮助定位瓶颈
        if self._frame_count % 100 == 1:
            logger.info(f"[Detector-{self.source_id}] 帧处理总耗时: {total_ms:.0f}ms (athletes={len(final_athletes)}, bibs={len(bibs)})")

        return crossing_events, final_athletes, bibs
