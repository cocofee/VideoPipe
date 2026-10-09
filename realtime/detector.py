"""
实时检测模块

复用bicycle_timing.py的检测逻辑：
- YOLO + ByteTrack 目标检测和跟踪
- 越线判定
- OCR号码识别
"""

import cv2
import numpy as np
from ultralytics import YOLO
try:
    from .independent_counting import FrameStamp, IndependentCounter
    from .count_observations import ObservedBoxCollector
except ImportError:
    from independent_counting import FrameStamp, IndependentCounter
    from count_observations import ObservedBoxCollector
import threading
import time
from pathlib import Path
from typing import Optional, Callable, Dict, Any, Tuple, List
from dataclasses import dataclass, field
from enum import Enum


class BibStatus(Enum):
    """号码识别状态"""
    UNRECOGNIZED = "unrecognized"  # 未识别
    NEEDS_REVIEW = "needs_review"  # 需复核
    RECOGNIZED = "recognized"      # 已识别


@dataclass
class CrossingEvent:
    """过线事件"""
    event_id: int
    rank: int
    track_id: int
    cross_time: float           # 过线时间（秒）
    cross_time_str: str         # 过线时间字符串
    bib_number: Optional[str]   # 号码
    bib_confidence: Optional[float]  # 识别置信度
    bib_status: BibStatus       # 识别状态
    detection_confidence: float # 检测置信度
    position: Tuple[int, int]   # 过线位置 (x, y)
    bbox: Tuple[int, int, int, int]  # 边界框
    frame: Optional[np.ndarray] = None  # 过线帧（完整）
    crop: Optional[np.ndarray] = None   # 骑手裁剪
    bib_crop: Optional[np.ndarray] = None  # 号码牌裁剪


@dataclass
class TrackState:
    """跟踪状态"""
    prev_x: int
    prev_y: int
    crossed: bool = False
    best_bib: Optional[str] = None
    best_bib_conf: float = 0.0
    best_bib_crop: Optional[np.ndarray] = None
    best_person_crop: Optional[np.ndarray] = None


def enhance_bib_image(img, scale_factor=4):
    """
    对BIB截图进行图像增强（从bicycle_timing.py复用）
    """
    if img is None or img.size == 0:
        return img

    h, w = img.shape[:2]

    # 1. 放大
    enlarged = cv2.resize(img, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_CUBIC)

    # 2. 转灰度
    if len(enlarged.shape) == 3:
        gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    else:
        gray = enlarged

    # 3. CLAHE对比度增强
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(gray)

    # 4. 锐化
    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    sharpened = cv2.filter2D(contrast_enhanced, -1, sharpen_kernel)

    # 5. 轻微去噪
    denoised = cv2.GaussianBlur(sharpened, (3, 3), 0)

    # 6. 再次锐化
    final = cv2.filter2D(denoised, -1, sharpen_kernel)

    # 转回BGR
    result = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)

    return result


def point_side_of_line(px: int, py: int, x1: int, y1: int, x2: int, y2: int) -> float:
    """
    计算点相对于线的位置（从bicycle_timing.py复用）

    返回值：正数表示在线的一侧，负数表示在另一侧，0表示在线上
    """
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


class Detector:
    """
    实时检测器

    处理视频帧，检测骑手过线事件
    """

    def __init__(self, model_path: str, *, model=None, ocr=None):
        """
        初始化检测器

        Args:
            model_path: YOLO模型路径
        """
        self.model_path = model_path
        self._model: Optional[YOLO] = model
        self._ocr = ocr

        # 终点线配置
        self._line_pt1 = (0, 400)
        self._line_pt2 = (1920, 400)

        # 检测参数
        self.detection_conf = 0.3
        self.process_imgsz = 640
        self.iou_threshold = 0.65
        self._frame_count = 0
        self.last_frame_metrics = {}
        self.ocr_conf_threshold = 0.7
        self.enable_enhancement = True
        self.bib_scale = 4

        # 状态
        self._track_states: Dict[int, TrackState] = {}
        self._event_id = 0
        self._start_time: Optional[float] = None

        # 回调
        self._on_crossing: Optional[Callable[[CrossingEvent], None]] = None
        self.independent_counter = None
        self._count_collector = None
        self._on_count_event = None
        self._on_count_diagnostics = None
        self.count_only = False

        # 线程锁
        self._lock = threading.Lock()

    def load_model(self) -> bool:
        """
        加载模型

        Returns:
            是否加载成功
        """
        try:
            print(f"[Detector] 加载模型: {self.model_path}")
            self._model = YOLO(self.model_path)
            from rapidocr_onnxruntime import RapidOCR
            self._ocr = RapidOCR()
            print(f"[Detector] 模型加载成功")
            return True
        except Exception as e:
            print(f"[Detector] 模型加载失败: {e}")
            return False

    def set_finish_line(self, pt1: Tuple[int, int], pt2: Tuple[int, int]):
        """
        设置终点线

        Args:
            pt1: 起点 (x, y)
            pt2: 终点 (x, y)
        """
        self._line_pt1 = pt1
        self._line_pt2 = pt2

    def set_on_crossing(self, callback: Callable[[CrossingEvent], None]):
        """设置过线事件回调"""
        self._on_crossing = callback

    def configure_counting(self, rules, event_sink=None, diagnostic_sink=None, *, count_only=False):
        """Enable the observed-event outlet before legacy OCR and filtering."""
        if self._model is None:
            raise ValueError("load the detection model before configuring independent counting")
        names = getattr(self._model, "names", {})
        labels = dict(enumerate(names)) if isinstance(names, (tuple, list)) else names
        if labels and any(class_id not in labels for class_id in rules.allowed_classes):
            raise ValueError("allowed count classes do not exist in the loaded model")
        if self._count_collector is not None:
            self._count_collector.close()
        self.independent_counter = IndependentCounter(rules)
        self._count_collector = ObservedBoxCollector(self._model, allowed_classes=rules.allowed_classes if count_only else None)
        self._on_count_event = event_sink
        self._on_count_diagnostics = diagnostic_sink
        self.count_only = count_only

    def stop(self):
        """Release the independent detection audit callback."""
        if self._count_collector is not None:
            self._count_collector.close()
            self._count_collector = None

    def reset(self, start_event_id: int = 0):
        """
        重置状态

        Args:
            start_event_id: 起始事件ID（用于续接之前的记录）
        """
        with self._lock:
            self._track_states.clear()
            self._event_id = start_event_id
            self._start_time = None

    def process_frame(self, frame: np.ndarray, timestamp: float = None, *, frame_stamp=None) -> Tuple[List[CrossingEvent], List[dict], List[dict]]:
        """
        处理一帧

        Args:
            frame: 视频帧
            timestamp: 时间戳（秒），如果为None则使用启动后的时间

        Returns:
            (过线事件列表, 检测到的骑手列表, 检测到的BIB列表)
        """
        if self._model is None:
            return [], [], []

        frame_started = time.monotonic()
        self._frame_count += 1
        if frame_stamp is None:
            now_ms = int(time.time() * 1000)
            frame_stamp = FrameStamp(self._frame_count, "legacy", now_ms, now_ms, "processing_clock")
        if self._count_collector is not None:
            self._count_collector.begin(frame_stamp)

        # 初始化开始时间
        if self._start_time is None:
            self._start_time = time.time()

        # 计算当前时间
        if timestamp is None:
            current_time = time.time() - self._start_time
        else:
            current_time = timestamp

        frame_original = frame.copy()
        height, width = frame.shape[:2]
        crossing_events = []

        # 运行跟踪检测
        track_kwargs = {"persist": True, "verbose": False, "tracker": "bytetrack.yaml", "conf": self.detection_conf}
        if self.independent_counter is not None:
            track_kwargs.update(imgsz=self.process_imgsz, iou=self.iou_threshold, augment=False)
            if self.count_only:
                track_kwargs["tracker"] = str(Path(__file__).with_name("count_bytetrack.yaml"))
        inference_started = time.monotonic()
        track_results = self._model.track(frame, **track_kwargs)
        inference_ms = (time.monotonic() - inference_started) * 1000

        if self.independent_counter is not None:
            boxes = []
            for result in track_results:
                if result.boxes is None:
                    continue
                for index in range(len(result.boxes)):
                    track_id = int(result.boxes.id[index]) if result.boxes.id is not None else None
                    boxes.append((int(result.boxes.cls[index]), float(result.boxes.conf[index]),
                                  result.boxes.xyxy[index].tolist(), track_id))
            observations = self._count_collector.link(boxes)
            raw = self._count_collector.raw
            count_events = self.independent_counter.process(frame_stamp, observations, raw_detections=raw)
            for event in count_events:
                if self._on_count_event is not None:
                    self._on_count_event(event)
            if self._on_count_diagnostics is not None:
                self._on_count_diagnostics(frame_stamp, raw, observations,
                                           self.independent_counter.last_diagnostics,
                                           self.independent_counter.last_quality)
            if self.count_only:
                total_ms = (time.monotonic() - frame_started) * 1000
                self.last_frame_metrics = {"raw_bikes": sum(row.class_id in self.independent_counter.rules.allowed_classes for row in raw),
                                           "raw_bibs": 0, "validated_tracks": len(observations), "synthetic_tracks": 0,
                                           "inference_ms": inference_ms, "postprocess_ms": max(0, total_ms - inference_ms),
                                           "total_ms": total_ms}
                return [], [], []

        # 运行普通检测（获取BIB）
        detect_results = self._model(frame, verbose=False, conf=self.detection_conf)

        # 收集骑手
        bikes = []
        for result in track_results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls = int(boxes.cls[i])
                if cls == 0:  # bike
                    x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                    conf = float(boxes.conf[i])
                    track_id = -1
                    if boxes.id is not None and i < len(boxes.id):
                        track_id = int(boxes.id[i])
                    bikes.append({
                        'bbox': [x1, y1, x2, y2],
                        'conf': conf,
                        'track_id': track_id,
                        'center_x': (x1 + x2) // 2,
                        'center_y': (y1 + y2) // 2,
                        'bottom_y': y2
                    })

        # 收集BIB
        bibs = []
        for result in detect_results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls = int(boxes.cls[i])
                if cls == 1:  # BIB
                    x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                    conf = float(boxes.conf[i])
                    bibs.append({
                        'bbox': [x1, y1, x2, y2],
                        'conf': conf,
                        'center_x': (x1 + x2) // 2,
                        'center_y': (y1 + y2) // 2
                    })

        # OCR识别BIB
        for bib in bibs:
            bib_x1, bib_y1, bib_x2, bib_y2 = bib['bbox']
            bib_cx, bib_cy = bib['center_x'], bib['center_y']

            # 找对应的骑手
            matched_bike = None
            for bike in bikes:
                bx1, by1, bx2, by2 = bike['bbox']
                if bx1 - 30 <= bib_cx <= bx2 + 30 and by1 - 30 <= bib_cy <= by2 + 30:
                    matched_bike = bike
                    break

            if matched_bike is None or matched_bike['track_id'] < 0:
                continue

            track_id = matched_bike['track_id']

            # 裁剪BIB
            padding = 10
            crop_x1 = max(0, bib_x1 - padding)
            crop_y1 = max(0, bib_y1 - padding)
            crop_x2 = min(width, bib_x2 + padding)
            crop_y2 = min(height, bib_y2 + padding)
            bib_crop = frame_original[crop_y1:crop_y2, crop_x1:crop_x2].copy()

            if bib_crop.size == 0:
                continue

            # 裁剪整个人
            px1, py1, px2, py2 = matched_bike['bbox']
            person_padding = 30
            person_x1 = max(0, px1 - person_padding)
            person_y1 = max(0, py1 - person_padding)
            person_x2 = min(width, px2 + person_padding)
            person_y2 = min(height, py2 + person_padding)
            person_crop = frame_original[person_y1:person_y2, person_x1:person_x2].copy()

            # 图像增强
            if self.enable_enhancement:
                bib_for_ocr = enhance_bib_image(bib_crop, self.bib_scale)
            else:
                bib_for_ocr = bib_crop

            # OCR识别
            try:
                ocr_result, _ = self._ocr(bib_for_ocr)
                if ocr_result:
                    for item in ocr_result:
                        box, text, ocr_conf = item
                        clean_text = ''.join(c for c in text if c.isalnum())

                        if clean_text and 1 <= len(clean_text) <= 5:
                            is_valid = (clean_text.isdigit() or
                                       (len(clean_text) > 1 and clean_text[0].isalpha() and clean_text[1:].isdigit()))

                            if is_valid and ocr_conf > 0.3:
                                # 更新最佳识别结果
                                with self._lock:
                                    if track_id not in self._track_states:
                                        self._track_states[track_id] = TrackState(
                                            prev_x=matched_bike['center_x'],
                                            prev_y=matched_bike['bottom_y']
                                        )

                                    state = self._track_states[track_id]
                                    if ocr_conf > state.best_bib_conf:
                                        state.best_bib = clean_text
                                        state.best_bib_conf = ocr_conf
                                        state.best_bib_crop = bib_crop.copy()
                                        state.best_person_crop = person_crop.copy()
            except Exception:
                pass

        # 处理过线
        line_x1, line_y1 = self._line_pt1
        line_x2, line_y2 = self._line_pt2

        for bike in bikes:
            track_id = bike['track_id']
            if track_id < 0:
                continue

            x1, y1, x2, y2 = bike['bbox']
            center_x = bike['center_x']
            bottom_y = bike['bottom_y']
            conf = bike['conf']

            with self._lock:
                if track_id not in self._track_states:
                    self._track_states[track_id] = TrackState(
                        prev_x=center_x,
                        prev_y=bottom_y
                    )

                state = self._track_states[track_id]

                if not state.crossed:
                    prev_x = state.prev_x
                    prev_y = state.prev_y

                    # 判断是否穿过终点线
                    prev_side = point_side_of_line(prev_x, prev_y, line_x1, line_y1, line_x2, line_y2)
                    curr_side = point_side_of_line(center_x, bottom_y, line_x1, line_y1, line_x2, line_y2)

                    # 从上到下穿过（prev_side < 0 变为 curr_side >= 0）
                    if prev_side < 0 and curr_side >= 0:
                        state.crossed = True
                        self._event_id += 1

                        # 时间字符串
                        minutes = int(current_time // 60)
                        seconds = current_time % 60
                        cross_time_str = f"{minutes:02d}:{seconds:06.3f}"

                        # 判断状态
                        if state.best_bib:
                            if state.best_bib_conf >= self.ocr_conf_threshold:
                                bib_status = BibStatus.RECOGNIZED
                            else:
                                bib_status = BibStatus.NEEDS_REVIEW
                        else:
                            bib_status = BibStatus.UNRECOGNIZED

                        # 过线时的骑手裁剪
                        person_padding = 50
                        crop_x1 = max(0, x1 - person_padding)
                        crop_y1 = max(0, y1 - person_padding)
                        crop_x2 = min(width, x2 + person_padding)
                        crop_y2 = min(height, y2 + person_padding)
                        person_crop_now = frame_original[crop_y1:crop_y2, crop_x1:crop_x2].copy()

                        # 创建事件
                        event = CrossingEvent(
                            event_id=self._event_id,
                            rank=self._event_id,
                            track_id=track_id,
                            cross_time=round(current_time, 3),
                            cross_time_str=cross_time_str,
                            bib_number=state.best_bib,
                            bib_confidence=round(state.best_bib_conf, 3) if state.best_bib else None,
                            bib_status=bib_status,
                            detection_confidence=round(conf, 3),
                            position=(center_x, bottom_y),
                            bbox=(x1, y1, x2, y2),
                            frame=frame_original.copy(),
                            crop=person_crop_now,
                            bib_crop=state.best_bib_crop
                        )

                        crossing_events.append(event)

                        # 触发回调
                        if self._on_crossing:
                            try:
                                self._on_crossing(event)
                            except Exception as e:
                                print(f"[Detector] 回调错误: {e}")

                # 更新位置
                state.prev_x = center_x
                state.prev_y = bottom_y

        return crossing_events, bikes, bibs

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                'event_count': self._event_id,
                'tracked_objects': len(self._track_states),
            }


# 测试代码
if __name__ == "__main__":
    print("=" * 50)
    print("  Detector 模块")
    print("=" * 50)
    print()
    print("此模块需要配合StreamReader使用")
    print("请运行 realtime/main.py 进行完整测试")
