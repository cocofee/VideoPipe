"""
事件记录器模块

保存过线事件截图和数据
"""

import cv2
import numpy as np
import threading
import queue
import time
import requests
from requests.auth import HTTPDigestAuth
import json
import os
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any
from datetime import datetime

try:
    from .logger import logger
except ImportError:
    import logging
    logger = logging.getLogger("EventRecorder")

try:
    from .io_utils import read_image_unicode, write_image_unicode
except ImportError:
    from io_utils import read_image_unicode, write_image_unicode

try:
    from .detector import CrossingEvent, BibStatus
    from .database import Database
except ImportError:
    from detector import CrossingEvent, BibStatus
    from database import Database


class EventRecorder:
    """
    事件记录器

    异步保存截图和数据到数据库，支持事件驱动的证据落盘
    """

    def __init__(self, output_dir: str, database: Database, enable_dedup: bool = True, camera_configs: Optional[dict] = None, source_count: int = 1):
        """
        初始化事件记录器

        Args:
            output_dir: 输出目录
            database: 数据库实例
            enable_dedup: 是否启用号码去重
            camera_configs: 摄像头高清抓拍配置 {source_id: {"ip": str, "user": str, "pass": str}}
        """
        self.output_dir = Path(output_dir).absolute()
        self.database = database
        self.enable_dedup = enable_dedup
        self.camera_configs = camera_configs or {}
        self.source_count = source_count

        # 创建基础目录
        self.evidence_root = self.output_dir / "evidence_photos"
        self.evidence_root.mkdir(parents=True, exist_ok=True)
        
        # 旧版目录保留兼容 (可选)
        self.screenshot_full_dir = self.output_dir / "screenshots_full"
        self.screenshot_clean_dir = self.output_dir / "screenshots_clean"
        self.screenshot_crop_dir = self.output_dir / "screenshots_crop"
        self.screenshot_bib_dir = self.output_dir / "screenshots_bib"
        self.hard_examples_dir = self.output_dir / "hard_examples"
        self.screenshot_isapi_dir = self.output_dir / "screenshots_isapi"

        self.screenshot_full_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_clean_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_crop_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_bib_dir.mkdir(parents=True, exist_ok=True)
        self.hard_examples_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_isapi_dir.mkdir(parents=True, exist_ok=True)

        # 事件保存队列（限制大小防止内存泄漏）
        self._queue: queue.Queue = queue.Queue(maxsize=100)
        # ISAPI 抓拍队列（限制大小防止内存泄漏）
        self._snapshot_queue: queue.Queue = queue.Queue(maxsize=20)
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._snapshot_thread: Optional[threading.Thread] = None

        # 回调
        self._on_event_saved: Optional[Callable[[int], None]] = None

        # 统计
        self._saved_count = 0
        self._dropped_critical_count = 0
        
        # 内存缓存：记录已处理的号码及其时间，防止异步 IO 延迟导致的重复记录
        self._processed_bibs = {}  # {bib_number: last_seen_timestamp}
        self._last_unknown_time = 0 # 最近一个未知号码记录的时间
        self._lock = threading.Lock()
        self._dedup_time_window = 60  # 有号码去重时间窗口（秒）- 增加到 60 秒
        self._unknown_dedup_window = 0.60 # 无号码去重时间窗口（秒）- 进一步收紧，优先不漏人
        self._unknown_dedup_dx = 80      # 水平距离阈值（像素）- 收紧，减少不同人的误合并
        self._unknown_dedup_dy = 90      # 垂直距离阈值（像素）- 收紧，减少不同人的误合并
        self._unknown_dedup_iou = 0.50   # IoU阈值 - 提高，要求更强重叠证据
        # UNKNOWN 跨 track_id 去重：仅在极强相似时才合并，避免“有框不刷新”
        # 默认关闭 UNKNOWN 跨 track_id 去重：人群密集时很容易把“不同人”误合并成“同一人”，造成漏刷新/漏人数
        # 如确实需要跨 ID 去重，可把该值改回 0.30 或更大，并配合 dx/dy/iou 调参
        self._unknown_cross_tid_window = 0.0
        self._unknown_cross_tid_dx = 8
        self._unknown_cross_tid_dy = 18
        self._unknown_cross_tid_iou = 0.86
        # 修正：确保这些参数一定被正确初始化（避免上一行注释串行导致赋值失效）
        self._unknown_dedup_window = 0.60
        self._unknown_dedup_dx = 80
        self._unknown_dedup_dy = 90
        self._unknown_dedup_iou = 0.50
        self._unknown_cross_tid_window = 0.0

        # 【多机位归并参数】
        self._cross_camera_merge_window = 5.0  # 跨机位归并时间窗口（秒）- 8米龙门需要更大窗口

        # 【新增】位置去重缓存（不依赖号码）
        self._recent_positions = []  # [(time, source_id, cx, cy, bbox, event_id), ...]
        self._position_dedup_window = 3.0  # 位置去重时间窗口（秒）
        self._position_dedup_dist = 100    # 位置去重距离阈值（像素）

        # 全局事件 ID 计数器 (初始化为数据库中最大的 ID)
        self._current_event_id = self.database.get_latest_event_id()
        logger.info(f"[EventRecorder] 初始化全局事件 ID 起点: {self._current_event_id}")

    def _write_image(self, img: np.ndarray, path: Path, quality: int = 90, encode_ext: Optional[str] = None) -> bool:
        """安全写入图片，支持中文路径"""
        try:
            return write_image_unicode(path, img, quality=quality, encode_ext=encode_ext)
        except Exception as e:
            logger.error(f"[EventRecorder] 写入图片失败 {path}: {e}")
            return False

    def _read_image(self, path: Path) -> Optional[np.ndarray]:
        """安全读取图片，支持中文路径"""
        try:
            return read_image_unicode(path)
        except Exception as e:
            logger.error(f"[EventRecorder] 读取图片失败 {path}: {e}")
            return None

    def set_on_event_saved(self, callback: Callable[[int], None]):
        """设置事件保存完成回调"""
        self._on_event_saved = callback

    def start(self):
        """启动记录器"""
        if self._running:
            return

        # 启动时同步一次数据库中的已存号码，初始化缓存（仅在启用去重时）
        if self.enable_dedup:
            try:
                # 加载最近N秒内的记录，避免加载历史数据
                events = self.database.get_all_events(include_void=False)
                current_time = 0.0
                with self._lock:
                    self._processed_bibs = {}
                    for e in events:
                        ct = e.get('cross_time')
                        try:
                            ct = float(ct)
                        except (TypeError, ValueError):
                            continue
                        if ct > current_time:
                            current_time = ct
                    for e in events:
                        bib = str(e.get('bib_number')) if e.get('bib_number') else None
                        if bib:
                            try:
                                cross_time = float(e.get('cross_time'))
                                # 只记录窗口内的号码
                                if current_time - cross_time < self._dedup_time_window:
                                    self._processed_bibs[bib] = cross_time
                            except (TypeError, ValueError):
                                pass
                    logger.info(f"[EventRecorder] 初始化完成，已加载 {len(self._processed_bibs)} 个近期号码")
            except Exception as e:
                logger.error(f"[EventRecorder] 初始化缓存失败: {e}")
        else:
            logger.info(f"[EventRecorder] 号码去重已禁用（耐力测试模式）")

        self._running = True
        
        # 启动保存线程
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

        # 启动抓拍线程
        self._snapshot_thread = threading.Thread(target=self._snapshot_loop, daemon=True)
        self._snapshot_thread.start()
            
        logger.info(f"[EventRecorder] 已启动，输出目录: {self.output_dir}")

    def stop(self):
        """停止记录器"""
        self._running = False
        if self._thread:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                with self._queue.mutex:
                    self._queue.queue.clear()
                    self._queue.unfinished_tasks = 0
                    self._queue.all_tasks_done.notify_all()
                try:
                    self._queue.put_nowait(None)
                except queue.Full:
                    logger.error("[EventRecorder] 停止时保存队列仍然满，跳过哨兵入队")
            self._thread.join(timeout=2)
            self._thread = None
        
        if self._snapshot_thread:
            try:
                self._snapshot_queue.put_nowait(None)
            except queue.Full:
                with self._snapshot_queue.mutex:
                    self._snapshot_queue.queue.clear()
                    self._snapshot_queue.unfinished_tasks = 0
                    self._snapshot_queue.all_tasks_done.notify_all()
                    self._snapshot_queue.not_full.notify_all()
                try:
                    self._snapshot_queue.put_nowait(None)
                except queue.Full:
                    logger.error("[EventRecorder] 停止时抓拍队列仍然满，跳过哨兵入队")
            self._snapshot_thread.join(timeout=2)
            self._snapshot_thread = None
            
        logger.info(f"[EventRecorder] 已停止，共保存 {self._saved_count} 个事件，丢弃关键事件 {self._dropped_critical_count} 个")

    def set_enable_dedup(self, enabled: bool):
        """设置是否启用号码去重"""
        self.enable_dedup = enabled
        logger.info(f"[EventRecorder] 号码去重已{'启用' if enabled else '禁用'}")
        if enabled:
            # 重新加载缓存
            self.reset()

    def reset(self):
        """重置记录器状态，清空缓存"""
        with self._lock:
            self._processed_bibs.clear()
            self._saved_count = 0
            # 重新从数据库加载已存在的号码（通常在重置后为空）
            if self.enable_dedup:
                try:
                    events = self.database.get_all_events(include_void=False)
                    current_time = 0.0
                    self._processed_bibs = {}
                    for e in events:
                        ct = e.get('cross_time')
                        try:
                            ct = float(ct)
                        except (TypeError, ValueError):
                            continue
                        if ct > current_time:
                            current_time = ct
                    for e in events:
                        bib = str(e.get('bib_number')) if e.get('bib_number') else None
                        if bib:
                            try:
                                cross_time = float(e.get('cross_time'))
                                if current_time - cross_time < self._dedup_time_window:
                                    self._processed_bibs[bib] = cross_time
                            except (TypeError, ValueError):
                                pass
                    logger.info(f"[EventRecorder] 状态已重置，重新加载了 {len(self._processed_bibs)} 个近期号码")
                except Exception as e:
                    logger.error(f"[EventRecorder] 重置缓存失败: {e}")
            else:
                logger.info("[EventRecorder] 状态已重置")

    def record(self, event: CrossingEvent):
        """
        记录事件（异步）
        Args:
            event: 过线事件
        """
        if not self._running:
            logger.warning("[EventRecorder] 记录器未运行，无法保存事件")
            return
        logger.debug(f"[EventRecorder] 收到事件并加入队列: ID={event.track_id}, Bib={event.bib_number}")
        self._enqueue_critical_event(event)

    def _enqueue_critical_event(self, event: CrossingEvent):
        try:
            self._queue.put_nowait(event)
            return
        except queue.Full:
            pass

        dropped = 0
        try:
            with self._queue.mutex:
                q = self._queue.queue
                kept = []
                while q:
                    it = q.popleft()
                    if it is None:
                        kept.append(it)
                        continue
                    if getattr(it, "is_update_only", False) or getattr(it, "is_info_update", False):
                        dropped += 1
                        continue
                    kept.append(it)
                for it in kept:
                    q.append(it)
                self._queue.unfinished_tasks = len(q)
                self._queue.all_tasks_done.notify_all()
                self._queue.not_full.notify_all()
        except Exception as e:
            logger.warning(f"[EventRecorder] 清理队列失败，将后台阻塞入队: {e}")

        if dropped:
            logger.warning(f"[EventRecorder] 队列满，已丢弃 {dropped} 条低优先更新以保过线事件")

        try:
            self._queue.put_nowait(event)
            return
        except queue.Full:
            pass

        try:
            self._queue.put(event, timeout=0.30)
            logger.warning(f"[EventRecorder] 队列持续满，等待后入队成功: tid={event.track_id}, bib={event.bib_number}")
            return
        except queue.Full:
            logger.error(
                f"[EventRecorder] 队列持续满，触发关键事件直写兜底: tid={event.track_id}, bib={event.bib_number}"
            )
            try:
                # 兜底策略：关键过线事件不再直接丢弃，改为当前线程直写。
                # 代价是高压时可能短时抖动，但能避免“漏记”。
                self._save_event(event)
                logger.warning(
                    f"[EventRecorder] 关键事件直写兜底成功: tid={event.track_id}, bib={event.bib_number}"
                )
                return
            except Exception:
                self._dropped_critical_count += 1
                logger.exception(
                    f"[EventRecorder] 关键事件直写兜底失败，事件被丢弃: tid={event.track_id}, bib={event.bib_number}, "
                    f"dropped={self._dropped_critical_count}"
                )

    def update_images(self, event_id: int, source_id: int, track_id: Optional[int] = None, bib_crop: Optional[np.ndarray] = None, 
                      athlete_crop: Optional[np.ndarray] = None, full_frame: Optional[np.ndarray] = None):
        """
        更新现有事件的截图 (异步)
        """
        if not self._running:
            return
            
        # 构造一个特殊的 CrossingEvent 用于更新
        update_event = CrossingEvent(
            event_id=event_id,
            rank=-1,
            track_id=track_id if track_id is not None else -1,
            cross_time=0.0,
            cross_time_str="",
            cross_realtime="",
            bib_number=None,
            bib_confidence=None,
            bib_status=BibStatus.UNRECOGNIZED,
            detection_confidence=0.0,
            position=(0, 0),
            bbox=(0, 0, 0, 0),
            source_id=source_id,
            frame=full_frame,
            crop=athlete_crop,
            bib_crop=bib_crop,
            is_update_only=True
        )
        try:
            self._queue.put_nowait(update_event)
        except queue.Full:
            logger.warning(f"[EventRecorder] 队列已满，丢弃补图更新: event_id={event_id}, source_id={source_id}")

    def update_event_info(self, event_id: int, bib: str, conf: float, status: Any, source_id: int, track_id: Optional[int] = None):
        """
        异步更新事件文本信息 (号码、置信度、状态)
        """
        if not self._running:
            return
            
        update_event = CrossingEvent(
            event_id=event_id,
            rank=-1,
            track_id=track_id if track_id is not None else -1,
            cross_time=0.0,
            cross_time_str="",
            cross_realtime="",
            bib_number=bib,
            bib_confidence=conf,
            bib_status=status,
            detection_confidence=0.0,
            position=(0, 0),
            bbox=(0, 0, 0, 0),
            source_id=source_id,
            is_info_update=True
        )
        try:
            self._queue.put_nowait(update_event)
        except queue.Full:
            logger.warning(f"[EventRecorder] 队列已满，丢弃补录信息更新: event_id={event_id}, source_id={source_id}")

    def save_hard_example(self, frame: np.ndarray, bib_crop: Optional[np.ndarray], bib_bbox: List[int], ocr_text: Optional[str]):
        """
        直接保存一个硬样本（用于照片批量模式）
        """
        try:
            time_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"hard_{time_str}.jpg"
            save_path = self.hard_examples_dir / filename
            
            # 1. 保存原图
            cv2.imwrite(str(save_path), frame)
            
            # 2. 保存标签 (YOLO 格式)
            if bib_bbox and len(bib_bbox) == 4:
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = bib_bbox
                xc = (x1 + x2) / 2.0 / w
                yc = (y1 + y2) / 2.0 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                
                label_path = save_path.with_suffix(".txt")
                with open(label_path, "w") as f:
                    # 类别 1 是号码牌
                    f.write(f"1 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
            
            # 3. 保存识别出的错误号码（如果有），存入一个同名的 .json 或记录在 log
            if ocr_text:
                log_path = self.hard_examples_dir / "ocr_results.log"
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"{filename}: {ocr_text}\n")
                    
            logger.info(f"[EventRecorder] 已保存硬样本: {filename}")
            return True
        except Exception as e:
            logger.error(f"[EventRecorder] 保存硬样本失败: {e}")
            return False

    def _process_loop(self):
        """处理循环"""
        while self._running:
            try:
                event = self._queue.get(timeout=1)
                if event is None:
                    break
                self._save_event(event)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[EventRecorder] 处理错误: {e}")

    def _snapshot_loop(self):
        """高清抓拍循环 (ISAPI)"""
        while self._running:
            try:
                task = self._snapshot_queue.get(timeout=1)
                if task is None:
                    break
                
                event_id = task.get('event_id')
                source_id = task.get('source_id')
                time_str = task.get('time_str')
                
                config = self.camera_configs.get(source_id)
                if not config:
                    continue
                
                # 海康 ISAPI 抓拍 URL (通道 101 通常是主码流)
                url = f"http://{config['ip']}/ISAPI/Streaming/channels/101/picture"
                
                try:
                    response = requests.get(
                        url, 
                        auth=HTTPDigestAuth(config['user'], config['pass']),
                        timeout=5
                    )
                    
                    if response.status_code == 200:
                        filename = f"isapi_{event_id:03d}_S{source_id}_{time_str}.jpg"
                        save_path = self.screenshot_isapi_dir / filename
                        
                        with open(save_path, "wb") as f:
                            f.write(response.content)
                        
                        # 将抓拍关联至证据链
                        evidence_data = {
                            'event_id': event_id,
                            'source_id': source_id,
                            'screenshot_full': str(save_path),
                            'confidence': 1.0
                        }
                        if self.source_count <= 1:
                            self.database.upsert_best_evidence(event_id, evidence_data)
                        else:
                            self.database.insert_evidence(evidence_data)
                        logger.info(f"[EventRecorder] 已保存 ISAPI 高清抓拍: {filename}")
                    else:
                        logger.warning(f"[EventRecorder] ISAPI 抓拍失败: HTTP {response.status_code}")
                except Exception as e:
                    logger.error(f"[EventRecorder] ISAPI 请求异常: {e}")
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[EventRecorder] 抓拍线程错误: {e}")

    def _save_evidence_to_disk(self, event: CrossingEvent) -> str:
        """
        将证据落盘：创建事件目录，保存图片和 meta.json
        返回事件目录路径
        """
        event_id_str = f"{event.event_id:06d}"
        event_dir = self.evidence_root / event_id_str
        event_dir.mkdir(parents=True, exist_ok=True)
        
        paths = {"full": None, "athlete": None, "bib": None, "bib_candidates": []}
        bib_candidate_metadata = []
        
        # 原子写函数
        def atomic_save(img: np.ndarray, target_path: Path):
            if img is None: return False
            try:
                target_path = target_path.absolute()
                tmp_path = Path(str(target_path) + ".tmp").absolute()
                
                # 使用类内置的安全写入方法
                if not self._write_image(img, tmp_path, quality=90, encode_ext=target_path.suffix):
                    tmp_exists = tmp_path.exists()
                    tmp_size = tmp_path.stat().st_size if tmp_exists else 0
                    logger.error(f"[EventRecorder] 原子保存写入失败: target={target_path}, tmp={tmp_path}, exists={tmp_exists}, size={tmp_size}")
                    return False
                    
                if target_path.exists():
                    try:
                        target_path.unlink()
                    except Exception:
                        pass
                if not tmp_path.exists():
                    logger.error(f"[EventRecorder] 原子保存缺失临时文件: target={target_path}, tmp={tmp_path}")
                    return False
                os.replace(str(tmp_path), str(target_path))
                return True
            except Exception as e:
                logger.error(f"[EventRecorder] 原子保存异常 {target_path}: {e}")
                return False

        # 1. 保存原图 (不带标注的 clean 图)
        if event.frame is not None:
            full_path = event_dir / "full.jpg"
            if atomic_save(event.frame, full_path):
                paths["full"] = "full.jpg"
                
        # 2. 保存运动员截图
        if event.crop is not None:
            athlete_path = event_dir / "athlete.jpg"
            if atomic_save(event.crop, athlete_path):
                paths["athlete"] = "athlete.jpg"
                
        # 3. 保存号码牌截图
        if event.bib_crop is not None:
            bib_path = event_dir / "bib.jpg"
            if atomic_save(event.bib_crop, bib_path):
                paths["bib"] = "bib.jpg"

        for index, candidate in enumerate(getattr(event, "bib_candidates", [])[:2], start=1):
            if len(candidate) < 2:
                continue
            candidate_image = getattr(candidate, "crop", candidate[1])
            if candidate_image is None or not isinstance(candidate_image, np.ndarray) or candidate_image.size == 0:
                continue
            candidate_name = f"bib_candidate_{index:02d}.jpg"
            if atomic_save(candidate_image, event_dir / candidate_name):
                paths["bib_candidates"].append(candidate_name)
                bib_candidate_metadata.append({
                    "path": candidate_name,
                    "frame_index": int(getattr(candidate, "frame_index", -1)),
                    "capture_time_ms": float(getattr(candidate, "capture_time_ms", 0.0)),
                    "athlete_bbox": list(getattr(candidate, "athlete_bbox", event.bbox) or []),
                    "bib_bbox": list(getattr(candidate, "bib_bbox", candidate[2] if len(candidate) > 2 else []) or []),
                    "source": str(getattr(candidate, "source", "detected") or "detected"),
                    "owner_validated": bool(getattr(candidate, "owner_validated", False)),
                    "quality": float(getattr(candidate, "quality", candidate[0] if len(candidate) > 0 else 0.0)),
                })
        
        # 4. 生成/更新 meta.json
        meta_path = event_dir / "meta.json"
        if meta_path.exists():
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                # 更新图片路径 (如果新保存了图片)
                for k, v in paths.items():
                    if v: meta["paths"][k] = v
                # 更新质量分和号码牌框 (如果是补录的高质量图)
                if event.quality_score > 0:
                    meta["quality"]["score"] = float(event.quality_score)
                if event.bib_bbox:
                    meta["bbox_bib"] = list(event.bib_bbox)
                if event.bib_evidence_kind:
                    meta["bib_evidence_kind"] = event.bib_evidence_kind
                if event.participant_id:
                    meta["participant_id"] = event.participant_id
                meta["raw_track_id"] = event.track_id
                meta["raw_track_ids"] = self._collect_raw_track_ids(
                    event,
                    meta.get("raw_track_ids", []),
                )
            except Exception as e:
                logger.error(f"[EventRecorder] 加载旧 meta.json 失败: {e}")
                meta = self._create_initial_meta(event, paths)
        else:
            meta = self._create_initial_meta(event, paths)
        if bib_candidate_metadata:
            meta["bib_candidate_metadata"] = bib_candidate_metadata
        elif "bib_candidate_metadata" not in meta:
            meta["bib_candidate_metadata"] = []
        
        tmp_meta_path = event_dir / "meta.json.tmp"
        with open(tmp_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        if os.path.exists(meta_path):
            os.remove(meta_path)
        os.rename(tmp_meta_path, meta_path)
        
        return str(event_dir)

    @staticmethod
    def _collect_raw_track_ids(event: CrossingEvent, existing: Optional[List[int]] = None) -> List[int]:
        raw_track_ids = set()
        for track_id in [*(existing or []), *(event.raw_track_ids or ()), event.track_id]:
            try:
                normalized = int(track_id)
            except (TypeError, ValueError):
                continue
            if normalized >= 0:
                raw_track_ids.add(normalized)
        return sorted(raw_track_ids)

    def _create_initial_meta(self, event: CrossingEvent, paths: dict) -> dict:
        """创建初始元数据字典"""
        return {
            "event_id": event.event_id,
            "track_id": event.track_id,
            "participant_id": event.participant_id,
            "raw_track_id": event.track_id,
            "raw_track_ids": self._collect_raw_track_ids(event),
            "cross_time_unix": event.cross_time,
            "cross_time_str": event.cross_time_str,
            "cross_realtime": event.cross_realtime,
            "camera_id": f"cam_{event.source_id}",
            "bbox_athlete": list(event.bbox) if event.bbox else None,
            "bbox_bib": list(event.bib_bbox) if event.bib_bbox else None,
            "bib_evidence_kind": event.bib_evidence_kind,
            "quality": {
                "score": float(event.quality_score)
            },
            "paths": paths,
            "status": {
                "ocr_state": "PENDING",
                "local_ocr_try": 0,
                "doubao_try": 0
            },
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        }

    def _save_event(self, event: CrossingEvent):
        """保存事件 (支持多机位归并与证据链)"""
        logger.info(f"[EventRecorder] 开始处理事件保存: ID={event.track_id}, Bib={event.bib_number}")
        try:
            # --- 0. 仅更新图片模式 ---
            if getattr(event, 'is_update_only', False):
                logger.debug(f"[EventRecorder] 仅更新图片模式: EventID={event.event_id}")
                self._update_event_images(event)
                return
                
            # --- 0.1 仅更新信息模式 ---
            if getattr(event, 'is_info_update', False):
                logger.debug(f"[EventRecorder] 仅更新信息模式: EventID={event.event_id}")
                self._update_event_info(event)
                return

            current_time = event.cross_time
            if not isinstance(event.cross_time_str, str):
                event.cross_time_str = str(event.cross_time_str or "")
            if event.bib_number is not None and not isinstance(event.bib_number, str):
                event.bib_number = str(event.bib_number)
            if not event.bbox or len(event.bbox) != 4:
                event.bbox = (0, 0, 0, 0)
            if not event.position or len(event.position) != 2:
                x1, y1, x2, y2 = event.bbox
                if x2 > x1 and y2 > y1:
                    event.position = ((x1 + x2) // 2, y2)
                else:
                    event.position = (0, 0)

            bib_status_str = event.bib_status.value if hasattr(event.bib_status, 'value') else str(event.bib_status)
            try:
                event_bib_conf = float(event.bib_confidence) if event.bib_confidence is not None else None
            except (TypeError, ValueError):
                event_bib_conf = None
            
            # --- 分配全局唯一 ID ---
            if not getattr(event, 'is_update_only', False) and not getattr(event, 'is_info_update', False):
                if event.event_id <= 0:
                    with self._lock:
                        self._current_event_id += 1
                        event.event_id = self._current_event_id
                    logger.debug(f"[EventRecorder] 分配新事件 ID: {event.event_id}")

            bib = event.bib_number.strip() if event.bib_number else ""
            is_unknown = bib == "" or bib.upper() == "UNKNOWN"
            bib_text = bib if bib else "unknown"

            match_event = None
            dedup_note = ""
            position_dedup_target_event_id = None

            # --- 0.5 位置去重（不依赖号码，最早拦截）---
            # 只在“强证据重复”时拦截，避免并排过线被误杀
            curr_cx, curr_cy = event.position
            with self._lock:
                # 清理过期记录
                self._recent_positions = [
                    r for r in self._recent_positions
                    if current_time - r[0] <= self._position_dedup_window
                ]
                # 检查是否重复
                for rp in self._recent_positions:
                    rp_time, rp_source_id, rp_cx, rp_cy, rp_bbox, rp_event_id = rp
                    if rp_source_id != event.source_id:
                        continue

                    prev_tid = None
                    if is_unknown and rp_event_id is not None:
                        try:
                            prev_ev = self.database.get_event(int(rp_event_id))
                        except Exception:
                            prev_ev = None
                        if prev_ev:
                            prev_tid = prev_ev.get('track_id')

                    dx = abs(curr_cx - rp_cx)
                    dy = abs(curr_cy - rp_cy)
                    time_diff = current_time - rp_time

                    x1, y1, x2, y2 = event.bbox
                    curr_bbox_valid = (x2 > x1 and y2 > y1)
                    curr_pos_valid = not (curr_cx == 0 and curr_cy == 0)
                    if not curr_bbox_valid or not curr_pos_valid:
                        continue

                    def _bbox_iou(a, b):
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

                    iou = _bbox_iou(list(event.bbox), rp_bbox)

                    is_strong_dup = (
                        time_diff <= 0.4 and (
                            iou >= 0.60 or
                            (dx <= 15 and dy <= 15 and iou >= 0.20)
                        )
                    )

                    # P0：UNKNOWN 去重更保守
                    # - 同一 track_id：允许正常短时去重
                    # - 不同 track_id：只允许“极强相似”去重，减少并排误吞
                    if is_unknown:
                        same_tid = False
                        if prev_tid is not None and event.track_id is not None:
                            try:
                                same_tid = int(prev_tid) == int(event.track_id)
                            except Exception:
                                same_tid = False

                        # UNKNOWN 场景更保守：
                        # - 同 track_id 也要满足“极高重叠+很短时间”
                        # - 不同 track_id 只在极强相似才去重
                        if same_tid:
                            is_strong_dup = (
                                time_diff <= 0.25 and
                                iou >= 0.90 and
                                dx <= 10 and
                                dy <= 22
                            )
                        else:
                            strict_cross_tid_dup = (
                                self._unknown_cross_tid_window > 0 and
                                time_diff <= self._unknown_cross_tid_window and
                                iou >= self._unknown_cross_tid_iou and
                                dx <= self._unknown_cross_tid_dx and
                                dy <= self._unknown_cross_tid_dy
                            )
                            is_strong_dup = strict_cross_tid_dup

                    if is_strong_dup:
                        logger.info(
                            f"[EventRecorder] 位置去重拦截: "
                            f"tid={event.track_id}, dx={dx:.0f}, dy={dy:.0f}, iou={iou:.2f}, dt={time_diff:.2f}s"
                        )
                        position_dedup_target_event_id = rp_event_id
                        dedup_note = f"DEDUP(POS,dt:{time_diff:.2f},iou:{iou:.2f})"
                        break

            if position_dedup_target_event_id is not None:
                match_event = self.database.get_event(position_dedup_target_event_id)
                if not match_event:
                    match_event = None
                    dedup_note = ""

            # --- 1. 异步事件归归并逻辑 ---
            if self.enable_dedup or is_unknown:
                # A. 已知号码：在较宽的时间窗口内寻找匹配（跨机位）
                if not is_unknown:
                    match_event = self.database.find_match_event(
                        event.bib_number, 
                        event.cross_time, 
                        window_seconds=self._dedup_time_window # 使用更大的窗口进行号码去重
                    )
                    if match_event:
                        dedup_note = f"DEDUP(BIB:{bib})"
                
                # B. 未知号码或需要跨机位归并：在较宽的时间窗口内寻找匹配
                if match_event is None:
                    # 使用跨机位归并时间窗口（默认5秒）
                    recent_events = self.database.get_events_since_time(
                        seconds=self._cross_camera_merge_window,
                        current_time=event.cross_time
                    )
                    
                    for ev in recent_events:
                        # 场景1：同一机位的去重（靠 IoU 和 坐标）
                        if ev.get('source_id') == event.source_id:
                            try:
                                import json
                                eb = json.loads(ev.get('bbox')) if isinstance(ev.get('bbox'), str) else ev.get('bbox')
                                if not eb or len(eb) != 4:
                                    continue
                                if not event.bbox or len(event.bbox) != 4:
                                    continue
                                if event.bbox[2] <= event.bbox[0] or event.bbox[3] <= event.bbox[1]:
                                    continue
                                
                                # 内部辅助函数计算 IoU
                                def _bbox_iou(a, b):
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

                                iou = _bbox_iou(event.bbox, eb if eb else event.bbox)

                                def _safe_pos(value, fallback):
                                    if value is None:
                                        return fallback
                                    try:
                                        return float(value)
                                    except (TypeError, ValueError):
                                        return fallback

                                ev_pos_x = _safe_pos(ev.get('position_x'), event.position[0])
                                ev_pos_y = _safe_pos(ev.get('position_y'), event.position[1])
                                dx = abs(event.position[0] - ev_pos_x)
                                dy = abs(event.position[1] - ev_pos_y)
                                ev_time = ev.get('cross_time')
                                try:
                                    ev_time = float(ev_time)
                                except (TypeError, ValueError):
                                    ev_time = None
                                curr_time = float(event.cross_time) if event.cross_time is not None else None
                                time_diff = abs(ev_time - curr_time) if (ev_time is not None and curr_time is not None) else None
                                ev_tid = ev.get('track_id')
                                
                                if is_unknown:
                                    if (ev_tid is not None and event.track_id is not None and
                                            int(ev_tid) == int(event.track_id) and
                                            time_diff is not None and
                                            time_diff <= 0.30 and
                                            dx <= 14 and dy <= 28 and
                                            iou >= 0.55):
                                        match_event = ev
                                        dedup_note = f"DEDUP(TID:{event.track_id})"
                                        logger.info(f"[EventRecorder] 发现同一机位匹配 (UNKNOWN同ID去重): 机位{event.source_id}, ID:{event.track_id}")
                                        break
                                    size_ratio_ok = True
                                    try:
                                        ebx1, eby1, ebx2, eby2 = eb
                                        ev_w = max(1, ebx2 - ebx1)
                                        ev_h = max(1, eby2 - eby1)
                                        cur_w = max(1, event.bbox[2] - event.bbox[0])
                                        cur_h = max(1, event.bbox[3] - event.bbox[1])
                                        ratio_w = cur_w / ev_w
                                        ratio_h = cur_h / ev_h
                                        size_ratio_ok = (0.6 <= ratio_w <= 1.8) and (0.6 <= ratio_h <= 1.8)
                                        dyn_dx = max(self._unknown_dedup_dx, int((ev_w + cur_w) * 0.35))
                                        dyn_dy = max(self._unknown_dedup_dy, int((ev_h + cur_h) * 0.55))
                                    except Exception:
                                        dyn_dx = self._unknown_dedup_dx
                                        dyn_dy = self._unknown_dedup_dy
                                        size_ratio_ok = True

                                    # P1收口：UNKNOWN 不同ID允许“极强相似”归并；其余仍按同ID归并
                                    same_tid = (ev_tid is not None and event.track_id is not None and int(ev_tid) == int(event.track_id))
                                    strict_cross_tid_dup = (
                                        (not same_tid) and
                                        time_diff is not None and
                                        self._unknown_cross_tid_window > 0 and
                                        time_diff <= self._unknown_cross_tid_window and
                                        size_ratio_ok and
                                        dx <= self._unknown_cross_tid_dx and
                                        dy <= self._unknown_cross_tid_dy and
                                        iou >= self._unknown_cross_tid_iou
                                    )

                                    same_tid_dup = (
                                        same_tid and
                                        time_diff is not None and
                                        time_diff <= min(self._unknown_dedup_window, 0.45) and
                                        size_ratio_ok and
                                        dx <= min(dyn_dx, 24) and
                                        dy <= min(dyn_dy, 48) and
                                        iou >= 0.55
                                    )

                                    if same_tid_dup or strict_cross_tid_dup:
                                        match_event = ev
                                        dedup_note = (
                                            f"DEDUP(UNK,dist:{int(dx)},{int(dy)})"
                                            if same_tid else f"DEDUP(UNKx,dist:{int(dx)},{int(dy)})"
                                        )
                                        logger.info(f"[EventRecorder] 发现同一机位匹配 (UNKNOWN近距去重): 机位{event.source_id}, dx:{dx:.1f}, dy:{dy:.1f}")
                                        break

                                    same_tid_high_iou = (
                                        same_tid and
                                        time_diff is not None and
                                        time_diff <= 0.30 and
                                        iou >= max(0.88, self._unknown_dedup_iou)
                                    )
                                    if same_tid_high_iou or strict_cross_tid_dup:
                                        match_event = ev
                                        dedup_note = (
                                            f"DEDUP(UNK,iou:{iou:.2f})"
                                            if same_tid else f"DEDUP(UNKx,iou:{iou:.2f})"
                                        )
                                        logger.info(f"[EventRecorder] 发现同一机位匹配 (UNKNOWN高IoU去重): 机位{event.source_id}, IoU:{iou:.2f}")
                                        break
                                else:
                                    # 如果 IoU 较大，或者水平/垂直位移都在合理范围内，则认为是同一人
                                    # 放宽阈值：dx 从 20 -> 40, dy 从 50 -> 80
                                    if iou >= 0.25 or (dx <= 40 and dy <= 80):
                                        match_event = ev
                                        dedup_note = f"DEDUP(ID:{event.track_id},dx:{int(dx)})"
                                        logger.info(f"[EventRecorder] 发现同一机位匹配 (ID跳变去重): 机位{event.source_id}, ID:{event.track_id} vs 旧ID:{ev.get('track_id')}, IoU:{iou:.2f}, dx:{dx:.1f}, dy:{dy:.1f}")
                                        break
                            except:
                                continue

                        # 场景2：跨机位归并（A+B方案核心逻辑）
                        # A: 号码关联 - 相同号码自动合并
                        # B: 时间窗口 - 在指定时间窗口内寻找匹配
                        else:
                            try:
                                ev_time = float(ev.get('cross_time'))
                                curr_time = float(event.cross_time)
                                time_diff = abs(ev_time - curr_time)

                                # 使用跨机位归并专用时间窗口（默认5秒，适配8米龙门）
                                if time_diff <= self._cross_camera_merge_window:
                                    ev_bib = ev.get('bib_number', '')
                                    ev_is_unknown = not ev_bib or ev_bib.upper() == "UNKNOWN"

                                    # 规则1：双方都是 UNKNOWN，不合并（避免误并导致漏记）
                                    if is_unknown and ev_is_unknown:
                                        continue

                                    # 规则2：双方都有号码且相同，合并
                                    if (not is_unknown) and (not ev_is_unknown) and ev_bib == bib:
                                        match_event = ev
                                        dedup_note = f"MERGE(S{ev.get('source_id')},dt:{time_diff:.1f}s,bib:{bib})"
                                        logger.info(f"[EventRecorder] 跨机位归并(号码匹配): 机位{ev.get('source_id')}→机位{event.source_id}, 号码={bib}, 时间差={time_diff:.2f}s")
                                        break

                                    # 规则3：一方有号码一方UNKNOWN，且时间窗口很近（≤2秒），可以合并
                                    # 这样可以实现：A机位识别到号码，B机位UNKNOWN，自动补充证据
                                    if time_diff <= 2.0:
                                        if (not is_unknown) and ev_is_unknown:
                                            # 当前有号码，旧记录UNKNOWN → 合并并补充号码
                                            match_event = ev
                                            dedup_note = f"MERGE(S{ev.get('source_id')},补号:{bib})"
                                            logger.info(f"[EventRecorder] 跨机位归并(号码补充): 机位{ev.get('source_id')}(UNKNOWN)→机位{event.source_id}({bib}), 时间差={time_diff:.2f}s")
                                            break
                                        elif is_unknown and (not ev_is_unknown):
                                            # 当前UNKNOWN，旧记录有号码 → 合并作为补充证据
                                            match_event = ev
                                            dedup_note = f"MERGE(S{ev.get('source_id')},补证据)"
                                            logger.info(f"[EventRecorder] 跨机位归并(证据补充): 机位{event.source_id}(UNKNOWN)→机位{ev.get('source_id')}({ev_bib}), 时间差={time_diff:.2f}s")
                                            break
                            except (TypeError, ValueError):
                                continue
            
            # --- 1.5 数据库层面强力去重 (Weak Unique Constraint) ---
            # 即使前面逻辑没挡住，入库前做最后一次拦截：同一秒内、同一机位、同一号码
            if self.enable_dedup and not match_event and not is_unknown:
                try:
                    try:
                        time_bucket = int(float(event.cross_time))
                    except (TypeError, ValueError):
                        time_bucket = None
                    
                    if time_bucket is not None and self.database.check_duplicate_record(event.source_id, time_bucket, bib):
                        logger.warning(f"[EventRecorder] 强力拦截重复入库: 机位{event.source_id}, 时间{time_bucket}, 号码{bib}")
                        return
                except Exception as e:
                    logger.error(f"[EventRecorder] 强力去重逻辑执行失败: {e}")
                
            # --- 2. 准备保存截图与证据落盘 ---
            # A. 新版事件驱动落盘 (按 event_id 分文件夹)
            evidence_dir = self._save_evidence_to_disk(event)
            
            # B. 旧版平铺保存 (保持兼容)
            time_for_filename = event.cross_time_str.replace(":", "-").replace(".", "-")
            source_suffix = f"_S{event.source_id}"

            screenshot_full = ""
            screenshot_clean = ""
            screenshot_crop = ""
            screenshot_bib = ""

            if event.frame is not None:
                full_filename = f"cross_{event.event_id:03d}_ID{event.track_id}{source_suffix}_{time_for_filename}_full.jpg"
                clean_filename = f"cross_{event.event_id:03d}_ID{event.track_id}{source_suffix}_{time_for_filename}_clean.jpg"
                full_path = self.screenshot_full_dir / full_filename
                clean_path = self.screenshot_clean_dir / clean_filename
                
                # 保存原图
                if self._write_image(event.frame, clean_path):
                    screenshot_clean = str(clean_path)

                # 绘制标注图
                frame_annotated = event.frame.copy()
                x1, y1, x2, y2 = event.bbox
                center_x, bottom_y = event.position
                cv2.rectangle(frame_annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.circle(frame_annotated, (center_x, bottom_y), 10, (0, 255, 0), -1)
                info_text = f"Rank:{event.rank} ID:{event.track_id} BIB:{bib_text} S:{event.source_id}"
                cv2.putText(frame_annotated, info_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                if self._write_image(frame_annotated, full_path):
                    screenshot_full = str(full_path)

            if event.crop is not None:
                crop_filename = f"cross_{event.event_id:03d}_ID{event.track_id}{source_suffix}_{time_for_filename}_crop.jpg"
                crop_path = self.screenshot_crop_dir / crop_filename
                crop_annotated = event.crop.copy()
                label = f"BIB:{bib_text} S:{event.source_id}"
                cv2.putText(crop_annotated, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                if self._write_image(crop_annotated, crop_path):
                    screenshot_crop = str(crop_path)

            if event.bib_crop is not None:
                bib_filename = f"bib_{event.event_id:03d}_ID{event.track_id}{source_suffix}_{time_for_filename}.jpg"
                bib_path = self.screenshot_bib_dir / bib_filename
                if self._write_image(event.bib_crop, bib_path):
                    screenshot_bib = str(bib_path)

            # --- 3. 归并或插入数据库 ---
            if match_event:
                target_event_id = match_event['event_id']
                logger.info(f"[EventRecorder] 发现匹配事件 ID:{target_event_id} (号码:{bib_text})，进行合并归并")
                
                updates = {}
                old_bib = match_event.get('bib_number')
                
                # A. 号码补完: 如果原记录是 UNKNOWN，当前有号码
                if (not old_bib or old_bib.upper() == "UNKNOWN") and (bib_text and bib_text.upper() != "UNKNOWN"):
                    updates['bib_number'] = bib_text
                    updates['bib_status'] = bib_status_str
                    updates['bib_confidence'] = event_bib_conf
                    logger.info(f"[EventRecorder] 归并更新: UNKNOWN -> {bib_text}")
                
                # B. 置信度提升: 如果当前置信度更高 (且号码相同)
                elif old_bib == bib_text and event_bib_conf is not None:
                    try:
                        old_conf = float(match_event.get('bib_confidence')) if match_event.get('bib_confidence') is not None else 0.0
                    except (TypeError, ValueError):
                        old_conf = 0.0
                    if event_bib_conf > old_conf:
                        updates['bib_confidence'] = event_bib_conf
                
                if screenshot_full and not match_event.get('screenshot_full'):
                    updates['screenshot_full'] = screenshot_full
                if screenshot_clean and not match_event.get('screenshot_clean'):
                    updates['screenshot_clean'] = screenshot_clean
                if screenshot_crop and not match_event.get('screenshot_crop'):
                    updates['screenshot_crop'] = screenshot_crop
                if screenshot_bib and not match_event.get('screenshot_bib'):
                    updates['screenshot_bib'] = screenshot_bib
                
                # 仅在目标事件尚无 evidence_dir 时写入，避免归并后覆盖已有证据目录
                if evidence_dir and not match_event.get('evidence_dir'):
                    updates['evidence_dir'] = evidence_dir

                if updates:
                    # 如果有去重说明，追加到备注中
                    if dedup_note:
                        old_notes = match_event.get('notes') or ""
                        if dedup_note not in old_notes:
                            updates['notes'] = f"{old_notes} {dedup_note}".strip()
                    self.database.update_event(target_event_id, updates)
            else:
                # 无匹配，创建新记录
                with self._lock:
                    if not is_unknown: self._processed_bibs[bib] = current_time
                    else: self._last_unknown_time = current_time

                logger.info(f"[EventRecorder] 创建新过线记录: ID={event.track_id}, Bib={event.bib_number}, EventID={event.event_id}")
                is_test = 1 if self.database.get_is_test_mode() else 0
                
                # 计算成绩
                finish_time = None
                if not is_test and event.bib_number:
                    athlete = self.database.get_athlete_by_bib(event.bib_number)
                    if athlete and athlete.get('category'):
                        finish_time = self.database.calculate_finish_time(event.cross_time, athlete['category'])
                
                db_data = {
                    'event_id': event.event_id,
                    'rank': event.rank,
                    'track_id': event.track_id,
                    'participant_id': event.participant_id,
                    'raw_track_ids': list(event.raw_track_ids or ()),
                    'sport_profile': event.sport_profile,
                    'passage_index': event.passage_index,
                    'cross_time': event.cross_time,
                    'cross_time_str': event.cross_time_str,
                    'cross_realtime': event.cross_realtime,
                    'bib_number': event.bib_number,
                    'original_bib': event.bib_number,
                    'bib_confidence': event_bib_conf,
                    'bib_status': bib_status_str,
                    'detection_confidence': event.detection_confidence,
                    'source_id': event.source_id,
                    'position_x': event.position[0],
                    'position_y': event.position[1],
                    'bbox': list(event.bbox),
                    'screenshot_full': screenshot_full,
                    'screenshot_clean': screenshot_clean,
                    'screenshot_crop': screenshot_crop,
                    'screenshot_bib': screenshot_bib,
                    'is_test': is_test,
                    'finish_time': finish_time,
                    'notes': dedup_note, # 保存去重说明
                    'evidence_dir': evidence_dir, # 新增
                    'ocr_state': 'PENDING' # 新增
                }
                self.database.insert_event(db_data, enable_dedup=False)
                logger.info(f"[EventRecorder] 数据库插入成功: EventID={event.event_id}")
                target_event_id = event.event_id
                self._saved_count += 1

                # 【新增】将位置添加到去重缓存
                with self._lock:
                    self._recent_positions.append((current_time, event.source_id, event.position[0], event.position[1], list(event.bbox), target_event_id))

            # --- 4. 统一插入证据记录 ---
            evidence_data = {
                'event_id': target_event_id,
                'source_id': event.source_id,
                'screenshot_full': screenshot_full,
                'screenshot_clean': screenshot_clean,
                'screenshot_crop': screenshot_crop,
                'screenshot_bib': screenshot_bib,
                'confidence': event_bib_conf
            }
            if self.source_count <= 1:
                self.database.upsert_best_evidence(target_event_id, evidence_data)
            else:
                self.database.insert_evidence(evidence_data)

            # --- 5. 触发高清抓拍 (ISAPI) ---
            if event.source_id in self.camera_configs:
                try:
                    self._snapshot_queue.put({
                        'event_id': target_event_id,
                        'source_id': event.source_id,
                        'time_str': time_for_filename
                    }, timeout=0.05)
                except queue.Full:
                    logger.warning(f"[EventRecorder] 抓拍队列已满，跳过本次抓拍: event_id={target_event_id}, source_id={event.source_id}")

            # --- 6. 内存清理 ---
            for attr in ("frame", "crop", "bib_crop"):
                if hasattr(event, attr):
                    setattr(event, attr, None)

            # --- 7. 触发回调 ---
            if self._on_event_saved:
                try:
                    self._on_event_saved(target_event_id)
                except Exception as e:
                    logger.error(f"[EventRecorder] 回调失败: {e}")

        except Exception as e:
            logger.exception(f"[EventRecorder] 保存事件异常: {e}")

    def _update_event_images(self, event: CrossingEvent):
        """仅更新现有事件的图片 (补录逻辑)"""
        try:
            event_id = event.event_id
            source_id = event.source_id
            
            # 获取原记录以构造文件名
            old_event = self.database.get_event(event_id)
            if not old_event:
                return

            if event.track_id is not None and event.track_id >= 0:
                old_tid = old_event.get('track_id')
                if old_tid is not None and int(old_tid) != int(event.track_id):
                    logger.warning(f"[EventRecorder] 补图忽略：track_id不匹配 event_id={event_id}, old_tid={old_tid}, new_tid={event.track_id}")
                    return
            
            # 使用原记录的时间戳构造文件名，确保覆盖旧文件或保持命名一致
            time_str = old_event.get('cross_time_str', 'unknown').replace(":", "-").replace(".", "-")
            source_suffix = f"_S{source_id}"
            
            updates = {}
            evidence_data = {
                'event_id': event_id,
                'source_id': source_id,
                'confidence': event.bib_confidence if event.bib_confidence is not None else 1.0
            }
            
            # 1. 保存/覆盖全图和清洗图
            if event.frame is not None:
                clean_filename = f"cross_{event_id:03d}_ID{old_event.get('track_id', 0)}{source_suffix}_{time_str}_clean.jpg"
                clean_path = self.screenshot_clean_dir / clean_filename
                cv2.imwrite(str(clean_path), event.frame)
                evidence_data['screenshot_clean'] = str(clean_path)
                
                # 如果是主机位且没有全图，也更新全图 (这里简化逻辑，通常补录只更新识别用的图)
                full_filename = f"cross_{event_id:03d}_ID{old_event.get('track_id', 0)}{source_suffix}_{time_str}_full.jpg"
                full_path = self.screenshot_full_dir / full_filename
                # 绘制简单的标注
                frame_annotated = event.frame.copy()
                cv2.putText(frame_annotated, f"RE-RECORD S{source_id}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imwrite(str(full_path), frame_annotated)
                evidence_data['screenshot_full'] = str(full_path)

            # 2. 保存/覆盖运动员裁剪图
            if event.crop is not None:
                crop_filename = f"cross_{event_id:03d}_ID{old_event.get('track_id', 0)}{source_suffix}_{time_str}_crop.jpg"
                crop_path = self.screenshot_crop_dir / crop_filename
                cv2.imwrite(str(crop_path), event.crop)
                evidence_data['screenshot_crop'] = str(crop_path)

            # 3. 保存/覆盖号码牌裁剪图
            if event.bib_crop is not None:
                bib_filename = f"bib_{event_id:03d}_ID{old_event.get('track_id', 0)}{source_suffix}_{time_str}.jpg"
                bib_path = self.screenshot_bib_dir / bib_filename
                cv2.imwrite(str(bib_path), event.bib_crop)
                evidence_data['screenshot_bib'] = str(bib_path)

            # 4. 更新证据库 (如果是单机位，upsert_best_evidence 会处理替换逻辑)
            if self.source_count <= 1:
                self.database.upsert_best_evidence(event_id, evidence_data)
            else:
                self.database.insert_evidence(evidence_data)
            
            # 5. 同时更新磁盘上的证据文件夹 (用于批量 OCR)
            self._save_evidence_to_disk(event)
                
            logger.info(f"[EventRecorder] 已完成 ID {event_id} 的图片补录更新")
            
        except Exception as e:
            logger.error(f"[EventRecorder] 图片补录更新失败: {e}")

    def _update_event_info(self, event: CrossingEvent):
        """执行数据库信息更新 (在后台线程执行)"""
        try:
            old_event = self.database.get_event(event.event_id)
            if old_event and event.track_id is not None and event.track_id >= 0:
                old_tid = old_event.get('track_id')
                if old_tid is not None and int(old_tid) != int(event.track_id):
                    logger.warning(f"[EventRecorder] 补录忽略：track_id不匹配 event_id={event.event_id}, old_tid={old_tid}, new_tid={event.track_id}")
                    return
            status_str = event.bib_status.value if hasattr(event.bib_status, 'value') else str(event.bib_status)
            updates = {
                'bib_number': event.bib_number,
                'bib_confidence': event.bib_confidence,
                'bib_status': status_str,
                'is_ai_correction': 1
            }
            if self.database.update_event(event.event_id, updates):
                logger.info(f"[EventRecorder] 异步更新号码成功: EventID {event.event_id} -> {event.bib_number}")
                
                # 触发保存完成回调，以便 UI 刷新列表
                if self._on_event_saved:
                    self._on_event_saved(event.event_id)
            else:
                logger.warning(f"[EventRecorder] 异步更新号码失败 (记录可能不存在): EventID {event.event_id}")
        except Exception as e:
            logger.error(f"[EventRecorder] 异步更新号码异常: {e}")

    def save_training_sample(self, event_id: int, bib_number: str):
        """
        当用户手动修正号码时，将该样本保存到训练集候选区
        保存两种格式：
        1. 裁剪后的号码牌 (用于识别模型训练)
        2. 完整画面及 YOLO 格式标签 (用于检测模型训练)
        """
        try:
            event = self.database.get_event(event_id)
            if not event:
                return

            # 1. 保存号码牌裁剪图 (识别模型用)
            if event.get('screenshot_bib'):
                bib_path = Path(event['screenshot_bib'])
                if bib_path.exists():
                    train_dir = self.output_dir / "training_samples" / "recognition"
                    train_dir.mkdir(parents=True, exist_ok=True)
                    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    train_filename = f"{bib_number}_{event_id}_{time_str}.jpg"
                    train_path = train_dir / train_filename
                    img = cv2.imread(str(bib_path))
                    if img is not None:
                        cv2.imwrite(str(train_path), img)
                        logger.info(f"[EventRecorder] 已保存识别训练样本: {train_path}")

            # 2. 保存完整画面及 YOLO 标签 (检测模型用)
            # 优先使用无标注的原图 (screenshot_clean)
            screenshot_path = event.get('screenshot_clean') or event.get('screenshot_full')
            if screenshot_path and event.get('bbox'):
                full_path = Path(screenshot_path)
                if full_path.exists():
                    detect_dir = self.output_dir / "training_samples" / "detection"
                    img_dir = detect_dir / "images"
                    lbl_dir = detect_dir / "labels"
                    img_dir.mkdir(parents=True, exist_ok=True)
                    lbl_dir.mkdir(parents=True, exist_ok=True)

                    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    base_name = f"detect_{event_id}_{time_str}"
                    
                    # 拷贝原图 (不带标注)
                    # 注意：screenshot_full 存储的是带标注的图，我们需要原始图
                    # 如果 Detector 能够提供原始帧，那就最好了。
                    # 暂时先用这张图，或者在保存 event 时也存一份原图。
                    
                    # 为了训练效果，我们最好保存原始未标注的图。
                    # 我们需要修改 CrossingEvent 结构或在 _save_event 时保存原图。
                    
                    # 简化处理：从 event 记录中读取 bbox，生成 YOLO 格式
                    # YOLO 格式: class_id x_center y_center width height (归一化)
                    try:
                        import json
                        bbox = json.loads(event['bbox']) # [x1, y1, x2, y2]
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            # 假设我们要训练的检测模型分辨率 (从数据库读取或固定)
                            # 这里需要原图的分辨率
                            full_img = cv2.imread(str(full_path))
                            if full_img is not None:
                                h, w = full_img.shape[:2]
                                dw = 1.0 / w
                                dh = 1.0 / h
                                x_center = (x1 + x2) / 2.0 * dw
                                y_center = (y1 + y2) / 2.0 * dh
                                width = (x2 - x1) * dw
                                height = (y2 - y1) * dh
                                
                                # 写入标签文件 (假设 class_id 为 0，代表骑手/号码牌)
                                label_path = lbl_dir / f"{base_name}.txt"
                                with open(label_path, "w") as f:
                                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                                
                                # 保存图片
                                cv2.imwrite(str(img_dir / f"{base_name}.jpg"), full_img)
                                logger.info(f"[EventRecorder] 已保存检测训练样本: {img_dir / f'{base_name}.jpg'}")
                    except Exception as e:
                        logger.error(f"[EventRecorder] 生成 YOLO 标签失败: {e}")

        except Exception as e:
            logger.error(f"[EventRecorder] 保存训练样本失败: {e}")

    @property
    def saved_count(self) -> int:
        """已保存事件数"""
        return self._saved_count

    @property
    def pending_count(self) -> int:
        """待处理队列长度"""
        return self._queue.qsize()


# 测试代码
if __name__ == "__main__":
    print("=" * 50)
    print("  EventRecorder 模块")
    print("=" * 50)
    print()
    print("此模块需要配合Detector使用")
    print("请运行 realtime/main.py 进行完整测试")
