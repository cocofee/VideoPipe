"""
实时计时系统 - 主窗口

一体化界面：左边实时画面，右边过线记录列表
"""

import cv2
import numpy as np
import sys
import time
import threading
import json
import shutil
import uuid
import subprocess
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QLabel, QPushButton, QSplitter, QStatusBar, QFileDialog,
    QMessageBox, QDialog, QScrollArea, QLineEdit, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QDateTimeEdit,
    QGroupBox, QFormLayout, QMenuBar, QMenu, QAction, QActionGroup,
    QComboBox, QRadioButton, QStackedWidget, QProgressDialog,
    QPlainTextEdit, QAbstractItemView, QSpinBox, QDialogButtonBox, QInputDialog
)
import logging
from ultralytics.utils import LOGGER
# 屏蔽 Ultralytics 的冗余警告（如 source 缺失等）
LOGGER.setLevel(logging.ERROR)
import os
from collections import deque


def _resolve_git_commit(path: str):
    try:
        result = subprocess.run(
            ["git", "-C", path, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _normalize_camera_clock_management_urls(value, source_count: int) -> list[str]:
    if isinstance(value, str):
        urls = [value]
    elif isinstance(value, (list, tuple)):
        urls = list(value)
    else:
        urls = []
    normalized = [str(item or "").strip() for item in urls[:source_count]]
    normalized.extend("" for _ in range(max(0, source_count - len(normalized))))
    return normalized

# 将当前目录和项目根目录添加到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QDateTime, QPoint, QEvent, QRect, QSize, QMetaObject, Q_ARG, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QIcon, QCursor, QPainter, QPen, QColor, QPolygon

class GuiLogHandler(logging.Handler):
    """自定义日志处理器，将日志发送到 GUI 信号"""
    def __init__(self, slot):
        super().__init__()
        self.slot = slot

    def emit(self, record):
        msg = self.format(record)
        self.slot(msg)


# 尝试导入用于检测USB摄像头的库
try:
    from pygrabber.dshow_graph import FilterGraph
    HAS_PYGRABBER = True
except ImportError:
    HAS_PYGRABBER = False

from typing import Optional, Any, List, Tuple, Dict
from datetime import datetime

try:
    from .logger import logger
except ImportError:
    import logging
    logger = logging.getLogger("MainWindow")

# 支持直接运行和作为模块运行
try:
    from .frame_envelope import FrameEnvelope
    from .stream_reader import StreamReader, StreamStatus
    from .detector import Detector, CrossingEvent, LOCAL_VIDEO_EVENT_SETTLE_SECONDS, resolve_athlete_validator_model
    from .database import Database
    from .event_recorder import EventRecorder
    from .event_list_widget import EventListWidget
    from .live_event_review import LiveEventReview
    from .ocr_manager import OCRManager
    from .io_utils import read_image_unicode, write_image_unicode
    from .field_issue_log import FIELD_ISSUE_CATEGORIES, FieldIssueLog
    from .event_profile import normalize_sport_profile
    from .runtime_paths import application_dir, find_model as find_runtime_model, resolve_output_dir, resolve_runtime_path, resolve_source
    from .stream_recorder import ManualRecordingManager, RecordingError, is_rtsp_source
    from .video_playback import VideoPlaybackDialog, find_recordings
    from .passage_receiver import DEFAULT_HOST, DEFAULT_PORT, PassageEventReceiver, PassageEventStore
    from .passage_review import PassageReviewDialog, lookup_status_text
    from .racetiger_source import RaceTigerClient, RaceTigerError, RaceTigerSource, RaceTigerStatus, parse_beijing_timestamp
    from .video_supplement import VideoSupplementStore, parse_beijing_datetime
    from .video_timeline import VideoTimelineError, VideoTimelineStore
    from .camera_clock import camera_clock_check_required, check_hikvision_camera_clock
    from .external_clip_import import (
        EXTERNAL_CLOCK_SOURCE,
        ExternalClipImportCancelled,
        ExternalClipImportError,
        import_verified_external_clips,
        load_external_clip_sidecar,
        race_id_from_passage_store,
    )
except ImportError:
    import sys
    import os
    # 确保当前目录在 sys.path 中
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    from frame_envelope import FrameEnvelope
    from stream_reader import StreamReader, StreamStatus
    from detector import Detector, CrossingEvent, LOCAL_VIDEO_EVENT_SETTLE_SECONDS, resolve_athlete_validator_model
    from database import Database
    from event_recorder import EventRecorder
    from event_list_widget import EventListWidget
    from live_event_review import LiveEventReview
    from ocr_manager import OCRManager
    from io_utils import read_image_unicode, write_image_unicode
    from field_issue_log import FIELD_ISSUE_CATEGORIES, FieldIssueLog
    from event_profile import normalize_sport_profile
    from runtime_paths import application_dir, find_model as find_runtime_model, resolve_output_dir, resolve_runtime_path, resolve_source
    from stream_recorder import ManualRecordingManager, RecordingError, is_rtsp_source
    from video_playback import VideoPlaybackDialog, find_recordings
    from passage_receiver import DEFAULT_HOST, DEFAULT_PORT, PassageEventReceiver, PassageEventStore
    from passage_review import PassageReviewDialog, lookup_status_text
    from racetiger_source import RaceTigerClient, RaceTigerError, RaceTigerSource, RaceTigerStatus, parse_beijing_timestamp
    from video_supplement import VideoSupplementStore, parse_beijing_datetime
    from video_timeline import VideoTimelineError, VideoTimelineStore
    from camera_clock import camera_clock_check_required, check_hikvision_camera_clock
    from external_clip_import import (
        EXTERNAL_CLOCK_SOURCE,
        ExternalClipImportCancelled,
        ExternalClipImportError,
        import_verified_external_clips,
        load_external_clip_sidecar,
        race_id_from_passage_store,
    )


class InteractiveVideoLabel(QLabel):
    """
    可交互的视频显示标签
    支持：
    - 终点线可视化绘制（支持斜线）
    - 坐标转换
    """
    # 终点线位置变化信号 (x1, y1, x2, y2)
    line_changed = pyqtSignal(int, int, int, int)
    # ROI 区域变化信号 (点列表: [(x, y), ...])
    roi_changed = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.drawing = False
        self.show_line = False  # 是否显示终点线
        self.show_roi = False   # 是否显示 ROI 区域
        self.draw_mode = 1      # 1: Line, 2: ROI
        
        # 终点线原始图像坐标
        self.pt1 = QPoint(0, 400)
        self.pt2 = QPoint(1920, 400)
        
        # ROI 区域原始图像坐标点列表
        self.roi_points = []
        self.roi_closed = False # ROI 是否已闭合
        
        # 临时点（用于绘制移动中的线）
        self.temp_pos = None
        
        # 原始视频分辨率
        self.orig_w = 1920
        self.orig_h = 1080
        
        self.setCursor(Qt.ArrowCursor) # 默认箭头

    def set_draw_mode(self, mode: int):
        """设置绘图模式: 1-终点线, 2-ROI区域"""
        self.draw_mode = mode
        if mode > 0:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
        self.update()

    def set_show_line(self, show: bool):
        """设置是否显示/启用终点线绘制"""
        self.show_line = show
        if show:
            self.draw_mode = 1
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
        self.update()

    def set_show_roi(self, show: bool):
        """设置是否显示/启用 ROI 绘制"""
        self.show_roi = show
        if show:
            self.draw_mode = 2
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
        self.update()

    def clear_line(self):
        """清除/隐藏终点线显示"""
        self.show_line = False
        self.update()

    def set_roi_points(self, points: list):
        """设置当前的 ROI 坐标点列表"""
        if self.drawing and self.draw_mode == 2:
            return
        self.roi_points = [QPoint(p[0], p[1]) for p in points]
        # 如果从外部设置（如加载配置），如果点数足够，认为已闭合
        if len(self.roi_points) >= 3:
            self.roi_closed = True
        else:
            self.roi_closed = False
        self.update()

    def clear_roi(self):
        """清除当前 ROI"""
        self.roi_points = []
        self.roi_closed = False
        self.roi_changed.emit([])
        self.update()

    def set_original_size(self, w, h):
        """设置原始视频分辨率"""
        if w > 0 and h > 0:
            self.orig_w = w
            self.orig_h = h

    def set_line_points(self, x1, y1, x2, y2):
        """设置当前的终点线坐标"""
        if self.drawing:
            return
        self.pt1 = QPoint(x1, y1)
        self.pt2 = QPoint(x2, y2)
        # self.show_line = True # 不要在这里自动显示，由外部开关控制
        self.update()

    def _get_image_pos(self, pos: QPoint):
        """将控件坐标转换为原始图像坐标"""
        if not self.pixmap() or self.pixmap().isNull():
            return None

        # 获取当前显示区域大小
        pix_size = self.pixmap().size()
        lbl_size = self.size()

        # 计算偏移（居中显示时的左上角坐标）
        offset_x = (lbl_size.width() - pix_size.width()) // 2
        offset_y = (lbl_size.height() - pix_size.height()) // 2

        # 转换为相对于pixmap的坐标
        rel_x = pos.x() - offset_x
        rel_y = pos.y() - offset_y

        # 限制在pixmap范围内
        rel_x = max(0, min(rel_x, pix_size.width() - 1))
        rel_y = max(0, min(rel_y, pix_size.height() - 1))

        # 实时计算当前显示比例
        sw = pix_size.width() / self.orig_w
        sh = pix_size.height() / self.orig_h

        # 转换为原始坐标
        if sw > 0 and sh > 0:
            orig_x = int(rel_x / sw)
            orig_y = int(rel_y / sh)
            return QPoint(orig_x, orig_y)
        return None

    def _get_label_pos(self, p: QPoint):
        """将原始图像坐标转换为控件坐标"""
        if not self.pixmap() or self.pixmap().isNull():
            return None
            
        pix_size = self.pixmap().size()
        lbl_size = self.size()
        offset_x = (lbl_size.width() - pix_size.width()) // 2
        offset_y = (lbl_size.height() - pix_size.height()) // 2
        
        # 实时计算当前显示比例
        sw = pix_size.width() / self.orig_w
        sh = pix_size.height() / self.orig_h
        
        if sw > 0 and sh > 0:
            x = int(p.x() * sw) + offset_x
            y = int(p.y() * sh) + offset_y
            return QPoint(x, y)
        return None

    def mousePressEvent(self, event):
        if not (self.show_line or self.show_roi): # 如果没开启，不响应点击
            return
            
        pos = self._get_image_pos(event.pos())
        if not pos:
            return

        if event.button() == Qt.LeftButton:
            if self.draw_mode == 1:
                self.drawing = True
                self.pt1 = pos
                self.pt2 = pos
            elif self.draw_mode == 2:
                # 多边形 ROI：点击第一个点附近表示闭合
                if len(self.roi_points) >= 3:
                    first_p = self._get_label_pos(self.roi_points[0])
                    if first_p:
                        dist = (event.pos() - first_p).manhattanLength()
                        if dist < 20: # 20像素范围内认为点击了起点
                            self.roi_closed = True
                            self.update()
                            return
                
                # 否则添加新点
                self.roi_closed = False
                self.roi_points.append(pos)
                self.roi_changed.emit([(p.x(), p.y()) for p in self.roi_points])
            self.update()
        elif event.button() == Qt.RightButton:
            if self.draw_mode == 2:
                # 右键删除最后一个点
                if self.roi_points:
                    self.roi_points.pop()
                    self.roi_closed = False
                    self.roi_changed.emit([(p.x(), p.y()) for p in self.roi_points])
                    self.update()
                else:
                    self.clear_roi()

    def mouseDoubleClickEvent(self, event):
        """双击清除所有"""
        if self.draw_mode == 2:
            self.clear_roi()
            self.update()

    def mouseMoveEvent(self, event):
        pos = self._get_image_pos(event.pos())
        if not pos:
            return
            
        self.temp_pos = pos # 记录临时位置用于绘制预览线
        
        if self.drawing:
            if self.draw_mode == 1:
                self.pt2 = pos
                self.line_changed.emit(self.pt1.x(), self.pt1.y(), self.pt2.x(), self.pt2.y())
            self.update()
        elif self.show_roi and self.draw_mode == 2:
            self.update() # 实时更新预览线

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            pos = self._get_image_pos(event.pos())
            if pos:
                if self.draw_mode == 1:
                    self.pt2 = pos
                    self.line_changed.emit(self.pt1.x(), self.pt1.y(), self.pt2.x(), self.pt2.y())
            self.drawing = False
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.pixmap() or self.pixmap().isNull():
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 1. 绘制终点线
        if self.show_line:
            p1 = self._get_label_pos(self.pt1)
            p2 = self._get_label_pos(self.pt2)
            if p1 and p2:
                # 绘制终点线 (更现代的样式：半透明发光感)
                pen = QPen(QColor(255, 50, 50, 200))
                pen.setWidth(3)
                painter.setPen(pen)
                painter.drawLine(p1, p2)
                
                # 绘制端点小圆点
                painter.setBrush(QColor(255, 50, 50))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(p1, 4, 4)
                painter.drawEllipse(p2, 4, 4)
                
                # 绘制 "FINISH" 标签
                label_text = "FINISH"
                font = painter.font()
                font.setBold(True)
                font.setPointSize(12)
                painter.setFont(font)
                
                metrics = painter.fontMetrics()
                # 兼容不同版本的 PyQt5
                if hasattr(metrics, 'horizontalAdvance'):
                    tw = metrics.horizontalAdvance(label_text)
                else:
                    tw = metrics.width(label_text)
                th = metrics.height()
                
                padding_x, padding_y = 6, 2
                rect = QRect(0, 0, tw + padding_x * 2, th + padding_y * 2)
                
                label_pos = p1 if p1.x() < p2.x() else p2
                rect.moveBottomLeft(label_pos + QPoint(5, -5))
                
                painter.setBrush(QColor(255, 50, 50, 220))
                painter.drawRoundedRect(rect, 2, 2)
                
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(rect, Qt.AlignCenter, label_text)

        # 2. 绘制 ROI 区域 (多边形)
        if self.show_roi and self.roi_points:
            label_points = []
            for p in self.roi_points:
                lp = self._get_label_pos(p)
                if lp:
                    label_points.append(lp)
            
            if label_points:
                # 1. 绘制已确定的边
                pen = QPen(QColor(50, 255, 50, 200))
                pen.setWidth(2)
                painter.setPen(pen)
                
                poly = QPolygon(label_points)
                if len(label_points) >= 3:
                    # 已闭合形状
                    painter.setBrush(QColor(50, 255, 50, 40))
                    painter.drawPolygon(poly)
                else:
                    # 只有 1-2 个点，画线
                    painter.drawPolyline(poly)
                
                # 2. 绘制各个顶点
                painter.setBrush(QColor(50, 255, 50))
                painter.setPen(Qt.NoPen)
                for lp in label_points:
                    painter.drawEllipse(lp, 3, 3)
                
                # 3. 绘制正在绘制中的预览线 (仅在未闭合时显示)
                if self.draw_mode == 2 and self.temp_pos and not self.roi_closed:
                    curr_lp = self._get_label_pos(self.temp_pos)
                    if curr_lp:
                        dash_pen = QPen(QColor(50, 255, 50, 150))
                        dash_pen.setStyle(Qt.DashLine)
                        painter.setPen(dash_pen)
                        painter.drawLine(label_points[-1], curr_lp)
                        if len(label_points) >= 2:
                            painter.drawLine(curr_lp, label_points[0])
                
                # 3. 绘制顶点
                painter.setBrush(QColor(50, 255, 50))
                painter.setPen(Qt.NoPen)
                for lp in label_points:
                    painter.drawEllipse(lp, 4, 4)
                
                # 4. 绘制标签（仅在 ROI 形成有效多边形时显示，避免 "ROI AREA (1 pts)" 误导）
                if len(label_points) >= 3:
                    label_text = f"ROI AREA ({len(label_points)} pts)"
                    font = painter.font()
                    font.setBold(True)
                    font.setPointSize(10)
                    painter.setFont(font)
                    
                    metrics = painter.fontMetrics()
                    if hasattr(metrics, 'horizontalAdvance'):
                        tw = metrics.horizontalAdvance(label_text)
                    else:
                        tw = metrics.width(label_text)
                    th = metrics.height()
                    
                    label_rect = QRect(0, 0, tw + 12, th + 4)
                    label_rect.moveTopLeft(label_points[0] + QPoint(10, 10))
                    
                    painter.setBrush(QColor(50, 255, 50, 220))
                    painter.drawRoundedRect(label_rect, 2, 2)
                    painter.setPen(QColor(0, 0, 0))
                    painter.drawText(label_rect, Qt.AlignCenter, label_text)
        
        painter.end()


class PreviewThread(QThread):
    """Publish the newest camera frame at a stable UI refresh rate."""

    frame_ready = pyqtSignal(object, int)

    def __init__(self, reader: StreamReader, source_id: int = 0, target_fps: float = 30.0):
        super().__init__()
        self.reader = reader
        self.source_id = source_id
        self.target_fps = max(1.0, float(target_fps))
        self._running = False
        self._pending_lock = threading.Lock()
        self._frame_pending = False

    def mark_consumed(self):
        with self._pending_lock:
            self._frame_pending = False

    def _reserve_publish_slot(self) -> bool:
        with self._pending_lock:
            if self._frame_pending:
                return False
            self._frame_pending = True
            return True

    def run(self):
        self._running = True
        last_frame_ts = 0.0
        frame_interval = 1.0 / self.target_fps

        while self._running:
            cycle_started = time.monotonic()
            if not self._reserve_publish_slot():
                time.sleep(min(0.005, frame_interval))
                continue
            try:
                frame, frame_ts = self.reader.get_frame_after(last_frame_ts)
            except Exception as exc:
                self.mark_consumed()
                logger.exception(f"[PreviewThread-{self.source_id}] get_frame_after failed: {exc}")
                time.sleep(0.01)
                continue

            if frame is None:
                self.mark_consumed()
                if self.reader.status == StreamStatus.ENDED:
                    break
                time.sleep(0.001)
                continue

            last_frame_ts = frame_ts
            self.frame_ready.emit(frame, self.source_id)

            remaining = frame_interval - (time.monotonic() - cycle_started)
            if remaining > 0:
                time.sleep(remaining)

        self._running = False

    def stop(self):
        self._running = False
        self.mark_consumed()
        self.wait(2000)


class VideoThread(QThread):
    """视频处理线程"""
    # frame, athletes, bibs, source_id
    frame_ready = pyqtSignal(object, list, list, int)
    detections_ready = pyqtSignal(list, list, int)
    event_detected = pyqtSignal(object)  # CrossingEvent
    bib_updated = pyqtSignal(int, str, float, object, int, object, object, object)  # track_id, bib, conf, status, source_id, bib_crop, athlete_crop, full_frame
    status_changed = pyqtSignal(object, int)  # StreamStatus, source_id
    roi_auto_disabled = pyqtSignal(int)  # source_id

    def __init__(self, reader: StreamReader, detector: Detector, source_id: int = 0, ui_skip: int = 2):
        super().__init__()
        self.reader = reader
        self.detector = detector
        self.source_id = source_id
        self.ui_skip = ui_skip  # UI刷新间隔（不影响检测，每帧都检测）
        self._running = False
        self._ui_pending_lock = threading.Lock()
        self._ui_frame_pending = False
        self.last_processed_envelope: Optional[FrameEnvelope] = None
        self.last_frame_metrics: Dict[str, Any] = {}
        
        # 绑定流状态回调
        self.reader.set_on_status_change(self._on_stream_status_change)
        # 绑定号码更新回调
        self.detector.set_on_bib_update(self._on_bib_update)

    def _on_stream_status_change(self, status):
        """转发流状态信号到主线程"""
        self.status_changed.emit(status, self.source_id)

    def _on_bib_update(self, track_id, bib, conf, status, source_id=None, bib_crop=None, athlete_crop=None, full_frame=None):
        """转发号码更新信号到主线程"""
        if not self._running: return
        sid = source_id if source_id is not None else self.source_id
        # 使用 QMetaObject 异步调用以确保线程安全
        QMetaObject.invokeMethod(self, "emit_bib_updated", Qt.QueuedConnection,
                               Q_ARG(int, track_id), Q_ARG(str, bib), Q_ARG(float, conf),
                               Q_ARG(object, status), Q_ARG(int, sid),
                               Q_ARG(object, bib_crop), Q_ARG(object, athlete_crop), Q_ARG(object, full_frame))

    @pyqtSlot(int, str, float, object, int, object, object, object)
    def emit_bib_updated(self, tid, bib, conf, status, sid, b_crop, a_crop, frame):
        self.bib_updated.emit(tid, bib, conf, status, sid, b_crop, a_crop, frame)

    def mark_ui_consumed(self):
        with self._ui_pending_lock:
            self._ui_frame_pending = False

    def _reserve_ui_publish_slot(self) -> bool:
        with self._ui_pending_lock:
            if self._ui_frame_pending:
                return False
            self._ui_frame_pending = True
            return True

    def run(self):
        logger.info(f"[VideoThread-{self.source_id}] 视频处理线程已启动")
        self._running = True
        frame_count = 0
        last_frame_ts = 0.0
        inference_timestamps = deque(maxlen=120)

        while self._running:
            envelope = None
            try:
                # Consume the bounded queue in order for both files and live
                # sources.  Clearing all pending live frames can skip the only
                # few frames where a sprinting athlete is detectable.
                get_frame_envelope = getattr(self.reader, "get_frame_envelope", None)
                if callable(get_frame_envelope):
                    envelope = get_frame_envelope()
                    if envelope is None:
                        frame, frame_ts = None, last_frame_ts
                    else:
                        frame = envelope.original_frame
                        frame_ts = envelope.capture_time_ms / 1000.0
                else:
                    frame, frame_ts = self.reader.get_frame_after(last_frame_ts)
            except Exception as e:
                logger.exception(f"[VideoThread-{self.source_id}] 获取新帧异常: {e}")
                time.sleep(0.01)
                continue

            if frame is None:
                if self.reader.status == StreamStatus.ENDED:
                    logger.info(f"[VideoThread-{self.source_id}] 本地视频处理完成")
                    break
                time.sleep(0.001)
                continue

            if envelope is None and frame_ts > 0:
                last_frame_ts = frame_ts

            frame_count += 1
            processing_started = time.monotonic()
            processing_started_wall_ms = time.time() * 1000.0

            # 每帧都检测（保持ByteTrack跟踪连续性）
            try:
                if envelope is None:
                    events, athletes, bibs = self.detector.process_frame(frame)
                else:
                    try:
                        events, athletes, bibs = self.detector.process_frame(
                            frame, timestamp=envelope.capture_time_ms / 1000.0
                        )
                    except TypeError as error:
                        if "unexpected keyword argument 'timestamp'" not in str(error):
                            raise
                        events, athletes, bibs = self.detector.process_frame(frame)
            except Exception as e:
                logger.exception(f"[VideoThread-{self.source_id}] process_frame 异常: {e}")
                events, athletes, bibs = [], [], []
            detector_metrics = dict(getattr(self.detector, "last_frame_metrics", {}) or {})
            processing_completed = time.monotonic()
            processing_time_ms = (processing_completed - processing_started) * 1000.0
            inference_timestamps.append(processing_completed)
            if len(inference_timestamps) >= 2:
                inference_window_seconds = inference_timestamps[-1] - inference_timestamps[0]
                inference_fps = (
                    (len(inference_timestamps) - 1) / inference_window_seconds
                    if inference_window_seconds > 0
                    else 0.0
                )
            else:
                inference_fps = 1000.0 / max(processing_time_ms, 0.001)
            try:
                reader_metrics = dict(self.reader.get_info() or {})
            except Exception:
                reader_metrics = {}
            if detector_metrics.get("roi_auto_disabled"):
                self.roi_auto_disabled.emit(self.source_id)
            if envelope is not None:
                envelope = envelope.with_processing_time(processing_time_ms)
                self.last_processed_envelope = envelope
                capture_latency_ms = envelope.arrival_time_ms - envelope.capture_time_ms
                if capture_latency_ms < 0 or capture_latency_ms > 60_000:
                    capture_latency_ms = None
                queue_latency_ms = max(0.0, processing_started_wall_ms - envelope.arrival_time_ms)
                self.last_frame_metrics = {
                    **detector_metrics,
                    "frame_index": envelope.frame_index,
                    "segment_id": envelope.segment_id,
                    "capture_time_ms": envelope.capture_time_ms,
                    "arrival_time_ms": envelope.arrival_time_ms,
                    "capture_latency_ms": capture_latency_ms,
                    "queue_latency_ms": queue_latency_ms,
                    "processing_time_ms": processing_time_ms,
                    "end_to_end_latency_ms": queue_latency_ms + processing_time_ms,
                    "queue_depth": reader_metrics.get("queue_depth"),
                    "capture_fps": float(reader_metrics.get("actual_fps", 0.0) or 0.0),
                    "inference_fps": inference_fps,
                    "dropped_frames": reader_metrics.get("dropped_frame_count"),
                    "consumer_skipped_frames": reader_metrics.get("consumer_skipped_frame_count"),
                    "discarded_frames": reader_metrics.get("discarded_frame_count"),
                    "drop_rate": float(reader_metrics.get("drop_rate", 0.0) or 0.0),
                }
            else:
                self.last_frame_metrics = {
                    **detector_metrics,
                    "processing_time_ms": processing_time_ms,
                    "queue_depth": reader_metrics.get("queue_depth"),
                    "capture_fps": float(reader_metrics.get("actual_fps", 0.0) or 0.0),
                    "inference_fps": inference_fps,
                    "dropped_frames": reader_metrics.get("dropped_frame_count"),
                    "consumer_skipped_frames": reader_metrics.get("consumer_skipped_frame_count"),
                    "discarded_frames": reader_metrics.get("discarded_frame_count"),
                    "drop_rate": float(reader_metrics.get("drop_rate", 0.0) or 0.0),
                }

            # 发送过线事件
            for event in events:
                # 已经在 detector 内部填充了 source_id
                logger.debug(f"[VideoThread-{self.source_id}] 发送过线信号: ID={event.track_id}, Bib={event.bib_number}")
                self.event_detected.emit(event)

            # UI刷新：降低刷新频率减轻GUI负担
            self.detections_ready.emit(athletes, bibs, self.source_id)
            if frame_count % self.ui_skip == 0 and self._reserve_ui_publish_slot():
                self.frame_ready.emit(frame, athletes, bibs, self.source_id)

        self._running = False

    def stop(self):
        self._running = False
        self.mark_ui_consumed()
        self.wait(2000)


class ImageViewerDialog(QDialog):
    """图片查看对话框 - 支持导航、缩放和实时修改"""

    # 信号：号码已修改 (event_id, new_bib)
    bib_changed = pyqtSignal(int, str)
    # 信号：作废状态变化 (event_id, is_void)
    void_changed = pyqtSignal(int, bool)
    # 信号：当前查看的事件已改变 (event_id)
    event_changed = pyqtSignal(int)

    def __init__(self, current_index: int, all_events: list, parent=None, database=None):
        super().__init__(parent)
        self.setWindowTitle("截图复核工作台")
        self.setMinimumSize(1200, 800)
        # 设置为非模态但始终保持在最前，方便与主界面联动
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        
        self.all_events = all_events
        self.current_index = current_index
        self.current_event = None
        self.database = database
        
        self.original_pixmap = None
        self.scale_factor = 1.0
        self.min_scale = 0.1
        self.max_scale = 10.0
        self.target_bbox = None
        
        # 拖拽相关
        self.dragging = False
        self.last_pos = None
        self.current_evidences = []
        self.evidence_index = 0

        self._init_ui()
        self._load_event(current_index)

    def _init_ui(self):
        """初始化主界面布局"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # 1. 顶部状态指示栏 (转播风格)
        top_status_bar = QWidget()
        top_status_bar.setFixedHeight(50)
        top_status_bar.setStyleSheet("background-color: #1a1a1a; border-radius: 4px;")
        top_status_layout = QHBoxLayout(top_status_bar)
        
        # 实时会话统计
        self.rank_label = QLabel("RANK --")
        self.rank_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #ffcc00; margin-left: 10px;")
        top_status_layout.addWidget(self.rank_label)
        
        top_status_layout.addSpacing(30)
        
        self.time_label = QLabel("TIME --:--:--.---")
        self.time_label.setStyleSheet("font-size: 20px; font-family: Consolas; color: white;")
        top_status_layout.addWidget(self.time_label)
        
        top_status_layout.addSpacing(30)
        
        self.index_label = QLabel("ID --")
        self.index_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #00ccff;")
        top_status_layout.addWidget(self.index_label)
        
        top_status_layout.addStretch()
        
        main_layout.addWidget(top_status_bar)

        # 2. 中间区域：左侧大图 + 右侧证据缩略图
        content_layout = QHBoxLayout()
        
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(False)
        self.scroll.setAlignment(Qt.AlignCenter)
        self.scroll.viewport().setCursor(Qt.OpenHandCursor)
        self.scroll.setStyleSheet("background-color: #111; border: none;")
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.scroll.setWidget(self.image_label)
        
        content_layout.addWidget(self.scroll, 5) # 占 5 份

        # 证据侧边栏
        self.evidence_panel = QWidget()
        self.evidence_panel.setFixedWidth(200)
        self.evidence_panel.setStyleSheet("background-color: #1a1a1a; border-left: 1px solid #444;")
        self.evidence_layout = QVBoxLayout(self.evidence_panel)
        self.evidence_layout.setContentsMargins(5, 5, 5, 5)
        
        evidence_title = QLabel("多机位证据链")
        evidence_title.setStyleSheet("color: #aaa; font-weight: bold; font-size: 14px; margin-bottom: 5px;")
        evidence_title.setAlignment(Qt.AlignCenter)
        self.evidence_layout.addWidget(evidence_title)
        
        self.evidence_scroll = QScrollArea()
        self.evidence_scroll.setWidgetResizable(True)
        self.evidence_scroll.setStyleSheet("border: none; background: transparent;")
        self.evidence_container = QWidget()
        self.evidence_list_layout = QVBoxLayout(self.evidence_container)
        self.evidence_list_layout.setSpacing(10)
        self.evidence_list_layout.addStretch()
        self.evidence_scroll.setWidget(self.evidence_container)
        
        self.evidence_layout.addWidget(self.evidence_scroll)
        
        content_layout.addWidget(self.evidence_panel, 1) # 占 1 份
        
        main_layout.addLayout(content_layout)

        # 3. 底部控制栏
        control_bar = QWidget()
        control_bar.setFixedHeight(65)
        control_bar.setStyleSheet("background-color: #262626; border-top: 1px solid #444; color: white;")
        bottom_layout = QHBoxLayout(control_bar)
        bottom_layout.setContentsMargins(10, 5, 10, 5)
        
        # 公共按钮样式
        btn_style = """
            QPushButton {
                background-color: #444;
                color: #ffffff;
                border: 1px solid #666;
                border-radius: 4px;
                font-weight: bold;
                font-size: 16px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #555;
            }
            QPushButton:pressed {
                background-color: #333;
            }
            QPushButton:disabled {
                background-color: #222;
                color: #555;
            }
        """

        # 导航按钮
        self.prev_btn = QPushButton("上一条 (↑)")
        self.prev_btn.setFixedSize(150, 50)
        self.prev_btn.setStyleSheet(btn_style)
        self.prev_btn.setToolTip("快捷键: 方向键上/左")
        self.prev_btn.clicked.connect(self._prev_event)
        bottom_layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("下一条 (↓)")
        self.next_btn.setFixedSize(150, 50)
        self.next_btn.setStyleSheet(btn_style)
        self.next_btn.setToolTip("快捷键: 方向键下/右")
        self.next_btn.clicked.connect(self._next_event)
        bottom_layout.addWidget(self.next_btn)
        
        bottom_layout.addSpacing(20)
        
        # 修改区域
        label_bib = QLabel("号码:")
        label_bib.setStyleSheet("color: white; font-weight: bold; font-size: 16px;")
        bottom_layout.addWidget(label_bib)
        
        self.bib_input = QLineEdit()
        self.bib_input.setPlaceholderText("号码")
        self.bib_input.setFixedWidth(110)
        self.bib_input.setFixedHeight(50)
        self.bib_input.setStyleSheet("font-size: 16px; font-weight: bold; padding: 2px; color: #ffcc00; background-color: #000; border: 1px solid #ffcc00;")
        bottom_layout.addWidget(self.bib_input)
        
        bottom_layout.addSpacing(5)

        self.save_btn = QPushButton("保存 (Enter)")
        self.save_btn.setFixedSize(150, 50)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #006644;
                color: white;
                font-weight: bold;
                font-size: 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #008855;
            }
        """)
        self.save_btn.clicked.connect(self._save_bib)
        bottom_layout.addWidget(self.save_btn)
        
        bottom_layout.addSpacing(10)
        
        self.void_btn = QPushButton("作废 (V)")
        self.void_btn.setFixedSize(140, 50)
        self.void_btn.setToolTip("快捷键: V")
        self.void_btn.clicked.connect(self._toggle_void)
        bottom_layout.addWidget(self.void_btn)
        
        bottom_layout.addStretch()
        
        # 缩放控制
        self.zoom_label = QLabel("100%")
        self.zoom_label.setStyleSheet("color: #aaa; font-size: 16px; margin-right: 5px;")
        bottom_layout.addWidget(self.zoom_label)
        
        reset_btn = QPushButton("对焦 (F)")
        reset_btn.setFixedSize(130, 50)
        reset_btn.setStyleSheet(btn_style)
        reset_btn.setToolTip("快捷键: F")
        reset_btn.clicked.connect(self._focus_on_target)
        bottom_layout.addWidget(reset_btn)
        
        main_layout.addWidget(control_bar)

        # 4. 悬浮反馈标签 (用于显示"已保存")
        self.feedback_label = QLabel(self)
        self.feedback_label.setAlignment(Qt.AlignCenter)
        self.feedback_label.setStyleSheet("""
            background-color: rgba(0, 102, 68, 200);
            color: white;
            font-size: 16px;
            font-weight: bold;
            padding: 20px;
            border-radius: 8px;
        """)
        self.feedback_label.hide()

        # 安装事件过滤器
        self.scroll.viewport().installEventFilter(self)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 居中显示反馈标签
        self.feedback_label.move(
            (self.width() - self.feedback_label.width()) // 2,
            (self.height() - self.feedback_label.height()) // 2
        )

    def _load_event(self, index):
        if not (0 <= index < len(self.all_events)):
            return
            
        self.current_index = index
        self.current_event = self.all_events[index]
        
        # 更新 UI 文本
        rank = self.current_event.get('rank', '-')
        time_str = self.current_event.get('cross_time_str', '-')
        bib = self.current_event.get('bib_number', '')
        is_void = self.current_event.get('is_void', 0)
        
        self.rank_label.setText(f"RANK {rank}")
        self.time_label.setText(f"TIME {time_str}")
        self.index_label.setText(f"{index + 1} / {len(self.all_events)}")
        
        self.bib_input.setText(bib if bib else "")
        self.void_btn.setText("取消作废" if is_void else "标记作废 (V)")
        if is_void:
            # 已作废状态：深灰色背景，白色文字
            self.void_btn.setStyleSheet("""
                QPushButton {
                    background-color: #444; 
                    color: white; 
                    border: 1px solid #666; 
                    font-weight: bold;
                    font-size: 16px;
                }
            """)
        else:
            # 正常状态：醒目的红色背景，白色文字（更容易看清）
            self.void_btn.setStyleSheet("""
                QPushButton {
                    background-color: #d32f2f; 
                    color: white; 
                    border: 1px solid #b71c1c; 
                    font-weight: bold;
                    font-size: 16px;
                }
                QPushButton:hover {
                    background-color: #e53935;
                }
            """)
        
        # 导航按钮状态
        self.prev_btn.setEnabled(index > 0)
        self.next_btn.setEnabled(index < len(self.all_events) - 1)
        
        # 加载图片 (优先使用 ISAPI 高清图)
        image_path = None
        event_id = self.current_event.get('event_id')
        self.current_evidences = []
        
        # 1. 基础证据（来自主记录）
        main_screenshot = self.current_event.get('screenshot_full') or self.current_event.get('screenshot_crop')
        if main_screenshot and Path(main_screenshot).exists():
            self.current_evidences.append({
                'path': main_screenshot,
                'source': f"机位 {self.current_event.get('source_id', 0) + 1} (主)",
                'type': 'main'
            })
        
        # 2. 额外证据（来自证据表）
        if self.database and event_id:
            db_evidences = self.database.get_event_evidences(event_id)
            for evd in db_evidences:
                path = evd.get('screenshot_full') or evd.get('screenshot_crop')
                if path and Path(path).exists():
                    # 避免重复添加主截图
                    if str(Path(path)) != str(Path(main_screenshot or "")):
                        source_id = evd.get('source_id', 0)
                        self.current_evidences.append({
                            'path': str(Path(path)),
                            'source': f"机位 {source_id + 1}",
                            'type': 'evidence',
                            'confidence': evd.get('confidence', 0)
                        })

        # 3. 证据目录回退（主截图缺失时）
        evidence_dir = self.current_event.get('evidence_dir')
        if (not evidence_dir) and event_id:
            parent = self.parent()
            if parent is not None and hasattr(parent, "output_dir"):
                candidate = Path(parent.output_dir) / "evidence_photos" / f"{int(event_id):06d}"
                if candidate.exists():
                    evidence_dir = str(candidate)

        if evidence_dir:
            evidence_root = Path(evidence_dir)
            meta_path = evidence_root / "meta.json"
            paths = {}
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    paths = meta.get("paths", {}) if isinstance(meta, dict) else {}
                except Exception:
                    paths = {}

            if paths:
                candidates = [
                    (paths.get("full"), "证据 full"),
                    (paths.get("athlete"), "证据 athlete"),
                    (paths.get("bib"), "证据 bib"),
                ]
                for rel_path, label in candidates:
                    if not rel_path:
                        continue
                    full_path = evidence_root / rel_path
                    if full_path.exists() and not any(Path(e['path']) == full_path for e in self.current_evidences):
                        self.current_evidences.append({
                            'path': str(full_path),
                            'source': label,
                            'type': 'evidence_dir'
                        })
            else:
                for key in ("full", "athlete", "bib"):
                    full_path = evidence_root / f"{key}.jpg"
                    if full_path.exists() and not any(Path(e['path']) == full_path for e in self.current_evidences):
                        self.current_evidences.append({
                            'path': str(full_path),
                            'source': f"证据 {key}",
                            'type': 'evidence_dir'
                        })

        # 更新侧边栏 UI
        self._update_evidence_list()
        
        # 单机位适配：如果只有一个证据，隐藏侧边栏
        if len(self.current_evidences) <= 1:
            self.evidence_panel.hide()
        else:
            self.evidence_panel.show()
            
        # 默认显示第一个证据
        if self.current_evidences:
            self.evidence_index = 0
            # 优先寻找 ISAPI 高清图作为默认显示
            for i, evd in enumerate(self.current_evidences):
                if "isapi_" in evd['path']:
                    self.evidence_index = i
                    break
            image_path = self.current_evidences[self.evidence_index]['path']
            
        self._load_and_mark_image(image_path)
        
        # 自动缩放聚焦
        QTimer.singleShot(50, self._focus_on_target)
        
        # 通知父窗口：当前选中的记录已变化，用于列表联动同步
        self.event_changed.emit(self.current_event.get('event_id'))

    def _update_evidence_list(self):
        """更新证据链侧边栏列表"""
        # 清理旧项
        while self.evidence_list_layout.count() > 1: # 保留最后的 stretch
            item = self.evidence_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # 添加新项
        for i, evd in enumerate(self.current_evidences):
            item_btn = QPushButton()
            item_btn.setFixedSize(180, 130)
            item_btn.setCursor(Qt.PointingHandCursor)
            
            # 设置样式（选中项高亮）
            border_color = "#1890ff" if i == self.evidence_index else "#444"
            item_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: #262626;
                    border: 2px solid {border_color};
                    border-radius: 4px;
                    color: white;
                    text-align: bottom center;
                    padding-bottom: 5px;
                }}
                QPushButton:hover {{
                    border-color: #40a9ff;
                }}
            """)
            
            # 缩略图预览
            pixmap = QPixmap(evd['path'])
            if not pixmap.isNull():
                icon_pixmap = pixmap.scaled(170, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                item_btn.setIcon(QIcon(icon_pixmap))
                item_btn.setIconSize(QSize(170, 100))
            
            item_btn.setText(evd['source'])
            item_btn.clicked.connect(lambda checked, idx=i: self._switch_evidence(idx))
            
            self.evidence_list_layout.insertWidget(self.evidence_list_layout.count() - 1, item_btn)

    def _switch_evidence(self, index):
        """切换显示的证据图片"""
        if 0 <= index < len(self.current_evidences):
            self.evidence_index = index
            self._update_evidence_list() # 更新高亮
            self._load_and_mark_image(self.current_evidences[index]['path'])
            # 切换后重新聚焦
            QTimer.singleShot(50, self._focus_on_target)

    def _load_and_mark_image(self, image_path: str):
        """加载图片并绘制专业的转播风格标注"""
        if not image_path or not Path(image_path).exists():
            self.original_pixmap = QPixmap()
            self.image_label.setText("截图文件不存在")
            return

        import json
        img = cv2.imread(image_path)
        if img is None:
            img = read_image_unicode(Path(image_path))
        if img is not None:
            # 1. 绘制目标框和 ID 标签
            bbox_data = self.current_event.get('bbox', '[]')
            if isinstance(bbox_data, str):
                try: bbox = json.loads(bbox_data)
                except: bbox = []
            else: bbox = bbox_data

            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                self.target_bbox = (x1, y1, x2, y2)
                
                # 绘制半透明蒙版（除了目标区域）- 可选，这里先只画框
                # 绘制专业红框
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                
                # 绘制 ID 盒子 (类似转播 ID)
                track_id = self.current_event.get('track_id', '?')
                id_label = f"ID:{track_id}"
                (tw, th), _ = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(img, (x1, y1-th-15), (x1+tw+10, y1), (0, 0, 255), -1)
                cv2.putText(img, id_label, (x1+5, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 2. 转换并显示
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.original_pixmap = QPixmap.fromImage(qt_image)
        else:
            self.original_pixmap = QPixmap(image_path)

        self._update_display()

    def _focus_on_target(self):
        """自动聚焦缩放"""
        if self.target_bbox is None or self.original_pixmap is None or self.original_pixmap.isNull():
            self._fit_window()
            return

        x1, y1, x2, y2 = self.target_bbox
        target_cx, target_cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        viewport = self.scroll.viewport()
        vw, vh = viewport.width(), viewport.height()

        # 理想缩放：让目标物体在屏幕中占据合适大小
        target_w = x2 - x1
        target_h = y2 - y1
        scale_w = (vw * 0.4) / target_w if target_w > 0 else 1.0
        scale_h = (vh * 0.4) / target_h if target_h > 0 else 1.0
        
        self.scale_factor = min(scale_w, scale_h)
        self.scale_factor = max(0.5, min(2.0, self.scale_factor)) # 限制在 0.5 到 2.0 之间
        
        self._update_display()

        # 居中
        scaled_cx = int(target_cx * self.scale_factor)
        scaled_cy = int(target_cy * self.scale_factor)
        self.scroll.horizontalScrollBar().setValue(max(0, scaled_cx - vw // 2))
        self.scroll.verticalScrollBar().setValue(max(0, scaled_cy - vh // 2))

    def _prev_event(self):
        if self.current_index > 0:
            self._load_event(self.current_index - 1)

    def _next_event(self):
        if self.current_index < len(self.all_events) - 1:
            self._load_event(self.current_index + 1)

    def _save_bib(self):
        new_bib = self.bib_input.text().strip()
        event_id = self.current_event.get('event_id')
        if event_id:
            self.bib_changed.emit(event_id, new_bib)
            self.current_event['bib_number'] = new_bib
            
            # 显示视觉反馈
            self.feedback_label.setText(f"号码 {new_bib} 已保存")
            self.feedback_label.adjustSize()
            self.feedback_label.move(
                (self.width() - self.feedback_label.width()) // 2,
                (self.height() - self.feedback_label.height()) // 2
            )
            self.feedback_label.show()
            QTimer.singleShot(1000, self.feedback_label.hide)
            
            # 按钮提示
            self.save_btn.setText("已保存!")
            QTimer.singleShot(1000, lambda: self.save_btn.setText("保存 (Enter)"))
            
            # 保存后焦点离开输入框，方便方向键导航
            self.save_btn.setFocus()

    def _toggle_void(self):
        event_id = self.current_event.get('event_id')
        current_void = self.current_event.get('is_void', 0)
        new_void = not current_void
        if event_id:
            self.void_changed.emit(event_id, new_void)
            self.current_event['is_void'] = 1 if new_void else 0
            self.void_btn.setText("取消作废" if new_void else "标记作废 (V)")
            if new_void:
                self.void_btn.setStyleSheet("background-color: #444; color: white; border: 1px solid #666; font-weight: bold;")
            else:
                self.void_btn.setStyleSheet("background-color: #ffcccc; color: #b00; border: 1px solid #f88; font-weight: bold;")
            
            # 显示视觉反馈
            status_text = "已作废" if new_void else "已恢复"
            self.feedback_label.setText(status_text)
            self.feedback_label.adjustSize()
            self.feedback_label.show()
            QTimer.singleShot(1000, self.feedback_label.hide)

    def keyPressEvent(self, event):
        # 1. 无论焦点在哪里都支持的快捷键
        if event.key() == Qt.Key_Escape:
            self.close()
            return

        # 2. 如果输入框正在输入，限制部分快捷键以免干扰
        has_focus = self.bib_input.hasFocus()
        
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self._save_bib()
            return
            
        if event.key() == Qt.Key_Up:
            self._prev_event()
            return
        elif event.key() == Qt.Key_Down:
            self._next_event()
            return
            
        # 左右键：如果输入框没焦点，则导航；有焦点则正常移动光标
        if not has_focus:
            if event.key() == Qt.Key_Left:
                self._prev_event()
                return
            elif event.key() == Qt.Key_Right:
                self._next_event()
                return
            elif event.key() == Qt.Key_V:
                self._toggle_void()
                return
            elif event.key() == Qt.Key_F:
                self._focus_on_target()
                return

        super().keyPressEvent(event)

    def _update_display(self):
        if self.original_pixmap is None or self.original_pixmap.isNull():
            return
        display_width = int(self.original_pixmap.width() * self.scale_factor)
        display_height = int(self.original_pixmap.height() * self.scale_factor)
        scaled_pixmap = self.original_pixmap.scaled(
            display_width, display_height,
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.resize(scaled_pixmap.size())
        self.zoom_label.setText(f"{int(self.scale_factor * 100)}%")

    def _fit_window(self):
        if self.original_pixmap is None or self.original_pixmap.isNull():
            return
        viewport = self.scroll.viewport()
        vw, vh = viewport.width() - 20, viewport.height() - 20
        iw, ih = self.original_pixmap.width(), self.original_pixmap.height()
        if iw <= 0 or ih <= 0:
            return
        self.scale_factor = min(vw / iw, vh / ih)
        self._update_display()

    def eventFilter(self, obj, event):
        """事件过滤器 - 处理滚轮和拖拽 (支持鼠标中心缩放)"""
        if obj == self.scroll.viewport():
            if event.type() == QEvent.Wheel:
                # 滚轮缩放 - 以鼠标位置为中心
                factor = 1.15 if event.angleDelta().y() > 0 else 1/1.15
                self._zoom_at(event.pos(), factor)
                return True

            elif event.type() == QEvent.MouseButtonPress:
                if event.button() == Qt.LeftButton:
                    self.dragging = True
                    self.last_pos = event.pos()
                    self.scroll.viewport().setCursor(Qt.ClosedHandCursor)
                    return True
            elif event.type() == QEvent.MouseMove:
                if self.dragging:
                    delta = event.pos() - self.last_pos
                    self.last_pos = event.pos()
                    self.scroll.horizontalScrollBar().setValue(self.scroll.horizontalScrollBar().value() - delta.x())
                    self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().value() - delta.y())
                    return True
            elif event.type() == QEvent.MouseButtonRelease:
                if event.button() == Qt.LeftButton:
                    self.dragging = False
                    self.scroll.viewport().setCursor(Qt.OpenHandCursor)
                    return True
        return super().eventFilter(obj, event)

    def _zoom_in(self):
        """点击按钮放大：以视图中心为基准"""
        viewport = self.scroll.viewport()
        center = QPoint(viewport.width() // 2, viewport.height() // 2)
        self._zoom_at(center, 1.2)

    def _zoom_out(self):
        """点击按钮缩小：以视图中心为基准"""
        viewport = self.scroll.viewport()
        center = QPoint(viewport.width() // 2, viewport.height() // 2)
        self._zoom_at(center, 1/1.2)

    def _zoom_at(self, mouse_pos: QPoint, factor: float):
        """以指定点为中心进行缩放"""
        if self.original_pixmap is None: return

        h_bar = self.scroll.horizontalScrollBar()
        v_bar = self.scroll.verticalScrollBar()

        # 1. 记录缩放前的图片原始坐标
        img_x = h_bar.value() + mouse_pos.x()
        img_y = v_bar.value() + mouse_pos.y()
        orig_x = img_x / self.scale_factor
        orig_y = img_y / self.scale_factor

        # 2. 更新缩放系数
        new_scale = self.scale_factor * factor
        self.scale_factor = max(self.min_scale, min(self.max_scale, new_scale))

        # 3. 刷新显示
        self._update_display()

        # 4. 调整滚动条，使之前的原始坐标点依然对应现在的鼠标位置
        new_scroll_x = int(orig_x * self.scale_factor - mouse_pos.x())
        new_scroll_y = int(orig_y * self.scale_factor - mouse_pos.y())
        h_bar.setValue(max(0, new_scroll_x))
        v_bar.setValue(max(0, new_scroll_y))

class ConnectionTester(QThread):
    """异步测试摄像头连接"""
    finished = pyqtSignal(bool, str)

    def __init__(
        self,
        source,
        *,
        verify_hikvision_clock=False,
        management_url="",
    ):
        super().__init__()
        self.source = source
        self.verify_hikvision_clock = bool(verify_hikvision_clock)
        self.management_url = str(management_url or "").strip()

    def run(self):
        try:
            # 针对不同源进行优化测试
            if isinstance(self.source, int):
                # USB 模式
                cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            else:
                # RTSP 模式，设置较短的超时
                cap = cv2.VideoCapture(self.source)
            
            if cap.isOpened():
                # 尝试读取一帧
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000) # 3秒超时
                ret, _ = cap.read()
                cap.release()
                if ret:
                    if self.verify_hikvision_clock:
                        clock_check = check_hikvision_camera_clock(
                            self.source,
                            management_url=self.management_url,
                        )
                        if clock_check.ok:
                            self.finished.emit(
                                True,
                                f"连接成功，画面读取正常；{clock_check.message}",
                            )
                        else:
                            self.finished.emit(
                                False,
                                "画面读取正常，但相机时间核验失败："
                                f"{clock_check.message}",
                            )
                    else:
                        self.finished.emit(True, "连接成功，画面读取正常")
                else:
                    self.finished.emit(False, "设备已打开但无法读取画面（可能是编码不支持或权限问题）")
            else:
                self.finished.emit(False, "无法打开设备（请检查IP、密码或USB连接）")
        except Exception as e:
            self.finished.emit(False, f"测试发生异常: {str(e)}")


class CameraClockPreflightThread(QThread):
    """Verify all credentialed RTSP camera clocks before capture starts."""

    completed = pyqtSignal(bool, str, object)

    def __init__(self, targets, parent=None):
        super().__init__(parent)
        self.targets = tuple(targets)

    def run(self):
        signature = tuple(self.targets)
        for camera_index, source, management_url in self.targets:
            if self.isInterruptionRequested():
                self.completed.emit(False, "相机时间核验已取消", signature)
                return
            result = check_hikvision_camera_clock(
                source,
                management_url=management_url,
            )
            if not result.ok:
                self.completed.emit(
                    False,
                    f"机位 {camera_index + 1}：{result.message}",
                    signature,
                )
                return
        self.completed.emit(
            True,
            f"{len(self.targets)} 路相机时间核验通过",
            signature,
        )


class ExternalClipProbeThread(QThread):
    """Validate external media without blocking the Qt event loop."""

    progress = pyqtSignal(int, int, str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, sidecar_path: str, expected_race_id: str, parent=None):
        super().__init__(parent)
        self.sidecar_path = str(sidecar_path)
        self.expected_race_id = str(expected_race_id)

    def run(self):
        try:
            clips = load_external_clip_sidecar(
                self.sidecar_path,
                expected_race_id=self.expected_race_id,
                progress_callback=self.progress.emit,
                cancel_check=self.isInterruptionRequested,
            )
        except ExternalClipImportCancelled:
            self.cancelled.emit()
        except (ExternalClipImportError, VideoTimelineError) as error:
            self.failed.emit(str(error))
        except Exception as error:
            logger.exception("[ExternalClipImport] media verification failed")
            self.failed.emit(str(error))
        else:
            if self.isInterruptionRequested():
                self.cancelled.emit()
            else:
                self.completed.emit(clips)


class MultiCameraManagementDialog(QDialog):
    """多摄像头管理对话框"""
    def __init__(
        self,
        current_sources,
        parent=None,
        *,
        camera_clock_management_urls=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("多摄像头管理")
        self.setMinimumSize(700, 450)
        if current_sources is None:
            self.sources = []
        elif isinstance(current_sources, (str, int)):
            self.sources = [current_sources]
        else:
            self.sources = list(current_sources)
        self.camera_clock_management_urls = _normalize_camera_clock_management_urls(
            camera_clock_management_urls,
            len(self.sources),
        )
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # 说明
        tip = QLabel("提示：支持 1-4 路摄像头并行检测。机位 1 为主相机，支持绘制终点线。")
        tip.setStyleSheet("color: #666; background-color: #f0f0f0; padding: 10px; border-radius: 4px;")
        layout.addWidget(tip)

        # 表格显示机位
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["机位", "连接地址/源", "操作"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.table)

        # 底部按钮
        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("添加机位")
        self.add_btn.clicked.connect(self._add_camera)
        self.add_btn.setStyleSheet("background-color: #1890ff; color: white; padding: 8px;")
        
        self.save_btn = QPushButton("确定并应用")
        self.save_btn.clicked.connect(self.accept)
        self.save_btn.setStyleSheet("background-color: #52c41a; color: white; padding: 8px;")
        
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        
        btn_layout.addWidget(self.add_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.save_btn)
        layout.addLayout(btn_layout)

        self._refresh_table()

    def _refresh_table(self):
        self.table.setRowCount(len(self.sources))
        for i, source in enumerate(self.sources):
            # 机位列
            name = f"机位 {i+1}"
            if i == 0: name += " (主)"
            self.table.setItem(i, 0, QTableWidgetItem(name))
            
            # 地址列
            self.table.setItem(i, 1, QTableWidgetItem(str(source)))
            
            # 操作列
            ops_widget = QWidget()
            ops_layout = QHBoxLayout(ops_widget)
            ops_layout.setContentsMargins(0, 0, 0, 0)
            
            edit_btn = QPushButton("修改")
            edit_btn.clicked.connect(lambda checked, idx=i: self._edit_camera(idx))
            
            del_btn = QPushButton("删除")
            del_btn.setEnabled(i > 0) # 不允许删除主机位，只能修改
            del_btn.clicked.connect(lambda checked, idx=i: self._delete_camera(idx))
            
            ops_layout.addWidget(edit_btn)
            ops_layout.addWidget(del_btn)
            self.table.setCellWidget(i, 2, ops_widget)

    def _add_camera(self):
        if len(self.sources) >= 4:
            QMessageBox.warning(self, "提示", "目前最多支持 4 路摄像头")
            return
            
        dialog = CameraConfigDialog("", self)
        if dialog.exec_() == QDialog.Accepted:
            self.sources.append(dialog.result_source)
            self.camera_clock_management_urls.append(
                dialog.result_management_url
            )
            self._refresh_table()

    def _edit_camera(self, index):
        dialog = CameraConfigDialog(
            self.sources[index],
            self,
            current_management_url=self.camera_clock_management_urls[index],
        )
        if dialog.exec_() == QDialog.Accepted:
            self.sources[index] = dialog.result_source
            self.camera_clock_management_urls[index] = dialog.result_management_url
            self._refresh_table()

    def _delete_camera(self, index):
        if index == 0: return
        self.sources.pop(index)
        self.camera_clock_management_urls.pop(index)
        self._refresh_table()

    def get_result(self):
        return self.sources

    def get_clock_management_urls(self):
        return self.camera_clock_management_urls

class CameraConfigDialog(QDialog):
    """摄像头配置对话框 - 支持模拟流、USB和大疆/海康网络摄像头"""

    def __init__(
        self,
        current_source,
        parent=None,
        *,
        current_management_url="",
    ):
        super().__init__(parent)
        self.setWindowTitle("摄像头配置")
        self.setMinimumSize(680, 550)
        self.result_source = current_source
        self.result_management_url = str(current_management_url or "").strip()
        self._current_management_url = self.result_management_url
        self._clock_check_passed = False
        self._init_ui()
        self._load_current(current_source)
        self._load_management_url(self._current_management_url)

    def _init_ui(self):
        # 统一字体样式
        self.setStyleSheet("""
            QDialog { background-color: #ffffff; }
            QLabel, QRadioButton, QLineEdit, QComboBox, QSpinBox, QPushButton, QGroupBox {
                font-size: 16px;
            }
            QGroupBox { font-weight: bold; }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # 1. 类型选择
        type_group = QGroupBox("选择连接方式")
        type_layout = QHBoxLayout(type_group)
        
        self.radio_mock = QRadioButton("模拟视频流")
        self.radio_usb = QRadioButton("大疆 Action 5 (USB)")
        self.radio_rtsp = QRadioButton("海康网络相机 (网线)")
        
        type_layout.addWidget(self.radio_mock)
        type_layout.addWidget(self.radio_usb)
        type_layout.addWidget(self.radio_rtsp)
        layout.addWidget(type_group)

        # 2. 参数堆栈
        self.stack = QStackedWidget()
        
        # --- 页面0: 模拟流 ---
        self.page_mock = QWidget()
        mock_layout = QFormLayout(self.page_mock)
        self.mock_path = QLineEdit()
        self.mock_path.setPlaceholderText("选择视频文件...")
        self.btn_browse = QPushButton("浏览文件")
        self.btn_browse.clicked.connect(self._browse_mock)
        mock_row = QHBoxLayout()
        mock_row.addWidget(self.mock_path)
        mock_row.addWidget(self.btn_browse)
        mock_layout.addRow("视频源:", mock_row)
        
        mock_tip = QLabel("提示：这里选择的文件将由软件直接读取。如果您想使用 mediamtx 推送的 RTSP 流，请选择右侧的“海康网络相机”。")
        mock_tip.setWordWrap(True)
        mock_tip.setStyleSheet("color: #666; font-size: 14px; background-color: #fffbe6; padding: 8px; border: 1px solid #ffe58f; border-radius: 4px;")
        mock_layout.addRow("", mock_tip)
        
        self.stack.addWidget(self.page_mock)

        # --- 页面1: USB (Action 5) ---
        self.page_usb = QWidget()
        usb_layout = QVBoxLayout(self.page_usb)
        usb_form = QFormLayout()
        self.usb_combo = QComboBox()
        self.usb_combo.setFixedHeight(40)
        self._refresh_usb_list()
        btn_refresh = QPushButton("刷新设备列表")
        btn_refresh.clicked.connect(self._refresh_usb_list)
        usb_row = QHBoxLayout()
        usb_row.addWidget(self.usb_combo, 1)
        usb_row.addWidget(btn_refresh)
        usb_form.addRow("摄像头:", usb_row)
        usb_layout.addLayout(usb_form)
        
        usb_tip = QLabel("提示：大疆 Action 5 连接电脑后，请在相机屏幕上选择“摄像头模式”。")
        usb_tip.setWordWrap(True)
        usb_tip.setStyleSheet("color: #666; font-size: 16px; background-color: #f0f0f0; padding: 10px; border-radius: 4px;")
        usb_layout.addWidget(usb_tip)
        self.stack.addWidget(self.page_usb)

        # --- 页面2: RTSP (海康) ---
        self.page_rtsp = QWidget()
        rtsp_layout = QVBoxLayout(self.page_rtsp)
        
        hik_box = QGroupBox("海康配置助手")
        hik_form = QFormLayout(hik_box)
        self.hik_ip = QLineEdit("192.168.1.64")
        self.hik_user = QLineEdit("admin")
        self.hik_pwd = QLineEdit()
        self.hik_pwd.setEchoMode(QLineEdit.Password)
        self.hik_pwd.setPlaceholderText("请输入摄像头的管理密码")
        
        hik_form.addRow("相机IP地址:", self.hik_ip)
        hik_form.addRow("登录用户名:", self.hik_user)
        hik_form.addRow("登录密码:", self.hik_pwd)
        rtsp_layout.addWidget(hik_box)

        # 高级选项（手动输入URL）
        self.advance_rtsp_check = QCheckBox("手动输入完整地址 (高级用户)")
        self.rtsp_url_input = QLineEdit()
        self.rtsp_url_input.setEnabled(False)
        self.rtsp_url_input.setPlaceholderText("rtsp://admin:password@IP:554/h264/ch1/main/av_stream")
        self.advance_rtsp_check.toggled.connect(self.rtsp_url_input.setEnabled)
        self.advance_rtsp_check.toggled.connect(lambda checked: hik_box.setEnabled(not checked))
        
        rtsp_layout.addWidget(self.advance_rtsp_check)
        rtsp_layout.addWidget(self.rtsp_url_input)

        management_box = QGroupBox("海康设备时间接口")
        management_form = QFormLayout(management_box)
        self.management_scheme_combo = QComboBox()
        self.management_scheme_combo.addItem("HTTP", "http")
        self.management_scheme_combo.addItem("HTTPS", "https")
        self.management_port_spin = QSpinBox()
        self.management_port_spin.setRange(1, 65535)
        self.management_port_spin.setValue(80)
        management_form.addRow("管理协议:", self.management_scheme_combo)
        management_form.addRow("管理端口:", self.management_port_spin)
        rtsp_layout.addWidget(management_box)

        self.camera_time_status = QLabel("相机时间：尚未核验（允许误差 3 秒）")
        self.camera_time_status.setStyleSheet(
            "color: #b54708; font-size: 14px; font-weight: 600;"
        )
        rtsp_layout.addWidget(self.camera_time_status)
        self.stack.addWidget(self.page_rtsp)

        layout.addWidget(self.stack)

        # 关联切换
        self.radio_mock.toggled.connect(lambda: self.stack.setCurrentIndex(0))
        self.radio_usb.toggled.connect(lambda: self.stack.setCurrentIndex(1))
        self.radio_rtsp.toggled.connect(lambda: self.stack.setCurrentIndex(2))
        for control in (
            self.hik_ip,
            self.hik_user,
            self.hik_pwd,
            self.rtsp_url_input,
        ):
            control.textChanged.connect(self._reset_clock_check)
        self.advance_rtsp_check.toggled.connect(self._reset_clock_check)
        self.management_scheme_combo.currentIndexChanged.connect(
            self._on_management_scheme_changed
        )
        self.management_port_spin.valueChanged.connect(self._reset_clock_check)

        # 3. 状态显示
        self.status_label = QLabel("等待测试...")
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(44)
        self.status_label.setStyleSheet("color: #888; font-size: 16px;")
        layout.addWidget(self.status_label)

        # 4. 底部按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        test_btn = QPushButton("测试连接")
        test_btn.setFixedWidth(120)
        test_btn.setStyleSheet("background-color: #fafafa; color: #666;")
        test_btn.clicked.connect(self._test_connection)
        
        save_btn = QPushButton("保存配置")
        save_btn.setFixedWidth(120)
        save_btn.setStyleSheet("background-color: #1890ff; color: white;")
        save_btn.clicked.connect(self._on_save)
        
        cancel_btn = QPushButton("取消")
        cancel_btn.setFixedWidth(80)
        cancel_btn.clicked.connect(self.reject)
        
        btn_layout.addWidget(test_btn)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def _test_connection(self):
        """异步测试连接"""
        source = self._get_current_source()
        if source is None: return
        self._clock_check_passed = False
        management_url = self._management_url_for_source(source)
        
        # 禁用按钮，显示进度
        self.status_label.setText("⏳ 正在尝试连接，请稍候...")
        self.status_label.setStyleSheet("color: #1890ff; font-weight: bold;")
        self._set_ui_enabled(False)
        
        self.tester = ConnectionTester(
            source,
            verify_hikvision_clock=self._requires_camera_clock_check(source),
            management_url=management_url,
        )
        self.tester.finished.connect(self._on_test_finished)
        self.tester.start()

    def _on_test_finished(self, success, message):
        """测试完成后的回调"""
        self._set_ui_enabled(True)
        requires_clock_check = self._requires_camera_clock_check()
        self._clock_check_passed = bool(success and requires_clock_check)
        if requires_clock_check and success:
            self.camera_time_status.setText("相机时间：核验通过")
            self.camera_time_status.setStyleSheet(
                "color: #247a52; font-size: 14px; font-weight: 600;"
            )
        elif requires_clock_check:
            self.camera_time_status.setText("相机时间：核验失败")
            self.camera_time_status.setStyleSheet(
                "color: #b54747; font-size: 14px; font-weight: 600;"
            )
        else:
            self.camera_time_status.setText("相机时间：当前源不需要海康设备校时")
            self.camera_time_status.setStyleSheet(
                "color: #667085; font-size: 14px; font-weight: 600;"
            )
        if success:
            self.status_label.setText(f"✅ {message}")
            self.status_label.setStyleSheet("color: #52c41a; font-weight: bold;")
            QMessageBox.information(self, "成功", message)
        else:
            self.status_label.setText(f"❌ {message}")
            self.status_label.setStyleSheet("color: #f5222d; font-weight: bold;")
            QMessageBox.critical(self, "失败", message)

    def _requires_camera_clock_check(self, source=None):
        if not self.radio_rtsp.isChecked():
            return False
        if source is None:
            source = self._get_current_source(show_warning=False)
        return camera_clock_check_required(source)

    def _on_management_scheme_changed(self, *_args):
        scheme = self.management_scheme_combo.currentData()
        port = self.management_port_spin.value()
        if scheme == "https" and port == 80:
            self.management_port_spin.setValue(443)
        elif scheme == "http" and port == 443:
            self.management_port_spin.setValue(80)
        self._reset_clock_check()

    def _reset_clock_check(self, *_args):
        self._clock_check_passed = False
        if not hasattr(self, "camera_time_status"):
            return
        if self._requires_camera_clock_check():
            text = "相机时间：尚未核验（允许误差 3 秒）"
        else:
            text = "相机时间：当前源不需要海康设备校时"
        self.camera_time_status.setText(text)
        self.camera_time_status.setStyleSheet(
            "color: #b54708; font-size: 14px; font-weight: 600;"
        )

    def _set_ui_enabled(self, enabled):
        """统一控制界面可用性"""
        self.radio_mock.setEnabled(enabled)
        self.radio_usb.setEnabled(enabled)
        self.radio_rtsp.setEnabled(enabled)
        self.stack.setEnabled(enabled)
        # 查找底部按钮并禁用/启用
        for btn in self.findChildren(QPushButton):
            if btn.text() in ["测试连接", "保存配置", "取消"]:
                btn.setEnabled(enabled)

    def _get_current_source(self, *, show_warning=True):
        """根据当前界面选择计算出source"""
        if self.radio_mock.isChecked():
            return self.mock_path.text().strip()
        elif self.radio_usb.isChecked():
            return self.usb_combo.currentData()
        else:
            if self.advance_rtsp_check.isChecked():
                source = self.rtsp_url_input.text().strip()
                if not source and show_warning:
                    QMessageBox.warning(self, "提示", "请填写完整 RTSP 地址")
                return source or None
            else:
                # 自动生成海康地址
                import urllib.parse
                ip = self.hik_ip.text().strip()
                user = self.hik_user.text().strip()
                pwd = self.hik_pwd.text().strip()
                if not ip or not pwd:
                    if show_warning:
                        QMessageBox.warning(self, "提示", "请填写海康相机的IP和密码")
                    return None
                
                # 对用户名和密码进行URL编码
                safe_user = urllib.parse.quote(user, safe="")
                safe_pwd = urllib.parse.quote(pwd, safe="")
                host = f"[{ip}]" if ":" in ip and not ip.startswith("[") else ip
                
                # 尝试两种常见的海康RTSP路径
                # 1. 现代标准路径: /Streaming/Channels/101
                # 2. 传统路径: /h264/ch1/main/av_stream
                return f"rtsp://{safe_user}:{safe_pwd}@{host}:554/Streaming/Channels/101"

    def _management_url_for_source(self, source):
        if not camera_clock_check_required(source):
            return ""
        from urllib.parse import urlsplit

        parsed = urlsplit(source)
        host = parsed.hostname
        if not host:
            return ""
        host_for_url = f"[{host}]" if ":" in host else host
        scheme = str(self.management_scheme_combo.currentData() or "http")
        port = self.management_port_spin.value()
        return f"{scheme}://{host_for_url}:{port}"

    def _refresh_usb_list(self):
        """刷新USB摄像头列表"""
        self.usb_combo.clear()
        if HAS_PYGRABBER:
            try:
                devices = FilterGraph().get_input_devices()
                for i, name in enumerate(devices):
                    self.usb_combo.addItem(f"{i}: {name}", i)
            except Exception:
                self.usb_combo.addItem("0: Default Camera", 0)
        else:
            # 备用方案：列出 0-5
            for i in range(5):
                self.usb_combo.addItem(f"设备 {i}", i)

    def _browse_mock(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Video (*.mp4 *.avi *.mkv);;All (*.*)")
        if path:
            self.mock_path.setText(path)
            # 自动切换到模拟视频单选按钮，确保保存时使用的是这个路径
            self.radio_mock.setChecked(True)
            self.stack.setCurrentIndex(0)

    def _load_current(self, source):
        """加载当前配置到界面"""
        if isinstance(source, int):
            self.radio_usb.setChecked(True)
            self.stack.setCurrentIndex(1)
            idx = self.usb_combo.findData(source)
            if idx >= 0: self.usb_combo.setCurrentIndex(idx)
        elif isinstance(source, str):
            if source.lower().startswith("rtsp://"):
                self.radio_rtsp.setChecked(True)
                self.stack.setCurrentIndex(2)
                # 使用结构化 URL 解析，避免已编码凭据再次保存时被二次编码。
                try:
                    from urllib.parse import unquote, urlsplit

                    parsed = urlsplit(source)
                    if not parsed.hostname:
                        raise ValueError("RTSP 地址缺少主机名")
                    self.hik_user.setText(unquote(parsed.username or ""))
                    self.hik_pwd.setText(unquote(parsed.password or ""))
                    self.hik_ip.setText(parsed.hostname)
                    self.rtsp_url_input.setText(source)
                    is_standard_hikvision = bool(
                        parsed.path.rstrip("/") == "/Streaming/Channels/101"
                        and parsed.port in {None, 554}
                        and not parsed.query
                        and not parsed.fragment
                    )
                    self.advance_rtsp_check.setChecked(not is_standard_hikvision)
                except (TypeError, ValueError):
                    self.rtsp_url_input.setText(source)
                    self.advance_rtsp_check.setChecked(True)
            else:
                self.radio_mock.setChecked(True)
                self.stack.setCurrentIndex(0)
                self.mock_path.setText(source)

    def _load_management_url(self, management_url):
        if not management_url:
            return
        try:
            from urllib.parse import urlsplit

            parsed = urlsplit(management_url)
            index = self.management_scheme_combo.findData(parsed.scheme.lower())
            if index < 0 or not parsed.hostname:
                return
            self.management_scheme_combo.setCurrentIndex(index)
            default_port = 443 if parsed.scheme.lower() == "https" else 80
            self.management_port_spin.setValue(parsed.port or default_port)
        except (TypeError, ValueError):
            return
        self._reset_clock_check()

    def _on_save(self):
        """保存并关闭"""
        source = self._get_current_source()
        if source is None:
            return
        if self._requires_camera_clock_check(source) and not self._clock_check_passed:
            QMessageBox.warning(
                self,
                "相机时间未核验",
                "请先点击“测试连接”，确认画面和相机时间均正常。",
            )
            return
        self.result_source = source
        self.result_management_url = self._management_url_for_source(source)
        self.accept()

class StartTimeDialog(QDialog):
    """发枪时间设置对话框"""

    def __init__(self, database, parent=None):
        super().__init__(parent)
        self.database = database
        self.setWindowTitle("发枪时间设置")
        self.setMinimumSize(800, 600)
        
        # 统一字体样式
        self.setStyleSheet("""
            QDialog { background-color: #ffffff; }
            QLabel, QTableWidget, QPushButton, QLineEdit, QGroupBox {
                font-size: 16px;
            }
            QGroupBox { font-weight: bold; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # 说明
        note = QLabel(
            "设置各组别的发枪时间（北京时间）\n"
            "成绩 = 过线时间 - 发枪时间\n"
            "提示：请确保电脑时间已校准！"
        )
        note.setStyleSheet("color: #666; padding: 10px; line-height: 1.5; font-size: 16px;")
        layout.addWidget(note)

        # 表格
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["组别", "发枪时间", "操作"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
        self.table.setColumnWidth(1, 350) # 调大时间列宽度
        self.table.setColumnWidth(2, 120) # 调大按钮列宽度
        self.table.verticalHeader().setDefaultSectionSize(60) # 增加行高
        self.table.setStyleSheet("""
            QTableWidget { 
                gridline-color: #f0f0f0;
                font-size: 16px;
            }
            QHeaderView::section {
                background-color: #fafafa;
                padding: 10px;
                font-weight: bold;
                font-size: 16px;
            }
        """)
        layout.addWidget(self.table)

        # 添加新组别
        add_group = QGroupBox("手动新增组别")
        add_group_layout = QHBoxLayout(add_group)
        add_group_layout.setContentsMargins(15, 15, 15, 15)
        
        self.new_category_input = QLineEdit()
        self.new_category_input.setPlaceholderText("输入组别名称（如：男子公开组）")
        self.new_category_input.setFixedHeight(50)
        add_group_layout.addWidget(self.new_category_input)
        
        add_btn = QPushButton("添加组别")
        add_btn.setFixedHeight(50)
        add_btn.setStyleSheet("background-color: #52c41a; color: white;") # 绿色代表新增
        add_btn.clicked.connect(self._add_category)
        add_group_layout.addWidget(add_btn)
        layout.addWidget(add_group)

        # 底部按钮
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(20)
        btn_layout.addStretch()

        save_btn = QPushButton("保存所有设置")
        save_btn.setFixedHeight(60)
        save_btn.setFixedWidth(220)
        save_btn.setStyleSheet("background-color: #1890ff; font-size: 16px; color: white;")
        save_btn.clicked.connect(self._save_all)
        btn_layout.addWidget(save_btn)

        close_btn = QPushButton("关闭")
        close_btn.setFixedHeight(60)
        close_btn.setFixedWidth(130)
        close_btn.setStyleSheet("background-color: #f5f5f5; color: #333; border: 1px solid #d9d9d9;")
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)

        layout.addLayout(btn_layout)

        # 加载数据
        self._load_categories()

    def _load_categories(self):
        """加载组别和发枪时间"""
        self.table.setRowCount(0)

        # 获取所有组别（从选手表）
        categories = self.database.get_all_categories()
        # 获取已设置的发枪时间
        start_times = self.database.get_all_start_times()
        start_time_dict = {st['category']: st['start_time'] for st in start_times}

        # 合并（可能有已设置发枪时间但没有选手的组别）
        all_categories = set(categories) | set(start_time_dict.keys())

        for category in sorted(all_categories):
            self._add_row(category, start_time_dict.get(category, ''))

    def _add_row(self, category: str, start_time: str = ''):
        """添加一行"""
        row = self.table.rowCount()
        self.table.insertRow(row)

        # 组别名称
        category_item = QTableWidgetItem(category)
        category_item.setFlags(category_item.flags() & ~Qt.ItemIsEditable)
        self.table.setItem(row, 0, category_item)

        # 发枪时间编辑器
        time_edit = QDateTimeEdit()
        time_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        time_edit.setCalendarPopup(True)
        if start_time:
            # 解析已有时间
            dt = QDateTime.fromString(start_time, "yyyy-MM-dd HH:mm:ss")
            if dt.isValid():
                time_edit.setDateTime(dt)
            else:
                time_edit.setDateTime(QDateTime.currentDateTime())
        else:
            time_edit.setDateTime(QDateTime.currentDateTime())
        self.table.setCellWidget(row, 1, time_edit)

        # 删除按钮
        del_btn = QPushButton("删除")
        del_btn.clicked.connect(lambda: self._delete_row(category))
        self.table.setCellWidget(row, 2, del_btn)

    def _add_category(self):
        """添加新组别"""
        category = self.new_category_input.text().strip()
        if not category:
            return

        # 检查是否已存在
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).text() == category:
                QMessageBox.warning(self, "提示", f"组别 '{category}' 已存在")
                return

        self._add_row(category)
        self.new_category_input.clear()

    def _delete_row(self, category: str):
        """删除行"""
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).text() == category:
                self.table.removeRow(row)
                # 从数据库删除
                self.database.delete_start_time(category)
                break

    def _save_all(self):
        """保存所有设置"""
        saved_count = 0
        for row in range(self.table.rowCount()):
            category = self.table.item(row, 0).text()
            time_edit = self.table.cellWidget(row, 1)
            if time_edit:
                start_time = time_edit.dateTime().toString("yyyy-MM-dd HH:mm:ss")
                self.database.set_start_time(category, start_time)
                saved_count += 1

        QMessageBox.information(self, "保存成功", f"已保存 {saved_count} 个组别的发枪时间")


class NewRaceDialog(QDialog):
    """新建赛事对话框：指定新赛事的文件夹名称和位置"""
    def __init__(
        self,
        parent=None,
        base_path: Optional[str] = None,
        allow_existing: bool = False,
    ):
        super().__init__(parent)
        self.allow_existing = allow_existing
        self.setWindowTitle("选择赛事" if allow_existing else "新建赛事")
        self.setFixedSize(620, 300)
        self.result_name = ""
        self.result_path = ""
        self.selected_race_dir: Optional[Path] = None
        self.base_path = base_path
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        
        tip_text = (
            "打开已有赛事，或创建一个全新的赛事文件夹。"
            if self.allow_existing
            else "创建一个全新的赛事文件夹，所有数据（数据库、照片）将独立存储。"
        )
        tip_label = QLabel(tip_text)
        tip_label.setStyleSheet("color: #666; font-size: 16px;")
        layout.addWidget(tip_label)
        
        # 输入区域
        form_widget = QWidget()
        form = QFormLayout(form_widget)
        form.setSpacing(15)
        
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("例如：2026年环湖赛")
        self.name_input.setFixedHeight(40)
        self.name_input.setText(datetime.now().strftime("%Y%m%d_新赛事"))
        form.addRow("赛事名称:", self.name_input)
        
        # 路径选择
        path_layout = QHBoxLayout()
        self.path_input = QLineEdit()
        self.path_input.setReadOnly(True)
        self.path_input.setFixedHeight(40)
        default_path = self.base_path or str(Path.cwd() / "RaceData")
        self.path_input.setText(default_path)
        
        browse_btn = QPushButton("选择位置...")
        browse_btn.setFixedHeight(40)
        browse_btn.clicked.connect(self._browse_path)
        
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(browse_btn)
        form.addRow("存储位置:", path_layout)
        
        layout.addWidget(form_widget)
        
        # 按钮
        btn_layout = QHBoxLayout()
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setFixedHeight(45)
        self.cancel_btn.clicked.connect(self.reject)

        if self.allow_existing:
            self.open_btn = QPushButton("打开已有赛事...")
            self.open_btn.setFixedHeight(45)
            self.open_btn.clicked.connect(self._open_existing)
            btn_layout.addWidget(self.open_btn)
            btn_layout.addStretch()
        
        self.ok_btn = QPushButton("创建并进入新赛事")
        self.ok_btn.setStyleSheet("""
            QPushButton {
                background-color: #1890ff;
                color: white;
                font-weight: bold;
                border: none;
            }
            QPushButton:hover {
                background-color: #40a9ff;
            }
        """)
        self.ok_btn.setFixedHeight(45)
        self.ok_btn.clicked.connect(self._on_ok)
        
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.ok_btn)
        layout.addLayout(btn_layout)

        if self.base_path:
            browse_btn.setEnabled(False)

    def _open_existing(self):
        start_path = self.base_path or self.path_input.text() or str(Path.cwd())
        selected = QFileDialog.getExistingDirectory(
            self,
            "打开已有赛事",
            start_path,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        if not selected:
            return

        race_dir = Path(selected).expanduser().absolute()
        if not (race_dir / "timing.db").is_file():
            QMessageBox.warning(
                self,
                "不是赛事目录",
                "请选择包含 timing.db 的具体赛事文件夹，不要选择 RaceData 根目录。",
            )
            return

        self.selected_race_dir = race_dir
        self.result_name = race_dir.name
        self.result_path = str(race_dir.parent)
        self.accept()

    def _browse_path(self):
        path = QFileDialog.getExistingDirectory(self, "选择赛事存储根目录", self.path_input.text())
        if path:
            self.path_input.setText(path)

    def _on_ok(self):
        self.selected_race_dir = None
        self.result_name = self.name_input.text().strip()
        self.result_path = self.path_input.text().strip()
        
        if not self.result_name:
            QMessageBox.warning(self, "错误", "请输入赛事名称")
            return
            
        full_path = Path(self.result_path) / self.result_name
        if full_path.exists():
            res = QMessageBox.question(self, "确认", f"文件夹 '{self.result_name}' 已存在，是否进入该赛事？\n(注意：这不会清空已有数据)", 
                                      QMessageBox.Yes | QMessageBox.No)
            if res != QMessageBox.Yes:
                return
                
        self.accept()

class VLMConfigDialog(QDialog):
    """VLM 辅助识别配置对话框"""
    def __init__(self, config: Dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("VLM 辅助识别配置")
        self.setMinimumWidth(500)
        self.config = config.get('vlm_config', {})
        
        layout = QVBoxLayout(self)
        
        form = QFormLayout()
        
        self.enabled_cb = QCheckBox("启用 VLM 辅助 (OpenAI兼容中转/通义千问/豆包)")
        self.enabled_cb.setChecked(self.config.get('enabled', False))
        form.addRow("总开关:", self.enabled_cb)
        
        self.model_type = QComboBox()
        self.model_type.addItems(["openai", "doubao", "qwen"])
        self.model_type.setCurrentText(self.config.get('model_type', 'qwen'))
        form.addRow("模型类型:", self.model_type)
        
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.Password)
        self.api_key.setText(self.config.get('api_key', ''))
        self.api_key.setPlaceholderText("从云端控制台获取的 API Key")
        form.addRow("API Key:", self.api_key)
        
        self.endpoint_label = QLabel("模型名称:")
        self.endpoint_id = QLineEdit()
        self.endpoint_id.setText(self.config.get('endpoint_id', ''))
        form.addRow(self.endpoint_label, self.endpoint_id)

        self.base_url_label = QLabel("中转地址:")
        self.base_url = QLineEdit()
        self.base_url.setText(self.config.get('base_url', ''))
        form.addRow(self.base_url_label, self.base_url)
        
        self.max_rpm = QSpinBox()
        self.max_rpm.setRange(1, 1000)
        self.max_rpm.setValue(self.config.get('max_calls_per_minute', 30))
        form.addRow("每分钟限速 (RPM):", self.max_rpm)

        self.ocr_mode = QComboBox()
        self.ocr_mode.addItem("仅兜底（推荐）", "fallback")
        self.ocr_mode.addItem("优先大模型", "first")
        self.ocr_mode.addItem("只用大模型（测试用）", "only")
        current_mode = str(self.config.get("ocr_mode", "fallback") or "fallback").lower()
        for i in range(self.ocr_mode.count()):
            if self.ocr_mode.itemData(i) == current_mode:
                self.ocr_mode.setCurrentIndex(i)
                break
        form.addRow("号码识别策略:", self.ocr_mode)
        
        layout.addLayout(form)

        def _refresh_endpoint_placeholder():
            model_type = self.model_type.currentText().strip().lower()
            if model_type == "openai":
                self.endpoint_label.setText("模型名称:")
                self.endpoint_id.setPlaceholderText("例如：gpt-4.1-mini")
                self.base_url_label.setVisible(True)
                self.base_url.setVisible(True)
                self.base_url.setPlaceholderText("例如：https://你的中转地址/v1")
            elif model_type == "qwen":
                self.endpoint_label.setText("模型名称:")
                self.endpoint_id.setPlaceholderText("通义模型名（可留空）：qwen3.5-ocr")
                self.base_url_label.setVisible(False)
                self.base_url.setVisible(False)
            else:
                self.endpoint_label.setText("推理接入点 ID:")
                self.endpoint_id.setPlaceholderText("豆包需提供推理接入点 ID (Endpoint ID)")
                self.base_url_label.setVisible(False)
                self.base_url.setVisible(False)

        self.model_type.currentTextChanged.connect(lambda _t: _refresh_endpoint_placeholder())
        _refresh_endpoint_placeholder()
        
        info_label = QLabel(
            "<font color='gray'>注：VLM 仅在 OCR 无法确定号码或过线并发较高时作为辅助判定。<br/>"
            "开启后会产生云端 API 调用费用，请注意限速设置。</font>"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def get_result(self) -> Dict:
        return {
            'enabled': self.enabled_cb.isChecked(),
            'model_type': self.model_type.currentText(),
            'api_key': self.api_key.text().strip(),
            'endpoint_id': self.endpoint_id.text().strip(),
            'base_url': self.base_url.text().strip(),
            'max_calls_per_minute': self.max_rpm.value(),
            'ocr_mode': self.ocr_mode.currentData() or "fallback",
        }


class MainWindow(QMainWindow):
    """
    实时计时系统主窗口

    布局：
    - 左边：实时视频画面
    - 右边：过线记录列表
    """

    # 定义信号，用于跨线程更新 UI
    event_saved_signal = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    field_issue_saved_signal = pyqtSignal(object)
    passage_received_signal = pyqtSignal(object)
    timing_status_signal = pyqtSignal(object)

    def __init__(
        self,
        config: dict,
        *,
        runtime_source_override=None,
        runtime_sport_profile_override=None,
    ):
        super().__init__()
        self.config = config
        self._runtime_source_override = runtime_source_override
        self._runtime_sport_profile_override = (
            normalize_sport_profile(runtime_sport_profile_override)
            if runtime_sport_profile_override is not None
            else None
        )
        self._yolo_only_mode = bool(config.get("yolo_only_mode", False))
        self.config["yolo_only_mode"] = self._yolo_only_mode
        self.sport_profile = normalize_sport_profile(config.get("sport_profile", "cycling"))
        self.config["sport_profile"] = self.sport_profile

        # 连接信号
        self.event_saved_signal.connect(self._on_event_saved_ui)
        self.log_signal.connect(self._on_log_received_ui)
        self.field_issue_saved_signal.connect(self._on_field_issue_saved_ui)
        self.passage_received_signal.connect(self._on_passage_received_ui)
        self.timing_status_signal.connect(self._on_timing_status_ui)

        # 组件 (支持多摄像头)
        self.readers = {}         # source_id -> StreamReader
        self.detectors = {}       # source_id -> Detector
        self.video_threads = {}   # source_id -> VideoThread
        self.preview_threads = {} # source_id -> PreviewThread
        self.video_labels = {}    # source_id -> InteractiveVideoLabel
        self.finish_line_checkboxes = {} # source_id -> QCheckBox
        
        self.shared_model = None  # 共享 YOLO 模型
        self.shared_athlete_validator = None  # 仅在无号码事件落库前确认自行车
        self._athlete_validator_checked = False
        self.shared_vlm = None    # 共享 VLM 助手
        self._ocr_runtime_state = 'idle'  # idle/loading/ready/failed/disabled
        self.database: Optional[Database] = None
        self.recorder: Optional[EventRecorder] = None
        self.recording_manager: Optional[ManualRecordingManager] = None
        self._recording_error_message = ""
        self._recording_timeline_warning = ""
        self._auto_record_live_sources = bool(config.get("auto_record_live_sources", True))
        self.config["auto_record_live_sources"] = self._auto_record_live_sources
        self.viewer_dialog: Optional[ImageViewerDialog] = None
        self.ocr_manager: Optional[OCRManager] = None
        self.field_issue_log: Optional[FieldIssueLog] = None
        self._field_issue_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="field-issue"
        )
        self._field_issue_closing = False
        self._latest_frame_observations: Dict[int, tuple[list, list]] = {}

        self.passage_receiver: Optional[PassageEventReceiver] = None
        self.racetiger_source: Optional[RaceTigerSource] = None
        self.passage_event_store: Optional[PassageEventStore] = None
        self.video_supplement_store: Optional[VideoSupplementStore] = None
        self.video_timeline_store: Optional[VideoTimelineStore] = None
        self.passage_review_dialog: Optional[PassageReviewDialog] = None
        self._external_clip_import_thread: Optional[ExternalClipProbeThread] = None
        self._external_clip_import_progress: Optional[QProgressDialog] = None
        self._external_clip_import_context = None
        self.external_clip_action: Optional[QAction] = None
        self._latest_passage_event = None
        self._timing_provider = str(config.get("timing_provider", "cyclerace") or "cyclerace").strip().lower()
        self.config["timing_provider"] = self._timing_provider
        self._racetiger_base_url = str(config.get("racetiger_base_url", "") or "").strip()
        self._racetiger_pc = str(config.get("racetiger_pc", "") or "").strip()
        self._racetiger_rid = str(config.get("racetiger_rid", "") or "").strip()
        self._racetiger_token = str(config.get("racetiger_token", "") or "").strip()
        try:
            self._racetiger_poll_interval = max(
                0.5, float(config.get("racetiger_poll_interval_seconds", 2.0))
            )
        except (TypeError, ValueError):
            self._racetiger_poll_interval = 2.0
        self._passage_receiver_enabled = bool(config.get("passage_receiver_enabled", False))
        self._passage_receiver_host = str(
            config.get("passage_receiver_host") or DEFAULT_HOST
        ).strip()
        try:
            self._passage_receiver_port = int(
                config.get("passage_receiver_port", DEFAULT_PORT)
            )
        except (TypeError, ValueError):
            self._passage_receiver_port = DEFAULT_PORT
        self.config["passage_receiver_enabled"] = self._passage_receiver_enabled
        self.config["passage_receiver_host"] = self._passage_receiver_host
        self.config["passage_receiver_port"] = self._passage_receiver_port
        self._sync_passage_video_config()

        # 配置
        # source 现在可以是一个列表，支持多路
        sources = config.get('sources')
        if sources is None:
            sources = config.get('source', 'rtsp://localhost:8554/test')
        if isinstance(sources, (str, int)):
            self.sources = [sources]
        else:
            self.sources = list(sources) if sources else []
        if not self.sources:
            self.sources = [config.get('source', 'rtsp://localhost:8554/test')]
        self.camera_clock_management_urls = _normalize_camera_clock_management_urls(
            config.get("camera_clock_management_urls"),
            len(self.sources),
        )
        self.config["camera_clock_management_urls"] = list(
            self.camera_clock_management_urls
        )
        self.model_path = config.get('model_path')
        self.race_root = Path(config.get("output_dir") or (Path.cwd() / "RaceData")).expanduser().absolute()
        self.output_dir = self.race_root.absolute()
        self._software_commit = config.get("software_commit") or _resolve_git_commit(project_root)
        requested_ocr_engine = (config.get('ocr_engine') or 'paddleocr').lower()
        if requested_ocr_engine != 'paddleocr':
            logger.warning(f"[Main] 当前版本已锁定 PaddleOCR，忽略配置 ocr_engine={requested_ocr_engine}")
        self.ocr_engine = 'paddleocr'
        self.config['ocr_engine'] = 'paddleocr'
        self._race_ready = False
        self._main_fps_value = 0.0
        self._gate_guard_enabled = True
        self.config['gate_guard_enabled'] = True

        # 轻量实时巡检（摄像头版）：只复用现有检测结果做滚动统计，不新增模型推理
        self._live_monitor_enabled = bool(config.get('live_monitor_enabled', True))
        self._live_monitor_window_seconds = float(config.get('live_monitor_window_seconds', 30.0))
        self._live_monitor_update_seconds = float(config.get('live_monitor_update_seconds', 1.2))
        self._live_monitor_min_fps = float(config.get('live_monitor_min_fps', 12.0))
        self._live_monitor_warn_drop_rate = float(config.get('live_monitor_warn_drop_rate', 0.05))
        self._live_monitor_severe_drop_rate = float(config.get('live_monitor_severe_drop_rate', 0.15))
        self._live_monitor_warn_cooldown_seconds = float(config.get('live_monitor_warn_cooldown_seconds', 8.0))
        self._live_monitor_samples: Dict[int, deque] = {}
        self._live_monitor_last_warn_ts: Dict[int, float] = {}
        self._live_monitor_last_render_ts = 0.0
        self._live_monitor_last_summary = ""
        self.config['live_monitor_enabled'] = self._live_monitor_enabled
        self.config['live_monitor_window_seconds'] = self._live_monitor_window_seconds
        self.config['live_monitor_update_seconds'] = self._live_monitor_update_seconds
        self.config['live_monitor_min_fps'] = self._live_monitor_min_fps
        self.config['live_monitor_warn_drop_rate'] = self._live_monitor_warn_drop_rate
        self.config['live_monitor_severe_drop_rate'] = self._live_monitor_severe_drop_rate
        self.config['live_monitor_warn_cooldown_seconds'] = self._live_monitor_warn_cooldown_seconds

        self._ocr_poll_timer = QTimer(self)
        self._ocr_poll_timer.setInterval(100)
        self._ocr_poll_timer.timeout.connect(self._poll_ocr_runtime)

        self._recording_poll_timer = QTimer(self)
        self._recording_poll_timer.setInterval(500)
        self._recording_poll_timer.timeout.connect(self._poll_recording_status)

        self.setObjectName("video_evidence_main_window")
        # 主窗口外壳保持克制，具体工作区样式由各自组件负责。
        self.setStyleSheet("""
            QMainWindow#video_evidence_main_window {
                background-color: #eef2f5;
                font-family: "Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI", sans-serif;
                font-size: 13px;
            }
            QMenuBar {
                min-height: 30px;
                background-color: #f8fafb;
                border-bottom: 1px solid #d8dee6;
                color: #344054;
                font-size: 13px;
            }
            QMenuBar::item {
                padding: 6px 10px;
                margin: 1px 2px;
                background-color: transparent;
                color: #344054;
            }
            QMenuBar::item:selected {
                background-color: #e8edf2;
                color: #182230;
            }
            QMenu {
                background-color: #ffffff;
                border: 1px solid #cfd6df;
                padding: 4px;
                color: #253246;
                font-size: 13px;
            }
            QMenu::item {
                min-width: 160px;
                padding: 7px 28px 7px 12px;
                margin: 1px;
                border-radius: 3px;
            }
            QMenu::item:selected {
                background-color: #edf2f6;
                color: #182230;
            }
            QMenu::separator { height: 1px; background: #e4e8ed; margin: 4px 8px; }
            QStatusBar {
                min-height: 22px;
                max-height: 24px;
                background-color: #f8fafb;
                border-top: 1px solid #d8dee6;
                color: #667085;
                font-size: 11px;
            }
            QStatusBar::item { border: none; }
            QToolTip {
                background: #182230;
                color: #ffffff;
                border: 1px solid #182230;
                padding: 5px 7px;
                font-size: 12px;
            }
        """)

        self._apply_line_and_roi_config()

        # 状态
        self._running = False
        self._start_time = None
        self._frame_count = 0
        self._initialized = False  # 是否已初始化组件
        self._session_event_count = 0  # 本次会话过线数
        self._camera_clock_preflight_thread: Optional[CameraClockPreflightThread] = None
        self._camera_clock_verified_signature = None

        self._init_ui()
        self._load_initial_config()
        self._setup_logging()
        
        # OCR model is initialized only inside the low-priority child process.
        if not self._yolo_only_mode:
            QTimer.singleShot(500, self._init_ocr_runtime)
        QTimer.singleShot(0, self._prompt_race_selection)

    def _load_initial_config(self):
        """从数据库加载持久化配置"""
        if self.database:
            if self.ocr_manager:
                self.ocr_manager.only_numeric = False
                self.ocr_manager._bib_ranges = []

            self._gate_guard_enabled = True
            self.config['gate_guard_enabled'] = True
            logger.info("[Main] 已启用自动号码格式与龙门误检保护")

            # 启动时同步一次给检测器
            self._update_detector_athletes()

    def _apply_line_and_roi_config(self):
        finish_line = self.config.get('finish_line', {})
        self.line_pt1 = (finish_line.get('x1', 0), finish_line.get('y1', 400))
        self.line_pt2 = (finish_line.get('x2', 1920), finish_line.get('y2', 400))
        self.finish_line_enabled = self.config.get('finish_line_enabled', {0: True})
        self.roi_points = self.config.get('roi_points', [])
        self.roi_enabled = self.config.get('roi_enabled', ({0: True} if self.roi_points else {0: False}))
        if isinstance(self.finish_line_enabled, list):
            self.finish_line_enabled = {i: True for i in self.finish_line_enabled}
        self.finish_line_enabled = {int(k): bool(v) for k, v in self.finish_line_enabled.items()}
        self.roi_enabled = {int(k): bool(v) for k, v in self.roi_enabled.items()}

    def _sync_passage_video_config(self):
        try:
            clock_offset_ms = int(self.config.get("passage_clock_offset_ms", 0))
        except (TypeError, ValueError):
            clock_offset_ms = 0
        try:
            pre_roll_ms = max(
                0, int(self.config.get("passage_video_preroll_ms", 3_000))
            )
        except (TypeError, ValueError):
            pre_roll_ms = 3_000
        try:
            timing_error_ms = max(
                0, int(self.config.get("video_timeline_timing_error_ms", 2_000))
            )
        except (TypeError, ValueError):
            timing_error_ms = 2_000

        self._passage_clock_offset_ms = clock_offset_ms
        self._passage_video_preroll_ms = pre_roll_ms
        self._video_timeline_timing_error_ms = timing_error_ms
        self.config["passage_clock_offset_ms"] = clock_offset_ms
        self.config["passage_video_preroll_ms"] = pre_roll_ms
        self.config["video_timeline_timing_error_ms"] = timing_error_ms

    def _apply_race_config(self, race_dir: Path):
        runtime_config = {
            key: self.config.get(key)
            for key in (
                'runtime_dir',
                'model_path',
                'yolo_only_mode',
                'ocr_cpu_threads',
            )
            if key in self.config
        }
        if self._runtime_source_override is not None:
            runtime_config.update(
                {
                    "source": self._runtime_source_override,
                    "sources": [self._runtime_source_override],
                }
            )
        if self._runtime_sport_profile_override is not None:
            runtime_config["sport_profile"] = self._runtime_sport_profile_override
        self.config.update(
            {
                "passage_clock_offset_ms": 0,
                "passage_video_preroll_ms": 3_000,
                "video_timeline_timing_error_ms": 2_000,
                "camera_clock_management_urls": [],
            }
        )
        config_file = race_dir / "config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    saved_config = json.load(f)
                for k, v in saved_config.items():
                    if k == 'ai_config' or k in runtime_config:
                        continue
                    self.config[k] = v
            except Exception as e:
                logger.warning(f"加载保存配置失败: {e}")
        self.config.update(runtime_config)
        self.config['output_dir'] = str(race_dir)
        self._sync_passage_video_config()
        self._timing_provider = str(
            self.config.get("timing_provider", "cyclerace") or "cyclerace"
        ).strip().lower()
        self._racetiger_base_url = str(
            self.config.get("racetiger_base_url", "") or ""
        ).strip()
        self._racetiger_pc = str(
            self.config.get("racetiger_pc", "") or ""
        ).strip()
        self._racetiger_rid = str(
            self.config.get("racetiger_rid", "") or ""
        ).strip()
        self._racetiger_token = str(
            self.config.get("racetiger_token", "") or ""
        ).strip()
        try:
            self._racetiger_poll_interval = max(
                0.5, float(self.config.get("racetiger_poll_interval_seconds", 2.0))
            )
        except (TypeError, ValueError):
            self._racetiger_poll_interval = 2.0
        self.sport_profile = normalize_sport_profile(self.config.get('sport_profile', 'cycling'))
        self.config['sport_profile'] = self.sport_profile
        self._gate_guard_enabled = True
        self.config['gate_guard_enabled'] = True
        self._sync_sport_profile_combo()
        sources = self.config.get('sources')
        if sources is None:
            sources = self.config.get('source', 'rtsp://localhost:8554/test')
        if isinstance(sources, (str, int)):
            self.sources = [sources]
        else:
            self.sources = list(sources) if sources else []
        if not self.sources:
            self.sources = [self.config.get('source', 'rtsp://localhost:8554/test')]
        self.camera_clock_management_urls = _normalize_camera_clock_management_urls(
            self.config.get("camera_clock_management_urls"),
            len(self.sources),
        )
        self.config["camera_clock_management_urls"] = list(
            self.camera_clock_management_urls
        )
        self._camera_clock_verified_signature = None
        self.model_path = self.config.get('model_path')
        requested_ocr_engine = (self.config.get('ocr_engine') or 'paddleocr').lower()
        if requested_ocr_engine != 'paddleocr':
            logger.warning(f"[Main] 当前版本已锁定 PaddleOCR，忽略配置 ocr_engine={requested_ocr_engine}")
        self.ocr_engine = 'paddleocr'
        self.config['ocr_engine'] = 'paddleocr'
        self._refresh_ocr_engine_badge()
        self._apply_line_and_roi_config()
        if len(self.sources) != len(self.video_labels):
            self._init_video_labels()

    def _activate_race_dir(self, race_dir: Path):
        self._close_passage_review()
        self.output_dir = race_dir.absolute()
        self.config['output_dir'] = str(self.output_dir)
        self.field_issue_log = FieldIssueLog(self.output_dir / "issues.jsonl")
        self._save_global_config({"last_race_dir": str(self.output_dir)})
        db_path = self.output_dir / "timing.db"
        logger.info(f"[Main] 激活赛事目录: {race_dir.name}, 数据库路径: {db_path}")
        self.database = Database(str(db_path))
        if hasattr(self, 'event_list') and self.event_list:
            try:
                self.event_list.set_database(self.database)
                # 强制重置 UI 状态，确保加载新数据库的数据
                if hasattr(self.event_list, 'reset_ui'):
                    self.event_list.reset_ui()
                self.event_list.refresh_data()
                logger.info("[Main] 已在赛事切换时绑定并重置事件列表")
            except Exception as e:
                logger.warning(f"[Main] 事件列表数据库绑定失败: {e}")
        if hasattr(self, 'live_event_review'):
            self.live_event_review.set_database(self.database)
            self.live_event_review.set_output_dir(self.output_dir)
            self.live_event_review.clear_event()
        self._load_video_timeline()
        self._start_passage_receiver()
        self.video_supplement_store = VideoSupplementStore(
            self.output_dir / "video_supplements.jsonl"
        )
        if self._yolo_only_mode:
            self.ocr_manager = None
        elif not self.ocr_manager:
            self.ocr_manager = OCRManager(self.database)
            self.ocr_manager.on_event_done = self._on_ocr_event_done
        else:
            self.ocr_manager.db = self.database
        self._load_initial_config()
        self._race_ready = True
        if self.ocr_manager:
            QTimer.singleShot(0, self._init_ocr_runtime)
        if hasattr(self, "field_issue_action"):
            self.field_issue_action.setEnabled(True)
        if hasattr(self, "race_name_label"):
            self.race_name_label.setText(race_dir.name)
        if hasattr(self, "passage_review_btn"):
            self.passage_review_btn.setEnabled(True)
        if hasattr(self, "video_supplement_btn"):
            self.video_supplement_btn.setEnabled(True)
        self._refresh_evidence_ui()
        self.statusBar().showMessage(f"当前赛事: {race_dir.name}")

    def _prompt_race_selection(self):
        if self._race_ready:
            return
        self.race_root.mkdir(parents=True, exist_ok=True)
        dialog = NewRaceDialog(
            self,
            base_path=str(self.race_root),
            allow_existing=True,
        )
        if dialog.exec_() != QDialog.Accepted:
            QMessageBox.warning(self, "提示", "必须先选择或创建赛事，系统将退出。")
            self.close()
            return
        race_dir = dialog.selected_race_dir or (
            Path(dialog.result_path) / dialog.result_name
        )
        race_name = race_dir.name
        is_existing = race_dir.exists()
        try:
            race_dir.mkdir(parents=True, exist_ok=True)
            (race_dir / "evidence_photos").mkdir(exist_ok=True)
            (race_dir / "results").mkdir(exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"创建赛事目录失败: {e}")
            self.close()
            return
        if is_existing:
            self._apply_race_config(race_dir)
        else:
            self.output_dir = race_dir.absolute()
            self.config['output_dir'] = str(race_dir)
            self._save_config()
        self._activate_race_dir(race_dir)

    def _setup_logging(self):
        """设置日志重定向到界面"""
        from PyQt5.QtCore import Q_ARG
        
        handler = GuiLogHandler(self._on_log_received)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', '%H:%M:%S'))
        
        # 将 handler 添加到根 logger
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
        # 保存引用以便关闭窗口时安全移除
        self._gui_log_handler = handler
        
        # 同时也确保我们的自定义 logger 也受到影响
        try:
            from .logger import logger as custom_logger
            custom_logger.addHandler(handler)
        except ImportError:
            pass
            
        logger.info("系统初始化完成")

    def _on_log_received(self, msg):
        """处理日志信号 (由 GuiLogHandler 在后台线程调用)"""
        self.log_signal.emit(msg)

    def _on_log_received_ui(self, msg):
        """处理日志更新 (UI 线程)"""
        # 组件可能已被销毁，需安全检查
        if not hasattr(self, 'log_output') or self.log_output is None:
            return
        try:
            import sip
            if sip.isdeleted(self.log_output):
                return
        except Exception:
            pass
        self.log_output.appendPlainText(msg)

    def _init_ui(self):
        """初始化现场视频工作台。"""
        self.setWindowTitle("VideoPipe 视频取证工作台")
        self.setMinimumSize(1200, 768)
        self.resize(1600, 960)
        self._setup_menubar()
        self._init_operator_controls()

        central = QWidget()
        central.setObjectName("live_console")
        central.setStyleSheet("""
            QWidget#live_console { background: #eef2f5; }
            QWidget#workspace_header {
                background: #fbfcfd;
                border-bottom: 1px solid #d8dee6;
            }
            QWidget#capture_bar {
                background: #fbfcfd;
                border-top: 1px solid #d8dee6;
            }
            QLabel#workspace_title { color: #182230; font-size: 17px; font-weight: 700; }
            QLabel#race_name {
                color: #52606d; font-size: 12px; background: #f0f3f6;
                border: 1px solid #d8dee6; border-radius: 3px; padding: 3px 8px;
            }
            QLabel#section_title { color: #253246; font-size: 14px; font-weight: 700; }
            QLabel#capture_state {
                border-radius: 3px; padding: 4px 9px; font-size: 12px; font-weight: 700;
            }
            QLabel#capture_detail { color: #667085; font-size: 12px; }
            QLabel#session_count { color: #344054; font-size: 12px; font-weight: 700; }
            QComboBox#sport_profile_combo {
                min-width: 250px; min-height: 32px; padding: 0 8px;
                background: #ffffff; color: #253246; border: 1px solid #c5ccd6;
                border-radius: 4px; font-size: 12px;
            }
            QComboBox#sport_profile_combo:disabled { background: #eef2f5; color: #7b8794; }
            QPushButton#start_btn {
                min-width: 108px; min-height: 36px; padding: 0 16px;
                background: #247a52; color: white; border: 1px solid #247a52;
                border-radius: 4px; font-size: 13px; font-weight: 700;
            }
            QPushButton#start_btn:hover { background: #1e6846; }
            QPushButton#start_btn:pressed { background: #18573b; }
            QPushButton#stop_btn {
                min-width: 108px; min-height: 36px; padding: 0 16px;
                background: #b54747; color: white; border: 1px solid #b54747;
                border-radius: 4px; font-size: 13px; font-weight: 700;
            }
            QPushButton#stop_btn:hover { background: #9f3d3d; }
            QPushButton#stop_btn:pressed { background: #8a3434; }
            QPushButton#record_btn {
                min-width: 96px; min-height: 34px; padding: 0 12px;
                background: #f8fafb; color: #8a3440; border: 1px solid #d6a3aa;
                border-radius: 4px; font-size: 12px; font-weight: 700;
            }
            QPushButton#record_btn:hover { background: #fdf2f3; border-color: #b96c77; }
            QPushButton#recording_btn {
                min-width: 96px; min-height: 34px; padding: 0 12px;
                background: #a33d4b; color: white; border: 1px solid #a33d4b;
                border-radius: 4px; font-size: 12px; font-weight: 700;
            }
            QPushButton#recording_btn:hover { background: #8a3440; }
            QPushButton#settings_btn {
                min-height: 34px; padding: 0 12px; background: #f8fafb; color: #344054;
                border: 1px solid #c5ccd6; border-radius: 4px; font-size: 12px; font-weight: 600;
            }
            QPushButton#settings_btn:hover { background: #eef2f5; border-color: #98a2b3; }
        """)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        workspace_header = QWidget()
        workspace_header.setObjectName("workspace_header")
        workspace_header.setFixedHeight(50)
        header_layout = QHBoxLayout(workspace_header)
        header_layout.setContentsMargins(14, 0, 14, 0)
        header_layout.setSpacing(12)

        workspace_title = QLabel("视频取证工作台")
        workspace_title.setObjectName("workspace_title")
        header_layout.addWidget(workspace_title)
        self.race_name_label = QLabel("未选择赛事")
        self.race_name_label.setObjectName("race_name")
        header_layout.addWidget(self.race_name_label)
        header_layout.addStretch()

        self.cyclerace_status_label = QLabel("计时源: 等待赛事")
        self.cyclerace_status_label.setStyleSheet("color: #667085; font-size: 12px; font-weight: 600;")
        header_layout.addWidget(self.cyclerace_status_label)
        self.evidence_status_label = QLabel("录像证据: 等待赛事")
        self.evidence_status_label.setStyleSheet(
            "color: #667085; font-size: 12px; font-weight: 600;"
        )
        header_layout.addWidget(self.evidence_status_label)
        self.passage_review_btn = QPushButton("终点核对")
        self.passage_review_btn.setObjectName("settings_btn")
        self.passage_review_btn.setEnabled(False)
        self.passage_review_btn.setToolTip("按号码快速核对普通录像和高速摄像")
        self.passage_review_btn.clicked.connect(self._show_passage_review)
        header_layout.addWidget(self.passage_review_btn)
        self.video_supplement_btn = QPushButton("视频补录")
        self.video_supplement_btn.setObjectName("settings_btn")
        self.video_supplement_btn.setEnabled(False)
        self.video_supplement_btn.setToolTip("记录赛虎没有对应芯片的录像观察")
        self.video_supplement_btn.clicked.connect(self._add_video_supplement)
        header_layout.addWidget(self.video_supplement_btn)
        self.external_clip_btn = QPushButton("导入高速")
        self.external_clip_btn.setObjectName("settings_btn")
        self.external_clip_btn.setEnabled(False)
        self.external_clip_btn.setToolTip("按北京时间 sidecar 导入高速摄像片段")
        self.external_clip_btn.clicked.connect(self._import_external_clips)
        header_layout.addWidget(self.external_clip_btn)

        self.capture_state_label = QLabel("采集未开始")
        self.capture_state_label.setObjectName("capture_state")
        header_layout.addWidget(self.capture_state_label)
        self.conn_status_indicator = QLabel("● 主相机未连接")
        self.conn_status_indicator.setStyleSheet("color: #8a3440; font-size: 13px; font-weight: 700;")
        header_layout.addWidget(self.conn_status_indicator)
        root_layout.addWidget(workspace_header)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(5)
        splitter.setChildrenCollapsible(False)
        splitter.setStyleSheet("QSplitter::handle { background: #e4e9ee; }")

        video_widget = QWidget()
        video_widget.setStyleSheet("background: #ffffff;")
        video_layout = QVBoxLayout(video_widget)
        video_layout.setContentsMargins(10, 8, 10, 8)
        video_layout.setSpacing(7)
        video_header = QHBoxLayout()
        video_title = QLabel("实时视频")
        video_title.setObjectName("section_title")
        video_header.addWidget(video_title)
        video_header.addStretch()
        video_layout.addLayout(video_header)

        self.video_container = QWidget()
        self.video_container.setStyleSheet("background-color: #0f1216;")
        self.video_grid = QGridLayout(self.video_container)
        self.video_grid.setContentsMargins(2, 2, 2, 2)
        self.video_grid.setSpacing(2)
        self._init_video_labels()
        video_layout.addWidget(self.video_container, 1)
        splitter.addWidget(video_widget)

        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.setHandleWidth(5)
        right_splitter.setChildrenCollapsible(False)
        right_splitter.setMinimumWidth(410)
        right_splitter.setStyleSheet("QSplitter::handle { background: #e4e9ee; }")

        self.event_list = EventListWidget(self.database)
        self.event_list.event_selected.connect(self._on_live_event_selected)
        self.event_list.view_screenshot.connect(self._view_screenshot)
        self.event_list.set_source_column_visible(len(self.sources) > 1)
        right_splitter.addWidget(self.event_list)

        self.live_event_review = LiveEventReview(self.database, self.output_dir)
        self.live_event_review.bib_changed.connect(self._on_bib_changed)
        self.live_event_review.void_changed.connect(self._on_void_changed)
        self.live_event_review.next_requested.connect(self._on_review_next_requested)
        right_splitter.addWidget(self.live_event_review)
        right_splitter.setSizes([470, 370])
        splitter.addWidget(right_splitter)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([1050, 550])
        root_layout.addWidget(splitter, 1)
        self.splitter = splitter
        self.right_splitter = right_splitter

        capture_bar = QWidget()
        capture_bar.setObjectName("capture_bar")
        capture_bar.setFixedHeight(54)
        capture_layout = QHBoxLayout(capture_bar)
        capture_layout.setContentsMargins(12, 0, 12, 0)
        capture_layout.setSpacing(10)

        self.start_btn = QPushButton("开始采集")
        self.start_btn.setObjectName("start_btn")
        self.start_btn.clicked.connect(self._toggle_running)
        capture_layout.addWidget(self.start_btn)

        self.capture_detail_label = QLabel("等待开始")
        self.capture_detail_label.setObjectName("capture_detail")
        capture_layout.addWidget(self.capture_detail_label)
        capture_layout.addStretch()

        self.sport_profile_combo = QComboBox()
        self.sport_profile_combo.setObjectName("sport_profile_combo")
        self.sport_profile_combo.setToolTip("选择运动员号码位置和检测规则")
        self.sport_profile_combo.addItem("自行车（车座下 / 背部号码）", "cycling")
        self.sport_profile_combo.addItem("胸前号码（跑步 / 铁三 / 五项 / 轮滑）", "running")
        self.sport_profile_combo.addItem("轮滑（头盔 / 大腿号码）", "speed_skating")
        self._sync_sport_profile_combo()
        self.sport_profile_combo.currentIndexChanged.connect(self._on_sport_profile_changed)
        capture_layout.addWidget(self.sport_profile_combo)

        self.record_btn = QPushButton("开始录像")
        self.record_btn.setObjectName("record_btn")
        self.record_btn.setToolTip("保存网络摄像头原始码流；再次点击停止并封装录像文件")
        self.record_btn.clicked.connect(self._toggle_recording)
        self.record_btn.setEnabled(False)
        capture_layout.addWidget(self.record_btn)

        self.session_count_label = QLabel("本轮新增 0")
        self.session_count_label.setObjectName("session_count")
        capture_layout.addWidget(self.session_count_label)

        self.settings_btn = QPushButton("现场设置")
        self.settings_btn.setObjectName("settings_btn")
        self.settings_btn.setMenu(self._create_live_settings_menu())
        capture_layout.addWidget(self.settings_btn)
        root_layout.addWidget(capture_bar)

        # 日志继续接收后台消息，但不占用现场工作区。
        self.log_output = QPlainTextEdit(self)
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumBlockCount(100)
        self.log_output.hide()

        self.conn_status_label = QLabel("主相机: 未知")
        self.conn_status_label.hide()
        self.statusBar().addPermanentWidget(self.conn_status_label)

        self.stats_status_label = QLabel("本轮新增: 0")
        self.stats_status_label.hide()
        self.statusBar().addPermanentWidget(self.stats_status_label)

        self.ocr_engine_label = QLabel("OCR引擎: Mobile Recognition", self)
        self.ocr_engine_label.hide()
        self._refresh_ocr_engine_badge()

        self.ocr_status_label = QPushButton("OCR: 未初始化")
        self.ocr_status_label.setFlat(True)
        self.ocr_status_label.setCursor(Qt.PointingHandCursor)
        self.ocr_status_label.setStyleSheet("color: #667085; font-size: 11px; border: none; padding: 0 7px;")
        self.ocr_status_label.clicked.connect(self._init_ocr_runtime)
        self.statusBar().addPermanentWidget(self.ocr_status_label)
        if self._yolo_only_mode:
            self._ocr_runtime_state = "disabled"
        self._refresh_ocr_runtime_ui()

        self.recording_status_label = QLabel("录像: 待机")
        self.recording_status_label.setStyleSheet(
            "margin-right: 6px; color: #667085; font-size: 11px;"
        )
        self.recording_status_label.setToolTip("手动录像状态、持续时间和当前文件大小")
        self.statusBar().addPermanentWidget(self.recording_status_label)

        self.live_monitor_label = QLabel("巡检: 待机")
        self.live_monitor_label.setStyleSheet("margin-right: 6px; color: #667085; font-size: 11px;")
        self.live_monitor_label.setToolTip("基于现有检测结果的轻量运行健康状态")
        self.statusBar().addPermanentWidget(self.live_monitor_label)
        self._refresh_live_monitor_badge(time.time())

        self.statusBar().setSizeGripEnabled(False)
        self.statusBar().setContentsMargins(8, 0, 4, 0)
        self.statusBar().showMessage("系统就绪")
        self._set_capture_state(False)

    def _init_operator_controls(self):
        """保留运行逻辑依赖的设置控件，但不放入现场主工作区。"""
        self._operator_control_store = QWidget(self)
        self._operator_control_store.hide()

        self.model_path_input = QLineEdit(self.model_path or "未选择模型", self._operator_control_store)
        self.model_path_input.setReadOnly(True)

        self.test_mode_checkbox = QCheckBox("模拟测试模式", self._operator_control_store)
        self.test_mode_checkbox.setChecked(True)
        self.test_mode_checkbox.stateChanged.connect(self._on_mode_changed)

        self.dedup_checkbox = QCheckBox("号码去重", self._operator_control_store)
        self.dedup_checkbox.setChecked(True)
        self.dedup_checkbox.stateChanged.connect(self._on_dedup_changed)

        self.athlete_filter_checkbox = QCheckBox("名单过滤", self._operator_control_store)
        self.athlete_filter_checkbox.setChecked(True)
        self.athlete_filter_checkbox.stateChanged.connect(self._update_detector_athletes)

    def _create_live_settings_menu(self):
        menu = QMenu(self)

        model_menu = menu.addMenu("检测模型")
        model_menu.addAction("选择模型...", self._browse_model)
        model_menu.addAction("重载模型", self._reload_model)

        preset_menu = menu.addMenu("标定预设")
        preset_menu.addAction("保存为默认方案", lambda: self._save_config_preset("default"))
        preset_menu.addAction("保存为救场方案", lambda: self._save_config_preset("rescue"))
        preset_menu.addSeparator()
        preset_menu.addAction("加载默认方案", lambda: self._load_config_preset("default"))
        preset_menu.addAction("加载救场方案", lambda: self._load_config_preset("rescue"))

        options_menu = menu.addMenu("识别选项")

        def add_checkbox_action(label, checkbox):
            action = QAction(label, self, checkable=True)
            action.setChecked(checkbox.isChecked())
            action.toggled.connect(checkbox.setChecked)
            checkbox.toggled.connect(action.setChecked)
            options_menu.addAction(action)
            return action

        self.test_mode_action = add_checkbox_action("模拟测试模式", self.test_mode_checkbox)
        self.dedup_action = add_checkbox_action("号码去重", self.dedup_checkbox)
        self.athlete_filter_action = add_checkbox_action("名单过滤", self.athlete_filter_checkbox)

        menu.addAction("发枪时间设置...", self._show_start_time_dialog)
        menu.addSeparator()
        self.reset_stats_action = menu.addAction("重置检测统计", self._reset_stats)
        self.clear_records_action = menu.addAction("清空全部事件...", self._clear_all_records)
        self.field_issue_action = menu.addAction("保存诊断问题标记...", self._mark_field_issue)
        self.field_issue_action.setEnabled(False)
        return menu

    def _sync_sport_profile_combo(self):
        combo = getattr(self, 'sport_profile_combo', None)
        if combo is None:
            return
        index = combo.findData(self.sport_profile)
        if index < 0:
            index = combo.findData('cycling')
        combo.blockSignals(True)
        combo.setCurrentIndex(index)
        combo.blockSignals(False)

    def _on_sport_profile_changed(self, index: int):
        combo = getattr(self, 'sport_profile_combo', None)
        if combo is None or index < 0:
            return
        selected = normalize_sport_profile(combo.itemData(index))
        if selected == self.sport_profile:
            return
        if self._running:
            self._sync_sport_profile_combo()
            return

        self.sport_profile = selected
        self.config['sport_profile'] = selected
        self._gate_guard_enabled = True
        self.config['gate_guard_enabled'] = True
        self._athlete_validator_checked = bool(self.shared_athlete_validator)

        for reader in self.readers.values():
            reader.stop()
        self.readers.clear()
        self.detectors.clear()
        self._initialized = False
        self._set_capture_state(False)
        self._save_config()
        logger.info(f"[Main] 赛事类型已切换: {selected}")
        self.statusBar().showMessage(f"赛事类型已切换: {combo.currentText()}")

    def _set_capture_state(self, running: bool):
        if hasattr(self, 'sport_profile_combo'):
            self.sport_profile_combo.setEnabled(not running)
        if running:
            self.capture_state_label.setText("正在采集")
            self.capture_state_label.setStyleSheet(
                "background: #e8f4ed; color: #246144; border: 1px solid #b9d8c6;"
            )
            self._update_capture_connection_detail()
        else:
            self.capture_state_label.setText("采集已停止" if self._initialized else "采集未开始")
            self.capture_state_label.setStyleSheet(
                "background: #f2f4f7; color: #52606d; border: 1px solid #d7dce2;"
            )
            self.capture_detail_label.setText("事件队列和证据核对保持可用" if self._initialized else "等待开始")
            standby_text = "● 主相机待机" if self._initialized else "● 主相机未连接"
            self.conn_status_indicator.setText(standby_text)
            self.conn_status_indicator.setStyleSheet(
                "color: #667085; font-weight: 700; font-size: 13px;"
            )
            for source_id, fps_label in getattr(self, 'cam_fps_labels', {}).items():
                setattr(self, f'_fps_{source_id}', 0.0)
                fps_label.setText("-- FPS")
                fps_label.setStyleSheet(
                    "color: #888; font-size: 12px; border: none; margin-left: 8px;"
                )
            for indicator in getattr(self, 'cam_status_indicators', {}).values():
                indicator.setText("●")
                indicator.setStyleSheet(
                    "color: #667085; font-size: 12px; border: none; margin-left: 8px;"
                )
                indicator.setToolTip("待机" if self._initialized else "未连接")

    def _connected_source_count(self) -> int:
        try:
            from .stream_reader import StreamStatus
        except ImportError:
            from stream_reader import StreamStatus
        return sum(
            1 for reader in self.readers.values()
            if getattr(reader, 'status', None) == StreamStatus.CONNECTED
        )

    def _update_capture_connection_detail(self):
        if not self._running:
            return
        connected = self._connected_source_count()
        total = max(1, len(self.sources))
        if connected >= total:
            self.capture_detail_label.setText(f"已连接 {connected} 路机位")
        elif connected == 0:
            self.capture_detail_label.setText("等待机位连接")
        else:
            self.capture_detail_label.setText(f"已连接 {connected}/{total} 路机位")

    def _on_live_event_selected(self, event: dict):
        self.live_event_review.set_event(event)

    def _on_review_next_requested(self, event_id: int):
        self.event_list.select_next_event(event_id)

    def _refresh_ocr_engine_badge(self):
        """刷新状态栏 OCR 引擎显示。"""
        if not hasattr(self, 'ocr_engine_label'):
            return
        engine = (self.ocr_engine or 'paddleocr').lower()
        if engine == 'paddleocr':
            state = getattr(self, '_ocr_runtime_state', 'idle')
            if state == 'ready':
                state_text = "绿灯"
                state_style = "background:#f6ffed; border:1px solid #b7eb8f; color:#389e0d;"
            elif state == 'loading':
                state_text = "黄灯"
                state_style = "background:#fffbe6; border:1px solid #ffe58f; color:#d48806;"
            elif state == 'failed':
                state_text = "红灯"
                state_style = "background:#fff2f0; border:1px solid #ffccc7; color:#cf1322;"
            else:
                state_text = "灰灯"
                state_style = "background:#fafafa; border:1px solid #d9d9d9; color:#595959;"

            self.ocr_engine_label.setText(f"OCR引擎: PaddleOCR ({state_text})")
            self.ocr_engine_label.setToolTip("当前版本固定使用 PaddleOCR；绿=就绪，黄=启动中，红=失败")
            self.ocr_engine_label.setStyleSheet(
                f"margin-right: 15px; font-weight: bold; font-size: 14px;"
                f"padding: 2px 8px; border-radius: 10px; {state_style}"
            )
        else:
            self.ocr_engine_label.setText(f"OCR引擎: {engine}")
            self.ocr_engine_label.setToolTip(f"当前引擎: {engine}")
            self.ocr_engine_label.setStyleSheet(
                "margin-right: 15px; color: #1677ff; font-weight: bold; font-size: 14px;"
            )

    def _update_live_monitor_sample(self, source_id: int, now_ts: float, athletes: list, bibs: list):
        """记录轻量巡检样本（仅统计，不新增推理）。"""
        if not self._live_monitor_enabled:
            return

        if source_id not in self._live_monitor_samples:
            self._live_monitor_samples[source_id] = deque(maxlen=360)

        athlete_count = len(athletes) if athletes else 0
        bib_count = len(bibs) if bibs else 0
        split_count = 0
        if athletes:
            for athlete in athletes:
                if isinstance(athlete, dict) and athlete.get('split_from_track') is not None:
                    split_count += 1

        mismatch_flag = 1 if (bib_count >= 2 and athlete_count <= 1) else 0
        bib_only_flag = 1 if (bib_count > 0 and athlete_count == 0) else 0
        fps_val = float(getattr(self, f'_fps_{source_id}', 0.0))
        thread = self.video_threads.get(source_id)
        frame_metrics = dict(getattr(thread, 'last_frame_metrics', {}) or {})
        frame_metrics['display_fps'] = fps_val

        self._live_monitor_samples[source_id].append(
            (
                now_ts,
                athlete_count,
                bib_count,
                split_count,
                mismatch_flag,
                bib_only_flag,
                fps_val,
                int(frame_metrics.get('participants', 0) or 0),
                int(frame_metrics.get('track_fragments_merged', 0) or 0),
                int(frame_metrics.get('identity_ambiguities', 0) or 0),
                int(frame_metrics.get('queue_depth', 0) or 0),
                int(frame_metrics.get('dropped_frames', 0) or 0),
                int(frame_metrics.get('consumer_skipped_frames', 0) or 0),
                int(frame_metrics.get('discarded_frames', 0) or 0),
                float(frame_metrics.get('drop_rate', 0.0) or 0.0),
                float(frame_metrics.get('capture_fps', 0.0) or 0.0),
                float(frame_metrics.get('inference_fps', fps_val) or 0.0),
                float(frame_metrics.get('queue_latency_ms', 0.0) or 0.0),
                float(frame_metrics.get('end_to_end_latency_ms', 0.0) or 0.0),
            )
        )

        cutoff_ts = now_ts - max(8.0, float(self._live_monitor_window_seconds))
        samples = self._live_monitor_samples[source_id]
        while samples and samples[0][0] < cutoff_ts:
            samples.popleft()

    def _evaluate_live_monitor_source(self, source_id: int, now_ts: float) -> Dict[str, Any]:
        """评估单机位巡检状态。"""
        samples = self._live_monitor_samples.get(source_id)
        if not samples:
            return {
                'ready': False,
                'level': 'idle',
                'label': f"机位{source_id + 1}:采样中",
                'note': '采样中'
            }

        n = len(samples)
        mismatch_ratio = float(sum(item[4] for item in samples)) / max(1, n)
        bib_only_ratio = float(sum(item[5] for item in samples)) / max(1, n)
        split_hits = int(sum(1 for item in samples if item[3] > 0))

        display_fps_values = [float(item[6]) for item in samples if float(item[6]) > 0.01]
        capture_fps_values = [
            float(item[15]) for item in samples
            if len(item) > 15 and float(item[15]) > 0.01
        ]
        inference_fps_values = [
            float(item[16]) if len(item) > 16 else float(item[6])
            for item in samples
            if (len(item) > 16 and float(item[16]) > 0.01) or float(item[6]) > 0.01
        ]
        fps_values = inference_fps_values
        avg_fps = float(sum(fps_values) / max(1, len(fps_values))) if fps_values else 0.0
        capture_fps = (
            float(sum(capture_fps_values) / len(capture_fps_values))
            if capture_fps_values else 0.0
        )
        display_fps = (
            float(sum(display_fps_values) / len(display_fps_values))
            if display_fps_values else 0.0
        )
        latest = samples[-1]
        participant_count = int(latest[7]) if len(latest) > 7 else 0
        fragment_merges = sum(int(item[8]) for item in samples if len(item) > 8)
        identity_ambiguities = sum(int(item[9]) for item in samples if len(item) > 9)
        queue_depth_max = max(
            (int(item[10]) for item in samples if len(item) > 10),
            default=0,
        )
        dropped_frames = max(
            (int(item[11]) for item in samples if len(item) > 11),
            default=0,
        )
        consumer_skipped_frames = max(
            (int(item[12]) for item in samples if len(item) > 12),
            default=0,
        )
        discarded_frames = max(
            (int(item[13]) for item in samples if len(item) > 13),
            default=dropped_frames + consumer_skipped_frames,
        )
        drop_rate = max(
            (float(item[14]) for item in samples if len(item) > 14),
            default=0.0,
        )
        queue_latency_ms = max(
            (float(item[17]) for item in samples if len(item) > 17),
            default=0.0,
        )
        end_to_end_latency_ms = max(
            (float(item[18]) for item in samples if len(item) > 18),
            default=0.0,
        )

        severe = (avg_fps > 0 and avg_fps < float(self._live_monitor_min_fps) * 0.80) or \
                 mismatch_ratio >= 0.30 or bib_only_ratio >= 0.22 or \
                 drop_rate > float(getattr(self, '_live_monitor_severe_drop_rate', 0.15))
        warning = (avg_fps > 0 and avg_fps < float(self._live_monitor_min_fps)) or \
                  mismatch_ratio >= 0.18 or bib_only_ratio >= 0.12 or \
                  drop_rate > float(getattr(self, '_live_monitor_warn_drop_rate', 0.05))

        if severe:
            level = 'danger'
            note = '告警'
        elif warning:
            level = 'warn'
            note = '注意'
        else:
            level = 'ok'
            note = '正常'

        detail_parts = []
        if capture_fps > 0:
            detail_parts.append(f"采集={capture_fps:.1f}FPS")
        if avg_fps > 0:
            detail_parts.append(f"推理={avg_fps:.1f}FPS")
        if display_fps > 0:
            detail_parts.append(f"显示={display_fps:.1f}FPS")
        if mismatch_ratio > 0.0:
            detail_parts.append(f"并排漏检风险={mismatch_ratio:.0%}")
        if bib_only_ratio > 0.0:
            detail_parts.append(f"号码孤立={bib_only_ratio:.0%}")
        if split_hits > 0:
            detail_parts.append(f"大框拆分={split_hits}")
        detail_parts.append(f"身份={participant_count}")
        detail_parts.append(f"轨迹合并={fragment_merges}")
        detail_parts.append(f"身份歧义={identity_ambiguities}")
        detail_parts.append(f"队列峰值={queue_depth_max}")
        detail_parts.append(f"丢帧={discarded_frames} ({drop_rate:.1%})")
        if queue_latency_ms > 0:
            detail_parts.append(f"队列延迟={queue_latency_ms:.0f}ms")
        if end_to_end_latency_ms > 0:
            detail_parts.append(f"总延迟={end_to_end_latency_ms:.0f}ms")

        detail_text = " | ".join(detail_parts) if detail_parts else "采样正常"
        return {
            'ready': True,
            'level': level,
            'label': f"机位{source_id + 1}:{note}",
            'note': note,
            'avg_fps': avg_fps,
            'capture_fps': capture_fps,
            'inference_fps': avg_fps,
            'display_fps': display_fps,
            'mismatch_ratio': mismatch_ratio,
            'bib_only_ratio': bib_only_ratio,
            'split_hits': split_hits,
            'participant_count': participant_count,
            'fragment_merges': fragment_merges,
            'identity_ambiguities': identity_ambiguities,
            'queue_depth_max': queue_depth_max,
            'dropped_frames': dropped_frames,
            'consumer_skipped_frames': consumer_skipped_frames,
            'discarded_frames': discarded_frames,
            'drop_rate': drop_rate,
            'queue_latency_ms': queue_latency_ms,
            'end_to_end_latency_ms': end_to_end_latency_ms,
            'detail': detail_text,
        }

    def _refresh_live_monitor_badge(self, now_ts: Optional[float] = None):
        """刷新轻量巡检状态栏。"""
        if not hasattr(self, 'live_monitor_label'):
            return

        if now_ts is None:
            now_ts = time.time()

        if not self._running:
            self.live_monitor_label.setText("巡检: 待机")
            self.live_monitor_label.setStyleSheet(
                "margin-right: 6px; color: #667085; font-size: 11px;"
            )
            self.live_monitor_label.setToolTip("采集开始后显示运行巡检状态")
            self._live_monitor_last_summary = "待机"
            return

        if not self._live_monitor_enabled:
            self.live_monitor_label.setText("巡检: 已关闭")
            self.live_monitor_label.setStyleSheet("margin-right: 6px; color: #667085; font-size: 11px;")
            self.live_monitor_label.setToolTip("轻量巡检已关闭")
            self._live_monitor_last_summary = ""
            return

        source_ids = sorted(self.video_labels.keys()) if hasattr(self, 'video_labels') else [0]
        evaluations = [self._evaluate_live_monitor_source(sid, now_ts) for sid in source_ids]
        ready_evals = [item for item in evaluations if item.get('ready')]

        if not ready_evals:
            self.live_monitor_label.setText("巡检: 采样中")
            self.live_monitor_label.setStyleSheet("margin-right: 6px; color: #667085; font-size: 11px;")
            self.live_monitor_label.setToolTip("轻量巡检正在积累样本")
            self._live_monitor_last_summary = "采样中"
            return

        level_rank = {'ok': 0, 'warn': 1, 'danger': 2}
        worst = max(ready_evals, key=lambda item: level_rank.get(item.get('level', 'ok'), 0))

        if worst.get('level') == 'danger':
            color = "#cf1322"
            title = "巡检: 告警"
        elif worst.get('level') == 'warn':
            color = "#d48806"
            title = "巡检: 注意"
        else:
            color = "#389e0d"
            title = "巡检: 正常"

        compact = " ".join([item.get('label', '') for item in evaluations])
        self.live_monitor_label.setText(f"{title} | {compact}")
        self.live_monitor_label.setStyleSheet(
            f"margin-right: 6px; color: {color}; font-weight: 600; font-size: 11px;"
        )
        self.live_monitor_label.setToolTip(
            "轻量巡检（滚动窗口）\n" + "\n".join([
                f"{item.get('label', '')} | {item.get('detail', '采样中')}" for item in evaluations
            ])
        )

        self._live_monitor_last_summary = worst.get('note', '正常')

        if worst.get('level') == 'danger':
            warn_source = 0
            if source_ids:
                try:
                    warn_source = source_ids[evaluations.index(worst)]
                except Exception:
                    warn_source = source_ids[0]

            last_warn = float(self._live_monitor_last_warn_ts.get(warn_source, 0.0))
            if now_ts - last_warn >= float(self._live_monitor_warn_cooldown_seconds):
                logger.warning(f"[Main] 轻量巡检告警({warn_source + 1}): {worst.get('detail', '异常')}")
                self._live_monitor_last_warn_ts[warn_source] = now_ts

    def _init_video_labels(self):
        """初始化或更新视频显示槽位 (支持多机位)"""
        # 清空现有布局
        for i in reversed(range(self.video_grid.count())):
            widget = self.video_grid.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        self.video_labels.clear()
        self.finish_line_checkboxes.clear()

        # 清空状态指示器和FPS标签字典
        if hasattr(self, 'cam_status_indicators'):
            self.cam_status_indicators.clear()
        if hasattr(self, 'cam_fps_labels'):
            self.cam_fps_labels.clear()

        # 动态创建机位槽位 (根据 sources 数量)
        num_sources = len(self.sources)
        cols = 2 if num_sources > 1 else 1
        
        for i in range(num_sources):
            # 每个机位一个容器
            container = QWidget()
            container.setStyleSheet("background-color: #000; border: 1px solid #333;")
            layout = QVBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)

            # 顶部控制条 (包含状态指示器和开关)
            ctrl_strip = QWidget()
            ctrl_strip.setFixedHeight(30)
            ctrl_strip.setStyleSheet("background-color: #222; border-bottom: 1px solid #333;")
            ctrl_layout = QHBoxLayout(ctrl_strip)
            ctrl_layout.setContentsMargins(10, 0, 10, 0)

            cam_label = QLabel(f"机位 {i+1}")
            cam_label.setStyleSheet("color: #aaa; font-size: 14px; font-weight: bold; border: none;")
            ctrl_layout.addWidget(cam_label)

            # 连接状态指示器
            status_indicator = QLabel("●")
            status_indicator.setStyleSheet("color: #666; font-size: 12px; border: none; margin-left: 8px;")
            status_indicator.setToolTip("连接状态")
            ctrl_layout.addWidget(status_indicator)
            if not hasattr(self, 'cam_status_indicators'):
                self.cam_status_indicators = {}
            self.cam_status_indicators[i] = status_indicator

            # FPS 显示
            fps_label = QLabel("-- FPS")
            fps_label.setStyleSheet("color: #888; font-size: 12px; border: none; margin-left: 8px;")
            ctrl_layout.addWidget(fps_label)
            if not hasattr(self, 'cam_fps_labels'):
                self.cam_fps_labels = {}
            self.cam_fps_labels[i] = fps_label

            ctrl_layout.addStretch()
            
            cb = QCheckBox("启用终点线")
            cb.setStyleSheet("color: #eee; font-size: 13px; border: none;")
            is_enabled = self.finish_line_enabled.get(i, False)
            cb.setChecked(is_enabled)
            cb.stateChanged.connect(lambda state, idx=i: self._on_finish_line_toggle(idx, state))
            ctrl_layout.addWidget(cb)
            self.finish_line_checkboxes[i] = cb
            
            roi_cb = QCheckBox("启用 ROI")
            roi_cb.setStyleSheet("color: #eee; font-size: 13px; border: none;")
            is_roi_enabled = self.roi_enabled.get(i, False)
            roi_cb.setChecked(is_roi_enabled)
            roi_cb.stateChanged.connect(lambda state, idx=i: self._on_roi_toggle(idx, state))
            ctrl_layout.addWidget(roi_cb)
            # 存储 ROI 复选框以便后续操作
            if not hasattr(self, 'roi_checkboxes'):
                self.roi_checkboxes = {}
            self.roi_checkboxes[i] = roi_cb
            
            layout.addWidget(ctrl_strip)

            # 视频显示标签
            label = InteractiveVideoLabel()
            label.set_original_size(1920, 1080)
            label.setMinimumSize(400, 225)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("border: none; color: #666;")
            label.setText(f"机位 {i+1} 未连接")
            
            # 明确设置是否显示
            label.set_show_line(is_enabled)
            label.set_show_roi(is_roi_enabled)
            
            # 如果启用了终点线，则连接绘图信号
            if is_enabled:
                label.line_changed.connect(self._on_line_changed)
                label.set_line_points(self.line_pt1[0], self.line_pt1[1], self.line_pt2[0], self.line_pt2[1])
            
            # 如果启用了 ROI，则连接绘图信号
            if is_roi_enabled:
                label.roi_changed.connect(self._on_roi_changed)
                label.set_roi_points(self.roi_points)
            
            if not is_enabled and not is_roi_enabled:
                label.setCursor(Qt.ArrowCursor)
                
            layout.addWidget(label, 1)
            
            row = i // cols
            col = i % cols
            self.video_grid.addWidget(container, row, col)
            self.video_labels[i] = label  # source_id -> label

    def _on_finish_line_toggle(self, source_id: int, state: int):
        """处理终点线开关切换"""
        is_enabled = state == Qt.Checked
        self.finish_line_enabled[source_id] = is_enabled
        
        # 如果开启了终点线，关闭 ROI 模式（或者互斥，根据需求，这里暂不互斥，但 UI 上切换模式）
        # 终点线模式和 ROI 模式共用 CrossCursor
        
        # 保存配置
        self.config['finish_line_enabled'] = self.finish_line_enabled
        self._save_config()
        
        # 更新显示标签状态
        if source_id in self.video_labels:
            label = self.video_labels[source_id]
            label.set_show_line(is_enabled)
            if is_enabled:
                try: label.line_changed.connect(self._on_line_changed)
                except: pass
                label.set_line_points(self.line_pt1[0], self.line_pt1[1], self.line_pt2[0], self.line_pt2[1])
            else:
                try: label.line_changed.disconnect(self._on_line_changed)
                except: pass
            label.update()
            
        # 同步到检测器
        if source_id in self.detectors:
            detector = self.detectors[source_id]
            if is_enabled:
                detector.set_finish_line(self.line_pt1, self.line_pt2)
            else:
                detector.set_finish_line(None, None)
        
        status = "启用" if is_enabled else "禁用"
        self.statusBar().showMessage(f"机位 {source_id+1} 终点线已{status}")

    def _init_shared_athlete_validator(self):
        """Load the optional event-only bicycle validator on CPU."""
        if self._athlete_validator_checked:
            return
        self._athlete_validator_checked = True

        if self.sport_profile != "cycling":
            logger.info(f"[Main] {self.sport_profile} profile does not require bicycle validation")
            return

        if not bool(self.config.get("athlete_validator_enabled", True)):
            logger.info("[Main] 运动员二次校验已禁用")
            return

        configured_validator_path = str(
            self.config.get("athlete_validator_model_path") or ""
        ).strip()
        if not configured_validator_path and self.shared_model is not None:
            names = getattr(self.shared_model, "names", None) or {}
            class_names = names.values() if isinstance(names, dict) else names
            if any(str(name).strip().lower() == "bicycle" for name in class_names):
                logger.info(
                    "[Main] 主模型已包含 bicycle 类，直接复用 YOLO11 人车证据，"
                    "不加载二次校验模型"
                )
                return

        validator_path = resolve_athlete_validator_model(
            configured_validator_path,
            [Path.cwd(), Path(__file__).resolve().parent.parent],
        )
        if validator_path is None:
            logger.warning(
                "[Main] 未找到可选的 YOLO11 自行车二次校验模型，"
                "非号码事件将使用主模型证据"
            )
            return

        try:
            from ultralytics import YOLO
            import numpy as np

            validator = YOLO(str(validator_path))
            validator.predict(
                source=np.zeros((320, 320, 3), dtype=np.uint8),
                imgsz=320,
                conf=0.20,
                verbose=False,
                device="cpu",
            )
            self.shared_athlete_validator = validator
            logger.info(f"[Main] 运动员二次校验模型已就绪: {validator_path.name} (CPU, event-only)")
        except Exception as exc:
            self.shared_athlete_validator = None
            logger.warning(f"[Main] 运动员二次校验模型加载失败，保持原有放行策略: {exc}")

    def _init_ocr_runtime(self):
        """Start recognition-only OCR in a low-priority child process."""
        if self._yolo_only_mode:
            self._ocr_runtime_state = "disabled"
            self._refresh_ocr_runtime_ui()
            return
        if not self._race_ready or self.ocr_manager is None:
            QTimer.singleShot(100, self._init_ocr_runtime)
            return
        if self.ocr_manager.runtime_state in {"loading", "ready"}:
            self._ocr_runtime_state = self.ocr_manager.runtime_state
            self._refresh_ocr_runtime_ui()
            return

        started = self.ocr_manager.start_process_runtime(
            cpu_threads=max(1, min(2, int(self.config.get("ocr_cpu_threads", 1)))),
        )
        self._ocr_runtime_state = self.ocr_manager.runtime_state
        self._refresh_ocr_runtime_ui()
        if started and not self._ocr_poll_timer.isActive():
            self._ocr_poll_timer.start()

    def _poll_ocr_runtime(self):
        if self._yolo_only_mode or self.ocr_manager is None:
            return
        self.ocr_manager.poll_process_results()
        state = self.ocr_manager.runtime_state
        if state != self._ocr_runtime_state:
            self._ocr_runtime_state = state
            self._refresh_ocr_runtime_ui()

    def _refresh_ocr_runtime_ui(self):
        self._refresh_ocr_engine_badge()
        if not hasattr(self, "ocr_status_label"):
            return
        state = self._ocr_runtime_state
        if state == "ready":
            text, color, tooltip = "OCR: 已就绪", "#52c41a", "独立轻量识别进程已就绪"
        elif state == "loading":
            text, color, tooltip = "OCR: 正在启动...", "#faad14", "正在子进程加载移动识别模型"
        elif state == "disabled":
            text, color, tooltip = "OCR: 已关闭", "#667085", "当前为仅视频检测模式，事件和证据保存不受影响"
        elif state == "failed":
            text, color, tooltip = "OCR: 启动失败 (点击重试)", "#ff4d4f", "OCR失败不影响检测和事件保存"
        else:
            text, color, tooltip = "OCR: 未启动 (点击启动)", "#666", "启动独立轻量识别进程"
        self.ocr_status_label.setText(text)
        self.ocr_status_label.setStyleSheet(
            f"color: {color}; font-weight: 600; font-size: 11px; border: none; "
            "text-align: left; padding: 0 7px;"
        )
        self.ocr_status_label.setToolTip(tooltip)
        self.ocr_status_label.setEnabled(state != "disabled")
        self.ocr_status_label.setCursor(
            Qt.ArrowCursor if state == "disabled" else Qt.PointingHandCursor
        )

    def _on_ocr_event_done(self, event_id, result):
        """单个 OCR 任务完成回调"""
        # 刷新事件列表中的特定行
        self.event_saved_signal.emit(event_id)

    def _on_event_saved_ui(self, event_id):
        """事件保存后的 UI 刷新 (UI 线程)"""
        # Prefer incremental refresh so OCR result updates do not redraw the full list.
        if hasattr(self.event_list, '_check_new_events'):
            self.event_list._check_new_events()
        else:
            self.event_list.refresh_data()

    def _refresh_evidence_ui(self):
        timeline = getattr(self, "video_timeline_store", None)
        recording = getattr(self, "recording_manager", None)
        recording_active = bool(recording and recording.is_recording)
        import_active = getattr(self, "_external_clip_import_thread", None) is not None

        if not getattr(self, "_race_ready", False):
            status_text = "录像证据: 等待赛事"
        elif timeline is None:
            status_text = "录像证据: 时间线不可用"
        else:
            segments = timeline.segments()
            high_speed_count = sum(
                segment.clock_source == EXTERNAL_CLOCK_SOURCE
                for segment in segments
            )
            standard_count = len(segments) - high_speed_count
            status_text = (
                f"录像证据: 普通 {standard_count} 段 | 高速 {high_speed_count} 段"
            )
            if recording_active:
                status_text += " | 普通录制中"
            if import_active:
                status_text += " | 高速导入中"

        label = getattr(self, "evidence_status_label", None)
        if label is not None:
            label.setText(status_text)
            label.setToolTip(
                "普通录像使用 VideoPipe 系统时钟；"
                "高速摄像使用北京时间 sidecar；两者仅作为辅助证据。"
            )

        can_import = bool(
            getattr(self, "_race_ready", False)
            and timeline is not None
            and getattr(self, "passage_event_store", None) is not None
            and not recording_active
            and not import_active
        )
        for control_name in ("external_clip_btn", "external_clip_action"):
            control = getattr(self, control_name, None)
            if control is not None:
                control.setEnabled(can_import)

    def _load_video_timeline(self):
        self.video_timeline_store = None
        try:
            store = VideoTimelineStore(self.output_dir / "video_timeline.jsonl")
            recovered_open_segments = store.recover_open_segments()
        except VideoTimelineError as exc:
            self._recording_timeline_warning = f"录像时间线不可用: {exc}"
            logger.error("[Main] 加载录像时间线失败: %s", exc)
            self.statusBar().showMessage(f"录像时间线不可用: {exc}")
            self._refresh_recording_ui()
            self._refresh_evidence_ui()
            return
        self.video_timeline_store = store
        self._recording_timeline_warning = ""
        self._refresh_recording_ui()
        self._refresh_evidence_ui()
        if store.recovered_incomplete_tail:
            logger.warning("[Main] 已恢复录像时间线未完整尾部")
        if recovered_open_segments:
            logger.warning(
                "[Main] 已收尾 %s 个上次异常退出遗留的录像段",
                recovered_open_segments,
            )

    def _start_passage_receiver(self):
        MainWindow._stop_passage_receiver(self, reset_status=False)
        self._latest_passage_event = None
        if self._timing_provider == "racetiger":
            self._start_racetiger_source()
            return
        journal_path = self.output_dir / "cyclerace_passage_events.jsonl"
        try:
            store = PassageEventStore(journal_path)
        except Exception as exc:
            self._set_passage_receiver_status(
                "CycleRace: 存储失败",
                "#b54747",
                str(exc),
            )
            logger.exception("[Main] 加载 CycleRace passage 存储失败: %s", exc)
            return
        self.passage_event_store = store
        recovered = "；已恢复未完整尾部" if store.recovered_incomplete_tail else ""
        if not self._passage_receiver_enabled:
            self._set_passage_receiver_status(
                "CycleRace: 未启用",
                "#667085",
                f"当前赛事未启用 CycleRace passage 接收；已保存 {len(store)} 条{recovered}",
            )
            return
        if not (1 <= self._passage_receiver_port <= 65535):
            self._set_passage_receiver_status(
                "CycleRace: 配置错误",
                "#b54747",
                f"无效监听端口: {self._passage_receiver_port}",
            )
            logger.error("[Main] CycleRace passage 接收端口无效: %s", self._passage_receiver_port)
            return

        receiver = None
        try:
            receiver = PassageEventReceiver(
                self._passage_receiver_host,
                self._passage_receiver_port,
                store,
                on_accepted=self._on_passage_received,
            )
            receiver.start()
            self.passage_receiver = receiver
            self._set_passage_receiver_status(
                f"CycleRace: 监听 {receiver.listen_port}",
                "#247a52",
                f"{self._passage_receiver_host}:{receiver.listen_port}/api/v1/passage-events"
                f"；已保存 {len(store)} 条{recovered}",
            )
        except Exception as exc:
            if receiver is not None:
                receiver.stop()
            self.passage_receiver = None
            self._set_passage_receiver_status(
                "CycleRace: 监听失败",
                "#b54747",
                str(exc),
            )
            logger.exception("[Main] 启动 CycleRace passage 接收失败: %s", exc)

    def _start_racetiger_source(self):
        journal_path = self.output_dir / "racetiger_passage_events.jsonl"
        try:
            store = PassageEventStore(journal_path)
            self.passage_event_store = store
            missing = [
                name
                for name, value in (
                    ("racetiger_base_url", self._racetiger_base_url),
                    ("racetiger_pc", self._racetiger_pc),
                    ("racetiger_rid", self._racetiger_rid),
                    ("racetiger_token", self._racetiger_token),
                )
                if not value
            ]
            if missing:
                self._set_passage_receiver_status(
                    "赛虎: 未配置",
                    "#b54747",
                    "请在赛事 config.json 中配置: " + ", ".join(missing),
                )
                return
            client = RaceTigerClient(
                self._racetiger_base_url,
                self._racetiger_token,
                pc=self._racetiger_pc,
                rid=self._racetiger_rid,
            )
            source = RaceTigerSource(
                client,
                store,
                race_id=f"racetiger:{self.output_dir.name}",
                poll_interval_seconds=self._racetiger_poll_interval,
                on_event=self._on_passage_received,
                on_status=self._on_racetiger_status,
            )
            self.racetiger_source = source
            source.start()
            self._set_passage_receiver_status(
                f"赛虎: 正在读取 {len(store)} 条",
                "#247a52",
                f"POST {self._racetiger_base_url}/Dif/*；已保存 {len(store)} 条",
            )
        except Exception as exc:
            self.passage_event_store = None
            self._set_passage_receiver_status("赛虎: 启动失败", "#b54747", str(exc))
            logger.exception("[Main] RaceTiger source start failed: %s", exc)

    def _on_racetiger_status(self, status: RaceTigerStatus):
        self.timing_status_signal.emit(status)

    def _on_timing_status_ui(self, status):
        if isinstance(status, RaceTigerStatus):
            color = "#247a52" if status.state == "ok" else "#b54747"
            self._set_passage_receiver_status(status.message, color, status.message)
            return
        self._set_passage_receiver_status(str(status), "#667085", str(status))

    def _stop_passage_receiver(self, *, reset_status=True):
        receiver = getattr(self, "passage_receiver", None)
        racetiger_source = getattr(self, "racetiger_source", None)
        self.passage_receiver = None
        self.racetiger_source = None
        self.passage_event_store = None
        if receiver is not None:
            try:
                receiver.stop()
            except Exception as exc:
                logger.warning("[Main] 停止 CycleRace passage 接收失败: %s", exc)
        if racetiger_source is not None:
            try:
                racetiger_source.stop()
            except Exception as exc:
                logger.warning("[Main] Stop RaceTiger source failed: %s", exc)
        if reset_status:
            MainWindow._set_passage_receiver_status(
                self,
                "CycleRace: 等待赛事",
                "#667085",
                "选择赛事后开始监听",
            )

    def _on_passage_received(self, event):
        self.passage_received_signal.emit(event)

    def _on_passage_received_ui(self, event):
        self._latest_passage_event = event
        store = self.passage_event_store
        count = len(store) if store is not None else 0
        identity = event.bib.strip() or event.chip_id.strip() or "未知"
        timeline = self.video_timeline_store
        if timeline is None:
            video_status = "录像时间线不可用"
        else:
            lookup = timeline.locate_passage(
                event.timeline_timestamp_ms,
                clock_offset_ms=self._passage_clock_offset_ms,
                pre_roll_ms=self._passage_video_preroll_ms,
                race_id=event.race_id,
            )
            video_status = lookup_status_text(lookup)
        self._set_passage_receiver_status(
            f"CycleRace: 已接收 {count}",
            "#247a52",
            f"最近通过: {identity}；组别 {event.group_id}；圈次 {event.lap}；"
            f"revision {event.revision}；{video_status}；"
            f"时钟偏移 {self._passage_clock_offset_ms:+d} ms",
        )
        provider_label = (
            "赛虎"
            if getattr(self, "_timing_provider", "cyclerace") == "racetiger"
            else "CycleRace"
        )
        self.statusBar().showMessage(
            f"收到 {provider_label} 通过记录: {identity}",
            3000,
        )
        dialog = self.passage_review_dialog
        if dialog is not None:
            dialog.refresh()
        logger.info(
            "[Main] 收到 CycleRace passage: event_id=%s, bib=%s, sequence=%s, revision=%s",
            event.event_id,
            event.bib,
            event.sequence,
            event.revision,
        )

    def _set_passage_receiver_status(self, text, color, tooltip):
        if getattr(self, "_timing_provider", "cyclerace") == "racetiger" and str(text).startswith("CycleRace"):
            text = str(text).replace("CycleRace", "赛虎", 1)
        label = getattr(self, "cyclerace_status_label", None)
        if label is None:
            return
        label.setText(text)
        label.setStyleSheet(f"color: {color}; font-size: 12px; font-weight: 600;")
        label.setToolTip(tooltip)

    def _show_passage_review(self):
        if not self._race_ready:
            QMessageBox.warning(self, "提示", "请先打开赛事。")
            return
        if self.passage_event_store is None:
            QMessageBox.warning(self, "无法打开", "CycleRace passage 存储不可用。")
            return
        if self.video_timeline_store is None:
            QMessageBox.warning(self, "无法打开", "当前赛事的录像时间线不可用。")
            return

        dialog = self.passage_review_dialog
        if dialog is not None:
            dialog.refresh()
            if dialog.isMinimized():
                dialog.showNormal()
            else:
                dialog.show()
            dialog.raise_()
            dialog.activateWindow()
            return

        dialog = PassageReviewDialog(
            self.passage_event_store,
            self.video_timeline_store,
            self,
            clock_offset_ms=self._passage_clock_offset_ms,
            pre_roll_ms=self._passage_video_preroll_ms,
            open_location=self._open_passage_location,
        )
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        dialog.finished.connect(
            lambda _result, current=dialog: self._on_passage_review_finished(current)
        )
        self.passage_review_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _add_video_supplement(self):
        if not self._race_ready:
            QMessageBox.warning(self, "提示", "请先打开赛事。")
            return
        store = getattr(self, "video_supplement_store", None)
        if store is None:
            QMessageBox.warning(self, "无法补录", "当前赛事的视频补录存储不可用。")
            return
        bib, accepted = QInputDialog.getText(
            self,
            "视频补录",
            "号码或临时编号：",
        )
        if not accepted:
            return
        observed_at, accepted = QInputDialog.getText(
            self,
            "视频补录",
            "通过北京时间（YYYY-MM-DD HH:MM:SS.mmm）：",
        )
        if not accepted:
            return
        observed_at_ms = parse_beijing_datetime(observed_at)
        if observed_at_ms is None:
            QMessageBox.warning(self, "补录失败", "北京时间格式无效。")
            return
        note, accepted = QInputDialog.getText(self, "视频补录", "备注（可选）：")
        if not accepted:
            return
        try:
            item = store.append(
                bib=bib,
                observed_at_ms=observed_at_ms,
                note=note,
            )
        except (OSError, TypeError, ValueError) as error:
            QMessageBox.warning(self, "补录失败", str(error))
            return
        message = f"已保存视频补录：{item.bib or '未知'}，不影响赛虎正式成绩。"
        self.statusBar().showMessage(message, 5000)
        logger.info("[Main] Video supplement saved: id=%s, bib=%s", item.supplement_id, item.bib)

    def _on_passage_review_finished(self, dialog):
        if dialog is not self.passage_review_dialog:
            return
        if dialog.clock_offset_ms != self._passage_clock_offset_ms:
            self._passage_clock_offset_ms = dialog.clock_offset_ms
            self.config["passage_clock_offset_ms"] = self._passage_clock_offset_ms
            self._save_config()
        self.passage_review_dialog = None

    def _close_passage_review(self):
        dialog = self.passage_review_dialog
        if dialog is None:
            return
        dialog.close()
        if self.passage_review_dialog is dialog:
            self.passage_review_dialog = None

    def _import_external_clips(self):
        thread = getattr(self, "_external_clip_import_thread", None)
        if thread is not None and thread.isRunning():
            QMessageBox.information(self, "正在导入", "高速摄像片段仍在验证中。")
            return
        if not self._race_ready:
            QMessageBox.warning(self, "提示", "请先打开赛事。")
            return
        manager = getattr(self, "recording_manager", None)
        if manager is not None and manager.is_recording:
            QMessageBox.warning(
                self,
                "无法导入",
                "请先停止当前录像，再导入高速摄像片段。",
            )
            return
        if self.video_timeline_store is None or self.passage_event_store is None:
            QMessageBox.warning(
                self,
                "无法导入",
                "当前赛事的 CycleRace passage 或录像时间线不可用。",
            )
            return

        sidecar_path, _ = QFileDialog.getOpenFileName(
            self,
            "导入高速摄像片段",
            str(self.output_dir),
            "JSON sidecar (*.json);;All files (*.*)",
        )
        if not sidecar_path:
            return
        try:
            race_id = race_id_from_passage_store(self.passage_event_store)
        except (ExternalClipImportError, VideoTimelineError) as error:
            QMessageBox.warning(self, "导入失败", str(error))
            return
        self._begin_external_clip_import(sidecar_path, race_id)

    def _begin_external_clip_import(self, sidecar_path: str, race_id: str):
        timeline_store = self.video_timeline_store
        passage_store = self.passage_event_store
        if timeline_store is None or passage_store is None:
            QMessageBox.warning(self, "导入失败", "当前赛事存储已不可用。")
            return

        progress = QProgressDialog(
            "正在验证高速摄像片段...",
            "取消",
            0,
            0,
            self,
        )
        progress.setWindowTitle("导入高速摄像片段")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)

        thread = ExternalClipProbeThread(sidecar_path, race_id, self)
        self._external_clip_import_thread = thread
        self._external_clip_import_progress = progress
        self._external_clip_import_context = {
            "race_dir": Path(self.output_dir).expanduser().absolute(),
            "race_id": str(race_id),
            "timeline_store": timeline_store,
            "passage_store": passage_store,
        }
        self._refresh_evidence_ui()

        progress.canceled.connect(thread.requestInterruption)
        thread.progress.connect(self._on_external_clip_import_progress)
        thread.completed.connect(self._on_external_clip_import_verified)
        thread.failed.connect(self._on_external_clip_import_failed)
        thread.cancelled.connect(self._on_external_clip_import_cancelled)
        thread.finished.connect(self._on_external_clip_import_finished)
        progress.show()
        thread.start()

    def _on_external_clip_import_progress(self, completed, total, video_path):
        progress = self._external_clip_import_progress
        if progress is None:
            return
        progress.setRange(0, max(1, int(total)))
        progress.setValue(int(completed))
        progress.setLabelText(f"正在验证 {Path(video_path).name}")

    def _on_external_clip_import_verified(self, clips):
        context = self._external_clip_import_context
        if context is None:
            return
        thread = getattr(self, "_external_clip_import_thread", None)
        if thread is not None and thread.isInterruptionRequested():
            self._on_external_clip_import_cancelled()
            return
        race_changed = (
            not self._race_ready
            or Path(self.output_dir).expanduser().absolute() != context["race_dir"]
            or self.video_timeline_store is not context["timeline_store"]
            or self.passage_event_store is not context["passage_store"]
        )
        if race_changed:
            QMessageBox.warning(
                self,
                "导入取消",
                "媒体验证期间赛事已切换，高速摄像片段未写入时间线。",
            )
            return
        try:
            current_race_id = race_id_from_passage_store(self.passage_event_store)
        except (ExternalClipImportError, VideoTimelineError) as error:
            QMessageBox.warning(self, "导入取消", str(error))
            return

        if current_race_id != context["race_id"]:
            QMessageBox.warning(
                self,
                "导入取消",
                "媒体验证期间赛事已切换，高速摄像片段未写入时间线。",
            )
            return

        try:
            result = import_verified_external_clips(
                context["timeline_store"],
                clips,
                expected_race_id=context["race_id"],
            )
        except (ExternalClipImportError, VideoTimelineError) as error:
            QMessageBox.warning(self, "导入失败", str(error))
            return

        dialog = self.passage_review_dialog
        if dialog is not None:
            dialog.refresh()
        refresh_evidence = getattr(self, "_refresh_evidence_ui", None)
        if callable(refresh_evidence):
            refresh_evidence()
        message = (
            f"已导入 {result.created_count} 个片段，"
            f"修复 {result.repaired_count} 个，"
            f"跳过重复 {result.duplicate_count} 个。"
        )
        self.statusBar().showMessage(message, 5000)
        QMessageBox.information(self, "导入完成", message)

    def _on_external_clip_import_failed(self, message: str):
        QMessageBox.warning(self, "导入失败", str(message))

    def _on_external_clip_import_cancelled(self):
        self.statusBar().showMessage("已取消高速摄像片段导入", 3000)

    def _on_external_clip_import_finished(self):
        thread = self.sender()
        if thread is not self._external_clip_import_thread:
            return
        progress = self._external_clip_import_progress
        if progress is not None:
            progress.close()
            progress.deleteLater()
        self._external_clip_import_progress = None
        self._external_clip_import_context = None
        self._external_clip_import_thread = None
        self._refresh_evidence_ui()
        thread.deleteLater()

    def _open_passage_location(self, event, location):
        segment_race_id = str(getattr(location.segment, "race_id", "") or "").strip()
        if segment_race_id and segment_race_id != event.race_id:
            QMessageBox.warning(self, "无法打开", "该录像片段属于其他赛事。")
            return
        if location.status not in {"located", "near_boundary", "unverified"}:
            QMessageBox.warning(
                self,
                "无法打开",
                f"当前录像定位状态不可打开: {location.status}",
            )
            return
        if not location.video_path.is_file():
            QMessageBox.warning(self, "录像缺失", str(location.video_path))
            return

        identity = event.bib.strip() or event.chip_id.strip() or "未知"
        self.statusBar().showMessage(
            f"打开机位 {location.segment.camera_index}：{identity} passage "
            f"约在 {location.passage_position_ms / 1000.0:.3f}s，"
            f"误差至少 ±{location.timing_error_ms}ms"
        )
        playback = VideoPlaybackDialog(
            location.video_path,
            self,
            initial_position_ms=location.playback_position_ms,
            target_position_ms=location.passage_position_ms,
            context_text=(
                f"{identity} | 机位 {location.segment.camera_index} | "
                f"定位误差至少 ±{location.timing_error_ms} ms"
            ),
            autoplay=False,
        )
        playback.exec_()

    def _on_roi_toggle(self, source_id: int, state: int):
        """处理 ROI 开关切换"""
        is_enabled = state == Qt.Checked
        self.roi_enabled[source_id] = is_enabled
        
        # 保存配置
        self.config['roi_enabled'] = self.roi_enabled
        self._save_config()
        
        # 更新显示标签状态
        if source_id in self.video_labels:
            label = self.video_labels[source_id]
            label.set_show_roi(is_enabled)
            if is_enabled:
                try:
                    label.roi_changed.disconnect(self._on_roi_changed)
                except Exception:
                    pass
                label.roi_changed.connect(self._on_roi_changed)
                label.set_roi_points(self.roi_points)
            else:
                try: label.roi_changed.disconnect(self._on_roi_changed)
                except: pass
            label.update()
            
        # 同步到检测器
        if source_id in self.detectors:
            detector = self.detectors[source_id]
            if is_enabled:
                detector.set_roi_polygon(self.roi_points)
            else:
                detector.set_roi_polygon(None)
        
        status = "启用" if is_enabled else "禁用"
        self.statusBar().showMessage(f"机位 {source_id+1} ROI 过滤已{status}")

    def _setup_menubar(self):
        """设置菜单栏"""
        menubar = self.menuBar()
        
        # Resource cleanup is centralized in closeEvent -> _cleanup_resources().
        
        # 数据管理菜单
        data_menu = menubar.addMenu("数据管理(&D)")
        
        # 1. 导入报名表
        import_reg_action = QAction("导入报名表...", self)
        import_reg_action.setStatusTip("从Excel文件导入选手报名信息（姓名、号码、组别）")
        import_reg_action.triggered.connect(self._import_registration)
        data_menu.addAction(import_reg_action)
        
        # 2. 导入个人成绩册
        import_score_action = QAction("导入个人成绩册...", self)
        import_score_action.setStatusTip("从Excel文件导入已有的个人成绩数据")
        import_score_action.triggered.connect(self._import_personal_scores)
        data_menu.addAction(import_score_action)
        
        data_menu.addSeparator()
        
        # 3. 导出完赛成绩册
        export_action = QAction("导出完赛成绩册...", self)
        export_action.setStatusTip("将当前比赛结果导出为Excel报表")
        export_action.triggered.connect(self._export_results)
        data_menu.addAction(export_action)

        # 终点线菜单
        line_menu = menubar.addMenu("终点线(&L)")
        dir_group = QActionGroup(self)
        dir_group.setExclusive(True)

        any_dir = QAction("任意方向", self, checkable=True)
        any_dir.setStatusTip("只要穿越终点线即判定过线")
        neg_pos = QAction("负到正", self, checkable=True)
        neg_pos.setStatusTip("仅允许从负侧到正侧方向过线")
        pos_neg = QAction("正到负", self, checkable=True)
        pos_neg.setStatusTip("仅允许从正侧到负侧方向过线")

        dir_group.addAction(any_dir)
        dir_group.addAction(neg_pos)
        dir_group.addAction(pos_neg)

        line_menu.addAction(any_dir)
        line_menu.addAction(neg_pos)
        line_menu.addAction(pos_neg)

        current_dir = self.config.get('crossing_direction')
        if current_dir == 'neg_to_pos':
            neg_pos.setChecked(True)
        elif current_dir == 'pos_to_neg':
            pos_neg.setChecked(True)
        else:
            any_dir.setChecked(True)

        def _apply_direction(direction: str):
            self.config['crossing_direction'] = direction
            self._save_config()
            for detector in self.detectors.values():
                detector.set_crossing_direction(direction)
            self.statusBar().showMessage(f"过线方向已设置为: {direction or '任意方向'}")

        any_dir.triggered.connect(lambda: _apply_direction(None))
        neg_pos.triggered.connect(lambda: _apply_direction('neg_to_pos'))
        pos_neg.triggered.connect(lambda: _apply_direction('pos_to_neg'))

        # 系统设置菜单
        settings_menu = menubar.addMenu("系统设置(&S)")
        
        cam_config_action = QAction("摄像头配置...", self)
        cam_config_action.setStatusTip("配置USB摄像头(Action 5)或网络摄像头(海康RTSP)")
        cam_config_action.triggered.connect(self._config_camera)
        settings_menu.addAction(cam_config_action)
        
        start_time_action = QAction("发枪时间设置...", self)
        start_time_action.setStatusTip("设置各组别的发枪时间")
        start_time_action.triggered.connect(self._show_start_time_dialog)
        settings_menu.addAction(start_time_action)

        vlm_config_action = QAction("VLM 辅助配置 (OpenAI中转/豆包/通义)...", self)
        vlm_config_action.setStatusTip("配置 VLM 大模型辅助号码识别")
        vlm_config_action.triggered.connect(self._config_vlm)
        settings_menu.addAction(vlm_config_action)
        
        settings_menu.addSeparator()
        
        data_menu.addSeparator()

        open_race_action = QAction("打开已有赛事...", self)
        open_race_action.setStatusTip("打开包含 timing.db 的已有赛事文件夹")
        open_race_action.triggered.connect(self._on_open_race_clicked)
        data_menu.addAction(open_race_action)

        playback_action = QAction("录像回放...", self)
        playback_action.setStatusTip("打开当前赛事 videos 目录中的录像")
        playback_action.triggered.connect(self._on_playback_clicked)
        data_menu.addAction(playback_action)

        passage_review_action = QAction("终点核对...", self)
        passage_review_action.setStatusTip("按号码快速核对普通录像和高速摄像")
        passage_review_action.triggered.connect(self._show_passage_review)
        data_menu.addAction(passage_review_action)

        self.external_clip_action = QAction("导入高速摄像片段...", self)
        self.external_clip_action.setStatusTip(
            "按北京时间 sidecar 将外部高速摄像片段加入当前赛事录像时间线"
        )
        self.external_clip_action.setEnabled(False)
        self.external_clip_action.triggered.connect(self._import_external_clips)
        data_menu.addAction(self.external_clip_action)

        init_race_action = QAction("新建赛事 (创建独立文件夹)...", self)
        init_race_action.setStatusTip("创建一个全新的赛事文件夹，所有数据独立存储")
        init_race_action.triggered.connect(self._on_new_race_clicked)
        data_menu.addAction(init_race_action)

    def _config_vlm(self):
        """打开 VLM 配置对话框"""
        dialog = VLMConfigDialog(self.config, self)
        if dialog.exec_() == QDialog.Accepted:
            vlm_config = dialog.get_result()
            self.config['vlm_config'] = vlm_config
            self._save_config()

            # Apply immediately so configuration changes also affect already
            # saved events; the next capture start will apply it again safely.
            self._apply_vlm_settings()
            retried = self._enqueue_pending_ocr_events()
            if retried:
                self.statusBar().showMessage(f"VLM 配置已应用，已重试 {retried} 个待识别事件")
            else:
                self.statusBar().showMessage("VLM 辅助配置已保存")

    def _enqueue_pending_ocr_events(self) -> int:
        """Requeue unresolved saved events after OCR/VLM settings change."""
        if self._yolo_only_mode or not self.ocr_manager or not self.database:
            return 0

        enqueued = 0
        try:
            events = self.database.get_all_events() or []
        except Exception as exc:
            logger.warning("[Main] 无法读取待识别事件: %s", exc)
            return 0

        for event in events:
            if int(event.get("manual_corrected") or 0) != 0:
                continue
            bib = str(event.get("bib_number") or "").strip().upper()
            ocr_state = str(event.get("ocr_state") or "PENDING").strip().upper()
            if bib and bib != "UNKNOWN" and ocr_state == "DONE":
                continue
            if self.ocr_manager.enqueue_live_event(
                event_id=int(event.get("event_id") or 0),
                evidence_dir=event.get("evidence_dir"),
                cross_time=event.get("cross_time"),
                participant_id=event.get("participant_id"),
                raw_track_id=event.get("track_id"),
            ):
                enqueued += 1

        if enqueued:
            logger.info("[Main] VLM 配置更新后重新入队待识别事件: %s", enqueued)
        return enqueued

    def _disable_vlm_runtime(self):
        self.shared_vlm = None
        for detector in self.detectors.values():
            if hasattr(detector, 'disable_vlm'):
                detector.disable_vlm()
        if self.ocr_manager:
            clear_vlm = getattr(self.ocr_manager, "clear_vlm", None)
            if callable(clear_vlm):
                clear_vlm()
            else:
                self.ocr_manager.vlm = None
                self.ocr_manager.vlm_mode = "fallback"

    def _apply_vlm_settings(self):
        """应用 VLM 设置到所有检测器及 OCR 管理器"""
        if self._yolo_only_mode:
            self._disable_vlm_runtime()
            return
        vlm_config = self.config.get('vlm_config', {})
        if not vlm_config.get('enabled'):
            self._disable_vlm_runtime()
            return

        api_key = vlm_config.get('api_key')
        if not api_key:
            self._disable_vlm_runtime()
            return

        endpoint_id = vlm_config.get('endpoint_id')
        base_url = vlm_config.get('base_url')
        max_rpm = vlm_config.get('max_calls_per_minute', 30)
        model_type = vlm_config.get('model_type', 'qwen')
        
        # 初始化共享 VLM
        try:
            from .detector import DoubaoVLMAssistant, QwenVLMAssistant
            from .vlm_utils import OpenAIVLMAssistant
            if model_type == "doubao":
                self.shared_vlm = DoubaoVLMAssistant(api_key, endpoint_id)
            elif model_type == "qwen":
                self.shared_vlm = QwenVLMAssistant(api_key, endpoint_id or "qwen3.5-ocr")
            elif model_type == "openai":
                self.shared_vlm = OpenAIVLMAssistant(
                    api_key, endpoint_id or "gpt-4.1-mini", base_url=base_url
                )
            else:
                logger.error("[Main] 不支持的 VLM 类型: %s", model_type)
                self._disable_vlm_runtime()
                return
        except ImportError:
            from detector import DoubaoVLMAssistant, QwenVLMAssistant
            from vlm_utils import OpenAIVLMAssistant
            if model_type == "doubao":
                self.shared_vlm = DoubaoVLMAssistant(api_key, endpoint_id)
            elif model_type == "qwen":
                self.shared_vlm = QwenVLMAssistant(api_key, endpoint_id or "qwen3.5-ocr")
            elif model_type == "openai":
                self.shared_vlm = OpenAIVLMAssistant(
                    api_key, endpoint_id or "gpt-4.1-mini", base_url=base_url
                )
            else:
                logger.error("[Main] 不支持的 VLM 类型: %s", model_type)
                self._disable_vlm_runtime()
                return
        try:
            logger.info(f"[Main] 全局 VLM 助手已就绪 ({model_type})")
            
            # 同步到 OCR 管理器
            if self.ocr_manager:
                self.ocr_manager.vlm = self.shared_vlm
                self.ocr_manager.vlm_mode = str(vlm_config.get("ocr_mode", "fallback") or "fallback").lower()
                self.ocr_manager.vlm_max_calls_per_minute = max(1, int(max_rpm))
        except Exception as e:
            logger.error(f"[Main] 初始化 VLM 失败: {e}")
            self._disable_vlm_runtime()

        # VLM only runs on saved events so network latency never enters the frame path.
        for detector in self.detectors.values():
            if hasattr(detector, 'disable_vlm'):
                detector.disable_vlm()

    def _config_camera(self):
        """打开摄像头配置对话框 (支持多机位)"""
        dialog = MultiCameraManagementDialog(
            self.sources,
            self,
            camera_clock_management_urls=self.camera_clock_management_urls,
        )
        if dialog.exec_() == QDialog.Accepted:
            new_sources = dialog.get_result()
            new_management_urls = dialog.get_clock_management_urls()
            if (
                new_sources != self.sources
                or new_management_urls != self.camera_clock_management_urls
            ):
                self.sources = new_sources
                self.camera_clock_management_urls = new_management_urls
                self._camera_clock_verified_signature = None
                
                # 保存到 config
                self.config['sources'] = self.sources
                self.config["camera_clock_management_urls"] = list(
                    self.camera_clock_management_urls
                )
                if self.sources:
                    self.config['source'] = self.sources[0] # 保持兼容性
                self._save_config()
                
                # 重新初始化视频显示标签
                self._init_video_labels()
                
                # 如果当前正在运行，询问是否立即重启
                if self._running:
                    reply = QMessageBox.question(
                        self, "配置已更新",
                        "摄像头配置已更新，是否立即重启以应用新配置？",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.Yes
                    )
                    if reply == QMessageBox.Yes:
                        self._stop()
                        self._start()
                else:
                    # 如果没运行，也要标记未初始化，确保下次点击“开始”时重新加载
                    self._initialized = False
                    self.statusBar().showMessage(f"摄像头配置已更新，当前共 {len(self.sources)} 路机位")
                    QMessageBox.information(self, "配置已更新", f"摄像头配置已更新 (共 {len(self.sources)} 路)，点击“开始”即可应用。")

    def _browse_model(self):
        """浏览并选择模型文件"""
        start_dir = str(Path(self.model_path).parent) if self.model_path else "."
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", start_dir, "YOLO Models (*.pt *.engine);;All Files (*)"
        )
        if path:
            self.model_path = path
            self.model_path_input.setText(path)
            self.config['model_path'] = path
            self._save_config()
            self.statusBar().showMessage(f"已选择模型: {Path(path).name}")
            
            # 询问是否立即重载
            reply = QMessageBox.question(
                self, "模型已更改",
                "模型路径已更新，是否立即重载引擎以应用新模型？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self._reload_model()

    def _reload_model(self):
        """热重载检测引擎"""
        if not self.model_path or not Path(self.model_path).exists():
            QMessageBox.warning(self, "错误", "未选择有效的模型文件，无法重载。")
            return

        # 记录当前运行状态
        was_running = self._running
        
        # 1. 停止当前流
        if was_running:
            self._stop()
        
        # 2. 显示加载中
        progress = QProgressDialog("正在加载 AI 模型，请稍候...", None, 0, 0, self)
        progress.setWindowTitle("加载中")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        QApplication.processEvents()

        try:
            logger.info(f"[Main] 正在重载模型: {self.model_path}")
            
            # 3. 重新加载共享资源
            from ultralytics import YOLO
            import numpy as np
            
            self.shared_model = YOLO(self.model_path)
            # 预热新模型
            logger.info("[Main] 正在预热新 YOLO 模型...")
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            self.shared_model.predict(source=dummy_frame, verbose=False)
            logger.info("[Main] 新 YOLO 模型预热完成")
            
            # 4. 更新所有检测器 (支持多机位)
            if self.detectors:
                for i, detector in self.detectors.items():
                    detector._model = self.shared_model
                    detector._ocr = None
                    detector.realtime_ocr_enabled = False
                    detector.model_path = self.model_path
                    # 重新应用配置
                    line_config = self.config.get('finish_lines', {}).get(str(i))
                    if not line_config and i == 0:
                        line_config = self.config.get('finish_line')
                    if line_config:
                        detector.set_finish_line((line_config['x1'], line_config['y1']), 
                                               (line_config['x2'], line_config['y2']))
                    
                    detector.set_crossing_direction(self.config.get('crossing_direction'))
                    detector.set_on_crossing(self._on_crossing_event)
            
            # 5. 恢复选手名单
            self._update_detector_athletes()
            
            self._initialized = True
            self.statusBar().showMessage(f"模型重载成功: {Path(self.model_path).name}")
            logger.info("模型重载成功")
            
        except Exception as e:
            logger.error(f"模型重载失败: {e}")
            QMessageBox.critical(self, "重载失败", f"模型加载出错:\n{str(e)}")
            self._initialized = False
        finally:
            progress.close()

        # 5. 如果之前在运行，则重启
        if was_running and self._initialized:
            self._start()

    def _save_config(self):
        """保存当前配置到文件"""
        config_path = self.output_dir / "config.json"
        try:
            # 创建一个可序列化的副本
            clean_config = {}
            for k, v in self.config.items():
                if k == 'ai_config':
                    continue
                if isinstance(v, Path):
                    clean_config[k] = str(v)
                else:
                    clean_config[k] = v
                    
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(clean_config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"保存配置失败: {e}")

    def _get_preset_path(self, preset_name: str) -> Path:
        safe_name = (preset_name or "").strip().lower() or "default"
        return self.output_dir / f"config_preset_{safe_name}.json"

    def _build_preset_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "preset_version": 1,
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # 核心：终点线/ROI/方向（尽量不包含摄像头地址、模型路径等复杂项）
            "finish_line": self.config.get("finish_line", {}),
            "finish_lines": self.config.get("finish_lines", {}),
            "finish_line_enabled": self.config.get("finish_line_enabled", {}),
            "roi_points": self.config.get("roi_points", []),
            "rois": self.config.get("rois", {}),
            "roi_enabled": self.config.get("roi_enabled", {}),
            "crossing_direction": self.config.get("crossing_direction"),
        }
        return payload

    def _apply_preset_payload(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return

        # 1) config.json 内的几何/方向配置
        for key in (
            "finish_line",
            "finish_lines",
            "finish_line_enabled",
            "roi_points",
            "rois",
            "roi_enabled",
            "crossing_direction",
        ):
            if key in payload:
                self.config[key] = payload[key]

        self._gate_guard_enabled = True
        self.config['gate_guard_enabled'] = True

        # 4) 应用到运行中的 detector（不强制重启，下一帧生效）
        try:
            self._apply_line_and_roi_config()
            for i, detector in getattr(self, "detectors", {}).items():
                line_config = self.config.get("finish_lines", {}).get(str(i))
                if not line_config and i == 0:
                    line_config = self.config.get("finish_line")
                if self.finish_line_enabled.get(i, False) and line_config:
                    detector.set_finish_line((line_config["x1"], line_config["y1"]), (line_config["x2"], line_config["y2"]))
                else:
                    detector.set_finish_line(None, None)

                roi_points = self.config.get("rois", {}).get(str(i))
                if not roi_points and i == 0:
                    roi_points = self.config.get("roi_points")
                if self.roi_enabled.get(i, False) and roi_points:
                    detector.set_roi_polygon(roi_points)
                else:
                    detector.set_roi_polygon(None)

                detector.set_crossing_direction(self.config.get("crossing_direction"))
                detector.set_gate_guard_enabled(bool(self._gate_guard_enabled))
        except Exception:
            pass

        self._save_config()

    def _save_config_preset(self, preset_name: str) -> None:
        try:
            if not getattr(self, "output_dir", None):
                QMessageBox.warning(self, "提示", "请先选择或创建赛事目录。")
                return

            preset_path = self._get_preset_path(preset_name)
            payload = self._build_preset_payload()
            with open(preset_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            self.statusBar().showMessage(f"预设已保存: {preset_path.name}")
            QMessageBox.information(self, "预设已保存", f"已保存预设文件:\n{preset_path}")
        except Exception as e:
            QMessageBox.warning(self, "保存失败", f"保存预设失败:\n{e}")

    def _load_config_preset(self, preset_name: str) -> None:
        if not getattr(self, "output_dir", None):
            QMessageBox.warning(self, "提示", "请先选择或创建赛事目录。")
            return

        preset_path = self._get_preset_path(preset_name)
        if not preset_path.exists():
            QMessageBox.warning(self, "未找到预设", f"未找到预设文件:\n{preset_path}")
            return

        reply = QMessageBox.question(
            self,
            "确认加载预设",
            f"确认加载预设: {preset_path.name}？\n"
            f"（会覆盖终点线/ROI/方向等标定配置）",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return

        try:
            with open(preset_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self._apply_preset_payload(payload)
            self.statusBar().showMessage(f"已加载预设: {preset_path.name}")
            QMessageBox.information(self, "预设已加载", "预设已加载并生效。\n如果正在运行，将从下一帧开始生效。")
        except Exception as e:
            QMessageBox.warning(self, "加载失败", f"加载预设失败:\n{e}")

    def _import_registration(self):
        """导入报名表"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择报名表文件", "", "Excel Files (*.xlsx *.xls);;All Files (*)"
        )
        if not file_path:
            return

        try:
            # 定义匹配关键词
            keywords = {
                'bib_number': ['号码', '编号', 'bib', 'no', '号'],
                'name': ['姓名', '选手', '名字', 'name', '人'],
                'category': ['组别', '分组', '类别', 'category', '组']
            }
            
            # 解析Excel
            data_list = self._parse_excel(file_path, keywords)
            if not data_list:
                QMessageBox.warning(self, "导入失败", "未能在Excel中找到有效的选手信息，请检查表头名称。")
                return
            
            # 存入数据库
            added, updated = self.database.upsert_athletes(data_list)
            
            # 立即同步到检测器
            self._update_detector_athletes()
            
            QMessageBox.information(self, "导入成功", 
                                  f"成功处理 {len(data_list)} 条数据：\n"
                                  f"- 新增选手: {added} 名\n"
                                  f"- 更新信息: {updated} 名")
            
            # 刷新列表显示
            if hasattr(self, 'event_list'):
                self.event_list.refresh_list()
                
        except Exception as e:
            QMessageBox.critical(self, "导入错误", f"解析Excel失败: {e}")

    def _import_personal_scores(self):
        """导入个人成绩册"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择个人成绩册文件", "", "Excel Files (*.xlsx *.xls);;All Files (*)"
        )
        if not file_path:
            return

        try:
            # 个人成绩册通常包含：号码、姓名、芯片成绩/完赛成绩
            keywords = {
                'bib_number': ['号码', '编号', 'bib', 'no', '号'],
                'name': ['姓名', '选手', '名字', 'name', '人'],
                'category': ['组别', '分组', '类别', 'category', '组'],
                'finish_time': ['成绩', '时间', '完赛', '芯片', 'result', 'time']
            }
            
            data_list = self._parse_excel(file_path, keywords)
            if not data_list:
                QMessageBox.warning(self, "导入失败", "未能在Excel中找到有效的成绩信息。")
                return
            
            # 存入芯片成绩表
            added, updated = self.database.upsert_chip_results(data_list)
            
            QMessageBox.information(self, "导入成功", 
                                  f"成功处理 {len(data_list)} 条成绩数据：\n"
                                  f"- 新增记录: {added}\n"
                                  f"- 更新记录: {updated}")
            
            # 刷新列表
            if hasattr(self, 'event_list'):
                self.event_list.refresh_list()
                
        except Exception as e:
            QMessageBox.critical(self, "导入错误", f"解析Excel失败: {e}")

    def _parse_excel(self, file_path: str, keyword_map: dict) -> list:
        """解析Excel并映射字段"""
        # 读取Excel
        df = pd.read_excel(file_path)
        # 清洗列名：转小写并去除空格
        cols = [str(c).strip().lower() for c in df.columns]
        
        # 建立映射： 目标字段 -> 实际Excel列名
        mapping = {}
        for target, keys in keyword_map.items():
            for col_name in df.columns:
                clean_col = str(col_name).strip().lower()
                if any(k.lower() in clean_col for k in keys):
                    mapping[target] = col_name
                    break
        
        if 'bib_number' not in mapping:
            return []
            
        # 提取数据
        results = []
        for _, row in df.iterrows():
            item = {}
            for target, excel_col in mapping.items():
                val = row[excel_col]
                # 处理空值
                if pd.isna(val):
                    val = ""
                else:
                    # 如果是号码，处理成字符串
                    if target == 'bib_number':
                        try:
                            # 去掉Excel自动加的 .0
                            if isinstance(val, (float, int)):
                                val = str(int(val))
                            else:
                                val = str(val).split('.')[0]
                        except:
                            val = str(val)
                    else:
                        val = str(val)
                item[target] = val
            
            if item.get('bib_number'):
                results.append(item)
        
        return results

    def _export_results(self, target_path: str = None):
        """导出完赛成绩册（按组别分页）"""
        # 获取所有有效数据
        events = self.database.get_export_data(include_void=False)
        if not events:
            if not target_path:
                QMessageBox.warning(self, "提示", "当前没有过线记录可以导出")
            return False

        if not target_path:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "导出成绩册", f"比赛成绩册_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx", 
                "Excel Files (*.xlsx);;All Files (*)"
            )
            if not file_path:
                return False
        else:
            file_path = target_path

        try:
            # 1. 准备基础数据
            all_data = []
            for ev in events:
                all_data.append({
                    '总名次': ev.get('rank', ''),
                    '号码': ev.get('bib_number', ''),
                    '姓名': ev.get('athlete_name', '未知'),
                    '组别': ev.get('athlete_category', '未分组'),
                    '过线时间': ev.get('cross_realtime', ''),
                    '比赛成绩': ev.get('finish_time', ''),
                    '芯片成绩': ev.get('chip_time', ''),
                    '备注': ev.get('notes', '')
                })
            
            df_all = pd.DataFrame(all_data)
            
            # 2. 使用 ExcelWriter 导出多页
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # 写入总表
                df_all.to_excel(writer, sheet_name='总排名', index=False)
                
                # 按组别拆分并排序
                categories = df_all['组别'].unique()
                for cat in categories:
                    df_cat = df_all[df_all['组别'] == cat].copy()
                    
                    # 在组内按成绩排序（假设 finish_time 格式支持字符串比较或已处理）
                    # 如果有比赛成绩，按成绩排；否则按过线时间排
                    sort_col = '比赛成绩' if df_cat['比赛成绩'].any() else '过线时间'
                    df_cat = df_cat.sort_values(by=sort_col)
                    
                    # 重新计算组内名次
                    df_cat.insert(0, '组内名次', range(1, len(df_cat) + 1))
                    
                    # 写入对应 Sheet
                    sheet_name = str(cat)[:30]  # Excel Sheet名长度限制
                    df_cat.to_excel(writer, sheet_name=sheet_name, index=False)
            
            if not target_path:
                QMessageBox.information(self, "导出成功", f"成绩册已成功导出！\n包含“总排名”及 {len(categories)} 个组别分页。\n保存路径：{file_path}")
            return True
            
        except Exception as e:
            if not target_path:
                QMessageBox.critical(self, "导出错误", f"保存Excel失败: {e}\n请确保文件未被其他程序占用。")
            return False

    def _release_current_race(self):
        """Stop race-bound workers and close the current database before switching."""
        MainWindow._cancel_camera_clock_preflight(self)
        self._camera_clock_verified_signature = None
        if self.recording_manager is not None:
            self._stop_manual_recording(show_message=False)

        for thread_group in (self.preview_threads, self.video_threads):
            for thread in list(thread_group.values()):
                try:
                    thread.stop()
                except Exception as exc:
                    logger.warning(f"[Main] 停止赛事线程失败: {exc}")
            thread_group.clear()

        for reader in list(self.readers.values()):
            try:
                reader.stop()
            except Exception as exc:
                logger.warning(f"[Main] 停止视频读取器失败: {exc}")
        self.readers.clear()
        self.detectors.clear()

        if self.recorder is not None:
            try:
                stopped_cleanly = self.recorder.stop()
            except Exception as exc:
                raise RuntimeError(f"事件记录器停止失败，未切换赛事: {exc}") from exc
            if stopped_cleanly is False:
                raise RuntimeError("事件记录器仍在写入，未关闭数据库或切换赛事")
            self.recorder = None

        MainWindow._stop_passage_receiver(self)
        self.video_timeline_store = None
        self.video_supplement_store = None
        if hasattr(self, "passage_review_btn"):
            self.passage_review_btn.setEnabled(False)
        if hasattr(self, "video_supplement_btn"):
            self.video_supplement_btn.setEnabled(False)

        if self.ocr_manager is not None:
            try:
                self.ocr_manager.stop()
            except Exception as exc:
                logger.warning(f"[Main] 停止 OCR 管理器失败: {exc}")
            self.ocr_manager = None
        if self._ocr_poll_timer.isActive():
            self._ocr_poll_timer.stop()

        if self.database is not None:
            try:
                self.database.close()
            except Exception as exc:
                logger.warning(f"[Main] 关闭赛事数据库失败: {exc}")
            self.database = None

        self.shared_model = None
        self.shared_athlete_validator = None
        self._athlete_validator_checked = False
        self.shared_vlm = None
        self.field_issue_log = None
        self._initialized = False
        self._race_ready = False
        refresh_evidence = getattr(self, "_refresh_evidence_ui", None)
        if callable(refresh_evidence):
            refresh_evidence()
        self._session_event_count = 0
        self._latest_frame_observations.clear()
        self._live_monitor_samples.clear()
        self._live_monitor_last_warn_ts.clear()
        self._ocr_runtime_state = "disabled" if self._yolo_only_mode else "idle"
        self._refresh_ocr_runtime_ui()

    def _switch_race_dir(self, race_dir: Path) -> bool:
        race_dir = race_dir.expanduser().absolute()
        try:
            self._release_current_race()
            self._apply_race_config(race_dir)
            self._activate_race_dir(race_dir)
            if not self._init_components():
                raise RuntimeError("重新初始化组件失败，请检查日志。")
            self.event_list.refresh_data()
            if hasattr(self, 'session_count_label'):
                self.session_count_label.setText("本轮新增 0")
            if hasattr(self, 'video_session_label'):
                self.video_session_label.setText("本轮新增 0")
            return True
        except Exception as exc:
            logger.exception(f"[Main] 切换赛事失败: {exc}")
            QMessageBox.critical(self, "切换赛事失败", str(exc))
            return False

    def _on_open_race_clicked(self):
        """Open an existing race directory without creating a new database."""
        if self._running:
            QMessageBox.warning(self, "警告", "请先停止当前计时任务，再打开其他赛事。")
            return

        start_path = self.output_dir if self.output_dir.exists() else self.race_root
        selected = QFileDialog.getExistingDirectory(
            self,
            "打开已有赛事",
            str(start_path),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        if not selected:
            return

        race_dir = Path(selected).expanduser().absolute()
        if not (race_dir / "timing.db").is_file():
            QMessageBox.warning(
                self,
                "不是赛事目录",
                "请选择包含 timing.db 的具体赛事文件夹，不要选择 RaceData 根目录。",
            )
            return
        if race_dir == self.output_dir:
            self.statusBar().showMessage(f"当前已经是赛事: {race_dir.name}")
            return

        if self._switch_race_dir(race_dir):
            QMessageBox.information(self, "成功", f"已打开赛事：{race_dir.name}")

    def _on_playback_clicked(self):
        """Open a recording from the current race with responsive review controls."""
        if self._running:
            QMessageBox.warning(self, "提示", "请先停止当前采集，再打开录像回放。")
            return
        if not self._race_ready:
            QMessageBox.warning(self, "提示", "请先打开赛事。")
            return

        recordings = find_recordings(self.output_dir)
        if not recordings:
            QMessageBox.information(
                self,
                "没有录像",
                f"当前赛事没有可回放录像：\n{self.output_dir / 'videos'}",
            )
            return

        latest = recordings[0]
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "选择赛事录像",
            str(latest),
            "Video (*.mkv *.mp4 *.avi *.mov *.m4v);;All (*.*)",
        )
        if not selected:
            return

        playback = VideoPlaybackDialog(Path(selected), self)
        playback.exec_()

    def _on_new_race_clicked(self):
        """新建赛事逻辑：创建新文件夹并切换"""
        if self._running:
            QMessageBox.warning(self, "警告", "请先停止当前计时任务，再新建赛事。")
            return
            
        dialog = NewRaceDialog(self, base_path=str(self.race_root))
        if dialog.exec_() != QDialog.Accepted:
            return
            
        race_name = dialog.result_name
        new_race_dir = Path(dialog.result_path) / race_name
        
        try:
            # 1. 创建目录结构
            new_race_dir.mkdir(parents=True, exist_ok=True)
            (new_race_dir / "evidence_photos").mkdir(exist_ok=True)
            (new_race_dir / "results").mkdir(exist_ok=True)
            
            # 2. 迁移配置：将当前的 config.json 复制过去（保留摄像头等设置）
            old_config = self.output_dir / "config.json"
            new_config = new_race_dir / "config.json"
            if old_config.exists() and old_config.absolute() != new_config.absolute():
                import shutil
                shutil.copy2(old_config, new_config)
            
            if not self._switch_race_dir(new_race_dir):
                return

            logger.info(f"[Main] 切换到新赛事目录: {self.output_dir}")
            QMessageBox.information(self, "成功", f"已成功切换至新赛事：{race_name}")
            
        except Exception as e:
            logger.error(f"[Main] 创建新赛事失败: {e}")
            QMessageBox.critical(self, "错误", f"创建新赛事失败: {e}")

    def _save_global_config(self, data: dict):
        """保存全局配置（用于记住上次打开的文件夹等）"""
        try:
            config_path = Path.cwd() / "global_config.json"
            config = {}
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            config.update(data)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"保存全局配置失败: {e}")


    def _init_components(self) -> bool:
        """初始化组件 (支持多机位)"""
        try:
            try:
                from .stream_reader import StreamStatus
            except ImportError:
                from stream_reader import StreamStatus
            
            # 0. 数据库 (全局唯一)
            if not self.database:
                db_path = self.output_dir / "timing.db"
                self.database = Database(str(db_path))
                logger.info(f"[Main] 数据库已初始化: {db_path}")
                # 同步更新 OCR 管理器的数据库引用
                if self.ocr_manager:
                    self.ocr_manager.db = self.database
                    logger.info("[Main] OCR 管理器数据库引用已更新")
                if hasattr(self, 'event_list') and self.event_list:
                    self.event_list.set_database(self.database)

            # 1. 事件记录器 (全局唯一)
            if not self.recorder:
                # 构造摄像头高清抓拍配置
                camera_configs = {}
                for i, source in enumerate(self.sources):
                    if isinstance(source, str) and source.startswith("rtsp://"):
                        try:
                            # 尝试从 RTSP URL 提取抓拍所需的 IP, User, Pass
                            # 格式: rtsp://user:pwd@ip:port/path
                            import urllib.parse
                            # 替换密码中的特殊字符以便解析，或者手动解析
                            parts = source.split("@")
                            if len(parts) == 2:
                                user_pwd_part = parts[0].replace("rtsp://", "")
                                ip_part = parts[1].split("/")[0]
                                
                                user_pwd = user_pwd_part.split(":")
                                if len(user_pwd) == 2:
                                    user = urllib.parse.unquote(user_pwd[0])
                                    pwd = urllib.parse.unquote(user_pwd[1])
                                    ip = ip_part.split(":")[0]
                                    
                                    camera_configs[i] = {
                                        "ip": ip,
                                        "user": user,
                                        "pass": pwd
                                    }
                                    logger.info(f"[Main] 已提取机位 {i} 的高清抓拍配置: IP={ip}, User={user}")
                        except Exception as e:
                            logger.warning(f"[Main] 提取机位 {i} 抓拍配置失败: {e}")
                
                self.recorder = EventRecorder(
                    str(self.output_dir), 
                    self.database,
                    enable_dedup=self.dedup_checkbox.isChecked(),
                    camera_configs=camera_configs,
                    source_count=max(1, len(self.sources))
                )
                self.recorder.set_on_event_saved(self._on_event_saved_callback)
                self.recorder.start()

            # 2. 共享 AI 模型实例 (节省显存)
            if not self.shared_model:
                if not self.model_path:
                    QMessageBox.critical(self, "错误", "未指定模型路径")
                    return False
                from ultralytics import YOLO
                import numpy as np
                self.shared_model = YOLO(self.model_path)
                # 预热模型
                logger.info("[Main] 正在预热 YOLO 模型...")
                dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
                self.shared_model.predict(source=dummy_frame, verbose=False)
                logger.info("[Main] YOLO 模型预热完成")

            self._init_shared_athlete_validator()
            
            if not self._yolo_only_mode:
                self._init_ocr_runtime()

            # 3. 初始化各机位组件
            max_event_id = self.database.get_latest_event_id()
            
            # 如果机位数量变化，重建 UI
            if len(self.sources) != len(self.video_labels):
                self._init_video_labels()
            
            for i, source in enumerate(self.sources):
                # 如果该机位已初始化且流正常，跳过
                if i in self.readers and self.readers[i].status == StreamStatus.CONNECTED:
                    continue

                # 检测器 (每个机位一个独立实例，但共享模型)
                detector = Detector(
                    self.model_path, 
                    source_id=i, 
                    model=self.shared_model, 
                    ocr=None,
                    ocr_engine=self.ocr_engine,
                    only_numeric=False,
                    realtime_ocr=False,
                    gate_guard_enabled=bool(self._gate_guard_enabled),
                    athlete_validator=self.shared_athlete_validator,
                    performance_profile=str(self.config.get("performance_profile") or "auto"),
                    sport_profile=self.sport_profile,
                    event_settle_seconds=(
                        LOCAL_VIDEO_EVENT_SETTLE_SECONDS
                        if StreamReader.is_video_file_source(source)
                        else None
                    ),
                    ocr_pipeline_enabled=not self._yolo_only_mode,
                )
                # 设置终点线 (优先加载该机位的独立配置)
                line_config = self.config.get('finish_lines', {}).get(str(i))
                if not line_config and i == 0:
                    line_config = self.config.get('finish_line')
                
                if self.finish_line_enabled.get(i, False) and line_config:
                    pt1 = (line_config['x1'], line_config['y1'])
                    pt2 = (line_config['x2'], line_config['y2'])
                    detector.set_finish_line(pt1, pt2)
                else:
                    detector.set_finish_line(None, None)
                    
                # 设置 ROI (优先加载该机位的独立配置)
                roi_points = self.config.get('rois', {}).get(str(i))
                if not roi_points and i == 0:
                    roi_points = self.config.get('roi_points')
                
                if self.roi_enabled.get(i, False) and roi_points:
                    detector.set_roi_polygon(roi_points)
                else:
                    detector.set_roi_polygon(None)
                
                detector.set_crossing_direction(self.config.get('crossing_direction'))
                detector.set_on_crossing(self._on_crossing_event)
                detector.set_on_bib_update(self._on_bib_updated) # 确保连接补录回调
                detector.reset(start_event_id=max_event_id)
                self.detectors[i] = detector

                # 流读取器
                is_video_file = StreamReader.is_video_file_source(source)
                reader = StreamReader(
                    source,
                    queue_size=4,
                    overflow_policy="block" if is_video_file else "drop_oldest",
                )
                if not reader.start():
                    logger.warning(f"[Main] 无法连接到机位 {i+1}: {source}")
                    # 继续尝试其他机位
                self.readers[i] = reader

            # 同步选手名单到所有检测器
            self._update_detector_athletes()

            # 应用 VLM 配置
            self._apply_vlm_settings()

            # 4. 同步测试模式状态
            is_test = self.database.get_is_test_mode()
            self.test_mode_checkbox.blockSignals(True)
            self.test_mode_checkbox.setChecked(is_test)
            color = "#ff6600" if is_test else "#00aa00"
            self.test_mode_checkbox.setStyleSheet(f"QCheckBox {{ font-weight: bold; color: {color}; font-size: 16px; }}")
            self.test_mode_checkbox.blockSignals(False)
            if hasattr(self, 'test_mode_action'):
                self.test_mode_action.setChecked(is_test)

            self._initialized = True
            return True

        except Exception as e:
            QMessageBox.critical(self, "错误", f"初始化失败: {e}")
            logger.exception("初始化组件时发生异常")
            return False

    def _update_detector_athletes(self):
        """同步数据库中的选手名单特征到所有检测器 (支持多机位)"""
        if not self.detectors or not self.database:
            return
            
        try:
            # 获取数据库中的选手名单特征 (包含号码段 bib_ranges_str)
            summary = self.database.get_athlete_summary()
            summary['bib_ranges_str'] = ''
            
            # 根据“名单过滤”开关决定是否加载名单
            if self.athlete_filter_checkbox.isChecked():
                num_bibs = len(summary.get('bibs', []))
                if num_bibs > 0 or summary.get('bib_ranges_str'):
                    for detector in self.detectors.values():
                        detector.set_athlete_list(summary)
                    logger.info(f"[Main] 已同步名单特征与号码段到所有检测器 (自行车模式)")
                else:
                    logger.warning("[Main] 开启了名单过滤但无名单和规则，引擎将采用宽泛识别模式")
                    for detector in self.detectors.values():
                        detector.set_athlete_list({}) # 清空名单
            else:
                for detector in self.detectors.values():
                    detector.set_athlete_list({})
                logger.info("[Main] 名单过滤已关闭，检测器使用自动号码格式")
                
        except Exception as e:
            logger.error(f"[Main] 同步选手名单失败: {e}")

    def _toggle_running(self):
        """切换运行状态"""
        if self._running:
            self._stop()
        else:
            self._start()

    @staticmethod
    def _format_recording_duration(seconds: float) -> str:
        total_seconds = max(0, int(seconds))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _capture_recording_timeline_warning(self, manager) -> Optional[str]:
        consume_warning = getattr(manager, "consume_timeline_warning", None)
        warning = consume_warning() if callable(consume_warning) else None
        if warning:
            self._recording_timeline_warning = str(warning)
        return warning

    def _refresh_recording_ui(self):
        manager = self.recording_manager
        active = bool(manager and manager.is_recording)
        refresh_evidence = getattr(self, "_refresh_evidence_ui", None)
        if callable(refresh_evidence):
            refresh_evidence()

        if hasattr(self, "record_btn"):
            self.record_btn.setText("停止录像" if active else "开始录像")
            self.record_btn.setObjectName("recording_btn" if active else "record_btn")
            self.record_btn.setEnabled(
                active or (
                    bool(self._running)
                    and any(is_rtsp_source(source) for source in self.sources)
                )
            )
            self.record_btn.setStyle(self.record_btn.style())

        if not hasattr(self, "recording_status_label"):
            return
        recording_error = getattr(self, "_recording_error_message", "")
        timeline_warning = getattr(self, "_recording_timeline_warning", "")
        if recording_error and not active:
            self.recording_status_label.setText("录像: 异常")
            self.recording_status_label.setToolTip(recording_error)
            self.recording_status_label.setStyleSheet(
                "margin-right: 12px; color: #cf1322; font-weight: bold; font-size: 13px;"
            )
        elif timeline_warning:
            self.recording_status_label.setText("录像: 时间线异常")
            self.recording_status_label.setToolTip(timeline_warning)
            self.recording_status_label.setStyleSheet(
                "margin-right: 12px; color: #b54708; font-weight: bold; font-size: 13px;"
            )
        elif active:
            self.recording_status_label.setToolTip("")
            duration = self._format_recording_duration(manager.elapsed_seconds)
            size_mb = manager.total_size_bytes / (1024 * 1024)
            self.recording_status_label.setText(f"录像: {duration} | {size_mb:.1f} MB")
            self.recording_status_label.setStyleSheet(
                "margin-right: 12px; color: #cf1322; font-weight: bold; font-size: 13px;"
            )
        else:
            self.recording_status_label.setToolTip("")
            self.recording_status_label.setText("录像: 待机")
            self.recording_status_label.setStyleSheet(
                "margin-right: 12px; color: #666; font-weight: bold; font-size: 13px;"
            )

    def _toggle_recording(self):
        if self.recording_manager is not None:
            self._stop_manual_recording()
        else:
            self._start_manual_recording()

    def _start_auto_recording_if_needed(self):
        if not getattr(self, '_auto_record_live_sources', False):
            return
        if any(is_rtsp_source(source) for source in self.sources):
            self._start_manual_recording(automatic=True)
            return
        if any(StreamReader.is_live_source(source) for source in self.sources):
            message = "当前实时源不是 RTSP，自动录像暂不可用；AI 采集继续运行。"
            self._recording_error_message = message
            logger.warning(f"[Recording] {message}")
            self.statusBar().showMessage(message)

    def _start_manual_recording(self, *, automatic: bool = False):
        if not self._race_ready:
            if not automatic:
                QMessageBox.warning(self, "提示", "请先选择或创建赛事。")
            return
        if not self._running:
            if not automatic:
                QMessageBox.warning(self, "提示", "请先启动AI检测，再开始录像。")
            return

        if not MainWindow._camera_clock_session_is_verified(self):
            message = "当前摄像头未通过本次采集的时间核验，请停止并重新开始采集。"
            self._recording_error_message = message
            logger.warning(f"[Recording] {message}")
            if automatic:
                self.statusBar().showMessage(message)
            else:
                QMessageBox.warning(self, "录像失败", message)
            return

        rtsp_sources = [source for source in self.sources if is_rtsp_source(source)]
        if not rtsp_sources:
            message = "当前实时源不是 RTSP，暂不支持自动录像。"
            self._recording_error_message = message
            logger.warning(f"[Recording] {message}")
            if automatic:
                self.statusBar().showMessage(message)
            else:
                QMessageBox.warning(self, "录像失败", message)
            return

        manager = ManualRecordingManager(
            self.sources,
            self.output_dir / "videos",
        )
        if hasattr(manager, "timeline_store"):
            manager.timeline_store = getattr(self, "video_timeline_store", None)
        if hasattr(manager, "timeline_timing_error_ms"):
            manager.timeline_timing_error_ms = getattr(
                self, "_video_timeline_timing_error_ms", 2_000
            )
        try:
            paths = manager.start()
        except RecordingError as exc:
            logger.error(f"[Recording] 开始录像失败: {exc}")
            self._recording_error_message = f"自动录像失败: {exc}" if automatic else str(exc)
            if automatic:
                self.statusBar().showMessage(self._recording_error_message)
            else:
                QMessageBox.warning(self, "录像失败", str(exc))
            self.recording_manager = None
            self._refresh_recording_ui()
            return

        self.recording_manager = manager
        self._recording_error_message = ""
        if getattr(self, "video_timeline_store", None) is None:
            self._recording_timeline_warning = (
                "录像已开始，但当前赛事录像时间线不可用，"
                "CycleRace passage 无法自动定位到本次录像。"
            )
        else:
            self._recording_timeline_warning = ""
        timeline_warning = MainWindow._capture_recording_timeline_warning(self, manager)
        self._recording_poll_timer.start()
        self._refresh_recording_ui()
        if timeline_warning:
            logger.warning(f"[Recording] {timeline_warning}")
        names = ", ".join(path.name for path in paths)
        mode = "自动" if automatic else "手动"
        logger.info(f"[Recording] 已开始{mode}录像: {names}")
        message = f"{mode}录像已开始，保存到: {self.output_dir / 'videos'}"
        if self._recording_timeline_warning:
            message = f"{message}；{self._recording_timeline_warning}"
        self.statusBar().showMessage(message)

    def _stop_manual_recording(self, *, show_message: bool = True):
        manager = self.recording_manager
        if manager is None:
            self._recording_poll_timer.stop()
            self._refresh_recording_ui()
            return tuple()

        elapsed = manager.elapsed_seconds
        total_size = manager.total_size_bytes
        self.recording_manager = None
        self._recording_poll_timer.stop()
        try:
            paths = manager.stop()
        except RecordingError as exc:
            logger.error(f"[Recording] 停止录像失败: {exc}")
            self._refresh_recording_ui()
            if show_message:
                QMessageBox.warning(self, "录像收尾异常", str(exc))
            return tuple()

        timeline_warning = MainWindow._capture_recording_timeline_warning(self, manager)
        if timeline_warning:
            logger.warning(f"[Recording] {timeline_warning}")

        self._refresh_recording_ui()
        names = ", ".join(path.name for path in paths)
        duration = self._format_recording_duration(elapsed)
        size_mb = total_size / (1024 * 1024)
        logger.info(
            f"[Recording] 录像已保存: {names} ({duration}, {size_mb:.1f} MB)"
        )
        if show_message:
            message = f"录像已保存: {names}"
            if self._recording_timeline_warning:
                message = f"{message}；{self._recording_timeline_warning}"
            self.statusBar().showMessage(message)
        return paths

    def _poll_recording_status(self):
        manager = self.recording_manager
        if manager is None:
            self._recording_poll_timer.stop()
            self._refresh_recording_ui()
            return

        error = manager.check_error()
        if error:
            logger.error(f"[Recording] {error}")
            self._recording_error_message = str(error)
            try:
                manager.stop()
            except RecordingError:
                pass
            timeline_warning = MainWindow._capture_recording_timeline_warning(self, manager)
            if timeline_warning:
                logger.warning(f"[Recording] {timeline_warning}")
            self.recording_manager = None
            self._recording_poll_timer.stop()
            self._refresh_recording_ui()
            QMessageBox.warning(self, "录像中断", error)
            return

        consume_notice = getattr(manager, "consume_recovery_notice", None)
        notice = consume_notice() if callable(consume_notice) else None
        if notice:
            logger.warning(f"[Recording] {notice}")
            self.statusBar().showMessage(notice)
        timeline_warning = MainWindow._capture_recording_timeline_warning(self, manager)
        if timeline_warning:
            logger.warning(f"[Recording] {timeline_warning}")
            self.statusBar().showMessage(timeline_warning)
        self._refresh_recording_ui()

    def start_when_race_ready(self):
        """Start once race selection has completed without opening a second dialog."""
        if self._race_ready:
            if not self._running:
                self._start()
            return
        if self.isVisible():
            QTimer.singleShot(100, self.start_when_race_ready)

    def _camera_clock_targets(self):
        sources = list(getattr(self, "sources", ()) or ())
        management_urls = _normalize_camera_clock_management_urls(
            getattr(self, "camera_clock_management_urls", None),
            len(sources),
        )
        return tuple(
            (index, source, management_urls[index])
            for index, source in enumerate(sources)
            if camera_clock_check_required(source)
        )

    def _camera_clock_session_is_verified(self):
        signature = MainWindow._camera_clock_targets(self)
        if not signature:
            return True
        return getattr(self, "_camera_clock_verified_signature", None) == signature

    def _cancel_camera_clock_preflight(self):
        thread = getattr(self, "_camera_clock_preflight_thread", None)
        if thread is None:
            return
        try:
            thread.completed.disconnect(self._on_camera_clock_preflight_finished)
        except (TypeError, RuntimeError):
            pass
        if thread.isRunning():
            thread.requestInterruption()
            thread.wait(4_000)
        thread.deleteLater()
        self._camera_clock_preflight_thread = None

    def _on_camera_clock_preflight_finished(self, success, message, signature):
        thread = self._camera_clock_preflight_thread
        self._camera_clock_preflight_thread = None
        if thread is not None:
            thread.deleteLater()
        if hasattr(self, "start_btn"):
            self.start_btn.setEnabled(True)

        current_signature = MainWindow._camera_clock_targets(self)
        if success and tuple(signature) != current_signature:
            success = False
            message = "摄像头配置在校时期间发生变化，请重新开始采集。"

        if not success:
            self._camera_clock_verified_signature = None
            logger.error("[CameraClock] 启动前核验失败: %s", message)
            self.statusBar().showMessage(f"相机时间核验失败：{message}")
            QMessageBox.critical(self, "无法开始采集", message)
            return

        self._camera_clock_verified_signature = current_signature
        logger.info("[CameraClock] %s", message)
        self.statusBar().showMessage(message)
        self._start_capture()

    def _start(self):
        """启动 (支持多机位)"""
        if not self._race_ready:
            QMessageBox.warning(self, "提示", "请先选择或创建赛事。")
            self._prompt_race_selection()
            return

        thread = getattr(self, "_camera_clock_preflight_thread", None)
        if thread is not None and thread.isRunning():
            self.statusBar().showMessage("正在核验相机时间，请稍候。")
            return

        targets = MainWindow._camera_clock_targets(self)
        if not targets:
            self._camera_clock_verified_signature = tuple()
            self._start_capture()
            return

        self._camera_clock_verified_signature = None
        self.start_btn.setEnabled(False)
        self.statusBar().showMessage(
            f"正在核验 {len(targets)} 路相机时间，请稍候。"
        )
        thread = CameraClockPreflightThread(targets, self)
        self._camera_clock_preflight_thread = thread
        thread.completed.connect(self._on_camera_clock_preflight_finished)
        thread.start()

    def _start_capture(self):
        """Start capture after the camera clock preflight has passed."""
        if self._running:
            return
        logger.info(f"正在启动 AI 引擎，机位数量: {len(self.sources)}")
        if not self._init_components():
            return

        self._running = True
        self._start_time = time.time()
        self._frame_count = 0
        self._fps_timestamps = deque(maxlen=30)  # 滑动窗口FPS：最近30帧

        self._live_monitor_samples.clear()
        self._live_monitor_last_warn_ts.clear()
        self._live_monitor_last_render_ts = 0.0
        self._refresh_live_monitor_badge(self._start_time)

        # 启动所有机位的视频线程
        for source_id in self.readers:
            reader = self.readers[source_id]
            detector = self.detectors.get(source_id)
            
            if reader and detector:
                ui_skip = 1 if source_id == 0 else 2
                thread = VideoThread(reader, detector, source_id=source_id, ui_skip=ui_skip)
                thread.detections_ready.connect(self._on_detections_ready)
                thread.frame_ready.connect(self._on_detected_frame)
                thread.event_detected.connect(self._on_event_detected)
                thread.bib_updated.connect(self._on_bib_updated)
                thread.status_changed.connect(self._update_conn_status)
                thread.roi_auto_disabled.connect(self._on_roi_auto_disabled)
                self.video_threads[source_id] = thread
                thread.start()
                
                # 初始状态同步
                self._update_conn_status(reader.status, source_id)

        self._start_auto_recording_if_needed()

        self.start_btn.setText("停止采集")
        self.start_btn.setObjectName("stop_btn")
        self.start_btn.setStyle(self.start_btn.style())  # 刷新样式
        self.reset_stats_action.setEnabled(True)
        self.clear_records_action.setEnabled(False)
        self._set_capture_state(True)
        self._refresh_recording_ui()
        
        # 强制刷新一次列表，确保显示最新状态
        if hasattr(self, 'event_list'):
            self.event_list.refresh_list()

        connected = self._connected_source_count()
        status = f"运行中 (已连接 {connected}/{len(self.sources)} 路机位)"
        recording_error = getattr(self, "_recording_error_message", "")
        self.statusBar().showMessage(f"{status} | {recording_error}" if recording_error else status)

    def _stop(self):
        """停止 (支持多机位)"""
        MainWindow._cancel_camera_clock_preflight(self)
        self._camera_clock_verified_signature = None
        self._running = False

        for preview_thread in self.preview_threads.values():
            try:
                preview_thread.frame_ready.disconnect()
            except Exception:
                pass
            preview_thread.stop()
        self.preview_threads.clear()

        # 停止所有视频线程
        for source_id, thread in self.video_threads.items():
            try:
                thread.detections_ready.disconnect()
                thread.frame_ready.disconnect()
                thread.event_detected.disconnect()
                thread.bib_updated.disconnect()
                thread.status_changed.disconnect()
            except:
                pass
            thread.stop()
        self.video_threads.clear()

        self._stop_manual_recording(show_message=False)

        # 停止所有读取器
        for reader in self.readers.values():
            reader.stop()

        if self.recorder:
            self.recorder.stop()

        self.start_btn.setText("开始采集")
        self.start_btn.setObjectName("start_btn")
        self.start_btn.setStyle(self.start_btn.style())  # 刷新样式
        self.reset_stats_action.setEnabled(True)
        self.clear_records_action.setEnabled(True)
        self._set_capture_state(False)
        self._refresh_recording_ui()
        self._refresh_live_monitor_badge(time.time())
        self.statusBar().showMessage("已停止")

    def _reset_stats(self):
        """重置统计 (支持多机位)"""
        max_id = self.database.get_latest_event_id() if self.database else 0
        
        for detector in self.detectors.values():
            detector.reset(start_event_id=max_id)
        
        # 重置记录器缓存
        if hasattr(self, 'recorder') and self.recorder:
            self.recorder.reset()
            
        self._frame_count = 0
        self._start_time = time.time()
        self._session_event_count = 0
        self._live_monitor_samples.clear()
        self._live_monitor_last_warn_ts.clear()
        self._live_monitor_last_render_ts = 0.0
        self._refresh_live_monitor_badge(self._start_time)
        if hasattr(self, 'session_count_label'):
            self.session_count_label.setText("本轮新增 0")
        if hasattr(self, 'video_session_label'):
            self.video_session_label.setText("本轮新增 0")
        if hasattr(self, 'stats_status_label'):
            self.stats_status_label.setText("本轮新增: 0")
        self.statusBar().showMessage("已重置统计 (所有机位追踪状态和记录缓存已清空)")

    def _clear_all_records(self):
        """清空所有记录 (支持多机位)"""
        reply = QMessageBox.question(
            self, "确认清空",
            "确定要清空所有过线记录吗？\n此操作不可撤销！",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            if self.database:
                self.database.clear_events()

            # 彻底重置所有检测引擎 (重建以清空追踪器记忆)
            logger.info("[Main] 正在彻底重建所有检测引擎以清空追踪器记忆...")
            for i in list(self.detectors.keys()):
                detector = Detector(
                    self.model_path, 
                    source_id=i, 
                    model=self.shared_model, 
                    ocr=None,
                    ocr_engine=self.ocr_engine,
                    only_numeric=False,
                    realtime_ocr=False,
                    gate_guard_enabled=bool(self._gate_guard_enabled),
                    athlete_validator=self.shared_athlete_validator,
                    performance_profile=str(self.config.get("performance_profile") or "auto"),
                    sport_profile=self.sport_profile,
                    event_settle_seconds=(
                        LOCAL_VIDEO_EVENT_SETTLE_SECONDS
                        if i < len(self.sources) and StreamReader.is_video_file_source(self.sources[i])
                        else None
                    ),
                    ocr_pipeline_enabled=not self._yolo_only_mode,
                )
                if i == 0:
                    detector.set_finish_line(self.line_pt1, self.line_pt2)
                detector.set_crossing_direction(self.config.get('crossing_direction'))
                detector.set_on_crossing(self._on_crossing_event)
                self.detectors[i] = detector
            
            # 恢复选手名单
            self._update_detector_athletes()
            
            # 重置记录器缓存
            if hasattr(self, 'recorder') and self.recorder:
                self.recorder.reset()

            # 重置计数
            self._session_event_count = 0

            # 强制重置列表 UI
            if hasattr(self, 'event_list'):
                self.event_list.reset_ui()
            if hasattr(self, 'live_event_review'):
                self.live_event_review.clear_event()
            if hasattr(self, 'session_count_label'):
                self.session_count_label.setText("本轮新增 0")
            if hasattr(self, 'video_session_label'):
                self.video_session_label.setText("本轮新增 0")
            if hasattr(self, 'stats_status_label'):
                self.stats_status_label.setText("本轮新增: 0")

            self.statusBar().showMessage("已清空所有记录并重置所有引擎状态")

    def _update_conn_status(self, status, source_id=0):
        """更新连接状态显示 (支持多机位)"""
        try:
            from .stream_reader import StreamStatus
        except ImportError:
            from stream_reader import StreamStatus
        
        text = "未知"
        style = "margin-right: 15px; font-weight: bold; font-size: 14px;" # 稍微调小一点字体以适应多路
        
        if status == StreamStatus.CONNECTED:
            text = "已连接"
            color = "#52c41a" # 绿色
        elif status == StreamStatus.CONNECTING:
            text = "正在连接..."
            color = "#faad14" # 橙色
        elif status == StreamStatus.RECONNECTING:
            text = "重连中..."
            color = "#faad14" # 橙色
        elif status == StreamStatus.ENDED:
            text = "视频已播放完成"
            color = "#1677ff"
        elif status == StreamStatus.ERROR:
            text = "连接错误"
            color = "#ff4d4f" # 红色
        elif status == StreamStatus.DISCONNECTED:
            text = "已断开"
            color = "#666666" # 灰色
            
        # 更新对应的视频标签文字（如果流未连接）
        if status != StreamStatus.CONNECTED and source_id in self.video_labels:
            self.video_labels[source_id].setText(f"机位 {source_id + 1}: {text}")

        # 更新机位状态指示器
        if hasattr(self, 'cam_status_indicators') and source_id in self.cam_status_indicators:
            indicator = self.cam_status_indicators[source_id]
            indicator.setText("●")
            indicator.setStyleSheet(f"color: {color}; font-size: 12px; border: none; margin-left: 8px;")
            indicator.setToolTip(text)

        # 更新状态栏或全局指示器（仅针对主机位或汇总显示）
        if source_id == 0:
            if hasattr(self, 'conn_status_label'):
                self.conn_status_label.setText(f"主相机: {text}")
                self.conn_status_label.setStyleSheet(f"{style} color: {color};")

            if hasattr(self, 'conn_status_indicator'):
                self.conn_status_indicator.setText(f"● 主相机{text}")
                self.conn_status_indicator.setStyleSheet(
                    f"color: {color}; font-weight: 700; font-size: 13px;"
                )

        self._update_capture_connection_detail()

    def _on_detections_ready(self, athletes, bibs, source_id=0):
        self._latest_frame_observations[source_id] = (
            [dict(item) if isinstance(item, dict) else item for item in (athletes or [])],
            [dict(item) if isinstance(item, dict) else item for item in (bibs or [])],
        )

    def _on_detected_frame(self, frame, athletes, bibs, source_id=0):
        try:
            self._on_frame_ready(frame, athletes, bibs, source_id)
        finally:
            thread = self.video_threads.get(source_id)
            if thread is not None:
                thread.mark_ui_consumed()

    def _on_preview_frame(self, frame, source_id=0):
        try:
            athletes, bibs = self._latest_frame_observations.get(source_id, ([], []))
            self._on_frame_ready(frame, athletes, bibs, source_id)
        finally:
            preview_thread = self.preview_threads.get(source_id)
            if preview_thread is not None:
                preview_thread.mark_consumed()

    def _on_frame_ready(self, frame, athletes, bibs, source_id=0):
        self._latest_frame_observations[source_id] = (
            [dict(item) if isinstance(item, dict) else item for item in (athletes or [])],
            [dict(item) if isinstance(item, dict) else item for item in (bibs or [])],
        )
        """收到新帧 (支持多机位)"""
        if source_id == 0:
            self._frame_count += 1

        # 1. 获取对应的视频标签
        video_label = self.video_labels.get(source_id)
        if not video_label:
            return

        # 2. 限制界面刷新率 (每个机位独立限制)
        now = time.time()
        last_update_attr = f'_last_ui_update_time_{source_id}'
        display_fps = max(1.0, float(self.config.get("preview_fps", 30.0)))
        if hasattr(self, last_update_attr):
            if now - getattr(self, last_update_attr) < (1.0 / display_fps):
                return
        setattr(self, last_update_attr, now)

        # 2.1 计算每个机位的独立FPS
        frame_count_attr = f'_frame_count_{source_id}'
        fps_time_attr = f'_fps_time_{source_id}'
        fps_attr = f'_fps_{source_id}'

        if not hasattr(self, frame_count_attr):
            setattr(self, frame_count_attr, 0)
            setattr(self, fps_time_attr, now)
            setattr(self, fps_attr, 0.0)

        setattr(self, frame_count_attr, getattr(self, frame_count_attr) + 1)

        # 每秒更新一次FPS
        elapsed = now - getattr(self, fps_time_attr)
        if elapsed >= 1.0:
            fps_val = getattr(self, frame_count_attr) / elapsed
            setattr(self, fps_attr, fps_val)
            setattr(self, frame_count_attr, 0)
            setattr(self, fps_time_attr, now)

            # 更新机位FPS标签
            if hasattr(self, 'cam_fps_labels') and source_id in self.cam_fps_labels:
                fps_label = self.cam_fps_labels[source_id]
                fps_color = "#52c41a" if fps_val >= 20 else ("#faad14" if fps_val >= 10 else "#ff4d4f")
                fps_label.setText(f"{fps_val:.0f} FPS")
                fps_label.setStyleSheet(f"color: {fps_color}; font-size: 12px; border: none; margin-left: 8px;")

        # 2.2 轻量巡检样本记录（不增加推理负担）
        self._update_live_monitor_sample(source_id, now, athletes, bibs)
        if source_id == 0 and (now - float(self._live_monitor_last_render_ts)) >= float(self._live_monitor_update_seconds):
            self._refresh_live_monitor_badge(now)
            self._live_monitor_last_render_ts = now

        # 3. 绘制检测结果
        orig_h, orig_w = frame.shape[:2]
        label_size = video_label.size()
        max_display_w = max(1, int(label_size.width()))
        max_display_h = max(1, int(label_size.height()))
        display_scale = min(
            max_display_w / max(1, orig_w),
            max_display_h / max(1, orig_h),
            1.0,
        )
        display_w = max(1, int(round(orig_w * display_scale)))
        display_h = max(1, int(round(orig_h * display_scale)))
        if display_w == orig_w and display_h == orig_h:
            display = frame.copy()
        else:
            display = cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_AREA)
        scale_x = display_w / max(1, orig_w)
        scale_y = display_h / max(1, orig_h)

        # 绘制运动员（绿色/红色）
        for athlete in athletes:
            x1, y1, x2, y2 = athlete['bbox']
            dx1, dy1 = int(round(x1 * scale_x)), int(round(y1 * scale_y))
            dx2, dy2 = int(round(x2 * scale_x)), int(round(y2 * scale_y))
            track_id = athlete.get('track_id', -1)
            bib_text = athlete.get('bib_text', "")
            
            # 统一使用绿色框
            is_crossed = athlete.get('is_crossed', False)
            color = (0, 255, 0) # 始终使用绿色
            thickness = 2
            
            cv2.rectangle(display, (dx1, dy1), (dx2, dy2), color, thickness)
            
            label = f"ID:{track_id}{bib_text}"
            if is_crossed:
                label += " [CROSSED]"
            
            # 减小字体大小和粗细
            font_scale = 0.6
            font_thickness = 1
            cv2.putText(display, label, (dx1, max(12, dy1 - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

        # 绘制BIB（蓝色）
        for bib in bibs:
            if not isinstance(bib, dict) or 'bbox' not in bib:
                continue
            x1, y1, x2, y2 = bib['bbox']
            dx1, dy1 = int(round(x1 * scale_x)), int(round(y1 * scale_y))
            dx2, dy2 = int(round(x2 * scale_x)), int(round(y2 * scale_y))
            cv2.rectangle(display, (dx1, dy1), (dx2, dy2), (255, 0, 0), 2)

        # 绘制终点线 (仅在未开启控件叠加绘制时，避免出现两条线)
        line_config = self.config.get('finish_lines', {}).get(str(source_id))
        if not line_config and source_id == 0:
            line_config = self.config.get('finish_line') # 兼容旧版
            
        if line_config and not getattr(video_label, 'show_line', False):
            lx1, ly1 = line_config.get('x1'), line_config.get('y1')
            lx2, ly2 = line_config.get('x2'), line_config.get('y2')
            if all(v is not None for v in [lx1, ly1, lx2, ly2]):
                dlx1, dly1 = int(round(lx1 * scale_x)), int(round(ly1 * scale_y))
                dlx2, dly2 = int(round(lx2 * scale_x)), int(round(ly2 * scale_y))
                cv2.line(display, (dlx1, dly1), (dlx2, dly2), (0, 0, 255), 2)
                cv2.putText(display, "FINISH LINE", (dlx1, max(12, dly1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        # 绘制叠加信息
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (display_w, min(display_h, 85)), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, display, 0.6, 0, display)

        # 只有主摄像头显示 FPS 和时间
        if source_id == 0:
            now = time.time()
            runtime = now - self._start_time if self._start_time else 0
            # 滑动窗口FPS：基于最近30帧的实际间隔，不受启动阶段拖累
            if not hasattr(self, '_fps_timestamps'):
                self._fps_timestamps = deque(maxlen=30)
            self._fps_timestamps.append(now)
            if len(self._fps_timestamps) >= 2:
                window = self._fps_timestamps[-1] - self._fps_timestamps[0]
                fps = (len(self._fps_timestamps) - 1) / window if window > 0.001 else 0
            else:
                fps = 0
            self._main_fps_value = float(fps)
            time_str = f"{int(runtime//3600):02d}:{int((runtime%3600)//60):02d}:{int(runtime%60):02d}"

            # --- 增强版 FPS 显示 ---
            fps_color = (0, 255, 0) if fps >= 10 else (0, 0, 255)
            
            # 获取 VLM 和 OCR 状态
            vlm_status = ""
            ocr_queue = 0
            bib_status = ""
            if source_id in self.detectors:
                det = self.detectors[source_id]
                ocr_queue = det.get_ocr_queue_size()
                crowd_status = None
                if hasattr(det, 'get_crowd_status'):
                    try:
                        crowd_status = det.get_crowd_status()
                    except Exception:
                        crowd_status = None
                v_stat = det.get_vlm_status()
                if v_stat:
                    if v_stat['quota_exhausted']:
                        vlm_status = "| VLM: ⛔"
                    elif v_stat['cooldown'] > 0:
                        vlm_status = f"| VLM: ⏳{v_stat['cooldown']}s"
                    else:
                        pending = v_stat['pending_ocr']
                        vlm_status = f"| VLM: 🟢{pending}"
                b_stat = getattr(det, "_last_bib_assign_stats", None)
                if isinstance(b_stat, dict):
                    assigned = b_stat.get("assigned", 0)
                    total_bibs = b_stat.get("bibs", 0)
                    rejected = b_stat.get("rejected", 0)
                    bib_status = f"| BIB: {assigned}/{total_bibs} R:{rejected}"

                if isinstance(crowd_status, dict):
                    crowd_level = str(crowd_status.get("level", "sparse"))
                    crowd_score = float(crowd_status.get("score", 0.0))
                    bib_status = f"{bib_status} | Crowd:{crowd_level[:1].upper()}({crowd_score:.2f})"
            
            live_text = f"FPS: {fps:.1f}"
            cv2.putText(display, live_text, (20, 35), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, fps_color, 1, cv2.LINE_AA)
            
            info_text = f"{time_str} | OCR Q:{ocr_queue} {vlm_status} {bib_status}"
            cv2.putText(display, info_text, (20, 70), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
            
            # 更新状态栏
            self.statusBar().showMessage(f"运行: {time_str} | FPS: {fps:.1f} | OCR队列: {ocr_queue}")
        else:
            # 非主机位也显示FPS
            cam_fps = getattr(self, f'_fps_{source_id}', 0)
            fps_color = (0, 255, 0) if cam_fps >= 20 else ((0, 165, 255) if cam_fps >= 10 else (0, 0, 255))
            cv2.putText(display, f"CAM {source_id+1} | FPS: {cam_fps:.0f}", (20, 35),
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, fps_color, 1, cv2.LINE_AA)

        # 4. 转换为Qt图像并显示
        h, w, ch = display.shape
        bytes_per_line = ch * w
        qt_image = QImage(display.data, w, h, bytes_per_line, QImage.Format_BGR888)

        scaled_pixmap = QPixmap.fromImage(qt_image)

        if not scaled_pixmap.isNull():
            if video_label.orig_w != orig_w:
                video_label.set_original_size(orig_w, orig_h)
            
            # --- 关键修复：如果用户正在绘画，不要同步坐标覆盖它 ---
            if not video_label.drawing and not video_label.show_roi:
                if line_config:
                    video_label.set_line_points(line_config['x1'], line_config['y1'], 
                                              line_config['x2'], line_config['y2'])

        video_label.setPixmap(scaled_pixmap)

    def _on_line_changed(self, x1, y1, x2, y2):
        """处理终点线变化"""
        sender = self.sender()
        source_id = -1
        
        # 找到是哪个 label 发出的信号
        for sid, label in self.video_labels.items():
            if label == sender:
                source_id = sid
                break
        
        if source_id == -1:
            return

        logger.info(f"更新机位 {source_id} 的终点线: ({x1}, {y1}) - ({x2}, {y2})")
        
        # 更新该机位的检测器
        if source_id in self.detectors:
            self.detectors[source_id].set_finish_line((x1, y1), (x2, y2))
        
        # 保存到配置
        if 'finish_lines' not in self.config:
            self.config['finish_lines'] = {}
        
        self.config['finish_lines'][str(source_id)] = {
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
        }
        
        # 兼容旧版主摄像头配置
        if source_id == 0:
            self.config['finish_line'] = self.config['finish_lines'][str(source_id)]
            
        self._save_config()
        self.statusBar().showMessage(f"机位 {source_id} 终点线已更新 (已自动保存)")

    def _on_roi_changed(self, points: list):
        """处理 ROI 区域变化信号"""
        sender = self.sender()
        source_id = -1
        for sid, label in self.video_labels.items():
            if label == sender:
                source_id = sid
                break
        
        if source_id == -1:
            return

        logger.info(f"更新机位 {source_id} 的 ROI 区域")
        
        # 同步到检测器
        if source_id in self.detectors:
            self.detectors[source_id].set_roi_polygon(points)
        
        # 保存到配置
        if 'rois' not in self.config:
            self.config['rois'] = {}
        
        self.config['rois'][str(source_id)] = [[p.x() if hasattr(p, 'x') else p[0], 
                                               p.y() if hasattr(p, 'y') else p[1]] for p in points]
        
        # 兼容旧版主摄像头配置
        if source_id == 0:
            self.config['roi_points'] = self.config['rois'][str(source_id)]
            
        self._save_config()
        self.statusBar().showMessage(f"机位 {source_id} ROI 区域已更新 (共 {len(points)} 个点，已自动保存)")

    def _on_roi_auto_disabled(self, source_id: int):
        """Keep the UI and persisted config aligned with detector ROI recovery."""
        if not self.roi_enabled.get(source_id, False):
            return

        self.roi_enabled[source_id] = False
        self.config['roi_enabled'] = dict(self.roi_enabled)

        checkbox = getattr(self, 'roi_checkboxes', {}).get(source_id)
        if checkbox is not None:
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(False)

        label = self.video_labels.get(source_id)
        if label is not None:
            label.set_show_roi(False)
            try:
                label.roi_changed.disconnect(self._on_roi_changed)
            except Exception:
                pass

        detector = self.detectors.get(source_id)
        if detector is not None:
            detector.set_roi_polygon(None)

        self._save_config()
        logger.warning(
            f"[Main] 机位 {source_id} ROI 过滤过强，已自动关闭并保存；终点线过滤仍保持启用"
        )
        self.statusBar().showMessage(
            f"机位 {source_id + 1} ROI 过滤过强，已自动关闭以避免漏拍运动员"
        )

    def _on_event_saved_callback(self, event_id: int):
        """记录器保存完事件后的回调 (在后台线程执行)"""
        # 记录器保存完后，触发列表增量刷新
        if hasattr(self, 'event_list'):
            # 使用 QMetaObject.invokeMethod 确保在 UI 线程执行
            # 触发 _check_new_events 进行增量刷新，而不是全量 refresh_list
            from PyQt5.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(self.event_list, "_check_new_events", Qt.QueuedConnection)

        if self._yolo_only_mode:
            return

        # 自动补号：事件落库后自动将 UNKNOWN/PENDING 事件加入 OCR 队列
        try:
            if self.ocr_manager and self.database:
                ev = self.database.get_event(event_id)
                if ev:
                    bib = str(ev.get('bib_number') or '').strip().upper()
                    ocr_state = str(ev.get('ocr_state') or 'PENDING').strip().upper()
                    should_enqueue = (not bib or bib == 'UNKNOWN' or ocr_state in {'PENDING', 'FAIL'})
                    if should_enqueue:
                        enqueued = self.ocr_manager.enqueue_live_event(
                            event_id=int(event_id),
                            evidence_dir=ev.get('evidence_dir'),
                            cross_time=ev.get('cross_time')
                        )
                        logger.info(
                            f"[Main] 自动入队OCR: event_id={event_id}, bib={bib or 'EMPTY'}, "
                            f"ocr_state={ocr_state}, enqueued={enqueued}"
                        )
        except Exception as e:
            logger.debug(f"[Main] 自动入队 OCR 失败: event_id={event_id}, err={e}")

    def _capture_field_issue_context(self) -> Optional[Dict[str, Any]]:
        candidates = []
        for source_id, thread in self.video_threads.items():
            envelope = getattr(thread, "last_processed_envelope", None)
            if envelope is not None:
                candidates.append((float(envelope.capture_time_ms), int(source_id), thread, envelope))
        if not candidates or self.field_issue_log is None:
            return None

        _, source_id, thread, envelope = max(candidates, key=lambda item: item[0])
        frame = np.array(envelope.original_frame, copy=True)
        if frame.size == 0:
            return None

        metrics = dict(getattr(thread, "last_frame_metrics", {}) or {})
        reader = self.readers.get(source_id)
        if reader is not None:
            try:
                reader_info = reader.get_info()
            except Exception:
                reader_info = {}
            metrics["queue_depth"] = reader_info.get("queue_depth")
            metrics["dropped_frames"] = reader_info.get("dropped_frame_count")

        participant_ids = set()
        raw_track_ids = set()
        event_ids = set()
        athletes, _ = self._latest_frame_observations.get(source_id, ([], []))
        for athlete in athletes:
            if not isinstance(athlete, dict):
                continue
            participant_id = athlete.get("participant_id")
            raw_track_id = athlete.get("raw_track_id", athlete.get("track_id"))
            event_id = athlete.get("event_id")
            if participant_id:
                participant_ids.add(str(participant_id))
            if raw_track_id is not None:
                try:
                    raw_track_ids.add(int(raw_track_id))
                except (TypeError, ValueError):
                    pass
            if event_id is not None:
                try:
                    event_ids.add(int(event_id))
                except (TypeError, ValueError):
                    pass

        model_identity = None
        if self.model_path:
            model_path = Path(self.model_path)
            model_identity = f"{model_path.name}:{model_path.stat().st_size}" if model_path.exists() else model_path.name

        return {
            "issue_log": self.field_issue_log,
            "session_id": self.output_dir.name,
            "source_id": source_id,
            "frame_index": int(envelope.frame_index),
            "capture_time_ms": float(envelope.capture_time_ms),
            "segment_id": int(envelope.segment_id),
            "frame": frame,
            "metrics": metrics,
            "participant_ids": tuple(sorted(participant_ids)),
            "raw_track_ids": tuple(sorted(raw_track_ids)),
            "event_ids": tuple(sorted(event_ids)),
            "software_commit": self._software_commit,
            "model_identity": model_identity,
            "event_profile": self.config.get("event_profile", self.config.get("sport_profile")),
        }

    def _mark_field_issue(self):
        if self._field_issue_closing or self._field_issue_executor is None:
            return
        context = self._capture_field_issue_context()
        if context is None:
            self.statusBar().showMessage("尚无可标记的已处理视频帧", 3000)
            return

        preferred_order = [
            "missed_athlete",
            "duplicate_athlete",
            "false_athlete",
            "wrong_bib",
            "missing_bib",
            "ui_freeze",
            "camera_reconnect",
            "other",
        ]
        categories = [item for item in preferred_order if item in FIELD_ISSUE_CATEGORIES]
        category, accepted = QInputDialog.getItem(
            self,
            "标记现场问题",
            "问题类型:",
            categories,
            0,
            False,
        )
        if not accepted:
            return
        note, accepted = QInputDialog.getText(
            self,
            "标记现场问题",
            "简短说明（可留空）:",
        )
        if not accepted:
            return

        token = uuid.uuid4().hex[:12]
        relative_screenshot = Path("field_issues") / "screenshots" / (
            f"source_{context['source_id']}_frame_{context['frame_index']}_{token}.jpg"
        )
        context.update(
            {
                "category": category,
                "note": note,
                "screenshot_relative": relative_screenshot,
                "screenshot_path": self.output_dir / relative_screenshot,
            }
        )
        self._field_issue_executor.submit(self._persist_field_issue, context)
        self.statusBar().showMessage(
            f"已锁定机位 {context['source_id'] + 1} 帧 {context['frame_index']}，正在保存问题标记",
            3000,
        )

    def _persist_field_issue(self, context: Dict[str, Any]):
        try:
            screenshot_saved = write_image_unicode(
                context["screenshot_path"], context["frame"], quality=95
            )
            marker = context["issue_log"].record(
                session_id=context["session_id"],
                source_id=context["source_id"],
                category=context["category"],
                frame_index=context["frame_index"],
                capture_time_ms=context["capture_time_ms"],
                segment_id=context["segment_id"],
                participant_ids=context["participant_ids"],
                raw_track_ids=context["raw_track_ids"],
                event_ids=context["event_ids"],
                screenshot_path=(
                    context["screenshot_relative"].as_posix() if screenshot_saved else None
                ),
                metrics=context["metrics"],
                software_commit=context["software_commit"],
                model_identity=context["model_identity"],
                event_profile=context["event_profile"],
                note=context["note"],
            )
            if not self._field_issue_closing:
                self.field_issue_saved_signal.emit(
                    {"marker": marker, "screenshot_saved": screenshot_saved, "error": None}
                )
        except Exception as exc:
            logger.exception(f"[FieldIssue] 保存失败: {exc}")
            if not self._field_issue_closing:
                self.field_issue_saved_signal.emit(
                    {"marker": None, "screenshot_saved": False, "error": str(exc)}
                )

    def _on_field_issue_saved_ui(self, result: Dict[str, Any]):
        error = result.get("error")
        if error:
            self.statusBar().showMessage(f"问题标记保存失败: {error}", 5000)
            return
        marker = result.get("marker")
        if marker is None:
            return
        screenshot_state = "含原始截图" if result.get("screenshot_saved") else "仅保存时间标记"
        self.statusBar().showMessage(
            f"问题标记已保存: {marker.category} ({screenshot_state})",
            5000,
        )
        logger.info(
            f"[FieldIssue] 已保存 {marker.issue_id}: source={marker.source_id}, "
            f"frame={marker.frame_index}, category={marker.category}"
        )

    def _on_event_detected(self, event: CrossingEvent):
        """检测到过线事件 (在主线程执行)"""
        if not self._running:
            return
        # 1. 更新本次会话计数
        self._session_event_count += 1
        logger.info(f"[MainWindow] 检测到过线事件: ID {event.track_id}, Bib {event.bib_number}, Source {event.source_id}")
        
        # 2. 保存到记录器 (异步处理 IO)
        if self.recorder:
            logger.debug(f"[MainWindow] 正在提交给 EventRecorder 保存...")
            if not getattr(self.recorder, "_running", False):
                logger.warning("[MainWindow] EventRecorder 未运行，尝试自动启动...")
                try:
                    self.recorder.start()
                except RuntimeError as exc:
                    logger.error(f"[MainWindow] EventRecorder 无法重新启动: {exc}")
                    self.statusBar().showMessage("事件记录器停止未完成，已阻止继续写入", 5000)
                    return
            self.recorder.record(event)
            
        # 3. 刷新右侧列表
        # 注意：此处不再直接刷新，而是等待 recorder 保存完后的回调触发，确保数据已落库
        # if hasattr(self, 'event_list'):
        #     self.event_list.refresh_list()
            
        # 4. 状态栏提示
        self.statusBar().showMessage(f"检测到过线: ID {event.track_id} | 号码: {event.bib_number or '未知'}", 3000)
        
        # 5. 更新统计信息
        if hasattr(self, 'stats_status_label'):
            # 不再实时查询数据库总数，避免主线程 IO 阻塞
            # 数据库总数可以通过其他方式异步更新，这里只更新本次会话计数
            self.stats_status_label.setText(f"本轮新增: {self._session_event_count}")
        if hasattr(self, 'session_count_label'):
            self.session_count_label.setText(f"本轮新增 {self._session_event_count}")
        if hasattr(self, 'video_session_label'):
            self.video_session_label.setText(f"本轮新增 {self._session_event_count}")

    def _on_bib_updated(self, track_id: int, bib: str, conf: float, status: Any, source_id: int, 
                        bib_crop: Optional[np.ndarray] = None, athlete_crop: Optional[np.ndarray] = None, 
                        full_frame: Optional[np.ndarray] = None):
        """处理过线后补录事件 (在检测线程调用，需确保不阻塞)"""
        # 1. 查找数据库中的对应记录
        if self.database:
            # 这里的 get_latest_event_by_track_id 依然是同步的，但它是读操作，通常较快
            # 如果依然卡顿，考虑将整个逻辑移至 EventRecorder
            event_id = self.database.get_latest_event_by_track_id(track_id, source_id)
            if event_id:
                # 2. 异步更新文本信息 (通过 recorder 队列)
                if self.recorder:
                    self.recorder.update_event_info(event_id, bib, conf, status, source_id, track_id)
                
                # 3. 异步更新截图文件 (如果提供了高质量图)
                if self.recorder and (bib_crop is not None or athlete_crop is not None):
                    self.recorder.update_images(
                        event_id, source_id, track_id,
                        bib_crop=bib_crop, 
                        athlete_crop=athlete_crop, 
                        full_frame=full_frame
                    )
                    
                    # 4. 刷新右侧列表
                    if hasattr(self, 'event_list'):
                        # 触发增量刷新
                        QMetaObject.invokeMethod(self.event_list, "_check_new_events", Qt.QueuedConnection)
                    
                    # 5. 状态栏提示
                    self.statusBar().showMessage(f"自动补录成功: ID {track_id} -> 号码 {bib}", 3000)
                else:
                    logger.warning(f"[UI] 数据库补录更新失败: EventID {event_id}")
            else:
                logger.debug(f"[UI] 未找到可补录的记录: TrackID {track_id}, Source {source_id}")

    def _on_crossing_event(self, event: CrossingEvent):
        """过线事件回调（在检测线程中）"""
        pass  # 通过信号处理

    def _view_screenshot(self, path: str, event_data: dict = None):
        """查看截图 - 支持列表联动同步"""
        if not (path and Path(path).exists()):
            if event_data is None:
                QMessageBox.warning(self, "提示", "截图文件不存在")
                return
            # 允许无主截图时继续打开查看器，内部将尝试证据目录回退
            path = ""

        # 获取当前列表中的所有事件，用于导航
        all_events = self.event_list.get_all_events()
        
        # 找到当前事件在列表中的索引
        current_index = 0
        if event_data:
            event_id = event_data.get('event_id')
            for i, ev in enumerate(all_events):
                if ev.get('event_id') == event_id:
                    current_index = i
                    break
        
        # 如果已经打开了一个查看器，则关闭旧的（或者更新它，这里选择重新打开以重置状态）
        if self.viewer_dialog:
            self.viewer_dialog.close()
            
        # 弹出增强版查看器 (非模态，支持联动)
        self.viewer_dialog = ImageViewerDialog(current_index, all_events, self, database=self.database)
        self.viewer_dialog.bib_changed.connect(self._on_bib_changed)
        self.viewer_dialog.void_changed.connect(self._on_void_changed)
        self.viewer_dialog.event_changed.connect(self._on_viewer_event_changed)
        self.viewer_dialog.finished.connect(self._on_viewer_closed)
        self.viewer_dialog.show()

    def _on_viewer_event_changed(self, event_id: int):
        """当查看器切换记录时，同步更新主界面列表的选中项"""
        if hasattr(self, 'event_list'):
            self.event_list.select_event_by_id(event_id)

    def _on_viewer_closed(self):
        """查看器关闭时的清理"""
        self.viewer_dialog = None

    def _on_void_changed(self, event_id: int, is_void: bool):
        """作废状态变化回调"""
        if self.database:
            if is_void:
                self.database.void_event(event_id)
            else:
                self.database.unvoid_event(event_id)
            # 刷新列表
            if hasattr(self, 'event_list'):
                self.event_list.refresh_list()

    def _on_bib_changed(self, event_id: int, new_bib: str):
        """号码修改回调"""
        if self.database:
            # 1. 更新号码
            self.database.update_event(event_id, {
                'bib_number': new_bib if new_bib else None,
                'bib_status': 'recognized' if new_bib else 'unrecognized'
            })
            
            # 2. 如果在正式模式，尝试更新成绩和过线时间
            if not self.test_mode_checkbox.isChecked():
                self.database.update_event_finish_time(event_id)
                
            # 3. 刷新列表
            if hasattr(self, 'event_list'):
                self.event_list.refresh_list()

    def _on_dedup_changed(self, state):
        """号码去重开关切换处理"""
        enabled = state == Qt.Checked
        if self.recorder:
            self.recorder.set_enable_dedup(enabled)
        self.statusBar().showMessage(f"号码去重已{'开启' if enabled else '关闭'}")

    def _on_mode_changed(self, state):
        """模式切换处理"""
        is_test = state == Qt.Checked

        # 如果数据库未初始化，只更新UI样式
        if not self.database:
            if is_test:
                self.test_mode_checkbox.setStyleSheet("""
                    QCheckBox {
                        font-weight: bold;
                        color: #ff6600;
                    }
                """)
            else:
                self.test_mode_checkbox.setStyleSheet("""
                    QCheckBox {
                        font-weight: bold;
                        color: #00aa00;
                    }
                """)
            return

        if is_test:
            # 切换到测试模式
            self.test_mode_checkbox.setStyleSheet("""
                QCheckBox {
                    font-weight: bold;
                    color: #ff6600;
                }
            """)
            if self.database:
                self.database.set_is_test_mode(True)
            self.statusBar().showMessage("已切换到测试模式")
        else:
            # 切换到正式模式
            # 检查是否有测试数据
            if self.database:
                test_count = self.database.get_test_event_count()
                if test_count > 0:
                    reply = QMessageBox.question(
                        self, "切换到正式模式",
                        f"当前有 {test_count} 条测试数据。\n是否清空测试数据？",
                        QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                        QMessageBox.Yes
                    )
                    if reply == QMessageBox.Cancel:
                        # 取消切换
                        self.test_mode_checkbox.blockSignals(True)
                        self.test_mode_checkbox.setChecked(True)
                        self.test_mode_checkbox.blockSignals(False)
                        if hasattr(self, 'test_mode_action'):
                            self.test_mode_action.setChecked(True)
                        return
                    elif reply == QMessageBox.Yes:
                        # 清空测试数据
                        self.database.clear_test_events()
                        if hasattr(self, 'event_list'):
                            self.event_list.refresh_list()

                # 检查是否设置了发枪时间
                categories = self.database.get_categories_without_start_time()
                if categories:
                    QMessageBox.warning(
                        self, "请设置发枪时间",
                        f"以下组别尚未设置发枪时间：\n{', '.join(categories)}\n\n请先设置发枪时间！"
                    )
                    self._show_start_time_dialog()
                    # 检查是否已设置
                    categories = self.database.get_categories_without_start_time()
                    if categories:
                        # 仍未设置，取消切换
                        self.test_mode_checkbox.blockSignals(True)
                        self.test_mode_checkbox.setChecked(True)
                        self.test_mode_checkbox.blockSignals(False)
                        if hasattr(self, 'test_mode_action'):
                            self.test_mode_action.setChecked(True)
                        QMessageBox.warning(self, "提示", "未设置发枪时间，保持测试模式")
                        return

                self.database.set_is_test_mode(False)

            self.test_mode_checkbox.setStyleSheet("""
                QCheckBox {
                    font-weight: bold;
                    color: #00aa00;
                }
            """)
            self.statusBar().showMessage("已切换到正式模式")

    def _show_start_time_dialog(self):
        """显示发枪时间设置对话框"""
        if not self.database:
            QMessageBox.information(self, "提示", "请先选择赛事。")
            return
        dialog = StartTimeDialog(self.database, self)
        dialog.exec_()

    def closeEvent(self, event):
        """关闭窗口"""
        self._stop()
        self._cleanup_resources()
        event.accept()

    def _cleanup_resources(self):
        """窗口关闭时清理资源（此时 Qt C++ 对象仍存活，可安全操作子线程）"""
        try:
            self._field_issue_closing = True
            MainWindow._close_passage_review(self)
            external_thread = getattr(self, "_external_clip_import_thread", None)
            if external_thread is not None and external_thread.isRunning():
                external_thread.requestInterruption()
                external_thread.wait()
            MainWindow._stop_passage_receiver(self)
            field_issue_executor = getattr(self, "_field_issue_executor", None)
            if field_issue_executor is not None:
                field_issue_executor.shutdown(wait=False, cancel_futures=False)
                self._field_issue_executor = None

            if getattr(self, "_ocr_poll_timer", None) and self._ocr_poll_timer.isActive():
                self._ocr_poll_timer.stop()
            if getattr(self, "_recording_poll_timer", None) and self._recording_poll_timer.isActive():
                self._recording_poll_timer.stop()
            if self.recording_manager is not None:
                self._stop_manual_recording(show_message=False)
            if self.ocr_manager is not None:
                self.ocr_manager.stop()

            # Remove the GUI handler before its Qt target is destroyed.
            import logging
            root_logger = logging.getLogger()
            gui_log_handler = getattr(self, '_gui_log_handler', None)
            if gui_log_handler:
                try:
                    root_logger.removeHandler(gui_log_handler)
                except Exception:
                    pass
                try:
                    from .logger import logger as custom_logger
                    custom_logger.removeHandler(gui_log_handler)
                except (ImportError, ValueError):
                    pass
                self._gui_log_handler = None
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


def find_model() -> str:
    """自动查找模型文件"""
    model_path = find_runtime_model(
        base_dir=application_dir(),
        project_root=Path(__file__).resolve().parent.parent,
    )
    return str(model_path) if model_path else None


def main():
    import argparse

    parser = argparse.ArgumentParser(description='实时计时系统GUI')
    parser.add_argument('--source', type=str, default=None,
                       help='视频源')
    parser.add_argument('--model', type=str, default=None,
                       help='模型路径（不指定则自动查找）')
    parser.add_argument('--output', type=str, default='RaceData',
                       help='输出目录')
    parser.add_argument('--line-pt1', type=str, default='0,850',
                       help='终点线起点 (x,y)')
    parser.add_argument('--line-pt2', type=str, default='1920,850',
                       help='终点线终点 (x,y)')
    parser.add_argument('--auto-start', action='store_true',
                       help='启动后自动开始')
    parser.add_argument('--yolo-only', action='store_true',
                       help='禁用 OCR，仅运行 YOLO 检测与事件保存')
    parser.add_argument('--ocr-cpu-threads', type=int, default=2,
                       help='本地 OCR 使用的 CPU 线程数（1-4）')
    parser.add_argument('--passage-host', type=str, default=None,
                       help='CycleRace passage 接收监听地址')
    parser.add_argument('--passage-port', type=int, default=None,
                       help='CycleRace passage 接收端口')
    parser.add_argument(
        '--disable-passage-receiver',
        action='store_true',
        help='禁用 CycleRace passage HTTP 接收',
    )

    parser.add_argument(
        '--sport-profile',
        choices=('cycling', 'running', 'speed_skating'),
        default=None,
        help='Event profile: cycling, running, or speed_skating',
    )

    args = parser.parse_args()
    runtime_root = application_dir()

    # 查找模型
    model_path = str(resolve_runtime_path(args.model, base_dir=runtime_root)) if args.model else None
    if not model_path:
        model_path = find_model()
        if not model_path:
            logger.error("未找到模型文件")
            sys.exit(1)
        logger.info(f"自动找到模型: {model_path}")

    # 初始配置
    # 解析坐标
    try:
        x1, y1 = map(int, args.line_pt1.split(','))
        x2, y2 = map(int, args.line_pt2.split(','))
    except:
        x1, y1, x2, y2 = 0, 850, 1920, 850

    output_path = resolve_output_dir(args.output, base_dir=runtime_root)
    runtime_source = (
        resolve_source(args.source, base_dir=runtime_root)
        if args.source is not None
        else None
    )
    config = {
        'source': runtime_source or 'rtsp://localhost:8554/test',
        'model_path': model_path,
        'output_dir': str(output_path),
        'runtime_dir': str(runtime_root),
        'yolo_only_mode': bool(args.yolo_only),
        'ocr_cpu_threads': max(1, min(4, int(args.ocr_cpu_threads))),
        'sport_profile': args.sport_profile or 'cycling',
        'passage_receiver_enabled': True,
        'passage_receiver_host': DEFAULT_HOST,
        'passage_receiver_port': DEFAULT_PORT,
        'timing_provider': 'cyclerace',
        'racetiger_base_url': '',
        'racetiger_pc': '',
        'racetiger_rid': '',
        'racetiger_token': '',
        'racetiger_poll_interval_seconds': 2.0,
        'finish_line': {
            'x1': x1, 'y1': y1,
            'x2': x2, 'y2': y2,
        },
        'vlm_config': {
            'enabled': False,
            'model_type': 'qwen',
            'api_key': '',
            'endpoint_id': 'qwen3.5-ocr',
            'max_calls_per_minute': 30,
            'ocr_mode': 'fallback'
        }
    }

    # 尝试加载上次保存的配置
    config_file = output_path / "config.json"
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                saved_config = json.load(f)
                # 合并配置，优先使用保存的配置，但命令行指定的参数可以覆盖（如果不是默认值）
                # 为了简单起见，这里我们让保存的配置优先
                for k, v in saved_config.items():
                    if k == 'ai_config':
                        continue
                    config[k] = v
                logger.info(f"已加载保存的配置: {config_file}")
        except Exception as e:
            logger.warning(f"加载保存配置失败: {e}")

    # Runtime/CLI choices must not be replaced by a stale race-root config.
    config['output_dir'] = str(output_path)
    config['runtime_dir'] = str(runtime_root)
    config['yolo_only_mode'] = bool(args.yolo_only)
    config['ocr_cpu_threads'] = max(1, min(4, int(args.ocr_cpu_threads)))
    if runtime_source is not None:
        config['source'] = runtime_source
        config['sources'] = [runtime_source]
    if args.model:
        config['model_path'] = model_path
    if args.sport_profile:
        config['sport_profile'] = args.sport_profile
    if args.passage_host is not None:
        config['passage_receiver_host'] = args.passage_host
    if args.passage_port is not None:
        config['passage_receiver_port'] = args.passage_port
    if args.disable_passage_receiver:
        config['passage_receiver_enabled'] = False

    app = QApplication(sys.argv)
    window = MainWindow(
        config,
        runtime_source_override=runtime_source,
        runtime_sport_profile_override=args.sport_profile,
    )
    window.show()
    
    if args.auto_start:
        logger.info("Auto-Start: 正在启动系统...")
        QTimer.singleShot(0, window.start_when_race_ready)
        
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
