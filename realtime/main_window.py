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
import pandas as pd
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QLabel, QPushButton, QSplitter, QStatusBar, QFileDialog,
    QMessageBox, QDialog, QScrollArea, QLineEdit, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QDateTimeEdit,
    QGroupBox, QFormLayout, QMenuBar, QMenu, QAction, QActionGroup,
    QComboBox, QRadioButton, QStackedWidget, QProgressDialog,
    QPlainTextEdit, QAbstractItemView, QSpinBox, QDialogButtonBox
)
import logging
from ultralytics.utils import LOGGER
# 屏蔽 Ultralytics 的冗余警告（如 source 缺失等）
LOGGER.setLevel(logging.ERROR)
import os
from collections import deque

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


class OCRInitThread(QThread):
    """OCR 后台初始化线程"""
    finished = pyqtSignal(object, str)  # (ocr_instance, error_msg)

    def __init__(self, ocr_engine="paddleocr"):
        super().__init__()
        self.ocr_engine = ocr_engine

    def run(self):
        try:
            logger.info(f"[OCR-Init] 正在初始化 {self.ocr_engine}...")
            if self.ocr_engine.lower() == "paddleocr":
                # 【关键】使用 paddlepaddle CPU-only 版本 (3.2.2)，GPU 100% 留给 YOLO
                # 原因：PaddlePaddle GPU 和 PyTorch GPU 共享同一块显卡时会严重互相干扰，
                # 导致实时流帧率从 30FPS 暴跌到 0-1 FPS。
                # 解决：安装 paddlepaddle==3.2.2 (CPU-only)，不装 paddlepaddle-gpu。
                # CPU OCR 约 1-2s/次，作为后台批处理完全可以接受。
                import os
                os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

                from paddleocr import PaddleOCR
                try:
                    from paddleocr import TextRecognition
                except ImportError:
                    TextRecognition = None
                try:
                    import paddle
                    import paddleocr as _paddleocr_mod
                    try:
                        import paddlex as _paddlex_mod
                    except Exception:
                        _paddlex_mod = None
                    logger.info(f"[OCR-Init] paddle={getattr(paddle, '__version__', None)}(CPU-only, cuda={paddle.is_compiled_with_cuda()}) paddleocr={getattr(_paddleocr_mod, '__version__', None)} paddlex={getattr(_paddlex_mod, '__version__', None) if _paddlex_mod else None}")
                except Exception:
                    pass
                ocr = PaddleOCR(
                    use_textline_orientation=False,
                    lang="en",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    text_rec_score_thresh=0.0,
                    # Paddle 3.3 CPU oneDNN fails on PP-OCRv5 PIR attributes on Windows.
                    enable_mkldnn=False,
                )
                recognizer = None
                if TextRecognition is not None:
                    try:
                        recognizer = TextRecognition(
                            model_name="en_PP-OCRv5_mobile_rec",
                            device="cpu",
                            enable_mkldnn=False,
                        )
                    except Exception as recognition_error:
                        logger.warning(
                            f"[OCR-Init] Recognition-only fallback unavailable: {recognition_error}"
                        )
                adapter = PaddleOcrAdapter(ocr, recognizer=recognizer)
                logger.info("[OCR-Init] PaddleOCR 初始化成功 (CPU模式，不影响YOLO实时检测)")
                self.finished.emit(adapter, "")
            else:
                self.finished.emit(None, f"不支持的 OCR 引擎: {self.ocr_engine}")
        except Exception as e:
            logger.error(f"[OCR-Init] 初始化失败: {e}")
            self.finished.emit(None, str(e))

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
    from .stream_reader import StreamReader, StreamStatus
    from .detector import Detector, CrossingEvent, PaddleOcrAdapter, resolve_athlete_validator_model
    from .database import Database
    from .event_recorder import EventRecorder
    from .event_list_widget import EventListWidget
    from .ocr_manager import OCRManager
    from .io_utils import read_image_unicode
except ImportError:
    import sys
    import os
    # 确保当前目录在 sys.path 中
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    from stream_reader import StreamReader, StreamStatus
    from detector import Detector, CrossingEvent, PaddleOcrAdapter, resolve_athlete_validator_model
    from database import Database
    from event_recorder import EventRecorder
    from event_list_widget import EventListWidget
    from ocr_manager import OCRManager
    from io_utils import read_image_unicode


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


class VideoThread(QThread):
    """视频处理线程"""
    # frame, athletes, bibs, source_id
    frame_ready = pyqtSignal(object, list, list, int)
    event_detected = pyqtSignal(object)  # CrossingEvent
    bib_updated = pyqtSignal(int, str, float, object, int, object, object, object)  # track_id, bib, conf, status, source_id, bib_crop, athlete_crop, full_frame
    status_changed = pyqtSignal(object, int)  # StreamStatus, source_id

    def __init__(self, reader: StreamReader, detector: Detector, source_id: int = 0, ui_skip: int = 2):
        super().__init__()
        self.reader = reader
        self.detector = detector
        self.source_id = source_id
        self.ui_skip = ui_skip  # UI刷新间隔（不影响检测，每帧都检测）
        self._running = False
        
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

    def run(self):
        logger.info(f"[VideoThread-{self.source_id}] 视频处理线程已启动")
        self._running = True
        frame_count = 0
        last_frame_ts = 0.0

        while self._running:
            try:
                frame, frame_ts = self.reader.get_frame_after(last_frame_ts)
            except Exception as e:
                logger.exception(f"[VideoThread-{self.source_id}] 获取新帧异常: {e}")
                time.sleep(0.01)
                continue

            if frame is None:
                time.sleep(0.001)
                continue

            if frame_ts > 0:
                last_frame_ts = frame_ts

            frame_count += 1

            # 每帧都检测（保持ByteTrack跟踪连续性）
            try:
                events, athletes, bibs = self.detector.process_frame(frame)
            except Exception as e:
                logger.exception(f"[VideoThread-{self.source_id}] process_frame 异常: {e}")
                events, athletes, bibs = [], [], []

            # 发送过线事件
            for event in events:
                # 已经在 detector 内部填充了 source_id
                logger.debug(f"[VideoThread-{self.source_id}] 发送过线信号: ID={event.track_id}, Bib={event.bib_number}")
                self.event_detected.emit(event)

            # UI刷新：降低刷新频率减轻GUI负担
            if frame_count % self.ui_skip == 0:
                self.frame_ready.emit(frame, athletes, bibs, self.source_id)

    def stop(self):
        self._running = False
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

    def __init__(self, source):
        super().__init__()
        self.source = source

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
                    self.finished.emit(True, "连接成功，画面读取正常")
                else:
                    self.finished.emit(False, "设备已打开但无法读取画面（可能是编码不支持或权限问题）")
            else:
                self.finished.emit(False, "无法打开设备（请检查IP、密码或USB连接）")
        except Exception as e:
            self.finished.emit(False, f"测试发生异常: {str(e)}")


class MultiCameraManagementDialog(QDialog):
    """多摄像头管理对话框"""
    def __init__(self, current_sources, parent=None):
        super().__init__(parent)
        self.setWindowTitle("多摄像头管理")
        self.setMinimumSize(700, 450)
        if current_sources is None:
            self.sources = []
        elif isinstance(current_sources, (str, int)):
            self.sources = [current_sources]
        else:
            self.sources = list(current_sources)
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
            self._refresh_table()

    def _edit_camera(self, index):
        dialog = CameraConfigDialog(self.sources[index], self)
        if dialog.exec_() == QDialog.Accepted:
            self.sources[index] = dialog.result_source
            self._refresh_table()

    def _delete_camera(self, index):
        if index == 0: return
        self.sources.pop(index)
        self._refresh_table()

    def get_result(self):
        return self.sources

class CameraConfigDialog(QDialog):
    """摄像头配置对话框 - 支持模拟流、USB和大疆/海康网络摄像头"""

    def __init__(self, current_source, parent=None):
        super().__init__(parent)
        self.setWindowTitle("摄像头配置")
        self.setMinimumSize(500, 400)
        self.result_source = current_source
        self._init_ui()
        self._load_current(current_source)

    def _init_ui(self):
        # 统一字体样式
        self.setStyleSheet("""
            QDialog { background-color: #ffffff; }
            QLabel, QRadioButton, QLineEdit, QComboBox, QPushButton, QGroupBox {
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
        self.stack.addWidget(self.page_rtsp)

        layout.addWidget(self.stack)

        # 关联切换
        self.radio_mock.toggled.connect(lambda: self.stack.setCurrentIndex(0))
        self.radio_usb.toggled.connect(lambda: self.stack.setCurrentIndex(1))
        self.radio_rtsp.toggled.connect(lambda: self.stack.setCurrentIndex(2))

        # 3. 状态显示
        self.status_label = QLabel("等待测试...")
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
        
        # 禁用按钮，显示进度
        self.status_label.setText("⏳ 正在尝试连接，请稍候...")
        self.status_label.setStyleSheet("color: #1890ff; font-weight: bold;")
        self._set_ui_enabled(False)
        
        self.tester = ConnectionTester(source)
        self.tester.finished.connect(self._on_test_finished)
        self.tester.start()

    def _on_test_finished(self, success, message):
        """测试完成后的回调"""
        self._set_ui_enabled(True)
        if success:
            self.status_label.setText(f"✅ {message}")
            self.status_label.setStyleSheet("color: #52c41a; font-weight: bold;")
            QMessageBox.information(self, "成功", message)
        else:
            self.status_label.setText(f"❌ {message}")
            self.status_label.setStyleSheet("color: #f5222d; font-weight: bold;")
            QMessageBox.critical(self, "失败", message)

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

    def _get_current_source(self):
        """根据当前界面选择计算出source"""
        if self.radio_mock.isChecked():
            return self.mock_path.text().strip()
        elif self.radio_usb.isChecked():
            return self.usb_combo.currentData()
        else:
            if self.advance_rtsp_check.isChecked():
                return self.rtsp_url_input.text().strip()
            else:
                # 自动生成海康地址
                import urllib.parse
                ip = self.hik_ip.text().strip()
                user = self.hik_user.text().strip()
                pwd = self.hik_pwd.text().strip()
                if not ip or not pwd:
                    QMessageBox.warning(self, "提示", "请填写海康相机的IP和密码")
                    return None
                
                # 对用户名和密码进行URL编码
                safe_user = urllib.parse.quote(user)
                safe_pwd = urllib.parse.quote(pwd)
                
                # 尝试两种常见的海康RTSP路径
                # 1. 现代标准路径: /Streaming/Channels/101
                # 2. 传统路径: /h264/ch1/main/av_stream
                return f"rtsp://{safe_user}:{safe_pwd}@{ip}:554/Streaming/Channels/101"

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
            if source.startswith("rtsp"):
                self.radio_rtsp.setChecked(True)
                self.stack.setCurrentIndex(2)
                # 尝试解析RTSP地址回填到助手
                try:
                    # rtsp://user:pwd@ip:554/...
                    parts = source.split("@")
                    if len(parts) == 2:
                        user_pwd = parts[0].replace("rtsp://", "").split(":")
                        ip_port = parts[1].split("/")[0].split(":")
                        if len(user_pwd) == 2:
                            self.hik_user.setText(user_pwd[0])
                            self.hik_pwd.setText(user_pwd[1])
                        if len(ip_port) >= 1:
                            self.hik_ip.setText(ip_port[0])
                    self.rtsp_url_input.setText(source)
                except:
                    self.rtsp_url_input.setText(source)
                    self.advance_rtsp_check.setChecked(True)
            else:
                self.radio_mock.setChecked(True)
                self.stack.setCurrentIndex(0)
                self.mock_path.setText(source)

    def _on_save(self):
        """保存并关闭"""
        self.result_source = self._get_current_source()
        if self.result_source is not None:
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
    def __init__(self, parent=None, base_path: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle("新建赛事")
        self.setFixedSize(550, 300)
        self.result_name = ""
        self.result_path = ""
        self.base_path = base_path
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        
        tip_label = QLabel("创建一个全新的赛事文件夹，所有数据（数据库、照片）将独立存储。")
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

    def _browse_path(self):
        path = QFileDialog.getExistingDirectory(self, "选择赛事存储根目录", self.path_input.text())
        if path:
            self.path_input.setText(path)

    def _on_ok(self):
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
        
        self.enabled_cb = QCheckBox("启用 VLM 辅助 (通义千问/豆包)")
        self.enabled_cb.setChecked(self.config.get('enabled', False))
        form.addRow("总开关:", self.enabled_cb)
        
        self.model_type = QComboBox()
        self.model_type.addItems(["doubao", "qwen"]) # 目前主要支持豆包和通义
        self.model_type.setCurrentText(self.config.get('model_type', 'doubao'))
        form.addRow("模型类型:", self.model_type)
        
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.Password)
        self.api_key.setText(self.config.get('api_key', ''))
        self.api_key.setPlaceholderText("从云端控制台获取的 API Key")
        form.addRow("API Key:", self.api_key)
        
        self.endpoint_id = QLineEdit()
        self.endpoint_id.setText(self.config.get('endpoint_id', ''))
        self.endpoint_id.setPlaceholderText("豆包需提供推理接入点 ID (Endpoint ID)")
        form.addRow("推理接入点 ID:", self.endpoint_id)
        
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
            if model_type == "qwen":
                self.endpoint_id.setPlaceholderText("通义请填模型名（可留空用默认）：qwen-vl-ocr-latest / qwen-vl-plus-latest / qwen-vl-max-latest")
            else:
                self.endpoint_id.setPlaceholderText("豆包需提供推理接入点 ID (Endpoint ID)")

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
    ocr_progress_signal = pyqtSignal(int, int, dict)
    ocr_done_signal = pyqtSignal(dict)
    event_saved_signal = pyqtSignal(int)
    log_signal = pyqtSignal(str)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # 连接信号
        self.ocr_progress_signal.connect(self._on_ocr_progress_ui)
        self.ocr_done_signal.connect(self._show_batch_ocr_done)
        self.event_saved_signal.connect(self._on_event_saved_ui)
        self.log_signal.connect(self._on_log_received_ui)

        # 组件 (支持多摄像头)
        self.readers = {}         # source_id -> StreamReader
        self.detectors = {}       # source_id -> Detector
        self.video_threads = {}   # source_id -> VideoThread
        self.video_labels = {}    # source_id -> InteractiveVideoLabel
        self.finish_line_checkboxes = {} # source_id -> QCheckBox
        
        self.shared_model = None  # 共享 YOLO 模型
        self.shared_athlete_validator = None  # 仅在无号码事件落库前确认自行车
        self._athlete_validator_checked = False
        self.shared_ocr = None    # 共享 OCR 引擎
        self.shared_vlm = None    # 共享 VLM 助手
        self._ocr_runtime_state = 'idle'  # idle/loading/ready/failed
        self.database: Optional[Database] = None
        self.recorder: Optional[EventRecorder] = None
        self.viewer_dialog: Optional[ImageViewerDialog] = None
        self.ocr_manager: Optional[OCRManager] = None
        self._manual_batch_ocr_running = False

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
        self.model_path = config.get('model_path')
        self.race_root = Path.cwd() / "RaceData"
        self.output_dir = self.race_root.absolute()
        requested_ocr_engine = (config.get('ocr_engine') or 'paddleocr').lower()
        if requested_ocr_engine != 'paddleocr':
            logger.warning(f"[Main] 当前版本已锁定 PaddleOCR，忽略配置 ocr_engine={requested_ocr_engine}")
        self.ocr_engine = 'paddleocr'
        self.config['ocr_engine'] = 'paddleocr'
        self.ocr_init_thread = None  # OCR 初始化线程引用
        self._race_ready = False

        # 自动巡检补号（全自动兜底）：定时扫描最近 UNKNOWN/PENDING 事件并重试 OCR 入队
        self._ocr_patrol_enabled = bool(config.get('ocr_patrol_enabled', True))
        self._ocr_patrol_interval_ms = int(config.get('ocr_patrol_interval_ms', 8000))
        self._ocr_patrol_recent_seconds = float(config.get('ocr_patrol_recent_seconds', 25.0))
        self._ocr_patrol_max_scan = int(config.get('ocr_patrol_max_scan', 8))
        self._ocr_patrol_retry_gap_seconds = float(config.get('ocr_patrol_retry_gap_seconds', 45.0))
        self._ocr_patrol_max_enqueue_per_tick = int(config.get('ocr_patrol_max_enqueue_per_tick', 1))
        self._ocr_patrol_max_retries = int(config.get('ocr_patrol_max_retries', 2))
        self._ocr_patrol_retry_cooldown_seconds = float(config.get('ocr_patrol_retry_cooldown_seconds', 120.0))
        self._ocr_patrol_last_enqueue = {}
        self._ocr_patrol_attempts = {}
        self._ocr_patrol_block_until = {}

        # 波次模式（不停机）：实时优先抓人，OCR按波次/空档分批执行
        self._wave_mode_enabled = bool(config.get('wave_mode_enabled', True))
        self._wave_trigger_count = int(config.get('wave_trigger_count', 25))
        self._wave_idle_seconds = float(config.get('wave_idle_seconds', 10.0))
        self._wave_batch_limit = int(config.get('wave_batch_limit', 10))
        self._wave_pause_fps = float(config.get('wave_pause_fps', 25.0))
        self._wave_resume_fps = float(config.get('wave_resume_fps', 28.0))
        self._wave_new_events_since_last_batch = 0
        self._wave_last_crossing_walltime = 0.0
        self._wave_ocr_paused_by_fps = False
        self._wave_pending_event_ids = []
        self._wave_pending_event_set = set()
        self._main_fps_value = 0.0
        self._gate_guard_enabled = bool(config.get('gate_guard_enabled', True))

        # 轻量实时巡检（摄像头版）：只复用现有检测结果做滚动统计，不新增模型推理
        self._live_monitor_enabled = bool(config.get('live_monitor_enabled', True))
        self._live_monitor_window_seconds = float(config.get('live_monitor_window_seconds', 30.0))
        self._live_monitor_update_seconds = float(config.get('live_monitor_update_seconds', 1.2))
        self._live_monitor_min_fps = float(config.get('live_monitor_min_fps', 12.0))
        self._live_monitor_warn_cooldown_seconds = float(config.get('live_monitor_warn_cooldown_seconds', 8.0))
        self._live_monitor_samples: Dict[int, deque] = {}
        self._live_monitor_last_warn_ts: Dict[int, float] = {}
        self._live_monitor_last_render_ts = 0.0
        self._live_monitor_last_summary = ""
        self.config['live_monitor_enabled'] = self._live_monitor_enabled
        self.config['live_monitor_window_seconds'] = self._live_monitor_window_seconds
        self.config['live_monitor_update_seconds'] = self._live_monitor_update_seconds
        self.config['live_monitor_min_fps'] = self._live_monitor_min_fps
        self.config['live_monitor_warn_cooldown_seconds'] = self._live_monitor_warn_cooldown_seconds

        self._ocr_patrol_timer = QTimer(self)
        self._ocr_patrol_timer.setInterval(max(1000, self._ocr_patrol_interval_ms))
        self._ocr_patrol_timer.timeout.connect(self._run_auto_ocr_patrol)

        # 应用全局样式：统一字体大小为 16px，确保布局不拥挤且清晰
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QMenuBar {
                background-color: #ffffff;
                border-bottom: 1px solid #d9d9d9;
                font-size: 16px;
            }
            QMenuBar::item {
                padding: 6px 12px;
                background-color: transparent;
                color: #333333;
            }
            QMenuBar::item:selected {
                background-color: #e6f7ff;
                color: #1890ff;
            }
            QMenu {
                background-color: #ffffff;
                border: 1px solid #d9d9d9;
                font-size: 16px;
            }
            QMenu::item {
                padding: 8px 25px;
            }
            QMenu::item:selected {
                background-color: #e6f7ff;
                color: #1890ff;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #d9d9d9;
                border-radius: 4px;
                margin-top: 15px;
                padding-top: 15px;
                font-size: 16px;
                background-color: #ffffff;
            }
            /* 基础按钮样式 - 扁平化 */
            QPushButton {
                background-color: #f0f0f0;
                color: #000000;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 16px;
                font-weight: bold;
                border: 1px solid #d9d9d9;
            }
            QPushButton:hover {
                background-color: #e6e6e6;
                border-color: #1890ff;
            }
            QPushButton:pressed {
                background-color: #d9d9d9;
            }
            QPushButton:disabled {
                background-color: #f5f5f5;
                color: #bfbfbf;
                border-color: #d9d9d9;
            }
            /* 特殊按钮样式 */
            #start_btn {
                background-color: #52c41a;
                color: white;
                border: none;
            }
            #start_btn:hover {
                background-color: #73d13d;
            }
            #stop_btn {
                background-color: #ff4d4f;
                color: white;
                border: none;
            }
            #stop_btn:hover {
                background-color: #ff7875;
            }
            #primary_btn {
                background-color: #1890ff;
                color: white;
                border: none;
            }
            #primary_btn:hover {
                background-color: #40a9ff;
            }
            QLabel {
                font-size: 16px;
            }
            QCheckBox {
                font-size: 16px;
            }
            QStatusBar {
                background-color: #ffffff;
                border-top: 1px solid #d9d9d9;
                font-size: 16px;
                color: #000000;
            }
        """)

        self._apply_line_and_roi_config()

        # 状态
        self._running = False
        self._start_time = None
        self._frame_count = 0
        self._initialized = False  # 是否已初始化组件
        self._session_event_count = 0  # 本次会话过线数

        self._init_ui()
        self._load_initial_config()
        self._setup_logging()
        
        # 启动后自动初始化 OCR (异步)
        QTimer.singleShot(500, self._init_shared_ocr)
        QTimer.singleShot(0, self._prompt_race_selection)

    def _load_initial_config(self):
        """从数据库加载持久化配置"""
        if self.database:
            ranges = self.database.get_config('bib_ranges', '')
            if hasattr(self, 'bib_ranges_input'):
                self.bib_ranges_input.setText(ranges)
                logger.info(f"[Main] 已加载持久化号段配置: {ranges}")
            
            # 加载纯数字开关状态
            is_numeric = self.database.get_config('numeric_only', '0') == '1'
            if hasattr(self, 'numeric_only_checkbox'):
                self.numeric_only_checkbox.blockSignals(True)
                self.numeric_only_checkbox.setChecked(is_numeric)
                self.numeric_only_checkbox.blockSignals(False)
                logger.info(f"[Main] 已加载纯数字模式状态: {is_numeric}")
            if self.ocr_manager:
                self.ocr_manager.only_numeric = bool(is_numeric)

            # 加载实时 OCR 开关状态
            is_realtime_ocr = self.database.get_config('realtime_ocr', '0') == '1'
            if hasattr(self, 'realtime_ocr_checkbox'):
                self.realtime_ocr_checkbox.blockSignals(True)
                self.realtime_ocr_checkbox.setChecked(is_realtime_ocr)
                self.realtime_ocr_checkbox.blockSignals(False)
                logger.info(f"[Main] 已加载实时 OCR 状态: {is_realtime_ocr}")

            gate_guard_enabled = self.database.get_config('gate_guard_enabled', '1') == '1'
            self._gate_guard_enabled = gate_guard_enabled
            self.config['gate_guard_enabled'] = gate_guard_enabled
            if hasattr(self, 'gate_guard_checkbox'):
                self.gate_guard_checkbox.blockSignals(True)
                self.gate_guard_checkbox.setChecked(gate_guard_enabled)
                self.gate_guard_checkbox.blockSignals(False)
                logger.info(f"[Main] 已加载龙门安全模式状态: {gate_guard_enabled}")

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

    def _apply_race_config(self, race_dir: Path):
        config_file = race_dir / "config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    saved_config = json.load(f)
                for k, v in saved_config.items():
                    if k == 'ai_config':
                        continue
                    self.config[k] = v
            except Exception as e:
                logger.warning(f"加载保存配置失败: {e}")
        self.config['output_dir'] = str(race_dir)
        sources = self.config.get('sources')
        if sources is None:
            sources = self.config.get('source', 'rtsp://localhost:8554/test')
        if isinstance(sources, (str, int)):
            self.sources = [sources]
        else:
            self.sources = list(sources) if sources else []
        if not self.sources:
            self.sources = [self.config.get('source', 'rtsp://localhost:8554/test')]
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
        self.output_dir = race_dir.absolute()
        self.config['output_dir'] = str(self.output_dir)
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
        if not self.ocr_manager:
            self.ocr_manager = OCRManager(self.database)
            self.ocr_manager.on_progress = self._on_ocr_progress
            self.ocr_manager.on_event_done = self._on_ocr_event_done
        else:
            self.ocr_manager.db = self.database

        # 若 OCR 已经初始化完成，确保 OCRManager 同步到同一引擎（用于自动补号）
        if self.ocr_manager and getattr(self, 'shared_ocr', None):
            self.ocr_manager.set_ocr(self.shared_ocr, getattr(self, 'shared_vlm', None))
        self._load_initial_config()
        self._race_ready = True
        self.statusBar().showMessage(f"当前赛事: {race_dir.name}")

    def _prompt_race_selection(self):
        if self._race_ready:
            return
        self.race_root.mkdir(parents=True, exist_ok=True)
        dialog = NewRaceDialog(self, base_path=str(self.race_root))
        if dialog.exec_() != QDialog.Accepted:
            QMessageBox.warning(self, "提示", "必须先选择或创建赛事，系统将退出。")
            self.close()
            return
        race_name = dialog.result_name
        race_dir = self.race_root / race_name
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
        """初始化界面"""
        self.setWindowTitle("运动比赛终点辅助计时系统 - 实时版")
        self.setMinimumSize(1200, 768) # 适配更多笔记本屏幕分辨率

        # 创建菜单栏
        self._setup_menubar()

        # 中央组件
        central = QWidget()
        self.setCentralWidget(central)

        # 主布局 - 使用分割器
        layout = QHBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)

        # 左边 - 视频区域
        video_widget = QWidget()
        video_layout = QVBoxLayout(video_widget)
        
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
        
        # 连接状态灯
        self.conn_status_indicator = QLabel("● 未连接")
        self.conn_status_indicator.setStyleSheet("color: #ff4444; font-weight: bold; font-size: 16px; margin-right: 10px;")
        top_status_layout.addWidget(self.conn_status_indicator)
        
        video_layout.addWidget(top_status_bar)

        # 1.5 模型选择工具栏
        model_bar = QWidget()
        model_bar.setFixedHeight(45)
        model_bar.setStyleSheet("background-color: #f0f0f0; border-bottom: 1px solid #d9d9d9;")
        model_layout = QHBoxLayout(model_bar)
        model_layout.setContentsMargins(15, 0, 15, 0)
        
        model_layout.addWidget(QLabel("当前模型:"))
        self.model_path_input = QLineEdit(self.model_path or "未选择模型")
        self.model_path_input.setReadOnly(True)
        self.model_path_input.setStyleSheet("background-color: #e8e8e8; color: #666; font-size: 14px;")
        model_layout.addWidget(self.model_path_input, 1)
        
        self.btn_browse_model = QPushButton("选择模型...")
        self.btn_browse_model.setFixedWidth(100)
        self.btn_browse_model.setStyleSheet("font-size: 14px; padding: 4px;")
        self.btn_browse_model.clicked.connect(self._browse_model)
        model_layout.addWidget(self.btn_browse_model)
        
        self.btn_reload_model = QPushButton("重载")
        self.btn_reload_model.setFixedWidth(60)
        self.btn_reload_model.setStyleSheet("background-color: #1890ff; color: white; font-size: 14px; padding: 4px;")
        self.btn_reload_model.clicked.connect(self._reload_model)
        model_layout.addWidget(self.btn_reload_model)

        # 预设：保存/加载“终点线 + ROI + 方向”等关键标定配置（给非程序员一键回滚）
        self.preset_btn = QPushButton("预设")
        self.preset_btn.setFixedWidth(60)
        self.preset_btn.setStyleSheet("font-size: 14px; padding: 4px;")
        preset_menu = QMenu(self)
        act_save_default = preset_menu.addAction("保存为默认方案")
        act_save_default.triggered.connect(lambda: self._save_config_preset("default"))
        act_save_rescue = preset_menu.addAction("保存为救场方案")
        act_save_rescue.triggered.connect(lambda: self._save_config_preset("rescue"))
        preset_menu.addSeparator()
        act_load_default = preset_menu.addAction("加载默认方案")
        act_load_default.triggered.connect(lambda: self._load_config_preset("default"))
        act_load_rescue = preset_menu.addAction("加载救场方案")
        act_load_rescue.triggered.connect(lambda: self._load_config_preset("rescue"))
        self.preset_btn.setMenu(preset_menu)
        model_layout.addWidget(self.preset_btn)
        
        video_layout.addWidget(model_bar)

        # 2. 视频显示区域 (支持多机位分屏 2-4 路)
        self.video_container = QWidget()
        self.video_container.setStyleSheet("background-color: #000;")
        self.video_grid = QGridLayout(self.video_container)
        self.video_grid.setContentsMargins(2, 2, 2, 2)
        self.video_grid.setSpacing(2)
        
        self._init_video_labels()
        
        video_layout.addWidget(self.video_container, 1) # 占据主要空间

        # 3. 控制按钮区域 - 容器 (使用 ID 选择器，避免影响子组件背景)
        self.control_bar = QWidget()
        self.control_bar.setObjectName("control_bar")
        self.control_bar.setFixedHeight(60)
        self.control_bar.setStyleSheet("#control_bar { background-color: #ffffff; border-top: 1px solid #d9d9d9; }")
        btn_layout = QHBoxLayout(self.control_bar)
        btn_layout.setContentsMargins(15, 0, 15, 0)
        btn_layout.setSpacing(10)

        # 开始/停止按钮 (精简文字)
        self.start_btn = QPushButton("开始计时")
        self.start_btn.setObjectName("start_btn")
        self.start_btn.setMinimumWidth(120)
        self.start_btn.setFixedHeight(40)
        self.start_btn.clicked.connect(self._toggle_running)
        btn_layout.addWidget(self.start_btn)

        # 发枪时间设置按钮
        self.start_time_btn = QPushButton("发枪时间")
        self.start_time_btn.setObjectName("primary_btn")
        self.start_time_btn.setFixedHeight(40)
        self.start_time_btn.clicked.connect(self._show_start_time_dialog)
        btn_layout.addWidget(self.start_time_btn)

        # 辅助功能按钮
        self.reset_btn = QPushButton("重置统计")
        self.reset_btn.setFixedHeight(40)
        self.reset_btn.clicked.connect(self._reset_stats)
        self.reset_btn.setEnabled(True)  # 默认开启
        btn_layout.addWidget(self.reset_btn)

        # 清空记录按钮 (补齐，解决崩溃)
        self.clear_btn = QPushButton("清空记录")
        self.clear_btn.setFixedHeight(40)
        self.clear_btn.clicked.connect(self._clear_all_records)
        self.clear_btn.setEnabled(True)  # 默认开启
        btn_layout.addWidget(self.clear_btn)

        btn_layout.addStretch()

        # 模式切换 (更简洁)
        self.test_mode_checkbox = QCheckBox("模拟测试模式")
        self.test_mode_checkbox.setChecked(True)
        self.test_mode_checkbox.setStyleSheet("color: #d46b08; font-weight: bold; font-size: 16px;") # 使用更深的橙色并加粗
        self.test_mode_checkbox.stateChanged.connect(self._on_mode_changed)
        btn_layout.addWidget(self.test_mode_checkbox)

        # 号码去重开关
        self.dedup_checkbox = QCheckBox("号码去重")
        self.dedup_checkbox.setChecked(True)
        self.dedup_checkbox.setStyleSheet("color: #0050b3; font-weight: bold; font-size: 16px;")
        self.dedup_checkbox.stateChanged.connect(self._on_dedup_changed)
        btn_layout.addWidget(self.dedup_checkbox)

        # 名单校验开关 (马拉松模式)
        self.athlete_filter_checkbox = QCheckBox("名单过滤")
        self.athlete_filter_checkbox.setChecked(True)
        self.athlete_filter_checkbox.setToolTip("开启：仅识别名单内的号码(自行车模式)；关闭：宽泛识别所有号码(马拉松模式)")
        self.athlete_filter_checkbox.setStyleSheet("color: #096dd9; font-weight: bold; font-size: 16px;")
        self.athlete_filter_checkbox.stateChanged.connect(self._update_detector_athletes)
        btn_layout.addWidget(self.athlete_filter_checkbox)

        # 仅限纯数字开关 (自行车模式优化)
        self.numeric_only_checkbox = QCheckBox("仅限纯数字")
        self.numeric_only_checkbox.setChecked(False) # 默认关闭，用户手动开启
        self.numeric_only_checkbox.setToolTip("针对自行车比赛优化：只识别纯数字号码，过滤掉带字母的干扰项")
        self.numeric_only_checkbox.setStyleSheet("color: #722ed1; font-weight: bold; font-size: 16px;")
        self.numeric_only_checkbox.stateChanged.connect(self._on_numeric_only_changed)
        btn_layout.addWidget(self.numeric_only_checkbox)

        # 实时识别开关 (默认关闭，防止卡顿)
        self.realtime_ocr_checkbox = QCheckBox("实时 OCR")
        self.realtime_ocr_checkbox.setChecked(False) 
        self.realtime_ocr_checkbox.setToolTip("开启：过线时实时识别号码(可能导致卡顿)；关闭：仅保存截图，由后台/手动处理(推荐)")
        self.realtime_ocr_checkbox.setStyleSheet("color: #eb2f96; font-weight: bold; font-size: 16px;")
        self.realtime_ocr_checkbox.stateChanged.connect(self._on_realtime_ocr_changed)
        btn_layout.addWidget(self.realtime_ocr_checkbox)

        # 龙门安全模式（默认开启，抑制龙门/拱门误检）
        self.gate_guard_checkbox = QCheckBox("龙门安全模式")
        self.gate_guard_checkbox.setChecked(bool(self._gate_guard_enabled))
        self.gate_guard_checkbox.setToolTip("开启：优先抑制龙门/拱门误检；关闭：放宽过滤，适合非龙门赛道")
        self.gate_guard_checkbox.setStyleSheet("color: #13c2c2; font-weight: bold; font-size: 16px;")
        self.gate_guard_checkbox.stateChanged.connect(self._on_gate_guard_changed)
        btn_layout.addWidget(self.gate_guard_checkbox)

        # 号码范围输入
        btn_layout.addSpacing(10)
        btn_layout.addWidget(QLabel("号码段:"))
        self.bib_ranges_input = QLineEdit()
        self.bib_ranges_input.setPlaceholderText("例如: A1001-A2000, 1-500")
        self.bib_ranges_input.setToolTip("输入合法号码段，用逗号分隔。符合范围的号码将获得更高识别权重。")
        self.bib_ranges_input.setFixedWidth(200)
        self.bib_ranges_input.editingFinished.connect(self._on_bib_ranges_changed)
        btn_layout.addWidget(self.bib_ranges_input)

        video_layout.addWidget(self.control_bar)

        splitter.addWidget(video_widget)

        # 右边 - 事件列表 + 系统日志
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(5)

        # 批处理工具栏
        batch_tools = QWidget()
        batch_tools_layout = QHBoxLayout(batch_tools)
        batch_tools_layout.setContentsMargins(5, 5, 5, 5)
        
        self.batch_ocr_btn = QPushButton("开始批处理 OCR")
        self.batch_ocr_btn.setToolTip("扫描 evidence_photos 目录，识别所有未完成的号码")
        self.batch_ocr_btn.setStyleSheet("""
            QPushButton {
                background-color: #1890ff;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #40a9ff; }
            QPushButton:disabled { background-color: #d9d9d9; }
        """)
        self.batch_ocr_btn.clicked.connect(self._start_batch_ocr)
        batch_tools_layout.addWidget(self.batch_ocr_btn)
        
        self.ocr_progress_label = QLabel("")
        self.ocr_progress_label.setStyleSheet("color: #666; font-size: 14px;")
        batch_tools_layout.addWidget(self.ocr_progress_label)
        batch_tools_layout.addStretch()
        
        right_layout.addWidget(batch_tools)

        self.event_list = EventListWidget(self.database)
        self.event_list.view_screenshot.connect(self._view_screenshot)
        # 单机位适配：如果只有一个机位，隐藏来源列
        self.event_list.set_source_column_visible(len(self.sources) > 1)
        right_layout.addWidget(self.event_list, 7) # 占 7/10 高度

        # 系统日志区域
        log_group = QGroupBox("系统运行状态")
        log_layout = QVBoxLayout(log_group)
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumBlockCount(100) # 最多保留100行
        self.log_output.setStyleSheet("font-family: Consolas; font-size: 14px; background-color: #f0f0f0;")
        log_layout.addWidget(self.log_output)
        right_layout.addWidget(log_group, 3) # 占 3/10 高度

        splitter.addWidget(right_panel)

        # 设置分割比例 (初始 2:1，适合大多数场景)
        splitter.setSizes([800, 400])

        layout.addWidget(splitter)
        self.splitter = splitter

        # 状态信息
        self.conn_status_label = QLabel("连接状态: 未知")
        self.conn_status_label.setStyleSheet("margin-right: 15px; color: #666; font-weight: bold; font-size: 16px;")
        self.statusBar().addPermanentWidget(self.conn_status_label)

        self.stats_status_label = QLabel("记录: 0")
        self.stats_status_label.setStyleSheet("margin-right: 15px; color: #000000; font-weight: bold; font-size: 16px;")
        self.statusBar().addPermanentWidget(self.stats_status_label)

        # OCR 引擎常驻标签：让用户一眼看到当前引擎
        self.ocr_engine_label = QLabel("OCR引擎: PaddleOCR (已锁定)")
        self.ocr_engine_label.setStyleSheet("margin-right: 15px; color: #1677ff; font-weight: bold; font-size: 14px;")
        self.ocr_engine_label.setToolTip("当前版本固定使用 PaddleOCR")
        self.statusBar().addPermanentWidget(self.ocr_engine_label)
        self._refresh_ocr_engine_badge()

        # OCR 状态标签
        self.ocr_status_label = QPushButton("OCR: 未初始化")
        self.ocr_status_label.setFlat(True)
        self.ocr_status_label.setCursor(Qt.PointingHandCursor)
        self.ocr_status_label.setStyleSheet("color: #666; font-weight: bold; font-size: 14px; border: none; text-align: left; padding: 0px 10px;")
        self.ocr_status_label.clicked.connect(self._init_shared_ocr)
        self.statusBar().addPermanentWidget(self.ocr_status_label)

        # 轻量巡检状态标签（摄像头实时健康）
        self.live_monitor_label = QLabel("巡检: 待机")
        self.live_monitor_label.setStyleSheet("margin-right: 12px; color: #666; font-weight: bold; font-size: 13px;")
        self.live_monitor_label.setToolTip("轻量实时巡检：基于现有检测结果做滚动统计，不额外跑模型")
        self.statusBar().addPermanentWidget(self.live_monitor_label)
        self._refresh_live_monitor_badge(time.time())

        self.statusBar().showMessage("系统就绪")

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

        self._live_monitor_samples[source_id].append(
            (now_ts, athlete_count, bib_count, split_count, mismatch_flag, bib_only_flag, fps_val)
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

        fps_values = [float(item[6]) for item in samples if float(item[6]) > 0.01]
        avg_fps = float(sum(fps_values) / max(1, len(fps_values))) if fps_values else 0.0

        severe = (avg_fps > 0 and avg_fps < float(self._live_monitor_min_fps) * 0.80) or \
                 mismatch_ratio >= 0.30 or bib_only_ratio >= 0.22
        warning = (avg_fps > 0 and avg_fps < float(self._live_monitor_min_fps)) or \
                  mismatch_ratio >= 0.18 or bib_only_ratio >= 0.12

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
        if avg_fps > 0:
            detail_parts.append(f"FPS={avg_fps:.1f}")
        if mismatch_ratio > 0.0:
            detail_parts.append(f"并排漏检风险={mismatch_ratio:.0%}")
        if bib_only_ratio > 0.0:
            detail_parts.append(f"号码孤立={bib_only_ratio:.0%}")
        if split_hits > 0:
            detail_parts.append(f"大框拆分={split_hits}")

        detail_text = " | ".join(detail_parts) if detail_parts else "采样正常"
        return {
            'ready': True,
            'level': level,
            'label': f"机位{source_id + 1}:{note}",
            'note': note,
            'avg_fps': avg_fps,
            'mismatch_ratio': mismatch_ratio,
            'bib_only_ratio': bib_only_ratio,
            'split_hits': split_hits,
            'detail': detail_text,
        }

    def _refresh_live_monitor_badge(self, now_ts: Optional[float] = None):
        """刷新轻量巡检状态栏。"""
        if not hasattr(self, 'live_monitor_label'):
            return

        if now_ts is None:
            now_ts = time.time()

        if not self._live_monitor_enabled:
            self.live_monitor_label.setText("巡检: 已关闭")
            self.live_monitor_label.setStyleSheet("margin-right: 12px; color: #8c8c8c; font-weight: bold; font-size: 13px;")
            self.live_monitor_label.setToolTip("轻量巡检已关闭")
            self._live_monitor_last_summary = ""
            return

        source_ids = sorted(self.video_labels.keys()) if hasattr(self, 'video_labels') else [0]
        evaluations = [self._evaluate_live_monitor_source(sid, now_ts) for sid in source_ids]
        ready_evals = [item for item in evaluations if item.get('ready')]

        if not ready_evals:
            self.live_monitor_label.setText("巡检: 采样中")
            self.live_monitor_label.setStyleSheet("margin-right: 12px; color: #595959; font-weight: bold; font-size: 13px;")
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
            f"margin-right: 12px; color: {color}; font-weight: bold; font-size: 13px;"
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

        if not bool(self.config.get("athlete_validator_enabled", True)):
            logger.info("[Main] 运动员二次校验已禁用")
            return

        validator_path = resolve_athlete_validator_model(
            str(self.config.get("athlete_validator_model_path") or ""),
            [Path.cwd(), Path(__file__).resolve().parent.parent],
        )
        if validator_path is None:
            logger.warning("[Main] 未找到 yolov8s.pt，非号码事件将保持原有放行策略")
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

    def _init_shared_ocr(self):
        """后台异步初始化 OCR"""
        if hasattr(self, 'ocr_init_thread') and self.ocr_init_thread and self.ocr_init_thread.isRunning():
            return
            
        self._ocr_runtime_state = 'loading'
        self._refresh_ocr_engine_badge()
        self.ocr_status_label.setText("OCR: 正在启动...")
        self.ocr_status_label.setStyleSheet("color: #faad14; font-weight: bold; font-size: 14px; border: none; text-align: left; padding: 0px 10px;")
        
        engine = (self.ocr_engine or "paddleocr").lower()
        self.ocr_init_thread = OCRInitThread(engine)
        self.ocr_init_thread.finished.connect(self._on_ocr_init_finished)
        self.ocr_init_thread.start()

    def _start_batch_ocr(self):
        """开始批处理 OCR"""
        if not self.shared_ocr:
            QMessageBox.warning(self, "警告", "OCR 引擎尚未就绪，请稍后再试")
            return
            
        # 严格使用当前赛事目录下的 evidence_photos
        evidence_root = self.output_dir / "evidence_photos"
        
        if not evidence_root.exists():
            # 如果不存在，尝试创建它（以防万一）
            try:
                evidence_root.mkdir(parents=True, exist_ok=True)
            except:
                pass
            
        if not any(evidence_root.iterdir()) if evidence_root.exists() else True:
            QMessageBox.information(self, "提示", f"当前赛事文件夹内尚未产生任何证据照片，无法开始批处理。\n目录: {evidence_root}")
            return
            
        self.ocr_manager.set_ocr(self.shared_ocr, self.shared_vlm)
        self._manual_batch_ocr_running = True
        self.batch_ocr_btn.setEnabled(False)
        self.batch_ocr_btn.setText("正在批处理...")
        self.ocr_manager.start_batch(str(evidence_root))
        logger.info(f"[Main] 启动 OCR 批处理，根目录: {evidence_root}")

    def _on_ocr_progress(self, current, total, stats):
        """OCR 进度回调 (由 OCR 线程调用)"""
        self.ocr_progress_signal.emit(current, total, stats)

    def _on_ocr_progress_ui(self, current, total, stats):
        """OCR 进度更新 (UI 线程)"""
        resolved_done = int(stats.get('resolved_done', stats.get('done', 0)))
        pending_result = int(stats.get('pending_result', 0))
        msg = (
            f"进度: {current}/{total} | 已补号: {resolved_done} | "
            f"待补(已处理): {pending_result} | 归并: {stats['merged']} | 失败: {stats['failed']}"
        )
        if stats.get('vlm_used', 0) > 0:
            msg += f" | VLM: {stats['vlm_used']}"
            
        self.ocr_progress_label.setText(msg)
        
        if current >= total:
            if self._manual_batch_ocr_running:
                self._manual_batch_ocr_running = False
                self.batch_ocr_btn.setEnabled(True)
                self.batch_ocr_btn.setText("开始批处理 OCR")
                # 不再弹窗，统一改为状态栏提示，避免任何情况下卡界面
                self.statusBar().showMessage("手工批处理 OCR 已完成", 2000)
            else:
                # 自动补号完成时仅更新状态栏，避免反复弹窗阻塞 UI
                self.statusBar().showMessage("自动补号已完成一轮", 1500)

    @pyqtSlot(dict)
    def _show_batch_ocr_done(self, stats):
        """兼容保留：批处理完成仅记录日志，不弹窗。"""
        resolved_done = int(stats.get('resolved_done', stats.get('done', 0)))
        pending_result = int(stats.get('pending_result', 0))
        logger.info(
            f"[Main] OCR批处理完成: total={stats.get('total', 0)}, "
            f"resolved={resolved_done}, pending={pending_result}, "
            f"merged={stats.get('merged', 0)}, failed={stats.get('failed', 0)}"
        )

    def _on_ocr_event_done(self, event_id, result):
        """单个 OCR 任务完成回调"""
        # 刷新事件列表中的特定行
        self.event_saved_signal.emit(event_id)

    def _on_event_saved_ui(self, event_id):
        """事件保存后的 UI 刷新 (UI 线程)"""
        # 优化：优先使用增量检查，避免全量刷新导致闪烁和批处理卡顿
        if hasattr(self.event_list, '_check_new_events'):
            self.event_list._check_new_events()
        else:
            self.event_list.refresh_data()

    def _on_ocr_init_finished(self, ocr_instance, error_msg):
        """OCR 初始化完成后的处理"""
        if ocr_instance:
            self.shared_ocr = ocr_instance
            self._ocr_runtime_state = 'ready'
            self._refresh_ocr_engine_badge()
            self.ocr_status_label.setText("OCR: 已就绪")
            self.ocr_status_label.setStyleSheet("color: #52c41a; font-weight: bold; font-size: 14px; border: none; text-align: left; padding: 0px 10px;")
            self.ocr_status_label.setToolTip("OCR 引擎已正常启动")
            
            # 将新 OCR 实例同步到所有已存在的检测器
            if hasattr(self, 'detectors') and self.detectors:
                for detector in self.detectors.values():
                    if hasattr(detector, 'set_ocr'):
                        detector.set_ocr(self.shared_ocr)
                    else:
                        detector._ocr = self.shared_ocr
                        detector.ensure_ocr()

            # 同步给 OCRManager，支持“事件保存后自动补号”
            if self.ocr_manager:
                self.ocr_manager.set_ocr(self.shared_ocr, getattr(self, 'shared_vlm', None))
            
            logger.info("[Main] OCR 引擎后台加载完成")
        else:
            self.shared_ocr = None
            self._ocr_runtime_state = 'failed'
            self._refresh_ocr_engine_badge()
            self.ocr_status_label.setText("OCR: 启动失败 (点击重试)")
            self.ocr_status_label.setStyleSheet("color: #ff4d4f; font-weight: bold; font-size: 14px; border: none; text-align: left; padding: 0px 10px;")
            self.ocr_status_label.setToolTip(f"错误: {error_msg}\n点击可尝试重新启动")
            logger.error(f"[Main] OCR 引擎加载失败: {error_msg}")

            if self.ocr_manager:
                self.ocr_manager.ocr = None

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
                try: label.roi_changed.connect(self._on_roi_changed)
                except: pass
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
        
        # 资源清理统一在 closeEvent -> _cleanup_resources() 中执行。
        # 注意：不要挂在 destroyed 信号上——该信号在 Qt C++ 对象销毁后才发出，
        # 此时再访问 ocr_init_thread 会触发 "wrapped C/C++ object has been deleted"。
        
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

        vlm_config_action = QAction("VLM 辅助配置 (豆包/通义)...", self)
        vlm_config_action.setStatusTip("配置 VLM 大模型辅助号码识别")
        vlm_config_action.triggered.connect(self._config_vlm)
        settings_menu.addAction(vlm_config_action)
        
        settings_menu.addSeparator()
        
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
            
            # 如果正在运行，立即应用
            if self._running:
                self._apply_vlm_settings()
            
            self.statusBar().showMessage("VLM 辅助配置已保存")

    def _apply_vlm_settings(self):
        """应用 VLM 设置到所有检测器及 OCR 管理器"""
        vlm_config = self.config.get('vlm_config', {})
        if not vlm_config.get('enabled'):
            self.shared_vlm = None
            for detector in self.detectors.values():
                if hasattr(detector, 'disable_vlm'):
                    detector.disable_vlm()
            if self.ocr_manager:
                self.ocr_manager.vlm = None
                self.ocr_manager.vlm_mode = "fallback"
            return

        api_key = vlm_config.get('api_key')
        if not api_key:
            self.shared_vlm = None
            return

        endpoint_id = vlm_config.get('endpoint_id')
        max_rpm = vlm_config.get('max_calls_per_minute', 30)
        model_type = vlm_config.get('model_type', 'doubao')
        
        # 初始化共享 VLM
        try:
            from .detector import DoubaoVLMAssistant, QwenVLMAssistant
            if model_type == "doubao":
                self.shared_vlm = DoubaoVLMAssistant(api_key, endpoint_id)
            else:
                self.shared_vlm = QwenVLMAssistant(api_key, endpoint_id or "qwen-vl-max")
            logger.info(f"[Main] 全局 VLM 助手已就绪 ({model_type})")
            
            # 同步到 OCR 管理器
            if self.ocr_manager:
                self.ocr_manager.vlm = self.shared_vlm
                self.ocr_manager.vlm_mode = str(vlm_config.get("ocr_mode", "fallback") or "fallback").lower()
        except Exception as e:
            logger.error(f"[Main] 初始化 VLM 失败: {e}")
            self.shared_vlm = None

        # 同步到所有检测器
        for detector in self.detectors.values():
            if hasattr(detector, 'enable_vlm'):
                detector.enable_vlm(
                    api_key=api_key,
                    endpoint_id=endpoint_id,
                    max_calls_per_minute=max_rpm,
                    model_type=model_type
                )

    def _config_camera(self):
        """打开摄像头配置对话框 (支持多机位)"""
        dialog = MultiCameraManagementDialog(self.sources, self)
        if dialog.exec_() == QDialog.Accepted:
            new_sources = dialog.get_result()
            if new_sources != self.sources:
                self.sources = new_sources
                
                # 保存到 config
                self.config['sources'] = self.sources
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
            
            # 确保 OCR 已初始化或正在初始化
            if not self.shared_ocr:
                if not self.ocr_init_thread or not self.ocr_init_thread.isRunning():
                    self._init_shared_ocr()
            
            # 4. 更新所有检测器 (支持多机位)
            if self.detectors:
                for i, detector in self.detectors.items():
                    detector._model = self.shared_model
                    # 注意：如果 OCR 还在后台初始化，这里的 _ocr 可能是 None
                    # 等 _on_ocr_init_finished 完成后会再次同步给所有 detector
                    detector._ocr = self.shared_ocr 
                    detector.ensure_ocr()
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
            "gate_guard_enabled": bool(self.config.get("gate_guard_enabled", True)),
        }
        if getattr(self, "database", None):
            payload["bib_ranges"] = self.database.get_config("bib_ranges", "")
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
            "gate_guard_enabled",
        ):
            if key in payload:
                self.config[key] = payload[key]

        # 2) 数据库持久化配置（号段）
        if getattr(self, "database", None) and "bib_ranges" in payload:
            try:
                ranges = str(payload.get("bib_ranges") or "").strip()
                self.database.set_config("bib_ranges", ranges)
                if hasattr(self, "bib_ranges_input"):
                    self.bib_ranges_input.setText(ranges)
            except Exception:
                pass

        # 3) gate_guard UI 同步（红线：默认开启，预设也只做“同步”）
        try:
            self._gate_guard_enabled = bool(self.config.get("gate_guard_enabled", True))
            if hasattr(self, "gate_guard_checkbox"):
                self.gate_guard_checkbox.blockSignals(True)
                self.gate_guard_checkbox.setChecked(bool(self._gate_guard_enabled))
                self.gate_guard_checkbox.blockSignals(False)
        except Exception:
            pass

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

    def _on_new_race_clicked(self):
        """新建赛事逻辑：创建新文件夹并切换"""
        if self._running:
            QMessageBox.warning(self, "警告", "请先停止当前计时任务，再新建赛事。")
            return
            
        dialog = NewRaceDialog(self, base_path=str(self.race_root))
        if dialog.exec_() != QDialog.Accepted:
            return
            
        race_name = dialog.result_name
        new_race_dir = self.race_root / race_name
        
        try:
            # 1. 创建目录结构
            new_race_dir.mkdir(parents=True, exist_ok=True)
            (new_race_dir / "evidence_photos").mkdir(exist_ok=True)
            (new_race_dir / "results").mkdir(exist_ok=True)
            
            # 2. 迁移配置：将当前的 config.json 复制过去（保留摄像头等设置）
            old_config = self.output_dir / "config.json"
            new_config = new_race_dir / "config.json"
            if old_config.exists():
                import shutil
                shutil.copy2(old_config, new_config)
            
            # 3. 停止当前所有组件
            if hasattr(self, 'recorder') and self.recorder:
                self.recorder.stop()
                self.recorder = None
            
            if self.database:
                self.database = None

            # 4. 更新工作目录
            self._activate_race_dir(new_race_dir)
            logger.info(f"[Main] 切换到新赛事目录: {self.output_dir}")

            if not self._init_components():
                QMessageBox.critical(self, "错误", "重新初始化组件失败，请检查日志。")
                return
            
            # 7. 刷新 UI
            self.event_list.refresh_data()
            self._session_event_count = 0
            self.statusBar().showMessage(f"当前赛事: {race_name}")
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
            
            if not self.shared_ocr:
                self._init_shared_ocr()

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
                    ocr=self.shared_ocr,
                    ocr_engine=self.ocr_engine,
                    only_numeric=self.numeric_only_checkbox.isChecked(),
                    realtime_ocr=(False if self._is_wave_mode_active() else self.realtime_ocr_checkbox.isChecked()),
                    gate_guard_enabled=bool(self._gate_guard_enabled),
                    athlete_validator=self.shared_athlete_validator,
                    performance_profile=str(self.config.get("performance_profile") or "auto"),
                )
                detector.ensure_ocr()
                
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
                reader = StreamReader(source)
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

            # 5. 同步号码段范围
            bib_ranges = self.database.get_config('bib_ranges', '')
            self.bib_ranges_input.setText(bib_ranges)

            self._initialized = True
            return True

        except Exception as e:
            QMessageBox.critical(self, "错误", f"初始化失败: {e}")
            logger.exception("初始化组件时发生异常")
            return False

    def _on_bib_ranges_changed(self):
        """当号码段输入框内容变化时保存并同步"""
        ranges = self.bib_ranges_input.text().strip()
        self.database.set_config('bib_ranges', ranges)
        logger.info(f"[Main] 已更新号码段规则并保存到数据库: {ranges}")
        self._update_detector_athletes()

    def _on_numeric_only_changed(self, state):
        """处理仅限纯数字开关切换"""
        is_numeric_only = (state == Qt.Checked)
        self.database.set_config('numeric_only', '1' if is_numeric_only else '0')

        # 同步到已启动的检测器
        if hasattr(self, 'detectors') and self.detectors:
            for detector in self.detectors.values():
                detector.only_numeric = is_numeric_only

        # 同步到批处理 OCR（否则会出现：检测是纯数字，但批处理仍按“字母+3~6位数字”过滤）
        if self.ocr_manager:
            self.ocr_manager.only_numeric = bool(is_numeric_only)
                
        status = "开启" if is_numeric_only else "关闭"
        logger.info(f"[Main] 仅限纯数字模式已{status}")
        self.statusBar().showMessage(f"仅限纯数字模式已{status}")

    def _on_realtime_ocr_changed(self, state):
        """处理实时 OCR 开关切换"""
        is_realtime_ocr = (state == Qt.Checked)

        if self._is_wave_mode_active() and is_realtime_ocr:
            is_realtime_ocr = False
            if hasattr(self, 'realtime_ocr_checkbox'):
                self.realtime_ocr_checkbox.blockSignals(True)
                self.realtime_ocr_checkbox.setChecked(False)
                self.realtime_ocr_checkbox.blockSignals(False)
            logger.info("[Main] 波次模式下已强制关闭实时 OCR")

        self.database.set_config('realtime_ocr', '1' if is_realtime_ocr else '0')
        
        # 同步到已启动的检测器
        if hasattr(self, 'detectors') and self.detectors:
            for detector in self.detectors.values():
                detector.realtime_ocr_enabled = is_realtime_ocr
                
        status = "开启" if is_realtime_ocr else "关闭"
        logger.info(f"[Main] 实时 OCR 已{status}")
        self.statusBar().showMessage(f"实时 OCR 已{status}")

    def _on_gate_guard_changed(self, state):
        """处理龙门安全模式开关切换"""
        enabled = (state == Qt.Checked)
        self._gate_guard_enabled = bool(enabled)
        self.config['gate_guard_enabled'] = bool(enabled)

        if self.database:
            self.database.set_config('gate_guard_enabled', '1' if enabled else '0')

        if hasattr(self, 'detectors') and self.detectors:
            for detector in self.detectors.values():
                if hasattr(detector, 'set_gate_guard_enabled'):
                    detector.set_gate_guard_enabled(enabled)
                else:
                    detector.enable_gate_guard = bool(enabled)

        self._save_config()
        status = "开启" if enabled else "关闭"
        logger.info(f"[Main] 龙门安全模式已{status}")
        self.statusBar().showMessage(f"龙门安全模式已{status}")

    def _is_wave_mode_active(self) -> bool:
        return bool(self._wave_mode_enabled)

    def _enqueue_wave_ocr_candidate(self, event_id: int):
        if event_id is None:
            return
        try:
            eid = int(event_id)
        except Exception:
            return
        if eid in self._wave_pending_event_set:
            return
        self._wave_pending_event_set.add(eid)
        self._wave_pending_event_ids.append(eid)

    def _drain_wave_ocr_batch(self, reason: str):
        if not self._is_wave_mode_active():
            return
        if not self.ocr_manager or not self.database:
            return

        if self._wave_ocr_paused_by_fps:
            return

        if not self._wave_pending_event_ids:
            return

        # 若 OCR 队列已有积压，避免继续加压
        try:
            task_queue = getattr(self.ocr_manager, 'task_queue', None)
            queue_size = task_queue.qsize() if task_queue is not None else 0
            if queue_size >= 2:
                return
        except Exception:
            pass

        batch_limit = max(1, int(self._wave_batch_limit))
        take_n = min(batch_limit, len(self._wave_pending_event_ids))
        batch_ids = self._wave_pending_event_ids[:take_n]
        self._wave_pending_event_ids = self._wave_pending_event_ids[take_n:]

        enqueued = 0
        for eid in batch_ids:
            self._wave_pending_event_set.discard(eid)
            try:
                ev = self.database.get_event(eid)
                if not ev:
                    continue
                bib = str(ev.get('bib_number') or '').strip().upper()
                ocr_state = str(ev.get('ocr_state') or 'PENDING').strip().upper()
                should_enqueue = (not bib or bib == 'UNKNOWN' or ocr_state in {'PENDING', 'FAIL'})
                if not should_enqueue:
                    continue
                ok = self.ocr_manager.enqueue_live_event(
                    event_id=eid,
                    evidence_dir=ev.get('evidence_dir'),
                    cross_time=ev.get('cross_time')
                )
                if ok:
                    enqueued += 1
            except Exception:
                continue

        if enqueued > 0:
            logger.info(f"[Main] 波次OCR入队: +{enqueued} (reason={reason})")

    def _update_detector_athletes(self):
        """同步数据库中的选手名单特征到所有检测器 (支持多机位)"""
        if not self.detectors or not self.database:
            return
            
        try:
            # 获取数据库中的选手名单特征 (包含号码段 bib_ranges_str)
            summary = self.database.get_athlete_summary()
            
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
                # 马拉松模式：虽然不启用名单严格过滤，但仍然同步号码段规则以提高权重
                # 我们传一个只含号码段规则的 summary
                ranges_only_summary = {
                    'bibs': set(),
                    'max_numeric': 0,
                    'allowed_chars': set(),
                    'has_alpha': False,
                    'bib_ranges_str': summary.get('bib_ranges_str', '')
                }
                for detector in self.detectors.values():
                    detector.set_athlete_list(ranges_only_summary)
                logger.info("[Main] 马拉松模式: 已同步号码段规则到所有检测器 (不启用严格名单过滤)")
                
        except Exception as e:
            logger.error(f"[Main] 同步选手名单失败: {e}")

    def _toggle_running(self):
        """切换运行状态"""
        if self._running:
            self._stop()
        else:
            self._start()

    def _start(self):
        """启动 (支持多机位)"""
        if not self._race_ready:
            QMessageBox.warning(self, "提示", "请先选择或创建赛事。")
            self._prompt_race_selection()
            return
        logger.info(f"正在启动 AI 引擎，机位数量: {len(self.sources)}")
        if not self._init_components():
            return

        self._running = True
        self._start_time = time.time()
        self._frame_count = 0
        self._fps_timestamps = deque(maxlen=30)  # 滑动窗口FPS：最近30帧

        # 每次启动都重置波次状态，避免上一次残留队列影响本次实时性
        self._wave_new_events_since_last_batch = 0
        self._wave_last_crossing_walltime = 0.0
        self._wave_ocr_paused_by_fps = False
        self._wave_pending_event_ids.clear()
        self._wave_pending_event_set.clear()
        self._live_monitor_samples.clear()
        self._live_monitor_last_warn_ts.clear()
        self._live_monitor_last_render_ts = 0.0
        self._refresh_live_monitor_badge(self._start_time)

        if self._is_wave_mode_active() and hasattr(self, 'realtime_ocr_checkbox'):
            self.realtime_ocr_checkbox.blockSignals(True)
            self.realtime_ocr_checkbox.setChecked(False)
            self.realtime_ocr_checkbox.blockSignals(False)
            if self.database:
                self.database.set_config('realtime_ocr', '0')
            for detector in self.detectors.values():
                detector.realtime_ocr_enabled = False
            logger.info("[Main] 波次模式已启用：实时OCR关闭，转为波次分批OCR")

        # 启动所有机位的视频线程
        for source_id in self.readers:
            reader = self.readers[source_id]
            detector = self.detectors.get(source_id)
            
            if reader and detector:
                ui_skip = 1 if source_id == 0 else 2
                thread = VideoThread(reader, detector, source_id=source_id, ui_skip=ui_skip)
                thread.frame_ready.connect(self._on_frame_ready)
                thread.event_detected.connect(self._on_event_detected)
                thread.bib_updated.connect(self._on_bib_updated)
                thread.status_changed.connect(self._update_conn_status)
                thread.start()
                self.video_threads[source_id] = thread
                
                # 初始状态同步
                self._update_conn_status(reader.status, source_id)

        self.start_btn.setText("停止")
        self.start_btn.setObjectName("stop_btn")
        self.start_btn.setStyle(self.start_btn.style())  # 刷新样式
        self.reset_btn.setEnabled(True)
        self.clear_btn.setEnabled(False)  # 运行中禁止清空记录
        
        # 强制刷新一次列表，确保显示最新状态
        if hasattr(self, 'event_list'):
            self.event_list.refresh_list()

        if self._ocr_patrol_enabled and not self._ocr_patrol_timer.isActive():
            self._ocr_patrol_timer.start()
            logger.info(
                f"[Main] 自动巡检补号已启动: interval={self._ocr_patrol_timer.interval()}ms, "
                f"recent={self._ocr_patrol_recent_seconds}s"
            )
            
        self.statusBar().showMessage(f"运行中 (已连接 {len(self.video_threads)} 路机位)")

    def _stop(self):
        """停止 (支持多机位)"""
        self._running = False

        if self._ocr_patrol_timer.isActive():
            self._ocr_patrol_timer.stop()

        # 停止所有视频线程
        for source_id, thread in self.video_threads.items():
            try:
                thread.frame_ready.disconnect()
                thread.event_detected.disconnect()
                thread.bib_updated.disconnect()
                thread.status_changed.disconnect()
            except:
                pass
            thread.stop()
        self.video_threads.clear()

        # 停止所有读取器
        for reader in self.readers.values():
            reader.stop()

        if self.recorder:
            self.recorder.stop()

        self.start_btn.setText("开始")
        self.start_btn.setObjectName("start_btn")
        self.start_btn.setStyle(self.start_btn.style())  # 刷新样式
        self.reset_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)   # 停止后可以清空
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
                    ocr=self.shared_ocr,
                    ocr_engine=self.ocr_engine,
                    only_numeric=self.numeric_only_checkbox.isChecked(),
                    realtime_ocr=(False if self._is_wave_mode_active() else self.realtime_ocr_checkbox.isChecked()),
                    gate_guard_enabled=bool(self._gate_guard_enabled),
                    athlete_validator=self.shared_athlete_validator,
                    performance_profile=str(self.config.get("performance_profile") or "auto"),
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
                self.conn_status_indicator.setText(f"● {text}")
                self.conn_status_indicator.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 16px; margin-right: 10px;")

    def _on_frame_ready(self, frame, athletes, bibs, source_id=0):
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
        if hasattr(self, last_update_attr):
            if now - getattr(self, last_update_attr) < 0.04:  # 1/25s
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
        display = frame.copy()
        orig_h, orig_w = display.shape[:2]

        # 绘制运动员（绿色/红色）
        for athlete in athletes:
            x1, y1, x2, y2 = athlete['bbox']
            track_id = athlete.get('track_id', -1)
            bib_text = athlete.get('bib_text', "")
            
            # 统一使用绿色框
            is_crossed = athlete.get('is_crossed', False)
            color = (0, 255, 0) # 始终使用绿色
            thickness = 2
            
            cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
            
            label = f"ID:{track_id}{bib_text}"
            if is_crossed:
                label += " [CROSSED]"
            
            # 减小字体大小和粗细
            font_scale = 0.6
            font_thickness = 1
            cv2.putText(display, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

        # 绘制BIB（蓝色）
        for bib in bibs:
            if not isinstance(bib, dict) or 'bbox' not in bib:
                continue
            x1, y1, x2, y2 = bib['bbox']
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # 绘制终点线 (仅在未开启控件叠加绘制时，避免出现两条线)
        line_config = self.config.get('finish_lines', {}).get(str(source_id))
        if not line_config and source_id == 0:
            line_config = self.config.get('finish_line') # 兼容旧版
            
        if line_config and not getattr(video_label, 'show_line', False):
            lx1, ly1 = line_config.get('x1'), line_config.get('y1')
            lx2, ly2 = line_config.get('x2'), line_config.get('y2')
            if all(v is not None for v in [lx1, ly1, lx2, ly2]):
                cv2.line(display, (lx1, ly1), (lx2, ly2), (0, 0, 255), 2)
                cv2.putText(display, "FINISH LINE", (lx1, ly1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        # 绘制叠加信息
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (orig_w, 85), (0, 0, 0), -1)
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

            if self._is_wave_mode_active():
                if (not self._wave_ocr_paused_by_fps) and self._main_fps_value < float(self._wave_pause_fps):
                    self._wave_ocr_paused_by_fps = True
                    logger.info(f"[Main] 波次OCR暂停: FPS={self._main_fps_value:.1f} < {self._wave_pause_fps:.1f}")
                elif self._wave_ocr_paused_by_fps and self._main_fps_value >= float(self._wave_resume_fps):
                    self._wave_ocr_paused_by_fps = False
                    logger.info(f"[Main] 波次OCR恢复: FPS={self._main_fps_value:.1f} >= {self._wave_resume_fps:.1f}")

                now_wall = time.time()
                idle_gap = now_wall - float(self._wave_last_crossing_walltime or 0.0)
                if self._wave_new_events_since_last_batch >= int(self._wave_trigger_count):
                    self._drain_wave_ocr_batch(reason="count")
                    self._wave_new_events_since_last_batch = 0
                elif self._wave_last_crossing_walltime > 0 and idle_gap >= float(self._wave_idle_seconds):
                    self._drain_wave_ocr_batch(reason="idle")
                    self._wave_new_events_since_last_batch = 0
            
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

        label_size = video_label.size()
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            label_size, Qt.KeepAspectRatio, Qt.FastTransformation
        )

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

    def _on_event_saved_callback(self, event_id: int):
        """记录器保存完事件后的回调 (在后台线程执行)"""
        # 记录器保存完后，触发列表增量刷新
        if hasattr(self, 'event_list'):
            # 使用 QMetaObject.invokeMethod 确保在 UI 线程执行
            # 触发 _check_new_events 进行增量刷新，而不是全量 refresh_list
            from PyQt5.QtCore import QMetaObject, Qt
            QMetaObject.invokeMethod(self.event_list, "_check_new_events", Qt.QueuedConnection)

        # 自动补号：事件落库后自动将 UNKNOWN/PENDING 事件加入 OCR 队列
        try:
            if self.ocr_manager and self.database:
                ev = self.database.get_event(event_id)
                if ev:
                    if self._is_wave_mode_active():
                        self._enqueue_wave_ocr_candidate(int(event_id))
                        return
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

    def _run_auto_ocr_patrol(self):
        """定时巡检：自动重试最近 UNKNOWN/PENDING 事件，减少偶发漏触发。"""
        if self._is_wave_mode_active():
            return
        if not self._running:
            return
        if not self._ocr_patrol_enabled:
            return
        if not self.ocr_manager or not self.database:
            return

        # OCR 正在处理且队列已有积压时，本轮不再补充，避免 UI 高频刷新
        try:
            task_queue = getattr(self.ocr_manager, 'task_queue', None)
            queue_size = task_queue.qsize() if task_queue is not None else 0
            if getattr(self.ocr_manager, '_running', False) and queue_size >= 2:
                return
        except Exception:
            pass

        try:
            events = self.database.get_events_since_time(seconds=float(self._ocr_patrol_recent_seconds))
            if not events:
                return

            now = time.time()
            enqueued_count = 0
            scanned = 0
            max_scan = max(1, int(self._ocr_patrol_max_scan))
            retry_gap = max(1.0, float(self._ocr_patrol_retry_gap_seconds))
            max_enqueue = max(1, int(self._ocr_patrol_max_enqueue_per_tick))
            max_retries = max(1, int(self._ocr_patrol_max_retries))
            retry_cooldown = max(10.0, float(self._ocr_patrol_retry_cooldown_seconds))

            for ev in events:
                if scanned >= max_scan:
                    break
                if enqueued_count >= max_enqueue:
                    break
                scanned += 1

                event_id = ev.get('event_id')
                if event_id is None:
                    continue

                try:
                    event_id_int = int(event_id)
                except Exception:
                    continue

                bib = str(ev.get('bib_number') or '').strip().upper()
                ocr_state = str(ev.get('ocr_state') or 'PENDING').strip().upper()
                should_enqueue = (not bib or bib == 'UNKNOWN' or ocr_state in {'PENDING', 'FAIL'})
                if not should_enqueue:
                    self._ocr_patrol_attempts.pop(event_id_int, None)
                    self._ocr_patrol_block_until.pop(event_id_int, None)
                    continue

                blocked_until = float(self._ocr_patrol_block_until.get(event_id_int, 0.0))
                if now < blocked_until:
                    continue

                last_try = float(self._ocr_patrol_last_enqueue.get(event_id_int, 0.0))
                if now - last_try < retry_gap:
                    continue

                attempts = int(self._ocr_patrol_attempts.get(event_id_int, 0))
                if attempts >= max_retries:
                    self._ocr_patrol_block_until[event_id_int] = now + retry_cooldown
                    self._ocr_patrol_attempts[event_id_int] = 0
                    continue

                enqueued = self.ocr_manager.enqueue_live_event(
                    event_id=event_id_int,
                    evidence_dir=ev.get('evidence_dir'),
                    cross_time=ev.get('cross_time')
                )
                self._ocr_patrol_last_enqueue[event_id_int] = now
                if enqueued:
                    self._ocr_patrol_attempts[event_id_int] = attempts + 1
                    if self._ocr_patrol_attempts[event_id_int] >= max_retries:
                        self._ocr_patrol_block_until[event_id_int] = now + retry_cooldown
                    enqueued_count += 1

            # 定期清理巡检缓存，避免无限增长
            expire_before = now - max(60.0, retry_gap * 8.0)
            stale_keys = [eid for eid, ts in self._ocr_patrol_last_enqueue.items() if float(ts) < expire_before]
            for eid in stale_keys:
                self._ocr_patrol_last_enqueue.pop(eid, None)
                self._ocr_patrol_attempts.pop(eid, None)
                self._ocr_patrol_block_until.pop(eid, None)

            if enqueued_count > 0:
                logger.info(f"[Main] 自动巡检补号入队: +{enqueued_count} (scan={scanned})")
        except Exception as e:
            logger.debug(f"[Main] 自动巡检补号异常: {e}")

    def _on_event_detected(self, event: CrossingEvent):
        """检测到过线事件 (在主线程执行)"""
        if not self._running:
            return
        # 1. 更新本次会话计数
        self._session_event_count += 1
        self._wave_last_crossing_walltime = time.time()
        if self._is_wave_mode_active():
            self._wave_new_events_since_last_batch += 1
        logger.info(f"[MainWindow] 检测到过线事件: ID {event.track_id}, Bib {event.bib_number}, Source {event.source_id}")
        
        # 2. 保存到记录器 (异步处理 IO)
        if self.recorder:
            logger.debug(f"[MainWindow] 正在提交给 EventRecorder 保存...")
            if not getattr(self.recorder, "_running", False):
                logger.warning("[MainWindow] EventRecorder 未运行，尝试自动启动...")
                self.recorder.start()
            self.recorder.record(event)
            
        # 3. 更新界面实时面板
        if hasattr(self, 'rank_label'):
            self.rank_label.setText(f"RANK {self._session_event_count}")
        if hasattr(self, 'time_label'):
            self.time_label.setText(event.cross_time_str)
        if hasattr(self, 'index_label'):
            self.index_label.setText(f"ID: {event.track_id}")
            
        # 4. 刷新右侧列表
        # 注意：此处不再直接刷新，而是等待 recorder 保存完后的回调触发，确保数据已落库
        # if hasattr(self, 'event_list'):
        #     self.event_list.refresh_list()
            
        # 5. 状态栏提示
        self.statusBar().showMessage(f"检测到过线: ID {event.track_id} | 号码: {event.bib_number or '未知'}", 3000)
        
        # 6. 更新统计信息
        if hasattr(self, 'stats_status_label'):
            # 不再实时查询数据库总数，避免主线程 IO 阻塞
            # 数据库总数可以通过其他方式异步更新，这里只更新本次会话计数
            self.stats_status_label.setText(f"本次会话: {self._session_event_count}")

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
                    if hasattr(self, 'event_list') and (not self._is_wave_mode_active()):
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
            # 1. 停止 OCR 初始化线程
            if getattr(self, 'ocr_init_thread', None):
                thread = self.ocr_init_thread
                if thread.isRunning():
                    logger.info("[Main] 正在停止 OCR 初始化线程...")
                    try:
                        thread.finished.disconnect()
                    except Exception:
                        pass
                    thread.terminate()
                    thread.wait(1000)
                self.ocr_init_thread = None

            # 2. 移除日志 handler，防止后台线程继续写入已销毁组件
            import logging
            root_logger = logging.getLogger()
            if getattr(self, '_gui_log_handler', None):
                try:
                    root_logger.removeHandler(self._gui_log_handler)
                except Exception:
                    pass
                self._gui_log_handler = None
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


def find_model() -> str:
    """自动查找模型文件"""
    from pathlib import Path

    # 获取脚本所在目录
    script_dir = Path(__file__).parent.parent

    # 搜索常见位置
    search_paths = [
        script_dir / "runs" / "detect",
        script_dir / "runs" / "detect" / "runs" / "detect",
        Path("runs/detect"),
        Path("runs/detect/runs/detect"),
    ]

    candidates = []
    for base in search_paths:
        if not base.exists():
            continue
        for pattern in ["**/best.engine", "**/best.pt"]:
            candidates.extend(list(base.glob(pattern)))
    if candidates:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(candidates[0])

    return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description='实时计时系统GUI')
    parser.add_argument('--source', type=str, default='rtsp://localhost:8554/test',
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

    args = parser.parse_args()

    # 查找模型
    model_path = args.model
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

    config = {
        'source': args.source,
        'model_path': model_path,
        'output_dir': args.output,
        'finish_line': {
            'x1': x1, 'y1': y1,
            'x2': x2, 'y2': y2,
        },
        'vlm_config': {
            'enabled': False,
            'model_type': 'doubao',
            'api_key': '',
            'endpoint_id': '',
            'max_calls_per_minute': 30
        }
    }

    # 尝试加载上次保存的配置
    output_path = Path(args.output)
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

    app = QApplication(sys.argv)
    window = MainWindow(config)
    window.show()
    
    if args.auto_start:
        logger.info("Auto-Start: 正在启动系统...")
        # 延迟一下，等 UI 稳定
        QTimer.singleShot(1000, window._start)
        
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
