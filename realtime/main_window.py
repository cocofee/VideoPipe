"""
实时计时系统 - 主窗口

一体化界面：左边实时画面，右边过线记录列表
"""

import cv2
import sys
import time
import threading
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QSplitter, QStatusBar, QFileDialog,
    QMessageBox, QDialog, QScrollArea, QLineEdit
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap
from typing import Optional

from .stream_reader import StreamReader, StreamStatus
from .detector import Detector, CrossingEvent
from .database import Database
from .event_recorder import EventRecorder
from .event_list_widget import EventListWidget


class VideoThread(QThread):
    """视频处理线程"""
    frame_ready = pyqtSignal(object, list, list)  # frame, bikes, bibs
    event_detected = pyqtSignal(object)  # CrossingEvent

    def __init__(self, reader: StreamReader, detector: Detector, skip_frames: int = 2):
        super().__init__()
        self.reader = reader
        self.detector = detector
        self.skip_frames = skip_frames  # 每N帧检测一次
        self._running = False

    def run(self):
        self._running = True
        frame_count = 0
        last_bikes = []
        last_bibs = []
        first_frame_ms = None
        last_sequence, last_generation = 0, ""

        while self._running:
            frame, stamp = self.reader.get_frame_packet_after(last_sequence, last_generation)
            if frame is None:
                time.sleep(0.01)
                continue

            last_sequence, last_generation = stamp.sequence, stamp.generation
            if first_frame_ms is None:
                first_frame_ms = stamp.observed_at_ms
            frame_count += 1

            # 跳帧处理：只有每N帧才进行检测
            if frame_count % self.skip_frames == 0:
                events, bikes, bibs = self.detector.process_frame(frame, timestamp=(stamp.observed_at_ms - first_frame_ms) / 1000, frame_stamp=stamp)
                last_bikes = bikes
                last_bibs = bibs

                # 发送事件
                for event in events:
                    self.event_detected.emit(event)
            else:
                # 非检测帧，使用上一次的检测结果
                bikes = last_bikes
                bibs = last_bibs

            # 发送帧（降低发送频率）
            if frame_count % 2 == 0:  # 每2帧发送一次显示
                self.frame_ready.emit(frame, bikes, bibs)

    def stop(self):
        self._running = False
        self.wait(2000)


class ImageViewerDialog(QDialog):
    """图片查看对话框 - 支持滚轮缩放和拖拽"""

    # 信号：号码已修改 (event_id, new_bib)
    bib_changed = pyqtSignal(int, str)
    # 信号：作废状态变化 (event_id, is_void)
    void_changed = pyqtSignal(int, bool)

    def __init__(self, image_path: str, parent=None, event_data: dict = None):
        super().__init__(parent)
        self.setWindowTitle("查看截图 - 滚轮缩放 | 拖拽移动")
        self.setMinimumSize(900, 700)

        self.image_path = image_path
        self.event_data = event_data  # 事件数据
        self.original_pixmap = None
        self.scale_factor = 1.0
        self.min_scale = 0.1
        self.max_scale = 10.0
        self.target_bbox = None  # 目标骑手边界框

        # 拖拽相关
        self.dragging = False
        self.last_pos = None

        layout = QVBoxLayout(self)

        # 顶部信息栏（如果有事件数据）
        if event_data:
            info_layout = QHBoxLayout()

            rank = event_data.get('rank', '-')
            time_str = event_data.get('cross_time_str', '-')
            current_bib = event_data.get('bib_number', '')
            is_void = event_data.get('is_void', 0)

            info_layout.addWidget(QLabel(f"名次: {rank}"))
            info_layout.addWidget(QLabel(f"时间: {time_str}"))
            info_layout.addWidget(QLabel("|"))
            info_layout.addWidget(QLabel("号码:"))

            self.bib_input = QLineEdit()
            self.bib_input.setText(current_bib if current_bib else "")
            self.bib_input.setPlaceholderText("输入号码")
            self.bib_input.setFixedWidth(100)
            info_layout.addWidget(self.bib_input)

            save_btn = QPushButton("保存号码")
            save_btn.clicked.connect(self._save_bib)
            info_layout.addWidget(save_btn)

            info_layout.addWidget(QLabel("|"))

            # 作废按钮
            self.void_btn = QPushButton("取消作废" if is_void else "标记作废")
            self.void_btn.setStyleSheet("background-color: #ffcccc;" if not is_void else "")
            self.void_btn.clicked.connect(self._toggle_void)
            info_layout.addWidget(self.void_btn)

            info_layout.addStretch()
            layout.addLayout(info_layout)

        # 滚动区域
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(False)
        self.scroll.setAlignment(Qt.AlignCenter)
        self.scroll.viewport().setCursor(Qt.OpenHandCursor)

        # 图片标签
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.scroll.setWidget(self.image_label)

        layout.addWidget(self.scroll)

        # 底部工具栏
        toolbar = QHBoxLayout()

        # 缩放信息
        self.zoom_label = QLabel("100%")
        toolbar.addWidget(self.zoom_label)

        toolbar.addStretch()

        # 缩放按钮
        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setFixedWidth(40)
        zoom_in_btn.clicked.connect(self._zoom_in)
        toolbar.addWidget(zoom_in_btn)

        zoom_out_btn = QPushButton("-")
        zoom_out_btn.setFixedWidth(40)
        zoom_out_btn.clicked.connect(self._zoom_out)
        toolbar.addWidget(zoom_out_btn)

        reset_btn = QPushButton("原始大小")
        reset_btn.clicked.connect(self._reset_zoom)
        toolbar.addWidget(reset_btn)

        fit_btn = QPushButton("适应窗口")
        fit_btn.clicked.connect(self._fit_window)
        toolbar.addWidget(fit_btn)

        focus_btn = QPushButton("定位目标")
        focus_btn.clicked.connect(self._focus_on_target)
        toolbar.addWidget(focus_btn)

        toolbar.addStretch()

        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        toolbar.addWidget(close_btn)

        layout.addLayout(toolbar)

        # 安装事件过滤器
        self.scroll.viewport().installEventFilter(self)

        # 加载图片并绘制标记框
        self._load_and_mark_image(image_path)

        # 延迟聚焦到目标（等待窗口完全显示后）
        QTimer.singleShot(100, self._focus_on_target)

    def _load_and_mark_image(self, image_path: str):
        """加载图片并在full截图上绘制目标骑手标记"""
        import json

        # 判断是否是full截图（需要标记）
        is_full_screenshot = image_path and 'full_' in image_path

        if is_full_screenshot and self.event_data:
            # 使用OpenCV加载并绘制标记
            img = cv2.imread(image_path)
            if img is not None:
                # 获取bbox
                bbox_data = self.event_data.get('bbox', '[]')
                if isinstance(bbox_data, str):
                    try:
                        bbox = json.loads(bbox_data)
                    except:
                        bbox = []
                else:
                    bbox = bbox_data

                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                    self.target_bbox = (x1, y1, x2, y2)  # 保存bbox用于居中
                    # 绘制醒目的红色框（粗线）
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4)
                    # 绘制"目标"标签
                    label = ">>> TARGET <<<"
                    font_scale = 1.0
                    thickness = 2
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    # 标签背景
                    cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), (0, 0, 255), -1)
                    # 标签文字
                    cv2.putText(img, label, (x1 + 5, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

                # 转换为QPixmap
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = img_rgb.shape
                bytes_per_line = ch * w
                qt_image = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.original_pixmap = QPixmap.fromImage(qt_image)
            else:
                self.original_pixmap = QPixmap(image_path)
        else:
            # 非full截图，直接加载
            self.original_pixmap = QPixmap(image_path)

        if not self.original_pixmap.isNull():
            self._update_display()

    def _focus_on_target(self):
        """聚焦到目标骑手：自动缩放并居中显示"""
        if self.target_bbox is None or self.original_pixmap is None:
            # 没有目标bbox，执行适应窗口
            self._fit_window()
            return

        x1, y1, x2, y2 = self.target_bbox
        target_w = x2 - x1
        target_h = y2 - y1
        target_cx = (x1 + x2) // 2
        target_cy = (y1 + y2) // 2

        # 获取视口大小
        viewport = self.scroll.viewport()
        vw, vh = viewport.width(), viewport.height()

        # 计算缩放比例：让目标区域占视口的60%左右，并留有上下文
        # 扩大目标区域以包含一些上下文
        padding = max(target_w, target_h) * 0.8  # 周围留80%的空间
        view_w = target_w + padding * 2
        view_h = target_h + padding * 2

        scale_w = vw / view_w
        scale_h = vh / view_h
        self.scale_factor = min(scale_w, scale_h)
        self.scale_factor = max(self.min_scale, min(self.max_scale, self.scale_factor))

        # 更新显示
        self._update_display()

        # 计算滚动位置使目标居中
        # 缩放后的目标中心位置
        scaled_cx = int(target_cx * self.scale_factor)
        scaled_cy = int(target_cy * self.scale_factor)

        # 滚动到使目标居中
        scroll_x = scaled_cx - vw // 2
        scroll_y = scaled_cy - vh // 2

        self.scroll.horizontalScrollBar().setValue(max(0, scroll_x))
        self.scroll.verticalScrollBar().setValue(max(0, scroll_y))

    def _toggle_void(self):
        """切换作废状态"""
        if not self.event_data:
            return

        event_id = self.event_data.get('event_id')
        current_void = self.event_data.get('is_void', 0)
        new_void = not current_void

        if event_id:
            self.void_changed.emit(event_id, new_void)
            # 更新本地数据
            self.event_data['is_void'] = 1 if new_void else 0
            # 更新按钮
            self.void_btn.setText("取消作废" if new_void else "标记作废")
            self.void_btn.setStyleSheet("" if new_void else "background-color: #ffcccc;")
            QMessageBox.information(self, "提示", "已标记作废" if new_void else "已取消作废")

    def _save_bib(self):
        """保存号码"""
        if not self.event_data:
            return

        new_bib = self.bib_input.text().strip()
        event_id = self.event_data.get('event_id')

        if event_id:
            self.bib_changed.emit(event_id, new_bib)
            QMessageBox.information(self, "提示", f"号码已保存: {new_bib if new_bib else '(空)'}")
            # 更新本地数据
            self.event_data['bib_number'] = new_bib

    def _update_display(self):
        """更新显示"""
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

    def _zoom_in(self):
        self.scale_factor *= 1.25
        self.scale_factor = min(self.max_scale, self.scale_factor)
        self._update_display()

    def _zoom_out(self):
        self.scale_factor /= 1.25
        self.scale_factor = max(self.min_scale, self.scale_factor)
        self._update_display()

    def _reset_zoom(self):
        self.scale_factor = 1.0
        self._update_display()

    def _fit_window(self):
        """适应窗口大小"""
        if self.original_pixmap is None:
            return

        viewport = self.scroll.viewport()
        vw, vh = viewport.width() - 20, viewport.height() - 20
        iw, ih = self.original_pixmap.width(), self.original_pixmap.height()

        scale_w = vw / iw
        scale_h = vh / ih
        self.scale_factor = min(scale_w, scale_h)
        self._update_display()

    def eventFilter(self, obj, event):
        """事件过滤器 - 处理滚轮和拖拽"""
        from PyQt5.QtCore import QEvent

        if obj == self.scroll.viewport():
            if event.type() == QEvent.Wheel:
                # 滚轮缩放 - 以鼠标位置为中心
                mouse_pos = event.pos()  # 鼠标在视口中的位置

                # 计算鼠标在图片上的位置（缩放前）
                h_bar = self.scroll.horizontalScrollBar()
                v_bar = self.scroll.verticalScrollBar()

                # 鼠标指向的图片坐标（缩放后的坐标）
                img_x = h_bar.value() + mouse_pos.x()
                img_y = v_bar.value() + mouse_pos.y()

                # 转换为原始图片坐标
                orig_x = img_x / self.scale_factor
                orig_y = img_y / self.scale_factor

                # 计算新缩放比例
                delta = event.angleDelta().y()
                if delta > 0:
                    self.scale_factor *= 1.15
                else:
                    self.scale_factor /= 1.15
                self.scale_factor = max(self.min_scale, min(self.max_scale, self.scale_factor))

                # 更新显示
                self._update_display()

                # 计算新的图片坐标（保持鼠标指向的原始位置不变）
                new_img_x = orig_x * self.scale_factor
                new_img_y = orig_y * self.scale_factor

                # 调整滚动条，使鼠标位置保持不变
                new_scroll_x = int(new_img_x - mouse_pos.x())
                new_scroll_y = int(new_img_y - mouse_pos.y())

                h_bar.setValue(max(0, new_scroll_x))
                v_bar.setValue(max(0, new_scroll_y))

                return True

            elif event.type() == QEvent.MouseButtonPress:
                if event.button() == Qt.LeftButton:
                    self.dragging = True
                    self.last_pos = event.pos()
                    self.scroll.viewport().setCursor(Qt.ClosedHandCursor)
                    return True

            elif event.type() == QEvent.MouseMove:
                if self.dragging and self.last_pos:
                    delta = event.pos() - self.last_pos
                    self.last_pos = event.pos()
                    self.scroll.horizontalScrollBar().setValue(
                        self.scroll.horizontalScrollBar().value() - delta.x())
                    self.scroll.verticalScrollBar().setValue(
                        self.scroll.verticalScrollBar().value() - delta.y())
                    return True

            elif event.type() == QEvent.MouseButtonRelease:
                if event.button() == Qt.LeftButton:
                    self.dragging = False
                    self.scroll.viewport().setCursor(Qt.OpenHandCursor)
                    return True

        return super().eventFilter(obj, event)


class MainWindow(QMainWindow):
    """
    实时计时系统主窗口

    布局：
    - 左边：实时视频画面
    - 右边：过线记录列表
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # 组件
        self.reader: Optional[StreamReader] = None
        self.detector: Optional[Detector] = None
        self.database: Optional[Database] = None
        self.recorder: Optional[EventRecorder] = None
        self.video_thread: Optional[VideoThread] = None

        # 配置
        self.source = config.get('source', 'rtsp://localhost:8554/test')
        self.model_path = config.get('model_path')
        self.output_dir = Path(config.get('output_dir', './realtime_output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 终点线
        finish_line = config.get('finish_line', {})
        self.line_pt1 = (finish_line.get('x1', 0), finish_line.get('y1', 400))
        self.line_pt2 = (finish_line.get('x2', 1920), finish_line.get('y2', 400))

        # 状态
        self._running = False
        self._start_time = None
        self._frame_count = 0
        self._initialized = False  # 是否已初始化组件
        self._session_event_count = 0  # 本次会话过线数

        self._init_ui()

    def _init_ui(self):
        """初始化界面"""
        self.setWindowTitle("自行车终点计时系统 - 实时版")
        self.setMinimumSize(1400, 800)

        # 中央组件
        central = QWidget()
        self.setCentralWidget(central)

        # 主布局 - 使用分割器
        layout = QHBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)

        # 左边 - 视频区域
        video_widget = QWidget()
        video_layout = QVBoxLayout(video_widget)

        # 视频显示
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000;")
        self.video_label.setText("等待连接...")
        video_layout.addWidget(self.video_label)

        # 控制按钮
        btn_layout = QHBoxLayout()

        self.start_btn = QPushButton("开始")
        self.start_btn.clicked.connect(self._toggle_running)
        btn_layout.addWidget(self.start_btn)

        self.reset_btn = QPushButton("重置统计")
        self.reset_btn.clicked.connect(self._reset_stats)
        self.reset_btn.setEnabled(False)
        btn_layout.addWidget(self.reset_btn)

        self.clear_btn = QPushButton("清空记录")
        self.clear_btn.clicked.connect(self._clear_all_records)
        self.clear_btn.setEnabled(False)
        btn_layout.addWidget(self.clear_btn)

        btn_layout.addStretch()

        # 状态信息
        self.info_label = QLabel("FPS: - | 运行: 00:00 | 过线: 0")
        btn_layout.addWidget(self.info_label)

        video_layout.addLayout(btn_layout)

        splitter.addWidget(video_widget)

        # 右边 - 事件列表（先创建空的，启动后再设置database）
        self.list_placeholder = QLabel("点击[开始]启动系统")
        self.list_placeholder.setAlignment(Qt.AlignCenter)
        self.list_placeholder.setMinimumWidth(400)
        splitter.addWidget(self.list_placeholder)

        # 设置分割比例
        splitter.setSizes([800, 400])

        layout.addWidget(splitter)
        self.splitter = splitter

        # 状态栏
        self.statusBar().showMessage("就绪")

        # 定时更新UI
        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self._update_ui)
        self.ui_timer.start(100)

    def _init_components(self) -> bool:
        """初始化组件"""
        try:
            # 如果已经初始化过，只需要重新连接流
            if self._initialized:
                # 重新连接流
                self.reader = StreamReader(self.source)
                if not self.reader.start():
                    QMessageBox.critical(self, "错误", f"无法连接到流: {self.source}")
                    return False

                # 更新event_id
                max_event_id = self.database.get_latest_event_id()
                self.detector.reset(start_event_id=max_event_id)

                # 刷新列表
                self.event_list.refresh_list()
                return True

            # 首次初始化
            # 数据库
            db_path = self.output_dir / "timing.db"
            self.database = Database(str(db_path))

            # 事件记录器
            self.recorder = EventRecorder(str(self.output_dir), self.database)
            self.recorder.start()

            # 检测器
            if not self.model_path:
                QMessageBox.critical(self, "错误", "未指定模型路径")
                return False

            self.detector = Detector(self.model_path)
            self.detector.set_finish_line(self.line_pt1, self.line_pt2)
            self.detector.set_on_crossing(self._on_crossing_event)

            if not self.detector.load_model():
                QMessageBox.critical(self, "错误", "模型加载失败")
                return False

            # 从数据库获取已有的最大event_id
            max_event_id = self.database.get_latest_event_id()
            if max_event_id > 0:
                self.detector.reset(start_event_id=max_event_id)

            # 流读取器
            self.reader = StreamReader(self.source)
            if not self.reader.start():
                QMessageBox.critical(self, "错误", f"无法连接到流: {self.source}")
                return False

            # 创建事件列表组件
            self.event_list = EventListWidget(self.database)
            self.event_list.view_screenshot.connect(self._view_screenshot)

            # 替换占位符
            self.splitter.replaceWidget(1, self.event_list)
            self.list_placeholder.deleteLater()

            self._initialized = True
            return True

        except Exception as e:
            QMessageBox.critical(self, "错误", f"初始化失败: {e}")
            return False

    def _toggle_running(self):
        """切换运行状态"""
        if self._running:
            self._stop()
        else:
            self._start()

    def _start(self):
        """启动"""
        if not self._init_components():
            return

        self._running = True
        self._start_time = time.time()
        self._frame_count = 0

        # 启动视频线程
        self.video_thread = VideoThread(self.reader, self.detector)
        self.video_thread.frame_ready.connect(self._on_frame_ready)
        self.video_thread.event_detected.connect(self._on_event_detected)
        self.video_thread.start()

        self.start_btn.setText("停止")
        self.reset_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)
        self.statusBar().showMessage("运行中...")

    def _stop(self):
        """停止"""
        self._running = False

        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None

        if self.reader:
            self.reader.stop()

        if self.recorder:
            self.recorder.stop()

        self.start_btn.setText("开始")
        self.reset_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        self.statusBar().showMessage("已停止")

    def _reset_stats(self):
        """重置统计（重置跟踪状态和本次会话计数）"""
        if self.detector:
            max_id = self.database.get_latest_event_id() if self.database else 0
            self.detector.reset(start_event_id=max_id)
        self._frame_count = 0
        self._start_time = time.time()
        self._session_event_count = 0
        self.statusBar().showMessage("已重置统计（跟踪状态已清空）")

    def _clear_all_records(self):
        """清空所有记录"""
        reply = QMessageBox.question(
            self, "确认清空",
            "确定要清空所有过线记录吗？\n此操作不可撤销！",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            if self.database:
                self.database.clear_events()

            # 重置检测器状态
            if self.detector:
                self.detector.reset(start_event_id=0)

            # 重置计数
            self._session_event_count = 0

            # 刷新列表
            if hasattr(self, 'event_list'):
                self.event_list.refresh_list()

            self.statusBar().showMessage("已清空所有记录")

    def _on_frame_ready(self, frame, bikes, bibs):
        """收到新帧"""
        self._frame_count += 1

        # 绘制检测结果
        display = frame.copy()

        # 绘制骑手（绿色）
        for bike in bikes:
            x1, y1, x2, y2 = bike['bbox']
            track_id = bike.get('track_id', -1)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, f"ID:{track_id}", (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 绘制BIB（蓝色）
        for bib in bibs:
            x1, y1, x2, y2 = bib['bbox']
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # 绘制终点线
        cv2.line(display, self.line_pt1, self.line_pt2, (0, 0, 255), 2)

        # 转换为Qt图像
        h, w, ch = display.shape
        bytes_per_line = ch * w
        qt_image = QImage(display.data, w, h, bytes_per_line, QImage.Format_BGR888)

        # 缩放到标签大小
        label_size = self.video_label.size()
        pixmap = QPixmap.fromImage(qt_image).scaled(
            label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)

    def _on_event_detected(self, event: CrossingEvent):
        """检测到过线事件"""
        # 更新本次会话计数
        self._session_event_count += 1
        # 保存到记录器
        self.recorder.record(event)

    def _on_crossing_event(self, event: CrossingEvent):
        """过线事件回调（在检测线程中）"""
        pass  # 通过信号处理

    def _update_ui(self):
        """更新UI信息"""
        if not self._running:
            return

        # 计算FPS
        if self._start_time:
            runtime = time.time() - self._start_time
            fps = self._frame_count / runtime if runtime > 0 else 0

            # 获取数据库总数
            total_count = 0
            if self.database:
                total_count = self.database.get_event_count()

            # 更新标签（显示本次会话过线数和数据库总数）
            self.info_label.setText(
                f"FPS: {fps:.1f} | 运行: {int(runtime//60):02d}:{int(runtime%60):02d} | 本次: {self._session_event_count} | 总计: {total_count}"
            )

    def _view_screenshot(self, path: str, event_data: dict = None):
        """查看截图"""
        if path and Path(path).exists():
            dialog = ImageViewerDialog(path, self, event_data)
            dialog.bib_changed.connect(self._on_bib_changed)
            dialog.void_changed.connect(self._on_void_changed)
            dialog.exec_()
        else:
            QMessageBox.warning(self, "提示", "截图文件不存在")

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
            self.database.update_event(event_id, {
                'bib_number': new_bib if new_bib else None,
                'bib_status': 'recognized' if new_bib else 'unrecognized'
            })
            # 刷新列表
            if hasattr(self, 'event_list'):
                self.event_list.refresh_list()

    def closeEvent(self, event):
        """关闭窗口"""
        self._stop()
        event.accept()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='实时计时系统GUI')
    parser.add_argument('--source', type=str, default='rtsp://localhost:8554/test',
                       help='视频源')
    parser.add_argument('--model', type=str, required=True,
                       help='模型路径')
    parser.add_argument('--output', type=str, default='./realtime_output',
                       help='输出目录')
    parser.add_argument('--line-y', type=int, default=850,
                       help='终点线Y坐标')

    args = parser.parse_args()

    config = {
        'source': args.source,
        'model_path': args.model,
        'output_dir': args.output,
        'finish_line': {
            'x1': 0, 'y1': args.line_y,
            'x2': 1920, 'y2': args.line_y,
        },
    }

    app = QApplication(sys.argv)
    window = MainWindow(config)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
