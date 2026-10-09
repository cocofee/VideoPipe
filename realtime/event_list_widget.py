"""
过线事件列表组件

实时显示过线记录，支持编辑和筛选
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QComboBox, QLabel, QLineEdit, QHeaderView, QMenu,
    QMessageBox, QAbstractItemView
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QColor, QBrush
from typing import List, Dict, Any, Optional
from .database import Database


class EventListWidget(QWidget):
    """
    过线事件列表组件

    功能：
    - 实时显示过线记录
    - 新记录自动插入顶部
    - 支持修改号码、作废、查看截图
    """

    # 信号
    event_selected = pyqtSignal(dict)  # 选中事件
    view_screenshot = pyqtSignal(str, dict)  # 查看截图 (path, event_data)
    bib_updated = pyqtSignal(int, str)  # 号码更新 (event_id, new_bib)

    def __init__(self, database: Database, parent=None):
        super().__init__(parent)
        self.database = database
        self._last_event_id = 0

        self._init_ui()
        self._setup_timer()

    def _init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # 顶部工具栏
        toolbar = QHBoxLayout()

        # 筛选下拉框
        self.filter_combo = QComboBox()
        self.filter_combo.addItems([
            "全部", "已识别", "需复核", "未识别", "已作废"
        ])
        self.filter_combo.currentTextChanged.connect(self._on_filter_changed)
        toolbar.addWidget(QLabel("筛选:"))
        toolbar.addWidget(self.filter_combo)

        toolbar.addStretch()

        # 统计标签
        self.stats_label = QLabel("共 0 条记录")
        toolbar.addWidget(self.stats_label)

        # 刷新按钮
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh_list)
        toolbar.addWidget(refresh_btn)

        layout.addLayout(toolbar)

        # 事件表格
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "名次", "号码", "状态", "过线时间", "置信度", "ID", "操作"
        ])

        # 表格设置
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)
        self.table.cellDoubleClicked.connect(self._on_cell_double_clicked)

        # 列宽
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # 名次
        header.setSectionResizeMode(1, QHeaderView.Stretch)          # 号码
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # 状态
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # 时间
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # 置信度
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # ID
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)  # 操作

        layout.addWidget(self.table)

    def _setup_timer(self):
        """设置定时刷新"""
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self._check_new_events)
        self.refresh_timer.start(1000)  # 每1000ms检查一次（降低刷新频率）

    def _check_new_events(self):
        """检查新事件"""
        latest_id = self.database.get_latest_event_id()
        if latest_id > self._last_event_id:
            self.refresh_list()

    def refresh_list(self):
        """刷新列表"""
        filter_text = self.filter_combo.currentText()

        # 获取数据
        include_void = (filter_text == "已作废" or filter_text == "全部")
        events = self.database.get_all_events(include_void=include_void)

        # 筛选
        if filter_text == "已识别":
            events = [e for e in events if e['bib_status'] == 'recognized']
        elif filter_text == "需复核":
            events = [e for e in events if e['bib_status'] == 'needs_review']
        elif filter_text == "未识别":
            events = [e for e in events if e['bib_status'] == 'unrecognized']
        elif filter_text == "已作废":
            events = [e for e in events if e['is_void']]

        # 更新表格
        self.table.setRowCount(len(events))

        for row, event in enumerate(events):
            self._set_row_data(row, event)

        # 更新统计
        self.stats_label.setText(f"共 {len(events)} 条记录")

        # 更新最后ID
        if events:
            self._last_event_id = max(e['event_id'] for e in events)

    def _set_row_data(self, row: int, event: dict):
        """设置行数据"""
        # 名次
        rank_item = QTableWidgetItem(str(event['rank']))
        rank_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, 0, rank_item)

        # 号码
        bib = event['bib_number'] if event['bib_number'] else "未识别"
        bib_item = QTableWidgetItem(bib)
        bib_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, 1, bib_item)

        # 状态
        status_map = {
            'recognized': '已识别',
            'needs_review': '需复核',
            'unrecognized': '未识别'
        }
        status = status_map.get(event['bib_status'], event['bib_status'])
        if event['is_void']:
            status = '已作废'
        status_item = QTableWidgetItem(status)
        status_item.setTextAlignment(Qt.AlignCenter)

        # 状态颜色
        if event['is_void']:
            status_item.setBackground(QBrush(QColor(200, 200, 200)))
        elif event['bib_status'] == 'recognized':
            status_item.setBackground(QBrush(QColor(200, 255, 200)))
        elif event['bib_status'] == 'needs_review':
            status_item.setBackground(QBrush(QColor(255, 255, 200)))
        else:
            status_item.setBackground(QBrush(QColor(255, 200, 200)))

        self.table.setItem(row, 2, status_item)

        # 过线时间
        time_item = QTableWidgetItem(event['cross_time_str'])
        time_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, 3, time_item)

        # 置信度
        conf = event['bib_confidence']
        conf_text = f"{conf:.2f}" if conf else "-"
        conf_item = QTableWidgetItem(conf_text)
        conf_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, 4, conf_item)

        # ID（隐藏数据）
        id_item = QTableWidgetItem(str(event['event_id']))
        id_item.setTextAlignment(Qt.AlignCenter)
        id_item.setData(Qt.UserRole, event)  # 存储完整事件数据
        self.table.setItem(row, 5, id_item)

        # 操作按钮
        btn_widget = QWidget()
        btn_layout = QHBoxLayout(btn_widget)
        btn_layout.setContentsMargins(2, 2, 2, 2)
        btn_layout.setSpacing(2)

        view_btn = QPushButton("查看")
        view_btn.setFixedWidth(40)
        view_btn.clicked.connect(lambda checked, e=event: self._view_event(e))
        btn_layout.addWidget(view_btn)

        self.table.setCellWidget(row, 6, btn_widget)

    def _on_filter_changed(self, text: str):
        """筛选变化"""
        self.refresh_list()

    def _on_cell_double_clicked(self, row: int, col: int):
        """双击单元格"""
        if col == 1:  # 号码列
            self._edit_bib_number(row)
        else:
            # 查看截图
            event = self._get_event_at_row(row)
            if event:
                self._view_event(event)

    def _get_event_at_row(self, row: int) -> Optional[dict]:
        """获取指定行的事件"""
        id_item = self.table.item(row, 5)
        if id_item:
            return id_item.data(Qt.UserRole)
        return None

    def _view_event(self, event: dict):
        """查看事件"""
        self.event_selected.emit(event)
        # 优先查看完整截图（可以看到目标骑手标记）
        screenshot = event.get('screenshot_full') or event.get('screenshot_crop')
        if screenshot:
            self.view_screenshot.emit(screenshot, event)

    def _edit_bib_number(self, row: int):
        """编辑号码"""
        event = self._get_event_at_row(row)
        if not event:
            return

        from PyQt5.QtWidgets import QInputDialog
        current = event['bib_number'] or ""
        new_bib, ok = QInputDialog.getText(
            self, "修改号码",
            f"请输入名次 {event['rank']} 的号码:",
            text=current
        )

        if ok and new_bib != current:
            # 更新数据库
            self.database.update_event(event['event_id'], {
                'bib_number': new_bib if new_bib else None,
                'bib_status': 'recognized' if new_bib else 'unrecognized'
            })
            self.refresh_list()

    def _show_context_menu(self, pos):
        """显示右键菜单"""
        row = self.table.rowAt(pos.y())
        if row < 0:
            return

        event = self._get_event_at_row(row)
        if not event:
            return

        menu = QMenu(self)

        # 修改号码
        edit_action = menu.addAction("修改号码")
        edit_action.triggered.connect(lambda: self._edit_bib_number(row))

        # 查看截图
        view_action = menu.addAction("查看截图")
        view_action.triggered.connect(lambda: self._view_event(event))

        menu.addSeparator()

        # 作废/取消作废
        if event['is_void']:
            unvoid_action = menu.addAction("取消作废")
            unvoid_action.triggered.connect(
                lambda: self._toggle_void(event['event_id'], False))
        else:
            void_action = menu.addAction("标记作废")
            void_action.triggered.connect(
                lambda: self._toggle_void(event['event_id'], True))

        menu.exec_(self.table.mapToGlobal(pos))

    def _toggle_void(self, event_id: int, void: bool):
        """切换作废状态"""
        if void:
            self.database.void_event(event_id)
        else:
            self.database.unvoid_event(event_id)
        self.refresh_list()

    def add_event(self, event: dict):
        """添加新事件（实时插入）"""
        # 插入到顶部
        self.table.insertRow(0)
        self._set_row_data(0, event)

        # 更新统计
        count = self.table.rowCount()
        self.stats_label.setText(f"共 {count} 条记录")

        # 更新最后ID
        self._last_event_id = max(self._last_event_id, event['event_id'])
