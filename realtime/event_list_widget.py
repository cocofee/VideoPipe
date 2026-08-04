"""
过线事件列表组件

实时显示过线记录，支持编辑和筛选
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QComboBox, QLabel, QLineEdit, QHeaderView, QMenu,
    QMessageBox, QAbstractItemView
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QThread, QMetaObject, pyqtSlot
from PyQt5.QtGui import QColor, QBrush, QFont
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional
try:
    from .database import Database
except ImportError:
    from database import Database


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
    void_requested = pyqtSignal(int, bool)  # 作废请求 (event_id, is_void)
    
    # 内部信号
    _new_events_fetched = pyqtSignal(list)
    _all_events_fetched = pyqtSignal(list)
    _events_modified_fetched = pyqtSignal(list)
    _row_update_requested = pyqtSignal(int, dict)  # row, updated_event_data

    def __init__(self, database: Database, parent=None):
        super().__init__(parent)
        self.database = database
        self._last_event_id = 0
        self._last_sync_time = datetime.now().isoformat()
        self._is_refreshing = False # 刷新状态锁
        self._db_missing_warned = False

        self._init_ui()
        self._connect_signals()
        
        # 初始加载已有数据
        QTimer.singleShot(0, self.refresh_list)
        
        # 启动定时增量刷新
        self._setup_timer()

    @pyqtSlot(object)
    def set_database(self, database: Database):
        self.database = database
        self._db_missing_warned = False
        if self.database:
            QTimer.singleShot(0, self.refresh_list)

    def _connect_signals(self):
        """连接内部信号"""
        self._new_events_fetched.connect(self._on_new_events_fetched)
        self._all_events_fetched.connect(self._on_all_events_fetched)
        self._events_modified_fetched.connect(self._on_events_modified_fetched)
        self._row_update_requested.connect(self._on_row_update_requested)

    def _init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # 样式表
        self.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
                font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
            }
            QTableWidget {
                border: 1px solid #d9d9d9;
                gridline-color: #f0f0f0;
                background-color: #ffffff;
                selection-background-color: #1890ff;
                selection-color: #ffffff;
                font-size: 16px;
                outline: none;
                color: #000000;
            }
            QTableWidget::item {
                padding: 10px;
            }
            QTableWidget::item:selected {
                background-color: #1890ff;
                color: #ffffff;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                color: #000000;
                padding: 12px;
                border: none;
                border-right: 1px solid #d9d9d9;
                border-bottom: 1px solid #d9d9d9;
                font-weight: bold;
                font-size: 16px;
            }
            QComboBox {
                border: 1px solid #d9d9d9;
                border-radius: 4px;
                padding: 5px 10px;
                min-width: 120px;
                background: white;
                font-size: 16px;
                color: #000000;
            }
            QComboBox:hover {
                border-color: #1890ff;
            }
            QPushButton {
                background-color: #1890ff;
                color: white;
                border-radius: 4px;
                padding: 5px 10px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #40a9ff;
            }
            QPushButton#action_btn {
                background-color: #f0f0f0;
                color: #000000;
                border: 1px solid #d9d9d9;
                border-radius: 4px;
                padding: 6px 15px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton#action_btn:hover {
                background-color: #ffffff;
                border-color: #1890ff;
                color: #1890ff;
            }
        """)

        # 顶部工具栏
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 5)

        # 筛选下拉框
        self.filter_combo = QComboBox()
        self.filter_combo.setPlaceholderText("所有分组")
        self._update_filter_categories()
        self.filter_combo.currentTextChanged.connect(self._on_filter_changed)
        self.filter_combo.setFixedWidth(150) # 固定宽度更整齐
        toolbar.addWidget(self.filter_combo)

        toolbar.addStretch()

        self.stats_label = QLabel("共 0 条记录")
        self.stats_label.setStyleSheet("color: #000000; font-size: 16px; font-weight: bold; margin-right: 10px;")
        toolbar.addWidget(self.stats_label)

        layout.addLayout(toolbar)

        # 事件表格
        self.table = QTableWidget()
        self.table.setColumnCount(12) # 增加一列：组名次 + 来源
        self.table.setHorizontalHeaderLabels([
            "序号", "号码", "姓名", "分组", "组名次", "状态", "过线时间", "芯片时间", "来源", "备注", "ID", "操作"
        ])

        # 表格设置
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.setShowGrid(False) # 隐藏网格线，更现代
        self.table.setAlternatingRowColors(True) # 启用交替行颜色，提高可读性
        self.table.setStyleSheet(self.table.styleSheet() + "QTableWidget { alternate-background-color: #f5f5f5; }")
        
        # 初始按过线时间倒序排列（最新的在最上面，符合直播/监控逻辑）
        # 注意：现在过线时间是第 6 列 (从0开始)
        self.table.horizontalHeader().setSortIndicator(6, Qt.DescendingOrder)
        self.table.sortByColumn(6, Qt.DescendingOrder)
        
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)
        self.table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        self.table.cellClicked.connect(self._on_cell_clicked)
        
        # 增加行高，视觉更清爽，适应更大的字体
        self.table.verticalHeader().setDefaultSectionSize(50)
        self.table.verticalHeader().setVisible(False)

        # 列宽与显示隐藏设置
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        
        # 重点突出：号码和名次列加宽
        self.table.setColumnWidth(0, 70)   # 序号
        self.table.setColumnWidth(1, 100)  # 号码
        self.table.setColumnWidth(4, 90)   # 组名次
        self.table.setColumnWidth(6, 180)  # 过线时间
        self.table.setColumnWidth(5, 140)  # 状态
        self.table.setColumnWidth(10, 120) # 操作列

        layout.addWidget(self.table)

    def _setup_timer(self):
        """设置定时刷新"""
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self._check_new_events)
        self.refresh_timer.start(1500)
        self._is_refreshing = False # 刷新状态锁

    @pyqtSlot()
    def _check_new_events(self):
        """检查新事件和更新的事件 (异步局部更新版)"""
        # 如果正在刷新，不要重叠
        if self._is_refreshing:
            return
        if not self.database:
            return
            
        self._is_refreshing = True # 加锁
        
        def fetch_task():
            try:
                if not self.database:
                    self._is_refreshing = False
                    return

                # 1. 检查新事件
                latest_id = self.database.get_latest_event_id()
                
                # 如果数据库 ID 小于当前记录的 ID，说明数据库已被重置
                if latest_id < self._last_event_id:
                    print(f"[EventList] 检测到数据库重置 (latest={latest_id}, last={self._last_event_id})")
                    self._last_event_id = 0
                    QMetaObject.invokeMethod(self, "refresh_list", Qt.QueuedConnection)
                    return # refresh_list 会重新加载所有并释放锁
                
                has_updates = False
                
                if latest_id > self._last_event_id:
                    new_events = self.database.get_events_since(self._last_event_id)
                    if new_events:
                        self._new_events_fetched.emit(new_events)
                        has_updates = True
                
                # 2. 检查最近修改过的事件 (例如 AI 修正后的记录)
                modified_events = self.database.get_modified_events_since(self._last_sync_time)
                if modified_events:
                    # 过滤掉已经是新事件的记录
                    new_ids = {e['event_id'] for e in (new_events if 'new_events' in locals() else [])}
                    really_modified = [e for e in modified_events if e['event_id'] not in new_ids]
                    if really_modified:
                        self._events_modified_fetched.emit(really_modified)
                        has_updates = True
                
                # 更新同步时间
                self._last_sync_time = datetime.now().isoformat()
                
                # 如果没有任何更新，手动释放锁 (如果有更新，信号处理函数会释放锁)
                if not has_updates:
                    self._is_refreshing = False
                    
            except Exception as e:
                print(f"[EventList] 检查更新错误: {e}")
                self._is_refreshing = False # 发生异常必须释放锁

        threading.Thread(target=fetch_task, daemon=True).start()

    def _update_filter_categories(self):
        """更新筛选下拉框的类别列表"""
        current = self.filter_combo.currentText()
        self.filter_combo.blockSignals(True)
        self.filter_combo.clear()
        
        # 基础状态筛选
        base_items = ["全部", "已识别", "需复核", "未识别", "已作废"]
        self.filter_combo.addItems(base_items)
        
        # 从数据库获取的分组
        try:
            categories = self.database.get_all_categories()
            if categories:
                self.filter_combo.insertSeparator(len(base_items))
                self.filter_combo.addItems(categories)
        except Exception:
            pass
            
        # 恢复之前的选中项
        index = self.filter_combo.findText(current)
        if index >= 0:
            self.filter_combo.setCurrentIndex(index)
        self.filter_combo.blockSignals(False)

    def _on_new_events_fetched(self, new_events):
        """主线程：处理新获取的事件"""
        if not new_events:
            self._is_refreshing = False
            return
            
        try:
            print(f"[EventList] 收到 {len(new_events)} 条新事件")
            
            # 暂时关闭排序，批量插入
            sorting_enabled = self.table.isSortingEnabled()
            self.table.setSortingEnabled(False)
            
            for event in new_events:
                self.add_event(event)
                
            # 更新最后获取的 ID
            current_max_new = max(e['event_id'] for e in new_events)
            self._last_event_id = max(self._last_event_id, current_max_new)
            
            # 恢复排序
            self.table.setSortingEnabled(sorting_enabled)
            
            # 只有当列表可见且数据量较小时，才立即重新计算排名，否则延迟计算
            if self.table.rowCount() < 200:
                self._recalculate_ranks()
                self._refresh_duplicate_highlights()
            
            # 异步更新类别（每隔一段时间更新一次，不要每次新事件都更新）
            if not hasattr(self, '_last_cat_update') or (datetime.now() - self._last_cat_update).total_seconds() > 10:
                self._update_filter_categories()
                self._last_cat_update = datetime.now()
                
        except Exception as e:
            print(f"[EventList] 处理新事件出错: {e}")
        finally:
            self._is_refreshing = False # 确保锁释放

    def _on_events_modified_fetched(self, modified_events):
        """主线程：处理修改过的事件"""
        try:
            if not modified_events:
                return
                
            print(f"[EventList] 收到 {len(modified_events)} 条修改事件")
            for event in modified_events:
                event_id = event.get('event_id')
                # 寻找该事件所在的行
                found = False
                for row in range(self.table.rowCount()):
                    item = self.table.item(row, 10) # ID 列
                    if not item:
                        continue
                    row_event_id = item.data(Qt.DisplayRole)
                    try:
                        same_event = int(row_event_id) == int(event_id)
                    except Exception:
                        same_event = str(row_event_id) == str(event_id)
                    if same_event:
                        # 更新这一行的数据
                        self._set_row_data(row, event)
                        found = True
                        break
                
                # 如果没找到（可能是新数据库还没加载），则强制全量刷新一次
                if not found:
                    print(f"[EventList] 事件 {event_id} 在列表中未找到，可能需要全量刷新")
                    # 先释放刷新锁，再异步触发全量刷新；避免锁内直接 refresh 导致失效
                    self._is_refreshing = False
                    QTimer.singleShot(0, self.refresh_list)
                    return
            
            # 更新完后刷新全局冲突和名次
            self._refresh_duplicate_highlights()
            self._recalculate_ranks()
        finally:
            self._is_refreshing = False # 确保锁释放

    def _recalculate_ranks(self):
        """重新计算当前视图中的序号和组内名次"""
        row_count = self.table.rowCount()
        if row_count == 0:
            return

        # 1. 序号 (顺序码)
        # 获取当前排序状态
        header = self.table.horizontalHeader()
        sort_col = header.sortIndicatorSection()
        sort_order = header.sortIndicatorOrder()
        
        # 如果是按过线时间或ID排序
        if sort_col in [6, 10]:
            if sort_order == Qt.DescendingOrder:
                # 最新在顶，序号应该是 N, N-1, ... 1
                for row in range(row_count):
                    item = self.table.item(row, 0)
                    if item: item.setData(Qt.DisplayRole, row_count - row)
            else:
                # 最早在顶，序号是 1, 2, ... N
                for row in range(row_count):
                    item = self.table.item(row, 0)
                    if item: item.setData(Qt.DisplayRole, row + 1)
        else:
            # 其他排序方式下，序号仅作为行号参考
            for row in range(row_count):
                item = self.table.item(row, 0)
                if item: item.setData(Qt.DisplayRole, row + 1)

        # 2. 组内名次
        # 收集所有非作废记录
        group_data = {} # {category: [(row, time)]}
        for row in range(row_count):
            event = self._get_event_at_row(row)
            if not event or event.get('is_void'):
                # 作废记录不计名次
                rank_item = self.table.item(row, 4)
                if rank_item: rank_item.setText("-")
                continue
                
            cat = event.get('athlete_category') or "未分组"
            if cat not in group_data:
                group_data[cat] = []
            
            # 使用 cross_time (float) 进行精确排序
            group_data[cat].append({
                'row': row,
                'time': event.get('cross_time', 0)
            })
            
        # 分组计算名次
        for cat, items in group_data.items():
            # 按过线时间升序排列（最早的第1名）
            items.sort(key=lambda x: x['time'])
            for i, item in enumerate(items):
                rank_item = self.table.item(item['row'], 4)
                if rank_item:
                    rank_item.setData(Qt.DisplayRole, i + 1)

    def reset_ui(self):
        """强制重置 UI 和所有同步计数器 (同步版，用于彻底解决清空后不刷新的问题)"""
        # 1. 停止当前可能的刷新
        self._is_refreshing = False
        
        # 2. 彻底重置所有追踪 ID
        self._last_event_id = 0
        self._last_sync_time = datetime.now().isoformat()
        
        # 3. 清空表格内容
        self.table.setRowCount(0)
        
        # 4. 更新统计和类别
        self.stats_label.setText("共 0 条记录")
        self._update_filter_categories()
        
        print("[EventList] UI 和计数器已强制重置为 0")

    @pyqtSlot(int)
    def add_event_by_id(self, event_id: int):
        """增量添加一个新事件（主线程）"""
        # 如果该 ID 已经存在于列表中，则跳过
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 10) # ID 列
            if item and item.data(Qt.DisplayRole) == event_id:
                return

        def fetch_task():
            try:
                if not self.database:
                    if not self._db_missing_warned:
                        print("[EventList] 获取单个事件失败: 数据库未初始化")
                        self._db_missing_warned = True
                    return
                event = self.database.get_event(event_id)
                if event:
                    self._new_events_fetched.emit([event])
            except Exception as e:
                print(f"[EventList] 获取单个事件失败: {e}")

        threading.Thread(target=fetch_task, daemon=True).start()

    @pyqtSlot()
    def set_source_column_visible(self, visible: bool):
        """设置来源列是否可见"""
        if visible:
            self.table.showColumn(8)
        else:
            self.table.hideColumn(8)

    @pyqtSlot()
    def refresh_data(self):
        """刷新数据的别名，与 main_window 兼容"""
        self.refresh_list()

    def refresh_list(self):
        """全量刷新列表（主线程）"""
        # 如果正在刷新，不要重叠，但如果是手动触发，我们可能需要强制重置
        if self._is_refreshing:
            return
        if not self.database:
            if not self._db_missing_warned:
                print("[EventList] 刷新失败: 数据库未初始化")
                self._db_missing_warned = True
            return
            
        filter_text = self.filter_combo.currentText()
        include_void = (filter_text == "已作废" or filter_text == "全部")
        
        def fetch_task():
            try:
                events = self.database.get_all_events(include_void=include_void)
                
                # 状态筛选
                if filter_text == "已识别":
                    events = [e for e in events if e['bib_status'] == 'recognized']
                elif filter_text == "需复核":
                    events = [e for e in events if e['bib_status'] == 'needs_review']
                elif filter_text == "未识别":
                    events = [e for e in events if e['bib_status'] == 'unrecognized']
                elif filter_text == "已作废":
                    events = [e for e in events if e['is_void']]
                elif filter_text not in ["全部", ""]:
                    events = [e for e in events if e.get('athlete_category') == filter_text]
                
                # 更新最后已知的 ID 和同步时间
                if events:
                    max_id = max(e['event_id'] for e in events)
                    self._last_event_id = max_id
                else:
                    self._last_event_id = 0
                
                self._last_sync_time = datetime.now().isoformat()
                self._all_events_fetched.emit(events)
            except Exception as e:
                print(f"[EventList] 刷新失败: {e}")
            finally:
                self._is_refreshing = False

        self._is_refreshing = True
        # 清空表格
        self.table.setRowCount(0)
        # 异步加载
        threading.Thread(target=fetch_task, daemon=True).start()

    def _on_all_events_fetched(self, events):
        """主线程：全量更新列表"""
        print(f"[EventList] 全量刷新，收到 {len(events)} 条记录")
        # 暂时关闭排序，避免插入数据时触发重排
        sorting_enabled = self.table.isSortingEnabled()
        self.table.setSortingEnabled(False)
        
        # 统计号码频率（用于高亮冲突）
        bib_counts = {}
        for e in events:
            bib = e.get('bib_number')
            if bib and not e.get('is_void'):
                bib_counts[bib] = bib_counts.get(bib, 0) + 1
        
        self.table.setRowCount(len(events))
        for row, event in enumerate(events):
            bib = event.get('bib_number')
            is_duplicate = bib and not event.get('is_void') and bib_counts.get(bib, 0) > 1
            self._set_row_data(row, event, is_duplicate)
            
            # 在全量加载时，row + 1 即为该组内的名次
            rank_item = self.table.item(row, 0)
            if rank_item:
                rank_item.setData(Qt.DisplayRole, row + 1)
            
        # 恢复排序设置
        self.table.setSortingEnabled(sorting_enabled)
        
        if events:
            self._last_event_id = max(e['event_id'] for e in events)
        else:
            self._last_event_id = 0
            # 如果列表为空，重置同步时间，确保能接收到任何新改动
            self._last_sync_time = datetime.now().isoformat()

    def _set_row_data(self, row: int, event: Dict[str, Any], is_duplicate: bool = False):
        """设置行数据 (12列适配版)"""
        # 暂时关闭排序，避免插入数据时触发重排
        sorting_enabled = self.table.isSortingEnabled()
        self.table.setSortingEnabled(False)

        # 显式设置基础字体
        base_font = QFont()
        base_font.setPointSize(14)

        # 0. 序号 (临时占位，由 _recalculate_ranks 统一计算)
        seq_item = QTableWidgetItem()
        seq_item.setTextAlignment(Qt.AlignCenter)
        seq_item.setFont(base_font)
        self.table.setItem(row, 0, seq_item)

        # 1. 号码
        bib = event.get('bib_number') or "未识别"
        bib_item = QTableWidgetItem()
        if bib.isdigit():
            bib_item.setData(Qt.DisplayRole, int(bib))
        else:
            bib_item.setText(bib)
            
        bib_item.setTextAlignment(Qt.AlignCenter)
        
        # 显式设置号码字体
        bib_font = QFont()
        bib_font.setPointSize(15) # 号码字号略大
        bib_font.setBold(True)     # 号码一律加粗
        
        if event.get('is_void'):
            bib_item.setForeground(QBrush(QColor("#8c8c8c"))) # 灰色
        elif is_duplicate:
            # 冲突号码显示红色
            bib_item.setForeground(QBrush(QColor("#f5222d")))
        elif event.get('bib_status') == 'recognized':
            bib_item.setForeground(QBrush(QColor("#52c41a"))) # 绿色
        elif event.get('bib_status') == 'needs_review':
            bib_item.setForeground(QBrush(QColor("#faad14"))) # 橙黄色
        elif bib == "未识别":
            bib_item.setForeground(QBrush(QColor("#ff4d4f"))) # 红色
            
        bib_item.setFont(bib_font)
        self.table.setItem(row, 1, bib_item)

        # 2. 姓名
        name = event.get('athlete_name') or "-"
        name_item = QTableWidgetItem(name)
        name_item.setTextAlignment(Qt.AlignCenter)
        name_item.setFont(base_font)
        self.table.setItem(row, 2, name_item)

        # 3. 分组
        category = event.get('athlete_category') or "-"
        cat_item = QTableWidgetItem(category)
        cat_item.setTextAlignment(Qt.AlignCenter)
        cat_item.setFont(base_font)
        self.table.setItem(row, 3, cat_item)

        # 4. 组名次 (临时占位，由 _recalculate_ranks 统一计算)
        g_rank_item = QTableWidgetItem()
        g_rank_item.setTextAlignment(Qt.AlignCenter)
        g_rank_item.setFont(base_font)
        self.table.setItem(row, 4, g_rank_item)

        # 5. 状态
        bg_color = None # 默认无背景色，使用交替行颜色

        if event.get('is_void'):
            status_text = "已作废"
            status_color = QColor("#8c8c8c") # 灰色
            bg_color = QColor("#f5f5f5") # 作废行灰色背景
        elif is_duplicate:
            status_text = "号码冲突"
            status_color = QColor("#f5222d") # 红色
            bg_color = QColor("#fff1f0") # 冲突行淡红背景
        elif event.get('is_ai_correction'):
            status_text = "AI已修正"
            status_color = QColor("#006644") # 深绿色
            bg_color = QColor("#e6f7ff") # AI修正行淡蓝背景 (或者淡绿色 #f6ffed)
        elif event.get('bib_status') == 'recognized':
            status_text = "已识别"
            status_color = QColor("#52c41a") # 绿色
        elif event.get('bib_status') == 'needs_review':
            status_text = "待复核"
            status_color = QColor("#faad14") # 橙黄色
            bg_color = QColor("#fffbe6") # 复核行淡黄背景
        else:
            status_text = "未识别"
            status_color = QColor("#ff4d4f") # 浅红
            bg_color = QColor("#fff1f0") # 未识别行淡红背景

        status_item = QTableWidgetItem(status_text)
        status_item.setTextAlignment(Qt.AlignCenter)
        status_item.setForeground(QBrush(status_color))
        # 加粗文字
        status_font = QFont()
        status_font.setPointSize(14)
        status_font.setBold(True)
        status_item.setFont(status_font)
        self.table.setItem(row, 5, status_item)

        # 6. 时间 (视频过线时间)
        time_item = QTableWidgetItem(event.get('cross_time_str', ''))
        time_item.setTextAlignment(Qt.AlignCenter)
        time_item.setFont(base_font)
        self.table.setItem(row, 6, time_item)

        # 7. 芯片时间 (个人芯片成绩)
        chip_time = event.get('chip_time') or "-"
        chip_item = QTableWidgetItem(chip_time)
        chip_item.setTextAlignment(Qt.AlignCenter)
        chip_item.setFont(base_font)
        # 如果有芯片成绩，用蓝色区分
        if chip_time != "-":
            chip_item.setForeground(QBrush(QColor("#1890ff"))) # Ant Design Blue
        self.table.setItem(row, 7, chip_item)

        # 8. 来源 (机位 ID)
        source_id = event.get('source_id', 0)
        source_text = f"机位 {source_id + 1}"
        source_item = QTableWidgetItem(source_text)
        source_item.setTextAlignment(Qt.AlignCenter)
        source_item.setFont(base_font)
        source_item.setForeground(QBrush(QColor("#722ed1"))) # 紫色区分来源
        self.table.setItem(row, 8, source_item)

        # 9. 备注
        notes = event.get('notes') or ""
        notes_item = QTableWidgetItem(notes)
        notes_item.setFont(base_font)
        # 如果备注中包含 DEDUP 或 MERGE，使用特殊颜色突出显示
        if "DEDUP" in notes or "MERGE" in notes:
            notes_item.setForeground(QBrush(QColor("#722ed1"))) # 紫色
            notes_item.setToolTip("系统已自动处理重复记录")
        self.table.setItem(row, 9, notes_item)

        # 10. ID (用于引用)
        id_val = event.get('event_id')
        id_item = QTableWidgetItem()
        if id_val is not None:
            id_item.setData(Qt.DisplayRole, int(id_val))
        id_item.setData(Qt.UserRole, event)  # 存储完整数据
        id_item.setTextAlignment(Qt.AlignCenter)
        id_item.setFont(base_font)
        self.table.setItem(row, 10, id_item)

        # 11. 操作 (查看截图文字链接)
        view_label = QLabel("<u>查看截图</u>")
        view_label.setStyleSheet("color: #1890ff; font-size: 16px;")
        view_label.setCursor(Qt.PointingHandCursor)
        view_label.setAlignment(Qt.AlignCenter)
        view_label.mousePressEvent = lambda e: self._view_event(event)
        self.table.setCellWidget(row, 11, view_label)

        # 设置整行背景色
        if bg_color:
            self._set_row_background(row, bg_color)
        else:
            # 清除可能存在的旧背景色，确保交替行颜色生效
            self._clear_row_background(row)

        # 恢复排序设置
        self.table.setSortingEnabled(sorting_enabled)

    def _clear_row_background(self, row: int):
        """清除整行背景色"""
        for col in range(self.table.columnCount()):
            item = self.table.item(row, col)
            if item:
                item.setBackground(QBrush(Qt.NoBrush))

    def _set_row_background(self, row: int, color: QColor):
        """设置整行背景色"""
        for col in range(self.table.columnCount()):
            item = self.table.item(row, col)
            if item:
                item.setBackground(QBrush(color))

    def _on_filter_changed(self):
        """筛选条件改变"""
        self.refresh_list()

    def _on_cell_double_clicked(self, row, column):
        """双击单元格"""
        event = self._get_event_at_row(row)
        if event:
            self._view_event(event)

    def _on_cell_clicked(self, row, column):
        """点击单元格（选中行同步）"""
        event = self._get_event_at_row(row)
        if event:
            self.event_selected.emit(event)

    def _get_event_at_row(self, row: int) -> Optional[dict]:
        """获取指定行的事件"""
        id_item = self.table.item(row, 10) # ID在第10列
        if id_item:
            return id_item.data(Qt.UserRole)
        return None

    def _view_event(self, event: dict):
        """查看事件"""
        self.event_selected.emit(event)
        # 优先查看完整截图（可以看到目标运动员标记）
        screenshot = event.get('screenshot_full') or event.get('screenshot_crop') or ""
        self.view_screenshot.emit(screenshot, event)

    def get_all_events(self) -> List[dict]:
        """获取当前列表中的所有事件（按表格当前顺序）"""
        events = []
        for row in range(self.table.rowCount()):
            event = self._get_event_at_row(row)
            if event:
                events.append(event)
        return events

    def select_event_by_id(self, event_id: int):
        """联动功能：根据 ID 选中列表中的行并滚动到视图中心"""
        # 取消所有之前的选中
        self.table.clearSelection()
        
        for row in range(self.table.rowCount()):
            event = self._get_event_at_row(row)
            if event and event.get('event_id') == event_id:
                # 选中该行
                self.table.selectRow(row)
                # 确保该行可见
                self.table.scrollToItem(self.table.item(row, 0), QAbstractItemView.PositionAtCenter)
                break

    def _refresh_duplicate_highlights(self):
        """重新扫描并更新所有行的冲突高亮状态"""
        row_count = self.table.rowCount()
        bib_counts = {}
        
        # 第一遍：统计
        for row in range(row_count):
            event = self._get_event_at_row(row)
            if event and not event.get('is_void'):
                bib = event.get('bib_number')
                if bib:
                    bib_counts[bib] = bib_counts.get(bib, 0) + 1
        
        # 第二遍：更新 UI
        for row in range(row_count):
            event = self._get_event_at_row(row)
            if event:
                bib = event.get('bib_number')
                is_duplicate = bib and not event.get('is_void') and bib_counts.get(bib, 0) > 1
                
                # 更新状态列 (第5列)
                status_item = self.table.item(row, 5)
                if status_item:
                    if event.get('is_void'):
                        status_text = "已作废"
                        status_color = QColor("#666666")
                    elif is_duplicate:
                        status_text = "逻辑冲突"
                        status_color = QColor("#DE350B")
                    elif event.get('bib_status') == 'recognized':
                        status_text = "已识别"
                        status_color = QColor("#006644")
                    elif event.get('bib_status') == 'needs_review':
                        status_text = "需复核"
                        status_color = QColor("#826A00")
                    else:
                        status_text = "未识别"
                        status_color = QColor("#BF2600")
                    
                    status_item.setText(status_text)
                    status_item.setForeground(QBrush(status_color))

                # 更新号码列颜色 (第1列)
                bib_item = self.table.item(row, 1)
                if bib_item:
                    if is_duplicate:
                        bib_item.setForeground(QBrush(QColor("#DE350B")))
                        font = bib_item.font()
                        font.setBold(True)
                        bib_item.setFont(font)
                    else:
                        # 恢复默认颜色
                        if event.get('is_void'):
                            bib_item.setForeground(QBrush(Qt.gray))
                        elif bib == "未识别":
                            bib_item.setForeground(QBrush(Qt.red))
                        else:
                            bib_item.setForeground(QBrush(Qt.black))
                        font = bib_item.font()
                        font.setBold(False)
                        bib_item.setFont(font)

                # 更新行背景
                bg_color = QColor("#FFFFFF")
                if event.get('is_void'):
                    bg_color = QColor("#F4F5F7")
                elif is_duplicate:
                    bg_color = QColor("#FFEBE6")
                elif event.get('bib_status') == 'needs_review':
                    bg_color = QColor("#FFF9E6")
                
                self._set_row_background(row, bg_color)

    def _on_row_update_requested(self, row: int, event_data: dict):
        """主线程：更新特定行的数据"""
        # 找到该 event_id 所在的当前行，因为排序可能已经改变了行号
        target_id = event_data.get('event_id')
        current_row = -1
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 10) # ID 列 (第10列)
            if item and item.data(Qt.DisplayRole) == target_id:
                current_row = r
                break
        
        if current_row >= 0:
            self._set_row_data(current_row, event_data)
            # 更新完某一行后，由于号码可能变化，需要刷新全局冲突状态
            self._refresh_duplicate_highlights()
            # 重新计算名次（因为号码变化可能影响分组名次）
            self._recalculate_ranks()

    def _edit_bib_number(self, row: int):
        """编辑号码（局部更新版）"""
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
            # 冲突检查逻辑
            if new_bib and self.database:
                existing_event = self.database.get_event_by_bib(new_bib, exclude_event_id=event['event_id'])
                if existing_event:
                    # 发现逻辑冲突
                    old_rank = existing_event.get('rank', '?')
                    old_time = existing_event.get('cross_time_str', '?')
                    
                    msg = QMessageBox(self)
                    msg.setIcon(QMessageBox.Warning)
                    msg.setWindowTitle("号码逻辑冲突")
                    msg.setText(f"号码 <font color='red'><b>{new_bib}</b></font> 已存在于第 <b>{old_rank}</b> 名 (时间: {old_time})。")
                    msg.setInformativeText("证明肯定有一个是错的，请选择处理方案：")
                    
                    void_old = msg.addButton("作废旧成绩 (推荐)", QMessageBox.ActionRole)
                    keep_both = msg.addButton("保留两者 (多圈赛)", QMessageBox.ActionRole)
                    cancel = msg.addButton("取消修改", QMessageBox.RejectRole)
                    
                    msg.setDefaultButton(void_old)
                    msg.exec_()
                    
                    if msg.clickedButton() == cancel:
                        return
                    elif msg.clickedButton() == void_old:
                        # 作废旧成绩
                        old_id = existing_event.get('event_id')
                        self.void_requested.emit(old_id, True)

            # 1. 立即更新 UI
            event['bib_number'] = new_bib if new_bib else None
            event['bib_status'] = 'recognized' if new_bib else 'unrecognized'
            self._set_row_data(row, event)
            
            # 2. 后台异步更新数据库并获取完整信息
            def update_db():
                try:
                    self.database.update_event_bib(event['event_id'], new_bib)
                    # 触发号码更新信号
                    self.bib_updated.emit(event['event_id'], new_bib)
                    # 获取更新后的完整数据（包括姓名、芯片成绩等）
                    updated_event = self.database.get_event(event['event_id'])
                    if updated_event:
                        self._row_update_requested.emit(row, updated_event)
                except Exception as e:
                    print(f"Error updating bib: {e}")
            
            threading.Thread(target=update_db, daemon=True).start()

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
                lambda: self._toggle_void(row, event, False))
        else:
            void_action = menu.addAction("标记作废")
            void_action.triggered.connect(
                lambda: self._toggle_void(row, event, True))

        menu.exec_(self.table.mapToGlobal(pos))

    def _toggle_void(self, row: int, event: dict, is_void: bool):
        """切换作废状态（局部更新版）"""
        event_id = event['event_id']
        
        # 1. 立即更新 UI (UI 先行)
        event['is_void'] = is_void
        self._set_row_data(row, event)
        
        # 2. 后台异步更新数据库
        def update_db():
            try:
                if is_void:
                    self.database.void_event(event_id)
                else:
                    self.database.unvoid_event(event_id)
                
                # 更新后获取完整数据
                updated_event = self.database.get_event(event_id)
                if updated_event:
                    self._row_update_requested.emit(row, updated_event)
            except Exception as e:
                print(f"Error toggling void: {e}")
        
        threading.Thread(target=update_db, daemon=True).start()

    def add_event(self, event: Dict[str, Any]):
        """添加新事件"""
        # 1. 检查筛选条件
        filter_text = self.filter_combo.currentText()
        if filter_text not in ["全部", ""]:
            # 状态检查
            if filter_text == "已识别" and event.get('bib_status') != 'recognized': return
            if filter_text == "需复核" and event.get('bib_status') != 'needs_review': return
            if filter_text == "未识别" and event.get('bib_status') != 'unrecognized': return
            if filter_text == "已作废" and not event.get('is_void'): return
            # 分组检查 (如果 filter_text 是一个分组名)
            if filter_text not in ["已识别", "需复核", "未识别", "已作废"]:
                if event.get('athlete_category') != filter_text:
                    return

        # 2. 插入行
        # 如果是 ASC 排序（按过线顺序），插在末尾
        # 如果是 DESC 排序（最新在顶），插在开头
        indicator_order = self.table.horizontalHeader().sortIndicatorOrder()
        indicator_column = self.table.horizontalHeader().sortIndicatorSection()
        
        if indicator_column == 8 and indicator_order == Qt.AscendingOrder:
            row = self.table.rowCount()
        else:
            row = 0
            
        self.table.insertRow(row)
        self._set_row_data(row, event)
        
        # 3. 重新计算所有名次并更新冲突高亮
        self._recalculate_ranks()
        self._refresh_duplicate_highlights()
        
        # 4. 自动滚动
        if row == 0:
            self.table.scrollToTop()
        else:
            self.table.scrollToBottom()
            
        count = self.table.rowCount()
        self.stats_label.setText(f"共 {count} 条记录")
        self._last_event_id = max(self._last_event_id, event['event_id'])
