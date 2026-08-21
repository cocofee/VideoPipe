"""Judge-facing review list for CycleRace passages and video locations."""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

try:
    from .passage_receiver import PassageEvent, PassageEventStore
    from .video_timeline import (
        PassageVideoLocation,
        PassageVideoLookup,
        VideoTimelineStore,
    )
except ImportError:
    from passage_receiver import PassageEvent, PassageEventStore
    from video_timeline import PassageVideoLocation, PassageVideoLookup, VideoTimelineStore


_STATUS_TEXT = {
    "no_segments": "没有录像时间线",
    "before_recording": "早于录像",
    "after_recording": "晚于录像",
    "recording_gap": "录像分段间隙",
    "recording": "对应机位仍在录像",
    "missing_file": "录像文件缺失",
    "unverified": "录像可打开，但时间范围未验证",
    "outside_media": "Passage 超出录像真实媒体范围",
}

_OPENABLE_STATUSES = {"located", "unverified"}


def format_passage_time(timestamp_ms: int) -> str:
    try:
        value = datetime.fromtimestamp(int(timestamp_ms) / 1000.0).astimezone()
    except (OSError, OverflowError, ValueError):
        return f"{int(timestamp_ms)} ms"
    return value.strftime("%H:%M:%S.%f")[:-3]


def lookup_status_text(lookup: PassageVideoLookup) -> str:
    located = [item for item in lookup.locations if item.status == "located"]
    if located:
        uncertainty = max(item.timing_error_ms for item in located)
        unverified_count = sum(
            item.status == "unverified" for item in lookup.locations
        )
        suffix = (
            f"，另有 {unverified_count} 个机位时间范围未验证"
            if unverified_count
            else ""
        )
        return f"{len(located)} 个机位，近似 ±{uncertainty} ms{suffix}"
    unverified = [item for item in lookup.locations if item.status == "unverified"]
    if unverified:
        return f"{len(unverified)} 个机位可打开，时间范围未验证"
    return _STATUS_TEXT.get(lookup.status, lookup.status)


class PassageReviewDialog(QDialog):
    clock_offset_changed = pyqtSignal(int)

    def __init__(
        self,
        passage_store: PassageEventStore,
        timeline_store: VideoTimelineStore,
        parent=None,
        *,
        clock_offset_ms: int = 0,
        pre_roll_ms: int = 3_000,
        open_location: Optional[
            Callable[[PassageEvent, PassageVideoLocation], None]
        ] = None,
    ):
        super().__init__(parent)
        self.passage_store = passage_store
        self.timeline_store = timeline_store
        self.clock_offset_ms = int(clock_offset_ms)
        self.pre_roll_ms = max(0, int(pre_roll_ms))
        self._open_location = open_location

        self.setWindowTitle("CycleRace 通过记录")
        self.resize(1060, 620)
        self.setMinimumSize(820, 480)
        self._init_ui()
        self.refresh()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        controls.addWidget(QLabel("CycleRace → VideoPipe 时钟偏移"))
        self.offset_spin = QSpinBox(self)
        self.offset_spin.setRange(-600_000, 600_000)
        self.offset_spin.setSingleStep(100)
        self.offset_spin.setSuffix(" ms")
        self.offset_spin.setMinimumWidth(130)
        self.offset_spin.setValue(self.clock_offset_ms)
        self.offset_spin.setToolTip("VideoPipe 时间 = CycleRace passage 时间 + 此偏移")
        self.offset_spin.valueChanged.connect(self._on_offset_changed)
        controls.addWidget(self.offset_spin)
        controls.addWidget(QLabel(f"打开位置提前 {self.pre_roll_ms / 1000.0:.1f} 秒"))
        controls.addStretch()
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh)
        controls.addWidget(refresh_btn)
        layout.addLayout(controls)

        self.table = QTableWidget(0, 7, self)
        self.table.setHorizontalHeaderLabels(
            ["序号", "号码 / 芯片", "组别", "圈次", "Passage 时间", "录像定位", "操作"]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.Stretch)
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)
        self.table.cellDoubleClicked.connect(self._open_row)
        layout.addWidget(self.table, 1)

        self.summary_label = QLabel()
        self.summary_label.setStyleSheet("color: #667085; font-size: 12px;")
        layout.addWidget(self.summary_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Close, parent=self)
        buttons.button(QDialogButtonBox.Close).setText("关闭")
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_offset_changed(self, value: int) -> None:
        self.clock_offset_ms = int(value)
        self.clock_offset_changed.emit(self.clock_offset_ms)
        self.refresh()

    def _lookup(self, event: PassageEvent) -> PassageVideoLookup:
        return self.timeline_store.locate_passage(
            event.passage_time_ms,
            clock_offset_ms=self.clock_offset_ms,
            pre_roll_ms=self.pre_roll_ms,
        )

    def refresh(self) -> None:
        events = self.passage_store.events()
        self.table.setRowCount(len(events))
        located_count = 0
        for row, event in enumerate(events):
            lookup = self._lookup(event)
            available = [
                item for item in lookup.locations if item.status in _OPENABLE_STATUSES
            ]
            if available:
                located_count += 1
            identity = event.bib.strip() or event.chip_id.strip() or "未知"
            values = (
                str(event.sequence),
                identity,
                event.group_id,
                str(event.lap),
                format_passage_time(event.passage_time_ms),
                lookup_status_text(lookup),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column in {0, 3}:
                    item.setTextAlignment(Qt.AlignCenter)
                if column == 0:
                    item.setData(Qt.UserRole, event.event_id)
                self.table.setItem(row, column, item)

            open_btn = QPushButton("打开证据")
            open_btn.setEnabled(bool(available))
            open_btn.clicked.connect(
                lambda _checked=False, event_id=event.event_id: self._open_event(event_id)
            )
            self.table.setCellWidget(row, 6, open_btn)

        self.summary_label.setText(
            f"共 {len(events)} 条 passage，{located_count} 条可定位；"
            f"当前时钟偏移 {self.clock_offset_ms:+d} ms"
        )

    def _open_row(self, row: int, _column: int) -> None:
        item = self.table.item(row, 0)
        if item is not None:
            self._open_event(str(item.data(Qt.UserRole) or ""))

    def _open_event(self, event_id: str) -> None:
        event = self.passage_store.get(event_id)
        if event is None:
            self.refresh()
            return
        lookup = self._lookup(event)
        available = [
            item for item in lookup.locations if item.status in _OPENABLE_STATUSES
        ]
        if not available:
            QMessageBox.information(self, "无法定位", lookup_status_text(lookup))
            return

        location = available[0]
        if len(available) > 1:
            labels = [
                f"机位 {item.segment.camera_index}: {item.video_path.name}"
                for item in available
            ]
            selected, accepted = QInputDialog.getItem(
                self,
                "选择机位",
                "可用录像",
                labels,
                0,
                False,
            )
            if not accepted:
                return
            location = available[labels.index(selected)]

        if self._open_location is not None:
            self._open_location(event, location)


__all__ = [
    "PassageReviewDialog",
    "format_passage_time",
    "lookup_status_text",
]
