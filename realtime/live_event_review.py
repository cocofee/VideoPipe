"""Embedded evidence review panel for the live operator workspace."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import Qt, QSize, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

try:
    from .io_utils import read_pending_ocr_candidate
except ImportError:
    from io_utils import read_pending_ocr_candidate


class LiveEventReview(QWidget):
    """Show saved event evidence without leaving the live workspace."""

    bib_changed = pyqtSignal(int, str)
    void_changed = pyqtSignal(int, bool)
    next_requested = pyqtSignal(int)

    def __init__(self, database=None, output_dir=None, parent=None):
        super().__init__(parent)
        self.database = database
        self.output_dir = Path(output_dir) if output_dir else None
        self.current_event: Optional[Dict[str, Any]] = None
        self.current_evidences: List[Dict[str, str]] = []
        self.evidence_index = -1
        self._current_pixmap = QPixmap()
        self._thumbnail_buttons: List[QPushButton] = []
        self._init_ui()
        self.clear_event()

    def _init_ui(self):
        self.setObjectName("live_event_review")
        self.setStyleSheet("""
            QWidget#live_event_review {
                background: #ffffff;
                font-family: "Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI", sans-serif;
            }
            QLabel#review_title {
                color: #182230;
                font-size: 15px;
                font-weight: 700;
            }
            QLabel#event_meta {
                color: #667085;
                font-size: 12px;
            }
            QLabel#evidence_view {
                background: #11161c;
                color: #aab4c0;
                border: 1px solid #252c35;
                border-radius: 3px;
                font-size: 13px;
            }
            QLabel#review_feedback {
                color: #667085;
                font-size: 11px;
            }
            QLabel#bib_label {
                color: #344054;
                font-size: 12px;
                font-weight: 700;
            }
            QLineEdit {
                min-height: 36px;
                border: 1px solid #c5ccd6;
                border-radius: 4px;
                padding: 0 10px;
                background: #ffffff;
                color: #182230;
                font-size: 15px;
                font-weight: 700;
            }
            QLineEdit:focus {
                border-color: #3b75a6;
            }
            QPushButton {
                min-height: 36px;
                border: 1px solid #c5ccd6;
                border-radius: 4px;
                padding: 0 12px;
                background: #f8fafb;
                color: #263445;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #eef2f5;
                border-color: #98a2b3;
            }
            QPushButton:disabled {
                background: #f2f4f7;
                color: #98a2b3;
                border-color: #e1e5ea;
            }
            QPushButton#save_next_btn {
                background: #286b9e;
                color: #ffffff;
                border-color: #286b9e;
            }
            QPushButton#save_next_btn:hover {
                background: #225b86;
                border-color: #225b86;
            }
            QPushButton#void_btn {
                color: #a23b3b;
                background: #ffffff;
                border-color: #d8aaaa;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        self.title_label = QLabel("快速核对")
        self.title_label.setObjectName("review_title")
        title_row.addWidget(self.title_label)
        title_row.addStretch()
        self.event_meta_label = QLabel("未选择事件")
        self.event_meta_label.setObjectName("event_meta")
        title_row.addWidget(self.event_meta_label)
        layout.addLayout(title_row)

        self.image_label = QLabel()
        self.image_label.setObjectName("evidence_view")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(180)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_label, 1)

        self.thumbnail_bar = QWidget()
        self.thumbnail_layout = QHBoxLayout(self.thumbnail_bar)
        self.thumbnail_layout.setContentsMargins(0, 0, 0, 0)
        self.thumbnail_layout.setSpacing(6)
        self.thumbnail_layout.addStretch()
        layout.addWidget(self.thumbnail_bar)

        edit_row = QHBoxLayout()
        edit_row.setSpacing(8)
        bib_label = QLabel("号码")
        bib_label.setObjectName("bib_label")
        edit_row.addWidget(bib_label)

        self.bib_input = QLineEdit()
        self.bib_input.setPlaceholderText("输入或修正号码")
        self.bib_input.returnPressed.connect(self._save_and_next)
        edit_row.addWidget(self.bib_input, 1)

        self.save_next_btn = QPushButton("保存并下一条")
        self.save_next_btn.setObjectName("save_next_btn")
        self.save_next_btn.clicked.connect(self._save_and_next)
        edit_row.addWidget(self.save_next_btn)

        self.void_btn = QPushButton("标记作废")
        self.void_btn.setObjectName("void_btn")
        self.void_btn.clicked.connect(self._toggle_void)
        edit_row.addWidget(self.void_btn)
        layout.addLayout(edit_row)

        self.feedback_label = QLabel("选择事件后可核对已保存证据")
        self.feedback_label.setObjectName("review_feedback")
        layout.addWidget(self.feedback_label)

    @pyqtSlot(object)
    def set_database(self, database):
        self.database = database

    @pyqtSlot(object)
    def set_output_dir(self, output_dir):
        self.output_dir = Path(output_dir) if output_dir else None

    @pyqtSlot(dict)
    def set_event(self, event: Dict[str, Any]):
        if not event:
            self.clear_event()
            return

        self.current_event = dict(event)
        candidate = read_pending_ocr_candidate(self.current_event, self.output_dir)
        if candidate:
            self.current_event["_ocr_candidate"] = candidate["bib"]
            self.current_event["_ocr_candidate_confidence"] = candidate["confidence"]
        sequence = self.current_event.get("_video_sequence") or self.current_event.get("event_id") or "-"
        time_text = self.current_event.get("cross_time_str") or "时间未知"
        source_id = int(self.current_event.get("source_id", 0) or 0)
        self.event_meta_label.setText(f"视频序号 {sequence}  |  {time_text}  |  机位 {source_id + 1}")
        display_bib = self.current_event.get("bib_number") or self.current_event.get("_ocr_candidate") or ""
        self.bib_input.setText(str(display_bib))
        self._refresh_void_button()

        self.current_evidences = self._collect_evidences(self.current_event)
        self._rebuild_thumbnails()
        if self.current_evidences:
            self._show_evidence(0)
            self.feedback_label.setText(f"已加载 {len(self.current_evidences)} 张已保存证据")
        else:
            self.evidence_index = -1
            self._current_pixmap = QPixmap()
            self.image_label.clear()
            self.image_label.setText("该事件尚无可用证据图片")
            self.feedback_label.setText("事件已保留，可继续录入号码或等待证据写入")

        if candidate:
            confidence = float(candidate.get("confidence") or 0.0)
            self.feedback_label.setText(
                f"OCR候选 {candidate['bib']}（置信度 {confidence:.0%}），请核对后保存"
            )

        self.bib_input.setEnabled(True)
        self.save_next_btn.setEnabled(True)
        self.void_btn.setEnabled(True)

    def clear_event(self):
        self.current_event = None
        self.current_evidences = []
        self.evidence_index = -1
        self._current_pixmap = QPixmap()
        self.event_meta_label.setText("未选择事件")
        self.image_label.clear()
        self.image_label.setText("从上方事件队列选择一条记录")
        self.bib_input.clear()
        self.bib_input.setEnabled(False)
        self.save_next_btn.setEnabled(False)
        self.void_btn.setEnabled(False)
        self.feedback_label.setText("选择事件后可核对已保存证据")
        self._rebuild_thumbnails()

    def _collect_evidences(self, event: Dict[str, Any]) -> List[Dict[str, str]]:
        evidences: List[Dict[str, str]] = []
        seen = set()

        def add(path_value, label):
            if not path_value or len(evidences) >= 5:
                return
            path = Path(str(path_value)).expanduser()
            if not path.is_absolute() and self.output_dir:
                path = self.output_dir / path
            try:
                key = str(path.resolve())
            except OSError:
                key = str(path.absolute())
            if key in seen or not path.is_file():
                return
            seen.add(key)
            evidences.append({"path": str(path), "label": label})

        direct_fields = (
            ("screenshot_full", "全景"),
            ("screenshot_clean", "原始帧"),
            ("screenshot_crop", "运动员"),
            ("screenshot_bib", "号码"),
        )
        for field, label in direct_fields:
            add(event.get(field), label)

        event_id = event.get("event_id")
        if self.database and event_id:
            try:
                for evidence in self.database.get_event_evidences(int(event_id)) or []:
                    source_id = int(evidence.get("source_id", 0) or 0) + 1
                    for field, label in direct_fields:
                        add(evidence.get(field), f"机位 {source_id} {label}")
            except Exception:
                pass

        evidence_dir = event.get("evidence_dir")
        if not evidence_dir and event_id and self.output_dir:
            candidate = self.output_dir / "evidence_photos" / f"{int(event_id):06d}"
            if candidate.is_dir():
                evidence_dir = candidate

        if evidence_dir and len(evidences) < 5:
            root = Path(str(evidence_dir)).expanduser()
            if not root.is_absolute() and self.output_dir:
                root = self.output_dir / root
            meta_path = root / "meta.json"
            paths: Dict[str, Any] = {}
            if meta_path.is_file():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    paths = meta.get("paths", {}) if isinstance(meta, dict) else {}
                except (OSError, ValueError, TypeError):
                    paths = {}

            if paths:
                for key, label in (
                    ("full", "证据全景"),
                    ("athlete", "运动员裁剪"),
                    ("bib", "号码裁剪"),
                ):
                    relative_path = paths.get(key)
                    if relative_path:
                        add(root / str(relative_path), label)
                for index, relative_path in enumerate(paths.get("bib_candidates") or [], start=1):
                    add(root / str(relative_path), f"号码候选 {index}")
            else:
                for name, label in (
                    ("full.jpg", "证据全景"),
                    ("athlete.jpg", "运动员裁剪"),
                    ("bib.jpg", "号码裁剪"),
                    ("bib_candidate_01.jpg", "号码候选 1"),
                    ("bib_candidate_02.jpg", "号码候选 2"),
                ):
                    add(root / name, label)

        return evidences[:5]

    def _rebuild_thumbnails(self):
        while self.thumbnail_layout.count() > 1:
            item = self.thumbnail_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._thumbnail_buttons = []

        for index, evidence in enumerate(self.current_evidences):
            button = QPushButton()
            button.setObjectName("evidence_thumbnail")
            button.setFixedSize(72, 52)
            button.setToolTip(evidence["label"])
            pixmap = QPixmap(evidence["path"])
            if not pixmap.isNull():
                button.setIcon(QIcon(pixmap.scaled(64, 42, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
                button.setIconSize(QSize(64, 42))
            button.clicked.connect(lambda _checked=False, idx=index: self._show_evidence(idx))
            self.thumbnail_layout.insertWidget(self.thumbnail_layout.count() - 1, button)
            self._thumbnail_buttons.append(button)

        self.thumbnail_bar.setVisible(bool(self.current_evidences))

    def _show_evidence(self, index: int):
        if not (0 <= index < len(self.current_evidences)):
            return
        self.evidence_index = index
        evidence = self.current_evidences[index]
        self._current_pixmap = QPixmap(evidence["path"])
        if self._current_pixmap.isNull():
            self.image_label.setText("证据图片无法读取")
        else:
            self._render_current_pixmap()

        for button_index, button in enumerate(self._thumbnail_buttons):
            border = "#286b9e" if button_index == index else "#c5ccd6"
            background = "#edf5fb" if button_index == index else "#f8fafb"
            button.setStyleSheet(
                "QPushButton#evidence_thumbnail {"
                f"background: {background}; border: 2px solid {border}; "
                "border-radius: 3px; padding: 2px;"
                "}"
            )
            button.setFixedSize(72, 52)
        self.feedback_label.setText(f"{evidence['label']}  |  {index + 1}/{len(self.current_evidences)}")

    def _render_current_pixmap(self):
        if self._current_pixmap.isNull():
            return
        target = self.image_label.size()
        if target.width() <= 4 or target.height() <= 4:
            return
        scaled = self._current_pixmap.scaled(
            max(1, target.width() - 4),
            max(1, target.height() - 4),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render_current_pixmap()

    def _save_and_next(self):
        if not self.current_event:
            return
        event_id = int(self.current_event.get("event_id") or 0)
        if not event_id:
            return
        bib = self.bib_input.text().strip()
        self.current_event["bib_number"] = bib or None
        self.current_event["bib_status"] = "recognized" if bib else "unrecognized"
        self.bib_changed.emit(event_id, bib)
        self.feedback_label.setText(f"号码 {bib or '空'} 已保存")
        self.next_requested.emit(event_id)

    def _toggle_void(self):
        if not self.current_event:
            return
        event_id = int(self.current_event.get("event_id") or 0)
        if not event_id:
            return
        new_void = not bool(self.current_event.get("is_void"))
        self.current_event["is_void"] = 1 if new_void else 0
        self.void_changed.emit(event_id, new_void)
        self._refresh_void_button()
        self.feedback_label.setText("事件已作废" if new_void else "事件已恢复")

    def _refresh_void_button(self):
        is_void = bool(self.current_event and self.current_event.get("is_void"))
        self.void_btn.setText("取消作废" if is_void else "标记作废")
        if is_void:
            self.void_btn.setStyleSheet(
                "QPushButton#void_btn { color: #344054; border-color: #98a2b3; background: #f2f4f7; }"
            )
        else:
            self.void_btn.setStyleSheet("")
