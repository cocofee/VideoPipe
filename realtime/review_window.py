"""Production finish console without detection or OCR dependencies."""

from __future__ import annotations

import logging
import inspect
import socket
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QShortcut,
    QStyle,
    QVBoxLayout,
)

try:
    from .external_clip_import import (
        EXTERNAL_CLOCK_SOURCE,
        ExternalClipImportError,
        import_external_clip_sidecar,
        race_id_from_passage_store,
    )
    from .passage_evidence import PassageEvidenceAssociationStore
    from .passage_receiver import (
        DEFAULT_HOST,
        DEFAULT_PORT,
        PassageEvent,
        PassageEventReceiver,
        PassageEventStore,
        RaceFocus,
    )
    from .passage_review import PassageReviewDialog
    from .race_metadata import RaceMetadata, RaceMetadataStore
    from .review_recorder import (
        ArchiveTimelinePublisher,
        FfmpegReviewRecorder,
        PassageReviewCoordinator,
        PassageReviewState,
        PassageReviewTimelinePublisher,
        PassageReviewWindow,
        ReviewRingBuffer,
        discover_directshow_video_devices,
        is_supported_review_source,
        load_archive_recording_sessions,
        make_directshow_source,
        parse_directshow_source,
    )
    from .stream_recorder import (
        RecordingError,
        sanitize_recording_message,
    )
    from .video_timeline import DEFAULT_TIMING_ERROR_MS, VideoTimelineStore
except ImportError:
    from external_clip_import import (
        EXTERNAL_CLOCK_SOURCE,
        ExternalClipImportError,
        import_external_clip_sidecar,
        race_id_from_passage_store,
    )
    from passage_evidence import PassageEvidenceAssociationStore
    from passage_receiver import (
        DEFAULT_HOST,
        DEFAULT_PORT,
        PassageEvent,
        PassageEventReceiver,
        PassageEventStore,
        RaceFocus,
    )
    from passage_review import PassageReviewDialog
    from race_metadata import RaceMetadata, RaceMetadataStore
    from review_recorder import (
        ArchiveTimelinePublisher,
        FfmpegReviewRecorder,
        PassageReviewCoordinator,
        PassageReviewState,
        PassageReviewTimelinePublisher,
        PassageReviewWindow,
        ReviewRingBuffer,
        discover_directshow_video_devices,
        is_supported_review_source,
        load_archive_recording_sessions,
        make_directshow_source,
        parse_directshow_source,
    )
    from stream_recorder import (
        RecordingError,
        sanitize_recording_message,
    )
    from video_timeline import DEFAULT_TIMING_ERROR_MS, VideoTimelineStore


logger = logging.getLogger("VideoPipe.FinishReview")


@dataclass(frozen=True, slots=True)
class FinishReviewSettings:
    source: str
    output_dir: Path
    passage_host: str
    passage_port: int
    camera_index: int


class FinishReviewLaunchDialog(QDialog):
    """Operator-facing device and race-directory settings."""

    def __init__(
        self,
        settings: FinishReviewSettings,
        parent=None,
        *,
        ffmpeg_path: Path | None = None,
        device_provider: Callable[[], tuple[str, ...]] | None = None,
    ):
        super().__init__(parent)
        self._source = str(settings.source).strip()
        self._output_dir = Path(settings.output_dir).expanduser().resolve()
        self._passage_host = settings.passage_host
        self._passage_port = settings.passage_port
        self._camera_index = settings.camera_index
        self._ffmpeg_path = Path(ffmpeg_path).resolve() if ffmpeg_path else None
        self._detected_device_names: set[str] = set()
        self._device_provider = device_provider or (
            lambda: discover_directshow_video_devices(self._ffmpeg_path)
        )
        self.setWindowTitle("设备与赛事设置")
        self.setMinimumWidth(680)
        self.setModal(True)
        self.setStyleSheet(
            "QDialog { background: #eef2f5; color: #17212b; }"
            "QLineEdit { min-height: 32px; padding: 0 8px; background: #ffffff; "
            "border: 1px solid #aeb8c2; border-radius: 4px; }"
            "QPushButton { min-height: 32px; padding: 0 12px; background: #ffffff; "
            "border: 1px solid #aeb8c2; border-radius: 4px; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(14)
        title = QLabel("终点设备与赛事设置", self)
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        layout.addWidget(title)

        form = QFormLayout()
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(12)
        device_row = QHBoxLayout()
        device_row.setSpacing(6)
        self.device_combo = QComboBox(self)
        self.device_combo.setMinimumWidth(360)
        self.device_combo.currentIndexChanged.connect(self._refresh_camera_status)
        device_row.addWidget(self.device_combo, 1)
        self.detect_button = QPushButton("重新检测", self)
        self.detect_button.clicked.connect(self._refresh_devices)
        device_row.addWidget(self.detect_button)
        form.addRow("录像设备", device_row)

        self.video_size_combo = QComboBox(self)
        self.video_size_combo.addItem("自动", None)
        for value in ("1920x1080", "2560x1440", "3840x2160"):
            self.video_size_combo.addItem(value, value)
        form.addRow("录像分辨率", self.video_size_combo)

        self.framerate_combo = QComboBox(self)
        self.framerate_combo.addItem("自动", None)
        for value in (25.0, 30.0, 50.0, 60.0):
            self.framerate_combo.addItem(f"{value:g} FPS", value)
        form.addRow("录像帧率", self.framerate_combo)

        output_row = QHBoxLayout()
        output_row.setSpacing(6)
        self.output_edit = QLineEdit(str(self._output_dir), self)
        self.output_edit.setReadOnly(True)
        output_row.addWidget(self.output_edit, 1)
        browse_button = QPushButton(self)
        browse_button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        browse_button.setToolTip("选择本机录像与证据保存目录")
        browse_button.setFixedWidth(42)
        browse_button.clicked.connect(self._browse_output_dir)
        output_row.addWidget(browse_button)
        form.addRow("录像证据保存", output_row)

        cycle_status = QLabel(
            f"自动发现本机“{socket.gethostname()}”，"
            "同机或局域网电脑都无需共享目录、无需填写IP",
            self,
        )
        cycle_status.setStyleSheet("color: #247a52; font-weight: 600;")
        cycle_status.setWordWrap(True)
        form.addRow("CycleRace", cycle_status)
        self.camera_status_label = QLabel(self)
        self.camera_status_label.setObjectName("recordingDeviceStatus")
        form.addRow("设备检查", self.camera_status_label)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            parent=self,
        )
        self.start_button = buttons.button(QDialogButtonBox.Ok)
        self.start_button.setText("保存设置")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._refresh_devices()

    def _refresh_devices(self) -> None:
        current_source = self._source
        parsed_source = parse_directshow_source(current_source)
        selected_name = parsed_source.device_name if parsed_source is not None else ""
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        if is_supported_review_source(current_source) and parsed_source is None:
            self.device_combo.addItem("网络摄像机（已配置）", current_source)
        try:
            devices = tuple(self._device_provider())
        except Exception:  # noqa: BLE001 - device discovery is best effort.
            devices = ()
        self._detected_device_names = set(devices)
        for device_name in devices:
            self.device_combo.addItem(device_name, device_name)
        if selected_name and self.device_combo.findData(selected_name) < 0:
            self.device_combo.addItem(f"{selected_name}（当前未检测到）", selected_name)
        selected_index = self.device_combo.findData(
            current_source if parsed_source is None else selected_name
        )
        self.device_combo.setCurrentIndex(max(0, selected_index))
        self.device_combo.blockSignals(False)
        if parsed_source is not None:
            size_index = self.video_size_combo.findData(parsed_source.video_size)
            fps_index = self.framerate_combo.findData(parsed_source.framerate)
            self.video_size_combo.setCurrentIndex(max(0, size_index))
            self.framerate_combo.setCurrentIndex(max(0, fps_index))
        self._refresh_camera_status()

    def _refresh_camera_status(self) -> None:
        configured = self.device_combo.currentIndex() >= 0
        selected = str(self.device_combo.currentData() or "")
        if not configured:
            text = "未检测到USB/Type-C摄像头"
        elif is_supported_review_source(selected):
            text = "网络摄像机配置已保留，开始录像后验证画面"
        elif selected not in self._detected_device_names:
            text = "录像设备已配置，但当前未检测到"
        else:
            text = "已检测到摄像头，开始录像后验证画面"
        ready = configured and (
            is_supported_review_source(selected)
            or selected in self._detected_device_names
        )
        self.camera_status_label.setText(text)
        self.camera_status_label.setStyleSheet(
            "color: #247a52; font-weight: 600;"
            if ready
            else "color: #b54747; font-weight: 600;"
        )
        self.start_button.setEnabled(configured)

    def _browse_output_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "选择本机录像与证据保存目录",
            str(self._output_dir),
        )
        if selected:
            self._output_dir = Path(selected).resolve()
            self.output_edit.setText(str(self._output_dir))

    @property
    def settings(self) -> FinishReviewSettings:
        selected = str(self.device_combo.currentData() or "").strip()
        if is_supported_review_source(selected):
            source = selected
        else:
            source = make_directshow_source(
                selected,
                video_size=self.video_size_combo.currentData(),
                framerate=self.framerate_combo.currentData(),
            )
        return FinishReviewSettings(
            source=source,
            output_dir=self._output_dir,
            passage_host=self._passage_host,
            passage_port=self._passage_port,
            camera_index=self._camera_index,
        )


class _PassageSignalBridge(QObject):
    accepted = pyqtSignal(object)
    metadata_accepted = pyqtSignal(object)
    focus_accepted = pyqtSignal(object)


class FinishReviewWindow(PassageReviewDialog):
    """Production console for recording, CycleRace intake, and evidence review."""

    def __init__(
        self,
        source: str,
        output_dir: str | Path,
        parent=None,
        *,
        passage_host: str = DEFAULT_HOST,
        passage_port: int = DEFAULT_PORT,
        camera_index: int = 1,
        ffmpeg_path: Path | None = None,
        review_retention_seconds: int = 90,
        timing_error_ms: int = DEFAULT_TIMING_ERROR_MS,
        refresh_interval_ms: int = 500,
        passage_batch_interval_ms: int = 150,
        recorder_factory: Callable[..., FfmpegReviewRecorder] = FfmpegReviewRecorder,
        receiver_factory: Callable[..., PassageEventReceiver] = PassageEventReceiver,
        settings_saver: Callable[[FinishReviewSettings], None] | None = None,
    ):
        self.source = str(source).strip()
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.passage_host = str(passage_host).strip()
        self.passage_port = int(passage_port)
        self.camera_index = max(1, int(camera_index))
        self.ffmpeg_path = Path(ffmpeg_path).resolve() if ffmpeg_path else None
        self.review_retention_seconds = max(6, int(review_retention_seconds))
        self.timing_error_ms = max(0, int(timing_error_ms))
        self._recorder_factory = recorder_factory
        self._receiver_factory = receiver_factory
        self._settings_saver = settings_saver

        passage_store = PassageEventStore(
            self.output_dir / "cyclerace_passage_events.jsonl"
        )
        metadata_store = RaceMetadataStore(
            self.output_dir / "cyclerace_race_metadata.json"
        )
        timeline_store = VideoTimelineStore(self.output_dir / "video_timeline.jsonl")
        super().__init__(
            passage_store,
            timeline_store,
            parent,
            metadata_store=metadata_store,
        )

        self.setWindowTitle("VideoPipe 终点多源核对")
        self.setMinimumSize(1180, 760)
        self._recorder: FfmpegReviewRecorder | None = None
        self._receiver: PassageEventReceiver | None = None
        self._ring_buffer: ReviewRingBuffer | None = None
        self._coordinator: PassageReviewCoordinator | None = None
        self._publisher: PassageReviewTimelinePublisher | None = None
        self._archive_publishers = [
            ArchiveTimelinePublisher(session, self.timeline_store)
            for session in load_archive_recording_sessions(self.output_dir)
        ]
        self._capture_windows: dict[str, PassageReviewWindow] = {}
        self._published_keys: set[tuple[str, int]] = set()
        self._unsupported_event_ids: set[str] = set()
        self._runtime_error = ""
        self._capture_error = ""
        self._receiver_error = ""
        self._started = False
        self._last_cleanup_at = 0.0
        self._recording_started_at = 0.0
        self._historical_passage_count = len(passage_store)
        self._received_passage_count = 0
        self._last_passage_monotonic = 0.0
        self._pending_focus: RaceFocus | None = None
        self._pending_passages: dict[str, PassageEvent] = {}

        self._signal_bridge = _PassageSignalBridge(self)
        self._signal_bridge.accepted.connect(self._on_passage_received)
        self._signal_bridge.metadata_accepted.connect(self._on_metadata_received)
        self._signal_bridge.focus_accepted.connect(self._on_focus_received)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(max(100, int(refresh_interval_ms)))
        self._refresh_timer.timeout.connect(self._refresh_capture_windows)
        self._passage_batch_timer = QTimer(self)
        self._passage_batch_timer.setSingleShot(True)
        self._passage_batch_timer.setInterval(
            max(0, int(passage_batch_interval_ms))
        )
        self._passage_batch_timer.timeout.connect(self._flush_passage_batch)
        self._clock_timer = QTimer(self)
        self._clock_timer.setInterval(1_000)
        self._clock_timer.timeout.connect(self._update_runtime_status)
        self._init_runtime_status()
        self._init_operator_controls()
        self.auto_advance_checkbox.setChecked(False)
        self.auto_advance_checkbox.hide()
        try:
            if self._publish_archive_segments():
                self._lookup_cache.clear()
                self.refresh()
        except Exception as exc:  # noqa: BLE001 - recovery remains operator-visible.
            self._capture_error = sanitize_recording_message(exc)
            logger.exception("Failed to recover archived recording sessions")
        self._clock_timer.start()
        self._update_runtime_status()

    @property
    def recorder(self) -> FfmpegReviewRecorder | None:
        return self._recorder

    @property
    def receiver(self) -> PassageEventReceiver | None:
        return self._receiver

    def _select_event(self, event_id: str) -> None:
        super()._select_event(event_id)
        self._update_operator_controls()

    def _clear_selection_details(self) -> None:
        super()._clear_selection_details()
        self._update_operator_controls()

    def _update_operator_controls(self) -> None:
        label = getattr(self, "operator_identity_label", None)
        if label is None:
            return
        event = self.passage_store.get(self._selected_event_id)
        identity = ""
        athlete_name = ""
        if event is not None:
            identity = event.bib.strip()
            athlete_name = event.athlete_name.strip()
        else:
            identity = self.selected_identity_value.text().strip()
            athlete_name = self.athlete_value.text().strip()
            if identity == "--":
                identity = ""
            if athlete_name == "--":
                athlete_name = ""
        athlete_summary = f"{identity} {athlete_name}".strip()
        label.setText(
            f"当前运动员：{athlete_summary}"
            if athlete_summary
            else "当前运动员：未选择"
        )
        regular_ready = bool(
            identity
            and getattr(self.regular_pane.video_view, "has_frame", False)
        )
        high_speed_ready = bool(
            identity
            and getattr(self.high_speed_pane.video_view, "has_frame", False)
        )
        self.mark_regular_button.setText(
            f"标线普通录像 {identity}" if identity else "标线普通录像"
        )
        self.mark_high_speed_button.setText(
            f"标线高速摄像 {identity}" if identity else "标线高速摄像"
        )
        self.mark_regular_button.setEnabled(regular_ready)
        self.mark_high_speed_button.setEnabled(high_speed_ready)
        has_pending_marker = bool(
            self.regular_pane.has_pending_marker
            or self.high_speed_pane.has_pending_marker
        )
        self.confirm_next_button.setEnabled(bool(identity and has_pending_marker))
        for shortcut in getattr(self, "confirm_marker_shortcuts", ()):
            shortcut.setEnabled(bool(identity and has_pending_marker))

    def _pending_marker_pane(self):
        if self.regular_pane.has_pending_marker:
            return self.regular_pane
        if self.high_speed_pane.has_pending_marker:
            return self.high_speed_pane
        return None

    def _confirm_current_marker(self) -> None:
        pane = self._pending_marker_pane()
        if pane is None:
            return
        self._confirm_pending_marker(pane)
        self._update_operator_controls()

    def _confirm_and_next(self) -> None:
        pane = self._pending_marker_pane()
        if pane is None:
            return
        row = self.table.currentRow()
        event_id = self._selected_event_id
        confirmed = self._confirm_pending_marker(pane)
        if (
            confirmed
            and self._selected_event_id == event_id
            and 0 <= row < self.table.rowCount() - 1
        ):
            self._move_selection(1)

    def _toggle_recording(self) -> None:
        recorder = self._recorder
        if recorder is not None and recorder.is_running:
            self.stop_recording()
            return
        if not is_supported_review_source(self.source):
            self._configure_devices()
            if not is_supported_review_source(self.source):
                return
        try:
            self.start_recording()
        except Exception as exc:  # noqa: BLE001 - GUI boundary reports device failures.
            self._runtime_error = sanitize_recording_message(exc)
            QMessageBox.critical(self, "无法开始录像", self._runtime_error)
            self._update_runtime_status()

    def _configure_devices(self) -> None:
        if self._recorder is not None and self._recorder.is_running:
            answer = QMessageBox.question(
                self,
                "停止录像并修改设置",
                "修改录像设备或赛事目录前需要停止当前录像，是否继续？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
            self.stop_recording()
        dialog = FinishReviewLaunchDialog(
            FinishReviewSettings(
                source=self.source,
                output_dir=self.output_dir,
                passage_host=self.passage_host,
                passage_port=self.passage_port,
                camera_index=self.camera_index,
            ),
            self,
            ffmpeg_path=self.ffmpeg_path,
        )
        if dialog.exec_() != QDialog.Accepted:
            return
        self._apply_settings(dialog.settings)

    def _apply_settings(self, settings: FinishReviewSettings) -> None:
        output_dir = Path(settings.output_dir).expanduser().resolve()
        output_changed = output_dir != self.output_dir
        if output_changed:
            self.stop_receiver()
            self._passage_batch_timer.stop()
            self._pending_passages.clear()
        self.source = str(settings.source).strip()
        self.passage_host = str(settings.passage_host).strip()
        self.passage_port = int(settings.passage_port)
        self.camera_index = max(1, int(settings.camera_index))
        if output_changed:
            output_dir.mkdir(parents=True, exist_ok=True)
            self.output_dir = output_dir
            self.passage_store = PassageEventStore(
                output_dir / "cyclerace_passage_events.jsonl"
            )
            self.metadata_store = RaceMetadataStore(
                output_dir / "cyclerace_race_metadata.json"
            )
            self.timeline_store = VideoTimelineStore(
                output_dir / "video_timeline.jsonl"
            )
            self.association_store = PassageEvidenceAssociationStore(
                output_dir / "passage_evidence_associations.jsonl"
            )
            self._lookup_cache.clear()
            self._timeline_signature = ()
            self._selected_event_id = ""
            self._capture_windows.clear()
            self._published_keys.clear()
            self._archive_publishers = [
                ArchiveTimelinePublisher(session, self.timeline_store)
                for session in load_archive_recording_sessions(self.output_dir)
            ]
            self._unsupported_event_ids.clear()
            self._historical_passage_count = len(self.passage_store)
            self._received_passage_count = 0
            self._capture_error = ""
            self.refresh()
            try:
                self.start_receiver()
            except Exception as exc:  # noqa: BLE001 - settings remain applied.
                self._runtime_error = sanitize_recording_message(exc)
                QMessageBox.warning(
                    self,
                    "CycleRace监听未启动",
                    self._runtime_error,
                )
        if self._settings_saver is not None:
            try:
                self._settings_saver(settings)
            except OSError as exc:
                QMessageBox.warning(self, "设置未保存", str(exc))
        self._runtime_error = ""
        self._update_runtime_status()

    def _import_high_speed_sidecar(self) -> None:
        try:
            race_id = self._current_archive_race_id()
        except ExternalClipImportError as exc:
            QMessageBox.information(self, "暂时无法导入", str(exc))
            return
        sidecar_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "导入高速摄像北京时间 sidecar",
            str(self.output_dir),
            "JSON sidecar (*.json);;所有文件 (*)",
        )
        if not sidecar_path:
            return
        try:
            result = import_external_clip_sidecar(
                self.timeline_store,
                sidecar_path,
                expected_race_id=race_id,
            )
        except ExternalClipImportError as exc:
            QMessageBox.warning(self, "高速摄像导入失败", str(exc))
            return
        self._lookup_cache.clear()
        self.refresh()
        self._update_runtime_status()
        QMessageBox.information(
            self,
            "高速摄像导入完成",
            f"新增 {result.created_count} 段，修复 {result.repaired_count} 段，"
            f"跳过重复 {result.duplicate_count} 段。",
        )

    def _init_runtime_status(self) -> None:
        panel = QFrame(self)
        panel.setObjectName("finishConsoleHeader")
        panel.setStyleSheet(
            "QFrame#finishConsoleHeader { background: #f7f9fb; "
            "border: 1px solid #cfd7df; border-radius: 4px; }"
            "QLabel { color: #44515d; font-size: 12px; font-weight: 600; }"
            "QLabel[statusChip='true'] { background: #ffffff; border: 1px solid #c7d0d9; "
            "border-radius: 4px; padding: 6px 9px; }"
            "QPushButton { min-height: 32px; padding: 0 12px; font-weight: 600; }"
        )
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(10, 8, 10, 8)
        panel_layout.setSpacing(7)

        title_row = QHBoxLayout()
        title_row.setSpacing(10)
        title = QLabel("终点多源核对", panel)
        title.setStyleSheet("font-size: 18px; font-weight: 700; color: #17212b;")
        self.race_dir_label = QLabel(panel)
        self.race_dir_label.setStyleSheet("color: #667085; font-weight: 500;")
        self.beijing_clock_label = QLabel(panel)
        self.beijing_clock_label.setStyleSheet(
            "font-family: Consolas; color: #17212b; font-size: 13px;"
        )
        title_row.addWidget(title)
        title_row.addWidget(self.race_dir_label)
        title_row.addStretch()
        title_row.addWidget(self.beijing_clock_label)
        panel_layout.addLayout(title_row)

        layout = QHBoxLayout()
        layout.setSpacing(8)
        self.camera_status_label = self._status_chip(panel)
        self.recording_status_label = self._status_chip(panel)
        self.receiver_status_label = self._status_chip(panel)
        self.high_speed_status_label = self._status_chip(panel)
        self.storage_status_label = self._status_chip(panel)
        self.capture_status_label = QLabel(panel)
        self.capture_status_label.setStyleSheet("color: #667085; font-weight: 500;")
        layout.addWidget(self.camera_status_label)
        layout.addWidget(self.recording_status_label)
        layout.addWidget(self.receiver_status_label)
        layout.addWidget(self.high_speed_status_label)
        layout.addWidget(self.storage_status_label)
        layout.addStretch(1)

        self.recheck_button = QPushButton("重新检查", panel)
        self.recheck_button.clicked.connect(self._recheck_connections)
        layout.addWidget(self.recheck_button)
        self.settings_button = QPushButton("设备设置", panel)
        self.settings_button.clicked.connect(self._configure_devices)
        layout.addWidget(self.settings_button)
        self.import_high_speed_button = QPushButton("导入高速", panel)
        self.import_high_speed_button.clicked.connect(self._import_high_speed_sidecar)
        layout.addWidget(self.import_high_speed_button)
        self.record_button = QPushButton("开始录像", panel)
        self.record_button.setObjectName("finishRecordButton")
        self.record_button.clicked.connect(self._toggle_recording)
        layout.addWidget(self.record_button)
        panel_layout.addLayout(layout)
        panel_layout.addWidget(self.capture_status_label)
        root_layout = self.layout()
        if root_layout is not None:
            root_layout.insertWidget(0, panel)

    @staticmethod
    def _status_chip(parent) -> QLabel:
        label = QLabel(parent)
        label.setProperty("statusChip", True)
        return label

    def _init_operator_controls(self) -> None:
        panel = QFrame(self)
        panel.setObjectName("finishOperatorBar")
        panel.setStyleSheet(
            "QFrame#finishOperatorBar { background: #ffffff; border: 1px solid #cfd7df; "
            "border-radius: 4px; }"
            "QPushButton { min-height: 32px; padding: 0 12px; font-weight: 600; }"
        )
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(8)
        self.operator_identity_label = QLabel("当前运动员：未选择", panel)
        self.operator_identity_label.setStyleSheet(
            "font-size: 14px; font-weight: 700; color: #17212b;"
        )
        layout.addWidget(self.operator_identity_label)
        layout.addStretch()
        self.mark_regular_button = QPushButton("标线普通录像", panel)
        self.mark_regular_button.clicked.connect(
            lambda: self._begin_marking(self.regular_pane)
        )
        layout.addWidget(self.mark_regular_button)
        self.mark_high_speed_button = QPushButton("标线高速摄像", panel)
        self.mark_high_speed_button.clicked.connect(
            lambda: self._begin_marking(self.high_speed_pane)
        )
        layout.addWidget(self.mark_high_speed_button)
        self.confirm_next_button = QPushButton("确认并下一条", panel)
        self.confirm_next_button.setShortcut("Ctrl+Return")
        self.confirm_next_button.clicked.connect(self._confirm_and_next)
        layout.addWidget(self.confirm_next_button)
        self.confirm_marker_shortcuts = []
        for key in (Qt.Key_Return, Qt.Key_Enter):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.activated.connect(self._confirm_current_marker)
            shortcut.setEnabled(False)
            self.confirm_marker_shortcuts.append(shortcut)
        for pane in (self.regular_pane, self.high_speed_pane):
            pane.video_view.marker_position_selected.connect(
                lambda _x, _y: QTimer.singleShot(0, self._update_operator_controls)
            )
            pane.confirmation_requested.connect(
                lambda _pane: QTimer.singleShot(0, self._update_operator_controls)
            )
            pane.cancel_requested.connect(
                lambda _pane: QTimer.singleShot(0, self._update_operator_controls)
            )
            pane.delete_requested.connect(
                lambda _pane: QTimer.singleShot(0, self._update_operator_controls)
            )
        root_layout = self.layout()
        if root_layout is not None:
            root_layout.insertWidget(max(0, root_layout.count() - 1), panel)
        self._update_operator_controls()

    def start_receiver(self) -> None:
        if self._receiver is not None and self._receiver.is_running:
            return
        receiver_kwargs = {
            "on_accepted": self._signal_bridge.accepted.emit,
        }
        parameters = inspect.signature(self._receiver_factory).parameters.values()
        supports_metadata = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        ) or "metadata_store" in {
            parameter.name for parameter in parameters
        }
        if supports_metadata:
            receiver_kwargs.update(
                metadata_store=self.metadata_store,
                on_metadata_accepted=self._signal_bridge.metadata_accepted.emit,
            )
        supports_focus = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        ) or "on_focus_accepted" in {
            parameter.name for parameter in parameters
        }
        if supports_focus:
            receiver_kwargs["on_focus_accepted"] = (
                self._signal_bridge.focus_accepted.emit
            )
        receiver = self._receiver_factory(
            self.passage_host,
            self.passage_port,
            self.passage_store,
            **receiver_kwargs,
        )
        try:
            receiver.start()
        except Exception as exc:
            self._receiver_error = sanitize_recording_message(exc)
            try:
                receiver.stop()
            except Exception as exc:  # noqa: BLE001 - rollback is best effort.
                logger.warning("Failed to stop CycleRace receiver: %s", exc)
            self._update_runtime_status()
            raise
        self._receiver = receiver
        self._receiver_error = ""
        self._refresh_timer.start()
        self._update_runtime_status()

    def _recheck_connections(self) -> None:
        receiver = self._receiver
        if receiver is None or not receiver.is_running:
            try:
                self.start_receiver()
            except Exception as exc:  # noqa: BLE001 - retry remains operator-visible.
                QMessageBox.warning(
                    self,
                    "CycleRace监听未启动",
                    sanitize_recording_message(exc),
                )
        self._update_runtime_status()

    def start_recording(self) -> None:
        if self._recorder is not None and self._recorder.is_running:
            return
        if not is_supported_review_source(self.source):
            raise RecordingError("请先在设备设置中选择录像摄像头")
        try:
            free_bytes = shutil.disk_usage(self.output_dir).free
        except OSError as exc:
            raise RecordingError(f"无法检查赛事存储空间: {exc}") from exc
        if free_bytes < 1024**3:
            raise RecordingError("赛事存储空间不足 1 GB，无法开始录像")
        recorder = self._recorder_factory(
            self.source,
            self.output_dir,
            camera_index=self.camera_index,
            ffmpeg_path=self.ffmpeg_path,
            review_retention_seconds=self.review_retention_seconds,
        )
        archive_publisher = None
        try:
            playlist_path = recorder.start()
            ring_buffer = ReviewRingBuffer(
                playlist_path,
                camera_index=self.camera_index,
                retention_seconds=self.review_retention_seconds,
            )
            ring_buffer.scan()
            coordinator = PassageReviewCoordinator(ring_buffer)
            publisher = PassageReviewTimelinePublisher(
                ring_buffer,
                self.timeline_store,
                timing_error_ms=self.timing_error_ms,
            )
            archive_publisher = ArchiveTimelinePublisher(
                recorder,
                self.timeline_store,
            )
            self._recorder = recorder
            self._ring_buffer = ring_buffer
            self._coordinator = coordinator
            self._publisher = publisher
            self._archive_publishers.append(archive_publisher)
            for event in self._events_for_current_metadata(
                self.passage_store.events()
            ):
                self._register_passage(event, scan=False)
            self._started = True
            self._recording_started_at = time.monotonic()
            self._runtime_error = ""
            self._capture_error = ""
            self._refresh_timer.start()
            self._refresh_capture_windows()
            self._lookup_cache.clear()
            self.refresh()
        except Exception:
            try:
                recorder.stop()
            except Exception as cleanup_error:  # noqa: BLE001
                logger.warning(
                    "Failed to stop recorder during startup rollback: %s",
                    cleanup_error,
                )
            self._recorder = None
            self._ring_buffer = None
            self._coordinator = None
            self._publisher = None
            if (
                archive_publisher is not None
                and archive_publisher in self._archive_publishers
            ):
                self._archive_publishers.remove(archive_publisher)
            raise
        finally:
            self._update_runtime_status()

    def start(self) -> None:
        """Compatibility helper used by automation that starts the full session."""

        self.start_receiver()
        self.start_recording()

    def _passage_timestamp(self, event: PassageEvent) -> int | None:
        timestamp_ms = int(event.timeline_timestamp_ms)
        if event.passage_timestamp_ms is None and timestamp_ms < 86_400_000:
            return None
        return timestamp_ms

    def _register_passage(self, event: PassageEvent, *, scan: bool = True) -> None:
        if not event.is_active:
            self._discard_registered_passage(event.event_id)
            return
        coordinator = self._coordinator
        if coordinator is None:
            return
        timestamp_ms = self._passage_timestamp(event)
        if timestamp_ms is None:
            self._unsupported_event_ids.add(event.event_id)
            self._capture_windows.pop(event.event_id, None)
            return
        self._unsupported_event_ids.discard(event.event_id)
        window = coordinator.register(
            event.event_id,
            passage_timestamp_ms=timestamp_ms,
            scan=scan,
        )
        self._capture_windows[event.event_id] = window
        self._publish_window(window, event)

    def _discard_registered_passage(self, event_id: str) -> None:
        coordinator = self._coordinator
        if coordinator is not None:
            coordinator.discard(event_id)
        self._capture_windows.pop(event_id, None)
        self._unsupported_event_ids.discard(event_id)

    def _publish_window(
        self,
        window: PassageReviewWindow,
        event: PassageEvent,
    ) -> bool:
        publisher = self._publisher
        key = (window.event_id, window.passage_timestamp_ms)
        if (
            publisher is None
            or window.state is not PassageReviewState.READY
            or key in self._published_keys
        ):
            return False
        publisher.publish(window, race_id=event.race_id)
        self._published_keys.add(key)
        return True

    def _on_passage_received(self, event: PassageEvent) -> None:
        self._received_passage_count += 1
        self._historical_passage_count = len(self.passage_store)
        self._last_passage_monotonic = time.monotonic()
        self._pending_passages[event.event_id] = event
        if not self._passage_batch_timer.isActive():
            self._passage_batch_timer.start()
        self._update_runtime_status()

    def _flush_passage_batch(self) -> None:
        pending_events = tuple(self._pending_passages.values())
        self._pending_passages.clear()
        if not pending_events:
            self._update_runtime_status()
            return
        metadata = self.metadata_store.current()

        def belongs_to_current_context(event: PassageEvent) -> bool:
            return metadata is None or (
                event.race_id == metadata.race_id
                and event.stage_id == metadata.stage_id
            )

        try:
            active_events = tuple(
                event
                for event in pending_events
                if event.is_active and belongs_to_current_context(event)
            )
            if active_events and self._ring_buffer is not None:
                self._ring_buffer.scan()
            archive_segments = (
                self._publish_archive_segments() if active_events else ()
            )
            changed_event_ids = {event.event_id for event in pending_events}
            for event in pending_events:
                if event.is_active and belongs_to_current_context(event):
                    self._register_passage(event, scan=False)
                else:
                    self._discard_registered_passage(event.event_id)
            if archive_segments:
                changed_event_ids.update(
                    item.event_id
                    for item in self._events_for_current_metadata(
                        self.passage_store.events()
                    )
                )
            self.refresh_events(changed_event_ids)
            self._apply_pending_focus()
            self._capture_error = ""
        except Exception as exc:
            self._capture_error = sanitize_recording_message(exc)
            logger.exception("Failed to prepare passage review evidence")
        self._update_runtime_status()

    def _on_metadata_received(self, metadata: RaceMetadata) -> None:
        pending_focus = self._pending_focus
        if pending_focus is not None and (
            pending_focus.race_id != metadata.race_id
            or pending_focus.stage_id != metadata.stage_id
        ):
            self._pending_focus = None
        self._lookup_cache.clear()
        self._selected_event_id = ""
        for event_id in tuple(self._capture_windows):
            event = self.passage_store.get(event_id)
            if event is None or not event.is_active or (
                event.race_id != metadata.race_id
                or event.stage_id != metadata.stage_id
            ):
                self._discard_registered_passage(event_id)
        self.refresh()
        self._apply_pending_focus()
        self._update_runtime_status()

    def _on_focus_received(self, focus: RaceFocus) -> None:
        self._pending_focus = focus
        self._apply_pending_focus()

    def _apply_pending_focus(self) -> bool:
        focus = self._pending_focus
        if focus is None:
            return False
        applied = self.focus_athlete(
            focus.race_id,
            focus.stage_id,
            athlete_id=focus.athlete_id,
            bib=focus.bib,
            group_id=focus.group_id,
        )
        if applied:
            self._update_operator_controls()
        return applied

    def _current_archive_race_id(self) -> str:
        metadata = self.metadata_store.current()
        if metadata is not None:
            return metadata.race_id
        return race_id_from_passage_store(self.passage_store)

    def _refresh_capture_windows(self) -> None:
        coordinator = self._coordinator
        ring_buffer = self._ring_buffer
        recorder = self._recorder
        if coordinator is None or ring_buffer is None:
            self._update_runtime_status()
            return
        changed_event_ids: set[str] = set()
        try:
            # Keep indexing completed camera segments even when no passage is
            # waiting, so device health and retention remain current.
            ring_buffer.scan()
            if self._publish_archive_segments():
                changed_event_ids.update(
                    event.event_id
                    for event in self._events_for_current_metadata(
                        self.passage_store.events()
                    )
                )
            for window in coordinator.refresh(scan=False):
                previous = self._capture_windows.get(window.event_id)
                self._capture_windows[window.event_id] = window
                event = self.passage_store.get(window.event_id)
                if event is not None:
                    if self._publish_window(window, event):
                        changed_event_ids.add(window.event_id)
                if previous is not None and previous.state is not window.state:
                    changed_event_ids.add(window.event_id)
            now = time.monotonic()
            if now - self._last_cleanup_at >= 5.0:
                ring_buffer.cleanup(current_time_ms=int(time.time() * 1000.0))
                self._last_cleanup_at = now
            if recorder is not None:
                recorder_error = recorder.check_error()
                if recorder_error:
                    self._runtime_error = recorder_error
            if changed_event_ids:
                self.refresh_events(changed_event_ids)
        except Exception as exc:
            self._capture_error = sanitize_recording_message(exc)
            logger.exception("Failed to refresh passage review capture")
        self._update_operator_controls()
        self._update_runtime_status()

    def _publish_archive_segments(
        self,
        *,
        race_id: str | None = None,
        recording: bool | None = None,
    ):
        publishers = tuple(self._archive_publishers)
        if not publishers:
            return ()
        if race_id is None:
            try:
                race_id = self._current_archive_race_id()
            except ExternalClipImportError:
                return ()
        if recording is None:
            recorder = self._recorder
            recording = bool(recorder is not None and recorder.is_running)
        published = []
        for publisher in publishers:
            publisher_recording = bool(
                recording and publisher.recorder is self._recorder
            )
            published.extend(
                publisher.publish_completed(
                    race_id=str(race_id),
                    recording=publisher_recording,
                )
            )
        return tuple(published)

    def _update_runtime_status(self) -> None:
        beijing_now = datetime.now(timezone(timedelta(hours=8)))
        self.beijing_clock_label.setText(
            beijing_now.strftime("北京时间 %Y-%m-%d %H:%M:%S")
        )
        self.race_dir_label.setText(f"证据目录：{self.output_dir.name}")
        self.race_dir_label.setToolTip(str(self.output_dir))

        recorder = self._recorder
        recording_active = bool(recorder is not None and recorder.is_running)
        configured = is_supported_review_source(self.source)
        ring_buffer = self._ring_buffer
        segments = ring_buffer.segments() if ring_buffer is not None else ()
        if recording_active and segments:
            newest_age_ms = int(time.time() * 1000.0) - segments[-1].ended_at_ms
            if newest_age_ms <= 8_000:
                camera_text, camera_color = "录像设备: 已连接", "#247a52"
                camera_tooltip = "FFmpeg 正常运行且持续生成可判读画面"
            else:
                camera_text, camera_color = "录像设备: 无新画面", "#b54747"
                camera_tooltip = "录像进程仍在运行，但超过 8 秒没有新的完整画面"
        elif recording_active:
            camera_text, camera_color = "录像设备: 正在检查", "#a56300"
            camera_tooltip = "等待首个 2 秒判读片段完成"
        elif configured:
            camera_text, camera_color = "录像设备: 已配置", "#526170"
            camera_tooltip = "开始录像后验证设备与画面是否真正联通"
        else:
            camera_text, camera_color = "录像设备: 未配置", "#b54747"
            camera_tooltip = "请打开设备设置并选择USB/Type-C摄像头"
        self.camera_status_label.setText(camera_text)
        self.camera_status_label.setToolTip(camera_tooltip)
        self.camera_status_label.setStyleSheet(f"color: {camera_color};")

        if self._runtime_error and not recording_active:
            recording_text, recording_color = "普通录像: 异常", "#b54747"
            recording_tooltip = self._runtime_error
        elif recording_active:
            elapsed = max(0, int(time.monotonic() - self._recording_started_at))
            hours, remainder = divmod(elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)
            recording_text = f"普通录像: {hours:02d}:{minutes:02d}:{seconds:02d}"
            recording_color = "#247a52"
            recording_tooltip = "5 分钟赛事存档 + 2 秒判读时间片"
        else:
            recording_text, recording_color = "普通录像: 待机", "#667085"
            recording_tooltip = "点击开始录像后持续保存整场赛事"
        self.recording_status_label.setText(recording_text)
        self.recording_status_label.setToolTip(recording_tooltip)
        self.recording_status_label.setStyleSheet(f"color: {recording_color};")
        self.record_button.setText("停止录像" if recording_active else "开始录像")
        self.record_button.setStyleSheet(
            "background: #a33d4b; color: white; border: 1px solid #a33d4b;"
            if recording_active
            else "background: #247a52; color: white; border: 1px solid #247a52;"
        )

        receiver = self._receiver
        if receiver is not None and receiver.is_running:
            metadata = self.metadata_store.current()
            pending_count = len(self._pending_passages)
            if pending_count:
                self.receiver_status_label.setText(
                    "CycleRace: 正在补同步，"
                    f"已接收 {self._received_passage_count}，待处理 {pending_count}"
                )
                self.receiver_status_label.setStyleSheet("color: #a56300;")
                self.receiver_status_label.setToolTip(
                    "通过记录已先写入本地审计日志，正在合并刷新录像定位和判读列表"
                )
            elif self._received_passage_count:
                self.receiver_status_label.setText(
                    f"CycleRace: 本次已接收 {self._received_passage_count}"
                )
                self.receiver_status_label.setStyleSheet("color: #247a52;")
                self.receiver_status_label.setToolTip(
                    "已收到CycleRace通过记录；当前协议没有持续心跳，不虚报长期在线"
                )
            elif metadata is not None:
                race_label = metadata.race_name.strip() or metadata.race_id
                stage_label = metadata.stage_name.strip() or metadata.stage_id
                self.receiver_status_label.setText(
                    f"CycleRace: 已同步 {race_label} / {stage_label}，等待通过"
                )
                self.receiver_status_label.setStyleSheet("color: #247a52;")
                self.receiver_status_label.setToolTip(
                    f"已读取 {len(metadata.groups)} 个组别、"
                    f"{len(metadata.athletes)} 名运动员；等待真实通过时间"
                )
            elif self._historical_passage_count:
                self.receiver_status_label.setText(
                    "CycleRace: 等待本次数据，"
                    f"已加载历史 {self._historical_passage_count} 条"
                )
                self.receiver_status_label.setStyleSheet("color: #a56300;")
                self.receiver_status_label.setToolTip(
                    "历史记录已加载，但本次运行还没有收到CycleRace新数据"
                )
            else:
                self.receiver_status_label.setText("CycleRace: 等待数据")
                self.receiver_status_label.setStyleSheet("color: #a56300;")
                self.receiver_status_label.setToolTip(
                    "接收服务已启动，收到首条通过记录后确认数据链路"
                )
        else:
            self.receiver_status_label.setText(
                "CycleRace: 异常" if self._receiver_error else "CycleRace: 未监听"
            )
            self.receiver_status_label.setStyleSheet("color: #b54747;")
            self.receiver_status_label.setToolTip(
                self._receiver_error or "CycleRace接收服务未启动"
            )

        high_speed_count = sum(
            segment.clock_source == EXTERNAL_CLOCK_SOURCE
            for segment in self.timeline_store.segments()
        )
        if high_speed_count:
            self.high_speed_status_label.setText(
                f"高速摄像: 已导入 {high_speed_count} 段"
            )
            self.high_speed_status_label.setStyleSheet("color: #247a52;")
        else:
            self.high_speed_status_label.setText("高速摄像: 等待导入")
            self.high_speed_status_label.setStyleSheet("color: #a56300;")
        self.high_speed_status_label.setToolTip(
            "无厂商硬件接口时只报告北京时间sidecar导入状态"
        )

        try:
            free_gb = shutil.disk_usage(self.output_dir).free / (1024**3)
            storage_color = "#b54747" if free_gb < 5 else "#a56300" if free_gb < 20 else "#247a52"
            self.storage_status_label.setText(f"存储: {free_gb:.1f} GB")
            self.storage_status_label.setStyleSheet(f"color: {storage_color};")
            self.storage_status_label.setToolTip(str(self.output_dir))
        except OSError as exc:
            self.storage_status_label.setText("存储: 不可用")
            self.storage_status_label.setStyleSheet("color: #b54747;")
            self.storage_status_label.setToolTip(str(exc))

        counts = {state: 0 for state in PassageReviewState}
        for window in self._capture_windows.values():
            counts[window.state] += 1
        if self._capture_error:
            self.capture_status_label.setText(f"证据处理异常：{self._capture_error}")
            self.capture_status_label.setToolTip(self._capture_error)
            self.capture_status_label.setStyleSheet(
                "color: #b54747; font-weight: 700;"
            )
        else:
            self.capture_status_label.setText(
                f"本次待封口 {counts[PassageReviewState.WAITING]}  |  "
                f"本次可核对 {counts[PassageReviewState.READY]}  |  "
                f"本次缺口 {counts[PassageReviewState.PARTIAL]}  |  "
                f"已有证据 {self._available_evidence_count}  |  "
                f"缺少绝对时间 {len(self._unsupported_event_ids)}"
            )
            self.capture_status_label.setToolTip("")
            self.capture_status_label.setStyleSheet(
                "color: #667085; font-weight: 500;"
            )
        self.import_high_speed_button.setEnabled(bool(self.passage_store.events()))
        self._update_operator_controls()

    def stop_recording(self) -> None:
        recorder = self._recorder
        if recorder is not None:
            try:
                recorder.stop()
            except RecordingError as exc:
                self._runtime_error = sanitize_recording_message(exc)
            try:
                self._publish_archive_segments(recording=False)
                self._refresh_capture_windows()
            except Exception:
                logger.exception("Failed to publish final review segments")
        self._recorder = None
        self._ring_buffer = None
        self._coordinator = None
        self._publisher = None
        self._started = False
        self._recording_started_at = 0.0
        self._update_runtime_status()

    def stop_receiver(self) -> None:
        receiver = self._receiver
        self._receiver = None
        if receiver is not None:
            try:
                receiver.stop()
            except Exception as exc:  # noqa: BLE001 - receiver factories may vary.
                logger.warning("Failed to stop CycleRace receiver: %s", exc)
        self._update_runtime_status()

    def stop(self) -> None:
        self._refresh_timer.stop()
        self._passage_batch_timer.stop()
        self._pending_passages.clear()
        self.stop_recording()
        self.stop_receiver()
        self._update_runtime_status()

    def closeEvent(self, event) -> None:
        self._clock_timer.stop()
        self.stop()
        super().closeEvent(event)


__all__ = [
    "FinishReviewLaunchDialog",
    "FinishReviewSettings",
    "FinishReviewWindow",
]
