import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import QEvent, QObject, QPoint, QPointF, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QMouseEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

import realtime.passage_review as passage_review
from realtime.passage_evidence import (
    HIGH_SPEED_SOURCE,
    REGULAR_SOURCE,
    PassageEvidenceAssociationStore,
)
from realtime.passage_receiver import PassageEvent, PassageEventStore
from realtime.passage_review import PassageReviewDialog, lookup_status_text
from realtime.video_timeline import VideoTimelineStore


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class _FakePlaybackWorker(QObject):
    metadata_ready = pyqtSignal(int, float, int, int, int)
    frame_ready = pyqtSignal(object, int, int)
    full_resolution_ready = pyqtSignal(object, int, int)
    playback_finished = pyqtSignal()
    playback_error = pyqtSignal(str)
    finished = pyqtSignal()
    instances = []

    def __init__(self, video_path, parent=None):
        super().__init__(parent)
        self.video_path = Path(video_path)
        self.running = False
        self.stopped = False
        self.seek_calls = []
        self.wait_calls = []
        self.speed_calls = []
        self.step_calls = []
        self.full_resolution_calls = []
        type(self).instances.append(self)

    def start(self):
        self.running = True
        fps = 250.0 if "high_speed" in self.video_path.name else 50.0
        self.metadata_ready.emit(60_000, fps, 2560, 1440, int(60 * fps))

    def isRunning(self):
        return self.running

    def pause(self):
        return None

    def seek(self, position_ms):
        self.seek_calls.append(int(position_ms))

    def set_shuttle_speed(self, speed):
        self.speed_calls.append(float(speed))

    def step(self, frame_delta):
        self.step_calls.append(int(frame_delta))

    def request_full_resolution(self, frame_index=None):
        self.full_resolution_calls.append(frame_index)

    def stop(self):
        self.stopped = True
        if self.running:
            self.running = False
            self.finished.emit()

    def wait(self, timeout_ms):
        self.wait_calls.append(int(timeout_ms))
        return True


@pytest.fixture
def fake_playback(monkeypatch):
    _FakePlaybackWorker.instances.clear()
    monkeypatch.setattr(passage_review, "VideoPlaybackWorker", _FakePlaybackWorker)
    return _FakePlaybackWorker


def _event(
    passage_time_ms=15_000,
    passage_timestamp_ms=None,
    *,
    event_id="passage-1",
    sequence=1,
    group_id="men-open",
    bib="23",
    chip_id="chip-23",
):
    return PassageEvent(
        event_id=event_id,
        race_id="race-1",
        stage_id="stage-1",
        group_id=group_id,
        sequence=sequence,
        chip_id=chip_id,
        bib=bib,
        passage_time_ms=passage_time_ms,
        lap=2,
        emitted_at_ms=passage_time_ms + 100,
        passage_timestamp_ms=passage_timestamp_ms,
    )


def _add_segment(
    timeline_store,
    video_path,
    *,
    source_id,
    camera_index,
    started_at_ms,
    ended_at_ms,
    clock_source="videopipe_system_clock",
    timing_error_ms=1_000,
    race_id="race-1",
):
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"video")
    segment = timeline_store.start_segment(
        source_id=source_id,
        camera_index=camera_index,
        video_path=video_path,
        started_at_ms=started_at_ms,
        clock_source=clock_source,
        timing_error_ms=timing_error_ms,
        race_id=race_id,
    )
    timeline_store.finish_segment(
        segment.segment_id,
        ended_at_ms=ended_at_ms,
        media_duration_ms=ended_at_ms - started_at_ms,
        media_started_at_ms=started_at_ms,
    )
    return segment


def test_review_uses_one_row_per_passage_and_opens_regular_video(
    qapp,
    tmp_path,
    fake_playback,
):
    passage_store = PassageEventStore(tmp_path / "passages.jsonl")
    passage_store.append(_event(passage_time_ms=5_000, passage_timestamp_ms=15_000))
    timeline_store = VideoTimelineStore(tmp_path / "video_timeline.jsonl")
    video_path = tmp_path / "videos" / "camera_01.mkv"
    _add_segment(
        timeline_store,
        video_path,
        source_id="camera_01",
        camera_index=1,
        started_at_ms=10_000,
        ended_at_ms=20_000,
    )
    opened = []

    dialog = PassageReviewDialog(
        passage_store,
        timeline_store,
        clock_offset_ms=500,
        pre_roll_ms=3_000,
        open_location=lambda event, location: opened.append((event, location)),
    )
    qapp.processEvents()

    assert dialog.table.rowCount() == 1
    assert dialog.table.item(0, 5).text() == "已定位"
    assert dialog.table.item(0, 6).text() == "未匹配"
    assert dialog.table.item(0, 7).text() == "缺少高速"
    assert dialog.regular_pane.location.video_path == video_path.absolute()
    assert dialog.high_speed_pane.location is None
    assert fake_playback.instances[0].seek_calls == [5_500]

    dialog.regular_pane.open_btn.click()
    assert opened[0][0].event_id == "passage-1"
    assert opened[0][1].passage_position_ms == 5_500
    assert opened[0][1].playback_position_ms == 2_500
    dialog.close()


def test_review_displays_absolute_passage_in_beijing_time(qapp, tmp_path):
    passage_store = PassageEventStore(tmp_path / "passages.jsonl")
    passage_store.append(
        _event(
            passage_time_ms=48_179_215,
            passage_timestamp_ms=1_786_252_979_215,
        )
    )
    timeline_store = VideoTimelineStore(tmp_path / "video_timeline.jsonl")

    dialog = PassageReviewDialog(passage_store, timeline_store)
    qapp.processEvents()

    assert dialog.table.item(0, 4).text() == "13:22:59.215"
    dialog.close()


def test_review_marks_legacy_video_as_unverified(qapp, tmp_path, fake_playback):
    passage_store = PassageEventStore(tmp_path / "passages.jsonl")
    passage_store.append(_event())
    timeline_store = VideoTimelineStore(tmp_path / "video_timeline.jsonl")
    video_path = tmp_path / "videos" / "legacy.mkv"
    video_path.parent.mkdir()
    video_path.write_bytes(b"video")
    segment = timeline_store.start_segment(
        source_id="camera_01",
        camera_index=1,
        video_path=video_path,
        started_at_ms=10_000,
    )
    timeline_store.finish_segment(segment.segment_id, ended_at_ms=20_000)

    dialog = PassageReviewDialog(passage_store, timeline_store)
    qapp.processEvents()

    assert dialog.table.item(0, 5).text() == "范围未验证"
    assert dialog.table.item(0, 7).text() == "缺少高速"
    assert dialog.regular_pane.open_btn.isEnabled()
    dialog.close()


def test_review_shows_missing_evidence_without_starting_workers(qapp, tmp_path):
    passage_store = PassageEventStore(tmp_path / "passages.jsonl")
    passage_store.append(_event())
    timeline_store = VideoTimelineStore(tmp_path / "video_timeline.jsonl")

    dialog = PassageReviewDialog(passage_store, timeline_store)
    qapp.processEvents()

    assert dialog.table.item(0, 5).text() == "未匹配"
    assert dialog.table.item(0, 6).text() == "未匹配"
    assert dialog.table.item(0, 7).text() == "无可用证据"
    assert dialog.regular_pane._worker is None
    assert dialog.high_speed_pane._worker is None
    dialog.close()


def test_review_shows_high_speed_boundary_independently(
    qapp,
    tmp_path,
    fake_playback,
):
    passage_store = PassageEventStore(tmp_path / "passages.jsonl")
    passage_store.append(_event(passage_time_ms=9_950))
    timeline_store = VideoTimelineStore(tmp_path / "video_timeline.jsonl")
    video_path = tmp_path / "videos" / "high_speed_01.mp4"
    _add_segment(
        timeline_store,
        video_path,
        source_id="high_speed_01",
        camera_index=1,
        started_at_ms=10_000,
        ended_at_ms=20_000,
        clock_source="external_test_clock",
        timing_error_ms=100,
    )

    dialog = PassageReviewDialog(passage_store, timeline_store)
    qapp.processEvents()

    assert dialog.table.rowCount() == 1
    assert dialog.table.item(0, 5).text() == "未匹配"
    assert dialog.table.item(0, 6).text() == "边界候选"
    assert dialog.table.item(0, 7).text() == "缺少普通录像"
    assert dialog.high_speed_pane.location.status == "near_boundary"
    assert dialog.high_speed_pane.location.playback_position_ms == 0
    dialog.close()


def test_review_shows_regular_and_high_speed_sources_on_one_row(
    qapp,
    tmp_path,
    fake_playback,
):
    passage_store = PassageEventStore(tmp_path / "passages.jsonl")
    passage_store.append(_event(passage_time_ms=20_050))
    timeline_store = VideoTimelineStore(tmp_path / "video_timeline.jsonl")
    videos_dir = tmp_path / "videos"
    standard_path = videos_dir / "camera_01.mkv"
    _add_segment(
        timeline_store,
        standard_path,
        source_id="camera_01",
        camera_index=1,
        started_at_ms=10_000,
        ended_at_ms=30_000,
    )
    high_speed_path = videos_dir / "high_speed_02.mp4"
    _add_segment(
        timeline_store,
        high_speed_path,
        source_id="high_speed_02",
        camera_index=2,
        started_at_ms=19_000,
        ended_at_ms=20_000,
        clock_source="external_clip_sidecar_beijing",
        timing_error_ms=100,
    )
    opened = []

    dialog = PassageReviewDialog(
        passage_store,
        timeline_store,
        open_location=lambda event, location: opened.append((event, location)),
    )
    qapp.processEvents()

    assert dialog.table.rowCount() == 1
    assert dialog.table.item(0, 5).text() == "已定位"
    assert dialog.table.item(0, 6).text() == "边界候选"
    assert dialog.table.item(0, 7).text() == "需核对"
    assert dialog.regular_pane.location.segment.source_id == "camera_01"
    assert dialog.high_speed_pane.location.segment.source_id == "high_speed_02"
    assert "另有 1 个机位位于误差边界" in lookup_status_text(
        dialog._lookup(passage_store.get("passage-1"))
    )

    dialog.high_speed_pane.open_btn.click()
    dialog.regular_pane.open_btn.click()
    assert [location.segment.source_id for _event, location in opened] == [
        "high_speed_02",
        "camera_01",
    ]
    dialog.close()


def test_identity_search_selects_bib_15_without_recomputing_lookups(
    qapp,
    tmp_path,
    fake_playback,
    monkeypatch,
):
    passage_store = PassageEventStore(tmp_path / "passages.jsonl")
    passage_store.append(_event(event_id="passage-9", sequence=1, bib="9"))
    passage_store.append(
        _event(event_id="passage-15", sequence=2, bib="15", chip_id="chip-15")
    )
    timeline_store = VideoTimelineStore(tmp_path / "video_timeline.jsonl")
    _add_segment(
        timeline_store,
        tmp_path / "videos" / "camera_01.mkv",
        source_id="camera_01",
        camera_index=1,
        started_at_ms=10_000,
        ended_at_ms=20_000,
    )
    locate_calls = []
    original_locate = timeline_store.locate_passage

    def locate_passage(*args, **kwargs):
        locate_calls.append((args, kwargs))
        return original_locate(*args, **kwargs)

    monkeypatch.setattr(timeline_store, "locate_passage", locate_passage)
    dialog = PassageReviewDialog(passage_store, timeline_store)
    qapp.processEvents()
    assert len(locate_calls) == 2

    dialog.identity_search.setText("15")
    dialog._find_identity()
    qapp.processEvents()

    assert dialog.table.currentRow() == 1
    assert dialog._selected_event_id == "passage-15"
    assert dialog.selected_identity_value.text() == "15"
    assert len(locate_calls) == 2
    dialog.refresh()
    assert dialog.table.currentRow() == 1
    assert dialog._selected_event_id == "passage-15"
    assert dialog.selected_identity_value.text() == "15"
    assert dialog.identity_search.text() == "15"
    assert len(locate_calls) == 2
    dialog.close()


def test_switching_video_files_does_not_wait_on_ui_thread(
    qapp,
    tmp_path,
    fake_playback,
):
    passage_store = PassageEventStore(tmp_path / "passages.jsonl")
    passage_store.append(
        _event(event_id="passage-1", sequence=1, passage_time_ms=15_000)
    )
    passage_store.append(
        _event(event_id="passage-2", sequence=2, passage_time_ms=35_000, bib="15")
    )
    timeline_store = VideoTimelineStore(tmp_path / "video_timeline.jsonl")
    _add_segment(
        timeline_store,
        tmp_path / "videos" / "camera_01_first.mkv",
        source_id="camera_01_first",
        camera_index=1,
        started_at_ms=10_000,
        ended_at_ms=20_000,
    )
    _add_segment(
        timeline_store,
        tmp_path / "videos" / "camera_01_second.mkv",
        source_id="camera_01_second",
        camera_index=1,
        started_at_ms=30_000,
        ended_at_ms=40_000,
    )

    dialog = PassageReviewDialog(passage_store, timeline_store)
    qapp.processEvents()
    first_worker = fake_playback.instances[0]

    dialog.table.selectRow(1)
    qapp.processEvents()

    assert first_worker.stopped
    assert first_worker.wait_calls == []
    assert len(fake_playback.instances) == 2
    assert fake_playback.instances[1].video_path.name == "camera_01_second.mkv"
    dialog.close()


def test_review_rejects_external_clip_from_another_race(qapp, tmp_path):
    passage_store = PassageEventStore(tmp_path / "passages.jsonl")
    passage_store.append(_event())
    timeline_store = VideoTimelineStore(tmp_path / "video_timeline.jsonl")
    _add_segment(
        timeline_store,
        tmp_path / "videos" / "other-race.mkv",
        source_id="high_speed_01",
        camera_index=1,
        started_at_ms=10_000,
        ended_at_ms=20_000,
        clock_source="external_test_clock",
        race_id="race-2",
    )

    dialog = PassageReviewDialog(passage_store, timeline_store)
    qapp.processEvents()

    assert dialog.table.item(0, 5).text() == "未匹配"
    assert dialog.table.item(0, 6).text() == "未匹配"
    assert dialog.table.item(0, 7).text() == "录像属于其他赛事"
    dialog.close()


def test_linked_frame_step_keeps_regular_and_high_speed_on_one_time_cursor(
    qapp,
    tmp_path,
    fake_playback,
):
    passage_store = PassageEventStore(tmp_path / "passages.jsonl")
    passage_store.append(_event(passage_time_ms=20_050))
    timeline_store = VideoTimelineStore(tmp_path / "video_timeline.jsonl")
    _add_segment(
        timeline_store,
        tmp_path / "videos" / "camera_01.mkv",
        source_id="camera_01",
        camera_index=1,
        started_at_ms=10_000,
        ended_at_ms=30_000,
    )
    _add_segment(
        timeline_store,
        tmp_path / "videos" / "high_speed_01.mp4",
        source_id="high_speed_01",
        camera_index=2,
        started_at_ms=19_000,
        ended_at_ms=22_000,
        clock_source="external_clip_sidecar_beijing",
        timing_error_ms=100,
    )

    dialog = PassageReviewDialog(passage_store, timeline_store)
    dialog.show()
    qapp.processEvents()
    regular_worker, high_speed_worker = fake_playback.instances

    QTest.keyClick(dialog.regular_pane.video_view, Qt.Key_Right)
    qapp.processEvents()

    assert dialog._shared_delta_ms == 4
    assert regular_worker.seek_calls[-1] == 10_054
    assert high_speed_worker.seek_calls[-1] == 1_054
    assert "Δ+4 ms" in dialog.current_time_label.text()
    dialog.close()


def test_zoom_requests_full_resolution_without_replacing_the_worker(
    qapp,
    tmp_path,
    fake_playback,
):
    passage_store = PassageEventStore(tmp_path / "passages.jsonl")
    passage_store.append(_event(passage_time_ms=15_000))
    timeline_store = VideoTimelineStore(tmp_path / "video_timeline.jsonl")
    _add_segment(
        timeline_store,
        tmp_path / "videos" / "camera_01.mkv",
        source_id="camera_01",
        camera_index=1,
        started_at_ms=10_000,
        ended_at_ms=20_000,
    )
    dialog = PassageReviewDialog(passage_store, timeline_store)
    qapp.processEvents()
    worker = fake_playback.instances[0]
    preview = QImage(1280, 720, QImage.Format_RGB888)
    preview.fill(0)

    worker.frame_ready.emit(preview, 5_000, 250)
    qapp.processEvents()
    dialog.regular_pane.video_view.set_actual_size()
    qapp.processEvents()

    assert fake_playback.instances == [worker]
    assert worker.full_resolution_calls[-1] == 250
    assert dialog.regular_pane.video_view.zoom_percent == 100
    assert dialog.regular_pane.video_view.sceneRect().width() == 2560
    dialog.close()


def test_manual_marker_uses_enter_while_space_keeps_linked_playback(
    qapp,
    tmp_path,
    fake_playback,
):
    passage_store = PassageEventStore(tmp_path / "passages.jsonl")
    passage_store.append(_event(passage_time_ms=20_050, bib="15", chip_id="chip-15"))
    timeline_store = VideoTimelineStore(tmp_path / "video_timeline.jsonl")
    _add_segment(
        timeline_store,
        tmp_path / "videos" / "camera_01.mkv",
        source_id="camera_01",
        camera_index=1,
        started_at_ms=10_000,
        ended_at_ms=30_000,
    )
    _add_segment(
        timeline_store,
        tmp_path / "videos" / "high_speed_01.mp4",
        source_id="high_speed_01",
        camera_index=2,
        started_at_ms=19_000,
        ended_at_ms=22_000,
        clock_source="external_clip_sidecar_beijing",
        timing_error_ms=100,
    )
    association_store = PassageEvidenceAssociationStore(
        tmp_path / "passage_evidence_associations.jsonl"
    )
    dialog = PassageReviewDialog(
        passage_store,
        timeline_store,
        association_store=association_store,
    )
    dialog.show()
    qapp.processEvents()
    regular_worker, high_speed_worker = fake_playback.instances
    frame = QImage(1280, 720, QImage.Format_RGB888)
    frame.fill(0)
    regular_worker.frame_ready.emit(frame, 10_050, 502)
    high_speed_worker.frame_ready.emit(frame, 1_050, 262)
    qapp.processEvents()

    regular_view = dialog.regular_pane.video_view
    QTest.mouseClick(
        regular_view.viewport(),
        Qt.LeftButton,
        pos=regular_view.viewport().rect().center(),
    )
    assert dialog.regular_pane.has_pending_marker
    pending_marker = regular_view._marker
    assert pending_marker is not None
    assert pending_marker[2:] == ("15", False)

    QTest.keyClick(dialog.regular_pane.video_view, Qt.Key_Return)
    qapp.processEvents()

    regular_association = association_store.get("passage-1", REGULAR_SOURCE)
    assert regular_association is not None
    assert regular_association.frame_index == 502
    assert regular_association.position_ms == 10_050
    assert regular_association.marker_x_normalized == pytest.approx(pending_marker[0])
    assert regular_association.marker_y_normalized == pytest.approx(pending_marker[1])
    assert dialog.table.item(0, 7).text() == "录像确认"

    QTest.keyClick(dialog.regular_pane.video_view, Qt.Key_Space)
    qapp.processEvents()
    assert dialog._sync_playing
    QTest.keyClick(dialog.regular_pane.video_view, Qt.Key_Space)
    qapp.processEvents()
    assert not dialog._sync_playing
    assert association_store.get("passage-1", HIGH_SPEED_SOURCE) is None
    dialog.close()


def test_left_drag_pans_image_without_placing_marker_while_click_places_marker(
    qapp,
    tmp_path,
    fake_playback,
):
    passage_store = PassageEventStore(tmp_path / "passages.jsonl")
    passage_store.append(_event(passage_time_ms=15_000, bib="12"))
    timeline_store = VideoTimelineStore(tmp_path / "video_timeline.jsonl")
    _add_segment(
        timeline_store,
        tmp_path / "videos" / "camera_01.mkv",
        source_id="camera_01",
        camera_index=1,
        started_at_ms=10_000,
        ended_at_ms=20_000,
    )
    dialog = PassageReviewDialog(passage_store, timeline_store)
    dialog.show()
    qapp.processEvents()
    worker = fake_playback.instances[0]
    frame = QImage(1280, 720, QImage.Format_RGB888)
    frame.fill(0)
    worker.frame_ready.emit(frame, 5_000, 250)
    qapp.processEvents()

    view = dialog.regular_pane.video_view
    view.set_actual_size()
    qapp.processEvents()
    horizontal_scrollbar = view.horizontalScrollBar()
    vertical_scrollbar = view.verticalScrollBar()
    before_pan = (horizontal_scrollbar.value(), vertical_scrollbar.value())
    center = view.viewport().rect().center()
    drag_target = center + QPoint(60, 40)

    QTest.mousePress(view.viewport(), Qt.LeftButton, pos=center)
    move_event = QMouseEvent(
        QEvent.MouseMove,
        QPointF(drag_target),
        QPointF(view.viewport().mapToGlobal(drag_target)),
        Qt.NoButton,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    QApplication.sendEvent(view.viewport(), move_event)
    QTest.mouseRelease(view.viewport(), Qt.LeftButton, pos=drag_target)
    qapp.processEvents()

    assert (horizontal_scrollbar.value(), vertical_scrollbar.value()) != before_pan
    assert not dialog.regular_pane.has_pending_marker
    assert view._marker is None

    QTest.mouseClick(view.viewport(), Qt.LeftButton, pos=center)
    assert dialog.regular_pane.has_pending_marker
    assert view._marker is not None
    dialog.close()


def test_manual_marker_restores_and_upgrades_to_dual_source_confirmation(
    qapp,
    tmp_path,
    fake_playback,
):
    passage_store = PassageEventStore(tmp_path / "passages.jsonl")
    passage_store.append(_event(passage_time_ms=20_050, bib="15", chip_id="chip-15"))
    timeline_store = VideoTimelineStore(tmp_path / "video_timeline.jsonl")
    regular_segment = _add_segment(
        timeline_store,
        tmp_path / "videos" / "camera_01.mkv",
        source_id="camera_01",
        camera_index=1,
        started_at_ms=10_000,
        ended_at_ms=30_000,
    )
    high_speed_segment = _add_segment(
        timeline_store,
        tmp_path / "videos" / "high_speed_01.mp4",
        source_id="high_speed_01",
        camera_index=2,
        started_at_ms=19_000,
        ended_at_ms=22_000,
        clock_source="external_clip_sidecar_beijing",
        timing_error_ms=100,
    )
    association_store = PassageEvidenceAssociationStore(
        tmp_path / "passage_evidence_associations.jsonl"
    )
    association_store.confirm(
        passage_event_id="passage-1",
        bib="15",
        confirmed_source=REGULAR_SOURCE,
        segment_id=regular_segment.segment_id,
        frame_index=502,
        position_ms=10_050,
        marker_x_normalized=0.25,
        marker_y_normalized=0.5,
        confirmed_at_ms=1_000,
    )

    dialog = PassageReviewDialog(
        passage_store,
        timeline_store,
        association_store=PassageEvidenceAssociationStore(
            association_store.journal_path
        ),
    )
    qapp.processEvents()
    regular_worker, high_speed_worker = fake_playback.instances
    assert regular_worker.seek_calls[-1] == 10_050
    assert high_speed_worker.seek_calls[-1] == 1_050
    assert dialog.regular_pane.association is not None
    assert dialog.table.item(0, 7).text() == "录像确认"

    frame = QImage(1280, 720, QImage.Format_RGB888)
    frame.fill(0)
    regular_worker.frame_ready.emit(frame, 10_050, 502)
    high_speed_worker.frame_ready.emit(frame, 1_050, 262)
    qapp.processEvents()
    assert dialog.regular_pane.video_view._marker == (0.25, 0.5, "15", True)

    high_speed_view = dialog.high_speed_pane.video_view
    high_speed_view.set_actual_size()
    high_speed_view.zoom_by(1.2)
    zoom_before_confirmation = high_speed_view.zoom_percent
    QTest.mouseClick(
        high_speed_view.viewport(),
        Qt.LeftButton,
        pos=high_speed_view.viewport().rect().center(),
    )
    QTest.keyClick(dialog.high_speed_pane.video_view, Qt.Key_Return)
    qapp.processEvents()

    high_speed_association = dialog.association_store.get(
        "passage-1", HIGH_SPEED_SOURCE
    )
    assert high_speed_association is not None
    assert high_speed_association.segment_id == high_speed_segment.segment_id
    assert high_speed_view.zoom_percent == zoom_before_confirmation
    assert dialog.table.item(0, 7).text() == "双源确认"
    assert dialog.source_value.text() == "双源确认"
    dialog.close()


def test_escape_cancels_pending_marker_and_delete_clears_confirmed_marker(
    qapp,
    tmp_path,
    fake_playback,
    monkeypatch,
):
    passage_store = PassageEventStore(tmp_path / "passages.jsonl")
    passage_store.append(_event(passage_time_ms=15_000, bib="15"))
    timeline_store = VideoTimelineStore(tmp_path / "video_timeline.jsonl")
    segment = _add_segment(
        timeline_store,
        tmp_path / "videos" / "camera_01.mkv",
        source_id="camera_01",
        camera_index=1,
        started_at_ms=10_000,
        ended_at_ms=20_000,
    )
    association_store = PassageEvidenceAssociationStore(
        tmp_path / "passage_evidence_associations.jsonl"
    )
    association_store.confirm(
        passage_event_id="passage-1",
        bib="15",
        confirmed_source=REGULAR_SOURCE,
        segment_id=segment.segment_id,
        frame_index=250,
        position_ms=5_000,
        marker_x_normalized=0.4,
        marker_y_normalized=0.6,
        confirmed_at_ms=1_000,
    )
    dialog = PassageReviewDialog(
        passage_store,
        timeline_store,
        association_store=association_store,
    )
    qapp.processEvents()
    worker = fake_playback.instances[0]
    frame = QImage(1280, 720, QImage.Format_RGB888)
    frame.fill(0)
    worker.frame_ready.emit(frame, 5_000, 250)
    qapp.processEvents()

    regular_view = dialog.regular_pane.video_view
    QTest.mouseClick(
        regular_view.viewport(),
        Qt.LeftButton,
        pos=regular_view.viewport().rect().center(),
    )
    QTest.keyClick(dialog.regular_pane.video_view, Qt.Key_Escape)
    assert not dialog.regular_pane.has_pending_marker
    assert dialog.regular_pane.video_view._marker == (0.4, 0.6, "15", True)

    monkeypatch.setattr(
        passage_review.QMessageBox,
        "question",
        lambda *args, **kwargs: passage_review.QMessageBox.Yes,
    )
    QTest.keyClick(dialog.regular_pane.video_view, Qt.Key_Delete)
    qapp.processEvents()
    assert association_store.get("passage-1", REGULAR_SOURCE) is None
    assert dialog.table.item(0, 7).text() == "缺少高速"
    dialog.close()


def test_enter_stays_by_default_and_opt_in_auto_advance_moves_to_next_passage(
    qapp,
    tmp_path,
    fake_playback,
):
    passage_store = PassageEventStore(tmp_path / "passages.jsonl")
    passage_store.append(
        _event(event_id="passage-12", sequence=1, passage_time_ms=15_000, bib="12")
    )
    passage_store.append(
        _event(event_id="passage-15", sequence=2, passage_time_ms=16_000, bib="15")
    )
    timeline_store = VideoTimelineStore(tmp_path / "video_timeline.jsonl")
    _add_segment(
        timeline_store,
        tmp_path / "videos" / "camera_01.mkv",
        source_id="camera_01",
        camera_index=1,
        started_at_ms=10_000,
        ended_at_ms=20_000,
    )
    association_store = PassageEvidenceAssociationStore(
        tmp_path / "passage_evidence_associations.jsonl"
    )
    dialog = PassageReviewDialog(
        passage_store,
        timeline_store,
        association_store=association_store,
    )
    dialog.show()
    qapp.processEvents()
    worker = fake_playback.instances[0]
    frame = QImage(1280, 720, QImage.Format_RGB888)
    frame.fill(0)
    worker.frame_ready.emit(frame, 5_000, 250)
    qapp.processEvents()

    view = dialog.regular_pane.video_view
    QTest.mouseClick(
        view.viewport(),
        Qt.LeftButton,
        pos=view.viewport().rect().center(),
    )
    QTest.keyClick(view, Qt.Key_Return)
    qapp.processEvents()

    assert not dialog.auto_advance_checkbox.isChecked()
    assert association_store.get("passage-12", REGULAR_SOURCE) is not None
    assert dialog.table.item(0, 7).text() == "录像确认"
    assert dialog.table.currentRow() == 0
    assert dialog._selected_event_id == "passage-12"

    worker.frame_ready.emit(frame, 5_000, 250)
    qapp.processEvents()
    dialog.auto_advance_checkbox.setChecked(True)
    QTest.mouseClick(
        view.viewport(),
        Qt.LeftButton,
        pos=view.viewport().rect().center(),
    )
    QTest.keyClick(view, Qt.Key_Return)
    qapp.processEvents()

    assert dialog.table.currentRow() == 1
    assert dialog._selected_event_id == "passage-15"
    assert dialog.selected_identity_value.text() == "15"
    assert dialog.identity_search.text() == "15"

    QTest.keyClick(view, Qt.Key_Up)
    qapp.processEvents()
    assert dialog.table.currentRow() == 0
    assert dialog._selected_event_id == "passage-12"
    dialog.close()


def test_opt_in_auto_advance_waits_for_all_available_sources(
    qapp,
    tmp_path,
    fake_playback,
):
    passage_store = PassageEventStore(tmp_path / "passages.jsonl")
    passage_store.append(
        _event(event_id="passage-12", sequence=1, passage_time_ms=15_000, bib="12")
    )
    passage_store.append(
        _event(event_id="passage-15", sequence=2, passage_time_ms=16_000, bib="15")
    )
    timeline_store = VideoTimelineStore(tmp_path / "video_timeline.jsonl")
    _add_segment(
        timeline_store,
        tmp_path / "videos" / "camera_01.mkv",
        source_id="camera_01",
        camera_index=1,
        started_at_ms=10_000,
        ended_at_ms=20_000,
    )
    _add_segment(
        timeline_store,
        tmp_path / "videos" / "high_speed_01.mp4",
        source_id="high_speed_01",
        camera_index=2,
        started_at_ms=14_000,
        ended_at_ms=17_000,
        clock_source="external_clip_sidecar_beijing",
        timing_error_ms=100,
    )
    association_store = PassageEvidenceAssociationStore(
        tmp_path / "passage_evidence_associations.jsonl"
    )
    dialog = PassageReviewDialog(
        passage_store,
        timeline_store,
        association_store=association_store,
    )
    dialog.show()
    dialog.auto_advance_checkbox.setChecked(True)
    qapp.processEvents()
    regular_worker, high_speed_worker = fake_playback.instances
    frame = QImage(1280, 720, QImage.Format_RGB888)
    frame.fill(0)
    regular_worker.frame_ready.emit(frame, 5_000, 250)
    high_speed_worker.frame_ready.emit(frame, 1_000, 250)
    qapp.processEvents()

    regular_view = dialog.regular_pane.video_view
    QTest.mouseClick(
        regular_view.viewport(),
        Qt.LeftButton,
        pos=regular_view.viewport().rect().center(),
    )
    QTest.keyClick(regular_view, Qt.Key_Return)
    qapp.processEvents()

    assert association_store.get("passage-12", REGULAR_SOURCE) is not None
    assert association_store.get("passage-12", HIGH_SPEED_SOURCE) is None
    assert dialog.table.currentRow() == 0
    assert dialog._selected_event_id == "passage-12"

    high_speed_worker.frame_ready.emit(frame, 1_000, 250)
    qapp.processEvents()
    high_speed_view = dialog.high_speed_pane.video_view
    QTest.mouseClick(
        high_speed_view.viewport(),
        Qt.LeftButton,
        pos=high_speed_view.viewport().rect().center(),
    )
    QTest.keyClick(high_speed_view, Qt.Key_Return)
    qapp.processEvents()

    assert association_store.get("passage-12", HIGH_SPEED_SOURCE) is not None
    assert dialog.table.item(0, 7).text() == "双源确认"
    assert dialog.table.currentRow() == 1
    assert dialog._selected_event_id == "passage-15"
    dialog.close()
