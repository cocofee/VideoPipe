import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtWidgets import QApplication, QPushButton

from realtime.passage_receiver import PassageEvent, PassageEventStore
from realtime.passage_review import PassageReviewDialog
from realtime.video_timeline import VideoTimelineStore


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _event(passage_time_ms=15_000, passage_timestamp_ms=None):
    return PassageEvent(
        event_id="passage-1",
        race_id="race-1",
        stage_id="stage-1",
        group_id="men-open",
        sequence=1,
        chip_id="chip-23",
        bib="23",
        passage_time_ms=passage_time_ms,
        lap=2,
        emitted_at_ms=passage_time_ms + 100,
        passage_timestamp_ms=passage_timestamp_ms,
    )


def test_review_opens_located_video_at_preroll_position(qapp, tmp_path):
    passage_store = PassageEventStore(tmp_path / "passages.jsonl")
    passage_store.append(
        _event(passage_time_ms=5_000, passage_timestamp_ms=15_000)
    )
    timeline_store = VideoTimelineStore(tmp_path / "video_timeline.jsonl")
    video_path = tmp_path / "videos" / "camera_01.mkv"
    video_path.parent.mkdir()
    video_path.write_bytes(b"video")
    segment = timeline_store.start_segment(
        source_id="camera_01",
        camera_index=1,
        video_path=video_path,
        started_at_ms=10_000,
        timing_error_ms=1_000,
    )
    timeline_store.finish_segment(
        segment.segment_id,
        ended_at_ms=20_000,
        media_duration_ms=10_000,
        media_started_at_ms=10_000,
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
    assert dialog.table.item(0, 5).text() == "1 个机位，近似 ±1000 ms"
    button = dialog.table.cellWidget(0, 6)
    assert isinstance(button, QPushButton)
    assert button.isEnabled()
    button.click()

    assert opened[0][0].event_id == "passage-1"
    assert opened[0][1].video_path == video_path.absolute()
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


def test_review_allows_legacy_video_with_unverified_time_range(qapp, tmp_path):
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

    assert dialog.table.item(0, 5).text() == "1 个机位可打开，时间范围未验证"
    assert dialog.table.cellWidget(0, 6).isEnabled()
    dialog.close()


def test_review_disables_open_when_passage_has_no_recording(qapp, tmp_path):
    passage_store = PassageEventStore(tmp_path / "passages.jsonl")
    passage_store.append(_event())
    timeline_store = VideoTimelineStore(tmp_path / "video_timeline.jsonl")

    dialog = PassageReviewDialog(passage_store, timeline_store)
    qapp.processEvents()

    assert dialog.table.item(0, 5).text() == "没有录像时间线"
    assert dialog.table.cellWidget(0, 6).isEnabled() is False
    dialog.close()


def test_review_allows_near_boundary_clip_with_warning(qapp, tmp_path):
    passage_store = PassageEventStore(tmp_path / "passages.jsonl")
    passage_store.append(_event(passage_time_ms=9_950))
    timeline_store = VideoTimelineStore(tmp_path / "video_timeline.jsonl")
    video_path = tmp_path / "videos" / "camera_01.mkv"
    video_path.parent.mkdir()
    video_path.write_bytes(b"video")
    segment = timeline_store.start_segment(
        source_id="camera_01",
        camera_index=1,
        video_path=video_path,
        started_at_ms=10_000,
        clock_source="external_test_clock",
        timing_error_ms=100,
        race_id="race-1",
    )
    timeline_store.finish_segment(
        segment.segment_id,
        ended_at_ms=20_000,
        media_duration_ms=10_000,
        media_started_at_ms=10_000,
    )
    opened = []
    dialog = PassageReviewDialog(
        passage_store,
        timeline_store,
        open_location=lambda event, location: opened.append((event, location)),
    )
    qapp.processEvents()

    button = dialog.table.cellWidget(0, 6)
    assert button.isEnabled()
    assert "误差边界" in dialog.table.item(0, 5).text()
    button.click()

    assert opened[0][1].status == "near_boundary"
    assert opened[0][1].playback_position_ms == 0
    dialog.close()


def test_review_rejects_external_clip_from_another_race(qapp, tmp_path):
    passage_store = PassageEventStore(tmp_path / "passages.jsonl")
    passage_store.append(_event())
    timeline_store = VideoTimelineStore(tmp_path / "video_timeline.jsonl")
    video_path = tmp_path / "videos" / "other-race.mkv"
    video_path.parent.mkdir()
    video_path.write_bytes(b"video")
    segment = timeline_store.start_segment(
        source_id="high_speed_01",
        camera_index=1,
        video_path=video_path,
        started_at_ms=10_000,
        clock_source="external_test_clock",
        race_id="race-2",
    )
    timeline_store.finish_segment(
        segment.segment_id,
        ended_at_ms=20_000,
        media_duration_ms=10_000,
        media_started_at_ms=10_000,
    )

    dialog = PassageReviewDialog(passage_store, timeline_store)
    qapp.processEvents()

    assert dialog.table.item(0, 5).text() == "录像属于其他赛事"
    assert dialog.table.cellWidget(0, 6).isEnabled() is False
    dialog.close()
