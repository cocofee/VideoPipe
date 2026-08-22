import pytest

from realtime.video_supplement import VideoSupplementStore, parse_beijing_datetime


def test_parse_beijing_datetime_and_persist_supplement(tmp_path):
    observed_at = parse_beijing_datetime("2026-08-22 10:00:01.250")
    assert observed_at is not None

    journal = tmp_path / "video_supplements.jsonl"
    store = VideoSupplementStore(journal)
    item = store.append(bib="X-1", observed_at_ms=observed_at, note="no chip in RaceTiger")

    restored = VideoSupplementStore(journal)
    assert restored.items() == (item,)
    assert restored.items()[0].bib == "X-1"


def test_corrupt_supplement_lines_do_not_block_reload(tmp_path):
    journal = tmp_path / "video_supplements.jsonl"
    journal.write_bytes(
        b"not-json\n"
        b'{"supplement_id":"video:ok","bib":"7","observed_at_ms":1}\n'
        b'{"supplement_id":"video:tail","bib":"8"'
    )

    restored = VideoSupplementStore(journal)

    assert [item.bib for item in restored.items()] == ["7"]


def test_video_supplement_requires_bib(tmp_path):
    store = VideoSupplementStore(tmp_path / "video_supplements.jsonl")

    with pytest.raises(ValueError, match="bib is required"):
        store.append(bib="", observed_at_ms=1)
