import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from realtime.passage_receiver import (
    PassageEvent,
    PassageEventReceiver,
    PassageEventStore,
)


def passage_payload(**overrides):
    payload = {
        "schema_version": 1,
        "message_type": "passage",
        "event_id": "race-1-stage-1-passage-7",
        "race_id": "race-1",
        "stage_id": "stage-1",
        "group_id": "men-open",
        "sequence": 7,
        "chip_id": "chip-23",
        "bib": "23",
        "passage_time_ms": 1_787_217_138_520,
        "lap": 1,
        "source": "cyclerace",
        "emitted_at_ms": 1_787_217_138_700,
        "revision": 1,
    }
    payload.update(overrides)
    return payload


def post_json(receiver, payload, path="/api/v1/passage-events"):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        f"http://127.0.0.1:{receiver.listen_port}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=2.0) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


@pytest.fixture
def running_receiver(tmp_path):
    accepted = []
    store = PassageEventStore(tmp_path / "passage-events.jsonl")
    receiver = PassageEventReceiver(
        "127.0.0.1",
        0,
        store,
        on_accepted=accepted.append,
    )
    receiver.start()
    try:
        yield receiver, store, accepted
    finally:
        receiver.stop()


def test_accepts_and_deduplicates_one_revision(running_receiver):
    receiver, store, accepted = running_receiver

    first_status, first_ack = post_json(receiver, passage_payload())
    second_status, second_ack = post_json(receiver, passage_payload())

    assert first_status == 201
    assert first_ack["status"] == "accepted"
    assert second_status == 200
    assert second_ack["status"] == "duplicate"
    assert len(store) == 1
    assert [event.event_id for event in accepted] == [
        "race-1-stage-1-passage-7"
    ]
    assert len(store.journal_path.read_text(encoding="utf-8").splitlines()) == 1


def test_optional_field_preserves_positional_constructor_contract():
    event = PassageEvent(
        "event-1",
        "race-1",
        "stage-1",
        "group-1",
        7,
        "chip-23",
        "23",
        123_456,
        2,
        "cyclerace",
        456_789,
        3,
        1,
        "passage",
    )

    assert event.lap == 2
    assert event.message_type == "passage"
    assert event.passage_timestamp_ms is None
    assert event.timeline_timestamp_ms == 123_456


def test_optional_absolute_passage_timestamp_is_preserved(running_receiver):
    receiver, store, accepted = running_receiver
    absolute_timestamp_ms = 1_786_252_979_215

    status, ack = post_json(
        receiver,
        passage_payload(
            passage_time_ms=48_179_215,
            passage_timestamp_ms=absolute_timestamp_ms,
        ),
    )

    assert status == 201
    assert ack["status"] == "accepted"
    event = store.get("race-1-stage-1-passage-7")
    assert event.passage_time_ms == 48_179_215
    assert event.passage_timestamp_ms == absolute_timestamp_ms
    assert event.timeline_timestamp_ms == absolute_timestamp_ms
    assert accepted == [event]


def test_same_revision_with_different_content_is_rejected(running_receiver):
    receiver, store, _ = running_receiver
    assert post_json(receiver, passage_payload())[0] == 201

    status, ack = post_json(receiver, passage_payload(bib="99"))

    assert status == 409
    assert ack["status"] == "rejected"
    assert store.get("race-1-stage-1-passage-7").bib == "23"


def test_race_journal_rejects_passage_from_another_race(running_receiver):
    receiver, store, accepted = running_receiver
    assert post_json(receiver, passage_payload())[0] == 201

    status, ack = post_json(
        receiver,
        passage_payload(
            event_id="race-2-stage-1-passage-1",
            race_id="race-2",
            sequence=1,
        ),
    )

    assert status == 409
    assert ack["status"] == "rejected"
    assert len(store) == 1
    assert [event.race_id for event in accepted] == ["race-1"]


def test_newer_revision_replaces_latest_and_stale_revision_is_duplicate(
    running_receiver,
):
    receiver, store, accepted = running_receiver
    assert post_json(receiver, passage_payload())[0] == 201

    corrected = passage_payload(revision=2, bib="24", emitted_at_ms=1_787_217_139_000)
    corrected_status, corrected_ack = post_json(receiver, corrected)
    stale_status, stale_ack = post_json(receiver, passage_payload())

    assert corrected_status == 201
    assert corrected_ack["status"] == "accepted"
    assert stale_status == 200
    assert stale_ack["status"] == "duplicate"
    assert store.get("race-1-stage-1-passage-7").revision == 2
    assert store.get("race-1-stage-1-passage-7").bib == "24"
    assert [event.revision for event in accepted] == [1, 2]
    assert len(store.journal_path.read_text(encoding="utf-8").splitlines()) == 2


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": 1},
        passage_payload(schema_version=2),
        passage_payload(message_type="result"),
        passage_payload(sequence=0),
        passage_payload(sequence="7"),
        passage_payload(chip_id="", bib=""),
        passage_payload(passage_time_ms=-1),
        passage_payload(passage_timestamp_ms=-1),
        passage_payload(lap=-1),
        passage_payload(revision=0),
    ],
)
def test_invalid_payload_is_rejected(running_receiver, payload):
    receiver, store, accepted = running_receiver

    status, ack = post_json(receiver, payload)

    assert status == 400
    assert ack["status"] == "rejected"
    assert len(store) == 0
    assert accepted == []


def test_wrong_endpoint_is_not_accepted(running_receiver):
    receiver, store, accepted = running_receiver

    status, ack = post_json(receiver, passage_payload(), path="/wrong")

    assert status == 404
    assert ack["status"] == "rejected"
    assert len(store) == 0
    assert accepted == []


def test_callback_failure_returns_retry_and_duplicate_retry_delivers(tmp_path):
    attempts = []

    def callback(event):
        attempts.append(event.event_id)
        if len(attempts) == 1:
            raise RuntimeError("review workspace unavailable")

    store = PassageEventStore(tmp_path / "passage-events.jsonl")
    receiver = PassageEventReceiver(
        "127.0.0.1",
        0,
        store,
        on_accepted=callback,
    )
    receiver.start()
    try:
        first_status, first_ack = post_json(receiver, passage_payload())
        second_status, second_ack = post_json(receiver, passage_payload())
    finally:
        receiver.stop()

    assert first_status == 503
    assert first_ack["status"] == "retry"
    assert second_status == 200
    assert second_ack["status"] == "duplicate"
    assert attempts == [
        "race-1-stage-1-passage-7",
        "race-1-stage-1-passage-7",
    ]
    assert len(store) == 1
    assert len(store.journal_path.read_text(encoding="utf-8").splitlines()) == 1


def test_store_recovers_incomplete_tail_and_restores_latest_revision(tmp_path):
    journal = tmp_path / "passage-events.jsonl"
    first = PassageEvent.from_payload(passage_payload())
    second = PassageEvent.from_payload(
        passage_payload(revision=2, bib="24", emitted_at_ms=1_787_217_139_000)
    )
    journal.write_bytes(
        json.dumps(first.to_payload(), separators=(",", ":")).encode("utf-8")
        + b"\n"
        + json.dumps(second.to_payload(), separators=(",", ":")).encode("utf-8")
        + b"\n"
        + b'{"schema_version":1,"message_type":"passage"'
    )

    store = PassageEventStore(journal)

    assert store.recovered_incomplete_tail is True
    assert len(store) == 1
    assert store.get(first.event_id).revision == 2
    assert store.get(first.event_id).bib == "24"
    assert journal.read_bytes().endswith(b"\n")
    assert len(journal.read_text(encoding="utf-8").splitlines()) == 2


def test_restart_duplicate_notifies_new_review_workspace(tmp_path):
    journal = tmp_path / "passage-events.jsonl"
    first_store = PassageEventStore(journal)
    first_store.append(PassageEvent.from_payload(passage_payload()))
    restored_store = PassageEventStore(journal)
    delivered = []
    receiver = PassageEventReceiver(
        "127.0.0.1",
        0,
        restored_store,
        on_accepted=delivered.append,
    )
    receiver.start()
    try:
        status, ack = post_json(receiver, passage_payload())
    finally:
        receiver.stop()

    assert status == 200
    assert ack["status"] == "duplicate"
    assert [event.event_id for event in delivered] == [
        "race-1-stage-1-passage-7"
    ]


def test_stop_is_idempotent(tmp_path):
    receiver = PassageEventReceiver(
        "127.0.0.1",
        0,
        PassageEventStore(tmp_path / "passage-events.jsonl"),
    )
    receiver.start()
    assert receiver.is_running is True

    receiver.stop()
    receiver.stop()

    assert receiver.is_running is False
