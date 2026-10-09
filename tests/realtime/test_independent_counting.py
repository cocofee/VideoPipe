from dataclasses import replace

from realtime.independent_counting import CountRules, FrameStamp, IndependentCounter, Observation


def rules(**changes):
    base = CountRules("session", "race", "stage", "finish", "camera-1",
                      ((100, 0), (100, 200)), "pos_to_neg", (0,),
                      roi=((0, 0), (250, 0), (250, 200), (0, 200)), side_observations=2)
    return replace(base, **changes)


def observation(x, tid=1, **changes):
    return Observation(tid, (x - 10, 20, x + 10, 60), 0, 0.8, f"det-{tid}-{x}", **changes)


def feed(counter, positions, *, tid=1, start=10000, generation="stream", **changes):
    result = []
    for index, x in enumerate(positions, 1):
        stamp = FrameStamp(index, generation, start + index * 40, 999999, "source_pts")
        result.extend(counter.process(stamp, [observation(x, tid, **changes)]))
    return result


def test_anonymous_crossing_uses_observed_interval_and_does_not_wait_for_ocr():
    counter = IndependentCounter(rules())
    events = feed(counter, [70, 80, 110, 120])
    assert len(events) == 1
    event = events[0]
    assert event.eligibility == "admitted"
    assert event.identity == ""
    assert (event.cross_time_start_ms, event.cross_time_end_ms) == (10080, 10120)
    assert event.estimated_cross_time_ms == 10107
    assert event.before["signed_distance"] > 0 and event.after["signed_distance"] < 0
    assert event.published_at_ms != event.estimated_cross_time_ms


def test_touch_return_and_deadband_jitter_do_not_count():
    counter = IndependentCounter(rules())
    assert feed(counter, [80, 90, 100, 101, 99, 100, 80, 70]) == []
    assert feed(IndependentCounter(rules()), [80, 90, 120, 80, 70]) == []


def test_parallel_sprint_counts_every_distinct_track_without_geometric_dedupe():
    counter = IndependentCounter(rules())
    events = []
    for sequence, x in enumerate([70, 80, 110, 120], 1):
        observations = [observation(x + delta, tid) for tid, delta in enumerate([0, 1, 2, 3, 4, 5], 1)]
        events.extend(counter.process(FrameStamp(sequence, "stream", 10000 + sequence * 40, 20000), observations))
    assert len(events) == 6
    assert len({row.event_uid for row in events}) == 6
    assert all(row.eligibility == "admitted" for row in events)


def test_reverse_is_diagnosed_and_repeated_laps_remain_possible():
    counter = IndependentCounter(rules())
    events = feed(counter, [70, 80, 110, 120, 80, 70, 110, 120])
    assert [event.eligibility for event in events] == ["admitted", "excluded", "admitted"]
    assert events[1].reasons == ("reverse_direction",)
    assert events[0].event_uid != events[2].event_uid


def test_prediction_and_first_seen_after_line_are_not_confirmed_passages():
    assert feed(IndependentCounter(rules()), [110, 120, 130]) == []
    counter = IndependentCounter(rules())
    assert feed(counter, [70, 80, 110, 120], observed=False) == []
    assert any(row["reason"] == "prediction_only" for row in counter.last_quality)


def test_long_observation_gap_stays_pending():
    counter = IndependentCounter(rules())
    counter.process(FrameStamp(1, "g", 1000, 1000), [observation(70)])
    counter.process(FrameStamp(2, "g", 1040, 1040), [observation(80)])
    counter.process(FrameStamp(20, "g", 1600, 1600), [observation(110)])
    event, = counter.process(FrameStamp(21, "g", 1640, 1640), [observation(120)])
    assert event.eligibility == "pending"
    assert "observation_insufficient" in event.quality


def test_staff_requires_reliable_identity_and_outside_course_is_excluded():
    anonymous = feed(IndependentCounter(rules()), [70, 80, 110, 120], excluded_identity_reason="staff")
    assert anonymous[0].eligibility == "admitted"
    staff = feed(IndependentCounter(rules()), [70, 80, 110, 120], identity="official-1", identity_reliable=True, excluded_identity_reason="staff")
    assert staff[0].reasons == ("staff",)
    counter = IndependentCounter(rules(roi=((0, 70), (250, 70), (250, 200), (0, 200))))
    assert feed(counter, [70, 80, 110, 120])[0].reasons == ("outside_course",)


def test_reconnect_never_joins_two_unrelated_track_ids_into_a_crossing():
    counter = IndependentCounter(rules())
    assert feed(counter, [70, 80]) == []
    assert feed(counter, [110, 120], generation="new", start=10200) == []
    assert any(row["reason"] == "first_observed_after_line" for row in counter.last_quality)


def test_horizontal_and_diagonal_motion_use_line_normal_without_bib_filter():
    diagonal = IndependentCounter(rules(line=((0, 0), (200, 200))))
    events = feed(diagonal, [20, 30, 80, 90])
    assert len(events) == 1
    assert events[0].eligibility == "admitted"


def test_reliable_identity_duplicate_is_reversible_evidence_not_geometry():
    counter = IndependentCounter(rules())
    events = []
    for sequence, x in enumerate([70, 80, 110, 120], 1):
        events.extend(counter.process(FrameStamp(sequence, "g", 1000 + sequence * 40, 9999),
                                      [observation(x, tid, identity="rider-1", identity_reliable=True) for tid in (1, 2)]))
    assert [row.eligibility for row in events] == ["admitted", "excluded"]
    assert events[1].duplicate_of == events[0].event_uid


def test_reliable_identity_does_not_suppress_new_lap_on_same_track():
    events = feed(IndependentCounter(rules()), [70, 80, 110, 120, 80, 70, 110, 120],
                  identity="rider-1", identity_reliable=True)
    assert [row.eligibility for row in events] == ["admitted", "excluded", "admitted"]


def test_expired_initial_candidate_and_preline_track_are_diagnosed_without_count():
    counter = IndependentCounter(rules())
    counter.process(FrameStamp(1, "g", 1000, 1000), [observation(70)])
    assert counter.process(FrameStamp(2, "g", 4000, 4000), []) == ()
    counter = IndependentCounter(rules())
    feed(counter, [70, 80], start=1000)
    assert counter.process(FrameStamp(3, "stream", 4000, 4000), []) == ()
    assert any(row["reason"] == "broken_track" for row in counter.last_quality)
