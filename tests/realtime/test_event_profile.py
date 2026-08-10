import subprocess
import sys
from pathlib import Path

import pytest

from realtime.event_profile import EventProfile, build_event_profile, build_sport_event_profile


def test_default_event_uses_the_common_athlete_bib_pipeline():
    profile = build_event_profile(name="current-event")

    assert profile.name == "current-event"
    assert profile.pipeline == "athlete_bib"
    assert profile.required_equipment is None
    assert profile.bib_regions == ("torso", "back")


def test_event_profile_accepts_real_competition_differences_as_configuration():
    profile = build_event_profile(
        name="current-cycling-event",
        required_equipment="bicycle",
        bib_regions=("torso", "back", "handlebar", "frame"),
        crossing_mode="multi_lap",
    )

    assert profile.pipeline == "athlete_bib"
    assert profile.required_equipment == "bicycle"
    assert profile.crossing_mode == "multi_lap"


def test_speed_skating_profile_uses_helmet_and_thigh_bibs_without_bicycle_gate():
    profile = build_sport_event_profile("roller_skating")

    assert profile.name == "speed_skating"
    assert profile.required_equipment is None
    assert profile.bib_regions == ("helmet", "left_thigh", "right_thigh")


def test_road_cycling_profile_keeps_bicycle_gate_and_rear_saddle_bib_region():
    profile = build_sport_event_profile("road_cycling")

    assert profile.name == "cycling"
    assert profile.required_equipment == "bicycle"
    assert profile.bib_regions == ("rear_saddle", "back")


def test_unknown_pipeline_fails_instead_of_silently_switching_algorithms():
    with pytest.raises(ValueError, match="pipeline"):
        build_event_profile(name="dense-test", pipeline="unknown")


def test_unknown_crossing_mode_fails_explicitly():
    with pytest.raises(ValueError, match="crossing mode"):
        build_event_profile(name="bad-mode", crossing_mode="unknown")


def test_unsupported_equipment_validator_fails_explicitly():
    with pytest.raises(ValueError, match="equipment"):
        build_event_profile(name="bad-equipment", required_equipment="motorcycle")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("pipeline", "unknown", "pipeline"),
        ("crossing_mode", "unknown", "crossing mode"),
        ("required_equipment", "motorcycle", "equipment"),
    ],
)
def test_direct_event_profile_construction_enforces_supported_values(field, value, message):
    kwargs = {field: value}

    with pytest.raises(ValueError, match=message):
        EventProfile(name="invalid-event", **kwargs)


def test_detector_remains_importable_as_a_top_level_module():
    project_root = Path(__file__).resolve().parents[2]
    realtime_dir = project_root / "realtime"
    command = (
        "import sys; "
        f"sys.path.insert(0, {str(realtime_dir)!r}); "
        "import detector"
    )

    result = subprocess.run(
        [sys.executable, "-c", command],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_direct_event_profile_normalizes_immutable_fields():
    profile = EventProfile(
        name=" current-event ",
        bib_regions=["torso", "back"],
    )

    assert profile.name == "current-event"
    assert profile.bib_regions == ("torso", "back")


def test_event_profile_rejects_blank_name_and_string_bib_regions():
    with pytest.raises(ValueError, match="name"):
        EventProfile(name="   ")

    with pytest.raises(ValueError, match="bib regions"):
        EventProfile(name="current-event", bib_regions="torso")
