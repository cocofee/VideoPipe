from dataclasses import dataclass


SUPPORTED_PIPELINES = {"athlete_bib"}
SUPPORTED_CROSSING_MODES = {"finish_once", "multi_lap"}
SUPPORTED_EQUIPMENT = {None, "bicycle"}


@dataclass(frozen=True)
class EventProfile:
    name: str
    required_equipment: str | None = None
    pipeline: str = "athlete_bib"
    crossing_mode: str = "finish_once"
    bib_regions: tuple[str, ...] = ("torso", "back")

    def __post_init__(self):
        normalized_name = str(self.name or "").strip()
        if not normalized_name:
            raise ValueError("Event profile name must not be blank")
        if isinstance(self.bib_regions, (str, bytes)):
            raise ValueError("Event profile bib regions must be a sequence of names")
        normalized_bib_regions = tuple(
            region.strip()
            for region in self.bib_regions
            if isinstance(region, str) and region.strip()
        )
        if len(normalized_bib_regions) != len(self.bib_regions):
            raise ValueError("Event profile bib regions must contain non-blank strings")
        if self.pipeline not in SUPPORTED_PIPELINES:
            raise ValueError(f"Unsupported event pipeline: {self.pipeline}")
        if self.crossing_mode not in SUPPORTED_CROSSING_MODES:
            raise ValueError(f"Unsupported crossing mode: {self.crossing_mode}")
        if self.required_equipment not in SUPPORTED_EQUIPMENT:
            raise ValueError(f"Unsupported required equipment: {self.required_equipment}")
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "bib_regions", normalized_bib_regions)


def build_event_profile(
    name: str,
    *,
    pipeline: str = "athlete_bib",
    crossing_mode: str = "finish_once",
    required_equipment: str | None = None,
    bib_regions: tuple[str, ...] = ("torso", "back"),
) -> EventProfile:
    return EventProfile(
        name=str(name or "").strip() or "current-event",
        required_equipment=required_equipment,
        pipeline=pipeline,
        crossing_mode=crossing_mode,
        bib_regions=bib_regions,
    )
