from __future__ import annotations

from dataclasses import dataclass, field

from .types import OFFICIAL_OBJECTS


@dataclass
class ScoreEvidence:
    table_objects_correct: dict[str, bool] = field(
        default_factory=lambda: {name: False for name in OFFICIAL_OBJECTS}
    )
    feeding_beans_present: bool = False
    feeding_hold_seconds: float = 0.0
    feeding_beans_returned: bool = False
    original_bean_mass: float = 0.0
    recovered_bean_mass: float = 0.0
    sink_objects: dict[str, bool] = field(
        default_factory=lambda: {name: False for name in OFFICIAL_OBJECTS}
    )


@dataclass(frozen=True)
class ScoreBreakdown:
    stage1: float
    stage2: float
    stage3: float
    stage4: float

    @property
    def total(self) -> float:
        return self.stage1 + self.stage2 + self.stage3 + self.stage4

    def as_dict(self) -> dict[str, float]:
        return {
            "stage1": self.stage1,
            "stage2": self.stage2,
            "stage3": self.stage3,
            "stage4": self.stage4,
            "total": self.total,
        }


def compute_score(evidence: ScoreEvidence) -> ScoreBreakdown:
    stage1 = float(sum(bool(evidence.table_objects_correct.get(name)) for name in OFFICIAL_OBJECTS))
    stage2 = 4.0 if (
        evidence.feeding_beans_present
        and evidence.feeding_hold_seconds >= 3.0
        and evidence.feeding_beans_returned
    ) else 0.0
    ratio = 0.0
    if evidence.original_bean_mass > 0.0:
        ratio = evidence.recovered_bean_mass / evidence.original_bean_mass
    stage3 = 4.0 * max(0.0, min(1.0, ratio))
    stage4 = float(sum(bool(evidence.sink_objects.get(name)) for name in OFFICIAL_OBJECTS))
    return ScoreBreakdown(stage1=stage1, stage2=stage2, stage3=stage3, stage4=stage4)


def highest_completed_stage(score: ScoreBreakdown) -> int:
    completed = 0
    for stage_number, stage_score in enumerate(
        (score.stage1, score.stage2, score.stage3, score.stage4),
        start=1,
    ):
        if stage_score >= 4.0 - 1e-9:
            completed = stage_number
    return completed
