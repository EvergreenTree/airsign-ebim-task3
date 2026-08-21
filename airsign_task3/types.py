from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


OFFICIAL_OBJECTS = ("plate", "cup", "bowl", "spoon")


class Lifecycle(str, Enum):
    READY = "READY"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    RECOVERY = "RECOVERY"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class Stage(str, Enum):
    TABLE_SETUP = "TABLE_SETUP"
    FEEDING = "FEEDING"
    BEAN_RECOVERY = "BEAN_RECOVERY"
    CLEANUP = "CLEANUP"


class Substate(str, Enum):
    IDLE = "IDLE"
    NAVIGATE = "NAVIGATE"
    APPROACH = "APPROACH"
    GRASP = "GRASP"
    CARRY = "CARRY"
    RELEASE = "RELEASE"
    VERIFY = "VERIFY"
    BACKOFF = "BACKOFF"
    REOPEN = "REOPEN"
    REAPPROACH = "REAPPROACH"
    REGRASP = "REGRASP"


@dataclass(frozen=True)
class Pose:
    position: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]


@dataclass(frozen=True)
class Bounds:
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]

    def contains(self, xyz: tuple[float, float, float], margin: float = 0.0) -> bool:
        return all(
            lo - margin <= value <= hi + margin
            for value, lo, hi in zip(xyz, self.minimum, self.maximum, strict=True)
        )


@dataclass
class SafetyTelemetry:
    peak_head_force_n: float = 0.0
    current_head_force_n: float = 0.0
    head_zone_active: bool = False
    watchdog_interventions: int = 0
    last_intervention: str | None = None


@dataclass
class EpisodeTelemetry:
    seed: int
    lifecycle: Lifecycle = Lifecycle.READY
    stage: Stage = Stage.TABLE_SETUP
    substate: Substate = Substate.IDLE
    simulated_seconds: float = 0.0
    wall_seconds: float = 0.0
    real_time_factor: float = 0.0
    score: float = 0.0
    highest_completed_stage: int = 0
    recovery_ratio: float = 0.0
    scene_ready: bool = False
    calibration_complete: bool = False
    assigned_seats: dict[str, str] = field(default_factory=dict)
    stage_scores: dict[str, float] = field(default_factory=dict)
    safety: SafetyTelemetry = field(default_factory=SafetyTelemetry)
    failure_reason: str | None = None
    message: str = "Initializing Isaac Sim"
    object_state: dict[str, Any] = field(default_factory=dict)
    robot_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    robot_orientation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
