from __future__ import annotations

import math
from dataclasses import dataclass

from .isaac_state import IsaacStateReader
from .scoring import ScoreBreakdown, ScoreEvidence, compute_score
from .types import Bounds, OFFICIAL_OBJECTS


@dataclass(frozen=True)
class AssignmentTarget:
    xy: tuple[float, float]
    tolerance_m: float = 0.22


class OfficialLiveScorer:
    """Official-rule-aligned scorer over read-only simulator state."""

    def __init__(
        self,
        reader: IsaacStateReader,
        assignments: dict[str, AssignmentTarget],
    ) -> None:
        self.reader = reader
        self.assignments = assignments
        self.evidence = ScoreEvidence()
        self.initial_bean_count = len(reader.bean_positions())
        self._feeding_had_beans = False
        self._feeding_hold_seconds = 0.0
        self._feeding_returned = False
        self._feeding_held_bean_indices: set[int] = set()
        self._feeding_active_bean_indices: set[int] = set()
        self._stage1_latched = {name: False for name in OFFICIAL_OBJECTS}

    @staticmethod
    def _inside_xy(bounds: Bounds, point: tuple[float, float, float], margin: float = 0.0) -> bool:
        return (
            bounds.minimum[0] - margin <= point[0] <= bounds.maximum[0] + margin
            and bounds.minimum[1] - margin <= point[1] <= bounds.maximum[1] + margin
        )

    @classmethod
    def _inside_recycling(cls, bounds: Bounds, point: tuple[float, float, float]) -> bool:
        return cls._inside_xy(bounds, point, margin=0.025) and (
            bounds.minimum[2] - 0.02 <= point[2] <= bounds.maximum[2] + 0.04
        )

    def record_feeding_hold(
        self,
        *,
        bean_indices: set[int],
        dt: float,
        in_head_zone: bool,
    ) -> None:
        if bean_indices and in_head_zone:
            if self._feeding_hold_seconds <= 0.0:
                self._feeding_active_bean_indices = set(bean_indices)
            else:
                self._feeding_active_bean_indices.intersection_update(bean_indices)
            if not self._feeding_active_bean_indices:
                self._feeding_hold_seconds = 0.0
                return
            self._feeding_hold_seconds += max(0.0, dt)
            if self._feeding_hold_seconds >= 3.0:
                self._feeding_had_beans = True
                self._feeding_held_bean_indices = set(
                    self._feeding_active_bean_indices
                )
        else:
            self._feeding_hold_seconds = 0.0
            self._feeding_active_bean_indices.clear()

    def record_feeding_return(self, bean_indices_in_bowl: set[int]) -> None:
        self._feeding_returned = bool(self._feeding_held_bean_indices) and (
            self._feeding_held_bean_indices <= bean_indices_in_bowl
        )

    def update(self) -> ScoreBreakdown:
        current_table: dict[str, bool] = {}
        for name in OFFICIAL_OBJECTS:
            target = self.assignments[name]
            position = self.reader.pose(name).position
            distance = math.hypot(position[0] - target.xy[0], position[1] - target.xy[1])
            current_table[name] = distance <= target.tolerance_m
        for name, correct in current_table.items():
            self._stage1_latched[name] = self._stage1_latched[name] or correct
        self.evidence.table_objects_correct.update(self._stage1_latched)

        beans = self.reader.bean_positions()
        recycling = self.reader.bounds("recycling")
        recovered = sum(self._inside_recycling(recycling, bean) for bean in beans)
        self.evidence.original_bean_mass = float(self.initial_bean_count)
        self.evidence.recovered_bean_mass = float(recovered)
        self.evidence.feeding_beans_present = self._feeding_had_beans
        self.evidence.feeding_hold_seconds = self._feeding_hold_seconds
        self.evidence.feeding_beans_returned = self._feeding_returned

        sink = self.reader.bounds("sink")
        for name in OFFICIAL_OBJECTS:
            position = self.reader.pose(name).position
            object_bounds = self.reader.bounds(name)
            center_inside = self._inside_xy(sink, position, margin=0.0)
            vertically_settled = object_bounds.minimum[2] >= sink.minimum[2] - 0.03
            self.evidence.sink_objects[name] = center_inside and vertically_settled
        return compute_score(self.evidence)

    @property
    def recovery_ratio(self) -> float:
        if self.evidence.original_bean_mass <= 0:
            return 0.0
        return max(0.0, min(1.0, self.evidence.recovered_bean_mass / self.evidence.original_bean_mass))

    @property
    def feeding_held_bean_indices(self) -> frozenset[int]:
        return frozenset(self._feeding_held_bean_indices)


def targets_from_head(
    head_position: tuple[float, float, float],
    *,
    table_center_xy: tuple[float, float] = (-2.1, 1.95),
) -> dict[str, AssignmentTarget]:
    """Initial assignment resolver, replaced by live marker prims when present."""
    dx = table_center_xy[0] - head_position[0]
    dy = table_center_xy[1] - head_position[1]
    norm = max(math.hypot(dx, dy), 1e-6)
    inward = (dx / norm, dy / norm)
    head_place = (
        head_position[0] + 0.34 * inward[0],
        head_position[1] + 0.34 * inward[1],
    )
    lateral = (-inward[1], inward[0])
    return {
        "bowl": AssignmentTarget(head_place),
        "spoon": AssignmentTarget(head_place, tolerance_m=0.30),
        "plate": AssignmentTarget(
            (head_place[0] - 0.42 * lateral[0], head_place[1] - 0.42 * lateral[1])
        ),
        "cup": AssignmentTarget(
            (head_place[0] + 0.42 * lateral[0], head_place[1] + 0.42 * lateral[1])
        ),
    }
