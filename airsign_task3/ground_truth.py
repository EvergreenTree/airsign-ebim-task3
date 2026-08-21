from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .types import Bounds, Pose


@dataclass(frozen=True)
class GroundTruthSnapshot:
    poses: dict[str, Pose]
    bounds: dict[str, Bounds]
    bean_positions: tuple[tuple[float, float, float], ...]


class GroundTruthReader:
    """Read-only façade around simulator transform and bounds queries."""

    def __init__(
        self,
        pose_query: Callable[[str], Pose],
        bounds_query: Callable[[str], Bounds],
        bean_query: Callable[[], tuple[tuple[float, float, float], ...]],
    ) -> None:
        self._pose_query = pose_query
        self._bounds_query = bounds_query
        self._bean_query = bean_query

    def snapshot(self, names: tuple[str, ...]) -> GroundTruthSnapshot:
        return GroundTruthSnapshot(
            poses={name: self._pose_query(name) for name in names},
            bounds={name: self._bounds_query(name) for name in names},
            bean_positions=self._bean_query(),
        )

