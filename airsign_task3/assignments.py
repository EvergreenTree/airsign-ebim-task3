from __future__ import annotations

from dataclasses import dataclass
import math

from .types import Pose


@dataclass(frozen=True)
class Seat:
    name: str
    pose: Pose


def assign_objects_to_seats(
    seats: list[Seat], head_position: tuple[float, float, float]
) -> dict[str, Seat]:
    """Resolve the live assignment without relying on hard-coded seat indices.

    Bowl and spoon share the head-adjacent assignment. Plate and cup occupy the
    two other closest distinct seats. Stable lexical ordering makes ties
    deterministic across machines.
    """
    if len(seats) < 3:
        raise ValueError("Task 3 requires at least three seat anchors")
    def distance(seat: Seat) -> float:
        return math.dist(seat.pose.position, head_position)

    ordered = sorted(
        seats,
        key=lambda seat: (
            distance(seat),
            seat.name,
        ),
    )
    return {
        "bowl": ordered[0],
        "spoon": ordered[0],
        "plate": ordered[1],
        "cup": ordered[2],
    }
