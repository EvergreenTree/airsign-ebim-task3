from airsign_task3.assignments import Seat, assign_objects_to_seats
from airsign_task3.types import Pose


def seat(name: str, x: float, y: float) -> Seat:
    return Seat(name, Pose((x, y, 0.0), (1.0, 0.0, 0.0, 0.0)))


def test_head_adjacent_bowl_and_spoon_share_assignment() -> None:
    seats = [seat("a", 2.0, 0.0), seat("b", 0.1, 0.0), seat("c", 1.0, 0.0)]
    result = assign_objects_to_seats(seats, (0.0, 0.0, 0.0))
    assert result["bowl"].name == "b"
    assert result["spoon"].name == "b"
    assert result["plate"].name != result["cup"].name

