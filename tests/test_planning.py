import math

from airsign_task3.planning import (
    RectObstacle,
    align_horizontal_corridor,
    clearance_egress_point,
    collision_cleared_waypoints,
    horizontal_corridor_entry_index,
    right_arm_facing_yaw,
    station_goal_candidates,
)


def test_waypoints_clear_expanded_obstacle() -> None:
    obstacle = RectObstacle((0.4, -0.2), (0.6, 0.2))
    path = collision_cleared_waypoints((0.0, 0.0), (1.0, 0.0), [obstacle], clearance=0.1, resolution=0.05)
    assert path[0] == (0.0, 0.0)
    assert path[-1] == (1.0, 0.0)
    assert all(not obstacle.contains(point, 0.1) for point in path)


def test_waypoints_do_not_cut_between_blocked_cardinal_cells() -> None:
    obstacles = [
        RectObstacle((1.0, 0.0), (1.0, 0.0)),
        RectObstacle((0.0, 1.0), (0.0, 1.0)),
    ]
    path = collision_cleared_waypoints(
        (0.0, 0.0),
        (1.0, 1.0),
        obstacles,
        clearance=0.0,
        resolution=1.0,
    )
    assert len(path) > 2
    assert path[1] != (1.0, 1.0)


def test_safe_endpoint_is_not_rejected_when_grid_rounding_enters_obstacle() -> None:
    obstacle = RectObstacle((-1.0, -1.0), (0.03, 1.0))
    start = (1.0, -1.5)
    goal = (0.42, 0.0)

    path = collision_cleared_waypoints(
        start,
        goal,
        [obstacle],
        clearance=0.38,
        resolution=0.10,
    )

    assert path[0] == start
    assert path[-1] == goal


def test_clearance_egress_moves_directly_away_from_supply_table() -> None:
    table = RectObstacle((-5.70, -2.47), (-4.96, -1.22))
    start = (-4.59, -1.65)

    egress = clearance_egress_point(
        start,
        [table],
        clearance=0.38,
        margin=0.10,
    )

    assert egress[0] > start[0]
    assert egress[1] == start[1]
    assert not table.contains(egress, clearance=0.38)


def test_long_horizontal_corridor_is_projected_to_live_centerline() -> None:
    path = [
        (-4.1, 1.4),
        (-3.5, 1.0),
        (-2.3, 1.0),
        (-2.2, 0.9),
        (-2.126, 0.941),
    ]

    aligned = align_horizontal_corridor(path, center_y=0.941)

    assert aligned == [
        (-4.1, 1.4),
        (-3.5, 0.941),
        (-2.3, 0.941),
        (-2.2, 0.9),
        (-2.126, 0.941),
    ]


def test_short_horizontal_segment_is_not_moved() -> None:
    path = [(-2.3, 1.0), (-2.2, 0.9), (-2.126, 0.941)]

    assert align_horizontal_corridor(path, center_y=0.941) == path


def test_horizontal_corridor_entry_index_finds_projected_segment() -> None:
    south = [
        (-4.1, 1.4),
        (-3.9, 1.4),
        (-3.5, 0.941),
        (-2.3, 0.941),
        (-2.126, 0.941),
    ]
    north_retry = [
        (-4.22, 0.99),
        (-3.5, 1.7),
        (-3.5, 2.941),
        (-2.126, 2.941),
    ]

    assert horizontal_corridor_entry_index(south) == 2
    assert horizontal_corridor_entry_index(north_retry) == 2


def test_horizontal_corridor_entry_index_returns_none_for_vertical_route() -> None:
    assert horizontal_corridor_entry_index([(0.0, 0.0), (0.0, 1.0)]) is None


def test_right_arm_yaw_uses_final_station_not_temporary_turn_point() -> None:
    target = (-2.126, 1.884)

    assert math.isclose(
        abs(right_arm_facing_yaw((-2.126, 0.941), target)),
        math.pi,
    )
    assert math.isclose(
        right_arm_facing_yaw((-2.126, 2.941), target),
        0.0,
        abs_tol=1e-12,
    )


def test_counter_station_uses_outer_support_not_nested_scale() -> None:
    counter = RectObstacle((-5.70, -2.47), (-4.96, -1.22))
    scale = RectObstacle((-5.21, -2.00), (-5.01, -1.85))

    candidates = station_goal_candidates(
        (-5.14, -1.92),
        [counter, scale],
        standoff_m=0.60,
        prefer_outermost=True,
    )

    assert candidates[0] == (-4.36, -1.92)
    assert not counter.contains(candidates[0], clearance=0.56)
