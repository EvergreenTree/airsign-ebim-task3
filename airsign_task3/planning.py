from __future__ import annotations

import heapq
import math
from dataclasses import dataclass


Point2 = tuple[float, float]


@dataclass(frozen=True)
class RectObstacle:
    minimum: Point2
    maximum: Point2

    def contains(self, point: Point2, clearance: float = 0.0) -> bool:
        x, y = point
        return (
            self.minimum[0] - clearance <= x <= self.maximum[0] + clearance
            and self.minimum[1] - clearance <= y <= self.maximum[1] + clearance
        )


def align_horizontal_corridor(
    waypoints: list[Point2],
    *,
    center_y: float,
    minimum_length: float = 0.50,
    maximum_vertical_delta: float = 0.15,
) -> list[Point2]:
    """Project long horizontal route segments onto a measured corridor line."""

    if minimum_length <= 0.0:
        raise ValueError("minimum_length must be positive")
    aligned = list(waypoints)
    for index, (start, end) in enumerate(zip(waypoints, waypoints[1:])):
        if abs(end[0] - start[0]) < minimum_length:
            continue
        if abs(end[1] - start[1]) > maximum_vertical_delta:
            continue
        aligned[index] = (start[0], center_y)
        aligned[index + 1] = (end[0], center_y)
    return aligned


def horizontal_corridor_entry_index(
    waypoints: list[Point2],
    *,
    minimum_length: float = 0.50,
    maximum_vertical_delta: float = 0.15,
) -> int | None:
    """Return the waypoint that begins the first long horizontal segment."""

    for index, (start, end) in enumerate(zip(waypoints, waypoints[1:])):
        if abs(end[0] - start[0]) < minimum_length:
            continue
        if abs(end[1] - start[1]) > maximum_vertical_delta:
            continue
        return index
    return None


def right_arm_facing_yaw(base: Point2, target: Point2) -> float:
    """Return base yaw that presents the robot's right side to a target."""

    dx = target[0] - base[0]
    dy = target[1] - base[1]
    if math.hypot(dx, dy) <= 1e-9:
        raise ValueError("base and target must be distinct")
    yaw = math.atan2(dy, dx) + 0.5 * math.pi
    return math.atan2(math.sin(yaw), math.cos(yaw))


def station_goal_candidates(
    target: Point2,
    supports: list[RectObstacle],
    *,
    standoff_m: float,
    prefer_outermost: bool = False,
    fallback_radius_m: float = 0.82,
) -> list[Point2]:
    if standoff_m <= 0.0:
        raise ValueError("standoff_m must be positive")
    target_x, target_y = target
    if supports:
        support = (max if prefer_outermost else min)(
            supports,
            key=lambda item: (
                item.maximum[0] - item.minimum[0]
            ) * (
                item.maximum[1] - item.minimum[1]
            ),
        )
        return [
            (support.maximum[0] + standoff_m, target_y),
            (target_x, support.maximum[1] + standoff_m),
            (target_x, support.minimum[1] - standoff_m),
            (support.minimum[0] - standoff_m, target_y),
        ]
    return [
        (
            target_x + fallback_radius_m * math.cos(angle),
            target_y + fallback_radius_m * math.sin(angle),
        )
        for angle in (
            0.0,
            math.pi / 2.0,
            -math.pi / 2.0,
            math.pi,
            math.pi / 4.0,
            -math.pi / 4.0,
        )
    ]


def clearance_egress_point(
    start: Point2,
    obstacles: list[RectObstacle],
    *,
    clearance: float,
    margin: float = 0.10,
) -> Point2:
    """Return the shortest axis-aligned point outside all expanded obstacles.

    A base can settle a few millimetres inside a conservative planning
    envelope after manipulation.  Dropping that obstacle allows a route to cut
    back through furniture.  Instead, move directly to the nearest exterior
    side with a bounded margin, then plan with the complete obstacle set.
    """
    containing = [
        obstacle for obstacle in obstacles if obstacle.contains(start, clearance)
    ]
    if not containing:
        return start
    candidates: list[Point2] = []
    x, y = start
    for obstacle in containing:
        candidates.extend(
            (
                (obstacle.minimum[0] - clearance - margin, y),
                (obstacle.maximum[0] + clearance + margin, y),
                (x, obstacle.minimum[1] - clearance - margin),
                (x, obstacle.maximum[1] + clearance + margin),
            )
        )
    safe = [
        candidate
        for candidate in candidates
        if not any(
            obstacle.contains(candidate, clearance) for obstacle in obstacles
        )
    ]
    if not safe:
        raise ValueError("no direct clearance egress from start")
    return min(safe, key=lambda candidate: math.dist(start, candidate))


def collision_cleared_waypoints(
    start: Point2,
    goal: Point2,
    obstacles: list[RectObstacle],
    *,
    clearance: float = 0.47,
    resolution: float = 0.10,
    max_nodes: int = 100_000,
) -> list[Point2]:
    """Plan a deterministic 8-connected path and reduce it to bend waypoints."""
    if resolution <= 0:
        raise ValueError("resolution must be positive")

    def cell(point: Point2) -> tuple[int, int]:
        return (round(point[0] / resolution), round(point[1] / resolution))

    def point(grid_cell: tuple[int, int]) -> Point2:
        return (grid_cell[0] * resolution, grid_cell[1] * resolution)

    def blocked(grid_cell: tuple[int, int]) -> bool:
        p = point(grid_cell)
        return any(obstacle.contains(p, clearance) for obstacle in obstacles)

    if any(obstacle.contains(start, clearance) for obstacle in obstacles) or any(
        obstacle.contains(goal, clearance) for obstacle in obstacles
    ):
        raise ValueError("start or goal is inside a clearance-expanded obstacle")

    def nearest_unblocked_cell(
        exact: Point2, rounded: tuple[int, int]
    ) -> tuple[int, int]:
        """Keep a safe continuous endpoint from being rounded into furniture."""
        if not blocked(rounded):
            return rounded
        candidates: list[tuple[float, tuple[int, int]]] = []
        for radius in range(1, 5):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy)) != radius:
                        continue
                    candidate = (rounded[0] + dx, rounded[1] + dy)
                    if blocked(candidate):
                        continue
                    candidates.append((math.dist(point(candidate), exact), candidate))
            if candidates:
                return min(candidates, key=lambda item: item[0])[1]
        raise ValueError("no free grid cell near safe start or goal")

    start_cell = nearest_unblocked_cell(start, cell(start))
    goal_cell = nearest_unblocked_cell(goal, cell(goal))

    neighbors = (
        (-1, -1), (-1, 0), (-1, 1), (0, -1),
        (0, 1), (1, -1), (1, 0), (1, 1),
    )
    queue: list[tuple[float, tuple[int, int]]] = [(0.0, start_cell)]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    cost = {start_cell: 0.0}
    visited = 0

    while queue:
        _, current = heapq.heappop(queue)
        visited += 1
        if visited > max_nodes:
            raise RuntimeError("waypoint search exceeded node budget")
        if current == goal_cell:
            break
        for dx, dy in neighbors:
            nxt = (current[0] + dx, current[1] + dy)
            if blocked(nxt):
                continue
            # A diagonal move must not squeeze between two occupied cardinal
            # cells.  A point robot can cross that mathematical corner, but
            # the mobile base's swept footprint cannot.
            if dx and dy and (
                blocked((current[0] + dx, current[1]))
                or blocked((current[0], current[1] + dy))
            ):
                continue
            step = math.sqrt(2.0) if dx and dy else 1.0
            new_cost = cost[current] + step
            if new_cost >= cost.get(nxt, math.inf):
                continue
            cost[nxt] = new_cost
            came_from[nxt] = current
            heuristic = math.hypot(goal_cell[0] - nxt[0], goal_cell[1] - nxt[1])
            heapq.heappush(queue, (new_cost + heuristic, nxt))
    else:
        raise RuntimeError("no collision-cleared base path")

    cells = [goal_cell]
    while cells[-1] != start_cell:
        cells.append(came_from[cells[-1]])
    cells.reverse()

    reduced = [cells[0]]
    previous_direction: tuple[int, int] | None = None
    for index in range(1, len(cells)):
        direction = (
            cells[index][0] - cells[index - 1][0],
            cells[index][1] - cells[index - 1][1],
        )
        if previous_direction is not None and direction != previous_direction:
            reduced.append(cells[index - 1])
        previous_direction = direction
    reduced.append(cells[-1])
    waypoints = [point(item) for item in reduced]
    waypoints[0] = start
    waypoints[-1] = goal
    return waypoints
