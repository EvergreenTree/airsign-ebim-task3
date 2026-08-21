from __future__ import annotations

import math
from collections.abc import Sequence


def head_mouth_target(
    head_position: Sequence[float],
    eye_positions: Sequence[Sequence[float]],
    *,
    table_center_xy: Sequence[float] = (-2.1, 1.95),
    face_surface_inward_m: float = 0.04,
    eye_to_mouth_drop_m: float = 0.045,
) -> tuple[float, float, float]:
    """Estimate the live mouth target from read-only head geometry.

    The skinned head mesh has unusably large authored bounds, but its eye
    descendants have stable world transforms.  Average those transforms for
    face height and move a small measured distance toward the dining-table
    center.  If eye geometry is unavailable, retain the same calibrated
    height relative to the head root instead of consulting static coordinates.
    """

    head = tuple(float(value) for value in head_position)
    if len(head) != 3:
        raise ValueError("head_position must have three components")
    table_xy = tuple(float(value) for value in table_center_xy)
    if len(table_xy) != 2:
        raise ValueError("table_center_xy must have two components")
    inward_x = table_xy[0] - head[0]
    inward_y = table_xy[1] - head[1]
    inward_norm = math.hypot(inward_x, inward_y)
    if inward_norm <= 1e-9:
        raise ValueError("head and table center must have distinct XY positions")
    inward_x /= inward_norm
    inward_y /= inward_norm

    eyes = [tuple(float(value) for value in position) for position in eye_positions]
    if any(len(position) != 3 for position in eyes):
        raise ValueError("each eye position must have three components")
    if eyes:
        face_center = tuple(
            sum(position[axis] for position in eyes) / len(eyes)
            for axis in range(3)
        )
    else:
        # Current Task 3 heads place the eye assembly 176 mm above the root.
        face_center = (head[0], head[1], head[2] + 0.176)
    return (
        face_center[0] + inward_x * float(face_surface_inward_m),
        face_center[1] + inward_y * float(face_surface_inward_m),
        face_center[2] - float(eye_to_mouth_drop_m),
    )


def rotate_vector_wxyz(
    quaternion_wxyz: Sequence[float],
    vector: Sequence[float],
) -> tuple[float, float, float]:
    """Rotate a 3-vector by a normalized wxyz quaternion."""

    w, x, y, z = (float(value) for value in quaternion_wxyz)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        raise ValueError("quaternion must be nonzero")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    vx, vy, vz = (float(value) for value in vector)
    return (
        (1 - 2 * (y * y + z * z)) * vx
        + 2 * (x * y - w * z) * vy
        + 2 * (x * z + w * y) * vz,
        2 * (x * y + w * z) * vx
        + (1 - 2 * (x * x + z * z)) * vy
        + 2 * (y * z - w * x) * vz,
        2 * (x * z - w * y) * vx
        + 2 * (y * z + w * x) * vy
        + (1 - 2 * (x * x + y * y)) * vz,
    )


def held_object_tcp_target(
    object_target: Sequence[float],
    current_tcp: Sequence[float],
    object_position: Sequence[float],
) -> tuple[float, float, float]:
    """Convert an object-space target into a TCP target for a live grasp."""

    target = tuple(float(value) for value in object_target)
    tcp = tuple(float(value) for value in current_tcp)
    position = tuple(float(value) for value in object_position)
    if not all(len(values) == 3 for values in (target, tcp, position)):
        raise ValueError("object target, TCP, and object position must be 3D")
    return tuple(
        target[axis] + tcp[axis] - position[axis]
        for axis in range(3)
    )


def co_moving_payload_indices(
    before_positions: Sequence[Sequence[float]],
    after_positions: Sequence[Sequence[float]],
    tool_before: Sequence[float],
    tool_after: Sequence[float],
    *,
    minimum_tool_lift_m: float = 0.035,
    minimum_payload_lift_m: float = 0.020,
    displacement_tolerance_m: float = 0.045,
    maximum_tool_distance_m: float = 0.12,
) -> set[int]:
    """Identify payload particles that physically moved with a lifted tool."""

    if len(before_positions) != len(after_positions):
        raise ValueError("payload snapshots must contain the same number of positions")
    tool_start = tuple(float(value) for value in tool_before)
    tool_end = tuple(float(value) for value in tool_after)
    if len(tool_start) != 3 or len(tool_end) != 3:
        raise ValueError("tool positions must be 3D")
    tool_delta = tuple(tool_end[axis] - tool_start[axis] for axis in range(3))
    if tool_delta[2] < float(minimum_tool_lift_m):
        return set()

    carried: set[int] = set()
    for index, (before, after) in enumerate(zip(before_positions, after_positions)):
        start = tuple(float(value) for value in before)
        end = tuple(float(value) for value in after)
        if len(start) != 3 or len(end) != 3:
            raise ValueError("payload positions must be 3D")
        payload_delta = tuple(end[axis] - start[axis] for axis in range(3))
        if payload_delta[2] < float(minimum_payload_lift_m):
            continue
        if math.dist(payload_delta, tool_delta) > float(displacement_tolerance_m):
            continue
        if math.dist(end, tool_end) > float(maximum_tool_distance_m):
            continue
        carried.add(index)
    return carried


def retained_payload_indices(
    positions: Sequence[Sequence[float]],
    candidate_indices: Sequence[int],
    tool_position: Sequence[float],
    *,
    maximum_tool_distance_m: float = 0.12,
) -> set[int]:
    """Return previously verified payload indices that remain with a tool."""

    tool = tuple(float(value) for value in tool_position)
    if len(tool) != 3:
        raise ValueError("tool_position must be 3D")
    retained: set[int] = set()
    for index in candidate_indices:
        if index < 0 or index >= len(positions):
            continue
        position = tuple(float(value) for value in positions[index])
        if len(position) != 3:
            raise ValueError("payload positions must be 3D")
        if math.dist(position, tool) <= float(maximum_tool_distance_m):
            retained.add(index)
    return retained


def top_clearance_path(
    start: Sequence[float],
    target: Sequence[float],
    *,
    clearance_m: float = 0.10,
    direct_xy_threshold_m: float = 0.12,
) -> list[tuple[float, float, float]]:
    """Build a lift/translate/descend path unless already over the target.

    A recovery that starts close to the requested pregrasp must not command a
    new overhead lift.  Near the edge of the arm workspace that artificial
    lift can be unreachable even though the actual pregrasp is reachable.
    """

    start_xyz = tuple(float(value) for value in start)
    target_xyz = tuple(float(value) for value in target)
    xy_distance = math.hypot(
        target_xyz[0] - start_xyz[0], target_xyz[1] - start_xyz[1]
    )
    if xy_distance <= direct_xy_threshold_m:
        return [target_xyz]
    transit_z = max(start_xyz[2], target_xyz[2] + clearance_m)
    return [
        (start_xyz[0], start_xyz[1], transit_z),
        (target_xyz[0], target_xyz[1], transit_z),
        target_xyz,
    ]


def circular_rim_inset(diameter_m: float, grasp_chord_m: float) -> float:
    """Return the rim inset that produces a requested circular chord width."""

    if diameter_m <= 0.0 or grasp_chord_m <= 0.0:
        raise ValueError("diameter and chord must be positive")
    radius = 0.5 * float(diameter_m)
    half_chord = min(0.5 * float(grasp_chord_m), radius)
    return radius - math.sqrt(max(radius * radius - half_chord * half_chord, 0.0))


def supported_assignment_reached(
    object_position: Sequence[float],
    assignment_xy: Sequence[float],
    tolerance_m: float,
    *,
    lowest_point_z: float,
    support_z_min_m: float = 0.70,
    support_z_max_m: float = 0.82,
    score_margin_m: float = 0.01,
) -> bool:
    """Return whether an object is supported inside its scored table region.

    The pose and bounds are observations only. Requiring the object's lowest
    point to be near the dining-table top prevents a dropped object on the
    floor, or an object still suspended above the table, from ending a lower
    command early.
    """

    effective_tolerance = max(0.0, float(tolerance_m) - float(score_margin_m))
    xy_distance = math.hypot(
        float(object_position[0]) - float(assignment_xy[0]),
        float(object_position[1]) - float(assignment_xy[1]),
    )
    return (
        xy_distance <= effective_tolerance
        and float(support_z_min_m) <= float(lowest_point_z) <= float(support_z_max_m)
    )


def post_release_stow_path(
    start: Sequence[float],
    target: Sequence[float],
    *,
    clearance_above_start_m: float = 0.18,
    clearance_above_target_m: float = 0.12,
) -> list[tuple[float, float, float]]:
    """Lift a released gripper vertically before translating toward stow.

    A generic clearance path may keep the start height when the body-relative
    stow target is lower. At table contact that begins horizontal travel while
    the fingers are still inside or beside the released object. This path
    always creates explicit vertical clearance first.
    """

    start_xyz = tuple(float(value) for value in start)
    target_xyz = tuple(float(value) for value in target)
    transit_z = max(
        start_xyz[2] + float(clearance_above_start_m),
        target_xyz[2] + float(clearance_above_target_m),
    )
    return [
        (start_xyz[0], start_xyz[1], transit_z),
        (target_xyz[0], target_xyz[1], transit_z),
        target_xyz,
    ]


def tcp_segment_reached(
    actual: Sequence[float],
    segment_target: Sequence[float],
    final_target: Sequence[float],
    *,
    phase: int,
    phase_count: int,
    final_tolerance_m: float = 0.025,
    clearance_final_vertical_tolerance_m: float = 0.030,
) -> bool:
    """Check a Cartesian path segment with clearance-aware tolerances."""
    horizontal_phase = phase_count - 2
    final_phase = phase_count - 1
    if phase_count in {3, 4} and phase == horizontal_phase:
        xy_error = math.hypot(
            float(actual[0]) - float(segment_target[0]),
            float(actual[1]) - float(segment_target[1]),
        )
        return xy_error <= 0.030 and float(actual[2]) >= float(final_target[2]) + 0.015
    if phase_count in {3, 4} and phase == final_phase:
        # Three-part paths terminate at a collision-clear pregrasp, not at
        # contact.  Keep lateral error tight while permitting a task-specific
        # vertical workspace-limit envelope; the subsequent one-part contact
        # move uses its separate strict tolerance.
        xy_error = math.hypot(
            float(actual[0]) - float(segment_target[0]),
            float(actual[1]) - float(segment_target[1]),
        )
        z_error = abs(float(actual[2]) - float(segment_target[2]))
        return xy_error <= 0.030 and z_error <= clearance_final_vertical_tolerance_m
    return (
        math.dist(
            tuple(float(value) for value in actual),
            tuple(float(value) for value in segment_target),
        )
        <= final_tolerance_m
    )
