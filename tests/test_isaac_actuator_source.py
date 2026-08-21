import ast
import math
from pathlib import Path

from airsign_task3.motion_geometry import (
    circular_rim_inset,
    co_moving_payload_indices,
    head_mouth_target,
    held_object_tcp_target,
    post_release_stow_path,
    retained_payload_indices,
    rotate_vector_wxyz,
    supported_assignment_reached,
    tcp_segment_reached,
    top_clearance_path,
)


def test_payload_detection_requires_lift_co_motion() -> None:
    carried = co_moving_payload_indices(
        before_positions=[(0.03, 0.0, 0.0), (0.08, 0.0, 0.0)],
        after_positions=[(0.03, 0.0, 0.09), (0.08, 0.0, 0.0)],
        tool_before=(0.0, 0.0, 0.0),
        tool_after=(0.0, 0.0, 0.09),
    )
    assert carried == {0}


def test_payload_retention_rejects_a_verified_bean_lost_in_transit() -> None:
    retained = retained_payload_indices(
        positions=[(0.03, 0.0, 0.09), (0.40, 0.0, 0.09)],
        candidate_indices=[0, 1],
        tool_position=(0.0, 0.0, 0.09),
    )
    assert retained == {0}


def test_held_object_target_preserves_the_live_grasp_offset() -> None:
    target = held_object_tcp_target(
        object_target=(-2.31, 1.75, 0.79),
        current_tcp=(-2.22, 1.62, 0.87),
        object_position=(-2.28, 1.62, 0.82),
    )
    assert all(
        math.isclose(actual, expected, abs_tol=1e-12)
        for actual, expected in zip(target, (-2.25, 1.75, 0.84), strict=True)
    )


def test_move_tcp_segment_completion_uses_the_active_path_target() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    move_tcp = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_move_tcp"
    )

    loaded_names = {
        node.id
        for node in ast.walk(move_tcp)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    assert "final_position" not in loaded_names

    assignments = [
        node
        for node in ast.walk(move_tcp)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "copy"
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "segment_position"
    ]
    assert assignments, "segment completion must copy the active path target"


def test_top_pregrasp_has_vertical_horizontal_and_descent_segments() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    move_tcp = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_move_tcp"
    )
    path_calls = [
        node
        for node in ast.walk(move_tcp)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "top_clearance_path"
    ]
    assert path_calls, "top pregrasp must use adaptive clearance path generation"

    path = top_clearance_path(
        (-4.345, -1.730, 0.565),
        (-5.164, -1.555, 0.872),
    )
    assert len(path) == 3
    assert path[0][2] == path[1][2] == 0.972


def test_nearby_pregrasp_recovery_skips_unreachable_overhead_lift() -> None:
    target = (-5.164, -1.555, 0.872)
    path = top_clearance_path((-5.154, -1.585, 0.882), target)
    assert path == [target]


def test_top_pregrasp_transit_allows_redundant_wrist_reconfiguration() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    source = source_path.read_text(encoding="utf-8")
    assert 'startswith("top_")' in source
    assert "orientation = None" in source
    assert "position_only=" in source


def test_precise_manipulation_yaw_keeps_turning_through_static_friction() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    source = source_path.read_text(encoding="utf-8")
    assert "BASE_MANIPULATION_MIN_YAW_SPEED_RADPS = 0.08" in source
    assert "abs(proportional_speed)" in source
    assert "math.copysign(" in source


def test_plate_wrist_rotation_starts_only_after_vertical_clearance() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    source = source_path.read_text(encoding="utf-8")
    assert 'primitive.orientation_hint == "top_plate"' in source
    assert "phase[side] == 0" in source
    assert "commanded_orientation = None" in source
    assert "step_targets[side] = (commanded[side], commanded_orientation)" in source
    assert "len(paths[side]) == 4" in source
    assert "phase[side] == 1" in source
    assert 'if primitive.orientation_hint == "top_plate"' in source
    assert "clearance_m=" in source
    assert "0.04" in source


def test_four_part_plate_path_uses_horizontal_and_final_envelopes() -> None:
    final_target = (-5.194, -1.488, 0.872)
    horizontal_target = (-5.194, -1.488, 0.972)
    assert tcp_segment_reached(
        (-5.215, -1.487, 0.930),
        horizontal_target,
        final_target,
        phase=2,
        phase_count=4,
    )
    assert tcp_segment_reached(
        (-5.170, -1.487, 0.825),
        final_target,
        final_target,
        phase=3,
        phase_count=4,
        clearance_final_vertical_tolerance_m=0.055,
    )


def test_plate_grasp_uses_a_live_near_rim_anchor() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    source = source_path.read_text(encoding="utf-8")
    assert 'primitive.target == "plate"' in source
    assert 'self.reader.bounds("plate")' in source
    assert "plate_bounds.maximum[0]" in source
    assert "plate_bounds.minimum[1] + plate_bounds.maximum[1]" in source
    assert "contact_or_carry" in source
    assert "PLATE_GRASP_CHORD_M = 0.075" in source
    assert "PLATE_LULA_INWARD_OFFSET_M = 0.007" in source
    assert "PLATE_GRASP_LATERAL_BIAS_M = 0.003" in source
    assert "+ PLATE_GRASP_LATERAL_BIAS_M" in source
    assert "circular_rim_inset" in source
    assert "if contact_or_carry" in source
    assert "PLATE_TCP_TO_PAD_M" in source
    assert "_quat_local_z(current_orientation)" in source
    assert 'primitive.orientation_hint == "top_plate"' in source
    assert "_quat_slerp" in source
    assert "PLATE_RIM_TILT_FRACTION" in source
    assert "PLATE_RIM_TILT_FRACTION = 0.75" in source
    assert "_quat_angular_error" in source


def test_plate_rim_orientation_is_top_down_with_world_y_physical_closing_axis() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "PLATE_RIM_QUAT_WXYZ" for target in node.targets)
    )
    assert isinstance(assignment.value, ast.Call)
    values = assignment.value.args[0]
    w, x, y, z = eval(  # noqa: S307 - evaluated AST is the repository's numeric constant
        compile(ast.Expression(values), str(source_path), "eval"),
        {"__builtins__": {}},
    )
    rotation = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
    assert abs(rotation[2][2] + 1.0) < 1e-8
    local_y_angle = math.atan2(rotation[1][1], rotation[0][1])
    physical_closing_angle = local_y_angle + 3.0 * math.pi / 4.0
    assert abs(math.cos(physical_closing_angle)) < 1e-8
    assert abs(math.sin(physical_closing_angle) - 1.0) < 1e-8


def test_top_down_yaw_family_keeps_tool_z_down() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    source = source_path.read_text(encoding="utf-8")
    assert "def _top_down_quat" in source
    for yaw in (-3.0, -1.0, 0.0, 1.0, 3.0):
        w, x, y, z = (0.0, -math.sin(0.5 * yaw), math.cos(0.5 * yaw), 0.0)
        local_z_world_z = 1 - 2 * (x * x + y * y)
        assert abs(w) < 1e-12
        assert abs(z) < 1e-12
        assert abs(local_z_world_z + 1.0) < 1e-12


def test_plate_orientation_is_selected_by_read_only_lula_scan() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    source = source_path.read_text(encoding="utf-8")
    assert "def select_reachable_plate_orientation" in source
    assert "compute_inverse_kinematics" in source
    assert "plate_orientation_selected" in source
    assert "successful_candidates" in source
    assert "position_only_succeeded" in source
    assert "self._plate_grasp_orientation" in source
    assert "PLATE_PREFERRED_TOP_DOWN_YAW_RAD" in source
    assert "preferred_yaw_error_rad" in source


def test_gripper_snapshot_records_exact_fingertip_collider_bounds() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    source = source_path.read_text(encoding="utf-8")
    assert '"PAD_OPEN_"' in source
    assert "UsdPhysics.CollisionAPI" in source
    assert "ComputeRelativeTransform" in source
    assert "live_world_corners" in source
    assert 'sample["fingertip_colliders"]' in source


def test_cup_contact_requires_secure_closure_after_first_touch() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    source = source_path.read_text(encoding="utf-8")
    assert "CUP_MIN_CONTACT_DRIVER = 0.74" in source
    assert 'if side_targets[side] == "cup" else 0.0' in source
    assert "closure_sufficient" in source
    assert "and effort >= force_threshold" in source
    assert 'primitive.orientation_hint == "top_cup"' in source
    assert '"cup_orientation_selected"' in source
    assert "float(cup_bounds.maximum[2]) + 0.10" in source
    assert "effort >= hard_force_threshold" in source


def test_horizontal_clearance_segment_uses_xy_and_safe_height() -> None:
    segment_target = (-5.164, -1.555, 0.972)
    final_target = (-5.164, -1.555, 0.872)

    assert tcp_segment_reached(
        (-5.182, -1.554, 0.893),
        segment_target,
        final_target,
        phase=1,
        phase_count=3,
    )
    assert not tcp_segment_reached(
        (-5.182, -1.554, 0.880),
        segment_target,
        final_target,
        phase=1,
        phase_count=3,
    )


def test_pregrasp_endpoint_uses_clearance_only_thirty_mm_tolerance() -> None:
    target = (-5.164, -1.555, 0.872)
    assert tcp_segment_reached(
        (-5.164, -1.555, 0.843), target, target, phase=2, phase_count=3
    )
    assert not tcp_segment_reached(
        (-5.164, -1.555, 0.841), target, target, phase=2, phase_count=3
    )
    assert not tcp_segment_reached(
        (-5.164, -1.555, 0.843), target, target, phase=0, phase_count=1
    )


def test_plate_pregrasp_can_use_wider_vertical_but_not_lateral_envelope() -> None:
    target = (-5.154, -1.488, 0.872)
    assert tcp_segment_reached(
        (-5.160, -1.494, 0.818),
        target,
        target,
        phase=2,
        phase_count=3,
        clearance_final_vertical_tolerance_m=0.055,
    )
    assert not tcp_segment_reached(
        (-5.185, -1.488, 0.872),
        target,
        target,
        phase=2,
        phase_count=3,
        clearance_final_vertical_tolerance_m=0.055,
    )
    assert not tcp_segment_reached(
        (-5.154, -1.488, 0.816),
        target,
        target,
        phase=2,
        phase_count=3,
        clearance_final_vertical_tolerance_m=0.055,
    )


def test_plate_contact_uses_calibrated_lula_reach_envelope() -> None:
    target = (-5.154, -1.488, 0.776)
    actual = (-5.115, -1.503, 0.770)
    assert tcp_segment_reached(
        actual,
        target,
        target,
        phase=0,
        phase_count=1,
        final_tolerance_m=0.045,
    )
    outside = (-5.108, -1.488, 0.776)
    assert not tcp_segment_reached(
        outside,
        target,
        target,
        phase=0,
        phase_count=1,
        final_tolerance_m=0.045,
    )


def test_only_single_segment_top_plate_motion_selects_wider_tolerance() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    source = source_path.read_text(encoding="utf-8")
    assert "final_tolerance_m=" in source
    assert 'primitive.orientation_hint == "top_plate"' in source
    assert "len(paths[side]) == 1" in source


def test_plate_contact_uses_named_reach_gate_before_physical_verification() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    source = source_path.read_text(encoding="utf-8")
    excerpt = source[source.index("final_tolerance_m=") : source.index("final_tolerance_m=") + 1200]
    assert "PLATE_CONTACT_REACH_TOLERANCE_M" in excerpt
    assert "PLATE_CONTACT_REACH_TOLERANCE_M = 0.055" in source
    assert "clearance_final_vertical_tolerance_m" in source
    assert "0.055" in source
    assert "lift_transport_verification" in source
    assert "if not lifted" in source


def test_tray_object_contact_height_uses_live_top_surface() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    source = source_path.read_text(encoding="utf-8")
    assert "TOP_GRASP_TCP_CLEARANCE_M = 0.012" in source
    assert 'primitive.target in {"cup", "bowl"}' in source
    assert 'primitive.target == "spoon"' in source
    assert "float(object_bounds.maximum[2])" in source


def test_live_plate_width_maps_to_robotiq_safe_chord() -> None:
    inset = circular_rim_inset(0.18551825, 0.075)
    assert 0.007 < inset < 0.009


def test_supported_assignment_requires_scoring_margin_and_table_height() -> None:
    assert supported_assignment_reached(
        (-2.13, 1.69, 0.75),
        (-2.13, 1.88),
        0.22,
        lowest_point_z=0.741,
    )
    assert not supported_assignment_reached(
        (-2.13, 1.665, 0.75),
        (-2.13, 1.88),
        0.22,
        lowest_point_z=0.741,
    )
    assert not supported_assignment_reached(
        (-2.13, 1.69, 0.06),
        (-2.13, 1.88),
        0.22,
        lowest_point_z=0.051,
    )


def test_wxyz_rotation_maps_local_spoon_handle_offset_to_world() -> None:
    half = math.sqrt(0.5)
    rotated = rotate_vector_wxyz(
        (half, 0.0, 0.0, half),
        (0.0, -0.06, 0.005),
    )
    assert math.dist(rotated, (0.06, 0.0, 0.005)) < 1e-12


def test_spoon_uses_live_pose_handle_anchor_and_two_pose_ik_scan() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    source = source_path.read_text(encoding="utf-8")
    assert "SPOON_HANDLE_LOCAL_OFFSET_M = (0.0, -0.085, 0.003)" in source
    assert "SPOON_HANDLE_LATERAL_BIAS_SEQUENCE_M = (-0.007, -0.0085, -0.0055, 0.0)" in source
    assert "SPOON_SIDE_GRASP_TCP_RETRACT_M = 0.035" in source
    assert "SPOON_SIDE_GRASP_TCP_HEIGHT_M = 0.020" in source
    assert "SPOON_SIDE_GRASP_ANGLE_RAD = math.radians(30.0)" in source
    assert "SPOON_SIDE_GRASP_RETRY_STEP_RAD = math.radians(15.0)" in source
    assert "SPOON_MIN_SIDE_GRASP_ANGLE_RAD = math.radians(10.0)" in source
    assert "SPOON_BILATERAL_CONTACT_CONFIRM_STEPS = 3" in source
    assert "SPOON_MIN_CONTACT_DRIVER = 0.78" in source
    assert "SPOON_GRASP_MAX_DISPLACEMENT_M = 0.08" in source
    assert "handle_lateral_xy * self._spoon_handle_lateral_bias_m" in source
    assert '"spoon_grasp_bias_advanced"' in source
    assert "handle_direction_xy * SPOON_SIDE_GRASP_TCP_RETRACT_M" in source
    assert "position[2] += SPOON_SIDE_GRASP_TCP_HEIGHT_M" in source
    spoon_branch = source.index('primitive.target == "spoon"')
    cup_branch = source.index('if primitive.target == "plate"', spoon_branch)
    assert 'self.reader.bounds("spoon")' not in source[spoon_branch:cup_branch]
    assert 'self.reader.pose("spoon")' in source
    assert "rotate_vector_wxyz" in source
    assert 'primitive.orientation_hint == "top_spoon"' in source
    assert "def select_reachable_spoon_orientation" in source
    assert "np.linspace(SPOON_SIDE_GRASP_ANGLE_RAD, 0.0, 13)" in source
    assert "pregrasp_solution, pregrasp_succeeded" in source
    assert "contact_solution, contact_succeeded" in source
    assert "maximum_angle_rad=self._spoon_grasp_angle_cap_rad" in source
    assert 'primitive.label == "approach spoon"' in source
    assert '"spoon_grasp_angle_reduced"' in source
    assert "self.rmp.select_reachable_spoon_orientation(" in source
    assert "contact_scan_position=contact_position.tolist()" in source
    assert '"spoon_orientation_selected"' in source
    assert "preferred_grasp_yaw_rad" in source
    assert "preferred_yaw_rad=_wrap_to_pi(" in source
    assert "+ math.pi / 4.0" in source
    assert 'grasp_mode="ik_scanned_side"' in source
    assert "spoon_orientation_wxyz=list(spoon_pose.orientation_wxyz)" in source
    assert "def _spoon_contact_report" in source
    assert '"spoon_bilateral_contact_confirmed"' in source
    assert '"spoon_grasp_displacement_exceeded"' in source
    assert '{"left", "right"} <= spoon_contact_sides' in source
    assert "finals[side] >= SPOON_MIN_CONTACT_DRIVER" in source
    assert "SPOON_STALLED_CONTACT_DRIVER = 0.74" in source
    assert "SPOON_STALLED_CONTACT_EFFORT_NM = 2.0" in source
    assert "def spoon_side_closed" in source
    assert "peak_efforts[side] >= SPOON_STALLED_CONTACT_EFFORT_NM" in source
    assert "all(spoon_side_closed(side) for side in sides)" in source
    assert "spoon_contact_pairs=[" in source
    assert '"world_corners"' in source
    assert "spoon_forward_xy = -handle_offset[:2]" in source
    assert '"placement_object_target_compensated"' in source
    assert 'primitive.metadata.get("placement_object_xy")' in source
    assert '"scoop_payload_verification"' in source
    assert 'mode="co_motion"' in source
    assert "co_moving_payload_indices" in source
    assert "retained_payload_indices" in source
    assert 'primitive.metadata["minimum_beans"]' in source


def test_live_eye_geometry_calibrates_a_table_facing_mouth_target() -> None:
    mouth = head_mouth_target(
        (-2.0, 2.2, 0.74659),
        [(-2.01, 2.20, 0.923), (-1.99, 2.20, 0.923)],
    )
    assert math.isclose(mouth[2], 0.878, abs_tol=1e-12)
    assert mouth[0] < -2.0
    assert mouth[1] < 2.2

    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    source = source_path.read_text(encoding="utf-8")
    assert 'descendant_geometry_samples("head")' in source
    assert '"feeding_target_calibrated"' in source
    assert 'if name == "mouth_standoff"' in source
    assert 'if name == "mouth_retract"' in source
    assert "HEAD_MOUTH_STANDOFF_M = 0.34" in source
    assert "HEAD_MOUTH_HOLD_M = 0.24" in source
    assert "HEAD_MOUTH_RETRACT_M = 0.40" in source
    assert 'primitive.orientation_hint == "spoon_level"' in source
    assert '"spoon_level_orientation_selected"' in source
    assert 'self.reader.pose("spoon")' in source
    assert "self.store.telemetry.safety.head_zone_active" in source


def test_loaded_carries_physically_retreat_from_dining_source_station() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    source = source_path.read_text(encoding="utf-8")
    assert 'primitive.metadata.get("retreat_from_station")' in source
    assert '"loaded_source_station_retreat_start"' in source
    assert "self._withdraw_base(" in source


def test_lower_motion_accepts_only_stable_supported_scored_placement() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    source = source_path.read_text(encoding="utf-8")
    assert "supported_assignment_reached" in source
    assert "PLACEMENT_STABLE_STEPS = 12" in source
    assert "PLACEMENT_MAX_STEP_M = 0.003" in source
    assert "PLACEMENT_SUPPORT_Z_MIN_M = 0.70" in source
    assert "PLACEMENT_SUPPORT_Z_MAX_M = 0.82" in source
    assert "PLACEMENT_SCORE_MARGIN_M = 0.01" in source
    assert '"placement_contact_verification"' in source
    assert 'completion_reason="stable_supported_assignment"' in source
    assert 'primitive.orientation_hint == "place_spoon"' in source
    assert '"spoon_placement_corridor"' in source
    assert "if placement_object_name is not None:" in source


def test_post_release_stow_forces_vertical_clearance_before_translation() -> None:
    path = post_release_stow_path(
        (-2.22, 1.70, 0.815),
        (-2.32, 0.95, 0.68),
    )
    expected = [
        (-2.22, 1.70, 0.995),
        (-2.32, 0.95, 0.995),
        (-2.32, 0.95, 0.68),
    ]
    assert all(
        math.dist(actual, wanted) < 1e-12
        for actual, wanted in zip(path, expected)
    )
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    source = source_path.read_text(encoding="utf-8")
    assert "post_release_stow_path" in source
    assert 'primitive.orientation_hint == "transit_stow"' in source


def test_sink_navigation_uses_the_right_arm_manipulation_yaw() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    source = source_path.read_text(encoding="utf-8")
    station_set = source[source.index('"plate_seat",') : source.index(
        "if not self._orient_base_for_right_arm", source.index('"plate_seat",')
    )]
    assert '"recycling",' in station_set
    assert '"sink",' in station_set


def test_navigation_retry_rank_advances_after_mid_route_failures() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    source = source_path.read_text(encoding="utf-8")
    assert "def _execute_navigation(" in source
    assert "previous_failures = self.navigation_failures.get(target, 0)" in source
    assert "self.navigation_failures[target] = previous_failures + 1" in source
    assert "PrimitiveKind.NAVIGATE: lambda: self._execute_navigation(" in source


def test_portal_yaw_occurs_after_egress_at_the_entry_waypoint() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    source = source_path.read_text(encoding="utf-8")
    navigate = source.index("def _navigate(")
    next_waypoint = source.index("next_waypoint =", navigate)
    portal_yaw = source.index("if not self._orient_base_for_portal_transit", navigate)
    assert portal_yaw > next_waypoint
    assert '"portal_entry_orientation_start"' in source
    assert '"portal_entry_orientation_complete"' in source


def test_manipulation_yaw_accepts_the_measured_static_friction_floor() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    source = source_path.read_text(encoding="utf-8")
    assert "BASE_MANIPULATION_YAW_TOLERANCE_RAD = math.radians(10.0)" in source
    assert "BASE_PORTAL_YAW_TOLERANCE_RAD = math.radians(12.0)" in source


def test_elevated_pregrasp_has_separate_noncontact_reach_tolerance() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    source = source_path.read_text(encoding="utf-8")
    assert "PREGRASP_REACH_TOLERANCE_M = 0.040" in source
    assert '(primitive.orientation_hint or "").startswith("top_")' in source
    assert "and primitive.offset_xyz[2] >= 0.08" in source
    assert "PLATE_CONTACT_REACH_TOLERANCE_M" in source
    assert "BOWL_INTERNAL_CONTACT_TOLERANCE_M" in source


def test_lift_tolerance_matches_measured_solver_floor_with_retention_checks() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    source = source_path.read_text(encoding="utf-8")
    assert "LIFT_REACH_TOLERANCE_M = 0.035" in source
    assert "LIFT_EARLY_ACCEPT_HEIGHT_M = 0.080" in source
    assert '"loaded_object_retention_failed"' in source


def test_object_lift_is_vertical_and_bounded() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    source = source_path.read_text(encoding="utf-8")
    assert 'primitive.label.startswith("lift ")' in source
    assert "position[:2] = current_position[:2]" in source
    assert "PLATE_LIFT_DELTA_M = 0.11" in source
    assert "SPOON_LIFT_DELTA_M = 0.07" in source
    assert "TRAY_OBJECT_LIFT_DELTA_M = 0.16" in source
    assert "LIFT_REACH_TOLERANCE_M = 0.035" in source
    assert "LIFT_EARLY_ACCEPT_HEIGHT_M = 0.080" in source
    assert "LIFT_EARLY_ACCEPT_STABLE_STEPS = 12" in source
    assert 'completion_reason="stable_object_height"' in source
    assert 'if primitive.target == "plate"' in source


def test_loaded_object_retracts_from_supply_table_before_base_motion() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    source = source_path.read_text(encoding="utf-8")
    assert 'primitive.label.startswith("retract ")' in source
    assert '"from supply table" in primitive.label' in source
    assert 'retreat_xy = np.asarray((1.0, 0.0), dtype=float)' in source
    assert '"loaded_transit_stow"' in source
    assert '(0.18, -0.20) if loaded_stow else (0.22, -0.10)' in source
    assert 'root_position[2] + (0.90 if loaded_stow else 0.68)' in source
    assert 'primitive.orientation_hint == "loaded_transit_stow"' in source
    assert 'orientation = current_orientation.copy()' in source
    assert 'loaded_stow and primitive.target in {"bowl", "plate"}' in source
    assert '(0.12, 0.0, 0.03)' in source
    assert "speed = min(speed, 0.06)" in source


def test_empty_arm_uses_body_relative_clearance_stow_after_placement() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    source = source_path.read_text(encoding="utf-8")
    assert 'primitive.orientation_hint == "transit_stow"' in source
    assert '(0.18, -0.20) if loaded_stow else (0.22, -0.10)' in source
    assert "root_position[2] + (0.90 if loaded_stow else 0.68)" in source
    assert "post_release_stow_path" in source


def test_bowl_grasp_expands_physical_pads_inside_live_rim() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    source = source_path.read_text(encoding="utf-8")
    assert "BOWL_INTERNAL_TCP_CLEARANCE_M = -0.030" in source
    assert "BOWL_INTERNAL_CONTACT_TOLERANCE_M = 0.050" in source
    assert "BOWL_INTERNAL_MAX_CONTACT_DRIVER = 0.15" in source
    assert "current <= BOWL_INTERNAL_MAX_CONTACT_DRIVER" in source
    assert '"top_bowl_internal"' in source
    assert '"bowl_internal"' in source
    assert "orientation = PLATE_RIM_QUAT_WXYZ.copy()" in source
    assert 'primitive.orientation_hint == "bowl_internal"' in source
    assert "object_bounds.minimum[0] + object_bounds.maximum[0]" in source
    assert "object_bounds.minimum[1] + object_bounds.maximum[1]" in source
    assert 'primitive.metadata.get("internal_spread", False)' in source
    assert "internal_release" in source


def test_loaded_stow_aborts_if_a_physically_carried_object_falls() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    source = source_path.read_text(encoding="utf-8")
    assert "LOADED_OBJECT_MIN_HEIGHT_M = 0.60" in source
    assert "LOADED_OBJECT_MAX_DROP_M = 0.12" in source
    assert '"loaded_object_retention_failed"' in source
    assert "contact_motion = (closing and not internal_release) or internal_spread" in source
    assert 'if primitive.label.startswith("release ")' in source


def test_spoon_strong_grip_waits_for_hard_force_confirmation() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    source = source_path.read_text(encoding="utf-8")
    assert 'primitive.metadata.get("strong_grip", False)' in source
    assert "if strong_grip" in source
    assert "hard_force_counts[side]" in source
    assert "strong_grip=strong_grip" in source
    assert "return contacted if strong_grip else contacted or reached" in source


def test_loaded_navigation_ends_an_episode_if_payload_is_lost() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    source = source_path.read_text(encoding="utf-8")
    assert "def _carried_objects_retained" in source
    assert '"navigation_object_retention_failed"' in source
    assert "self.unrecoverable_failure_reason" in source
    assert 'primitive.metadata.get("carried_objects", ())' in source
    assert "loaded_navigation = bool(carried_objects)" in source
    assert "if loaded_navigation" in source


def test_bimanual_pair_targets_each_live_object_and_retracts_from_table() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    source = source_path.read_text(encoding="utf-8")
    assert 'paired_names = (' in source
    assert 'paired_index = 0 if side is Arm.LEFT else 1' in source
    assert 'self._target_position(paired_names[paired_index])' in source
    assert '"from supply table" in primitive.label' in source
    assert 'np.asarray(root_position[:2], dtype=float) - target_center[:2]' in source
    assert 'for object_name in (primitive.target or "").split("+")' in source
    assert 'if all(' in source
    assert 'for height in lifted_heights.values()' in source
    assert 'paired_targets[0 if side is Arm.LEFT else 1]' in source
    assert 'CUP_MIN_CONTACT_DRIVER if side_targets[side] == "cup" else 0.0' in source


def test_thin_plate_has_a_physical_object_specific_lift_gate() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    source = source_path.read_text(encoding="utf-8")
    assert 'name: 0.025 if name == "plate" else 0.035' in source
    assert '"minimum_lift_m": minimum_lifts[name]' in source


def test_head_safety_uses_live_mouth_geometry_and_a_hard_tcp_stop() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    source = source_path.read_text(encoding="utf-8")
    assert "HEAD_MOUTH_HOLD_M = 0.24" in source
    assert "HEAD_MOUTH_TCP_STOP_M = 0.035" in source
    assert 'if name == "mouth_hold"' in source
    assert "commanded[side] - self._head_mouth_position" in source
    assert "distance_to_head < HEAD_MOUTH_TCP_STOP_M" in source


def test_plate_can_be_carried_on_the_physically_grasped_empty_tray() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    source = source_path.read_text(encoding="utf-8")
    assert "TRAY_GRASP_X_INSET_M = 0.060" in source
    assert "TRAY_GRASP_Y_INSET_M = 0.018" in source
    assert 'if primitive.target == "tray"' in source
    assert 'primitive.orientation_hint in {"top_plate", "top_tray"}' in source


def test_actuator_long_loops_observe_shutdown_request() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    source = source_path.read_text(encoding="utf-8")
    assert source.count("self.store.stop_requested") >= 9


def test_supply_navigation_preserves_clearance_and_physically_holds_base() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    source = source_path.read_text(encoding="utf-8")
    assert "BASE_STATION_STANDOFF_M = 0.42" in source
    assert "BASE_MAX_SUPPLY_MANIPULATION_REACH_M = 0.75" in source
    assert "BASE_SUPPLY_STATION_ACCEPTANCE_M = 0.06" in source
    assert "BASE_LOADED_FOOTPRINT_CLEARANCE_M = 0.52" in source
    assert "loaded_navigation = bool(carried_objects)" in source
    assert 'loaded_station = target_name.endswith("_seat")' in source
    assert "base_clearance + attempt * BASE_RETRY_CLEARANCE_STEP_M" in source
    assert "DINING_STATION_STANDOFF_M = 0.60" in source
    assert "COUNTER_STATION_STANDOFF_M = 0.60" in source
    assert 'counter_station = target_name in {"recycling", "sink"}' in source
    assert "prefer_outermost=counter_station" in source
    assert "DINING_STATION_ACCEPTANCE_M = 0.03" in source
    assert "DINING_CORRIDOR_ENTRY_TOLERANCE_M = 0.03" in source
    assert "BASE_LOADED_SPEED_MPS = 0.08" in source
    assert "BASE_MAX_DINING_MANIPULATION_REACH_M = 1.10" in source
    assert "align_horizontal_corridor" in source
    assert "center_y=float(candidate[1])" in source
    assert "horizontal_corridor_entry_index" in source
    assert '"dining_corridor_orientation_start"' in source
    assert '"dining_corridor_orientation_complete"' in source
    assert "max(0, corridor_entry_index - 2)" in source
    assert "base_position_hint: np.ndarray | None = None" in source
    assert "base_position_hint=goal" in source
    assert "aiming_base_xy" in source
    assert "right_arm_facing_yaw" in source
    assert "mid-route turn, so release that hold" in source
    assert source.count("self._base_hold_enabled = False") >= 6
    assert "begins_dining_corridor" in source
    assert "abs(float(next_waypoint[0] - waypoint[0])) >= 0.50" in source
    assert "abs(float(next_waypoint[1] - waypoint[1])) <= 0.15" in source
    assert "min(limits.base_speed_mps, BASE_LOADED_SPEED_MPS)" in source
    assert "CARRY_NAVIGATION_TIMEOUT_S = 180.0" in source
    assert "DINING_STATION_FINAL_ADVANCE_M = 0.28" in source
    assert "DINING_STATION_FINAL_SPEED_MPS = 0.03" in source
    assert "DINING_STATION_FINAL_TIMEOUT_S = 32.0" in source
    assert "def _advance_dining_station" in source
    assert "dining_station: bool = False" in source
    assert "dining_station = loaded_station or dining_station" in source
    assert 'primitive.metadata.get("dining_station", False)' in source
    assert 'primitive.metadata.get(\n                                "position_tolerance_m"' in source
    assert 'primitive.metadata.get("dither_timeout_s", 20.0)' in source
    assert '"dining_station_advance_complete"' in source
    assert '"dining_station_advance_incomplete"' in source
    assert "PLACEMENT_LOST_LOWEST_Z_M = 0.55" in source
    assert '"placement_object_lost"' in source
    assert "self.defer_scope_reason" in source
    assert 'primitive.target in {"bowl", "plate"}' in source
    assert '"payload lost during loaded arm stow: "' in source
    assert "advanced = self._advance_dining_station(target, limits)" in source
    assert "if dining_station:" in source
    assert '"nearby_dining_station_accepted"' in source
    assert '"dining_station_final_advance_skipped"' in source
    assert 'primitive.metadata.get("dining_final_advance", True)' in source
    assert 'primitive.metadata["nearby_station_acceptance_m"]' in source
    assert 'primitive.metadata.get("hold_current_position")' in source
    assert '"recovery_payload_released"' in source
    assert '"recovery_payload_unmet"' in source
    assert "def _retained_bowl_bean_count" in source
    assert "BASE_MANIPULATION_CREEP_M" not in source
    assert "def _creep_base_toward" not in source
    assert "- PLATE_LULA_INWARD_OFFSET_M" in source
    assert "def _apply_base_hold" in source
    assert "self._apply_base_hold()" in source
    assert "joint_velocities=np.zeros(len(self.drive_ids)" in source
    assert "self._base_hold_enabled = False" in source
    assert "clearance_egress_point" in source
    assert "planning_obstacles = list(self.base_obstacles)" in source
    assert "BASE_CLEARANCE_EGRESS_MARGIN_M = 0.22" in source
    assert "margin=BASE_CLEARANCE_EGRESS_MARGIN_M" in source
    assert "BASE_CLEARANCE_EGRESS_ACCEPTANCE_M = 0.16" in source
    assert "if needs_egress and waypoint_index == 1" in source
    assert "if self._base_hold_enabled:" in source
    assert "BASE_STALL_TIMEOUT_S = 12.0" in source
    assert "BASE_DEFAULT_WITHDRAW_M = 0.18" in source
    assert "BASE_LOADED_STATION_MIN_WITHDRAW_M = 0.35" in source
    assert "BASE_LOADED_STATION_MAX_WITHDRAW_M = 1.00" in source
    assert "BASE_LOADED_WITHDRAW_MARGIN_M = 0.10" in source
    assert "BASE_WITHDRAW_TIMEOUT_S = 22.0" in source
    assert "def _loaded_station_withdraw_distance" in source
    assert "clearance + BASE_LOADED_WITHDRAW_MARGIN_M" in source
    assert "if dining_station:" in source
    assert '"base_drive_progress"' in source
    assert "BASE_WHEEL_VELOCITY_KD = 10.0" in source
    assert "BASE_WHEEL_MAX_EFFORT_NM = 20.0" in source
    assert 'switch_dof_control_mode(index, "velocity")' in source
    assert 'set_effort_modes(\n            "force"' in source
    assert "max_efforts_readback" in source
    assert '"base_drive_configured"' in source
    assert "def _orient_base_for_right_arm" in source
    assert "BASE_MANIPULATION_YAW_SPEED_RADPS = 0.25" in source
    assert "BASE_MANIPULATION_YAW_TOLERANCE_RAD = math.radians(10.0)" in source
    assert "yaw_tolerance_rad: float | None = None" in source
    assert 'primitive.metadata["manipulation_yaw_tolerance_deg"]' in source
    assert "BASE_MANIPULATION_YAW_TIMEOUT_S = 180.0" in source
    assert "BASE_PORTAL_YAW_TOLERANCE_RAD = math.radians(12.0)" in source
    assert "desired_yaw = right_arm_facing_yaw(" in source
    assert "def _orient_base_for_portal_transit" in source
    assert '"base_portal_orientation_complete"' in source
    assert '"orienting base for portal transit: "' in source
    assert "manipulation_reach = float(np.linalg.norm(candidate - target))" in source
    assert "manipulation_reach > BASE_MAX_SUPPLY_MANIPULATION_REACH_M" in source
    assert "ranked_routes = sorted(routes, key=lambda item: (item[0], item[1]))" in source
    assert "selected_route_rank = min(attempt, len(ranked_routes) - 1)" in source
    plate_branch = 'if target_name == "plate" and not dining_station:'
    plate_retry_excerpt = source[
        source.index(plate_branch):
        source.index('if target_name in {', source.index(plate_branch))
    ]
    assert "self.navigation_failures[target_name] = attempt + 1" in plate_retry_excerpt
    assert "0 if advanced else attempt + 1" in source
    assert 'candidate_routes=route_diagnostics' in source
    assert 'rejected_candidates=route_errors' in source


def test_plate_station_uses_bounded_face_on_physical_advance() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    source = source_path.read_text(encoding="utf-8")
    assert "PLATE_STATION_FINAL_ADVANCE_M = 0.095" in source
    assert "PLATE_STATION_FINAL_SPEED_MPS = 0.035" in source
    assert "PLATE_STATION_FINAL_TIMEOUT_S = 5.0" in source
    assert "PLATE_STATION_REACH_MARGIN_M = 0.020" in source
    assert '"plate_station_advance_skipped"' in source
    assert "def _advance_plate_station" in source
    assert 'target_name == "plate"' in source
    assert "_compute_drive_targets" in source
    assert "plate_station_advance_complete" in source


def test_rmp_uses_pinned_task3_franka_hand_model() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    source = source_path.read_text(encoding="utf-8")
    assert 'benchmark_root / "task3_isaacsim"' in source
    assert '"mobile_fr3_duo_v0_2_franka_hand.urdf"' in source
    assert '"left_fr3v2_hand_tcp"' in source
    assert '"right_fr3v2_hand_tcp"' in source
    controller_source = source[
        source.index("class BimanualRmpController") :
        source.index("class IsaacPhysicalActuator")
    ]
    assert 'benchmark_root / "task2_isaacsim"' not in controller_source


def test_signal_handler_wakes_paused_actuator_for_shutdown() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_native.py"
    source = source_path.read_text(encoding="utf-8")
    assert "store.request_stop()" in source
    assert 'store.queue_command("resume")' in source


def test_dynamic_ground_truth_uses_read_only_physx_views() -> None:
    state_source = (
        Path(__file__).parents[1] / "airsign_task3" / "isaac_state.py"
    ).read_text(encoding="utf-8")
    native_source = (
        Path(__file__).parents[1] / "airsign_task3" / "isaac_native.py"
    ).read_text(encoding="utf-8")
    assert "def bind_live_physics" in state_source
    assert "UsdPhysics.RigidBodyAPI" in state_source
    assert "get_world_poses(clone=True, usd=False)" in state_source
    assert '"/World/Scene/CoffeeBeans/Bean_.*"' in state_source
    assert "set_world_poses" not in state_source
    assert "RigidPrim, SingleArticulation" in native_source
    assert "state_reader.bind_live_physics(RigidPrim)" in native_source


def test_gripper_requires_travel_or_sustained_contact_and_lift_is_verified() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_actuator.py"
    source = source_path.read_text(encoding="utf-8")
    assert "stall_counts" in source
    assert "gripper_motion_complete" in source
    assert "return contacted if strong_grip else contacted or reached" in source
    assert "lift_transport_verification" in source
    assert "if not lifted" in source
    assert "gripper_geometry_snapshot" in source
    assert 'phase="before"' in source
    assert 'phase="after"' in source
    assert "articulation_live_pose_api" in source
    assert "articulation_live_pose_probe" in source
    assert "get_link_transforms" in source
    assert "closing =" in source
    assert "target = 0.8 * (1.0 - requested_opening)" in source
    assert "contacted = contact_motion" in source
    assert "force_counts" in source
    assert "force_limited" in source
    assert "def _apply_gripper_hold" in source
    assert "self._apply_gripper_hold()" in source


def test_robot_pad_physics_material_is_bound_before_world_reset() -> None:
    source = (
        Path(__file__).parents[1] / "airsign_task3" / "isaac_native.py"
    ).read_text(encoding="utf-8")
    assert "def _configure_gripper_pad_material" in source
    assert "CreateStaticFrictionAttr(1.4)" in source
    assert "CreateDynamicFrictionAttr(1.2)" in source
    assert '"physics"' in source
    assert source.index("gripper_pad_links = _configure_gripper_pad_material") < source.index(
        "world.reset()"
    )


def test_ready_pose_starts_both_physical_grippers_open() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_native.py"
    source = source_path.read_text(encoding="utf-8")
    assert '"left_right_finger_joint": 0.0' in source
    assert '"right_right_finger_joint": 0.0' in source
    assert '"left_robotiq_85_left_finger_tip_joint": 0.0' in source
    assert '"right_robotiq_85_right_finger_tip_joint": 0.0' in source


def test_visible_calibration_is_short_but_physically_actuated() -> None:
    source_path = Path(__file__).parents[1] / "airsign_task3" / "isaac_native.py"
    source = source_path.read_text(encoding="utf-8")
    assert 'parser.add_argument("--calibration-steps", type=int, default=60)' in source
    assert "total_steps = max(12, int(total_steps))" in source
    assert "controller.apply_action" in source
    assert "render_step = step % 4 == 0" in source
    assert "world.step(render=render_step)" in source
