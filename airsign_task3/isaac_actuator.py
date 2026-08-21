from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .isaac_state import IsaacStateReader
from .live_scorer import AssignmentTarget, OfficialLiveScorer
from .motion_geometry import (
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
from .planning import (
    RectObstacle,
    align_horizontal_corridor,
    clearance_egress_point,
    collision_cleared_waypoints,
    horizontal_corridor_entry_index,
    right_arm_facing_yaw,
    station_goal_candidates,
)
from .policy import Arm, PhysicalActuator, Primitive, PrimitiveKind
from .runtime import RuntimeStore
from .safety import SafetyLimits
from .types import Lifecycle, Stage, Substate


DRIVE_MODULES = (
    ("tmrv0_2_joint_0", "tmrv0_2_joint_1", 0.3, -0.2),
    ("tmrv0_2_joint_2", "tmrv0_2_joint_3", -0.3, 0.2),
)
WHEEL_RADIUS_M = 0.05
MAX_WHEEL_SPEED_RADPS = 18.0
BASE_FOOTPRINT_CLEARANCE_M = 0.38
BASE_LOADED_FOOTPRINT_CLEARANCE_M = 0.52
BASE_RETRY_CLEARANCE_STEP_M = 0.01
BASE_MAX_CLEARANCE_M = 0.56
# The static dining-table envelope ends near y=1.54. A 10 cm mathematical
# egress margin left the articulated footprint touching it at y~=1.08; use the
# measured contact-free lane instead of relying on a forceful retry.
BASE_CLEARANCE_EGRESS_MARGIN_M = 0.22
# The dining booth's physical corridor bottoms out near y=1.05 for the full
# articulated footprint, while the conservative inflated-map egress point is
# y=0.94.  Once the base is within this bound, the following westbound
# waypoint moves it away from the table and clears the corridor.  Requiring
# the impossible final 11 cm of pure lateral travel caused repeated stalls
# after a successful cup placement.
BASE_CLEARANCE_EGRESS_ACCEPTANCE_M = 0.16
BASE_STATION_STANDOFF_M = 0.42
BASE_MAX_SUPPLY_MANIPULATION_REACH_M = 0.75
BASE_SUPPLY_STATION_ACCEPTANCE_M = 0.06
DINING_STATION_STANDOFF_M = 0.60
COUNTER_STATION_STANDOFF_M = 0.60
DINING_STATION_ACCEPTANCE_M = 0.03
DINING_CORRIDOR_ENTRY_TOLERANCE_M = 0.03
BASE_LOADED_SPEED_MPS = 0.08
BASE_MAX_DINING_MANIPULATION_REACH_M = 1.10
CARRY_NAVIGATION_TIMEOUT_S = 180.0
# The loaded arm pushes the passive mobile base back by about 70 mm during
# table contact. A 280 mm slow face-on advance preserves the post-recoil arm
# workspace while the base/table collision response remains authoritative.
DINING_STATION_FINAL_ADVANCE_M = 0.28
DINING_STATION_FINAL_SPEED_MPS = 0.03
DINING_STATION_FINAL_TIMEOUT_S = 32.0
# Preserve the collision-cleared diagonal route to the supply table, then make
# a slow face-on correction only after the base is aligned with the plate.
# This closes the measured Robotiq reach gap without cutting the table corner.
PLATE_STATION_FINAL_ADVANCE_M = 0.095
PLATE_STATION_FINAL_SPEED_MPS = 0.035
PLATE_STATION_FINAL_TIMEOUT_S = 5.0
PLATE_STATION_REACH_MARGIN_M = 0.020
BASE_PORTAL_LONGITUDINAL_CLEARANCE_M = 1.05
BASE_STALL_TIMEOUT_S = 12.0
BASE_DEFAULT_WITHDRAW_M = 0.18
BASE_LOADED_STATION_MIN_WITHDRAW_M = 0.35
BASE_LOADED_STATION_MAX_WITHDRAW_M = 1.00
BASE_LOADED_WITHDRAW_STEP_M = 0.05
BASE_LOADED_WITHDRAW_MARGIN_M = 0.10
BASE_WITHDRAW_TIMEOUT_S = 22.0
BASE_WHEEL_VELOCITY_KD = 10.0
BASE_WHEEL_MAX_EFFORT_NM = 20.0
BASE_MANIPULATION_YAW_SPEED_RADPS = 0.25
# A purely proportional command falls below the measured static-friction
# threshold near the spoon's precise final yaw. Keep a small signed floor so
# the base can cross the final few degrees instead of timing out at ~3.9 deg.
BASE_MANIPULATION_MIN_YAW_SPEED_RADPS = 0.08
BASE_MANIPULATION_YAW_TOLERANCE_RAD = math.radians(10.0)
BASE_PORTAL_YAW_TOLERANCE_RAD = math.radians(12.0)
# The swerve modules first have to steer under a loaded, high-friction base;
# the measured supply turn can need roughly 50 seconds even though commanded
# yaw speed is bounded at 0.25 rad/s.  Leave margin for the first steering set.
# Loaded in-place yaw is deliberately slow with the conservative wheel limits;
# the measured bowl carry can need roughly 150 wall seconds for the worst-case
# turn while another Isaac renderer is active.
BASE_MANIPULATION_YAW_TIMEOUT_S = 180.0
# The installed Robotiq is clocked 135 degrees around the Task 3 Franka-hand
# Lula TCP.  A -135 degree top-down tool yaw therefore makes the *physical*
# finger closing axis tangent to the plate rim (world Y at the supply table).
PLATE_PREFERRED_TOP_DOWN_YAW_RAD = -3.0 * math.pi / 4.0
PLATE_RIM_QUAT_WXYZ = np.asarray(
    (0.0, 0.9238795325112867, 0.38268343236508984, 0.0), dtype=float
)
# The position-only pregrasp reaches a wrist branch roughly 66 degrees from
# top-down.  Rotate most of that residual only during the short contact
# approach so the fingers straddle the rim chord rather than sweep the table.
PLATE_RIM_TILT_FRACTION = 0.75
# Keep the circular plate chord inside the Robotiq's measured 85 mm opening.
# A 75 mm chord leaves contact margin without demanding a diameter-spanning
# grasp.  The corresponding rim inset is derived from the live plate bounds.
PLATE_GRASP_CHORD_M = 0.075
# The measured approach finishes about 10 mm outward of its commanded Lula
# target.  A 7 mm command-space inset compensates that residual while leaving
# the physical pad centers on the computed 75 mm chord instead of the wider
# inner chord that let one jaw cam the plate away from the other.
PLATE_LULA_INWARD_OFFSET_M = 0.007
# With the jaws pre-shaped to driver 0.04, a +3 mm world-Y bias makes the two
# inner pad faces equidistant from the live circular chord.
PLATE_GRASP_LATERAL_BIAS_M = 0.003
# This gate advances to force-observed closure; it does not declare a grasp.
# Contact and transport still require gripper effort/stall and object lift.
PLATE_CONTACT_REACH_TOLERANCE_M = 0.055
# Cup, bowl, and spoon rigid-body origins are authored near their support
# surfaces rather than at the upper grasp surface.  Place the Lula TCP just
# above the live top bound so the downward fingers straddle the object without
# commanding the wrist through the object or tray.
TOP_GRASP_TCP_CLEARANCE_M = 0.012
PREGRASP_REACH_TOLERANCE_M = 0.040
# Target the wider handle neck while staying clear of the spoon bowl.
SPOON_HANDLE_LOCAL_OFFSET_M = (0.0, -0.085, 0.003)
SPOON_HANDLE_LATERAL_BIAS_SEQUENCE_M = (-0.007, -0.0085, -0.0055, 0.0)
SPOON_SIDE_GRASP_TCP_RETRACT_M = 0.035
SPOON_SIDE_GRASP_TCP_HEIGHT_M = 0.020
SPOON_SIDE_GRASP_ANGLE_RAD = math.radians(30.0)
SPOON_SIDE_GRASP_RETRY_STEP_RAD = math.radians(15.0)
SPOON_MIN_SIDE_GRASP_ANGLE_RAD = math.radians(10.0)
SPOON_BILATERAL_CONTACT_CONFIRM_STEPS = 3
SPOON_MIN_CONTACT_DRIVER = 0.78
SPOON_GRASP_MAX_DISPLACEMENT_M = 0.08
# The mouth target is derived from the live eye assembly.  These offsets keep
# the TCP on the table-facing side of the face, with a collision-clear approach
# and retract segment.  The spoon itself is the only body intended to bridge
# the remaining mouth standoff.
HEAD_MOUTH_STANDOFF_M = 0.34
HEAD_MOUTH_HOLD_M = 0.24
HEAD_MOUTH_RETRACT_M = 0.40
HEAD_MOUTH_STANDOFF_Z_M = 0.08
HEAD_MOUTH_HOLD_Z_M = 0.06
HEAD_MOUTH_RETRACT_Z_M = 0.10
HEAD_MOUTH_TCP_STOP_M = 0.035
# Insert closed fingers into the upper bowl cavity, then expand them against
# the inner walls.  This avoids the outward-sloping exterior camming a pinch up.
BOWL_INTERNAL_TCP_CLEARANCE_M = -0.030
BOWL_INTERNAL_CONTACT_TOLERANCE_M = 0.050
# A brief force spike can occur before both pads have reached the bowl's
# opposing inner walls. Require the measured driver to reach the empirically
# stable spread from the successful carry, unless the hard effort limit fires.
BOWL_INTERNAL_MAX_CONTACT_DRIVER = 0.15
BOWL_INTERNAL_HARD_FORCE_CONFIRM_STEPS = 2
LOADED_OBJECT_MIN_HEIGHT_M = 0.60
LOADED_OBJECT_MAX_DROP_M = 0.12
PLATE_LIFT_DELTA_M = 0.11
SPOON_LIFT_DELTA_M = 0.07
TRAY_OBJECT_LIFT_DELTA_M = 0.16
LIFT_REACH_TOLERANCE_M = 0.035
LIFT_EARLY_ACCEPT_HEIGHT_M = 0.080
LIFT_EARLY_ACCEPT_STABLE_STEPS = 12
PLACEMENT_STABLE_STEPS = 12
PLACEMENT_MAX_STEP_M = 0.003
PLACEMENT_SUPPORT_Z_MIN_M = 0.70
PLACEMENT_SUPPORT_Z_MAX_M = 0.82
PLACEMENT_SCORE_MARGIN_M = 0.01
PLACEMENT_LOST_LOWEST_Z_M = 0.55
CUP_MIN_CONTACT_DRIVER = 0.74
# Keep zero until the installed Robotiq USD geometry snapshot establishes the
# physical pad center relative to Lula's different-family 220 mm TCP model.
PLATE_TCP_TO_PAD_M = 0.0
TRAY_GRASP_X_INSET_M = 0.060
TRAY_GRASP_Y_INSET_M = 0.018
TRAY_GRASP_TCP_CLEARANCE_M = 0.015
GRIPPER_DRIVERS = {
    Arm.LEFT: "left_right_finger_joint",
    Arm.RIGHT: "right_right_finger_joint",
}
GRIPPER_COUPLED = {
    Arm.LEFT: {
        "left_robotiq_85_left_knuckle_joint": 1.0,
        "left_robotiq_85_right_knuckle_joint": -1.0,
        "left_robotiq_85_left_inner_knuckle_joint": 1.0,
        "left_robotiq_85_right_inner_knuckle_joint": -1.0,
        "left_robotiq_85_left_finger_tip_joint": -1.0,
        "left_robotiq_85_right_finger_tip_joint": 1.0,
    },
    Arm.RIGHT: {
        "right_robotiq_85_left_knuckle_joint": 1.0,
        "right_robotiq_85_right_knuckle_joint": -1.0,
        "right_robotiq_85_left_inner_knuckle_joint": 1.0,
        "right_robotiq_85_right_inner_knuckle_joint": -1.0,
        "right_robotiq_85_left_finger_tip_joint": -1.0,
        "right_robotiq_85_right_finger_tip_joint": 1.0,
    },
}


def _quat_to_yaw(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = (float(value) for value in quat_wxyz)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = (float(value) for value in left)
    rw, rx, ry, rz = (float(value) for value in right)
    return np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=float,
    )


def _axis_angle_quat(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=float)
    vector /= max(float(np.linalg.norm(vector)), 1e-9)
    half = 0.5 * angle
    return np.asarray((math.cos(half), *(math.sin(half) * vector)), dtype=float)


def _quat_angular_error(left: np.ndarray, right: np.ndarray) -> float:
    """Shortest unsigned angular distance between two wxyz quaternions."""

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left /= max(float(np.linalg.norm(left)), 1e-9)
    right /= max(float(np.linalg.norm(right)), 1e-9)
    return 2.0 * math.acos(float(np.clip(abs(np.dot(left, right)), 0.0, 1.0)))


def _quat_slerp(start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
    """Shortest-path spherical interpolation for normalized wxyz quaternions."""

    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    start /= max(float(np.linalg.norm(start)), 1e-9)
    end /= max(float(np.linalg.norm(end)), 1e-9)
    dot = float(np.dot(start, end))
    if dot < 0.0:
        end = -end
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = start + fraction * (end - start)
        return result / max(float(np.linalg.norm(result)), 1e-9)
    theta = math.acos(dot)
    scale = math.sin(theta)
    result = (
        math.sin((1.0 - fraction) * theta) / scale * start
        + math.sin(fraction * theta) / scale * end
    )
    return result / max(float(np.linalg.norm(result)), 1e-9)


def _top_down_quat(yaw: float) -> np.ndarray:
    """Return wxyz for tool-Z down with a selectable world-Z yaw."""
    half = 0.5 * float(yaw)
    return np.asarray((0.0, -math.sin(half), math.cos(half), 0.0), dtype=float)


def _quat_local_z(quaternion: np.ndarray) -> np.ndarray:
    """World-space direction of a wxyz quaternion's local positive Z axis."""

    w, x, y, z = (float(value) for value in quaternion)
    return np.asarray(
        (
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
        dtype=float,
    )


def _wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _physx_continuous_target(current: float, delta: float) -> float:
    target = current + delta
    while target >= 2.0 * math.pi:
        target -= 2.0 * math.pi
    while target <= -2.0 * math.pi:
        target += 2.0 * math.pi
    return target


def _compute_drive_targets(
    joint_positions: np.ndarray,
    steering_ids: list[int],
    vx: float,
    vy: float,
    wz: float,
) -> tuple[np.ndarray, np.ndarray]:
    steering_targets = np.zeros(len(DRIVE_MODULES), dtype=np.float32)
    drive_targets = np.zeros(len(DRIVE_MODULES), dtype=np.float32)
    vectors = []
    maximum = 0.0
    for _, _, x, y in DRIVE_MODULES:
        wheel_vx, wheel_vy = vx - wz * y, vy + wz * x
        speed = math.hypot(wheel_vx, wheel_vy)
        vectors.append((wheel_vx, wheel_vy, speed))
        maximum = max(maximum, speed)
    allowed = MAX_WHEEL_SPEED_RADPS * WHEEL_RADIUS_M
    scale = min(1.0, allowed / maximum) if maximum else 1.0
    for index, (wheel_vx, wheel_vy, speed) in enumerate(vectors):
        wheel_vx, wheel_vy, speed = wheel_vx * scale, wheel_vy * scale, speed * scale
        current = float(joint_positions[steering_ids[index]])
        if speed < 1e-4:
            steering_targets[index] = current
            continue
        raw = math.atan2(wheel_vy, wheel_vx)
        direct = _wrap_to_pi(raw - current)
        flipped = _wrap_to_pi(raw + math.pi - current)
        use_flipped = abs(flipped) < abs(direct)
        delta = flipped if use_flipped else direct
        steering_targets[index] = _physx_continuous_target(current, delta)
        alignment = min(max((math.radians(35.0) - abs(delta)) / math.radians(27.0), 0.0), 1.0)
        wheel_speed = speed / WHEEL_RADIUS_M * alignment
        drive_targets[index] = -wheel_speed if use_flipped else wheel_speed
    return steering_targets, drive_targets


class BimanualRmpController:
    def __init__(self, robot: Any, benchmark_root: Path, physics_dt: float, action_type: Any) -> None:
        from isaacsim.core.utils.rotations import rot_matrix_to_quat
        from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
        from isaacsim.robot_motion.motion_generation.articulation_motion_policy import ArticulationMotionPolicy
        from isaacsim.robot_motion.motion_generation.lula.motion_policies import RmpFlow

        self.robot = robot
        self.physics_dt = physics_dt
        self.action_type = action_type
        self.rot_matrix_to_quat = rot_matrix_to_quat
        assets = benchmark_root / "task3_isaacsim" / "assets" / "lula" / "mobile_fr3_duo"
        urdf = assets / "mobile_fr3_duo_v0_2_franka_hand.urdf"
        configs = {
            Arm.LEFT: (
                assets / "left_arm_description.yaml",
                assets / "left_arm_rmpflow_config.yaml",
                "left_fr3v2_hand_tcp",
            ),
            Arm.RIGHT: (
                assets / "right_arm_description.yaml",
                assets / "right_arm_rmpflow_config.yaml",
                "right_fr3v2_hand_tcp",
            ),
        }
        missing = [str(path) for triple in configs.values() for path in triple[:2] if not Path(path).is_file()]
        if not urdf.is_file() or missing:
            raise FileNotFoundError(f"missing Lula assets: urdf={urdf.is_file()} configs={missing}")
        self.arms: dict[Arm, dict[str, Any]] = {}
        for side, (description, config, frame_name) in configs.items():
            rmpflow = RmpFlow(
                robot_description_path=str(description),
                urdf_path=str(urdf),
                rmpflow_config_path=str(config),
                end_effector_frame_name=frame_name,
                maximum_substep_size=0.0034,
                ignore_robot_state_updates=True,
            )
            policy = ArticulationMotionPolicy(robot, rmpflow, physics_dt)
            self.arms[side] = {
                "rmpflow": rmpflow,
                "policy": policy,
                "ik_solver": LulaKinematicsSolver(str(description), str(urdf)),
                "frame_name": frame_name,
                "joint_indices": np.asarray(
                    [
                        list(robot.dof_names).index(f"{side.value}_fr3v2_joint{joint}")
                        for joint in range(1, 8)
                    ],
                    dtype=np.int64,
                ),
            }

    def _base_pose(self) -> tuple[np.ndarray, np.ndarray]:
        position, orientation = self.robot.get_world_pose()
        position = np.asarray(position, dtype=float)
        names = list(self.robot.dof_names)
        if "franka_spine_vertical_joint" in names:
            position[2] += float(self.robot.get_joint_positions()[names.index("franka_spine_vertical_joint")])
        return position, np.asarray(orientation, dtype=float)

    def current_world_pose(self, side: Arm) -> tuple[np.ndarray, np.ndarray]:
        controller = self.arms[side]
        rmpflow, policy = controller["rmpflow"], controller["policy"]
        base_position, base_orientation = self._base_pose()
        rmpflow.set_robot_base_pose(base_position, base_orientation)
        active = policy.get_active_joints_subset().get_joint_positions()
        local_position, rotation = rmpflow.get_end_effector_pose(active)
        # RmpFlow returns the pose in the world frame after its live robot
        # base pose has been set.  Applying the base transform again here
        # double-translates mobile robots by several metres.
        world_position = np.asarray(local_position, dtype=float)
        world_quat = np.asarray(self.rot_matrix_to_quat(rotation), dtype=float)
        return (
            world_position,
            world_quat / max(float(np.linalg.norm(world_quat)), 1e-9),
        )

    def select_reachable_plate_orientation(
        self,
        side: Arm,
        position: np.ndarray,
        current_orientation: np.ndarray,
        *,
        preferred_yaw_rad: float = PLATE_PREFERRED_TOP_DOWN_YAW_RAD,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Select a reachable top-down yaw without moving the robot."""
        arm = self.arms[side]
        solver = arm["ik_solver"]
        base_position, base_orientation = self._base_pose()
        solver.set_robot_base_pose(base_position, base_orientation)
        joint_positions = np.asarray(self.robot.get_joint_positions(), dtype=float)
        warm_start = joint_positions[arm["joint_indices"]]
        best_orientation: np.ndarray | None = None
        best_score = (-math.inf, -math.inf, -math.inf)
        best_details: dict[str, Any] = {}
        successful = 0
        failures = 0
        try:
            _, position_only_succeeded = solver.compute_inverse_kinematics(
                arm["frame_name"],
                np.asarray(position, dtype=float),
                None,
                warm_start.copy(),
            )
        except Exception:
            position_only_succeeded = False
        # Prefer exact top-down solutions.  Only consider modest residual tilt
        # when the articulated workspace has no exact-yaw solution.
        for fraction in (1.0, 0.90, 0.75, 0.60, 0.45, 0.30, 0.0):
            yaw_values = (
                (0.0,)
                if fraction == 0.0
                else np.linspace(-math.pi, math.pi, 24, endpoint=False)
            )
            for yaw in yaw_values:
                candidate = _quat_slerp(
                    current_orientation,
                    _top_down_quat(float(yaw)),
                    fraction,
                )
                try:
                    solution, succeeded = solver.compute_inverse_kinematics(
                        arm["frame_name"],
                        np.asarray(position, dtype=float),
                        candidate,
                        warm_start.copy(),
                    )
                except Exception:
                    failures += 1
                    continue
                if not succeeded:
                    failures += 1
                    continue
                successful += 1
                solution_array = np.asarray(solution, dtype=float)
                joint_delta = float(np.linalg.norm(solution_array - warm_start))
                yaw_error = abs(
                    _wrap_to_pi(float(yaw) - float(preferred_yaw_rad))
                )
                # IK success is mandatory.  Among exact-top-down solutions,
                # preserve the physical rim-tangent closing axis before using
                # joint travel as the final tie breaker.
                score = (fraction, -yaw_error, -joint_delta)
                if score > best_score:
                    best_score = score
                    best_orientation = candidate
                    best_details = {
                        "fraction": fraction,
                        "yaw_rad": float(yaw),
                        "preferred_yaw_error_rad": yaw_error,
                        "joint_delta_rad": joint_delta,
                    }
        if best_orientation is None:
            fallback = _quat_slerp(
                current_orientation,
                _top_down_quat(float(preferred_yaw_rad)),
                PLATE_RIM_TILT_FRACTION,
            )
            return fallback, {
                "selected": False,
                "position_only_succeeded": bool(position_only_succeeded),
                "successful_candidates": successful,
                "failed_candidates": failures,
            }
        return best_orientation, {
            "selected": True,
            "position_only_succeeded": bool(position_only_succeeded),
            "successful_candidates": successful,
            "failed_candidates": failures,
            **best_details,
        }

    def select_reachable_spoon_orientation(
        self,
        side: Arm,
        pregrasp_position: np.ndarray,
        contact_position: np.ndarray,
        current_orientation: np.ndarray,
        *,
        closing_axis: np.ndarray,
        preferred_yaw_rad: float,
        maximum_angle_rad: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Select the steepest spoon grasp reachable at clearance and contact."""
        top_down, top_down_scan = self.select_reachable_plate_orientation(
            side,
            contact_position,
            current_orientation,
            preferred_yaw_rad=preferred_yaw_rad,
        )
        arm = self.arms[side]
        solver = arm["ik_solver"]
        base_position, base_orientation = self._base_pose()
        solver.set_robot_base_pose(base_position, base_orientation)
        joint_positions = np.asarray(self.robot.get_joint_positions(), dtype=float)
        warm_start = joint_positions[arm["joint_indices"]]
        best_orientation: np.ndarray | None = None
        best_score = (-math.inf, -math.inf)
        best_details: dict[str, Any] = {}
        pregrasp_failures = 0
        contact_failures = 0
        successful_angles_deg: list[float] = []
        for angle_rad in np.linspace(SPOON_SIDE_GRASP_ANGLE_RAD, 0.0, 13):
            if float(angle_rad) > float(maximum_angle_rad) + 1e-9:
                continue
            candidate = _quat_mul(
                _axis_angle_quat(tuple(closing_axis), float(angle_rad)),
                top_down,
            )
            try:
                pregrasp_solution, pregrasp_succeeded = solver.compute_inverse_kinematics(
                    arm["frame_name"],
                    np.asarray(pregrasp_position, dtype=float),
                    candidate,
                    warm_start.copy(),
                )
            except Exception:
                pregrasp_failures += 1
                continue
            if not pregrasp_succeeded:
                pregrasp_failures += 1
                continue
            try:
                contact_solution, contact_succeeded = solver.compute_inverse_kinematics(
                    arm["frame_name"],
                    np.asarray(contact_position, dtype=float),
                    candidate,
                    np.asarray(pregrasp_solution, dtype=float),
                )
            except Exception:
                contact_failures += 1
                continue
            if not contact_succeeded:
                contact_failures += 1
                continue
            angle_deg = math.degrees(float(angle_rad))
            successful_angles_deg.append(angle_deg)
            joint_delta = float(
                np.linalg.norm(np.asarray(contact_solution, dtype=float) - warm_start)
            )
            score = (float(angle_rad), -joint_delta)
            if score > best_score:
                best_score = score
                best_orientation = candidate
                best_details = {
                    "grasp_angle_rad": float(angle_rad),
                    "grasp_angle_deg": angle_deg,
                    "joint_delta_rad": joint_delta,
                }
        if best_orientation is None:
            return top_down, {
                "selected": False,
                "grasp_angle_rad": 0.0,
                "grasp_angle_deg": 0.0,
                "successful_angles_deg": successful_angles_deg,
                "pregrasp_failed_candidates": pregrasp_failures,
                "contact_failed_candidates": contact_failures,
                "top_down_scan": top_down_scan,
            }
        return best_orientation, {
            "selected": True,
            "successful_angles_deg": successful_angles_deg,
            "pregrasp_failed_candidates": pregrasp_failures,
            "contact_failed_candidates": contact_failures,
            "top_down_scan": top_down_scan,
            **best_details,
        }

    def step(self, targets: dict[Arm, tuple[np.ndarray, np.ndarray | None]]) -> None:
        positions: dict[int, float] = {}
        velocities: dict[int, float] = {}
        base_position, base_orientation = self._base_pose()
        for side, (target_position, target_orientation) in targets.items():
            arm = self.arms[side]
            arm["rmpflow"].set_robot_base_pose(base_position, base_orientation)
            arm["rmpflow"].set_end_effector_target(target_position, target_orientation)
            action = arm["policy"].get_next_articulation_action(self.physics_dt)
            if action is None or action.joint_positions is None:
                continue
            action_velocities = action.joint_velocities
            if action_velocities is None:
                action_velocities = np.zeros_like(action.joint_positions)
            for index, position, velocity in zip(
                action.joint_indices, action.joint_positions, action_velocities, strict=True
            ):
                positions[int(index)] = float(position)
                velocities[int(index)] = float(velocity)
        if not positions:
            return
        indices = np.asarray(sorted(positions), dtype=np.int64)
        self.robot.get_articulation_controller().apply_action(
            self.action_type(
                joint_positions=np.asarray([positions[i] for i in indices], dtype=np.float32),
                joint_velocities=np.asarray([velocities[i] for i in indices], dtype=np.float32),
                joint_indices=indices,
            )
        )


class IsaacPhysicalActuator(PhysicalActuator):
    def __init__(
        self,
        *,
        world: Any,
        robot: Any,
        action_type: Any,
        benchmark_root: Path,
        reader: IsaacStateReader,
        scorer: OfficialLiveScorer,
        assignments: dict[str, AssignmentTarget],
        store: RuntimeStore,
        render_callback: Callable[[], None],
        physics_dt: float,
        control_dt: float,
    ) -> None:
        self.world = world
        self.robot = robot
        self.action_type = action_type
        self.reader = reader
        self.scorer = scorer
        self.assignments = assignments
        self.store = store
        self.render_callback = render_callback
        self.physics_dt = physics_dt
        self.control_dt = control_dt
        self.rmp = BimanualRmpController(robot, benchmark_root, physics_dt, action_type)
        self.names = list(robot.dof_names)
        self.name_to_index = {name: index for index, name in enumerate(self.names)}
        required = {
            *(item[0] for item in DRIVE_MODULES),
            *(item[1] for item in DRIVE_MODULES),
            *GRIPPER_DRIVERS.values(),
        }
        missing = sorted(required - self.name_to_index.keys())
        if missing:
            raise RuntimeError("robot articulation missing required DOFs: " + ", ".join(missing))
        self.steering_ids = [self.name_to_index[item[0]] for item in DRIVE_MODULES]
        self.drive_ids = [self.name_to_index[item[1]] for item in DRIVE_MODULES]
        drive_controller = self.robot.get_articulation_controller()
        original_kps, original_kds = drive_controller.get_gains()
        tuned_kps = np.asarray(original_kps, dtype=float).copy()
        tuned_kds = np.asarray(original_kds, dtype=float).copy()
        drive_gains_before = {
            self.names[index]: {
                "kp": float(tuned_kps[index]),
                "kd": float(tuned_kds[index]),
            }
            for index in self.drive_ids
        }
        tuned_kps[self.drive_ids] = 0.0
        tuned_kds[self.drive_ids] = np.maximum(
            tuned_kds[self.drive_ids], BASE_WHEEL_VELOCITY_KD
        )
        drive_controller.set_gains(tuned_kps, tuned_kds, save_to_usd=False)
        for index in self.drive_ids:
            drive_controller.switch_dof_control_mode(index, "velocity")
        drive_controller.set_effort_modes(
            "force", joint_indices=list(self.drive_ids)
        )
        drive_controller.set_max_efforts(
            np.full(len(self.drive_ids), BASE_WHEEL_MAX_EFFORT_NM, dtype=np.float32),
            joint_indices=list(self.drive_ids),
        )
        configured_max_efforts = np.asarray(
            drive_controller.get_max_efforts(), dtype=float
        )
        configured_effort_modes = drive_controller.get_effort_modes()
        self.store.event(
            "base_drive_configured",
            gains_before=drive_gains_before,
            velocity_kd=BASE_WHEEL_VELOCITY_KD,
            max_effort_nm=BASE_WHEEL_MAX_EFFORT_NM,
            drive_joints=[self.names[index] for index in self.drive_ids],
            max_efforts_readback=configured_max_efforts[self.drive_ids].tolist(),
            effort_modes_readback=[
                configured_effort_modes[index] for index in self.drive_ids
            ],
        )
        # The omni base is not statically braked by the upstream asset.  Arm
        # reaction forces otherwise roll it several centimetres away from a
        # station during a long Cartesian motion.  Hold the authored steering
        # angles and command zero wheel velocity while manipulating; navigation
        # and physical withdrawal explicitly release this actuator hold.
        current_joints = np.asarray(self.robot.get_joint_positions(), dtype=float)
        self._gripper_hold_targets = {
            side: float(current_joints[self.name_to_index[GRIPPER_DRIVERS[side]]])
            for side in (Arm.LEFT, Arm.RIGHT)
        }
        self._base_hold_enabled = True
        self._base_hold_steering = current_joints[self.steering_ids].copy()
        self.base_obstacles, obstacle_details = self._build_base_obstacles()
        self.base_obstacle_details = obstacle_details
        self.navigation_failures: dict[str, int] = {}
        self.unrecoverable_failure_reason: str | None = None
        self.defer_scope_reason: str | None = None
        self._plate_grasp_orientation: np.ndarray | None = None
        self._cup_grasp_orientation: np.ndarray | None = None
        self._spoon_grasp_orientation: np.ndarray | None = None
        self._spoon_grasp_bias_index = 0
        self._spoon_handle_lateral_bias_m = SPOON_HANDLE_LATERAL_BIAS_SEQUENCE_M[0]
        self._spoon_grasp_angle_cap_rad = SPOON_SIDE_GRASP_ANGLE_RAD
        self._physx_simulation_interface: Any | None = None
        self._physics_schema_tools: Any | None = None
        try:
            from omni.physx import get_physx_simulation_interface
            from pxr import PhysicsSchemaTools

            self._physx_simulation_interface = get_physx_simulation_interface()
            self._physics_schema_tools = PhysicsSchemaTools
        except Exception as exc:
            self.store.event("spoon_contact_telemetry_unavailable", error=repr(exc))
        self._feeding_payload_indices: set[int] = set()
        head_position = np.asarray(self.reader.pose("head").position, dtype=float)
        head_samples = self.reader.descendant_geometry_samples("head")
        eye_positions: list[tuple[float, float, float]] = []
        for sample in head_samples:
            path = str(sample.get("path", "")).lower()
            if "/eye_" not in path:
                continue
            bounds = sample.get("bounds")
            if isinstance(bounds, dict):
                minimum = np.asarray(bounds.get("minimum", ()), dtype=float)
                maximum = np.asarray(bounds.get("maximum", ()), dtype=float)
                if minimum.shape == (3,) and maximum.shape == (3,):
                    eye_positions.append(
                        tuple(float(value) for value in 0.5 * (minimum + maximum))
                    )
                    continue
            position = np.asarray(sample.get("position", ()), dtype=float)
            if position.shape == (3,):
                eye_positions.append(tuple(float(value) for value in position))
        self._head_mouth_position = np.asarray(
            head_mouth_target(head_position, eye_positions), dtype=float
        )
        table_center_xy = np.asarray((-2.1, 1.95), dtype=float)
        self._head_inward_xy = table_center_xy - head_position[:2]
        self._head_inward_xy /= max(
            float(np.linalg.norm(self._head_inward_xy)), 1e-9
        )
        self.store.event(
            "feeding_target_calibrated",
            head_position=head_position.tolist(),
            eye_positions=[list(position) for position in eye_positions],
            mouth_position=self._head_mouth_position.tolist(),
            inward_xy=self._head_inward_xy.tolist(),
        )
        self.store.event(
            "base_obstacles_built",
            count=len(self.base_obstacles),
            obstacles=obstacle_details,
        )
        articulation_view = getattr(robot, "_articulation_view", None)
        physics_view = getattr(articulation_view, "_physics_view", None)
        method_filter = lambda name: any(  # noqa: E731 - compact introspection predicate
            token in name.lower() for token in ("link", "body", "pose", "transform")
        )
        self.store.event(
            "articulation_live_pose_api",
            body_names=list(getattr(robot, "body_names", []) or []),
            robot_methods=sorted(name for name in dir(robot) if method_filter(name)),
            articulation_methods=sorted(
                name for name in dir(articulation_view) if method_filter(name)
            )
            if articulation_view is not None
            else [],
            physics_methods=sorted(name for name in dir(physics_view) if method_filter(name))
            if physics_view is not None
            else [],
        )
        if physics_view is not None and hasattr(physics_view, "get_link_transforms"):
            link_transforms = np.asarray(physics_view.get_link_transforms())
            self.store.event(
                "articulation_live_pose_probe",
                articulation_body_names=list(getattr(articulation_view, "body_names", []) or []),
                link_paths_repr=repr(getattr(physics_view, "link_paths", None))[:24000],
                transform_shape=list(link_transforms.shape),
                transform_head=link_transforms.reshape(-1)[:28].tolist(),
            )

    def _build_base_obstacles(self) -> tuple[list[RectObstacle], list[dict[str, object]]]:
        from pxr import Usd, UsdPhysics

        excluded_roots = ["/World/Robot", "/World/Scene/CoffeeBeans"]
        excluded_roots.extend(
            self.reader.resolve(name)
            for name in ("plate", "cup", "bowl", "spoon", "tray", "recycling", "sink", "head")
        )
        purposes = [
            self.reader.UsdGeom.Tokens.default_,
            self.reader.UsdGeom.Tokens.render,
            self.reader.UsdGeom.Tokens.proxy,
        ]
        cache = self.reader.UsdGeom.BBoxCache(self.reader.Usd.TimeCode.Default(), purposes)
        obstacles: list[RectObstacle] = []
        details: list[dict[str, object]] = []
        prim_range = Usd.PrimRange.Stage(self.reader.stage, Usd.TraverseInstanceProxies())
        for prim in prim_range:
            path = str(prim.GetPath())
            if any(path == root or path.startswith(root + "/") for root in excluded_roots):
                continue
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            minimum, maximum = aligned.GetMin(), aligned.GetMax()
            values = [float(value) for value in (*minimum, *maximum)]
            if not all(math.isfinite(value) for value in values):
                continue
            min_x, min_y, min_z, max_x, max_y, max_z = values
            # Ignore floor/ceiling-only colliders.  Keep walls and furniture
            # that intersect the mobile base's swept vertical volume.
            if max_z <= 0.08 or min_z >= 0.95:
                continue
            if max_x - min_x < 0.015 and max_y - min_y < 0.015:
                continue
            obstacles.append(RectObstacle((min_x, min_y), (max_x, max_y)))
            details.append(
                {
                    "path": path,
                    "minimum": [min_x, min_y, min_z],
                    "maximum": [max_x, max_y, max_z],
                }
            )
        return obstacles, details

    def _goal_candidates(
        self,
        target: np.ndarray,
        *,
        standoff_m: float = BASE_STATION_STANDOFF_M,
        prefer_outermost: bool = False,
    ) -> list[np.ndarray]:
        """Generate reachable base poses around the live supporting station."""
        x, y = (float(value) for value in target[:2])
        supports: list[RectObstacle] = []
        for detail in self.base_obstacle_details:
            minimum = detail["minimum"]
            maximum = detail["maximum"]
            min_x, min_y, min_z = (float(value) for value in minimum)
            max_x, max_y, max_z = (float(value) for value in maximum)
            if not (min_x <= x <= max_x and min_y <= y <= max_y):
                continue
            if not (0.45 <= max_z <= 0.95 and max_z - min_z <= 0.30):
                continue
            supports.append(RectObstacle((min_x, min_y), (max_x, max_y)))
        return [
            np.asarray(candidate, dtype=float)
            for candidate in station_goal_candidates(
                (x, y),
                supports,
                standoff_m=standoff_m,
                prefer_outermost=prefer_outermost,
            )
        ]

    def _room_portal(self, target_x: float) -> tuple[float, float, float] | None:
        """Derive the room-divider portal from the live authored wall bounds."""
        horizontal: list[tuple[float, float, float, float]] = []
        for detail in self.base_obstacle_details:
            minimum = detail["minimum"]
            maximum = detail["maximum"]
            min_x, min_y, min_z = (float(value) for value in minimum)
            max_x, max_y, max_z = (float(value) for value in maximum)
            if (
                max_x - min_x >= 0.8
                and max_y - min_y <= 0.35
                and min_z <= 0.08
                and max_z >= 0.95
            ):
                horizontal.append((min_x, max_x, min_y, max_y))
        if not horizontal:
            return None
        divider_y = min(
            ((min_y + max_y) * 0.5 for _, _, min_y, max_y in horizontal),
            key=abs,
        )
        spans = sorted(
            (min_x, max_x, min_y, max_y)
            for min_x, max_x, min_y, max_y in horizontal
            if min_y - 0.03 <= divider_y <= max_y + 0.03
        )
        merged: list[list[float]] = []
        for min_x, max_x, min_y, max_y in spans:
            if merged and min_x <= merged[-1][1] + 0.03:
                merged[-1][1] = max(merged[-1][1], max_x)
                merged[-1][2] = min(merged[-1][2], min_y)
                merged[-1][3] = max(merged[-1][3], max_y)
            else:
                merged.append([min_x, max_x, min_y, max_y])
        gaps: list[tuple[float, float, float]] = []
        for left, right in zip(merged, merged[1:], strict=False):
            gap_min, gap_max = left[1], right[0]
            if gap_max - gap_min >= 0.85:
                gaps.append(((gap_min + gap_max) * 0.5, min(left[2], right[2]), max(left[3], right[3])))
        if not gaps:
            return None
        return min(gaps, key=lambda item: abs(item[0] - target_x))

    def _plan_route(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: list[RectObstacle],
        clearance: float,
    ) -> list[tuple[float, float]]:
        portal = self._room_portal(float(goal[0]))
        if portal is None:
            return collision_cleared_waypoints(
                tuple(start), tuple(goal), obstacles, clearance=clearance, resolution=0.10, max_nodes=150_000
            )
        portal_x, divider_min_y, divider_max_y = portal
        crosses_down = float(start[1]) > divider_max_y and float(goal[1]) < divider_min_y
        crosses_up = float(start[1]) < divider_min_y and float(goal[1]) > divider_max_y
        if not (crosses_down or crosses_up):
            return collision_cleared_waypoints(
                tuple(start), tuple(goal), obstacles, clearance=clearance, resolution=0.10, max_nodes=150_000
            )
        upper = np.asarray((portal_x, divider_max_y + BASE_PORTAL_LONGITUDINAL_CLEARANCE_M))
        lower = np.asarray((portal_x, divider_min_y - BASE_PORTAL_LONGITUDINAL_CLEARANCE_M))
        entry, exit_point = (upper, lower) if crosses_down else (lower, upper)
        first = collision_cleared_waypoints(
            tuple(start), tuple(entry), obstacles, clearance=clearance, resolution=0.10, max_nodes=150_000
        )
        second = collision_cleared_waypoints(
            tuple(exit_point), tuple(goal), obstacles, clearance=clearance, resolution=0.10, max_nodes=150_000
        )
        return [*first, tuple(exit_point), *second[1:]]

    def _step(self, render: bool = True) -> None:
        for command in self.store.drain_commands():
            if command in {"pause", "resume"}:
                self.store.apply_command(command)
            else:
                self.store.queue_command(command)
        while (
            self.store.telemetry.lifecycle is Lifecycle.PAUSED
            and not self.store.stop_requested
        ):
            time.sleep(0.02)
            for command in self.store.drain_commands():
                if command == "resume":
                    self.store.apply_command(command)
                else:
                    self.store.queue_command(command)
        if self.store.stop_requested:
            return
        self._apply_base_hold()
        # A gripper position action sent after the wheel-velocity action can
        # replace the active command on this single articulation while the
        # jaws are loaded.  During navigation the existing PhysX joint-drive
        # targets already persist, so only refresh them while the base brake is
        # active for arm manipulation.
        if self._base_hold_enabled:
            self._apply_gripper_hold()
        self.world.step(render=render)
        if self.world.current_time_step_index % 4 == 0:
            self.render_callback()
        root_position, root_orientation = self.robot.get_world_pose()
        self.store.update(
            simulated_seconds=self.world.current_time,
            robot_position=tuple(float(value) for value in root_position),
            robot_orientation_wxyz=tuple(float(value) for value in root_orientation),
        )
        if self.world.current_time_step_index % 12 == 0:
            self.store.update(object_state=self.reader.snapshot())

    def _target_position(self, name: str) -> np.ndarray:
        if "+" in name:
            values = [self._target_position(part) for part in name.split("+")]
            return np.mean(values, axis=0)
        if name == "plate_seat":
            xy = self.assignments["plate"].xy
            return np.asarray((xy[0], xy[1], 0.80), dtype=float)
        if name == "cup_seat":
            xy = self.assignments["cup"].xy
            return np.asarray((xy[0], xy[1], 0.80), dtype=float)
        if name == "head_seat":
            xy = self.assignments["bowl"].xy
            return np.asarray((xy[0], xy[1], 0.80), dtype=float)
        if name == "mouth":
            return self._head_mouth_position.copy()
        if name == "mouth_standoff":
            return self._head_mouth_position + np.asarray(
                (
                    self._head_inward_xy[0] * HEAD_MOUTH_STANDOFF_M,
                    self._head_inward_xy[1] * HEAD_MOUTH_STANDOFF_M,
                    HEAD_MOUTH_STANDOFF_Z_M,
                ),
                dtype=float,
            )
        if name == "mouth_hold":
            return self._head_mouth_position + np.asarray(
                (
                    self._head_inward_xy[0] * HEAD_MOUTH_HOLD_M,
                    self._head_inward_xy[1] * HEAD_MOUTH_HOLD_M,
                    HEAD_MOUTH_HOLD_Z_M,
                ),
                dtype=float,
            )
        if name == "mouth_retract":
            return self._head_mouth_position + np.asarray(
                (
                    self._head_inward_xy[0] * HEAD_MOUTH_RETRACT_M,
                    self._head_inward_xy[1] * HEAD_MOUTH_RETRACT_M,
                    HEAD_MOUTH_RETRACT_Z_M,
                ),
                dtype=float,
            )
        return np.asarray(self.reader.pose(name).position, dtype=float)

    def _carried_objects_retained(self, object_names: tuple[str, ...]) -> bool:
        if not object_names:
            return True
        root_position, _ = self.robot.get_world_pose()
        root_xy = np.asarray(root_position[:2], dtype=float)
        lost: dict[str, dict[str, object]] = {}
        for name in object_names:
            position = np.asarray(self.reader.pose(name).position, dtype=float)
            distance_xy = float(np.linalg.norm(position[:2] - root_xy))
            if position[2] < LOADED_OBJECT_MIN_HEIGHT_M or distance_xy > 1.15:
                lost[name] = {
                    "position": position.tolist(),
                    "robot_position": np.asarray(root_position, dtype=float).tolist(),
                    "distance_xy_m": distance_xy,
                }
        if not lost:
            return True
        self.unrecoverable_failure_reason = (
            "physically carried object lost during navigation: "
            + ", ".join(sorted(lost))
        )
        self.store.event(
            "navigation_object_retention_failed",
            objects=lost,
            failure_reason=self.unrecoverable_failure_reason,
        )
        return False

    def _retained_bowl_bean_count(self) -> int:
        bowl_position = np.asarray(self.reader.pose("bowl").position, dtype=float)
        return sum(
            float(np.linalg.norm(np.asarray(bean, dtype=float) - bowl_position))
            <= 0.11
            for bean in self.reader.bean_positions()
        )

    def _navigate(
        self,
        target_name: str,
        limits: SafetyLimits,
        *,
        carried_objects: tuple[str, ...] = (),
        dining_station: bool = False,
        dining_final_advance: bool = True,
        nearby_station_acceptance_m: float | None = None,
        manipulation_yaw_tolerance_rad: float | None = None,
    ) -> bool:
        target = self._target_position(target_name)[:2]
        root_position, root_orientation = self.robot.get_world_pose()
        start = np.asarray(root_position[:2], dtype=float)
        portal = self._room_portal(float(target[0]))
        attempt = self.navigation_failures.get(target_name, 0)
        loaded_station = target_name.endswith("_seat")
        dining_station = loaded_station or dining_station
        loaded_navigation = bool(carried_objects)
        if (
            dining_station
            and nearby_station_acceptance_m is not None
            and float(np.linalg.norm(start - target))
            <= float(nearby_station_acceptance_m)
        ):
            self.store.event(
                "nearby_dining_station_accepted",
                target=target_name,
                start=start.tolist(),
                target_position=target.tolist(),
                distance_m=float(np.linalg.norm(start - target)),
                acceptance_m=float(nearby_station_acceptance_m),
            )
            oriented = self._orient_base_for_right_arm(
                target,
                limits,
                yaw_tolerance_rad=manipulation_yaw_tolerance_rad,
            )
            self.navigation_failures[target_name] = 0 if oriented else attempt + 1
            return oriented
        base_clearance = (
            BASE_LOADED_FOOTPRINT_CLEARANCE_M
            if loaded_navigation
            else BASE_FOOTPRINT_CLEARANCE_M
        )
        clearance = min(
            BASE_MAX_CLEARANCE_M,
            base_clearance + attempt * BASE_RETRY_CLEARANCE_STEP_M,
        )
        planning_obstacles = list(self.base_obstacles)
        try:
            planning_start = np.asarray(
                clearance_egress_point(
                    tuple(start),
                    planning_obstacles,
                    clearance=clearance,
                    margin=BASE_CLEARANCE_EGRESS_MARGIN_M,
                ),
                dtype=float,
            )
        except ValueError as exc:
            self.store.event(
                "clearance_egress_failed",
                target=target_name,
                start=start.tolist(),
                error=repr(exc),
            )
            return False
        needs_egress = float(np.linalg.norm(planning_start - start)) > 1e-6
        routes: list[
            tuple[float, float, np.ndarray, list[tuple[float, float]]]
        ] = []
        route_diagnostics: list[dict[str, object]] = []
        route_errors: list[str] = []
        counter_station = target_name in {"recycling", "sink"}
        station_standoff = (
            COUNTER_STATION_STANDOFF_M
            if counter_station
            else DINING_STATION_STANDOFF_M
            if dining_station
            else BASE_STATION_STANDOFF_M
        )
        for candidate in self._goal_candidates(
            target,
            standoff_m=station_standoff,
            prefer_outermost=counter_station,
        ):
            manipulation_reach = float(np.linalg.norm(candidate - target))
            if (
                dining_station
                and manipulation_reach > BASE_MAX_DINING_MANIPULATION_REACH_M
            ):
                route_errors.append(
                    f"{candidate.tolist()}: manipulation reach "
                    f"{manipulation_reach:.3f} m exceeds "
                    f"{BASE_MAX_DINING_MANIPULATION_REACH_M:.3f} m"
                )
                continue
            if (
                not dining_station
                and target_name in {"plate", "cup", "bowl", "spoon"}
                and manipulation_reach > BASE_MAX_SUPPLY_MANIPULATION_REACH_M
            ):
                route_errors.append(
                    f"{candidate.tolist()}: manipulation reach "
                    f"{manipulation_reach:.3f} m exceeds "
                    f"{BASE_MAX_SUPPLY_MANIPULATION_REACH_M:.3f} m"
                )
                continue
            if any(obstacle.contains(tuple(candidate), clearance=clearance) for obstacle in planning_obstacles):
                route_errors.append(f"{candidate.tolist()}: blocked")
                continue
            try:
                route = self._plan_route(
                    planning_start,
                    candidate,
                    planning_obstacles,
                    clearance,
                )
                if dining_station:
                    route = align_horizontal_corridor(
                        route,
                        center_y=float(candidate[1]),
                    )
                if needs_egress:
                    route = [tuple(start), *route]
            except Exception as exc:
                route_errors.append(f"{candidate.tolist()}: {exc!r}")
                continue
            length = sum(
                math.dist(left, right)
                for left, right in zip(route, route[1:], strict=False)
            )
            routes.append((manipulation_reach, length, candidate, route))
            route_diagnostics.append(
                {
                    "candidate": candidate.tolist(),
                    "manipulation_reach_m": manipulation_reach,
                    "route_length_m": length,
                }
            )
        if not routes:
            self.store.event(
                "waypoint_planning_failed", target=target_name, errors=route_errors
            )
            return False
        ranked_routes = sorted(routes, key=lambda item: (item[0], item[1]))
        # A physically stalled station approach is evidence that its authored
        # collision geometry is incomplete or too optimistic. On a retry,
        # advance to the next collision-cleared station candidate instead of
        # driving the loaded base into the same contact again.
        selected_route_rank = min(attempt, len(ranked_routes) - 1)
        _, _, goal, waypoints = ranked_routes[selected_route_rank]
        self._base_hold_enabled = False
        self.store.event(
            "navigation_planned",
            target=target_name,
            attempt=attempt + 1,
            clearance_m=clearance,
            start=start.tolist(),
            planning_start=planning_start.tolist(),
            goal=goal.tolist(),
            waypoints=[list(point) for point in waypoints],
            candidate_routes=route_diagnostics,
            rejected_candidates=route_errors,
            selected_route_rank=selected_route_rank,
        )
        waypoint_index = 1 if len(waypoints) > 1 else 0
        corridor_entry_index = (
            horizontal_corridor_entry_index(waypoints)
            if dining_station
            else None
        )
        corridor_orientation_index = (
            max(0, corridor_entry_index - 2)
            if corridor_entry_index is not None
            else None
        )
        corridor_oriented = False
        portal_oriented = False
        previous_speed = np.zeros(2, dtype=float)
        started = time.monotonic()
        navigation_timeout_s = (
            CARRY_NAVIGATION_TIMEOUT_S
            if loaded_navigation
            else limits.command_timeout_s
        )
        last_progress_at = started
        last_progress_position = start.copy()
        if corridor_orientation_index == 0:
            self.store.event(
                "dining_corridor_orientation_start",
                target=target_name,
                waypoint=list(waypoints[0]),
            )
            if not self._orient_base_for_right_arm(
                target,
                limits,
                base_position_hint=goal,
                yaw_tolerance_rad=manipulation_yaw_tolerance_rad,
            ):
                return False
            # _orient_base_for_right_arm brakes on completion.  This is a
            # mid-route turn, so release that hold before translating toward
            # the corridor; final arrival will brake again for manipulation.
            self._base_hold_enabled = False
            corridor_oriented = True
            started = time.monotonic()
            last_progress_at = started
            root_position, _ = self.robot.get_world_pose()
            last_progress_position = np.asarray(root_position[:2], dtype=float)
            self.store.event(
                "dining_corridor_orientation_complete",
                target=target_name,
                waypoint=list(waypoints[0]),
            )
        iteration = 0
        while (
            time.monotonic() - started <= navigation_timeout_s
            and not self.store.stop_requested
        ):
            iteration += 1
            if not self._carried_objects_retained(carried_objects):
                self._stop_base()
                return False
            root_position, root_orientation = self.robot.get_world_pose()
            current_xy = np.asarray(root_position[:2], dtype=float)
            waypoint = np.asarray(waypoints[waypoint_index], dtype=float)
            error = waypoint - current_xy
            next_waypoint = (
                np.asarray(waypoints[waypoint_index + 1], dtype=float)
                if waypoint_index < len(waypoints) - 1
                else None
            )
            begins_dining_corridor = (
                dining_station
                and next_waypoint is not None
                and abs(float(next_waypoint[0] - waypoint[0])) >= 0.50
                and abs(float(next_waypoint[1] - waypoint[1])) <= 0.15
            )
            crosses_portal_segment = False
            if portal is not None and next_waypoint is not None:
                _, divider_min_y, divider_max_y = portal
                crosses_portal_segment = (
                    float(waypoint[1]) > divider_max_y
                    and float(next_waypoint[1]) < divider_min_y
                ) or (
                    float(waypoint[1]) < divider_min_y
                    and float(next_waypoint[1]) > divider_max_y
                )
            if needs_egress and waypoint_index == 1:
                tolerance = BASE_CLEARANCE_EGRESS_ACCEPTANCE_M
            elif waypoint_index == len(waypoints) - 1 and dining_station:
                tolerance = DINING_STATION_ACCEPTANCE_M
            elif (
                waypoint_index == len(waypoints) - 1
                and target_name in {"plate", "cup", "bowl", "spoon"}
            ):
                tolerance = BASE_SUPPLY_STATION_ACCEPTANCE_M
            elif begins_dining_corridor or (
                waypoint_index == len(waypoints) - 2 and dining_station
            ):
                # Enter the narrow 1.20 m wall/table corridor on its measured
                # centerline before commanding longitudinal travel.  The old
                # 10 cm bend tolerance let the loaded swerve base turn east
                # while still 15-18 cm toward the table and make contact.
                tolerance = DINING_CORRIDOR_ENTRY_TOLERANCE_M
            else:
                tolerance = 0.045 if waypoint_index == len(waypoints) - 1 else 0.10
            if float(np.linalg.norm(error)) <= tolerance:
                if waypoint_index < len(waypoints) - 1:
                    if crosses_portal_segment and not portal_oriented:
                        self._stop_base()
                        self.store.event(
                            "portal_entry_orientation_start",
                            target=target_name,
                            waypoint=waypoint.tolist(),
                        )
                        if not self._orient_base_for_portal_transit(limits):
                            return False
                        self._base_hold_enabled = False
                        portal_oriented = True
                        started = time.monotonic()
                        last_progress_at = started
                        root_position, _ = self.robot.get_world_pose()
                        last_progress_position = np.asarray(
                            root_position[:2], dtype=float
                        )
                        self.store.event(
                            "portal_entry_orientation_complete",
                            target=target_name,
                            waypoint=waypoint.tolist(),
                        )
                    if (
                        corridor_orientation_index == waypoint_index
                        and not corridor_oriented
                    ):
                        self._stop_base()
                        self.store.event(
                            "dining_corridor_orientation_start",
                            target=target_name,
                            waypoint=waypoint.tolist(),
                        )
                        if not self._orient_base_for_right_arm(
                            target,
                            limits,
                            base_position_hint=goal,
                            yaw_tolerance_rad=manipulation_yaw_tolerance_rad,
                        ):
                            return False
                        self._base_hold_enabled = False
                        corridor_oriented = True
                        started = time.monotonic()
                        last_progress_at = started
                        root_position, _ = self.robot.get_world_pose()
                        last_progress_position = np.asarray(
                            root_position[:2], dtype=float
                        )
                        self.store.event(
                            "dining_corridor_orientation_complete",
                            target=target_name,
                            waypoint=waypoint.tolist(),
                        )
                    waypoint_index += 1
                    previous_speed = np.zeros(2, dtype=float)
                    continue
                self._stop_base()
                settled_position, _ = self.robot.get_world_pose()
                self.store.event(
                    "navigation_complete",
                    target=target_name,
                    start=start.tolist(),
                    goal=goal.tolist(),
                    end=np.asarray(settled_position[:2], dtype=float).tolist(),
                )
                if target_name == "plate" and not dining_station:
                    if not self._advance_plate_station(target, limits):
                        self.navigation_failures[target_name] = attempt + 1
                        return False
                if target_name in {
                    "plate",
                    "cup",
                    "bowl",
                    "spoon",
                    "plate_seat",
                    "cup_seat",
                    "head_seat",
                    "recycling",
                    "sink",
                }:
                    if not self._orient_base_for_right_arm(
                        target,
                        limits,
                        yaw_tolerance_rad=manipulation_yaw_tolerance_rad,
                    ):
                        self.navigation_failures[target_name] = attempt + 1
                        return False
                    if dining_station:
                        if not dining_final_advance:
                            self.store.event(
                                "dining_station_final_advance_skipped",
                                target=target_name,
                                start=np.asarray(settled_position[:2], dtype=float).tolist(),
                                target_position=target.tolist(),
                            )
                            self.navigation_failures[target_name] = 0
                            return True
                        advanced = self._advance_dining_station(target, limits)
                        if not self._carried_objects_retained(carried_objects):
                            self.navigation_failures[target_name] = attempt + 1
                            return False
                        self.navigation_failures[target_name] = (
                            0 if advanced else attempt + 1
                        )
                        return advanced
                    self.navigation_failures[target_name] = 0
                    return True
                self.navigation_failures[target_name] = 0
                return True
            if float(np.linalg.norm(current_xy - last_progress_position)) >= 0.035:
                last_progress_position = current_xy.copy()
                last_progress_at = time.monotonic()
            elif time.monotonic() - last_progress_at >= BASE_STALL_TIMEOUT_S:
                self._stop_base()
                self.store.event(
                    "navigation_stalled",
                    target=target_name,
                    attempt=attempt + 1,
                    clearance_m=clearance,
                    waypoint=waypoint.tolist(),
                    start=start.tolist(),
                    goal=goal.tolist(),
                    end=current_xy.tolist(),
                )
                self.navigation_failures[target_name] = attempt + 1
                withdraw_distance = BASE_DEFAULT_WITHDRAW_M
                if dining_station:
                    withdraw_distance = self._loaded_station_withdraw_distance(
                        current_xy=current_xy,
                        away_from_world=-error,
                        clearance=clearance,
                    )
                self._withdraw_base(
                    away_from_world=-error,
                    distance_m=withdraw_distance,
                    limits=limits,
                )
                return False
            self.store.update(
                message=(
                    f"navigate to {target_name}: waypoint {waypoint_index + 1}/"
                    f"{len(waypoints)}, {float(np.linalg.norm(error)):.2f} m"
                )
            )
            navigation_speed = (
                min(limits.base_speed_mps, BASE_LOADED_SPEED_MPS)
                if loaded_navigation
                else limits.base_speed_mps
            )
            desired_world = (
                error / max(float(np.linalg.norm(error)), 1e-6)
                * navigation_speed
            )
            max_delta = limits.base_accel_mps2 * self.control_dt
            delta = np.clip(desired_world - previous_speed, -max_delta, max_delta)
            world_velocity = previous_speed + delta
            previous_speed = world_velocity
            yaw = _quat_to_yaw(np.asarray(root_orientation, dtype=float))
            vx = math.cos(yaw) * world_velocity[0] + math.sin(yaw) * world_velocity[1]
            vy = -math.sin(yaw) * world_velocity[0] + math.cos(yaw) * world_velocity[1]
            joints = np.asarray(self.robot.get_joint_positions(), dtype=float)
            steering, drive = _compute_drive_targets(joints, self.steering_ids, vx, vy, 0.0)
            controller = self.robot.get_articulation_controller()
            controller.apply_action(
                self.action_type(
                    joint_positions=steering,
                    joint_indices=np.asarray(self.steering_ids, dtype=np.int64),
                )
            )
            controller.apply_action(
                self.action_type(
                    joint_velocities=drive,
                    joint_indices=np.asarray(self.drive_ids, dtype=np.int64),
                )
            )
            if iteration % 60 == 0:
                measured_velocities = np.asarray(
                    self.robot.get_joint_velocities(), dtype=float
                )
                measured_efforts = self.robot.get_measured_joint_efforts()
                self.store.event(
                    "base_drive_progress",
                    target=target_name,
                    waypoint=waypoint.tolist(),
                    current=current_xy.tolist(),
                    steering_measured=joints[self.steering_ids].tolist(),
                    steering_targets=steering.tolist(),
                    drive_targets=drive.tolist(),
                    drive_measured=measured_velocities[self.drive_ids].tolist(),
                    drive_efforts=(
                        []
                        if measured_efforts is None
                        else np.asarray(measured_efforts, dtype=float)[
                            self.drive_ids
                        ].tolist()
                    ),
                )
            self._step()
        self._stop_base()
        final_position, _ = self.robot.get_world_pose()
        self.store.event(
            "navigation_timeout",
            target=target_name,
            attempt=attempt + 1,
            clearance_m=clearance,
            start=start.tolist(),
            goal=goal.tolist(),
            end=np.asarray(final_position[:2], dtype=float).tolist(),
        )
        self.navigation_failures[target_name] = attempt + 1
        return False

    def _orient_base_for_right_arm(
        self,
        target: np.ndarray,
        limits: SafetyLimits,
        *,
        base_position_hint: np.ndarray | None = None,
        yaw_tolerance_rad: float | None = None,
    ) -> bool:
        """Rotate in place so the right arm's table-facing side aims at target."""
        started = time.monotonic()
        self._base_hold_enabled = False
        yaw_tolerance_rad = (
            BASE_MANIPULATION_YAW_TOLERANCE_RAD
            if yaw_tolerance_rad is None
            else float(yaw_tolerance_rad)
        )
        while (
            time.monotonic() - started <= BASE_MANIPULATION_YAW_TIMEOUT_S
            and not self.store.stop_requested
        ):
            root_position, root_orientation = self.robot.get_world_pose()
            current_xy = np.asarray(root_position[:2], dtype=float)
            aiming_base_xy = (
                current_xy
                if base_position_hint is None
                else np.asarray(base_position_hint[:2], dtype=float)
            )
            target_delta = np.asarray(target[:2], dtype=float) - aiming_base_xy
            if float(np.linalg.norm(target_delta)) <= 1e-6:
                return False
            desired_yaw = right_arm_facing_yaw(
                tuple(aiming_base_xy),
                tuple(np.asarray(target[:2], dtype=float)),
            )
            current_yaw = _quat_to_yaw(np.asarray(root_orientation, dtype=float))
            yaw_error = _wrap_to_pi(desired_yaw - current_yaw)
            if abs(yaw_error) <= yaw_tolerance_rad:
                self._stop_base()
                self.store.event(
                    "base_manipulation_orientation_complete",
                    target=np.asarray(target[:2], dtype=float).tolist(),
                    base_position_hint=(
                        None
                        if base_position_hint is None
                        else np.asarray(base_position_hint[:2], dtype=float).tolist()
                    ),
                    desired_yaw_rad=desired_yaw,
                    actual_yaw_rad=current_yaw,
                    error_rad=yaw_error,
                )
                return True
            proportional_speed = 0.8 * yaw_error
            angular_speed = math.copysign(
                min(
                    max(
                        abs(proportional_speed),
                        BASE_MANIPULATION_MIN_YAW_SPEED_RADPS,
                    ),
                    BASE_MANIPULATION_YAW_SPEED_RADPS,
                ),
                yaw_error,
            )
            joints = np.asarray(self.robot.get_joint_positions(), dtype=float)
            steering, drive = _compute_drive_targets(
                joints, self.steering_ids, 0.0, 0.0, angular_speed
            )
            controller = self.robot.get_articulation_controller()
            controller.apply_action(
                self.action_type(
                    joint_positions=steering,
                    joint_indices=np.asarray(self.steering_ids, dtype=np.int64),
                )
            )
            controller.apply_action(
                self.action_type(
                    joint_velocities=drive,
                    joint_indices=np.asarray(self.drive_ids, dtype=np.int64),
                )
            )
            self.store.update(
                message=f"orienting base for manipulation: {math.degrees(yaw_error):.1f} deg"
            )
            self._step()
        self._stop_base()
        self.store.event(
            "base_manipulation_orientation_failed",
            target=np.asarray(target[:2], dtype=float).tolist(),
        )
        return False

    def _orient_base_for_portal_transit(self, limits: SafetyLimits) -> bool:
        """Rotate the compact base to its measured narrow portal footprint."""
        del limits  # Fixed angular bounds are stricter than the task limits.
        desired_yaw = -0.5 * math.pi
        started = time.monotonic()
        self._base_hold_enabled = False
        while (
            time.monotonic() - started <= BASE_MANIPULATION_YAW_TIMEOUT_S
            and not self.store.stop_requested
        ):
            _, root_orientation = self.robot.get_world_pose()
            current_yaw = _quat_to_yaw(np.asarray(root_orientation, dtype=float))
            yaw_error = _wrap_to_pi(desired_yaw - current_yaw)
            if abs(yaw_error) <= BASE_PORTAL_YAW_TOLERANCE_RAD:
                self._stop_base()
                self.store.event(
                    "base_portal_orientation_complete",
                    desired_yaw_rad=desired_yaw,
                    actual_yaw_rad=current_yaw,
                    error_rad=yaw_error,
                )
                return True
            angular_speed = float(
                np.clip(
                    0.8 * yaw_error,
                    -BASE_MANIPULATION_YAW_SPEED_RADPS,
                    BASE_MANIPULATION_YAW_SPEED_RADPS,
                )
            )
            joints = np.asarray(self.robot.get_joint_positions(), dtype=float)
            steering, drive = _compute_drive_targets(
                joints, self.steering_ids, 0.0, 0.0, angular_speed
            )
            controller = self.robot.get_articulation_controller()
            controller.apply_action(
                self.action_type(
                    joint_positions=steering,
                    joint_indices=np.asarray(self.steering_ids, dtype=np.int64),
                )
            )
            controller.apply_action(
                self.action_type(
                    joint_velocities=drive,
                    joint_indices=np.asarray(self.drive_ids, dtype=np.int64),
                )
            )
            self.store.update(
                message=(
                    "orienting base for portal transit: "
                    f"{math.degrees(yaw_error):.1f} deg"
                )
            )
            self._step()
        self._stop_base()
        self.store.event("base_portal_orientation_failed")
        return False

    def _advance_plate_station(self, target: np.ndarray, limits: SafetyLimits) -> bool:
        """Make a bounded physical final approach after clearing the table corner."""
        start_position, _ = self.robot.get_world_pose()
        start_xy = np.asarray(start_position[:2], dtype=float)
        delta = np.asarray(target, dtype=float) - start_xy
        norm = float(np.linalg.norm(delta))
        if norm <= 1e-6:
            return False
        if norm <= BASE_MAX_SUPPLY_MANIPULATION_REACH_M - PLATE_STATION_REACH_MARGIN_M:
            self.store.event(
                "plate_station_advance_skipped",
                start=start_xy.tolist(),
                target=np.asarray(target, dtype=float).tolist(),
                manipulation_reach_m=norm,
                maximum_reach_m=BASE_MAX_SUPPLY_MANIPULATION_REACH_M,
            )
            return True
        direction = delta / norm
        self._base_hold_enabled = False
        started = time.monotonic()
        self.store.update(message="plate station final approach")
        while (
            time.monotonic() - started <= PLATE_STATION_FINAL_TIMEOUT_S
            and not self.store.stop_requested
        ):
            root_position, root_orientation = self.robot.get_world_pose()
            current_xy = np.asarray(root_position[:2], dtype=float)
            traveled = float(np.dot(current_xy - start_xy, direction))
            if traveled >= PLATE_STATION_FINAL_ADVANCE_M:
                self._stop_base()
                final_position, _ = self.robot.get_world_pose()
                final_xy = np.asarray(final_position[:2], dtype=float)
                actual = float(np.dot(final_xy - start_xy, direction))
                self.store.event(
                    "plate_station_advance_complete",
                    start=start_xy.tolist(),
                    end=final_xy.tolist(),
                    requested_distance_m=PLATE_STATION_FINAL_ADVANCE_M,
                    actual_distance_m=actual,
                )
                return actual >= 0.025
            remaining = PLATE_STATION_FINAL_ADVANCE_M - traveled
            speed = min(
                PLATE_STATION_FINAL_SPEED_MPS,
                limits.base_speed_mps,
                max(0.012, 0.8 * remaining),
            )
            yaw = _quat_to_yaw(np.asarray(root_orientation, dtype=float))
            world_velocity = direction * speed
            vx = math.cos(yaw) * world_velocity[0] + math.sin(yaw) * world_velocity[1]
            vy = -math.sin(yaw) * world_velocity[0] + math.cos(yaw) * world_velocity[1]
            joints = np.asarray(self.robot.get_joint_positions(), dtype=float)
            steering, drive = _compute_drive_targets(
                joints, self.steering_ids, vx, vy, 0.0
            )
            controller = self.robot.get_articulation_controller()
            controller.apply_action(
                self.action_type(
                    joint_positions=steering,
                    joint_indices=np.asarray(self.steering_ids, dtype=np.int64),
                )
            )
            controller.apply_action(
                self.action_type(
                    joint_velocities=drive,
                    joint_indices=np.asarray(self.drive_ids, dtype=np.int64),
                )
            )
            self._step()
        self._stop_base()
        final_position, _ = self.robot.get_world_pose()
        final_xy = np.asarray(final_position[:2], dtype=float)
        actual = float(np.dot(final_xy - start_xy, direction))
        self.store.event(
            "plate_station_advance_incomplete",
            start=start_xy.tolist(),
            end=final_xy.tolist(),
            requested_distance_m=PLATE_STATION_FINAL_ADVANCE_M,
            actual_distance_m=actual,
        )
        return actual >= 0.025

    def _advance_dining_station(
        self,
        target: np.ndarray,
        limits: SafetyLimits,
    ) -> bool:
        """Make a short normal approach after the loaded corridor is clear."""

        start_position, _ = self.robot.get_world_pose()
        start_xy = np.asarray(start_position[:2], dtype=float)
        delta = np.asarray(target, dtype=float) - start_xy
        norm = float(np.linalg.norm(delta))
        if norm <= 1e-6:
            return False
        direction = delta / norm
        self._base_hold_enabled = False
        started = time.monotonic()
        self.store.update(message="dining station final approach")
        while (
            time.monotonic() - started <= DINING_STATION_FINAL_TIMEOUT_S
            and not self.store.stop_requested
        ):
            root_position, root_orientation = self.robot.get_world_pose()
            current_xy = np.asarray(root_position[:2], dtype=float)
            traveled = float(np.dot(current_xy - start_xy, direction))
            if traveled >= DINING_STATION_FINAL_ADVANCE_M:
                self._stop_base()
                self.store.event(
                    "dining_station_advance_complete",
                    start=start_xy.tolist(),
                    end=current_xy.tolist(),
                    requested_distance_m=DINING_STATION_FINAL_ADVANCE_M,
                    actual_distance_m=traveled,
                )
                return True
            remaining = DINING_STATION_FINAL_ADVANCE_M - traveled
            speed = min(
                DINING_STATION_FINAL_SPEED_MPS,
                limits.base_speed_mps,
                max(0.012, 0.8 * remaining),
            )
            yaw = _quat_to_yaw(np.asarray(root_orientation, dtype=float))
            world_velocity = direction * speed
            vx = math.cos(yaw) * world_velocity[0] + math.sin(yaw) * world_velocity[1]
            vy = -math.sin(yaw) * world_velocity[0] + math.cos(yaw) * world_velocity[1]
            joints = np.asarray(self.robot.get_joint_positions(), dtype=float)
            steering, drive = _compute_drive_targets(
                joints, self.steering_ids, vx, vy, 0.0
            )
            controller = self.robot.get_articulation_controller()
            controller.apply_action(
                self.action_type(
                    joint_positions=steering,
                    joint_indices=np.asarray(self.steering_ids, dtype=np.int64),
                )
            )
            controller.apply_action(
                self.action_type(
                    joint_velocities=drive,
                    joint_indices=np.asarray(self.drive_ids, dtype=np.int64),
                )
            )
            self._step()
        self._stop_base()
        final_position, _ = self.robot.get_world_pose()
        final_xy = np.asarray(final_position[:2], dtype=float)
        actual = float(np.dot(final_xy - start_xy, direction))
        self.store.event(
            "dining_station_advance_incomplete",
            start=start_xy.tolist(),
            end=final_xy.tolist(),
            requested_distance_m=DINING_STATION_FINAL_ADVANCE_M,
            actual_distance_m=actual,
        )
        return actual >= 0.08

    def _withdraw_base(
        self,
        *,
        away_from_world: np.ndarray,
        distance_m: float,
        limits: SafetyLimits,
    ) -> bool:
        """Physically withdraw after contact before planning a wider retry."""
        norm = float(np.linalg.norm(away_from_world))
        if norm <= 1e-6:
            return False
        self._base_hold_enabled = False
        direction = np.asarray(away_from_world, dtype=float) / norm
        start_position, _ = self.robot.get_world_pose()
        start_xy = np.asarray(start_position[:2], dtype=float)
        speed = min(0.10, limits.base_speed_mps)
        started = time.monotonic()
        self.store.update(substate=Substate.BACKOFF, message="withdrawing base from obstacle")
        while (
            time.monotonic() - started <= BASE_WITHDRAW_TIMEOUT_S
            and not self.store.stop_requested
        ):
            root_position, root_orientation = self.robot.get_world_pose()
            current_xy = np.asarray(root_position[:2], dtype=float)
            traveled = float(np.dot(current_xy - start_xy, direction))
            if traveled >= distance_m:
                self._stop_base()
                self.store.event(
                    "base_withdraw_complete",
                    start=start_xy.tolist(),
                    end=current_xy.tolist(),
                    requested_distance_m=distance_m,
                )
                return True
            yaw = _quat_to_yaw(np.asarray(root_orientation, dtype=float))
            world_velocity = direction * speed
            vx = math.cos(yaw) * world_velocity[0] + math.sin(yaw) * world_velocity[1]
            vy = -math.sin(yaw) * world_velocity[0] + math.cos(yaw) * world_velocity[1]
            joints = np.asarray(self.robot.get_joint_positions(), dtype=float)
            steering, drive = _compute_drive_targets(joints, self.steering_ids, vx, vy, 0.0)
            controller = self.robot.get_articulation_controller()
            controller.apply_action(
                self.action_type(
                    joint_positions=steering,
                    joint_indices=np.asarray(self.steering_ids, dtype=np.int64),
                )
            )
            controller.apply_action(
                self.action_type(
                    joint_velocities=drive,
                    joint_indices=np.asarray(self.drive_ids, dtype=np.int64),
                )
            )
            self._step()
        self._stop_base()
        final_position, _ = self.robot.get_world_pose()
        self.store.event(
            "base_withdraw_incomplete",
            start=start_xy.tolist(),
            end=np.asarray(final_position[:2], dtype=float).tolist(),
            requested_distance_m=distance_m,
        )
        return False

    def _loaded_station_withdraw_distance(
        self,
        *,
        current_xy: np.ndarray,
        away_from_world: np.ndarray,
        clearance: float,
    ) -> float:
        """Return a bounded physical backoff that exits the loaded envelope."""

        norm = float(np.linalg.norm(away_from_world))
        if norm <= 1e-6:
            return BASE_LOADED_STATION_MIN_WITHDRAW_M
        direction = np.asarray(away_from_world, dtype=float) / norm
        distance = BASE_LOADED_STATION_MIN_WITHDRAW_M
        expanded_clearance = clearance + BASE_LOADED_WITHDRAW_MARGIN_M
        while distance <= BASE_LOADED_STATION_MAX_WITHDRAW_M:
            candidate = np.asarray(current_xy, dtype=float) + direction * distance
            if not any(
                obstacle.contains(tuple(candidate), clearance=expanded_clearance)
                for obstacle in self.base_obstacles
            ):
                return min(
                    BASE_LOADED_STATION_MAX_WITHDRAW_M,
                    distance + BASE_LOADED_WITHDRAW_STEP_M,
                )
            distance += BASE_LOADED_WITHDRAW_STEP_M
        return BASE_LOADED_STATION_MAX_WITHDRAW_M

    def _stop_base(self) -> None:
        joints = np.asarray(self.robot.get_joint_positions(), dtype=float)
        self._base_hold_steering = joints[self.steering_ids].copy()
        self._base_hold_enabled = True
        for _ in range(8):
            self._step()

    def _apply_base_hold(self) -> None:
        """Physically brake the omni wheels while the arms manipulate."""
        if not self._base_hold_enabled:
            return
        controller = self.robot.get_articulation_controller()
        controller.apply_action(
            self.action_type(
                joint_positions=np.asarray(self._base_hold_steering, dtype=np.float32),
                joint_indices=np.asarray(self.steering_ids, dtype=np.int64),
            )
        )
        controller.apply_action(
            self.action_type(
                joint_velocities=np.zeros(len(self.drive_ids), dtype=np.float32),
                joint_indices=np.asarray(self.drive_ids, dtype=np.int64),
            )
        )

    def _apply_gripper_hold(self) -> None:
        """Persist the last physically commanded jaw targets during arm motion."""

        desired: dict[int, float] = {}
        for side, target in self._gripper_hold_targets.items():
            driver_index = self.name_to_index[GRIPPER_DRIVERS[side]]
            desired[driver_index] = target
            for name, multiplier in GRIPPER_COUPLED[side].items():
                if name in self.name_to_index:
                    desired[self.name_to_index[name]] = target * multiplier
        indices = np.asarray(sorted(desired), dtype=np.int64)
        self.robot.get_articulation_controller().apply_action(
            self.action_type(
                joint_positions=np.asarray([desired[index] for index in indices], dtype=np.float32),
                joint_indices=indices,
            )
        )

    def _move_tcp(self, primitive: Primitive, limits: SafetyLimits) -> bool:
        target_center = self._target_position(primitive.target or "") + np.asarray(primitive.offset_xyz)
        sides = (Arm.LEFT, Arm.RIGHT) if primitive.arm is Arm.BOTH else (primitive.arm or Arm.LEFT,)
        paired_names = (
            (primitive.target or "").split("+")
            if primitive.arm is Arm.BOTH and "+" in (primitive.target or "")
            else []
        )
        targets: dict[Arm, tuple[np.ndarray, np.ndarray | None]] = {}
        start_positions: dict[Arm, np.ndarray] = {}
        start_orientations: dict[Arm, np.ndarray] = {}
        lift_object_starts: dict[str, np.ndarray] = {}
        retained_object_starts: dict[str, np.ndarray] = {}
        placement_object_name: str | None = None
        placement_previous_position: np.ndarray | None = None
        stable_placement_steps = 0
        if primitive.label.startswith("lower "):
            candidate = primitive.label.removeprefix("lower ").strip()
            if candidate in self.assignments:
                placement_object_name = candidate
                placement_previous_position = np.asarray(
                    self.reader.pose(candidate).position, dtype=float
                )
        if primitive.label.startswith("lift "):
            for object_name in (primitive.target or "").split("+"):
                if object_name in {"plate", "cup", "bowl", "spoon", "tray"}:
                    lift_object_starts[object_name] = np.asarray(
                        self.reader.pose(object_name).position, dtype=float
                    )
        if primitive.orientation_hint == "loaded_transit_stow":
            for object_name in (primitive.target or "").split("+"):
                if object_name in {"plate", "cup", "bowl", "spoon", "tray"}:
                    retained_object_starts[object_name] = np.asarray(
                        self.reader.pose(object_name).position, dtype=float
                    )
        for side in sides:
            current_position, current_orientation = self.rmp.current_world_pose(side)
            start_positions[side] = current_position
            start_orientations[side] = current_orientation
            if len(paired_names) == 2:
                paired_index = 0 if side is Arm.LEFT else 1
                position = self._target_position(paired_names[paired_index]) + np.asarray(
                    primitive.offset_xyz
                )
            else:
                position = target_center.copy()
            if primitive.orientation_hint in {
                "spoon_scoop_entry",
                "spoon_scoop_exit",
            }:
                spoon_pose = self.reader.pose("spoon")
                handle_offset = np.asarray(
                    rotate_vector_wxyz(
                        spoon_pose.orientation_wxyz,
                        SPOON_HANDLE_LOCAL_OFFSET_M,
                    ),
                    dtype=float,
                )
                spoon_forward_xy = -handle_offset[:2]
                spoon_forward_xy /= max(
                    float(np.linalg.norm(spoon_forward_xy)), 1e-9
                )
                target_center = self._target_position(
                    primitive.target or ""
                ) + np.asarray(
                    (
                        spoon_forward_xy[0] * float(primitive.offset_xyz[0]),
                        spoon_forward_xy[1] * float(primitive.offset_xyz[0]),
                        float(primitive.offset_xyz[2]),
                    )
                )
                spoon_position = np.asarray(
                    spoon_pose.position, dtype=float
                )
                position = np.asarray(
                    held_object_tcp_target(
                        target_center,
                        current_position,
                        spoon_position,
                    ),
                    dtype=float,
                )
                self.store.event(
                    "spoon_scoop_target_compensated",
                    label=primitive.label,
                    object_target=target_center.tolist(),
                    spoon_position=spoon_position.tolist(),
                    spoon_forward_xy=spoon_forward_xy.tolist(),
                    tcp_position=current_position.tolist(),
                    tcp_target=position.tolist(),
                )
            if primitive.metadata.get("placement_object_xy"):
                placement_position = np.asarray(
                    self.reader.pose(placement_object_name or "").position,
                    dtype=float,
                )
                position[:2] = (
                    current_position[:2]
                    + target_center[:2]
                    - placement_position[:2]
                )
                self.store.event(
                    "placement_object_target_compensated",
                    label=primitive.label,
                    object_position=placement_position.tolist(),
                    object_target_xy=target_center[:2].tolist(),
                    tcp_target=position.tolist(),
                )
            if primitive.label.startswith("lift "):
                # Lift vertically from the pose that was physically reached.
                # Re-anchoring X/Y to an object center or rim while also
                # lifting can demand extra reach at a table-edge singularity.
                lift_cap = {
                    "plate": PLATE_LIFT_DELTA_M,
                    "spoon": SPOON_LIFT_DELTA_M,
                }.get(primitive.target, TRAY_OBJECT_LIFT_DELTA_M)
                position[:2] = current_position[:2]
                position[2] = current_position[2] + min(
                    max(float(primitive.offset_xyz[2]), 0.08), lift_cap
                )
            if (
                primitive.target in {"cup", "bowl"}
                and (primitive.orientation_hint or "").startswith("top_")
                and primitive.offset_xyz[2] < 0.08
            ):
                object_bounds = self.reader.bounds(primitive.target)
                position[2] = (
                    float(object_bounds.maximum[2])
                    + TOP_GRASP_TCP_CLEARANCE_M
                )
            if (
                primitive.target == "cup"
                and primitive.orientation_hint == "top_cup"
                and primitive.offset_xyz[2] >= 0.08
            ):
                cup_bounds = self.reader.bounds("cup")
                position[2] = float(cup_bounds.maximum[2]) + 0.10
            if (
                primitive.target == "spoon"
                and primitive.orientation_hint == "top_spoon"
            ):
                spoon_pose = self.reader.pose("spoon")
                handle_offset = np.asarray(
                    rotate_vector_wxyz(
                        spoon_pose.orientation_wxyz,
                        SPOON_HANDLE_LOCAL_OFFSET_M,
                    ),
                    dtype=float,
                )
                position = np.asarray(spoon_pose.position, dtype=float) + handle_offset
                handle_direction_xy = handle_offset[:2].copy()
                handle_direction_xy /= max(
                    float(np.linalg.norm(handle_direction_xy)), 1e-9
                )
                position[:2] += (
                    handle_direction_xy * SPOON_SIDE_GRASP_TCP_RETRACT_M
                )
                handle_lateral_xy = np.asarray(
                    (-handle_offset[1], handle_offset[0]), dtype=float
                )
                handle_lateral_xy /= max(
                    float(np.linalg.norm(handle_lateral_xy)), 1e-9
                )
                position[:2] += (
                    handle_lateral_xy * self._spoon_handle_lateral_bias_m
                )
                if primitive.offset_xyz[2] >= 0.08:
                    position[2] += float(primitive.offset_xyz[2])
                else:
                    # The spoon bowl is the mesh's global high point, while
                    # the thin handle being pinched sits lower.  Using the
                    # global top bound left both live fingertip pads 10-20 mm
                    # above the handle and the jaws closed without contact.
                    # Anchor contact height to the transformed handle point.
                    position[2] += SPOON_SIDE_GRASP_TCP_HEIGHT_M
            if primitive.orientation_hint == "bowl_internal":
                object_bounds = self.reader.bounds("bowl")
                position[0] = 0.5 * float(
                    object_bounds.minimum[0] + object_bounds.maximum[0]
                )
                position[1] = 0.5 * float(
                    object_bounds.minimum[1] + object_bounds.maximum[1]
                )
                position[2] = (
                    float(object_bounds.maximum[2])
                    + BOWL_INTERNAL_TCP_CLEARANCE_M
                )
            if primitive.target == "plate":
                # The Robotiq opening cannot span the plate diameter.  Approach
                # the near rim and choose the reachable point along that rim
                # from the live object bounds and the active arm pose.
                plate_bounds = self.reader.bounds("plate")
                contact_or_carry = (
                    primitive.orientation_hint == "top_plate"
                    and primitive.offset_xyz[2] < 0.08
                ) or primitive.orientation_hint == "carry_plate"
                diameter_x = float(
                    plate_bounds.maximum[0] - plate_bounds.minimum[0]
                )
                rim_inset = circular_rim_inset(
                    diameter_x, PLATE_GRASP_CHORD_M
                )
                if not primitive.label.startswith("lift "):
                    position[0] = (
                        float(plate_bounds.maximum[0])
                        - rim_inset
                        - PLATE_LULA_INWARD_OFFSET_M
                    )
                    # The live reachable wrist closes almost along world Y.
                    # Centering that axis on the plate realizes the bounded
                    # chord instead of an off-center, diameter-scale span.
                    position[1] = 0.5 * float(
                        plate_bounds.minimum[1] + plate_bounds.maximum[1]
                    ) + PLATE_GRASP_LATERAL_BIAS_M
                if contact_or_carry:
                    position += PLATE_TCP_TO_PAD_M * _quat_local_z(current_orientation)
            if primitive.target == "tray":
                tray_bounds = self.reader.bounds("tray")
                position[0] = (
                    float(tray_bounds.maximum[0]) - TRAY_GRASP_X_INSET_M
                )
                position[1] = (
                    float(tray_bounds.maximum[1]) - TRAY_GRASP_Y_INSET_M
                )
                if primitive.offset_xyz[2] >= 0.08:
                    position[2] = (
                        float(tray_bounds.maximum[2])
                        + float(primitive.offset_xyz[2])
                    )
                else:
                    position[2] = (
                        float(tray_bounds.maximum[2])
                        + TRAY_GRASP_TCP_CLEARANCE_M
                    )
            if primitive.arm is Arm.BOTH and len(paired_names) != 2:
                position[1] += -0.09 if side is Arm.LEFT else 0.09
                if primitive.orientation_hint == "bowl_pour_115deg":
                    position[2] += 0.10 if side is Arm.LEFT else -0.10
            if primitive.label.startswith("retract "):
                # At the official supply station world +X points away from the
                # table.  Pull the secured object and gripper completely past
                # the table edge before asking the loaded mobile base to move.
                if "from supply table" in primitive.label:
                    retreat_xy = np.asarray((1.0, 0.0), dtype=float)
                else:
                    root_position, _ = self.robot.get_world_pose()
                    retreat_xy = np.asarray(root_position[:2], dtype=float) - target_center[:2]
                    retreat_xy /= max(float(np.linalg.norm(retreat_xy)), 1e-9)
                position = current_position + np.asarray(
                    (0.20 * retreat_xy[0], 0.20 * retreat_xy[1], 0.02)
                )
            elif primitive.orientation_hint in {"transit_stow", "loaded_transit_stow"}:
                # Fold the empty right arm into a compact body-relative pose
                # before navigating back through the room.  This is reached by
                # ordinary RMPFlow joint actuation after assignment has been
                # verified; no object transform is touched.
                root_position, root_orientation = self.robot.get_world_pose()
                root_position = np.asarray(root_position, dtype=float)
                yaw = _quat_to_yaw(np.asarray(root_orientation, dtype=float))
                loaded_stow = primitive.orientation_hint == "loaded_transit_stow"
                if loaded_stow and primitive.target in {"bowl", "plate"}:
                    # The bowl's internal support and the plate's shallow rim
                    # grasp are stable through lift and table retraction but
                    # can unload under the generic 30 cm diagonal tuck. Use the
                    # already collision-cleared live pose and make only the
                    # extra 12 cm world-X withdrawal needed for portal transit.
                    position = current_position + np.asarray(
                        (0.12, 0.0, 0.03), dtype=float
                    )
                else:
                    local_xy = np.asarray(
                        (0.18, -0.20) if loaded_stow else (0.22, -0.10),
                        dtype=float,
                    )
                    world_xy = np.asarray(
                        (
                            math.cos(yaw) * local_xy[0]
                            - math.sin(yaw) * local_xy[1],
                            math.sin(yaw) * local_xy[0]
                            + math.cos(yaw) * local_xy[1],
                        )
                    )
                    position = np.asarray(
                        (
                            root_position[0] + world_xy[0],
                            root_position[1] + world_xy[1],
                            root_position[2] + (0.90 if loaded_stow else 0.68),
                        ),
                        dtype=float,
                    )
            if primitive.metadata.get("hold_current_position"):
                position = current_position.copy()
            orientation: np.ndarray | None = current_orientation
            if primitive.orientation_hint in {
                "top_bowl_internal",
                "bowl_internal",
                "top_bowl_support",
                "bowl_side_support",
            }:
                # Establish a level wrist above the bowl so both closed fingers
                # enter to the same depth before the force-aware outward spread.
                orientation = PLATE_RIM_QUAT_WXYZ.copy()
            elif primitive.orientation_hint == "bowl_level":
                orientation = PLATE_RIM_QUAT_WXYZ.copy()
            elif primitive.orientation_hint == "top_spoon":
                if (
                    self._spoon_grasp_orientation is None
                    or primitive.offset_xyz[2] >= 0.08
                ):
                    closing_axis = np.asarray(
                        (-handle_direction_xy[1], handle_direction_xy[0], 0.0),
                        dtype=float,
                    )
                    contact_position = position.copy()
                    contact_position[2] += (
                        SPOON_SIDE_GRASP_TCP_HEIGHT_M
                        - float(primitive.offset_xyz[2])
                    )
                    selected, scan = self.rmp.select_reachable_spoon_orientation(
                        side,
                        position,
                        contact_position,
                        current_orientation,
                        closing_axis=closing_axis,
                        preferred_yaw_rad=_wrap_to_pi(
                            math.atan2(
                                float(handle_offset[1]),
                                float(handle_offset[0]),
                            )
                            + math.pi / 4.0
                        ),
                        maximum_angle_rad=self._spoon_grasp_angle_cap_rad,
                    )
                    self._spoon_grasp_orientation = selected
                    self.store.event(
                        "spoon_orientation_selected",
                        side=side.value,
                        scan_position=position.tolist(),
                        spoon_position=list(spoon_pose.position),
                        spoon_orientation_wxyz=list(spoon_pose.orientation_wxyz),
                        handle_point_world=(
                            np.asarray(spoon_pose.position, dtype=float) + handle_offset
                        ).tolist(),
                        handle_offset_world=handle_offset.tolist(),
                        handle_yaw_rad=math.atan2(
                            float(handle_offset[1]),
                            float(handle_offset[0]),
                        ),
                        preferred_grasp_yaw_rad=_wrap_to_pi(
                            math.atan2(
                                float(handle_offset[1]),
                                float(handle_offset[0]),
                            )
                            + math.pi / 4.0
                        ),
                        contact_scan_position=contact_position.tolist(),
                        grasp_mode="ik_scanned_side",
                        orientation_wxyz=selected.tolist(),
                        **scan,
                    )
                orientation = self._spoon_grasp_orientation.copy()
            elif primitive.orientation_hint == "top_cup":
                if (
                    self._cup_grasp_orientation is None
                    or primitive.offset_xyz[2] >= 0.08
                ):
                    selected, scan = self.rmp.select_reachable_plate_orientation(
                        side,
                        position,
                        current_orientation,
                    )
                    self._cup_grasp_orientation = selected
                    self.store.event(
                        "cup_orientation_selected",
                        side=side.value,
                        scan_position=position.tolist(),
                        orientation_wxyz=selected.tolist(),
                        **scan,
                    )
                orientation = self._cup_grasp_orientation.copy()
            elif primitive.orientation_hint in {"top_plate", "top_tray"}:
                # Establish the plate wrist branch during the elevated
                # horizontal pregrasp segment.  The same target is then held
                # through the short contact descent; rotating the long tilted
                # finger geometry at table height can push the mobile base away.
                if (
                    self._plate_grasp_orientation is None
                    or primitive.offset_xyz[2] >= 0.08
                ):
                    scan_position = position.copy()
                    scan_position[2] = max(scan_position[2], position[2] + 0.04)
                    selected, scan = self.rmp.select_reachable_plate_orientation(
                        side,
                        scan_position,
                        current_orientation,
                    )
                    self._plate_grasp_orientation = selected
                    self.store.event(
                        "plate_orientation_selected",
                        side=side.value,
                        scan_position=scan_position.tolist(),
                        orientation_wxyz=selected.tolist(),
                        **scan,
                    )
                orientation = self._plate_grasp_orientation.copy()
            elif (
                (primitive.orientation_hint or "").startswith("top_")
                and primitive.offset_xyz[2] >= 0.08
            ):
                # Let the redundant wrist reconfigure during the collision-clear
                # transit.  The subsequent short contact approach then holds the
                # physically reached wrist orientation instead of the stowed one.
                orientation = None
            elif primitive.orientation_hint == "transit_stow":
                orientation = None
            elif primitive.orientation_hint == "loaded_transit_stow":
                # Preserve the grasp orientation while tucking a payload; a
                # position-only solve may unnecessarily roll the wrist and
                # unload an internal bowl grasp.
                orientation = current_orientation.copy()
            elif primitive.orientation_hint in {
                "carry_spoon",
                "spoon_scoop_entry",
                "spoon_scoop_exit",
            }:
                orientation = current_orientation.copy()
            elif primitive.orientation_hint == "spoon_level":
                # Use the live rigid spoon direction, not an authored yaw, so
                # its bowl points toward the calibrated mouth while preserving
                # the physically achieved pitch/roll from the scoop.
                spoon_position = np.asarray(
                    self.reader.pose("spoon").position, dtype=float
                )
                spoon_xy = spoon_position[:2] - current_position[:2]
                mouth_xy = self._head_mouth_position[:2] - current_position[:2]
                if (
                    float(np.linalg.norm(spoon_xy)) > 1e-4
                    and float(np.linalg.norm(mouth_xy)) > 1e-4
                ):
                    yaw_delta = _wrap_to_pi(
                        math.atan2(float(mouth_xy[1]), float(mouth_xy[0]))
                        - math.atan2(float(spoon_xy[1]), float(spoon_xy[0]))
                    )
                    orientation = _quat_mul(
                        _axis_angle_quat((0.0, 0.0, 1.0), yaw_delta),
                        current_orientation,
                    )
                    self.store.event(
                        "spoon_level_orientation_selected",
                        side=side.value,
                        spoon_direction_xy=spoon_xy.tolist(),
                        mouth_direction_xy=mouth_xy.tolist(),
                        yaw_delta_rad=yaw_delta,
                        orientation_wxyz=orientation.tolist(),
                    )
            elif primitive.orientation_hint == "bowl_pour_115deg":
                orientation = _quat_mul(
                    _axis_angle_quat((1.0, 0.0, 0.0), math.radians(115.0)),
                    PLATE_RIM_QUAT_WXYZ,
                )
            elif primitive.orientation_hint == "spoon_tip_down":
                orientation = _quat_mul(_axis_angle_quat((0.0, 1.0, 0.0), math.radians(55.0)), orientation)
            if orientation is not None:
                orientation /= max(float(np.linalg.norm(orientation)), 1e-9)
            targets[side] = (position, orientation)
        self.store.event(
            "tcp_motion_target",
            label=primitive.label,
            starts={side.value: start_positions[side].tolist() for side in sides},
            start_orientations={side.value: start_orientations[side].tolist() for side in sides},
            targets={side.value: targets[side][0].tolist() for side in sides},
            target_orientations={
                side.value: None if targets[side][1] is None else targets[side][1].tolist()
                for side in sides
            },
            position_only={side.value: targets[side][1] is None for side in sides},
        )
        paths: dict[Arm, list[np.ndarray]] = {
            side: [targets[side][0].copy()]
            for side in sides
        }
        if primitive.orientation_hint == "place_spoon":
            for side in sides:
                robot_side_waypoint = targets[side][0].copy()
                robot_side_waypoint[1] = start_positions[side][1]
                paths[side] = [
                    robot_side_waypoint,
                    targets[side][0].copy(),
                ]
            self.store.event(
                "spoon_placement_corridor",
                label=primitive.label,
                paths={
                    side.value: [position.tolist() for position in paths[side]]
                    for side in sides
                },
            )
        if (
            (
                (primitive.orientation_hint or "").startswith("top_")
                and primitive.offset_xyz[2] >= 0.08
            )
            or primitive.orientation_hint in {"transit_stow", "loaded_transit_stow"}
        ):
            for side in sides:
                path_positions = (
                    post_release_stow_path(
                        start_positions[side],
                        targets[side][0],
                    )
                    if primitive.orientation_hint == "transit_stow"
                    else top_clearance_path(
                        start_positions[side],
                        targets[side][0],
                        clearance_m=(
                            0.04 if primitive.orientation_hint == "top_plate"
                            else (
                                0.08
                                if primitive.orientation_hint == "loaded_transit_stow"
                                else 0.10
                            )
                        ),
                    )
                )
                paths[side] = [
                    np.asarray(position, dtype=float)
                    for position in path_positions
                ]
                if (
                    primitive.orientation_hint == "loaded_transit_stow"
                    and primitive.target == "bowl"
                ):
                    paths[side] = [
                        start_positions[side]
                        + np.asarray((0.0, 0.0, 0.04), dtype=float),
                        targets[side][0].copy(),
                    ]
                if primitive.orientation_hint in {"top_plate", "top_tray"} and len(paths[side]) == 3:
                    # Insert an orientation-only waypoint at the elevated
                    # start XY.  The solver first rises on its live branch,
                    # then rotates in place before extending toward the rim.
                    paths[side] = [
                        paths[side][0].copy(),
                        paths[side][0].copy(),
                        paths[side][1].copy(),
                        paths[side][2].copy(),
                    ]
            self.store.event(
                "tcp_clearance_path",
                label=primitive.label,
                paths={
                    side.value: [position.tolist() for position in paths[side]]
                    for side in sides
                },
            )
        started = time.monotonic()
        commanded = {side: position.copy() for side, position in start_positions.items()}
        phase = {side: 0 for side in sides}
        errors = [math.inf]
        iteration = 0
        stable_lift_steps = 0
        feeding_payload_start_spoon = (
            np.asarray(self.reader.pose("spoon").position, dtype=float)
            if primitive.metadata.get("capture_feeding_payload")
            else None
        )
        feeding_payload_start_beans = (
            tuple(
                np.asarray(position, dtype=float)
                for position in self.reader.bean_positions()
            )
            if primitive.metadata.get("capture_feeding_payload")
            else None
        )
        if primitive.metadata.get("capture_feeding_payload"):
            self._feeding_payload_indices.clear()
        while (
            time.monotonic() - started <= limits.command_timeout_s
            and not self.store.stop_requested
        ):
            iteration += 1
            step_targets: dict[Arm, tuple[np.ndarray, np.ndarray | None]] = {}
            head_zone_active = False
            for side, (_, final_orientation) in targets.items():
                segment_position = paths[side][phase[side]]
                remaining = segment_position - commanded[side]
                distance_to_head = float(
                    np.linalg.norm(commanded[side] - self._head_mouth_position)
                )
                speed = limits.limited_tcp_speed(distance_to_head)
                head_zone_active = (
                    head_zone_active
                    or distance_to_head <= limits.head_zone_radius_m
                )
                if primitive.orientation_hint == "loaded_transit_stow":
                    speed = min(speed, 0.06)
                max_step = speed * self.control_dt
                distance = float(np.linalg.norm(remaining))
                if distance > max_step:
                    commanded[side] += remaining / distance * max_step
                else:
                    commanded[side] = segment_position.copy()
                if distance_to_head < HEAD_MOUTH_TCP_STOP_M:
                    self.store.telemetry.safety.watchdog_interventions += 1
                    self.store.telemetry.safety.last_intervention = "TCP entered head stop radius"
                    return False
                commanded_orientation = final_orientation
                if (
                    primitive.orientation_hint == "top_plate"
                    and len(paths[side]) > 1
                    and phase[side] == 0
                ):
                    # First rise vertically using the live wrist branch; begin
                    # the large rotation only once the fingers have clearance.
                    commanded_orientation = None
                step_targets[side] = (commanded[side], commanded_orientation)
            self.store.telemetry.safety.head_zone_active = head_zone_active
            self.rmp.step(step_targets)
            self._step()
            actual_positions = {
                side: self.rmp.current_world_pose(side)[0]
                for side in sides
            }
            actual_orientations = {
                side: self.rmp.current_world_pose(side)[1]
                for side in sides
            }
            if retained_object_starts:
                retained_currents = {
                    name: np.asarray(self.reader.pose(name).position, dtype=float)
                    for name in retained_object_starts
                }
                lost = {
                    name: {
                        "start": retained_object_starts[name].tolist(),
                        "current": retained_currents[name].tolist(),
                        "drop_m": float(
                            retained_object_starts[name][2]
                            - retained_currents[name][2]
                        ),
                    }
                    for name in retained_object_starts
                    if (
                        retained_currents[name][2] < LOADED_OBJECT_MIN_HEIGHT_M
                        or retained_object_starts[name][2]
                        - retained_currents[name][2] > LOADED_OBJECT_MAX_DROP_M
                    )
                }
                if lost:
                    self.defer_scope_reason = (
                        "payload lost during loaded arm stow: "
                        + ", ".join(sorted(lost))
                    )
                    self.store.event(
                        "loaded_object_retention_failed",
                        label=primitive.label,
                        objects=lost,
                    )
                    return False
            if lift_object_starts:
                object_currents = {
                    name: np.asarray(self.reader.pose(name).position, dtype=float)
                    for name in lift_object_starts
                }
                lifted_heights = {
                    name: float(object_currents[name][2] - start[2])
                    for name, start in lift_object_starts.items()
                }
                stable_lift_steps = (
                    stable_lift_steps + 1
                    if all(
                        height >= LIFT_EARLY_ACCEPT_HEIGHT_M
                        for height in lifted_heights.values()
                    )
                    else 0
                )
                if stable_lift_steps >= LIFT_EARLY_ACCEPT_STABLE_STEPS:
                    if not self._spoon_motion_evidence_satisfied(
                        primitive,
                        feeding_payload_start_spoon,
                        feeding_payload_start_beans,
                    ):
                        return False
                    self.store.event(
                        "lift_transport_verification",
                        label=primitive.label,
                        target=primitive.target,
                        objects={
                            name: {
                                "start": lift_object_starts[name].tolist(),
                                "end": object_currents[name].tolist(),
                                "displacement_m": float(
                                    np.linalg.norm(
                                        object_currents[name] - lift_object_starts[name]
                                    )
                                ),
                                "lifted_height_m": lifted_heights[name],
                            }
                            for name in lift_object_starts
                        },
                        lifted=True,
                        completion_reason="stable_object_height",
                        stable_steps=stable_lift_steps,
                    )
                    self.store.event(
                        "tcp_motion_complete",
                        label=primitive.label,
                        elapsed_s=time.monotonic() - started,
                        errors_m=errors,
                        completion_reason="stable_object_height",
                    )
                    return True
            if placement_object_name is not None:
                placement_position = np.asarray(
                    self.reader.pose(placement_object_name).position, dtype=float
                )
                placement_bounds = self.reader.bounds(placement_object_name)
                if float(placement_bounds.minimum[2]) < PLACEMENT_LOST_LOWEST_Z_M:
                    self.defer_scope_reason = (
                        f"{placement_object_name} fell below the dining support during placement"
                    )
                    self.store.event(
                        "placement_object_lost",
                        label=primitive.label,
                        object=placement_object_name,
                        object_position=placement_position.tolist(),
                        lowest_point_z=float(placement_bounds.minimum[2]),
                        minimum_retained_z=PLACEMENT_LOST_LOWEST_Z_M,
                    )
                    return False
                assignment = self.assignments[placement_object_name]
                placement_target_xy = (
                    target_center[:2]
                    if primitive.metadata.get("placement_object_xy")
                    else np.asarray(assignment.xy, dtype=float)
                )
                placement_tolerance_m = float(
                    primitive.metadata.get(
                        "placement_object_xy_tolerance_m",
                        assignment.tolerance_m,
                    )
                )
                settled_step = (
                    placement_previous_position is not None
                    and float(
                        np.linalg.norm(
                            placement_position - placement_previous_position
                        )
                    )
                    <= PLACEMENT_MAX_STEP_M
                )
                inside_supported_assignment = supported_assignment_reached(
                    placement_position,
                    placement_target_xy,
                    placement_tolerance_m,
                    lowest_point_z=float(placement_bounds.minimum[2]),
                    support_z_min_m=PLACEMENT_SUPPORT_Z_MIN_M,
                    support_z_max_m=PLACEMENT_SUPPORT_Z_MAX_M,
                    score_margin_m=PLACEMENT_SCORE_MARGIN_M,
                )
                stable_placement_steps = (
                    stable_placement_steps + 1
                    if settled_step and inside_supported_assignment
                    else 0
                )
                placement_previous_position = placement_position
                if stable_placement_steps >= PLACEMENT_STABLE_STEPS:
                    distance_m = float(
                        np.linalg.norm(
                            placement_position[:2]
                            - placement_target_xy
                        )
                    )
                    self.store.event(
                        "placement_contact_verification",
                        label=primitive.label,
                        object=placement_object_name,
                        object_position=placement_position.tolist(),
                        lowest_point_z=float(placement_bounds.minimum[2]),
                        assignment_xy=placement_target_xy.tolist(),
                        assignment_tolerance_m=placement_tolerance_m,
                        score_margin_m=PLACEMENT_SCORE_MARGIN_M,
                        distance_m=distance_m,
                        stable_steps=stable_placement_steps,
                        supported=True,
                        completion_reason="stable_supported_assignment",
                    )
                    self.store.event(
                        "tcp_motion_complete",
                        label=primitive.label,
                        elapsed_s=time.monotonic() - started,
                        errors_m=errors,
                        completion_reason="stable_supported_assignment",
                    )
                    return True
            errors = [
                float(
                    np.linalg.norm(
                        actual_positions[side] - paths[side][phase[side]]
                    )
                )
                for side in sides
            ]
            if iteration % 20 == 0:
                self.store.update(
                    message=f"{primitive.label}: TCP error {max(errors):.3f} m"
                )
            if iteration % 100 == 0:
                self.store.event(
                    "tcp_motion_progress",
                    label=primitive.label,
                    elapsed_s=time.monotonic() - started,
                    errors_m=errors,
                    actuals={
                        side.value: actual_positions[side].tolist()
                        for side in sides
                    },
                )
            segment_reached = {
                side: (
                    tcp_segment_reached(
                        actual_positions[side],
                        paths[side][phase[side]],
                        targets[side][0],
                        phase=phase[side],
                        phase_count=len(paths[side]),
                        final_tolerance_m=float(
                            primitive.metadata.get(
                                "position_tolerance_m",
                                PREGRASP_REACH_TOLERANCE_M
                                if (primitive.orientation_hint or "").startswith("top_")
                                and primitive.offset_xyz[2] >= 0.08
                                else (
                                    PLATE_CONTACT_REACH_TOLERANCE_M
                                    if primitive.orientation_hint == "top_plate"
                                    and len(paths[side]) == 1
                                    else (
                                        BOWL_INTERNAL_CONTACT_TOLERANCE_M
                                        if primitive.orientation_hint == "bowl_internal"
                                        and len(paths[side]) == 1
                                        else (
                                            LIFT_REACH_TOLERANCE_M
                                            if primitive.label.startswith("lift ")
                                            else (
                                                0.045
                                                if primitive.orientation_hint == "loaded_transit_stow"
                                                else 0.025
                                            )
                                        )
                                    )
                                ),
                            )
                        ),
                        clearance_final_vertical_tolerance_m=(
                            0.055
                            if primitive.orientation_hint == "top_plate"
                            else 0.030
                        ),
                    )
                    and (
                        targets[side][1] is None
                        or (
                            phase[side] + 1 < len(paths[side])
                            and not (
                                primitive.orientation_hint == "top_plate"
                                and len(paths[side]) == 4
                                and phase[side] == 1
                            )
                        )
                        or _quat_angular_error(actual_orientations[side], targets[side][1])
                        <= math.radians(10.0)
                    )
                )
                for side in sides
            }
            if all(segment_reached.values()):
                advanced = False
                for side in sides:
                    if phase[side] + 1 < len(paths[side]):
                        phase[side] += 1
                        advanced = True
                if advanced:
                    self.store.event(
                        "tcp_segment_complete",
                        label=primitive.label,
                        next_phases={side.value: phase[side] for side in sides},
                        actuals={
                            side.value: actual_positions[side].tolist()
                            for side in sides
                        },
                        actual_orientations={
                            side.value: actual_orientations[side].tolist()
                            for side in sides
                        },
                    )
                    errors = [math.inf]
                    continue
                if placement_object_name is not None:
                    continue
                if "target_recovery_ratio" not in primitive.metadata:
                    if not self._spoon_motion_evidence_satisfied(
                        primitive,
                        feeding_payload_start_spoon,
                        feeding_payload_start_beans,
                    ):
                        return False
                    if lift_object_starts:
                        object_ends = {
                            name: np.asarray(self.reader.pose(name).position, dtype=float)
                            for name in lift_object_starts
                        }
                        minimum_lifts = {
                            name: 0.025 if name == "plate" else 0.035
                            for name in lift_object_starts
                        }
                        lifted = all(
                            float(object_ends[name][2] - start[2])
                            >= minimum_lifts[name]
                            for name, start in lift_object_starts.items()
                        )
                        self.store.event(
                            "lift_transport_verification",
                            label=primitive.label,
                            target=primitive.target,
                            objects={
                                name: {
                                    "start": lift_object_starts[name].tolist(),
                                    "end": object_ends[name].tolist(),
                                    "displacement_m": float(
                                        np.linalg.norm(
                                            object_ends[name] - lift_object_starts[name]
                                        )
                                    ),
                                    "lifted_height_m": float(
                                        object_ends[name][2] - lift_object_starts[name][2]
                                    ),
                                    "minimum_lift_m": minimum_lifts[name],
                                }
                                for name in lift_object_starts
                            },
                            lifted=lifted,
                        )
                        if not lifted:
                            if (
                                primitive.target == "spoon"
                                and self._spoon_grasp_bias_index + 1
                                < len(SPOON_HANDLE_LATERAL_BIAS_SEQUENCE_M)
                            ):
                                previous_bias = self._spoon_handle_lateral_bias_m
                                self._spoon_grasp_bias_index += 1
                                self._spoon_handle_lateral_bias_m = (
                                    SPOON_HANDLE_LATERAL_BIAS_SEQUENCE_M[
                                        self._spoon_grasp_bias_index
                                    ]
                                )
                                self.store.event(
                                    "spoon_grasp_bias_advanced",
                                    previous_bias_m=previous_bias,
                                    next_bias_m=self._spoon_handle_lateral_bias_m,
                                    bias_index=self._spoon_grasp_bias_index,
                                )
                            return False
                    self.store.event(
                        "tcp_motion_complete",
                        label=primitive.label,
                        elapsed_s=time.monotonic() - started,
                        errors_m=errors,
                    )
                    return True
                break
        else:
            self.store.event(
                "tcp_motion_failed",
                label=primitive.label,
                elapsed_s=time.monotonic() - started,
                errors_m=errors,
                actuals={
                    side.value: actual_positions[side].tolist()
                    for side in sides
                },
            )
            if primitive.label == "approach spoon":
                previous_angle = self._spoon_grasp_angle_cap_rad
                self._spoon_grasp_angle_cap_rad = max(
                    SPOON_MIN_SIDE_GRASP_ANGLE_RAD,
                    previous_angle - SPOON_SIDE_GRASP_RETRY_STEP_RAD,
                )
                self._spoon_grasp_orientation = None
                self.store.event(
                    "spoon_grasp_angle_reduced",
                    previous_angle_rad=previous_angle,
                    next_angle_rad=self._spoon_grasp_angle_cap_rad,
                    previous_angle_deg=math.degrees(previous_angle),
                    next_angle_deg=math.degrees(
                        self._spoon_grasp_angle_cap_rad
                    ),
                )
            return False

        target_ratio = float(primitive.metadata["target_recovery_ratio"])
        max_retained_beans = primitive.metadata.get("max_retained_beans")
        dither_deadline = time.monotonic() + float(
            primitive.metadata.get("dither_timeout_s", 20.0)
        )
        dither_step = 0
        while time.monotonic() <= dither_deadline and not self.store.stop_requested:
            recovery_ratio = self.scorer.update().stage3 / 4.0
            retained_beans = self._retained_bowl_bean_count()
            if recovery_ratio >= target_ratio and (
                max_retained_beans is None
                or retained_beans <= int(max_retained_beans)
            ):
                self.store.event(
                    "recovery_payload_released",
                    recovery_ratio=recovery_ratio,
                    retained_beans=retained_beans,
                    max_retained_beans=max_retained_beans,
                )
                return True
            dither_step += 1
            dither_targets: dict[Arm, tuple[np.ndarray, np.ndarray]] = {}
            oscillation = 0.025 * math.sin(2.0 * math.pi * dither_step / 40.0)
            for side, (position, orientation) in targets.items():
                moved = position.copy()
                moved[2] += oscillation if side is Arm.LEFT else -oscillation
                dither_targets[side] = (moved, orientation)
            self.rmp.step(dither_targets)
            self._step()
        self.store.event(
            "recovery_payload_unmet",
            recovery_ratio=self.scorer.update().stage3 / 4.0,
            retained_beans=self._retained_bowl_bean_count(),
            max_retained_beans=max_retained_beans,
        )
        return False

    @staticmethod
    def _contact_components(value: Any) -> list[float]:
        try:
            return [float(component) for component in value]
        except TypeError:
            return [float(value)]

    def _spoon_contact_report(self) -> list[dict[str, Any]]:
        if (
            self._physx_simulation_interface is None
            or self._physics_schema_tools is None
        ):
            return []
        try:
            headers, contact_data = (
                self._physx_simulation_interface.get_contact_report()
            )
        except Exception:
            return []
        reports: list[dict[str, Any]] = []
        for header in headers:
            actor0 = str(self._physics_schema_tools.intToSdfPath(header.actor0))
            actor1 = str(self._physics_schema_tools.intToSdfPath(header.actor1))
            collider0 = str(
                self._physics_schema_tools.intToSdfPath(header.collider0)
            )
            collider1 = str(
                self._physics_schema_tools.intToSdfPath(header.collider1)
            )
            if not any(
                "spoon2_01" in path
                for path in (actor0, actor1, collider0, collider1)
            ):
                continue
            points: list[dict[str, Any]] = []
            start = int(header.contact_data_offset)
            stop = start + int(header.num_contact_data)
            for index in range(start, stop):
                contact = contact_data[index]
                impulse = self._contact_components(contact.impulse)
                points.append(
                    {
                        "position": self._contact_components(contact.position),
                        "normal": self._contact_components(contact.normal),
                        "impulse": impulse,
                        "impulse_magnitude": float(np.linalg.norm(impulse)),
                        "separation": float(contact.separation),
                    }
                )
            reports.append(
                {
                    "type": str(header.type),
                    "actor0": actor0,
                    "actor1": actor1,
                    "collider0": collider0,
                    "collider1": collider1,
                    "points": points,
                }
            )
        return reports

    def _gripper(self, primitive: Primitive) -> bool:
        sides = (Arm.LEFT, Arm.RIGHT) if primitive.arm is Arm.BOTH else (primitive.arm or Arm.LEFT,)
        paired_targets = (primitive.target or "").split("+")
        side_targets = {
            side: (
                paired_targets[0 if side is Arm.LEFT else 1]
                if len(paired_targets) == 2 and len(sides) == 2
                else primitive.target
            )
            for side in sides
        }
        requested_opening = float(np.clip(primitive.opening or 0.0, 0.0, 1.0))
        closing = requested_opening < 0.5
        internal_spread = bool(primitive.metadata.get("internal_spread", False))
        internal_release = bool(primitive.metadata.get("internal_release", False))
        strong_grip = bool(primitive.metadata.get("strong_grip", False))
        # An internal release closes the fingers away from the bowl wall. It
        # must reach the requested driver target instead of terminating early
        # on residual contact while still supporting the bowl.
        contact_motion = (closing and not internal_release) or internal_spread
        # Exact live collision-pad telemetry establishes the physical mapping:
        # driver 0.0 spreads the fingertip pads and driver 0.8 brings them
        # together.  Preserve intermediate semantic openings for bowl support.
        target = 0.8 * (1.0 - requested_opening)
        steps = 80
        controller = self.robot.get_articulation_controller()
        measured = np.asarray(self.robot.get_joint_positions(), dtype=float)
        starts = {
            side: float(measured[self.name_to_index[GRIPPER_DRIVERS[side]]])
            for side in sides
        }
        previous = starts.copy()
        minimum_contact_driver = {
            side: CUP_MIN_CONTACT_DRIVER if side_targets[side] == "cup" else 0.0
            for side in sides
        }
        stall_counts = {side: 0 for side in sides}
        force_counts = {side: 0 for side in sides}
        hard_force_counts = {side: 0 for side in sides}
        peak_efforts = {side: 0.0 for side in sides}
        spoon_contact_pairs: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        spoon_grasp = closing and "spoon" in side_targets.values()
        spoon_start_position = (
            np.asarray(self.reader.pose("spoon").position, dtype=float)
            if spoon_grasp
            else None
        )
        spoon_contact_sides: set[str] = set()
        spoon_displacement_m = 0.0
        spoon_displacement_exceeded = False
        self.store.event(
            "gripper_motion_start",
            label=primitive.label,
            starts={side.value: starts[side] for side in sides},
            target=target,
            side_targets={side.value: side_targets[side] for side in sides},
        )
        self.store.event(
            "gripper_geometry_snapshot",
            label=primitive.label,
            phase="before",
            prims=self._gripper_geometry_snapshot(sides),
        )
        for step in range(steps):
            if self.store.stop_requested:
                return False
            fraction = (step + 1) / steps
            desired: dict[int, float] = {}
            for side in sides:
                driver_index = self.name_to_index.get(GRIPPER_DRIVERS[side])
                if driver_index is None:
                    return False
                desired[driver_index] = starts[side] + (target - starts[side]) * fraction
                self._gripper_hold_targets[side] = desired[driver_index]
                for name, multiplier in GRIPPER_COUPLED[side].items():
                    if name in self.name_to_index:
                        desired[self.name_to_index[name]] = desired[driver_index] * multiplier
            indices = np.asarray(sorted(desired), dtype=np.int64)
            controller.apply_action(
                self.action_type(
                    joint_positions=np.asarray([desired[i] for i in indices], dtype=np.float32),
                    joint_indices=indices,
                )
            )
            self._step()
            if "spoon" in side_targets.values():
                for report in self._spoon_contact_report():
                    key = (
                        report["actor0"],
                        report["actor1"],
                        report["collider0"],
                        report["collider1"],
                    )
                    aggregate = spoon_contact_pairs.setdefault(
                        key,
                        {
                            "actor0": report["actor0"],
                            "actor1": report["actor1"],
                            "collider0": report["collider0"],
                            "collider1": report["collider1"],
                            "event_types": [],
                            "steps": 0,
                            "point_count": 0,
                            "max_impulse": 0.0,
                            "minimum_separation": math.inf,
                            "last_points": [],
                        },
                    )
                    if report["type"] not in aggregate["event_types"]:
                        aggregate["event_types"].append(report["type"])
                    points = report["points"]
                    aggregate["steps"] += 1
                    aggregate["point_count"] += len(points)
                    if points:
                        aggregate["max_impulse"] = max(
                            aggregate["max_impulse"],
                            *(point["impulse_magnitude"] for point in points),
                        )
                        aggregate["minimum_separation"] = min(
                            aggregate["minimum_separation"],
                            *(point["separation"] for point in points),
                        )
                        aggregate["last_points"] = points[:4]
                if spoon_grasp:
                    spoon_contact_sides.clear()
                    for pair in spoon_contact_pairs.values():
                        if (
                            pair["steps"]
                            < SPOON_BILATERAL_CONTACT_CONFIRM_STEPS
                        ):
                            continue
                        actors = f'{pair["actor0"]} {pair["actor1"]}'
                        if "/spoon2_01/Tea_Spoon" not in actors:
                            continue
                        for finger_side in ("left", "right"):
                            if f"/{finger_side}_inner_finger" in actors:
                                spoon_contact_sides.add(finger_side)
                    current_spoon_position = np.asarray(
                        self.reader.pose("spoon").position, dtype=float
                    )
                    spoon_displacement_m = float(
                        np.linalg.norm(
                            current_spoon_position - spoon_start_position
                        )
                    )
                    if spoon_displacement_m > SPOON_GRASP_MAX_DISPLACEMENT_M:
                        spoon_displacement_exceeded = True
                        self.store.event(
                            "spoon_grasp_displacement_exceeded",
                            label=primitive.label,
                            displacement_m=spoon_displacement_m,
                            maximum_displacement_m=SPOON_GRASP_MAX_DISPLACEMENT_M,
                            contact_sides=sorted(spoon_contact_sides),
                        )
                        break
            measured = np.asarray(self.robot.get_joint_positions(), dtype=float)
            efforts = self.robot.get_measured_joint_efforts()
            for side in sides:
                index = self.name_to_index[GRIPPER_DRIVERS[side]]
                current = float(measured[index])
                effort = 0.0 if efforts is None else abs(float(efforts[index]))
                peak_efforts[side] = max(peak_efforts[side], effort)
                effort_threshold = max(0.25, 0.012 * float(primitive.max_force_n or 20.0))
                force_threshold = max(0.8, 0.14 * float(primitive.max_force_n or 20.0))
                hard_force_threshold = max(
                    2.0, 0.30 * float(primitive.max_force_n or 20.0)
                )
                closure_sufficient = current >= minimum_contact_driver[side]
                stalled = (
                    closure_sufficient
                    and abs(current - previous[side]) < 5e-4
                    and effort >= effort_threshold
                )
                stall_counts[side] = stall_counts[side] + 1 if stalled else 0
                force_detected = effort >= hard_force_threshold or (
                    closure_sufficient
                    and (
                        not internal_spread
                        or current <= BOWL_INTERNAL_MAX_CONTACT_DRIVER
                    )
                    and effort >= force_threshold
                )
                force_counts[side] = (
                    force_counts[side] + 1 if force_detected else 0
                )
                hard_force_counts[side] = (
                    hard_force_counts[side] + 1
                    if effort >= hard_force_threshold
                    else 0
                )
                previous[side] = current
            if (
                spoon_grasp
                and {"left", "right"} <= spoon_contact_sides
                and all(
                    float(measured[self.name_to_index[GRIPPER_DRIVERS[side]]])
                    >= SPOON_MIN_CONTACT_DRIVER
                    for side in sides
                )
            ):
                self.store.event(
                    "spoon_bilateral_contact_confirmed",
                    label=primitive.label,
                    contact_sides=sorted(spoon_contact_sides),
                    displacement_m=spoon_displacement_m,
                    confirmation_steps=SPOON_BILATERAL_CONTACT_CONFIRM_STEPS,
                    driver_positions={
                        side.value: float(
                            measured[
                                self.name_to_index[GRIPPER_DRIVERS[side]]
                            ]
                        )
                        for side in sides
                    },
                    minimum_contact_driver=SPOON_MIN_CONTACT_DRIVER,
                )
                break
            if contact_motion and step >= 8 and all(
                (
                    hard_force_counts[side]
                    >= BOWL_INTERNAL_HARD_FORCE_CONFIRM_STEPS
                    if strong_grip
                    else (
                        stall_counts[side] >= 8
                        or force_counts[side] >= 2
                        or hard_force_counts[side]
                        >= BOWL_INTERNAL_HARD_FORCE_CONFIRM_STEPS
                    )
                )
                for side in sides
            ):
                break
        measured = np.asarray(self.robot.get_joint_positions(), dtype=float)
        finals = {
            side: float(measured[self.name_to_index[GRIPPER_DRIVERS[side]]])
            for side in sides
        }
        contacted = contact_motion and all(
            stall_counts[side] >= 8 or force_counts[side] >= 2 for side in sides
        )
        reached = all(abs(finals[side] - target) <= 0.05 for side in sides)
        self.store.event(
            "gripper_motion_complete",
            label=primitive.label,
            target=target,
            finals={side.value: finals[side] for side in sides},
            peak_efforts={side.value: peak_efforts[side] for side in sides},
            stall_counts={side.value: stall_counts[side] for side in sides},
            force_counts={side.value: force_counts[side] for side in sides},
            hard_force_counts={
                side.value: hard_force_counts[side] for side in sides
            },
            force_limited=contact_motion and any(
                force_counts[side] >= 2 for side in sides
            ),
            closing=closing,
            internal_spread=internal_spread,
            internal_release=internal_release,
            strong_grip=strong_grip,
            contacted=contacted,
            reached=reached,
            spoon_contact_sides=sorted(spoon_contact_sides),
            spoon_bilateral_contact=(
                {"left", "right"} <= spoon_contact_sides
            ),
            spoon_displacement_m=spoon_displacement_m,
            spoon_displacement_exceeded=spoon_displacement_exceeded,
            spoon_contact_pairs=[
                {
                    **pair,
                    "minimum_separation": (
                        pair["minimum_separation"]
                        if math.isfinite(pair["minimum_separation"])
                        else None
                    ),
                }
                for pair in spoon_contact_pairs.values()
            ],
        )
        self.store.event(
            "gripper_geometry_snapshot",
            label=primitive.label,
            phase="after",
            prims=self._gripper_geometry_snapshot(sides),
        )
        if spoon_grasp:
            return (
                {"left", "right"} <= spoon_contact_sides
                and all(
                    finals[side] >= SPOON_MIN_CONTACT_DRIVER
                    for side in sides
                )
                and not spoon_displacement_exceeded
            )
        return contacted if strong_grip else contacted or reached

    def _gripper_geometry_snapshot(self, sides: tuple[Arm, ...]) -> list[dict[str, Any]]:
        """Read-only live PhysX poses for installed Robotiq bodies and Lula TCPs."""

        from pxr import Gf, UsdPhysics

        articulation_view = getattr(self.robot, "_articulation_view", None)
        physics_view = getattr(articulation_view, "_physics_view", None)
        if physics_view is None or not hasattr(physics_view, "get_link_transforms"):
            return []
        transforms = np.asarray(physics_view.get_link_transforms())
        paths = list(physics_view.link_paths[0])
        if transforms.ndim != 3 or transforms.shape[0] != 1 or transforms.shape[1] != len(paths):
            return []
        local_bounds_cache = self.reader.UsdGeom.BBoxCache(
            self.reader.Usd.TimeCode.Default(),
            [
                self.reader.UsdGeom.Tokens.default_,
                self.reader.UsdGeom.Tokens.render,
                self.reader.UsdGeom.Tokens.proxy,
            ],
        )
        xform_cache = self.reader.UsdGeom.XformCache(self.reader.Usd.TimeCode.Default())
        samples: list[dict[str, Any]] = []
        for side in sides:
            root = f"/World/Robot/Asset/{side.value}_Robotiq_2F_85/"
            for index, path in enumerate(paths):
                if not path.startswith(root):
                    continue
                value = transforms[0, index]
                qx, qy, qz, qw = (float(component) for component in value[3:7])
                rotation = np.asarray(
                    (
                        (1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
                        (2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)),
                        (2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)),
                    ),
                    dtype=float,
                )
                sample: dict[str, Any] = {
                    "path": path,
                    "type": "PhysXLink",
                    "position": [float(component) for component in value[:3]],
                    "orientation_xyzw": [float(component) for component in value[3:7]],
                }
                prim = self.reader.stage.GetPrimAtPath(path)
                if prim and prim.IsValid():
                    local_range = local_bounds_cache.ComputeLocalBound(prim).ComputeAlignedRange()
                    local_minimum = np.asarray(local_range.GetMin(), dtype=float)
                    local_maximum = np.asarray(local_range.GetMax(), dtype=float)
                    if (
                        np.all(np.isfinite(local_minimum))
                        and np.all(np.isfinite(local_maximum))
                        and float(np.max(np.abs(np.concatenate((local_minimum, local_maximum))))) < 10.0
                    ):
                        corners = np.asarray(
                            [
                                (x, y, z)
                                for x in (local_minimum[0], local_maximum[0])
                                for y in (local_minimum[1], local_maximum[1])
                                for z in (local_minimum[2], local_maximum[2])
                            ],
                            dtype=float,
                        )
                        world_corners = corners @ rotation.T + np.asarray(value[:3], dtype=float)
                        sample["bounds"] = {
                            "minimum": np.min(world_corners, axis=0).tolist(),
                            "maximum": np.max(world_corners, axis=0).tolist(),
                        }
                    if "inner_finger" in path:
                        fingertip_colliders: list[dict[str, Any]] = []
                        prim_range = self.reader.Usd.PrimRange(
                            prim, self.reader.Usd.TraverseInstanceProxies()
                        )
                        for collider_prim in prim_range:
                            collider_path = str(collider_prim.GetPath())
                            if (
                                "PAD_OPEN_" not in collider_path
                                or not collider_prim.HasAPI(UsdPhysics.CollisionAPI)
                            ):
                                continue
                            collider_range = local_bounds_cache.ComputeLocalBound(
                                collider_prim
                            ).ComputeAlignedRange()
                            collider_minimum = collider_range.GetMin()
                            collider_maximum = collider_range.GetMax()
                            relative_transform, _ = xform_cache.ComputeRelativeTransform(
                                collider_prim, prim
                            )
                            link_origin = np.asarray(
                                tuple(relative_transform.Transform(Gf.Vec3d(0.0))),
                                dtype=float,
                            )
                            link_axes = np.asarray(
                                [
                                    np.asarray(
                                        tuple(
                                            relative_transform.Transform(
                                                Gf.Vec3d(
                                                    1.0 if axis == 0 else 0.0,
                                                    1.0 if axis == 1 else 0.0,
                                                    1.0 if axis == 2 else 0.0,
                                                )
                                            )
                                        ),
                                        dtype=float,
                                    )
                                    - link_origin
                                    for axis in range(3)
                                ],
                                dtype=float,
                            )
                            link_corners = np.asarray(
                                [
                                    tuple(
                                        relative_transform.Transform(
                                            Gf.Vec3d(float(x), float(y), float(z))
                                        )
                                    )
                                    for x in (collider_minimum[0], collider_maximum[0])
                                    for y in (collider_minimum[1], collider_maximum[1])
                                    for z in (collider_minimum[2], collider_maximum[2])
                                ],
                                dtype=float,
                            )
                            live_world_corners = (
                                link_corners @ rotation.T
                                + np.asarray(value[:3], dtype=float)
                            )
                            live_world_origin = (
                                link_origin @ rotation.T
                                + np.asarray(value[:3], dtype=float)
                            )
                            live_world_axes = link_axes @ rotation.T
                            fingertip_colliders.append(
                                {
                                    "path": collider_path,
                                    "local_minimum": [
                                        float(component)
                                        for component in collider_minimum
                                    ],
                                    "local_maximum": [
                                        float(component)
                                        for component in collider_maximum
                                    ],
                                    "world_origin": live_world_origin.tolist(),
                                    "world_axes": live_world_axes.tolist(),
                                    "world_corners": live_world_corners.tolist(),
                                    "minimum": np.min(live_world_corners, axis=0).tolist(),
                                    "maximum": np.max(live_world_corners, axis=0).tolist(),
                                }
                            )
                        sample["fingertip_colliders"] = fingertip_colliders
                samples.append(sample)
            tcp_position, tcp_orientation = self.rmp.current_world_pose(side)
            samples.append(
                {
                    "path": f"lula:{side.value}_tcp",
                    "type": "LulaFrame",
                    "position": tcp_position.tolist(),
                    "orientation_wxyz": tcp_orientation.tolist(),
                }
            )
        return samples

    def _retained_feeding_payload_indices(self) -> set[int]:
        spoon_position = np.asarray(self.reader.pose("spoon").position, dtype=float)
        return retained_payload_indices(
            self.reader.bean_positions(),
            sorted(self._feeding_payload_indices),
            spoon_position,
        )

    def _spoon_motion_evidence_satisfied(
        self,
        primitive: Primitive,
        payload_start_spoon: np.ndarray | None,
        payload_start_beans: tuple[np.ndarray, ...] | None,
    ) -> bool:
        maximum_vertical_extent = primitive.metadata.get(
            "max_spoon_vertical_extent_m"
        )
        if maximum_vertical_extent is not None:
            spoon_bounds = self.reader.bounds("spoon")
            vertical_extent = float(
                spoon_bounds.maximum[2] - spoon_bounds.minimum[2]
            )
            self.store.event(
                "spoon_orientation_verification",
                label=primitive.label,
                vertical_extent_m=vertical_extent,
                maximum_vertical_extent_m=float(maximum_vertical_extent),
            )
            if vertical_extent > float(maximum_vertical_extent):
                if primitive.metadata.get("capture_feeding_payload"):
                    self._feeding_payload_indices.clear()
                return False

        if primitive.metadata.get("capture_feeding_payload"):
            if payload_start_spoon is None or payload_start_beans is None:
                raise RuntimeError("feeding payload capture requires start snapshots")
            current_spoon = np.asarray(
                self.reader.pose("spoon").position, dtype=float
            )
            current_beans = tuple(
                np.asarray(position, dtype=float)
                for position in self.reader.bean_positions()
            )
            self._feeding_payload_indices = co_moving_payload_indices(
                payload_start_beans,
                current_beans,
                payload_start_spoon,
                current_spoon,
            )
            self.store.event(
                "scoop_payload_verification",
                label=primitive.label,
                mode="co_motion",
                bean_indices=sorted(self._feeding_payload_indices),
                bean_count=len(self._feeding_payload_indices),
                spoon_start=payload_start_spoon.tolist(),
                spoon_end=current_spoon.tolist(),
                spoon_displacement_m=float(
                    np.linalg.norm(current_spoon - payload_start_spoon)
                ),
            )

        if "minimum_beans" in primitive.metadata:
            retained = self._retained_feeding_payload_indices()
            minimum_beans = int(primitive.metadata["minimum_beans"])
            self.store.event(
                "feeding_payload_retention_verification",
                label=primitive.label,
                verified_indices=sorted(self._feeding_payload_indices),
                retained_indices=sorted(retained),
                retained_count=len(retained),
                minimum_beans=minimum_beans,
            )
            if len(retained) < minimum_beans:
                return False
        return True

    def _wait(self, primitive: Primitive, limits: SafetyLimits) -> bool:
        steps = max(1, int(primitive.duration_s / self.control_dt))
        for _ in range(steps):
            if self.store.stop_requested:
                return False
            if primitive.metadata.get("require_beans"):
                spoon = np.asarray(self.reader.pose("spoon").position)
                head = np.asarray(self.reader.pose("head").position)
                tcp_position, _ = self.rmp.current_world_pose(
                    primitive.arm or Arm.RIGHT
                )
                self.store.telemetry.safety.head_zone_active = (
                    float(np.linalg.norm(tcp_position - self._head_mouth_position))
                    <= limits.head_zone_radius_m
                )
                bean_indices = self._retained_feeding_payload_indices()
                self.scorer.record_feeding_hold(
                    bean_indices=bean_indices,
                    dt=self.control_dt,
                    in_head_zone=float(np.linalg.norm(spoon - head)) <= 0.48,
                )
                if not bean_indices:
                    self.store.event(
                        "feeding_payload_lost",
                        verified_indices=sorted(self._feeding_payload_indices),
                    )
                    return False
            self._step()
        return True

    def _verify(self, primitive: Primitive) -> bool:
        score = self.scorer.update()
        if primitive.target == "beans" and "minimum_ratio" in primitive.metadata:
            return self.scorer.recovery_ratio >= float(primitive.metadata["minimum_ratio"])
        if primitive.metadata.get("region") == "sink":
            return all(self.scorer.evidence.sink_objects[name] for name in (primitive.target or "").split("+"))
        if primitive.metadata.get("returned_to") == "bowl":
            bowl = self.reader.bounds("bowl")
            bean_indices_in_bowl = {
                index
                for index, bean in enumerate(self.reader.bean_positions())
                if (
                bowl.minimum[0] - 0.02 <= bean[0] <= bowl.maximum[0] + 0.02
                and bowl.minimum[1] - 0.02 <= bean[1] <= bowl.maximum[1] + 0.02
                and bowl.minimum[2] - 0.02 <= bean[2] <= bowl.maximum[2] + 0.12
                )
            }
            self.scorer.record_feeding_return(bean_indices_in_bowl)
            held_bean_indices = self.scorer.feeding_held_bean_indices
            self.store.event(
                "feeding_return_verification",
                held_bean_indices=sorted(held_bean_indices),
                returned_bean_indices=sorted(
                    held_bean_indices & bean_indices_in_bowl
                ),
            )
            score = self.scorer.update()
            return score.stage2 >= 4.0
        if primitive.target in self.scorer.evidence.table_objects_correct:
            return self.scorer.evidence.table_objects_correct[primitive.target]
        return True

    def _execute_navigation(
        self,
        primitive: Primitive,
        limits: SafetyLimits,
    ) -> bool:
        target = primitive.target or ""
        previous_failures = self.navigation_failures.get(target, 0)
        retreat_from_station = primitive.metadata.get("retreat_from_station")
        if retreat_from_station is not None:
            root_position, _ = self.robot.get_world_pose()
            current_xy = np.asarray(root_position[:2], dtype=float)
            station_xy = self._target_position(str(retreat_from_station))[:2]
            away_from_station = current_xy - station_xy
            current_clearance = float(np.linalg.norm(away_from_station))
            required_clearance = float(
                primitive.metadata.get("source_station_clearance_m", 0.90)
            )
            if current_clearance < required_clearance:
                self.store.event(
                    "loaded_source_station_retreat_start",
                    target=target,
                    source_station=str(retreat_from_station),
                    start=current_xy.tolist(),
                    station=station_xy.tolist(),
                    current_clearance_m=current_clearance,
                    required_clearance_m=required_clearance,
                )
                retreated = self._withdraw_base(
                    away_from_world=away_from_station,
                    distance_m=required_clearance - current_clearance,
                    limits=limits,
                )
                self.store.event(
                    "loaded_source_station_retreat_complete",
                    target=target,
                    source_station=str(retreat_from_station),
                    success=retreated,
                )
                if not retreated:
                    return False
        success = self._navigate(
            target,
            limits,
            carried_objects=tuple(primitive.metadata.get("carried_objects", ())),
            dining_station=bool(primitive.metadata.get("dining_station", False)),
            dining_final_advance=bool(
                primitive.metadata.get("dining_final_advance", True)
            ),
            nearby_station_acceptance_m=(
                float(primitive.metadata["nearby_station_acceptance_m"])
                if "nearby_station_acceptance_m" in primitive.metadata
                else None
            ),
            manipulation_yaw_tolerance_rad=(
                math.radians(
                    float(primitive.metadata["manipulation_yaw_tolerance_deg"])
                )
                if "manipulation_yaw_tolerance_deg" in primitive.metadata
                else None
            ),
        )
        if (
            not success
            and self.navigation_failures.get(target, 0) <= previous_failures
        ):
            self.navigation_failures[target] = previous_failures + 1
        return success

    def execute(self, primitive: Primitive, limits: SafetyLimits) -> bool:
        if self.store.stop_requested:
            return False
        self.defer_scope_reason = None
        substate = {
            PrimitiveKind.NAVIGATE: Substate.NAVIGATE,
            PrimitiveKind.MOVE_TCP: Substate.APPROACH,
            PrimitiveKind.GRIPPER: (
                Substate.RELEASE
                if primitive.label.startswith("release ")
                else Substate.GRASP
            ),
            PrimitiveKind.WAIT: Substate.CARRY,
            PrimitiveKind.VERIFY: Substate.VERIFY,
        }[primitive.kind]
        self.store.update(substate=substate, message=primitive.label)
        self.store.event("primitive_start", label=primitive.label, kind=primitive.kind.value)
        handlers = {
            PrimitiveKind.NAVIGATE: lambda: self._execute_navigation(
                primitive, limits
            ),
            PrimitiveKind.MOVE_TCP: lambda: self._move_tcp(primitive, limits),
            PrimitiveKind.GRIPPER: lambda: self._gripper(primitive),
            PrimitiveKind.WAIT: lambda: self._wait(primitive, limits),
            PrimitiveKind.VERIFY: lambda: self._verify(primitive),
        }
        success = handlers[primitive.kind]()
        self.store.event("primitive_end", label=primitive.label, success=success)
        return success

    def backoff_and_open(self, arm: Arm) -> None:
        if self.store.stop_requested:
            return
        self.store.update(lifecycle=Lifecycle.RECOVERY, substate=Substate.BACKOFF)
        primitive = Primitive(PrimitiveKind.GRIPPER, "recovery open", arm=arm, opening=1.0)
        self._gripper(primitive)

    def emergency_stop(self, reason: str) -> None:
        self._stop_base()
        self.store.update(lifecycle=Lifecycle.FAILED, failure_reason=reason, message=reason)
        self.store.event("emergency_stop", reason=reason)
