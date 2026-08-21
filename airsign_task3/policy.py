from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from .safety import SafetyLimits
from .types import Lifecycle, Stage, Substate


class Arm(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"


class PrimitiveKind(str, Enum):
    NAVIGATE = "navigate"
    MOVE_TCP = "move_tcp"
    GRIPPER = "gripper"
    WAIT = "wait"
    VERIFY = "verify"


@dataclass(frozen=True)
class Primitive:
    kind: PrimitiveKind
    label: str
    target: str | None = None
    arm: Arm | None = None
    offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation_hint: str | None = None
    duration_s: float = 0.0
    opening: float | None = None
    max_force_n: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyObservation:
    lifecycle: Lifecycle
    stage: Stage
    substate: Substate
    poses: dict[str, Any]
    bounds: dict[str, Any]
    score: float
    recovery_ratio: float
    safety: dict[str, float]


class ActionPolicy(Protocol):
    """Common seam for deterministic, RL, or VLA policies."""

    def propose(self, observation: PolicyObservation) -> Primitive: ...


class PhysicalActuator(Protocol):
    """Only authorized policy-to-simulator mutation surface.

    Implementations may command base, joints, and grippers. They must never
    expose object transform or rigid-body kinematic mutation methods.
    """

    def execute(self, primitive: Primitive, limits: SafetyLimits) -> bool: ...

    def backoff_and_open(self, arm: Arm) -> None: ...

    def emergency_stop(self, reason: str) -> None: ...


def build_table_setup_plan(*, include_plate: bool = False) -> list[Primitive]:
    plan: list[Primitive] = []
    handling = [
        # With the official -90 degree supply-table base yaw, the right arm is
        # the table-facing manipulator.  The left TCP cannot reach even the
        # elevated cup pregrasp from the collision-cleared base standoff.
        # Pick tray objects sequentially with the right arm; the left arm
        # remains available for later bimanual bowl support and carrying.
        # Carry the bean-filled bowl first, before other supply-table motion
        # can disturb it or its contents.
        ("bowl", Arm.RIGHT, "head_seat"),
        # Remove the neighboring cup before approaching the spoon handle.
        ("cup", Arm.RIGHT, "cup_seat"),
        # Place the spoon after the bowl because both share the head-adjacent
        # assignment; the spoon may settle against or inside the bowl without
        # a later bowl placement sweeping it away.
        ("spoon", Arm.RIGHT, "head_seat"),
    ]
    if include_plate:
        handling.insert(2, ("plate", Arm.RIGHT, "plate_seat"))
    for object_name, arm, destination in handling:
        contact_height = 0.025
        grasp_force_n = (
            40.0
            if object_name == "bowl"
            else (40.0 if object_name == "plate" else 30.0)
        )
        pregrasp_hint = (
            "top_bowl_internal"
            if object_name == "bowl"
            else f"top_{object_name}"
        )
        approach_hint = (
            "bowl_internal"
            if object_name == "bowl"
            else f"top_{object_name}"
        )
        grasp_opening = 0.95 if object_name == "bowl" else 0.0
        grasp_metadata = {"internal_spread": True} if object_name == "bowl" else {}
        placement_offset = (0.0, -0.17, 0.055) if object_name == "spoon" else (0.0, 0.0, 0.055)
        object_plan = [
                # Supply routing can include a physical egress, portal transit,
                # and final manipulation yaw. The actuator keeps independent
                # stall bounds; this outer allowance must cover the complete
                # successfully bounded sequence.
                Primitive(
                    PrimitiveKind.NAVIGATE,
                    f"navigate to {object_name}",
                    target=object_name,
                    metadata={
                        "timeout_s": 180.0,
                        **(
                            {"manipulation_yaw_tolerance_deg": 2.5}
                            if object_name == "spoon"
                            else {}
                        ),
                    },
                ),
                Primitive(
                    PrimitiveKind.MOVE_TCP,
                    f"pregrasp {object_name}",
                    target=object_name,
                    arm=arm,
                    offset_xyz=(0.0, 0.0, 0.12),
                    orientation_hint=pregrasp_hint,
                    metadata={"position_tolerance_m": 0.055}
                    if object_name == "spoon"
                    else {},
                ),
                Primitive(PrimitiveKind.MOVE_TCP, f"approach {object_name}", target=object_name, arm=arm,
                          offset_xyz=(0.0, 0.0, contact_height), orientation_hint=approach_hint,
                          metadata={"position_tolerance_m": 0.020}
                          if object_name == "spoon"
                          else {}),
                Primitive(PrimitiveKind.GRIPPER, f"grasp {object_name}", target=object_name, arm=arm,
                          opening=grasp_opening, max_force_n=grasp_force_n,
                          metadata=grasp_metadata),
                Primitive(PrimitiveKind.MOVE_TCP, f"lift {object_name}", target=object_name, arm=arm,
                          offset_xyz=(0.0, 0.0, 0.18), orientation_hint=f"carry_{object_name}",
                          metadata={"position_tolerance_m": 0.020}
                          if object_name == "spoon"
                          else {}),
                # The spoon is taken with a scanned side grasp, so the wrist
                # can end up in a configuration from which the nominal
                # clearance and transit poses are out of reach. Both are
                # posture moves, not task requirements: treat them as best
                # effort for the spoon so an unreachable pose costs a slightly
                # wider carry rather than dropping a spoon that is physically
                # held and restarting the whole contact chain. Runs
                # seed-{0,2}-20260821T23* each lost the spoon this way, with
                # the stow stalling 0.13-0.29 m short while the grasp itself
                # was sound.
                Primitive(PrimitiveKind.MOVE_TCP, f"retract {object_name} from supply table",
                          target=object_name, arm=arm, offset_xyz=(0.20, 0.0, 0.02),
                          orientation_hint=f"carry_{object_name}",
                          metadata={"best_effort": True} if object_name == "spoon" else {}),
                Primitive(PrimitiveKind.MOVE_TCP, f"stow loaded {object_name} for carry",
                          target=object_name, arm=arm,
                          orientation_hint="loaded_transit_stow",
                          metadata={"best_effort": True} if object_name == "spoon" else {}),
                # The measured bowl carry needs about 188 wall-seconds: the
                # navigation itself remains bounded, but loaded portal yaw,
                # corridor alignment, and the final table approach are all
                # part of this primitive. Leave deterministic margin so a
                # physically completed carry is not replayed as a timeout.
                Primitive(PrimitiveKind.NAVIGATE, f"carry {object_name} to seat", target=destination,
                          metadata={
                              "timeout_s": 300.0,
                              "carried_objects": [object_name],
                          }),
                Primitive(PrimitiveKind.MOVE_TCP, f"lower {object_name}", target=destination, arm=arm,
                          offset_xyz=placement_offset, orientation_hint=f"place_{object_name}",
                          metadata={
                              "placement_object_xy": True,
                              "placement_object_xy_tolerance_m": 0.045,
                          }
                          if object_name == "spoon"
                          else {}),
                Primitive(
                    PrimitiveKind.GRIPPER,
                    f"release {object_name}",
                    target=destination,
                    arm=arm,
                    # The bowl is retained by spreading the closed fingers
                    # against its inner wall. Releasing it therefore retracts
                    # the fingers inward; exterior grasps release outward.
                    opening=0.0 if object_name == "bowl" else 1.0,
                    metadata={"internal_release": True} if object_name == "bowl" else {},
                ),
                Primitive(PrimitiveKind.VERIFY, f"verify {object_name} assignment", target=object_name,
                          metadata={"region": destination}),
                Primitive(PrimitiveKind.MOVE_TCP, f"stow right arm after placing {object_name}",
                          target=destination, arm=arm, orientation_hint="transit_stow"),
        ]
        if object_name == "bowl":
            object_plan.insert(
                2,
                Primitive(
                    PrimitiveKind.GRIPPER,
                    "preshape bowl internal",
                    target="bowl",
                    arm=arm,
                    opening=0.0,
                    max_force_n=12.0,
                ),
            )
        if object_name == "plate":
            object_plan.insert(
                2,
                Primitive(
                    PrimitiveKind.GRIPPER,
                    "preshape plate",
                    target="plate",
                    arm=arm,
                    opening=0.95,
                    max_force_n=10.0,
                ),
            )
        if object_name in {"cup", "spoon"}:
            object_plan.insert(
                2,
                Primitive(
                    PrimitiveKind.GRIPPER,
                    f"preshape {object_name} open",
                    target=object_name,
                    arm=arm,
                    opening=1.0,
                    max_force_n=10.0,
                ),
            )
        plan.extend(object_plan)
    return plan


def build_feeding_plan() -> list[Primitive]:
    return [
        Primitive(
            PrimitiveKind.VERIFY,
            "check bowl staged for feeding",
            target="bowl",
            metadata={"skip_stage_on_failure": True},
        ),
        Primitive(
            PrimitiveKind.VERIFY,
            "check spoon staged for feeding",
            target="spoon",
            metadata={"skip_stage_on_failure": True},
        ),
        Primitive(
            PrimitiveKind.NAVIGATE,
            "navigate to head-adjacent seat",
            target="head_seat",
            metadata={
                "timeout_s": 180.0,
                "dining_station": True,
                "nearby_station_acceptance_m": 0.75,
            },
        ),
        Primitive(PrimitiveKind.MOVE_TCP, "right spoon pregrasp", target="spoon", arm=Arm.RIGHT,
                  offset_xyz=(0.0, 0.0, 0.10), orientation_hint="top_spoon"),
        Primitive(PrimitiveKind.MOVE_TCP, "right spoon approach", target="spoon", arm=Arm.RIGHT,
                  offset_xyz=(0.0, 0.0, 0.018), orientation_hint="top_spoon",
                  metadata={"position_tolerance_m": 0.020}),
        Primitive(PrimitiveKind.GRIPPER, "grasp spoon", target="spoon", arm=Arm.RIGHT,
                  opening=0.0, max_force_n=30.0,
                  metadata={"strong_grip": True}),
        Primitive(PrimitiveKind.MOVE_TCP, "lift spoon for feeding", target="spoon", arm=Arm.RIGHT,
                  offset_xyz=(0.0, 0.0, 0.10), orientation_hint="carry_spoon",
                  metadata={"max_spoon_vertical_extent_m": 0.075,
                            "position_tolerance_m": 0.020}),
        Primitive(PrimitiveKind.MOVE_TCP, "scoop entry", target="bowl", arm=Arm.RIGHT,
                  offset_xyz=(-0.03, 0.0, 0.075), orientation_hint="spoon_scoop_entry",
                  metadata={"position_tolerance_m": 0.055}),
        Primitive(PrimitiveKind.MOVE_TCP, "scoop sweep", target="bowl", arm=Arm.RIGHT,
                  offset_xyz=(0.03, 0.0, 0.070), orientation_hint="spoon_scoop_exit",
                  metadata={"position_tolerance_m": 0.055}),
        Primitive(PrimitiveKind.MOVE_TCP, "lift loaded spoon from bowl", target="spoon", arm=Arm.RIGHT,
                  offset_xyz=(0.0, 0.0, 0.15), orientation_hint="carry_spoon",
                  metadata={
                      "capture_feeding_payload": True,
                      "minimum_beans": 1,
                      "max_spoon_vertical_extent_m": 0.075,
                      "position_tolerance_m": 0.020,
                  }),
        Primitive(PrimitiveKind.MOVE_TCP, "feeding approach", target="mouth_standoff", arm=Arm.RIGHT,
                  metadata={"head_safety_zone": True, "minimum_beans": 1,
                            "position_tolerance_m": 0.070,
                            "max_spoon_vertical_extent_m": 0.075}),
        Primitive(PrimitiveKind.WAIT, "feeding hold", target="mouth_standoff", arm=Arm.RIGHT,
                  duration_s=3.2, metadata={"require_beans": True, "head_safety_zone": True}),
        Primitive(PrimitiveKind.MOVE_TCP, "feeding retract", target="mouth_retract", arm=Arm.RIGHT,
                  metadata={"head_safety_zone": True}),
        Primitive(PrimitiveKind.MOVE_TCP, "return beans", target="bowl", arm=Arm.RIGHT,
                  offset_xyz=(0.0, 0.0, 0.04), orientation_hint="spoon_tip_down"),
        Primitive(PrimitiveKind.GRIPPER, "release feeding spoon", target="bowl", arm=Arm.RIGHT,
                  opening=1.0),
        Primitive(PrimitiveKind.VERIFY, "verify feeding and return", target="beans",
                  metadata={"hold_seconds": 3.0, "returned_to": "bowl"}),
    ]


def build_bean_recovery_plan() -> list[Primitive]:
    return [
        # Accept the station the robot is already standing at, but still make
        # the 0.28 m final advance -- the actuator now advances on the nearby
        # path too. Without the advance the right arm stalls 0.16 m short of
        # the bowl pregrasp pose at its reach limit
        # (seed-0-20260821T225102); without the nearby acceptance the full
        # re-route from the head seat timed out on all six attempts
        # (seed-1-20260821T232050). Stage 1 reaches this same seat because its
        # carry navigation ends with that advance.
        Primitive(
            PrimitiveKind.NAVIGATE,
            "navigate to recovery bowl",
            target="bowl",
            metadata={
                "timeout_s": 180.0,
                "dining_station": True,
                "nearby_station_acceptance_m": 0.75,
            },
        ),
        Primitive(
            PrimitiveKind.MOVE_TCP,
            "recovery bowl pregrasp",
            target="bowl",
            arm=Arm.RIGHT,
            offset_xyz=(0.0, 0.0, 0.09),
            orientation_hint="top_bowl_internal",
        ),
        Primitive(
            PrimitiveKind.GRIPPER,
            "preshape recovery bowl internal",
            target="bowl",
            arm=Arm.RIGHT,
            opening=0.0,
            max_force_n=12.0,
        ),
        Primitive(
            PrimitiveKind.MOVE_TCP,
            "recovery bowl approach",
            target="bowl",
            arm=Arm.RIGHT,
            offset_xyz=(0.0, 0.0, 0.025),
            orientation_hint="bowl_internal",
        ),
        Primitive(
            PrimitiveKind.GRIPPER,
            "support recovery bowl",
            target="bowl",
            arm=Arm.RIGHT,
            opening=0.95,
            max_force_n=40.0,
            metadata={"internal_spread": True},
        ),
        Primitive(
            PrimitiveKind.MOVE_TCP,
            "lift recovery bowl",
            target="bowl",
            arm=Arm.RIGHT,
            offset_xyz=(0.0, 0.0, 0.18),
            orientation_hint="carry_bowl",
        ),
        Primitive(
            PrimitiveKind.MOVE_TCP,
            "retract recovery bowl from dining table",
            target="bowl",
            arm=Arm.RIGHT,
            offset_xyz=(0.0, 0.0, 0.02),
            orientation_hint="carry_bowl",
        ),
        Primitive(
            PrimitiveKind.NAVIGATE,
            "carry bowl to recycling",
            target="recycling",
            metadata={
                "timeout_s": 300.0,
                "carried_objects": ["bowl"],
                "retreat_from_station": "head_seat",
                "source_station_clearance_m": 0.90,
            },
        ),
        Primitive(PrimitiveKind.MOVE_TCP, "position bowl over recycling", target="recycling", arm=Arm.RIGHT,
                  offset_xyz=(0.0, 0.0, 0.28), orientation_hint="bowl_level",
                  metadata={"position_tolerance_m": 0.055}),
        Primitive(PrimitiveKind.MOVE_TCP, "controlled bowl tilt", target="recycling", arm=Arm.RIGHT,
                  orientation_hint="bowl_pour_115deg",
                  metadata={
                      "hold_current_position": True,
                      "dither_degrees": 8,
                      "target_recovery_ratio": 0.90,
                      "max_retained_beans": 15,
                      "timeout_s": 90.0,
                      "dither_timeout_s": 20.0,
                      "position_tolerance_m": 0.03,
                      "on_exhaustion_label": "upright recovered bowl",
                      "on_exhaustion_open": False,
                  }),
        Primitive(PrimitiveKind.WAIT, "settle poured beans", target="recycling",
                  arm=Arm.RIGHT, duration_s=3.0),
        Primitive(PrimitiveKind.MOVE_TCP, "upright recovered bowl", target="recycling", arm=Arm.RIGHT,
                  orientation_hint="bowl_level",
                  metadata={
                      "hold_current_position": True,
                      "position_tolerance_m": 0.03,
                  }),
        Primitive(PrimitiveKind.VERIFY, "measure continuous bean recovery", target="beans",
                  metadata={"minimum_ratio": 0.80, "best_effort": True}),
        Primitive(PrimitiveKind.MOVE_TCP, "position recovered bowl over sink", target="sink",
                  arm=Arm.RIGHT, offset_xyz=(0.08, 0.06, 0.10),
                  orientation_hint="carry_bowl",
                  metadata={
                      "position_tolerance_m": 0.08,
                      "on_exhaustion_label": "release recovered bowl",
                      "on_exhaustion_open": False,
                  }),
        Primitive(PrimitiveKind.GRIPPER, "release recovered bowl", target="sink", arm=Arm.RIGHT,
                  opening=0.0, metadata={"internal_release": True}),
        Primitive(PrimitiveKind.VERIFY, "verify recovered bowl inside sink", target="bowl",
                  metadata={"region": "sink"}),
        Primitive(PrimitiveKind.MOVE_TCP, "stow after bean recovery", target="sink", arm=Arm.RIGHT,
                  orientation_hint="transit_stow"),
    ]


def build_cleanup_plan() -> list[Primitive]:
    plan: list[Primitive] = []
    sink_offsets = {
        "bowl": (0.08, 0.06, 0.10),
        "spoon": (-0.10, 0.08, 0.07),
        "cup": (0.08, -0.08, 0.10),
        "plate": (-0.08, -0.08, 0.10),
    }
    for object_name in ("bowl", "spoon", "cup", "plate"):
        internal = object_name == "bowl"
        contact_height = 0.018 if object_name == "spoon" else 0.025
        pregrasp_hint = "top_bowl_internal" if internal else f"top_{object_name}"
        approach_hint = "bowl_internal" if internal else f"top_{object_name}"
        grasp_opening = 0.95 if internal else 0.0
        grasp_force = 40.0 if object_name in {"bowl", "plate"} else 30.0
        plan.extend(
            [
                Primitive(
                    PrimitiveKind.VERIFY,
                    f"check {object_name} already inside sink",
                    target=object_name,
                    metadata={"region": "sink", "skip_scope_on_success": True},
                ),
                Primitive(PrimitiveKind.NAVIGATE, f"navigate to cleanup {object_name}",
                          target=object_name,
                          metadata={
                              # Resolve the station from where the object
                              # actually is: a deferred Stage 1 object is still
                              # on the kitchen supply table, and dining
                              # geometry parks the base too far to lift it.
                              "timeout_s": 180.0,
                              "dining_station": "auto",
                              "manipulation_yaw_tolerance_deg": 15.0,
                              # Cleanup starts from wherever the previous scope
                              # ended, usually already beside the table. Route
                              # planning from there found no viable candidate
                              # at all 14 times in seed-0-20260822T002705,
                              # deferring every one of the four objects. Accept
                              # the station already occupied and let the
                              # actuator orient and make its final approach,
                              # exactly as feeding and bean recovery do.
                              "nearby_station_acceptance_m": 0.75,
                          }),
                # A dining-table pickup is at the right arm's extension limit:
                # the pregrasp converged with z on target and xy 0.13-0.19 m
                # short in seed-{0,1}-20260821T23*. These targets sit above
                # shoulder height, so a lower standoff buys horizontal reach.
                # It still clears the cup and the bowl rim before the descent.
                Primitive(PrimitiveKind.MOVE_TCP, f"pregrasp cleanup {object_name}",
                          target=object_name, arm=Arm.RIGHT,
                          offset_xyz=(0.0, 0.0, 0.09), orientation_hint=pregrasp_hint),
                Primitive(PrimitiveKind.GRIPPER, f"preshape cleanup {object_name}",
                          target=object_name, arm=Arm.RIGHT,
                          opening=0.0 if internal else (0.95 if object_name == "plate" else 1.0),
                          max_force_n=12.0),
                Primitive(PrimitiveKind.MOVE_TCP, f"approach cleanup {object_name}",
                          target=object_name, arm=Arm.RIGHT,
                          offset_xyz=(0.0, 0.0, contact_height), orientation_hint=approach_hint),
                Primitive(PrimitiveKind.GRIPPER, f"grasp cleanup {object_name}",
                          target=object_name, arm=Arm.RIGHT, opening=grasp_opening,
                          max_force_n=grasp_force,
                          metadata={"internal_spread": True} if internal else {}),
                Primitive(PrimitiveKind.MOVE_TCP, f"lift cleanup {object_name}",
                          target=object_name, arm=Arm.RIGHT,
                          offset_xyz=(0.0, 0.0, 0.16), orientation_hint=f"carry_{object_name}"),
                Primitive(PrimitiveKind.MOVE_TCP, f"retract cleanup {object_name}",
                          target=object_name, arm=Arm.RIGHT,
                          offset_xyz=(0.0, 0.0, 0.02), orientation_hint=f"carry_{object_name}"),
                Primitive(PrimitiveKind.MOVE_TCP, f"stow loaded cleanup {object_name}",
                          target=object_name, arm=Arm.RIGHT,
                          orientation_hint="loaded_transit_stow"),
                Primitive(
                    PrimitiveKind.NAVIGATE,
                    f"carry cleanup {object_name} to sink",
                    target="sink",
                    metadata={
                        "timeout_s": 300.0,
                        "carried_objects": [object_name],
                        **(
                            {
                                "retreat_from_station": (
                                    "cup_seat"
                                    if object_name == "cup"
                                    else "head_seat"
                                ),
                                "source_station_clearance_m": 0.90,
                            }
                            if object_name in {"bowl", "spoon", "cup"}
                            else {}
                        ),
                    },
                ),
                Primitive(PrimitiveKind.MOVE_TCP, f"lower {object_name}", target="sink", arm=Arm.RIGHT,
                          offset_xyz=sink_offsets[object_name], orientation_hint=f"place_{object_name}"),
                Primitive(PrimitiveKind.GRIPPER, f"release cleanup {object_name}", target="sink",
                          arm=Arm.RIGHT, opening=0.0 if internal else 1.0,
                          metadata={"internal_release": True} if internal else {}),
                Primitive(PrimitiveKind.VERIFY, f"verify cleanup {object_name} inside sink",
                          target=object_name,
                          metadata={"region": "sink"}),
                Primitive(PrimitiveKind.MOVE_TCP, f"stow after cleanup {object_name}", target="sink",
                          arm=Arm.RIGHT, orientation_hint="transit_stow"),
            ]
        )
    return plan


def build_full_plan() -> dict[Stage, list[Primitive]]:
    return {
        Stage.TABLE_SETUP: build_table_setup_plan(),
        Stage.FEEDING: build_feeding_plan(),
        Stage.BEAN_RECOVERY: build_bean_recovery_plan(),
        Stage.CLEANUP: build_cleanup_plan(),
    }


class HierarchicalPolicyRunner:
    def __init__(self, actuator: PhysicalActuator, limits: SafetyLimits | None = None) -> None:
        self.actuator = actuator
        self.limits = limits or SafetyLimits()
        self.stage_plans = build_full_plan()
        self.stage_index = 0
        self.primitive_index = 0
        self.retries = 0
        self.max_retries = 3
        self.navigation_max_retries = 6
        self.failed = False
        self.deferred_failures: list[str] = []
        self._recovery_resume_index: int | None = None

    @property
    def stage(self) -> Stage:
        return tuple(self.stage_plans)[self.stage_index]

    @property
    def current(self) -> Primitive:
        return self.stage_plans[self.stage][self.primitive_index]

    def _scope_end(self, start_index: int) -> int:
        plan = self.stage_plans[self.stage]
        if self.stage is Stage.TABLE_SETUP:
            boundary = "navigate to "
        elif self.stage is Stage.CLEANUP:
            boundary = "check "
        else:
            return len(plan)
        return next(
            (
                index
                for index in range(start_index + 1, len(plan))
                if plan[index].label.startswith(boundary)
            ),
            len(plan),
        )

    def _index_for_label(self, label: str) -> int:
        return next(
            index
            for index, primitive in enumerate(self.stage_plans[self.stage])
            if primitive.label == label
        )

    def _advance_stage_if_needed(self) -> bool:
        if self.primitive_index < len(self.stage_plans[self.stage]):
            return False
        self.stage_index += 1
        self.primitive_index = 0
        self.retries = 0
        self._recovery_resume_index = None
        if self.stage_index < len(self.stage_plans):
            return False
        self.stage_index = len(self.stage_plans) - 1
        self.primitive_index = len(self.stage_plans[self.stage]) - 1
        return True

    def _table_setup_recovery(self, failed_index: int) -> tuple[int, bool]:
        """Return a safe replay point and whether the jaws should be opened.

        Contact-chain failures must replay pregrasp, optional preshape,
        approach, and grasp; retrying a lift after opening can never reacquire
        the object.  Carry/lower failures retain the grasp.
        """
        plan = self.stage_plans[Stage.TABLE_SETUP]
        group_start = failed_index
        while group_start > 0 and not plan[group_start].label.startswith("navigate to "):
            group_start -= 1
        failed_label = plan[failed_index].label
        if failed_label.startswith(
            (
                "pregrasp ",
                "preshape ",
                "approach ",
                "grasp ",
                "lift ",
                "stow loaded ",
            )
        ):
            return group_start + 1, True
        if failed_label.startswith("lower "):
            return group_start, True
        if failed_label.startswith("verify "):
            return group_start, True
        return failed_index, False

    def _recovery_action(self, failed_index: int) -> tuple[int, bool]:
        if self.stage is Stage.TABLE_SETUP:
            return self._table_setup_recovery(failed_index)
        if self.stage is Stage.FEEDING:
            plan = self.stage_plans[Stage.FEEDING]
            label = plan[failed_index].label
            if plan[failed_index].kind is PrimitiveKind.NAVIGATE:
                return failed_index, False
            spoon_pregrasp = next(
                index
                for index, item in enumerate(plan)
                if item.label == "right spoon pregrasp"
            )
            scoop_entry = next(
                index
                for index, item in enumerate(plan)
                if item.label == "scoop entry"
            )
            if label in {
                "scoop entry",
                "scoop sweep",
                "feeding approach",
                "feeding hold",
                "feeding retract",
            }:
                # The spoon is still physically held; move it back to the bowl
                # and repeat the bounded scoop without opening the gripper.
                return scoop_entry, False
            if label == "verify feeding and return":
                # Both objects have been released, so rebuild the whole stage.
                return 0, True
            if failed_index >= spoon_pregrasp:
                return spoon_pregrasp, True
            return spoon_pregrasp, True
        if self.stage is Stage.BEAN_RECOVERY:
            plan = self.stage_plans[Stage.BEAN_RECOVERY]
            label = plan[failed_index].label
            pregrasp = next(
                index
                for index, item in enumerate(plan)
                if item.label == "recovery bowl pregrasp"
            )
            if failed_index < pregrasp:
                return failed_index, False
            if label in {
                "recovery bowl pregrasp",
                "preshape recovery bowl internal",
                "recovery bowl approach",
                "support recovery bowl",
                "lift recovery bowl",
                "retract recovery bowl from dining table",
            }:
                return pregrasp, True
            return failed_index, False
        if self.stage is Stage.CLEANUP:
            plan = self.stage_plans[Stage.CLEANUP]
            group_start = failed_index
            while group_start > 0 and not plan[group_start].label.startswith("check "):
                group_start -= 1
            navigate = group_start + 1
            label = plan[failed_index].label
            if label.startswith(
                (
                    "pregrasp cleanup ",
                    "preshape cleanup ",
                    "approach cleanup ",
                    "grasp cleanup ",
                    "lift cleanup ",
                    "retract cleanup ",
                    "stow loaded cleanup ",
                )
            ):
                return navigate + 1, True
            if label.startswith("verify cleanup "):
                return group_start, True
            return failed_index, False
        return failed_index, True

    def _defer_current_scope(self, primitive: Primitive) -> tuple[bool, str]:
        # A station the base cannot reach is a routing failure, not a safety
        # hazard. Ending the episode here forfeits every later stage: in seed-2
        # of the 2026-08-22 acceptance set the cup carry exhausted its retries
        # during Stage 1 and the run stopped at 1.0 points with feeding,
        # recovery and cleanup untried. Set the object down and carry on with
        # the next scope instead. Genuine hazards -- a lost carried object, a
        # lost actuator -- still reach `tick` through
        # `unrecoverable_failure_reason` and still emergency-stop there.
        carried_objects = tuple(primitive.metadata.get("carried_objects", ()))
        destination_label = primitive.metadata.get("on_exhaustion_label")
        if destination_label is None:
            self.primitive_index = self._scope_end(self.primitive_index)
        else:
            self.primitive_index = self._index_for_label(str(destination_label))
        # Always free the jaws when giving up mid-carry, so the next scope
        # starts with an empty gripper rather than an object it cannot use.
        should_open = bool(
            primitive.metadata.get("on_exhaustion_open", True)
        ) or bool(carried_objects)
        if should_open:
            arm = {
                Stage.TABLE_SETUP: Arm.RIGHT,
                Stage.FEEDING: Arm.BOTH,
                Stage.BEAN_RECOVERY: Arm.RIGHT,
                Stage.CLEANUP: Arm.RIGHT,
            }[self.stage]
            self.actuator.backoff_and_open(primitive.arm or arm)
        failure = f"{self.stage.value}:{primitive.label}"
        if carried_objects:
            failure += " (set down " + ", ".join(str(n) for n in carried_objects) + ")"
        self.deferred_failures.append(failure)
        self.retries = 0
        self._recovery_resume_index = None
        complete = self._advance_stage_if_needed()
        return True, "complete" if complete else f"deferred {failure}; continuing"

    def tick(self) -> tuple[bool, str]:
        if self.failed:
            return False, "policy already failed"
        primitive = self.current
        started = time.monotonic()
        ok = self.actuator.execute(primitive, self.limits)
        elapsed = time.monotonic() - started
        timeout = float(primitive.metadata.get("timeout_s", self.limits.command_timeout_s))
        if elapsed > timeout:
            ok = False
        if not ok:
            unrecoverable = getattr(
                self.actuator, "unrecoverable_failure_reason", None
            )
            if unrecoverable:
                self.failed = True
                self.actuator.emergency_stop(str(unrecoverable))
                return False, str(unrecoverable)
            defer_scope_reason = getattr(
                self.actuator, "defer_scope_reason", None
            )
            if defer_scope_reason:
                setattr(self.actuator, "defer_scope_reason", None)
                return self._defer_current_scope(primitive)
        if primitive.metadata.get("skip_scope_on_success"):
            if ok:
                self.primitive_index = self._scope_end(self.primitive_index)
                message = f"already satisfied: {primitive.label}"
            else:
                self.primitive_index += 1
                message = f"action required: {primitive.label}"
            self.retries = 0
            self._recovery_resume_index = None
            complete = self._advance_stage_if_needed()
            return True, "complete" if complete else message
        if primitive.metadata.get("skip_stage_on_failure"):
            if ok:
                self.primitive_index += 1
                message = primitive.label
            else:
                failure = f"{self.stage.value}:{primitive.label}"
                self.deferred_failures.append(failure)
                self.primitive_index = len(self.stage_plans[self.stage])
                message = f"deferred {failure}; continuing"
            self.retries = 0
            self._recovery_resume_index = None
            complete = self._advance_stage_if_needed()
            return True, "complete" if complete else message
        if not ok and primitive.metadata.get("best_effort"):
            self.primitive_index += 1
            self.retries = 0
            self._recovery_resume_index = None
            complete = self._advance_stage_if_needed()
            return True, "complete" if complete else f"best effort unmet: {primitive.label}"
        if not ok:
            retry_limit = (
                self.navigation_max_retries
                if primitive.kind is PrimitiveKind.NAVIGATE
                else self.max_retries
            )
            if self.retries >= retry_limit:
                return self._defer_current_scope(primitive)
            self.retries += 1
            failed_index = self.primitive_index
            recovery_index, should_open = self._recovery_action(failed_index)
            if should_open:
                self.actuator.backoff_and_open(primitive.arm or Arm.BOTH)
            self.primitive_index = recovery_index
            self._recovery_resume_index = max(
                failed_index,
                self._recovery_resume_index
                if self._recovery_resume_index is not None
                else failed_index,
            )
            return False, f"recovery {self.retries}/{retry_limit}: {primitive.label}"
        self.primitive_index += 1
        if (
            self._recovery_resume_index is None
            or self.primitive_index > self._recovery_resume_index
        ):
            self.retries = 0
            self._recovery_resume_index = None
        if self._advance_stage_if_needed():
            return True, "complete"
        return True, primitive.label
