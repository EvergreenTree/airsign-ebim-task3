from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

from .benchmark import BENCHMARK_COMMIT, policy_source_hash, validate_benchmark_revision
from .dashboard import start_dashboard
from .runtime import RuntimeStore
from .types import EpisodeTelemetry, Lifecycle, Stage, Substate


ROBOT_PRIM_PATH = "/World/Robot"
ARM_READY_POSE = {
    # Symmetric transit fold selected by the development-only --pose-scan.
    # Its measured world-X span at the Task 3 start yaw is 0.770 m versus
    # 1.875 m for the upstream manipulation default and a 1.20 m doorway.
    "left_fr3v2_joint1": 2.4,
    "left_fr3v2_joint2": -1.5,
    "left_fr3v2_joint3": 2.4,
    "left_fr3v2_joint4": -2.2,
    "left_fr3v2_joint5": 0.0,
    "left_fr3v2_joint6": 1.5,
    "left_fr3v2_joint7": 0.785,
    "right_fr3v2_joint1": -2.4,
    "right_fr3v2_joint2": -1.5,
    "right_fr3v2_joint3": -2.4,
    "right_fr3v2_joint4": -2.2,
    "right_fr3v2_joint5": 0.0,
    "right_fr3v2_joint6": 1.5,
    "right_fr3v2_joint7": 0.785,
    # Exact live collision-pad calibration: driver/coupled value 0.0 spreads
    # the installed Robotiq pads; 0.8 brings them together.  Start fully open
    # so the first approach can physically straddle the plate chord.
    "left_right_finger_joint": 0.0,
    "right_right_finger_joint": 0.0,
    "left_robotiq_85_left_knuckle_joint": 0.0,
    "left_robotiq_85_right_knuckle_joint": 0.0,
    "left_robotiq_85_left_inner_knuckle_joint": 0.0,
    "left_robotiq_85_right_inner_knuckle_joint": 0.0,
    "left_robotiq_85_left_finger_tip_joint": 0.0,
    "left_robotiq_85_right_finger_tip_joint": 0.0,
    "right_robotiq_85_left_knuckle_joint": 0.0,
    "right_robotiq_85_right_knuckle_joint": 0.0,
    "right_robotiq_85_left_inner_knuckle_joint": 0.0,
    "right_robotiq_85_right_inner_knuckle_joint": 0.0,
    "right_robotiq_85_left_finger_tip_joint": 0.0,
    "right_robotiq_85_right_finger_tip_joint": 0.0,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AirSign EBiM Task 3 native Isaac Sim runner")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ui-port", type=int, default=18091)
    parser.add_argument("--native-webrtc", action="store_true")
    parser.add_argument("--livestream-public-endpoint", default="127.0.0.1")
    parser.add_argument("--livestream-signal-port", type=int, default=49100)
    parser.add_argument("--livestream-media-port", type=int, default=47998)
    parser.add_argument("--benchmark-root", type=Path,
                        default=Path(os.environ.get(
                            "AIRSIGN_TASK3_BENCHMARK_ROOT",
                            "/mnt/nas/evergreen/ebim-task3/benchmark",
                        )))
    parser.add_argument("--run-root", type=Path,
                        default=Path(os.environ.get(
                            "AIRSIGN_TASK3_RUN_ROOT",
                            "/mnt/nas/evergreen/ebim-task3/runs",
                        )))
    parser.add_argument("--head-placement", default="random")
    parser.add_argument("--physics-hz", type=float, default=120.0)
    parser.add_argument("--render-hz", type=float, default=20.0)
    parser.add_argument("--calibration-steps", type=int, default=60)
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--wait-for-start", action="store_true")
    parser.add_argument("--pose-scan", action="store_true")
    parser.add_argument(
        "--development-table-object",
        choices=("plate", "cup", "bowl", "spoon"),
        default=None,
        help="Run only one physically actuated Stage 1 object chain for calibration.",
    )
    parser.add_argument(
        "--development-feeding",
        action="store_true",
        help="Physically stage only bowl and spoon, then run the feeding stage.",
    )
    parser.add_argument(
        "--development-recovery",
        action="store_true",
        help="Physically stage only the bowl, then run the bean-recovery stage.",
    )
    return parser


def _iter_prims_under(root_prim: Any):
    yield root_prim
    for child in root_prim.GetChildren():
        yield from _iter_prims_under(child)


def _fix_single_articulation_root(stage: Any, robot_prim_path: str, usd_physics: Any) -> None:
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    roots = [
        prim for prim in _iter_prims_under(robot_prim)
        if prim.HasAPI(usd_physics.ArticulationRootAPI)
    ]
    if len(roots) <= 1:
        return
    preferred = stage.GetPrimAtPath(f"{robot_prim_path}/base")
    keep = preferred if preferred in roots else roots[0]
    for prim in roots:
        if prim != keep:
            prim.RemoveAPI(usd_physics.ArticulationRootAPI)


def _configure_gripper_pad_material(
    stage: Any,
    robot_prim_path: str,
    usd: Any,
    usd_physics: Any,
) -> int:
    """Bind a rubber-like physics material to the four inner-finger links."""

    from pxr import UsdShade

    material = UsdShade.Material.Define(stage, "/World/PhysicsMaterials/AirSignRobotiqPad")
    physics_material = usd_physics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr(1.4)
    physics_material.CreateDynamicFrictionAttr(1.2)
    physics_material.CreateRestitutionAttr(0.0)
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    bound = 0
    for prim in usd.PrimRange(robot_prim, usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if not path.endswith("_inner_finger") or not prim.HasAPI(usd_physics.RigidBodyAPI):
            continue
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material,
            UsdShade.Tokens.strongerThanDescendants,
            "physics",
        )
        bound += 1
    return bound


def _configure_spoon_contact_reporting(
    stage: Any,
    usd: Any,
    usd_physics: Any,
    physx_schema: Any,
) -> list[str]:
    root = stage.GetPrimAtPath("/World/Environment/RobotRoom/Asset/spoon2_01")
    configured: list[str] = []
    for prim in usd.PrimRange(root, usd.TraverseInstanceProxies()):
        if not prim.HasAPI(usd_physics.RigidBodyAPI):
            continue
        report = physx_schema.PhysxContactReportAPI.Apply(prim)
        report.CreateThresholdAttr().Set(0.0)
        configured.append(str(prim.GetPath()))
    return configured


def _find_articulation_root(stage: Any, robot_prim_path: str, usd_physics: Any) -> str:
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    for prim in _iter_prims_under(robot_prim):
        if prim.HasAPI(usd_physics.ArticulationRootAPI):
            return str(prim.GetPath())
    return robot_prim_path


def _find_physics_scene(stage: Any) -> str:
    for prim in stage.Traverse():
        if str(prim.GetTypeName()) == "PhysicsScene":
            return str(prim.GetPath())
    return "/physicsScene"


def _configure_base_drives(stage: Any, robot_prim_path: str, usd_physics: Any) -> None:
    gains = {
        "tmrv0_2_joint_0": (500.0, 50.0, 200.0),
        "tmrv0_2_joint_2": (500.0, 50.0, 200.0),
        "tmrv0_2_joint_1": (0.0, 5.0, 500.0),
        "tmrv0_2_joint_3": (0.0, 5.0, 500.0),
    }
    for prim in _iter_prims_under(stage.GetPrimAtPath(robot_prim_path)):
        values = gains.get(prim.GetName())
        if values is None or not prim.IsA(usd_physics.Joint):
            continue
        schema = "linear" if prim.IsA(usd_physics.PrismaticJoint) else "angular"
        drive = usd_physics.DriveAPI.Apply(prim, schema)
        drive.CreateStiffnessAttr().Set(values[0])
        drive.CreateDampingAttr().Set(values[1])
        drive.CreateMaxForceAttr().Set(values[2])


def _apply_ready_pose(robot: Any, articulation_action: Any) -> tuple[np.ndarray, np.ndarray]:
    actual = {name: index for index, name in enumerate(robot.dof_names)}
    indices = np.asarray([actual[name] for name in ARM_READY_POSE if name in actual], dtype=np.int64)
    positions = np.asarray([ARM_READY_POSE[name] for name in ARM_READY_POSE if name in actual], dtype=np.float32)
    if len(indices):
        robot.set_joint_positions(positions, joint_indices=indices)
        robot.get_articulation_controller().apply_action(
            articulation_action(joint_positions=positions, joint_indices=indices)
        )
    return indices, positions


def _robot_world_bounds(stage: Any, usd: Any, usd_geom: Any) -> dict[str, list[float]]:
    aligned = usd_geom.BBoxCache(
        usd.TimeCode.Default(),
        [usd_geom.Tokens.default_, usd_geom.Tokens.render, usd_geom.Tokens.proxy],
    ).ComputeWorldBound(stage.GetPrimAtPath(ROBOT_PRIM_PATH)).ComputeAlignedRange()
    minimum = [float(value) for value in aligned.GetMin()]
    maximum = [float(value) for value in aligned.GetMax()]
    return {
        "minimum": minimum,
        "maximum": maximum,
        "size": [
            upper - lower
            for lower, upper in zip(minimum, maximum, strict=True)
        ],
    }


def _scan_compact_pose(world: Any, robot: Any, stage: Any, usd: Any, usd_geom: Any) -> None:
    """Development-only coarse scan; final runs command the chosen pose physically."""
    name_to_index = {name: index for index, name in enumerate(robot.dof_names)}
    required = [
        "left_fr3v2_joint1",
        "right_fr3v2_joint1",
        "left_fr3v2_joint3",
        "right_fr3v2_joint3",
    ]
    if any(name not in name_to_index for name in required):
        raise RuntimeError("pose scan requires both FR3 joint1/joint3 pairs")
    all_positions = np.asarray(robot.get_joint_positions(), dtype=np.float32)
    values = np.linspace(-2.4, 2.4, 7)
    results: list[dict[str, Any]] = []
    for joint1_sign in (-1.0, 1.0):
        for joint3_sign in (-1.0, 1.0):
            for joint1 in values:
                for joint3 in values:
                    candidate = all_positions.copy()
                    candidate[name_to_index["left_fr3v2_joint1"]] = joint1
                    candidate[name_to_index["right_fr3v2_joint1"]] = joint1_sign * joint1
                    candidate[name_to_index["left_fr3v2_joint3"]] = joint3
                    candidate[name_to_index["right_fr3v2_joint3"]] = joint3_sign * joint3
                    robot.set_joint_positions(candidate)
                    world.step(render=False)
                    bounds = _robot_world_bounds(stage, usd, usd_geom)
                    results.append(
                        {
                            "joint1": [
                                float(joint1),
                                float(joint1_sign * joint1),
                            ],
                            "joint3": [
                                float(joint3),
                                float(joint3_sign * joint3),
                            ],
                            "size": bounds["size"],
                        }
                    )
    ranked = sorted(results, key=lambda row: (row["size"][0], row["size"][1]))
    print("POSE_SCAN " + json.dumps(ranked[:20], separators=(",", ":")), flush=True)


def _create_overview_capture(rep: Any, stage: Any):
    # The pinned room builder positions this camera using its vetted
    # INITIAL_VIEW_POSE.  Capturing the same prim keeps the browser view
    # identical to the upstream interactive viewport and avoids placing a
    # second camera outside the room shell.
    camera_path = "/OmniverseKit_Persp"
    camera_prim = stage.GetPrimAtPath(camera_path)
    if not camera_prim or not camera_prim.IsValid():
        raise RuntimeError(f"official overview camera missing: {camera_path}")
    render_product = rep.create.render_product(camera_path, (960, 540))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach(render_product)
    return annotator, render_product


def _create_supply_capture(rep: Any, stage: Any, usd_geom: Any, gf: Any):
    """Create a fixed close view of the supply table and gripper workspace."""

    camera_path = "/World/AirSignCameras/Supply"
    camera = usd_geom.Camera.Define(stage, camera_path)
    camera.CreateFocalLengthAttr(30.0)
    camera.CreateHorizontalApertureAttr(20.955)
    eye = gf.Vec3d(-4.65, -2.15, 2.05)
    target = gf.Vec3d(-5.20, -1.49, 0.79)
    transform = gf.Matrix4d(1.0).SetLookAt(eye, target, gf.Vec3d(0.0, 0.0, 1.0)).GetInverse()
    xformable = usd_geom.Xformable(camera.GetPrim())
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp().Set(transform)
    render_product = rep.create.render_product(camera_path, (640, 360))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach(render_product)
    return annotator, render_product


def _create_head_capture(
    rep: Any,
    stage: Any,
    usd_geom: Any,
    gf: Any,
    head_position: tuple[float, float, float],
):
    """Create a close safety view from the dining-table side of the head."""

    camera_path = "/World/AirSignCameras/HeadSafety"
    camera = usd_geom.Camera.Define(stage, camera_path)
    camera.CreateFocalLengthAttr(46.0)
    camera.CreateHorizontalApertureAttr(20.955)
    head = np.asarray(head_position, dtype=float)
    table_center = np.asarray((-2.1, 1.95), dtype=float)
    inward = table_center - head[:2]
    inward /= max(float(np.linalg.norm(inward)), 1e-9)
    # The dining booth has a tall upholstered divider on the table side.  A
    # near-horizontal camera at 1.16 m sees that divider rather than the bust.
    # Observe from above its envelope and aim at the live eye/mouth height.
    eye_xy = head[:2] + 0.75 * inward
    eye = gf.Vec3d(float(eye_xy[0]), float(eye_xy[1]), 1.75)
    target = gf.Vec3d(float(head[0]), float(head[1]), 0.90)
    transform = gf.Matrix4d(1.0).SetLookAt(
        eye, target, gf.Vec3d(0.0, 0.0, 1.0)
    ).GetInverse()
    xformable = usd_geom.Xformable(camera.GetPrim())
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp().Set(transform)
    render_product = rep.create.render_product(camera_path, (640, 360))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach(render_product)
    return annotator, render_product


def _frame_to_bgr(data: Any) -> np.ndarray | None:
    if data is None:
        return None
    array = np.asarray(data)
    if array.ndim != 3 or array.shape[2] < 3 or array.size == 0:
        return None
    rgb = array[..., :3]
    if np.issubdtype(rgb.dtype, np.floating):
        peak = float(np.nanmax(rgb))
        if peak <= 1.0:
            rgb = np.clip(rgb * 255.0, 0.0, 255.0)
        rgb = rgb.astype(np.uint8)
    elif rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb[..., ::-1])


def _read_annotator_frame(annotator: Any) -> np.ndarray | None:
    """Read a camera frame without letting transient RTX metadata abort a run."""
    try:
        data = annotator.get_data()
    except Exception:
        # Replicator can briefly expose a render product before its overscan
        # metadata is populated, especially when another Kit process is using
        # the GPU.  A later rendered frame is valid; policy physics must not be
        # coupled to this optional review stream.
        return None
    return _frame_to_bgr(data)


def _calibration_motion(
    world: Any,
    robot: Any,
    action_type: Any,
    indices: np.ndarray,
    ready: np.ndarray,
    store: RuntimeStore,
    annotator: Any,
    total_steps: int = 60,
) -> None:
    """Small joint-drive motion; task objects are never modified."""
    if not len(indices):
        raise RuntimeError("no FR3 arm joints found for calibration")
    actual_names = [robot.dof_names[int(index)] for index in indices]
    selected_positions = [i for i, name in enumerate(actual_names) if name.endswith("joint1")]
    if not selected_positions:
        selected_positions = [0]
    controller = robot.get_articulation_controller()
    total_steps = max(12, int(total_steps))
    for step in range(total_steps):
        phase = 2.0 * math.pi * step / (total_steps - 1)
        targets = ready.copy()
        for selected in selected_positions:
            targets[selected] += 0.055 * math.sin(phase)
        controller.apply_action(action_type(joint_positions=targets, joint_indices=indices))
        render_step = step % 4 == 0 or step + 1 == total_steps
        world.step(render=render_step)
        if render_step:
            frame = _read_annotator_frame(annotator)
            if frame is not None:
                store.set_frame("overview", frame)
        store.update(
            simulated_seconds=world.current_time,
            message=f"Actuator calibration {100 * (step + 1) / total_steps:.0f}%",
        )
    store.update(calibration_complete=True, message="Scene ready; calibration motion complete")
    store.event("calibration_complete", joints=[actual_names[i] for i in selected_positions])


def run(args: argparse.Namespace) -> int:
    root = args.benchmark_root.resolve()
    validate_benchmark_revision(root)
    run_dir = args.run_root / f"seed-{args.seed}-{time.strftime('%Y%m%dT%H%M%S')}"
    telemetry = EpisodeTelemetry(seed=args.seed)
    store = RuntimeStore(telemetry, run_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("PRIVACY_CONSENT", "Y")

    import isaacsim
    from isaacsim import SimulationApp

    launch_config: dict[str, Any] = {
        "headless": args.headless,
        "width": 1280,
        "height": 720,
        "renderer": "RaytracedLighting",
    }
    experience = ""
    if args.native_webrtc:
        experience = str(
            Path(next(iter(isaacsim.__path__)))
            / "apps"
            / "isaacsim.exp.full.streaming.kit"
        )
        launch_config["extra_args"] = [
            (
                "--/app/livestream/publicEndpointAddress="
                f"{args.livestream_public_endpoint}"
            ),
            f"--/app/livestream/port={args.livestream_signal_port}",
            # Isaac Sim 5.1 separates the UDP socket it binds from the UDP
            # port advertised alongside publicEndpointAddress.  Keep both
            # deterministic so an SSH/VPN relay can target the documented
            # media endpoint.
            f"--/app/livestream/fixedHostPort={args.livestream_media_port}",
            f"--/app/livestream/publicEndpointPort={args.livestream_media_port}",
        ]
    simulation_app = SimulationApp(launch_config, experience=experience)

    scripts_dir = root / "scripts" / "scenes"
    task3_scripts = root / "task3_isaacsim" / "scripts"
    for path in (scripts_dir, task3_scripts):
        sys.path.insert(0, str(path))

    import omni.kit.app
    import omni.replicator.core as rep
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.prims import RigidPrim, SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

    import scene_robot_room_keyboard as room_scene
    from gripper_profiles import get_gripper_profile

    from .isaac_state import IsaacStateReader
    from .isaac_actuator import IsaacPhysicalActuator
    from .live_scorer import OfficialLiveScorer, targets_from_head
    from .policy import HierarchicalPolicyRunner
    from .scoring import highest_completed_stage

    profile = get_gripper_profile("robotiq")
    room_path = room_scene.asset_path("robot_room.usd")
    robot_path = profile.robot_usd
    for path in (room_path, robot_path):
        if not Path(path).is_file():
            raise FileNotFoundError(f"required competition asset missing: {path}")

    resolved_head, _, _ = room_scene.resolve_head_placement(args.head_placement)
    room_scene.build_stage(
        omni.kit.app.get_app(),
        room_path=room_path,
        robot_path=robot_path,
        task="task3",
        robot_position=room_scene.TASK_ROBOT_POSES["task3"]["position"],
        robot_rotation=room_scene.yaw_to_quat(room_scene.TASK_ROBOT_POSES["task3"]["yaw"]),
        robot_yaw=room_scene.TASK_ROBOT_POSES["task3"]["yaw"],
        head_placement=resolved_head,
        dynamic_beans=True,
    )
    stage = omni.usd.get_context().get_stage()
    _fix_single_articulation_root(stage, ROBOT_PRIM_PATH, UsdPhysics)
    gripper_pad_links = _configure_gripper_pad_material(
        stage, ROBOT_PRIM_PATH, Usd, UsdPhysics
    )
    spoon_contact_bodies = _configure_spoon_contact_reporting(
        stage, Usd, UsdPhysics, PhysxSchema
    )
    _configure_base_drives(stage, ROBOT_PRIM_PATH, UsdPhysics)
    articulation_root = _find_articulation_root(stage, ROBOT_PRIM_PATH, UsdPhysics)
    world = World(
        physics_prim_path=_find_physics_scene(stage),
        stage_units_in_meters=1.0,
        physics_dt=1.0 / args.physics_hz,
        rendering_dt=1.0 / args.render_hz,
        sim_params={"use_fabric": not args.pose_scan},
    )
    robot = SingleArticulation(prim_path=articulation_root, name="task3_robot")
    world.scene.add(robot)
    world.reset()
    joint_indices, ready_positions = _apply_ready_pose(robot, ArticulationAction)
    for _ in range(30):
        world.step(render=True)
    if args.pose_scan:
        _scan_compact_pose(world, robot, stage, Usd, UsdGeom)
        simulation_app.close()
        return 0

    state_reader = IsaacStateReader(stage)
    live_physics_details = state_reader.bind_live_physics(RigidPrim)
    assignment_targets = targets_from_head(state_reader.pose("head").position)
    live_scorer = OfficialLiveScorer(state_reader, assignment_targets)
    store.update(
        assigned_seats={name: f"{target.xy[0]:.3f},{target.xy[1]:.3f}" for name, target in assignment_targets.items()},
        object_state=state_reader.snapshot(),
    )

    annotator, render_product = _create_overview_capture(rep, stage)
    detail_annotator, detail_render_product = _create_supply_capture(rep, stage, UsdGeom, Gf)
    head_annotator, head_render_product = _create_head_capture(
        rep,
        stage,
        UsdGeom,
        Gf,
        state_reader.pose("head").position,
    )
    for _ in range(20):
        world.step(render=True)
    frame = _read_annotator_frame(annotator)
    for _ in range(120):
        if frame is not None:
            break
        world.step(render=True)
        frame = _read_annotator_frame(annotator)
    camera_capture_degraded = frame is None
    if frame is None:
        # Keep physics and policy control available when the optional review
        # camera is temporarily unavailable.  The main loop continues probing
        # and replaces this placeholder as soon as RTX capture recovers.
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
    store.set_frame("overview", frame)
    video_path = run_dir / "evidence.mp4"
    video_writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, float(args.render_hz) / 4.0),
        (960, 540),
    )
    if not video_writer.isOpened():
        video_writer.release()
        video_writer = None

    def record_overview(image: np.ndarray) -> None:
        canvas = image
        if canvas.shape[:2] != (540, 960):
            canvas = cv2.resize(canvas, (960, 540), interpolation=cv2.INTER_AREA)
        canvas = canvas.copy()
        state = store.state()
        safety = state["safety"]
        cv2.rectangle(canvas, (0, 0), (960, 62), (12, 12, 12), thickness=-1)
        cv2.putText(
            canvas,
            (
                f"{state['lifecycle']}  {state['stage']}  "
                f"score {state['score']:.2f}/16  sim {state['simulated_seconds']:.1f}s"
            ),
            (16, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            (
                f"{state['substate']}  RTF {state['real_time_factor']:.2f}  "
                f"head force {safety['current_head_force_n']:.1f}N  "
                f"watchdogs {safety['watchdog_interventions']}"
            ),
            (16, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (210, 230, 255),
            1,
            cv2.LINE_AA,
        )
        store.set_frame("overview", canvas)
        if video_writer is not None:
            video_writer.write(canvas)

    record_overview(frame)
    detail_frame = _read_annotator_frame(detail_annotator)
    if detail_frame is not None:
        store.set_frame("left_wrist", detail_frame)
    head_frame = _read_annotator_frame(head_annotator)
    if head_frame is not None:
        store.set_frame("head", head_frame)
    store.update(scene_ready=True, message="Scene rendered; running actuator calibration")
    # Start HTTP only after Kit has created its main-thread asyncio integration.
    # Starting uvicorn before SimulationApp can replace the process-wide event
    # loop policy while Kit is bootstrapping and corrupt omni.kit.async_engine.
    start_dashboard(store, args.ui_port)
    robot_bounds = _robot_world_bounds(stage, Usd, UsdGeom)
    store.event(
        "scene_ready",
        benchmark_commit=BENCHMARK_COMMIT,
        policy_source_sha256=policy_source_hash(),
        articulation_root=articulation_root,
        room_usd=str(room_path),
        robot_usd=str(robot_path),
        dof_names=list(robot.dof_names),
        frame_mean=float(frame.mean()),
        frame_min=int(frame.min()),
        frame_max=int(frame.max()),
        robot_bounds=robot_bounds,
        live_physics=live_physics_details,
        gripper_pad_material_links=gripper_pad_links,
        spoon_contact_report_bodies=spoon_contact_bodies,
        camera_capture_degraded=camera_capture_degraded,
        evidence_video=str(video_path),
        evidence_video_enabled=video_writer is not None,
    )
    store.event(
        "head_geometry_inventory",
        root=state_reader.resolve("head"),
        samples=state_reader.descendant_geometry_samples("head"),
    )
    _calibration_motion(
        world,
        robot,
        ArticulationAction,
        joint_indices,
        ready_positions,
        store,
        annotator,
        args.calibration_steps,
    )
    store.update(lifecycle=Lifecycle.READY, stage=Stage.TABLE_SETUP, substate=Substate.IDLE)
    if not args.preview_only and not args.wait_for_start:
        store.apply_command("start")
    root_position, root_orientation = robot.get_world_pose()
    store.update(
        robot_position=tuple(float(value) for value in root_position),
        robot_orientation_wxyz=tuple(float(value) for value in root_orientation),
    )

    def refresh_overview() -> None:
        current = _read_annotator_frame(annotator)
        if current is not None:
            record_overview(current)
        detail = _read_annotator_frame(detail_annotator)
        if detail is not None:
            store.set_frame("left_wrist", detail)
        head_detail = _read_annotator_frame(head_annotator)
        if head_detail is not None:
            store.set_frame("head", head_detail)

    policy_runner: HierarchicalPolicyRunner | None = None

    stopping = False

    def stop_handler(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True
        store.request_stop()
        # Wake an actuator paused inside its physics-step loop so it can
        # observe stop_requested and return without completing the primitive.
        store.queue_command("resume")

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    wall_started = time.monotonic()
    while (
        simulation_app.is_running()
        and not stopping
        and not store.stop_requested
    ):
        for command in store.drain_commands():
            store.apply_command(command)
        if store.telemetry.lifecycle is Lifecycle.PAUSED:
            simulation_app.update()
            time.sleep(0.02)
            continue
        if store.telemetry.lifecycle in {Lifecycle.RUNNING, Lifecycle.RECOVERY} and not args.preview_only:
            if policy_runner is None:
                try:
                    actuator = IsaacPhysicalActuator(
                        world=world,
                        robot=robot,
                        action_type=ArticulationAction,
                        benchmark_root=root,
                        reader=state_reader,
                        scorer=live_scorer,
                        assignments=assignment_targets,
                        store=store,
                        render_callback=refresh_overview,
                        physics_dt=1.0 / args.physics_hz,
                        control_dt=1.0 / args.render_hz,
                    )
                    policy_runner = HierarchicalPolicyRunner(actuator)
                    if args.development_recovery:
                        table_plan = policy_runner.stage_plans[Stage.TABLE_SETUP]
                        start = next(
                            index
                            for index, item in enumerate(table_plan)
                            if item.label == "navigate to bowl"
                        )
                        end = next(
                            (
                                index
                                for index in range(start + 1, len(table_plan))
                                if table_plan[index].label.startswith("navigate to ")
                            ),
                            len(table_plan),
                        )
                        policy_runner.stage_plans = {
                            Stage.TABLE_SETUP: table_plan[start:end],
                            Stage.BEAN_RECOVERY: policy_runner.stage_plans[
                                Stage.BEAN_RECOVERY
                            ],
                        }
                        store.event(
                            "development_recovery_plan_override",
                            table_objects=["bowl"],
                            primitive_count=sum(
                                len(plan) for plan in policy_runner.stage_plans.values()
                            ),
                        )
                    elif args.development_feeding:
                        table_plan = policy_runner.stage_plans[Stage.TABLE_SETUP]
                        feeding_table_plan = []
                        for object_name in ("bowl", "cup", "spoon"):
                            start = next(
                                index
                                for index, item in enumerate(table_plan)
                                if item.label == f"navigate to {object_name}"
                            )
                            end = next(
                                (
                                    index
                                    for index in range(start + 1, len(table_plan))
                                    if table_plan[index].label.startswith("navigate to ")
                                ),
                                len(table_plan),
                            )
                            feeding_table_plan.extend(table_plan[start:end])
                        policy_runner.stage_plans = {
                            Stage.TABLE_SETUP: feeding_table_plan,
                            Stage.FEEDING: policy_runner.stage_plans[Stage.FEEDING],
                        }
                        store.event(
                            "development_feeding_plan_override",
                            table_objects=["bowl", "cup", "spoon"],
                            primitive_count=sum(
                                len(plan) for plan in policy_runner.stage_plans.values()
                            ),
                        )
                    elif args.development_table_object is not None:
                        table_plan = policy_runner.stage_plans[Stage.TABLE_SETUP]
                        start = next(
                            index
                            for index, item in enumerate(table_plan)
                            if item.label
                            == f"navigate to {args.development_table_object}"
                        )
                        end = next(
                            (
                                index
                                for index in range(start + 1, len(table_plan))
                                if table_plan[index].label.startswith("navigate to ")
                            ),
                            len(table_plan),
                        )
                        policy_runner.stage_plans = {
                            Stage.TABLE_SETUP: table_plan[start:end]
                        }
                        store.event(
                            "development_plan_override",
                            table_object=args.development_table_object,
                            primitive_count=end - start,
                        )
                    store.event(
                        "policy_initialized",
                        mode="deterministic",
                        development_table_object=args.development_table_object,
                        development_feeding=args.development_feeding,
                        development_recovery=args.development_recovery,
                    )
                except Exception as exc:
                    store.update(
                        lifecycle=Lifecycle.FAILED,
                        failure_reason=f"policy initialization: {exc}",
                        message=f"Policy initialization failed: {exc}",
                    )
                    store.event("policy_initialization_failed", error=repr(exc))
            if policy_runner is not None and store.telemetry.lifecycle in {Lifecycle.RUNNING, Lifecycle.RECOVERY}:
                success, message = policy_runner.tick()
                breakdown = live_scorer.update()
                store.update(
                    lifecycle=Lifecycle.RUNNING if success else store.telemetry.lifecycle,
                    stage=policy_runner.stage,
                    score=breakdown.total,
                    stage_scores=breakdown.as_dict(),
                    recovery_ratio=live_scorer.recovery_ratio,
                    highest_completed_stage=highest_completed_stage(breakdown),
                    message=message,
                )
                if message == "complete":
                    store.update(lifecycle=Lifecycle.COMPLETE)
                    store.event(
                        "episode_complete",
                        deferred_failures=policy_runner.deferred_failures,
                    )
                    # Finalize summary/video deterministically. A completed
                    # run must not sit forever in Kit with an unflushed MP4.
                    stopping = True
                elif not success and policy_runner.failed:
                    store.update(lifecycle=Lifecycle.FAILED, failure_reason=message)
                    store.event(
                        "episode_failed",
                        failure_reason=message,
                        deferred_failures=policy_runner.deferred_failures,
                    )
                    stopping = True
        world.step(render=True)
        if world.current_time_step_index % 4 == 0:
            frame = _read_annotator_frame(annotator)
            if frame is not None:
                record_overview(frame)
            detail_frame = _read_annotator_frame(detail_annotator)
            if detail_frame is not None:
                store.set_frame("left_wrist", detail_frame)
            head_frame = _read_annotator_frame(head_annotator)
            if head_frame is not None:
                store.set_frame("head", head_frame)
        if world.current_time_step_index % 12 == 0:
            breakdown = live_scorer.update()
            store.update(
                score=breakdown.total,
                stage_scores=breakdown.as_dict(),
                recovery_ratio=live_scorer.recovery_ratio,
            )
        wall_elapsed = time.monotonic() - wall_started
        store.update(
            simulated_seconds=world.current_time,
            wall_seconds=wall_elapsed,
            real_time_factor=world.current_time / max(wall_elapsed, 1e-6),
        )

    store.request_stop()
    store.write_summary()
    if video_writer is not None:
        video_writer.release()
    annotator.detach(render_product)
    detail_annotator.detach(detail_render_product)
    head_annotator.detach(head_render_product)
    # Kit's fast-shutdown path can terminate the interpreter with status 0
    # from inside SimulationApp.close(), before Python returns our intended
    # exit status.  Persist the reset intent first so the shell supervisor can
    # still distinguish a requested full reset from an ordinary clean exit.
    if store.reset_requested:
        reset_marker = args.run_root / ".reset-requested"
        reset_marker.write_text(f"seed={args.seed}\n", encoding="utf-8")
    simulation_app.close()
    return 75 if store.reset_requested else 0


def main() -> None:
    args = build_parser().parse_args()
    try:
        raise SystemExit(run(args))
    except Exception as exc:
        failure_root = args.run_root / "launcher-failures"
        failure_root.mkdir(parents=True, exist_ok=True)
        (failure_root / f"seed-{args.seed}-{int(time.time())}.json").write_text(
            json.dumps({"error": repr(exc), "seed": args.seed}, indent=2) + "\n",
            encoding="utf-8",
        )
        raise
