#!/usr/bin/env python3
"""Print authored collision configuration for selected USD prims.

This tool intentionally opens a stage without starting Kit or PhysX.  It is
safe to run while an Isaac Sim episode is active and helps distinguish visual
geometry from collision geometry in competition assets.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_simulation_app = None
try:
    from pxr import Usd, UsdGeom, UsdPhysics
except ModuleNotFoundError:
    # The pip distribution of Isaac Sim exposes USD only after Kit has been
    # initialized.  Boot a minimal headless app so this remains usable as a
    # standalone, read-only asset diagnostic on the managed runtime.
    from isaacsim import SimulationApp

    _simulation_app = SimulationApp({"headless": True})
    from pxr import Usd, UsdGeom, UsdPhysics


INTERESTING_ATTRIBUTE_PARTS = (
    "approximation",
    "collision",
    "contactoffset",
    "kinematic",
    "restoffset",
    "rigidbody",
    "simulationowner",
)


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def describe_prim(prim: Usd.Prim) -> dict[str, object]:
    attributes: dict[str, object] = {}
    for attribute in prim.GetAttributes():
        compact_name = attribute.GetName().lower().replace(":", "")
        if any(part in compact_name for part in INTERESTING_ATTRIBUTE_PARTS):
            attributes[attribute.GetName()] = _json_value(attribute.Get())

    relationships: dict[str, object] = {}
    is_collision_group = "collisiongroup" in str(prim.GetTypeName()).lower()
    for relationship in prim.GetRelationships():
        compact_name = relationship.GetName().lower().replace(":", "")
        if is_collision_group or "filter" in compact_name or "collision" in compact_name:
            relationships[relationship.GetName()] = [
                str(target) for target in relationship.GetTargets()
            ]

    return {
        "path": str(prim.GetPath()),
        "type": prim.GetTypeName(),
        "applied_schemas": list(prim.GetAppliedSchemas()),
        "has_collision_api": prim.HasAPI(UsdPhysics.CollisionAPI),
        "has_mesh_collision_api": prim.HasAPI(UsdPhysics.MeshCollisionAPI),
        "has_rigid_body_api": prim.HasAPI(UsdPhysics.RigidBodyAPI),
        "attributes": attributes,
        "relationships": relationships,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=Path)
    parser.add_argument(
        "--match",
        default=r"(finger|plate)",
        help="case-insensitive regular expression matched against prim paths",
    )
    parser.add_argument(
        "--all-matches",
        action="store_true",
        help="include matching prims even when they have no physics API",
    )
    parser.add_argument(
        "--collision-groups",
        action="store_true",
        help="also include collision-group prims and prims with filter relationships",
    )
    parser.add_argument(
        "--reference-at",
        help="compose the asset into a new stage at this prim path before auditing",
    )
    args = parser.parse_args()

    if args.reference_at:
        stage = Usd.Stage.CreateInMemory()
        reference_prim = UsdGeom.Xform.Define(stage, args.reference_at).GetPrim()
        reference_prim.GetReferences().AddReference(str(args.stage.resolve()))
        stage.Load()
    else:
        stage = Usd.Stage.Open(str(args.stage))
    if stage is None:
        raise RuntimeError(f"failed to open USD stage: {args.stage}")
    matcher = re.compile(args.match, re.IGNORECASE)
    # The official Robotiq asset instances its detailed visual/collision
    # geometry.  Normal Stage.Traverse() intentionally skips instance proxies,
    # which would make the audit incorrectly report that finger meshes are
    # absent.
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        relationships = prim.GetRelationships()
        is_collision_group = "collisiongroup" in str(prim.GetTypeName()).lower()
        has_filter_relationship = any(
            "filter" in relationship.GetName().lower()
            for relationship in relationships
        )
        selected_by_group_mode = args.collision_groups and (
            is_collision_group or has_filter_relationship
        )
        if not matcher.search(str(prim.GetPath())) and not selected_by_group_mode:
            continue
        has_physics = any(
            (
                prim.HasAPI(UsdPhysics.CollisionAPI),
                prim.HasAPI(UsdPhysics.MeshCollisionAPI),
                prim.HasAPI(UsdPhysics.RigidBodyAPI),
                bool(relationships),
            )
        )
        if args.all_matches or has_physics:
            print(json.dumps(describe_prim(prim), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    finally:
        if _simulation_app is not None:
            _simulation_app.close()
    raise SystemExit(exit_code)
