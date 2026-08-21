#!/usr/bin/env python3
"""Print collision bounds near a room-space point without launching Kit."""

from __future__ import annotations

import argparse
import json
import math

from pxr import Usd, UsdGeom, UsdPhysics


def _distance_to_rect(x: float, y: float, bounds: tuple[float, float, float, float]) -> float:
    min_x, min_y, max_x, max_y = bounds
    dx = max(min_x - x, 0.0, x - max_x)
    dy = max(min_y - y, 0.0, y - max_y)
    return math.hypot(dx, dy)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("room_usd")
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--radius", type=float, default=1.5)
    args = parser.parse_args()

    stage = Usd.Stage.Open(args.room_usd)
    if stage is None:
        raise SystemExit(f"could not open {args.room_usd}")
    purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes)
    rows: list[tuple[float, dict[str, object]]] = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        minimum, maximum = aligned.GetMin(), aligned.GetMax()
        values = [float(value) for value in (*minimum, *maximum)]
        if not all(math.isfinite(value) for value in values):
            continue
        min_x, min_y, min_z, max_x, max_y, max_z = values
        if max_z <= 0.08 or min_z >= 1.35:
            continue
        bounds = (min_x, min_y, max_x, max_y)
        distance = _distance_to_rect(args.x, args.y, bounds)
        if distance > args.radius:
            continue
        rows.append(
            (
                distance,
                {
                    "distance": round(distance, 4),
                    "path": str(prim.GetPath()),
                    "minimum": [round(min_x, 4), round(min_y, 4), round(min_z, 4)],
                    "maximum": [round(max_x, 4), round(max_y, 4), round(max_z, 4)],
                },
            )
        )
    for _, row in sorted(rows, key=lambda item: (item[0], str(item[1]["path"]))):
        print(json.dumps(row, separators=(",", ":")))


if __name__ == "__main__":
    main()
