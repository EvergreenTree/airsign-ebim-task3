from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .types import Bounds, Pose


PRIM_ALIASES = {
    "plate": "plate2_01",
    "cup": "cup",
    "bowl": "bowl2",
    "spoon": "spoon2_01",
    "tray": "simple_tray",
    "recycling": "ikea_knock_box",
    "sink": "sink_boundary",
    "head": "head",
}


@dataclass(frozen=True)
class PrimSample:
    path: str
    pose: Pose
    bounds: Bounds


class IsaacStateReader:
    """Simulator ground truth reader with no mutation methods."""

    def __init__(self, stage: Any) -> None:
        from pxr import Usd, UsdGeom, UsdPhysics

        self.stage = stage
        self.Usd = Usd
        self.UsdGeom = UsdGeom
        self.UsdPhysics = UsdPhysics
        self._paths: dict[str, str] = {}
        self._rigid_paths: dict[str, str] = {}
        self._rigid_views: dict[str, Any] = {}
        self._rigid_local_bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._bean_view: Any | None = None

    def bind_live_physics(self, rigid_prim_type: Any) -> dict[str, object]:
        """Bind read-only PhysX tensor views for dynamic task bodies and beans."""

        import numpy as np
        from pxr import UsdPhysics

        details: dict[str, object] = {}
        purposes = [
            self.UsdGeom.Tokens.default_,
            self.UsdGeom.Tokens.render,
            self.UsdGeom.Tokens.proxy,
        ]
        for name in ("plate", "cup", "bowl", "spoon", "tray"):
            root = self.stage.GetPrimAtPath(self.resolve(name))
            rigid_paths = []
            for prim in self.Usd.PrimRange(root, self.Usd.TraverseInstanceProxies()):
                if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    continue
                rigid_api = UsdPhysics.RigidBodyAPI(prim)
                enabled = rigid_api.GetRigidBodyEnabledAttr().Get()
                if enabled is not False:
                    rigid_paths.append(str(prim.GetPath()))
            if len(rigid_paths) != 1:
                details[name] = {"candidates": rigid_paths, "bound": False}
                continue
            rigid_path = rigid_paths[0]
            view = rigid_prim_type(
                prim_paths_expr=rigid_path,
                name=f"airsign_{name}_live_view",
                reset_xform_properties=False,
                prepare_contact_sensors=False,
            )
            view.initialize()
            rigid_prim = self.stage.GetPrimAtPath(rigid_path)
            cache = self.UsdGeom.BBoxCache(self.Usd.TimeCode.Default(), purposes)
            local_range = cache.ComputeLocalBound(rigid_prim).ComputeAlignedRange()
            self._rigid_paths[name] = rigid_path
            self._rigid_views[name] = view
            self._rigid_local_bounds[name] = (
                np.asarray(local_range.GetMin(), dtype=float),
                np.asarray(local_range.GetMax(), dtype=float),
            )
            details[name] = {"path": rigid_path, "bound": True}

        bean_view = rigid_prim_type(
            prim_paths_expr="/World/Scene/CoffeeBeans/Bean_.*",
            name="airsign_beans_live_view",
            reset_xform_properties=False,
            prepare_contact_sensors=False,
        )
        bean_view.initialize()
        self._bean_view = bean_view
        details["beans"] = {"count": int(bean_view.count), "bound": True}
        return details

    @staticmethod
    def _rotation_matrix(quaternion_wxyz: tuple[float, float, float, float]) -> np.ndarray:
        import numpy as np

        w, x, y, z = quaternion_wxyz
        return np.asarray(
            (
                (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
                (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
                (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
            ),
            dtype=float,
        )

    def _live_pose(self, semantic_name: str) -> Pose | None:
        import numpy as np

        view = self._rigid_views.get(semantic_name)
        if view is None:
            return None
        positions, orientations = view.get_world_poses(clone=True, usd=False)
        position = np.asarray(positions, dtype=float)[0]
        orientation = np.asarray(orientations, dtype=float)[0]
        return Pose(
            position=tuple(float(value) for value in position),
            orientation_wxyz=tuple(float(value) for value in orientation),
        )

    def resolve(self, semantic_name: str) -> str:
        if semantic_name in self._paths:
            return self._paths[semantic_name]
        name = PRIM_ALIASES.get(semantic_name, semantic_name)
        candidates = (
            f"/World/Environment/RobotRoom/Asset/{name}",
            f"/World/Environment/RobotRoom/Asset/root/{name}",
            f"/root/{name}",
            f"/World/Scene/{name}",
        )
        for candidate in candidates:
            prim = self.stage.GetPrimAtPath(candidate)
            if prim and prim.IsValid():
                self._paths[semantic_name] = candidate
                return candidate
        suffix = f"/{name}"
        prim_range = self.Usd.PrimRange.Stage(
            self.stage, self.Usd.TraverseInstanceProxies()
        )
        matches = [
            str(prim.GetPath())
            for prim in prim_range
            if str(prim.GetPath()).endswith(suffix)
        ]
        if len(matches) != 1:
            raise RuntimeError(f"could not uniquely resolve {semantic_name!r}: {matches}")
        self._paths[semantic_name] = matches[0]
        return matches[0]

    def pose(self, semantic_name: str) -> Pose:
        live_pose = self._live_pose(semantic_name)
        if live_pose is not None:
            return live_pose
        prim = self.stage.GetPrimAtPath(self.resolve(semantic_name))
        matrix = self.UsdGeom.XformCache(self.Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
        translation = matrix.ExtractTranslation()
        rotation = matrix.ExtractRotationQuat()
        imaginary = rotation.GetImaginary()
        return Pose(
            position=(float(translation[0]), float(translation[1]), float(translation[2])),
            orientation_wxyz=(
                float(rotation.GetReal()),
                float(imaginary[0]),
                float(imaginary[1]),
                float(imaginary[2]),
            ),
        )

    def bounds(self, semantic_name: str) -> Bounds:
        live_pose = self._live_pose(semantic_name)
        if live_pose is not None:
            import numpy as np

            local_minimum, local_maximum = self._rigid_local_bounds[semantic_name]
            corners = np.asarray(
                [
                    (x, y, z)
                    for x in (local_minimum[0], local_maximum[0])
                    for y in (local_minimum[1], local_maximum[1])
                    for z in (local_minimum[2], local_maximum[2])
                ],
                dtype=float,
            )
            world_corners = (
                corners @ self._rotation_matrix(live_pose.orientation_wxyz).T
                + np.asarray(live_pose.position, dtype=float)
            )
            minimum = np.min(world_corners, axis=0)
            maximum = np.max(world_corners, axis=0)
            return Bounds(
                minimum=tuple(float(value) for value in minimum),
                maximum=tuple(float(value) for value in maximum),
            )
        prim = self.stage.GetPrimAtPath(self.resolve(semantic_name))
        purposes = [
            self.UsdGeom.Tokens.default_,
            self.UsdGeom.Tokens.render,
            self.UsdGeom.Tokens.proxy,
        ]
        cache = self.UsdGeom.BBoxCache(self.Usd.TimeCode.Default(), purposes)
        aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        minimum, maximum = aligned.GetMin(), aligned.GetMax()
        return Bounds(
            minimum=(float(minimum[0]), float(minimum[1]), float(minimum[2])),
            maximum=(float(maximum[0]), float(maximum[1]), float(maximum[2])),
        )

    def sample(self, semantic_name: str) -> PrimSample:
        return PrimSample(
            path=self._rigid_paths.get(semantic_name, self.resolve(semantic_name)),
            pose=self.pose(semantic_name),
            bounds=self.bounds(semantic_name),
        )

    def bean_positions(self) -> tuple[tuple[float, float, float], ...]:
        if self._bean_view is not None:
            import numpy as np

            positions, _ = self._bean_view.get_world_poses(clone=True, usd=False)
            return tuple(
                tuple(float(value) for value in position)
                for position in np.asarray(positions, dtype=float)
            )
        root = self.stage.GetPrimAtPath("/World/Scene/CoffeeBeans")
        if not root or not root.IsValid():
            return ()
        cache = self.UsdGeom.XformCache(self.Usd.TimeCode.Default())
        positions = []
        for prim in root.GetChildren():
            translation = cache.GetLocalToWorldTransform(prim).ExtractTranslation()
            positions.append((float(translation[0]), float(translation[1]), float(translation[2])))
        return tuple(positions)

    def descendant_geometry_samples(
        self,
        semantic_name: str,
        *,
        max_extent_m: float = 2.0,
        max_samples: int = 128,
    ) -> list[dict[str, Any]]:
        """Return read-only world bounds for plausible descendant geometry.

        Referenced assets can give a useful semantic root an unusably broad
        aggregate bound.  Enumerating boundable instance proxies lets policy
        calibration identify the actual face/head surface without changing the
        stage or relying on an authored child name.
        """

        root = self.stage.GetPrimAtPath(self.resolve(semantic_name))
        purposes = [
            self.UsdGeom.Tokens.default_,
            self.UsdGeom.Tokens.render,
            self.UsdGeom.Tokens.proxy,
        ]
        bounds_cache = self.UsdGeom.BBoxCache(self.Usd.TimeCode.Default(), purposes)
        xform_cache = self.UsdGeom.XformCache(self.Usd.TimeCode.Default())
        samples: list[dict[str, Any]] = []
        for prim in self.Usd.PrimRange(root, self.Usd.TraverseInstanceProxies()):
            if not prim.IsA(self.UsdGeom.Boundable):
                continue
            aligned = bounds_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            minimum = tuple(float(value) for value in aligned.GetMin())
            maximum = tuple(float(value) for value in aligned.GetMax())
            values = (*minimum, *maximum)
            if not all(math.isfinite(value) for value in values):
                continue
            extent = tuple(maximum[index] - minimum[index] for index in range(3))
            if max(extent) <= 0.0 or max(extent) > float(max_extent_m):
                continue
            translation = xform_cache.GetLocalToWorldTransform(prim).ExtractTranslation()
            samples.append(
                {
                    "path": str(prim.GetPath()),
                    "type": prim.GetTypeName(),
                    "position": tuple(float(value) for value in translation),
                    "bounds": {"minimum": minimum, "maximum": maximum},
                    "extent": extent,
                    "collision_api": prim.HasAPI(self.UsdPhysics.CollisionAPI),
                    "rigid_body_api": prim.HasAPI(self.UsdPhysics.RigidBodyAPI),
                }
            )
            if len(samples) >= int(max_samples):
                break
        return samples

    def snapshot(self) -> dict[str, Any]:
        names = ("plate", "cup", "bowl", "spoon", "tray", "recycling", "sink", "head")
        return {
            name: {
                "path": self._rigid_paths.get(name, self.resolve(name)),
                "position": self.pose(name).position,
                "bounds": {
                    "minimum": self.bounds(name).minimum,
                    "maximum": self.bounds(name).maximum,
                },
            }
            for name in names
        }
