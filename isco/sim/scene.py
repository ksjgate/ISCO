"""Scene construction: ground, lights, NVIDIA conveyor, part spawner, ISCO<->stage frame.

Frames
------
The Isaac stage is Z-up, metres, and the official ``ConveyorBelt_A06`` asset runs along
stage +X (measured in the PoC). The ISCO world frame (``isco.geometry.transforms``)
therefore differs from the stage frame **only by a translation**::

    p_stage = p_world + origin_stage
    origin_stage = (focus_x, belt_centre_y, belt_top_z)

with ``focus_x = conveyor_aabb.xmin + 0.4 * conveyor_length`` (the PoC's capture point,
40 % along the belt), ``belt_centre_y`` the centre of the ``Belt`` prim's AABB and
``belt_top_z`` the top of the ``Belt`` prim's AABB (the plane z = 0 in ISCO world).
Rotations and directions (camera axes, normals) are identical in both frames.

All Isaac/pxr imports are lazy (``SimulationApp`` must be running).
"""
from __future__ import annotations

import glob
import os
import random
from dataclasses import dataclass, field

import numpy as np

#: fixed colours, one per spawned part (PoC)
PART_COLORS = [
    (0.75, 0.55, 0.15),  # brass
    (0.55, 0.15, 0.15),  # brick
    (0.20, 0.45, 0.65),  # teal
    (0.55, 0.35, 0.60),  # purple
    (0.35, 0.60, 0.30),  # green
    (0.70, 0.70, 0.70),  # steel grey
]

FOCUS_X_FRACTION = 0.4
FOCUS_X_RULE = "focus_x = conveyor_aabb.xmin + 0.4 * (conveyor_aabb.xmax - conveyor_aabb.xmin)"
DEFAULT_CONVEYOR_ASSET = "ConveyorBelt_A06"


# ----------------------------------------------------------------------------- frame
@dataclass
class IscoStageFrame:
    """Translation between the ISCO world frame and the Isaac stage frame."""

    origin_stage: np.ndarray = field(default_factory=lambda: np.zeros(3))
    focus_x_rule: str = FOCUS_X_RULE

    def __post_init__(self) -> None:
        self.origin_stage = np.asarray(self.origin_stage, dtype=float).reshape(3)

    @property
    def T_stage_world(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, 3] = self.origin_stage
        return T

    def to_stage_T(self, T_world: np.ndarray) -> np.ndarray:
        T = np.array(T_world, dtype=float, copy=True)
        T[:3, 3] = T[:3, 3] + self.origin_stage
        return T

    def to_world_T(self, T_stage: np.ndarray) -> np.ndarray:
        T = np.array(T_stage, dtype=float, copy=True)
        T[:3, 3] = T[:3, 3] - self.origin_stage
        return T

    def to_stage_pts(self, p_world: np.ndarray) -> np.ndarray:
        return np.asarray(p_world, dtype=float) + self.origin_stage

    def to_world_pts(self, p_stage: np.ndarray) -> np.ndarray:
        return np.asarray(p_stage, dtype=float) - self.origin_stage

    def to_dict(self) -> dict:
        return {"origin_stage": self.origin_stage.tolist(), "focus_x_rule": self.focus_x_rule}

    @classmethod
    def from_dict(cls, d: dict) -> "IscoStageFrame":
        return cls(np.asarray(d["origin_stage"], dtype=float), d.get("focus_x_rule", FOCUS_X_RULE))

    @classmethod
    def from_conveyor_info(cls, info: dict, focus_fraction: float = FOCUS_X_FRACTION) -> "IscoStageFrame":
        whole, belt = info["whole_aabb"], info["belt_aabb"]
        focus_x = float(whole[0] + (whole[3] - whole[0]) * focus_fraction)
        y = float(0.5 * (belt[1] + belt[4]))
        return cls(np.array([focus_x, y, float(info["belt_top_z"])]),
                   FOCUS_X_RULE.replace("0.4", f"{focus_fraction:g}"))


# ----------------------------------------------------------------------------- USD helpers
def apply_physics_material(stage, prim_path, static_friction=0.7, dynamic_friction=0.6, restitution=0.0):
    """Create a physics material with the given friction and bind it to ``prim_path``.

    Surface velocity pushes objects through contact friction, so (near) zero friction
    would make parts slide on a running belt without moving (PoC)."""
    from pxr import UsdPhysics, UsdShade

    mat_path = f"{prim_path}/PhysicsMaterial"
    material = UsdShade.Material.Define(stage, mat_path)
    physics_mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_mat.CreateStaticFrictionAttr().Set(static_friction)
    physics_mat.CreateDynamicFrictionAttr().Set(dynamic_friction)
    physics_mat.CreateRestitutionAttr().Set(restitution)
    prim = stage.GetPrimAtPath(prim_path)
    binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
    binding_api.Bind(material, bindingStrength=UsdShade.Tokens.weakerThanDescendants, materialPurpose="physics")
    return material


def add_official_conveyor(stage, prim_path, asset_name=DEFAULT_CONVEYOR_ASSET, speed=0.3):
    """Reference an official NVIDIA conveyor (``Isaac/Props/Conveyors/<asset>.usd``,
    streamed from the asset CDN) and drive its ``Belt`` prim with PhysX surface velocity.

    The asset's ``Belt`` prim already is a kinematic rigid body with collision; only the
    surface velocity (local +X = belt travel) and a friction material are added.
    Returns ``(conveyor_path, belt_path)``."""
    from isaacsim.storage.native import nucleus
    from pxr import Gf, PhysxSchema

    root = nucleus.get_assets_root_path()
    if not root:
        raise RuntimeError("cannot reach the NVIDIA asset server (Nucleus/CDN) - check the internet connection")
    asset_path = f"{root}/Isaac/Props/Conveyors/{asset_name}.usd"
    conveyor_xform = stage.DefinePrim(prim_path, "Xform")
    if not conveyor_xform.GetReferences().AddReference(asset_path):
        raise RuntimeError(f"failed to reference asset: {asset_path}")
    belt_path = f"{prim_path}/Belt"
    belt_prim = stage.GetPrimAtPath(belt_path)
    if not belt_prim.IsValid():
        raise RuntimeError(f"{asset_name} has no 'Belt' prim - only BELT/DUAL style conveyors are supported "
                           "(see --list-assets)")
    surface_vel = PhysxSchema.PhysxSurfaceVelocityAPI.Apply(belt_prim)
    surface_vel.CreateSurfaceVelocityEnabledAttr().Set(True)
    surface_vel.CreateSurfaceVelocityLocalSpaceAttr().Set(True)
    surface_vel.CreateSurfaceVelocityAttr().Set(Gf.Vec3f(float(speed), 0.0, 0.0))
    apply_physics_material(stage, belt_path, static_friction=0.8, dynamic_friction=0.7)
    return prim_path, belt_path


def set_belt_speed(stage, belt_path, speed: float) -> None:
    """Change the belt surface velocity (m/s along local +X) at run time."""
    from pxr import Gf, PhysxSchema

    api = PhysxSchema.PhysxSurfaceVelocityAPI(stage.GetPrimAtPath(belt_path))
    api.GetSurfaceVelocityAttr().Set(Gf.Vec3f(float(speed), 0.0, 0.0))


def get_belt_speed(stage, belt_path) -> float:
    from pxr import PhysxSchema

    v = PhysxSchema.PhysxSurfaceVelocityAPI(stage.GetPrimAtPath(belt_path)).GetSurfaceVelocityAttr().Get()
    return float(v[0]) if v is not None else 0.0


def measure_conveyor(prim_path, belt_path) -> dict:
    """Overall AABB and belt-top height of the loaded conveyor (stage frame)."""
    from isaacsim.core.utils.bounds import compute_aabb, create_bbox_cache

    cache = create_bbox_cache()
    whole_aabb = np.asarray(compute_aabb(cache, prim_path), dtype=float)
    belt_aabb = np.asarray(compute_aabb(cache, belt_path), dtype=float)
    return {
        "whole_aabb": whole_aabb,
        "belt_aabb": belt_aabb,
        "belt_top_z": float(belt_aabb[5]),
        "center": np.array([(whole_aabb[0] + whole_aabb[3]) / 2.0, (whole_aabb[1] + whole_aabb[4]) / 2.0,
                            (whole_aabb[2] + whole_aabb[5]) / 2.0]),
        "size": np.array([whole_aabb[3] - whole_aabb[0], whole_aabb[4] - whole_aabb[1],
                          whole_aabb[5] - whole_aabb[2]]),
    }


def add_lights(stage):
    """Dome light + key distant light (render/video quality only; depth is unaffected)."""
    from pxr import Gf, UsdGeom, UsdLux

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
    dome.CreateIntensityAttr(900.0)
    key_light = UsdLux.DistantLight.Define(stage, "/World/Lights/KeyLight")
    key_light.CreateIntensityAttr(3000.0)
    key_light.CreateAngleAttr(1.0)
    UsdGeom.Xformable(key_light).AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 30.0, 0.0))


def find_part_stl_files(parts_dir) -> list[str]:
    files = sorted(glob.glob(os.path.join(str(parts_dir), "*.stl")))
    if not files:
        raise RuntimeError(f"no part STL in {parts_dir} - the Fusion 360 Gallery STLs live in "
                           "data/parts/raw/fusion360/ (not redistributable, ADR-0004)")
    return files


def _euler_z_to_quat_wxyz(yaw_degrees):
    half = np.radians(yaw_degrees) / 2.0
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


class PartSpawner:
    """Drops a random part at the belt entry every ``spawn_interval`` and deletes parts
    that passed the belt end (PoC behaviour, used by the demo/video loop).

    Poses are read with plain USD queries instead of the tensor API: tensor physics views
    are invalidated whenever the number of rigid bodies changes (every spawn/despawn).
    """

    def __init__(self, stage, part_files, entry_pos, exit_x, part_size, spawn_interval, physics_fps, rng=None):
        self.stage = stage
        self.part_files = list(part_files)
        self.entry_pos = np.asarray(entry_pos, dtype=float)
        self.exit_x = exit_x
        self.part_size = part_size
        self.spawn_every_steps = max(1, int(round(spawn_interval * physics_fps)))
        self.step_count = 0
        self.spawn_count = 0
        self.active: list[str] = []
        self.rng = rng or random.Random()

    def _spawn_one(self):
        from pxr import Gf, UsdGeom

        from .part_prim import load_part_mesh_stl

        stl_path = self.rng.choice(self.part_files)
        color = PART_COLORS[self.spawn_count % len(PART_COLORS)]
        prim_path = f"/World/Parts/Part_{self.spawn_count:04d}"
        self.spawn_count += 1
        jitter_y = self.rng.uniform(-0.6, 0.6) * 0.15
        pos = self.entry_pos + np.array([0.0, jitter_y, 0.0])
        load_part_mesh_stl(self.stage, prim_path, stl_path, self.part_size, color)
        xformable = UsdGeom.Xformable(self.stage.GetPrimAtPath(prim_path))
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(*pos.tolist()))
        w, x, y, z = _euler_z_to_quat_wxyz(self.rng.uniform(0, 360))
        xformable.AddOrientOp().Set(Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z))))
        self.active.append(prim_path)

    def step(self):
        """Call once per physics step."""
        from pxr import Usd, UsdGeom

        self.step_count += 1
        if self.step_count % self.spawn_every_steps == 0:
            self._spawn_one()
        still_active = []
        for prim_path in self.active:
            prim = self.stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
            t = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
            if t[0] > self.exit_x or t[2] < -0.5:
                self.stage.RemovePrim(prim_path)
            else:
                still_active.append(prim_path)
        self.active = still_active


# ----------------------------------------------------------------------------- scene
class IscoScene:
    """World + ground + lights + official conveyor, measured once.

    ``IscoScene.build()`` creates (or adopts) an ``isaacsim.core.api.World``; the
    resulting ``frame`` maps ISCO world <-> stage coordinates.
    """

    def __init__(self, world, conveyor_path: str, belt_path: str, info: dict, frame: IscoStageFrame,
                 asset_name: str, speed: float):
        self.world = world
        self.stage = world.stage
        self.conveyor_path = conveyor_path
        self.belt_path = belt_path
        self.info = info
        self.frame = frame
        self.asset_name = asset_name
        self.speed = float(speed)

    @classmethod
    def build(cls, conveyor_asset: str = DEFAULT_CONVEYOR_ASSET, speed: float = 0.0, ground: bool = True,
              lights: bool = True, world=None, physics_dt: float | None = None, rendering_dt: float | None = None,
              warmup_steps: int = 5) -> "IscoScene":
        from isaacsim.core.api import World

        if world is None:
            kw = {"stage_units_in_meters": 1.0}
            if physics_dt is not None:
                kw["physics_dt"] = physics_dt
            if rendering_dt is not None:
                kw["rendering_dt"] = rendering_dt
            world = World(**kw)
        stage = world.stage
        if ground:
            from isaacsim.core.api.objects.ground_plane import GroundPlane

            GroundPlane(prim_path="/World/GroundPlane", z_position=0.0, size=10.0)
        if lights:
            add_lights(stage)
        conveyor_path, belt_path = add_official_conveyor(stage, "/World/Conveyor", conveyor_asset, speed)
        world.reset()
        for _ in range(warmup_steps):  # let the asset reference compose fully
            world.step(render=False)
        info = measure_conveyor(conveyor_path, belt_path)
        frame = IscoStageFrame.from_conveyor_info(info)
        return cls(world, conveyor_path, belt_path, info, frame, conveyor_asset, speed)

    # -- helpers -------------------------------------------------------------------
    @property
    def physics_fps(self) -> int:
        return int(round(1.0 / self.world.get_physics_dt()))

    def set_belt_speed(self, speed: float) -> None:
        set_belt_speed(self.stage, self.belt_path, speed)
        self.speed = float(speed)

    def belt_speed(self) -> float:
        return get_belt_speed(self.stage, self.belt_path)

    def conveyor_aabb_world(self) -> np.ndarray:
        w = np.asarray(self.info["whole_aabb"], dtype=float)
        o = self.frame.origin_stage
        return np.concatenate([w[:3] - o, w[3:] - o])

    def meta(self) -> dict:
        info = self.info
        belt = np.asarray(info["belt_aabb"], dtype=float)
        return {
            "asset_name": self.asset_name,
            "conveyor_prim": self.conveyor_path,
            "belt_prim": self.belt_path,
            "belt_top_z_stage": float(info["belt_top_z"]),
            "belt_width_m": float(belt[4] - belt[1]),
            "belt_length_m": float(belt[3] - belt[0]),
            "belt_aabb_stage": belt.tolist(),
            "conveyor_aabb_stage": np.asarray(info["whole_aabb"], dtype=float).tolist(),
            "conveyor_aabb_world": self.conveyor_aabb_world().tolist(),
            "belt_aabb_world": (np.concatenate([belt[:3], belt[3:]]) - np.tile(self.frame.origin_stage, 2)).tolist(),
            "isco_origin_stage": self.frame.origin_stage.tolist(),
            "focus_x_rule": self.frame.focus_x_rule,
            "belt_travel_direction": "+X (stage and ISCO world)",
            "frame_mapping": "p_stage = p_world + isco_origin_stage (translation only)",
            "physics_dt": float(self.world.get_physics_dt()),
        }
