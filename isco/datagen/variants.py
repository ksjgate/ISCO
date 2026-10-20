"""Variant sets: the fixed population of part instances every configuration is
evaluated on (paired comparisons across configs, plan "샘플 수").

Roles
-----
bank         normal instances used to build the memory bank
val          normal instances used to set the 1 %-FPR threshold
search       defective instances the optimiser sees
report       held-out defective instances used for every reported number (D3)
unreachable  control defects on surfaces no upper-hemisphere camera can see (k = 0);
             reported as the visibility ceiling, excluded from the main FNR

Layout: ``data/variants/<part_id>/<set_id>/{manifest.json, meshes/<inst_id>.npz}``.
"""
from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from ..geometry.mesh_io import LabeledMesh
from ..utils.hashing import content_hash
from ..utils.io import data_rel, read_json, write_json
from ..utils.logging import get_logger
from ..utils.seeds import rng_for
from .defects import DEFECT_TYPES, DefectDictionary, DefectGenerator, defect_area
from .normal_variation import NormalVariationParams, PoseParams, apply_normal_variation, load_settle, sample_pose
from .parts import data_root, load_part, load_part_mesh

log = get_logger(__name__)


class VariantSetSpec(BaseModel):
    set_id: str = "mvp"
    n_bank: int = 40
    n_val: int = 100
    per_type_search: int = 6
    per_type_report: int = 6
    n_unreachable: int = 6
    types: list[str] = list(DEFECT_TYPES)
    unreachable_types: list[str] = ["dent", "hole", "crack", "bulge", "cut"]
    seed: int = 0
    normal: NormalVariationParams = NormalVariationParams()
    pose: PoseParams = PoseParams()
    dictionary: DefectDictionary = DefectDictionary()

    @classmethod
    def preset(cls, name: str, **kw) -> "VariantSetSpec":
        if name == "mvp":
            return cls(set_id="mvp", **kw)
        if name == "full":
            return cls(set_id="full", n_bank=60, n_val=200, per_type_search=12, per_type_report=12, n_unreachable=12, **kw)
        if name == "smoke":
            return cls(set_id="smoke", n_bank=8, n_val=12, per_type_search=1, per_type_report=1, n_unreachable=1, **kw)
        raise ValueError(name)


def set_dir(part_id: str, set_id: str) -> Path:
    return data_root() / "variants" / part_id / set_id


def load_manifest(part_id: str, set_id: str) -> dict:
    return read_json(set_dir(part_id, set_id) / "manifest.json")


def _largest_remainder(n: int, mix: dict[str, float]) -> list[str]:
    keys = list(mix)
    w = np.array([mix[k] for k in keys], float)
    w = w / w.sum() * n
    base = np.floor(w).astype(int)
    rem = n - base.sum()
    order = np.argsort(-(w - base), kind="stable")
    base[order[:rem]] += 1
    return [k for k, c in zip(keys, base) for _ in range(c)]


def placeable_unreachable(gen, spec: VariantSetSpec) -> list[str]:
    """Unreachable-control types this part can host: every type needs a hidden surface
    point, except a cut, which can also sit on a hidden sharp edge (without one it is a
    surface chip, ``DefectGenerator.make``)."""
    tab = gen.difficulty_table()["unreachable"]
    return [t for t in spec.unreachable_types if tab["surface"] > 0 or (t == "cut" and tab["edge"] > 0)]


def plan_instances(spec: VariantSetSpec, unreachable_types: list[str] | None = None) -> list[dict]:
    """Deterministic instance plan (ids, roles, types, tiers, difficulties)."""
    rng = rng_for("plan", spec.seed)
    plan: list[dict] = []
    for i in range(spec.n_bank):
        plan.append({"inst_id": f"n_bank_{i:04d}", "role": "bank", "type": "normal"})
    for i in range(spec.n_val):
        plan.append({"inst_id": f"n_val_{i:04d}", "role": "val", "type": "normal"})
    tiers = ["S", "M", "L"]
    for role, n in (("search", spec.per_type_search), ("report", spec.per_type_report)):
        for t in spec.types:
            diffs = _largest_remainder(n, spec.dictionary.difficulty_mix) if t != "warp" else ["global"] * n
            rng.shuffle(diffs)
            for i in range(n):
                plan.append({"inst_id": f"d_{role}_{t}_{i:02d}", "role": role, "type": t,
                             "tier": tiers[i % 3], "difficulty": diffs[i]})
    u_types = spec.unreachable_types if unreachable_types is None else unreachable_types
    for i in range(spec.n_unreachable if u_types else 0):
        t = u_types[i % len(u_types)]
        plan.append({"inst_id": f"u_{t}_{i:02d}", "role": "unreachable", "type": t, "tier": tiers[i % 3],
                     "difficulty": "unreachable"})
    return plan


# -- worker process state ---------------------------------------------------------------
_W: dict = {}


def _init_worker(part_id: str, spec_json: str) -> None:
    spec = VariantSetSpec.model_validate_json(spec_json)
    nominal = load_part_mesh(part_id, "fine")
    _W.update(part_id=part_id, spec=spec, nominal=nominal,
              gen=DefectGenerator(nominal, spec.dictionary, seed=spec.seed),
              settle=load_settle(part_id, data_root()))


def _make_instance(item: dict) -> dict:
    spec: VariantSetSpec = _W["spec"]
    nominal: LabeledMesh = _W["nominal"]
    rng = rng_for(_W["part_id"], spec.set_id, spec.seed, item["inst_id"])
    t0 = time.time()
    mesh, var_info = apply_normal_variation(nominal, rng, spec.normal)
    dmeta = None
    if item["type"] != "normal":
        mesh, dmeta = _W["gen"].make(mesh, item["type"], item["difficulty"], item["tier"], rng)
        dmeta["area_m2"] = defect_area(mesh)
    T, pose_info = sample_pose(rng, spec.pose, mesh, _W["settle"])
    out_dir = set_dir(_W["part_id"], spec.set_id) / "meshes"
    mesh.meta = {"inst_id": item["inst_id"], "part_id": _W["part_id"]}
    path = out_dir / f"{item['inst_id']}.npz"
    mesh.save(path)
    rec = dict(item)
    if dmeta is not None:
        rec["difficulty"] = dmeta.get("difficulty", item.get("difficulty"))
    rec.update({"mesh": data_rel(path), "T_world_part": T.tolist(), "pose": pose_info,
                "variation": var_info, "defect": dmeta, "n_faces": int(len(mesh.faces)),
                "gen_s": round(time.time() - t0, 3)})
    return rec


def build_variant_set(part_id: str, spec: VariantSetSpec, n_workers: int = 8, overwrite: bool = False) -> dict:
    out = set_dir(part_id, spec.set_id)
    man_path = out / "manifest.json"
    if man_path.exists() and not overwrite:
        man = read_json(man_path)
        if man.get("spec_hash") == content_hash(spec.model_dump(mode="json")):
            return man
        raise RuntimeError(f"{man_path} exists with a different spec; pass overwrite=True")
    load_part(part_id)  # existence check
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    _init_worker(part_id, spec.model_dump_json())          # main-process generator: candidate table
    u_types = placeable_unreachable(_W["gen"], spec)
    plan = plan_instances(spec, u_types)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers, initializer=_init_worker, initargs=(part_id, spec.model_dump_json())) as pool:
        records = pool.map(_make_instance, plan, chunksize=4)
    man = {
        "part_id": part_id, "set_id": spec.set_id, "spec": spec.model_dump(mode="json"),
        "spec_hash": content_hash(spec.model_dump(mode="json")),
        "template": load_part(part_id).mesh_fine,
        "difficulty_table": _W["gen"].difficulty_table(),
        "unreachable_types": u_types,
        "settle_used": _W["settle"] is not None,
        "instances": records, "build_s": round(time.time() - t0, 1),
        "counts": {r: sum(1 for x in records if x["role"] == r) for r in ("bank", "val", "search", "report", "unreachable")},
    }
    write_json(man_path, man)
    log.info(f"variant set {part_id}/{spec.set_id}: {man['counts']} in {man['build_s']}s")
    return man


def instances_by_role(man: dict, *roles: str) -> list[dict]:
    return [r for r in man["instances"] if r["role"] in roles]
