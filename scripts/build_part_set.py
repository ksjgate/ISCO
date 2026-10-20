"""Build variant set + slot cache + visibility table for parts (CLI-independent helper).

    python scripts/build_part_set.py --preset mvp fusion_51876 fusion_22340 ...
"""
from __future__ import annotations

import argparse
import time

from isco.datagen.variants import VariantSetSpec, build_variant_set
from isco.render.slot_cache import build_slot_cache
from isco.utils.logging import get_logger
from isco.visibility.raycast import build_visibility

log = get_logger("build_part_set")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="+")
    ap.add_argument("--preset", default="mvp")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--yaw-mode", default="guided", choices=["guided", "random"])
    ap.add_argument("--set-id", default=None)
    a = ap.parse_args()
    for pid in a.parts:
        t0 = time.time()
        spec = VariantSetSpec.preset(a.preset)
        spec.pose.yaw_mode = a.yaw_mode
        if a.set_id:
            spec.set_id = a.set_id
        man = build_variant_set(pid, spec, n_workers=8, overwrite=a.overwrite)
        build_slot_cache(pid, spec.set_id, n_workers=a.workers, overwrite=a.overwrite)
        build_visibility(pid, spec.set_id, n_workers=8, overwrite=a.overwrite)
        log.info(f"{pid}/{spec.set_id} ready in {time.time() - t0:.0f}s ({man['counts']})")


if __name__ == "__main__":
    main()
