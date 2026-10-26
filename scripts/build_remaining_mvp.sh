#!/usr/bin/env bash
# Resume the MVP data build for the paper part set (variant sets / caches / visibility are
# skipped or resumed when already on disk). Appends to runs/logs/build_all_mvp.log.
set -euo pipefail
export PYTHONPATH=""
source "${CONDA_BASE:-$HOME/anaconda3}/etc/profile.d/conda.sh"
conda activate env_isaaclab
cd "$(dirname "$0")/.."
isco data build tless_29 tless_30 tless_13 tless_18 --preset mvp
isco data build mulsen_nut mulsen_flat_pad mulsen_capsule --preset mvp
isco data build real3d_candybar real3d_starfish dowel --preset mvp
echo "[build_all_mvp] done"
