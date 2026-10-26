#!/usr/bin/env bash
# Build MVP variant sets + slot caches + visibility tables for the whole part set.
# (ROS 2 on this machine exports a Python-3.10 PYTHONPATH; clear it for env_isaaclab.)
set -euo pipefail
export PYTHONPATH=""
source "${CONDA_BASE:-$HOME/anaconda3}/etc/profile.d/conda.sh"
conda activate env_isaaclab
cd "$(dirname "$0")/.."
isco data build fusion_51876 fusion_22340 fusion_41229 fusion_26768 --preset mvp --overwrite
isco data build tless_29 tless_30 tless_13 tless_18 --preset mvp
isco data build mulsen_nut mulsen_flat_pad mulsen_capsule --preset mvp
isco data build real3d_candybar real3d_starfish dowel --preset mvp
echo "[build_all_mvp] done"
