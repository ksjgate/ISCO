#!/usr/bin/env bash
# Install the extra Python packages ISCO needs into the existing `env_isaaclab`
# conda environment (Isaac Sim 5.1, Python 3.11). Never creates a new env.
#
# Usage:  bash scripts/install_extra_deps.sh [--dry-run]
set -euo pipefail

source "${CONDA_BASE:-$HOME/anaconda3}/etc/profile.d/conda.sh"
conda activate env_isaaclab

PKGS=(
  "open3d==0.20.0"          # FPFH, ICP, RaycastingScene (embree), Poisson meshing
  "optuna>=4.5"             # NSGA-II / multi-objective TPE search
  "faiss-cpu==1.11.0"       # memory-bank kNN (CPU fallback). >=1.12 imports numpy._core (numpy-2 only)
  "scikit-learn"            # ROC / AUROC
  "pymeshlab"               # isotropic remeshing
  "manifold3d"              # robust mesh booleans (hole / cut / crack defects)
  "embreex"                 # fast trimesh ray queries (optional accel)
  "point-cloud-utils"       # chamfer / sampling helpers (optional)
  "seaborn"                 # figures
  "pymoo"                   # hypervolume indicator
  "gdown"                   # Google-Drive dataset downloads (Real3D-AD)
  "markdown"                # report HTML rendering
  "tifffile<2025.6"         # MVTec 3D-AD xyz TIFF (float32 x3); newer releases need numpy>=2
)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Constraints keep Isaac Sim's pins intact (e.g. psutil==5.9.8: newer ipython,
# pulled in by open3d, would otherwise upgrade it to 7.x).
CONSTRAINTS=(-c "$HERE/constraints-isaacsim.txt")

if [[ "${1:-}" == "--dry-run" ]]; then
  pip install --dry-run "${CONSTRAINTS[@]}" "${PKGS[@]}"
  exit 0
fi

pip install "${CONSTRAINTS[@]}" "${PKGS[@]}"
pip check || true
python - <<'PY'
import importlib
mods = ["open3d", "optuna", "faiss", "sklearn", "pymeshlab", "manifold3d",
        "embreex", "point_cloud_utils", "seaborn", "pymoo", "gdown", "markdown",
        "trimesh", "torch", "scipy", "numpy"]
for m in mods:
    mod = importlib.import_module(m)
    print(f"{m:20s} {getattr(mod, '__version__', 'ok')}")
PY
