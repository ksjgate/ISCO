#!/usr/bin/env bash
# Download the public datasets ISCO uses into data/datasets/<name>/raw/ and
# extract them there. Each dataset lives in its own directory (downloaded
# archives are treated as untrusted data; nothing inside them is executed).
#
# Usage: bash scripts/download_datasets.sh [tless] [mulsen] [anomaly_shapenet] [real3d]
#        (no argument = all four)
#
# MVTec 3D-AD needs a registration form, so it is NOT downloaded here:
#   https://www.mvtec.com/company/research/datasets/mvtec-3d-ad/downloads
#   -> put the extracted category folders (at least dowel/ and cable_gland/) in
#      data/datasets/mvtec3d/raw/
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT/data/datasets"
source "${CONDA_BASE:-$HOME/anaconda3}/etc/profile.d/conda.sh"
conda activate env_isaaclab

want() { [[ $# -eq 0 ]] && return 0; return 1; }
TARGETS=("$@"); [[ ${#TARGETS[@]} -eq 0 ]] && TARGETS=(tless mulsen anomaly_shapenet real3d)

hf_get() {  # repo file outdir
  mkdir -p "$3"
  hf download "$1" "$2" --repo-type dataset --local-dir "$3"
}

for t in "${TARGETS[@]}"; do
  case "$t" in
    tless)
      # T-LESS CAD + reconstructed models only (33.5 MB). CC BY 4.0.
      d="$DATA/tless/raw"; hf_get bop-benchmark/tless tless_models.zip "$d"
      (cd "$d" && unzip -o -q tless_models.zip -d extracted) ;;
    mulsen)
      # MulSen-AD (CVPR 2025), MIT. ~9 GB zip.
      d="$DATA/mulsen_ad/raw"; hf_get orgjy314159/MulSen_AD MulSen_AD_new.zip "$d"
      (cd "$d" && unzip -o -q MulSen_AD_new.zip -d extracted) ;;
    anomaly_shapenet)
      # Anomaly-ShapeNet v2 (CVPR 2024), CC BY 4.0. ~3.8 GB rar.
      d="$DATA/anomaly_shapenet/raw"; hf_get Chopper233/Anomaly-ShapeNet Anomaly-ShapeNet-v2.rar "$d"
      # p7zip cannot decode this RAR method ("Unsupported Method"); libarchive can.
      (cd "$d" && mkdir -p extracted && "${CONDA_BASE:-$HOME/anaconda3}/bin/bsdtar" -xf Anomaly-ShapeNet-v2.rar -C extracted) ;;
    real3d)
      # Real3D-AD (NeurIPS 2023 D&B), CC BY 4.0. Google Drive (PCD release).
      d="$DATA/real3d_ad/raw"; mkdir -p "$d"
      gdown "1oM4qjhlIMsQc_wiFIFIVBvuuR8nyk2k0" -O "$d/Real3D-AD-PCD.zip"   # gdown>=6 dropped --fuzzy; pass the file id
      (cd "$d" && unzip -o -q Real3D-AD-PCD.zip -d extracted) ;;
    *) echo "unknown dataset: $t" >&2; exit 2 ;;
  esac
  echo "[download_datasets] done: $t"
done
