#!/usr/bin/env bash
# Week-5 gate in one command: Isaac worker -> render 3 instances x 2 slots (checked against
# the embree cache) -> detector-in-the-loop evaluation of one 2-camera configuration ->
# JSON with FNR@1%FPR and cost.   Usage: bash scripts/smoke_gate.sh [part] [slots]
set -euo pipefail
export PYTHONPATH=""
source "${CONDA_BASE:-$HOME/anaconda3}/etc/profile.d/conda.sh"
conda activate env_isaaclab
cd "$(dirname "$0")/.."
python scripts/smoke_gate.py --part "${1:-fusion_51876}" --slots "${2:-12,20}" --stop-worker
