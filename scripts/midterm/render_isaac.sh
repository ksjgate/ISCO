#!/usr/bin/env bash
# Render the Isaac shots of the mid-term video, one Isaac process per group of shots
# (a long session crashed in the rig-camera capture after several products).
#   bash scripts/midterm/render_isaac.sh [shot ...]     # default: all groups
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=""
export ISCO_DATA_ROOT="${ISCO_DATA_ROOT:-/home/juk0324/jk/KU/ISCO/data}"
PY=/home/juk0324/anaconda3/envs/env_isaaclab/bin/python
LOG=runs/midterm_reports/isaac.log
mkdir -p runs/midterm_reports
if [ $# -gt 0 ]; then
  GROUPS_=("$@")
else
  GROUPS_=("line lineup thumbs" "defect_dent defect_bulge defect_hole defect_cut defect_crack defect_warp"
           "slots turntable" part_tless_13 part_tless_18 part_tless_29 part_tless_30 part_mulsen_nut
           part_mulsen_flat_pad)
fi
for g in "${GROUPS_[@]}"; do
  for attempt in 1 2; do
    echo "[render_isaac] $g (attempt $attempt) $(date -u)" >> "$LOG"
    # shellcheck disable=SC2086
    timeout 1500 "$PY" scripts/midterm/isaac_shots.py --only $g >> "$LOG" 2>&1
    rc=$?
    echo "[render_isaac] $g exit $rc $(date -u)" >> "$LOG"
    [ $rc -eq 0 ] && break
  done
done
echo "[render_isaac] done $(date -u)" >> "$LOG"
