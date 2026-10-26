#!/usr/bin/env bash
# Steps run after the pre-registered queue (scripts/run_experiments.sh) has finished
# (pre-registration §12):
#   real     rebuild the MulSen-AD real sets with the fixed registration (2026-10-24)
#   e5       E5 / H3 rerun on them (the 10-23 run crashed on mulsen_capsule)
#   e7_n0    exploratory: E7 noise ablation at N0 next to N2 (E5b: the real Zivid scans are at N0)
#   e1_full  exploratory: E1 with seeds 0,1,2 and budget 60 per seed (the "full" search settings)
#   spec_sets       held-out report sets (144 defects, 200 validation normals) of the six MVP parts
#   specialization  part-specific optimum vs universal configuration (registered 2026-10-26)
#   feature_map     part shape features vs optimal configurations, 16 parts (registered 2026-10-26)
# Logs: runs/logs/exp_<step>.log, status lines on stdout.
#
#   bash scripts/run_followups.sh [first_step]        # waits for the queue, then runs
set -uo pipefail
export PYTHONPATH=""
source "${CONDA_BASE:-$HOME/anaconda3}/etc/profile.d/conda.sh"
conda activate env_isaaclab
cd "$(dirname "$0")/.."
mkdir -p runs/logs
W=${ISCO_WORKERS:-22}
MVP6="--parts tless_13 --parts tless_18 --parts tless_29 --parts tless_30 --parts mulsen_nut --parts mulsen_flat_pad"
FMAP16="$MVP6 --parts mulsen_capsule --parts real3d_candybar --parts real3d_starfish --parts dowel"
FMAP16="$FMAP16 --parts tless_16 --parts tless_10 --parts tless_24 --parts tless_08 --parts tless_17 --parts tless_14"

QPID=$(pgrep -f "^bash scripts/run_experiments.sh" | head -1)
echo "[run_followups] waiting for the experiment queue (pid ${QPID:-none}) $(date)"
while [[ -n "$QPID" ]] && kill -0 "$QPID" 2>/dev/null; do sleep 300; done
echo "[run_followups] queue finished: $(tail -n 1 runs/logs/run_experiments.log) $(date)"

STEPS=(real e5 e7_n0 e1_full spec_sets specialization feature_map)
START="${1:-real}"
run=0
for s in "${STEPS[@]}"; do
  [[ "$s" == "$START" ]] && run=1
  [[ $run -eq 0 ]] && continue
  echo "[run_followups] === $s === $(date)"
  case "$s" in
    real)    python -c "
from isco.eval.real_parts import build_mulsen_real_set
from isco.render.slot_cache import build_slot_cache
from isco.visibility.raycast import build_visibility
for c in ('nut', 'flat_pad', 'capsule'):
    build_mulsen_real_set(c, overwrite=True)
    build_slot_cache('mulsen_' + c, 'real', n_workers=6)
    build_visibility('mulsen_' + c, 'real')
" ;;
    e5)      isco exp run e5 --parts mulsen_nut --parts mulsen_flat_pad --parts mulsen_capsule --workers $W ;;
    e7_n0)   for p in tless_13 tless_29; do
               [[ -s runs/exp/e2/evals_$p.jsonl ]] || { echo "missing E2 evaluations for $p"; exit 4; }
             done && python -c "
from isco.eval.hypotheses import e7
e7(parts=['tless_13', 'tless_29'], noises=('N0', 'N2'), n_workers=$W, out='runs/exp/e7_n0')
" && python scripts/analyze_noise_levels.py ;;
    e1_full) isco exp run e1 $MVP6 --seeds 0,1,2 --budget 60 --out runs/exp/e1_full --workers $W ;;
    spec_sets) python -c "
from isco.datagen.variants import build_report_extension
from isco.render.slot_cache import build_slot_cache
from isco.visibility.raycast import build_visibility
for p in ('tless_13', 'tless_18', 'tless_29', 'tless_30', 'mulsen_nut', 'mulsen_flat_pad'):
    build_report_extension(p)
    build_slot_cache(p, 'mvp_r144', n_workers=$W)
    build_visibility(p, 'mvp_r144', n_workers=$W)
" ;;
    specialization) [[ -s runs/exp/e1_full/summary.json ]] || { echo "E1-full has no summary"; exit 4; }
             isco exp run specialization $MVP6 --workers $W ;;
    feature_map) isco exp run feature_map $FMAP16 --workers $W ;;
  esac > "runs/logs/exp_$s.log" 2>&1
  echo "[run_followups] $s exit $? $(date)"
done
echo "[run_followups] done $(date)"
