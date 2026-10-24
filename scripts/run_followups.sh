#!/usr/bin/env bash
# Steps run after the pre-registered queue (scripts/run_experiments.sh) has finished
# (pre-registration §12):
#   real     rebuild the MulSen-AD real sets with the fixed registration (2026-10-24)
#   e5       E5 / H3 rerun on them (the 10-23 run crashed on mulsen_capsule)
#   e7_n0    exploratory: E7 noise ablation at N0 next to N2 (E5b: the real Zivid scans are at N0)
#   e1_full  exploratory: E1 with seeds 0,1,2 and budget 60 per seed (the "full" search settings)
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

QPID=$(pgrep -f "^bash scripts/run_experiments.sh" | head -1)
echo "[run_followups] waiting for the experiment queue (pid ${QPID:-none}) $(date)"
while [[ -n "$QPID" ]] && kill -0 "$QPID" 2>/dev/null; do sleep 300; done
echo "[run_followups] queue finished: $(tail -n 1 runs/logs/run_experiments.log) $(date)"

STEPS=(real e5 e7_n0 e1_full)
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
" ;;
    e1_full) isco exp run e1 $MVP6 --seeds 0,1,2 --budget 60 --out runs/exp/e1_full --workers $W ;;
  esac > "runs/logs/exp_$s.log" 2>&1
  echo "[run_followups] $s exit $? $(date)"
done
echo "[run_followups] done $(date)"
