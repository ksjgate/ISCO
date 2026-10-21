#!/usr/bin/env bash
# Post-hoc exploratory follow-ups (pre-registration §12, 2026-10-21), run after the
# pre-registered queue (scripts/run_experiments.sh) has finished:
#   e7_n0    E7 noise ablation at N0 next to N2 (E5b: the real Zivid scans are at N0)
#   e1_full  E1 with seeds 0,1,2 and budget 60 per seed (the "full" search settings)
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

STEPS=(e7_n0 e1_full)
START="${1:-e7_n0}"
run=0
for s in "${STEPS[@]}"; do
  [[ "$s" == "$START" ]] && run=1
  [[ $run -eq 0 ]] && continue
  echo "[run_followups] === $s === $(date)"
  case "$s" in
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
