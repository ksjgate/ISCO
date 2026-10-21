#!/usr/bin/env bash
# Prioritised experiment queue (plan: mid-term demo -> E0 calibration -> hypotheses).
# Every stage-2 evaluation is cached (data/evalcache), so the queue can be stopped and
# re-run; finished work is reused. Logs: runs/logs/exp_<step>.log
#
#   bash scripts/run_experiments.sh [first_step]      # e.g. start from "e1"
set -uo pipefail
export PYTHONPATH=""
source "${CONDA_BASE:-$HOME/anaconda3}/etc/profile.d/conda.sh"
conda activate env_isaaclab
cd "$(dirname "$0")/.."
mkdir -p runs/logs
W=${ISCO_WORKERS:-22}
# part sets and budgets follow the pre-registration: search budget 40 per run (E2: 60
# stratified configurations) = the experiment functions' defaults; C1 is decided on >= 4/6
# parts (E3 on the six MVP parts), H4 on >= 2/3 parts (E4 on three T-LESS parts)
TLESS="--parts tless_13 --parts tless_18 --parts tless_29 --parts tless_30"
MVP6="$TLESS --parts mulsen_nut --parts mulsen_flat_pad"
TRIPLE="--parts tless_13 --parts tless_29 --parts tless_30"
PAIR="--parts tless_13 --parts tless_29"
# calibration first: E0.5 decides the defect scale; "calib" stops the queue when it is out
# of target (variant sets must then be rebuilt with the recommended scale_k)
STEPS=(waitbuild e0_3 e0_4 e0_5 calib demo assets gate sanity e0_2 e1 e2 e3 e4 real e5 e6 e7 e7b e8 dino)
START="${1:-waitbuild}"
run=0
for s in "${STEPS[@]}"; do
  [[ "$s" == "$START" ]] && run=1
  [[ $run -eq 0 ]] && continue
  echo "[run_experiments] === $s === $(date)"
  case "$s" in
    demo)   isco run fusion_51876 --constraint "n<=3,T<=1.5" --search demo --out runs/demo_fusion_51876 --workers $W ;;
    assets) python scripts/make_demo_assets.py --workers $W --rig-run runs/demo_fusion_51876 ;;
    gate)   bash scripts/smoke_gate.sh fusion_51876 12,20 ;;
    waitbuild) until grep -q "\[build_all_mvp\] done" runs/logs/build_all_mvp.log 2>/dev/null; do sleep 30; done ;;
    e0_3)   isco exp run e0_3 $TLESS --parts mulsen_nut --parts fusion_51876 ;;
    e0_4)   isco exp run e0_4 --parts tless_13 --parts fusion_51876 --workers $W ;;
    e0_5)   isco exp run e0_5 $TLESS --parts mulsen_nut --parts mulsen_flat_pad --workers $W ;;
    calib)  python -c "
import json, sys
s = json.load(open('runs/exp/e0_5/summary.json'))
print('mean FNR at N2', s['mean_fnr_N2'], 'in target', s['in_target'], 'recommended scale_k', s['scale_k_recommended'])
sys.exit(0 if s['in_target'] else 3)" || { echo "[run_experiments] calibration out of target: rebuild variant sets, then rerun from e0_5 $(date)"; exit 3; } ;;
    sanity) isco exp run result_sanity --parts fusion_22340 --parts fusion_41229 --parts fusion_51876 --workers $W ;;
    e0_2)   bash scripts/start_worker.sh && isco exp run e0_2 --parts tless_29 --parts tless_13 --workers $W; \
            python -c "from isco.render.queue import FileJobQueue as Q; Q('queue').request_shutdown()" ;;
    e1)     isco exp run e1 $MVP6 --workers $W ;;
    e2)     isco exp run e2 $TLESS --workers $W ;;
    e3)     isco exp run e3 $MVP6 --workers $W ;;
    e4)     isco exp run e4 $TRIPLE --workers $W ;;
    real)   python -c "
from isco.eval.real_parts import build_mulsen_real_set
from isco.render.slot_cache import build_slot_cache
from isco.visibility.raycast import build_visibility
for c in ('nut', 'flat_pad', 'capsule'):
    build_mulsen_real_set(c)
    build_slot_cache('mulsen_' + c, 'real', n_workers=6)
    build_visibility('mulsen_' + c, 'real')
" ;;
    e5)     isco exp run e5 --parts mulsen_nut --parts mulsen_flat_pad --parts mulsen_capsule --workers $W ;;
    e6)     isco exp run e6 $PAIR --workers $W ;;
    e7)     isco exp run e7 $PAIR --workers $W ;;
    e7b)    isco data build tless_29 --preset mvp --yaw-mode random --set-id mvp_yawrand && \
            isco exp run e7b --parts tless_29 --workers $W ;;
    e8)     isco exp run e8 $PAIR --seeds 0,1 --workers $W ;;
    dino)   python -c "
from isco.eval.hypotheses import e6
e6(parts=['tless_13', 'tless_29'], n_workers=$W, out='runs/exp/e6_dino', detectors=('dinov2_bank',))
" ;;
  esac > "runs/logs/exp_$s.log" 2>&1
  echo "[run_experiments] $s exit $? $(date)"
done
echo "[run_experiments] done $(date)"
