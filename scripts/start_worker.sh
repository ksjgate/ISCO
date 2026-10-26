#!/usr/bin/env bash
# Start the long-lived Isaac Sim render worker in the background (ADR-0002).
#
#   bash scripts/start_worker.sh [QUEUE_DIR] [extra worker args...]   # default QUEUE_DIR=<repo>/queue
#   tail -f queue/worker.log
#   python -c "from isco.render.queue import FileJobQueue as Q; Q('queue').request_shutdown()"   # stop
#
# Only one Isaac process should use the GPU at a time: refuses to start if a worker with
# a fresh heartbeat (and a live pid) is already serving this queue.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUEUE="${1:-$REPO/queue}"
shift || true
mkdir -p "$QUEUE"
QUEUE="$(cd "$QUEUE" && pwd)"
# shellcheck disable=SC1091
source ~/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
# ~/.bashrc sources ROS 2 Humble (Python 3.10 packages on PYTHONPATH); keep them out of Python 3.11
export PYTHONPATH=""

if [ -f "$QUEUE/worker.pid" ] && kill -0 "$(cat "$QUEUE/worker.pid")" 2>/dev/null; then
  if python - "$QUEUE" <<'EOF'
import sys
from isco.render.queue import FileJobQueue
sys.exit(0 if FileJobQueue(sys.argv[1]).worker_alive(max_age_s=60) else 1)
EOF
  then
    echo "worker already running (pid $(cat "$QUEUE/worker.pid")) for $QUEUE"
    exit 0
  fi
fi

cd "$REPO"
nohup python -m isco.sim.worker --queue "$QUEUE" --headless "$@" >> "$QUEUE/worker.log" 2>&1 &
echo $! > "$QUEUE/worker.pid"
echo "started isco.sim.worker pid $(cat "$QUEUE/worker.pid") queue=$QUEUE log=$QUEUE/worker.log"
