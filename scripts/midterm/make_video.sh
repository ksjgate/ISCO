#!/usr/bin/env bash
# Compose the mid-term demo video and encode the two delivery copies:
#   *_고화질.mp4 (CRF 22, for playback) and *_압축.mp4 (CRF 28, embedded in the slides / web page).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=""
export ISCO_DATA_ROOT="${ISCO_DATA_ROOT:-/home/juk0324/jk/KU/ISCO/data}"
PY=/home/juk0324/anaconda3/envs/env_isaaclab/bin/python
FF=/home/juk0324/anaconda3/envs/env_isaaclab/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
W=runs/midterm_reports
"$PY" scripts/midterm/video.py
for q in "고화질 22" "압축 28"; do
  set -- $q
  "$FF" -y -loglevel error -i "$W/ISCO_중간발표_데모영상.mp4" -c:v libx264 -crf "$2" -preset slow -pix_fmt yuv420p \
    -movflags +faststart "$W/ISCO_중간발표_데모영상_$1.mp4"
done
ls -la "$W"/*.mp4
