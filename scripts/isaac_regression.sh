#!/usr/bin/env bash
# PoC regression for the isco.sim refactor of isaac_sim/create_conveyor.py
# (the three isaac_sim/todo.md commands + before/after of the ADR-0008 capture race).
#
#   bash scripts/isaac_regression.sh            # outputs under runs/regression/
#   OUT=runs/regression_x bash scripts/isaac_regression.sh
#
# Never writes to the git-tracked isaac_sim/output/* or isaac_sim/assets/conveyor_belt.usd
# (every command gets --save / --video-out / --views-out-dir under $OUT).
# GUI note: `python create_conveyor.py` (interactive GUI) cannot be opened here; its code
# path (world.play() loop + spawner + save on exit) is exercised headlessly with
# --gui-loop-seconds. The interactive window itself is NOT tested by this script.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
# shellcheck disable=SC1091
source ~/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
# ~/.bashrc sources ROS 2 Humble (Python 3.10 packages on PYTHONPATH); keep them out of Python 3.11
export PYTHONPATH=""
OUT="${OUT:-runs/regression}"
PART="${PART:-41229_16283ae1_0018.stl}"
PARTS_DIR="data/parts/raw/fusion360"
CC="python isaac_sim/create_conveyor.py"
mkdir -p "$OUT"
declare -A RC

run() {  # name, command...
  local name="$1"; shift
  mkdir -p "$OUT/$name"
  echo "=== [$name] $*"
  local t0=$SECONDS
  "$@" > "$OUT/$name/log.txt" 2>&1
  RC[$name]=$?
  echo "    exit=${RC[$name]} ($((SECONDS - t0)) s)"
  grep -E "^\[create_conveyor\]|Traceback" "$OUT/$name/log.txt" | grep -v "saved view" | tail -n 8 | sed 's/^/    /'
}

# 0. --list-assets (no SimulationApp; finds track_types.json through the installed isaacsim package)
run list_assets $CC --list-assets

# 1. todo.md: `python create_conveyor.py` (GUI) -> same loop, headless, bounded to 12 s of sim time
run gui_path $CC --headless --gui-loop-seconds 12 --save "$OUT/gui_path/conveyor_belt.usd"

# 2. todo.md: `python create_conveyor.py --headless --video`
run video $CC --headless --video --video-out "$OUT/video/conveyor_belt.mp4" --save "$OUT/video/conveyor_belt.usd"

# 3. todo.md: `python create_conveyor.py --headless --capture-views --views-part 41229_16283ae1_0018.stl`
run capture_fixed $CC --headless --capture-views --views-part "$PART" \
    --views-out-dir "$OUT/capture_fixed" --save "$OUT/capture_fixed/stage.usd"
#    3b. the refactor reproducing the original race (comparison only)
run capture_legacy_race $CC --headless --capture-views --legacy-race --views-part "$PART" \
    --views-out-dir "$OUT/capture_legacy_race" --save "$OUT/capture_legacy_race/stage.usd"
#    3c. the original PoC script from git HEAD (8b28c36) = "before"
mkdir -p "$OUT/original"
git show 8b28c36:isaac_sim/create_conveyor.py > "$OUT/original/create_conveyor_original.py"
run capture_before_original python "$OUT/original/create_conveyor_original.py" --headless --capture-views \
    --views-part "$PART" --parts-dir "$PARTS_DIR" --views-out-dir "$OUT/capture_before_original" \
    --save "$OUT/capture_before_original/stage.usd"

# 4. summary (pure python, no Isaac): metrics at 2 mm and 0.2 mm for every capture
python - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
import numpy as np, trimesh
from isco.sim.capture import compare_reconstruction
out = Path(sys.argv[1])
rows = []
for name, label in [("capture_before_original", "before: original PoC script (git 8b28c36)"),
                    ("capture_legacy_race", "refactor --legacy-race (PoC logic)"),
                    ("capture_fixed", "after: refactor, race fixed (default)")]:
    d = out / name
    if not (d / "merged_pointcloud.npy").exists():
        rows.append({"run": name, "label": label, "error": "missing outputs"}); continue
    pts = np.load(d / "merged_pointcloud.npy")
    gt = trimesh.load(d / "ground_truth_mesh.ply", process=False)
    c = compare_reconstruction(np.asarray(gt.vertices), np.asarray(gt.faces), pts, 0.002, seed=0)
    s = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
    rows.append({"run": name, "label": label, "merged_points": int(len(pts)),
                 "completeness@2mm": c["completeness"], "accuracy@2mm": c["accuracy"],
                 "accuracy_exact@2mm": c["accuracy_exact"], "completeness@0.2mm": c["completeness_fine"],
                 "accuracy@0.2mm": c["accuracy_exact_fine"],
                 "accuracy_exact_median_mm": c["accuracy_exact_median_m"] * 1e3,
                 "part_drift_mm": s.get("part_drift_between_gt_read_and_capture_mm"),
                 "summary_txt": (d / "summary.txt").read_text() if (d / "summary.txt").exists() else ""})
checks = {}
v = out / "video" / "conveyor_belt.mp4"
checks["video_mp4_bytes"] = v.stat().st_size if v.exists() else 0
try:
    import imageio
    r = imageio.get_reader(str(v)); checks["video_frames"] = r.count_frames(); r.close()
except Exception as e:
    checks["video_frames"] = f"n/a ({e})"
for n in ("gui_path", "video"):
    p = out / n / "conveyor_belt.usd"
    checks[f"{n}_usd_bytes"] = p.stat().st_size if p.exists() else 0
gl = (out / "gui_path" / "log.txt").read_text(errors="ignore") if (out / "gui_path" / "log.txt").exists() else ""
checks["gui_loop_line"] = next((l for l in gl.splitlines() if "GUI loop ran" in l), None)
la = (out / "list_assets" / "log.txt").read_text(errors="ignore") if (out / "list_assets" / "log.txt").exists() else ""
checks["list_assets_count"] = sum(1 for l in la.splitlines() if "style=" in l)
res = {"rows": rows, "checks": checks}
(out / "regression_summary.json").write_text(json.dumps(res, indent=2))
md = ["| run | points | comp@2mm | acc@2mm (PoC) | acc@2mm (exact) | comp@0.2mm | acc@0.2mm (exact) | median err (mm) | drift (mm) |",
      "|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    if "error" in r:
        md.append(f"| {r['label']} | {r['error']} |"); continue
    drift = "-" if r["part_drift_mm"] is None else f"{r['part_drift_mm']:.1f}"
    md.append(f"| {r['label']} | {r['merged_points']} | {r['completeness@2mm']*100:.1f}% | {r['accuracy@2mm']*100:.1f}% | "
              f"{r['accuracy_exact@2mm']*100:.1f}% | {r['completeness@0.2mm']*100:.1f}% | {r['accuracy@0.2mm']*100:.2f}% | "
              f"{r['accuracy_exact_median_mm']:.4f} | {drift} |")
md.append("")
md.append("checks: " + json.dumps(checks))
(out / "regression_summary.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
EOF

fail=0
for k in "${!RC[@]}"; do [ "${RC[$k]}" -ne 0 ] && { echo "FAILED: $k (exit ${RC[$k]})"; fail=1; }; done
exit $fail
