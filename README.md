# ISCO — Inspection Setup Configuration Optimizer

ISCO searches the configuration of an **inline 3D inspection cell** — how many depth
cameras, where to put them on a hemisphere around the conveyor, whether to use a rotation
stage, how fast the belt runs and how the cameras are triggered — so that a
**normal-only 3D anomaly detector** misses as few defects as possible at a fixed false
alarm rate (**FNR @ 1 % FPR**) for the least cost (**camera count, cycle time**).
The cell is a digital twin built in **NVIDIA Isaac Sim** (conveyor, parts, cameras); the
output is a performance–cost **Pareto front**, a **recommended configuration** under user
constraints, and per-defect-type visibility/detection rates.

```
part CAD / 3D scan ──► rest-frame mesh ──► variant set (normal variation + defect dictionary)
                                                │
                    96 camera slots ──► slot cache (render once per instance x slot)
                                                │
   stage 1: ray-cast defect visibility (ms/config) ──► shortlist
   stage 2: compose views ─► sensor & process model ─► merge + ICP ─► FPFH memory bank
            ─► FNR@1%FPR (held-out defects, bootstrap CI)        (~0.5-1 min/config)
                                                │
   multi-objective search (NSGA-II / MOTPE) ──► Pareto front + recommended config + report
```

## Installation

ISCO runs in the existing conda environment `env_isaaclab` (Isaac Sim 5.1, Python 3.11,
PyTorch cu128). Do not create a new environment.

```bash
conda activate env_isaaclab
bash scripts/install_extra_deps.sh          # open3d, optuna, faiss-cpu, pymeshlab, manifold3d, ...
pip install -e . --no-deps                  # installs the `isco` package and CLI
```

`scripts/constraints-isaacsim.txt` pins the versions Isaac Sim relies on (`numpy==1.26.0`,
`psutil==5.9.8`, `torch==2.7.0+cu128`) so the extra packages never upgrade them.

If ROS 2 is sourced in your shell, it exports a Python 3.10 `PYTHONPATH`; clear it for
ISCO (`export PYTHONPATH=`). The pytest configuration already disables the ROS plugins.

## Data

```bash
bash scripts/download_datasets.sh            # T-LESS models, MulSen-AD, Anomaly-ShapeNet, Real3D-AD
```

| dataset | use in ISCO | licence |
|---|---|---|
| T-LESS (CAD models) | redistributable industrial CAD parts (`tless_13/18/29/30`) | CC BY 4.0 |
| MulSen-AD | real industrial parts and **real defective scans** (`mulsen_nut/flat_pad/capsule`) | MIT |
| Real3D-AD | 360° prototype scans → Poisson meshes (`real3d_candybar/starfish`) | CC BY 4.0 |
| Anomaly-ShapeNet v2 | defect taxonomy reference | CC BY 4.0 |
| MVTec 3D-AD | sensor gap (real scans re-rendered through the sensor model) / BTF detector reproduction (manual download, registration form) | CC BY-NC-SA 4.0 |
| Fusion 360 Gallery bodies | original proof-of-concept parts (`fusion_*`), kept **locally only** in `data/parts/raw/fusion360/` | non-commercial research, no redistribution |

MVTec 3D-AD must be downloaded by hand from
<https://www.mvtec.com/company/research/datasets/mvtec-3d-ad/downloads>; extract the
category folders (at least `dowel/`, `cable_gland/`) into `data/datasets/mvtec3d/raw/`.
Everything under `data/`, `runs/` and `queue/` is generated and git-ignored.

## Quick start

```bash
isco parts prepare-fusion                        # proof-of-concept parts (local data)
isco parts prepare-tless                         # T-LESS 13, 18, 29, 30
isco parts prepare-mulsen --categories nut --categories flat_pad --categories capsule
isco parts prepare-real3d --categories candybar --categories starfish
isco parts prepare-dowel
isco parts list

isco data build fusion_51876 --preset mvp        # variants + slot cache + visibility (~5 min/part)

# one configuration in -> FNR + cost out
isco evaluate fusion_51876 --slots 12,16,20 --rotation none --speed 1 --trigger simultaneous

# full search + report (stage 1 screening, stage 2 detector-in-the-loop)
isco search fusion_51876 --search full --constraint "n<=3,T<=1.5"

# demo flow: builds data if missing, searches, writes runs/<id>/report/index.html
isco run fusion_51876 --constraint "n<=3,T<=1.5" --out runs/demo_fusion_51876
```

Slots are numbered `k = (i_dist * 4 + i_el) * 12 + i_az` over azimuth 0..330° (step 30°),
elevation {20, 40, 60, 80}° and distance {0.25, 0.40} m; e.g. 12/16/20 is the reference
rig: three cameras at 40° elevation, 120° apart, 0.25 m from the part.

A run directory contains `run.json` (all settings), `env.json` (versions, git sha),
`stage1.jsonl`, `stage2.jsonl` and `report/` (`index.md`, `index.html`,
`recommended.json`, Pareto / rig / per-type figures). Stage-2 evaluations are cached in
`data/evalcache/`, so interrupted searches resume and experiments share work.

## Isaac Sim

```bash
bash scripts/start_worker.sh                     # long-lived headless worker (filesystem queue)
isco worker status
isco sim export-occluder                         # conveyor meshes -> data/conveyor/occluder.npz
isco sim settle fusion_51876                     # physics drops -> data/settle/<part>.json
isco sim rgb fusion_51876 --slots 12,16,20       # RGB snapshots of a rig
isco sim check-rig fusion_51876                  # render all 96 slots, cross-check slot validity
isco worker stop
```

`bash scripts/smoke_gate.sh [part] [slots]` is the one-command gate: it starts the worker,
renders three instances from two slots in Isaac Sim and compares them with the slot
cache, evaluates one two-camera configuration and prints FNR@1%FPR and cost as JSON.

The worker renders `distance_to_image_plane` depth, semantic labels (part / defect) and
normals with the same pinhole conventions as the embree renderer used for the bulk slot
cache; the two agree to sub-micrometre depth (`scripts/e01_isaac_vs_embree.py`).

The original proof-of-concept script still works as a thin wrapper over `isco.sim`:

```bash
cd isaac_sim
python create_conveyor.py                                    # GUI, parts keep flowing
python create_conveyor.py --headless --video                 # demo video
python create_conveyor.py --headless --capture-views --views-part 41229_16283ae1_0018.stl
```

`--capture-views` now freezes the part before the camera warm-up and filters points by
semantic label (the earlier version read the part pose before rendering while the belt
kept moving it). `bash scripts/isaac_regression.sh` reruns the three commands.

## Experiments

`isco exp run <name> --parts <part> [--parts ...] --budget N --seeds 0,1`

| name | question |
|---|---|
| `e0_2` | slot-cache approximation vs direct Isaac inline render (worker required) |
| `e0_3` | ground-truth defect masks (labelled vs nominal area) |
| `e0_4` | timing (cache build, stage 1, stage 2) |
| `e0_5` | severity calibration of the defect dictionary at a reference configuration |
| `e0_6`, `e5b` | MVTec 3D-AD: BTF detector reproduction; sensor gap (real scans re-rendered through the sensor model) |
| `e1` | are optimal configurations part-specific? (cross-application matrix) |
| `e2` | is ray-cast visibility a valid surrogate for detection? |
| `e3` | is the detection objective non-additive over views (greedy coverage vs BO)? |
| `e4` | does the inline process change the recommendation vs static capture? |
| `e5` | are configuration rankings preserved on real defective scans (MulSen-AD)? |
| `e6`, `e7`, `e7b` | detector, sensor-noise and part-pose robustness |
| `e8` | search efficiency (hypervolume vs evaluations) |

Results go to `runs/exp/<name>/summary.json` (+ figures). Real-defect variant sets for
`e5` are built with `isco.eval.real_parts.build_mulsen_real_set`.
`bash scripts/run_experiments.sh [first_step]` runs the whole prioritised queue (E0
calibration, demo, demo assets, gate, result sanity, E1–E8). It stops after E0.5 when the
reference configuration's FNR is outside the calibration target, so the variant sets can
be rebuilt with the recommended defect scale first. Every stage-2 evaluation is cached in
`data/evalcache/`, so the queue can be stopped and restarted. `scripts/make_demo_assets.py`
writes the mid-term demo figures (baseline rigs per part, defect examples, coverage maps,
Isaac RGB of the recommended rig).

Detectors (`configs/detector/`): `fpfh_bank` (default, multi-scale FPFH memory bank),
`cad_distance` (training-free baseline: distance to the nominal CAD after ICP) and
`dinov2_bank` (DINOv2-S/14 features of rendered normal-map views back-projected to points).
Set `ISCO_WANDB=1` to also log search runs to Weights & Biases (optional; `wandb` is not
installed by default).

## Configuration

`configs/` holds named YAML files: `sensor/` (1024², 20° HFOV structured-light model),
`noise/` (N0–N3, N2MS: axial noise, incidence dropout, flying pixels, the camera's on-board
edge filter and exposure blur), `process/` (`inline_default`, `static_capture`), `grid/`,
`detector/` (`fpfh_bank`, `cad_distance`), `defects/` (defect dictionary), `variants/`
(`smoke`, `mvp`, `full`) and `search/` (`smoke`, `demo`, `full`).

## Repository layout

```
isco/
  schemas/      pydantic contracts (config, sensor/noise, process, part, worker jobs)
  geometry/     frames, pinhole model + slot grid, labeled meshes, remeshing
  datagen/      parts registry, normal variation, defect generator, variant sets, sensor noise
  render/       embree renderer, occluder, slot cache, view-pixel format, job queue
  visibility/   per-slot defect visibility + stage-1 surrogate metrics
  detect/       view composition/registration, FPFH features, memory bank, coverage and
                CAD-distance scorers, metrics, stage-2 pipeline
  optim/        search space, cost model, evaluator (+cache), search strategies, Pareto, runs
  report/       results store, figures, markdown/HTML report
  eval/         real-data parts (T-LESS, MulSen-AD, Real3D-AD), E0-E8 experiments, MVTec 3D-AD
  sim/          Isaac Sim: app, scene, part prims, camera rig, capture, settling, worker
  cli/          `isco` command line
isaac_sim/      original proof-of-concept entry point (wrapper) and its outputs
configs/        YAML configs            scripts/   install, download, worker, regression, builds
tests/unit      pytest (no Isaac)       tests/isaac  `pytest -m isaac tests/isaac`
```

## Conventions

* World frame: +Z up, +X = belt travel, belt top at z = 0, focus point above the origin.
* Parts are stored in a rest frame (most stable resting pose by lowest centre of mass,
  long side along +X, lowest point at z = 0) and normalised to a 40 mm longest side.
* Cameras use OpenCV axes; pixel centres at (u + 0.5, v + 0.5); depth is the distance to
  the image plane.
* A rotation stage turns the part about the vertical axis through the focus point, so a
  rotated view equals the cached view of the azimuth-shifted slot.

## Tests

```bash
pytest                                   # unit tests, ~30 s, no Isaac Sim
pytest -m isaac tests/isaac              # Isaac Sim tests (headless)
```

## Licence and data notice

Code: Apache-2.0. Datasets keep their own licences (table above); download scripts are
provided instead of redistributing data. The NVIDIA conveyor asset is referenced from the
Isaac Sim asset server, not copied.
