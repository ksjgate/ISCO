"""Figures and data for the mid-term report (``docs/report.md``) and its web page.

* ``docs/report_assets/*.png``   - Isaac stills, detector view and Korean charts for the report
* ``runs/report_web/``           - the same images (JPEG), a web-sized demo video and
  ``data.json`` for the page's configuration explorer: the 96 camera slots, which
  search-set defect samples each slot sees (bit-packed, the stage-1 visibility rule), the
  cost model and every configuration of the demo run that was evaluated with the detector.

    python scripts/make_report_assets.py      # after scripts/make_demo_video.py
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_video_isaac import TYPES  # noqa: E402

KO = {"dent": "눌림", "bulge": "돌출", "hole": "구멍", "cut": "절단", "crack": "균열", "warp": "휨"}
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]   # camera count 1..6
INK, MUTED, GRID = "#1d252b", "#5b6670", "#e3e7ea"


def mpl():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager

    for f in ("NanumGothic.ttf", "NanumGothicExtraBold.ttf"):
        font_manager.fontManager.addfont(str(Path.home() / ".fonts" / f))
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "NanumGothic", "axes.unicode_minus": False, "font.size": 11,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white"})
    return plt


def frame(path: Path, i: int) -> np.ndarray:
    import imageio

    r = imageio.get_reader(str(path))
    n = r.count_frames()
    f = r.get_data(min(i, n - 1))
    r.close()
    return f


def stills(vid: Path, out: Path) -> None:
    from PIL import Image

    sys.path.insert(0, str(REPO / "scripts"))
    import make_demo_video as M

    iso = vid / "isaac"
    Image.fromarray(frame(iso / "line.mp4", 120)).save(out / "isaac_line.png")
    Image.fromarray(frame(iso / "inspect.mp4", 48)).save(out / "isaac_inspect.png")
    M.camera_panel(iso / "capture").save(out / "capture_views.png")
    tiles = [Image.fromarray(frame(iso / f"defect_{t}.mp4", 36)).resize((640, 360)) for t in TYPES]
    from PIL import ImageDraw

    sheet = Image.new("RGB", (1920, 720), "white")
    for k, (t, im) in enumerate(zip(TYPES, tiles)):
        d = ImageDraw.Draw(im)
        d.rounded_rectangle([14, 14, 130, 56], radius=8, fill=(20, 28, 34))
        d.text((28, 20), KO[t], font=M.font(26, True), fill=(240, 165, 0))
        sheet.paste(im, ((k % 3) * 640, (k // 3) * 360))
    sheet.save(out / "defects.png")
    meta = json.loads((vid / "scores.json").read_text())
    img = next(iter(M.pointcloud_frames(vid / "scores.npz", meta, "cut", 1)))
    img.save(out / "detection.png")


def charts(out: Path, run: Path) -> None:
    plt = mpl()
    s2 = [json.loads(line) for line in open(run / "stage2.jsonl")]
    rec = json.loads((run / "report" / "recommended.json").read_text())
    base = [b for b in json.loads((REPO / "runs" / "demo_assets" / "baselines.json").read_text())
            if b["part"] == "fusion_51876"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    for n in range(1, 7):
        rows = [r for r in s2 if r["n_cam"] == n]
        if rows:
            ax.scatter([r["objectives_report"]["cycle_time_s"] for r in rows],
                       [r["objectives_report"]["fnr"] for r in rows], s=28 + 14 * n, color=RAMP[n - 1],
                       edgecolors="white", linewidths=0.8, label=f"{n}대", zorder=3)
    names = {"top_1cam": "위 1대", "ring3_el40": "등간격 3대", "ring6_el40": "등간격 6대"}
    for b in base:
        ax.scatter([b["cycle_time_s"]], [b["fnr_report"]], s=60, facecolors="none", edgecolors=ORANGE,
                   linewidths=1.4, zorder=4)
        ax.annotate(names[b["rig"]], (b["cycle_time_s"], b["fnr_report"]), xytext=(8, -3),
                    textcoords="offset points", fontsize=9, color=ORANGE)
    ax.scatter([rec["cycle_time_s"]], [rec["fnr_report"]], s=260, marker="*", color="#eda100", edgecolors=INK,
               linewidths=0.8, zorder=5)
    ax.annotate("권장 설정", (rec["cycle_time_s"], rec["fnr_report"]), xytext=(10, 8), textcoords="offset points",
                fontsize=10, color=INK)
    ax.set_xlabel("사이클 시간 (초)")
    ax.set_ylabel("놓친 결함 비율 (과검 1% 기준)")
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1.05)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    leg = ax.legend(title="카메라 수 (탐색 평가)", fontsize=9, title_fontsize=9, loc="upper right", frameon=False,
                    ncol=2)
    leg.get_title().set_color(INK)
    ax.set_title("브래킷 부품: 검출기로 평가한 설정 20개 + 사람이 정한 기본 배치 3개(주황 테두리)", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out / "pareto.png")
    plt.close(fig)

    # camera count vs FNR, per detector (fixed ring rigs, inline 0.3 m/s, noise level N2)
    cams = [1, 2, 3, 4, 6]
    cad = {"브래킷": [0.31, 0.25, 0.33, 0.22, 0.25], "막대": [0.44, 0.42, 0.39, 0.42, 0.36], "얇은 판": [0.33, 0.33, 0.31, 0.36, 0.36]}
    bank = {"브래킷": [0.36, 0.25, 0.31, 0.25, 0.22], "막대": [0.89, 1.00, 0.83, 0.64, 0.72], "얇은 판": [0.50, 0.58, 0.69, 0.47, 0.78]}
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.6), dpi=150, sharey=True)
    for ax, data, title in ((axs[0], bank, "메모리 뱅크 검출기 (주 검출기)"), (axs[1], cad, "설계도 거리 비교 (단순 기준)")):
        for (name, ys), col, mk in zip(data.items(), (BLUE, ORANGE, AQUA), ("o", "s", "^")):
            ax.plot(cams, ys, color=col, lw=2, marker=mk, ms=6, label=name)
            if data is bank:                    # end labels only where the lines end apart
                ax.annotate(name, (cams[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", fontsize=9,
                            color=col, va="center")
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xticks(cams)
        ax.set_xlabel("카메라 수 (고도 40° 등간격)")
        ax.set_xlim(0.6, 7.2)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axs[0].set_ylabel("놓친 결함 비율")
    axs[0].set_ylim(0, 1.05)
    axs[1].legend(fontsize=9, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "cameras_vs_fnr.png")
    plt.close(fig)

    # the sensor-model fix (design-distance detector, bar part)
    fig, ax = plt.subplots(figsize=(5.6, 3.5), dpi=150)
    c3 = [1, 3, 6]
    for ys, col, mk, lab in (([0.39, 0.72, 0.92], ORANGE, "s", "수정 전 (정지, 잡음 '중')"),
                             ([0.39, 0.31, 0.28], BLUE, "o", "수정 후 (정지, 잡음 '중')"),
                             ([0.31, 0.31, 0.22], AQUA, "^", "이상 조건 (잡음 없음, 정지)")):
        ax.plot(c3, ys, color=col, lw=2, marker=mk, ms=6, label=lab)
    ax.set_xticks(c3)
    ax.set_xlabel("카메라 수")
    ax.set_ylabel("놓친 결함 비율")
    ax.set_ylim(0, 1.0)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.set_title("막대 부품, 설계도 거리 비교: 센서 모델 수정 전후", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out / "sensor_fix.png")
    plt.close(fig)


def explorer_data(part: str, set_id: str, run: Path) -> dict:
    from isco import config_loader as C
    from isco.datagen.parts import load_part
    from isco.geometry.camera_model import SlotGrid
    from isco.optim.space import SearchSpace
    from isco.render.occluder import Occluder
    from isco.render.slot_cache import SlotCache
    from isco.schemas.config import InspectionConfig
    from isco.schemas.process import ROTATION_STAGES
    from isco.visibility.surrogate import VisibilityTable

    spec = load_part(part)
    cache = SlotCache.open(part, set_id, "sl_default")
    grid: SlotGrid = cache.grid
    vt = VisibilityTable.for_set(part, set_id)
    sel = np.isin(vt.role, ["search"])
    ok = vt.ok[sel]                                         # (J, K, S)
    J, K, S = ok.shape
    sp = SearchSpace(grid, np.asarray(spec.focus_m), Occluder.default(), None, cache.sensor.working_range_m)
    proc = C.process("inline_default")
    slots = []
    for k in range(grid.n_slots):
        s = grid.slot(k)
        e = grid.eye(k, np.asarray(spec.focus_m))
        slots.append({"k": k, "az": s.azimuth_deg, "el": s.elevation_deg, "d": s.distance_m,
                      "eye": [round(float(x), 4) for x in e], "valid": k in sp.valid_slots,
                      "why": sp.invalid_reason.get(k) if hasattr(sp, "invalid_reason") else None})
    evaluated = []
    for line in open(run / "stage2.jsonl"):
        r = json.loads(line)
        evaluated.append({"slots": r["config"]["slots"], "rotation": r["config"]["rotation"],
                          "speed": r["config"]["speed_level"], "trigger": r["config"]["trigger_mode"],
                          "fnr": r["objectives_report"]["fnr"], "ci": r["report"]["fnr_ci"], "auroc": r["report"]["auroc"],
                          "cycle": r["objectives_report"]["cycle_time_s"], "V_area": r["surrogate"]["V_area"],
                          "by_type": r["report"]["fnr_by_type"], "source": "탐색 평가"})
    names = {"top_1cam": "위에서 1대", "ring3_el40": "고도 40° 등간격 3대", "ring6_el40": "고도 40° 등간격 6대"}
    for b in json.loads((REPO / "runs" / "demo_assets" / "baselines.json").read_text()):
        if b["part"] == part:
            evaluated.append({"slots": b["slots"], "rotation": "none", "speed": 1, "trigger": "simultaneous",
                              "fnr": b["fnr_report"], "ci": b["fnr_ci"], "auroc": b["auroc"], "cycle": b["cycle_time_s"],
                              "V_area": b["V_area"], "by_type": b["fnr_by_type"], "source": names[b["rig"]]})
    rec = json.loads((run / "report" / "recommended.json").read_text())
    # reference values for checking the page's visibility arithmetic
    checks = []
    for cfg in (rec["config"], {"slots": [36], "rotation": "none", "speed_level": 1},
                {"slots": [25], "rotation": "r90x3", "speed_level": 2}):
        c = InspectionConfig.model_validate(cfg)
        checks.append({"slots": c.slots, "rotation": c.rotation, "V_area": vt.metrics(c)["V_area"]})
    bits = np.packbits(ok.reshape(-1).astype(np.uint8))
    return {"part": part, "extents_mm": [round(x * 1e3, 1) for x in spec.extents_m], "focus": spec.focus_m,
            "grid": grid.to_dict(), "slots": slots, "min_angle_deg": 30.0,
            "rotation_stages": {k: list(v) for k, v in ROTATION_STAGES.items()},
            "rotation_time_s": proc.rotation_time_s, "speeds_mps": proc.speeds_mps, "pitch_m": proc.pitch_m,
            "defects": {"n": J, "samples": S, "types": [str(t) for t in vt.type[sel]]},
            "seen_bits": base64.b64encode(bits.tobytes()).decode(), "seen_shape": [J, K, S],
            "evaluated": evaluated, "recommended": rec, "checks": checks}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="fusion_51876")
    ap.add_argument("--set-id", default="mvp")
    ap.add_argument("--run", default=str(REPO / "runs" / "demo_fusion_51876"))
    ap.add_argument("--video", default=str(REPO / "runs" / "demo_video"))
    a = ap.parse_args()
    docs = REPO / "docs" / "report_assets"
    web = REPO / "runs" / "report_web"
    docs.mkdir(parents=True, exist_ok=True)
    web.mkdir(parents=True, exist_ok=True)
    stills(Path(a.video), docs)
    charts(docs, Path(a.run))
    from PIL import Image

    for p in docs.glob("*.png"):
        im = Image.open(p).convert("RGB")
        if im.width > 1600:
            im = im.resize((1600, round(im.height * 1600 / im.width)))
        im.save(web / (p.stem + ".jpg"), quality=84)
    (web / "data.json").write_text(json.dumps(explorer_data(a.part, a.set_id, Path(a.run))))
    import imageio_ffmpeg

    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i",
                    str(Path(a.video) / "isco_demo.mp4"), "-c:v", "libx264", "-crf", "27", "-preset", "slow",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(web / "isco_demo.mp4")], check=True)
    for p in sorted(web.iterdir()):
        print(f"{p.relative_to(REPO)}  {p.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
