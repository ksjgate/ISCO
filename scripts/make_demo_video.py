"""Demo video: Isaac Sim footage (``scripts/demo_video_isaac.py``) + detector and search
results of the demo run, with Korean captions, into ``runs/demo_video/isco_demo.mp4``.

    python scripts/make_demo_video.py scores     # per-point anomaly scores of the hero parts (~5 min, 1 core)
    python scripts/make_demo_video.py compose    # assemble the video (needs the Isaac shots and scores)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_video_isaac import FPS, TYPES, pick_instances  # noqa: E402

W, H = 1280, 720
BG = (30, 42, 50)
PANEL = (40, 55, 65)
ACCENT = (240, 165, 0)
INK = (238, 242, 245)
MUTED = (160, 174, 184)
RED = (229, 50, 45)
FONT_DIR = Path.home() / ".fonts"
KO = {"dent": "눌림", "bulge": "돌출", "hole": "구멍", "cut": "절단(모서리 깨짐)", "crack": "균열", "warp": "휨"}


# ------------------------------------------------------------------------------ scores
def cmd_scores(a) -> None:
    from isco import config_loader as C
    from isco.datagen.variants import load_manifest
    from isco.detect import pipeline as P
    from isco.detect.memory_bank import MemoryBank
    from isco.schemas.config import InspectionConfig

    rec = json.loads((Path(a.run) / "report" / "recommended.json").read_text())
    cfg = InspectionConfig.model_validate(rec["config"])
    det = C.detector("fpfh_bank").model_copy(update={"keep_point_scores": True})
    P._init(a.part, a.set_id, "sl_default", det.model_dump_json(), C.process("inline_default").model_dump_json(),
            C.noise("N2").model_dump_json())
    man = load_manifest(a.part, a.set_id)
    insts = man["instances"]
    cj = cfg.model_dump_json()
    bank = [P._work((r, cj, 0)) for r in insts if r["role"] == "bank"]
    mb = MemoryBank(det.coreset_ratio, det.coreset_max, "cuda", 0, det.bank_presample).fit(
        [o["feats"].astype(np.float32) for o in bank])
    print(f"[scores] bank of {mb.size} features from {len(bank)} normal parts", flush=True)

    def scored(r):
        o = P._work((r, cj, 0))
        s = mb.score(o["feats"].astype(np.float32))
        return o, (s[o["nbr"]].mean(axis=1) if o["nbr"] is not None else s)

    arrays, meta = {}, {"config": rec["config"], "val_image_scores": [], "parts": {}}
    vals = [r for r in insts if r["role"] == "val"][: a.n_val]
    normal_pts = []
    for j, r in enumerate(vals):
        o, s = scored(r)
        meta["val_image_scores"].append(float(s.max()))
        normal_pts.append(s)
        if j == 0:
            arrays.update({"normal_xyz": o["xyz"], "normal_label": o["label"], "normal_score": s.astype(np.float32)})
            meta["parts"]["normal"] = {"inst_id": r["inst_id"], "image_score": float(s.max())}
    meta["normal_point_p50"] = float(np.median(np.concatenate(normal_pts)))
    meta["val_image_p99"] = float(np.quantile(meta["val_image_scores"], 0.99))
    for t, r in pick_instances(man).items():
        o, s = scored(r)
        arrays.update({f"{t}_xyz": o["xyz"], f"{t}_label": o["label"], f"{t}_score": s.astype(np.float32)})
        meta["parts"][t] = {"inst_id": r["inst_id"], "image_score": float(s.max()),
                            "n_defect_points": int((o["label"] >= 2).sum())}
        print(f"[scores] {t}: {r['inst_id']} max {s.max():.2f} (val p99 {meta['val_image_p99']:.2f})", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "scores.npz", **arrays)
    (out / "scores.json").write_text(json.dumps(meta, indent=2))


# ------------------------------------------------------------------------------ drawing
def rgb01(c: tuple) -> tuple:
    return tuple(v / 255 for v in c)


def font(size: int, bold: bool = False):
    from PIL import ImageFont

    return ImageFont.truetype(str(FONT_DIR / ("NanumGothicExtraBold.ttf" if bold else "NanumGothic.ttf")), size)


def canvas():
    from PIL import Image

    return Image.new("RGB", (W, H), BG)


def caption(img, text: str, chapter: str | None = None, sub: str | None = None):
    """Lower-third caption bar + chapter tag (top left)."""
    from PIL import Image, ImageDraw

    img = img.convert("RGBA")
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    h = 92 if sub else 64
    d.rectangle([0, H - h, W, H], fill=(20, 28, 34, 215))
    d.rectangle([0, H - h, 8, H], fill=ACCENT + (255,))
    if chapter:
        f = font(22, True)
        tw = d.textlength(chapter, font=f)
        d.rounded_rectangle([24, 22, 24 + tw + 28, 60], radius=6, fill=(20, 28, 34, 215))
        d.text((38, 29), chapter, font=f, fill=ACCENT + (255,))
    img = Image.alpha_composite(img, over)
    d = ImageDraw.Draw(img)
    d.text((32, H - h + 16), text, font=font(28, True), fill=INK)
    if sub:
        d.text((32, H - h + 56), sub, font=font(21), fill=MUTED)
    return img.convert("RGB")


def text_card(lines: list[tuple[str, int, bool, tuple]], top: int = 150, left: int = 90, gap: int = 18):
    from PIL import ImageDraw

    img = canvas()
    d = ImageDraw.Draw(img)
    y = top
    for s, size, bold, color in lines:
        d.text((left, y), s, font=font(size, bold), fill=color)
        y += size + gap
    return img


def fig_to_array(fig) -> np.ndarray:
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    return a


def mpl_setup():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager

    for f in ("NanumGothic.ttf", "NanumGothicExtraBold.ttf"):
        font_manager.fontManager.addfont(str(FONT_DIR / f))
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "NanumGothic", "axes.unicode_minus": False,
                         "figure.facecolor": rgb01(BG), "axes.facecolor": rgb01(PANEL),
                         "axes.edgecolor": rgb01(MUTED), "axes.labelcolor": rgb01(INK),
                         "xtick.color": rgb01(MUTED), "ytick.color": rgb01(MUTED),
                         "text.color": rgb01(INK), "font.size": 13})
    return plt


# ------------------------------------------------------------------------------ compose
class Video:
    def __init__(self, path: Path):
        import imageio

        self.w = imageio.get_writer(str(path), fps=FPS, codec="libx264", quality=8, macro_block_size=8,
                                    ffmpeg_params=["-pix_fmt", "yuv420p"])
        self.n = 0

    def add(self, img, seconds: float | None = None, frames: int | None = None) -> None:
        a = np.asarray(img.convert("RGB") if hasattr(img, "convert") else img, dtype=np.uint8)
        for _ in range(frames if frames is not None else int(round(seconds * FPS))):
            self.w.append_data(a)
            self.n += 1

    def fade(self, a_img, b_img, seconds: float = 0.4) -> None:
        a = np.asarray(a_img, dtype=np.float32)
        b = np.asarray(b_img, dtype=np.float32)
        n = int(round(seconds * FPS))
        for i in range(n):
            u = (i + 1) / (n + 1)
            self.w.append_data(((1 - u) * a + u * b).astype(np.uint8))
            self.n += 1

    def close(self) -> None:
        self.w.close()


def clip_frames(path: Path):
    import imageio

    r = imageio.get_reader(str(path))
    for f in r:
        yield f
    r.close()


def camera_panel(cap_dir: Path):
    """Three rig views: RGB (top) and depth (bottom)."""
    import matplotlib.cm as cm
    from PIL import Image, ImageDraw

    info = json.loads((cap_dir / "capture.json").read_text())
    img = canvas()
    d = ImageDraw.Draw(img)
    size, gap = 300, 26
    x0 = (W - 3 * size - 2 * gap) // 2
    for i, v in enumerate(info["views"]):
        rgb = Image.open(cap_dir / f"cam_{i}_rgb.png").convert("RGB")
        depth = np.load(cap_dir / f"cam_{i}_depth.npy").astype(np.float64)
        sem_p = cap_dir / f"cam_{i}_sem.npy"
        ok = np.isfinite(depth) & (depth > 0)
        part = ok
        if sem_p.exists():
            sem = np.load(sem_p)
            labels = json.loads((cap_dir / f"cam_{i}_sem_labels.json").read_text())
            ids = [int(k) for k, lab in labels.items() if lab.get("class") in ("part", "defect")]
            if sem.shape == depth.shape and ids:
                part = ok & np.isin(sem, ids)
        lo, hi = np.quantile(depth[part], [0.01, 0.99]) if part.any() else (depth[ok].min(), depth[ok].max())
        z = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
        col = (cm.get_cmap("turbo")(1 - z)[:, :, :3] * 255).astype(np.uint8)
        col[~ok] = 0
        col[ok & ~part] = (col[ok & ~part] * 0.35).astype(np.uint8)
        dimg = Image.fromarray(col)
        x = x0 + i * (size + gap)
        img.paste(rgb.resize((size, size)), (x, 92))
        img.paste(dimg.resize((size, size)), (x, 92 + size + 18))
        lab = f"카메라 {i + 1} · 방위 {v['azimuth_deg']:.0f}° · 고도 {v['elevation_deg']:.0f}°"
        d.text((x, 60), lab, font=font(19, True), fill=INK)
    d.text((x0 - 8 - 70, 92 + size // 2 - 12), "컬러", font=font(18), fill=MUTED)
    d.text((x0 - 8 - 70, 92 + size + 18 + size // 2 - 12), "깊이", font=font(18), fill=MUTED)
    return img


def pointcloud_frames(scores_npz: Path, meta: dict, t: str, n: int):
    plt = mpl_setup()
    from matplotlib.colors import LinearSegmentedColormap

    z = np.load(scores_npz)
    xyz, lab, s = z[f"{t}_xyz"] * 1e3, z[f"{t}_label"], z[f"{t}_score"]
    lo, hi = meta["normal_point_p50"], meta["val_image_p99"]
    cmap = LinearSegmentedColormap.from_list("isco", ["#5b6b78", "#9fb0bb", "#f0a500", "#e5322d"])
    sn = np.clip((s - lo) / max(hi - lo, 1e-9), 0, 1)
    gt = np.where(lab >= 2, 1.0, 0.0)
    rng = np.random.default_rng(0)
    keep = rng.permutation(len(xyz))[:16000]
    keep = np.union1d(keep, np.flatnonzero(lab >= 2))
    P, G, S = xyz[keep], gt[keep], sn[keep]
    o_g, o_s = np.argsort(G), np.argsort(S)
    c = (P.min(axis=0) + P.max(axis=0)) / 2
    rad = 0.5 * np.ptp(P[:, :2], axis=0).max() * 1.04
    for f in range(n):
        fig = plt.figure(figsize=(W / 100, (H - 150) / 100), dpi=100)
        az = -60 + 90 * f / max(n - 1, 1)
        for k, (vals, order, title, cm_) in enumerate((
                (G, o_g, "정답: 시뮬레이터가 넣은 결함 위치", LinearSegmentedColormap.from_list("gt", ["#5b6b78", "#e5322d"])),
                (S, o_s, "검출기: 점마다 계산한 이상 점수", cmap))):
            ax = fig.add_subplot(1, 2, k + 1, projection="3d")
            ax.set_facecolor(rgb01(BG))
            ax.scatter(P[order, 0], P[order, 1], P[order, 2], c=vals[order], cmap=cm_, vmin=0, vmax=1, s=2.6,
                       linewidths=0, depthshade=False)
            ax.set_xlim(c[0] - rad, c[0] + rad)
            ax.set_ylim(c[1] - rad, c[1] + rad)
            ax.set_zlim(c[2] - rad * 0.45, c[2] + rad * 0.45)
            ax.set_box_aspect((1, 1, 0.45), zoom=1.25)
            ax.view_init(elev=38, azim=az)
            ax.set_axis_off()
            ax.set_title(title, color=rgb01(INK), fontsize=17, pad=2)
        fig.subplots_adjust(left=0, right=1, top=0.94, bottom=0, wspace=0)
        a = fig_to_array(fig)
        plt.close(fig)
        img = canvas()
        from PIL import Image

        img.paste(Image.fromarray(a), (0, 40))
        yield img


def search_frames(run: Path, n_stage1: int, n_stage2: int):
    plt = mpl_setup()
    s1 = [json.loads(line) for line in open(run / "stage1.jsonl")]
    s2 = [json.loads(line) for line in open(run / "stage2.jsonl")]
    rec = json.loads((run / "report" / "recommended.json").read_text())
    v1 = np.array([r["surrogate"]["V_area"] for r in s1])
    t1 = np.array([r["objectives"]["cycle_time_s"] for r in s1])
    n1 = np.array([r["n_cam"] for r in s1])
    rng = np.random.default_rng(0)
    jit = rng.uniform(-0.06, 0.06, len(t1))
    order = rng.permutation(len(s1))
    shortlist = {r["config_id"] for r in s2}
    cols = ["#7fb3d5", "#5dade2", "#f0a500", "#e67e22", "#e5322d", "#c39bd3"]
    f2 = np.array([r["objectives_report"]["fnr"] for r in s2])
    t2 = np.array([r["objectives_report"]["cycle_time_s"] for r in s2])
    n2 = np.array([r["n_cam"] for r in s2])

    def frame(k1: int, k2: int, final: bool):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(W / 100, (H - 150) / 100), dpi=100)
        idx = order[:k1]
        for n in range(1, 7):
            m = idx[n1[idx] == n]
            a1.scatter(t1[m] + jit[m], v1[m], s=7, color=cols[n - 1], alpha=0.75, linewidths=0, label=f"{n}대")
        if k2 > 0:
            sl = [i for i in range(len(s1)) if s1[i]["config_id"] in shortlist]
            a1.scatter(t1[sl] + jit[sl], v1[sl], s=60, facecolors="none", edgecolors="white", linewidths=1.2)
        a1.set_xlabel("사이클 시간 (초)")
        a1.set_ylabel("결함 표면이 보이는 비율")
        a1.set_title(f"1단계: 가시성으로 빠르게 거르기 ({k1:,}/{len(s1):,}개)", fontsize=15)
        a1.set_ylim(0, 1)
        a1.set_xlim(0, 5)
        leg = a1.legend(title="카메라", fontsize=10, title_fontsize=10, loc="lower right", markerscale=2,
                        facecolor=rgb01(PANEL), edgecolor="none")
        for txt in leg.get_texts() + [leg.get_title()]:
            txt.set_color(rgb01(INK))
        a2.set_xlim(0, 5)
        a2.set_ylim(0, 1.05)
        a2.set_xlabel("사이클 시간 (초)")
        a2.set_ylabel("놓친 결함 비율 (과검 1% 기준)")
        a2.set_title(f"2단계: 검출기로 실제 평가 ({k2}/{len(s2)}개)", fontsize=15)
        if k2 > 0:
            m = np.arange(k2)
            a2.scatter(t2[m], f2[m], s=40 + 25 * n2[m], c=[cols[n - 1] for n in n2[m]], edgecolors="white",
                       linewidths=0.6, zorder=3)
        if final:
            a2.scatter([rec["cycle_time_s"]], [rec["fnr_report"]], s=420, marker="*", color=rgb01(ACCENT),
                       edgecolors="white", linewidths=1.2, zorder=5)
            a2.annotate("권장: 카메라 3대, 회전 없음\n0.6 m/s, 사이클 0.5초", (rec["cycle_time_s"], rec["fnr_report"]),
                        xytext=(2.0, 0.12), color=rgb01(INK), fontsize=13,
                        arrowprops=dict(arrowstyle="->", color=rgb01(ACCENT)))
        for ax in (a1, a2):
            ax.grid(alpha=0.15)
        fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.12, wspace=0.22)
        a = fig_to_array(fig)
        plt.close(fig)
        from PIL import Image

        img = canvas()
        img.paste(Image.fromarray(a), (0, 30))
        return img

    for f in range(n_stage1):
        yield frame(int(len(s1) * (f + 1) / n_stage1), 0, False)
    per = max(1, n_stage2 // len(s2))
    for k in range(1, len(s2) + 1):
        img = frame(len(s1), k, False)
        for _ in range(per):
            yield img
    yield frame(len(s1), len(s2), True)


def cmd_compose(a) -> None:
    from PIL import Image

    run, iso = Path(a.run), Path(a.out) / "isaac"
    rec = json.loads((run / "report" / "recommended.json").read_text())
    meta = json.loads((Path(a.out) / "scores.json").read_text())
    dmeta = json.loads((iso / "defects.json").read_text())
    base = json.loads((REPO / "runs" / "demo_assets" / "baselines.json").read_text()) \
        if (REPO / "runs" / "demo_assets" / "baselines.json").exists() else None
    v = Video(Path(a.out) / "isco_demo.mp4")

    title = text_card([("2026-2 산학캡스톤디자인 · 팀 강화실패", 24, True, ACCENT),
                       ("ISCO 시뮬레이터 데모", 64, True, INK),
                       ("부품이 들어오면, 그 부품을 가장 적은 비용으로 잘 검사하는", 30, False, INK),
                       ("카메라 배치·회전·컨베이어 속도를 찾아 주는 가상 공정", 30, False, INK),
                       ("", 10, False, INK),
                       ("Isaac Sim 5.1 컨베이어 디지털 트윈 · 브래킷 부품(40 mm) 예시", 22, False, MUTED)], top=170)
    v.add(title, 3.5)

    problem = text_card([("라인 설계자가 정해야 하는 것", 40, True, INK),
                         ("카메라를 몇 대 둘 것인가 · 어느 각도와 거리에 둘 것인가", 28, False, INK),
                         ("부품을 돌려서 다시 찍을 것인가 · 컨베이어를 얼마나 빨리 돌릴 것인가", 28, False, INK),
                         ("", 10, False, INK),
                         ("ISCO는 이 조합을 시뮬레이터 안에서 직접 시험해서", 28, False, ACCENT),
                         ("\"정상품 오판정 1%일 때 놓치는 불량 비율\"과 비용을 함께 보고 고른다", 28, False, ACCENT)],
                        top=190)
    v.fade(title, problem)
    v.add(problem, 4.5)

    # 1. the line
    first = None
    for f in clip_frames(iso / "line.mp4"):
        img = caption(Image.fromarray(f), "가상 공정: 부품이 컨베이어를 따라 0.3 m/s로 흘러간다",
                      "1  가상 공정 (Isaac Sim)",
                      "파란 사각뿔 = 탐색이 고른 카메라 3대 위치 · 빨간 부분 = 시뮬레이터가 넣은 결함(설명용 색)")
        if first is None:
            first = img
            v.fade(problem, img)
        v.add(img, frames=1)
    # 2. defect generator
    for t in TYPES:
        d = dmeta[t]["defect"]
        size = []
        if d.get("diameter_m"):
            size.append(f"지름 {d['diameter_m'] * 1e3:.1f} mm")
        if d.get("length_m"):
            size.append(f"길이 {d['length_m'] * 1e3:.1f} mm")
        if d.get("width_m"):
            size.append(f"폭 {d['width_m'] * 1e3:.2f} mm")
        if d.get("depth_m"):
            size.append(("처짐 " if t == "warp" else "깊이 ") + f"{d['depth_m'] * 1e3:.2f} mm")
        for f in clip_frames(iso / f"defect_{t}.mp4"):
            v.add(caption(Image.fromarray(f), f"결함 생성기: {KO[t]}  ({', '.join(size)})", "2  결함 만들기",
                          "부품 형상(메시)을 직접 깎거나 눌러서 6종 결함을 만들고, 결함 위치는 자동으로 정답 표시"), frames=1)
    # 3. inspection moment
    last = None
    for f in clip_frames(iso / "inspect.mp4"):
        last = caption(Image.fromarray(f), "검사 지점: 부품이 카메라 아래를 지나는 순간 3대가 동시에 촬영",
                       "3  촬영", "벨트 속도·트리거 지연·노출 중 움직임은 센서 모델로 반영")
        v.add(last, frames=1)
    flash = Image.blend(last, Image.new("RGB", (W, H), (255, 255, 255)), 0.7)
    v.add(flash, frames=3)
    cams = caption(camera_panel(iso / "capture"), "각 카메라가 얻는 것: 컬러 영상 + 깊이 영상(가까울수록 붉게)",
                   "3  촬영", "실제 계산에는 깊이만 쓰고, 측정 잡음·결측·가장자리 튐을 센서 모델로 덧입힌다")
    v.fade(flash, cams, 0.3)
    v.add(cams, 4.5)
    # 4. detection
    t = "cut"
    pc_first = None
    for img in pointcloud_frames(Path(a.out) / "scores.npz", meta, t, n=int(5.5 * FPS)):
        img = caption(img, "검출: 깊이 영상들을 합친 3D 점군에서, 정상품만 배운 검출기가 이상 부위를 찾는다",
                      "4  검출", f"정상품 40개로 기억장치(메모리 뱅크)를 만들고, 처음 보는 결함은 '정상과 다른 정도'로 점수화 · 예: {KO[t]} 결함")
        if pc_first is None:
            pc_first = img
            v.fade(cams, img)
        v.add(img, frames=1)
    # 5. search
    s_first = None
    for img in search_frames(run, n_stage1=int(3 * FPS), n_stage2=int(4 * FPS)):
        img = caption(img, "탐색: 후보 1,355개를 가시성으로 거르고, 유망한 20개만 검출기로 실제 평가",
                      "5  설정 탐색", "점 색·크기 = 카메라 수 · 흰 테두리 = 2단계로 넘어간 후보 · 2단계 점수는 탐색에 쓰지 않은 결함으로 매김")
        if s_first is None:
            s_first = img
            v.fade(pc_first, img)
        v.add(img, frames=1)
    v.add(img, 3.0)
    # 6. result
    lines = [("결과: 브래킷 부품의 권장 검사 설정 (제약: 카메라 3대 이하, 사이클 1.5초 이하)", 30, True, INK),
             ("", 6, False, INK),
             (f"카메라 3대 (방위 240°·330° 고도 40°, 방위 270° 고도 80°, 거리 0.25 m) · 회전 없음 · 0.6 m/s", 25, False, INK),
             (f"놓친 결함 비율 {rec['fnr_report'] * 100:.0f}%  (95% 신뢰구간 {rec['fnr_report_ci'][0] * 100:.0f}–"
              f"{rec['fnr_report_ci'][1] * 100:.0f}%)  ·  사이클 {rec['cycle_time_s']:.1f}초", 25, True, ACCENT),
             ("", 6, False, INK)]
    if base:
        rows = {r["rig"]: r for r in base if r["part"] == a.part}
        names = (("top_1cam", "위에서 1대"), ("ring3_el40", "고도 40° 등간격 3대"), ("ring6_el40", "등간격 6대"))
        lines.append(("비교: 사람이 정한 기본 배치 (0.3 m/s, 사이클 1.0초) — " + " · ".join(
            f"{ko} {rows[k]['fnr_report'] * 100:.0f}%" for k, ko in names if k in rows), 22, False, MUTED))
        lines.append(("→ 권장 설정은 등간격 3대와 비슷한 미검율을 벨트 속도 2배(사이클 절반)에서 낸다", 22, False, MUTED))
    lines += [("결함 종류별로 놓친 비율: " + " · ".join(f"{KO[k].split('(')[0]} {rec['fnr_by_type_report'][k] * 100:.0f}%"
                                              for k in TYPES), 22, False, MUTED),
              ("같은 사이클에서 카메라 1대(고도 60°)도 28%로 신뢰구간이 겹침 → 카메라 수를 줄일 여지", 22, False, MUTED)]
    res = text_card(lines, top=150, gap=16)
    v.fade(img, res)
    v.add(res, 6.0)
    end = text_card([("지금 진행 중", 40, True, INK),
                     ("검증 실험을 사전에 정한 규칙대로 야간 대기열에서 실행 중 (6개 부품, 총 약 65–70시간)", 25, False, INK),
                     ("· 부품마다 최적 설정이 정말 다른가  · 가시성만으로 골라도 되는가", 24, False, MUTED),
                     ("· 카메라를 늘리면 항상 좋아지는가  · 공정 속도를 넣으면 답이 바뀌는가", 24, False, MUTED),
                     ("· 실제 스캔한 결함에서도 순위가 유지되는가", 24, False, MUTED),
                     ("", 10, False, INK),
                     ("ISCO · Inspection Setup Configuration Optimizer", 22, True, ACCENT)], top=170)
    v.fade(res, end)
    v.add(end, 4.5)
    v.close()
    print(f"[make_demo_video] {Path(a.out) / 'isco_demo.mp4'}: {v.n} frames ({v.n / FPS:.1f}s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scores", "compose"])
    ap.add_argument("--part", default="fusion_51876")
    ap.add_argument("--set-id", default="mvp")
    ap.add_argument("--run", default=str(REPO / "runs" / "demo_fusion_51876"))
    ap.add_argument("--out", default=str(REPO / "runs" / "demo_video"))
    ap.add_argument("--n-val", type=int, default=40)
    a = ap.parse_args()
    {"scores": cmd_scores, "compose": cmd_compose}[a.cmd](a)


if __name__ == "__main__":
    main()
