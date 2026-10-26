"""Mid-term demo video: Isaac Sim footage (``isaac_shots.py``), per-product detection
(``scores.py``), the search of one product (``search_data.py``) and the experiment figures
(``figures.py``/``assets.py``, dark theme), with Korean captions.

    python scripts/midterm/video.py   -> runs/midterm_reports/ISCO_중간발표_데모영상.mp4
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import KO_PART, KO_ROT, KO_TYPE, PARTS, SPEEDS, TYPES, WORK, best_record  # noqa: E402

W, H, FPS = 1280, 720, 30
BG = (26, 26, 25)            # dark chart surface #1a1a19
PANEL = (44, 44, 42)
INK = (240, 239, 236)
INK2 = (195, 194, 183)
MUTED = (137, 135, 129)
ACCENT = (237, 161, 0)
GOOD = (12, 163, 12)
RED = (230, 103, 103)
SERIES = [(57, 135, 229), (25, 158, 112), (201, 133, 0)]
FONT_DIR = Path.home() / ".fonts"
ISAAC = WORK / "isaac"
FIG = WORK / "figures" / "dark"


# ------------------------------------------------------------------------------ drawing
def font(size: int, bold: bool = False):
    from PIL import ImageFont

    return ImageFont.truetype(str(FONT_DIR / ("NanumGothicExtraBold.ttf" if bold else "NanumGothic.ttf")), size)


def canvas(color=BG):
    from PIL import Image

    return Image.new("RGB", (W, H), color)


def caption(img, text: str, chapter: str | None = None, sub: str | None = None):
    """Lower-third caption bar + chapter tag (top left)."""
    from PIL import Image, ImageDraw

    img = img.convert("RGBA")
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    h = 94 if sub else 66
    d.rectangle([0, H - h, W, H], fill=(14, 14, 13, 225))
    d.rectangle([0, H - h, 8, H], fill=ACCENT + (255,))
    if chapter:
        f = font(21, True)
        tw = d.textlength(chapter, font=f)
        d.rounded_rectangle([24, 20, 24 + tw + 28, 58], radius=6, fill=(14, 14, 13, 215))
        d.text((38, 27), chapter, font=f, fill=ACCENT + (255,))
    img = Image.alpha_composite(img, over)
    d = ImageDraw.Draw(img)
    d.text((32, H - h + 15), text, font=font(27, True), fill=INK)
    if sub:
        d.text((32, H - h + 56), sub, font=font(20), fill=INK2)
    return img.convert("RGB")


def info_box(img, lines: list[tuple[str, int, bool, tuple]], xy=(24, 74), pad=16):
    """Semi-transparent text panel (top left) over footage."""
    from PIL import Image, ImageDraw

    img = img.convert("RGBA")
    d0 = ImageDraw.Draw(img)
    wmax = max(d0.textlength(s, font=font(sz, b)) for s, sz, b, _ in lines)
    hsum = sum(sz + 10 for _, sz, _, _ in lines)
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    x, y = xy
    d.rounded_rectangle([x, y, x + wmax + 2 * pad, y + hsum + 2 * pad - 6], radius=10, fill=(14, 14, 13, 200))
    img = Image.alpha_composite(img, over)
    d = ImageDraw.Draw(img)
    yy = y + pad
    for s, sz, b, col in lines:
        d.text((x + pad, yy), s, font=font(sz, b), fill=col)
        yy += sz + 10
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


def fit(im, w: int, h: int):
    """Resize keeping the aspect ratio to fit inside w x h."""
    r = min(w / im.width, h / im.height)
    return im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))))


def figure_card(name: str, title: str | None = None):
    from PIL import Image

    img = canvas()
    fig = fit(Image.open(FIG / f"{name}.png").convert("RGB"), W - 60, H - 110 - (40 if title else 0))
    img.paste(fig, ((W - fig.width) // 2, 20 + (40 if title else 0)))
    if title:
        from PIL import ImageDraw

        ImageDraw.Draw(img).text((40, 18), title, font=font(24, True), fill=INK)
    return img


def mpl():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager

    for f in ("NanumGothic.ttf", "NanumGothicExtraBold.ttf"):
        font_manager.fontManager.addfont(str(FONT_DIR / f))
    import matplotlib.pyplot as plt

    to01 = lambda c: tuple(v / 255 for v in c)  # noqa: E731
    plt.rcParams.update({"font.family": "NanumGothic", "axes.unicode_minus": False, "figure.facecolor": to01(BG),
                         "axes.facecolor": to01(BG), "axes.edgecolor": to01(MUTED), "axes.labelcolor": to01(INK2),
                         "xtick.color": to01(MUTED), "ytick.color": to01(MUTED), "text.color": to01(INK),
                         "font.size": 13, "axes.spines.top": False, "axes.spines.right": False})
    return plt, to01


def fig_array(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()


# ------------------------------------------------------------------------------ writer
class Video:
    def __init__(self, path: Path):
        import imageio

        self.w = imageio.get_writer(str(path), fps=FPS, codec="libx264", quality=8, macro_block_size=8,
                                    ffmpeg_params=["-pix_fmt", "yuv420p"])
        self.n = 0
        self.last = None

    def add(self, img, seconds: float | None = None, frames: int | None = None) -> None:
        a = np.asarray(img.convert("RGB") if hasattr(img, "convert") else img, dtype=np.uint8)
        for _ in range(frames if frames is not None else int(round(seconds * FPS))):
            self.w.append_data(a)
            self.n += 1
        self.last = a

    def fade_to(self, img, seconds: float = 0.4) -> None:
        if self.last is None:
            return
        a = self.last.astype(np.float32)
        b = np.asarray(img.convert("RGB") if hasattr(img, "convert") else img, dtype=np.float32)
        n = int(round(seconds * FPS))
        for i in range(n):
            u = (i + 1) / (n + 1)
            self.w.append_data(((1 - u) * a + u * b).astype(np.uint8))
            self.n += 1

    def close(self) -> None:
        self.w.close()


def clip(path: Path):
    import imageio

    r = imageio.get_reader(str(path))
    for f in r:
        yield f
    r.close()


def flash(img, k: float = 0.65):
    from PIL import Image

    return Image.blend(img, Image.new("RGB", (W, H), (255, 255, 255)), k)


# ------------------------------------------------------------------------------ pieces
def lineup_frames():
    from PIL import Image, ImageDraw

    meta = json.loads((ISAAC / "lineup.json").read_text())
    short = {"tless_13": "T-LESS 13번\n전구 소켓", "tless_18": "T-LESS 18번\n원뿔형 하우징",
             "tless_29": "T-LESS 29번\n사각 전기 박스", "tless_30": "T-LESS 30번\n원형 전기 박스",
             "mulsen_nut": "너트\n(실제 스캔)", "mulsen_flat_pad": "평판 패드\n(실제 스캔)"}
    for f, anchors in zip(clip(ISAAC / "lineup.mp4"), meta["anchors"]):
        img = Image.fromarray(f)
        d = ImageDraw.Draw(img)
        for p, a in zip(meta["parts"], anchors):
            if a is None or not (40 < a[0] < W - 40):
                continue
            x, y = a[0], max(70, a[1] - 92)
            for i, line in enumerate(short[p].split("\n")):
                fnt = font(22 if i == 0 else 18, i == 0)
                tw = d.textlength(line, font=fnt)
                d.rounded_rectangle([x - tw / 2 - 8, y - 4 + i * 28, x + tw / 2 + 8, y + 24 + i * 28], radius=5,
                                    fill=(14, 14, 13))
                d.text((x - tw / 2, y + i * 28), line, font=fnt, fill=INK if i == 0 else INK2)
        yield img


def camera_panel(pid: str):
    """Rig views of the demo part: per rotation stage, RGB and depth of every camera."""
    import matplotlib
    from PIL import Image, ImageDraw

    cdir = ISAAC / f"capture_{pid}"
    info = json.loads((cdir / "capture.json").read_text())
    stages = sorted({v["stage"] for v in info["views"]})
    cams = sorted({v["cam"] for v in info["views"]})
    cols, rows = 2 * len(cams), len(stages)
    gap, pair_gap, lab_h = 8, 34, 56
    size = int(min((W - 100 - gap * len(cams) - pair_gap * (len(cams) - 1)) / cols,
                   (H - 190 - lab_h * rows) / rows))
    img = canvas()
    d = ImageDraw.Draw(img)
    x0 = (W - cols * size - gap * len(cams) - pair_gap * (len(cams) - 1)) // 2
    y0 = max(70, (H - 100 - rows * (size + lab_h) - 30) // 2)
    for v in info["views"]:
        s, i = v["stage"], v["cam"]
        rgb = Image.open(cdir / f"s{s}_cam{i}_rgb.png").convert("RGB")
        depth = np.load(cdir / f"s{s}_cam{i}_depth.npy").astype(np.float64)
        ok = np.isfinite(depth) & (depth > 0)
        part = ok
        semp = cdir / f"s{s}_cam{i}_sem.npy"
        if semp.exists():
            sem = np.load(semp)
            labels = json.loads((cdir / f"s{s}_cam{i}_sem_labels.json").read_text())
            ids = [int(k) for k, lab in labels.items() if lab.get("class") in ("part", "defect")]
            if sem.shape == depth.shape and ids:
                part = ok & np.isin(sem, ids)
        lo, hi = np.quantile(depth[part], [0.01, 0.99]) if part.any() else (0, 1)
        z = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
        col = (matplotlib.colormaps["turbo"](1 - z)[:, :, :3] * 255).astype(np.uint8)
        col[~ok] = 0
        col[ok & ~part] = (col[ok & ~part] * 0.3).astype(np.uint8)
        x = x0 + i * (2 * size + gap + pair_gap)
        y = y0 + 30 + s * (size + lab_h)
        img.paste(rgb.resize((size, size)), (x, y))
        img.paste(Image.fromarray(col).resize((size, size)), (x + size + gap, y))
        lab = f"카메라 {i + 1} · 방위 {v['azimuth_deg']:.0f}° 고도 {v['elevation_deg']:.0f}°"
        if len(stages) > 1:
            lab += f" · 회전 {v['angle_deg']:.0f}°"
        fl = font(17 if size > 200 else 15, True)
        while d.textlength(lab, font=fl) > 2 * size + gap and fl.size > 11:
            fl = font(fl.size - 1, True)
        d.text((x, y - 26), lab, font=fl, fill=INK)
        d.text((x, y + size + 4), "컬러", font=font(14), fill=INK2)
        d.text((x + size + gap, y + size + 4), "깊이 (가까울수록 붉게)", font=font(14), fill=INK2)
    return img


def pointcloud_frames(pid: str, n: int):
    plt, to01 = mpl()
    from matplotlib.colors import LinearSegmentedColormap

    z = np.load(WORK / "scores" / f"{pid}.npz")
    m = json.loads((WORK / "scores" / f"{pid}.json").read_text())
    xyz, lab, s = z["xyz"] * 1e3, z["label"], z["score"]
    sn = np.clip((s - m["normal_point_median"]) / max(m["feat_threshold"] - m["normal_point_median"], 1e-9), 0, 1)
    rng = np.random.default_rng(0)
    keep = np.union1d(rng.permutation(len(xyz))[:14000], np.flatnonzero(lab >= 2))
    P, G, S = xyz[keep], (lab[keep] >= 2).astype(float), sn[keep]
    cm_gt = LinearSegmentedColormap.from_list("gt", ["#5c5b57", "#e66767"])
    cm_s = LinearSegmentedColormap.from_list("s", ["#5c5b57", "#3987e5", "#c98500", "#e66767"])
    c = (P.min(0) + P.max(0)) / 2
    rad = 0.5 * np.ptp(P[:, :2], axis=0).max() * 1.05
    from PIL import Image

    for f in range(n):
        fig = plt.figure(figsize=(W / 100, (H - 160) / 100), dpi=100)
        az = -75 + 80 * f / max(n - 1, 1)
        for k, (vals, cm_, title) in enumerate(((G, cm_gt, "정답: 시뮬레이터가 넣은 결함 위치"),
                                                (S, cm_s, "검출기: 점마다 계산한 이상 점수"))):
            ax = fig.add_subplot(1, 2, k + 1, projection="3d")
            ax.set_facecolor(to01(BG))
            o = np.argsort(vals)
            ax.scatter(P[o, 0], P[o, 1], P[o, 2], c=vals[o], cmap=cm_, vmin=0, vmax=1, s=2.4, linewidths=0,
                       depthshade=False)
            ax.set_xlim(c[0] - rad, c[0] + rad)
            ax.set_ylim(c[1] - rad, c[1] + rad)
            ax.set_zlim(c[2] - rad * 0.5, c[2] + rad * 0.5)
            ax.set_box_aspect((1, 1, 0.5), zoom=1.2)
            ax.view_init(elev=38, azim=az)
            ax.set_axis_off()
            ax.set_title(title, color=to01(INK), fontsize=16, pad=0)
        fig.subplots_adjust(left=0, right=1, top=0.93, bottom=0, wspace=0)
        a = fig_array(fig)
        plt.close(fig)
        img = canvas()
        img.paste(Image.fromarray(a), (0, 50))
        yield img


def result_card(pid: str):
    from PIL import ImageDraw

    rec = best_record(pid)
    cfg = rec["config"]
    from isco.geometry.camera_model import SlotGrid

    g = SlotGrid()
    cams = " · ".join(f"방위 {g.slot(k).azimuth_deg:.0f}°/고도 {g.slot(k).elevation_deg:.0f}°/{g.slot(k).distance_m:.2f} m"
                      for k in cfg["slots"])
    rot = "회전대 없음" if cfg["rotation"] == "none" else f"회전대 {KO_ROT[cfg['rotation']]}"
    rep = rec["report"]
    img = text_card([(f"{KO_PART[pid]} — 권장 검사 설정", 36, True, INK),
                     (f"카메라 {len(cfg['slots'])}대: {cams}", 22, False, INK2),
                     (f"{rot} · 벨트 {SPEEDS[cfg['speed_level']]} m/s · 사이클 {rec['objectives']['cycle_time_s']:.1f}초"
                      f" (제약: 카메라 3대 이하, 1.5초 이하)", 22, False, INK2),
                     ("", 4, False, INK),
                     (f"놓친 결함 비율 {rep['fnr'] * 100:.0f}%   (95% 신뢰구간 {rep['fnr_ci'][0] * 100:.0f}–"
                      f"{rep['fnr_ci'][1] * 100:.0f}%, 정상품 오판정 1% 기준)", 28, True, ACCENT),
                     (f"탐색에 쓰지 않은 결함 36개로 매긴 성적 · 정상/불량 분리 능력 {rep['auroc']:.2f}", 20, False, MUTED)],
                    top=95, left=80, gap=16)
    d = ImageDraw.Draw(img)
    x0, y0, bw, bh = 110, 430, 150, 150
    d.text((80, y0 - 52), "결함 종류별로 놓친 비율", font=font(22, True), fill=INK)
    for i, t in enumerate(TYPES):
        v = rep["fnr_by_type"][t]
        x = x0 + i * (bw + 30)
        d.rectangle([x, y0, x + bw - 40, y0 + bh], outline=PANEL, width=2)
        hh = int(bh * v)
        col = GOOD if v < 0.34 else (ACCENT if v < 0.67 else RED)
        if hh > 1:
            d.rectangle([x + 2, y0 + bh - hh, x + bw - 42, y0 + bh - 1], fill=col)
        d.text((x, y0 + bh + 8), KO_TYPE[t], font=font(20, True), fill=INK)
        d.text((x + 52, y0 + bh + 10), f"{v * 100:.0f}%", font=font(18), fill=INK2)
    return img


def search_frames(data: dict, n1: int, n2: int):
    plt, to01 = mpl()
    from PIL import Image

    s1, s2 = data["stage1"], data["stage2"]
    V = np.array([c["V"] for c in s1])
    T = np.array([c["T"] for c in s1])
    N = np.array([c["n"] for c in s1])
    rng = np.random.default_rng(0)
    jit = rng.uniform(-0.05, 0.05, len(T))
    order = rng.permutation(len(s1))
    sl = set(data["shortlist"])
    sl_idx = np.array([i for i, c in enumerate(s1) if c["id"] in sl])
    F2 = np.array([r["fnr"] for r in s2])
    T2 = np.array([r["T"] for r in s2])
    N2 = np.array([r["n"] for r in s2])
    cols = ["#3987e5", "#199e70", "#c98500", "#d95926", "#d55181", "#9085e9"]
    best = data["best"]

    def frame(k1, k2, final):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(W / 100, (H - 150) / 100), dpi=100)
        idx = order[:k1]
        for n in range(1, 7):
            m = idx[N[idx] == n]
            a1.scatter(T[m] + jit[m], V[m], s=7, color=cols[n - 1], alpha=0.8, linewidths=0, label=f"{n}대")
        if k2 > 0:
            a1.scatter(T[sl_idx] + jit[sl_idx], V[sl_idx], s=70, facecolors="none", edgecolors="white", linewidths=1.3)
        a1.set_xlabel("사이클 시간 (초)")
        a1.set_ylabel("결함 표면이 보이는 비율 (가시성)")
        a1.set_title(f"1단계: 가시성으로 빠르게 거르기 ({k1:,}/{len(s1):,}개)", fontsize=15, color=to01(INK))
        a1.set_xlim(0, 5)
        a1.set_ylim(0, 0.85)
        leg = a1.legend(title="카메라", fontsize=10, title_fontsize=10, loc="lower right", markerscale=2,
                        facecolor=to01(PANEL), edgecolor="none", labelcolor=to01(INK2))
        leg.get_title().set_color(to01(INK2))
        a2.set_xlim(0, 5)
        a2.set_ylim(0, 1.05)
        a2.set_xlabel("사이클 시간 (초)")
        a2.set_ylabel("놓친 결함 비율 (과검 1% 기준)")
        a2.set_title(f"2단계: 검출기로 실제 평가 ({k2}/{len(s2)}회)", fontsize=15, color=to01(INK))
        a2.axvspan(1.5, 5, color=to01(RED), alpha=0.08, lw=0)
        a2.text(1.55, 0.02, "제약(1.5초) 밖", color=to01(RED), fontsize=11)
        if k2 > 0:
            m = np.arange(k2)
            a2.scatter(T2[m], F2[m], s=40 + 25 * N2[m], c=[cols[n - 1] for n in N2[m]], edgecolors="white",
                       linewidths=0.6, zorder=3)
        if final:
            a2.scatter([best["T"]], [best["fnr"]], s=480, marker="*", color=to01(ACCENT), edgecolors="white",
                       linewidths=1.2, zorder=5)
            a2.annotate(f"권장: 카메라 {best['n']}대 + 90° 회전, 0.6 m/s\n사이클 {best['T']:.1f}초, 놓친 비율 {best['fnr'] * 100:.0f}%",
                        (best["T"], best["fnr"]), xytext=(2.0, 0.12), color=to01(INK), fontsize=13,
                        arrowprops=dict(arrowstyle="->", color=to01(ACCENT)))
        for ax in (a1, a2):
            ax.grid(alpha=0.12)
        fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.12, wspace=0.22)
        a = fig_array(fig)
        plt.close(fig)
        img = canvas()
        img.paste(Image.fromarray(a), (0, 40))
        return img

    for f in range(n1):
        yield frame(int(len(s1) * (f + 1) / n1), 0, False)
    per = max(1, n2 // len(s2))
    for k in range(1, len(s2) + 1):
        img = frame(len(s1), k, False)
        for _ in range(per):
            yield img
    yield frame(len(s1), len(s2), True)


# ------------------------------------------------------------------------------ compose
def main() -> None:
    from PIL import Image

    out = WORK / "ISCO_중간발표_데모영상.mp4"
    v = Video(out)

    title = text_card([("2026-2 산학캡스톤디자인 · 팀 강화실패", 24, True, ACCENT),
                       ("ISCO 중간 점검 데모", 66, True, INK),
                       ("부품이 들어오면, 그 부품을 가장 적은 비용으로 잘 검사하는", 30, False, INK),
                       ("카메라 배치·회전대·컨베이어 속도를 찾아 주는 가상 공정", 30, False, INK),
                       ("", 10, False, INK),
                       ("Isaac Sim 5.1 컨베이어 디지털 트윈 · 검증 제품 6종 · 2026년 10월 26일 기준", 22, False, MUTED)],
                      top=170)
    v.add(title, 4.0)
    problem = text_card([("검사 라인 설계자가 정해야 하는 것", 40, True, INK),
                         ("카메라를 몇 대, 어느 각도와 거리에 둘 것인가", 28, False, INK),
                         ("부품을 회전대로 돌려 다시 찍을 것인가 · 컨베이어를 얼마나 빨리 돌릴 것인가", 28, False, INK),
                         ("", 10, False, INK),
                         ("ISCO는 이 조합을 시뮬레이터 안에서 직접 시험해서", 28, False, ACCENT),
                         ("\"정상품 오판정 1%일 때 놓치는 불량 비율\"과 비용(카메라 수·사이클 시간)을 함께 보고 고른다", 26,
                          False, ACCENT)], top=190)
    v.fade_to(problem)
    v.add(problem, 5.5)

    # 1. the process environment
    first = True
    for f in clip(ISAAC / "line.mp4"):
        img = caption(Image.fromarray(f), "가상 공정: 실제 컨베이어를 본뜬 Isaac Sim 디지털 트윈", "1  가상 공정 환경",
                      "벨트 폭 0.9 m · 부품 간격 0.3 m · 여기서는 0.3 m/s로 흐름 · 레일·기둥은 카메라를 가리는 물체로 계산")
        if first:
            v.fade_to(img)
            first = False
        v.add(img, frames=1)
    # 2. products
    first = True
    for img in lineup_frames():
        img = caption(img, "검사할 제품 6종: 공개 데이터의 전기 부품 4종 + 실제 스캔 부품 2종", "2  검사할 제품",
                      "모두 가장 긴 변 40 mm로 맞춤 · T-LESS = 전기 부품 설계 모델, MulSen-AD = 실제 산업 부품 360° 스캔")
        if first:
            v.fade_to(img)
            first = False
        v.add(img, frames=1)
    # 3. defects
    dmeta = json.loads((ISAAC / "defects.json").read_text())
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
        for f in clip(ISAAC / f"defect_{t}.mp4"):
            v.add(caption(Image.fromarray(f), f"결함 생성기: {KO_TYPE[t]}  ({', '.join(size)})", "3  결함 만들기",
                          "부품 형상을 직접 깎거나 밀어서 6종 결함을 만들고, 결함 면에는 자동으로 정답 표시(빨강)"),
                  frames=1)
    # 4. configuration space
    first = True
    for f in clip(ISAAC / "slots.mp4"):
        img = caption(Image.fromarray(f), "공정 설정 ①: 카메라 후보 자리 96곳 (방위 12 × 고도 4 × 거리 2)",
                      "4  공정 환경 설정", "파랑 = 0.25 m, 주황 = 0.40 m · 카메라 1–6대를 이 중에서 고름 (서로 30° 이상 떨어지게)")
        if first:
            v.fade_to(img)
            first = False
        v.add(img, frames=1)
    tt = json.loads((ISAAC / "turntable.json").read_text())
    for i, f in enumerate(clip(ISAAC / "turntable.mp4")):
        img = caption(Image.fromarray(f), "공정 설정 ②: 회전대 — 부품을 멈추고 돌려 가며 찍는다", "4  공정 환경 설정",
                      "없음 / 90° 한 번(0.9초) / 180° 한 번(1.1초) / 90°씩 세 번(1.7초) · 다시 놓일 때 0.5 mm·1° 오차")
        v.add(flash(img) if any(abs(i - k) <= 1 for k in tt["flash_frames"]) else img, frames=1)
    settings = text_card([("공정 설정 ③: 바꿔 보는 값과 비용", 38, True, INK),
                          ("카메라 수 1–6대 · 위치 96곳 · 회전대 4가지 · 벨트 0.1/0.3/0.6 m/s · 동시/순차 촬영", 25, False, INK),
                          ("사이클 시간 = 부품 간격 0.3 m ÷ 벨트 속도 + 회전 시간  (0.5–4.7초)", 25, False, INK),
                          ("벨트가 빠르면: 촬영 신호 지연 → 위치 오차(0.6 m/s에서 3 mm), 노출 중 번짐 1.2 mm", 25, False, INK),
                          ("센서 잡음: 0.25 m에서 0.15 mm (거리 제곱에 비례) · 비스듬한 면 측정 실패 · 경계 튐 제거", 25, False, INK),
                          ("", 8, False, INK),
                          ("가능한 조합은 수백만 개 이상 → 1단계(가시성)로 거르고 2단계(검출기)로 정밀 평가", 25, True, ACCENT)],
                         top=130, gap=20)
    v.fade_to(settings)
    v.add(settings, 6.0)
    # 5. search (one product)
    data = json.loads((WORK / "search_tless_29.json").read_text())
    first = True
    img = None
    for img in search_frames(data, n1=int(3 * FPS), n2=int(5 * FPS)):
        img = caption(img, f"설정 탐색 (T-LESS 29번): 후보 {len(data['stage1']):,}개를 가시성으로 거르고, 40개를 검출기로 평가",
                      "5  설정 찾기", "흰 테두리 = 2단계 출발점으로 뽑힌 후보 · 오른쪽 점수는 탐색에 쓰지 않은 결함으로 매긴 성적")
        if first:
            v.fade_to(img)
            first = False
        v.add(img, frames=1)
    v.add(img, 3.0)

    # 6. per product
    for k, pid in enumerate(PARTS):
        rec = best_record(pid)
        cfg = rec["config"]
        meta = json.loads((ISAAC / f"part_{pid}.json").read_text())
        sc = json.loads((WORK / "scores" / f"{pid}.json").read_text())
        rot = "회전대 없음" if cfg["rotation"] == "none" else f"회전대 {KO_ROT[cfg['rotation']]}"
        chapter = f"6  제품별 검사 시뮬레이션 ({k + 1}/6)"
        box = [(KO_PART[pid], 28, True, INK),
               (f"카메라 {len(cfg['slots'])}대 · {rot}", 21, False, INK2),
               (f"벨트 {meta['speed_mps']} m/s · 사이클 {rec['objectives']['cycle_time_s']:.1f}초", 21, False, INK2),
               (f"이 부품의 결함: {KO_TYPE[sc['type']]}", 21, False, ACCENT)]
        sub = ("회전대: 검사 지점에서 멈춰 찍고, 90° 돌려 다시 찍는다 (번쩍임 = 촬영)" if cfg["rotation"] != "none"
               else "부품이 검사 지점을 지나는 순간 카메라가 촬영한다 (번쩍임 = 촬영)")
        first = True
        for i, f in enumerate(clip(ISAAC / f"part_{pid}.mp4")):
            img = info_box(Image.fromarray(f), box)
            img = caption(img, "공정: 탐색이 찾은 권장 설정으로 부품이 검사 지점을 지난다 (사각뿔 = 카메라)", chapter, sub)
            if any(0 <= i - c < 3 for c in meta["capture_frames"]):
                img = flash(img, 0.55)
            if first:
                v.fade_to(img)
                first = False
            v.add(img, frames=1)
        cams = caption(camera_panel(pid), "촬영: 각 카메라가 얻은 컬러 영상과 깊이 영상", chapter,
                       "실제 계산에는 깊이만 쓰고, 센서 잡음·측정 실패·벨트 움직임을 덧입힌 뒤 3D 점으로 합친다")
        v.fade_to(cams, 0.3)
        v.add(cams, 3.2)
        first = True
        for img in pointcloud_frames(pid, int(4.2 * FPS)):
            img = caption(img, f"검출: 결함 점수 {sc['score']:.1f} > 판정 기준선 {sc['tau']:.1f} → 불량으로 검출", chapter,
                          "정상품 40개만 배운 검출기가 처음 보는 결함을 '정상과 다른 정도'로 찾아낸다 (기준선 = 정상품 오판정 1%)")
            if first:
                v.fade_to(img)
                first = False
            v.add(img, frames=1)
        res = caption(result_card(pid), "결과: 이 제품의 권장 설정과 성적", chapter)
        v.fade_to(res)
        v.add(res, 4.0)

    # 7. across products and the verification experiments
    for name, text, sub, sec in (
            ("recommended_rigs", "제품마다 찾은 권장 설정 (위에서 본 카메라 위치)", "6개 중 4개가 \"0.6 m/s + 90° 한 번 회전\" — 같은 사이클이면 카메라를 늘리기보다 회전대로 시점을 늘리는 선택이 자주 이겼다", 6.0),
            ("cross_application", "검증 ①: 다른 제품의 최적 설정을 써도 될까?", "중앙값 9.7%p 손해 (기준 10%p 이상) · 확실한 차이는 30쌍 중 2쌍 → \"제품마다 다르다\"는 기준 미달", 6.5),
            ("visibility_vs_miss", "검증 ②: 결함이 잘 보이는 설정이면 덜 놓칠까?", "4개 중 1개 제품만 기준 충족 → 보여도 놓친다: 잡음 대비 작은 변형과 위치 맞추기 품질이 더 중요", 6.5),
            ("static_vs_inline", "검증 ③: 정지 촬영으로 찾은 설정을 컨베이어 라인에 써도 될까?", "3개 제품 모두 사이클 1.5초를 넘어 못 쓴다 → 공정 제한을 탐색에 넣어야 한다 (지키는 대가 8–42%p)", 6.5),
            ("detector_robustness", "검증 ④: 검출기를 바꿔도 같은 설정이 좋을까?", "2개 중 1개 제품에서만 순위가 유지 → 권장 설정에는 어떤 검출기를 쓰는지 함께 적는다", 5.5),
            ("sensor_gap", "검증 ⑤: 시뮬레이터 센서는 실제 3D 카메라와 얼마나 다를까?", "실제 카메라가 훨씬 깨끗하다 → 우리 센서는 보수적 가정, 절대 수치가 아니라 설정 간 순위를 주장", 5.5),
            ("search_efficiency", "검증 ⑥: 두 단계 탐색은 효율적인가? (진행 중, T-LESS 13번)", "1단계 후보로 시작한 최적화가 무작위 탐색보다 30회 시점에서 59% 앞섬", 5.0)):
        img = caption(figure_card(name), text, "7  검증 실험 결과", sub)
        v.fade_to(img)
        v.add(img, sec)

    summary = text_card([("지금까지의 결론", 40, True, INK),
                         ("✓ 시뮬레이터·계산 방법 점검 9가지 통과 (검출기는 공개 논문 수치 0.883을 그대로 재현)", 24, False, INK),
                         ("✓ 공정의 사이클 시간 제한을 반영하면 권장 설정이 달라진다 (3/3 제품)", 24, False, INK),
                         ("✓ 결함이 보이는 것과 잡히는 것은 다르다 → 검출기 성능으로 직접 탐색해야 한다", 24, False, INK),
                         ("✗ \"제품마다 최적 설정이 다르다\", \"가시성만으로 충분하다\"는 미리 정한 기준에 못 미침", 24, False, INK2),
                         ("… 실제 결함 스캔에서의 순위 유지 실험은 정렬 오류를 고쳐 다시 실행 예정", 24, False, INK2),
                         ("", 8, False, INK),
                         ("다음: 실제 결함 순위 실험 · 잡음 없는 센서 비교 · 탐색 확대 재실험 → 기말 발표, 2027년 3월 학회 투고", 23,
                          True, ACCENT),
                         ("ISCO · Inspection Setup Configuration Optimizer", 21, True, MUTED)], top=120, gap=20)
    v.fade_to(summary)
    v.add(summary, 8.0)
    v.close()
    print(f"[video] {out}: {v.n} frames ({v.n / FPS:.1f}s)")


if __name__ == "__main__":
    main()
