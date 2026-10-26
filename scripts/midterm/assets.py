"""Picture assets of the mid-term report and slides: system flow diagram, product stills,
defect close-ups and per-product detection examples (point clouds: ground truth vs anomaly score).

    python scripts/midterm/assets.py   -> runs/midterm_reports/figures/{light,dark}/*.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import KO_PART, KO_TYPE, PARTS, TYPES, WORK, best_record  # noqa: E402
from figures import THEMES, save, setup  # noqa: E402

ISAAC = WORK / "isaac"


def frame_of(path: Path, frac: float) -> np.ndarray:
    import imageio

    r = imageio.get_reader(str(path))
    n = r.count_frames()
    f = np.asarray(r.get_data(min(n - 1, int(frac * n))))
    r.close()
    return f


def fig_system(theme: str) -> None:
    plt, c = setup(theme)
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(12.8, 5.2) if theme == "dark" else (12, 5.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    s = c["series"]
    boxes = [
        (0.2, 2.75, 2.2, 1.95, "입력", ["부품 3D 모델", "(설계도·실제 스캔)", "결함 종류 목록", "제약: 3대·1.5초"], s[0]),
        (2.8, 2.75, 2.6, 1.95, "데이터 생성기", ["정상품 140개 + 결함 72개", "결함 6종 자동 생성", "정답 표시 자동"], s[1]),
        (5.8, 2.75, 2.8, 1.95, "가상 공정 (Isaac Sim)", ["컨베이어·카메라 자리 96곳", "미리 찍어 두기 (부품당 2만 장)", "센서 잡음·벨트 움직임 모델"], s[2]),
        (9.0, 2.75, 2.8, 1.95, "검출 파이프라인", ["3D 점 합치기·위치 맞추기", "정상품만 배운 검출기", "과검률 1%에서의 미검률"], s[3]),
        (2.8, 0.4, 5.8, 1.7, "설정 최적화기 (두 단계 탐색)", ["1단계: 가시성으로 후보 약 1,400개를 몇 초에 거르기",
                                                   "2단계: 유망한 40개를 검출기로 평가 + 베이지안 최적화"], s[4]),
        (9.0, 0.4, 2.8, 1.7, "결과 리포트", ["권장 설정", "성능–비용 곡선", "결함 종류별 가시성·미검률"], s[5]),
    ]
    for x, y, w, h, title, lines, col in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=c["surface"],
                                    ec=col, lw=2.2))
        ax.add_patch(FancyBboxPatch((x, y + h - 0.42), w, 0.42, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=col, ec=col, lw=0))
        ax.text(x + w / 2, y + h - 0.21, title, ha="center", va="center", fontsize=12.5, fontweight="bold",
                color="#ffffff" if col not in ("#eda100", "#c98500") else "#0b0b0b")
        for i, line in enumerate(lines):
            ax.text(x + w / 2, y + h - 0.68 - i * 0.33, line, ha="center", va="center", fontsize=10.5, color=c["ink"])

    def arrow(p, q, text=None):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, color=c["muted"], lw=1.6))
        if text:
            ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.12, text, ha="center", fontsize=9.5, color=c["ink2"])

    arrow((2.4, 3.7), (2.8, 3.7))
    arrow((5.4, 3.7), (5.8, 3.7))
    arrow((8.6, 3.7), (9.0, 3.7))
    arrow((10.4, 2.75), (10.4, 2.1))
    arrow((8.6, 1.25), (9.0, 1.25))
    ax.add_patch(FancyArrowPatch((8.3, 2.1), (9.35, 2.75), arrowstyle="<|-|>", mutation_scale=16, color=c["muted"],
                                 lw=1.6))
    ax.text(8.2, 2.42, "시험할 설정 ⇄ 미검률", fontsize=9.5, color=c["ink2"], ha="right")
    fig.tight_layout()
    save(fig, "system_flow", theme)


def fig_products(theme: str) -> None:
    plt, c = setup(theme)
    from PIL import Image

    notes = {"tless_13": "안쪽 나사산 전구 소켓 · 자기 가림 큼", "tless_18": "원뿔형 하우징 · 자기 가림 가장 큼",
             "tless_29": "뚜껑 달린 사각 전기 박스 · 단순형", "tless_30": "원형 전기 박스 · 단순형",
             "mulsen_nut": "육각 너트 · 실제 스캔", "mulsen_flat_pad": "고리 모양 패드 · 실제 스캔, 얇음"}
    fig, axs = plt.subplots(2, 3, figsize=(12, 5.6) if theme == "light" else (12.8, 5.7))
    for ax, p in zip(axs.ravel(), PARTS):
        im = Image.open(ISAAC / "thumbs" / f"{p}.png").convert("RGB")
        w, h = im.size
        ax.imshow(np.asarray(im.crop((int(w * 0.18), int(h * 0.05), int(w * 0.82), int(h * 0.95)))))
        ax.set_title(f"{KO_PART[p]}\n{notes[p]}", fontsize=11.5, color=c["ink"], fontweight="normal")
        ax.axis("off")
    fig.suptitle("검증 실험의 제품 6종 (모두 가장 긴 변 40 mm로 맞춤)", color=c["ink"], fontweight="bold")
    fig.tight_layout()
    save(fig, "products", theme)


def fig_defects(theme: str) -> None:
    plt, c = setup(theme)
    meta = json.loads((ISAAC / "defects.json").read_text())
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.2) if theme == "light" else (12.8, 5.9))
    for ax, t in zip(axs.ravel(), TYPES):
        ax.imshow(frame_of(ISAAC / f"defect_{t}.mp4", 0.5)[60:690, 200:1080])
        d = meta[t]["defect"]
        size = []
        if d.get("diameter_m"):
            size.append(f"지름 {d['diameter_m'] * 1e3:.1f} mm")
        if d.get("length_m"):
            size.append(f"길이 {d['length_m'] * 1e3:.1f} mm")
        if d.get("width_m"):
            size.append(f"폭 {d['width_m'] * 1e3:.2f} mm")
        if d.get("depth_m"):
            size.append(("처짐 " if t == "warp" else "깊이 ") + f"{d['depth_m'] * 1e3:.2f} mm")
        ax.set_title(f"{KO_TYPE[t]}  ({', '.join(size)})", fontsize=12, color=c["ink"])
        ax.axis("off")
    fig.suptitle("결함 생성기가 만든 6종 결함 (T-LESS 29번, 빨간 면 = 자동 정답 표시)", color=c["ink"], fontweight="bold")
    fig.tight_layout(h_pad=1.6)
    save(fig, "defects", theme)


def fig_detection(theme: str) -> None:
    """Per product: merged point cloud of the demo defect part, ground truth vs anomaly score."""
    plt, c = setup(theme)
    from matplotlib.colors import LinearSegmentedColormap

    fig = plt.figure(figsize=(13, 8.2) if theme == "light" else (12.8, 6.0))
    cm_gt = LinearSegmentedColormap.from_list("gt", [c["axis"], c["red"]])
    cm_s = LinearSegmentedColormap.from_list("s", [c["axis"], "#86b6ef" if theme == "light" else "#3987e5",
                                                   c["series"][3], c["red"]])
    for k, p in enumerate(PARTS):
        z = np.load(WORK / "scores" / f"{p}.npz")
        m = json.loads((WORK / "scores" / f"{p}.json").read_text())
        xyz, lab, s = z["xyz"] * 1e3, z["label"], z["score"]
        lo, hi = m["normal_point_median"], m["feat_threshold"]
        sn = np.clip((s - lo) / max(hi - lo, 1e-9), 0, 1)
        rng = np.random.default_rng(0)
        keep = np.union1d(rng.permutation(len(xyz))[:9000], np.flatnonzero(lab >= 2))
        P = xyz[keep]
        cc = (P.min(0) + P.max(0)) / 2
        rad = 0.5 * np.ptp(P[:, :2], axis=0).max() * 1.05
        for j, (vals, cm_, title) in enumerate(((np.where(lab[keep] >= 2, 1.0, 0.0), cm_gt, "정답"),
                                                (sn[keep], cm_s, "검출기 점수"))):
            ax = fig.add_subplot(3, 4, 2 * k + j + 1, projection="3d")
            ax.set_facecolor(c["surface"])
            o = np.argsort(vals)
            ax.scatter(P[o, 0], P[o, 1], P[o, 2], c=vals[o], cmap=cm_, vmin=0, vmax=1, s=1.6, linewidths=0,
                       depthshade=False)
            ax.set_xlim(cc[0] - rad, cc[0] + rad)
            ax.set_ylim(cc[1] - rad, cc[1] + rad)
            ax.set_zlim(cc[2] - rad * 0.5, cc[2] + rad * 0.5)
            ax.set_box_aspect((1, 1, 0.5), zoom=1.15)
            ax.view_init(elev=42, azim=-60)
            ax.set_axis_off()
            name = f"{KO_PART[p]} · {KO_TYPE[m['type']]}" if j == 0 else \
                f"점수 {m['score']:.1f} > 기준선 {m['tau']:.1f} → 검출"
            ax.set_title(f"{title}: {name}", fontsize=10.5, color=c["ink"], pad=0)
    fig.suptitle("제품별 검출 예시: 권장 설정으로 찍은 3D 점 묶음 (왼쪽 = 시뮬레이터가 넣은 결함, 오른쪽 = 정상품만 배운 검출기의 점수)",
                 color=c["ink"], fontweight="bold", fontsize=12.5)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01, wspace=0.0, hspace=0.12)
    save(fig, "detection_examples", theme)


def main() -> None:
    for theme in THEMES:
        for fn in (fig_system, fig_products, fig_defects, fig_detection):
            fn(theme)
            print(f"[assets] {theme} {fn.__name__}", flush=True)


if __name__ == "__main__":
    main()
