"""Korean figures of the mid-term report, slides and video, from the experiment results.

    python scripts/midterm/figures.py      -> runs/midterm_reports/figures/{light,dark}/*.png

Colours follow the validated reference palette (light: categorical slots 1-4 pass the
adjacent CVD / normal-vision gates; aqua and yellow are below 3:1 on the light surface, so
those series always carry direct labels or marker shapes).
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EXP, KO_PART, KO_ROT, KO_TYPE, PARTS, ROOT, SPEEDS, TYPES, WORK, best_record  # noqa: E402

FONT_DIR = Path.home() / ".fonts"
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "axis": "#c3c2b7", "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"],
              "mid": "#f0efec", "red": "#e34948", "good": "#0ca30c", "critical": "#d03b3b"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "axis": "#383835", "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"],
             "mid": "#383835", "red": "#e66767", "good": "#0ca30c", "critical": "#d03b3b"},
}
MARKERS = ["o", "s", "^", "D", "v", "P"]
SHORT = {"tless_13": "T-LESS 13", "tless_18": "T-LESS 18", "tless_29": "T-LESS 29", "tless_30": "T-LESS 30",
         "mulsen_nut": "너트", "mulsen_flat_pad": "평판 패드"}


def setup(theme: str):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager

    for f in ("NanumGothic.ttf", "NanumGothicExtraBold.ttf"):
        font_manager.fontManager.addfont(str(FONT_DIR / f))
    import matplotlib.pyplot as plt

    c = THEMES[theme]
    plt.rcParams.update({
        "font.family": "NanumGothic", "axes.unicode_minus": False, "font.size": 12,
        "figure.facecolor": c["surface"], "axes.facecolor": c["surface"], "savefig.facecolor": c["surface"],
        "axes.edgecolor": c["axis"], "axes.labelcolor": c["ink2"], "text.color": c["ink"],
        "xtick.color": c["muted"], "ytick.color": c["muted"], "axes.titlecolor": c["ink"],
        "axes.grid": True, "grid.color": c["grid"], "grid.linewidth": 0.7, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
        "legend.frameon": False, "legend.labelcolor": c["ink2"], "axes.titleweight": "bold",
        "axes.titlesize": 13, "axes.titlepad": 10})
    return plt, c


def save(fig, name: str, theme: str) -> None:
    d = WORK / "figures" / theme
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.png", dpi=200 if theme == "light" else 100)
    import matplotlib.pyplot as plt

    plt.close(fig)


def size(theme: str, w: float = 10.0, h: float = 5.4):
    return (w, h) if theme == "light" else (12.8, 5.7)


def jl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path)]


# ------------------------------------------------------------------------------ figures
def fig_cross(theme: str) -> None:
    """Miss rate on each part (columns) of each part's optimum (rows); diagonal = own optimum."""
    plt, c = setup(theme)
    from matplotlib.colors import LinearSegmentedColormap

    s = json.loads((EXP / "e1" / "summary.json").read_text())
    M = np.array(s["fnr_matrix_report"])
    D = M - np.diag(M)[None, :]
    cmap = LinearSegmentedColormap.from_list("div", [c["series"][0], c["mid"], c["red"]])
    fig, ax = plt.subplots(figsize=size(theme, 9.2, 6.4))
    ax.grid(False)
    im = ax.imshow(D, cmap=cmap, vmin=-0.4, vmax=0.4)
    for i in range(6):
        for j in range(6):
            txt = f"{M[i, j] * 100:.0f}%" if i == j else f"{D[i, j] * 100:+.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=12 if i == j else 11,
                    fontweight="bold" if i == j else "normal",
                    color=c["ink"] if abs(D[i, j]) < 0.25 else "#ffffff")
            if i == j:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=c["ink"], lw=1.6))
    names = [SHORT[p] for p in PARTS]
    ax.set_xticks(range(6), [f"{n}에 적용" for n in names], rotation=25, ha="right")
    ax.set_yticks(range(6), [f"{n} 최적 설정" for n in names])
    ax.tick_params(colors=c["ink2"], length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("자기 최적 대비 미검률 변화 (%p, 빨강 = 손해)", color=c["ink2"])
    cb.outline.set_visible(False)
    cb.ax.yaxis.set_tick_params(color=c["muted"])
    cb.set_ticks([-0.4, -0.2, 0, 0.2, 0.4], labels=["-40", "-20", "0", "+20", "+40"])
    ax.set_title("다른 부품의 최적 설정을 썼을 때 미검률이 얼마나 달라지나\n(대각선 = 자기 최적 설정의 미검률, 나머지 = 그 대비 차이)")
    fig.tight_layout()
    save(fig, "cross_application", theme)


def fig_visibility(theme: str) -> None:
    plt, c = setup(theme)
    s = json.loads((EXP / "e2" / "summary.json").read_text())
    fig, axs = plt.subplots(1, 4, figsize=size(theme, 13, 4.2), sharey=True)
    for ax, pid in zip(axs, ["tless_13", "tless_18", "tless_29", "tless_30"]):
        R = jl(EXP / "e2" / f"evals_{pid}.jsonl")
        V = np.array([r["surrogate"]["V_area"] for r in R])
        F = np.array([r["report"]["fnr"] for r in R])
        rot = np.array([r["config"]["rotation"] != "none" for r in R])
        ax.scatter(V[~rot], F[~rot], s=34, color=c["series"][0], marker="o", label="회전대 없음",
                   edgecolors=c["surface"], linewidths=0.8)
        ax.scatter(V[rot], F[rot], s=38, color=c["series"][1], marker="^", label="회전대 사용",
                   edgecolors=c["surface"], linewidths=0.8)
        rho = s[pid]["spearman_V_area"]
        ok = s[pid]["supported"]
        ax.set_title(f"{SHORT[pid]}  순위상관 {rho:+.2f}", color=c["ink"])
        ax.text(0.03, 0.04, "기준 충족" if ok else "기준 미달", transform=ax.transAxes, fontsize=11,
                color=c["good"] if ok else c["critical"], fontweight="bold")
        ax.set_xlabel("결함 표면이 보이는 비율 (가시성)")
        ax.set_xlim(0, 0.8)
        ax.set_ylim(0, 1.05)
    axs[0].set_ylabel("미검률 (과검률 1% 기준)")
    axs[0].yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    axs[-1].legend(loc="lower right", fontsize=10)
    fig.suptitle("가시성이 높다고 덜 놓치는가 — 무작위 설정 60개씩 (기준: 순위상관 -0.5 이하, 4개 중 3개 부품)",
                 color=c["ink"], fontweight="bold", fontsize=13)
    fig.tight_layout()
    save(fig, "visibility_vs_miss", theme)


def fig_camera_addition(theme: str) -> None:
    plt, c = setup(theme)
    s = json.loads((EXP / "e3" / "summary.json").read_text())
    names = {"greedy_vis": "가시성 기준으로 한 대씩 추가", "greedy_fnr": "미검률 기준으로 한 대씩 추가",
             "random": "무작위 탐색", "bo_warm": "베이지안 최적화"}
    fig, axs = plt.subplots(2, 3, figsize=size(theme, 12.5, 7.0), sharex=True, sharey=True)
    for ax, pid in zip(axs.ravel(), PARTS):
        b = s[pid]["fpfh_bank"]["best_fnr_by_n"]
        for k, (m, lab) in enumerate(names.items()):
            ns = [int(n) for n, v in b[m].items() if v is not None and v == v]
            vs = [b[m][str(n)] for n in ns]
            ax.plot(ns, vs, color=c["series"][k], marker=MARKERS[k], markersize=6, label=lab,
                    markeredgecolor=c["surface"], markeredgewidth=0.8)
        cad = s[pid]["cad_distance"]["best_fnr_by_n"]["greedy_vis"]
        ns = [int(n) for n, v in cad.items() if v is not None]
        ax.plot(ns, [cad[str(n)] for n in ns], color=c["muted"], ls="--", lw=1.4, label="(비교) 설계도 거리 검출기, 가시성 순 추가")
        gain = -s[pid]["bo_minus_greedy_vis_at_n3"]
        ax.set_title(f"{SHORT[pid]}  · 3대에서 최적화 이득 {gain * 100:.0f}%p", color=c["ink"], fontsize=12)
        ax.set_xticks(range(1, 7))
        ax.set_ylim(0, 1.0)
    for ax in axs[1]:
        ax.set_xlabel("카메라 수")
    for ax in axs[:, 0]:
        ax.set_ylabel("가장 낮은 미검률")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=10.5, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("같은 평가 40회로 찾은 카메라 수별 최선의 설정 (회전대 없음, 0.3 m/s)", color=c["ink"], fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "camera_addition", theme)


def fig_static_inline(theme: str) -> None:
    plt, c = setup(theme)
    s = json.loads((EXP / "e4" / "summary.json").read_text())
    fig, ax = plt.subplots(figsize=size(theme, 9, 5.4))
    ax.axvspan(1.5, 2.95, color=c["red"], alpha=0.08, lw=0)
    ax.axvline(1.5, color=c["red"], lw=1.2, ls="--")
    ax.text(1.53, 0.04, "사이클 시간 제한\n(1.5초) 초과", color=c["critical"], fontsize=10.5)
    for k, pid in enumerate(["tless_13", "tless_29", "tless_30"]):
        v = s[pid]
        a = (v["static_opt_deployed_inline"]["2"]["cycle_time_s"], v["static_opt_deployed_inline"]["2"]["fnr_report"])
        b = (v["inline_opt_cycle_time_s"], v["inline_opt_fnr_report"])
        col = c["series"][k]
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color=col, lw=1.6))
        ax.scatter(*a, s=90, color=col, marker="o", edgecolors=c["surface"], zorder=3)
        ax.scatter(*b, s=110, color=col, marker="^", edgecolors=c["surface"], zorder=3)
        box = dict(facecolor=c["surface"], edgecolor="none", alpha=0.9, pad=1.5)
        ax.text(a[0] + 0.04, a[1] - 0.035, f"{SHORT[pid]}: 정지 촬영 최적 ({KO_ROT[v['static_opt']['rotation']]})",
                fontsize=10.5, color=c["ink2"], bbox=box, zorder=4)
        ax.text(b[0] - 0.05, b[1] - 0.008, f"{SHORT[pid]}: 인라인 최적 ({KO_ROT[v['inline_opt']['rotation']]})",
                fontsize=10.5, color=c["ink2"], ha="right", bbox=box, zorder=4)
    ax.set_xlim(0.45, 2.95)
    ax.set_ylim(0, 0.75)
    ax.set_xlabel("부품 1개 검사 시간 = 사이클 시간 (초, 0.6 m/s 기준)")
    ax.set_ylabel("미검률 (과검률 1% 기준)")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    ax.set_title("정지 촬영으로 찾은 최적 설정은 인라인 라인의 사이클 제한을 넘는다\n(●: 정지 촬영 최적을 라인에 그대로 쓴 경우 → ▲: 제한을 지키는 인라인 최적)")
    fig.tight_layout()
    save(fig, "static_vs_inline", theme)


def fig_detectors(theme: str) -> None:
    plt, c = setup(theme)
    s = json.loads((EXP / "e6" / "summary.json").read_text())
    fig, axs = plt.subplots(1, 2, figsize=size(theme, 10.5, 4.8), sharex=True, sharey=True)
    for ax, pid in zip(axs, ["tless_13", "tless_29"]):
        f = np.array(s[pid]["fnr"]["fpfh_bank"])
        g = np.array(s[pid]["fnr"]["cad_distance"])
        ax.plot([0, 1], [0, 1], color=c["axis"], lw=1)
        ax.scatter(g, f, s=36, color=c["series"][0], edgecolors=c["surface"], linewidths=0.8)
        i = int(np.argmin(f))
        ax.scatter(g[i], f[i], s=150, marker="*", color=c["series"][3], edgecolors=c["ink"], linewidths=0.6, zorder=3)
        ax.annotate("메모리 뱅크 기준 최적 설정", (g[i], f[i]), xytext=(g[i] + 0.12, f[i] - 0.1), fontsize=10,
                    color=c["ink2"], arrowprops=dict(arrowstyle="-", color=c["muted"]))
        tau = s[pid]["pairs"]["fpfh_bank~cad_distance"]["kendall_tau"]
        ok = s[pid]["robust"]
        ax.set_title(f"{SHORT[pid]}  순위 일치도 {tau:.2f} → {'강건' if ok else '검출기에 따라 달라짐'}", color=c["ink"])
        ax.set_xlabel("설계도 거리 검출기의 미검률")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    axs[0].set_ylabel("메모리 뱅크 검출기의 미검률")
    axs[0].yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    fig.suptitle("같은 설정 40개를 두 검출기로 평가 (대각선 위 = 메모리 뱅크가 더 놓침)", color=c["ink"], fontweight="bold")
    fig.tight_layout()
    save(fig, "detector_robustness", theme)


def fig_noise(theme: str) -> None:
    plt, c = setup(theme)
    nl = json.loads((EXP / "e7" / "noise_levels.json").read_text())
    levels = [k for k in ("N0", "N1", "N2", "N3") if all(k in nl[p] for p in nl)]
    lab = {"N0": "잡음 없음", "N1": "약", "N2": "기본(중)", "N3": "강"}
    fig, ax = plt.subplots(figsize=size(theme, 7.8, 4.8))
    for k, pid in enumerate(["tless_13", "tless_29"]):
        y = [nl[pid][lv]["mean_fnr"] for lv in levels]
        ax.plot(range(len(levels)), y, color=c["series"][k], marker=MARKERS[k], markersize=8, label=SHORT[pid])
        ax.text(len(levels) - 1 + 0.08, y[-1], SHORT[pid], color=c["ink2"], va="center", fontsize=11)
        if "N2MS" in nl[pid]:
            ax.scatter([levels.index("N2")], [nl[pid]["N2MS"]["mean_fnr"]], marker="x", s=70, color=c["series"][k])
    ax.set_xticks(range(len(levels)), [lab[lv] for lv in levels])
    ax.set_xlim(-0.3, len(levels) - 0.3 + 0.6)
    ax.set_ylim(0, 1)
    ax.set_xlabel("센서 잡음 수준 (× = 기본 수준 + 노출 10배 다중 촬영)")
    ax.set_ylabel("설정 40개의 평균 미검률")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    ax.set_title("잡음이 커질수록 미검률이 단조롭게 커진다")
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    save(fig, "noise_levels", theme)


def fig_sensor_gap(theme: str) -> None:
    plt, c = setup(theme)
    s = json.loads((EXP / "e5b" / "summary.json").read_text())["dowel"]
    conds = [("real", "실제 스캔 (Zivid)"), ("N0", "다시 찍기: 잡음 없음"), ("N1@native", "약한 잡음 (실제 거리 기준)"),
             ("N2@native", "기본 잡음 (실제 거리 기준)"), ("N3@native", "강한 잡음 (실제 거리 기준)"),
             ("N2", "기본 잡음 (정의대로 0.55 m로 늘림)")]
    fig, ax = plt.subplots(figsize=size(theme, 9.5, 5.0))
    y = np.arange(len(conds))[::-1]
    h = 0.36
    for k, (det, lab) in enumerate((("btf_fpfh", "공개 기준 검출기 (BTF)"), ("isco_fpfh", "ISCO 검출기"))):
        vals = [s["detection"][cd][det]["auroc_test"] for cd, _ in conds]
        ax.barh(y + (h / 2 if k == 0 else -h / 2), vals, height=h - 0.04, color=c["series"][k], label=lab)
        for yy, v in zip(y, vals):
            ax.text(v + 0.006, yy + (h / 2 if k == 0 else -h / 2), f"{v:.2f}", va="center", fontsize=10, color=c["ink2"])
    ax.axvline(0.5, color=c["axis"], lw=1)
    ax.text(0.505, y[0] + 0.45, "0.5 = 동전 던지기", fontsize=10, color=c["muted"])
    ax.set_yticks(y, [lab for _, lab in conds])
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("정상·불량 분리 능력 (1이면 완벽, 0.5면 무작위)")
    ax.set_title("MVTec 3D-AD dowel(벽 앵커) 실제 스캔을 시뮬레이터 센서로 다시 찍었을 때")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    save(fig, "sensor_gap", theme)


def fig_search(theme: str) -> None:
    plt, c = setup(theme)
    e8 = json.loads((WORK / "e8_interim.json").read_text())
    names = {"bo_warm": "베이지안 최적화 (1단계 후보로 시작)", "bo_cold": "베이지안 최적화 (처음부터)",
             "nsga2": "유전 알고리즘", "random": "무작위"}
    fig, ax = plt.subplots(figsize=size(theme, 8.5, 4.8))
    cur = e8["tless_13"]["curves_median"]
    for k, (m, lab) in enumerate(names.items()):
        y = np.array(cur[m])
        ax.plot(np.arange(1, len(y) + 1), y, color=c["series"][k], label=lab, marker=MARKERS[k], markevery=8,
                markersize=6)
    ax.axvline(30, color=c["axis"], lw=1, ls="--")
    ax.text(30.5, 0.05, "평가 30회 시점", color=c["muted"], fontsize=10)
    ax.set_xlabel("검출기로 평가한 설정 수")
    ax.set_ylabel("찾아낸 성능–비용 곡선의 넓이 (클수록 좋음)")
    ax.set_ylim(0, 0.75)
    ax.set_title("T-LESS 13번: 같은 평가 횟수에서 탐색 방법별로 찾은 설정의 질 (시드 2개 중앙값)")
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    save(fig, "search_efficiency", theme)


def fig_rigs(theme: str) -> None:
    """Top view of each part's recommended rig on the camera hemisphere."""
    plt, c = setup(theme)
    from isco.geometry.camera_model import SlotGrid

    g = SlotGrid()
    fig, axs = plt.subplots(2, 3, figsize=size(theme, 12.5, 8.2), subplot_kw={"projection": "polar"})
    for ax, pid in zip(axs.ravel(), PARTS):
        rec = best_record(pid)
        cfg = rec["config"]
        ax.set_theta_zero_location("E")
        ax.set_ylim(0, 90)
        ax.set_yticks([10, 30, 50, 70], ["80°", "60°", "40°", "20°"], fontsize=8, color=c["muted"])
        ax.set_xticks(np.radians(np.arange(0, 360, 90)), ["0°", "90°", "180°", "270°"], fontsize=9)
        ax.grid(color=c["grid"])
        for k in range(g.n_slots):
            s = g.slot(k)
            ax.scatter(np.radians(s.azimuth_deg), 90 - s.elevation_deg, s=6, color=c["axis"], zorder=1)
        for i, k in enumerate(cfg["slots"]):
            s = g.slot(k)
            ax.scatter(np.radians(s.azimuth_deg), 90 - s.elevation_deg, s=160 if s.distance_m < 0.3 else 90,
                       color=c["series"][i], marker=MARKERS[i], edgecolors=c["surface"], zorder=3)
        ax.annotate("", xy=(np.radians(0), 88), xytext=(np.radians(180), 88),
                    arrowprops=dict(arrowstyle="->", color=c["muted"], lw=0.8))
        rot = "회전대 없음" if cfg["rotation"] == "none" else f"회전대 {KO_ROT[cfg['rotation']]}"
        ax.set_title(f"{KO_PART[pid]}\n카메라 {len(cfg['slots'])}대 · {rot} · {SPEEDS[cfg['speed_level']]} m/s\n"
                     f"사이클 {rec['objectives']['cycle_time_s']:.1f}초 · 미검률 {rec['report']['fnr'] * 100:.0f}%",
                     fontsize=11, color=c["ink"], pad=14)
    fig.suptitle("부품별 권장 검사 설정 (위에서 본 카메라 위치: 중심 = 바로 위, 바깥 = 낮은 각도, 큰 표시 = 0.25 m, 작은 표시 = 0.40 m,"
                 " 화살표 = 벨트 진행 방향)", color=c["ink"], fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "recommended_rigs", theme)


def fig_types(theme: str) -> None:
    """Miss rate by defect type under each part's recommended configuration."""
    plt, c = setup(theme)
    fig, ax = plt.subplots(figsize=size(theme, 10.5, 4.8))
    M = np.array([[best_record(p)["report"]["fnr_by_type"][t] for t in TYPES] for p in PARTS])
    w = 0.13
    for k, pid in enumerate(PARTS):
        ax.bar(np.arange(len(TYPES)) + (k - 2.5) * w, M[k], width=w - 0.015, color=c["series"][k], label=SHORT[pid])
    mean = M.mean(axis=0)
    for i, m in enumerate(mean):
        ax.text(i, 1.04, f"평균 {m * 100:.0f}%", ha="center", fontsize=10.5, color=c["ink"], fontweight="bold")
    ax.set_xticks(range(len(TYPES)), [KO_TYPE[t] for t in TYPES], fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("미검률 (결함 6개 중 놓친 비율)")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=6, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.set_title("권장 설정에서 결함 종류별로 놓친 비율 — 구멍·절단은 잘 잡고 균열·휨·눌림은 어렵다")
    fig.tight_layout()
    save(fig, "miss_by_type", theme)


def fig_timeline(theme: str) -> None:
    plt, c = setup(theme)
    pat = re.compile(r"\[run_experiments\] (\S+) exit (\d+) (.+)$")
    pat_s = re.compile(r"\[run_experiments\] === (\S+) === (.+)$")
    start, spans = {}, []
    for line in open(ROOT / "runs" / "logs" / "run_experiments.log"):
        line = line.strip()
        m = pat_s.match(line)
        if m:
            start[m.group(1)] = datetime.strptime(m.group(2), "%a %b %d %I:%M:%S %p UTC %Y")
            continue
        m = pat.match(line)
        if m and m.group(1) in start:
            spans.append((m.group(1), start[m.group(1)], datetime.strptime(m.group(3), "%a %b %d %I:%M:%S %p UTC %Y"),
                          int(m.group(2))))
    names = {"e1": "부품 간 교차 적용",
             "e2": "가시성 대리지표 검증", "e3": "카메라 추가 효과", "e4": "정지 vs 인라인", "real": "실스캔 세트 구성",
             "e5": "실스캔 순위 유지 (중단)", "e6": "검출기 교체", "e7": "잡음 수준 비교", "e7b": "부품 놓임 방향"}
    keep = [(n, a, b, e) for n, a, b, e in spans if n in names and (b - a).total_seconds() > 1800]
    seen, rows = set(), []
    for n, a, b, e in keep[::-1]:
        if n not in seen:
            seen.add(n)
            rows.append((n, a, b, e))
    rows = rows[::-1]
    now = datetime.utcnow()
    if "e8" in start and not any(n == "e8" for n, *_ in spans):
        rows.append(("e8", start["e8"], now, -1))
        names["e8"] = "탐색 방법 효율"
    fig, ax = plt.subplots(figsize=size(theme, 11, 5.6))
    t0 = min(a for _, a, _, _ in rows)
    for i, (n, a, b, e) in enumerate(rows):
        x0 = (a - t0).total_seconds() / 3600
        dx = (b - a).total_seconds() / 3600
        col = c["series"][0] if e == 0 else (c["series"][3] if e < 0 else c["critical"])
        ax.barh(i, dx, left=x0, color=col, height=0.6)
        ax.text(x0 + dx + 1, i, f"{names[n]}  {dx:.0f}시간" + (" (실행 중)" if e < 0 else ""), va="center",
                fontsize=10.5, color=c["ink2"])
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xlabel(f"시작({t0:%m-%d %H시} UTC)부터 경과 시간 (시간)")
    ax.set_xlim(0, (now - t0).total_seconds() / 3600 + 40)
    ax.grid(axis="y", visible=False)
    ax.set_title("검증 실험 대기열 진행 기록 (파랑 = 완료, 노랑 = 실행 중, 빨강 = 오류로 중단)")
    fig.tight_layout()
    save(fig, "timeline", theme)


def main() -> None:
    for theme in ("light", "dark"):
        for fn in (fig_cross, fig_visibility, fig_camera_addition, fig_static_inline, fig_detectors, fig_noise,
                   fig_sensor_gap, fig_search, fig_rigs, fig_types, fig_timeline):
            fn(theme)
            print(f"[figures] {theme} {fn.__name__}", flush=True)


if __name__ == "__main__":
    main()
