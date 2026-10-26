"""Paper figures that do not depend on experiment results:
Fig. 1 system overview, Fig. 2 configuration space (slot hemisphere + process variables).
Result figures (Fig. 3-9) are written by the experiments into runs/exp/*.

    python scripts/make_paper_figures.py  ->  runs/paper_figures/
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from isco import config_loader as C
from isco.report import style

OUT = Path(__file__).resolve().parents[1] / "runs" / "paper_figures"


def fig1_overview() -> None:
    style.apply()
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    boxes = [
        (0.1, 1.9, 1.7, 1.2, "Part CAD /\n3D scan\n+ defect dictionary"),
        (2.1, 1.9, 1.9, 1.2, "Variant population\nnormal variation,\n6 geometric defects"),
        (4.3, 1.9, 2.0, 1.2, "Isaac Sim conveyor twin\nrender once per\ninstance x 96 slots"),
        (6.6, 1.9, 2.0, 1.2, "Stage 1: ray-cast\ndefect visibility\n(~0.3 ms / config)"),
        (8.9, 1.9, 2.0, 1.2, "Stage 2: detector in\nthe loop, MOTPE\n(FNR@1%FPR, n, T)"),
        (4.3, 0.2, 2.0, 1.2, "Sensor + process model\nnoise, dropout, blur,\ntrigger, turntable"),
        (6.6, 0.2, 2.0, 1.2, "Merge + ICP + FPFH\nmemory bank (normal\nparts only)"),
        (8.9, 0.2, 2.0, 1.2, "Pareto front +\nrecommended rig +\nper-type report"),
    ]
    for x, y, w, h, t in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor="#eef4fc", edgecolor=style.CAT[0], linewidth=1.2))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8.5, color=style.TEXT)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, color=style.TEXT_2, linewidth=1.0))

    arrow(1.8, 2.5, 2.1, 2.5)
    arrow(4.0, 2.5, 4.3, 2.5)
    arrow(6.3, 2.5, 6.6, 2.5)
    arrow(8.6, 2.5, 8.9, 2.5)
    arrow(5.3, 1.9, 5.3, 1.4)
    arrow(6.3, 0.8, 6.6, 0.8)
    arrow(9.9, 1.9, 9.9, 1.4)                        # stage 2 -> Pareto / report
    arrow(8.6, 1.2, 8.95, 1.9)                       # detector scores feed stage 2
    ax.text(8.75, 3.22, "shortlist", fontsize=7.5, color=style.TEXT_2, ha="center")
    fig.savefig(OUT / "fig1_system.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig2_config_space() -> None:
    style.apply()
    import matplotlib.pyplot as plt

    grid = C.grid()
    focus = np.array([0, 0, 0.01])
    fig = plt.figure(figsize=(10, 4.2))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    from isco.report.figures import ORDINAL5

    for k in range(grid.n_slots):
        s = grid.slot(k)
        e = grid.eye(k, focus)
        col = ORDINAL5[1] if s.i_dist == 0 else ORDINAL5[4]
        ax.scatter(*e, color=col, s=14, depthshade=False)
    u = np.linspace(-0.45, 0.45, 2)
    X, Y = np.meshgrid(np.linspace(-0.5, 0.5, 2), np.linspace(-0.2, 0.2, 2))
    ax.plot_surface(X, Y, np.zeros_like(X), color="#d9d8d3", alpha=0.6)
    ax.quiver(0, -0.3, 0.0, 0.25, 0, 0, color=style.CAT[1], linewidth=1.5, arrow_length_ratio=0.2)
    ax.text(0.12, -0.36, 0.0, "belt", color=style.TEXT_2, fontsize=8)
    ax.scatter(*focus, color=style.CAT[1], s=30)
    ax.set_box_aspect((1, 1, 0.55))
    ax.set_axis_off()
    ax.set_title("96 camera slots: 12 azimuths x 4 elevations x 2 distances", fontsize=9, loc="left")
    ax.scatter([], [], color=ORDINAL5[1], s=14, label="0.25 m")
    ax.scatter([], [], color=ORDINAL5[4], s=14, label="0.40 m")
    ax.legend(loc="upper right", fontsize=8)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis("off")
    rows = [("cameras", "1-6 distinct slots; >= 30 deg apart; >= 8 cm spacing"),
            ("distance", "0.25 m (0.086 mm/px) or 0.40 m (0.138 mm/px)"),
            ("rotation stage", "none, 90 x1, 180 x1, 90 x3 (turntable at focus)"),
            ("belt speed", "0.1 / 0.3 / 0.6 m/s (blur, trigger misregistration)"),
            ("trigger", "simultaneous or staggered"),
            ("objectives", "FNR@1%FPR, camera count, cycle time")]
    for i, (a, b) in enumerate(rows):
        ax2.text(0.0, 0.9 - i * 0.15, a, fontsize=9, color=style.TEXT, fontweight="bold")
        ax2.text(0.33, 0.9 - i * 0.15, b, fontsize=9, color=style.TEXT_2)
    fig.savefig(OUT / "fig2_config_space.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig1_overview()
    fig2_config_space()
    print(OUT)
