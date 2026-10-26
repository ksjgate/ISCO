"""Chart style tokens (reference palette of the data-viz method, light theme) and
matplotlib defaults: one axis, thin marks, recessive grid, text in ink colours."""
from __future__ import annotations

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
#: categorical slots, fixed order (never cycled)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
#: single-hue sequential/ordinal ramp (blue 250 -> 700)
BLUE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
DIVERGING = ("#2a78d6", "#f0efec", "#e34948")   # blue <-> gray <-> red
MUTED = "#a9a8a2"


def apply() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "text.color": TEXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
        "lines.linewidth": 2.0, "lines.markersize": 6, "font.size": 9, "axes.titleweight": "regular",
    })
    for f in ("Noto Sans CJK KR", "NanumGothic"):
        if f in {x.name for x in matplotlib.font_manager.fontManager.ttflist}:
            plt.rcParams["font.family"] = f
            break


def ordinal(n_levels: int) -> list[str]:
    """``n_levels`` steps from the ordinal blue ramp (light -> dark)."""
    if n_levels <= 1:
        return [BLUE_RAMP[4]]
    idx = [round(i * (len(BLUE_RAMP) - 1) / (n_levels - 1)) for i in range(n_levels)]
    return [BLUE_RAMP[i] for i in idx]


def diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("isco_div", list(DIVERGING))


def sequential_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("isco_seq", ["#cde2fb"] + BLUE_RAMP)
