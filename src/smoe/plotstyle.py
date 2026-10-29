"""Figure style for the paper (static PDF). Colours follow the dataviz reference palette,
first three categorical slots (validated all-pairs for CVD); experts keep the same colour everywhere."""

import matplotlib as mpl
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
EXPERT_COLORS = {"trend": SERIES[0], "reversal": SERIES[1], "crisis": SERIES[2]}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"


def use():
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "lines.linewidth": 1.2, "legend.frameon": False, "legend.fontsize": 8,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
