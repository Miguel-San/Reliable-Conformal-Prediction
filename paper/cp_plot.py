"""Plot style of the figures of the paper.

Colour roles (checked with a colour-vision-deficiency validator):
- standard procedure: blue; new procedure: orange; C^min reference line: pink;
- true quantities (the true threshold t*, known in the synthetic case): grey dashed; calibration data: dark grey;
- where the new procedure is shown at several confidence levels, each level has its own hue (CONF_COL) and line
  style (LS_CONF).
"""
from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl

COL = dict(std="#1f77b4", new="#ff7f0e", cmin="#e377c2", oracle="0.35", cal="0.25")
LABEL = dict(std="Standard procedure", new="New procedure")
LS_CONF = {0.9: (0, (4, 1.5)), 0.95: "-", 0.99: (0, (1, 1))}
CONF_COL = {0.9: "#ff7f0e", 0.95: "#d62728", 0.99: "#6a3d9a"}  # distinct hues, darker with confidence
XLAB_N = "Calibration set size $N_{\\mathrm{cal}}$"
TEXT_WIDTH = 5.4  # inches; figures are designed at final size


@contextmanager
def style(fontsize=7):
    """Matplotlib rc context for paper figures (no global state is changed)."""
    rc = {"font.size": fontsize, "axes.titlesize": fontsize, "axes.labelsize": fontsize,
          "figure.labelsize": fontsize, "figure.titlesize": fontsize,
          "legend.fontsize": fontsize - 0.5, "xtick.labelsize": fontsize - 0.5, "ytick.labelsize": fontsize - 0.5,
          "lines.linewidth": 1.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
          "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "axes.grid": True, "grid.linewidth": 0.4,
          "grid.color": "0.9", "grid.linestyle": "-", "axes.spines.top": False, "axes.spines.right": False,
          "legend.frameon": True, "legend.handlelength": 2.2, "pdf.fonttype": 42, "ps.fonttype": 42,
          "savefig.dpi": 300, "figure.dpi": 150, "axes.titlelocation": "left"}
    with mpl.rc_context(rc):
        yield


def savefig(fig, stem, figdir="figures"):
    """Save the figure as <figdir>/<stem>.png at 300 dpi."""
    figdir = Path(figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / f"{stem}.png", dpi=300, bbox_inches="tight")
