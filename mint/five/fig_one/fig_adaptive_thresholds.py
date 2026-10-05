"""Supplementary "adaptive thresholds" figure: ROC curves across outcome cutoffs.

Two panels of AUROC/ROC curves that show how MINT's discrimination shifts as the
clinical definition of the outcome is tightened or loosened:

    A. Hypoxia          -- SpO2 cutoffs 88 / 90 / 92 / 94
    B. Oxygen support   -- escalating support intensity (r1, r2, r4, PPV, ventilator)

Each curve is one softmax prediction CSV; the AUROC point estimate and 95% CI in
the legend are read verbatim from the per-directory ``results_softmax_test.jsonl``
(the same numbers reported elsewhere -- nothing is recomputed here).

RUN (from repo root):
    python -m mint.five.fig_one.fig_adaptive_thresholds \
        --final_dir artifacts/fig1-final \
        --output_dir artifacts/fig1_nejm_adaptive

OUTPUT (in --output_dir):
    fig_adaptive_thresholds_auroc.png   1x2 ROC-curve grid (hypoxia | oxygen support)

INPUT LAYOUT (under --final_dir)
    fig1-multi-hypoxia/{task}_softmax.csv      hypoxia88 / hypoxia90 / hypoxia92 / hypoxia94
    fig1-multi-resp/{task}_softmax.csv         r1 / r2 / r4 / ppv / ventilator
    fig1-multi-*/results_softmax_test.jsonl    AUROC point estimates + CIs, incidence, n
    Every CSV must carry ``probs`` and ``labels`` columns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve
from tap import tapify


plt.style.use(str(Path(__file__).resolve().parents[1] / "design-skill" / "nature.mplstyle"))
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    }
)

BASELINE_COLOR = "#B8B8B8"
TITLE_FS = 12
AXIS_LABEL_FS = 11
TICK_FS = 10
LEGEND_FS = 8

# Ordered thresholds per panel. Curves are drawn (and legended) in this order,
# from the most permissive definition to the most stringent. Labels follow the
# clinical reading requested for the figure.
HYPOXIA_TASKS = [
    ("hypoxia88", "SpO$_2$ ≤ 88%"),
    ("hypoxia90", "SpO$_2$ ≤ 90%"),
    ("hypoxia92", "SpO$_2$ ≤ 92%"),
    ("hypoxia94", "SpO$_2$ ≤ 94%"),
]
RESP_TASKS = [
    ("r1", "Any respiratory support"),
    ("r2", "Nasal cannula or more"),
    ("r4", "Mask or more intensive"),
    ("ppv", "Positive pressure or more intensive"),
    ("ventilator", "Ventilator"),
]

PANELS = [
    ("Hypoxia", "fig1-multi-hypoxia", HYPOXIA_TASKS),
    ("Respiratory support", "fig1-multi-resp", RESP_TASKS),
]


@dataclass
class Args:
    final_dir: str = "artifacts/fig1-final"
    output_dir: str = "artifacts/fig1_nejm_adaptive"
    dpi: int = 600


def _load_summary(subdir: Path) -> dict[str, dict]:
    """AUROC point estimates + CIs (and incidence/n) keyed by task."""
    path = subdir / "results_softmax_test.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {rec["task"]: rec for rec in records}


def _load_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """False-positive-rate / true-positive-rate for one prediction CSV."""
    df = pd.read_csv(path)
    for col in ("probs", "labels"):
        if col not in df.columns:
            raise RuntimeError(f"column '{col}' missing from {path}")
    fpr, tpr, _ = roc_curve(df["labels"].to_numpy(dtype=int), df["probs"].to_numpy(dtype=float))
    return fpr, tpr


def _colors(n: int) -> list:
    """Sequential viridis samples -- ordered thresholds read as a progression and
    the ramp is perceptually uniform + colorblind-safe. Cap short of the pale end."""
    return [plt.cm.viridis(x) for x in np.linspace(0.0, 0.85, n)]


def _draw_panel(ax, subdir: Path, tasks: list[tuple[str, str]], summary: dict[str, dict]) -> None:
    ax.plot([0, 1], [0, 1], color=BASELINE_COLOR, lw=0.8, ls="--", zorder=1)
    colors = _colors(len(tasks))
    handles = []
    for (task, label), color in zip(tasks, colors):
        csv_path = subdir / f"{task}_softmax.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing prediction CSV: {csv_path}")
        if task not in summary:
            raise KeyError(f"No summary record for task '{task}' in {subdir}")
        fpr, tpr = _load_curve(csv_path)
        ax.plot(fpr, tpr, color=color, lw=1.8, zorder=3)
        rec = summary[task]
        auroc = rec["softmax_auroc"]
        ci = rec["softmax_auroc_ci"]
        handles.append(
            Line2D([0], [0], color=color, lw=1.8,
                   label=f"{label}: {auroc:.3f} ({ci[0]:.3f}–{ci[1]:.3f})")
        )
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=LEGEND_FS,
              handlelength=1.3, labelspacing=0.35, borderaxespad=0.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def main() -> None:
    args = tapify(Args)
    final_dir = Path(args.final_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.5), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.13, wspace=0.20)

    for ax, (title, sub, tasks), letter in zip(axes, PANELS, "ab"):
        subdir = final_dir / sub
        summary = _load_summary(subdir)
        _draw_panel(ax, subdir, tasks, summary)
        ax.set_title(title, fontsize=TITLE_FS, pad=6)
        ax.set_xlabel("False positive rate", fontsize=AXIS_LABEL_FS)
        ax.tick_params(axis="both", labelsize=TICK_FS, length=2)
        ax.text(-0.15, 1.05, letter, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")
    axes[0].set_ylabel("True positive rate", fontsize=AXIS_LABEL_FS)

    out_path = output_dir / "fig_adaptive_thresholds_auroc.png"
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
