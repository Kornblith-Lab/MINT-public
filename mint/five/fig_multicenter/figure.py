"""Multicenter external validation composite figure.

Layout (2x2 grid):
  Panel A: top row, both columns -- grouped dot plot of paired AUROC deltas
           (MINT zero-shot minus each XGBoost baseline) by site and outcome
  Panel B: bottom-left  -- AUROC gain over the locally-trained XGBoost against
           the number of positive cases, coloured by outcome
  Panel C: bottom-right -- reader-study ROC curves for MINT, GPT-5.5,
           physicians with MINT, and physicians alone

Usage:
    uv run --with pandas --with numpy --with scipy --with matplotlib --with scikit-learn --with openpyxl python -m mint.five.fig_multicenter
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[3]
STYLE_PATH = Path(__file__).resolve().parents[1] / "design-skill" / "nature.mplstyle"
INPUT_DIR = ROOT / "multicenter"
READER_DIR = ROOT / "artifacts" / "fig1" / "human_review"
SAVE_DIR = ROOT / "artifacts" / "multicenter_analysis_v2"
PNG_DPI = 1200

SITE_ORDER = ["UC 1", "UC 2", "UC 3", "UC 4", "UC 5"]
OUTCOME_ORDER = ["hypoxia", "tachypnea", "periarrest", "tachycardia", "hypotension"]
OUTCOME_LABELS = {
    "hypoxia": "Hypoxia",
    "tachypnea": "Tachypnea",
    "periarrest": "Peri-arrest",
    "tachycardia": "Tachycardia",
    "hypotension": "Hypotension",
}
# Panel B outcome palette. Deliberately avoids the navy and magenta that encode
# the two contrasts in panel A, and pairs each colour with a distinct marker so
# the categories survive greyscale and colour-vision deficiency.
OUTCOME_COLORS = {
    "hypoxia": "#CC79A7",
    "tachypnea": "#56B4E9",
    "periarrest": "#D55E00",
    "tachycardia": "#7570B3",
    "hypotension": "#E69F00",
}
OUTCOME_MARKERS = {
    "hypoxia": "o",
    "tachypnea": "s",
    "periarrest": "D",
    "tachycardia": "^",
    "hypotension": "v",
}

# ``n_pos`` is small-cell suppressed as "<10" for one site-outcome pair; it is
# plotted at the suppression threshold.
CENSORED_N_POS = 10.0

# Paired contrasts, each MINT zero-shot minus one XGBoost baseline.
CONTRAST_ORDER = ["GlobalXGBoost", "XGBoost"]
CONTRAST_COLORS = {
    "GlobalXGBoost": "#1D5B7E",
    "XGBoost": "#DE369D",
}
CONTRAST_LABELS = {
    "GlobalXGBoost": "MINT − UCSF-trained XGBoost",
    "XGBoost": "MINT − locally-trained XGBoost",
}

# Panel C: the 100-case reader study, four cohorts of 25 read by different
# physicians. ``gpt_5`` is the gpt-5.5 comparator (see fig_one/comparator/llm.py).
READER_COHORTS = ["A", "B", "C", "D"]
READER_ORDER = ["mint", "gpt_5", "human_mint", "human_alone"]
READER_LABELS = {
    "mint": "MINT",
    "gpt_5": "GPT-5.5",
    "human_mint": "Physicians + MINT",
    "human_alone": "Physicians alone",
}
# Physicians with and without MINT share a hue and split on dash so the panel
# reads as one paired contrast plus two standalone models.
READER_COLORS = {
    # Same green as MINT (zero-shot) in fig1_composite.png (_OKABE["green"]).
    "mint": "#009E73",
    "gpt_5": "#000000",
    "human_mint": "#D55E00",
    "human_alone": "#D55E00",
}
READER_DASHES = {
    "mint": "-",
    "gpt_5": "-",
    "human_mint": "-",
    "human_alone": (0, (3, 1.6)),
}
# The curves are averaged over cohorts at fixed false-positive rate, which makes
# the area under each one the macro AUROC that comparator_metrics.csv reports.
FPR_GRID = np.linspace(0.0, 1.0, 501)

FONT_SIZES = {
    "panel": 9,
    "facet": 7.5,
    "label": 8,
    "tick": 6.5,
    "legend": 7,
}

DODGE = 0.19
YLIM = (-0.25, 0.85)
YTICKS = [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8]


def _load_deltas() -> pd.DataFrame:
    """Paired AUROC deltas as MINT zero-shot minus each XGBoost baseline.

    ``bootstrap_diffs_summary.csv`` stores ``method`` minus ``target`` with
    ``target == "Softmax"``, so the sign is flipped and the CI bounds swapped.
    """
    df = pd.read_csv(INPUT_DIR / "bootstrap_diffs_summary.csv")
    df = df[(df["target"] == "Softmax") & (df["method"].isin(CONTRAST_ORDER))].copy()
    df["delta"] = -df["auroc_diff"]
    df["delta_lo"] = -df["auroc_diff_ci_hi"]
    df["delta_hi"] = -df["auroc_diff_ci_lo"]
    expected = len(SITE_ORDER) * len(OUTCOME_ORDER) * len(CONTRAST_ORDER)
    if len(df) != expected:
        raise ValueError(f"expected {expected} delta rows, found {len(df)}")
    return df


def _panel_a(axes: list[plt.Axes], df: pd.DataFrame) -> None:
    """Grouped dot plot of paired AUROC deltas, faceted by outcome."""
    for j, (ax, outcome) in enumerate(zip(axes, OUTCOME_ORDER)):
        sub = df[df["outcome"] == outcome]

        for y in YTICKS:
            ax.axhline(y, color="#EDEDED", lw=0.4, zorder=0)
        for x in range(len(SITE_ORDER) - 1):
            ax.axvline(x + 0.5, color="#F0F0F0", lw=0.4, zorder=0)
        ax.axhline(0.0, color="#4D4D4D", lw=0.6, ls=(0, (3, 2)), zorder=1)

        for i, site in enumerate(SITE_ORDER):
            for k, method in enumerate(CONTRAST_ORDER):
                row = sub[(sub["hospital"] == site) & (sub["method"] == method)]
                if row.empty:
                    continue
                row = row.iloc[0]
                x = i + (k - 0.5) * 2 * DODGE
                color = CONTRAST_COLORS[method]
                ax.plot(
                    [x, x],
                    [row["delta_lo"], row["delta_hi"]],
                    color=color,
                    lw=0.9,
                    solid_capstyle="butt",
                    zorder=2,
                )
                ax.plot(
                    x,
                    row["delta"],
                    marker="o",
                    ms=3.2,
                    color=color if row["auroc_sig"] else "white",
                    mec=color if row["auroc_sig"] else color,
                    mew=0.5 if row["auroc_sig"] else 0.9,
                    zorder=3,
                )

        ax.set_xlim(-0.55, len(SITE_ORDER) - 0.45)
        ax.set_ylim(*YLIM)
        ax.set_xticks(range(len(SITE_ORDER)))
        ax.set_xticklabels([s.replace(" ", "") for s in SITE_ORDER], fontsize=FONT_SIZES["tick"])
        ax.set_title(OUTCOME_LABELS[outcome], fontsize=FONT_SIZES["facet"], pad=3)
        ax.tick_params(axis="both", labelsize=FONT_SIZES["tick"])
        ax.set_yticks(YTICKS)

        if j == 0:
            ax.set_ylabel("Δ AUROC (95% CI)", fontsize=FONT_SIZES["label"])
        else:
            ax.set_yticklabels([])
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)

    axes[2].set_xlabel("Site", fontsize=FONT_SIZES["label"])

    axes[0].annotate(
        "MINT better", xy=(0.02, 0.985), xycoords="axes fraction",
        fontsize=FONT_SIZES["tick"], color="#4D4D4D", va="top", ha="left",
    )
    axes[0].annotate(
        "MINT worse", xy=(0.02, 0.015), xycoords="axes fraction",
        fontsize=FONT_SIZES["tick"], color="#4D4D4D", va="bottom", ha="left",
    )


def _panel_a_legend(fig: plt.Figure, y: float) -> None:
    """Figure-level legend placed in the gap below the panel A facet strip.

    Anchored in figure coordinates so constrained_layout does not reserve
    width for it and squeeze the facets. ``y`` is measured from the rendered
    panel A axes by the caller so the legend cannot land on the x-axis label.
    """
    handles = [
        Line2D([], [], marker="o", ls="none", ms=3.2, color=CONTRAST_COLORS[m],
               mec="white", mew=0.5, label=CONTRAST_LABELS[m])
        for m in CONTRAST_ORDER
    ]
    handles.append(
        Line2D([], [], marker="o", ls="none", ms=3.2, color="white",
               mec="#4D4D4D", mew=0.9, label="95% CI includes 0")
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        bbox_transform=fig.transFigure,
        ncol=len(handles),
        fontsize=FONT_SIZES["legend"],
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.8,
        borderpad=0.0,
    )


def _load_positives() -> pd.DataFrame:
    """Positive-case count per site-outcome pair, with the suppressed cell flagged."""
    df = pd.read_csv(INPUT_DIR / "bootstrap_summary.csv")
    df = df[["hospital", "outcome", "n_pos"]].drop_duplicates(["hospital", "outcome"])
    df["censored"] = df["n_pos"] == "<10"
    df["n_pos_plot"] = pd.to_numeric(
        df["n_pos"].replace("<10", str(CENSORED_N_POS)), errors="raise"
    )
    return df


def _panel_b(ax: plt.Axes, deltas: pd.DataFrame, positives: pd.DataFrame) -> None:
    """AUROC gain over the locally-trained XGBoost against positive-case count."""
    sub = deltas[deltas["method"] == "XGBoost"].merge(
        positives, on=["hospital", "outcome"], how="inner"
    )
    expected = len(SITE_ORDER) * len(OUTCOME_ORDER)
    if len(sub) != expected:
        raise ValueError(f"expected {expected} points in panel B, found {len(sub)}")

    ax.axhline(0.0, color="#BFBFBF", lw=0.5, zorder=1)

    for outcome in OUTCOME_ORDER:
        pts = sub[sub["outcome"] == outcome]
        ax.scatter(
            pts["n_pos_plot"],
            pts["delta"],
            s=24,
            marker=OUTCOME_MARKERS[outcome],
            color=OUTCOME_COLORS[outcome],
            edgecolor="white",
            linewidth=0.4,
            label=OUTCOME_LABELS[outcome],
            zorder=3,
        )

    lx = np.log10(sub["n_pos_plot"].to_numpy(dtype=float))
    y = sub["delta"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(lx, y, 1)
    x_lo, x_hi = 8.0, 14_000.0
    grid = np.logspace(np.log10(x_lo), np.log10(x_hi), 200)
    ax.plot(grid, slope * np.log10(grid) + intercept, color="#444444", lw=1.0, ls="--", zorder=2)

    # Spearman is reported rather than Pearson: it is invariant both to the log
    # transform and to the value imputed for the suppressed n_pos cell.
    rho = float(spearmanr(sub["n_pos_plot"], y).statistic)
    ax.annotate(
        f"Spearman $\\rho$ = {rho:.2f}",
        xy=(0.035, 0.04),
        xycoords="axes fraction",
        fontsize=FONT_SIZES["legend"],
        va="bottom",
        ha="left",
    )

    ax.set_xscale("log")
    ax.set_xlim(x_lo, x_hi)
    ax.set_xticks([10, 100, 1_000, 10_000])
    ax.set_xticklabels(["10", "100", "1,000", "10,000"])
    ax.set_xlabel("Positive cases (log scale)", fontsize=FONT_SIZES["label"])
    # Same quantity as the panel A magenta series; label it from the same source
    # so the two panels cannot drift apart.
    ax.set_ylabel(f"Δ AUROC ({CONTRAST_LABELS['XGBoost']})", fontsize=FONT_SIZES["label"])
    ax.set_ylim(-0.25, 0.7)
    ax.set_yticks([-0.2, 0.0, 0.2, 0.4, 0.6])
    ax.tick_params(axis="both", labelsize=FONT_SIZES["tick"])
    ax.minorticks_off()
    ax.legend(
        title="Outcome",
        title_fontsize=FONT_SIZES["legend"],
        fontsize=FONT_SIZES["legend"],
        frameon=False,
        loc="upper right",
        handletextpad=0.3,
        labelspacing=0.35,
        borderpad=0.0,
    )


def _load_reader_auroc() -> pd.DataFrame:
    """Macro AUROC and 95% CI per method from the reader-study metrics table."""
    df = pd.read_csv(READER_DIR / "comparator_metrics.csv")
    df = df[(df["analysis"] == "macro_absolute") & (df["metric"] == "AUROC")]
    df = df.set_index("method")[["value", "ci_low", "ci_high"]]
    missing = [m for m in READER_ORDER if m not in df.index]
    if missing:
        raise ValueError(f"comparator_metrics.csv is missing macro AUROC for {missing}")
    return df


def _load_reader_sig() -> set[str]:
    """Methods whose macro AUROC differs from MINT's (paired 95% CI excludes 0)."""
    df = pd.read_csv(READER_DIR / "comparator_metrics.csv")
    df = df[(df["analysis"] == "macro_delta") & (df["reference"] == "mint")
            & (df["metric"] == "AUROC")]
    return set(df.loc[df["significant"] == True, "method"])  # noqa: E712


def _mean_roc(scores: pd.DataFrame, method: str) -> np.ndarray:
    """Sensitivity at each grid FPR, averaged over the four reader cohorts."""
    tprs = []
    for cohort in READER_COHORTS:
        sub = scores[scores["subgroup"] == cohort]
        fpr, tpr, _ = roc_curve(sub["label"].to_numpy(), sub[method].to_numpy())
        tprs.append(np.interp(FPR_GRID, fpr, tpr))
    return np.mean(tprs, axis=0)


def _panel_c(ax: plt.Axes) -> None:
    """Reader-study ROC curves, labelled with the macro AUROC of each method."""
    auroc = _load_reader_auroc()
    sig = _load_reader_sig()
    scores = pd.read_excel(READER_DIR / "ppv.xlsx", sheet_name="model")
    if sorted(scores["subgroup"].unique()) != READER_COHORTS:
        raise ValueError(f"expected reader cohorts {READER_COHORTS} in ppv.xlsx")

    ax.plot([0, 1], [0, 1], color="#BFBFBF", lw=0.5, ls=(0, (2, 2)), zorder=1)

    for method in READER_ORDER:
        tpr = _mean_roc(scores, method)
        # The plotted curve must carry the same number the legend quotes.
        area = float(np.sum(np.diff(FPR_GRID) * (tpr[1:] + tpr[:-1]) / 2))
        if abs(area - auroc.loc[method, "value"]) > 0.005:
            raise ValueError(
                f"{method}: curve area {area:.3f} disagrees with the reported "
                f"macro AUROC {auroc.loc[method, 'value']:.3f}"
            )
        row = auroc.loc[method]
        ax.plot(
            FPR_GRID,
            tpr,
            color=READER_COLORS[method],
            ls=READER_DASHES[method],
            lw=1.1,
            solid_joinstyle="round",
            label=f"{READER_LABELS[method]}: {row['value']:.3f} "
                  f"({row['ci_low']:.3f}–{row['ci_high']:.3f})"
                  f"{'*' if method in sig else ''}",
            zorder=2,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("1 − specificity", fontsize=FONT_SIZES["label"])
    ax.set_ylabel("Sensitivity", fontsize=FONT_SIZES["label"])
    ax.tick_params(axis="both", labelsize=FONT_SIZES["tick"])
    legend = ax.legend(
        title="AUROC (95% CI)",
        title_fontsize=FONT_SIZES["legend"],
        fontsize=FONT_SIZES["legend"],
        frameon=False,
        loc="lower right",
        markerfirst=False,
        handlelength=1.6,
        handletextpad=0.4,
        labelspacing=0.35,
        borderpad=0.0,
    )
    legend.set_alignment("right")


def main() -> None:
    plt.style.use(STYLE_PATH)
    df = _load_deltas()

    fig = plt.figure(figsize=(7.2, 5.4), constrained_layout=True)
    # hspace opens a band between the rows wide enough for the panel A legend,
    # which is placed into it from measured coordinates further down.
    outer = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.42, wspace=0.16)

    top = outer[0, :].subgridspec(1, len(OUTCOME_ORDER), wspace=0.10)
    axes_a = [fig.add_subplot(top[0, j]) for j in range(len(OUTCOME_ORDER))]
    _panel_a(axes_a, df)

    ax_b = fig.add_subplot(outer[1, 0])
    ax_c = fig.add_subplot(outer[1, 1])
    _panel_b(ax_b, df, _load_positives())
    _panel_c(ax_c)

    # The legend and the panel letters are figure-owned, not axes-owned:
    # constrained_layout folds axes text into each axes' bounding box, so a
    # letter pinned to the left margin would make the engine reserve that strip
    # and push the plots right. Draw once to settle the layout, freeze it, then
    # place both from measured coordinates -- a and b share the x of the leftmost
    # y-label so they align in a column, and the legend sits just under panel A's
    # lowest ink so it cannot land on the "Site" label.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_x = min(
        ax.yaxis.label.get_window_extent(renderer).x0 / fig.bbox.width
        for ax in (axes_a[0], ax_b)
    )
    panel_a_bottom = min(
        ax.get_tightbbox(renderer).y0 for ax in axes_a
    ) / fig.bbox.height

    fig.set_layout_engine("none")
    _panel_a_legend(fig, panel_a_bottom - 0.008)
    label_x_c = ax_c.yaxis.label.get_window_extent(renderer).x0 / fig.bbox.width
    for ax, letter, x in ((axes_a[0], "a", label_x), (ax_b, "b", label_x),
                          (ax_c, "c", label_x_c)):
        fig.text(
            x,
            ax.get_position().y1 + 0.012,
            letter,
            fontsize=FONT_SIZES["panel"],
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAVE_DIR / "fig_multicenter.png", dpi=PNG_DPI)
    fig.savefig(SAVE_DIR / "fig_multicenter.pdf")
    plt.close(fig)
    print(f"wrote {SAVE_DIR / 'fig_multicenter.png'}")


if __name__ == "__main__":
    main()
