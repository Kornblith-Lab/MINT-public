"""Supplementary method comparison for the multicenter external validation.

Compares all eight methods across the 25 site-outcome comparisons (5 external
sites x 5 physiologic deterioration outcomes), against both the locally-trained
XGBoost and MINT zero-shot references.

Running
-------
From the repository root::

    uv run --with pandas --with matplotlib python -m mint.five.fig_multicenter.fig3_comp

or, if the ``delphi`` conda environment is available::

    conda run -n delphi python -m mint.five.fig_multicenter.fig3_comp

No arguments and no model checkpoint required; everything is read from the
bootstrap CSVs already written by ``mint.five.multicenter.analysis``. Runs in a
few seconds and prints each method's volume correlation to stdout.

Inputs (``multicenter/``)
-------------------------
bootstrap_summary.csv                AUROC/AUPRC with 95% CI, per method x cell.
bootstrap_diffs_summary.csv          Visit-paired deltas, target = MINT zero-shot.
bootstrap_diffs_xgboost_summary.csv  Visit-paired deltas, target = local XGBoost.

Note that paired bootstraps exist only for those two anchors, so no method can
be tested against UCSF-trained XGBoost; adding that would require re-running the
bootstrap upstream.

Outputs (``artifacts/multicenter_analysis_v2/``)
-----------------------------------------------
Each figure is written as a vector PDF for submission and a 1200 dpi PNG.

supp_comp_heatmap        Figure X, full page. Paired delta-AUROC heatmaps, 25
                         rows x 6 methods, one panel per reference. Asterisk
                         marks a 95% CI excluding 0; colour saturates at
                         +/- DELTA_CLIP, shown by the colorbar arrowheads, while
                         the printed value is always exact.
supp_comp_determinants   Figure Y, three panels. (a) pairwise dominance matrix,
                         (b) dependence on available positive cases, (c) adaptation
                         ladder faceted by site.
supp_comp_table.csv      Reference table, 25 rows x 8 methods, AUROC (95% CI)
                         with case counts and incidence.

Conventions
-----------
Panel a and Figure X share one fixed method ordering (``FIGURE_METHODS``) with
MINT zero-shot pinned first. Panel b sorts independently by its own correlation
so the sign split reads directly, and so carries its own axis labels. Triage
logistic regression appears only in the reference table.
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
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[3]
STYLE_PATH = Path(__file__).resolve().parents[1] / "design-skill" / "nature.mplstyle"
INPUT_DIR = ROOT / "multicenter"
OUT_DIR = ROOT / "artifacts" / "multicenter_analysis_v2"
SAVE_DPI = 1200  # Nature's line-art maximum; PDF is vector regardless

SITE_ORDER = ["UC 1", "UC 2", "UC 3", "UC 4", "UC 5"]
OUTCOME_ORDER = ["hypoxia", "tachypnea", "periarrest", "tachycardia", "hypotension"]
OUTCOME_LABELS = {
    "hypoxia": "Hypoxia",
    "tachypnea": "Tachypnea",
    "periarrest": "Peri-arrest",
    "tachycardia": "Tachycardia",
    "hypotension": "Hypotension",
}

METHOD_LABELS = {
    "Softmax": "MINT zero-shot",
    "MINT_LR": "MINT probe",
    "MINT_LR_FT": "FT MINT probe",
    "ClassHead": "End-to-end FT MINT",
    "Softmax_FT": "FT MINT zero-shot",
    "XGBoost": "Local XGBoost",
    "GlobalXGBoost": "UCSF XGBoost",
    "Triage": "Triage LR",
}
# Shared ordering across both figures, MINT zero-shot pinned first, then
# increasing site-specific adaptation, then baselines. Triage appears only in
# the reference table.
FIGURE_METHODS = [
    "Softmax",
    "MINT_LR",
    "MINT_LR_FT",
    "ClassHead",
    "Softmax_FT",
    "XGBoost",
    "GlobalXGBoost",
]
TABLE_METHODS = FIGURE_METHODS + ["Triage"]

METHOD_COLORS = {
    "Softmax": "#0072B2",
    "MINT_LR": "#56B4E9",
    "MINT_LR_FT": "#8FD3F4",
    "ClassHead": "#CC79A7",
    "Softmax_FT": "#D55E00",
    "XGBoost": "#009E73",
    "GlobalXGBoost": "#7A9E4F",
    "Triage": "#9A9A9A",
}
SITE_COLORS = {
    "UC 1": "#0072B2",
    "UC 2": "#E69F00",
    "UC 3": "#009E73",
    "UC 4": "#CC79A7",
    "UC 5": "#D55E00",
}

# Increasing site-specific adaptation, for the ladder panel.
LADDER = ["Softmax", "MINT_LR", "Softmax_FT", "MINT_LR_FT", "ClassHead"]
LADDER_LABELS = {
    "Softmax": "Frozen\nzero-shot",
    "MINT_LR": "Frozen\n+ probe",
    "Softmax_FT": "Full FT\nzero-shot",
    "MINT_LR_FT": "Full FT\n+ probe",
    "ClassHead": "Full FT\nend-to-end",
}

# Deltas beyond this magnitude are clipped for color only; text shows true value.
DELTA_CLIP = 0.25
CENSORED_NPOS = 10.0  # matches multicenter/analysis.py handling of "<10"

FONT = {"panel": 9, "title": 8, "label": 7.5, "tick": 6, "cell": 5.4, "legend": 6.5}

DELTA_CMAP = LinearSegmentedColormap.from_list(
    "delta", ["#B2182B", "#E9A3A9", "#F7F7F7", "#9DC3DE", "#0B5D9E"]
)


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(INPUT_DIR / "bootstrap_summary.csv")
    diffs_mint = pd.read_csv(INPUT_DIR / "bootstrap_diffs_summary.csv")
    diffs_xgb = pd.read_csv(INPUT_DIR / "bootstrap_diffs_xgboost_summary.csv")
    return summary, diffs_mint, diffs_xgb


def _cells() -> list[tuple[str, str]]:
    return [(s, o) for s in SITE_ORDER for o in OUTCOME_ORDER]


def _auroc_matrix(summary: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    m = summary.pivot_table(index=["hospital", "outcome"], columns="method", values="auroc")
    return m.reindex(index=_cells())[methods]


def _npos(summary: pd.DataFrame) -> pd.Series:
    n = summary[summary["method"] == "Softmax"].set_index(["hospital", "outcome"])["n_pos"]
    n = pd.to_numeric(n.astype(str).str.replace("<10", "0", regex=False), errors="coerce")
    n = n.where(n > 0, CENSORED_NPOS)
    return n.reindex(_cells())


def _volume_rho(summary: pd.DataFrame) -> dict[str, float]:
    """Spearman rho between AUROC and positive cases, per method.

    Computed as Pearson correlation of the ranks, which is exactly Spearman and
    avoids a scipy dependency. Being rank-based it needs no log transform, and
    the censored UC 5 peri-arrest cell only has to rank lowest, not be exact.
    """
    m = _auroc_matrix(summary, FIGURE_METHODS)
    npos_rank = _npos(summary).rank().to_numpy(dtype=float)
    return {
        k: float(np.corrcoef(npos_rank, m[k].rank().to_numpy(dtype=float))[0, 1])
        for k in FIGURE_METHODS
    }


def _save(fig: plt.Figure, stem: str) -> None:
    """Vector PDF for submission plus a 1200 dpi PNG for drafts and slides."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def _row_labels() -> list[str]:
    return [f"{s} · {OUTCOME_LABELS[o]}" for s, o in _cells()]


LETTER_DX = -0.075  # offset left of the axes, shared so letters sit alike
LETTER_DY = 0.035


def _panel_letter(fig: plt.Figure, ax: plt.Axes, letter: str, x: float | None = None) -> float:
    """Place a panel letter and return its x, so other panels can align to it."""
    box = ax.get_position()
    x = box.x0 + LETTER_DX if x is None else x
    fig.text(x, box.y1 + LETTER_DY, letter, fontsize=FONT["panel"], fontweight="bold",
             va="top", ha="left")
    return x


# ------------------------------------------------------------------ reference table


def write_table(summary: pd.DataFrame) -> None:
    rows = []
    npos = _npos(summary)
    raw_npos = summary[summary["method"] == "Softmax"].set_index(["hospital", "outcome"])["n_pos"]
    for site, outcome in _cells():
        sub = summary[(summary["hospital"] == site) & (summary["outcome"] == outcome)].set_index("method")
        row = {
            "Site": site,
            "Outcome": OUTCOME_LABELS[outcome],
            "Cases": int(sub["n_total"].iloc[0]),
            "Positives": raw_npos.loc[(site, outcome)],
            "Incidence, %": round(100.0 * npos.loc[(site, outcome)] / sub["n_total"].iloc[0], 2),
        }
        for method in TABLE_METHODS:
            r = sub.loc[method]
            label = METHOD_LABELS[method]
            row[label] = f"{r['auroc']:.3f} ({r['auroc_ci_lo']:.3f}-{r['auroc_ci_hi']:.3f})"
        rows.append(row)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / "supp_comp_table.csv", index=False)


# --------------------------------------------------------------------- Figure X


def _delta_panel(
    ax: plt.Axes, diffs: pd.DataFrame, reference: str, title: str,
    order: list[str], show_ylabels: bool,
) -> plt.cm.ScalarMappable:
    methods = [m for m in order if m != reference]
    cells = _cells()
    grid = np.full((len(cells), len(methods)), np.nan)
    sig = np.zeros_like(grid, dtype=bool)
    idx = diffs.set_index(["hospital", "outcome", "method"])
    for i, key_cell in enumerate(cells):
        for j, method in enumerate(methods):
            key = (*key_cell, method)
            if key not in idx.index:
                continue
            grid[i, j] = idx.loc[key, "auroc_diff"]
            sig[i, j] = bool(idx.loc[key, "auroc_sig"])

    norm = Normalize(vmin=-DELTA_CLIP, vmax=DELTA_CLIP)
    im = ax.imshow(np.clip(grid, -DELTA_CLIP, DELTA_CLIP), cmap=DELTA_CMAP, norm=norm, aspect="auto")

    for i in range(len(cells)):
        for j in range(len(methods)):
            if np.isnan(grid[i, j]):
                continue
            shade = abs(np.clip(grid[i, j], -DELTA_CLIP, DELTA_CLIP)) / DELTA_CLIP
            label = f"{grid[i, j]:+.02f}".replace("+0.", "+.").replace("-0.", "−.")
            ax.text(
                j, i, label + ("*" if sig[i, j] else ""),
                ha="center", va="center", fontsize=FONT["cell"],
                color="white" if shade > 0.62 else "#222222",
            )

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=40, ha="right",
                       rotation_mode="anchor", fontsize=FONT["tick"])
    ax.set_yticks(range(len(cells)))
    if show_ylabels:
        ax.set_yticklabels(_row_labels(), fontsize=FONT["tick"])
    else:
        ax.set_yticklabels([])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for k in range(1, len(SITE_ORDER)):
        ax.axhline(k * len(OUTCOME_ORDER) - 0.5, color="white", lw=1.6)
    ax.set_title(title, fontsize=FONT["title"], pad=5)
    return im


def figure_x(diffs_mint: pd.DataFrame, diffs_xgb: pd.DataFrame, order: list[str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 7.4), constrained_layout=True)
    im = _delta_panel(axes[0], diffs_mint, "Softmax", "Δ AUROC vs MINT zero-shot", order, True)
    _delta_panel(axes[1], diffs_xgb, "XGBoost", "Δ AUROC vs locally-trained XGBoost", order, False)

    cbar = fig.colorbar(
        im, ax=axes, orientation="horizontal", extend="both",
        fraction=0.022, pad=0.015, aspect=45,
    )
    cbar.set_ticks([-0.2, -0.1, 0.0, 0.1, 0.2])
    cbar.set_ticklabels(["−0.20", "−0.10", "0", "+0.10", "+0.20"], fontsize=FONT["tick"])
    cbar.set_label(
        "Δ AUROC, method minus reference; *95% CI excludes 0",
        fontsize=FONT["legend"],
    )
    cbar.outline.set_visible(False)
    _save(fig, "supp_comp_heatmap")


# --------------------------------------------------------------------- Figure Y


def _dominance_panel(
    ax: plt.Axes, summary: pd.DataFrame, diffs_mint: pd.DataFrame,
    diffs_xgb: pd.DataFrame, order: list[str],
) -> None:
    m = _auroc_matrix(summary, order)
    n = len(order)
    wins = np.full((n, n), np.nan)
    for i, ri in enumerate(order):
        for j, rj in enumerate(order):
            if i != j:
                wins[i, j] = int((m[ri] > m[rj]).sum())

    # Significant wins are only computable for the two bootstrap anchors.
    sig_wins: dict[tuple[str, str], int] = {}
    for anchor, diffs in [("Softmax", diffs_mint), ("XGBoost", diffs_xgb)]:
        for method, g in diffs.groupby("method"):
            sig_wins[(method, anchor)] = int(((g["auroc_diff"] > 0) & g["auroc_sig"]).sum())
            sig_wins[(anchor, method)] = int(((g["auroc_diff"] < 0) & g["auroc_sig"]).sum())

    ax.imshow(wins, cmap="Blues", vmin=0, vmax=25, aspect="auto")
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=FONT["tick"], color="#BBBBBB")
                continue
            s = sig_wins.get((order[i], order[j]))
            txt = f"{int(wins[i, j])}" if s is None else f"{int(wins[i, j])}\n({s})"
            ax.text(j, i, txt, ha="center", va="center", fontsize=FONT["cell"],
                    color="white" if wins[i, j] > 15 else "#222222", linespacing=0.95)

    ax.set_xticks(range(n))
    ax.set_xticklabels([METHOD_LABELS[x] for x in order], rotation=40, ha="right",
                       rotation_mode="anchor", fontsize=FONT["tick"])
    ax.set_yticks(range(n))
    ax.set_yticklabels([METHOD_LABELS[x] for x in order], fontsize=FONT["tick"])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Loses to (column)", fontsize=FONT["label"])
    ax.set_title("Comparisons won of 25, by AUROC point estimate", fontsize=FONT["title"], pad=5)


def _volume_panel(ax: plt.Axes, rs: dict[str, float], order: list[str]) -> None:
    ypos = np.arange(len(order))
    ax.barh(ypos, [rs[k] for k in order], color=[METHOD_COLORS[k] for k in order], height=0.6)
    ax.axvline(0, color="#4D4D4D", lw=0.6)
    for y, k in zip(ypos, order):
        off = 0.04 if rs[k] >= 0 else -0.04
        ax.text(rs[k] + off, y, f"{rs[k]:+.2f}", va="center",
                ha="left" if rs[k] >= 0 else "right", fontsize=FONT["tick"], color="#333333")
    ax.set_yticks(ypos)
    ax.set_yticklabels([METHOD_LABELS[k] for k in order], fontsize=FONT["tick"])
    ax.set_ylim(len(order) - 0.5, -0.5)  # strongest correlation at the top
    ax.set_xlim(-0.95, 1.15)
    ax.set_xticks([-0.5, 0.0, 0.5, 1.0])
    ax.set_xlabel("Spearman ρ (absolute AUROC vs positive cases)", fontsize=FONT["label"])
    ax.tick_params(labelsize=FONT["tick"], length=0)
    ax.spines["left"].set_visible(False)
    # Positive counts vary more across outcomes than across sites, so this is a
    # dependence on available positives, not specifically on site volume.
    ax.set_title("Dependence on available positive cases", fontsize=FONT["title"], pad=5)
    ax.annotate("more data-dependent →", xy=(0.98, 0.02), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=FONT["legend"], color="#777777")


def _ladder_panel(axes: list[plt.Axes], summary: pd.DataFrame) -> None:
    m = _auroc_matrix(summary, FIGURE_METHODS)
    x = np.arange(len(LADDER))
    for ax, site in zip(axes, SITE_ORDER):
        sub = m.loc[site]
        for outcome in OUTCOME_ORDER:
            y = sub.loc[outcome, LADDER].to_numpy(dtype=float)
            ax.plot(x, y, color=SITE_COLORS[site], lw=0.7, alpha=0.42, marker="o", ms=2.2, mew=0)
        ax.plot(x, sub[LADDER].mean().to_numpy(dtype=float), color=SITE_COLORS[site],
                lw=1.8, marker="o", ms=4, mec="white", mew=0.6, zorder=4)
        ax.axhline(float(sub["XGBoost"].mean()), color="#4D4D4D", lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.axhline(0.5, color="#DDDDDD", lw=0.5, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels([LADDER_LABELS[s] for s in LADDER], fontsize=FONT["tick"] - 0.5,
                           rotation=45, ha="right", rotation_mode="anchor", linespacing=0.9)
        ax.set_xlim(-0.4, len(LADDER) - 0.6)
        ax.set_ylim(0.32, 1.0)
        ax.set_title(site, fontsize=FONT["title"], pad=3, color=SITE_COLORS[site])
        ax.tick_params(labelsize=FONT["tick"])
    axes[0].set_ylabel("AUROC", fontsize=FONT["label"])
    for ax in axes[1:]:
        ax.set_yticklabels([])


def figure_y(
    summary: pd.DataFrame, diffs_mint: pd.DataFrame, diffs_xgb: pd.DataFrame,
    rs: dict[str, float], order: list[str],
) -> None:
    fig = plt.figure(figsize=(7.2, 7.6), constrained_layout=True)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.5, 1.0], hspace=0.10)
    top = outer[0].subgridspec(1, 2, width_ratios=[1.0, 0.62], wspace=0.03)

    ax_a = fig.add_subplot(top[0])
    ax_b = fig.add_subplot(top[1])
    _dominance_panel(ax_a, summary, diffs_mint, diffs_xgb, order)
    # Panel b carries its own ordering, sorted so the sign split reads directly.
    _volume_panel(ax_b, rs, sorted(order, key=lambda k: -rs[k]))

    bottom = outer[1].subgridspec(1, len(SITE_ORDER), wspace=0.10)
    axes_c = [fig.add_subplot(bottom[j]) for j in range(len(SITE_ORDER))]
    _ladder_panel(axes_c, summary)

    handles = [
        Line2D([], [], color="#4D4D4D", lw=1.8, marker="o", ms=4, label="Site mean across outcomes"),
        Line2D([], [], color="#4D4D4D", lw=0.7, alpha=0.42, marker="o", ms=2.2, label="Individual outcome"),
        Line2D([], [], color="#4D4D4D", lw=0.7, ls=(0, (3, 2)), label="Locally-trained XGBoost (site mean)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=FONT["legend"], bbox_to_anchor=(0.5, -0.028))

    fig.canvas.draw()
    letter_x = _panel_letter(fig, ax_a, "a")
    _panel_letter(fig, ax_b, "b")
    _panel_letter(fig, axes_c[0], "c", x=letter_x)  # flush with panel a
    _save(fig, "supp_comp_determinants")


def main() -> None:
    plt.style.use(str(STYLE_PATH))
    summary, diffs_mint, diffs_xgb = _load()
    rs = _volume_rho(summary)

    write_table(summary)
    figure_x(diffs_mint, diffs_xgb, FIGURE_METHODS)
    figure_y(summary, diffs_mint, diffs_xgb, rs, FIGURE_METHODS)
    for k in FIGURE_METHODS:
        print(f"  rho = {rs[k]:+.2f}  {METHOD_LABELS[k]}")
    print(f"wrote 2 figures + supp_comp_table.csv to {OUT_DIR}")


if __name__ == "__main__":
    main()
