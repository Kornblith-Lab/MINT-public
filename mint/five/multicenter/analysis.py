"""Multicenter validation summaries and figures.

Reads the exported multicenter CSVs in ``multicenter/`` and writes:
  - a forest plot for MINT vs UCSF-trained XGBoost and MINT vs locally-trained XGBoost
  - a separate forest plot for fine-tuned MINT vs locally-trained XGBoost
  - two scatter plots relating AUROC gain to site volume and n_pos
  - a compact method summary table/figure for win/tie/loss counts
  - a site characteristics CSV in the format requested for the manuscript

All outputs are written to ``artifacts/multicenter_analysis``.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


ROOT = Path(__file__).resolve().parents[3]
STYLE_PATH = Path(__file__).resolve().parents[1] / "design-skill" / "nature.mplstyle"
INPUT_DIR = ROOT / "multicenter"
OUT_DIR = ROOT / "artifacts" / "multicenter_analysis"

SITE_ORDER = ["UC 1", "UC 2", "UC 3", "UC 4", "UC 5"]
OUTCOME_ORDER = ["hypoxia", "tachypnea", "periarrest", "tachycardia", "hypotension"]
OUTCOME_COLORS = {
    "hypoxia": "#0072B2",
    "tachypnea": "#009E73",
    "periarrest": "#D55E00",
    "tachycardia": "#CC79A7",
    "hypotension": "#E69F00",
}

FONT_SIZES = {
    "title": 12,
    "label": 11,
    "tick": 9,
    "panel": 10,
    "annotation": 8.5,
    "legend": 8.5,
}

METHOD_COLORS = {
    "better": "#5E8C61",
    "equivalent": "#B3B3B3",
    "worse": "#DC4151",
}

SITE_COLORS: dict[str, str] = {
    "UC 1": "#0072B2",
    "UC 2": "#E69F00",
    "UC 3": "#009E73",
    "UC 4": "#CC79A7",
    "UC 5": "#D55E00",
}

METHOD_ORDER = ["Softmax",
                "Softmax_FT",
                "XGBoost",
                "GlobalXGBoost",
                # "Triage",
                "MINT_LR",
                "MINT_LR_FT",
                "ClassHead"]

# Fill this in to rename method labels and panel titles in the supplementary
# method summary figure.
SUPP_MULTICENTER_METHOD_SUMMARY_RENAMES: dict[str, str] = {
    "Softmax": "MINT zero-shot",
    "Softmax_FT": "Fine-tuned\nMINT zero-shot",
    "XGBoost": "Locally-trained\nXGBoost",
    "GlobalXGBoost": "UCSF-trained\nXGBoost",
    # "Triage": "Triage",
    "MINT_LR": "MINT linear probe",
    "MINT_LR_FT": "Fine-tuned\nMINT linear probe",
    "ClassHead": "Fine-tuned\nend-to-end MINT"
}

SUPP_MULTICENTER_METHOD_SUMMARY_SKIP_METHODS: list[str] = ["Triage"]

SHOW_INCIDENCE = True


def _load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(INPUT_DIR / name)


def _fmt_pct_count(n: int, denom: int) -> str:
    if denom <= 0:
        return f"0.0% ({n})"
    return f"{100.0 * n / denom:.1f}% ({n})"


def _fmt_range(lo: int, hi: int) -> str:
    return f"{lo}-{hi}"


def _save_fig(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _site_totals() -> pd.DataFrame:
    token = _load_csv("site_token_dist.csv")
    token = token.rename(columns={"Site": "hospital"})
    token["Encounters"] = token["Encounters"].astype(int)
    return token.set_index("hospital")


def _site_year_ranges() -> dict[str, str]:
    years = _load_csv("site_years.csv").set_index("care_site_name")
    year_cols = [c for c in years.columns if c.isdigit()]
    ranges: dict[str, str] = {}
    for site, row in years[year_cols].iterrows():
        vals = []
        for col in year_cols:
            raw = row[col]
            vals.append(0 if str(raw).strip() == "<10" else int(raw))
        active = [int(year_cols[i]) for i, val in enumerate(vals) if val > 0]
        if not active:
            ranges[site] = ""
        else:
            ranges[site] = _fmt_range(min(active), max(active))
    return ranges


def build_site_characteristics() -> pd.DataFrame:
    totals = _site_totals()
    years = _site_year_ranges()
    race = _load_csv("site_race.csv").set_index("care_site_name")
    eth = _load_csv("site_eth.csv").set_index("care_site_name")
    gender = _load_csv("site_gender.csv").set_index("care_site_name")

    rows: list[dict[str, str]] = []

    def add_row(label: str, values: dict[str, str]) -> None:
        row = {"Metric": label}
        row.update(values)
        rows.append(row)

    add_row("Total visits", {site: f"{int(totals.loc[site, 'Encounters'])}" for site in SITE_ORDER})
    add_row("Date range", {site: years.get(site, "") for site in SITE_ORDER})
    add_row(
        "Tokens, median [IQR]",
        {site: str(totals.loc[site, "Tokens per encounter"]).replace("–", "-") for site in SITE_ORDER},
    )

    add_row("Race", {site: "" for site in SITE_ORDER})
    race_groups = [
        "Asian",
        "Black or African American",
        "White",
    ]
    for grp in race_groups:
        label = "Black" if grp == "Black or African American" else ("Other/Unknown" if grp == "Unknown" else grp)
        values = {
            site: _fmt_pct_count(int(race.loc[site, grp]), int(totals.loc[site, "Encounters"]))
            if grp in race.columns
            else ""
            for site in SITE_ORDER
        }
        add_row(label, values)

    # Collapse the remaining race columns into Other/Unknown so the rows sum to 100%.
    other_cols = [
        "American Indian or Alaska Native",
        "Multirace",
        "Native Hawaiian or Other Pacific Islander",
        "Other Race",
        "Unknown",
    ]
    add_row(
        "Other/Unknown",
        {
            site: _fmt_pct_count(
                int(race.loc[site, other_cols].sum()) + int(race.loc[site, "Unknown"]),
                int(totals.loc[site, "Encounters"]),
            )
            for site in SITE_ORDER
        },
    )

    add_row("Ethnicity", {site: "" for site in SITE_ORDER})
    add_row(
        "Hispanic or Latino",
        {site: _fmt_pct_count(int(eth.loc[site, "Hispanic or Latino"]), int(totals.loc[site, "Encounters"])) for site in SITE_ORDER},
    )
    add_row(
        "Not Hispanic or Latino",
        {site: _fmt_pct_count(int(eth.loc[site, "Not Hispanic or Latino"]), int(totals.loc[site, "Encounters"])) for site in SITE_ORDER},
    )
    add_row(
        "Unknown",
        {site: _fmt_pct_count(int(eth.loc[site, "Unknown"]), int(totals.loc[site, "Encounters"])) for site in SITE_ORDER},
    )

    add_row("Sex", {site: "" for site in SITE_ORDER})
    add_row(
        "Female",
        {site: _fmt_pct_count(int(gender.loc[site, "FEMALE"]), int(totals.loc[site, "Encounters"])) for site in SITE_ORDER},
    )
    add_row(
        "Not female",
        {site: _fmt_pct_count(int(gender.loc[site, "NOT FEMALE"]), int(totals.loc[site, "Encounters"])) for site in SITE_ORDER},
    )

    return pd.DataFrame(rows, columns=["Metric", *SITE_ORDER])


def _prepare_summary() -> pd.DataFrame:
    summary = _load_csv("bootstrap_summary.csv")
    diffs = _load_csv("bootstrap_diffs_summary.csv")
    diffs_xgb = _load_csv("bootstrap_diffs_xgboost_summary.csv")
    return summary, diffs, diffs_xgb


def _invert_diff_row(row: pd.Series) -> dict[str, float]:
    lo = -float(row["auroc_diff_ci_hi"])
    hi = -float(row["auroc_diff_ci_lo"])
    return {
        "auroc_diff": -float(row["auroc_diff"]),
        "auroc_diff_ci_lo": min(lo, hi),
        "auroc_diff_ci_hi": max(lo, hi),
        "auroc_sig": bool((lo > 0 and hi > 0) or (lo < 0 and hi < 0)),
    }


def _to_long_diff_frame(summary: pd.DataFrame, diffs: pd.DataFrame, baseline: str) -> pd.DataFrame:
    rows = []
    for _, row in diffs.iterrows():
        if baseline == "Softmax":
            # diffs_summary stores method - Softmax, so for a MINT-centric plot
            # the comparator rows must be inverted.
            if row["method"] in {"GlobalXGBoost", "XGBoost"}:
                continue
        rows.append(row.to_dict())
    return pd.DataFrame(rows)


def _make_forest_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    title: str,
    show_yticklabels: bool = True,
) -> None:
    sites = [s for s in SITE_ORDER if s in data["hospital"].unique()]
    site_totals = _site_totals()["Encounters"].to_dict()
    order_rows = []
    for site in [s for s in SITE_ORDER if s in sites]:
        sub = data[data["hospital"] == site].copy()
        sub["outcome_order"] = sub["outcome"].map({o: i for i, o in enumerate(OUTCOME_ORDER)})
        sub = sub.sort_values(["outcome_order", "outcome"])
        for _, r in sub.iterrows():
            order_rows.append(r)
    ordered = pd.DataFrame(order_rows)
    ordered = ordered.reset_index(drop=True)
    row_step = 0.82
    ordered["y"] = np.arange(len(ordered))[::-1] * row_step

    for _, r in ordered.iterrows():
        point_color = "#1f1f1f"
        if bool(r.get("auroc_sig", False)) and float(r["auroc_diff"]) > 0:
            point_color = METHOD_COLORS["better"]
        elif bool(r.get("auroc_sig", False)) and float(r["auroc_diff"]) < 0:
            point_color = METHOD_COLORS["worse"]
        ax.plot(
            [r["auroc_diff_ci_lo"], r["auroc_diff_ci_hi"]],
            [r["y"], r["y"]],
            color=point_color,
            lw=1.2,
            solid_capstyle="round",
            zorder=1,
        )
        ax.scatter(
            r["auroc_diff"],
            r["y"],
            s=28,
            color=point_color,
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )

    # Add subtle site separators and labels.
    y_lookup = {row.hospital: [] for row in ordered.itertuples()}
    for _, r in ordered.iterrows():
        y_lookup.setdefault(r["hospital"], []).append(r["y"])
    x_min = float(min(ordered["auroc_diff"].min(), ordered["auroc_diff_ci_lo"].min())) - 0.01
    x_max = max(0.45, float(ordered["auroc_diff_ci_hi"].max()) + 0.02)
    ax.set_xlim(x_min, x_max)
    for site in [s for s in SITE_ORDER if s in y_lookup]:
        ys = y_lookup[site]
        if not ys:
            continue
        ax.axhline(max(ys) + 0.5, color="#e0e0e0", lw=0.8, zorder=0)
        hypoxia_y = max(ys)
        ax.text(
            x_max,
            hypoxia_y,
            f"{site} (n={site_totals.get(site, 0):,})",
            ha="right",
            va="center",
            fontsize=FONT_SIZES["annotation"],
            color="black",
        )

    ax.axvline(0, color="#333333", lw=1.0, ls="--", zorder=0)
    ax.set_title(title, fontsize=FONT_SIZES["title"])
    ax.set_ylim(-0.5 * row_step, (len(ordered) - 1) * row_step + 0.5 * row_step)
    ax.set_xlabel("AUROC gain", fontsize=FONT_SIZES["label"])
    if show_yticklabels:
        ax.set_yticks(ordered["y"])
        labels = []
        for r in ordered.itertuples():
            label = r.outcome
            if SHOW_INCIDENCE and hasattr(r, "n_pos") and hasattr(r, "n_total"):
                n_pos: int = int(r.n_pos) if r.n_pos != "<10" else 10 # type: ignore
                n_total: int = int(r.n_total) # type: ignore
                incidence = _format_incidence(int(n_pos), int(n_total))
                incidence = n_pos / n_total
                label = f"{label} ({incidence:.1%})"
            labels.append(label)
        ax.set_yticklabels(labels, fontsize=FONT_SIZES["tick"])
        ax.tick_params(axis="y", pad=2, labelleft=True)
    else:
        ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=FONT_SIZES["tick"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)


def _make_compact_forest_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    title: str,
    site_totals: dict[str, int],
    show_legend: bool = False,
) -> None:
    """Compact grouped forest plot: one row per outcome, sites dodged within the row."""
    outcomes = [o for o in OUTCOME_ORDER if o in data["outcome"].unique()]
    sites = [s for s in SITE_ORDER if s in data["hospital"].unique()]
    n_sites = len(sites)

    site_size = {s: 40.0 for s in sites}

    # Vertical dodge: UC 1 (biggest) at top, UC 5 (smallest) at bottom.
    dodge_range = 0.7
    if n_sites > 1:
        offsets = np.linspace(dodge_range / 2, -dodge_range / 2, n_sites)
    else:
        offsets = np.array([0.0])
    site_offset = dict(zip(sites, offsets))

    # Row centres: one integer per outcome, plotted bottom-to-top.
    outcome_y = {o: i for i, o in enumerate(outcomes)}

    for site_idx, site in enumerate(sites):
        sub = data[data["hospital"] == site]
        for _, r in sub.iterrows():
            outcome = r["outcome"]
            if outcome not in outcome_y:
                continue
            y = outcome_y[outcome] + site_offset[site]
            color = SITE_COLORS.get(site, "#555555")
            # CI whisker
            ax.plot(
                [r["auroc_diff_ci_lo"], r["auroc_diff_ci_hi"]],
                [y, y],
                color=color,
                lw=1.0,
                solid_capstyle="round",
                zorder=1,
                alpha=0.85,
            )
            # Point, sized by hospital volume
            ax.scatter(
                r["auroc_diff"],
                y,
                s=site_size[site],
                color=color,
                edgecolor="white",
                linewidth=0.5,
                zorder=2,
                alpha=0.92,
            )

    # Outcome row labels on y-axis.
    incidence_map: dict[str, float] = {}
    for o in outcomes:
        sub = data[data["outcome"] == o]
        if "n_pos" in sub.columns and "n_total" in sub.columns:
            n_pos = pd.to_numeric(sub["n_pos"].replace("<10", "0"), errors="coerce").fillna(0)
            n_total = pd.to_numeric(sub["n_total"], errors="coerce").fillna(0)
            denom = n_total.sum()
            if denom > 0:
                incidence_map[o] = n_pos.sum() / denom

    ax.set_yticks(list(outcome_y.values()))
    labels = []
    for o in outcomes:
        label = o
        if o in incidence_map:
            label = f"{o} ({incidence_map[o]:.1%})"
        labels.append(label)
    ax.set_yticklabels(labels, fontsize=FONT_SIZES["tick"])

    # Zero-reference line and axes.
    ax.axvline(0, color="#333333", lw=1.0, ls="--", zorder=0)
    ax.set_title(title, fontsize=FONT_SIZES["title"])
    ax.set_xlabel("AUROC gain", fontsize=FONT_SIZES["label"])
    ax.set_ylim(-0.7, len(outcomes) - 1 + 0.7)
    all_lo = data["auroc_diff_ci_lo"].min()
    all_hi = data["auroc_diff_ci_hi"].max()
    ax.set_xlim(min(all_lo, 0.0) - 0.02, max(all_hi, 0.0) + 0.04)
    ax.tick_params(axis="x", labelsize=FONT_SIZES["tick"])
    ax.tick_params(axis="y", pad=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    if show_legend:
        from matplotlib.lines import Line2D
        handles = [
            Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor=SITE_COLORS.get(s, "#555555"),
                markersize=np.sqrt(site_size[s]),
                label=f"{s} (n={site_totals.get(s, 0):,})",
            )
            for s in sites
        ]
        ax.legend(
            handles=handles,
            title="Site",
            title_fontsize=FONT_SIZES["legend"],
            fontsize=FONT_SIZES["legend"],
            frameon=False,
            loc="lower right",
        )


def make_compact_forest_figure_lr(summary: pd.DataFrame, diffs_xgb: pd.DataFrame) -> None:
    """Compact grouped forest plot for MINT linear probe vs locally-trained XGBoost."""
    lr_local = _forest_data_lr(diffs_xgb)
    incidence = summary.loc[:, ["hospital", "outcome", "n_pos", "n_total"]].drop_duplicates(["hospital", "outcome"])
    lr_local = lr_local.merge(incidence, on=["hospital", "outcome"], how="left")

    site_totals = _site_totals()["Encounters"].to_dict()

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.6), constrained_layout=True)
    _make_compact_forest_panel(ax, lr_local, "MINT linear probe vs locally-trained XGBoost", site_totals, show_legend=True)
    _save_fig(fig, "compact_supp_multicenter_forest_lr")


def make_compact_forest_figure(summary: pd.DataFrame, diffs: pd.DataFrame, diffs_xgb: pd.DataFrame) -> None:
    """Side-by-side compact grouped forest plot; outputs compact_supp_multicenter_forest_zero_shot."""
    zero_shot_global, zero_shot_local = _forest_data_zero_shot(diffs, diffs_xgb)
    incidence = summary.loc[:, ["hospital", "outcome", "n_pos", "n_total"]].drop_duplicates(["hospital", "outcome"])
    zero_shot_global = zero_shot_global.merge(incidence, on=["hospital", "outcome"], how="left")
    zero_shot_local = zero_shot_local.merge(incidence, on=["hospital", "outcome"], how="left")

    site_totals = _site_totals()["Encounters"].to_dict()

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6), constrained_layout=True)
    _make_compact_forest_panel(axes[0], zero_shot_global, "MINT vs UCSF-trained XGBoost", site_totals, show_legend=False)
    _make_compact_forest_panel(axes[1], zero_shot_local, "MINT vs locally-trained XGBoost", site_totals, show_legend=True)
    _save_fig(fig, "compact_supp_multicenter_forest_zero_shot")


def _forest_data_zero_shot(diffs: pd.DataFrame, diffs_xgb: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # MINT vs UCSF-trained XGBoost comes from the diffs-vs-Softmax table by
    # inverting GlobalXGBoost - Softmax.
    mint_vs_global = diffs.loc[diffs["method"] == "GlobalXGBoost", ["hospital", "outcome", "auroc_diff", "auroc_diff_ci_lo", "auroc_diff_ci_hi", "auroc_sig"]].copy()
    mint_vs_global["auroc_diff"] = -mint_vs_global["auroc_diff"]
    lo = -mint_vs_global["auroc_diff_ci_hi"]
    hi = -mint_vs_global["auroc_diff_ci_lo"]
    mint_vs_global["auroc_diff_ci_lo"] = np.minimum(lo, hi)
    mint_vs_global["auroc_diff_ci_hi"] = np.maximum(lo, hi)

    # MINT vs locally-trained XGBoost comes directly from the diffs-vs-XGBoost table.
    mint_vs_local = diffs_xgb.loc[diffs_xgb["method"] == "Softmax", ["hospital", "outcome", "auroc_diff", "auroc_diff_ci_lo", "auroc_diff_ci_hi", "auroc_sig"]].copy()
    return mint_vs_global, mint_vs_local


def _forest_data_lr(diffs_xgb: pd.DataFrame) -> pd.DataFrame:
    return diffs_xgb.loc[diffs_xgb["method"] == "MINT_LR", ["hospital", "outcome", "auroc_diff", "auroc_diff_ci_lo", "auroc_diff_ci_hi", "auroc_sig"]].copy()


def _forest_data_finetuned(diffs_xgb: pd.DataFrame) -> pd.DataFrame:
    ft = diffs_xgb.loc[diffs_xgb["method"] == "Softmax_FT", ["hospital", "outcome", "auroc_diff", "auroc_diff_ci_lo", "auroc_diff_ci_hi", "auroc_sig"]].copy()
    return ft


def make_forest_figures(summary: pd.DataFrame, diffs: pd.DataFrame, diffs_xgb: pd.DataFrame) -> None:
    zero_shot_global, zero_shot_local = _forest_data_zero_shot(diffs, diffs_xgb)
    ft_local = _forest_data_finetuned(diffs_xgb)
    incidence = summary.loc[:, ["hospital", "outcome", "n_pos", "n_total"]].drop_duplicates(["hospital", "outcome"])
    zero_shot_global = zero_shot_global.merge(incidence, on=["hospital", "outcome"], how="left")
    zero_shot_local = zero_shot_local.merge(incidence, on=["hospital", "outcome"], how="left")
    ft_local = ft_local.merge(incidence, on=["hospital", "outcome"], how="left")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 7.0), sharey=False, constrained_layout=True)
    _make_forest_panel(axes[0], zero_shot_global, "MINT vs UCSF-trained XGBoost", show_yticklabels=True)
    _make_forest_panel(axes[1], zero_shot_local, "MINT vs locally-trained XGBoost", show_yticklabels=False)
    _save_fig(fig, "supp_multicenter_forest_zero_shot")

    fig, ax = plt.subplots(figsize=(6.3, 6.9), constrained_layout=True)
    _make_forest_panel(ax, ft_local, "Fine-tuned MINT vs locally-trained XGBoost", show_yticklabels=True)
    _save_fig(fig, "supp_multicenter_forest_finetuned")


def _pearson_text(x: np.ndarray, y: np.ndarray) -> str:
    if len(x) < 2:
        return "Pearson r = NA, p = NA"
    r, p = pearsonr(x, y)
    return f"Pearson r = {r:.2f}, p{_fmt_p_value(p)}"


def _fmt_p_value(p: float) -> str:
    return "<0.01" if p < 0.01 else f" = {p:.3g}"


def _set_standard_log_ticks(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10))
    ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10))
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())


def _counts_for_target(df: pd.DataFrame, method: str) -> tuple[int, int, int]:
    sub = df.loc[df["method"] == method]
    better = int((((sub["auroc_diff"] > 0) & (sub["auroc_sig"] == True))).sum())
    worse = int((((sub["auroc_diff"] < 0) & (sub["auroc_sig"] == True))).sum())
    equiv = int((~sub["auroc_sig"].astype(bool)).sum())
    return better, equiv, worse


def _format_incidence(n_pos: int, n_total: int) -> str:
    if n_total <= 0:
        return "0/0"
    return f"{n_pos:,}/{n_total:,}"


def _draw_segment_labels(ax: plt.Axes, left: float, width: float, y: float, value: int, color: Optional[str] = "#1f1f1f") -> None:
    if value <= 0:
        return
    x = left + width / 2
    ax.text(
        x,
        y,
        f"{value}",
        ha="center",
        va="center",
        fontsize=FONT_SIZES["legend"],
        color=color,
        fontweight="bold",
    )


def _plot_method_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str,
    baseline: str,
    method_order: list[str],
) -> None:
    rename = SUPP_MULTICENTER_METHOD_SUMMARY_RENAMES.get
    skip_methods = set(SUPP_MULTICENTER_METHOD_SUMMARY_SKIP_METHODS)
    # Show all non-baseline methods in a stable order.
    methods = [m for m in method_order if m in set(df["method"]) and m != baseline and m not in skip_methods]
    rows = []
    for method in methods:
        better, equiv, worse = _counts_for_target(df, method)
        rows.append({"method": method, "better": better, "equivalent": equiv, "worse": worse})
    out = pd.DataFrame(rows)

    y = np.arange(len(out))[::-1]
    left = np.zeros(len(out))
    for key in ["better", "equivalent", "worse"]:
        vals = out[key].to_numpy()
        ax.barh(
            y,
            vals,
            left=left,
            color=METHOD_COLORS[key],
            edgecolor="white",
            height=0.75,
            label=key.capitalize(),
        )
        for yi, lft, val in zip(y, left, vals):
            _draw_segment_labels(ax, lft, val, yi, int(val), color="#FFFFFF" if key in ["better", "worse"] else None)
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels([rename(method, method) for method in out["method"]], fontsize=FONT_SIZES["tick"])
    ax.set_xlim(0, 25)
    ax.set_xlabel("Count of site-outcome comparisons", fontsize=FONT_SIZES["label"])
    ax.set_title(rename(title, title), fontsize=FONT_SIZES["title"])
    ax.tick_params(axis="x", labelsize=FONT_SIZES["tick"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_scatter_figures(summary: pd.DataFrame, diffs_xgb: pd.DataFrame) -> None:
    # Site volume vs mean gain across outcomes.
    site_totals = _site_totals()
    gain = diffs_xgb.loc[diffs_xgb["method"] == "Softmax", ["hospital", "outcome", "auroc_diff"]].copy()
    site_gain = gain.groupby("hospital", as_index=False)["auroc_diff"].mean()
    site_gain["Encounters"] = site_gain["hospital"].map(site_totals["Encounters"].to_dict())
    site_gain["site_order"] = site_gain["hospital"].map({site: i for i, site in enumerate(SITE_ORDER)})
    site_gain = site_gain.sort_values("site_order")
    x = site_gain["Encounters"].to_numpy(dtype=float)
    y = site_gain["auroc_diff"].to_numpy(dtype=float)
    pearson_text = _pearson_text(x, y)

    fig, ax = plt.subplots(figsize=(5.2, 4.0), constrained_layout=True)
    ax.scatter(x, y, s=60, color="#1f77b4", edgecolor="white", linewidth=0.7, zorder=3)
    for _, row in site_gain.iterrows():
        ax.annotate(row["hospital"], (row["Encounters"], row["auroc_diff"]), xytext=(4, 4), textcoords="offset points", fontsize=FONT_SIZES["annotation"])
    xx = np.linspace(x.min() * 0.9, x.max() * 1.05, 100)
    lx = np.log10(x)
    m, b = np.polyfit(lx, y, 1)
    ax.plot(xx, m * np.log10(xx) + b, color="#444444", lw=1.2, zorder=2)
    ax.set_xscale("log")
    _set_standard_log_ticks(ax)
    ax.set_xlabel("Site volume (encounters, log scale)", fontsize=FONT_SIZES["label"])
    ax.set_ylabel("Mean AUROC gain vs locally-trained XGBoost", fontsize=FONT_SIZES["label"])
    ax.text(0.03, 0.03, pearson_text, transform=ax.transAxes, ha="left", va="bottom", fontsize=FONT_SIZES["legend"], bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#dddddd"))
    ax.tick_params(axis="both", labelsize=FONT_SIZES["tick"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_fig(fig, "supp_multicenter_gain_vs_volume")

    # n_pos vs gain across all 25 site-outcome comparisons.
    merged = summary.loc[summary["method"] == "XGBoost", ["hospital", "outcome", "n_pos"]].merge(
        gain, on=["hospital", "outcome"], how="inner"
    )
    merged["n_pos"] = pd.to_numeric(merged["n_pos"].replace("<10", "0"), errors="coerce")
    x = merged["n_pos"].to_numpy(dtype=float)
    y = merged["auroc_diff"].to_numpy(dtype=float)
    pearson_text = _pearson_text(x, y)
    x_min = 9.0
    censored_x = 10.0
    plot_x = np.where(x > 0, x, censored_x)

    fig, ax = plt.subplots(figsize=(5.6, 4.2), constrained_layout=True)
    for outcome, sub in merged.groupby("outcome"):
        sub_x = pd.to_numeric(sub["n_pos"].replace("<10", "0"), errors="coerce").to_numpy(dtype=float)
        ax.scatter(
            np.where(sub_x > 0, sub_x, censored_x),
            sub["auroc_diff"],
            s=44,
            color=OUTCOME_COLORS.get(outcome, "#555555"),
            edgecolor="white",
            linewidth=0.6,
            label=outcome,
            alpha=0.95,
        )
    x_fit = np.log10(plot_x)
    m, b = np.polyfit(x_fit, y, 1)
    x_max = plot_x.max() * 1.12
    xx = np.logspace(np.log10(x_min), np.log10(x_max), 200)
    ax.plot(xx, m * np.log10(xx) + b, color="#444444", lw=1.2, ls="--")
    ax.set_xscale("log")
    _set_standard_log_ticks(ax)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Positive cases", fontsize=FONT_SIZES["label"])
    ax.set_ylabel("AUROC gain vs locally-trained XGBoost", fontsize=FONT_SIZES["label"])
    ax.text(0.03, 0.03, pearson_text, transform=ax.transAxes, ha="left", va="bottom", fontsize=FONT_SIZES["legend"], bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#dddddd"))
    ax.legend(frameon=False, fontsize=FONT_SIZES["legend"], loc="upper right", title="Outcome", title_fontsize=FONT_SIZES["legend"])
    ax.tick_params(axis="both", labelsize=FONT_SIZES["tick"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save_fig(fig, "supp_multicenter_gain_vs_npos")


def make_method_summary(summary: pd.DataFrame, diffs_xgb: pd.DataFrame) -> pd.DataFrame:
    # Two-panel stacked bar figure: comparison against Softmax and comparison
    # against locally-trained XGBoost.
    diffs_softmax = _load_csv("bootstrap_diffs_summary.csv")
    softmax_order = ["XGBoost", "Softmax_FT", "GlobalXGBoost", "Triage", "MINT_LR", "MINT_LR_FT", "ClassHead"]
    xgb_order = ["Softmax", "Softmax_FT", "GlobalXGBoost", "Triage", "MINT_LR", "MINT_LR_FT", "ClassHead"]
    methods = ["Softmax", "Softmax_FT", "GlobalXGBoost", "Triage", "MINT_LR", "MINT_LR_FT", "ClassHead"]
    rows = []
    for target_name, df in [("Softmax", diffs_softmax), ("XGBoost", diffs_xgb)]:
        for method in methods:
            sub = df.loc[df["method"] == method]
            rows.append({
                "target": target_name,
                "method": method,
                "better": int((((sub["auroc_diff"] > 0) & (sub["auroc_sig"] == True))).sum()),
                "equivalent": int((~sub["auroc_sig"].astype(bool)).sum()),
                "worse": int((((sub["auroc_diff"] < 0) & (sub["auroc_sig"] == True))).sum()),
                "total": int(len(sub)),
            })
    pd.DataFrame(rows).to_csv(OUT_DIR / "method_win_tie_loss.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)
    _plot_method_panel(axes[0], diffs_softmax, "vs MINT zero-shot", baseline="Softmax", method_order=softmax_order)
    _plot_method_panel(axes[1], diffs_xgb, "vs locally-trained XGBoost", baseline="XGBoost", method_order=xgb_order)
    handles, labels = axes[1].get_legend_handles_labels()
    for ax in axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.07), fontsize=FONT_SIZES["legend"])
    _save_fig(fig, "supp_multicenter_method_summary")
    return pd.DataFrame()


def write_outputs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary, diffs, diffs_xgb = _prepare_summary()

    site_table = build_site_characteristics()
    site_table.to_csv(OUT_DIR / "site_characteristics.csv", index=False)

    make_forest_figures(summary, diffs, diffs_xgb)
    make_compact_forest_figure(summary, diffs, diffs_xgb)
    make_compact_forest_figure_lr(summary, diffs_xgb)
    make_scatter_figures(summary, diffs_xgb)
    make_method_summary(summary, diffs_xgb)


def main() -> None:
    plt.style.use(str(STYLE_PATH))
    plt.rcParams.update({
        "font.size": FONT_SIZES["tick"],
        "axes.titlesize": FONT_SIZES["title"],
        "axes.labelsize": FONT_SIZES["label"],
        "xtick.labelsize": FONT_SIZES["tick"],
        "ytick.labelsize": FONT_SIZES["tick"],
        "legend.fontsize": FONT_SIZES["legend"],
    })
    write_outputs()


if __name__ == "__main__":
    main()
