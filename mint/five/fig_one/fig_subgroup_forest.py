"""Demographic-bias forest plot: MINT AUROC by subgroup, per outcome.

Replaces the earlier subgroup_incidence.py + fig_subgroup_heatmaps.py. Instead
of a sex x race x age heatmap grid, this draws a two-panel forest plot -- one
panel for ICU AUROC, one for Hypoxia AUROC -- with a row per subgroup, grouped
into categories (Race, Ethnicity, Gender, Payor, Age), matching the layout of
the reference forest_demo.png but using the local nature.mplstyle.

    python -m mint.five.fig_one.fig_subgroup_forest

Metric convention matches fig1_nejm.py / the old heatmap script: AUROC is
computed over ALL prediction rows (every time-window) in a subgroup using the
MINT softmax predictions, and the 95% CI is an ENCOUNTER-CLUSTERED bootstrap
(all of a sampled encounter's rows travel together).

CATEGORIES (each a distinct demographic dimension, drawn separately)
    Race       cdw/demos_v3.csv FirstRace, collapsed to
               White / Black or African American / Asian / Other-Unknown.
    Ethnicity  cdw/demos_v3.csv Ethnicity: Hispanic/Latinx vs
               Not Hispanic or Unknown (Race and Ethnicity are kept SEPARATE
               here, unlike combine_race_ethnicity).
    Gender     cdw/demos_v3.csv Sex: Female vs Not Female.
    Payor      cdw/demos_v3.csv BenefitPlanProductType, collapsed to
               Medicaid / Commercial / Other-Unknown (Other-Unknown pools all
               remaining plan types -- Medicare*, TriCare, etc. -- with missing
               / *Unspecified). (CoverageFinancialClass is ~98% *Unspecified so
               is unusable; BenefitPlanProductType has near-complete coverage.)
    Age        cdw/all_pediatric_ed_visits_with_note.csv Age, binned by
               create_age_category (the same 5 existing age groups).

Within each category, rows are ordered: Age by its clinical bins, Race /
Ethnicity / Payor alphabetically, Gender by AUROC. Both demographic sources are
inner-joined onto the prediction encounter_keys. Every prediction encounter_key
is present exactly once in demos_v3.csv and in
all_pediatric_ed_visits_with_note.csv, so the joins drop no encounter_keys; this
is asserted at run time.

Outputs (in artifacts/fig_subgroups/):
    fig_subgroup_forest.png / .pdf
    subgroup_forest.csv   (tidy per-subgroup AUROC + 95% CI + n, both outcomes)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tap import tapify

from mint.helpers.demos import create_age_category

# Reuse the Fig-1 nature style for a consistent look.
plt.style.use(str(Path(__file__).resolve().parents[1] / "design-skill" / "nature.mplstyle"))

ROOT = Path(__file__).resolve().parents[3]
# Large data (cdw/, artifacts/) is gitignored and lives only in the main
# checkout, not in worktrees under .claude/worktrees/.
if ".claude/worktrees" in str(ROOT):
    DATA_ROOT = Path(str(ROOT).split("/.claude/worktrees")[0])
else:
    DATA_ROOT = ROOT
OUT = DATA_ROOT / "artifacts" / "fig_subgroups"

OUTCOME_PATHS = {
    "icu": DATA_ROOT / "artifacts/fig1_operational/icu_MINT_softmax.csv",
    "hypoxia": DATA_ROOT / "artifacts/fig1-final/fig1-final-softmax/hypoxia_softmax.csv",
}
OUTCOME_LABELS = {"icu": "ICU", "hypoxia": "Hypoxia"}

AGE_ORDER = ["Infant (0-<1)", "Toddler (1-<3)", "Early childhood (3-<6)",
             "Middle childhood (6-<12)", "Adolescent (12-<18)"]

# The category dimensions, drawn top-to-bottom, matching forest_demo.png.
CATEGORY_ORDER = ["Race", "Ethnicity", "Gender", "Payor", "Age"]
# Categories whose subgroup rows are ordered alphabetically (rather than by
# mean AUROC).
ALPHABETICAL_CATEGORIES = {"Race", "Ethnicity", "Payor"}


def _payer(x: object) -> str:
    """Collapse BenefitPlanProductType into four coarse payer buckets."""
    if pd.isna(x):
        return "Other/Unknown"
    s = str(x)
    if s == "*Unspecified":
        return "Other/Unknown"
    if "Medi-Cal" in s or "Medicaid" in s or s in ("CCS", "MIA", "CMSP", "GHPP"):
        return "Medicaid"
    if s in ("PPO", "POS", "HMO", "EPO", "IPA", "Indemnity"):
        return "Commercial"
    # Everything else -- Medicare*, TriCare, GOV, Worker's Comp, Charity,
    # Research Government -- plus missing/*Unspecified above, into one bucket.
    return "Other/Unknown"


def _race(x: object) -> str:
    if x in ("White", "Asian", "Black or African American"):
        return str(x)
    return "Other/Unknown"


def load_demographics() -> pd.DataFrame:
    """Per-encounter subgroup labels for every category dimension."""
    d = pd.read_csv(DATA_ROOT / "cdw/demos_v3.csv",
                    usecols=["EncounterKey", "Sex", "FirstRace", "Ethnicity",
                             "BenefitPlanProductType"])
    v = pd.read_csv(DATA_ROOT / "cdw/all_pediatric_ed_visits_with_note.csv",
                    usecols=["EncounterKey", "Age"])

    d["Race"] = d["FirstRace"].apply(_race)
    d["Ethnicity_grp"] = np.where(d["Ethnicity"] == "Hispanic or Latino",
                                  "Hispanic/Latinx", "Not Hispanic or Unknown")
    d["Gender"] = np.where(d["Sex"] == "Female", "Female", "Not Female")
    d["Payor"] = d["BenefitPlanProductType"].apply(_payer)

    v["Age_grp"] = v["Age"].apply(create_age_category)

    d = d.set_index("EncounterKey")[["Race", "Ethnicity_grp", "Gender", "Payor"]]
    v = v.set_index("EncounterKey")[["Age_grp"]]
    demos = d.join(v, how="outer").rename(columns={"Ethnicity_grp": "Ethnicity",
                                                   "Age_grp": "Age"})
    return demos


def _bootstrap_auroc_ci(sub: pd.DataFrame, n_boot: int, seed: int) -> tuple[float, float]:
    """95% CI for AUROC by ENCOUNTER-clustered bootstrap (matches fig1_nejm.py)."""
    codes, uniq = pd.factorize(sub["encounter_key"].to_numpy())
    n_enc = len(uniq)
    order = np.argsort(codes, kind="stable")
    codes_s = codes[order]
    probs_s = sub["probs"].to_numpy()[order]
    labels_s = sub["labels"].to_numpy()[order]
    counts = np.bincount(codes_s, minlength=n_enc)
    offsets = np.zeros(n_enc + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)

    rng = np.random.default_rng(seed)
    aurocs = np.full(n_boot, np.nan)
    for b in range(n_boot):
        sampled = rng.integers(0, n_enc, size=n_enc)
        sel = counts[sampled]
        total = int(sel.sum())
        starts = offsets[sampled]
        base = np.repeat(starts, sel)
        within = np.arange(total) - np.repeat(np.cumsum(sel) - sel, sel)
        idx = base + within
        y = labels_s[idx]
        if y.min() == y.max():
            continue
        aurocs[b] = roc_auc_score(y, probs_s[idx])
    aurocs = aurocs[~np.isnan(aurocs)]
    if len(aurocs) < max(20, n_boot // 20):
        return np.nan, np.nan
    return float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))


# Which demographics column backs each category, and how to order its groups.
CATEGORY_COL = {"Race": "Race", "Ethnicity": "Ethnicity", "Gender": "Gender",
                "Payor": "Payor", "Age": "Age"}


def subgroup_metrics(outcome: str, demos: pd.DataFrame, n_boot: int,
                     seed: int) -> pd.DataFrame:
    """Per-(category, group) AUROC + 95% CI (encounter-clustered) and n."""
    pr = pd.read_csv(OUTCOME_PATHS[outcome])
    n_keys_before = pr["encounter_key"].nunique()
    pr = pr.join(demos, on="encounter_key")
    # Assert the demographic join dropped no encounter_keys.
    n_keys_after = pr.dropna(subset=list(CATEGORY_COL.values()))["encounter_key"].nunique()
    assert n_keys_after == n_keys_before, (
        f"[{outcome}] demographic join dropped encounter_keys: "
        f"{n_keys_before} -> {n_keys_after}")

    rows = []
    for cat in CATEGORY_ORDER:
        col = CATEGORY_COL[cat]
        for grp, sub in pr.groupby(col, observed=True):
            y = sub["labels"].to_numpy()
            if y.min() == y.max():
                auroc, ci_lo, ci_hi = np.nan, np.nan, np.nan
            else:
                auroc = roc_auc_score(y, sub["probs"].to_numpy())
                ci_lo, ci_hi = _bootstrap_auroc_ci(sub, n_boot, seed)
            rows.append({"outcome": outcome, "category": cat, "group": str(grp),
                         "n": int(len(sub)),
                         "n_enc": int(sub["encounter_key"].nunique()),
                         "n_pos": int(y.sum()), "auroc": float(auroc),
                         "auroc_ci_lo": float(ci_lo), "auroc_ci_hi": float(ci_hi)})
    return pd.DataFrame(rows)


def _row_layout(metrics: pd.DataFrame) -> tuple[list[dict], list[float]]:
    """Assign a y-position to every subgroup row and record category band edges.

    Rows are grouped by category (in CATEGORY_ORDER) with a gap between
    categories. Age keeps its clinical ordering; Race, Ethnicity, and Payor are
    ordered alphabetically; any remaining category falls back to mean AUROC
    across the two outcomes (descending). Returns (rows, band_boundaries) where
    each row is {category, group, y} and band_boundaries are the y-values of the
    dashed separators between categories.
    """
    # Mean AUROC per (category, group) across outcomes, for the fallback order.
    mean_au = metrics.groupby(["category", "group"])["auroc"].mean()

    rows: list[dict] = []
    boundaries: list[float] = []
    y = 0.0
    gap = 1.0  # extra vertical gap between categories
    for ci, cat in enumerate(CATEGORY_ORDER):
        if ci > 0:
            boundaries.append(y + gap / 2)
            y += gap
        groups = metrics[metrics.category == cat]["group"].unique().tolist()
        if cat == "Age":
            groups = [g for g in AGE_ORDER if g in groups]
        elif cat in ALPHABETICAL_CATEGORIES:
            groups = sorted(groups)
        else:
            groups = sorted(groups, key=lambda g: mean_au.loc[(cat, g)], reverse=True)
        for grp in groups:
            rows.append({"category": cat, "group": grp, "y": y})
            y += 1.0
    return rows, boundaries


def plot_forest(metrics: pd.DataFrame, dpi: int) -> Path:
    outcomes = ["icu", "hypoxia"]
    rows, boundaries = _row_layout(metrics)
    y_of = {(r["category"], r["group"]): r["y"] for r in rows}
    y_max = max(r["y"] for r in rows)

    # Okabe-Ito blue for the point estimate; black CI whiskers.
    point_color = "#0072B2"

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 5.6), sharey=True, layout="none")
    fig.subplots_adjust(left=0.46, right=0.97, top=0.94, bottom=0.10, wspace=0.12)

    for ax, outcome in zip(axes, outcomes):
        m = metrics[metrics.outcome == outcome]
        val = {(r.category, r.group): r for r in m.itertuples()}
        for (cat, grp), r in val.items():
            y = y_of[(cat, grp)]
            if np.isnan(r.auroc):
                continue
            if not np.isnan(r.auroc_ci_lo):
                ax.plot([r.auroc_ci_lo, r.auroc_ci_hi], [y, y], color="black",
                        lw=0.9, zorder=2, solid_capstyle="round")
            ax.plot(r.auroc, y, "o", color=point_color, ms=4, zorder=3,
                    markeredgewidth=0)

        ax.set_ylim(y_max + 0.8, -0.8)  # invert: first category on top
        for b in boundaries:
            ax.axhline(b, color="0.7", ls="--", lw=0.5, zorder=1)
        ax.set_xlabel("AUROC")
        ax.set_title(OUTCOME_LABELS[outcome])
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="0.9", lw=0.4, zorder=0)
        ax.set_axisbelow(True)

    # y tick labels: the subgroup name for every row (shared axis -> set once).
    axes[0].set_yticks([r["y"] for r in rows])
    axes[0].set_yticklabels([r["group"] for r in rows], fontsize=6)

    # Category labels in their own gutter to the far left, clear of the
    # subgroup tick labels (right-aligned at a fixed axes-fraction x).
    for cat in CATEGORY_ORDER:
        ys = [r["y"] for r in rows if r["category"] == cat]
        yc = (min(ys) + max(ys)) / 2
        axes[0].text(-0.66, yc, cat, transform=axes[0].get_yaxis_transform(),
                     ha="right", va="center", fontsize=8, fontweight="bold")

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "fig_subgroup_forest.png"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(OUT / "fig_subgroup_forest.pdf", bbox_inches="tight")
    plt.close(fig)
    return png


@dataclass
class Args:
    n_boot: int = 1000  # bootstrap iterations for per-subgroup AUROC CIs
    seed: int = 42
    dpi: int = 600


def main() -> None:
    args = tapify(Args)
    OUT.mkdir(parents=True, exist_ok=True)
    demos = load_demographics()

    all_metrics = []
    for outcome in ["icu", "hypoxia"]:
        print(f"[{outcome}] computing per-subgroup AUROC + {args.n_boot}x bootstrap CIs...")
        all_metrics.append(subgroup_metrics(outcome, demos, args.n_boot, args.seed))
    metrics = pd.concat(all_metrics, ignore_index=True)

    png = plot_forest(metrics, args.dpi)
    metrics.to_csv(OUT / "subgroup_forest.csv", index=False)
    print(f"Wrote {png.name} and subgroup_forest.csv ({len(metrics)} subgroup rows)")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
