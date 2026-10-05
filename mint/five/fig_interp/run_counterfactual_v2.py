"""Counterfactual Figure v2 — clean, data-driven rewrite of fig_counterfactual.

Produces (no CDF code — softmax over vital tokens throughout):
  1. distribution_regression.pdf — 6 histograms (panel a) stacked above 4
                                    predicted-vs-actual Δ scatters (panel b)
  2. forest_supplementary.pdf    — per-vital forest of single-drug counterfactual Δ
                                    (panels a/b/c/d = HR / BP / Resp / GCS)

The forest run also writes forest_or_summary.csv (the P(correct) ~ log10(train
count) logistic model + odds ratio) and forest_or.csv (per-case detail).

The individual distribution.pdf / regression.pdf panels are still available via
--output distribution / --output regression.

See NOTES_v2.md for the full plan. Intervention cases are plain CounterfactualCase
records; the forest plot is generated automatically from the immediate-physiologic
driver CSV.

Usage:
    python -m mint.five.fig_interp.run_counterfactual_v2
    python -m mint.five.fig_interp.run_counterfactual_v2 --debug
    python -m mint.five.fig_interp.run_counterfactual_v2 --output distribution
    python -m mint.five.fig_interp.run_counterfactual_v2 --output distribution --epinephrine_mode triple
"""

import argparse
import hashlib
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mint.five.fig_interp.core import (
    DATA_DIR,
    DEVICE,
    DISPOSITION_TOKENS,
    build_cohort_sequences,
    load_test_cohort,
    load_vocab,
    load_model,
)
from mint.five.fig_interp.fig_counterfactual import (
    STYLE_PATH,
    OUTPUT_DIR,
    _apply_style,
    append_timed_tokens,
    bootstrap_delta_ci,
    build_hypoxia_sequences,
    build_trauma_sequences,
    compute_expected_vital_distribution,
    compute_pvalue,
    find_triplets,
    format_pvalue,
    predict_expected_vital_for_triplets,
)

# Checkpoint override per NOTES_v2: always the foundation model, never ckpt_100000.pt.
CHECKPOINT_PATH = "output/mint/ckpt.pt"
CACHE_DIR = OUTPUT_DIR / "cache"
ACUITY_FILTER = ("Acuity_Immediate", "Acuity_Emergent")

DRIVER_CSV = Path("mint/five/fig_interp/drug_physiologic.csv")

# CSV column -> vital token prefix (systolic only for BP; SpO2 has no forest axis).
COLUMN_TO_VITAL = {
    "HR": "Vital_Pulse",
    "BP": "Vital_Systolic",
    "RR": "Vital_Resp",
    "SpO2": "Vital_SpO2",
    "GCS": "Vital_Glasgow Coma Scale Score",
}

# Palette shared with sibling panels.
COLOR_BASELINE = "#FFACAC"
COLOR_CF = "#ACCCFF"
COLOR_ACCENT = "#166FFF"
COLOR_DARK = "#194791"

N_BOOTSTRAP = 1000  # overridden to 50 in --debug
TRIPLE_EPINEPHRINE_SCHEDULE = [
    ("Med_epinephrine", 1.0),
    ("Med_epinephrine", 6.0),
    ("Med_epinephrine", 11.0),
]


# ---------------------------------------------------------------------------
# Intervention API
# ---------------------------------------------------------------------------

@dataclass
class CounterfactualCase:
    label: str                       # display label
    vital: str                       # vital prefix, e.g. "Vital_Pulse"
    tokens: list = field(default_factory=list)  # [(token_name, dt_minutes), ...]
    cohort: str = "general"          # "general" | "trauma" | "hypoxia_mask"
    mechanism: str = "add"           # "add" | "mask"
    expected_dir: int = None         # +1 / -1 / None (direction marker)


# ---------------------------------------------------------------------------
# Driver CSV parsing
# ---------------------------------------------------------------------------

def cell_direction(cell) -> int:
    """Directional strength cell -> +1 / -1 / None."""
    if not isinstance(cell, str):
        return None
    s = cell.strip()
    if "+" in s:
        return +1
    if "-" in s:
        return -1
    return None


def load_driver(name_to_id):
    """Parse the driver CSV into per-drug records with a canonical token.

    Canonical token = the single vocab token (among the drug's `; `-separated
    candidates) with the highest occurrence count in the cohort. Occurrence
    counts are computed lazily by the caller and injected via `token_counts`;
    here we just keep the in-vocab candidate list.
    """
    df = pd.read_csv(DRIVER_CSV)
    drugs = []
    for _, row in df.iterrows():
        candidates = [t.strip() for t in str(row["vocab_token"]).split(";")]
        candidates = [t for t in candidates if t in name_to_id]
        if not candidates:
            continue
        directions = {
            col: cell_direction(row.get(col))
            for col in COLUMN_TO_VITAL
        }
        drugs.append({
            "name": row["supplementary_name"],
            "candidates": candidates,
            "directions": directions,
        })
    return drugs


def canonical_token(candidates, token_counts):
    """Pick the candidate token that occurs most in the cohort."""
    return max(candidates, key=lambda t: token_counts.get(t, 0))


def count_tokens_in_cohort(sequences, id_to_name):
    """Count occurrences of each token name across all cohort sequences."""
    counts = {}
    for seq in sequences:
        for eid in seq["events"]:
            name = id_to_name.get(eid)
            if name is not None:
                counts[name] = counts.get(name, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def _ckpt_signature():
    p = Path(CHECKPOINT_PATH)
    return (str(p), p.stat().st_mtime if p.exists() else None)


def _cache_path(key_parts):
    """Deterministic cache file. The strength CSV is never part of the key."""
    blob = repr(key_parts).encode()
    digest = hashlib.sha256(blob).hexdigest()[:20]
    return CACHE_DIR / f"{digest}.pkl"


def _data_signature():
    p = DATA_DIR / "test.feather"
    return (str(p), p.stat().st_mtime if p.exists() else None,
            p.stat().st_size if p.exists() else None)


def cached_expected(model, config, name_to_id, sequences, vital, schedule,
                    cohort_name, n_patients, batch_size, desc):
    """Expected vital per patient, cached. schedule=[] means baseline (no append)."""
    key = (
        _ckpt_signature(), cohort_name, vital, "add",
        tuple(schedule), n_patients,
    )
    path = _cache_path(key)
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)

    if schedule:
        seqs = append_timed_tokens(sequences, schedule, name_to_id)
    else:
        seqs = sequences
    values = compute_expected_vital_distribution(
        model, config, seqs, vital, name_to_id,
        batch_size=batch_size, device=DEVICE, desc=desc,
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(values, f)
    return values


def cached_mask_spo2(model, config, name_to_id, df, n_patients, batch_size):
    """Hypoxia-cohort SpO2 baseline (pre-O2) vs counterfactual (with O2 device)."""
    key = (_ckpt_signature(), "hypoxia_mask", "Vital_SpO2", "mask", n_patients)
    path = _cache_path(key)
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)

    baseline_seqs, with_o2_seqs = build_hypoxia_sequences(df, name_to_id)
    baseline_seqs, with_o2_seqs = _subsample_pair(baseline_seqs, with_o2_seqs, n_patients)
    baseline = compute_expected_vital_distribution(
        model, config, baseline_seqs, "Vital_SpO2", name_to_id,
        batch_size=batch_size, device=DEVICE, desc="  SpO2 baseline (no O2)")
    cf = compute_expected_vital_distribution(
        model, config, with_o2_seqs, "Vital_SpO2", name_to_id,
        batch_size=batch_size, device=DEVICE, desc="  SpO2 + O2 device")
    result = (baseline, cf, len(baseline_seqs))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(result, f)
    return result


def _round_regression_predictions(predicted, vital):
    """Round predicted vital values to the requested reporting granularity."""
    if vital == "Vital_Pulse":
        step = 5
    elif vital == "Vital_SpO2":
        step = 1
    elif vital == "Vital_Systolic":
        step = 2
    else:
        return predicted
    return np.round(predicted / step) * step


def cached_regression_result(model, config, name_to_id, df, med_name, med_tokens,
                             vital, window, batch_size, prediction_mode="raw"):
    """Cached triplet regression result for one med/vital pair."""
    key = (
        _ckpt_signature(), _data_signature(), med_name, tuple(med_tokens), vital,
        float(window), batch_size, prediction_mode,
    )
    path = _cache_path(key)
    if path.exists():
        print(f"    Cache hit -> {path.name}")
        with open(path, "rb") as f:
            return pickle.load(f)

    triplets = find_triplets(df, name_to_id, med_tokens, vital, max_dt=window)
    result = {
        "med_name": med_name,
        "vital": vital,
        "n": len(triplets),
        "skipped": len(triplets) < MIN_TRIPLETS,
    }

    if len(triplets) < MIN_TRIPLETS:
        result["triplets"] = triplets
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(result, f)
        return result

    predicted = predict_expected_vital_for_triplets(
        model, config, triplets, vital, name_to_id,
        batch_size=batch_size, device=DEVICE)
    if prediction_mode == "rounded":
        predicted = _round_regression_predictions(predicted, vital)
    val_before = np.array([t["val_before"] for t in triplets], dtype=np.float32)
    val_after = np.array([t["val_after"] for t in triplets], dtype=np.float32)
    actual = val_after - val_before
    forecast = predicted - val_before

    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = 1 - np.sum((forecast - actual) ** 2) / ss_tot if ss_tot > 0 else 0.0
    r2_baseline = 1 - np.sum(actual ** 2) / ss_tot if ss_tot > 0 else 0.0
    _, pearson_p = stats.pearsonr(actual, forecast)

    result.update({
        "skipped": False,
        "actual": actual,
        "forecast": forecast,
        "r2": r2,
        "r2_baseline": r2_baseline,
        "pearson_p": pearson_p,
    })

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(result, f)
    return result


# ---------------------------------------------------------------------------
# Cohort building (deterministic; n_patients is part of the cache key)
# ---------------------------------------------------------------------------

def _subsample(sequences, n_patients):
    if n_patients is None or len(sequences) <= n_patients:
        return sequences
    rng = np.random.default_rng(0)
    idx = rng.choice(len(sequences), size=n_patients, replace=False)
    return [sequences[i] for i in idx]


def _subsample_pair(a, b, n_patients):
    if n_patients is None or len(a) <= n_patients:
        return a, b
    rng = np.random.default_rng(0)
    idx = rng.choice(len(a), size=n_patients, replace=False)
    return [a[i] for i in idx], [b[i] for i in idx]


def build_cohorts(df, name_to_id, n_patients):
    """Build the two token-appending cohorts once (general + trauma)."""
    print("Building general cohort...")
    general = _subsample(build_cohort_sequences(df, name_to_id), n_patients)
    print(f"  {len(general)} patients")
    print("Building trauma cohort...")
    trauma = _subsample(build_trauma_sequences(df, name_to_id), n_patients)
    print(f"  {len(trauma)} patients")
    return {"general": general, "trauma": trauma}


# ---------------------------------------------------------------------------
# Case evaluation
# ---------------------------------------------------------------------------

def evaluate_case(model, config, name_to_id, cohorts, df, case, n_patients, batch_size):
    """Return (baseline_array, cf_array, n) for a CounterfactualCase."""
    if case.mechanism == "mask":
        baseline, cf, n = cached_mask_spo2(
            model, config, name_to_id, df, n_patients, batch_size)
        return baseline, cf, n

    seqs = cohorts[case.cohort]
    baseline = cached_expected(
        model, config, name_to_id, seqs, case.vital, [],
        case.cohort, n_patients, batch_size, desc=f"  {case.label} baseline")
    cf = cached_expected(
        model, config, name_to_id, seqs, case.vital, case.tokens,
        case.cohort, n_patients, batch_size, desc=f"  {case.label} + intervention")
    return baseline, cf, len(seqs)


# ---------------------------------------------------------------------------
# Default distribution cases
# ---------------------------------------------------------------------------

def massive_transfusion_schedule():
    """6× transfusion @ 5-min; TXA with 1st & 6th; Ca-chloride + Ca-gluconate with 1st."""
    schedule = []
    for i in range(6):
        dt = i * 5.0
        schedule.append(("Procedure_BLOOD TRANSFUSION ORDERABLES", dt))
        if i in (0, 5):
            schedule.append(("Med_tranexamic acid", dt))
        if i == 0:
            schedule.append(("Med_calcium chloride", dt))
            schedule.append(("Med_calcium gluconate", dt))
    return schedule


def default_distribution_cases():
    return [
        CounterfactualCase(
            label="Pulse after epinephrine", vital="Vital_Pulse",
            tokens=[("Med_epinephrine", 1.0)], expected_dir=+1),
        CounterfactualCase(
            label="GCS after rapid sequence\nintubation medications",
            vital="Vital_Glasgow Coma Scale Score",
            tokens=[("Med_ketamine", 1.0), ("Med_rocuronium", 2.0),
                    ("Med_propofol", 3.0)], expected_dir=-1),
        CounterfactualCase(
            label="SpO2 after respiratory support\n(children with hypoxia)",
            vital="Vital_SpO2", cohort="hypoxia_mask", mechanism="mask",
            expected_dir=+1),
        CounterfactualCase(
            label="Systolic BP after transfusion", vital="Vital_Systolic",
            tokens=[("Procedure_BLOOD TRANSFUSION ORDERABLES", 1.0)],
            expected_dir=+1),
        CounterfactualCase(
            label="Systolic BP after massive transfusion\n(children with injury)",
            vital="Vital_Systolic", cohort="trauma",
            tokens=massive_transfusion_schedule(), expected_dir=+1),
        CounterfactualCase(
            label="Respiratory rate after naloxone", vital="Vital_Resp",
            tokens=[("Med_naloxone", 1.0)], expected_dir=+1),
    ]


def triple_epinephrine_schedule():
    """Three epinephrine doses separated by 5 minutes."""
    return TRIPLE_EPINEPHRINE_SCHEDULE


# ---------------------------------------------------------------------------
# Output 1: distribution.pdf
# ---------------------------------------------------------------------------

def _draw_distribution(axes_flat, model, config, name_to_id, cohorts, df, cases,
                       n_patients, batch_size, n_boot, panel_labels=None,
                       epinephrine_mode="single"):
    """Draw baseline-vs-counterfactual histograms into the given axes.

    panel_labels: list of bold panel letters per axis (e.g. ['a','b','c']),
                  or None to skip labelling, or a single string to label
                  only the first axis (legacy behaviour).
    """
    if isinstance(panel_labels, str):
        panel_labels = [panel_labels] + [None] * (len(cases) - 1)

    hist_kwargs = dict(bins=100, alpha=0.65, density=True, edgecolor="none")

    for idx, case in enumerate(cases):
        ax = axes_flat[idx]
        print(f"\n[distribution] {case.label.replace(chr(10), ' ')}")
        baseline, cf, n = evaluate_case(
            model, config, name_to_id, cohorts, df, case, n_patients, batch_size)

        xlabel = VITAL_DISPLAY.get(case.vital, case.vital.replace("Vital_", ""))
        ax.hist(baseline, color=COLOR_BASELINE, label="Baseline", **hist_kwargs)
        ax.hist(cf, color=COLOR_CF, label="+ Intervention", **hist_kwargs)

        if idx == 0 and epinephrine_mode == "triple":
            cf3 = cached_expected(
                model, config, name_to_id, cohorts[case.cohort], case.vital,
                triple_epinephrine_schedule(), case.cohort, n_patients,
                batch_size, desc=f"  {case.label} + 3x epinephrine")
            ax.hist(cf3, color=COLOR_DARK, label="+ Intervention (3×)",
                    bins=100, alpha=0.25, density=True, edgecolor="none")

        ax.set_xlabel(f"Expected {xlabel}", fontsize=7)
        if idx % 3 == 0:
            ax.set_ylabel("Density", fontsize=7)
        ax.set_title(f"{case.label} (n={n:,} visits)", fontsize=8, fontweight="bold")
        ax.tick_params(axis="both", which="major", length=2, width=0.4)

        # Legend always upper left; stat box always lower right — keeps them apart.
        ax.legend(fontsize=6, frameon=False, loc="upper left")

        delta = float((cf - baseline).mean())
        ci_lo, ci_hi = bootstrap_delta_ci(baseline, cf, n_boot=n_boot)
        p = compute_pvalue(baseline, cf)
        stat_lines = [f"Δ = {delta:+.1f} ({ci_lo:+.1f} to {ci_hi:+.1f})"]
        # PVALUE VIZ
        # stat_lines.append(format_pvalue(p))
        if idx == 0 and epinephrine_mode == "triple":
            delta3 = float((cf3 - baseline).mean())
            ci3_lo, ci3_hi = bootstrap_delta_ci(baseline, cf3, n_boot=n_boot)
            p3 = compute_pvalue(baseline, cf3)
            stat_lines.append(f"Δ (3×) = {delta3:+.1f} ({ci3_lo:+.1f} to {ci3_hi:+.1f})")
            # PVALUE VIZ
            # stat_lines.append(format_pvalue(p3))
        stat_text = "\n".join(stat_lines)
        ax.text(0.03, 0.03, stat_text, transform=ax.transAxes, fontsize=6.5,
                ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          alpha=0.9, edgecolor="#cccccc", linewidth=0.5))
        print(f"    Δ = {delta:+.2f} ({ci_lo:+.2f} to {ci_hi:+.2f}), {format_pvalue(p)}")
        if idx == 0 and epinephrine_mode == "triple":
            print(f"    Δ₃ = {delta3:+.2f} ({ci3_lo:+.2f} to {ci3_hi:+.2f}), {format_pvalue(p3)}")

        lbl = panel_labels[idx] if panel_labels and idx < len(panel_labels) else None
        if lbl:
            ax.text(-0.10, 1.08, lbl, transform=ax.transAxes,
                    fontsize=11, fontweight="bold", va="top")


def run_distribution(model, config, name_to_id, cohorts, df, cases,
                     n_patients, batch_size, n_boot, epinephrine_mode="single"):
    _apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5))
    _draw_distribution(axes.flatten(), model, config, name_to_id, cohorts, df,
                       cases, n_patients, batch_size, n_boot, panel_labels="a",
                       epinephrine_mode=epinephrine_mode)
    fig.subplots_adjust(wspace=0.35, hspace=0.55)
    _save(fig, "distribution")


def default_distribution_supp_cases():
    return [
        CounterfactualCase(
            label="Systolic BP after dopamine", vital="Vital_Systolic",
            tokens=[("Med_dopamine", 1.0)], expected_dir=+1),
        CounterfactualCase(
            label="Heart rate after adenosine", vital="Vital_Pulse",
            tokens=[("Med_adenosine", 1.0)], expected_dir=-1),
        CounterfactualCase(
            label="Respiratory rate after propofol", vital="Vital_Resp",
            tokens=[("Med_propofol", 1.0)], expected_dir=-1),
    ]


def _train_vital_stats(vital_prefix, drug_token=None):
    """Compute population and (optionally) pre-drug mean/SD from train.feather.

    vital_prefix: e.g. "Vital_Systolic"
    drug_token:   e.g. "Med_dopamine" — if given, restricts to encounters where
                  this drug appears before disposition, and uses last vital before
                  first drug occurrence (one value per encounter).

    Returns (pop_mean, pop_sd, predrug_mean, predrug_sd).
    predrug_* are None when drug_token is None.
    """
    df = pd.read_feather(DATA_DIR / "train.feather", columns=["encounter_key", "name", "t"])

    # Truncate each encounter at first disposition token.
    disp = df[df["name"].isin(DISPOSITION_TOKENS)]
    disp_time = disp.groupby("encounter_key", sort=False)["t"].min()
    df["_disp_t"] = df["encounter_key"].map(disp_time).fillna(np.inf)
    df = df[df["t"] < df["_disp_t"]]

    # Parse numeric vital values from token names like "Vital_Systolic_104".
    vital_rows = df[df["name"].str.startswith(vital_prefix + "_")].copy()
    vital_rows["_val"] = pd.to_numeric(
        vital_rows["name"].str[len(vital_prefix) + 1:], errors="coerce")
    vital_rows = vital_rows.dropna(subset=["_val"])

    # Population mean/SD: last vital per encounter.
    pop = vital_rows.sort_values("t").groupby("encounter_key")["_val"].last()
    pop_mean, pop_sd = float(pop.mean()), float(pop.std())

    if drug_token is None:
        return pop_mean, pop_sd, None, None

    # Pre-drug mean/SD: encounters where the drug appears; last vital before
    # the first dose.
    drug_rows = df[df["name"] == drug_token]
    first_drug_t = drug_rows.groupby("encounter_key")["t"].min()
    drug_encs = first_drug_t.index

    predrug_vals = []
    for enc in drug_encs:
        t_drug = first_drug_t[enc]
        enc_vitals = vital_rows[(vital_rows["encounter_key"] == enc) &
                                (vital_rows["t"] < t_drug)]
        if enc_vitals.empty:
            continue
        predrug_vals.append(float(enc_vitals.sort_values("t")["_val"].iloc[-1]))

    if not predrug_vals:
        return pop_mean, pop_sd, None, None
    arr = np.array(predrug_vals)
    return pop_mean, pop_sd, float(arr.mean()), float(arr.std())


def _add_stats_table(fig, axes, cases):
    """Append a compact 2-row table below the panel-a axes."""
    # Map each case to its drug token name and vital prefix.
    vital_prefix_map = {
        "Vital_Systolic": "Vital_Systolic",
        "Vital_Pulse":    "Vital_Pulse",
        "Vital_Resp":     "Vital_Resp",
    }
    # Drug token: first token in the case schedule.
    rows = [["Population mean (SD)", ""], ["Pre-drug mean (SD)", ""]]
    col_labels = []
    cell_data = [[], []]  # [row][col]

    for case in cases:
        vital_prefix = vital_prefix_map.get(case.vital, case.vital)
        drug_token = case.tokens[0][0] if case.tokens else None
        print(f"[stats table] computing train stats for {vital_prefix} / {drug_token}")
        pop_mean, pop_sd, pre_mean, pre_sd = _train_vital_stats(vital_prefix, drug_token)
        xlabel = VITAL_DISPLAY.get(case.vital, case.vital.replace("Vital_", ""))
        col_labels.append(xlabel)
        cell_data[0].append(f"{pop_mean:.1f} ({pop_sd:.1f})")
        if pre_mean is not None:
            cell_data[1].append(f"{pre_mean:.1f} ({pre_sd:.1f})")
        else:
            cell_data[1].append("—")

    # Place a table in figure coordinates spanning the same x-range as the axes.
    # Use a single wide axes for the table drawn below panel a.
    row_labels = ["Population\nmean (SD)", "Pre-drug\nmean (SD)"]
    tbl_ax = fig.add_axes([0.07, 0.01, 0.90, 0.14])  # [left, bottom, width, height]
    tbl_ax.axis("off")
    tbl = tbl_ax.table(
        cellText=cell_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    # Minimal row height.
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_text_props(fontweight="bold")
        cell.set_height(0.38)


def run_distribution_supp(model, config, name_to_id, cohorts, df, cases,
                          n_patients, batch_size, n_boot):
    _apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    _draw_distribution(axes.flatten(), model, config, name_to_id, cohorts, df,
                       cases, n_patients, batch_size, n_boot, panel_labels=None)
    fig.subplots_adjust(wspace=0.38, left=0.07, right=0.97, top=0.88, bottom=0.18)
    _save(fig, "distribution_supp")

    # Write population and pre-drug mean/SD to CSV (no model needed).
    vital_prefix_map = {
        "Vital_Systolic": "Vital_Systolic",
        "Vital_Pulse":    "Vital_Pulse",
        "Vital_Resp":     "Vital_Resp",
    }
    rows = []
    for case in cases:
        vital_prefix = vital_prefix_map.get(case.vital, case.vital)
        drug_token = case.tokens[0][0] if case.tokens else None
        xlabel = VITAL_DISPLAY.get(case.vital, case.vital.replace("Vital_", ""))
        print(f"[stats] computing train stats for {xlabel} / {drug_token}")
        pop_mean, pop_sd, pre_mean, pre_sd = _train_vital_stats(vital_prefix, drug_token)
        rows.append({
            "vital": xlabel,
            "drug": drug_token,
            "population_mean": round(pop_mean, 2),
            "population_sd":   round(pop_sd, 2),
            "predrug_mean":    round(pre_mean, 2) if pre_mean is not None else None,
            "predrug_sd":      round(pre_sd, 2)   if pre_sd  is not None else None,
        })
    # Extra row: Naloxone vs Respiratory Rate (from main distribution figure)
    print("[stats] computing train stats for Respiratory rate / Med_naloxone")
    pop_mean, pop_sd, pre_mean, pre_sd = _train_vital_stats("Vital_Resp", "Med_naloxone")
    rows.append({
        "vital": "Respiratory rate",
        "drug": "Med_naloxone",
        "population_mean": round(pop_mean, 2),
        "population_sd":   round(pop_sd, 2),
        "predrug_mean":    round(pre_mean, 2) if pre_mean is not None else None,
        "predrug_sd":      round(pre_sd, 2)   if pre_sd  is not None else None,
    })

    csv_path = OUTPUT_DIR / "distribution_supp_stats.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")


# ---------------------------------------------------------------------------
# Output 2: regression.pdf
# ---------------------------------------------------------------------------

REGRESSION_PAIRS = [
    ("Epinephrine", ["Med_epinephrine"], "Vital_Pulse"),
    ("Albuterol", ["Med_albuterol sulfate", "Med_albuterol sulfate concentrate",
                   "Med_albuterol sulfate hfa"], "Vital_SpO2"),
    ("Ketamine", ["Med_ketamine"], "Vital_Systolic"),
]
MIN_TRIPLETS = 10


def _draw_regression(axes_flat, model, config, name_to_id, df, window, batch_size,
                     panel_label="a", prediction_mode="raw"):
    """Draw the 4 predicted-vs-actual Δ scatters into the given axes."""
    for idx, (med_name, med_tokens, vital) in enumerate(REGRESSION_PAIRS):
        ax = axes_flat[idx]
        vital_label = vital.replace("Vital_", "")
        vital_label = "Systolic BP" if vital_label == "Systolic" else vital_label
        print(f"\n[regression] {med_name} -> {vital_label}")
        result = cached_regression_result(
            model, config, name_to_id, df, med_name, med_tokens, vital, window,
            batch_size, prediction_mode=prediction_mode)
        print(f"    {result['n']} eligible triplets")

        if result["skipped"]:
            print(f"    Skipping — below minimum N={MIN_TRIPLETS}")
            ax.text(0.5, 0.5, f"Insufficient data\n(n={result['n']} < {MIN_TRIPLETS})",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_title(f"{med_name} vs {vital_label}", fontsize=9, fontweight="bold")
            continue

        actual = result["actual"]
        forecast = result["forecast"]
        r2 = result["r2"]
        r2_baseline = result["r2_baseline"]
        pearson_p = result["pearson_p"]
        print(f"    R² = {r2:.3f} (baseline Δ=0: {r2_baseline:.3f}), {format_pvalue(pearson_p)}")

        ax.scatter(actual, forecast, s=12, alpha=0.5, color=COLOR_ACCENT,
                   edgecolors="none", rasterized=True)
        all_vals = np.concatenate([actual, forecast])
        lo, hi = np.percentile(all_vals, [1, 99])
        margin = (hi - lo) * 0.1
        lo, hi = lo - margin, hi + margin
        ax.plot([lo, hi], [lo, hi], color="#333333", linewidth=1, linestyle="--",
                alpha=0.7, zorder=0, label="Perfect")
        ax.axhline(0, color="#888888", linewidth=1, linestyle=":", alpha=0.7,
                   zorder=0, label="Baseline (Δ=0)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"True Δ {vital_label}", fontsize=7)
        ax.set_ylabel(f"Predicted Δ {vital_label}", fontsize=7)
        ax.set_title(f"{med_name} vs {vital_label} (n={result['n']} administrations)",
                     fontsize=9, fontweight="bold")
        ax.tick_params(axis="both", which="major", length=2, width=0.4)
        stat_text = (f"MINT R² = {r2:.3f}\nBaseline (Δ = 0) R² = {r2_baseline:.3f}")
        # PVALUE VIZ
        # stat_text += f"\n{format_pvalue(pearson_p)}"
        ax.text(0.05, 0.95, stat_text, transform=ax.transAxes, fontsize=7,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.9, edgecolor="#cccccc", linewidth=0.5))
        ax.legend(fontsize=6, frameon=False, loc="lower right")

    axes_flat[0].text(-0.10, 1.06, panel_label, transform=axes_flat[0].transAxes,
                      fontsize=11, fontweight="bold", va="top")


def run_regression(model, config, name_to_id, df, window, batch_size,
                   prediction_mode="raw"):
    _apply_style()
    fig, axes = plt.subplots(1, len(REGRESSION_PAIRS), figsize=(10.5, 3.6))
    _draw_regression(np.atleast_1d(axes).flatten(), model, config, name_to_id,
                     df, window, batch_size, panel_label="a",
                     prediction_mode=prediction_mode)
    fig.subplots_adjust(wspace=0.35)
    _save(fig, "regression")


# ---------------------------------------------------------------------------
# Stacked output: distribution (top) + regression (bottom) in one figure
# ---------------------------------------------------------------------------

def run_stacked(model, config, name_to_id, cohorts, df, cases,
                n_patients, batch_size, n_boot, window, epinephrine_mode="single",
                df_regression=None, regression_prediction_mode="raw"):
    """Single figure: 2×3 distribution histograms (panel a) stacked above the
    1×4 regression scatters (panel b)."""
    _apply_style()
    if df_regression is None:
        df_regression = df

    # Use one 3×3 outer grid so every panel cell has the same physical size.
    # Panel A occupies the first two rows; Panel B uses the third row.
    fig = plt.figure(figsize=(10, 11), layout="tight")
    fig.subplots_adjust(left=0.035, right=0.99, top=0.975, bottom=0.045)
    gs = fig.add_gridspec(3, 3, hspace=0.25, wspace=0.11)

    dist_axes = np.array([fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)])
    _draw_distribution(dist_axes, model, config, name_to_id, cohorts, df, cases,
                       n_patients, batch_size, n_boot, panel_labels="a",
                       epinephrine_mode=epinephrine_mode)
    for ax in dist_axes:
        ax.set_box_aspect(1)

    reg_axes = np.array([fig.add_subplot(gs[2, j]) for j in range(len(REGRESSION_PAIRS))])
    _draw_regression(reg_axes, model, config, name_to_id, df_regression, window, batch_size,
                     panel_label="b", prediction_mode=regression_prediction_mode)
    for ax in reg_axes:
        ax.set_box_aspect(1)

    _save(fig, "distribution_regression")


# ---------------------------------------------------------------------------
# Output 3: forest_supplementary.pdf
# ---------------------------------------------------------------------------

FOREST_VITALS = [("HR", "Vital_Pulse"), ("BP", "Vital_Systolic"),
                 ("RR", "Vital_Resp"), ("GCS", "Vital_Glasgow Coma Scale Score")]


def compute_forest_panels(model, config, name_to_id, id_to_name, cohorts,
                          drugs, n_patients, batch_size, n_boot):
    """Single-drug counterfactual Δ per (vital, drug), with direction match.

    Returns (panels, n_eval, token_counts). Reused by both run_forest and
    forest_or; the shared pickle cache means the second caller recomputes only
    the (cheap) bootstrap CI, not the model forward passes.
    """
    general = cohorts["general"]
    n_eval = len(general)  # every drug is evaluated on the full cohort (one case/visit)
    token_counts = count_tokens_in_cohort(general, id_to_name)

    panels = []
    for col, vital in FOREST_VITALS:
        print(f"\n[forest] {col} ({vital})")
        baseline = cached_expected(
            model, config, name_to_id, general, vital, [],
            "general", n_patients, batch_size, desc=f"  {col} baseline")

        rows = []
        for drug in drugs:
            exp_dir = drug["directions"][col]
            if exp_dir is None:
                continue
            token = canonical_token(drug["candidates"], token_counts)
            cf = cached_expected(
                model, config, name_to_id, general, vital, [(token, 1.0)],
                "general", n_patients, batch_size, desc=f"  {drug['name']}")
            delta = cf - baseline
            mean_delta = float(delta.mean())
            ci_lo, ci_hi = bootstrap_delta_ci(baseline, cf, n_boot=n_boot)
            match = np.sign(mean_delta) == exp_dir
            rows.append({"name": drug["name"], "token": token, "vital_col": col,
                         "mean": mean_delta, "ci_lo": ci_lo, "ci_hi": ci_hi,
                         "match": match, "n": n_eval})
            print(f"    {drug['name']:20s} Δ={mean_delta:+.2f} "
                  f"({ci_lo:+.2f}, {ci_hi:+.2f}) n={n_eval} "
                  f"{'match' if match else 'MISMATCH'} (expected {exp_dir:+d})")
        panels.append((col, vital, sorted(rows, key=lambda r: r["mean"])))
    return panels, n_eval, token_counts


# Drugs seen fewer than this many times in the pre-disposition training set are
# flagged (italic label) as an implicit exclusion.
FOREST_TRAIN_MIN = 1000

# Drugs seen fewer than this many times in the pre-disposition training set are
# dropped from the forest entirely (too few administrations to trust).
FOREST_TRAIN_EXCLUDE = 10


def _prepare_forest_panels(model, config, name_to_id, id_to_name, cohorts,
                           drugs, n_patients, batch_size, n_boot):
    """Compute, filter, and write CSVs. Returns (panels, n_eval, train_counts, or_fit)."""
    panels, n_eval, _ = compute_forest_panels(
        model, config, name_to_id, id_to_name, cohorts,
        drugs, n_patients, batch_size, n_boot)

    forest_tokens = sorted({r["token"] for _, _, rows in panels for r in rows})
    train_counts = train_token_counts(forest_tokens)

    dropped = sorted({r["name"] for _, _, rows in panels for r in rows
                      if train_counts.get(r["token"], 0) < FOREST_TRAIN_EXCLUDE})
    if dropped:
        print(f"[forest] excluding {len(dropped)} drug(s) with "
              f"< {FOREST_TRAIN_EXCLUDE} train administrations: {', '.join(dropped)}")
    panels = [(col, vital,
               [r for r in rows
                if train_counts.get(r["token"], 0) >= FOREST_TRAIN_EXCLUDE])
              for col, vital, rows in panels]

    all_cases = [r for _, _, rows in panels for r in rows]
    or_fit = fit_correct_odds_ratio(all_cases, train_counts)
    _write_forest_or_csv(all_cases, train_counts, or_fit)
    return panels, n_eval, train_counts, or_fit


def _print_or(or_fit):
    print("\n--- Odds ratio summary ---")
    print(f"  n_correct / n_total : {or_fit['n_correct']} / {or_fit['n_total']}")
    acc = or_fit['n_correct'] / or_fit['n_total'] if or_fit['n_total'] else float('nan')
    print(f"  Accuracy            : {acc:.1%}")
    print(f"  OR per 10× training : {or_fit['odds_ratio']:.3f} "
          f"(95% CI {or_fit['or_lo']:.3f}–{or_fit['or_hi']:.3f})")
    print(f"  p-value             : {or_fit['pvalue']:.4f}")
    print("--------------------------\n")


def run_forest(model, config, name_to_id, id_to_name, cohorts,
               drugs, n_patients, batch_size, n_boot):
    panels, n_eval, train_counts, or_fit = _prepare_forest_panels(
        model, config, name_to_id, id_to_name, cohorts,
        drugs, n_patients, batch_size, n_boot)
    _print_or(or_fit)

    # Two rows, two columns: (HR, BP) top; (RR, GCS) bottom.
    _apply_style()
    panel_list = [p for p in panels]   # HR, BP, RR, GCS

    top_panels = panel_list[:2]   # HR, BP
    bot_panels = panel_list[2:]   # RR, GCS
    max_top_rows = max(max(len(p[2]) for p in top_panels), 1)
    max_bot_rows = max(max(len(p[2]) for p in bot_panels), 1)

    row_h = 0.32
    row_pad = 1.4
    top_h = max(max_top_rows * row_h + row_pad, 4.0)
    bot_h = max(max_bot_rows * row_h + row_pad, 4.0)
    fig_w = 12.0

    fig = plt.figure(figsize=(fig_w, top_h + bot_h + 0.5), constrained_layout=True)
    sub_top, sub_bot = fig.subfigures(2, 1, height_ratios=[top_h, bot_h], hspace=0.05)

    gs_top = sub_top.add_gridspec(max_top_rows, 2, hspace=0.0, wspace=0.08)
    top_axes = [sub_top.add_subplot(gs_top[:, c]) for c in range(2)]

    gs_bot = sub_bot.add_gridspec(max_bot_rows, 2, hspace=0.0, wspace=0.08)
    bot_axes = [sub_bot.add_subplot(gs_bot[:, c]) for c in range(2)]

    all_axes = top_axes + bot_axes
    for ax, (col, vital, rows) in zip(all_axes, panel_list):
        _plot_forest_axis(ax, col, vital, rows, n_eval, train_counts, compact=True)
        ax.set_xlim(xmax=max(0.1, ax.get_xlim()[-1]))

    _save(fig, "forest_supplementary")


def _build_forest_axes(sub, panels, train_counts, n_eval):
    """Draw four forest panels side-by-side inside a subfigure.

    Each panel gets its own column; column widths are equal so the four
    panels sit neatly next to each other.  The y-extent of every axis is
    set so that per-drug row heights are identical across panels.
    """
    hr_panel, bp_panel, rr_panel, gcs_panel = panels
    panel_list = [hr_panel, bp_panel, rr_panel, gcs_panel]
    max_rows = max(max(len(p[2]) for p in panel_list), 1)

    gs = sub.add_gridspec(max_rows, 4, hspace=0.0, wspace=1.2)
    axes = [sub.add_subplot(gs[:, c]) for c in range(4)]

    panel_labels = ["d", "e", "f", "g"]
    for ax, (col, vital, rows), lbl in zip(axes, panel_list, panel_labels):
        _plot_forest_axis(ax, col, vital, rows, n_eval, train_counts, compact=True)
        ax.text(-0.32, 1.06, lbl, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top")

    return axes


def run_composite_supp(model, config, name_to_id, id_to_name, cohorts, df,
                       dist_cases, drugs, n_patients, batch_size, n_boot):
    """Composite supplementary figure: distribution top row + forest bottom row."""
    panels, n_eval, train_counts, or_fit = _prepare_forest_panels(
        model, config, name_to_id, id_to_name, cohorts,
        drugs, n_patients, batch_size, n_boot)
    _print_or(or_fit)

    _apply_style()

    # Figure sizing: top row is fixed height; bottom scales with drug count.
    max_forest_rows = max(max(len(p[2]) for p in panels), 1)
    top_h = 3.0        # histogram row (inches)
    bot_h = max(max_forest_rows * 0.32 + 1.4, 4.0)
    fig_w = 19.0       # wide for 4 panels + drug-name labels

    fig = plt.figure(figsize=(fig_w, top_h + bot_h), constrained_layout=False)
    sub_top, sub_bot = fig.subfigures(2, 1, height_ratios=[top_h, bot_h],
                                      hspace=0.06)

    # --- Top row: 3 distribution histograms ---
    dist_axes = sub_top.subplots(1, 3)
    sub_top.subplots_adjust(left=0.055, right=0.985, top=0.85, bottom=0.20,
                            wspace=0.35)
    _draw_distribution(dist_axes.flatten(), model, config, name_to_id, cohorts, df,
                       dist_cases, n_patients, batch_size, n_boot,
                       panel_labels=["a", "b", "c"])

    # --- Bottom row: 4 forest panels ---
    sub_bot.subplots_adjust(left=0.10, right=0.99, top=0.91, bottom=0.09,
                            wspace=1.3)
    _build_forest_axes(sub_bot, panels, train_counts, n_eval)

    _save(fig, "forest_supplementary")


def _write_forest_or_csv(cases, tok_counts, fit):
    """Write the correctness-vs-frequency summary + per-case detail to CSV.

    Two files: forest_or_summary.csv (one row: the logistic model, counts, and
    odds ratio) and forest_or.csv (per (drug, vital) case detail).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame([{
        "model": "P(correct) ~ log10(train_ed_count)",
        "n_correct": fit["n_correct"],
        "n_total": fit["n_total"],
        "accuracy": round(fit["n_correct"] / fit["n_total"], 4) if fit["n_total"] else float("nan"),
        "odds_ratio_per_10x": round(fit["odds_ratio"], 4),
        "or_ci_lo": round(fit["or_lo"], 4),
        "or_ci_hi": round(fit["or_hi"], 4),
        "beta": round(fit["beta"], 4),
        "pvalue": round(fit["pvalue"], 4),
    }])
    summary.to_csv(OUTPUT_DIR / "forest_or_summary.csv", index=False)
    print(f"  Saved forest_or_summary.csv -> {OUTPUT_DIR}/forest_or_summary.csv")

    rows = sorted(cases, key=lambda r: tok_counts[r["token"]], reverse=True)
    detail = pd.DataFrame([{
        "drug": r["name"],
        "vital": r["vital_col"],
        "token": r["token"],
        "train_ed_count": tok_counts[r["token"]],
        "delta": round(r["mean"], 2),
        "correct": bool(r["match"]),
    } for r in rows])
    detail.to_csv(OUTPUT_DIR / "forest_or.csv", index=False)
    print(f"  Saved forest_or.csv -> {OUTPUT_DIR}/forest_or.csv")


def _forest_label(name, train_count):
    """Drug label with its training-set occurrence count, e.g. 'Ketamine (n=18,262)'."""
    return f"{name} (n={train_count:,})"


VITAL_DISPLAY = {
    "Vital_Pulse": "Pulse",
    "Vital_Systolic": "Systolic BP",
    "Vital_Resp": "Respiratory rate",
    "Vital_Glasgow Coma Scale Score": "Glasgow Coma Scale Score",
    "Vital_SpO2": "SpO2",
}


def _plot_forest_axis(ax, col, vital, rows, n_eval, train_counts, compact=False):
    vital_label = VITAL_DISPLAY.get(vital, vital.replace("Vital_", ""))
    if not rows:
        ax.text(0.5, 0.5, "No drugs", ha="center", va="center",
                transform=ax.transAxes, fontsize=10)
        ax.set_title(vital_label, fontsize=10, fontweight="bold")
        return

    y = np.arange(len(rows))
    means = np.array([r["mean"] for r in rows])
    err_lo = means - np.array([r["ci_lo"] for r in rows])
    err_hi = np.array([r["ci_hi"] for r in rows]) - means

    ax.axvline(0, color="#888888", linewidth=0.6, linestyle="--", alpha=0.7, zorder=1)
    ax.errorbar(means, y, xerr=[err_lo, err_hi], fmt="none", ecolor="black",
                capsize=2.5, linewidth=1.0, capthick=1.0, zorder=2)
    for i, r in enumerate(rows):
        color = COLOR_DARK if r["match"] else "#CC4444"
        ax.scatter(r["mean"], i, s=34, color=color, marker="o",
                   edgecolors="none", zorder=3)

    xmax = float(np.max(means + err_hi))
    xmin = float(np.min(means - err_lo))
    if vital == "Vital_Glasgow Coma Scale Score":
        xmax = min(xmax, 0.1)
    span = xmax - xmin if xmax > xmin else 1.0
    ax.set_xlim(xmin - 0.08 * span, xmax + 0.08 * span)

    counts = [train_counts.get(r["token"], 0) for r in rows]
    lbl_size = 9
    ax.set_yticks(y)
    labels = ax.set_yticklabels(
        [_forest_label(r["name"], c) for r, c in zip(rows, counts)], fontsize=lbl_size)
    for lbl, c in zip(labels, counts):
        if c < FOREST_TRAIN_MIN:
            lbl.set_fontstyle("italic")
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel(f"Δ {vital_label} (cf − baseline)", fontsize=10)
    ax.set_title(f"{vital_label}\n(n={n_eval:,})", fontsize=11, fontweight="bold", pad=3)
    ax.tick_params(axis="both", which="major", length=2.5, width=0.5, labelsize=lbl_size)


# ---------------------------------------------------------------------------
# Training-set frequency vs counterfactual correctness
# ---------------------------------------------------------------------------

def train_token_counts(tokens):
    """Total occurrences of each token in train.feather before the first
    disposition (Admit/Discharge/ICU Start) of its encounter.

    Returns {token_name: count}. Streams once over the training feather.
    """
    wanted = set(tokens)
    print("Loading train.feather for ED token counts...")
    df = pd.read_feather(DATA_DIR / "train.feather", columns=["encounter_key", "name", "t"])

    # First-disposition time per encounter (inf if none).
    disp = df[df["name"].isin(DISPOSITION_TOKENS)]
    disp_time = disp.groupby("encounter_key", sort=False)["t"].min()

    df = df[df["name"].isin(wanted)].copy()
    df["disp_t"] = df["encounter_key"].map(disp_time).fillna(np.inf)
    before = df[df["t"] < df["disp_t"]]
    counts = before["name"].value_counts().to_dict()
    return {tok: int(counts.get(tok, 0)) for tok in wanted}


def fit_correct_odds_ratio(cases, tok_counts, n_boot=None):
    """Pooled logistic model  P(correct) ~ log10(train count).

    One observation per (drug, vital) case; outcome = correct direction (match).
    Fit with an unregularized MLE (statsmodels Logit) so the coefficient is not
    shrunk toward zero; the CI is the analytic Wald interval from the fit (no
    bootstrap — a bootstrap over sklearn's default L2-penalised fit was pinning
    the lower bound artificially at OR≈1). Returns the odds ratio per 10× more
    training samples, its 95% CI, the Wald p-value, and the correct count.
    OR > 1 means more training data raises the odds of a correct direction.
    """
    import statsmodels.api as sm

    log_count = np.array([np.log10(tok_counts[r["token"]] + 1) for r in cases])
    y = np.array([1 if r["match"] else 0 for r in cases], dtype=int)  # 1 = correct
    n_correct = int(y.sum())

    if len(np.unique(y)) < 2:
        return {"odds_ratio": 1.0, "or_lo": float("nan"), "or_hi": float("nan"),
                "beta": 0.0, "pvalue": float("nan"),
                "n_correct": n_correct, "n_total": len(y)}

    X = sm.add_constant(log_count)
    res = sm.Logit(y, X).fit(disp=0)
    beta = float(res.params[1])
    ci_lo, ci_hi = res.conf_int(alpha=0.05)[1]  # Wald CI on the slope
    return {"odds_ratio": float(np.exp(beta)),
            "or_lo": float(np.exp(ci_lo)), "or_hi": float(np.exp(ci_hi)),
            "beta": beta, "pvalue": float(res.pvalues[1]),
            "n_correct": n_correct, "n_total": len(y)}


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save(fig, name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name} -> {OUTPUT_DIR}/{name}.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global N_BOOTSTRAP
    parser = argparse.ArgumentParser(description="Counterfactual Figure v2")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--debug", action="store_true",
                        help="N_BOOT=50 and cap the cohort at 200 patients")
    parser.add_argument("--n_patients", type=int, default=None,
                        help="Cap cohort size (part of the cache key)")
    parser.add_argument("--window", type=float, default=30.0,
                        help="Regression triplet window in minutes")
    parser.add_argument("--epinephrine_mode", type=str, default="triple",
                        choices=["single", "triple"],
                        help="Distribution mode for epinephrine: one dose or three doses 5 min apart")
    parser.add_argument("--regression_prediction_mode", type=str, default="raw",
                        choices=["raw", "rounded"],
                        help="Regression forecast mode: raw vital values or rounded reporting values")
    parser.add_argument("--output",
                        choices=["stacked", "distribution", "distribution_supp",
                                 "regression", "forest"],
                        default=None, help="Run only one output")
    args = parser.parse_args()

    n_boot = 50 if args.debug else N_BOOTSTRAP
    n_patients = args.n_patients
    if n_patients is None and args.debug:
        n_patients = 200

    print("Loading vocab, model, cohort...")
    vocab, name_to_id, id_to_name = load_vocab()
    model, config = load_model(checkpoint_path=CHECKPOINT_PATH)
    df = load_test_cohort(acuity_filter=ACUITY_FILTER)
    df_full = load_test_cohort(acuity_filter=None)
    print(f"  {df['encounter_key'].nunique()} encounters")
    print(f"  {df_full['encounter_key'].nunique()} full-cohort encounters")

    cohorts = None

    def ensure_cohorts():
        nonlocal cohorts
        if cohorts is None:
            cohorts = build_cohorts(df, name_to_id, n_patients)
        return cohorts

    if args.output in (None, "stacked"):
        ensure_cohorts()
        run_stacked(model, config, name_to_id, cohorts, df,
                    default_distribution_cases(), n_patients,
                    args.batch_size, n_boot, args.window,
                    epinephrine_mode=args.epinephrine_mode,
                    df_regression=df_full,
                    regression_prediction_mode=args.regression_prediction_mode)

    if args.output == "distribution":
        ensure_cohorts()
        run_distribution(model, config, name_to_id, cohorts, df,
                         default_distribution_cases(), n_patients,
                         args.batch_size, n_boot,
                         epinephrine_mode=args.epinephrine_mode)

    if args.output == "regression":
        run_regression(model, config, name_to_id, df_full, args.window, args.batch_size,
                       prediction_mode=args.regression_prediction_mode)

    if args.output in (None, "distribution_supp"):
        ensure_cohorts()
        run_distribution_supp(model, config, name_to_id, cohorts, df,
                              default_distribution_supp_cases(), n_patients,
                              args.batch_size, n_boot)

    if args.output in (None, "forest"):
        ensure_cohorts()
        drugs = load_driver(name_to_id)
        run_forest(model, config, name_to_id, id_to_name, cohorts,
                   drugs, n_patients, args.batch_size, n_boot)

    print("\nDONE. Outputs in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
