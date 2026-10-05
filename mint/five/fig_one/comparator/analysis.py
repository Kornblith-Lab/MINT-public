"""
analysis.py — Per-subcohort and macro AUROC/AUPRC for human review comparator.

Usage:
    uv run --with pandas --with numpy --with scikit-learn --with openpyxl python -m mint.five.fig_one.comparator.analysis --excel artifacts/fig1/human_review/ppv.xlsx

Also writes every computed point estimate + 95% CI (and, for paired deltas, a
significance flag) to <out-dir>/comparator_metrics.csv in long/tidy form.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

SUBCOHORTS = ["A", "B", "C", "D"]
DEFAULT_METHODS = ["mint", "gpt_5_mini", "gpt_5", "human_alone", "human_mint"]
N_BOOT = 1000
RNG_SEED = 42
DEFAULT_OUT_DIR = "artifacts/fig1/human_review"

CSV_FIELDS = [
    "section", "analysis", "method", "reference", "cohort", "n", "n_pos",
    "metric", "value", "ci_low", "ci_high", "significant",
]


# ── Bootstrap helpers ────────────────────────────────────────────────────────

def _boot_metric(labels, scores, metric_fn, n, rng):
    """Percentile bootstrap CI for a single metric."""
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(labels), size=len(labels))
        y, s = labels[idx], scores[idx]
        if len(np.unique(y)) < 2:
            continue
        vals.append(metric_fn(y, s))
    v = np.array(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def _boot_paired_diff(labels, scores_a, scores_b, metric_fn, n, rng):
    """Percentile bootstrap CI for metric(a) - metric(b), paired by row."""
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(labels), size=len(labels))
        y = labels[idx]
        if len(np.unique(y)) < 2:
            continue
        diffs.append(metric_fn(y, scores_a[idx]) - metric_fn(y, scores_b[idx]))
    d = np.array(diffs)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def _macro_boot(df, method, metric_fn, n, rng):
    """
    Macro CI: each iteration resamples each subcohort independently, computes
    the metric per cohort, then averages. Returns (lo, hi).
    """
    vals = []
    for _ in range(n):
        cohort_vals = []
        for sub in SUBCOHORTS:
            sub_df = df[df["subgroup"] == sub]
            idx = rng.integers(0, len(sub_df), size=len(sub_df))
            y, s = sub_df["label"].values[idx], sub_df[method].values[idx]
            if len(np.unique(y)) < 2:
                break
            cohort_vals.append(metric_fn(y, s))
        if len(cohort_vals) == len(SUBCOHORTS):
            vals.append(np.mean(cohort_vals))
    v = np.array(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def _macro_boot_paired_diff(df, method_a, method_b, metric_fn, n, rng):
    """
    Paired macro CI for metric(a) - metric(b): each iteration resamples the
    same rows within each cohort for both methods, computes per-cohort diffs,
    then averages. Returns (lo, hi).
    """
    diffs = []
    for _ in range(n):
        cohort_diffs = []
        for sub in SUBCOHORTS:
            sub_df = df[df["subgroup"] == sub]
            idx = rng.integers(0, len(sub_df), size=len(sub_df))
            y = sub_df["label"].values[idx]
            if len(np.unique(y)) < 2:
                break
            sa = sub_df[method_a].values[idx]
            sb = sub_df[method_b].values[idx]
            cohort_diffs.append(metric_fn(y, sa) - metric_fn(y, sb))
        if len(cohort_diffs) == len(SUBCOHORTS):
            diffs.append(np.mean(cohort_diffs))
    d = np.array(diffs)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ── CSV collection ───────────────────────────────────────────────────────────

def _add_row(rows, section, analysis, method, cohort, metric, val, lo, hi,
             reference="", n="", n_pos="", is_diff=False):
    """Append one long-format record. For deltas, flag CIs excluding 0."""
    rows.append({
        "section": section,
        "analysis": analysis,
        "method": method,
        "reference": reference,
        "cohort": cohort,
        "n": n,
        "n_pos": n_pos,
        "metric": metric,
        "value": round(float(val), 6),
        "ci_low": round(float(lo), 6),
        "ci_high": round(float(hi), 6),
        "significant": (not (lo <= 0 <= hi)) if is_diff else "",
    })


def _write_csv(rows, out_dir):
    out_path = Path(out_dir) / "comparator_metrics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


# ── Macro paired Δ section ───────────────────────────────────────────────────

def _print_macro_diff_section(df, complete_methods, ref, section_num, n_boot, rng, W, rows):
    complete_others = [m for m in complete_methods if m != ref]

    print("=" * 80)
    print(f"SECTION {section_num} — MACRO PAIRED Δ vs {ref.upper()}  (method − {ref}, 95% CI)")
    if ref not in complete_methods:
        print(f"  '{ref}' is not complete across all cohorts — cannot compute macro diffs.")
    elif not complete_others:
        print(f"  No other complete methods to compare against {ref}.")
    else:
        print("=" * 80)
        hdr = f"{'Method':<{W}}  {'Δ Macro AUROC':>26}  {'Δ Macro AUPRC':>26}"
        print(hdr)
        print("-" * len(hdr))
        for method in complete_others:
            cohort_auroc_diffs, cohort_auprc_diffs = [], []
            for sub in SUBCOHORTS:
                sub_df = df[df["subgroup"] == sub]
                y = sub_df["label"].values
                sr, so = sub_df[ref].values, sub_df[method].values
                cohort_auroc_diffs.append(roc_auc_score(y, so) - roc_auc_score(y, sr))
                cohort_auprc_diffs.append(average_precision_score(y, so) - average_precision_score(y, sr))
            d_auroc = float(np.mean(cohort_auroc_diffs))
            d_auprc = float(np.mean(cohort_auprc_diffs))
            alo, ahi = _macro_boot_paired_diff(df, method, ref, roc_auc_score, n_boot, rng)
            plo, phi = _macro_boot_paired_diff(df, method, ref, average_precision_score, n_boot, rng)
            _add_row(rows, section_num, "macro_delta", method, "macro", "AUROC",
                     d_auroc, alo, ahi, reference=ref, is_diff=True)
            _add_row(rows, section_num, "macro_delta", method, "macro", "AUPRC",
                     d_auprc, plo, phi, reference=ref, is_diff=True)
            print(f"{method:<{W}}  {fmt(d_auroc, alo, ahi):>26}  {fmt(d_auprc, plo, phi):>26}")
    print()


# ── Formatting ───────────────────────────────────────────────────────────────

def fmt(val, lo, hi):
    return f"{val:+.3f} ({lo:+.3f}–{hi:+.3f})"


def fmt_abs(val, lo, hi):
    return f"{val:.3f} ({lo:.3f}–{hi:.3f})"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", default="artifacts/fig1/human_review/ppv.xlsx")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    df = pd.read_excel(args.excel, sheet_name="model")
    rng = np.random.default_rng(RNG_SEED)

    methods = [m for m in args.methods if m in df.columns]
    missing = [m for m in args.methods if m not in df.columns]
    if missing:
        print(f"[WARN] columns not found, skipping: {missing}\n")

    rows: list[dict] = []

    ref = "mint"
    if ref not in methods:
        print(f"[ERROR] reference method '{ref}' not in data — cannot compute diffs")
        return

    # ── Section 1: Per-subcohort AUROC + AUPRC ──────────────────────────────
    W = 16
    print("=" * 80)
    print("SECTION 1 — PER-SUBCOHORT AUROC / AUPRC  (95% CI, percentile bootstrap)")
    print("=" * 80)
    hdr = f"{'Method':<{W}}  {'Cohort':>6}  {'n':>4}  {'pos':>3}  {'AUROC':>24}  {'AUPRC':>24}"
    print(hdr)
    print("-" * len(hdr))

    # cache point estimates for use in diff section
    _cache: dict[tuple, dict] = {}

    for method in methods:
        for sub in SUBCOHORTS:
            sub_df = df[df["subgroup"] == sub].dropna(subset=[method])
            n, pos = len(sub_df), int(sub_df["label"].sum())
            key = (method, sub)
            if n == 0:
                print(f"{method:<{W}}  {sub:>6}  {'—':>4}  {'—':>3}  {'no data':>24}  {'no data':>24}")
                continue
            if len(np.unique(sub_df["label"].values)) < 2:
                print(f"{method:<{W}}  {sub:>6}  {n:>4}  {pos:>3}  {'only 1 class':>24}  {'only 1 class':>24}")
                continue
            y, s = sub_df["label"].values, sub_df[method].values
            auroc = roc_auc_score(y, s)
            auprc = average_precision_score(y, s)
            alo, ahi = _boot_metric(y, s, roc_auc_score, args.n_boot, rng)
            plo, phi = _boot_metric(y, s, average_precision_score, args.n_boot, rng)
            _cache[key] = {"auroc": auroc, "auprc": auprc, "y": y, "s": s}
            _add_row(rows, 1, "per_cohort_absolute", method, sub, "AUROC",
                     auroc, alo, ahi, n=n, n_pos=pos)
            _add_row(rows, 1, "per_cohort_absolute", method, sub, "AUPRC",
                     auprc, plo, phi, n=n, n_pos=pos)
            print(f"{method:<{W}}  {sub:>6}  {n:>4}  {pos:>3}  {fmt_abs(auroc, alo, ahi):>24}  {fmt_abs(auprc, plo, phi):>24}")
        print()

    # ── Section 2: Per-subcohort paired Δ vs mint ───────────────────────────
    other_methods = [m for m in methods if m != ref]

    print("=" * 80)
    print(f"SECTION 2 — PER-SUBCOHORT PAIRED Δ vs {ref.upper()}  (method − mint, 95% CI)")
    print("Paired on the same rows available for both methods in each cohort.")
    print("=" * 80)
    hdr2 = f"{'Method':<{W}}  {'Cohort':>6}  {'n':>4}  {'ΔAUROC':>26}  {'ΔAUPRC':>26}"
    print(hdr2)
    print("-" * len(hdr2))

    for method in other_methods:
        for sub in SUBCOHORTS:
            # paired: rows where BOTH mint and method are non-null
            sub_df = df[df["subgroup"] == sub].dropna(subset=[ref, method])
            n = len(sub_df)
            if n == 0:
                print(f"{method:<{W}}  {sub:>6}  {'—':>4}  {'no data':>26}  {'no data':>26}")
                continue
            if len(np.unique(sub_df["label"].values)) < 2:
                print(f"{method:<{W}}  {sub:>6}  {n:>4}  {'only 1 class':>26}  {'only 1 class':>26}")
                continue
            y  = sub_df["label"].values
            sm = sub_df[ref].values
            so = sub_df[method].values
            d_auroc = roc_auc_score(y, so) - roc_auc_score(y, sm)
            d_auprc = average_precision_score(y, so) - average_precision_score(y, sm)
            alo, ahi = _boot_paired_diff(y, so, sm, roc_auc_score, args.n_boot, rng)
            plo, phi = _boot_paired_diff(y, so, sm, average_precision_score, args.n_boot, rng)
            _add_row(rows, 2, "per_cohort_delta", method, sub, "AUROC", d_auroc, alo, ahi,
                     reference=ref, n=n, n_pos=int(sub_df["label"].sum()), is_diff=True)
            _add_row(rows, 2, "per_cohort_delta", method, sub, "AUPRC", d_auprc, plo, phi,
                     reference=ref, n=n, n_pos=int(sub_df["label"].sum()), is_diff=True)
            print(f"{method:<{W}}  {sub:>6}  {n:>4}  {fmt(d_auroc, alo, ahi):>26}  {fmt(d_auprc, plo, phi):>26}")
        print()

    # ── Section 3: Macro AUROC + AUPRC ─────────────────────────────────────
    complete_methods = [m for m in methods if df[m].isna().sum() == 0]

    print("=" * 80)
    print("SECTION 3 — MACRO (mean-of-cohorts) AUROC / AUPRC  (95% CI)")
    if not complete_methods:
        print("  No methods with complete data across all cohorts.")
    else:
        print(f"Complete methods: {', '.join(complete_methods)}")
        print("=" * 80)
        hdr3 = f"{'Method':<{W}}  {'Macro AUROC':>24}  {'Macro AUPRC':>24}"
        print(hdr3)
        print("-" * len(hdr3))
        for method in complete_methods:
            cohort_aurocs, cohort_auprcs = [], []
            for sub in SUBCOHORTS:
                sub_df = df[df["subgroup"] == sub]
                y, s = sub_df["label"].values, sub_df[method].values
                cohort_aurocs.append(roc_auc_score(y, s))
                cohort_auprcs.append(average_precision_score(y, s))
            macro_auroc = float(np.mean(cohort_aurocs))
            macro_auprc = float(np.mean(cohort_auprcs))
            alo, ahi = _macro_boot(df, method, roc_auc_score, args.n_boot, rng)
            plo, phi = _macro_boot(df, method, average_precision_score, args.n_boot, rng)
            _add_row(rows, 3, "macro_absolute", method, "macro", "AUROC", macro_auroc, alo, ahi)
            _add_row(rows, 3, "macro_absolute", method, "macro", "AUPRC", macro_auprc, plo, phi)
            print(f"{method:<{W}}  {fmt_abs(macro_auroc, alo, ahi):>24}  {fmt_abs(macro_auprc, plo, phi):>24}")
    print()

    # ── Section 4: Macro paired Δ vs mint ───────────────────────────────────
    _print_macro_diff_section(df, complete_methods, ref, 4, args.n_boot, rng, W, rows)

    # ── Section 5: Macro paired Δ vs human_alone ────────────────────────────
    human_ref = "human_alone"
    if human_ref in methods:
        _print_macro_diff_section(df, complete_methods, human_ref, 5, args.n_boot, rng, W, rows)

    out_path = _write_csv(rows, args.out_dir)
    print(f"[OK] wrote {len(rows)} rows → {out_path}")


if __name__ == "__main__":
    main()
