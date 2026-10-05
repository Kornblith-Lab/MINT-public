#!/usr/bin/env bash
# Production run commands for the NEJM Fig. 1 panels.
#
# Two outputs, both in BAR mode (bar plots of AUROC/AUPRC per task with 95%
# paired-bootstrap CIs; a * marks methods that differ significantly from the
# target, mint_softmax, on that task):
#
#   1. Main text     -> MINT, XGBoost, Triage        (legend "MINT")
#   2. Supplementary -> Softmax, Test-time softmax, CDF
#
# Bootstraps are cached under <output_dir>/cache, so re-runs are near-instant.
# Run from the repo root:  bash mint/five/fig_one/run_production.sh
set -euo pipefail

# ── Main text ────────────────────────────────────────────────────────────────
python -m mint.five.fig_one.fig1_nejm \
    --final_dir artifacts/fig1-final \
    --operational_dir artifacts/fig1_operational \
    --output_dir artifacts/fig1_nejm_main \
    --style curve \
    --methods mint_softmax xgboost triage \
    --labels mint_softmax=MINT \
    --target_estimator mint_softmax

# ── Supplementary ─────────────────────────────────────────────────────────────
python -m mint.five.fig_one.fig1_nejm \
    --final_dir artifacts/fig1-final \
    --operational_dir artifacts/fig1_operational \
    --output_dir artifacts/fig1_nejm_supp \
    --style curve \
    --methods mint_softmax mint_tt_mean mint_cdf \
    --labels mint_softmax=Softmax "mint_tt_mean=Test-time softmax" \
    --target_estimator mint_softmax

# ── Supplementary: temporal-fidelity ablation (token resolution) ──────────────
# MINT full (minute) vs MINT Hourly (fig1-60) vs MINT Daily (fig1-1440); diffs vs
# full MINT. Restricted to the five physiologic outcomes (hypoxia, tachypnea,
# tachycardia, hypotension, periarrest) since the ablation only speaks to those.
python -m mint.five.fig_one.fig1_nejm \
    --final_dir artifacts/fig1-final \
    --output_dir artifacts/fig1_nejm_ablation \
    --style curve \
    --methods mint_softmax mint_hour mint_day \
    --tasks hypoxia tachypnea tachycardia hypotension periarrest \
    --labels mint_softmax=MINT \
    --target_estimator mint_softmax \
    --skip_operational

# ── Supplementary: adaptive thresholds (ROC across outcome cutoffs) ────────────
# Hypoxia SpO2 cutoffs (88/90/92/94) | escalating oxygen support (r1/r2/r4/ppv/vent).
python -m mint.five.fig_one.fig_adaptive_thresholds \
    --final_dir artifacts/fig1-final \
    --output_dir artifacts/fig1_nejm_adaptive

# ── Supplementary: token ablations (Δ AUROC/AUPRC vs full MINT) ────────────────
# Heatmap blocks: token-type ablations (Vital/O2 device/Med/Procedure/Lab) and
# temporal fidelity (Hourly/Daily), each ablation - full MINT with paired-diff CIs.
python -m mint.five.fig_one.fig_token_ablation \
    --final_dir artifacts/fig1-final \
    --output_dir artifacts/fig1_nejm_token_ablation

# ── Composite (fig1_composite.png) ───────────────────────────────────────────
# Left: macro-average AUROC and AUPRC curves (mean across all 10 outcomes, with
# dim per-outcome lines behind).  Right: individual AUROC panels — airway column
# (left) and circulation column (right).  Methods: MINT + XGBoost by default.
# CIs on the average are encounter-level (element-wise mean of the 1000-iteration
# bootstrap arrays across tasks — see paired_bootstrap).  First run generates the
# raw-arrays cache (~same time as a regular bootstrap run); subsequent runs are
# instant.  Pass --no-show_xgboost or --show_triage to adjust shown methods, and
# --no-show_dim_curves to hide the per-outcome background lines.
python -m mint.five.fig_one.fig1_nejm \
    --final_dir artifacts/fig1-final \
    --operational_dir artifacts/fig1_operational \
    --output_dir artifacts/fig1_nejm_main \
    --style curve \
    --methods mint_softmax xgboost triage \
    --labels mint_softmax=MINT \
    --target_estimator mint_softmax \
    --show_triage
    # Composite flags are all store_true (default off = the good default):
    #   --hide_xgboost     remove XGBoost from composite
    #   --show_triage      add Triage to composite
    #   --hide_dim_curves  suppress per-outcome background lines
