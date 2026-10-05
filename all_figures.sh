#!/bin/bash

# Figure 2
./mint/five/fig_one/run_production.sh

# Figure 3
python mint/five/multicenter/analysis.py

# Figure 4
python -m mint.five.fig_dynamic_v2 --stage figures --cdf

# Figure 5
python -m mint.five.fig_interp.run_counterfactual_v2

# ############################################

# Figure 1 supplementary materials

python -m mint.five.fig_one.fig_adaptive_thresholds \
    --final_dir artifacts/fig1-final \
    --output_dir artifacts/fig1_nejm_adaptive

python -m mint.five.fig_one.fig_token_ablation \
    --final_dir artifacts/fig1-final \
    --output_dir artifacts/fig1_nejm_token_ablation

python -m mint.five.fig_one.sample_eff --save_path artifacts/sample_eff --plot_only

# Figure 5 supplementary materials

python -m mint.five.fig_tokenomics.token_figures
