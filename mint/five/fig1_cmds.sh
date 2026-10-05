!/bin/bash

# core run
python -m mint.five.fig_one.fig1 --lookahead 5 --save_path artifacts/fig1-final-softmax --max_len 512 --mode vitals_all --skip_exclusion && python -m mint.five.fig_one.fig1 --lookahead 5 --save_path artifacts/fig1-final-softmax --max_len 512 --mode int_all --skip_exclusion --first && python -m mint.five.fig_one.fig1 --lookahead 5 --save_path artifacts/fig1-final-softmax-tt --estimator test_time_softmax --max_len 512 --mode vitals_all --skip_exclusion --skip_xgboost && python -m mint.five.fig_one.fig1 --lookahead 5 --save_path artifacts/fig1-final-softmax-tt --estimator test_time_softmax --max_len 512 --mode int_all --skip_exclusion --first --skip_xgboost && python -m mint.five.fig_one.fig1 --lookahead 5 --save_path artifacts/fig1-final-cdf --estimator cdf --max_len 512 --mode vitals_all --skip_exclusion --skip_xgboost && python -m mint.five.fig_one.fig1 --lookahead 5 --save_path artifacts/fig1-final-cdf --estimator cdf --max_len 512 --mode int_all --skip_exclusion --first --skip_xgboost && python -m mint.five.fig_one.fig1 --lookahead 5 --save_path artifacts/fig1-final-triage --estimator triage --max_len 512 --mode vitals_all --skip_exclusion --skip_xgboost && python -m mint.five.fig_one.fig1 --lookahead 5 --save_path artifacts/fig1-final-triage --estimator triage --max_len 512 --mode int_all --skip_exclusion --first --skip_xgboost && python -m mint.five.fig_one.fig1 --lookahead 5 --save_path artifacts/fig1-final-cdf-tt --estimator test_time_cdf --max_len 512 --mode vitals_all --skip_exclusion --skip_xgboost && python -m mint.five.fig_one.fig1 --lookahead 5 --save_path artifacts/fig1-final-cdf-tt --estimator test_time_cdf --max_len 512 --mode int_all --skip_exclusion --first --skip_xgboost

# multiple thresholds dynamic run
python -m mint.five.fig_one.fig1 --lookahead 5 --save_path artifacts/fig1-multi-hypoxia --max_len 512 --mode h88 h90 h92 h94 --skip_exclusion --skip_xgboost && python -m mint.five.fig_one.fig1 --lookahead 5 --save_path artifacts/fig1-multi-resp --max_len 512 --mode r1 r2 r3 r4 ppv ventilator --skip_exclusion --first --skip_xgboost

# ablations of 60 min and 1440 min
python -m mint.five.fig_one.fig1 --lookahead 5 --save_path artifacts/fig1-60-softmax --max_len 512 --mode vitals_all --skip_exclusion --checkpoint_path output/mint60/ckpt.pt --skip_xgboost && python -m mint.five.fig_one.fig1 --lookahead 5 --save_path artifacts/fig1-60-softmax --max_len 512 --mode int_all --skip_exclusion --first --checkpoint_path output/mint60/ckpt.pt --skip_xgboost && python -m mint.five.fig_one.fig1 --lookahead 5 --save_path artifacts/fig1-1440-softmax --max_len 512 --mode vitals_all --skip_exclusion --checkpoint_path output/mint1440/ckpt.pt --skip_xgboost && python -m mint.five.fig_one.fig1 --lookahead 5 --save_path artifacts/fig1-1440-softmax --max_len 512 --mode int_all --skip_exclusion --first --checkpoint_path output/mint1440/ckpt.pt --skip_xgboost

# ablations of meds, vitals, and procedures
for ABLATION in Vital O2Device Med Procedure Lab; do
  python -m mint.five.fig_one.fig1 \
    --lookahead 5 \
    --save_path "artifacts/fig1-ablate-${ABLATION}" \
    --max_len 512 \
    --mode vitals_all \
    --skip_exclusion \
    --skip_xgboost \
    --ablate "${ABLATION}" && \
  python -m mint.five.fig_one.fig1 \
    --lookahead 5 \
    --save_path "artifacts/fig1-ablate-${ABLATION}" \
    --max_len 512 \
    --mode int_all \
    --skip_exclusion \
    --skip_xgboost \
    --first \
    --ablate "${ABLATION}"
done
