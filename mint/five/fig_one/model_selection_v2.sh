#!/bin/bash

find output -path "*/mint_v2_nl12_*/ckpt.pt" | while read -r path; do
    identifier=$(basename "$(dirname "$path")")
    python -m mint.five.fig_one.fig1 \
        --lookahead 5 \
        --save_path "artifacts/model_selection_v2/${identifier}" \
        --max_len 512 \
        --mode vitals_all \
        --skip_exclusion \
	    --skip_xgboost \
        --val \
	    --estimator softmax \
        --checkpoint_path "$path"
done
