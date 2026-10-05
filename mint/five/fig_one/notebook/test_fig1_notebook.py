"""
Integration test for fig1_notebook_init.py and fig1_notebook_run.py.

Sets up the prerequisite variables (model, tokens, vocab) on a small subset
of real data (first 100 encounters), enables DEBUG mode, then executes both
blocks and verifies the pipeline produces expected outputs.

Usage:
    python -m mint.five.fig_one.notebook.test_fig1_notebook
"""

import sys
import tempfile
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]

import numpy as np
import pandas as pd
import torch

# ─── Load real data (small subset) ──────────────────────────────────────────

# Data lives in the main checkout's output/ dir (not in worktrees)
DATA_DIR = PROJECT_ROOT / "output"
if not DATA_DIR.exists():
    # If running from a worktree, look at the main checkout
    import subprocess
    main_root = Path(subprocess.check_output(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=str(PROJECT_ROOT), text=True
    ).strip()).parent
    DATA_DIR = main_root / "output"

VOCAB_CSV = DATA_DIR / "vocab.csv"
TOKENS_FEATHER = DATA_DIR / "tokens.feather"

assert VOCAB_CSV.exists(), f"Missing {VOCAB_CSV}"
assert TOKENS_FEATHER.exists(), f"Missing {TOKENS_FEATHER}"

vocab = pd.read_csv(VOCAB_CSV)

all_tokens = pd.read_feather(TOKENS_FEATHER)
N_ENCOUNTERS = int(os.environ.get("TEST_N_ENCOUNTERS", 100))
first_encounters = all_tokens["encounter_key"].unique()[:N_ENCOUNTERS]
tokens_subset = all_tokens[all_tokens["encounter_key"].isin(first_encounters)].copy()
tokens = {"test_hospital": tokens_subset}

print(f"Test data: {len(first_encounters)} encounters, {len(tokens_subset)} tokens")
print(f"Vocab: {len(vocab)} entries")

# ─── Load model ─────────────────────────────────────────────────────────────

REPO_ROOT = DATA_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
from model import Delphi, DelphiConfig

CHECKPOINT_PATH = DATA_DIR / "mint" / "ckpt.pt"
assert CHECKPOINT_PATH.exists(), f"Missing {CHECKPOINT_PATH}"

checkpoint = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=False)
config = DelphiConfig(**checkpoint["model_args"])
model = Delphi(config)
model.load_state_dict(checkpoint["model"])
model.eval()

print(f"Model loaded: {config.n_embd}d, {config.n_layer} layers, vocab={config.vocab_size}")

# ─── Execute init block (with DEBUG=True override) ───────────────────────────

# Use a temp directory for outputs so we don't pollute the real output dir
tmp_dir = tempfile.mkdtemp(prefix="fig1_test_")
print(f"Test output dir: {tmp_dir}")

SCRIPT_DIR = Path(__file__).resolve().parent
init_code = (SCRIPT_DIR / "fig1_notebook_init.py").read_text()
# Force DEBUG mode
init_code = init_code.replace("DEBUG = False", "DEBUG = True")
# Redirect output dir
init_code = init_code.replace(
    'OUTPUT_DIR = Path("output")',
    f'OUTPUT_DIR = Path("{tmp_dir}")',
)
# Limit outcomes via env var (default: hypoxia + tachypnea)
test_outcomes = os.environ.get("TEST_OUTCOMES", "hypoxia,tachypnea").split(",")
init_code = init_code.replace(
    'OUTCOMES = ["hypoxia", "tachypnea", "periarrest", "tachycardia", "hypotension"]',
    f'OUTCOMES = {test_outcomes}',
)

init_globals = {"model": model, "tokens": tokens, "vocab": vocab}
exec(compile(init_code, "fig1_notebook_init.py", "exec"), init_globals)

print("\n--- Init block completed ---")
print(f"Hospitals: {list(init_globals['hospital_splits'].keys())}")
print(f"Outcomes: {init_globals['OUTCOMES']}")

# ─── Execute peft block (defines backbone_finetune) ─────────────────────────

peft_code = (SCRIPT_DIR / "fig1_notebook_peft.py").read_text()
exec(compile(peft_code, "fig1_notebook_peft.py", "exec"), init_globals)

# ─── Execute run block ───────────────────────────────────────────────────────

run_code = (SCRIPT_DIR / "fig1_notebook_run.py").read_text()

exec(compile(run_code, "fig1_notebook_run.py", "exec"), init_globals)

print("\n--- Run block completed ---")

# ─── Verify outputs ─────────────────────────────────────────────────────────

all_results = init_globals["all_results"]
print(f"\nResults collected: {len(all_results)} outcome(s)")

# Check that we got at least some results
assert len(all_results) > 0, "No results produced!"

# Verify each result has the expected keys
required_keys = [
    "hospital", "task", "lookahead_min", "incidence",
    "n_train", "n_test", "n_pos_train", "n_pos_test",
    # Dataset stats for interpretability
    "xgb_train_n_total", "xgb_train_n_pos", "xgb_test_n_total", "xgb_test_n_pos",
    "mint_probe_train_n_total", "mint_probe_train_n_pos", "mint_probe_test_n_total", "mint_probe_test_n_pos",
    "Softmax_auprc", "Softmax_auroc",
    "XGBoost_auprc", "XGBoost_auroc",
    "MINT_SVM_auprc", "MINT_SVM_auroc",
    "MINT_LR_auprc", "MINT_LR_auroc",
    "equivalence_samples_multiple",
]

for r in all_results:
    for key in required_keys:
        assert key in r, f"Missing key '{key}' in result for {r.get('task', '?')}"

    # Verify metrics are in valid ranges (all methods may be None if class imbalance)
    for method in ["Softmax", "XGBoost", "MINT_SVM", "MINT_LR"]:
        auprc = r.get(f"{method}_auprc")
        auroc = r.get(f"{method}_auroc")
        if auprc is not None:
            assert 0 <= auprc <= 1, f"{method} AUPRC={auprc} out of range for {r['task']}"
        if auroc is not None:
            assert 0 <= auroc <= 1, f"{method} AUROC={auroc} out of range for {r['task']}"

    # Verify dataset stats are present and sensible
    for prefix in ["xgb", "mint_probe"]:
        train_n = r.get(f"{prefix}_train_n_total")
        train_pos = r.get(f"{prefix}_train_n_pos")
        test_n = r.get(f"{prefix}_test_n_total")
        test_pos = r.get(f"{prefix}_test_n_pos")
        assert train_n is not None and train_n >= 0, f"Invalid {prefix}_train_n_total"
        assert train_pos is not None and 0 <= train_pos <= train_n, f"Invalid {prefix}_train_n_pos"
        assert test_n is not None and test_n >= 0, f"Invalid {prefix}_test_n_total"
        assert test_pos is not None and 0 <= test_pos <= test_n, f"Invalid {prefix}_test_n_pos"

    def _fmt(val):
        return f"{val:.4f}" if val is not None else "N/A"

    print(f"  {r['task']}: Softmax={_fmt(r.get('Softmax_auprc'))} XGB={_fmt(r.get('XGBoost_auprc'))} "
          f"SVM={_fmt(r.get('MINT_SVM_auprc'))} LR={_fmt(r.get('MINT_LR_auprc'))} "
          f"[train: {r['xgb_train_n_total']}({r['xgb_train_n_pos']}+), test: {r['xgb_test_n_total']}({r['xgb_test_n_pos']}+)]")

# Check output files exist
hosp_dir = Path(tmp_dir) / "test_hospital"
if hosp_dir.exists():
    csv_files = list(hosp_dir.glob("*.csv"))
    print(f"\nOutput CSVs: {len(csv_files)}")
    for f in sorted(csv_files):
        df = pd.read_csv(f)
        print(f"  {f.name}: {len(df)} rows, columns={df.columns.tolist()}")
        assert "probs" in df.columns, f"Missing 'probs' column in {f.name}"
        assert "labels" in df.columns, f"Missing 'labels' column in {f.name}"

# Check summary CSV
summary_path = Path(tmp_dir) / "results_summary.csv"
assert summary_path.exists(), "results_summary.csv not created"
summary = pd.read_csv(summary_path)
print(f"\nSummary CSV: {len(summary)} rows")
assert len(summary) == len(all_results)

# Check JSONL
jsonl_path = hosp_dir / "results.jsonl"
if jsonl_path.exists():
    import json
    with open(jsonl_path) as f:
        lines = f.readlines()
    print(f"JSONL results: {len(lines)} entries")
    for line in lines:
        parsed = json.loads(line)
        assert "MINT_SVM_auprc" in parsed, "Missing MINT_SVM_auprc key in JSONL"
        assert "MINT_LR_auprc" in parsed, "Missing MINT_LR_auprc key in JSONL"
        # Check dataset stats are in the output
        assert "xgb_train_n_total" in parsed, "Missing xgb_train_n_total in JSONL"
        assert "xgb_train_n_pos" in parsed, "Missing xgb_train_n_pos in JSONL"
        assert "xgb_test_n_total" in parsed, "Missing xgb_test_n_total in JSONL"
        assert "xgb_test_n_pos" in parsed, "Missing xgb_test_n_pos in JSONL"
        assert "mint_probe_train_n_total" in parsed, "Missing mint_probe_train_n_total in JSONL"
        assert "n_pos_train" in parsed, "Missing n_pos_train in JSONL"

# Verify at least one outcome trained the linear probes successfully
any_probes_trained = any(r["MINT_SVM_auprc"] is not None for r in all_results)
assert any_probes_trained, "No outcome successfully trained linear probes!"

# ─── Execute stats block (Block 4) ──────────────────────────────────────────

print("\n--- Running stats block (Block 4) ---")

stats_code = (SCRIPT_DIR / "fig1_notebook_stats.py").read_text()
# Use fewer bootstrap iterations for speed in testing
stats_code = stats_code.replace("STATS_N_BOOTSTRAP = 1000", "STATS_N_BOOTSTRAP = 50")

exec(compile(stats_code, "fig1_notebook_stats.py", "exec"), init_globals)

print("\n--- Stats block completed ---")

# Verify bootstrap stats outputs
bootstrap_stats_path = hosp_dir / "bootstrap_stats.csv"
if bootstrap_stats_path.exists():
    bootstrap_stats = pd.read_csv(bootstrap_stats_path)
    print(f"\nBootstrap stats CSV: {len(bootstrap_stats)} rows")
    required_stats_cols = ["hospital", "outcome", "method", "auroc", "auroc_ci_lo", "auroc_ci_hi",
                           "auprc", "auprc_ci_lo", "auprc_ci_hi", "n_encounters", "n_pos"]
    for col in required_stats_cols:
        assert col in bootstrap_stats.columns, f"Missing column '{col}' in bootstrap_stats.csv"

    # Verify CIs are sensible (lo <= point <= hi)
    for _, row in bootstrap_stats.iterrows():
        if not np.isnan(row["auroc"]):
            assert row["auroc_ci_lo"] <= row["auroc"] <= row["auroc_ci_hi"], \
                f"Invalid AUROC CI for {row['method']}/{row['outcome']}"
        if not np.isnan(row["auprc"]):
            assert row["auprc_ci_lo"] <= row["auprc"] <= row["auprc_ci_hi"], \
                f"Invalid AUPRC CI for {row['method']}/{row['outcome']}"
    print("  Bootstrap stats CIs validated")
else:
    print("  Warning: bootstrap_stats.csv not created (may be expected if <2 methods)")

bootstrap_diffs_path = hosp_dir / "bootstrap_diffs.csv"
if bootstrap_diffs_path.exists():
    bootstrap_diffs = pd.read_csv(bootstrap_diffs_path)
    print(f"Bootstrap diffs CSV: {len(bootstrap_diffs)} rows")
    required_diffs_cols = ["hospital", "outcome", "method", "target",
                           "auroc_diff", "auroc_diff_ci_lo", "auroc_diff_ci_hi", "auroc_sig",
                           "auprc_diff", "auprc_diff_ci_lo", "auprc_diff_ci_hi", "auprc_sig"]
    for col in required_diffs_cols:
        assert col in bootstrap_diffs.columns, f"Missing column '{col}' in bootstrap_diffs.csv"
    print("  Bootstrap diffs columns validated")
else:
    print("  Warning: bootstrap_diffs.csv not created (may be expected if target method missing)")

# Check combined summary
bootstrap_summary_path = Path(tmp_dir) / "bootstrap_summary.csv"
if bootstrap_summary_path.exists():
    summary_stats = pd.read_csv(bootstrap_summary_path)
    print(f"Combined bootstrap summary: {len(summary_stats)} rows")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
