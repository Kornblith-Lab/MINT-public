# SHAP interpretability analysis for fig1 outcomes using MINT foundation model.
#
# Computes per-token SHAP contributions to each outcome's predicted probability,
# then aggregates across patients to produce global token importance rankings.
#
# Usage:
#   conda run -n delphi python -m mint.five.fig1_shap
#   conda run -n delphi python -m mint.five.fig1_shap --outcomes hypoxia hypotension periarrest
#   conda run -n delphi python -m mint.five.fig1_shap --n_patients 100
#   conda run -n delphi python -m mint.five.fig1_shap --plot_only  # skip compute, just re-plot

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
from tqdm import tqdm

from mint.five.fig_one.fig1 import Definition, VOCAB
from model import Delphi, DelphiConfig

SAVE_DIR = Path("artifacts/fig1-shap")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("output")
CHECKPOINT = "output/mint/ckpt_100000.pt"
LOOKAHEAD = 5
MAX_LEN = 512

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def load_model(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    config = DelphiConfig(**checkpoint["model_args"])
    model = Delphi(config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.to(DEVICE)
    return model, config

ALL_OUTCOMES = Definition.TASKS + Definition.INTERVENTION_TASKS


def build_predict_fn(model, token_ids, times, pos_ids, horizon_minutes):
    """SHAP-compatible prediction: returns predicted CDF probability for the outcome."""
    token_ids_arr = np.array(token_ids)
    times_arr = np.array(times)
    pos_idx = torch.tensor(pos_ids, dtype=torch.long, device=DEVICE)

    def predict(masked_inputs):
        if isinstance(masked_inputs, np.ndarray):
            rows = [list(row) for row in masked_inputs] if masked_inputs.ndim > 1 else [str(masked_inputs).split()]
        else:
            rows = [str(row).split() for row in masked_inputs]

        results = []
        for tokens_str in rows:
            keep_mask = np.array([str(t) != "0" for t in tokens_str[:len(token_ids_arr)]])
            kept_ids = token_ids_arr[keep_mask]
            kept_times = times_arr[keep_mask]

            if len(kept_ids) < 2:
                results.append(0.0)
                continue

            events_t = torch.tensor(kept_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
            times_t = torch.tensor(kept_times, dtype=torch.float, device=DEVICE).unsqueeze(0)

            with torch.no_grad():
                logits, _, _ = model(events_t, times_t)

            last_logits = logits[0, -1, :]
            rates = torch.exp(last_logits).clamp(min=1e-10)
            lambda_pos = rates[pos_idx].sum()
            lambda_total = rates.sum() - rates[:2].sum()
            p = (lambda_pos / lambda_total) * (-torch.expm1(-lambda_total * horizon_minutes))
            results.append(p.clamp(0, 1).item())

        return np.array(results).reshape(-1, 1)

    return predict


def compute_shap_for_case(model, case, pos_ids, horizon_minutes, max_len):
    """Compute SHAP values for a single patient case."""
    events = case.events
    times = case.times

    if max_len and len(events) > max_len:
        events = events[-max_len:]
        times = times[-max_len:]

    token_ids = events.tolist()
    times_list = times.tolist()

    predict_fn = build_predict_fn(model, token_ids, times_list, pos_ids, horizon_minutes)

    input_text = " ".join(str(tid) for tid in token_ids)

    def tokenizer(s, return_offsets_mapping=True):
        tokens = s.split()
        offsets = []
        pos = 0
        for t in tokens:
            start = s.index(t, pos)
            offsets.append((start, start + len(t)))
            pos = start + len(t)
        out = {"input_ids": tokens}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out

    masker = shap.maskers.Text(tokenizer, mask_token="0", output_type="str", collapse_mask_token=False)
    explainer = shap.Explainer(predict_fn, masker, output_names=["prob"])
    shap_values = explainer([input_text])

    vals = shap_values.values[0, :, 0]
    base_value = float(shap_values.base_values[0, 0])

    return vals, base_value, token_ids


def run_outcome(model, outcome_name, n_patients, seed=42):
    """Compute SHAP for one outcome across n_patients."""
    print(f"\n{'='*60}")
    print(f"  Outcome: {outcome_name}")
    print(f"{'='*60}")

    task = Definition.get(outcome_name, LOOKAHEAD)

    train = pd.read_feather(DATA_DIR / "train.feather")
    val = pd.read_feather(DATA_DIR / "val.feather")
    test = pd.read_feather(DATA_DIR / "test.feather")

    train_cases, val_cases, test_cases = task.build_cases(
        train, val, test,
        use_only_first_event=False,
        use_ed_data_only=True,
        use_negative_exclusion=False,
        limit_pos_patients=False,
        do_exclusion=False,
    )

    pos_cases = test_cases["positive"]
    neg_cases = test_cases["negative"]

    print(f"  Test: {len(pos_cases)} positive, {len(neg_cases)} negative cases")

    rng = np.random.default_rng(seed)

    n_pos = min(n_patients // 2, len(pos_cases))
    n_neg = min(n_patients - n_pos, len(neg_cases))

    selected_pos = [pos_cases[i] for i in rng.choice(len(pos_cases), size=n_pos, replace=False)]
    selected_neg = [neg_cases[i] for i in rng.choice(len(neg_cases), size=n_neg, replace=False)]
    selected = selected_pos + selected_neg

    print(f"  Selected: {n_pos} positive + {n_neg} negative = {len(selected)} patients")

    age_to_pos_id, _ = task.label.get_age_to_token_ids()

    id_to_name = {0: "PAD", 1: "NO_EVENT"}
    for _, row in VOCAB.iterrows():
        id_to_name[int(row["index"]) + 1] = row["name"]

    all_shap = []
    all_token_names = []
    all_labels = []

    for case in tqdm(selected, desc=f"SHAP [{outcome_name}]"):
        pos_ids = age_to_pos_id[case.age]
        try:
            vals, base_value, token_ids = compute_shap_for_case(
                model, case, pos_ids, LOOKAHEAD, MAX_LEN
            )
        except Exception as e:
            print(f"  Skipping case: {e}")
            continue

        token_names = [id_to_name.get(tid, f"UNK_{tid}") for tid in token_ids]
        all_shap.append((vals, token_names))
        all_labels.append(case.label)

    print(f"  Completed: {len(all_shap)} patients")
    return all_shap, all_labels, id_to_name


def aggregate_shap(all_shap, top_n=30):
    """Aggregate SHAP across patients → mean |SHAP| per unique token."""
    token_shap_sum = {}
    token_shap_abs_sum = {}
    token_count = {}

    for vals, token_names in all_shap:
        seen = {}
        for v, name in zip(vals, token_names):
            if name in ("PAD", "NO_EVENT"):
                continue
            if name not in seen:
                seen[name] = 0.0
            seen[name] += v

        for name, total_v in seen.items():
            token_shap_sum[name] = token_shap_sum.get(name, 0.0) + total_v
            token_shap_abs_sum[name] = token_shap_abs_sum.get(name, 0.0) + abs(total_v)
            token_count[name] = token_count.get(name, 0) + 1

    n_patients = len(all_shap)
    rows = []
    for name in token_shap_abs_sum:
        rows.append({
            "token": name,
            "mean_abs_shap": token_shap_abs_sum[name] / n_patients,
            "mean_shap": token_shap_sum[name] / n_patients,
            "n_patients_with_token": token_count[name],
            "fraction_patients": token_count[name] / n_patients,
        })

    df = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return df.head(top_n) if top_n else df


def categorize_token(name):
    if name.startswith("Vital_SpO2_"):
        return "SpO2"
    if name.startswith("Vital_Pulse_"):
        return "Pulse"
    if name.startswith("Vital_Resp_"):
        return "Resp Rate"
    if name.startswith("Vital_Systolic_") or name.startswith("Vital_Diastolic_") or name.startswith("Vital_MAP"):
        return "Blood Pressure"
    if name.startswith("Vital_Temp_"):
        return "Temperature"
    if name.startswith("Vital_O2 Device_"):
        return "O2 Device"
    if name.startswith("Vital_Glasgow"):
        return "GCS"
    if name.startswith("Med_"):
        return "Medication"
    if name.startswith("Procedure_"):
        return "Procedure"
    if name.startswith("Lab_"):
        return "Lab"
    if name.startswith("Age_"):
        return "Age"
    if name.startswith("Sex_"):
        return "Sex"
    if name.startswith("Arrival_"):
        return "Arrival"
    if name.startswith("Acuity_"):
        return "Acuity"
    return "Other"


def short_name(token):
    """Strip prefix for display."""
    prefixes = [
        "Vital_SpO2_", "Vital_Pulse_", "Vital_Resp_",
        "Vital_Systolic_", "Vital_Diastolic_", "Vital_MAP (mmHg)_",
        "Vital_Temp_", "Vital_O2 Device_", "Vital_Glasgow Coma Scale Score_",
        "Med_", "Procedure_", "Lab_", "Age_", "Sex_",
        "Arrival_Method_", "Acuity_",
    ]
    for p in prefixes:
        if token.startswith(p):
            return token[len(p):]
    return token


TASK_LABELS = {
    "hypoxia": "Hypoxia",
    "tachypnea": "Tachypnea",
    "ppv": "PPV",
    "resp_rescue": "Airway Med",
    "ventilator": "Ventilator",
    "periarrest": "Periarrest",
    "tachycardia": "Tachycardia",
    "hypotension": "Hypotension",
    "vasopressor": "Vasopressor",
    "cardio_rescue": "CV Med",
    "transfusion": "Transfusion",
}


def plot_summary(all_results, save_dir):
    """Create summary heatmap and per-outcome bar charts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use(str(Path(__file__).parent / "design-skill" / "nature.mplstyle"))

    # --- Per-outcome bar charts (top 15 tokens) ---
    outcomes = list(all_results.keys())
    n_outcomes = len(outcomes)
    ncols = min(3, n_outcomes)
    nrows = (n_outcomes + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 2.2 * nrows))
    if n_outcomes == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, outcome in enumerate(outcomes):
        ax = axes[i]
        df = all_results[outcome].head(15).iloc[::-1]

        colors = []
        for _, row in df.iterrows():
            colors.append("#D55E00" if row["mean_shap"] > 0 else "#0072B2")

        ax.barh(
            range(len(df)),
            df["mean_abs_shap"].values,
            color=colors,
            height=0.7,
        )
        labels = [f"{short_name(t)} [{categorize_token(t)}]" for t in df["token"]]
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(labels, fontsize=5.5)
        ax.set_xlabel("Mean |SHAP|", fontsize=7)
        ax.set_title(TASK_LABELS.get(outcome, outcome), fontsize=8, fontweight="bold")
        ax.tick_params(axis="x", labelsize=6)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("MINT Foundation Model: Top Predictive Tokens per Outcome", fontsize=9, fontweight="bold", y=1.01)

    # Legend for direction
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#D55E00", label="Increases risk"),
        Patch(facecolor="#0072B2", label="Decreases risk"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    plt.savefig(save_dir / "fig1_shap_bars.pdf", dpi=600, bbox_inches="tight")
    plt.savefig(save_dir / "fig1_shap_bars.png", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved bar charts: {save_dir / 'fig1_shap_bars.png'}")

    # --- Cross-outcome heatmap of top tokens ---
    top_n_heatmap = 10
    all_tokens = set()
    for outcome, df in all_results.items():
        all_tokens.update(df.head(top_n_heatmap)["token"].tolist())

    token_list = sorted(all_tokens)

    heatmap_data = np.zeros((len(token_list), len(outcomes)))
    for j, outcome in enumerate(outcomes):
        df = all_results[outcome]
        token_to_shap = dict(zip(df["token"], df["mean_shap"]))
        for i, token in enumerate(token_list):
            heatmap_data[i, j] = token_to_shap.get(token, 0.0)

    row_importance = np.abs(heatmap_data).max(axis=1)
    sort_idx = np.argsort(row_importance)[::-1]
    heatmap_data = heatmap_data[sort_idx]
    token_list = [token_list[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(7.2, max(4, len(token_list) * 0.22)))

    vmax = np.abs(heatmap_data).max()
    im = ax.imshow(heatmap_data, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(outcomes)))
    ax.set_xticklabels([TASK_LABELS.get(o, o) for o in outcomes], fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(token_list)))
    ax.set_yticklabels([f"{short_name(t)} [{categorize_token(t)}]" for t in token_list], fontsize=5.5)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean SHAP (directional)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    ax.set_title("Cross-Outcome Token Importance (MINT Model)", fontsize=9, fontweight="bold")

    plt.savefig(save_dir / "fig1_shap_heatmap.pdf", dpi=600, bbox_inches="tight")
    plt.savefig(save_dir / "fig1_shap_heatmap.png", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap: {save_dir / 'fig1_shap_heatmap.png'}")


def main():
    parser = argparse.ArgumentParser(description="SHAP analysis for fig1 outcomes (MINT model)")
    parser.add_argument("--outcomes", nargs="+", default=ALL_OUTCOMES,
                        help="Outcomes to analyze (default: all 11)")
    parser.add_argument("--n_patients", type=int, default=50,
                        help="Patients per outcome (default: 50)")
    parser.add_argument("--top_n", type=int, default=30,
                        help="Top N tokens to report per outcome")
    parser.add_argument("--plot_only", action="store_true",
                        help="Skip compute, re-plot from saved CSVs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    save_dir = SAVE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        all_results = {}
        for outcome in args.outcomes:
            csv_path = save_dir / f"{outcome}_top_tokens.csv"
            if csv_path.exists():
                all_results[outcome] = pd.read_csv(csv_path)
        if all_results:
            plot_summary(all_results, save_dir)
        else:
            print("No CSVs found. Run without --plot_only first.")
        return

    print(f"Loading model from {CHECKPOINT}...")
    model, _ = load_model(CHECKPOINT)

    all_results = {}

    for outcome in args.outcomes:
        all_shap, all_labels, id_to_name = run_outcome(
            model, outcome, args.n_patients, seed=args.seed
        )

        df = aggregate_shap(all_shap, top_n=args.top_n)
        df["category"] = df["token"].apply(categorize_token)
        df["direction"] = df["mean_shap"].apply(lambda x: "risk" if x > 0 else "protective")
        df["short_name"] = df["token"].apply(short_name)

        csv_path = save_dir / f"{outcome}_top_tokens.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

        all_results[outcome] = df

        print(f"\n  Top 10 tokens for {outcome}:")
        print(f"  {'Rank':<5} {'Token':<45} {'|SHAP|':<10} {'Dir':<6}")
        print(f"  {'-'*5} {'-'*45} {'-'*10} {'-'*6}")
        for rank_i, (_, row) in enumerate(df.head(10).iterrows(), 1):
            arrow = "+" if row["mean_shap"] > 0 else "-"
            print(f"  {rank_i:<5} {row['token']:<45} {row['mean_abs_shap']:.5f}  {arrow}")

    # Save combined summary
    summary_rows = []
    for outcome, df in all_results.items():
        for _, row in df.head(10).iterrows():
            summary_rows.append({"outcome": outcome, **row.to_dict()})
    pd.DataFrame(summary_rows).to_csv(save_dir / "summary_all_outcomes.csv", index=False)

    plot_summary(all_results, save_dir)

    print(f"\n{'='*60}")
    print(f"  All results saved to {save_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
