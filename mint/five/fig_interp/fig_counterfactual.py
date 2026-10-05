"""Figure: Counterfactual Analysis (Panels B-F).

Panel B: 4x4 O2 device replacement heatmap — delta risk of admission
Panel C: Respiratory therapy escalation bar plot (fold change vs control)
Panel D: Vital sign distribution shifts (pulse, RR, GCS)
Panel E: Predicted vs actual vital sign changes after interventions
Panel F: Extended counterfactuals — dose-response vital sign shifts

Usage:
    python -m mint.five.fig_interp.fig_counterfactual
    python -m mint.five.fig_interp.fig_counterfactual --panel b
    python -m mint.five.fig_interp.fig_counterfactual --panel c
    python -m mint.five.fig_interp.fig_counterfactual --panel d
    python -m mint.five.fig_interp.fig_counterfactual --panel e
    python -m mint.five.fig_interp.fig_counterfactual --panel f
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mint.five.fig_interp.core import (
    ARTIFACTS_DIR,
    DEVICE,
    HORIZON_MINUTES,
    TRUNCATION_TIME,
    DISPOSITION_TOKENS,
    build_outcome_definitions,
    compute_cdf_probabilities_batch,
    get_model_token_id,
    load_model,
    load_test_cohort,
    load_vocab,
)
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "respiratory",
    Path(__file__).resolve().parents[2] / "model" / "respiratory.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
RespiratoryEscalation = _mod.RespiratoryEscalation

STYLE_PATH = Path(__file__).resolve().parent.parent / "design-skill" / "nature.mplstyle"
OUTPUT_DIR = ARTIFACTS_DIR / "fig_counterfactual"
N_BOOTSTRAP = 1000

# ---------------------------------------------------------------------------
# O2 Device category definitions (4 groups)
# ---------------------------------------------------------------------------

O2_CATEGORIES = {
    "Room Air": {0},
    "Non-PPV": {1, 2, 3, 4},
    "PPV": {5, 6},
    "Ventilator": {7},
}

O2_CATEGORY_ORDER = ["Room Air", "Non-PPV", "PPV", "Ventilator"]

RESPIRATORY_CC_TOKENS = [
    "CC_APNEA", "CC_ASTHMA", "CC_BREATHING PROBLEM", "CC_COUGH", "CC_CROUP",
    "CC_LABORED BREATHING", "CC_PNEUMONIA", "CC_RESPIRATORY DISTRESS",
    "CC_SHORTNESS OF BREATH", "CC_STRIDOR", "CC_WHEEZING",
]

TRAUMA_CC_TOKENS = [
    "CC_TRAUMA", "CC_HEAD INJURY", "CC_GUN SHOT WOUND", "CC_ASSAULT VICTIM",
    "CC_FALL", "CC_CHEST INJURY", "CC_ABDOMINAL INJURY", "CC_BURN",
    "CC_FACIAL INJURY", "CC_FACIAL LACERATION", "CC_PUNCTURE WOUND",
    "CC_ARM INJURY", "CC_LEG INJURY", "CC_BACK INJURY", "CC_NECK INJURY",
    "CC_RIB INJURY", "CC_SHOULDER INJURY", "CC_HIP INJURY",
]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _apply_style():
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))


def bootstrap_mean_ci(values, n_boot=N_BOOTSTRAP, ci=0.95):
    rng = np.random.default_rng(42)
    n = len(values)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = values[idx].mean()
    alpha = (1 - ci) / 2
    return float(values.mean()), float(np.percentile(boot_means, alpha * 100)), float(np.percentile(boot_means, (1 - alpha) * 100))


def get_o2_severity_map(name_to_id):
    """Return {token_name: severity} for O2 tokens that exist in vocab."""
    sev = RespiratoryEscalation.SEVERITY
    return {k: v for k, v in sev.items() if k in name_to_id}


def get_tokens_in_category(cat_name, o2_sev_map, name_to_id):
    """Get list of vocab-space IDs for O2 tokens in a category."""
    sevs = O2_CATEGORIES[cat_name]
    return [name_to_id[k] for k, v in o2_sev_map.items() if v in sevs and k in name_to_id]


def build_sequences_with_o2(df, name_to_id, max_time=TRUNCATION_TIME, min_tokens=5):
    """Build sequences for encounters that have O2 device tokens before disposition.

    Returns list of dicts with 'events', 'times', 'encounter_key', 'age',
    and 'token_names' (original names for O2 identification).
    """
    sequences = []
    grouped = df.groupby("encounter_key", sort=False)

    for enc_key, group in tqdm(grouped, desc="Building O2 sequences"):
        group = group.sort_values("t")

        disp_mask = group["name"].isin(DISPOSITION_TOKENS)
        if disp_mask.any():
            disp_time = group.loc[disp_mask, "t"].iloc[0]
            cutoff = min(max_time, disp_time)
        else:
            cutoff = max_time

        truncated = group[group["t"] < cutoff]
        if len(truncated) < min_tokens:
            continue

        names_list = truncated["name"].tolist()
        times_list = truncated["t"].tolist()

        # Must have at least one O2 device token in the severity map
        has_o2 = any(
            n.startswith("Vital_O2 Device_") and n in RespiratoryEscalation.SEVERITY
            for n in names_list
        )
        if not has_o2:
            continue

        age = None
        for n in names_list:
            if n.startswith("Age_"):
                try:
                    age = int(n.split("_")[1])
                    break
                except ValueError:
                    continue
        if age is None:
            continue

        events = [name_to_id[n] for n in names_list if n in name_to_id]
        times_clean = [times_list[i] for i, n in enumerate(names_list) if n in name_to_id]
        names_clean = [n for n in names_list if n in name_to_id]

        if len(events) < min_tokens:
            continue

        sequences.append({
            "encounter_key": enc_key,
            "events": events,
            "times": times_clean,
            "age": age,
            "token_names": names_clean,
        })

    return sequences


def build_respiratory_sequences(df, name_to_id, max_time=60.0, min_tokens=5):
    """Build sequences for respiratory complaint patients (first 60 min or disposition)."""
    resp_encs = df[df["name"].isin(RESPIRATORY_CC_TOKENS)]["encounter_key"].unique()
    resp_df = df[df["encounter_key"].isin(resp_encs)]

    sequences = []
    grouped = resp_df.groupby("encounter_key", sort=False)

    for enc_key, group in tqdm(grouped, desc="Building respiratory sequences"):
        group = group.sort_values("t")

        disp_mask = group["name"].isin(DISPOSITION_TOKENS)
        if disp_mask.any():
            disp_time = group.loc[disp_mask, "t"].iloc[0]
            cutoff = min(max_time, disp_time)
        else:
            cutoff = max_time

        truncated = group[group["t"] < cutoff]
        if len(truncated) < min_tokens:
            continue

        names_list = truncated["name"].tolist()
        times_list = truncated["t"].tolist()

        age = None
        for n in names_list:
            if n.startswith("Age_"):
                try:
                    age = int(n.split("_")[1])
                    break
                except ValueError:
                    continue
        if age is None:
            continue

        events = [name_to_id[n] for n in names_list if n in name_to_id]
        times_clean = [times_list[i] for i, n in enumerate(names_list) if n in name_to_id]

        if len(events) < min_tokens:
            continue

        sequences.append({
            "encounter_key": enc_key,
            "events": events,
            "times": times_clean,
            "age": age,
        })

    return sequences



def compute_expected_vital_distribution(
    model, model_config, sequences, vital_prefix, name_to_id,
    batch_size=64, device=DEVICE, desc="Vital distribution",
):
    """Compute per-patient expected vital value using CDF-based softmax over vital tokens.

    Method: get logits -> exp() for rates -> select vital tokens -> softmax -> dot product with values.
    Returns array of shape (n_patients,) with expected vital value per patient.
    """
    vital_tokens = []
    vital_values = []
    for name, idx in name_to_id.items():
        if name.startswith(vital_prefix + "_"):
            try:
                val = int(name[len(vital_prefix) + 1:])
                vital_tokens.append(get_model_token_id(idx))
                vital_values.append(val)
            except ValueError:
                continue

    if not vital_tokens:
        return np.zeros(len(sequences), dtype=np.float32)

    vital_token_ids = torch.tensor(vital_tokens, dtype=torch.long, device=device)
    vital_vals = torch.tensor(vital_values, dtype=torch.float, device=device)

    n = len(sequences)
    expected_values = np.zeros(n, dtype=np.float32)

    model.eval()
    for start in tqdm(range(0, n, batch_size), desc=desc, leave=False):
        end = min(start + batch_size, n)
        batch = sequences[start:end]
        bs = len(batch)

        max_len = max(len(s["events"]) for s in batch)
        padded_events = torch.zeros(bs, max_len, dtype=torch.long, device=device)
        padded_times = torch.zeros(bs, max_len, dtype=torch.float, device=device)

        for j, seq in enumerate(batch):
            seq_len = min(len(seq["events"]), max_len)
            shifted = [e + 1 for e in seq["events"][-seq_len:]]
            padded_events[j, :seq_len] = torch.tensor(shifted, dtype=torch.long)
            padded_times[j, :seq_len] = torch.tensor(
                seq["times"][-seq_len:], dtype=torch.float)

        with torch.no_grad():
            logits, _, _ = model(padded_events, padded_times)

        lengths = (padded_events > 0).sum(dim=1) - 1
        lengths = lengths.clamp(min=0)

        for j in range(bs):
            last_logits = logits[j, lengths[j], :]
            # CDF rates for vital tokens
            rates = torch.exp(last_logits).clamp(min=1e-10)
            vital_rates = rates.index_select(0, vital_token_ids)
            # Softmax over just the vital token rates
            probs = torch.softmax(vital_rates.log(), dim=0)
            expected_values[start + j] = (probs * vital_vals).sum().item()

    return expected_values


# ---------------------------------------------------------------------------
# Panel B: O2 Device Replacement Heatmap
# ---------------------------------------------------------------------------

def run_panel_b(model, model_config, name_to_id, sequences, outcome_defs, batch_size):
    """4x4 heatmap: replace O2 tokens from original category with counterfactual category."""
    print("\n" + "=" * 60)
    print("Panel B: O2 Device Replacement Heatmap")
    print("=" * 60)

    o2_sev_map = get_o2_severity_map(name_to_id)
    rng = np.random.default_rng(42)

    admission_def = {"admission": outcome_defs["admission"]}
    n_cats = len(O2_CATEGORY_ORDER)
    fc_matrix = np.full((n_cats, n_cats), np.nan, dtype=np.float32)
    ci_lo_matrix = np.full((n_cats, n_cats), np.nan, dtype=np.float32)
    ci_hi_matrix = np.full((n_cats, n_cats), np.nan, dtype=np.float32)

    for i, orig_cat in enumerate(O2_CATEGORY_ORDER):
        orig_sevs = O2_CATEGORIES[orig_cat]
        orig_token_ids = set(get_tokens_in_category(orig_cat, o2_sev_map, name_to_id))

        # Find patients who have tokens in this category
        cat_sequences = []
        cat_indices = []
        for s_idx, seq in enumerate(sequences):
            has_match = any(eid in orig_token_ids for eid in seq["events"])
            if has_match:
                cat_sequences.append(seq)
                cat_indices.append(s_idx)

        if len(cat_sequences) < 10:
            print(f"  {orig_cat}: only {len(cat_sequences)} patients, skipping")
            continue
        print(f"  {orig_cat}: {len(cat_sequences)} patients")

        # Compute baseline admission probability for these patients (original sequences)
        baseline_probs = compute_cdf_probabilities_batch(
            model, model_config, cat_sequences, admission_def,
            horizon=HORIZON_MINUTES, batch_size=batch_size, device=DEVICE,
            desc=f"  Baseline [{orig_cat.replace(chr(10), ' ')}]",
        )["admission"]

        for j, cf_cat in enumerate(O2_CATEGORY_ORDER):
            if i == j:
                continue  # diagonal = no change

            cf_sevs = O2_CATEGORIES[cf_cat]
            cf_token_names = [k for k, v in o2_sev_map.items() if v in cf_sevs]
            cf_token_ids = [name_to_id[k] for k in cf_token_names]

            if not cf_token_ids:
                continue

            # Build counterfactual sequences: replace orig-category tokens with cf-category tokens
            cf_sequences = []
            for seq in cat_sequences:
                new_events = list(seq["events"])
                for pos, eid in enumerate(new_events):
                    if eid in orig_token_ids:
                        # Replace with random token from counterfactual category
                        new_events[pos] = rng.choice(cf_token_ids)
                cf_sequences.append({
                    "encounter_key": seq["encounter_key"],
                    "events": new_events,
                    "times": list(seq["times"]),
                    "age": seq["age"],
                })

            cf_probs = compute_cdf_probabilities_batch(
                model, model_config, cf_sequences, admission_def,
                horizon=HORIZON_MINUTES, batch_size=batch_size, device=DEVICE,
                desc=f"  CF [{orig_cat.replace(chr(10), ' ')} -> {cf_cat.replace(chr(10), ' ')}]",
            )["admission"]

            safe_baseline = np.maximum(baseline_probs, 1e-10)
            fold_changes = cf_probs / safe_baseline
            mean_fc, ci_lo, ci_hi = bootstrap_mean_ci(fold_changes)
            fc_matrix[i, j] = mean_fc
            ci_lo_matrix[i, j] = ci_lo
            ci_hi_matrix[i, j] = ci_hi
            print(f"    {orig_cat.replace(chr(10), ' ')} -> {cf_cat.replace(chr(10), ' ')}: "
                  f"{mean_fc:.2f}x [{ci_lo:.2f}, {ci_hi:.2f}]")

    return fc_matrix, ci_lo_matrix, ci_hi_matrix


def plot_panel_b(fc_matrix, ci_lo_matrix, ci_hi_matrix, output_dir):
    _apply_style()
    fig, ax = plt.subplots(figsize=(4.5, 3.8))

    labels = [c.replace("\n", " ") for c in O2_CATEGORY_ORDER]
    display_matrix = fc_matrix.copy()

    # Center colormap on 1.0 (no change)
    max_dev = np.nanmax(np.abs(display_matrix - 1.0))
    vmin, vmax = 1.0 - max_dev, 1.0 + max_dev
    masked = np.ma.masked_where(np.isnan(display_matrix), display_matrix)

    im = ax.imshow(masked, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Counterfactual Respiratory Support", fontsize=8)
    ax.set_ylabel("Original Respiratory Support", fontsize=8)

    # Annotate cells with fold change and CI
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="gray")
            elif not np.isnan(fc_matrix[i, j]):
                val = fc_matrix[i, j]
                lo = ci_lo_matrix[i, j]
                hi = ci_hi_matrix[i, j]
                color = "white" if abs(val - 1.0) > 0.6 * max_dev else "black"
                ax.text(j, i, f"{val:.2f}x\n[{lo:.2f}, {hi:.2f}]",
                        ha="center", va="center", fontsize=5.5, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Fold change P(admission)", fontsize=7)

    ax.text(-0.15, 1.05, "b", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "panel_b_o2_heatmap.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "panel_b_o2_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Panel B to {output_dir}/panel_b_o2_heatmap.pdf")


# ---------------------------------------------------------------------------
# Panel C: Respiratory Therapy Escalation
# ---------------------------------------------------------------------------

THERAPY_INTERVENTIONS = {
    "Baseline": [],
    "Mild": [("Med_albuterol sulfate", 0)],
    "Moderate": [
        ("Med_albuterol sulfate", 0),
        ("Med_dexamethasone sodium phosphate", 0),
        ("Med_albuterol sulfate", 20),
        ("Med_albuterol sulfate", 40),
    ],
    "Severe": [
        ("Med_albuterol sulfate", 0),
        ("Med_dexamethasone sodium phosphate", 0),
        ("Med_albuterol sulfate", 20),
        ("Med_magnesium sulfate", 40),
        ("Med_epinephrine", 40),
        ("Med_albuterol sulfate", 40),
    ],
}


def append_timed_tokens(sequences, interventions, name_to_id):
    """Append tokens with specified time offsets relative to last event."""
    cf_sequences = []
    for seq in sequences:
        new_events = list(seq["events"])
        new_times = list(seq["times"])
        last_t = new_times[-1] if new_times else 0

        for token_name, dt in interventions:
            if token_name not in name_to_id:
                continue
            new_events.append(name_to_id[token_name])
            new_times.append(last_t + dt)

        cf_sequences.append({
            "encounter_key": seq["encounter_key"],
            "events": new_events,
            "times": new_times,
            "age": seq["age"],
        })
    return cf_sequences


def run_panel_c(model, model_config, name_to_id, sequences, outcome_defs, batch_size):
    """Respiratory therapy escalation: fold change vs control."""
    print("\n" + "=" * 60)
    print("Panel C: Respiratory Therapy Escalation")
    print("=" * 60)

    admission_def = {"admission": outcome_defs["admission"]}
    results = {}

    # Compute control (no intervention) = baseline
    baseline_probs = compute_cdf_probabilities_batch(
        model, model_config, sequences, admission_def,
        horizon=HORIZON_MINUTES, batch_size=batch_size, device=DEVICE,
        desc="  Control baseline",
    )["admission"]

    for therapy_name, interventions in THERAPY_INTERVENTIONS.items():
        if not interventions:
            # Control: fold change = 1.0 for everyone
            fold_changes = np.ones(len(sequences))
        else:
            cf_sequences = append_timed_tokens(sequences, interventions, name_to_id)
            cf_probs = compute_cdf_probabilities_batch(
                model, model_config, cf_sequences, admission_def,
                horizon=HORIZON_MINUTES, batch_size=batch_size, device=DEVICE,
                desc=f"  {therapy_name}",
            )["admission"]
            safe_baseline = np.maximum(baseline_probs, 1e-10)
            fold_changes = cf_probs / safe_baseline

        mean_fc, ci_lo, ci_hi = bootstrap_mean_ci(fold_changes)
        results[therapy_name] = {"mean": mean_fc, "ci_lo": ci_lo, "ci_hi": ci_hi}
        print(f"  {therapy_name}: fold change = {mean_fc:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")

    return results


def plot_panel_c(results, output_dir):
    _apply_style()
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    names = list(THERAPY_INTERVENTIONS.keys())
    means = [results[n]["mean"] for n in names]
    ci_lo = [results[n]["mean"] - results[n]["ci_lo"] for n in names]
    ci_hi = [results[n]["ci_hi"] - results[n]["mean"] for n in names]
    colors = ["#FFACAC", "#ACCCFF", "#166FFF", "#194791"]

    x = np.arange(len(names))
    bars = ax.bar(x, means, color=colors, edgecolor="none", width=0.6)
    ax.errorbar(x, means, yerr=[ci_lo, ci_hi], fmt="none", ecolor="#333333",
                capsize=3, linewidth=0.7, capthick=0.7)

    ax.axhline(1.0, color="#888888", linewidth=0.4, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7)
    ax.set_xlabel("Respiratory Therapy Escalation", fontsize=8)
    ax.set_ylabel("Fold change P(admission)", fontsize=8)
    ax.text(-0.15, 1.05, "c", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
    ax.tick_params(axis="both", which="major", length=2, width=0.4)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "panel_c_resp_therapy.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "panel_c_resp_therapy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Panel C to {output_dir}/panel_c_resp_therapy.pdf")


# ---------------------------------------------------------------------------
# Panel D: Vital Sign Distribution Shifts
# ---------------------------------------------------------------------------

def run_panel_d_pulse(model, model_config, name_to_id, sequences, batch_size):
    """Pulse distribution shift from epinephrine."""
    print("\n" + "=" * 60)
    print("Panel D.1: Pulse Distribution (Epinephrine)")
    print("=" * 60)

    baseline_pulse = compute_expected_vital_distribution(
        model, model_config, sequences, "Vital_Pulse", name_to_id,
        batch_size=batch_size, device=DEVICE, desc="  Baseline pulse",
    )

    cf_sequences = []
    for seq in sequences:
        new_events = list(seq["events"])
        new_times = list(seq["times"])
        last_t = new_times[-1] if new_times else 0
        new_events.append(name_to_id["Med_epinephrine"])
        new_times.append(last_t + 1.0)
        cf_sequences.append({
            "encounter_key": seq["encounter_key"],
            "events": new_events,
            "times": new_times,
            "age": seq["age"],
        })

    cf_pulse = compute_expected_vital_distribution(
        model, model_config, cf_sequences, "Vital_Pulse", name_to_id,
        batch_size=batch_size, device=DEVICE, desc="  + Epinephrine",
    )

    delta = cf_pulse - baseline_pulse
    print(f"  Epinephrine: mean delta = {delta.mean():+.2f} bpm "
          f"[{np.percentile(delta, 2.5):+.2f}, {np.percentile(delta, 97.5):+.2f}]")

    return {"baseline": baseline_pulse, "counterfactual": cf_pulse}


def run_panel_d_triple_epi(model, model_config, name_to_id, sequences, batch_size):
    """Pulse distribution shift from 3 epinephrine doses separated by 5 minutes."""
    print("\n" + "=" * 60)
    print("Panel D.1b: Pulse Distribution (3x Epinephrine)")
    print("=" * 60)

    epi_id = name_to_id["Med_epinephrine"]

    cf_sequences = []
    for seq in sequences:
        new_events = list(seq["events"])
        new_times = list(seq["times"])
        last_t = new_times[-1] if new_times else 0
        for dose in range(3):
            new_events.append(epi_id)
            new_times.append(last_t + 1.0 + dose * 5.0)
        cf_sequences.append({
            "encounter_key": seq["encounter_key"],
            "events": new_events,
            "times": new_times,
            "age": seq["age"],
        })

    cf_pulse = compute_expected_vital_distribution(
        model, model_config, cf_sequences, "Vital_Pulse", name_to_id,
        batch_size=batch_size, device=DEVICE, desc="  + 3x Epinephrine",
    )

    # Use same baseline from single-epi (will be passed externally)
    return {"counterfactual": cf_pulse}


def run_panel_d_gcs(model, model_config, name_to_id, sequences, batch_size):
    """GCS distribution shift from RSI (ketamine + rocuronium + propofol)."""
    print("\n" + "=" * 60)
    print("Panel D.3: GCS Distribution (RSI)")
    print("=" * 60)

    rsi_meds = ["Med_ketamine", "Med_rocuronium", "Med_propofol"]

    baseline_gcs = compute_expected_vital_distribution(
        model, model_config, sequences, "Vital_Glasgow Coma Scale Score", name_to_id,
        batch_size=batch_size, device=DEVICE, desc="  Baseline GCS",
    )

    cf_sequences = []
    for seq in sequences:
        new_events = list(seq["events"])
        new_times = list(seq["times"])
        last_t = new_times[-1] if new_times else 0
        for k, med in enumerate(rsi_meds):
            if med in name_to_id:
                new_events.append(name_to_id[med])
                new_times.append(last_t + 1.0 * (k + 1))
        cf_sequences.append({
            "encounter_key": seq["encounter_key"],
            "events": new_events,
            "times": new_times,
            "age": seq["age"],
        })

    cf_gcs = compute_expected_vital_distribution(
        model, model_config, cf_sequences, "Vital_Glasgow Coma Scale Score", name_to_id,
        batch_size=batch_size, device=DEVICE, desc="  + RSI",
    )

    delta = cf_gcs - baseline_gcs
    print(f"  RSI effect on GCS: mean delta = {delta.mean():+.2f} "
          f"[{np.percentile(delta, 2.5):+.2f}, {np.percentile(delta, 97.5):+.2f}]")

    return {"baseline": baseline_gcs, "counterfactual": cf_gcs}



def run_panel_d_midazolam_rr(model, model_config, name_to_id, sequences, batch_size):
    """Respiratory rate distribution shift from midazolam."""
    print("\n" + "=" * 60)
    print("Panel D: Resp Rate Distribution (Midazolam)")
    print("=" * 60)

    baseline_rr = compute_expected_vital_distribution(
        model, model_config, sequences, "Vital_Resp", name_to_id,
        batch_size=batch_size, device=DEVICE, desc="  Baseline RR",
    )

    cf_sequences = []
    for seq in sequences:
        new_events = list(seq["events"])
        new_times = list(seq["times"])
        last_t = new_times[-1] if new_times else 0
        new_events.append(name_to_id["Med_midazolam"])
        new_times.append(last_t + 1.0)
        cf_sequences.append({
            "encounter_key": seq["encounter_key"],
            "events": new_events,
            "times": new_times,
            "age": seq["age"],
        })

    cf_rr = compute_expected_vital_distribution(
        model, model_config, cf_sequences, "Vital_Resp", name_to_id,
        batch_size=batch_size, device=DEVICE, desc="  + Midazolam",
    )

    delta = cf_rr - baseline_rr
    print(f"  Midazolam effect on RR: mean delta = {delta.mean():+.2f} "
          f"[{np.percentile(delta, 2.5):+.2f}, {np.percentile(delta, 97.5):+.2f}]")

    return {"baseline": baseline_rr, "counterfactual": cf_rr}


def build_hypoxia_sequences(df, name_to_id, spo2_threshold=92, min_tokens=5):
    """Build sequences for children with hypoxia (SpO2 <= threshold) before first O2 device.

    Each sequence is truncated to include the first O2 device token after hypoxia.
    Returns sequences with and without the final O2 device token for counterfactual comparison.
    """
    o2_device_tokens = {n for n in name_to_id if n.startswith("Vital_O2 Device_")
                        and "None (Room air)" not in n}
    hypoxic_spo2_tokens = set()
    for val in range(0, spo2_threshold + 1):
        tok = f"Vital_SpO2_{val}"
        if tok in name_to_id:
            hypoxic_spo2_tokens.add(tok)

    sequences_baseline = []
    sequences_with_o2 = []
    grouped = df.groupby("encounter_key", sort=False)

    for enc_key, group in tqdm(grouped, desc="Building hypoxia sequences"):
        group = group.sort_values("t")

        disp_mask = group["name"].isin(DISPOSITION_TOKENS)
        if disp_mask.any():
            disp_time = group.loc[disp_mask, "t"].iloc[0]
        else:
            disp_time = float("inf")

        names_list = group["name"].tolist()
        times_list = group["t"].tolist()

        hypoxia_idx = None
        o2_idx = None

        for i, name in enumerate(names_list):
            if times_list[i] >= disp_time:
                break
            if hypoxia_idx is None and name in hypoxic_spo2_tokens:
                hypoxia_idx = i
            elif hypoxia_idx is not None and o2_idx is None and name in o2_device_tokens:
                o2_idx = i
                break

        if hypoxia_idx is None or o2_idx is None:
            continue

        # Truncate including the O2 device token
        truncated_names = names_list[:o2_idx + 1]
        truncated_times = times_list[:o2_idx + 1]

        age = None
        for n in truncated_names:
            if n.startswith("Age_"):
                try:
                    age = int(n.split("_")[1])
                    break
                except ValueError:
                    continue
        if age is None:
            continue

        events_with = [name_to_id[n] for n in truncated_names if n in name_to_id]
        times_with = [truncated_times[i] for i, n in enumerate(truncated_names) if n in name_to_id]

        # Baseline: everything up to but NOT including the O2 device
        baseline_names = names_list[:o2_idx]
        baseline_times = times_list[:o2_idx]
        events_base = [name_to_id[n] for n in baseline_names if n in name_to_id]
        times_base = [baseline_times[i] for i, n in enumerate(baseline_names) if n in name_to_id]

        if len(events_base) < min_tokens:
            continue

        sequences_baseline.append({
            "encounter_key": enc_key,
            "events": events_base,
            "times": times_base,
            "age": age,
        })
        sequences_with_o2.append({
            "encounter_key": enc_key,
            "events": events_with,
            "times": times_with,
            "age": age,
        })

    return sequences_baseline, sequences_with_o2


def run_panel_d_resp_support(model, model_config, name_to_id, df, batch_size):
    """SpO2 distribution shift from respiratory support in hypoxic children."""
    print("\n" + "=" * 60)
    print("Panel D: SpO2 Distribution (Respiratory Support — Hypoxic Children)")
    print("=" * 60)

    sequences_baseline, sequences_with_o2 = build_hypoxia_sequences(df, name_to_id)
    print(f"  {len(sequences_baseline)} hypoxic children with subsequent O2 device")

    baseline_spo2 = compute_expected_vital_distribution(
        model, model_config, sequences_baseline, "Vital_SpO2", name_to_id,
        batch_size=batch_size, device=DEVICE, desc="  Baseline SpO2 (no O2)",
    )

    cf_spo2 = compute_expected_vital_distribution(
        model, model_config, sequences_with_o2, "Vital_SpO2", name_to_id,
        batch_size=batch_size, device=DEVICE, desc="  + O2 Device",
    )

    delta = cf_spo2 - baseline_spo2
    print(f"  O2 device effect on SpO2: mean delta = {delta.mean():+.2f}% "
          f"[{np.percentile(delta, 2.5):+.2f}, {np.percentile(delta, 97.5):+.2f}]")

    return {"baseline": baseline_spo2, "counterfactual": cf_spo2,
            "n_patients": len(sequences_baseline)}


def run_panel_d_transfusion(model, model_config, name_to_id, sequences, batch_size):
    """Systolic BP distribution shift from a single blood transfusion."""
    print("\n" + "=" * 60)
    print("Panel D: Systolic BP Distribution (Transfusion)")
    print("=" * 60)

    baseline_sys = compute_expected_vital_distribution(
        model, model_config, sequences, "Vital_Systolic", name_to_id,
        batch_size=batch_size, device=DEVICE, desc="  Baseline systolic",
    )

    blood_id = name_to_id["Procedure_BLOOD TRANSFUSION ORDERABLES"]

    cf_sequences = []
    for seq in sequences:
        new_events = list(seq["events"])
        new_times = list(seq["times"])
        last_t = new_times[-1] if new_times else 0
        new_events.append(blood_id)
        new_times.append(last_t + 1.0)
        cf_sequences.append({
            "encounter_key": seq["encounter_key"],
            "events": new_events,
            "times": new_times,
            "age": seq["age"],
        })

    cf_sys = compute_expected_vital_distribution(
        model, model_config, cf_sequences, "Vital_Systolic", name_to_id,
        batch_size=batch_size, device=DEVICE, desc="  + Transfusion",
    )

    delta = cf_sys - baseline_sys
    print(f"  Transfusion effect on SBP: mean delta = {delta.mean():+.2f} mmHg "
          f"[{np.percentile(delta, 2.5):+.2f}, {np.percentile(delta, 97.5):+.2f}]")

    return {"baseline": baseline_sys, "counterfactual": cf_sys}


def run_panel_d_morphine_rr(model, model_config, name_to_id, sequences, batch_size):
    """Respiratory rate distribution shift from morphine."""
    print("\n" + "=" * 60)
    print("Panel D: Resp Rate Distribution (Morphine)")
    print("=" * 60)

    baseline_rr = compute_expected_vital_distribution(
        model, model_config, sequences, "Vital_Resp", name_to_id,
        batch_size=batch_size, device=DEVICE, desc="  Baseline RR",
    )

    cf_sequences = []
    for seq in sequences:
        new_events = list(seq["events"])
        new_times = list(seq["times"])
        last_t = new_times[-1] if new_times else 0
        new_events.append(name_to_id["Med_morphine"])
        new_times.append(last_t + 1.0)
        cf_sequences.append({
            "encounter_key": seq["encounter_key"],
            "events": new_events,
            "times": new_times,
            "age": seq["age"],
        })

    cf_rr = compute_expected_vital_distribution(
        model, model_config, cf_sequences, "Vital_Resp", name_to_id,
        batch_size=batch_size, device=DEVICE, desc="  + Morphine",
    )

    delta = cf_rr - baseline_rr
    print(f"  Morphine effect on RR: mean delta = {delta.mean():+.2f} "
          f"[{np.percentile(delta, 2.5):+.2f}, {np.percentile(delta, 97.5):+.2f}]")

    return {"baseline": baseline_rr, "counterfactual": cf_rr}


def run_panel_d_morphine_sbp(model, model_config, name_to_id, sequences, batch_size):
    """Systolic BP distribution shift from morphine."""
    print("\n" + "=" * 60)
    print("Panel D: Systolic BP Distribution (Morphine)")
    print("=" * 60)

    baseline_sys = compute_expected_vital_distribution(
        model, model_config, sequences, "Vital_Systolic", name_to_id,
        batch_size=batch_size, device=DEVICE, desc="  Baseline systolic",
    )

    cf_sequences = []
    for seq in sequences:
        new_events = list(seq["events"])
        new_times = list(seq["times"])
        last_t = new_times[-1] if new_times else 0
        new_events.append(name_to_id["Med_morphine"])
        new_times.append(last_t + 1.0)
        cf_sequences.append({
            "encounter_key": seq["encounter_key"],
            "events": new_events,
            "times": new_times,
            "age": seq["age"],
        })

    cf_sys = compute_expected_vital_distribution(
        model, model_config, cf_sequences, "Vital_Systolic", name_to_id,
        batch_size=batch_size, device=DEVICE, desc="  + Morphine",
    )

    delta = cf_sys - baseline_sys
    print(f"  Morphine effect on SBP: mean delta = {delta.mean():+.2f} mmHg "
          f"[{np.percentile(delta, 2.5):+.2f}, {np.percentile(delta, 97.5):+.2f}]")

    return {"baseline": baseline_sys, "counterfactual": cf_sys}


def build_trauma_sequences(df, name_to_id, max_time=TRUNCATION_TIME, min_tokens=5):
    """Build sequences for trauma chief complaint patients."""
    trauma_encs = df[df["name"].isin(TRAUMA_CC_TOKENS)]["encounter_key"].unique()
    trauma_df = df[df["encounter_key"].isin(trauma_encs)]

    sequences = []
    grouped = trauma_df.groupby("encounter_key", sort=False)

    for enc_key, group in tqdm(grouped, desc="Building trauma sequences"):
        group = group.sort_values("t")

        disp_mask = group["name"].isin(DISPOSITION_TOKENS)
        if disp_mask.any():
            disp_time = group.loc[disp_mask, "t"].iloc[0]
            cutoff = min(max_time, disp_time)
        else:
            cutoff = max_time

        truncated = group[group["t"] < cutoff]
        if len(truncated) < min_tokens:
            continue

        names_list = truncated["name"].tolist()
        times_list = truncated["t"].tolist()

        age = None
        for n in names_list:
            if n.startswith("Age_"):
                try:
                    age = int(n.split("_")[1])
                    break
                except ValueError:
                    continue
        if age is None:
            continue

        events = [name_to_id[n] for n in names_list if n in name_to_id]
        times_clean = [times_list[i] for i, n in enumerate(names_list) if n in name_to_id]

        if len(events) < min_tokens:
            continue

        sequences.append({
            "encounter_key": enc_key,
            "events": events,
            "times": times_clean,
            "age": age,
        })

    return sequences


def run_panel_d_massive_transfusion(model, model_config, name_to_id, sequences, batch_size):
    """Systolic BP shift from massive transfusion protocol on trauma subset.

    Protocol: 6 blood transfusions separated by 5 min, TXA with 1st and 6th,
    calcium chloride + calcium gluconate with the first.
    """
    print("\n" + "=" * 60)
    print("Panel D.6: Systolic BP (Massive Transfusion Protocol — Trauma)")
    print("=" * 60)
    print(f"  Trauma cohort: {len(sequences)} patients")

    baseline_sys = compute_expected_vital_distribution(
        model, model_config, sequences, "Vital_Systolic", name_to_id,
        batch_size=batch_size, device=DEVICE, desc="  Baseline systolic",
    )

    blood_id = name_to_id["Procedure_BLOOD TRANSFUSION ORDERABLES"]
    txa_id = name_to_id["Med_tranexamic acid"]
    cacl_id = name_to_id["Med_calcium chloride"]
    cagluc_id = name_to_id["Med_calcium gluconate"]

    cf_sequences = []
    for seq in sequences:
        new_events = list(seq["events"])
        new_times = list(seq["times"])
        last_t = new_times[-1] if new_times else 0

        for i in range(6):
            t_offset = i * 5.0  # 5-min intervals
            # Blood transfusion
            new_events.append(blood_id)
            new_times.append(last_t + t_offset)
            # TXA with 1st and 6th
            if i == 0 or i == 5:
                new_events.append(txa_id)
                new_times.append(last_t + t_offset)
            # Calcium chloride + calcium gluconate with the first
            if i == 0:
                new_events.append(cacl_id)
                new_times.append(last_t + t_offset)
                new_events.append(cagluc_id)
                new_times.append(last_t + t_offset)

        cf_sequences.append({
            "encounter_key": seq["encounter_key"],
            "events": new_events,
            "times": new_times,
            "age": seq["age"],
        })

    cf_sys = compute_expected_vital_distribution(
        model, model_config, cf_sequences, "Vital_Systolic", name_to_id,
        batch_size=batch_size, device=DEVICE, desc="  + Massive Transfusion",
    )

    delta = cf_sys - baseline_sys
    print(f"  Massive transfusion effect on SBP: mean delta = {delta.mean():+.2f} mmHg "
          f"[{np.percentile(delta, 2.5):+.2f}, {np.percentile(delta, 97.5):+.2f}]")

    return {"baseline": baseline_sys, "counterfactual": cf_sys}


def compute_pvalue(baseline, counterfactual):
    """Paired Wilcoxon signed-rank test for baseline vs counterfactual distributions."""
    diff = counterfactual - baseline
    if np.all(diff == 0):
        return 1.0
    stat, p = stats.wilcoxon(diff, alternative="two-sided")
    return p


def format_pvalue(p):
    """Format p-value for display on plot."""
    if p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return f"p = {p:.3f}"
    elif p < 0.05:
        return f"p = {p:.2f}"
    else:
        return f"p = {p:.2f}"


def bootstrap_delta_ci(baseline, counterfactual, n_boot=N_BOOTSTRAP, ci=0.95):
    """Bootstrap 95% CI for the mean delta (counterfactual - baseline)."""
    rng = np.random.default_rng(42)
    delta = counterfactual - baseline
    n = len(delta)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = delta[idx].mean()
    alpha = (1 - ci) / 2
    return float(np.percentile(boot_means, alpha * 100)), float(np.percentile(boot_means, (1 - alpha) * 100))


def plot_panel_d(pulse_results, gcs_results,
                 midazolam_rr_results, transfusion_results,
                 massive_results, resp_support_results, output_dir,
                 triple_epi_results=None):
    """Plot 6 histograms (2x3) with p-values and optional triple-epi overlay."""
    _apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5))
    axes_flat = axes.flatten()

    COLOR_BASELINE = "#FFACAC"
    COLOR_CF = "#ACCCFF"
    COLOR_TRIPLE = "#194791"

    hist_kwargs = dict(bins=30, alpha=0.65, density=True, edgecolor="none")

    n_pulse = len(pulse_results["baseline"])
    n_gcs = len(gcs_results["baseline"])
    n_midazolam = len(midazolam_rr_results["baseline"])
    n_transfusion = len(transfusion_results["baseline"])
    n_massive = len(massive_results["baseline"])
    n_resp = resp_support_results["n_patients"]

    # Indices where legend/delta should be on the left side
    left_side_indices = {1, 5}  # RSI and respiratory support

    panels = [
        ("Pulse after epinephrine", pulse_results["baseline"],
         pulse_results["counterfactual"], "Expected Pulse (bpm)",
         "Baseline", "+ Epinephrine"),
        ("GCS after rapid sequence\nintubation medications",
         gcs_results["baseline"],
         gcs_results["counterfactual"], "Expected GCS",
         "Baseline", "+ RSI medications"),
        ("Resp rate after midazolam",
         midazolam_rr_results["baseline"],
         midazolam_rr_results["counterfactual"], "Expected Resp Rate",
         "Baseline", "+ Midazolam"),
        ("Systolic BP after transfusion",
         transfusion_results["baseline"],
         transfusion_results["counterfactual"], "Expected Systolic (mmHg)",
         "Baseline", "+ Transfusion"),
        (f"Systolic BP after massive transfusion\n(children with trauma; n={n_massive:,})",
         massive_results["baseline"],
         massive_results["counterfactual"], "Expected Systolic (mmHg)",
         "Baseline", "+ Massive transfusion"),
        (f"SpO2 after respiratory support\n(children with hypoxia; n={n_resp:,})",
         resp_support_results["baseline"],
         resp_support_results["counterfactual"], "Expected SpO2 (%)",
         "Baseline", "+ Respiratory support"),
    ]

    def format_ci(delta_val, ci_lo_val, ci_hi_val):
        return f"Δ = {delta_val:+.1f} ({ci_lo_val:+.1f} to {ci_hi_val:+.1f})"

    for idx, (title, bl, cf, xlabel, bl_label, cf_label) in enumerate(panels):
        ax = axes_flat[idx]
        ax.hist(bl, color=COLOR_BASELINE, label=bl_label, **hist_kwargs)
        ax.hist(cf, color=COLOR_CF, label=cf_label, **hist_kwargs)

        # Overlay triple-epi on the first panel (lighter opacity)
        if idx == 0 and triple_epi_results is not None:
            ax.hist(triple_epi_results["counterfactual"], color=COLOR_TRIPLE,
                    label="+ 3× Epinephrine", bins=30, alpha=0.45,
                    density=True, edgecolor="none")

        ax.set_xlabel(xlabel, fontsize=7)
        if idx % 3 == 0:
            ax.set_ylabel("Density", fontsize=7)
        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.tick_params(axis="both", which="major", length=2, width=0.4)

        use_left = idx in left_side_indices
        legend_loc = "upper left" if use_left else "upper right"
        ax.legend(fontsize=6, frameon=False, loc=legend_loc)

        # Compute delta, CI, and p-value
        p = compute_pvalue(bl, cf)
        delta = (cf - bl).mean()
        ci_lo, ci_hi = bootstrap_delta_ci(bl, cf)

        # Build stat text — include triple-epi stats on first panel
        stat_lines = [format_ci(delta, ci_lo, ci_hi), format_pvalue(p)]
        if idx == 0 and triple_epi_results is not None:
            delta3 = (triple_epi_results["counterfactual"] - bl).mean()
            ci3_lo, ci3_hi = bootstrap_delta_ci(bl, triple_epi_results["counterfactual"])
            p3 = compute_pvalue(bl, triple_epi_results["counterfactual"])
            stat_lines.append(f"Δ₃ = {delta3:+.1f} ({ci3_lo:+.1f} to {ci3_hi:+.1f})")
            stat_lines.append(format_pvalue(p3))
        stat_text = "\n".join(stat_lines)

        # Stat box: bottom-left for left_side_indices, bottom-right otherwise
        ha = "left" if use_left else "right"
        x_pos = 0.03 if use_left else 0.97
        ax.text(x_pos, 0.03, stat_text,
                transform=ax.transAxes, fontsize=6.5, ha=ha, va="bottom",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          alpha=0.9, edgecolor="#cccccc", linewidth=0.5))

    axes_flat[0].text(-0.15, 1.05, "d", transform=axes_flat[0].transAxes,
                      fontsize=11, fontweight="bold", va="top")

    fig.subplots_adjust(wspace=0.35, hspace=0.55)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "panel_d_vitals.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "panel_d_vitals.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Panel D to {output_dir}/panel_d_vitals.pdf")


# ---------------------------------------------------------------------------
# Panel E: Predicted vs Actual Vital Sign Changes
# ---------------------------------------------------------------------------

PANEL_E_VITAL_PREFIXES = [
    "Vital_Pulse",
    "Vital_Resp",
    "Vital_SpO2",
    "Vital_Systolic",
    "Vital_Diastolic",
    "Vital_Temp",
    "Vital_Glasgow Coma Scale Score",
]

PANEL_E_MEDS = {
    "Epinephrine": ["Med_epinephrine"],
    "Albuterol": ["Med_albuterol sulfate", "Med_albuterol sulfate concentrate",
                   "Med_albuterol sulfate hfa"],
    "Ketamine": ["Med_ketamine"],
    "Midazolam": ["Med_midazolam", "Med_midazolam (pf)"],
}

PANEL_E_PAIRS = [
    ("Epinephrine", "Vital_Pulse"),
    ("Albuterol", "Vital_SpO2"),
    ("Ketamine", "Vital_Systolic"),
    ("Midazolam", "Vital_Pulse"),
]


def _get_vital_type_e(token_name):
    for prefix in PANEL_E_VITAL_PREFIXES:
        if token_name.startswith(prefix + "_"):
            try:
                int(token_name[len(prefix) + 1:])
                return prefix
            except ValueError:
                continue
    return None


def _is_med_or_procedure(token_name):
    return token_name.startswith("Med_") or token_name.startswith("Procedure_")


def find_triplets(df, name_to_id, med_tokens, vital_prefix, max_dt=30.0):
    """Find B-X-A triplets for a given medication and vital type.

    Returns list of dicts with:
      - events: token IDs up to and including X (excludes A)
      - times: corresponding timestamps
      - val_before: vital value B
      - val_after: vital value A (ground truth)
    """
    triplets = []
    grouped = df.groupby("encounter_key", sort=False)
    med_set = set(med_tokens)

    for enc_key, group in grouped:
        group = group.sort_values("t")

        disp_mask = group["name"].isin(DISPOSITION_TOKENS)
        if disp_mask.any():
            disp_time = group.loc[disp_mask, "t"].iloc[0]
            cutoff = min(TRUNCATION_TIME, disp_time)
        else:
            cutoff = TRUNCATION_TIME

        truncated = group[group["t"] < cutoff]
        if len(truncated) < 3:
            continue

        names = truncated["name"].tolist()
        times = truncated["t"].tolist()
        n_tokens = len(names)

        for i, (name_i, t_i) in enumerate(zip(names, times)):
            if name_i not in med_set:
                continue

            # Look backward for closest vital B (no Med_/Procedure_ between)
            val_b = None
            for j in range(i - 1, -1, -1):
                if _is_med_or_procedure(names[j]):
                    break
                vtype = _get_vital_type_e(names[j])
                if vtype == vital_prefix:
                    dt = t_i - times[j]
                    if dt <= max_dt:
                        val_b = int(names[j][len(vital_prefix) + 1:])
                    break

            if val_b is None:
                continue

            # Look forward for closest vital A (no Med_/Procedure_ between)
            val_a = None
            for j in range(i + 1, n_tokens):
                if _is_med_or_procedure(names[j]):
                    break
                vtype = _get_vital_type_e(names[j])
                if vtype == vital_prefix:
                    dt = times[j] - t_i
                    if dt <= max_dt:
                        val_a = int(names[j][len(vital_prefix) + 1:])
                    break

            if val_a is None:
                continue

            # Build sequence up to and including Med_X (exclude everything after)
            seq_names = names[:i + 1]
            seq_times = times[:i + 1]

            events = [name_to_id[n] for n in seq_names if n in name_to_id]
            times_clean = [seq_times[k] for k, n in enumerate(seq_names)
                           if n in name_to_id]

            if len(events) < 5:
                continue

            triplets.append({
                "events": events,
                "times": times_clean,
                "val_before": val_b,
                "val_after": val_a,
                "encounter_key": enc_key,
            })

    return triplets


def predict_expected_vital_for_triplets(
    model, model_config, triplets, vital_prefix, name_to_id,
    batch_size=64, device=DEVICE,
):
    """Run model on each triplet's sequence and predict expected vital value."""
    vital_tokens = []
    vital_values = []
    for name, idx in name_to_id.items():
        if name.startswith(vital_prefix + "_"):
            try:
                val = int(name[len(vital_prefix) + 1:])
                vital_tokens.append(get_model_token_id(idx))
                vital_values.append(val)
            except ValueError:
                continue

    if not vital_tokens:
        return np.zeros(len(triplets), dtype=np.float32)

    vital_token_ids = torch.tensor(vital_tokens, dtype=torch.long, device=device)
    vital_vals = torch.tensor(vital_values, dtype=torch.float, device=device)

    n = len(triplets)
    predicted = np.zeros(n, dtype=np.float32)

    model.eval()
    for start in tqdm(range(0, n, batch_size), desc="  Predicting vitals", leave=False):
        end = min(start + batch_size, n)
        batch = triplets[start:end]
        bs = len(batch)

        max_len = max(len(t["events"]) for t in batch)
        padded_events = torch.zeros(bs, max_len, dtype=torch.long, device=device)
        padded_times = torch.zeros(bs, max_len, dtype=torch.float, device=device)

        for j, tri in enumerate(batch):
            seq_len = min(len(tri["events"]), max_len)
            shifted = [e + 1 for e in tri["events"][-seq_len:]]
            padded_events[j, :seq_len] = torch.tensor(shifted, dtype=torch.long)
            padded_times[j, :seq_len] = torch.tensor(
                tri["times"][-seq_len:], dtype=torch.float)

        with torch.no_grad():
            logits, _, _ = model(padded_events, padded_times)

        lengths = (padded_events > 0).sum(dim=1) - 1
        lengths = lengths.clamp(min=0)

        for j in range(bs):
            last_logits = logits[j, lengths[j], :]
            rates = torch.exp(last_logits).clamp(min=1e-10)
            vital_rates = rates.index_select(0, vital_token_ids)
            probs = torch.softmax(vital_rates.log(), dim=0)
            predicted[start + j] = (probs * vital_vals).sum().item()

    return predicted


def run_panel_e(model, model_config, name_to_id, df, batch_size, max_dt=30.0):
    """Run Panel E: predicted vs actual vital changes for each med-vital pair."""
    print(f"\n{'=' * 60}")
    print(f"Panel E: Predicted vs Actual Vital Changes (max_dt={max_dt:.0f} min)")
    print("=" * 60)

    results = {}
    for med_name, vital_prefix in PANEL_E_PAIRS:
        med_tokens = PANEL_E_MEDS[med_name]
        print(f"\n  {med_name} → {vital_prefix.replace('Vital_', '')}:")

        triplets = find_triplets(df, name_to_id, med_tokens, vital_prefix, max_dt=max_dt)
        print(f"    Found {len(triplets)} eligible triplets")

        if len(triplets) < 5:
            print("    Skipping — too few cases")
            results[(med_name, vital_prefix)] = None
            continue

        predicted_vals = predict_expected_vital_for_triplets(
            model, model_config, triplets, vital_prefix, name_to_id,
            batch_size=batch_size, device=DEVICE,
        )

        val_before = np.array([t["val_before"] for t in triplets], dtype=np.float32)
        val_after = np.array([t["val_after"] for t in triplets], dtype=np.float32)

        actual_delta = val_after - val_before
        forecast_delta = predicted_vals - val_before

        ss_res = np.sum((forecast_delta - actual_delta) ** 2)
        ss_tot = np.sum((actual_delta - actual_delta.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Baseline: "no change" (predict Δ=0)
        ss_res_baseline = np.sum(actual_delta ** 2)
        r2_baseline = 1 - ss_res_baseline / ss_tot if ss_tot > 0 else 0.0

        print(f"    R² = {r2:.3f} (baseline Δ=0: {r2_baseline:.3f})")
        print(f"    Mean actual Δ = {actual_delta.mean():+.1f}, "
              f"Mean forecast Δ = {forecast_delta.mean():+.1f}")

        results[(med_name, vital_prefix)] = {
            "actual_delta": actual_delta,
            "forecast_delta": forecast_delta,
            "r2": r2,
            "r2_baseline": r2_baseline,
            "n": len(triplets),
            "med_name": med_name,
            "vital_prefix": vital_prefix,
        }

    return results


def plot_panel_e(results, output_dir, max_dt, suffix=""):
    """Plot 2x2 scatter: actual delta (X) vs forecast delta (Y) with baseline."""
    _apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(7, 6))
    axes_flat = axes.flatten()

    COLOR_DOTS = "#166FFF"
    COLOR_DIAG = "#333333"
    COLOR_BASELINE = "#CC4444"

    for idx, (med_name, vital_prefix) in enumerate(PANEL_E_PAIRS):
        ax = axes_flat[idx]
        key = (med_name, vital_prefix)
        res = results.get(key)

        if res is None:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
            ax.set_title(f"{med_name} → {vital_prefix.replace('Vital_', '')}",
                         fontsize=9, fontweight="bold")
            continue

        actual = res["actual_delta"]
        forecast = res["forecast_delta"]
        r2 = res["r2"]
        r2_baseline = res["r2_baseline"]
        n = res["n"]
        vital_label = vital_prefix.replace("Vital_", "")

        ax.scatter(actual, forecast, s=12, alpha=0.5, color=COLOR_DOTS,
                   edgecolors="none", rasterized=True)

        # Axis limits
        all_vals = np.concatenate([actual, forecast])
        lo, hi = np.percentile(all_vals, [1, 99])
        margin = (hi - lo) * 0.1
        lo -= margin
        hi += margin

        # Diagonal line (perfect agreement)
        ax.plot([lo, hi], [lo, hi], color=COLOR_DIAG, linewidth=1,
                linestyle="--", alpha=0.7, zorder=0, label="Perfect")

        # Baseline: horizontal line at y=0 ("no change" prediction)
        ax.axhline(0, color=COLOR_BASELINE, linewidth=1, linestyle=":",
                   alpha=0.7, zorder=0, label="Baseline (Δ=0)")

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel(f"Actual Δ {vital_label}", fontsize=7)
        ax.set_ylabel(f"Predicted Δ {vital_label}", fontsize=7)
        ax.set_title(f"{med_name} → {vital_label} (n={n})",
                     fontsize=9, fontweight="bold")
        ax.tick_params(axis="both", which="major", length=2, width=0.4)

        # R² annotation with baseline comparison
        stat_text = f"Model R² = {r2:.3f}\nBaseline R² = {r2_baseline:.3f}"
        ax.text(0.05, 0.95, stat_text, transform=ax.transAxes,
                fontsize=7, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.9, edgecolor="#cccccc", linewidth=0.5))

        ax.legend(fontsize=6, frameon=False, loc="lower right")

    axes_flat[0].text(-0.18, 1.08, "e", transform=axes_flat[0].transAxes,
                      fontsize=11, fontweight="bold", va="top")

    fig.subplots_adjust(wspace=0.4, hspace=0.5)

    output_dir.mkdir(parents=True, exist_ok=True)
    fname = f"panel_e_predicted_vs_actual{suffix}"
    fig.savefig(output_dir / f"{fname}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{fname}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Panel E to {output_dir}/{fname}.pdf")


# ---------------------------------------------------------------------------
# Panel F: Extended Counterfactuals — Dose-Response Vital Sign Shifts
# ---------------------------------------------------------------------------

VENTILATOR_TOKENS = [
    "Vital_O2 Device_Ventilator",
    "Vital_O2 Device_CPAP;Ventilator",
    "Vital_O2 Device_Ventilator;CPAP",
    "Vital_O2 Device_BiPAP;Ventilator",
    "Vital_O2 Device_Ventilator;BiPAP",
    "Vital_O2 Device_Ventilator;Other (Comment)",
    "Vital_O2 Device_None (Room air);Ventilator",
    "Vital_O2 Device_Ventilator;None (Room air)",
    "Vital_O2 Device_Other (Comment);Ventilator",
]

LOW_RR_TOKENS = [f"Vital_Resp_{v}" for v in range(0, 13, 2)]


def _build_dose_sequences(sequences, med_token_id, n_doses, name_to_id, rng=None,
                          prepend_token_pool=None, prepend_dt=0.0):
    """Build counterfactual sequences with n_doses of a medication (5 min apart).

    If prepend_token_pool is provided, a random token from that pool is appended
    first (at prepend_dt minutes after last event), and doses start 5 min after that.
    """
    cf_sequences = []
    for seq in sequences:
        new_events = list(seq["events"])
        new_times = list(seq["times"])
        last_t = new_times[-1] if new_times else 0

        if prepend_token_pool is not None:
            tok_id = rng.choice(prepend_token_pool)
            new_events.append(tok_id)
            new_times.append(last_t + prepend_dt)
            last_t = last_t + prepend_dt

        for dose in range(n_doses):
            new_events.append(med_token_id)
            new_times.append(last_t + 5.0 * (dose + 1))

        cf_sequences.append({
            "encounter_key": seq["encounter_key"],
            "events": new_events,
            "times": new_times,
            "age": seq["age"],
        })
    return cf_sequences


def run_panel_f_naloxone(model, model_config, name_to_id, sequences, batch_size):
    """Naloxone (1 and 3 dose) effect on respiratory rate.

    Special: prepend a low-RR token to condition the model, then add naloxone.
    Baseline = expected RR after seeing the low-RR token (before naloxone).
    """
    print("\n" + "=" * 60)
    print("Panel F.1: Resp Rate (Naloxone — low RR conditioning)")
    print("=" * 60)

    rng = np.random.default_rng(42)
    naloxone_id = name_to_id["Med_naloxone"]
    low_rr_ids = [name_to_id[t] for t in LOW_RR_TOKENS if t in name_to_id]

    # Build baseline: append a random low-RR token
    baseline_seqs = []
    for seq in sequences:
        new_events = list(seq["events"])
        new_times = list(seq["times"])
        last_t = new_times[-1] if new_times else 0
        new_events.append(rng.choice(low_rr_ids))
        new_times.append(last_t + 1.0)
        baseline_seqs.append({
            "encounter_key": seq["encounter_key"],
            "events": new_events,
            "times": new_times,
            "age": seq["age"],
        })

    baseline_rr = compute_expected_vital_distribution(
        model, model_config, baseline_seqs, "Vital_Resp", name_to_id,
        batch_size=batch_size, device=DEVICE, desc="  Baseline (low RR)",
    )

    # 1 dose: low-RR then naloxone 5 min later
    cf_1dose = _build_dose_sequences(
        sequences, naloxone_id, 1, name_to_id, rng=rng,
        prepend_token_pool=low_rr_ids, prepend_dt=1.0)
    cf_rr_1 = compute_expected_vital_distribution(
        model, model_config, cf_1dose, "Vital_Resp", name_to_id,
        batch_size=batch_size, device=DEVICE, desc="  + 1× Naloxone",
    )

    # 3 doses: low-RR then 3× naloxone (5 min apart)
    cf_3dose = _build_dose_sequences(
        sequences, naloxone_id, 3, name_to_id, rng=rng,
        prepend_token_pool=low_rr_ids, prepend_dt=1.0)
    cf_rr_3 = compute_expected_vital_distribution(
        model, model_config, cf_3dose, "Vital_Resp", name_to_id,
        batch_size=batch_size, device=DEVICE, desc="  + 3× Naloxone",
    )

    delta1 = cf_rr_1 - baseline_rr
    delta3 = cf_rr_3 - baseline_rr
    print(f"  1× Naloxone: Δ RR = {delta1.mean():+.2f} [{np.percentile(delta1, 2.5):+.2f}, {np.percentile(delta1, 97.5):+.2f}]")
    print(f"  3× Naloxone: Δ RR = {delta3.mean():+.2f} [{np.percentile(delta3, 2.5):+.2f}, {np.percentile(delta3, 97.5):+.2f}]")

    return {"baseline": baseline_rr, "cf_1dose": cf_rr_1, "cf_3dose": cf_rr_3}


def run_panel_f_atropine(model, model_config, name_to_id, sequences, batch_size):
    """Atropine (1 and 3 dose) effect on pulse."""
    print("\n" + "=" * 60)
    print("Panel F.2: Pulse (Atropine)")
    print("=" * 60)

    atropine_id = name_to_id["Med_atropine"]

    baseline_pulse = compute_expected_vital_distribution(
        model, model_config, sequences, "Vital_Pulse", name_to_id,
        batch_size=batch_size, device=DEVICE, desc="  Baseline pulse",
    )

    cf_1dose = _build_dose_sequences(sequences, atropine_id, 1, name_to_id)
    cf_pulse_1 = compute_expected_vital_distribution(
        model, model_config, cf_1dose, "Vital_Pulse", name_to_id,
        batch_size=batch_size, device=DEVICE, desc="  + 1× Atropine",
    )

    cf_3dose = _build_dose_sequences(sequences, atropine_id, 3, name_to_id)
    cf_pulse_3 = compute_expected_vital_distribution(
        model, model_config, cf_3dose, "Vital_Pulse", name_to_id,
        batch_size=batch_size, device=DEVICE, desc="  + 3× Atropine",
    )

    delta1 = cf_pulse_1 - baseline_pulse
    delta3 = cf_pulse_3 - baseline_pulse
    print(f"  1× Atropine: Δ pulse = {delta1.mean():+.2f} [{np.percentile(delta1, 2.5):+.2f}, {np.percentile(delta1, 97.5):+.2f}]")
    print(f"  3× Atropine: Δ pulse = {delta3.mean():+.2f} [{np.percentile(delta3, 2.5):+.2f}, {np.percentile(delta3, 97.5):+.2f}]")

    return {"baseline": baseline_pulse, "cf_1dose": cf_pulse_1, "cf_3dose": cf_pulse_3}


def run_panel_f_norepinephrine(model, model_config, name_to_id, sequences, batch_size):
    """Norepinephrine (1 and 3 dose) effect on systolic BP."""
    print("\n" + "=" * 60)
    print("Panel F.3: Systolic BP (Norepinephrine)")
    print("=" * 60)

    norepi_id = name_to_id["Med_norepinephrine bitartrate"]

    baseline_sys = compute_expected_vital_distribution(
        model, model_config, sequences, "Vital_Systolic", name_to_id,
        batch_size=batch_size, device=DEVICE, desc="  Baseline systolic",
    )

    cf_1dose = _build_dose_sequences(sequences, norepi_id, 1, name_to_id)
    cf_sys_1 = compute_expected_vital_distribution(
        model, model_config, cf_1dose, "Vital_Systolic", name_to_id,
        batch_size=batch_size, device=DEVICE, desc="  + 1× Norepinephrine",
    )

    cf_3dose = _build_dose_sequences(sequences, norepi_id, 3, name_to_id)
    cf_sys_3 = compute_expected_vital_distribution(
        model, model_config, cf_3dose, "Vital_Systolic", name_to_id,
        batch_size=batch_size, device=DEVICE, desc="  + 3× Norepinephrine",
    )

    delta1 = cf_sys_1 - baseline_sys
    delta3 = cf_sys_3 - baseline_sys
    print(f"  1× Norepinephrine: Δ SBP = {delta1.mean():+.2f} [{np.percentile(delta1, 2.5):+.2f}, {np.percentile(delta1, 97.5):+.2f}]")
    print(f"  3× Norepinephrine: Δ SBP = {delta3.mean():+.2f} [{np.percentile(delta3, 2.5):+.2f}, {np.percentile(delta3, 97.5):+.2f}]")

    return {"baseline": baseline_sys, "cf_1dose": cf_sys_1, "cf_3dose": cf_sys_3}


def run_panel_f_dopamine(model, model_config, name_to_id, sequences, batch_size):
    """Dopamine (1 and 3 dose) effect on systolic BP."""
    print("\n" + "=" * 60)
    print("Panel F.4: Systolic BP (Dopamine)")
    print("=" * 60)

    dopamine_id = name_to_id["Med_dopamine"]

    baseline_sys = compute_expected_vital_distribution(
        model, model_config, sequences, "Vital_Systolic", name_to_id,
        batch_size=batch_size, device=DEVICE, desc="  Baseline systolic",
    )

    cf_1dose = _build_dose_sequences(sequences, dopamine_id, 1, name_to_id)
    cf_sys_1 = compute_expected_vital_distribution(
        model, model_config, cf_1dose, "Vital_Systolic", name_to_id,
        batch_size=batch_size, device=DEVICE, desc="  + 1× Dopamine",
    )

    cf_3dose = _build_dose_sequences(sequences, dopamine_id, 3, name_to_id)
    cf_sys_3 = compute_expected_vital_distribution(
        model, model_config, cf_3dose, "Vital_Systolic", name_to_id,
        batch_size=batch_size, device=DEVICE, desc="  + 3× Dopamine",
    )

    delta1 = cf_sys_1 - baseline_sys
    delta3 = cf_sys_3 - baseline_sys
    print(f"  1× Dopamine: Δ SBP = {delta1.mean():+.2f} [{np.percentile(delta1, 2.5):+.2f}, {np.percentile(delta1, 97.5):+.2f}]")
    print(f"  3× Dopamine: Δ SBP = {delta3.mean():+.2f} [{np.percentile(delta3, 2.5):+.2f}, {np.percentile(delta3, 97.5):+.2f}]")

    return {"baseline": baseline_sys, "cf_1dose": cf_sys_1, "cf_3dose": cf_sys_3}


def build_ventilator_sequences(df, name_to_id, max_time=TRUNCATION_TIME, min_tokens=5):
    """Build sequences for patients who have a Ventilator O2 device token."""
    vent_token_set = set(t for t in VENTILATOR_TOKENS if t in name_to_id)
    vent_encs = df[df["name"].isin(vent_token_set)]["encounter_key"].unique()
    vent_df = df[df["encounter_key"].isin(vent_encs)]

    sequences = []
    grouped = vent_df.groupby("encounter_key", sort=False)

    for enc_key, group in tqdm(grouped, desc="Building ventilator sequences"):
        group = group.sort_values("t")

        disp_mask = group["name"].isin(DISPOSITION_TOKENS)
        if disp_mask.any():
            disp_time = group.loc[disp_mask, "t"].iloc[0]
            cutoff = min(max_time, disp_time)
        else:
            cutoff = max_time

        truncated = group[group["t"] < cutoff]
        if len(truncated) < min_tokens:
            continue

        names_list = truncated["name"].tolist()
        times_list = truncated["t"].tolist()

        age = None
        for n in names_list:
            if n.startswith("Age_"):
                try:
                    age = int(n.split("_")[1])
                    break
                except ValueError:
                    continue
        if age is None:
            continue

        events = [name_to_id[n] for n in names_list if n in name_to_id]
        times_clean = [times_list[i] for i, n in enumerate(names_list) if n in name_to_id]

        if len(events) < min_tokens:
            continue

        sequences.append({
            "encounter_key": enc_key,
            "events": events,
            "times": times_clean,
            "age": age,
        })

    return sequences


def run_panel_f_extubation(model, model_config, name_to_id, sequences, batch_size):
    """Extubation effect on GCS (ventilator-subset patients)."""
    print("\n" + "=" * 60)
    print("Panel F.5: GCS (Extubation — ventilator patients)")
    print("=" * 60)
    print(f"  Ventilator cohort: {len(sequences)} patients")

    extubation_id = name_to_id["Procedure_EXTUBATION"]

    baseline_gcs = compute_expected_vital_distribution(
        model, model_config, sequences, "Vital_Glasgow Coma Scale Score", name_to_id,
        batch_size=batch_size, device=DEVICE, desc="  Baseline GCS",
    )

    cf_sequences = []
    for seq in sequences:
        new_events = list(seq["events"])
        new_times = list(seq["times"])
        last_t = new_times[-1] if new_times else 0
        new_events.append(extubation_id)
        new_times.append(last_t + 5.0)
        cf_sequences.append({
            "encounter_key": seq["encounter_key"],
            "events": new_events,
            "times": new_times,
            "age": seq["age"],
        })

    cf_gcs = compute_expected_vital_distribution(
        model, model_config, cf_sequences, "Vital_Glasgow Coma Scale Score", name_to_id,
        batch_size=batch_size, device=DEVICE, desc="  + Extubation",
    )

    delta = cf_gcs - baseline_gcs
    print(f"  Extubation effect on GCS: Δ = {delta.mean():+.2f} [{np.percentile(delta, 2.5):+.2f}, {np.percentile(delta, 97.5):+.2f}]")

    return {"baseline": baseline_gcs, "counterfactual": cf_gcs}


def plot_panel_f(naloxone_results, atropine_results, norepinephrine_results,
                 dopamine_results, output_dir):
    """Plot 2x2 grid of extended counterfactual histograms."""
    _apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))
    axes_flat = axes.flatten()

    COLOR_BASELINE = "#FFACAC"
    COLOR_1DOSE = "#ACCCFF"
    COLOR_3DOSE = "#194791"

    hist_kwargs = dict(bins=30, alpha=0.65, density=True, edgecolor="none")

    panels = [
        ("Resp rate after naloxone\n(low-RR conditioned)",
         naloxone_results, "Expected Resp Rate",
         "Naloxone", "Baseline (low RR)"),
        ("Pulse after atropine",
         atropine_results, "Expected Pulse (bpm)",
         "Atropine", "Baseline"),
        ("Systolic BP after norepinephrine",
         norepinephrine_results, "Expected Systolic (mmHg)",
         "Norepinephrine", "Baseline"),
        ("Systolic BP after dopamine",
         dopamine_results, "Expected Systolic (mmHg)",
         "Dopamine", "Baseline"),
    ]

    for idx, (title, res, xlabel, med_name, bl_label) in enumerate(panels):
        ax = axes_flat[idx]
        bl = res["baseline"]
        cf1 = res["cf_1dose"]
        cf3 = res["cf_3dose"]

        ax.hist(bl, color=COLOR_BASELINE, label=bl_label, **hist_kwargs)
        ax.hist(cf1, color=COLOR_1DOSE, label=f"+ 1× {med_name}", **hist_kwargs)
        ax.hist(cf3, color=COLOR_3DOSE, label=f"+ 3× {med_name}",
                bins=30, alpha=0.45, density=True, edgecolor="none")

        p1 = compute_pvalue(bl, cf1)
        delta1 = (cf1 - bl).mean()
        ci1_lo, ci1_hi = bootstrap_delta_ci(bl, cf1)

        p3 = compute_pvalue(bl, cf3)
        delta3 = (cf3 - bl).mean()
        ci3_lo, ci3_hi = bootstrap_delta_ci(bl, cf3)

        stat_lines = [
            f"Δ₁ = {delta1:+.1f} ({ci1_lo:+.1f} to {ci1_hi:+.1f})",
            format_pvalue(p1),
            f"Δ₃ = {delta3:+.1f} ({ci3_lo:+.1f} to {ci3_hi:+.1f})",
            format_pvalue(p3),
        ]

        ax.set_xlabel(xlabel, fontsize=7)
        if idx % 2 == 0:
            ax.set_ylabel("Density", fontsize=7)
        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.tick_params(axis="both", which="major", length=2, width=0.4)
        ax.legend(fontsize=6, frameon=False, loc="upper right")

        stat_text = "\n".join(stat_lines)
        ax.text(0.97, 0.03, stat_text,
                transform=ax.transAxes, fontsize=6.5, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          alpha=0.9, edgecolor="#cccccc", linewidth=0.5))

    axes_flat[0].text(-0.15, 1.05, "f", transform=axes_flat[0].transAxes,
                      fontsize=11, fontweight="bold", va="top")

    fig.subplots_adjust(wspace=0.35, hspace=0.55)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "panel_f_extended.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "panel_f_extended.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Panel F to {output_dir}/panel_f_extended.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Counterfactual Figure (Panels B-F)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--panel", type=str, default=None,
                        choices=["b", "c", "d", "e", "f"], help="Run only one panel")
    parser.add_argument("--n_patients", type=int, default=None,
                        help="Limit patients for debugging")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading vocab...")
    vocab, name_to_id, id_to_name = load_vocab()

    print("Loading model...")
    model, model_config = load_model()

    print("Loading test cohort (Immediate/Emergent)...")
    df = load_test_cohort(acuity_filter=("Acuity_Immediate", "Acuity_Emergent"))
    print(f"  {df['encounter_key'].nunique()} encounters")

    print("Building outcome definitions...")
    outcome_defs = build_outcome_definitions(vocab, name_to_id)

    # ---- Panel B ----
    if args.panel is None or args.panel == "b":
        print("\nBuilding O2 device sequences...")
        o2_sequences = build_sequences_with_o2(df, name_to_id)
        print(f"  {len(o2_sequences)} patients with O2 device tokens")
        if args.n_patients and len(o2_sequences) > args.n_patients:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(o2_sequences), size=args.n_patients, replace=False)
            o2_sequences = [o2_sequences[i] for i in idx]
            print(f"  Subsampled to {len(o2_sequences)} patients")

        fc_matrix, ci_lo_matrix, ci_hi_matrix = run_panel_b(
            model, model_config, name_to_id, o2_sequences, outcome_defs, args.batch_size)
        np.savez(OUTPUT_DIR / "panel_b_matrix.npz",
                 fc=fc_matrix, ci_lo=ci_lo_matrix, ci_hi=ci_hi_matrix)
        plot_panel_b(fc_matrix, ci_lo_matrix, ci_hi_matrix, OUTPUT_DIR)

    # ---- Panel C ----
    if args.panel is None or args.panel == "c":
        print("\nBuilding respiratory complaint sequences...")
        resp_sequences = build_respiratory_sequences(df, name_to_id)
        print(f"  {len(resp_sequences)} respiratory patients")
        if args.n_patients and len(resp_sequences) > args.n_patients:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(resp_sequences), size=args.n_patients, replace=False)
            resp_sequences = [resp_sequences[i] for i in idx]
            print(f"  Subsampled to {len(resp_sequences)} patients")

        panel_c_results = run_panel_c(
            model, model_config, name_to_id, resp_sequences, outcome_defs, args.batch_size)
        np.savez(OUTPUT_DIR / "panel_c_results.npz",
                 **{k: [v["mean"], v["ci_lo"], v["ci_hi"]] for k, v in panel_c_results.items()})
        plot_panel_c(panel_c_results, OUTPUT_DIR)

    # ---- Panel D ----
    if args.panel is None or args.panel == "d":
        # D.1 & D.3: Use full cohort (truncated at 120 min)
        from mint.five.fig_interp.core import build_cohort_sequences
        print("\nBuilding general cohort sequences for Panel D...")
        general_sequences = build_cohort_sequences(df, name_to_id)
        print(f"  {len(general_sequences)} patients")
        if args.n_patients and len(general_sequences) > args.n_patients:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(general_sequences), size=args.n_patients, replace=False)
            general_sequences = [general_sequences[i] for i in idx]
            print(f"  Subsampled to {len(general_sequences)} patients")

        # D.1: Pulse (Epinephrine)
        pulse_results = run_panel_d_pulse(
            model, model_config, name_to_id, general_sequences, args.batch_size)

        # D.1b: Pulse (3× Epinephrine — overlay)
        triple_epi_results = run_panel_d_triple_epi(
            model, model_config, name_to_id, general_sequences, args.batch_size)

        # D.2: GCS (RSI)
        gcs_results = run_panel_d_gcs(
            model, model_config, name_to_id, general_sequences, args.batch_size)

        # D.3: Resp Rate (Midazolam)
        midazolam_rr_results = run_panel_d_midazolam_rr(
            model, model_config, name_to_id, general_sequences, args.batch_size)

        # D.4: Transfusion → Systolic BP (full cohort)
        transfusion_results = run_panel_d_transfusion(
            model, model_config, name_to_id, general_sequences, args.batch_size)

        # D.5: Massive Transfusion Protocol → Systolic BP (Trauma subset)
        print("\nBuilding trauma subset sequences for massive transfusion...")
        trauma_sequences = build_trauma_sequences(df, name_to_id)
        print(f"  {len(trauma_sequences)} trauma patients")
        if args.n_patients and len(trauma_sequences) > args.n_patients:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(trauma_sequences), size=args.n_patients, replace=False)
            trauma_sequences = [trauma_sequences[i] for i in idx]
            print(f"  Subsampled to {len(trauma_sequences)} patients")
        massive_results = run_panel_d_massive_transfusion(
            model, model_config, name_to_id, trauma_sequences, args.batch_size)

        # D.6: Respiratory Support → SpO2 (Hypoxic children)
        resp_support_results = run_panel_d_resp_support(
            model, model_config, name_to_id, df, args.batch_size)

        # Save numeric results
        np.savez(OUTPUT_DIR / "panel_d_pulse.npz",
                 baseline=pulse_results["baseline"],
                 counterfactual=pulse_results["counterfactual"])
        np.savez(OUTPUT_DIR / "panel_d_triple_epi.npz",
                 counterfactual=triple_epi_results["counterfactual"])
        np.savez(OUTPUT_DIR / "panel_d_gcs.npz",
                 baseline=gcs_results["baseline"],
                 counterfactual=gcs_results["counterfactual"])
        np.savez(OUTPUT_DIR / "panel_d_midazolam_rr.npz",
                 baseline=midazolam_rr_results["baseline"],
                 counterfactual=midazolam_rr_results["counterfactual"])
        np.savez(OUTPUT_DIR / "panel_d_transfusion.npz",
                 baseline=transfusion_results["baseline"],
                 counterfactual=transfusion_results["counterfactual"])
        np.savez(OUTPUT_DIR / "panel_d_massive.npz",
                 baseline=massive_results["baseline"],
                 counterfactual=massive_results["counterfactual"])
        np.savez(OUTPUT_DIR / "panel_d_resp_support.npz",
                 baseline=resp_support_results["baseline"],
                 counterfactual=resp_support_results["counterfactual"],
                 n_patients=np.array([resp_support_results["n_patients"]]))

        plot_panel_d(pulse_results, gcs_results,
                     midazolam_rr_results, transfusion_results,
                     massive_results, resp_support_results, OUTPUT_DIR,
                     triple_epi_results=triple_epi_results)

    # ---- Panel E ----
    if args.panel is None or args.panel == "e":
        # 30-minute window version
        results_30 = run_panel_e(
            model, model_config, name_to_id, df, args.batch_size, max_dt=30.0)
        for key, res in results_30.items():
            if res is not None:
                tag = f"{res['med_name']}_{res['vital_prefix'].replace('Vital_', '')}"
                np.savez(OUTPUT_DIR / f"panel_e_{tag}_30min.npz",
                         actual_delta=res["actual_delta"],
                         forecast_delta=res["forecast_delta"],
                         r2=np.array([res["r2"]]),
                         n=np.array([res["n"]]))
        plot_panel_e(results_30, OUTPUT_DIR, max_dt=30.0, suffix="_30min")

        # 15-minute window version
        results_15 = run_panel_e(
            model, model_config, name_to_id, df, args.batch_size, max_dt=15.0)
        for key, res in results_15.items():
            if res is not None:
                tag = f"{res['med_name']}_{res['vital_prefix'].replace('Vital_', '')}"
                np.savez(OUTPUT_DIR / f"panel_e_{tag}_15min.npz",
                         actual_delta=res["actual_delta"],
                         forecast_delta=res["forecast_delta"],
                         r2=np.array([res["r2"]]),
                         n=np.array([res["n"]]))
        plot_panel_e(results_15, OUTPUT_DIR, max_dt=15.0, suffix="_15min")

    # ---- Panel F ----
    if args.panel is None or args.panel == "f":
        from mint.five.fig_interp.core import build_cohort_sequences
        if "general_sequences" not in locals():
            print("\nBuilding general cohort sequences for Panel F...")
            general_sequences = build_cohort_sequences(df, name_to_id)
            print(f"  {len(general_sequences)} patients")
            if args.n_patients and len(general_sequences) > args.n_patients:
                rng = np.random.default_rng(0)
                idx = rng.choice(len(general_sequences), size=args.n_patients, replace=False)
                general_sequences = [general_sequences[i] for i in idx]
                print(f"  Subsampled to {len(general_sequences)} patients")

        # F.1: Naloxone → Resp Rate (low-RR conditioned)
        naloxone_results = run_panel_f_naloxone(
            model, model_config, name_to_id, general_sequences, args.batch_size)

        # F.2: Atropine → Pulse
        atropine_results = run_panel_f_atropine(
            model, model_config, name_to_id, general_sequences, args.batch_size)

        # F.3: Norepinephrine → Systolic BP
        norepinephrine_results = run_panel_f_norepinephrine(
            model, model_config, name_to_id, general_sequences, args.batch_size)

        # F.4: Dopamine → Systolic BP
        dopamine_results = run_panel_f_dopamine(
            model, model_config, name_to_id, general_sequences, args.batch_size)

        # Save numeric results
        np.savez(OUTPUT_DIR / "panel_f_naloxone.npz",
                 baseline=naloxone_results["baseline"],
                 cf_1dose=naloxone_results["cf_1dose"],
                 cf_3dose=naloxone_results["cf_3dose"])
        np.savez(OUTPUT_DIR / "panel_f_atropine.npz",
                 baseline=atropine_results["baseline"],
                 cf_1dose=atropine_results["cf_1dose"],
                 cf_3dose=atropine_results["cf_3dose"])
        np.savez(OUTPUT_DIR / "panel_f_norepinephrine.npz",
                 baseline=norepinephrine_results["baseline"],
                 cf_1dose=norepinephrine_results["cf_1dose"],
                 cf_3dose=norepinephrine_results["cf_3dose"])
        np.savez(OUTPUT_DIR / "panel_f_dopamine.npz",
                 baseline=dopamine_results["baseline"],
                 cf_1dose=dopamine_results["cf_1dose"],
                 cf_3dose=dopamine_results["cf_3dose"])

        plot_panel_f(naloxone_results, atropine_results, norepinephrine_results,
                     dopamine_results, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("DONE. All outputs in:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
