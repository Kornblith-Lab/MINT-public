"""Supplementary confounding-by-indication analyses.

Produces two figures:
  1. supp_predrug_vitals.pdf  (Analysis 1)
     Training-set vital distributions immediately before drug administration:
     CBI drugs vs. matched correctly-learned controls.
     Shows the spurious co-occurrence the model learned from.

  2. supp_dose_response.pdf  (Analysis 4)
     Dose-response line plots (0/1/2/3 doses) for CBI drugs vs. their correctly-
     learned pharmacologic counterparts, on the same axis per vital type.

Usage:
    conda run -n delphi python -m mint.five.fig_interp.run_supp
    conda run -n delphi python -m mint.five.fig_interp.run_supp --output predrug
    conda run -n delphi python -m mint.five.fig_interp.run_supp --output dose_response
    conda run -n delphi python -m mint.five.fig_interp.run_supp --debug
"""

import argparse
import hashlib
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mint.five.fig_interp.core import (
    DATA_DIR,
    DEVICE,
    DISPOSITION_TOKENS,
    build_cohort_sequences,
    load_model,
    load_test_cohort,
    load_vocab,
)
from mint.five.fig_interp.fig_counterfactual import (
    STYLE_PATH,
    OUTPUT_DIR,
    _apply_style,
    append_timed_tokens,
    bootstrap_delta_ci,
    compute_expected_vital_distribution,
)

CHECKPOINT_PATH = "output/mint/ckpt.pt"
CACHE_DIR = OUTPUT_DIR / "cache"
ACUITY_FILTER = ("Acuity_Immediate", "Acuity_Emergent")
N_BOOTSTRAP = 1000

# ---------------------------------------------------------------------------
# CBI and control drug definitions
# ---------------------------------------------------------------------------

# Each entry: (label, drug_token, vital_prefix, expected_dir, is_cbi, paired_group)
# paired_group links a CBI drug and its correctly-learned pharmacologic counterpart.
# Analysis 1 uses all of these; Analysis 4 uses the pairs.

DRUG_VITAL_PAIRS = [
    # HR: adenosine (CBI) vs. propranolol (correct)
    {
        "label": "Adenosine",
        "token": "Med_adenosine",
        "vital": "Vital_Pulse",
        "expected_dir": -1,
        "is_cbi": True,
        "group": "HR",
    },
    {
        "label": "Propranolol",
        "token": "Med_propranolol",
        "vital": "Vital_Pulse",
        "expected_dir": -1,
        "is_cbi": False,
        "group": "HR",
    },
    # SBP: dopamine (CBI) vs. norepinephrine (correct)
    {
        "label": "Dopamine",
        "token": "Med_dopamine",
        "vital": "Vital_Systolic",
        "expected_dir": +1,
        "is_cbi": True,
        "group": "BP",
    },
    {
        "label": "Norepinephrine",
        "token": "Med_norepinephrine bitartrate",
        "vital": "Vital_Systolic",
        "expected_dir": +1,
        "is_cbi": False,
        "group": "BP",
    },
    # RR: naloxone (CBI) vs. morphine (correct; decreases RR — CBI pair on same vital)
    # naloxone should increase RR; morphine should decrease RR
    {
        "label": "Naloxone",
        "token": "Med_naloxone",
        "vital": "Vital_Resp",
        "expected_dir": +1,
        "is_cbi": True,
        "group": "RR",
    },
    {
        "label": "Morphine",
        "token": "Med_morphine",
        "vital": "Vital_Resp",
        "expected_dir": -1,
        "is_cbi": False,
        "group": "RR",
    },
    # RR: propofol (CBI) — no single obvious control, use fentanyl (also decreases RR, correctly)
    {
        "label": "Propofol",
        "token": "Med_propofol",
        "vital": "Vital_Resp",
        "expected_dir": -1,
        "is_cbi": True,
        "group": "RR2",
    },
    {
        "label": "Fentanyl",
        "token": "Med_fentanyl (pf)",
        "vital": "Vital_Resp",
        "expected_dir": -1,
        "is_cbi": False,
        "group": "RR2",
    },
]

VITAL_DISPLAY = {
    "Vital_Pulse": "Heart Rate (bpm)",
    "Vital_Systolic": "Systolic BP (mmHg)",
    "Vital_Resp": "Respiratory Rate",
}

DOSE_LEVELS = [0, 1, 2, 3]

# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------

def _ckpt_sig():
    p = Path(CHECKPOINT_PATH)
    return (str(p), p.stat().st_mtime if p.exists() else None)


def _cache_path(key_parts):
    blob = repr(key_parts).encode()
    digest = hashlib.sha256(blob).hexdigest()[:20]
    return CACHE_DIR / f"{digest}.pkl"


def _cached_expected(model, config, name_to_id, sequences, vital, schedule,
                     cohort_name, n_patients, batch_size, desc):
    key = (_ckpt_sig(), cohort_name, vital, "add", tuple(schedule), n_patients)
    path = _cache_path(key)
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    seqs = append_timed_tokens(sequences, schedule, name_to_id) if schedule else sequences
    values = compute_expected_vital_distribution(
        model, config, seqs, vital, name_to_id,
        batch_size=batch_size, device=DEVICE, desc=desc,
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(values, f)
    return values


# ---------------------------------------------------------------------------
# Analysis 1: Pre-drug vital distributions from training data
# ---------------------------------------------------------------------------

def _is_vital_for(token_name, vital_prefix):
    if not token_name.startswith(vital_prefix + "_"):
        return False
    try:
        int(token_name[len(vital_prefix) + 1:])
        return True
    except ValueError:
        return False


def _is_med_or_procedure(name):
    return name.startswith("Med_") or name.startswith("Procedure_")


def compute_predrug_vitals(train_path, drug_token, vital_prefix):
    """Pull the vital value immediately before each administration of drug_token
    in the training set (before the encounter's first disposition token).

    Returns (predrug_values: np.ndarray, general_values: np.ndarray) where
    general_values is a random sample of all pre-disposition vital measurements
    for that vital type, matched in size to predrug_values.
    """
    print(f"  Loading train.feather for {drug_token} / {vital_prefix}...")
    df = pd.read_feather(train_path, columns=["encounter_key", "name", "t"])

    # First disposition time per encounter
    disp = df[df["name"].isin(DISPOSITION_TOKENS)]
    disp_time = disp.groupby("encounter_key", sort=False)["t"].min()

    predrug_vals = []
    general_vals = []

    for enc_key, group in tqdm(df.groupby("encounter_key", sort=False),
                               desc=f"  {drug_token}", leave=False):
        group = group.sort_values("t")
        cutoff = disp_time.get(enc_key, np.inf)

        names = group["name"].tolist()
        times = group["t"].tolist()
        n = len(names)

        for i in range(n):
            if times[i] >= cutoff:
                break

            # Collect general population vital values (pre-disposition)
            if _is_vital_for(names[i], vital_prefix):
                general_vals.append(int(names[i][len(vital_prefix) + 1:]))

            # Find drug administrations and look back for the immediately preceding vital
            if names[i] != drug_token:
                continue

            # Walk backward; stop at any other med/procedure
            for j in range(i - 1, -1, -1):
                if _is_med_or_procedure(names[j]):
                    break
                if _is_vital_for(names[j], vital_prefix):
                    predrug_vals.append(int(names[j][len(vital_prefix) + 1:]))
                    break

    predrug_vals = np.array(predrug_vals, dtype=np.float32)

    # Sub-sample general to same size for visual comparability
    rng = np.random.default_rng(42)
    if len(general_vals) > len(predrug_vals) and len(predrug_vals) > 0:
        idx = rng.choice(len(general_vals), size=len(predrug_vals), replace=False)
        general_vals = np.array(general_vals, dtype=np.float32)[idx]
    else:
        general_vals = np.array(general_vals, dtype=np.float32)

    return predrug_vals, general_vals


def plot_predrug_vitals(drug_data, output_dir):
    """2×4 grid: one panel per (drug, control) pair showing pre-drug vital histograms.

    Rows = groups (HR, BP, RR-naloxone, RR-propofol).
    Each row has 2 panels: CBI drug | control drug.
    """
    _apply_style()

    groups = ["HR", "BP", "RR", "RR2"]
    group_labels = [
        ("Adenosine", "Propranolol"),
        ("Dopamine", "Norepinephrine"),
        ("Naloxone", "Morphine"),
        ("Propofol", "Fentanyl"),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(7, 10))

    COLOR_DRUG = "#FFACAC"    # CBI or control drug
    COLOR_GEN = "#ACCCFF"     # general population

    for row, (group, (left_name, right_name)) in enumerate(zip(groups, group_labels)):
        for col, drug_name in enumerate([left_name, right_name]):
            ax = axes[row, col]
            entry = drug_data[drug_name]
            predrug = entry["predrug"]
            general = entry["general"]
            vital_label = VITAL_DISPLAY[entry["vital"]]
            is_cbi = entry["is_cbi"]
            expected_dir = entry["expected_dir"]

            if len(predrug) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9)
                ax.set_title(drug_name, fontsize=8, fontweight="bold")
                continue

            bins = 40
            ax.hist(general, bins=bins, density=True, alpha=0.6,
                    color=COLOR_GEN, edgecolor="none", label="General population")
            ax.hist(predrug, bins=bins, density=True, alpha=0.6,
                    color=COLOR_DRUG, edgecolor="none", label=f"Pre-{drug_name}")

            mean_pre = predrug.mean()
            mean_gen = general.mean()
            direction_word = "↑" if expected_dir > 0 else "↓"

            cbi_tag = " [CBI]" if is_cbi else ""
            ax.set_title(f"{drug_name}{cbi_tag}\n(expected {direction_word} {vital_label.split()[0]})",
                         fontsize=7.5, fontweight="bold")
            ax.set_xlabel(f"Pre-drug {vital_label}", fontsize=7)
            if col == 0:
                ax.set_ylabel("Density", fontsize=7)
            ax.tick_params(axis="both", which="major", length=2, width=0.4)

            # Annotation: pre-drug mean vs general mean
            diff = mean_pre - mean_gen
            sign = "+" if diff >= 0 else ""
            stat_text = (
                f"Pre-drug mean: {mean_pre:.1f}\n"
                f"General mean: {mean_gen:.1f}\n"
                f"Diff: {sign}{diff:.1f}"
            )
            ax.text(0.97, 0.97, stat_text, transform=ax.transAxes,
                    fontsize=6, ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              alpha=0.9, edgecolor="#cccccc", linewidth=0.5))

            ax.legend(fontsize=6, frameon=False, loc="upper left")

    # Panel label
    axes[0, 0].text(-0.18, 1.12, "a", transform=axes[0, 0].transAxes,
                    fontsize=11, fontweight="bold", va="top")

    fig.subplots_adjust(wspace=0.4, hspace=0.6, left=0.1, right=0.97,
                        top=0.95, bottom=0.06)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "supp_predrug_vitals.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "supp_predrug_vitals.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved supp_predrug_vitals -> {output_dir}/supp_predrug_vitals.pdf")


# ---------------------------------------------------------------------------
# Analysis 4: Dose-response line plots
# ---------------------------------------------------------------------------

def compute_dose_response(model, config, name_to_id, sequences, drug_entry,
                          n_patients, batch_size, n_boot):
    """Compute mean Δ vital (± bootstrap CI) for 0/1/2/3 doses of a drug.

    Returns list of (dose, mean_delta, ci_lo, ci_hi).
    """
    vital = drug_entry["vital"]
    token = drug_entry["token"]
    label = drug_entry["label"]

    baseline = _cached_expected(
        model, config, name_to_id, sequences, vital, [],
        "general", n_patients, batch_size, desc=f"  {label} baseline")

    results = []
    for n_doses in DOSE_LEVELS:
        if n_doses == 0:
            mean_d = 0.0
            ci_lo = 0.0
            ci_hi = 0.0
        else:
            schedule = [(token, float(d * 5)) for d in range(1, n_doses + 1)]
            cf = _cached_expected(
                model, config, name_to_id, sequences, vital, schedule,
                "general", n_patients, batch_size,
                desc=f"  {label} {n_doses}x")
            delta = cf - baseline
            mean_d = float(delta.mean())
            ci_lo, ci_hi = bootstrap_delta_ci(baseline, cf, n_boot=n_boot)
        results.append((n_doses, mean_d, ci_lo, ci_hi))

    return results


def plot_dose_response(dose_data, output_dir):
    """2×2 grid: one panel per vital group (HR, BP, RR-pair1, RR-pair2).

    Each panel overlays the CBI drug and its correct-pharmacology control,
    x = dose count, y = mean Δ vital with shaded 95% CI.
    """
    _apply_style()

    groups = [
        ("HR", "Adenosine", "Propranolol", "Heart Rate (bpm)"),
        ("BP", "Dopamine", "Norepinephrine", "Systolic BP (mmHg)"),
        ("RR", "Naloxone", "Morphine", "Respiratory Rate"),
        ("RR2", "Propofol", "Fentanyl", "Respiratory Rate"),
    ]

    COLOR_CBI = "#CC4444"
    COLOR_CTRL = "#194791"

    fig, axes = plt.subplots(2, 2, figsize=(7, 6))
    axes_flat = axes.flatten()

    for idx, (group, cbi_name, ctrl_name, ylabel) in enumerate(groups):
        ax = axes_flat[idx]

        cbi_res = dose_data[cbi_name]
        ctrl_res = dose_data[ctrl_name]

        for drug_name, res, color, linestyle in [
            (cbi_name, cbi_res, COLOR_CBI, "-"),
            (ctrl_name, ctrl_res, COLOR_CTRL, "--"),
        ]:
            doses = [r[0] for r in res]
            means = [r[1] for r in res]
            lo = [r[2] for r in res]
            hi = [r[3] for r in res]

            is_cbi = (drug_name == cbi_name)
            lbl = f"{drug_name} [CBI]" if is_cbi else drug_name
            ax.plot(doses, means, color=color, linestyle=linestyle,
                    linewidth=1.4, marker="o", markersize=4, label=lbl)
            ax.fill_between(doses, lo, hi, color=color, alpha=0.15)

        ax.axhline(0, color="#888888", linewidth=0.5, linestyle=":", alpha=0.7)
        ax.set_xticks(DOSE_LEVELS)
        ax.set_xlabel("Number of doses", fontsize=7)
        ax.set_ylabel(f"Mean Δ {ylabel}", fontsize=7)
        ax.set_title(f"{cbi_name} vs {ctrl_name}", fontsize=8, fontweight="bold")
        ax.tick_params(axis="both", which="major", length=2, width=0.4)
        ax.legend(fontsize=6, frameon=False, loc="upper left")

    axes_flat[0].text(-0.18, 1.1, "b", transform=axes_flat[0].transAxes,
                      fontsize=11, fontweight="bold", va="top")

    fig.subplots_adjust(wspace=0.4, hspace=0.5)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "supp_dose_response.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "supp_dose_response.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved supp_dose_response -> {output_dir}/supp_dose_response.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global N_BOOTSTRAP
    parser = argparse.ArgumentParser(description="Supplementary CBI analyses")
    parser.add_argument("--output", choices=["predrug", "dose_response"], default=None,
                        help="Run only one analysis (default: both)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_patients", type=int, default=None)
    parser.add_argument("--debug", action="store_true",
                        help="N_BOOT=50, cap cohort at 200 patients")
    args = parser.parse_args()

    n_boot = 50 if args.debug else N_BOOTSTRAP
    n_patients = args.n_patients
    if n_patients is None and args.debug:
        n_patients = 200

    run_predrug = args.output in (None, "predrug")
    run_dose = args.output in (None, "dose_response")

    # --- Analysis 1: pre-drug vitals (training data only, no model needed) ---
    if run_predrug:
        train_path = DATA_DIR / "train.feather"
        print("\n=== Analysis 1: Pre-drug vital distributions (training set) ===")
        drug_data = {}
        for entry in DRUG_VITAL_PAIRS:
            name = entry["label"]
            print(f"\n  {name} ({entry['token']}) -> {entry['vital']}")
            predrug, general = compute_predrug_vitals(
                train_path, entry["token"], entry["vital"])
            print(f"    Pre-drug values: n={len(predrug)}, "
                  f"mean={predrug.mean():.1f}" if len(predrug) > 0 else "    No data")
            print(f"    General values:  n={len(general)}, "
                  f"mean={general.mean():.1f}" if len(general) > 0 else "    No general data")
            drug_data[name] = {
                "predrug": predrug,
                "general": general,
                "vital": entry["vital"],
                "is_cbi": entry["is_cbi"],
                "expected_dir": entry["expected_dir"],
            }
        plot_predrug_vitals(drug_data, OUTPUT_DIR)

    # --- Analysis 4: dose-response (model inference) ---
    if run_dose:
        print("\n=== Analysis 4: Dose-response line plots ===")
        print("Loading vocab, model, cohort...")
        vocab, name_to_id, id_to_name = load_vocab()
        model, config = load_model(checkpoint_path=CHECKPOINT_PATH)
        df = load_test_cohort(acuity_filter=ACUITY_FILTER)
        print(f"  {df['encounter_key'].nunique()} encounters")

        print("Building general cohort sequences...")
        sequences = build_cohort_sequences(df, name_to_id)
        if n_patients and len(sequences) > n_patients:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(sequences), size=n_patients, replace=False)
            sequences = [sequences[i] for i in idx]
        print(f"  {len(sequences)} patients")

        dose_data = {}
        for entry in DRUG_VITAL_PAIRS:
            name = entry["label"]
            token = entry["token"]
            if token not in name_to_id:
                print(f"  WARNING: {token} not in vocab, skipping")
                dose_data[name] = [(d, 0.0, 0.0, 0.0) for d in DOSE_LEVELS]
                continue
            print(f"\n  {name}")
            dose_data[name] = compute_dose_response(
                model, config, name_to_id, sequences, entry,
                n_patients, args.batch_size, n_boot)
            for n_doses, mean_d, ci_lo, ci_hi in dose_data[name]:
                print(f"    {n_doses}x: Δ = {mean_d:+.2f} ({ci_lo:+.2f} to {ci_hi:+.2f})")

        plot_dose_response(dose_data, OUTPUT_DIR)

    print(f"\nDONE. Outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
