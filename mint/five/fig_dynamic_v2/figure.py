"""Dynamic patient reprioritization v2.

This script replaces the old ``mint.five.fig_dynamic`` package with a single
entry point that can:

* compute the cached data products needed for the figure
* render the main composite figure
* render the supplementary SHAP figure

Usage:
    python -m mint.five.fig_dynamic_v2 --stage all
    python -m mint.five.fig_dynamic_v2 --stage data
    python -m mint.five.fig_dynamic_v2 --stage figures

    python -m mint.five.fig_dynamic_v2 --stage figures --cdf
"""

from __future__ import annotations

import argparse
import pickle
from string import ascii_lowercase
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.transforms import blended_transform_factory, offset_copy
from scipy.stats import wilcoxon
from torch.utils.data import DataLoader
from tqdm import tqdm

from mint.five.data import CategoricalLabel, Case, CaseDataset, Task, case_collate_fn
from mint.five.fig_one.fig1 import DEVICE, VOCAB, load_model

CHECKPOINT = "output/mint/ckpt.pt"
DATA_DIR = Path("output")
NOTES_PATH = Path("cdw/all_pediatric_ed_visits_with_note.csv")
SAVE_DIR = Path("artifacts/fig_dynamic_v2")
STYLE_PATH = Path(__file__).parent.parent / "design-skill" / "nature.mplstyle"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = SAVE_DIR / "shap_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

COHORT_SEED = 330
SIM_SEED = 0
N_SIMS = 1000
N_ADMIT = 20
N_DISCHARGE = 80
K = 15
T_RANK = 30
T_MAX_BUMP = 30
HORIZON_MINUTES = 60 * 12

PROBS_T30_CACHE = SAVE_DIR / "_cache_probs_t30_softmaxbin.csv"
RESOURCE_CACHE = SAVE_DIR / "_cache_encounter_resources.csv"
RESOURCE_SIM_CACHE = SAVE_DIR / "_cache_resource_sims.csv"
DYNAMIC_RANKS_CACHE = SAVE_DIR / "dynamic_ranks.csv"
COHORT_LABELS_CACHE = SAVE_DIR / "cohort_labels.csv"
COHORT_METADATA_CACHE = SAVE_DIR / "cohort_metadata.csv"

PANEL_A_NOTES = {
    "D239C1FB5DEB6A": ("Hx: Reactive airway disease (ex 34-weeker)", 0),
    "DDA282660A3B9A": ("Hx: Medically complex (ex 29-weeker)", 5),
    "D3504C361B1841": ("Respiratory decompensation at t=60 min", 5),
}

PANEL_A_NOTES = { k:(textwrap.fill(v[0], width=40), v[1]) for k,v in PANEL_A_NOTES.items() }

PANEL_B_RANKS = [(1, 5), (16, 20)]
PANEL_B_DESCRIPTIONS = {
    "D34A13E0398468": "Infant girl, s/p fall with subdural hematoma; transfer by air",
    "D517C0B71EC7E6": "4 y.o. girl, referral for appendicitis; surgical admit",
    "DEE1C0E437FAFC": "Infant male, seizures",
    "D10773AEA35DDB": "Infant boy, hypoxic episode; history of trach-vent dependence",
    "DFBBB304A29490": "5 y.o. girl, fevers and chills; history of acute myeloid leukemia",
    "DB88C0A0F683C0": "1 y.o. boy, eyebrow laceration",
    "D6D9E65D3059DC": "3 y.o. girl, vomiting and fever (Tmax 104F)"
}

PANEL_B_DESCRIPTIONS = { k:textwrap.fill(v, width=40) for k,v in PANEL_B_DESCRIPTIONS.items() }


SHAP_CASES = [
    {
        "encounter_key": "DDE88C1CCB015D",
        "t_max": 90,
        "step": 1,
        "clean": "mode1_bw",
        # "clean": "nothing",
        "destination": "composite",
        "vlines": [
            {"time": 5, "label": "Ventilator; SpO2 92%"},
            {"time": 13, "label": "Midazolam"},
            {"time": 16, "label": "Fosphenytoin"},
            {"time": 25, "label": "GCS 7"},
            {"time": 39, "label": "X-ray of chest"},
            {"time": 43, "label": "Keppra"},
            {"time": 53, "label": "Abnormal albumin"},
            {"time": 82, "label": "Abnormal ketones"},
        ],
    },
    {
        "encounter_key": "D93401F5610C29",
        "title": "6-year-old boy with an extremity laceration",
        "t_max": 180,
        "step": 1,
        # "clean": "nothing",
        "destination": "supplement",
        "vlines": [
            {"time": 4, "label": "Normal vitals"},
            {"time": 33, "label": "X-ray of foot"},
            {"time": 76, "label": "Acetaminophen"},
            {"time": 107, "label": "Lidocaine"},
            {"time": 148, "label": "Midazolam"},
            {"time": 177, "label": "Laceration repair"},
        ],
    },
    {
        "encounter_key": "DDC7AFE4078836",
        "title": "Infant girl with difficulty breathing",
        "t_max": 100,
        "step": 1,
        "destination": "supplement",
        # "clean": "nothing",
        "vlines": [
            {"time": 2, "label": "Blow-by oxygen", "side": "right"},
            {"time": 6, "label": "High-flow nasal cannula", "side": "right"},
            {"time": 15, "label": "Blood gas order"},
            {"time": 21, "label": "X-ray of chest/abdomen", "side": "right"},
            {"time": 38, "label": "Ipatropium"},
            {"time": 44, "label": "Levalbuterol"},
            {"time": 44, "label": "Levalbuterol"},
            {"time": 52, "label": "Respiratory rate, 84"},
            {"time": 59, "label": "IV fluids"},
            {"time": 69, "label": "Respiratory rate, 68"},
            {"time": 73, "label": "EKG"},
            {"time": 95, "label": "Abnormal labs", "side": "right"},
            # 6 HFNC
            # 2 blowby
            # 15 blood gas
            # Med_levalbuterol    44
            # 73 EKG
        ],
    },
    {
        "encounter_key": "DDA63F03DDB90C",
        "title": "15-year-old girl with supraventricular tachycardia (SVT)",
        "t_max": 110,
        "step": 1,
        "clean": "nothing",
        "destination": "supplement",
        "vlines": [
            {"time": 9, "label": "Pulse, 200 bpm"},
            {"time": 15, "label": "EKG"},
            {"time": 18, "label": "Adenosine"},
            {"time": 24, "label": "Pulse, 140 bpm"},
            {"time": 31, "label": "Pulse, 100 bpm"},
            {"time": 41, "label": "X-ray of chest"},
            {"time": 47, "label": "Pulse, 90 bpm"},
            {"time": 67, "label": "Pulse, 95 bpm"},
            {"time": 55, "label": "Pulse, 100 bpm"},
            {"time": 95, "label": "Abnormal labs"},
        ],
    }
    # D503AC2A635528 -- zero year-old sick trauma
]

CLEAN_MODES = {
    "mode1": {
        "O2 Device": {"rename": "Ventilator", "show_tokens": False, "offset": (-170, 6)},
        "Labs": {"rename": "Abnormal Labs", "show_tokens": False, "offset": (-67, 10)},
        "Medications": {"rename": "Medications", "show_tokens": False, "offset": (-30, 2)},
        "Blood Pressure": {"rename": "BP", "show_tokens": False, "offset": (-7, -1.5)},
        "SpO2": {"rename": "SpO2", "offset": (0, 0), "show_tokens": False},
        "Procedures": {
            "rename": "Procedures\n(X-ray of chest, Blood gas)",
            "offset": (0, 0),
            "show_tokens": False,
        },
    },
    "mode2": {
        "Other": {"rename": "Triage information", "show_tokens": False},
        "O2 Device": {"rename": "Room air", "show_tokens": False},
        "Blood Pressure": {"rename": "BP", "show_tokens": False},
        "Procedures": {"rename": "Procedures", "show_tokens": False },
        "Medications": {"rename": "Medications", "show_tokens": False},
    },
    "mode1_bw": {
        "version": 2,
        "groups": {
            "O2 Device": {"rename": "Ventilator", "show_tokens": False, "label_side": "both", "anchor_time": 20, "offset": (0,-2)},
            "Labs": {"rename": "Abnormal\nLabs", "show_tokens": False, "label_side": "positive", "anchor_time": 59},
            "Medications": {"rename": "Medications", "show_tokens": False, "label_side": "positive", "anchor_time": 68},
            "Pulse": {"rename": "Improving\nPulse", "show_tokens": False, "label_side": "negative", "anchor_time": 32},
            "SpO2": {"rename": "Improving\nSpO2", "show_tokens": False, "label_side": "negative", "anchor_time": 20, "offset": (0,-3)},
        },
        "palette": {
            "positive": ["#ff4d6a", "#ff6b8a", "#ff89a5", "#ffa7bf", "#ffc5d9", "#e84570", "#d63060"],
            "negative": ["#4da6ff", "#6bb5ff", "#89c4ff", "#a7d3ff", "#c5e2ff", "#3090e8", "#1a7ad6"],
            "gray_positive": ["#d9d9d9", "#bfbfbf", "#a6a6a6", "#8c8c8c", "#737373"],
            "gray_negative": ["#d0d0d0", "#b6b6b6", "#9d9d9d", "#848484", "#6b6b6b"],
        },
    },
    "nothing": {}
}

SHAP_GROUPS = [
    ("SpO2", lambda name: name.startswith("Vital_SpO2_")),
    ("Pulse", lambda name: name.startswith("Vital_Pulse_")),
    ("Resp Rate", lambda name: name.startswith("Vital_Resp_")),
    (
        "Blood Pressure",
        lambda name: (
            name.startswith("Vital_Systolic_")
            or name.startswith("Vital_Diastolic_")
            or name.startswith("Vital_MAP")
        ),
    ),
    ("Temperature", lambda name: name.startswith("Vital_Temp_")),
    ("O2 Device", lambda name: name.startswith("Vital_O2 Device_")),
    ("Medications", lambda name: name.startswith("Med_")),
    ("Procedures", lambda name: name.startswith("Procedure_")),
    ("Labs", lambda name: name.startswith("Lab_")),
    ("Age/Sex", lambda name: name.startswith("Age_") or name.startswith("Sex_")),
    ("Other", lambda name: True),
]

METHODS = ("softmax_binary",)


def _resolve_clean_mode(clean_config):
    if not isinstance(clean_config, dict):
        return 1, {}, {}
    version = clean_config.get("version", 1)
    if version == 2:
        group_configs = clean_config.get("groups")
        if group_configs is None:
            group_configs = {
                k: v for k, v in clean_config.items() if k not in {"version", "groups", "palette"}
            }
        palette = clean_config.get("palette", {})
        return version, group_configs, palette
    return version, clean_config, {}


def plot_dynamic_ranks(ax=None, t_max=60, clean=False, superclean=False, smooth=False):
    """Draw the bump chart onto ax. If ax is None, create a standalone figure."""
    from scipy.ndimage import uniform_filter1d

    standalone = ax is None
    if standalone:
        apply_nature_style()
        fig, ax = plt.subplots(figsize=(5.5, 7.5))
    else:
        fig = ax.get_figure()

    ranks = pd.read_csv(DYNAMIC_RANKS_CACHE)
    labels = pd.read_csv(COHORT_LABELS_CACHE)
    ranks = ranks.merge(labels[["encounter_key", "admit"]], on="encounter_key", how="left")

    if superclean:
        display_times = [0, 10] + list(range(11, t_max + 1))
    elif clean:
        display_times = [0, 5, 10] + list(range(11, t_max + 1))
    else:
        display_times = list(range(0, t_max + 1))
    display_df = ranks[ranks.t_min.isin(display_times)].copy()

    n_pat = display_df.groupby("t_min")["encounter_key"].nunique().max()
    x_positions = {t: t for t in display_times}

    ax.axhspan(0, K + 0.5, facecolor="#FFF0F0", edgecolor="none", zorder=0)

    for enc_key, grp in display_df.groupby("encounter_key"):
        grp = grp.sort_values("t_min")
        times_available = grp["t_min"].values
        ranks_available = grp["rank"].values.astype(float)
        is_admitted = grp["admit"].values[0] == 1

        xs = np.array([x_positions[t] for t in times_available])
        ys = ranks_available
        if smooth and len(ys) >= 5:
            ys = uniform_filter1d(ys, size=3, mode="nearest")

        if not is_admitted:
            ax.plot(
                xs,
                ys,
                color="#acacac",
                linewidth=2,
                alpha=0.3,
                solid_capstyle="round",
                zorder=1,
            )
        else:
            ax.plot(
                xs,
                ys,
                color="#005FA3",
                linewidth=2,
                alpha=0.6,
                solid_capstyle="round",
                zorder=3,
            )

    final_rank_df = display_df[display_df.t_min == t_max].copy()
    final_rank_df = final_rank_df.sort_values("rank").reset_index(drop=True)
    rank_to_enc = final_rank_df.set_index("rank")["encounter_key"].to_dict()

    for enc_key, (label_text, offset) in PANEL_A_NOTES.items():
        rank_row = final_rank_df[final_rank_df["encounter_key"] == enc_key]
        if rank_row.empty:
            continue
        rank = int(rank_row.iloc[0]["rank"])
        ax.text(
            x_positions[t_max],
            rank + offset - 1,
            label_text,
            fontsize=9,
            va="bottom",
            ha="right",
            color="0.15",
            style="italic",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.5),
        )

    ax.axhline(K + 0.5, color="0.6", linewidth=0.4, linestyle="-", zorder=5)

    ax.text(
        x_positions[0],
        -3,
        f"Top-{K} admission enrichment",
        ha="left",
        va="bottom",
        fontsize=11,
        color="0.15",
        style="italic",
    )

    n_total_admitted = int(display_df[display_df.t_min == 0]["admit"].sum())
    expected = n_total_admitted * K / n_pat
    for t in [0, t_max]:
        t_data = display_df[display_df.t_min == t]
        top_k = t_data[t_data["rank"] <= K]
        n_admitted_top_k = int(top_k["admit"].sum())
        enrichment = n_admitted_top_k / expected if expected else 0.0
        x = x_positions[t]
        ha = "left" if t == 0 else "right"
        ax.text(
            x,
            -0.5,
            f"{n_admitted_top_k}/{K} ({enrichment:.1f}×)",
            ha=ha,
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#005FA3",
        )

    ax.set_xlim(-0.5, t_max + 0.5)
    ax.set_ylim(n_pat + 2, -5)
    tick_times = list(range(0, t_max + 1, 10))
    ax.set_xticks([x_positions[t] for t in tick_times])
    ax.set_xticklabels([f"{t}" for t in tick_times], fontsize=11)
    ax.set_xlabel("Minutes into visit", fontsize=12)
    ax.set_ylabel("Patient rank (1 = highest priority)", fontsize=12)
    ax.set_yticks([1, 20, 40, 60, 80, 100])
    ax.tick_params(axis="y", labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#005FA3", alpha=0.45, label="Admitted"),
        Patch(facecolor="#acacac", alpha=0.25, label="Discharged"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower left",
        fontsize=11,
        frameon=True,
        framealpha=0.9,
        edgecolor="0.85",
    )

    if standalone:
        plt.tight_layout()
        plt.savefig(SAVE_DIR / "fig_dynamic.pdf", dpi=600, bbox_inches="tight")
        plt.savefig(SAVE_DIR / "fig_dynamic.png", dpi=600, bbox_inches="tight")
        plt.close()
        print("Saved fig_dynamic.pdf and fig_dynamic.png")


def build_disposition_task() -> Task:
    label = CategoricalLabel(
        positive_tokens=["Admit"],
        negative_token_match="Discharge",
        vocab=VOCAB,
    )
    return Task(
        name="admit",
        label=label,
        lookahead_min=1,
        exclusion_window_min=999_999,
        tokens_min=3,
    )


def get_ed_only_df(df: pd.DataFrame) -> pd.DataFrame:
    events = {"Admit", "ICU Start", "Discharge"}
    after_event = df.groupby("encounter_key")["name"].transform(lambda s: s.isin(events).cummax())
    return df[~after_event].copy()


def build_time_windowed_cases(df: pd.DataFrame, encounter_keys: set[str], max_time: float):
    name_to_id = {k: v + 1 for k, v in zip(VOCAB["name"], VOCAB["index"])}

    full_df = df.sort_values(["encounter_key", "t"])
    pos_encounters = set(full_df[full_df.name == "Admit"].encounter_key.unique())

    ed_df = get_ed_only_df(full_df)
    ed_df = ed_df[ed_df.encounter_key.isin(encounter_keys)].copy()
    ed_df["token_id"] = ed_df["name"].map(name_to_id)
    ed_df = ed_df[ed_df.t <= max_time]

    age_df = ed_df[ed_df["name"].str.startswith("Age_")].copy()
    age_df["age"] = age_df["name"].str.removeprefix("Age_").astype(int)
    encounter_to_age = dict(zip(age_df.encounter_key, age_df.age))

    task = build_disposition_task()
    age_to_pos_id, age_to_neg_id = task.label.get_age_to_token_ids()

    cases = []
    for enc_key, grp in ed_df.groupby("encounter_key"):
        if enc_key not in encounter_to_age:
            continue
        events = grp["token_id"].values
        times = grp["t"].values
        if len(events) < 1:
            continue
        label = 1 if enc_key in pos_encounters else 0
        cases.append(
            Case(
                encounter_key=enc_key,
                label=label,
                events=events,
                times=times,
                age=encounter_to_age[enc_key],
            )
        )

    return cases, age_to_pos_id, age_to_neg_id


def run_methods_on_cases(model, cases, age_to_pos_id, age_to_neg_id, max_len=512, batch_size=32):
    ds = CaseDataset(cases, age_to_pos_id, age_to_neg_id, max_len=max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=case_collate_fn)

    probs = {method: [] for method in METHODS}
    labels_list = []
    enc_keys_list = []

    model.eval()
    for batch in loader:
        events = batch["events"].to(DEVICE)
        times = batch["times"].to(DEVICE)
        lengths = batch["lengths"].to(DEVICE) - 1
        pos_ids_batch = batch["pos_ids"]
        neg_ids_batch = batch["neg_ids"]

        labels_list.extend(batch["labels"].cpu().numpy())
        enc_keys_list.extend(batch["encounter_key"])

        with torch.no_grad():
            logits, _, _ = model(events, times)

        for j in range(events.size(0)):
            last_logits = logits[j, lengths[j], :]
            pos_idx = torch.as_tensor(pos_ids_batch[j], dtype=torch.long, device=DEVICE)
            neg_idx = torch.as_tensor(neg_ids_batch[j], dtype=torch.long, device=DEVICE)
            sel_logits = last_logits.index_select(0, torch.cat([pos_idx, neg_idx]))
            sel_sm = torch.softmax(sel_logits, dim=-1)
            probs["softmax_binary"].append(sel_sm[: pos_idx.numel()].sum().item())

    return (
        {method: np.asarray(values, dtype=np.float32) for method, values in probs.items()},
        np.asarray(labels_list, dtype=np.int32),
        enc_keys_list,
    )


def compute_dynamic_cohort(model):
    test = pd.read_feather(DATA_DIR / "test.feather")
    rng = np.random.default_rng(COHORT_SEED)

    admit_encounters = set(test[test.name == "Admit"].encounter_key.unique())
    all_encounters = set(test.encounter_key.unique())
    discharged_encounters = all_encounters - admit_encounters

    selected_admit = rng.choice(sorted(admit_encounters), size=N_ADMIT, replace=False).tolist()
    selected_discharged = rng.choice(sorted(discharged_encounters), size=N_DISCHARGE, replace=False).tolist()
    cohort = selected_admit + selected_discharged
    cohort_set = set(cohort)

    print(f"Selected cohort: {len(cohort)} patients (20 admit, 80 discharged)")

    all_rows = []
    for t_min in tqdm(range(0, T_MAX_BUMP + 1), desc="Cohort timepoints"):
        cases, age_to_pos_id, age_to_neg_id = build_time_windowed_cases(test, cohort_set, t_min)
        if not cases:
            continue

        probs_by_method, labels, enc_keys = run_methods_on_cases(
            model, cases, age_to_pos_id, age_to_neg_id
        )
        probs = probs_by_method["softmax_binary"]

        for enc_key, prob, label in zip(enc_keys, probs, labels):
            all_rows.append(
                {
                    "encounter_key": enc_key,
                    "t_min": t_min,
                    "admit_prob": float(prob),
                    "admit_label": int(label),
                }
            )

    df_all = pd.DataFrame(all_rows)
    df_all["rank"] = df_all.groupby("t_min")["admit_prob"].rank(
        ascending=False, method="first"
    ).astype(int)
    t30_unique_scores = df_all.loc[df_all["t_min"] == T_RANK, "admit_prob"].nunique()
    print(f"Selected cohort t={T_RANK}: {t30_unique_scores} unique admit_prob values")
    df_all.to_csv(DYNAMIC_RANKS_CACHE, index=False)
    print(f"Saved {DYNAMIC_RANKS_CACHE.name}")

    cohort_labels = pd.DataFrame(
        {
            "encounter_key": cohort,
            "admit": [1 if e in admit_encounters else 0 for e in cohort],
        }
    )
    cohort_labels.to_csv(COHORT_LABELS_CACHE, index=False)
    print(f"Saved {COHORT_LABELS_CACHE.name}")

    pivot = df_all.pivot(index="encounter_key", columns="t_min", values="rank")
    pivot.columns = [f"rank_t{int(c)}" for c in pivot.columns]
    pivot = pivot.reset_index()
    pivot = pivot.merge(cohort_labels, on="encounter_key", how="left")
    pivot["final_rank"] = pivot[f"rank_t{T_MAX_BUMP}"]

    notes = pd.read_csv(
        NOTES_PATH,
        usecols=["EncounterKey", "note_text", "ArrivalMethod", "AcuityLevel", "Age"],
    )
    notes = notes.rename(columns={"EncounterKey": "encounter_key"})
    pivot = pivot.merge(notes, on="encounter_key", how="left")

    rank_cols = sorted([c for c in pivot.columns if c.startswith("rank_t")], key=lambda c: int(c[6:]))
    ordered = [
        "encounter_key",
        "admit",
        "final_rank",
        *rank_cols,
        "note_text",
        "ArrivalMethod",
        "AcuityLevel",
        "Age",
    ]
    pivot = pivot[ordered].sort_values("final_rank")
    pivot.to_csv(COHORT_METADATA_CACHE, index=False)
    print(f"Saved {COHORT_METADATA_CACHE.name}")

    return df_all, cohort_labels, pivot


def compute_encounter_resources():
    test = pd.read_feather(DATA_DIR / "test.feather")
    test["encounter_key"] = test["encounter_key"].astype(str)

    stats = test.groupby("encounter_key").agg(
        n_meds=("name", lambda x: x.str.startswith("Med_").sum()),
        n_procedures=("name", lambda x: x.str.startswith("Procedure_").sum()),
        n_labs=("name", lambda x: x.str.startswith("Lab_").sum()),
        max_t=("t", "max"),
    ).reset_index()

    admit_encounters = set(test[test.name == "Admit"].encounter_key.unique())
    stats["admit"] = stats["encounter_key"].isin(admit_encounters).astype(int)
    stats["n_resources"] = stats["n_meds"] + stats["n_procedures"] + stats["n_labs"]
    stats.to_csv(RESOURCE_CACHE, index=False)
    print(f"Saved {RESOURCE_CACHE.name}")
    return stats


def compute_t30_probs(model):
    test = pd.read_feather(DATA_DIR / "test.feather")
    all_encounters = set(test.encounter_key.unique())
    cases, age_to_pos_id, age_to_neg_id = build_time_windowed_cases(test, all_encounters, T_RANK)
    probs_by_method, labels, enc_keys = run_methods_on_cases(
        model, cases, age_to_pos_id, age_to_neg_id
    )

    df = pd.DataFrame(
        {
            "encounter_key": enc_keys,
            "admit_prob": probs_by_method["softmax_binary"].astype(float),
            "admit_label": labels.astype(int),
        }
    )
    df.to_csv(PROBS_T30_CACHE, index=False)
    print(f"Saved {PROBS_T30_CACHE.name}")
    return df


def compute_resource_sims(prob_df: pd.DataFrame, resource_df: pd.DataFrame):
    labels_df = prob_df.copy()
    admit_list = sorted(labels_df[labels_df.admit_label == 1]["encounter_key"].unique())
    discharge_list = sorted(labels_df[labels_df.admit_label == 0]["encounter_key"].unique())
    print(f"Pool: {len(admit_list)} admit, {len(discharge_list)} discharged")

    prob_lookup = prob_df.set_index("encounter_key")["admit_prob"]
    resource_lookup = resource_df.set_index("encounter_key")

    rng = np.random.default_rng(SIM_SEED)
    rows = []
    for sim_idx in tqdm(range(N_SIMS), desc="Resource sims"):
        selected_admit = rng.choice(admit_list, size=N_ADMIT, replace=False)
        selected_discharge = rng.choice(discharge_list, size=N_DISCHARGE, replace=False)
        cohort = list(selected_admit) + list(selected_discharge)

        cohort_probs = prob_lookup.loc[cohort].dropna()
        if len(cohort_probs) < K:
            continue

        order = np.argsort(-cohort_probs.values)
        top_encs = [cohort_probs.index[i] for i in order[:K]]
        bot_encs = [cohort_probs.index[i] for i in order[K:]]

        top_res = resource_lookup.loc[top_encs]
        bot_res = resource_lookup.loc[bot_encs]

        for metric in ["admit", "n_meds", "n_procedures", "n_labs", "max_t"]:
            rows.append(
                {
                    "sim": sim_idx,
                    "metric": metric,
                    "top_mean": float(top_res[metric].mean()),
                    "bot_mean": float(bot_res[metric].mean()),
                }
            )

    resource_sim_df = pd.DataFrame(rows)
    resource_sim_df.to_csv(RESOURCE_SIM_CACHE, index=False)
    print(f"Saved {RESOURCE_SIM_CACHE.name}")
    return resource_sim_df


def _ensure_data_stage():
    need_model = not (
        DYNAMIC_RANKS_CACHE.exists()
        and COHORT_LABELS_CACHE.exists()
        and COHORT_METADATA_CACHE.exists()
        and PROBS_T30_CACHE.exists()
    )

    model = None
    if need_model:
        print(f"Loading model from {CHECKPOINT}...")
        model, _ = load_model(CHECKPOINT)

    if not DYNAMIC_RANKS_CACHE.exists() or not COHORT_LABELS_CACHE.exists() or not COHORT_METADATA_CACHE.exists():
        if model is None:
            print(f"Loading model from {CHECKPOINT}...")
            model, _ = load_model(CHECKPOINT)
        compute_dynamic_cohort(model)
    else:
        print(f"[cache hit] {DYNAMIC_RANKS_CACHE.name}")
        print(f"[cache hit] {COHORT_LABELS_CACHE.name}")
        print(f"[cache hit] {COHORT_METADATA_CACHE.name}")

    if not RESOURCE_CACHE.exists():
        compute_encounter_resources()
    else:
        print(f"[cache hit] {RESOURCE_CACHE.name}")

    if not PROBS_T30_CACHE.exists():
        if model is None:
            print(f"Loading model from {CHECKPOINT}...")
            model, _ = load_model(CHECKPOINT)
        compute_t30_probs(model)
    else:
        print(f"[cache hit] {PROBS_T30_CACHE.name}")

    if not RESOURCE_SIM_CACHE.exists():
        prob_df = pd.read_csv(PROBS_T30_CACHE)
        resource_df = pd.read_csv(RESOURCE_CACHE)
        compute_resource_sims(prob_df, resource_df)
    else:
        print(f"[cache hit] {RESOURCE_SIM_CACHE.name}")


def classify_token(name: str) -> str:
    for group, matcher in SHAP_GROUPS:
        if group == "Other":
            continue
        if matcher(name):
            return group
    return "Other"


def _admit_prob_from_logits(last_logits, pos_idx, neg_idx, use_cdf=False):
    if use_cdf:
        rates = torch.exp(last_logits).clamp(min=1e-10)
        lambda_pos = rates.index_select(0, pos_idx).sum()
        lambda_total = rates.sum() - rates[:2].sum()
        p = (lambda_pos / lambda_total) * (-torch.expm1(-lambda_total * HORIZON_MINUTES))
        return p.clamp(1e-8, 1 - 1e-8).item()

    sel_logits = last_logits.index_select(0, torch.cat([pos_idx, neg_idx]))
    sel_sm = torch.softmax(sel_logits, dim=-1)
    return sel_sm[: pos_idx.numel()].sum().clamp(1e-8, 1 - 1e-8).item()


def compute_baseline_log_score(model, enc_df, name_to_id, pos_ids, neg_ids, use_cdf=False):
    context = enc_df[enc_df["t"] <= 0].copy()
    context["token_id"] = context["name"].map(name_to_id)
    context = context.dropna(subset=["token_id"])
    context["token_id"] = context["token_id"].astype(int)

    if len(context) < 2:
        context = enc_df.head(3).copy()
        context["token_id"] = context["name"].map(name_to_id)
        context = context.dropna(subset=["token_id"])
        context["token_id"] = context["token_id"].astype(int)

    token_ids = context["token_id"].values.tolist()
    times_list = context["t"].values.tolist()

    pos_idx = torch.tensor(pos_ids, dtype=torch.long, device=DEVICE)
    neg_idx = torch.tensor(neg_ids, dtype=torch.long, device=DEVICE)
    events_t = torch.tensor(token_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    times_t = torch.tensor(times_list, dtype=torch.float, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        logits, _, _ = model(events_t, times_t)

    last_logits = logits[0, -1, :]
    p = _admit_prob_from_logits(last_logits, pos_idx, neg_idx, use_cdf=use_cdf)
    if use_cdf:
        return np.log(p)
    return np.log(p) - np.log1p(-p)


def make_shap_predict_fn(model, token_ids, times, pos_ids, neg_ids, baseline_score, use_cdf=False):
    token_ids_arr = np.array(token_ids)
    times_arr = np.array(times)
    pos_idx = torch.tensor(pos_ids, dtype=torch.long, device=DEVICE)
    neg_idx = torch.tensor(neg_ids, dtype=torch.long, device=DEVICE)

    def predict(masked_inputs):
        if isinstance(masked_inputs, np.ndarray):
            if masked_inputs.ndim == 1:
                rows = [str(row).split() for row in masked_inputs]
            else:
                rows = [list(row) for row in masked_inputs]
        else:
            rows = [str(row).split() for row in masked_inputs]

        results = []
        for tokens_str in rows:
            keep_mask = np.array([str(t) != "0" for t in tokens_str[: len(token_ids_arr)]])
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
            p = _admit_prob_from_logits(last_logits, pos_idx, neg_idx, use_cdf=use_cdf)
            if use_cdf:
                score = np.log(p)
            else:
                score = np.log(p) - np.log1p(-p)
            results.append(score - baseline_score)

        return np.array(results).reshape(-1, 1)

    return predict


def compute_shap_at_time(
    model,
    enc_df,
    name_to_id,
    pos_ids,
    neg_ids,
    eval_t,
    baseline_score,
    use_cdf=False,
    max_len=512,
):
    import shap

    context = enc_df[enc_df["t"] <= eval_t].copy()
    context["token_id"] = context["name"].map(name_to_id)
    context = context.dropna(subset=["token_id"])
    context["token_id"] = context["token_id"].astype(int)

    if len(context) < 3:
        return None

    token_names = context["name"].values.tolist()
    token_ids = context["token_id"].values.tolist()
    times = context["t"].values.tolist()

    if len(token_ids) > max_len:
        token_ids = token_ids[-max_len:]
        times = times[-max_len:]
        token_names = token_names[-max_len:]

    predict_fn = make_shap_predict_fn(
        model, token_ids, times, pos_ids, neg_ids, baseline_score, use_cdf=use_cdf
    )

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

    masker = shap.maskers.Text(
        tokenizer,
        mask_token="0",
        output_type="str",
        collapse_mask_token=False,
    )
    explainer = shap.Explainer(
        predict_fn,
        masker,
        output_names=["delta_log_risk"] if use_cdf else ["delta_log_odds"],
    )
    shap_values = explainer([input_text])

    vals = shap_values.values[0, :, 0]
    base_value = float(shap_values.base_values[0, 0])

    return {
        "shap_values": vals,
        "token_names": token_names,
        "base_value": base_value,
        "predicted": base_value + float(vals.sum()),
    }


def aggregate_per_token(shap_results):
    all_tokens = set()
    for result in shap_results:
        all_tokens.update(result["token_names"])
    token_labels = sorted(all_tokens)
    token_to_idx = {t: i for i, t in enumerate(token_labels)}

    n_times = len(shap_results)
    values = np.zeros((n_times, len(token_labels)))
    base_values = np.zeros(n_times)

    for i, result in enumerate(shap_results):
        base_values[i] = result["base_value"]
        for val, name in zip(result["shap_values"], result["token_names"]):
            values[i, token_to_idx[name]] += val

    token_importance = np.abs(values).mean(axis=0)
    return values, base_values, token_labels, token_importance


def apply_nature_style():
    plt.style.use(str(STYLE_PATH))
    plt.rcParams.update(
        {
            "figure.constrained_layout.use": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica"],
            "axes.linewidth": 0.5,
            "axes.edgecolor": "black",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.size": 2.5,
            "xtick.minor.size": 1.5,
            "xtick.major.width": 0.5,
            "xtick.minor.width": 0.4,
            "xtick.direction": "out",
            "ytick.major.size": 2.5,
            "ytick.minor.size": 1.5,
            "ytick.major.width": 0.5,
            "ytick.minor.width": 0.4,
            "ytick.direction": "out",
            "lines.linewidth": 1.2,
        }
    )


def plot_stacked_shap(
    shap_results,
    eval_times,
    encounter_key,
    rank,
    is_admitted,
    output_path,
    use_cdf=False,
    per_token=False,
    jump_events=None,
    manual_vlines=None,
    clean_config=None,
    ax=None,
):
    apply_nature_style()

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 3.5))
    else:
        fig = ax.get_figure()

    clean_version, clean_groups, clean_palette = _resolve_clean_mode(clean_config)

    if per_token:
        values, base_values, feature_names, token_importance = aggregate_per_token(shap_results)
    else:
        group_names = [g for g, _ in SHAP_GROUPS]
        n_times = len(shap_results)
        values = np.zeros((n_times, len(group_names)))
        base_values = np.zeros(n_times)
        group_token_importance = {g: {} for g in group_names}
        for i, result in enumerate(shap_results):
            base_values[i] = result["base_value"]
            group_sums = {g: 0.0 for g in group_names}
            for val, name in zip(result["shap_values"], result["token_names"]):
                group = classify_token(name)
                group_sums[group] += val
                group_token_importance[group][name] = group_token_importance[group].get(name, 0.0) + abs(val)
            for j, g in enumerate(group_names):
                values[i, j] = group_sums[g]
        feature_names = group_names

    times = np.array(eval_times)
    base = base_values
    pos_colors = [
        "#ff4d6a",
        "#ff6b8a",
        "#ff89a5",
        "#ffa7bf",
        "#ffc5d9",
        "#e84570",
        "#d63060",
        "#c41b50",
        "#b20640",
        "#990030",
        "#800020",
    ]
    neg_colors = [
        "#4da6ff",
        "#6bb5ff",
        "#89c4ff",
        "#a7d3ff",
        "#c5e2ff",
        "#3090e8",
        "#1a7ad6",
        "#0564c4",
        "#004eb2",
        "#003890",
        "#002070",
    ]
    gray_pos_colors = clean_palette.get(
        "gray_positive",
        ["#d9d9d9", "#bfbfbf", "#a6a6a6", "#8c8c8c", "#737373"],
    )
    gray_neg_colors = clean_palette.get(
        "gray_negative",
        ["#d0d0d0", "#b6b6b6", "#9d9d9d", "#848484", "#6b6b6b"],
    )
    custom_pos_colors = clean_palette.get("positive", pos_colors)
    custom_neg_colors = clean_palette.get("negative", neg_colors)

    def pretty_token_name(raw_name):
        prefixes = [
            "Vital_SpO2_",
            "Vital_Pulse_",
            "Vital_Resp_",
            "Vital_Systolic_",
            "Vital_Diastolic_",
            "Vital_MAP_",
            "Vital_Temp_",
            "Vital_O2 Device_",
            "Med_",
            "Procedure_",
            "Lab_",
            "Age_",
            "Sex_",
        ]
        for p in prefixes:
            if raw_name.startswith(p):
                return raw_name[len(p):]
        return raw_name

    predicted = base + values.sum(axis=1)

    abs_importance = np.abs(values).sum(axis=0)
    sort_order = np.argsort(abs_importance)[::-1]
    pos_importance = np.maximum(values, 0).sum(axis=0)
    neg_importance = np.abs(np.minimum(values, 0)).sum(axis=0)
    cumulative_pos = base.copy()
    cumulative_neg = base.copy()
    band_labels = []
    pos_idx = 0
    neg_idx = 0
    gray_pos_idx = 0
    gray_neg_idx = 0

    def display_group_name(group_name):
        if clean_version == 2 and group_name in clean_groups:
            return clean_groups[group_name].get("rename", group_name)
        if clean_config is not None and group_name in clean_config:
            return clean_config[group_name].get("rename", group_name)
        return group_name

    pos_order = np.argsort(pos_importance)[::-1]
    neg_order = np.argsort(neg_importance)[::-1]
    top_pos = [
        f"{display_group_name(feature_names[j])} ({pos_importance[j]:.3f})"
        for j in pos_order[:4]
        if pos_importance[j] > 1e-8
    ]
    top_neg = [
        f"{display_group_name(feature_names[j])} ({neg_importance[j]:.3f})"
        for j in neg_order[:4]
        if neg_importance[j] > 1e-8
    ]
    print(
        "Top SHAP contributors | positive: "
        + ", ".join(top_pos if top_pos else ["none"])
        + " | negative: "
        + ", ".join(top_neg if top_neg else ["none"])
    )
    print("Top tokens by group:")
    for group_name in feature_names:
        token_imp = group_token_importance.get(group_name, {})
        if not token_imp:
            continue
        sorted_tokens = sorted(token_imp.items(), key=lambda kv: kv[1], reverse=True)
        top_tokens = ", ".join(
            f"{pretty_token_name(token)} ({importance:.3f})"
            for token, importance in sorted_tokens[:4]
        )
        print(f"  {display_group_name(group_name)}: {top_tokens}")

    for j in sort_order:
        g = feature_names[j]
        col = values[:, j]
        pos_part = np.maximum(col, 0)
        neg_part = np.minimum(col, 0)
        is_highlighted = clean_version != 2 or g in clean_groups
        cfg = clean_groups[g] if clean_version == 2 and g in clean_groups else None
        label_side = cfg.get("label_side", "positive") if cfg is not None else None

        if pos_part.max() > 0.01:
            bottom = cumulative_pos.copy()
            top = cumulative_pos + pos_part
            if clean_version == 2 and (not is_highlighted or label_side == "negative"):
                color = gray_pos_colors[gray_pos_idx % len(gray_pos_colors)]
                gray_pos_idx += 1
                label = None
            else:
                color = custom_pos_colors[pos_idx % len(custom_pos_colors)]
                pos_idx += 1
                label = g if pos_part.max() > 0.05 else None
            ax.fill_between(
                times,
                bottom,
                top,
                color=color,
                alpha=0.7,
                label=label,
                linewidth=0.2,
                edgecolor="white",
            )
            band_labels.append((g, pos_part, bottom, top))
            cumulative_pos = top

        if neg_part.min() < -0.01:
            bottom = cumulative_neg + neg_part
            top = cumulative_neg.copy()
            if clean_version == 2 and (not is_highlighted or label_side == "positive"):
                color = gray_neg_colors[gray_neg_idx % len(gray_neg_colors)]
                gray_neg_idx += 1
                label = None
            else:
                color = custom_neg_colors[neg_idx % len(custom_neg_colors)]
                neg_idx += 1
                label = g if neg_part.min() < -0.05 else None
            ax.fill_between(
                times,
                bottom,
                top,
                color=color,
                alpha=0.7,
                label=label,
                linewidth=0.2,
                edgecolor="white",
            )
            band_labels.append((g, -neg_part, bottom, top))
            cumulative_neg = cumulative_neg + neg_part

    ax.plot(times, predicted, color="black", linewidth=1.2, zorder=10)

    if not per_token:
        label_min_height = 0.08
        for g, thickness, bottom, top in band_labels:
            if clean_config is not None:
                if clean_version == 2:
                    if g not in clean_groups:
                        continue
                elif g not in clean_config:
                    continue
                cfg = clean_groups[g] if clean_version == 2 else clean_config[g]
                band_side = "positive" if (bottom + top).mean() >= base.mean() else "negative"
                label_side = cfg.get("label_side", "positive")
                if label_side not in ("both", band_side):
                    continue

            if thickness.max() < label_min_height:
                continue
            area = np.trapz(thickness, times)
            if area <= 1e-8:
                continue
            x_centroid = np.trapz(thickness * times, times) / area
            nearest_idx = np.argmin(np.abs(times - x_centroid))
            x = times[nearest_idx]
            y = (bottom[nearest_idx] + top[nearest_idx]) / 2

            if clean_config is not None:
                cfg = clean_groups[g] if clean_version == 2 else clean_config[g]
                display_name = cfg.get("rename", g)
                if cfg.get("show_tokens", True):
                    token_imp = group_token_importance.get(g, {})
                    if token_imp:
                        sorted_tokens = sorted(token_imp.items(), key=lambda kv: kv[1], reverse=True)
                        top_names = [pretty_token_name(t) for t, _ in sorted_tokens[:3]]
                        label_text = f"{display_name} ({', '.join(top_names)})"
                    else:
                        label_text = display_name
                else:
                    label_text = display_name
                dx, dy = cfg.get("offset", (0, 0))
                anchor_time = cfg.get("anchor_time", cfg.get("time"))
            else:
                token_imp = group_token_importance.get(g, {})
                if token_imp:
                    sorted_tokens = sorted(token_imp.items(), key=lambda kv: kv[1], reverse=True)
                    top_names = [pretty_token_name(t) for t, _ in sorted_tokens[:3]]
                    label_text = f"{g} ({', '.join(top_names)})"
                else:
                    label_text = g
                dx, dy = 0, 0
                anchor_time = None

            trans = ax.transData
            if dx != 0 or dy != 0:
                trans = offset_copy(ax.transData, fig=fig, x=dx, y=dy, units="points")
            if anchor_time is not None:
                x = times[np.argmin(np.abs(times - anchor_time))]
                y = (bottom[np.argmin(np.abs(times - anchor_time))] + top[np.argmin(np.abs(times - anchor_time))]) / 2
            ax.text(
                x,
                y,
                label_text,
                fontsize=8,
                ha="center",
                va="center",
                color="black",
                fontweight="normal",
                zorder=11,
                transform=trans,
            )

    if jump_events:
        for evt in jump_events:
            ax.axvline(evt["time"], color="black", linestyle=":", linewidth=1.0, alpha=0.7, zorder=5)

    if manual_vlines:
        for vl in manual_vlines:
            ax.axvline(vl["time"], color="black", linestyle="-", linewidth=1.0, alpha=0.8, zorder=5)
            ax.text(
                vl["time"] - 0.3,
                ax.get_ylim()[1] * 0.98,
                vl["label"],
                fontsize=10,
                rotation=90,
                ha="right",
                va="top",
                color="black",
                zorder=12,
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=1),
            )

    ax.set_xlabel("Minutes into visit", fontsize=11)
    ax.set_ylabel(
        "Change in log(admission probability)" if use_cdf else "Change in log odds of admission",
        fontsize=11,
    )

    ax.tick_params(axis="x", labelsize=10)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.set_xlim(times[0], times[-1])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, axis="y")

    if standalone:
        plt.tight_layout()
        png_path = output_path.with_suffix(".png")
        plt.savefig(png_path, dpi=600, bbox_inches="tight")
        pdf_path = output_path.with_suffix(".pdf")
        plt.savefig(pdf_path, bbox_inches="tight")
        plt.close()
        print(f"Saved {png_path}")
        print(f"Saved {pdf_path}")


def _get_chief_complaints(encounter_keys):
    notes = pd.read_csv(NOTES_PATH, usecols=["EncounterKey", "PrimaryChiefComplaintName"])
    notes = notes[notes["EncounterKey"].isin(set(encounter_keys))]
    cc_map = dict(zip(notes["EncounterKey"], notes["PrimaryChiefComplaintName"]))

    test = pd.read_feather(DATA_DIR / "test.feather")
    for enc in encounter_keys:
        if enc not in cc_map or pd.isna(cc_map.get(enc)):
            enc_df = test[test.encounter_key == enc]
            cc_rows = enc_df[enc_df["name"].str.startswith("CC_")]
            if len(cc_rows) > 0:
                cc_map[enc] = cc_rows.iloc[0]["name"].removeprefix("CC_")

    return cc_map


def _get_vitals_for_cohort(encounter_keys, t_max=60):
    test = pd.read_feather(DATA_DIR / "test.feather")
    cohort_df = test[test.encounter_key.isin(set(encounter_keys))].copy()
    cohort_df = cohort_df[cohort_df.t <= t_max].sort_values(["encounter_key", "t"])

    vitals = {}
    for enc_key, grp in cohort_df.groupby("encounter_key"):
        pulse_rows = grp[grp["name"].str.startswith("Vital_Pulse_")]
        rr_rows = grp[grp["name"].str.startswith("Vital_Resp_")]
        spo2_rows = grp[grp["name"].str.startswith("Vital_SpO2_")]

        pulse = pulse_rows.iloc[-1]["name"].split("_")[-1] if len(pulse_rows) > 0 else ""
        rr = rr_rows.iloc[-1]["name"].split("_")[-1] if len(rr_rows) > 0 else ""
        spo2 = spo2_rows.iloc[-1]["name"].split("_")[-1] if len(spo2_rows) > 0 else ""
        vitals[enc_key] = {"Pulse": pulse, "RR": rr, "SpO2": spo2}

    return vitals


ESI_MAP = {"Immediate": 1, "Emergent": 2, "Urgent": 3, "Less Urgent": 4, "Non-Urgent": 5}
ESI_COLORS = {1: "#E53935", 2: "#FB8C00", 3: "#FDD835", 4: "#66BB6A", 5: "#66BB6A"}


def render_trackboard(ax, t_max=T_MAX_BUMP, rank_ranges=None):
    if rank_ranges is None:
        rank_ranges = PANEL_B_RANKS

    meta = pd.read_csv(COHORT_METADATA_CACHE)
    rank_col = f"rank_t{t_max}"
    if rank_col not in meta.columns:
        rank_col = "final_rank"
    meta = meta.sort_values(rank_col).reset_index(drop=True)
    meta["_rank"] = range(1, len(meta) + 1)

    selected_ranks = []
    for start, end in rank_ranges:
        selected_ranks.extend(range(start, end + 1))
    selected = meta[meta["_rank"].isin(selected_ranks)].copy()
    encounter_keys = selected["encounter_key"].tolist()

    cc_map = _get_chief_complaints(encounter_keys)
    _get_vitals_for_cohort(encounter_keys, t_max=t_max)

    ranks_df = pd.read_csv(DYNAMIC_RANKS_CACHE)
    t_earlier = max(0, t_max - 30)
    rank_at_now = ranks_df[ranks_df.t_min == t_max].set_index("encounter_key")["rank"]
    rank_at_earlier = ranks_df[ranks_df.t_min == t_earlier].set_index("encounter_key")["rank"]
    delta_map = (rank_at_earlier - rank_at_now).to_dict()

    display_rows = []
    for range_idx, (start, end) in enumerate(rank_ranges):
        if range_idx > 0:
            display_rows.append({"_separator": True})
        range_rows = selected[selected["_rank"].between(start, end)].sort_values("_rank")
        for _, row in range_rows.iterrows():
            enc = row["encounter_key"]
            cc = cc_map.get(enc, "")
            if pd.isna(cc):
                cc = ""
            cc = cc.strip("*").title()

            acuity_str = row["AcuityLevel"] if pd.notna(row["AcuityLevel"]) else ""
            esi_num = ESI_MAP.get(acuity_str, None)
            age = int(row["Age"]) if pd.notna(row["Age"]) else ""

            delta = delta_map.get(enc, 0)
            if delta > 0:
                delta_str = f"▲ {int(delta)}"
            elif delta < 0:
                delta_str = f"▼ {int(abs(delta))}"
            else:
                delta_str = "-"

            rank_num = int(row["_rank"])
            display_rows.append(
                {
                    "_separator": False,
                    "rank": rank_num,
                    "delta": delta_str,
                    "delta_val": delta,
                    "age": age,
                    "chief_complaint": cc,
                    "esi_num": esi_num,
                    "description": PANEL_B_DESCRIPTIONS.get(enc, ""),
                    "is_admit": row["admit"] == 1,
                }
            )

    col_labels = ["#", "Delta 30m", "Age", "ESI", "Chief Complaint"]
    col_widths_raw = [0.01, 0.05, 0.018, 0.022, 0.084]
    table_width = 0.58
    desc_x = table_width + 0.02
    total_raw = sum(col_widths_raw)
    col_widths = [w / total_raw * table_width for w in col_widths_raw]
    col_x = [0.0]
    for w in col_widths[:-1]:
        col_x.append(col_x[-1] + w)

    n_visual_rows = len(display_rows)
    row_height = 1.0 / (n_visual_rows + 1.2)
    header_y = 1.0 - row_height * 0.4
    left_pad = 0.008
    fontsize_cell = 9.5
    fontsize_hdr = 10

    for j, label in enumerate(col_labels):
        if j < 2:
            x_hdr = col_x[j] + col_widths[j] / 2
            ha_hdr = "center"
        else:
            x_hdr = col_x[j] + left_pad
            ha_hdr = "left"
        ax.text(
            x_hdr,
            header_y,
            label,
            ha=ha_hdr,
            va="center",
            fontsize=fontsize_hdr,
            fontweight="bold",
            color="white",
            transform=ax.transAxes,
            zorder=10,
        )

    ax.text(
        desc_x,
        header_y,
        "Additional clinical information:",
        ha="left",
        va="center",
        fontsize=fontsize_hdr,
        fontweight="bold",
        color="0",
        transform=ax.transAxes,
    )

    header_rect = Rectangle(
        (0, header_y - row_height * 0.45),
        table_width,
        row_height * 0.9,
        transform=ax.transAxes,
        facecolor="#4A90D9",
        edgecolor="none",
        clip_on=False,
        zorder=5,
    )
    ax.add_patch(header_rect)

    admit_bg = "#D6EAF8"
    for i, r in enumerate(display_rows):
        y = header_y - row_height * (i + 1.05)

        if r["_separator"]:
            ax.text(
                table_width / 2,
                y,
                "...",
                ha="center",
                va="center",
                fontsize=fontsize_cell + 2,
                color="0.4",
                transform=ax.transAxes,
            )
            continue

        below_threshold = r["rank"] > K
        if r["is_admit"]:
            bg_color = "#C8DCF0" if below_threshold else admit_bg
        elif below_threshold:
            bg_color = "#F5F5F0"
        else:
            bg_color = "#F9F9F9" if i % 2 == 0 else "white"

        rect = Rectangle(
            (0, y - row_height * 0.45),
            table_width,
            row_height * 0.9,
            transform=ax.transAxes,
            facecolor=bg_color,
            edgecolor="#E0E0E0",
            linewidth=0.5,
            clip_on=False,
        )
        ax.add_patch(rect)

        if r["is_admit"]:
            indicator = Rectangle(
                (0, y - row_height * 0.45),
                0.005,
                row_height * 0.9,
                transform=ax.transAxes,
                facecolor="#005FA3",
                edgecolor="none",
                clip_on=False,
            )
            ax.add_patch(indicator)

        values = [
            str(r["rank"]),
            r["delta"],
            str(r["age"]) if r["age"] != "" else "",
            None,
            r["chief_complaint"],
        ]

        esi_col_idx = 3
        for j, val in enumerate(values):
            if j == esi_col_idx:
                esi_num = r["esi_num"]
                if esi_num is not None:
                    pill_x = col_x[j] + left_pad + 0.012
                    pill_color = ESI_COLORS.get(esi_num, "#BDBDBD")
                    pill = FancyBboxPatch(
                        (pill_x - 0.012, y - row_height * 0.25),
                        0.024,
                        row_height * 0.5,
                        boxstyle="round,pad=0.003",
                        transform=ax.transAxes,
                        facecolor=pill_color,
                        edgecolor="none",
                        alpha=0.7,
                        clip_on=False,
                    )
                    ax.add_patch(pill)
                    ax.text(
                        pill_x,
                        y,
                        str(esi_num),
                        ha="center",
                        va="center",
                        fontsize=fontsize_cell,
                        fontweight="bold",
                        color="0.15",
                        transform=ax.transAxes,
                    )
                continue

            if j == 0:
                x = col_x[j] + col_widths[j] * 0.65
                ha = "center"
            elif j == 1:
                x = col_x[j] + col_widths[j] / 2
                ha = "center"
            else:
                x = col_x[j] + left_pad
                ha = "left"

            fontweight = "bold" if j == 0 else "normal"
            if j == 1:
                if r["delta_val"] > 0:
                    color = "#2E7D32"
                elif r["delta_val"] < 0:
                    color = "#C62828"
                else:
                    color = "0.5"
            else:
                color = "0.15"

            ax.text(
                x,
                y,
                val,
                ha=ha,
                va="center",
                fontsize=fontsize_cell,
                fontweight=fontweight,
                color=color,
                transform=ax.transAxes,
            )

        if r["description"]:
            dash_x = table_width - 0.005
            ax.plot(
                [dash_x, dash_x + 0.02],
                [y, y],
                color="0.5",
                linewidth=0.8,
                transform=ax.transAxes,
                clip_on=False,
            )
            ax.text(
                desc_x,
                y,
                r["description"],
                ha="left",
                va="center",
                fontsize=fontsize_cell,
                color="black",
                style="italic",
                transform=ax.transAxes,
            )


def render_resource_table(ax, resource_sim_df):
    ax.axis("off")

    metric_labels = {
        "admit": "Admission rate",
        "n_meds": "Medications",
        "n_procedures": "Procedures",
        "n_labs": "Lab orders",
        "max_t": "Total LOS (min)",
    }
    metric_order = ["admit", "n_meds", "n_procedures", "n_labs", "max_t"]

    col_x = [0.02, 0.42, 0.67, 0.88]
    col_headers = ["Metric", f"Top {K}", f"Bottom {N_ADMIT + N_DISCHARGE - K}", "p-value"]
    n_rows = len(metric_order)
    row_height = 1.0 / (n_rows + 2.5)
    header_y = 1.0 - row_height * 0.6
    fs_hdr = 9.5
    fs_data = 8.5

    for j, hdr in enumerate(col_headers):
        ha = "left" if j == 0 else "center"
        ax.text(
            col_x[j],
            header_y,
            hdr,
            transform=ax.transAxes,
            fontsize=fs_hdr,
            fontweight="bold",
            ha=ha,
            va="center",
        )

    line_y = header_y - row_height * 0.5
    ax.plot([0.01, 0.99], [line_y, line_y], color="0.4", linewidth=0.5, transform=ax.transAxes, clip_on=False)

    for i, metric in enumerate(metric_order):
        y = header_y - row_height * (i + 1.2)
        sub = resource_sim_df[resource_sim_df.metric == metric]
        top_vals = sub["top_mean"].values
        bot_vals = sub["bot_mean"].values

        top_median = np.median(top_vals)
        bot_median = np.median(bot_vals)
        top_q1, top_q3 = np.percentile(top_vals, [25, 75])
        bot_q1, bot_q3 = np.percentile(bot_vals, [25, 75])

        diffs = top_vals - bot_vals
        if len(diffs) == 0 or np.allclose(diffs, 0):
            p_val = 1.0
        else:
            _, p_val = wilcoxon(diffs, alternative="two-sided")

        if metric == "admit":
            top_str = f"{top_median:.0%} [{top_q1:.0%}-{top_q3:.0%}]"
            bot_str = f"{bot_median:.0%} [{bot_q1:.0%}-{bot_q3:.0%}]"
        elif metric == "max_t":
            top_str = f"{top_median:.0f} [{top_q1:.0f}-{top_q3:.0f}]"
            bot_str = f"{bot_median:.0f} [{bot_q1:.0f}-{bot_q3:.0f}]"
        else:
            top_str = f"{top_median:.1f} [{top_q1:.1f}-{top_q3:.1f}]"
            bot_str = f"{bot_median:.1f} [{bot_q1:.1f}-{bot_q3:.1f}]"

        if p_val < 0.001:
            p_str = "<0.001"
        elif p_val < 0.01:
            p_str = "<0.01"
        else:
            p_str = f"{p_val:.3f}"

        ax.text(
            col_x[0],
            y,
            metric_labels[metric],
            transform=ax.transAxes,
            fontsize=fs_data,
            fontweight="bold",
            ha="left",
            va="center",
        )
        ax.text(
            col_x[1],
            y,
            top_str,
            transform=ax.transAxes,
            fontsize=fs_data,
            ha="center",
            va="center",
        )
        ax.text(
            col_x[2],
            y,
            bot_str,
            transform=ax.transAxes,
            fontsize=fs_data,
            ha="center",
            va="center",
        )
        ax.text(
            col_x[3],
            y,
            p_str,
            transform=ax.transAxes,
            fontsize=fs_data,
            ha="center",
            va="center",
            fontweight="bold" if p_val < 0.05 else "normal",
        )


def _shap_step(case, debug=False):
    if not debug:
        return case["step"]
    return max(case["step"], 5)


def _ensure_shap_cache(case, debug=False, use_cdf=False):
    step = _shap_step(case, debug=debug)
    suffix = "_debug" if debug else ""
    mode = "cdf" if use_cdf else "softmax"
    cache_key = f"{case['encounter_key']}_step{step}_max{case['t_max']}_{mode}_t0_logratio{suffix}.pkl"
    cache_path = CACHE_DIR / cache_key
    if cache_path.exists():
        return cache_path

    print(f"Computing SHAP cache for {case['encounter_key']}...")
    test = pd.read_feather(DATA_DIR / "test.feather")
    model, _ = load_model(CHECKPOINT)
    name_to_id = {k: v + 1 for k, v in zip(VOCAB["name"], VOCAB["index"])}

    enc_df = test[test.encounter_key == case["encounter_key"]].sort_values("t")
    enc_df = get_ed_only_df(enc_df)
    age_row = enc_df[enc_df["name"].str.startswith("Age_")]
    if len(age_row) == 0:
        raise ValueError(f"No age token found for {case['encounter_key']}")
    age = int(age_row["name"].iloc[0].removeprefix("Age_"))

    task = build_disposition_task()
    age_to_pos_id, age_to_neg_id = task.label.get_age_to_token_ids()
    pos_ids = age_to_pos_id[age]
    neg_ids = age_to_neg_id[age]
    baseline_score = compute_baseline_log_score(
        model, enc_df, name_to_id, pos_ids, neg_ids, use_cdf=use_cdf
    )

    eval_times_all = list(range(0, case["t_max"] + 1, step))
    eval_times = []
    shap_results = []
    for eval_t in tqdm(eval_times_all, desc=f"SHAP {case['encounter_key']}"):
        result = compute_shap_at_time(
            model,
            enc_df,
            name_to_id,
            pos_ids,
            neg_ids,
            eval_t,
            baseline_score,
            use_cdf=use_cdf,
        )
        if result is None:
            continue
        eval_times.append(eval_t)
        shap_results.append(result)

    with open(cache_path, "wb") as f:
        pickle.dump({"eval_times": eval_times, "shap_results": shap_results}, f)
    print(f"Saved {cache_path}")
    return cache_path


def _load_shap_cache(case, debug=False, use_cdf=False):
    cache_path = _ensure_shap_cache(case, debug=debug, use_cdf=use_cdf)
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def build_composite(debug=False, use_cdf=False):
    _ensure_data_stage()
    composite_case = next(case for case in SHAP_CASES if case["destination"] == "composite")
    cached = _load_shap_cache(composite_case, debug=debug, use_cdf=use_cdf)

    apply_nature_style()
    fig = plt.figure(figsize=(15, 9.5))
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[0.7, 1.4],
        height_ratios=[1.3, 1],
        hspace=0.08,
        wspace=0.08,
    )

    ax_bump = fig.add_subplot(gs[:, 0])
    bump_pos = ax_bump.get_position()
    table_h = 0.10
    gap = 0.06
    ax_bump.set_position(
        [
            bump_pos.x0,
            bump_pos.y0 + table_h + gap,
            bump_pos.width,
            bump_pos.height - table_h - gap,
        ]
    )
    table_y = bump_pos.y0 - 0.015
    ax_table = fig.add_axes([bump_pos.x0, table_y, bump_pos.width, table_h])
    resource_sim_df = pd.read_csv(RESOURCE_SIM_CACHE)
    render_resource_table(ax_table, resource_sim_df)
    ax_table.text(
        -0.05,
        1.02,
        "c",
        transform=ax_table.transAxes,
        fontsize=14,
        fontweight="bold",
        va="bottom",
        ha="right",
    )
    plot_dynamic_ranks(ax=ax_bump, t_max=T_MAX_BUMP, smooth=False)

    ax_board = fig.add_subplot(gs[0, 1])
    ax_board.axis("off")
    render_trackboard(ax_board, t_max=T_MAX_BUMP, rank_ranges=PANEL_B_RANKS)

    ax_shap = fig.add_subplot(gs[1, 1])
    plot_stacked_shap(
        cached["shap_results"],
        cached["eval_times"],
        encounter_key=composite_case["encounter_key"],
        rank=5,
        is_admitted=True,
        output_path=None,
        use_cdf=use_cdf,
        manual_vlines=composite_case["vlines"],
        clean_config=CLEAN_MODES.get(composite_case.get("clean") or ""),
        ax=ax_shap,
    )

    ax_board.text(
        -0.02,
        1.02,
        "b",
        transform=ax_board.transAxes,
        fontsize=14,
        fontweight="bold",
        va="bottom",
        ha="right",
    )
    ax_shap.text(
        -0.05,
        1.05,
        "d",
        transform=ax_shap.transAxes,
        fontsize=14,
        fontweight="bold",
        va="bottom",
        ha="right",
    )
    b_fig_y = ax_board.get_position().y1 + 0.01
    a_trans = blended_transform_factory(ax_bump.transAxes, fig.transFigure)
    ax_bump.text(
        -0.05,
        b_fig_y,
        "a",
        transform=a_trans,
        fontsize=14,
        fontweight="bold",
        va="bottom",
        ha="right",
    )

    pos = ax_shap.get_position()
    ax_shap.set_position([pos.x0 + 0.015, pos.y0, pos.width - 0.015, pos.height])

    out_base = SAVE_DIR / "fig_dynamic_composite"
    fig.savefig(out_base.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    fig.savefig(out_base.with_suffix(".pdf"), dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved {out_base.with_suffix('.png')}")
    print(f"Saved {out_base.with_suffix('.pdf')}")


def build_additional_shap(debug=False, use_cdf=False):
    supplement_cases = [case for case in SHAP_CASES if case["destination"] != "composite"]
    if not supplement_cases:
        print("No supplementary SHAP cases configured; skipping.")
        return

    cached_cases = [(case, _load_shap_cache(case, debug=debug, use_cdf=use_cdf)) for case in supplement_cases]

    apply_nature_style()
    fig, axes = plt.subplots(
        len(cached_cases),
        1,
        figsize=(10, 3.8 * len(cached_cases)),
        squeeze=False,
        gridspec_kw={"hspace": 0.35},
    )

    for idx, (case, cached) in enumerate(cached_cases):
        ax = axes[idx, 0]
        plot_stacked_shap(
            cached["shap_results"],
            cached["eval_times"],
            encounter_key=case["encounter_key"],
            rank=0,
            is_admitted=False,
            output_path=None,
            use_cdf=use_cdf,
            manual_vlines=case["vlines"],
            clean_config=CLEAN_MODES.get(case.get("clean") or ""),
            ax=ax,
        )
        display_title = case.get("title") or case["encounter_key"]
        ax.set_title(display_title, loc="left", fontsize=11, fontweight="bold")
        ax.text(
            -0.03,
            1.05,
            ascii_lowercase[idx],
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="bottom",
            ha="right",
        )

    out_base = SAVE_DIR / "additional_shap"
    fig.savefig(out_base.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    fig.savefig(out_base.with_suffix(".pdf"), dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved {out_base.with_suffix('.png')}")
    print(f"Saved {out_base.with_suffix('.pdf')}")


def main():
    parser = argparse.ArgumentParser(description="Dynamic patient reprioritization v2")
    parser.add_argument(
        "--stage",
        choices=["all", "data", "figures"],
        default="all",
        help="Run data generation, figure generation, or both",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use a coarser SHAP grid to speed up figure generation",
    )
    parser.add_argument(
        "--cdf",
        dest="use_cdf",
        action="store_true",
        help="Use the CDF estimator for SHAP only (default: on)",
    )
    parser.add_argument(
        "--softmax",
        dest="use_cdf",
        action="store_false",
        help="Use softmax_binary for SHAP instead of CDF",
    )
    parser.set_defaults(use_cdf=True)
    args = parser.parse_args()

    if args.stage in {"all", "data"}:
        _ensure_data_stage()

    if args.stage in {"all", "figures"}:
        build_composite(debug=args.debug, use_cdf=args.use_cdf)
        build_additional_shap(debug=args.debug, use_cdf=args.use_cdf)


if __name__ == "__main__":
    main()
