"""LLM-based cluster naming using trajectory + triage note exemplars.

Uses Claude (Sonnet) via subagents to generate clinically sensible names
for each cluster based on 10 exemplar cases per cluster.
"""

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SAVE_DIR = REPO_ROOT / "artifacts/fig_atlas"
NOTES_CSV = REPO_ROOT / "cdw/finalized_cohorts/all_pediatric_ed_visits_with_note.csv"
DATA_DIR = REPO_ROOT / "output"


def is_note_valid(text: str):
    """Validate note and return (start, end) indices of triage section, or False."""
    text_joined = ' '.join(text.split())
    text_lower = text_joined.lower()
    text_upper = text_joined

    cc = text_lower.find("chief complaint")
    hpi = text_lower.find("hpi")
    cc_star = text_lower.find("***** complaint")
    cc_only = text_lower.find("cc:")
    history = text_lower.find("history")
    options = [x for x in [cc, hpi, cc_star, cc_only, history] if x != -1]
    if not options:
        return False

    start = min(options)

    end = text_lower.find("% physical exam", start)
    if end != -1:
        return (start, end + 1)

    ros = text_lower.find("review of systems", start)
    ros = ros if ros != -1 else text_upper.find("ROS", start)
    if ros == -1:
        return False

    triage_vital_signs = text_lower.find("triage vital signs", ros)
    triage_vital_signs = triage_vital_signs if triage_vital_signs != -1 else text_lower.find("vital signs", ros)

    if triage_vital_signs != -1:
        end = text_lower.find("%", triage_vital_signs)
        if end == -1:
            resp = text_lower.find("resp:", triage_vital_signs)
            if resp != -1:
                end = text_lower.find("physical exam", resp) - 1
        end = end if end > 0 else text_lower.find("physical exam", triage_vital_signs) - 1
        if end > 0 and (end - triage_vital_signs) < 200:
            return (start, end + 1)

    end = text_lower.find("physical exam", ros)
    if end == -1:
        return False
    return (start, end)


def get_triage_note(note_text: str):
    """Extract triage section of a clinical note."""
    result = is_note_valid(note_text)
    if result is False:
        return None
    start, end = result
    text = ' '.join(note_text.split())
    return text[start:end]


def build_naming_context():
    """Build the context dict for LLM naming: exemplar trajectories + triage notes.

    Returns list of dicts, one per cluster, each containing:
        cluster_id, N, admit_rate, icu_rate, median_age, exemplar_cases (list of dicts)
    """
    cluster_stats = pd.read_csv(SAVE_DIR / "cluster_stats.csv")
    exemplars_data = np.load(SAVE_DIR / "exemplars.npz", allow_pickle=True)
    enc_df = pd.read_parquet(SAVE_DIR / "encounters.parquet")

    # Load trajectories
    traj_df = pd.read_feather(DATA_DIR / "test.feather")
    traj_df = traj_df.sort_values(["encounter_key", "t"])

    # Load notes
    print("  Loading notes for exemplar cases...")
    all_exemplar_keys = set()
    for cid in range(len(cluster_stats)):
        key = f"cluster_{cid}"
        if key in exemplars_data:
            all_exemplar_keys.update(exemplars_data[key].tolist())

    notes_df = pd.read_csv(
        NOTES_CSV, usecols=["EncounterKey", "note_text"],
    )
    notes_df = notes_df[notes_df["EncounterKey"].isin(all_exemplar_keys)]
    notes_map = notes_df.set_index("EncounterKey")["note_text"].to_dict()

    # Build context per cluster
    clusters = []
    for _, row in cluster_stats.iterrows():
        cid = int(row["cluster_id"])
        key = f"cluster_{cid}"
        exemplar_keys = exemplars_data[key].tolist() if key in exemplars_data else []

        cases = []
        for ek in exemplar_keys:
            # Trajectory (first 30 min)
            enc_traj = traj_df[traj_df["encounter_key"] == ek]
            enc_traj_30 = enc_traj[enc_traj["t"] <= 30]
            traj_str = " -> ".join(enc_traj_30["name"].values[:150])

            # Triage note
            raw_note = notes_map.get(ek, "")
            triage = get_triage_note(raw_note) if raw_note else None

            # Outcome
            has_admit = "Admit" in enc_traj["name"].values
            has_icu = "ICU Start" in enc_traj["name"].values

            cases.append({
                "encounter_key": ek,
                "trajectory": traj_str,
                "triage_note": triage or "(not available)",
                "admitted": has_admit,
                "icu": has_icu,
            })

        clusters.append({
            "cluster_id": cid,
            "N": int(row["N"]),
            "admit_rate": float(row["admit_rate"]),
            "icu_rate": float(row["icu_rate"]),
            "median_age": float(row["median_age"]),
            "age_iqr": row["age_iqr"],
            "mean_pews": float(row["mean_pews"]) if not pd.isna(row["mean_pews"]) else None,
            "exemplar_cases": cases,
        })

    return clusters


def format_cluster_prompt(cluster_batch):
    """Format a prompt for naming a batch of clusters."""
    parts = []
    parts.append(
        "You are a pediatric emergency medicine physician. "
        "Below are groups of children who visited the ED, clustered by similarity "
        "of their clinical trajectories (vitals, medications, procedures, chief complaints). "
        "For each cluster, I provide:\n"
        "- Summary statistics (N, admission rate, ICU rate, median age)\n"
        "- 10 exemplar cases with their first-30-minute trajectory tokens and triage notes\n\n"
        "For EACH cluster, provide:\n"
        "1. A short clinical name (3-6 words, e.g., 'Febrile infant workup', "
        "'Adolescent psychiatric crisis', 'Mild asthma exacerbation')\n"
        "2. A one-sentence description of the dominant clinical phenotype\n\n"
        "Format your response as:\n"
        "CLUSTER <id>: <name>\n"
        "Description: <one sentence>\n\n"
        "Be specific and clinically precise. Use the trajectory tokens, chief complaints, "
        "vitals, medications, and triage notes to identify the phenotype."
    )

    for cluster in cluster_batch:
        parts.append(f"\n{'='*60}")
        parts.append(f"CLUSTER {cluster['cluster_id']}")
        parts.append(f"  N={cluster['N']}, Admission rate={cluster['admit_rate']:.1%}, "
                     f"ICU rate={cluster['icu_rate']:.1%}")
        parts.append(f"  Median age={cluster['median_age']:.0f}y "
                     f"(IQR {cluster['age_iqr']})")
        if cluster['mean_pews'] is not None:
            parts.append(f"  Mean ED-PEWS={cluster['mean_pews']:.1f}")
        parts.append("")

        for i, case in enumerate(cluster["exemplar_cases"]):
            parts.append(f"  --- Case {i+1} ---")
            parts.append(f"  Trajectory: {case['trajectory'][:500]}")
            parts.append(f"  Triage note: {case['triage_note'][:800]}")
            outcome = "ICU" if case["icu"] else ("Admitted" if case["admitted"] else "Discharged")
            parts.append(f"  Outcome: {outcome}")
            parts.append("")

    return "\n".join(parts)


def run_naming():
    """Build naming context and save prompt files for workflow execution."""
    print("Building naming context...")
    clusters = build_naming_context()
    print(f"  {len(clusters)} clusters to name")

    # Split into batches of 5
    batch_size = 5
    batches = []
    for i in range(0, len(clusters), batch_size):
        batches.append(clusters[i:i + batch_size])

    # Save prompts for each batch
    prompts_dir = SAVE_DIR / "naming_prompts"
    prompts_dir.mkdir(exist_ok=True)
    for i, batch in enumerate(batches):
        prompt = format_cluster_prompt(batch)
        (prompts_dir / f"batch_{i}.txt").write_text(prompt)

    print(f"  Saved {len(batches)} prompt files to {prompts_dir}")
    print("  Run the naming workflow to generate names.")

    # Also save full context as JSON for the workflow
    import json
    context_path = SAVE_DIR / "naming_context.json"
    with open(context_path, "w") as f:
        json.dump(clusters, f, indent=2, default=str)
    print(f"  Saved naming_context.json ({context_path})")
