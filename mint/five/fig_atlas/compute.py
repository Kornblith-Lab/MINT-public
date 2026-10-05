"""Compute pipeline for fig_atlas: clustering, UMAP, PEWS scoring.

Temporal split:
  - 2024 encounters → development (UMAP, clustering, Panel B)
  - 2025 encounters → held-out evaluation (Panel D)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from umap import UMAP

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[3]
REP_PATH = REPO_ROOT / "artifacts/fig_umap/analysis/representations/test_t512tok.npz"
DATA_DIR = REPO_ROOT / "output"
PEWS_CSV = REPO_ROOT / "cdw/finalized_cohorts/pews_predictors.csv"
NOTES_CSV = REPO_ROOT / "cdw/finalized_cohorts/all_pediatric_ed_visits_with_note.csv"
SAVE_DIR = REPO_ROOT / "artifacts/fig_atlas"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

UMAP_NEIGHBORS = 30
UMAP_MIN_DIST = 0.3
RANDOM_STATE = 42
MAX_TIME_PEWS = 30  # minutes — PEWS is a triage tool
DISPOSITION_TOKENS = {"Admit", "ICU Start", "Discharge"}


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_representations():
    """Load 512-token representations, filter NaN rows."""
    data = np.load(REP_PATH, allow_pickle=True)
    keys = data["encounter_keys"]
    reps = data["last_layer"]
    valid = ~np.isnan(reps[:, 0])
    return keys[valid], reps[valid]


def load_encounter_dates(cohort_keys):
    """Load service dates for encounters, return {enc_key: year}."""
    visits = pd.read_csv(
        NOTES_CSV, usecols=["EncounterKey", "deid_service_date"],
    )
    visits = visits[visits["EncounterKey"].isin(cohort_keys)]
    visits["date"] = pd.to_datetime(visits["deid_service_date"], errors="coerce")
    visits["year"] = visits["date"].dt.year
    return visits.set_index("EncounterKey")["year"].to_dict()


def temporal_split(keys, reps, year_map):
    """Split into 2024 (development) and 2025 (evaluation) cohorts."""
    dev_mask = np.array([year_map.get(k, 0) == 2024 for k in keys])
    eval_mask = np.array([year_map.get(k, 0) == 2025 for k in keys])
    return (keys[dev_mask], reps[dev_mask]), (keys[eval_mask], reps[eval_mask])


def load_trajectories(cohort_keys: set):
    """Load test.feather filtered to cohort encounters."""
    df = pd.read_feather(DATA_DIR / "test.feather")
    return df[df["encounter_key"].isin(cohort_keys)].copy()


def load_pews_predictors(cohort_keys: set):
    """Load note-derived PEWS predictors (WOB, consciousness, cap refill)."""
    cols = ["EncounterKey", "Age", "work_of_breathing",
            "decreased_level_of_conciousness", "increased_capillary_refill"]
    pews = pd.read_csv(PEWS_CSV, usecols=cols)
    pews = pews[pews["EncounterKey"].isin(cohort_keys)]
    pews["wob"] = (pews["work_of_breathing"] == "Yes").astype(int)
    pews["consciousness"] = (pews["decreased_level_of_conciousness"] == "Yes").astype(int)
    pews["cap_refill"] = (pews["increased_capillary_refill"] == "Yes").astype(int)
    return pews.set_index("EncounterKey")


# ──────────────────────────────────────────────────────────────────────────────
# ED-PEWS scoring (standard rule)
# ──────────────────────────────────────────────────────────────────────────────

def pews_age_score(age: int) -> int:
    if age <= 4:
        return 0
    elif age <= 11:
        return 4
    elif age <= 15:
        return 6
    else:
        return 6


def pews_rr_score(rr: int) -> int:
    if rr < 30:
        return 0
    elif rr < 40:
        return 3
    elif rr < 60:
        return 5
    else:
        return 9


def pews_spo2_score(spo2: int) -> int:
    if spo2 >= 98:
        return 0
    elif spo2 >= 94:
        return 4
    elif spo2 >= 88:
        return 9
    else:
        return 15


def pews_hr_score(hr: int) -> int:
    if hr < 100:
        return 0
    elif hr < 140:
        return 3
    elif hr < 180:
        return 6
    else:
        return 9


def compute_pews_total(age, consciousness, wob, rr, spo2, hr, cap_refill):
    score = pews_age_score(age)
    score += 14 if consciousness else 0
    score += 12 if wob else 0
    score += pews_rr_score(rr) if rr is not None else 0
    score += pews_spo2_score(spo2) if spo2 is not None else 0
    score += pews_hr_score(hr) if hr is not None else 0
    score += 3 if cap_refill else 0
    return score


# ──────────────────────────────────────────────────────────────────────────────
# Adjusted ED-PEWS scoring (Options B + C)
#   B: WOB 12→4 if SpO2 >= 94 (compensated respiratory distress)
#   C: Age-adjusted HR thresholds for age <= 4 (shifted up by 20)
# ──────────────────────────────────────────────────────────────────────────────

def pews_hr_score_young(hr: int) -> int:
    """HR scoring for age <= 4: thresholds shifted up by 20."""
    if hr < 120:
        return 0
    elif hr < 160:
        return 3
    elif hr < 200:
        return 6
    else:
        return 9


TRANSFER_TOKENS = {
    "Arrival_Method_Ground Ambulance Interfacility Transfer",
    "Arrival_Method_Fixed Wing",
    "Arrival_Method_Rotor Wing",
}
TRANSFER_BONUS = 5


def compute_adjusted_pews(age, consciousness, wob, rr, spo2, hr, cap_refill,
                          has_transfer=False):
    score = pews_age_score(age)
    score += 14 if consciousness else 0

    # Option B: conditional WOB reduction
    if wob:
        if spo2 is not None and spo2 >= 94:
            score += 4  # compensated: reduced from 12
        else:
            score += 12  # decompensated or unknown: full weight

    score += pews_rr_score(rr) if rr is not None else 0
    score += pews_spo2_score(spo2) if spo2 is not None else 0

    # Option C: age-adjusted HR for young children
    if hr is not None:
        if age <= 4:
            score += pews_hr_score_young(hr)
        else:
            score += pews_hr_score(hr)

    score += 3 if cap_refill else 0

    # Option D: transfer arrival bonus
    if has_transfer:
        score += TRANSFER_BONUS

    return score


# ──────────────────────────────────────────────────────────────────────────────
# Vital extraction (first 30 min, before disposition)
# ──────────────────────────────────────────────────────────────────────────────

def extract_encounter_features(traj_df):
    """Extract vitals at t<=30 (before disposition) and outcome labels.

    Returns:
        vitals: {enc_key: (hr, rr, spo2)} — worst-case vitals in first 30 min
        enc_flags: {enc_key: {has_admit, has_icu, trajectory, ...}}
        complete_keys: set of encounter keys that have HR, RR, AND SpO2
    """
    vitals = {}
    enc_flags = {}
    complete_keys = set()

    for enc_key, grp in traj_df.groupby("encounter_key"):
        names = grp["name"].values
        times = grp["t"].values

        has_icu = "ICU Start" in names
        has_admit = "Admit" in names

        # Find first disposition time
        disp_time = np.inf
        for n, t in zip(names, times):
            if n in DISPOSITION_TOKENS:
                disp_time = t
                break

        effective_cutoff = min(MAX_TIME_PEWS, disp_time - 0.001)

        best_hr, best_rr, best_spo2 = None, None, None

        for name, t in zip(names, times):
            if t > effective_cutoff:
                continue
            if name in DISPOSITION_TOKENS:
                continue

            if name.startswith("Vital_Pulse_"):
                val = int(name.split("_")[-1])
                if best_hr is None or val > best_hr:
                    best_hr = val
            elif name.startswith("Vital_Resp_"):
                val = int(name.split("_")[-1])
                if best_rr is None or val > best_rr:
                    best_rr = val
            elif name.startswith("Vital_SpO2_"):
                val = int(name.split("_")[-1])
                if best_spo2 is None or val < best_spo2:
                    best_spo2 = val

        vitals[enc_key] = (best_hr, best_rr, best_spo2)

        # Build trajectory string (first 30 min, for LLM naming)
        mask = (times <= effective_cutoff)
        traj_tokens = names[mask]
        traj_str = " -> ".join(traj_tokens[:200])

        has_transfer = bool(set(names) & TRANSFER_TOKENS)

        enc_flags[enc_key] = {
            "has_admit": has_admit,
            "has_icu": has_icu,
            "has_transfer": has_transfer,
            "trajectory": traj_str,
        }

        if best_hr is not None and best_rr is not None and best_spo2 is not None:
            complete_keys.add(enc_key)

    return vitals, enc_flags, complete_keys


# ──────────────────────────────────────────────────────────────────────────────
# Clustering
# ──────────────────────────────────────────────────────────────────────────────

N_CLUSTERS = 50


def cluster_representations(reps):
    """KMeans clustering on the full 120-dim representation space."""
    km = MiniBatchKMeans(
        n_clusters=N_CLUSTERS, random_state=RANDOM_STATE,
        batch_size=4096, n_init=10,
    )
    labels = km.fit_predict(reps)
    sizes = np.bincount(labels)
    print(f"  {N_CLUSTERS} clusters: min={sizes.min()}, max={sizes.max()}, "
          f"median={int(np.median(sizes))}")
    return labels, km


# ──────────────────────────────────────────────────────────────────────────────
# UMAP
# ──────────────────────────────────────────────────────────────────────────────

def compute_umap(reps):
    reducer = UMAP(
        n_neighbors=UMAP_NEIGHBORS, min_dist=UMAP_MIN_DIST,
        n_components=2, random_state=RANDOM_STATE, metric="cosine"
    )
    return reducer.fit_transform(reps)


# ──────────────────────────────────────────────────────────────────────────────
# Score encounters (original + adjusted)
# ──────────────────────────────────────────────────────────────────────────────

def compute_all_pews_scores(keys, pews_lookup, vitals, complete_keys,
                            enc_flags=None, adjusted=False):
    """Compute ED-PEWS for encounters with complete vitals at 30 min."""
    scores = {}
    for enc_key in keys:
        if enc_key not in complete_keys:
            scores[enc_key] = np.nan
            continue
        if enc_key not in pews_lookup.index:
            scores[enc_key] = np.nan
            continue
        row = pews_lookup.loc[enc_key]
        hr, rr, spo2 = vitals[enc_key]
        if adjusted:
            has_transfer = enc_flags.get(enc_key, {}).get("has_transfer", False) if enc_flags else False
            scores[enc_key] = compute_adjusted_pews(
                age=int(row["Age"]),
                consciousness=bool(row["consciousness"]),
                wob=bool(row["wob"]),
                rr=rr, spo2=spo2, hr=hr,
                cap_refill=bool(row["cap_refill"]),
                has_transfer=has_transfer,
            )
        else:
            scores[enc_key] = compute_pews_total(
                age=int(row["Age"]),
                consciousness=bool(row["consciousness"]),
                wob=bool(row["wob"]),
                rr=rr, spo2=spo2, hr=hr,
                cap_refill=bool(row["cap_refill"]),
            )
    return scores


# ──────────────────────────────────────────────────────────────────────────────
# Cluster statistics
# ──────────────────────────────────────────────────────────────────────────────

def build_cluster_stats(keys, cluster_labels, enc_flags, pews_scores, pews_lookup):
    """Build per-cluster summary table."""
    key_to_cluster = dict(zip(keys, cluster_labels))
    n_clusters = cluster_labels.max() + 1

    rows = []
    for cid in range(n_clusters):
        cluster_keys = [k for k in keys if key_to_cluster[k] == cid]
        n = len(cluster_keys)
        if n == 0:
            continue

        admit_count = sum(
            1 for k in cluster_keys
            if enc_flags.get(k, {}).get("has_admit", False)
        )
        icu_count = sum(
            1 for k in cluster_keys
            if enc_flags.get(k, {}).get("has_icu", False)
        )

        pews_vals = [pews_scores.get(k, np.nan) for k in cluster_keys]
        valid_pews = [v for v in pews_vals if not np.isnan(v)]

        ages = []
        for k in cluster_keys:
            if k in pews_lookup.index:
                ages.append(int(pews_lookup.loc[k, "Age"]))
        ages = np.array(ages) if ages else np.array([0])

        row = {
            "cluster_id": cid,
            "N": n,
            "admit_rate": admit_count / n,
            "icu_rate": icu_count / n,
            "mean_pews": np.nanmean(pews_vals) if valid_pews else np.nan,
            "median_age": float(np.median(ages)),
            "age_iqr": f"{np.percentile(ages, 25):.0f}-{np.percentile(ages, 75):.0f}",
            "n_with_pews": len(valid_pews),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def select_exemplars_by_centroid(reps, keys, cluster_labels, n_exemplars=10):
    """Select n_exemplars closest to centroid for each cluster."""
    n_clusters = cluster_labels.max() + 1
    exemplar_map = {}

    for cid in range(n_clusters):
        mask = cluster_labels == cid
        if not mask.any():
            exemplar_map[cid] = []
            continue

        cluster_reps = reps[mask]
        cluster_keys = keys[mask]
        centroid = cluster_reps.mean(axis=0)

        dists = np.linalg.norm(cluster_reps - centroid, axis=1)
        n_select = min(n_exemplars, len(cluster_keys))
        top_idx = np.argsort(dists)[:n_select]
        exemplar_map[cid] = cluster_keys[top_idx].tolist()

    return exemplar_map


# ──────────────────────────────────────────────────────────────────────────────
# Main compute pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_compute():
    print("=" * 60)
    print("Fig Atlas: Compute Pipeline")
    print("=" * 60)

    print("\n1. Loading representations...")
    keys, reps = load_representations()
    cohort_keys = set(keys)
    print(f"   Full test cohort: {len(keys)} encounters, {reps.shape[1]}-dim")

    print("\n2. Loading encounter dates for temporal split...")
    year_map = load_encounter_dates(cohort_keys)
    (dev_keys, dev_reps), (eval_keys, eval_reps) = temporal_split(keys, reps, year_map)
    print(f"   2024 (development): {len(dev_keys)}")
    print(f"   2025 (evaluation):  {len(eval_keys)}")

    # ── Development set (2024): clustering + UMAP + Panel B ──
    print("\n3. Clustering development set (KMeans, k=50)...")
    dev_labels, km_model = cluster_representations(dev_reps)
    np.savez(SAVE_DIR / "clusters.npz",
             keys=dev_keys, labels=dev_labels)

    print("\n4. Computing UMAP on development set...")
    embedding = compute_umap(dev_reps)
    np.savez(SAVE_DIR / "umap.npz", embedding=embedding)
    print(f"   UMAP shape: {embedding.shape}")

    print("\n5. Loading trajectory data (development)...")
    dev_cohort = set(dev_keys)
    traj_df_dev = load_trajectories(dev_cohort)

    print("\n6. Extracting encounter features (first 30 min, development)...")
    vitals_dev, enc_flags_dev, complete_keys_dev = extract_encounter_features(traj_df_dev)
    print(f"   Complete vitals at 30 min: {len(complete_keys_dev)}/{len(dev_keys)}")

    print("\n7. Loading PEWS predictors (development)...")
    pews_lookup_dev = load_pews_predictors(dev_cohort)
    print(f"   Matched PEWS records: {len(pews_lookup_dev)}")

    print("\n8. Computing ED-PEWS scores (development, original)...")
    pews_scores_dev = compute_all_pews_scores(
        dev_keys, pews_lookup_dev, vitals_dev, complete_keys_dev,
        enc_flags=enc_flags_dev, adjusted=False)
    valid_dev = {k: v for k, v in pews_scores_dev.items() if not np.isnan(v)}
    print(f"   Valid PEWS scores: {len(valid_dev)}/{len(dev_keys)}")

    print("\n9. Selecting exemplars (development)...")
    exemplar_map = select_exemplars_by_centroid(dev_reps, dev_keys, dev_labels, n_exemplars=10)
    np.savez(SAVE_DIR / "exemplars.npz",
             **{f"cluster_{cid}": np.array(ks) for cid, ks in exemplar_map.items()})

    print("\n10. Building cluster statistics (development)...")
    cluster_df = build_cluster_stats(
        dev_keys, dev_labels, enc_flags_dev, pews_scores_dev, pews_lookup_dev
    )
    cluster_df.to_csv(SAVE_DIR / "cluster_stats.csv", index=False)
    print(f"   Saved cluster_stats.csv ({len(cluster_df)} clusters)")

    # Save development encounter-level data
    enc_df_dev = pd.DataFrame({
        "encounter_key": dev_keys,
        "cluster_id": dev_labels,
        "pews_score": [pews_scores_dev.get(k, np.nan) for k in dev_keys],
        "has_admit": [enc_flags_dev.get(k, {}).get("has_admit", False) for k in dev_keys],
        "has_icu": [enc_flags_dev.get(k, {}).get("has_icu", False) for k in dev_keys],
        "has_complete_vitals": [k in complete_keys_dev for k in dev_keys],
    })
    enc_df_dev.to_parquet(SAVE_DIR / "encounters_dev.parquet", index=False)

    # Reference table
    ref_cols = ["cluster_id", "N", "admit_rate", "icu_rate", "mean_pews",
                "median_age", "age_iqr", "n_with_pews"]
    ref_table = cluster_df[ref_cols].copy()
    ref_table.to_csv(SAVE_DIR / "reference_table.csv", index=False)

    # ── Evaluation set (2025): compute original + adjusted scores for Panel D ──
    print("\n" + "=" * 60)
    print("Evaluation Set (2025)")
    print("=" * 60)

    print("\n11. Loading trajectory data (evaluation)...")
    eval_cohort = set(eval_keys)
    traj_df_eval = load_trajectories(eval_cohort)

    print("\n12. Extracting encounter features (evaluation)...")
    vitals_eval, enc_flags_eval, complete_keys_eval = extract_encounter_features(traj_df_eval)
    print(f"   Complete vitals at 30 min: {len(complete_keys_eval)}/{len(eval_keys)}")

    print("\n13. Loading PEWS predictors (evaluation)...")
    pews_lookup_eval = load_pews_predictors(eval_cohort)
    print(f"   Matched PEWS records: {len(pews_lookup_eval)}")

    print("\n14. Computing original ED-PEWS scores (evaluation)...")
    pews_orig_eval = compute_all_pews_scores(
        eval_keys, pews_lookup_eval, vitals_eval, complete_keys_eval,
        enc_flags=enc_flags_eval, adjusted=False)
    valid_orig = {k: v for k, v in pews_orig_eval.items() if not np.isnan(v)}
    print(f"   Valid original PEWS: {len(valid_orig)}/{len(eval_keys)}")

    print("\n15. Computing adjusted ED-PEWS scores (evaluation)...")
    pews_adj_eval = compute_all_pews_scores(
        eval_keys, pews_lookup_eval, vitals_eval, complete_keys_eval,
        enc_flags=enc_flags_eval, adjusted=True)
    valid_adj = {k: v for k, v in pews_adj_eval.items() if not np.isnan(v)}
    print(f"   Valid adjusted PEWS: {len(valid_adj)}/{len(eval_keys)}")

    # Save evaluation encounter data
    enc_df_eval = pd.DataFrame({
        "encounter_key": eval_keys,
        "pews_original": [pews_orig_eval.get(k, np.nan) for k in eval_keys],
        "pews_adjusted": [pews_adj_eval.get(k, np.nan) for k in eval_keys],
        "has_admit": [enc_flags_eval.get(k, {}).get("has_admit", False) for k in eval_keys],
        "has_icu": [enc_flags_eval.get(k, {}).get("has_icu", False) for k in eval_keys],
        "has_complete_vitals": [k in complete_keys_eval for k in eval_keys],
    })
    enc_df_eval.to_parquet(SAVE_DIR / "encounters_eval.parquet", index=False)
    print(f"\n   Saved encounters_eval.parquet ({len(enc_df_eval)} rows)")

    # Also assign 2025 encounters to clusters using the fitted KMeans model
    eval_cluster_labels = km_model.predict(eval_reps)
    np.savez(SAVE_DIR / "clusters_eval.npz",
             keys=eval_keys, labels=eval_cluster_labels)

    print("\n" + "=" * 60)
    print("Compute complete.")
    print("  Development (2024): clusters.npz, umap.npz, encounters_dev.parquet")
    print("  Evaluation (2025): encounters_eval.parquet")
    print("=" * 60)
