"""
Operational prediction with MINT vs XGBoost for ED outcomes.
Predicts Admit, ICU, Sepsis, and Septic Shock using first K minutes of ED data.

  Key Features:

  Outcomes Supported:
  - Admit (60-minute window) - predicts ED discharge vs admission
  - ICU (60-minute window) - predicts ICU transfer for admitted patients
  - Sepsis (240-minute window) - external label CSV (cdw/sepsis.csv)
  - Septic Shock (240-minute window) - external label CSV (cdw/septic_shock.csv)

  Comparison Methods:
  1. MINT Zero-Shot (Softmax) - For Admit/ICU only
    - Uses token-level softmax probabilities of label tokens (Admit, ICU Start)
    - Skipped for sepsis/septic_shock (out-of-distribution)
  2. MINT Linear Probes - For all outcomes
    - Extracts mean-pooled embeddings from MINT model
    - Trains Logistic Regression on embeddings
    - Trains XGBoost on scaled embeddings
  3. XGBoost Baseline - For all outcomes
    - Bag-of-words feature engineering
    - Direct XGBoost training on token counts
  4. ESI (Emergency Severity Index) - For Admit/ICU only
    - Extracts Acuity_* tokens (1-5 severity scale)
    - Inverts scores (1.0/score) for probability interpretation
    - Skips encounters without valid ESI
  5. ED-PEWS (Pediatric Early Warning System) - For Admit/ICU only
    - Computes 0-68 score from age, vitals (HR, RR, SpO2), and predictors
    - Vitals extracted from first K minutes (worst-case values)
    - Predictors loaded from cdw/pews_predictors.csv (work_of_breathing, consciousness, capillary_refill)
    - Skips encounters without all required vitals or PEWS predictor data

  Data Processing:

  - Truncation Rule: First K minutes (configurable per outcome) AND before any cutoff token (Discharge/Admit/ICU Start)
  - Token Count Filter: Minimum 12 tokens (configurable), maximum 512 tokens (configurable)
  - Label Leakage Prevention: Labels computed from raw encounter, predictions only on truncated data

  Output Format:

  Matches fig1.py exactly:
  - CSV files per method: {outcome}_{method}.csv with encounter_key, probabilities, labels
  - Results JSONL: results_operational.jsonl with AUROC, AUPRC, confidence intervals, incidence, n_pos
  - Results CSV: results_operational.csv for easy viewing

  Output Files by Outcome (in save_path directory):

  Admit/ICU:
    - {outcome}_MINT_softmax.csv: MINT zero-shot predictions
    - {outcome}_MINT_probes.csv: MINT LR and XGBoost on embeddings
    - {outcome}_XGBoost_baseline.csv: XGBoost on bag-of-words
    - {outcome}_ESI.csv: ESI (Emergency Severity Index) baseline
    - {outcome}_ED-PEWS.csv: ED-PEWS (Pediatric Early Warning System) baseline

  Sepsis/Septic Shock:
    - {outcome}_MINT_probes.csv: MINT LR and XGBoost on embeddings
    - {outcome}_XGBoost_baseline.csv: XGBoost on bag-of-words

  Summary:
    - results_operational.csv: All metrics in tabular format
    - results_operational.jsonl: All metrics in JSONL format (encounter-level detail)

  Command Line Arguments:

  --save_path ARTIFACTS_DIR           # Output directory (default: artifacts/fig1_operational)
  --checkpoint_path CKPT_PATH         # MINT checkpoint (default: output/mint/ckpt.pt)
  --max_len MAX_TOKENS                # Max token limit (default: 512)
  --min_tokens MIN_TOKENS             # Min token requirement (default: 12)
  --outcomes OUTCOME1 OUTCOME2 ...    # Subset of outcomes to run
  --debug                             # Debug mode with subsampled data

  Example Usage:

  # Go time! Full run (all outcomes with all comparators)
  python -m mint.five.fig_one.fig1_operational \
    --save_path artifacts/fig1_operational \
    --max_len 512 \
    --min_tokens 12

  # Debug run (subsample 1000 encounters, test ESI/ED-PEWS for Admit only)
  python -m mint.five.fig_one.fig1_operational \
    --debug \
    --outcomes admit \
    --save_path artifacts/fig1_operational_test \
    --max_len 512 \
    --min_tokens 12

  # Test Admit and ICU with ESI/ED-PEWS, no other outcomes
  python -m mint.five.fig_one.fig1_operational \
    --outcomes admit icu \
    --save_path artifacts/fig1_operational_clinical_baselines

  Design Choices:

  0. Truncation direction: This file uses events[:max_len] (first N tokens) unlike fig1.py which
     uses events[-max_len:] (last N tokens). This is intentional. Operational evals use only early
     information (first K minutes), so we keep tokens from the start of the encounter. Dynamic/continuous
     evals in fig1.py predict relative to an event and need the most recent context before it.
  1. External label CSVs for sepsis/septic shock - these aren't token-based like Admit/ICU, they come from diagnostic annotations
  2. Mean pooling for embeddings - averages representations across all tokens as requested
  3. Stratified metrics - Bootstrap confidence intervals on AUROC/AUPRC (1000 samples)
  4. Scalable architecture - can easily add new outcomes by updating OUTCOMES dict
  5. ESI baseline - inverts scores (1.0/score) so higher values = higher prediction of admission (matching the model direction)
  6. ED-PEWS baseline - uses raw 0-68 scores directly without normalization for AUROC computation
  7. Both ESI and ED-PEWS only for Admit/ICU - skipped for sepsis/septic_shock (out-of-distribution outcomes)
  8. ED-PEWS filtering - encounters must have:
     - All three vital signs (HR, RR, SpO2) within K-minute window
     - Entry in cdw/pews_predictors.csv with valid predictor flags
  9. Worst-case vitals - HR and RR use maximum values, SpO2 uses minimum value (consistent with clinical severity)

"""

import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from tqdm import tqdm
from tap import tapify
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from mint.five.data import CategoricalLabel
from model import Delphi, DelphiConfig


def _cache_path(cache_dir: Path, name: str) -> Path:
    """Get path for a cache file."""
    return cache_dir / f"{name}.pkl"


def _load_cache(cache_dir: Path, name: str):
    """Load from cache if it exists, else return None."""
    path = _cache_path(cache_dir, name)
    if path.exists():
        print(f"  [cache hit] Loading {name} from {path}")
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def _save_cache(cache_dir: Path, name: str, obj):
    """Save object to cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, name)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  [cache save] Saved {name} to {path}")

DATA_DIR = Path("output")
VOCAB_CSV = DATA_DIR / "vocab.csv"
VOCAB = pd.read_csv(VOCAB_CSV)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {DEVICE}")


@dataclass
class Args:
    save_path: str = "artifacts/fig1_operational"
    checkpoint_path: str = "output/mint/ckpt.pt"
    max_len: int = 512
    min_tokens: int = 12
    debug: bool = False
    outcomes: list[str] = None

    def __post_init__(self):
        if self.outcomes is None:
            self.outcomes = ["admit", "icu", "sepsis", "septic_shock"]


class OperationalConfig:
    """Configuration for operational outcomes."""

    def __init__(self, name: str, label_tokens: list[str], cutoff_tokens: list[str],
                 time_cutoff_min: int, include_zero_shot: bool = True, label_csv_path: Optional[str] = None,
                 include_csv_path: Optional[str] = None, exclude_csv_path: Optional[str] = None):
        self.name = name
        self.label_tokens = label_tokens
        self.cutoff_tokens = cutoff_tokens
        self.time_cutoff_min = time_cutoff_min
        self.include_zero_shot = include_zero_shot  # Whether to include MINT zero-shot probe
        self.label_csv_path = label_csv_path
        self.include_csv_path = include_csv_path
        self.exclude_csv_path = exclude_csv_path
        self._label_set = None
        self._include_set = None
        self._exclude_set = None

        # Load external label CSV if provided
        if label_csv_path:
            label_df = pd.read_csv(label_csv_path)
            self._label_set = set(label_df["EncounterKey"].values)

        # Load inclusion criteria
        if include_csv_path:
            include_df = pd.read_csv(include_csv_path)
            self._include_set = set(include_df["EncounterKey"].values)

        # Load exclusion criteria
        if exclude_csv_path:
            exclude_df = pd.read_csv(exclude_csv_path)
            self._exclude_set = set(exclude_df["EncounterKey"].values)

    def is_eligible(self, encounter_key: str) -> bool:
        """Check if encounter passes inclusion/exclusion criteria."""
        # Must be in inclusion set if specified
        if self._include_set is not None:
            if encounter_key not in self._include_set:
                return False

        # Must not be in exclusion set if specified
        if self._exclude_set is not None:
            if encounter_key in self._exclude_set:
                return False

        return True

    def get_label(self, encounter_key: str, df: pd.DataFrame = None) -> int:
        """Compute binary label from encounter_key or dataframe."""
        # If using external CSV, check membership
        if self._label_set is not None:
            return 1 if encounter_key in self._label_set else 0

        # Otherwise check for label tokens
        if df is not None:
            for token in self.label_tokens:
                if token in df["name"].values:
                    return 1
        return 0

    def truncate_to_cutoff(self, df: pd.DataFrame) -> pd.DataFrame:
        """Truncate to first K minutes or before cutoff token, whichever comes first."""
        # t is already in minutes
        df = df[df["t"] <= self.time_cutoff_min].copy()

        # Also truncate before any cutoff token
        cutoff_idx = None
        for i, token in enumerate(df["name"].values):
            if token in self.cutoff_tokens:
                cutoff_idx = i
                break

        if cutoff_idx is not None:
            df = df.iloc[:cutoff_idx]

        return df


# Define operational outcomes
OUTCOMES = {
    "admit": OperationalConfig(
        name="admit",
        label_tokens=["Admit"],
        cutoff_tokens=["Discharge", "Admit"],
        time_cutoff_min=60,
        include_zero_shot=True,
    ),
    "icu": OperationalConfig(
        name="icu",
        label_tokens=["ICU Start"],
        cutoff_tokens=["Discharge", "Admit", "ICU Start"],
        time_cutoff_min=60,
        include_zero_shot=True,
    ),
    "sepsis": OperationalConfig(
        name="sepsis",
        label_tokens=[],  # Uses external CSV
        cutoff_tokens=["Discharge", "Admit"],
        time_cutoff_min=240,
        include_zero_shot=False,
        label_csv_path="cdw/sepsis.csv",
        include_csv_path="cdw/microbio.csv",
        exclude_csv_path="cdw/exclude_sepsis.csv",
    ),
    "septic_shock": OperationalConfig(
        name="septic_shock",
        label_tokens=[],  # Uses external CSV
        cutoff_tokens=["Discharge", "Admit"],
        time_cutoff_min=240,
        include_zero_shot=False,
        label_csv_path="cdw/septic_shock.csv",
        include_csv_path="cdw/microbio.csv",
        exclude_csv_path="cdw/exclude_sepsis.csv",
    ),
}


def load_model(checkpoint_path: str):
    """Load MINT model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    config = DelphiConfig(**checkpoint["model_args"])
    model = Delphi(config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.to(DEVICE)
    return model, config


def get_token_id_map():
    """Create token name to ID mapping."""
    return {k: v + 1 for k, v in zip(VOCAB["name"], VOCAB["index"])}


# ESI mapping
ESI_MAP = {
    "Acuity_Immediate": 1,
    "Acuity_Emergent": 2,
    "Acuity_Urgent": 3,
    "Acuity_Less Urgent": 4,
    "Acuity_Non-Urgent": 5,
}


def extract_esi_scores(df: pd.DataFrame) -> dict:
    """Extract numeric ESI (1-5) per encounter from Acuity_ tokens.

    Returns dict {encounter_key: esi_int}. Encounters without valid Acuity are excluded.
    """
    acuity_rows = df[df["name"].str.startswith("Acuity_")][["encounter_key", "name"]].drop_duplicates("encounter_key")
    acuity_rows = acuity_rows[acuity_rows["name"].isin(ESI_MAP.keys())]
    acuity_rows["esi"] = acuity_rows["name"].map(ESI_MAP)
    return dict(zip(acuity_rows["encounter_key"], acuity_rows["esi"]))


def pews_age_score(age: int) -> int:
    """ED-PEWS age component score."""
    if age <= 4:
        return 0
    elif age <= 11:
        return 4
    else:  # age > 11
        return 6


def pews_rr_score(rr: int) -> int:
    """ED-PEWS respiratory rate component score."""
    if rr < 30:
        return 0
    elif rr < 40:
        return 3
    elif rr < 60:
        return 5
    else:
        return 9


def pews_spo2_score(spo2: int) -> int:
    """ED-PEWS SpO2 component score."""
    if spo2 >= 98:
        return 0
    elif spo2 >= 94:
        return 4
    elif spo2 >= 88:
        return 9
    else:
        return 15


def pews_hr_score(hr: int) -> int:
    """ED-PEWS heart rate component score."""
    if hr < 100:
        return 0
    elif hr < 140:
        return 3
    elif hr < 180:
        return 6
    else:
        return 9


def compute_pews_total(age: int, consciousness: bool, wob: bool, rr: Optional[int],
                       spo2: Optional[int], hr: Optional[int], cap_refill: bool) -> Optional[int]:
    """Compute total ED-PEWS score from components.

    Returns None if critical vitals are missing.
    """
    # Critical vitals must be present
    if rr is None or spo2 is None or hr is None:
        return None

    score = pews_age_score(age)
    score += 14 if consciousness else 0
    score += 12 if wob else 0
    score += pews_rr_score(rr)
    score += pews_spo2_score(spo2)
    score += pews_hr_score(hr)
    score += 3 if cap_refill else 0
    return score


def extract_pews_scores(df: pd.DataFrame, config: OperationalConfig, max_len: int) -> dict:
    """Extract ED-PEWS scores per encounter.

    Returns dict {encounter_key: pews_score} - only for encounters with all required data.
    Must be in pews_predictors.csv and have all vitals within K-minute window.
    """
    # Load PEWS predictor data (vectorized)
    pews_df = pd.read_csv("cdw/pews_predictors.csv")
    pews_df = pews_df.set_index("EncounterKey")
    pews_df["wob"] = pews_df["work_of_breathing"].str.lower() == "yes"
    pews_df["consciousness"] = pews_df["decreased_level_of_conciousness"].str.lower() == "yes"
    pews_df["cap_refill"] = pews_df["increased_capillary_refill"].str.lower() == "yes"
    pews_encounter_set = set(pews_df.index)

    # Filter to time window and encounters that have PEWS predictor data
    vitals_df = df[(df["t"] <= config.time_cutoff_min) & (df["encounter_key"].isin(pews_encounter_set))].copy()

    # Extract vital values vectorized via string parsing
    # Heart rate (max per encounter)
    hr_mask = vitals_df["name"].str.startswith("Vital_Pulse_")
    hr_df = vitals_df.loc[hr_mask, ["encounter_key", "name"]].copy()
    hr_df["value"] = pd.to_numeric(hr_df["name"].str.removeprefix("Vital_Pulse_"), errors="coerce")
    hr_max = hr_df.groupby("encounter_key")["value"].max()

    # Respiratory rate (max per encounter)
    rr_mask = vitals_df["name"].str.startswith("Vital_Resp_")
    rr_df = vitals_df.loc[rr_mask, ["encounter_key", "name"]].copy()
    rr_df["value"] = pd.to_numeric(rr_df["name"].str.removeprefix("Vital_Resp_"), errors="coerce")
    rr_max = rr_df.groupby("encounter_key")["value"].max()

    # SpO2 (min per encounter)
    spo2_mask = vitals_df["name"].str.startswith("Vital_SpO2_")
    spo2_df = vitals_df.loc[spo2_mask, ["encounter_key", "name"]].copy()
    spo2_df["value"] = pd.to_numeric(spo2_df["name"].str.removeprefix("Vital_SpO2_"), errors="coerce")
    spo2_min = spo2_df.groupby("encounter_key")["value"].min()

    # Only encounters with all three vitals present
    complete_keys = hr_max.index.intersection(rr_max.index).intersection(spo2_min.index)

    # Compute PEWS scores vectorized
    pews_scores = {}
    for enc_key in complete_keys:
        if enc_key not in pews_encounter_set:
            continue
        row = pews_df.loc[enc_key]
        pews_score = compute_pews_total(
            age=int(row["Age"]),
            consciousness=bool(row["consciousness"]),
            wob=bool(row["wob"]),
            rr=int(rr_max[enc_key]),
            spo2=int(spo2_min[enc_key]),
            hr=int(hr_max[enc_key]),
            cap_refill=bool(row["cap_refill"]),
        )
        if pews_score is not None:
            pews_scores[enc_key] = pews_score

    return pews_scores


def build_dataset(df: pd.DataFrame, config: OperationalConfig, token_id_map: dict,
                  max_len: int, min_tokens: int) -> tuple[list, list, list]:
    """
    Build dataset for an outcome.

    Returns:
        encounters: list of tuples (events, times, label, encounter_key)
        labels: list of binary labels
        encounter_keys: list of encounter identifiers
    """
    # Precompute token_id column for the entire dataframe (vectorized)
    df = df.copy()
    df["token_id"] = df["name"].map(token_id_map)
    df = df.sort_values(["encounter_key", "t"])

    # Pre-filter to eligible encounters
    unique_keys = df["encounter_key"].unique()
    eligible_keys = np.array([k for k in unique_keys if config.is_eligible(k)])
    df = df[df["encounter_key"].isin(set(eligible_keys))]

    # Precompute labels per encounter using the raw (untruncated) data
    # For CSV-based labels, config.get_label only needs encounter_key
    # For token-based labels, we need to check if label_tokens appear in each encounter
    cutoff_token_set = set(config.cutoff_tokens)

    if config._label_set is not None:
        # CSV-based label: vectorized lookup
        encounter_to_label = {k: (1 if k in config._label_set else 0) for k in eligible_keys}
    else:
        # Token-based label: check if any label_token appears per encounter
        label_token_set = set(config.label_tokens)
        has_label = df[df["name"].isin(label_token_set)]["encounter_key"].unique()
        has_label_set = set(has_label)
        encounter_to_label = {k: (1 if k in has_label_set else 0) for k in eligible_keys}

    encounters = []
    labels = []
    encounter_keys = []

    # Process encounters in parallel via groupby
    for encounter_key, group in tqdm(df.groupby("encounter_key", sort=False), total=len(eligible_keys)):
        # Truncate to time cutoff
        group = group[group["t"] <= config.time_cutoff_min]

        # Truncate before any cutoff token
        cutoff_mask = group["name"].isin(cutoff_token_set)
        if cutoff_mask.any():
            first_cutoff_idx = cutoff_mask.idxmax()
            group = group.loc[:first_cutoff_idx].iloc[:-1]

        # Filter to rows with valid token_ids and extract arrays directly
        valid = group["token_id"].notna()
        events = group.loc[valid, "token_id"].values.astype(np.int64)
        times = group.loc[valid, "t"].values.astype(np.int64)

        if len(events) < min_tokens:
            continue

        # Truncate to max_len
        if len(events) > max_len:
            events = events[:max_len]
            times = times[:max_len]

        label = encounter_to_label[encounter_key]

        encounters.append((events, times, label, encounter_key))
        labels.append(label)
        encounter_keys.append(encounter_key)

    return encounters, labels, encounter_keys


def build_shared_dataset(df: pd.DataFrame, configs: list[OperationalConfig], token_id_map: dict,
                         max_len: int, min_tokens: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Build a shared dataset for multiple outcomes that share the same time_cutoff_min.
    Uses the union of cutoff tokens so all outcomes get identical truncated sequences.

    Returns:
        dict mapping encounter_key → (events, times) for all encounters passing min_tokens
        from any outcome's eligibility set.
    """
    # Merge cutoff tokens across all configs in this K-group
    merged_cutoff_tokens = set()
    for config in configs:
        merged_cutoff_tokens.update(config.cutoff_tokens)

    # Use the shared time_cutoff_min (all configs in the group have the same K)
    time_cutoff_min = configs[0].time_cutoff_min

    # Precompute token_id column
    df = df.copy()
    df["token_id"] = df["name"].map(token_id_map)
    df = df.sort_values(["encounter_key", "t"])

    # Union of eligible encounters across all outcomes
    unique_keys = df["encounter_key"].unique()
    eligible_keys = set()
    for config in configs:
        eligible_keys.update(k for k in unique_keys if config.is_eligible(k))
    df = df[df["encounter_key"].isin(eligible_keys)]

    shared = {}

    for encounter_key, group in tqdm(df.groupby("encounter_key", sort=False), total=len(eligible_keys)):
        # Truncate to time cutoff
        group = group[group["t"] <= time_cutoff_min]

        # Truncate before any cutoff token (using merged superset)
        cutoff_mask = group["name"].isin(merged_cutoff_tokens)
        if cutoff_mask.any():
            first_cutoff_idx = cutoff_mask.idxmax()
            group = group.loc[:first_cutoff_idx].iloc[:-1]

        # Filter to valid token_ids
        valid = group["token_id"].notna()
        events = group.loc[valid, "token_id"].values.astype(np.int64)
        times = group.loc[valid, "t"].values.astype(np.int64)

        if len(events) < min_tokens:
            continue

        # Truncate to max_len
        if len(events) > max_len:
            events = events[:max_len]
            times = times[:max_len]

        shared[encounter_key] = (events, times)

    return shared


def build_shared_bow(shared_dataset: dict[str, tuple[np.ndarray, np.ndarray]],
                     vocab_size: int) -> dict[str, np.ndarray]:
    """
    Build bag-of-words features from a shared dataset.

    Returns:
        dict mapping encounter_key → bow vector
    """
    shared_bow = {}
    for encounter_key, (events, _) in shared_dataset.items():
        bow = np.bincount(events - 1, minlength=vocab_size).astype(np.float32)
        shared_bow[encounter_key] = bow
    return shared_bow


def extract_embeddings_and_softmax(model: Delphi, events_list: list, times_list: list,
                                    device: torch.device = DEVICE,
                                    batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """
    Batched forward pass: extract mean-pooled embeddings AND last-token softmax probs.

    Left-pads sequences with token_id=0, time=-10_000 so the last real token is always
    at the rightmost position. Mean-pooling only considers non-padded positions.

    Returns:
        embeddings: (n_samples, embedding_dim) array
        softmax_probs: (n_samples, vocab_size) array of last-token softmax distributions
    """
    from torch.nn.utils.rnn import pad_sequence

    embeddings = []
    all_softmax = []

    prev_return_reps = model.config.return_reps
    model.config.return_reps = True
    model.eval()

    n = len(events_list)

    with torch.no_grad():
        n_batches = (n + batch_size - 1) // batch_size
        for start in tqdm(range(0, n, batch_size), total=n_batches, desc="Model inference"):
            batch_events = events_list[start:start + batch_size]
            batch_times = times_list[start:start + batch_size]

            # Flip sequences, right-pad, then flip back → left-padded
            events_tensors = [torch.tensor(e, dtype=torch.long).flip(0) for e in batch_events]
            times_tensors = [torch.tensor(t, dtype=torch.float).flip(0) for t in batch_times]

            padded_events = pad_sequence(events_tensors, batch_first=True, padding_value=0).flip(1)
            padded_times = pad_sequence(times_tensors, batch_first=True, padding_value=-10_000).flip(1)

            padded_events = padded_events.to(device)
            padded_times = padded_times.to(device)

            lengths = torch.tensor([len(e) for e in batch_events], dtype=torch.long)

            logits, _, reps, _ = model(padded_events, padded_times)

            # For each sample in the batch:
            B, T, D = reps.shape
            for i in range(B):
                seq_len = lengths[i].item()
                # Non-padded positions are the last `seq_len` positions (left-padded)
                real_reps = reps[i, T - seq_len:, :]  # (seq_len, D)
                mean_rep = real_reps.mean(dim=0).cpu().numpy()
                embeddings.append(mean_rep)

                # Last-token softmax is always at position -1 (rightmost)
                last_logits = logits[i, -1, :]
                probs = torch.softmax(last_logits, dim=0).cpu().numpy()
                all_softmax.append(probs)

    model.config.return_reps = prev_return_reps

    return np.vstack(embeddings), np.vstack(all_softmax)


def softmax_probs_for_tokens(softmax_matrix: np.ndarray, label_tokens: list[str],
                             token_id_map: dict) -> np.ndarray:
    """
    Extract softmax probabilities for specific label tokens from precomputed softmax matrix.

    Args:
        softmax_matrix: (n_samples, vocab_size) precomputed softmax distributions
        label_tokens: tokens to sum probabilities for
        token_id_map: token name → ID mapping

    Returns:
        (n_samples,) array of summed probabilities for the label tokens
    """
    pos_ids = [token_id_map[t] for t in label_tokens if t in token_id_map]
    if not pos_ids:
        raise ValueError(f"No label tokens found in vocabulary: {label_tokens}")
    return softmax_matrix[:, pos_ids].sum(axis=1).astype(np.float32)


def train_linear_probes(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Train logistic regression and XGBoost on embeddings.

    Returns:
        (lr_probs, xgb_probs)
    """
    # Scale embeddings
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Logistic regression
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42, solver="lbfgs")
    lr.fit(X_train_scaled, y_train)
    lr_probs = lr.predict_proba(X_test_scaled)[:, 1]

    # XGBoost
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    xgb_clf = xgb.XGBClassifier(
        random_state=42,
        n_jobs=1,
        eval_metric="aucpr",
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        verbose=0,
    )
    xgb_clf.fit(X_train_scaled, y_train, verbose=False)
    xgb_probs = xgb_clf.predict_proba(X_test_scaled)[:, 1]

    return lr_probs, xgb_probs


def train_xgboost_baseline(df: pd.DataFrame, config: OperationalConfig, token_id_map: dict,
                           max_len: int, min_tokens: int) -> tuple[np.ndarray, list, list]:
    """
    Build bag-of-words features and train XGBoost directly on them.
    """
    bow_features = []
    labels = []
    encounter_keys = []

    vocab_size = max(token_id_map.values())

    # Precompute token_id column for the entire dataframe (vectorized)
    df = df.copy()
    df["token_id"] = df["name"].map(token_id_map)
    df = df.sort_values(["encounter_key", "t"])

    # Pre-filter to eligible encounters
    unique_keys = df["encounter_key"].unique()
    eligible_keys = np.array([k for k in unique_keys if config.is_eligible(k)])
    df = df[df["encounter_key"].isin(set(eligible_keys))]

    # Precompute labels
    cutoff_token_set = set(config.cutoff_tokens)

    if config._label_set is not None:
        encounter_to_label = {k: (1 if k in config._label_set else 0) for k in eligible_keys}
    else:
        label_token_set = set(config.label_tokens)
        has_label = df[df["name"].isin(label_token_set)]["encounter_key"].unique()
        has_label_set = set(has_label)
        encounter_to_label = {k: (1 if k in has_label_set else 0) for k in eligible_keys}

    for encounter_key, group in tqdm(df.groupby("encounter_key", sort=False), total=len(eligible_keys)):
        # Truncate to time cutoff
        group = group[group["t"] <= config.time_cutoff_min]

        # Truncate before any cutoff token
        cutoff_mask = group["name"].isin(cutoff_token_set)
        if cutoff_mask.any():
            first_cutoff_idx = cutoff_mask.idxmax()
            group = group.loc[:first_cutoff_idx].iloc[:-1]

        # Filter to valid token_ids and build BoW directly via bincount
        valid_ids = group["token_id"].dropna().values.astype(np.int64)
        seq_len = len(valid_ids)

        if seq_len < min_tokens:
            continue

        # Truncate to max_len
        if seq_len > max_len:
            valid_ids = valid_ids[:max_len]

        # token IDs are 1-indexed, shift to 0-indexed for bincount
        bow = np.bincount(valid_ids - 1, minlength=vocab_size).astype(np.float32)

        bow_features.append(bow)
        labels.append(encounter_to_label[encounter_key])
        encounter_keys.append(encounter_key)

    return np.vstack(bow_features), np.array(labels), encounter_keys


def compute_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    """Compute AUROC and AUPRC with 95% CI."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.metrics import roc_curve, precision_recall_curve
    from scipy import stats

    auroc = roc_auc_score(labels, probs)
    auprc = average_precision_score(labels, probs)

    # Bootstrap CI
    n_bootstraps = 1000
    aurocs = []
    auprcs = []
    rng = np.random.RandomState(42)

    for _ in range(n_bootstraps):
        indices = rng.choice(len(labels), size=len(labels), replace=True)
        labels_boot = labels[indices]
        probs_boot = probs[indices]

        if len(np.unique(labels_boot)) > 1:
            aurocs.append(roc_auc_score(labels_boot, probs_boot))
            auprcs.append(average_precision_score(labels_boot, probs_boot))

    auroc_ci = [np.percentile(aurocs, 2.5), np.percentile(aurocs, 97.5)]
    auprc_ci = [np.percentile(auprcs, 2.5), np.percentile(auprcs, 97.5)]

    incidence = labels.mean()
    n_pos = int(labels.sum())

    return {
        "auroc": auroc,
        "auprc": auprc,
        "auroc_ci": auroc_ci,
        "auprc_ci": auprc_ci,
        "incidence": float(incidence),
        "n_pos": n_pos,
        "n_total": len(labels),
    }


def main():
    args = tapify(Args)

    save_dir = Path(args.save_path)
    save_dir.mkdir(exist_ok=True, parents=True)

    print(f"Loading data from {DATA_DIR}...")
    train_df = pd.read_feather(str(DATA_DIR / "train.feather"))
    test_df = pd.read_feather(str(DATA_DIR / "test.feather"))

    if args.debug:
        print("Debug mode: subsampling to 1000 encounters")
        rng = np.random.default_rng(42)
        train_keys = rng.choice(train_df["encounter_key"].unique(), size=1000, replace=False)
        test_keys = rng.choice(test_df["encounter_key"].unique(), size=1000, replace=False)
        train_df = train_df[train_df.encounter_key.isin(train_keys)]
        test_df = test_df[test_df.encounter_key.isin(test_keys)]

    print(f"Loading model from {args.checkpoint_path}...")
    model, _ = load_model(args.checkpoint_path)

    token_id_map = get_token_id_map()
    vocab_size = max(token_id_map.values())  # max token ID (1-indexed); bincount needs this after subtracting 1

    results_list = []

    # Group outcomes by time_cutoff_min so we can share forward passes
    k_groups: dict[int, list[tuple[str, OperationalConfig]]] = defaultdict(list)
    for outcome_name in args.outcomes:
        if outcome_name not in OUTCOMES:
            print(f"Warning: outcome {outcome_name} not defined, skipping")
            continue
        config = OUTCOMES[outcome_name]
        k_groups[config.time_cutoff_min].append((outcome_name, config))

    cache_dir = save_dir / "cache"

    for k_value, group_outcomes in k_groups.items():
        configs = [c for _, c in group_outcomes]
        outcome_names = [n for n, _ in group_outcomes]

        print(f"\n{'='*60}")
        print(f"K-group: {k_value}min | Outcomes: {outcome_names}")
        print(f"{'='*60}")

        # Build shared datasets (single groupby pass, merged cutoff tokens)
        cache_name = f"shared_dataset_K{k_value}_max{args.max_len}_min{args.min_tokens}"
        cached = _load_cache(cache_dir, cache_name)
        if cached is not None:
            shared_train, shared_test = cached
        else:
            print(f"Building shared dataset for K={k_value}min...")
            shared_train = build_shared_dataset(train_df, configs, token_id_map, args.max_len, args.min_tokens)
            shared_test = build_shared_dataset(test_df, configs, token_id_map, args.max_len, args.min_tokens)
            _save_cache(cache_dir, cache_name, (shared_train, shared_test))
        print(f"  Shared train encounters: {len(shared_train)}")
        print(f"  Shared test encounters: {len(shared_test)}")

        # Build shared BoW features
        cache_name_bow = f"shared_bow_K{k_value}_max{args.max_len}_min{args.min_tokens}"
        cached = _load_cache(cache_dir, cache_name_bow)
        if cached is not None:
            shared_train_bow, shared_test_bow = cached
        else:
            shared_train_bow = build_shared_bow(shared_train, vocab_size)
            shared_test_bow = build_shared_bow(shared_test, vocab_size)
            _save_cache(cache_dir, cache_name_bow, (shared_train_bow, shared_test_bow))

        # Shared model forward pass (embeddings + softmax) — one pass for all outcomes at this K
        # Order encounters consistently for indexing
        train_encounter_keys = list(shared_train.keys())
        test_encounter_keys = list(shared_test.keys())

        cache_name_emb = f"embeddings_softmax_K{k_value}_max{args.max_len}_min{args.min_tokens}"
        cached = _load_cache(cache_dir, cache_name_emb)
        if cached is not None:
            train_embeddings, test_embeddings, test_softmax = cached
        else:
            train_events_list = [shared_train[k][0] for k in train_encounter_keys]
            train_times_list = [shared_train[k][1] for k in train_encounter_keys]
            test_events_list = [shared_test[k][0] for k in test_encounter_keys]
            test_times_list = [shared_test[k][1] for k in test_encounter_keys]

            print(f"Extracting MINT embeddings + softmax for K={k_value}min (single pass)...")
            train_embeddings, _ = extract_embeddings_and_softmax(
                model, train_events_list, train_times_list, device=DEVICE
            )
            test_embeddings, test_softmax = extract_embeddings_and_softmax(
                model, test_events_list, test_times_list, device=DEVICE
            )
            _save_cache(cache_dir, cache_name_emb, (train_embeddings, test_embeddings, test_softmax))

        # Build index maps for fast lookup
        train_key_to_idx = {k: i for i, k in enumerate(train_encounter_keys)}
        test_key_to_idx = {k: i for i, k in enumerate(test_encounter_keys)}

        # Now process each outcome using the shared representations
        for outcome_name, config in group_outcomes:
            print(f"\n  --- {outcome_name} ---")

            # Filter to eligible encounters for this outcome and assign labels
            train_keys_outcome = [k for k in train_encounter_keys if config.is_eligible(k)]
            test_keys_outcome = [k for k in test_encounter_keys if config.is_eligible(k)]

            # Compute labels (from raw df, not truncated)
            if config._label_set is not None:
                train_labels = np.array([1 if k in config._label_set else 0 for k in train_keys_outcome])
                test_labels = np.array([1 if k in config._label_set else 0 for k in test_keys_outcome])
            else:
                label_token_set = set(config.label_tokens)
                train_pos_keys = set(train_df[train_df["name"].isin(label_token_set)]["encounter_key"].unique())
                test_pos_keys = set(test_df[test_df["name"].isin(label_token_set)]["encounter_key"].unique())
                train_labels = np.array([1 if k in train_pos_keys else 0 for k in train_keys_outcome])
                test_labels = np.array([1 if k in test_pos_keys else 0 for k in test_keys_outcome])

            test_keys_arr = np.array(test_keys_outcome)

            print(f"  Train: {len(train_labels)} samples ({train_labels.mean():.4f} incidence)")
            print(f"  Test: {len(test_labels)} samples ({test_labels.mean():.4f} incidence)")

            # Index into shared embeddings/softmax for this outcome's encounters
            train_idx = np.array([train_key_to_idx[k] for k in train_keys_outcome])
            test_idx = np.array([test_key_to_idx[k] for k in test_keys_outcome])

            X_train = train_embeddings[train_idx]
            X_test = test_embeddings[test_idx]
            test_softmax_outcome = test_softmax[test_idx]

            # MINT zero-shot (softmax) for outcomes with include_zero_shot
            if config.include_zero_shot:
                print(f"  Computing MINT zero-shot (softmax) for {outcome_name}...")
                probs_softmax = softmax_probs_for_tokens(test_softmax_outcome, config.label_tokens, token_id_map)

                metrics = compute_metrics(probs_softmax, test_labels)
                results_list.append({"outcome": outcome_name, "method": "MINT_softmax", **metrics})
                print(f"    AUROC: {metrics['auroc']:.4f} ({metrics['auroc_ci'][0]:.4f}-{metrics['auroc_ci'][1]:.4f})")
                print(f"    AUPRC: {metrics['auprc']:.4f} ({metrics['auprc_ci'][0]:.4f}-{metrics['auprc_ci'][1]:.4f})")

                pred_df = pd.DataFrame({"encounter_key": test_keys_arr, "probs": probs_softmax, "labels": test_labels})
                pred_df.to_csv(save_dir / f"{outcome_name}_MINT_softmax.csv", index=False)

            # MINT linear probes
            print(f"  Training MINT linear probes for {outcome_name}...")
            lr_probs, xgb_probs = train_linear_probes(X_train, train_labels, X_test)

            metrics_lr = compute_metrics(lr_probs, test_labels)
            results_list.append({"outcome": outcome_name, "method": "MINT_LR", **metrics_lr})
            print(f"    MINT_LR AUROC: {metrics_lr['auroc']:.4f}")

            metrics_xgb_emb = compute_metrics(xgb_probs, test_labels)
            results_list.append({"outcome": outcome_name, "method": "MINT_XGBoost", **metrics_xgb_emb})
            print(f"    MINT_XGBoost AUROC: {metrics_xgb_emb['auroc']:.4f}")

            pred_df = pd.DataFrame({"encounter_key": test_keys_arr, "probs_lr": lr_probs, "probs_xgb": xgb_probs, "labels": test_labels})
            pred_df.to_csv(save_dir / f"{outcome_name}_MINT_probes.csv", index=False)

            # XGBoost baseline (bag of words) — reuse shared BoW
            print(f"  Training XGBoost baseline for {outcome_name}...")
            X_train_bow = np.vstack([shared_train_bow[k] for k in train_keys_outcome])
            X_test_bow = np.vstack([shared_test_bow[k] for k in test_keys_outcome])

            n_pos = train_labels.sum()
            n_neg = len(train_labels) - n_pos
            scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

            xgb_baseline = xgb.XGBClassifier(
                random_state=42, n_jobs=1, eval_metric="aucpr",
                objective='binary:logistic', scale_pos_weight=scale_pos_weight, verbose=0,
            )
            xgb_baseline.fit(X_train_bow, train_labels, verbose=False)
            xgb_baseline_probs = xgb_baseline.predict_proba(X_test_bow)[:, 1]

            metrics_baseline = compute_metrics(xgb_baseline_probs, test_labels)
            results_list.append({"outcome": outcome_name, "method": "XGBoost_baseline", **metrics_baseline})
            print(f"    XGBoost_baseline AUROC: {metrics_baseline['auroc']:.4f}")

            pred_df = pd.DataFrame({"encounter_key": test_keys_arr, "probs": xgb_baseline_probs, "labels": test_labels})
            pred_df.to_csv(save_dir / f"{outcome_name}_XGBoost_baseline.csv", index=False)

            # ESI and ED-PEWS comparators (for Admit and ICU only)
            if outcome_name in ["admit", "icu"]:
                print(f"  Computing ESI and ED-PEWS scores for {outcome_name}...")

                cache_name_esi = f"esi_scores_K{k_value}"
                esi_scores = _load_cache(cache_dir, cache_name_esi)
                if esi_scores is None:
                    esi_scores = extract_esi_scores(test_df)
                    _save_cache(cache_dir, cache_name_esi, esi_scores)
                esi_mask = np.array([k in esi_scores for k in test_keys_arr])

                if esi_mask.sum() > 0:
                    esi_labels = test_labels[esi_mask]
                    esi_probs = np.array([1.0 / esi_scores[k] for k in test_keys_arr[esi_mask]])

                    metrics_esi = compute_metrics(esi_probs, esi_labels)
                    results_list.append({"outcome": outcome_name, "method": "ESI", **metrics_esi})
                    print(f"    ESI AUROC: {metrics_esi['auroc']:.4f} (n={esi_mask.sum()})")

                    pred_df_esi = pd.DataFrame({"encounter_key": test_keys_arr[esi_mask], "probs": esi_probs, "labels": esi_labels})
                    pred_df_esi.to_csv(save_dir / f"{outcome_name}_ESI.csv", index=False)

                cache_name_pews = f"pews_scores_{outcome_name}_K{k_value}"
                pews_scores = _load_cache(cache_dir, cache_name_pews)
                if pews_scores is None:
                    pews_scores = extract_pews_scores(test_df, config, args.max_len)
                    _save_cache(cache_dir, cache_name_pews, pews_scores)
                pews_mask = np.array([k in pews_scores for k in test_keys_arr])

                if pews_mask.sum() > 0:
                    pews_labels = test_labels[pews_mask]
                    pews_probs = np.array([float(pews_scores[k]) for k in test_keys_arr[pews_mask]])

                    metrics_pews = compute_metrics(pews_probs, pews_labels)
                    results_list.append({"outcome": outcome_name, "method": "ED-PEWS", **metrics_pews})
                    print(f"    ED-PEWS AUROC: {metrics_pews['auroc']:.4f} (n={pews_mask.sum()})")

                    pred_df_pews = pd.DataFrame({"encounter_key": test_keys_arr[pews_mask], "probs": pews_probs, "labels": pews_labels})
                    pred_df_pews.to_csv(save_dir / f"{outcome_name}_ED-PEWS.csv", index=False)

    # Save all results
    print(f"\n{'='*60}")
    print("Saving results...")

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(save_dir / "results_operational.csv", index=False)

    with open(save_dir / "results_operational.jsonl", "w") as f:
        for result in results_list:
            f.write(json.dumps(result) + "\n")

    print(f"Results saved to {save_dir}")
    print(results_df.to_string())


if __name__ == "__main__":
    main()
