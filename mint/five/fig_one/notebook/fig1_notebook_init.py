# =============================================================================
# BLOCK 1: INITIALIZATION
# =============================================================================
# These files are Jupyter notebook cells executed sequentially in a shared
# namespace. DO NOT add import statements between blocks — all names defined
# in earlier blocks are already in scope.
#
# Execution order:
#   Cell 1: fig1_notebook_init.py   (imports, config, helpers, data splitting)
#   Cell 2: fig1_notebook_peft.py   (defines backbone_finetune, EncounterDataset)
#   Cell 3: fig1_notebook_run.py    (evaluation loop, uses all of the above)
#
# Prerequisites (already in scope before Cell 1):
#   - model: trained Delphi checkpoint, eval mode, CPU
#   - tokens: dict[str, pd.DataFrame] mapping hospital name -> DataFrame
#             Each DataFrame has columns [encounter_key, name, t].
#             NOTE: these token streams are always truncated at ED disposition
#             (admit / ICU start / discharge) upstream, so the model never sees
#             any event past disposition. This is why build_cases here does not
#             re-apply the ED-only truncation that fig1.py does via
#             use_ed_data_only — it is already baked into the input.
#   - vocab: pd.DataFrame of vocab.csv with columns [index, name, count]
#
# Dependencies: torch, numpy, pandas, sklearn, xgboost
#
# Testing:
#   Run the integration test after any changes to this file:
#     TEST_OUTCOMES=tachypnea python -m mint.five.fig_one.notebook.test_fig1_notebook
#
#   Environment variables:
#     TEST_OUTCOMES  - comma-separated outcomes (default: hypoxia,tachypnea)
#     TEST_N_ENCOUNTERS - number of encounters to use (default: 100)
# =============================================================================

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# ─── Configuration ───────────────────────────────────────────────────────────

DEBUG = False              # set True to run on small subset (fast sanity check)
DEBUG_N_PER_LABEL = 100    # max cases per label in debug mode

LOOKAHEAD_MIN = 5          # prediction horizon in minutes
MAX_LEN = 128              # max sequence length (tokens)
BATCH_SIZE = 32            # inference batch size
SEED = 42                  # random seed for reproducibility
TOKENS_MIN = 6             # minimum history tokens per case (notebook diverges from fig1.py's 12)
EXCLUSION_WINDOW_MIN = 60  # exclusion window for repeated events
SKIP_EXCLUSION = True      # skip exclusion filtering
EQUIVALENCE_SEEDS = 5      # number of seeds for equivalence search
EQUIVALENCE_PRECISION = 100

PEFT_BACKBONE = False      # set True to enable backbone fine-tuning per hospital
PEFT_CLASSIFICATION = False  # set True to fine-tune MINT + classification head (per outcome)
GLOBAL_XGBOOST = False     # set True to evaluate pre-trained global XGBoost models
TRIAGE = True              # set True to evaluate the triage-time logistic regression baseline
# If GLOBAL_XGBOOST is True, `global_xgb_json` must be in scope:
#   global_xgb_json: dict[str, str|Path] mapping task name -> path to XGBoost JSON checkpoint

OUTCOMES = ["hypoxia", "tachypnea", "periarrest", "tachycardia", "hypotension"]

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ─── Logging setup ───────────────────────────────────────────────────────────

log_path = OUTPUT_DIR / "fig1_notebook.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("fig1_notebook")
logger.info(f"Initialized. Output dir: {OUTPUT_DIR}")
logger.info(f"Config: LOOKAHEAD_MIN={LOOKAHEAD_MIN}, MAX_LEN={MAX_LEN}, BATCH_SIZE={BATCH_SIZE}")

# ─── Device ──────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logger.info(f"Device: {DEVICE}")

# ─── Vocab mapping ───────────────────────────────────────────────────────────

name_to_id = {k: v + 1 for k, v in zip(vocab["name"], vocab["index"])}
vocab_tokens = vocab["name"].unique()
logger.info(f"Vocab size: {len(vocab_tokens)}")

# ─── Task definitions (vitals only) ─────────────────────────────────────────

def _numeric_positive_ids(token_prefix, positive_range_by_age):
    """Build age -> list[token_id] for numeric thresholds."""
    age_to_pos = {}
    age_to_neg = {}
    for age, (lo, hi) in positive_range_by_age.items():
        pos_names = [n for n in vocab_tokens if token_prefix in n and lo <= int(n.replace(token_prefix + "_", "")) <= hi]
        neg_names = [n for n in vocab_tokens if token_prefix in n and not (lo <= int(n.replace(token_prefix + "_", "")) <= hi)]
        age_to_pos[age] = [name_to_id[n] for n in pos_names]
        age_to_neg[age] = [name_to_id[n] for n in neg_names]
    return age_to_pos, age_to_neg

TASK_DEFS = {}

# Periarrest: Pulse 0-59 for ages 0-11
TASK_DEFS["periarrest"] = {
    "token": "Vital_Pulse",
    "positive_range": {age: (0, 59) for age in range(0, 12)},
}

# Hypoxia: SpO2 0-92 for all ages
TASK_DEFS["hypoxia"] = {
    "token": "Vital_SpO2",
    "positive_range": {age: (0, 92) for age in range(0, 18)},
}

# Hypotension: systolic BP thresholds by age
TASK_DEFS["hypotension"] = {
    "token": "Vital_Systolic",
    "positive_range": {
        0: (0, 69), 1: (0, 71), 2: (0, 73), 3: (0, 75), 4: (0, 77), 5: (0, 79), 6: (0, 81), 7: (0, 83), 8: (0, 85), 9: (0, 87),
        10: (0, 89), 11: (0, 89), 12: (0, 89), 13: (0, 89), 14: (0, 89), 15: (0, 89), 16: (0, 89), 17: (0, 89),
    },
}

# Tachycardia: Pulse >180 for ages 1-11, >220 for age 0
TASK_DEFS["tachycardia"] = {
    "token": "Vital_Pulse",
    "positive_range": {**{age: (180, 999) for age in range(1, 12)}, **{0: (220, 999)}},
}

# Tachypnea: Resp rate thresholds by age
TASK_DEFS["tachypnea"] = {
    "token": "Vital_Resp",
    "positive_range": {
        0: (54, 999),
        1: (38, 999), 2: (38, 999),
        3: (38, 999), 4: (29, 999), 5: (29, 999),
        6: (26, 999), 7: (26, 999), 8: (26, 999), 9: (26, 999), 10: (26, 999), 11: (26, 999),
        12: (26, 999), 13: (21, 999), 14: (21, 999), 15: (21, 999), 16: (21, 999), 17: (21, 999),
    },
}

# Pre-compute age->token_id maps for each task
TASK_TOKEN_MAPS = {}
for task_name, tdef in TASK_DEFS.items():
    pos_map, neg_map = _numeric_positive_ids(tdef["token"], tdef["positive_range"])
    TASK_TOKEN_MAPS[task_name] = {"pos": pos_map, "neg": neg_map}

logger.info(f"Task definitions built for: {list(TASK_DEFS.keys())}")

# ─── Debug mode overrides ────────────────────────────────────────────────────

if DEBUG:
    logger.info(f"*** DEBUG MODE: limiting to {DEBUG_N_PER_LABEL} cases per label ***")
    EQUIVALENCE_SEEDS = 1

# ─── Data splitting (per hospital) ───────────────────────────────────────────

logger.info(f"Hospitals: {list(tokens.keys())}")

hospital_splits = {}
for hospital_name, hospital_df in tokens.items():
    logger.info(f"Splitting {hospital_name} into train/test (50/50 by encounter)...")
    all_encounters = hospital_df["encounter_key"].unique()
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(all_encounters)
    split_idx = len(shuffled) // 2
    train_encounters = set(shuffled[:split_idx])
    test_encounters = set(shuffled[split_idx:])

    tokens_train = hospital_df[hospital_df["encounter_key"].isin(train_encounters)].copy()
    tokens_test = hospital_df[hospital_df["encounter_key"].isin(test_encounters)].copy()

    # Val is 20% of train for XGBoost early stopping
    train_enc_arr = np.array(list(train_encounters))
    val_split = int(len(train_enc_arr) * 0.2)
    rng2 = np.random.default_rng(SEED + 1)
    shuffled_train = rng2.permutation(train_enc_arr)
    val_encounters = set(shuffled_train[:val_split])
    pure_train_encounters = set(shuffled_train[val_split:])

    tokens_val = tokens_train[tokens_train["encounter_key"].isin(val_encounters)].copy()
    tokens_pure_train = tokens_train[tokens_train["encounter_key"].isin(pure_train_encounters)].copy()

    hospital_splits[hospital_name] = {
        "tokens_pure_train": tokens_pure_train,
        "tokens_val": tokens_val,
        "tokens_test": tokens_test,
    }
    logger.info(f"  {hospital_name}: Train={len(pure_train_encounters)}, Val={len(val_encounters)}, Test={len(test_encounters)}")
    logger.info(f"  {hospital_name}: Train tokens={len(tokens_pure_train)}, Val={len(tokens_val)}, Test={len(tokens_test)}")

# ─── Helper functions ────────────────────────────────────────────────────────

def get_encounter_ages(df):
    """Extract encounter -> age mapping from Age_* tokens."""
    age_rows = df[df["name"].str.startswith("Age_")]
    age_vals = age_rows["name"].str.replace("Age_", "").astype(int)
    return dict(zip(age_rows["encounter_key"], age_vals))


def build_cases(df, task_name, encounter_to_age, use_first_only=True):
    """Build positive/negative cases for a task from a token DataFrame.

    Returns dict with keys 'positive' and 'negative', each a list of dicts:
      {'encounter_key': str, 'events': np.array, 'times': np.array, 'age': int, 'label': int}
    """
    tdef = TASK_DEFS[task_name]
    token_prefix = tdef["token"]
    pos_range = tdef["positive_range"]

    df = df.sort_values(["encounter_key", "t"]).copy()
    df["age"] = df["encounter_key"].map(encounter_to_age)
    df = df.dropna(subset=["age"])
    df["age"] = df["age"].astype(int)

    # Vectorized positive/negative detection
    is_token_mask = df["name"].str.startswith(token_prefix + "_")
    token_df = df[is_token_mask].copy()

    if len(token_df) == 0:
        logger.warning(f"No outcome tokens found for {task_name}")
        return {"positive": [], "negative": []}

    # Extract numeric values
    token_df["val"] = token_df["name"].str.replace(token_prefix + "_", "").astype(int)

    pos_mask_parts = []
    for age, (lo, hi) in pos_range.items():
        mask = (token_df["age"] == age) & (token_df["val"] >= lo) & (token_df["val"] <= hi)
        pos_mask_parts.append(mask)
    pos_mask = np.logical_or.reduce(pos_mask_parts) if pos_mask_parts else pd.Series(False, index=token_df.index)

    positive_df = token_df[pos_mask]
    negative_df = token_df[~pos_mask]

    # Apply exclusion window
    if not SKIP_EXCLUSION:
        excl_mask = (
            positive_df.groupby("encounter_key")["t"]
            .diff()
            .ge(EXCLUSION_WINDOW_MIN)
            .fillna(True)
        )
        positive_df = positive_df[excl_mask]

    # First event only
    if use_first_only:
        positive_df = positive_df.groupby("encounter_key").first().reset_index()
        negative_df = negative_df.groupby("encounter_key").sample(1, random_state=SEED).reset_index()

    # Map token names to IDs for the full df
    df["token_id"] = df["name"].map(name_to_id)

    # Vectorized case building using searchsorted (matches original _get_indicies)
    def _get_indices(full_df, target_df):
        encounters = np.union1d(full_df["encounter_key"].values, target_df["encounter_key"].values)
        full_code = np.searchsorted(encounters, full_df["encounter_key"].values)
        target_code = np.searchsorted(encounters, target_df["encounter_key"].values)
        M = max(full_df["t"].max(), target_df["t"].max()) + LOOKAHEAD_MIN + 1
        full_key = full_code * M + full_df["t"].values
        target_key = target_code * M + target_df["t"].values
        begin = np.searchsorted(full_key, target_code * M, side="left")
        end = np.searchsorted(full_key, target_key - LOOKAHEAD_MIN, side="left")
        return begin, end

    def _make_cases(target_df, label_val):
        if len(target_df) == 0:
            return []
        begin, end = _get_indices(df, target_df)
        token_ids = df["token_id"].values
        t_vals = df["t"].values.astype(np.float32)
        enc_keys = df["encounter_key"].values
        target_ages = target_df["age"].values
        target_encs = target_df["encounter_key"].values

        cases = []
        for i in range(len(target_df)):
            b, e = begin[i], end[i]
            if (e - b) < TOKENS_MIN:
                continue
            events = token_ids[b:e].astype(np.int64)
            times = t_vals[b:e]
            cases.append({
                "encounter_key": target_encs[i],
                "events": events,
                "times": times,
                "age": int(target_ages[i]),
                "label": label_val,
            })
        return cases

    logger.info(f"  Building positive cases ({len(positive_df)} candidates)...")
    pos_cases = _make_cases(positive_df, 1)
    logger.info(f"  Building negative cases ({len(negative_df)} candidates)...")
    neg_cases = _make_cases(negative_df, 0)

    if DEBUG:
        pos_cases = pos_cases[:DEBUG_N_PER_LABEL]
        neg_cases = neg_cases[:DEBUG_N_PER_LABEL]

    logger.info(f"  {task_name}: {len(pos_cases)} positive, {len(neg_cases)} negative cases")
    return {"positive": pos_cases, "negative": neg_cases}


class CaseDataset(Dataset):
    def __init__(self, cases, age_to_pos_id, age_to_neg_id, max_len=None):
        self.cases = cases
        self.age_to_pos_id = age_to_pos_id
        self.age_to_neg_id = age_to_neg_id
        self.max_len = max_len

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        c = self.cases[idx]
        events = c["events"]
        times = c["times"]
        if self.max_len and len(events) > self.max_len:
            events = events[-self.max_len:]
            times = times[-self.max_len:]
        age = c["age"]
        return {
            "encounter_key": c["encounter_key"],
            "label": torch.tensor(c["label"], dtype=torch.long),
            "events": torch.tensor(events, dtype=torch.long),
            "times": torch.tensor(times, dtype=torch.float),
            "pos_ids": self.age_to_pos_id.get(age, []),
            "neg_ids": self.age_to_neg_id.get(age, []),
        }


def collate_fn(batch):
    events = [x["events"] for x in batch]
    times = [x["times"] for x in batch]
    padded_events = pad_sequence(events, batch_first=True, padding_value=0)
    padded_times = pad_sequence(times, batch_first=True, padding_value=-10000)
    lengths = torch.tensor([len(x) for x in events], dtype=torch.long)
    return {
        "encounter_key": [x["encounter_key"] for x in batch],
        "labels": torch.stack([x["label"] for x in batch]),
        "events": padded_events,
        "times": padded_times,
        "lengths": lengths,
        "pos_ids": [x["pos_ids"] for x in batch],
        "neg_ids": [x["neg_ids"] for x in batch],
    }


def build_loader(cases_dict, age_to_pos_id, age_to_neg_id, max_len, batch_size):
    all_cases = cases_dict["positive"] + cases_dict["negative"]
    ds = CaseDataset(all_cases, age_to_pos_id, age_to_neg_id, max_len=max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


def build_bag_of_words(cases_dict, max_len, vocab_size):
    all_cases = cases_dict["positive"] + cases_dict["negative"]
    X = []
    y = []
    for c in all_cases:
        events = c["events"]
        if max_len:
            events = events[-max_len:]
        bow = np.bincount(events, minlength=vocab_size + 2)
        X.append(bow)
        y.append(c["label"])
    return np.vstack(X), np.asarray(y)


# Triage-time feature families: age, sex, ESI/acuity, chief complaint,
# arrival method, and vitals. token_id = vocab index + 1 (see name_to_id), so
# columns are index-shifted by 1. Mirrors Task.build_triage_features in
# mint/five/data.py — keep the two in sync.
TRIAGE_PREFIXES = ("Age_", "Sex_", "Acuity_", "CC_", "Arrival_", "Vital_")
TRIAGE_WINDOW_MIN = 30
_triage_mask = vocab["name"].astype(str).str.startswith(TRIAGE_PREFIXES)
TRIAGE_COLS = (vocab.loc[_triage_mask, "index"].to_numpy() + 1)
logger.info(f"Triage feature columns: {TRIAGE_COLS.size}")


def build_triage_features(cases_dict, vocab_size):
    """Binary bag-of-words over triage-time tokens only, restricted to the first
    TRIAGE_WINDOW_MIN minutes of the (already outcome-truncated) trajectory.

    Unlike build_bag_of_words this ignores max_len — triage tokens sit at the head
    of the sequence while max_len keeps the tail. Anything after the triage window
    is treated as unavailable/missing and dropped. The upstream horizon guard in
    build_cases already excludes tokens within LOOKAHEAD_MIN of the outcome, so a
    very early outcome still cannot leak future information."""
    all_cases = cases_dict["positive"] + cases_dict["negative"]
    X = []
    y = []
    for c in all_cases:
        window = c["times"] <= TRIAGE_WINDOW_MIN
        events = c["events"][window]
        bow = np.bincount(events, minlength=vocab_size + 2)
        X.append((bow[TRIAGE_COLS] > 0).astype(np.float32))
        y.append(c["label"])
    return np.vstack(X), np.asarray(y)


def softmax_probe(model, dataloader, device=DEVICE):
    """Next-token softmax probability probe: sum of softmax probs over positive indices.
    Also extracts embeddings in a single forward pass."""
    probs = []
    labels = []
    encounter_keys = []
    embeddings = []
    model.eval()

    prev_return_reps = model.config.return_reps
    model.config.return_reps = True

    for batch in dataloader:
        events = batch["events"].to(device)
        times = batch["times"].to(device)
        lengths = batch["lengths"].to(device) - 1
        pos_ids_batch = batch["pos_ids"]

        labels.extend(batch["labels"].cpu().numpy())
        encounter_keys.extend(batch["encounter_key"])

        with torch.no_grad():
            logits, _, reps, _ = model(events, times)

        for j in range(events.size(0)):
            last_logits = logits[j, lengths[j], :]
            softmax_probs = torch.softmax(last_logits, dim=0)
            pos_idx = torch.as_tensor(pos_ids_batch[j], dtype=torch.long, device=device)
            p = softmax_probs.index_select(0, pos_idx).sum()
            probs.append(p.item())
            embeddings.append(reps[j, lengths[j], :].cpu().numpy())

    model.config.return_reps = prev_return_reps

    return (
        np.asarray(probs, dtype=np.float32),
        np.asarray(labels),
        encounter_keys,
        np.vstack(embeddings),
    )


def compute_metrics(probs, labels, method_name, task_name):
    """Compute AUPRC, AUROC point estimates.

    Bootstrap CIs are computed separately in Block 4 (fig1_notebook_stats.py)
    using paired encounter-level resampling for proper statistical comparison.
    """
    auprc = average_precision_score(labels, probs)
    auroc = roc_auc_score(labels, probs)
    prevalence = labels.mean()
    improvement = auprc / prevalence if prevalence > 0 else 0

    logger.info(f"  {method_name} | AUROC={auroc:.4f} | AUPRC={auprc:.4f} | Improvement={improvement:.2f}x | Prevalence={prevalence:.4f}")

    return {
        f"{method_name}_auprc": float(auprc),
        f"{method_name}_auroc": float(auroc),
        f"{method_name}_improvement": float(improvement),
    }


def stratified_sample(y, size, seed=42):
    """Sample `size` indices from y, stratified by label to preserve class ratio."""
    y = np.asarray(y).ravel()
    rng_s = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos = max(1, int(round(size * len(pos_idx) / len(y))))
    n_neg = size - n_pos
    n_pos = min(n_pos, len(pos_idx))
    n_neg = min(n_neg, len(neg_idx))
    sampled_pos = rng_s.choice(pos_idx, size=n_pos, replace=False)
    sampled_neg = rng_s.choice(neg_idx, size=n_neg, replace=False)
    return np.concatenate([sampled_pos, sampled_neg])


def equivalence_search(X_train_full, y_train_full, X_val, y_val, X_test, y_test, mint_auroc, task_name, seed=42):
    """Binary search for smallest training set size where XGBoost outperforms MINT by 0.5% AUROC or more."""
    if y_test.sum() == 0 or y_test.sum() == len(y_test):
        return None

    n_total = len(y_train_full)

    def _train_and_score(X_sub, y_sub):
        if y_sub.sum() == 0 or y_sub.sum() == len(y_sub):
            return None
        n_pos = y_sub.sum()
        spw = (len(y_sub) - n_pos) / n_pos if n_pos > 0 else 1.0
        clf = xgb.XGBClassifier(random_state=42, n_jobs=1, eval_metric="aucpr",
                                objective="binary:logistic", scale_pos_weight=spw)
        clf.fit(X_sub, y_sub, eval_set=[(X_val, y_val)], verbose=False)
        score = clf.predict_proba(X_test)[:, 1]
        return float(roc_auc_score(y_test, score))

    def eval_size(size, s):
        if size >= n_total:
            return _train_and_score(X_train_full, y_train_full)
        idx = stratified_sample(y_train_full, size, seed=s)
        return _train_and_score(X_train_full[idx], y_train_full[idx])

    # Verify full dataset beats target before searching
    target_auroc = mint_auroc + 0.005  # 0.5% improvement
    full_auroc = eval_size(n_total, seed)
    if full_auroc is None or full_auroc < target_auroc:
        logger.info(f"    [equiv {task_name}] full set ({n_total:,}) AUROC={full_auroc} < target={target_auroc:.4f}, skipping")
        return None

    coarse_fractions = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    coarse_sizes = sorted(set(max(EQUIVALENCE_PRECISION, int(f * n_total)) for f in coarse_fractions))

    hi = n_total
    lo = 0
    for size in coarse_sizes:
        if size >= n_total:
            break
        auroc = eval_size(size, seed)
        if auroc is None:
            continue
        logger.info(f"    [equiv {task_name}] n={size:,} AUROC={auroc:.4f} (target={target_auroc:.4f})")
        if auroc >= target_auroc:
            hi = size
            break
        lo = size

    while hi - lo > EQUIVALENCE_PRECISION:
        mid = (lo + hi) // 2
        auroc = eval_size(mid, seed)
        if auroc is not None and auroc >= target_auroc:
            hi = mid
        else:
            lo = mid

    return hi


def check_two_classes(y_train, y_test, method_name, outcome, hospital, logger):
    """Check that both train and test sets have at least two classes.

    Returns:
        tuple: (is_valid, stats_dict)
            is_valid: True if both sets have two classes, False otherwise.
            stats_dict: Statistics about the datasets (n_total, n_pos for both).
    """
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    n_train = len(y_train)
    n_train_pos = int(np.sum(y_train == 1))
    n_train_neg = n_train - n_train_pos

    n_test = len(y_test)
    n_test_pos = int(np.sum(y_test == 1))
    n_test_neg = n_test - n_test_pos

    stats = {
        "train_n_total": n_train,
        "train_n_pos": n_train_pos,
        "train_n_neg": n_train_neg,
        "test_n_total": n_test,
        "test_n_pos": n_test_pos,
        "test_n_neg": n_test_neg,
    }

    # Check train set
    if n_train_pos < 1 or n_train_neg < 1:
        logger.warning(
            f"    [{hospital}] Skipping {method_name} for {outcome}: "
            f"train set has only one class (n_pos={n_train_pos}, n_neg={n_train_neg})"
        )
        return False, stats

    # Check test set
    if n_test_pos < 1 or n_test_neg < 1:
        logger.warning(
            f"    [{hospital}] Skipping {method_name} for {outcome}: "
            f"test set has only one class (n_pos={n_test_pos}, n_neg={n_test_neg})"
        )
        return False, stats

    return True, stats


logger.info("=" * 60)
logger.info("INITIALIZATION COMPLETE - Ready to run tasks")
logger.info("=" * 60)
