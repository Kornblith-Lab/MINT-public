"""Core infrastructure for interpretability experiments.

Shared model loading, CDF computation, outcome definitions, and cohort loading.
All experiments truncate patients at min(t=120, disposition) and use the
competing-risks CDF with a 60-minute horizon.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from model import Delphi, DelphiConfig
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "respiratory",
    Path(__file__).resolve().parents[2] / "model" / "respiratory.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
RespiratoryEscalation = _mod.RespiratoryEscalation

DATA_DIR = Path("output")
VOCAB_CSV = DATA_DIR / "vocab.csv"
CHECKPOINT_PATH = "output/mint/ckpt_100000.pt"
ARTIFACTS_DIR = Path("artifacts/fig_interp")
HORIZON_MINUTES = 60.0
TRUNCATION_TIME = 120.0
DISPOSITION_TOKENS = ["Admit", "Discharge", "ICU Start"]

DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


def load_vocab():
    vocab = pd.read_csv(VOCAB_CSV)
    name_to_id = {row["name"]: int(row["index"]) for _, row in vocab.iterrows()}
    id_to_name = {int(row["index"]): row["name"] for _, row in vocab.iterrows()}
    return vocab, name_to_id, id_to_name


def load_model(checkpoint_path=CHECKPOINT_PATH, device=DEVICE):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = DelphiConfig(**checkpoint["model_args"])
    model = Delphi(config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.to(device)
    return model, config


def get_model_token_id(vocab_index: int) -> int:
    """Convert vocab CSV index (1-based) to model-space ID (+1 shift)."""
    return vocab_index + 1


# ---------------------------------------------------------------------------
# Outcome definitions
# ---------------------------------------------------------------------------

class OutcomeDefinition:
    """Defines positive token IDs for a single outcome in model-space."""

    def __init__(self, name: str, positive_ids: list[int]):
        self.name = name
        self.positive_ids = positive_ids

    def __repr__(self):
        return f"OutcomeDefinition({self.name}, n_tokens={len(self.positive_ids)})"


def build_outcome_definitions(vocab: pd.DataFrame, name_to_id: dict) -> dict[str, OutcomeDefinition]:
    """Build outcome definitions for all 13 outcomes using model-space IDs."""
    outcomes = {}

    def _numeric_ids(prefix, lo, hi):
        ids = []
        for name, idx in name_to_id.items():
            if name.startswith(prefix + "_"):
                try:
                    val = int(name[len(prefix) + 1:])
                    if lo <= val <= hi:
                        ids.append(get_model_token_id(idx))
                except ValueError:
                    continue
        return ids

    # 1. Hypoxia: SpO2 0-92
    outcomes["hypoxia"] = OutcomeDefinition("hypoxia", _numeric_ids("Vital_SpO2", 0, 92))

    # 2. Tachycardia: Pulse >= 180 (simplified, age 1-11 threshold)
    outcomes["tachycardia"] = OutcomeDefinition("tachycardia", _numeric_ids("Vital_Pulse", 180, 999))

    # 3. Hypotension: Systolic 0-90 (conservative, covers all ages)
    outcomes["hypotension"] = OutcomeDefinition("hypotension", _numeric_ids("Vital_Systolic", 0, 90))

    # 4. Tachypnea: Resp >= 28 (simplified threshold covering most ages)
    outcomes["tachypnea"] = OutcomeDefinition("tachypnea", _numeric_ids("Vital_Resp", 28, 999))

    # 5. Peri-arrest bradycardia: Pulse 0-59
    outcomes["periarrest"] = OutcomeDefinition("periarrest", _numeric_ids("Vital_Pulse", 0, 59))

    # 6. Vasopressor
    vaso_tokens = ["Med_epinephrine", "Med_milrinone", "Med_norepinephrine bitartrate",
                   "Med_dopamine", "Med_dobutamine"]
    outcomes["vasopressor"] = OutcomeDefinition(
        "vasopressor", [get_model_token_id(name_to_id[t]) for t in vaso_tokens if t in name_to_id])

    # 7. Ventilator (severity == 7)
    vent_tokens = [k for k, v in RespiratoryEscalation.SEVERITY.items() if v == 7]
    outcomes["ventilator"] = OutcomeDefinition(
        "ventilator", [get_model_token_id(name_to_id[t]) for t in vent_tokens if t in name_to_id])

    # 8. PPV (severity >= 5)
    ppv_tokens = [k for k, v in RespiratoryEscalation.SEVERITY.items() if v >= 5]
    outcomes["ppv"] = OutcomeDefinition(
        "ppv", [get_model_token_id(name_to_id[t]) for t in ppv_tokens if t in name_to_id])

    # 9. Resp rescue meds
    resp_rescue_tokens = ["Med_albuterol sulfate", "Med_albuterol sulfate concentrate",
                          "Med_albuterol sulfate hfa", "Med_levalbuterol",
                          "Med_ipratropium", "Med_ipratropium bromide",
                          "Med_racepinephrine", "Med_dexamethasone sodium phosphate",
                          "Med_dexamethasone", "Med_magnesium sulfate"]
    outcomes["resp_rescue"] = OutcomeDefinition(
        "resp_rescue", [get_model_token_id(name_to_id[t]) for t in resp_rescue_tokens if t in name_to_id])

    # 10. Cardio rescue
    cardio_tokens = ["Med_adenosine", "Med_amiodarone", "Med_atropine"] + vaso_tokens
    outcomes["cardio_rescue"] = OutcomeDefinition(
        "cardio_rescue", [get_model_token_id(name_to_id[t]) for t in cardio_tokens if t in name_to_id])

    # 11. Transfusion
    outcomes["transfusion"] = OutcomeDefinition(
        "transfusion", [get_model_token_id(name_to_id["Procedure_BLOOD TRANSFUSION ORDERABLES"])])

    # 12. ICU
    outcomes["icu"] = OutcomeDefinition("icu", [get_model_token_id(name_to_id["ICU Start"])])

    # 13. Admission
    outcomes["admission"] = OutcomeDefinition("admission", [get_model_token_id(name_to_id["Admit"])])

    return outcomes


# ---------------------------------------------------------------------------
# Cohort loading and truncation
# ---------------------------------------------------------------------------

def load_test_cohort(feather_path=None, acuity_filter=("Acuity_Immediate", "Acuity_Emergent")):
    """Load test feather, filter to Immediate/Emergent, group by encounter."""
    if feather_path is None:
        feather_path = DATA_DIR / "test.feather"
    df = pd.read_feather(feather_path)

    if acuity_filter:
        enc_with_acuity = df[df["name"].isin(acuity_filter)]["encounter_key"].unique()
        df = df[df["encounter_key"].isin(enc_with_acuity)]

    return df


def truncate_encounter(group_df, max_time=TRUNCATION_TIME, disposition_tokens=DISPOSITION_TOKENS):
    """Truncate a single encounter at min(max_time, first disposition token).

    Returns (events_list, times_list) with vocab-space IDs (NOT model-space).
    """
    disp_mask = group_df["name"].isin(disposition_tokens)
    if disp_mask.any():
        disp_time = group_df.loc[disp_mask, "t"].iloc[0]
        cutoff = min(max_time, disp_time)
    else:
        cutoff = max_time

    truncated = group_df[group_df["t"] < cutoff]
    if len(truncated) == 0:
        return None, None

    return truncated["name"].tolist(), truncated["t"].tolist()


def build_cohort_sequences(df, name_to_id, max_time=TRUNCATION_TIME, min_tokens=5):
    """Build truncated sequences for all encounters in the cohort.

    Returns list of dicts with keys: encounter_key, events (vocab-space), times, age.
    """
    sequences = []
    grouped = df.groupby("encounter_key", sort=False)

    for enc_key, group in tqdm(grouped, desc="Building cohort sequences"):
        group = group.sort_values("t")
        names, times = truncate_encounter(group, max_time)
        if names is None or len(names) < min_tokens:
            continue

        # Extract age from tokens
        age = None
        for n in names:
            if n.startswith("Age_"):
                try:
                    age = int(n.split("_")[1])
                    break
                except ValueError:
                    continue
        if age is None:
            continue

        # Convert to vocab-space IDs
        events = [name_to_id[n] for n in names if n in name_to_id]
        times_clean = [times[i] for i, n in enumerate(names) if n in name_to_id]

        if len(events) < min_tokens:
            continue

        sequences.append({
            "encounter_key": enc_key,
            "events": events,
            "times": times_clean,
            "age": age,
        })

    return sequences


# ---------------------------------------------------------------------------
# CDF probability computation (batched, multi-outcome)
# ---------------------------------------------------------------------------

def compute_cdf_probabilities_batch(
    model,
    model_config: DelphiConfig,
    sequences: list[dict],
    outcome_defs: dict[str, OutcomeDefinition],
    horizon: float = HORIZON_MINUTES,
    batch_size: int = 64,
    device=DEVICE,
    desc="Computing CDF probabilities",
) -> dict[str, np.ndarray]:
    """Compute competing-risks CDF probabilities for multiple outcomes in a single forward pass.

    Args:
        sequences: list of dicts with 'events' (vocab-space IDs) and 'times'
        outcome_defs: dict mapping outcome name -> OutcomeDefinition

    Returns:
        dict mapping outcome name -> np.ndarray of shape (n_sequences,)
    """
    n = len(sequences)
    results = {name: np.zeros(n, dtype=np.float32) for name in outcome_defs}

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
            # +1 shift from vocab-space to model-space
            shifted = [e + 1 for e in seq["events"][-seq_len:]]
            padded_events[j, :seq_len] = torch.tensor(shifted, dtype=torch.long)
            padded_times[j, :seq_len] = torch.tensor(
                seq["times"][-seq_len:], dtype=torch.float)

        with torch.no_grad():
            logits, _, _ = model(padded_events, padded_times)

        # Get last valid position for each sequence
        lengths = (padded_events > 0).sum(dim=1) - 1
        lengths = lengths.clamp(min=0)

        for j in range(bs):
            last_logits = logits[j, lengths[j], :]
            rates = torch.exp(last_logits).clamp(min=1e-10)
            lambda_total = rates.sum() - rates[:2].sum()

            for name, outcome_def in outcome_defs.items():
                pos_idx = torch.tensor(outcome_def.positive_ids, dtype=torch.long, device=device)
                lambda_pos = rates.index_select(0, pos_idx).sum()
                p = (lambda_pos / lambda_total) * (-torch.expm1(-lambda_total * horizon))
                results[name][start + j] = p.item()

    return results


