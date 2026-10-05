# NOTE: The MINT linear probe specifications (SVM, LR) in this file must stay
# aligned with mint/five/fig_one/notebook/fig1_notebook_run.py. Both use model
# embeddings (not bag-of-words), StandardScaler, and identical hyperparameters.
# If you change the probe methodology here, update the notebook and vice versa.

# NEJM-style AUROC/AUPRC grids are implemented in fig1_nejm.py.

import json
import os
from pathlib import Path
from typing import Callable, Optional

from torch.utils.data import DataLoader
import torch

import xgboost as xgb

import numpy as np
import pandas as pd
from tqdm import tqdm
from mint.five.data import CategoricalLabel, NumericLabel, Task
from mint.five.fig_one.equiv_helpers import equivalence_search
from mint.model.respiratory import RespiratoryEscalation
from model import Delphi, DelphiConfig

from dataclasses import dataclass
from tap import tapify

from functools import partial
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

DATA_DIR = Path("output")
VOCAB_CSV = DATA_DIR / "vocab.csv"
VOCAB = pd.read_csv(VOCAB_CSV)

@dataclass
class Args:
    mode: list[str]
    lookahead: list[int]
    checkpoint_path: str = "output/mint/ckpt.pt"
    save_path: str = "artifacts/fig1"
    debug: bool = False
    first: bool = False
    all_data: bool = False
    limit_pos_patients: bool = False
    max_len: Optional[int] = None
    skip_xgboost: bool = False
    skip_xgboost_ci: bool = False
    skip_exclusion: bool = False
    use_neg: bool = False
    high_acuity: bool = False
    estimator: str = "softmax"
    val: bool = False
    ablate: Optional[list[str]] = None
    style: str = "curve"

def task_factory(name, label, exclusion):
    return partial(
        Task,
        name=name,
        label=label,
        exclusion_window_min=exclusion,
    )

class Definition:
    NEVER_BEFORE = 999_999

    # TASKS = { "periarrest", "hypoxia", "hypotension", "vasopressor", "ventilator", "ppv", "cardio_rescue", "resp_rescue", "transfusion", "tachypnea", "bradypnea", "narcan", "tachycardia" }


    # TASKS = [ "hypoxia", "hypotension", "tachypnea", "bradypnea", "tachycardia", "periarrest" ] + INTERVENTION_TASKS

    TASKS = [ "hypoxia", "tachypnea", "periarrest", "tachycardia", "hypotension" ]
    INTERVENTION_TASKS = [ "ventilator", "ppv", "resp_rescue", "vasopressor", "transfusion" ]
    
    periarrest: Callable[[int], Task] = lambda lookahead_min: Task(name="periarrest", label=NumericLabel("Vital_Pulse", positive_range={age: (0,59) for age in range(0,12)}, vocab=VOCAB), lookahead_min=lookahead_min, exclusion_window_min=60)

    hypoxia: Callable[[int], Task] = lambda lookahead_min: Task(name="hypoxia", label=NumericLabel("Vital_SpO2", positive_range={age: (0,92) for age in range(0,18)}, vocab=VOCAB), lookahead_min=lookahead_min, exclusion_window_min=60)

    # h88 h90 h92 h94
    h88: Callable[[int], Task] = lambda lookahead_min: Task(name="hypoxia88", label=NumericLabel("Vital_SpO2", positive_range={age: (0,88) for age in range(0,18)}, vocab=VOCAB), lookahead_min=lookahead_min, exclusion_window_min=60)
    h90: Callable[[int], Task] = lambda lookahead_min: Task(name="hypoxia90", label=NumericLabel("Vital_SpO2", positive_range={age: (0,90) for age in range(0,18)}, vocab=VOCAB), lookahead_min=lookahead_min, exclusion_window_min=60)
    h92: Callable[[int], Task] = lambda lookahead_min: Task(name="hypoxia92", label=NumericLabel("Vital_SpO2", positive_range={age: (0,92) for age in range(0,18)}, vocab=VOCAB), lookahead_min=lookahead_min, exclusion_window_min=60)
    h94: Callable[[int], Task] = lambda lookahead_min: Task(name="hypoxia94", label=NumericLabel("Vital_SpO2", positive_range={age: (0,94) for age in range(0,18)}, vocab=VOCAB), lookahead_min=lookahead_min, exclusion_window_min=60)


    # https://airwayjedi.com/2022/12/06/pediatric-hypotension-think-hypovolemia/

    hypotension: Callable[[int], Task] = lambda lookahead_min: Task(name="hypotension", label=NumericLabel("Vital_Systolic", positive_range={
        0: (0, 69), 1: (0, 71), 2: (0, 73), 3: (0, 75), 4: (0, 77), 5: (0, 79), 6: (0, 81), 7: (0, 83), 8: (0, 85), 9: (0, 87),
        10: (0, 89), 11: (0, 89), 12: (0, 89), 13: (0, 89), 14: (0, 89), 15: (0, 89), 16: (0, 89), 17: (0, 89),
    }, vocab=VOCAB), lookahead_min=lookahead_min, exclusion_window_min=60)

    tachycardia: Callable[[int], Task] = lambda lookahead_min: Task(name="tachycardia", label=NumericLabel("Vital_Pulse", positive_range={**{age: (180,999) for age in range(1,12)}, **{0: (220,999)}}, vocab=VOCAB), lookahead_min=lookahead_min, exclusion_window_min=60)

    # https://acls-algorithms.com/wp-content/uploads/2021/07/PALS-Vital-Signs.pdf

    tachypnea: Callable[[int], Task] = lambda lookahead_min: Task(name="tachypnea", label=NumericLabel("Vital_Resp", positive_range={
        0: (54, 999),
        1: (38, 999), 2: (38, 999),
        3: (38, 999), 4: (29, 999), 5: (29, 999),
        6: (26, 999), 7: (26, 999), 8: (26, 999), 9: (26, 999), 10: (26, 999), 11: (26, 999),
        12: (26, 999), 13: (21, 999), 14: (21, 999), 15: (21, 999), 16: (21, 999), 17: (21, 999),
    }, vocab=VOCAB), lookahead_min=lookahead_min, exclusion_window_min=60)

    bradypnea: Callable[[int], Task] = lambda lookahead_min: Task(name="bradypnea", label=NumericLabel("Vital_Resp", positive_range={
        0: (0, 29),
        1: (0, 21), 2: (0, 21),
        3: (0, 21), 4: (0, 19), 5: (0, 19),
        6: (0, 17), 7: (0, 17), 8: (0, 17), 9: (0, 17), 10: (0, 17), 11: (0, 17),
        12: (0, 17), 13: (0, 11), 14: (0, 11), 15: (0, 11), 16: (0, 11), 17: (0, 11),
    }, vocab=VOCAB), lookahead_min=lookahead_min, exclusion_window_min=60)

    narcan: Callable[[int], Task] = lambda lookahead_min: Task(name="narcan", label=CategoricalLabel(positive_tokens=["Med_naloxone"], negative_token_match="Med_*", vocab=VOCAB), lookahead_min=lookahead_min, exclusion_window_min=60)

    vasopressor: Callable[[int], Task] = lambda lookahead_min: Task(name="vasopressor", label=CategoricalLabel(positive_tokens=["Med_epinephrine", "Med_milrinone", "Med_norepinephrine bitartrate", "Med_dopamine", "Med_dobutamine"], negative_token_match="Med_*", vocab=VOCAB), lookahead_min=lookahead_min, exclusion_window_min=60) # "Med_vasopressin", "Med_racepinephrine"

    resp_rescue: Callable[[int], Task] = lambda lookahead_min: Task(name="resp_rescue", label=CategoricalLabel(positive_tokens=["Med_albuterol sulfate", "Med_albuterol sulfate concentrate", "Med_albuterol sulfate hfa", "Med_levalbuterol", "Med_ipratropium", "Med_ipratropium bromide", "Med_racepinephrine", "Med_dexamethasone sodium phosphate", "Med_dexamethasone", "Med_magnesium sulfate"], negative_token_match="Med_*", vocab=VOCAB), lookahead_min=lookahead_min, exclusion_window_min=60)

    cardio_rescue: Callable[[int], Task] = lambda lookahead_min: Task(name="cardio_rescue", label=CategoricalLabel(positive_tokens=["Med_adenosine","Med_amiodarone", "Med_atropine"] + ["Med_epinephrine", "Med_milrinone", "Med_norepinephrine bitartrate", "Med_dopamine", "Med_dobutamine"], negative_token_match="Med_*", vocab=VOCAB), lookahead_min=lookahead_min, exclusion_window_min=60)

    transfusion: Callable[[int], Task] = lambda lookahead_min: Task(name="transfusion", label=CategoricalLabel(positive_tokens=["Procedure_BLOOD TRANSFUSION ORDERABLES"], negative_token_match="Procedure_*", vocab=VOCAB), lookahead_min=lookahead_min, exclusion_window_min=60)

    ventilator: Callable[[int], Task] = lambda lookahead_min: Task(name="ventilator", label=CategoricalLabel(positive_tokens=[k for k,v in RespiratoryEscalation.SEVERITY.items() if v == 7], negative_token_match="Vital_O2 Device_*", vocab=VOCAB), lookahead_min=lookahead_min, exclusion_window_min=60)

    ppv: Callable[[int], Task] = lambda lookahead_min: Task(name="ppv", label=CategoricalLabel(positive_tokens=[k for k,v in RespiratoryEscalation.SEVERITY.items() if v in (5, 6, 7)], negative_token_match="Vital_O2 Device_*", vocab=VOCAB), lookahead_min=lookahead_min, exclusion_window_min=60)

    r4: Callable[[int], Task] = lambda lookahead_min: Task(name="r4", label=CategoricalLabel(positive_tokens=[k for k,v in RespiratoryEscalation.SEVERITY.items() if v in (4, 5, 6, 7)], negative_token_match="Vital_O2 Device_*", vocab=VOCAB), lookahead_min=lookahead_min, exclusion_window_min=60)

    r3: Callable[[int], Task] = lambda lookahead_min: Task(name="r3", label=CategoricalLabel(positive_tokens=[k for k,v in RespiratoryEscalation.SEVERITY.items() if v in (3, 4, 5, 6, 7)], negative_token_match="Vital_O2 Device_*", vocab=VOCAB), lookahead_min=lookahead_min, exclusion_window_min=60)

    r2: Callable[[int], Task] = lambda lookahead_min: Task(name="r2", label=CategoricalLabel(positive_tokens=[k for k,v in RespiratoryEscalation.SEVERITY.items() if v in (2, 3, 4, 5, 6, 7)], negative_token_match="Vital_O2 Device_*", vocab=VOCAB), lookahead_min=lookahead_min, exclusion_window_min=60)

    r1: Callable[[int], Task] = lambda lookahead_min: Task(name="r1", label=CategoricalLabel(positive_tokens=[k for k,v in RespiratoryEscalation.SEVERITY.items() if v in (1, 2, 3, 4, 5, 6, 7)], negative_token_match="Vital_O2 Device_*", vocab=VOCAB), lookahead_min=lookahead_min, exclusion_window_min=60)

    @staticmethod
    def get(name: str, lookahead_min: int) -> Task:
        fn = getattr(Definition, name)
        return fn(lookahead_min)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(DEVICE)

def load_model(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    config = DelphiConfig(**checkpoint["model_args"])
    model = Delphi(config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.to(DEVICE)
    return model, config

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def temporal_cdf_probe_new(
    model,
    model_config: DelphiConfig,
    dataloader: DataLoader,
    horizon_minutes: float,
    device: torch.device = DEVICE,
    use_neg: bool = False,
    exclude_special_tokens: bool = True,
    extract_embeddings: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray | None]:
    """Competing-risks CDF probe over a DataLoader."""
    probs = []
    labels = []
    encounter_keys = []
    embeddings = []

    assert use_neg == False and exclude_special_tokens == True

    model.eval()

    if extract_embeddings:
        prev_return_reps = model.config.return_reps
        model.config.return_reps = True

    desc = " (use_neg enabled)" if use_neg else ""

    pbar = tqdm(dataloader)

    for batch in pbar:
        events = batch["events"].to(device)
        times = batch["times"].to(device)
        lengths = batch["lengths"].to(device) - 1  # last valid token index
        pos_ids_batch = batch["pos_ids"]           # list[list[int]]
        neg_ids_batch = batch["neg_ids"]

        pbar.set_description(f"token_len: {events.shape[-1]}" + desc)

        labels.extend(batch["labels"].cpu().numpy())
        encounter_keys.extend(batch["encounter_key"])

        with torch.no_grad():
            if extract_embeddings:
                logits, _, reps, _ = model(events, times)
            else:
                logits, _, _ = model(events, times)

        for j in range(events.size(0)):
            last_logits = logits[j, lengths[j], :]  # [V]

            pos_idx = torch.as_tensor(
                pos_ids_batch[j],
                dtype=torch.long,
                device=device,
            )

            neg_idx = torch.as_tensor(
                neg_ids_batch[j],
                dtype=torch.long,
                device=device,
            )

            rates = torch.exp(last_logits).clamp(min=1e-10)

            lambda_pos = rates.index_select(0, pos_idx).sum()
            lambda_total = lambda_pos + rates.index_select(0, neg_idx).sum() if use_neg else rates.sum()

            if exclude_special_tokens:
                lambda_total -= rates[:2].sum()

            p = (lambda_pos / lambda_total) * (
                -torch.expm1(-lambda_total * horizon_minutes)
            )
            probs.append(p.item())

            if extract_embeddings:
                embeddings.append(reps[j, lengths[j], :].cpu().numpy())

    if extract_embeddings:
        model.config.return_reps = prev_return_reps

    emb_array = np.vstack(embeddings) if extract_embeddings else None
    return np.asarray(probs, dtype=np.float32), np.asarray(labels), encounter_keys, emb_array

def softmax_probe(
    model,
    model_config: DelphiConfig,
    dataloader: DataLoader,
    device: torch.device = DEVICE,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Next-token softmax probability probe: sum of softmax probs over positive indices."""
    probs = []
    labels = []
    encounter_keys = []

    model.eval()

    for batch in tqdm(dataloader, desc="softmax probe"):
        events = batch["events"].to(device)
        times = batch["times"].to(device)
        lengths = batch["lengths"].to(device) - 1
        pos_ids_batch = batch["pos_ids"]

        labels.extend(batch["labels"].cpu().numpy())
        encounter_keys.extend(batch["encounter_key"])

        with torch.no_grad():
            logits, _, _ = model(events, times)

        for j in range(events.size(0)):
            last_logits = logits[j, lengths[j], :]
            softmax_probs = torch.softmax(last_logits, dim=0)

            pos_idx = torch.as_tensor(
                pos_ids_batch[j], dtype=torch.long, device=device
            )
            p = softmax_probs.index_select(0, pos_idx).sum()
            probs.append(p.item())

    return np.asarray(probs, dtype=np.float32), np.asarray(labels), encounter_keys


def _get_exclusion_mask(pos_ids, neg_ids, vocab_size, ignore_tokens, device):
    """Build a boolean mask of tokens to exclude from top-k selection."""
    exclude = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    exclude[torch.as_tensor(pos_ids, dtype=torch.long, device=device)] = True
    exclude[torch.as_tensor(neg_ids, dtype=torch.long, device=device)] = True
    exclude[torch.as_tensor(ignore_tokens, dtype=torch.long, device=device)] = True

    disposition_names = ["Admit", "ICU Start", "Discharge"]
    name_to_id = {k: v + 1 for k, v in zip(VOCAB["name"], VOCAB["index"])}
    for name in disposition_names:
        if name in name_to_id:
            exclude[name_to_id[name]] = True
    return exclude


def test_time_probe(
    model,
    model_config: DelphiConfig,
    dataloader: DataLoader,
    horizon_minutes: float,
    mode: str = "cdf",
    n_trajectories: int = 20,
    device: torch.device = DEVICE,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    """Test-time inference probe: extend each trajectory with top-k neutral tokens,
    then estimate probability via CDF or softmax on the extended sequence.

    Returns dict with 'mean', 'min', 'max' pooled probabilities."""
    all_probs_per_sample = []
    labels = []
    encounter_keys = []

    model.eval()
    ignore_tokens = model.config.ignore_tokens

    for batch in tqdm(dataloader, desc=f"test-time ({mode})"):
        events = batch["events"].to(device)
        times = batch["times"].to(device)
        lengths = batch["lengths"].to(device) - 1
        pos_ids_batch = batch["pos_ids"]
        neg_ids_batch = batch["neg_ids"]

        labels.extend(batch["labels"].cpu().numpy())
        encounter_keys.extend(batch["encounter_key"])

        with torch.no_grad():
            logits, _, _ = model(events, times)

        for j in range(events.size(0)):
            seq_len = lengths[j].item() + 1
            last_logits = logits[j, lengths[j], :]
            vocab_size = last_logits.shape[0]

            pos_ids = pos_ids_batch[j]
            neg_ids = neg_ids_batch[j]

            exclude_mask = _get_exclusion_mask(pos_ids, neg_ids, vocab_size, ignore_tokens, device)

            masked_logits = last_logits.clone()
            masked_logits[exclude_mask] = -torch.inf

            top_k_indices = masked_logits.topk(min(n_trajectories, (~exclude_mask).sum().item())).indices

            rates = torch.exp(last_logits).clamp(min=1e-10)
            rates[:2] = 0
            lambda_total = rates.sum()
            dt = 1.0 / lambda_total.item()

            last_time = times[j, lengths[j]].item()
            next_time = last_time + dt

            sample_events = events[j, :seq_len]
            sample_times = times[j, :seq_len]

            k = top_k_indices.shape[0]
            if k == 0:
                all_probs_per_sample.append([0.0])
                continue

            # Batch all k trajectories together: each is the shared prefix with one
            # candidate token appended, so they share length seq_len+1 and run in one pass.
            ext_events_batch = torch.cat([
                sample_events.unsqueeze(0).expand(k, seq_len),
                top_k_indices.unsqueeze(1),
            ], dim=1)
            ext_times_batch = torch.cat([
                sample_times.unsqueeze(0).expand(k, seq_len),
                torch.full((k, 1), next_time, dtype=sample_times.dtype, device=device),
            ], dim=1)

            with torch.no_grad():
                ext_logits, _, _ = model(ext_events_batch, ext_times_batch)

            ext_last_logits = ext_logits[:, -1, :]  # [k, V]
            pos_idx = torch.as_tensor(pos_ids, dtype=torch.long, device=device)

            if mode == "cdf":
                ext_rates = torch.exp(ext_last_logits).clamp(min=1e-10)
                lambda_pos = ext_rates.index_select(1, pos_idx).sum(dim=1)
                lambda_tot = ext_rates.sum(dim=1) - ext_rates[:, :2].sum(dim=1)
                p = (lambda_pos / lambda_tot) * (-torch.expm1(-lambda_tot * horizon_minutes))
            else:
                softmax_p = torch.softmax(ext_last_logits, dim=1)
                p = softmax_p.index_select(1, pos_idx).sum(dim=1)

            all_probs_per_sample.append(p.tolist())

    arr = np.array([np.array(x) for x in all_probs_per_sample], dtype=object)
    mean_probs = np.array([np.mean(x) for x in arr], dtype=np.float32)
    min_probs = np.array([np.min(x) for x in arr], dtype=np.float32)
    max_probs = np.array([np.max(x) for x in arr], dtype=np.float32)

    return {"mean": mean_probs, "min": min_probs, "max": max_probs}, np.asarray(labels), encounter_keys


def main():
    args = tapify(Args)

    valid_estimators = {"cdf", "softmax", "test_time_cdf", "test_time_softmax", "triage"}
    assert args.estimator in valid_estimators, f"--estimator must be one of {valid_estimators}"

    assert args.style in {"curve", "bar"}, "--style must be 'curve' or 'bar'"

    if args.ablate:
        valid_ablate = set(Task.ABLATION_PREFIXES)
        assert set(args.ablate) <= valid_ablate, f"--ablate must be a subset of {valid_ablate}"
        print(f"*** Ablating token prefixes: {args.ablate} ***")

    save_dir = Path(args.save_path)
    save_dir.mkdir(exist_ok=True, parents=True)

    train_path = str(DATA_DIR / f"train.feather")
    val_path = str(DATA_DIR / f"val.feather")
    test_path = str(DATA_DIR / f"test.feather")
    vocab_path = str(VOCAB_CSV)

    train = pd.read_feather(train_path)
    val = pd.read_feather(val_path)
    test = pd.read_feather(test_path)

    if args.debug:
        print("Starting debug mode, running only 1000 encounters each")

        n_groups = 1000
        rng = np.random.default_rng(42)

        train_groups = rng.choice(train["encounter_key"].unique(), size=n_groups, replace=False)
        val_groups = rng.choice(val["encounter_key"].unique(), size=n_groups, replace=False)
        test_groups = rng.choice(test["encounter_key"].unique(), size=n_groups, replace=False)

        train = train[train.encounter_key.isin(train_groups)]
        val = val[val.encounter_key.isin(val_groups)]
        test = test[test.encounter_key.isin(test_groups)]

    print(f"Loading model from {args.checkpoint_path}...")
    model, model_config = load_model(args.checkpoint_path)

    definitions: list[Task] = []

    modes = args.mode
    if modes[0] == "all": modes = Definition.TASKS + Definition.INTERVENTION_TASKS
    if modes[0] == "vitals_all": modes = Definition.TASKS
    if modes[0] == "int_all": modes = Definition.INTERVENTION_TASKS

    for name in modes:
        for lookahead in args.lookahead:
            print(f"{name}@{lookahead}min")
            definition = Definition.get(name, lookahead)
            definition.style = args.style
            definitions.append(definition)

    save_rows = []
    multi_lookahead = len(args.lookahead) > 1
    eval_split = "val" if args.val else "test"
    ablate_tag = f"_ablate-{'-'.join(args.ablate)}" if args.ablate else ""
    results_filename = f"results_{args.estimator}_{eval_split}{ablate_tag}.jsonl"

    if args.val:
        print("*** Using validation cohort as evaluation set ***")

    for definition in definitions:
        task_key = f"{definition.name}_{definition.lookahead_min}"
        train_cases, val_cases, test_cases = definition.build_cases(train, val, test, use_only_first_event=args.first, use_ed_data_only=not args.all_data, use_negative_exclusion=False, limit_pos_patients=args.limit_pos_patients, do_exclusion=not args.skip_exclusion, high_acuity_only=args.high_acuity, ablate=args.ablate)

        eval_cases = val_cases if args.val else test_cases

        batch_size = 32 if torch.backends.mps.is_available() else 32

        print("building data loader...")
        train_loader, test_loader = definition.build_loader(train_cases, eval_cases, max_len=args.max_len, batch_size=batch_size)
        print("building bag of words...")
        (X_train, y_train, _), (X_val, y_val, _), (X_test, y_test, xgb_encounter_keys) = definition.build_bag_of_words(train_cases, val_cases, eval_cases, max_len=args.max_len)

        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        print(f"training (incidence {y_train.mean():.4f})...")

        if not args.skip_xgboost:
            clf = xgb.XGBClassifier(
                random_state=42,
                n_jobs=1,
                eval_metric="aucpr",
                objective='binary:logistic',
                scale_pos_weight=scale_pos_weight
            )

            clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

            if definition.name in Definition.TASKS:
                ckpt_path = save_dir / f"{task_key}_xgb.json"
                clf.save_model(str(ckpt_path))
                print(f"saved XGBoost checkpoint to {ckpt_path}")

            print("predicting xgboost...")

            probs = clf.predict_proba(X_test)[:, 1]
            xgb_dict = definition.report_summary_scoring("XGBoost", probs, y_test, save_dir=args.save_path)

            if multi_lookahead:
                pd.DataFrame({
                    "probs": probs,
                    "labels": y_test,
                    "encounter_key": xgb_encounter_keys
                }).to_csv(save_dir / f"{definition.name}_{definition.lookahead_min}min_XGBoost.csv", index=False)
            else:
                pd.DataFrame({
                    "probs": probs,
                    "labels": y_test,
                    "encounter_key": xgb_encounter_keys
                }).to_csv(save_dir / f"{definition.name}_XGBoost.csv", index=False)

        is_vitals_task = definition.name in Definition.TASKS

        def run_xgb_equivalence(mint_auroc: float) -> dict:
            """Binary-search the XGBoost training-set size that matches MINT's AUROC.

            Independent of the estimator; only the MINT AUROC target differs.
            Returns an empty dict when XGBoost or its CI is skipped."""
            if args.skip_xgboost or args.skip_xgboost_ci:
                return {}
            print("doing xgboost equivalence (5 seeds)...")
            equivalence_samples_multiple = []
            for seed_i in range(5):
                result = equivalence_search(X_train, y_train, X_val, y_val, X_test, y_test, mint_auroc=mint_auroc, task_key=task_key, verbose=True, precision=100, seed=seed_i)
                equivalence_samples_multiple.append(result)
                print(f"  seed {seed_i}: {result}")
            return { "xgb_total_samples": len(X_train), "equivalence_samples_multiple": equivalence_samples_multiple, **xgb_dict }

        if args.estimator == "cdf":
            print("predicting CDF...")
            probs, labels, encounter_keys, emb_test = temporal_cdf_probe_new(
                model,
                model_config,
                test_loader,
                horizon_minutes=definition.lookahead_min,
                use_neg=args.use_neg,
                extract_embeddings=is_vitals_task,
            )

            if is_vitals_task:
                print("extracting train embeddings for linear probes...")
                _, y_emb_train, _, emb_train = temporal_cdf_probe_new(
                    model,
                    model_config,
                    train_loader,
                    horizon_minutes=definition.lookahead_min,
                    use_neg=args.use_neg,
                    extract_embeddings=True,
                )

                scaler = StandardScaler()
                emb_train_scaled = scaler.fit_transform(emb_train)
                emb_test_scaled = scaler.transform(emb_test)

                n_pos_train_emb = int(y_emb_train.sum())
                n_neg_train_emb = int(len(y_emb_train) - n_pos_train_emb)

                print("training MINT logistic regression probe...")
                lr_clf = LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                    solver="lbfgs",
                )
                lr_clf.fit(emb_train_scaled, y_emb_train)
                lr_probs = lr_clf.predict_proba(emb_test_scaled)[:, 1]
                lr_dict = definition.report_summary_scoring("MINT_LR", lr_probs, labels, save_dir=args.save_path)

                print("training MINT SVM probe...")
                svm_base = LinearSVC(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=42,
                    dual="auto",
                )
                n_min_class = min(n_pos_train_emb, n_neg_train_emb)
                svm_cv = min(3, n_min_class)
                if svm_cv >= 2:
                    svm_cal = CalibratedClassifierCV(svm_base, cv=svm_cv, method="sigmoid")
                    svm_cal.fit(emb_train_scaled, y_emb_train)
                    svm_probs = svm_cal.predict_proba(emb_test_scaled)[:, 1]
                else:
                    svm_base.fit(emb_train_scaled, y_emb_train)
                    svm_probs = svm_base.decision_function(emb_test_scaled)
                    svm_probs = 1.0 / (1.0 + np.exp(-svm_probs))
                svm_dict = definition.report_summary_scoring("MINT_SVM", svm_probs, labels, save_dir=args.save_path)

            csv_suffix = f"_{definition.lookahead_min}min" if multi_lookahead else ""
            pd.DataFrame({
                "probs": probs,
                "labels": labels,
                "encounter_key": encounter_keys
            }).to_csv(save_dir / f"{definition.name}{csv_suffix}_CDF.csv", index=False)

            cdf_dict = definition.report_summary_scoring("CDF", probs, labels, save_dir=args.save_path)
            combined_dict = { "estimator": "cdf", "lookahead_min": definition.lookahead_min, "task": definition.name, **cdf_dict, "incidence": float(y_test.mean()), "n": int(y_test.sum()) }

            if is_vitals_task:
                combined_dict = { **combined_dict, **lr_dict, **svm_dict }

            combined_dict = { **combined_dict, **run_xgb_equivalence(cdf_dict["CDF_auroc"]) }

        elif args.estimator == "softmax":
            print("predicting softmax...")
            probs, labels, encounter_keys = softmax_probe(
                model, model_config, test_loader
            )

            csv_suffix = f"_{definition.lookahead_min}min" if multi_lookahead else ""
            pd.DataFrame({
                "probs": probs,
                "labels": labels,
                "encounter_key": encounter_keys
            }).to_csv(save_dir / f"{definition.name}{csv_suffix}_softmax.csv", index=False)

            score_dict = definition.report_summary_scoring("softmax", probs, labels, save_dir=args.save_path)
            combined_dict = { "estimator": "softmax", "lookahead_min": definition.lookahead_min, "task": definition.name, **score_dict, "incidence": float(y_test.mean()), "n": int(y_test.sum()) }
            combined_dict = { **combined_dict, **run_xgb_equivalence(score_dict["softmax_auroc"]) }

        elif args.estimator == "triage":
            print("building triage features (first 30 min of triage-time tokens)...")
            (Xt_train, yt_train, _), (Xt_val, yt_val, _), (Xt_test, yt_test, triage_keys) = \
                definition.build_triage_features(train_cases, val_cases, eval_cases)

            scaler = StandardScaler()
            Xt_train_scaled = scaler.fit_transform(Xt_train)
            Xt_test_scaled = scaler.transform(Xt_test)

            print(f"training triage logistic regression ({Xt_train.shape[1]} features)...")
            triage_clf = LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
                solver="lbfgs",
            )
            triage_clf.fit(Xt_train_scaled, yt_train)
            probs = triage_clf.predict_proba(Xt_test_scaled)[:, 1]

            csv_suffix = f"_{definition.lookahead_min}min" if multi_lookahead else ""
            pd.DataFrame({
                "probs": probs,
                "labels": yt_test,
                "encounter_key": triage_keys
            }).to_csv(save_dir / f"{definition.name}{csv_suffix}_triage.csv", index=False)

            score_dict = definition.report_summary_scoring("triage", probs, yt_test, save_dir=args.save_path)
            combined_dict = { "estimator": "triage", "lookahead_min": definition.lookahead_min, "task": definition.name, **score_dict, "incidence": float(yt_test.mean()), "n": int(yt_test.sum()) }

        elif args.estimator in ("test_time_cdf", "test_time_softmax"):
            tt_mode = "cdf" if args.estimator == "test_time_cdf" else "softmax"
            print(f"predicting test-time ({tt_mode})...")

            probs_dict, labels, encounter_keys = test_time_probe(
                model, model_config, test_loader,
                horizon_minutes=definition.lookahead_min,
                mode=tt_mode,
            )

            csv_suffix = f"_{definition.lookahead_min}min" if multi_lookahead else ""
            pd.DataFrame({
                "probs_mean": probs_dict["mean"],
                "probs_min": probs_dict["min"],
                "probs_max": probs_dict["max"],
                "labels": labels,
                "encounter_key": encounter_keys
            }).to_csv(save_dir / f"{definition.name}{csv_suffix}_{args.estimator}.csv", index=False)

            combined_dict = { "estimator": args.estimator, "lookahead_min": definition.lookahead_min, "task": definition.name, "incidence": float(y_test.mean()), "n": int(y_test.sum()) }
            for pool_name, pool_probs in probs_dict.items():
                pool_dict = definition.report_summary_scoring(f"{args.estimator}_{pool_name}", pool_probs, labels, save_dir=args.save_path)
                combined_dict = { **combined_dict, **pool_dict }

            combined_dict = { **combined_dict, **run_xgb_equivalence(combined_dict[f"{args.estimator}_mean_auroc"]) }

        jsonl = Path(save_dir / results_filename)
        if not jsonl.exists(): jsonl.write_text("")
        jsonl.write_text(jsonl.read_text() + json.dumps(combined_dict) + "\n")
        save_rows.append(combined_dict)

    pd.DataFrame(save_rows).to_csv(save_dir / f"results_{args.estimator}_{eval_split}{ablate_tag}.csv")

if __name__ == "__main__":
    main()
