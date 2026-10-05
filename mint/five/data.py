from functools import partial
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score, roc_auc_score, precision_recall_curve
import torch
from fnmatch import fnmatch
from tqdm import tqdm
from abc import ABC, abstractmethod

from dataclasses import dataclass

from mint.model.respiratory import RespiratoryEscalation

@dataclass(frozen=True, slots=True)
class Case:
    encounter_key: str
    label: int
    age: int
    events: np.typing.NDArray
    times: np.typing.NDArray
    outcome_time: Optional[float] = None

class Label(ABC):
    name_to_id: dict[str, int]
    vocab: pd.DataFrame

    @abstractmethod
    def get_age_to_token_names(self) -> tuple[dict, dict]:
        pass
    @abstractmethod
    def get_age_to_token_ids(self) -> tuple[dict, dict]:
        pass

@dataclass
class CategoricalLabel(Label):
    positive_tokens: list[str]
    negative_token_match: str
    vocab: pd.DataFrame

    def __post_init__(self):
        self.name_to_id = {k:v+1 for k,v in zip(self.vocab["name"], self.vocab["index"])}
        self.ages = range(0,18)

        vocab_tokens = self.vocab.name.unique()

        pos_names = self.positive_tokens
        pos_ids = [self.name_to_id[n] for n in pos_names]

        neg_names = [n for n in vocab_tokens if fnmatch(n, self.negative_token_match) and n not in pos_names]
        neg_ids = [self.name_to_id[n] for n in neg_names]

        self._age_to_pos_id = self._ageify(pos_ids)
        self._age_to_neg_id = self._ageify(neg_ids)

        self._age_to_pos_name = self._ageify(pos_names)
        self._age_to_neg_name = self._ageify(neg_names)
    def _ageify(self, l: list):
        return {a:l for a in self.ages}
    def get_age_to_token_names(self):
        return self._age_to_pos_name, self._age_to_neg_name
    def get_age_to_token_ids(self):
        return self._age_to_pos_id, self._age_to_neg_id

@dataclass
class NumericLabel(Label):
    token: str
    positive_range: dict
    vocab: pd.DataFrame

    def __post_init__(self):
        age_to_pos_name = {}
        age_to_neg_name = {}

        age_to_pos_id = {}
        age_to_neg_id = {}

        self.name_to_id = {k:v+1 for k,v in zip(self.vocab["name"], self.vocab["index"])}
        ages = self.positive_range.keys() if "all" not in self.positive_range else range(0,18)

        vocab_tokens = self.vocab.name.unique()
        for age in ages:
            minimum, maximum = self.positive_range[age] if "all" not in self.positive_range else self.positive_range["all"]
            age_tokens = [v for v in vocab_tokens if self.token in v and minimum <= int(v.replace(self.token + "_", "")) <= maximum]
            negative_age_tokens = [v for v in vocab_tokens if self.token in v and not (minimum <= int(v.replace(self.token + "_", "")) <= maximum)]
            age_to_pos_name[age] = age_tokens
            age_to_neg_name[age] = negative_age_tokens
            age_to_pos_id[age] = [self.name_to_id[name] for name in age_tokens]
            age_to_neg_id[age] = [self.name_to_id[name] for name in negative_age_tokens]
        
        self._age_to_pos_name = age_to_pos_name
        self._age_to_neg_name = age_to_neg_name
        self._age_to_pos_id = age_to_pos_id
        self._age_to_neg_id = age_to_neg_id
    def get_age_to_token_names(self):
        return self._age_to_pos_name, self._age_to_neg_name
    def get_age_to_token_ids(self):
        return self._age_to_pos_id, self._age_to_neg_id

@dataclass
class Task:
    name: str
    label: Label
    lookahead_min: int
    exclusion_window_min: int
    tokens_min: int = 12
    style: str = "curve"  # figure style for report_summary_scoring: "curve" (PRC) or "bar"
    # forward_confirmation_window: Optional[int] = None

    # Prefix-based ablation groups: each maps a CLI key to (include, exclude)
    # startswith rules over vocab names. "Vital" excludes the O2 Device
    # sub-prefix so it can be ablated independently.
    ABLATION_PREFIXES = {
        "Vital": (("Vital_",), ("Vital_O2 Device",)),
        "O2Device": (("Vital_O2 Device",), ()),
        "Med": (("Med_",), ()),
        "Procedure": (("Procedure_",), ()),
        "Lab": (("Lab_",), ()),
    }

    def _ablated_token_ids(self, ablate: Optional[list[str]]) -> set[int]:
        """Token ids to strip from case histories, per ABLATION_PREFIXES groups."""
        if not ablate:
            return set()
        names = self.label.vocab["name"].astype(str)
        keep = np.zeros(len(names), dtype=bool)
        for key in ablate:
            include, exclude = self.ABLATION_PREFIXES[key]
            mask = names.str.startswith(include)
            if exclude:
                mask &= ~names.str.startswith(exclude)
            keep |= mask.to_numpy()
        return {self.label.name_to_id[n] for n in names[keep]}

    def _create_age_mapper(self, df):
        age_df = df[df["name"].str.startswith("Age_")]
        age_df["age"] = age_df.name.str.removeprefix("Age_").astype(int)
        encounter_to_age = dict(zip(age_df.encounter_key, age_df.age)) # type: ignore
        return encounter_to_age
    
    def report_summary_scoring(self, method_name, probs, labels, bootstrapped=True, save_dir=None):
        N_BOOT = 1000
        print(f"Performance for {method_name} on {self.name}@{self.lookahead_min}min")
        auprc = average_precision_score(labels, probs)
        auroc = roc_auc_score(labels, probs)
        prevalance = labels.mean()
        improvement = auprc/prevalance
        print(f"AUROC: {roc_auc_score(labels, probs):.4f}")
        print(f"AUPRC: {auprc:.4f}")
        print(f"Improvement over baseline: {improvement:.2f}x (prevalance: {prevalance:.2%})")

        precision, recall, thresholds = precision_recall_curve(labels, probs)
        for target_prec in [0.70, 0.30]:
            mask = precision >= target_prec
            if mask.any():
                recall_at_prec = recall[mask].max()
            else:
                recall_at_prec = 0.0
            print(f"Recall at {target_prec:.0%} precision: {recall_at_prec:.4f}")

        results = {
            f"{method_name}_auprc": auprc,
            f"{method_name}_auroc": auroc,
            f"{method_name}_improvement": improvement
        }
        if bootstrapped:
            rng = np.random.default_rng(42)
            n = len(labels)
            auprc_boots = np.empty(N_BOOT)
            auroc_boots = np.empty(N_BOOT)
            improvement_boots = np.empty(N_BOOT)
            for i in tqdm(range(N_BOOT), desc=f"Bootstrapping {method_name}"):
                idx = rng.integers(0, n, size=n)
                b_labels = labels[idx]
                b_probs = probs[idx]
                if b_labels.sum() == 0 or b_labels.sum() == n:
                    auprc_boots[i] = np.nan
                    auroc_boots[i] = np.nan
                    improvement_boots[i] = np.nan
                    continue
                b_auprc = average_precision_score(b_labels, b_probs)
                b_auroc = roc_auc_score(b_labels, b_probs)
                b_prevalance = b_labels.mean()
                auprc_boots[i] = b_auprc
                auroc_boots[i] = b_auroc
                improvement_boots[i] = b_auprc / b_prevalance
            auprc_ci = (np.nanpercentile(auprc_boots, 2.5), np.nanpercentile(auprc_boots, 97.5))
            auroc_ci = (np.nanpercentile(auroc_boots, 2.5), np.nanpercentile(auroc_boots, 97.5))
            improvement_ci = (np.nanpercentile(improvement_boots, 2.5), np.nanpercentile(improvement_boots, 97.5))
            print(f"AUPRC 95% CI: [{auprc_ci[0]:.4f}, {auprc_ci[1]:.4f}]")
            print(f"AUROC 95% CI: [{auroc_ci[0]:.4f}, {auroc_ci[1]:.4f}]")
            print(f"Improvement 95% CI: [{improvement_ci[0]:.2f}x, {improvement_ci[1]:.2f}x]")
            results[f"{method_name}_auprc_ci"] = auprc_ci
            results[f"{method_name}_auroc_ci"] = auroc_ci
            results[f"{method_name}_improvement_ci"] = improvement_ci

        if save_dir is not None:
            self._render_scoring_figure(method_name, labels, probs, results, save_dir)
        return results

    def _render_scoring_figure(self, method_name, labels, probs, results, save_dir):
        """Save a per-method figure: PRC ("curve") or AUROC/AUPRC bars with 95% CIs ("bar")."""
        import os
        import matplotlib.pyplot as plt
        os.makedirs(save_dir, exist_ok=True)

        if self.style == "bar":
            metrics = ["auroc", "auprc"]
            labels_txt = ["AUROC", "AUPRC"]
            vals = [results[f"{method_name}_{m}"] for m in metrics]
            cis = [results.get(f"{method_name}_{m}_ci") for m in metrics]
            # Asymmetric error bars from bootstrap CIs (None if not bootstrapped)
            if all(ci is not None for ci in cis):
                yerr = np.array([[v - ci[0] for v, ci in zip(vals, cis)],
                                 [ci[1] - v for v, ci in zip(vals, cis)]])
            else:
                yerr = None
            fig, ax = plt.subplots(figsize=(4, 5))
            x = np.arange(len(metrics))
            ax.bar(x, vals, width=0.6, color=["#005FA3", "#7FBFDF"], alpha=0.9,
                   yerr=yerr, capsize=4, error_kw=dict(lw=0.8, capthick=0.8))
            for xi, v in zip(x, vals):
                ax.text(xi, min(v + 0.03, 1.0), f"{v:.3f}", ha="center", va="bottom", fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels(labels_txt)
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("Score")
            ax.set_title(f"{method_name} — {self.name}@{self.lookahead_min}min")
            fig.tight_layout()
            out_path = os.path.join(save_dir, f"{self.name}_{method_name}_bar.png")
        else:
            fig, ax = plt.subplots(figsize=(6, 5))
            PrecisionRecallDisplay.from_predictions(labels, probs, ax=ax, name=method_name)
            fig.tight_layout()
            out_path = os.path.join(save_dir, f"{self.name}_{method_name}_prc.png")

        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {self.style} figure to {out_path}")

    def _get_selectors(self, df):
        positive_selectors = []
        negative_selectors = []
        age_to_positive_tokens, age_to_negative_tokens = self.label.get_age_to_token_names()
        for age, tokens in age_to_positive_tokens.items():
            selector = (df.age == age) & (df.name.isin(tokens))
            positive_selectors.append(selector)
        for age, tokens in age_to_negative_tokens.items():
            selector = (df.age == age) & (df.name.isin(tokens))
            negative_selectors.append(selector)
        return positive_selectors, negative_selectors

    def build_loader(self, *case_sets: dict[str, list[Case]], max_len: Optional[int] = None, batch_size: int=64):
        all_loaders = []
        for case_set in case_sets:
            all_cases = case_set["positive"] + case_set["negative"]
            id_to_name = {v:k for k,v in self.label.name_to_id.items()}

            age_to_pos_id, age_to_neg_id = self.label.get_age_to_token_ids()

            ds = CaseDataset(all_cases, age_to_pos_id, age_to_neg_id, max_len=max_len)
            loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=case_collate_fn,
            )
            all_loaders.append(loader)
        return all_loaders
    
    def build_bag_of_words(self, *case_sets: dict[str, list[Case]], max_len: Optional[int] = None) -> list[tuple[np.ndarray, np.ndarray, list[str]]]:
        all_X_y_pairs = []
        for case_set in case_sets:
            all_cases = case_set["positive"] + case_set["negative"]
            X = []
            y = []
            keys = []
            for c in tqdm(all_cases):
                events = c.events
                if max_len: events = events[-max_len:]
                bow = np.bincount(events, minlength=len(self.label.vocab) + 2)
                y.append(c.label)
                X.append(bow)
                keys.append(c.encounter_key)
            X = np.vstack(X)
            y = np.asarray(y)
            all_X_y_pairs.append((X, y, keys))

        return all_X_y_pairs

    # Token-id columns of the binary bag-of-words that correspond to triage-time
    # features: age, sex, ESI/acuity, chief complaint, arrival method, and vitals.
    # token_id = vocab index + 1 (see build_cases), so columns are index-shifted by 1.
    TRIAGE_PREFIXES = ("Age_", "Sex_", "Acuity_", "CC_", "Arrival_", "Vital_")
    TRIAGE_WINDOW_MIN = 30

    def _triage_columns(self) -> np.ndarray:
        vocab = self.label.vocab
        mask = vocab["name"].astype(str).str.startswith(self.TRIAGE_PREFIXES)
        return (vocab.loc[mask, "index"].to_numpy() + 1)

    def build_triage_features(self, *case_sets: dict[str, list[Case]]) -> list[tuple[np.ndarray, np.ndarray, list[str]]]:
        """Binary bag-of-words over triage-time tokens only, restricted to the first
        TRIAGE_WINDOW_MIN minutes of the (already outcome-truncated) trajectory.

        Unlike build_bag_of_words, this ignores max_len (triage tokens are at the head
        of the sequence, while max_len keeps the tail). Anything after the triage window
        is treated as unavailable/missing and dropped."""
        triage_cols = self._triage_columns()
        vocab_dim = len(self.label.vocab) + 2
        all_X_y_pairs = []
        for case_set in case_sets:
            all_cases = case_set["positive"] + case_set["negative"]
            X = []
            y = []
            keys = []
            for c in tqdm(all_cases):
                window = c.times <= self.TRIAGE_WINDOW_MIN
                events = c.events[window]
                bow = np.bincount(events, minlength=vocab_dim)
                X.append((bow[triage_cols] > 0).astype(np.float32))
                y.append(c.label)
                keys.append(c.encounter_key)
            X = np.vstack(X)
            y = np.asarray(y)
            all_X_y_pairs.append((X, y, keys))

        return all_X_y_pairs

    def build_cases(self, *dfs: pd.DataFrame, use_ed_data_only: bool = False, use_only_first_event: bool = True, use_negative_exclusion: bool = True, limit_pos_patients: bool = False, do_exclusion: bool = True, high_acuity_only: bool = False, return_outcome_time: bool = False, ablate: Optional[list[str]] = None):
        ablate_ids = np.fromiter(self._ablated_token_ids(ablate), dtype=np.int64) if ablate else None
        all_cases = []
        for df in tqdm(dfs, total=len(dfs)):
            df = df.sort_values(["encounter_key", "t"])

            EARLY_PERIOD_T = 30

            if high_acuity_only:
                early_period = (df.t <= EARLY_PERIOD_T)
                supp_oxygen = df.name.isin(set([k for k,v in RespiratoryEscalation.SEVERITY.items() if v != 0])) & early_period
                bolus = (df.name == "Med_IV fluid") & early_period
                high_acuity_encounters = df[supp_oxygen | bolus].encounter_key.tolist()
                # high_acuity_encounters = df[df.name.isin(set(["Arrival_Method_Walk In", "Arrival_Method_Car"]))].encounter_key.tolist()
                # high_acuity_encounters = df[df.name.isin(set(["Acuity_Immediate"]))].encounter_key.tolist()
                # Task name: baseline versus ESI versus RESP_60 -- RESP - 60min -- RESP - 30 mins -- bolus/supp
                # hypoxia: 0.5% versus 1.2% versus 3.1% -- 3.1% -- 2.9% --  2.2%
                # tachypnea: X versus 15.7% versus 33.2% -- 33% -- 32.3% -- 26.8%
                # bradypnea: X versus 4.3% versus 1.5% -- 1.5% -- 1.6% -- 2.0%
                # ppv: X versus 1.8% versus 7.2% -- 7.2% -- 8.0% -- 5.4%
                # periarrest: versus 0.0003
                # tachycardia: versus 0.00565
                before = df.encounter_key.nunique()
                print(before)
                df = df[df.encounter_key.isin(set(high_acuity_encounters))]
                after = df.encounter_key.nunique()
                print(before, after)
                print(f"{after / before:.2f}")

            encounter_to_age = self._create_age_mapper(df)
            df["age"] = df.encounter_key.map(encounter_to_age)

            if use_ed_data_only:
                events = {"Admit", "ICU Start", "Discharge"}

                # True at and after the first event row within each encounter
                after_event = df.groupby("encounter_key")["name"].transform(
                    lambda s: s.isin(events).cummax()
                )

                # keep only rows before the first event row
                df = df[~after_event]
            else:
                # Use a 10 day data limit
                DAY = 60 * 24
                df = df[df.t <= (10 * DAY)]

            positive_selectors, negative_selectors = self._get_selectors(df)
            positive_selector = np.logical_or.reduce(positive_selectors)
            negative_selector = np.logical_or.reduce(negative_selectors)

            # initial positive case dataset
            positive_df: pd.DataFrame = df[positive_selector]
            negative_df: pd.DataFrame = df[negative_selector]

            if do_exclusion:
                mask = (
                    positive_df.groupby("encounter_key")["t"]
                    .diff()
                    .ge(self.exclusion_window_min)
                    .fillna(True)
                )

                # positive cases after filtering with the exclusion window
                positive_df = positive_df[mask]

            if high_acuity_only:
                positive_df = positive_df[positive_df.t > EARLY_PERIOD_T]
                negative_df = negative_df[negative_df.t > EARLY_PERIOD_T]

            if use_only_first_event:
                # This was designed carefully. The idea here is that if you are predicting positive events, you take only the first one from a given encounter (initiation). To avoid negative events diluting the incidence, we will take only 1 procedure from negative encounters.

                # Filter out events without enough history before selecting first
                # Commented out because I am worried that the pre-emptive filter introduces a bias where we are no longer capturing just initiation, but instead that a patient could have been initiated in the masked section, so the model learns a shortcut
                # begin, end = self._get_indicies(df, positive_df)
                # positive_df = positive_df[(end - begin) >= self.tokens_min]
                # begin, end = self._get_indicies(df, negative_df)
                # negative_df = negative_df[(end - begin) >= self.tokens_min]

                positive_df = positive_df.sort_values("t")

                positive_df = positive_df.groupby("encounter_key").nth(0) # type: ignore
                negative_df = negative_df.groupby("encounter_key").sample(1, random_state=42)

            df["token_id"] = df["name"].map(self.label.name_to_id)

            target_dfs = [("positive", positive_df), ("negative", negative_df)]
            cases: dict[str, list[Case]] = {l[0]: [] for l in target_dfs}

            for label, target_df in target_dfs:
                assert not all([use_negative_exclusion, limit_pos_patients]), "use_negative_exclusion and limit_pos_patients are mutually exclusive"
                if use_negative_exclusion and label == "negative":
                    positive_encounter_keys = set(positive_df.encounter_key.tolist())
                    target_df = target_df[~target_df.encounter_key.isin(positive_encounter_keys)]
                if limit_pos_patients:
                    positive_encounter_keys = set(positive_df.encounter_key.tolist())
                    target_df = target_df[target_df.encounter_key.isin(positive_encounter_keys)]

                if label == "positive":
                    print(target_df.t.describe())
                    print(len(target_df) / len(negative_df))

                begin, end = self._get_indicies(df, target_df)
                outcome_times = target_df["t"].values if return_outcome_time else None
                for i, (b, e) in tqdm(enumerate(zip(begin, end)), total=len(begin), leave=False):
                    history = df.iloc[b:e]

                    if len(history) < self.tokens_min: continue

                    events = history["token_id"].values
                    times = history["t"].values

                    if ablate_ids is not None:
                        keep = ~np.isin(events, ablate_ids)
                        events = events[keep]
                        times = times[keep]

                    encounter_key = history["encounter_key"].values[0]
                    ot = float(outcome_times[i]) if return_outcome_time else None
                    cases[label].append(Case(encounter_key=encounter_key, label=int(label == "positive"), events=events, times=times, age=encounter_to_age[encounter_key], outcome_time=ot)) # type: ignore
                # print(label)
                # print(len([c for c in cases[label] if len(c.events) > 2048]) / len(cases[label]))
            all_cases.append(cases)
        # exit()
        return all_cases
    
    def _get_indicies(self, full, test):
        # Integer-encode encounters
        encounters = np.union1d(full["encounter_key"], test["encounter_key"])
        full_code = np.searchsorted(encounters, full["encounter_key"])
        test_code = np.searchsorted(encounters, test["encounter_key"])

        # Offset each encounter so searchsorted never crosses encounters
        M = max(full["t"].max(), test["t"].max()) + self.lookahead_min + 1

        full_key = full_code * M + full["t"].to_numpy()
        test_key = test_code * M + test["t"].to_numpy()

        # Find the beginning and end of each training window
        begin = np.searchsorted(full_key, test_code * M, side="left")
        end = np.searchsorted(full_key, test_key - self.lookahead_min, side="left")

        return begin, end

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class CaseDataset(Dataset):
    def __init__(self, cases: list[Case], age_to_pos_id, age_to_neg_id, max_len: Optional[int] = None):
        self.cases = cases
        self.age_to_pos_id = age_to_pos_id
        self.age_to_neg_id = age_to_neg_id
        self.max_len = max_len

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]

        events = case.events
        times = case.times

        if self.max_len:
            events = events[-self.max_len:]
            times = times[-self.max_len:]

        return {
            "encounter_key": case.encounter_key,
            "label": torch.tensor(case.label, dtype=torch.long),
            "events": torch.tensor(events, dtype=torch.long),
            "times": torch.tensor(times, dtype=torch.float),
            "age": case.age,
            "pos_ids": self.age_to_pos_id[case.age],
            "neg_ids": self.age_to_neg_id[case.age]
        }

def case_collate_fn(batch):
    labels = torch.stack([x["label"] for x in batch])
    encounter_keys = [x["encounter_key"] for x in batch]

    events = [x["events"] for x in batch]
    times = [x["times"] for x in batch]
    
    ages = [x["age"] for x in batch]
    pos_ids = [x["pos_ids"] for x in batch]
    neg_ids = [x["neg_ids"] for x in batch]

    # Pad on the right; use 0 for events and a sentinel for times
    padded_events = pad_sequence(events, batch_first=True, padding_value=0)
    padded_times = pad_sequence(times, batch_first=True, padding_value=-10_000)

    # Optional: lengths for masking
    lengths = torch.tensor([len(x) for x in events], dtype=torch.long)

    return {
        "encounter_key": encounter_keys,
        "labels": labels,
        "events": padded_events,
        "times": padded_times,
        "lengths": lengths,
        "ages": ages,
        "pos_ids": pos_ids,
        "neg_ids": neg_ids
    }