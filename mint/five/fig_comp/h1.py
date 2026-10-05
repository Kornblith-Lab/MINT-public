"""Standalone H1 analysis for septic shock and sepsis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SAVE_DIR = Path("artifacts/fig_comp_h1")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
H1_OUTCOMES = ("septic_shock", "sepsis")
DATA_DIR = Path("output")
TEST_PATH = DATA_DIR / "test.feather"
CACHE_DIR = SAVE_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ABBREV = {
    "septic_shock": "Septic shock",
    "sepsis": "Sepsis",
}

OUTCOME_SPECS = {
    "septic_shock": {
        "label": "Septic shock",
        "mint_path": Path("artifacts/fig1_operational/septic_shock_MINT_probes.csv"),
        "xgb_path": Path("artifacts/fig1_operational/septic_shock_XGBoost_baseline.csv"),
        "mint_col": "probs_lr",
        "xgb_col": "probs",
    },
    "sepsis": {
        "label": "Sepsis",
        "mint_path": Path("artifacts/fig1_operational/sepsis_MINT_probes.csv"),
        "xgb_path": Path("artifacts/fig1_operational/sepsis_XGBoost_baseline.csv"),
        "mint_col": "probs_lr",
        "xgb_col": "probs",
    },
}

ABX_TERMS = (
    "cef", "penicillin", "amoxicillin", "ampicillin", "azithromycin", "clindamycin",
    "vancomycin", "piperacillin", "metronidazole", "trimethoprim", "sulfamethoxazole",
    "gentamicin", "tobramycin", "doxycycline", "acyclovir", "fluconazole", "linezolid",
    "meropenem", "imipenem", "ertapenem", "cephalexin", "cefdinir", "cefotaxime",
    "ceftriaxone", "cefepime", "ceftazidime", "cefixime", "ceftaroline", "rifampin",
)

VITAL_PREFIXES = {
    "pulse": "Vital_Pulse_",
    "resp": "Vital_Resp_",
    "sbp": "Vital_Systolic_",
    "spo2": "Vital_SpO2_",
}


class OutcomeFrame:
    def __init__(self, outcome: str, label: str, df: pd.DataFrame):
        self.outcome = outcome
        self.label = label
        self.df = df

plt.style.use(str(Path(__file__).resolve().parents[1] / "design-skill" / "nature.mplstyle"))
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }
)


def _save_figure(fig: plt.Figure, path_base: Path, dpi: int = 300) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _tertiles(series: pd.Series) -> pd.Series:
    try:
        return pd.qcut(series.rank(method="first"), 3, labels=False, duplicates="drop")
    except ValueError:
        return pd.Series(np.nan, index=series.index)


def _subplot_label(ax, label: str) -> None:
    ax.text(-0.12, 1.04, label, transform=ax.transAxes, fontweight="bold", fontsize=10, va="top")


def _bootstrap_diff(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, n_boot: int = 500,
                    seed: int = 42) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    valid = ~np.isnan(y_true) & ~np.isnan(score_a) & ~np.isnan(score_b)
    y_true = y_true[valid]
    score_a = score_a[valid]
    score_b = score_b[valid]
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return {"diff": np.nan, "lo": np.nan, "hi": np.nan}
    diff = roc_auc_score(y_true, score_a) - roc_auc_score(y_true, score_b)
    rng = np.random.default_rng(seed)
    boot = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        b_true = y_true[idx]
        if np.unique(b_true).size < 2:
            continue
        boot.append(roc_auc_score(b_true, score_a[idx]) - roc_auc_score(b_true, score_b[idx]))
    if not boot:
        return {"diff": diff, "lo": np.nan, "hi": np.nan}
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return {"diff": diff, "lo": float(lo), "hi": float(hi)}


def _age_from_group(group: pd.DataFrame) -> float:
    rows = group[group["name"].str.startswith("Age_")]
    if rows.empty:
        return np.nan
    return pd.to_numeric(rows.iloc[0]["name"].removeprefix("Age_"), errors="coerce")


def _encounter_chief_complaint(group: pd.DataFrame) -> str:
    rows = group[group["name"].str.startswith("CC_")]
    if rows.empty:
        return "CC_OTHER"
    return str(rows.iloc[0]["name"]).removeprefix("CC_")


def _first_numeric_value(group: pd.DataFrame, prefix: str) -> float:
    rows = group[group["name"].str.startswith(prefix)]
    if rows.empty:
        return np.nan
    return pd.to_numeric(rows.iloc[0]["name"].removeprefix(prefix), errors="coerce")


def _slope(times: np.ndarray, values: np.ndarray) -> float:
    valid = ~np.isnan(times) & ~np.isnan(values)
    times = np.asarray(times)[valid]
    values = np.asarray(values)[valid]
    if times.size < 2 or np.unique(times).size < 2:
        return np.nan
    try:
        return float(np.polyfit(times, values, 1)[0])
    except np.linalg.LinAlgError:
        return np.nan


def _age_thresholds() -> dict[str, dict[int, float]]:
    pulse = {0: 220, **{age: 180 for age in range(1, 18)}}
    resp = {
        0: 54, 1: 38, 2: 38, 3: 38, 4: 29, 5: 29, 6: 26, 7: 26, 8: 26,
        9: 26, 10: 26, 11: 26, 12: 26, 13: 21, 14: 21, 15: 21, 16: 21, 17: 21,
    }
    spo2 = {age: 92 for age in range(0, 18)}
    sbp = {
        0: 69, 1: 71, 2: 73, 3: 75, 4: 77, 5: 79, 6: 81, 7: 83, 8: 85, 9: 87,
        10: 89, 11: 89, 12: 89, 13: 89, 14: 89, 15: 89, 16: 89, 17: 89,
    }
    return {"pulse": pulse, "resp": resp, "spo2": spo2, "sbp": sbp}


AGE_THRESHOLDS = _age_thresholds()


def _is_abnormal(value: float, age: float, kind: str) -> bool:
    if pd.isna(value) or pd.isna(age):
        return False
    age = int(max(0, min(17, age)))
    if kind == "pulse":
        return value >= AGE_THRESHOLDS[kind][age]
    if kind == "resp":
        return value >= AGE_THRESHOLDS[kind][age]
    if kind == "spo2":
        return value <= AGE_THRESHOLDS[kind][age]
    if kind == "sbp":
        return value <= AGE_THRESHOLDS[kind][age]
    return False


def _compute_encounter_features() -> pd.DataFrame:
    cache = CACHE_DIR / "encounter_features.csv"
    if cache.exists():
        return pd.read_csv(cache)

    df = pd.read_feather(TEST_PATH, columns=["encounter_key", "name", "t", "pair_id"])
    if "pair_id" in df.columns:
        df["_pair_id"] = pd.to_numeric(df["pair_id"], errors="coerce").fillna(-1)
        df = df.sort_values(["encounter_key", "t", "_pair_id"], kind="mergesort")
    else:
        df = df.sort_values(["encounter_key", "t"], kind="mergesort")

    rows = []
    for encounter_key, group in df.groupby("encounter_key", sort=False):
        names = group["name"].to_numpy()
        times = pd.to_numeric(group["t"], errors="coerce").to_numpy(dtype=float)

        pulse_mask = group["name"].str.startswith(VITAL_PREFIXES["pulse"]).to_numpy()
        resp_mask = group["name"].str.startswith(VITAL_PREFIXES["resp"]).to_numpy()
        sbp_mask = group["name"].str.startswith(VITAL_PREFIXES["sbp"]).to_numpy()
        spo2_mask = group["name"].str.startswith(VITAL_PREFIXES["spo2"]).to_numpy()

        pulse_slope = _slope(times[pulse_mask], pd.to_numeric(group.loc[pulse_mask, "name"].str.removeprefix(VITAL_PREFIXES["pulse"]), errors="coerce").to_numpy(dtype=float))
        resp_slope = _slope(times[resp_mask], pd.to_numeric(group.loc[resp_mask, "name"].str.removeprefix(VITAL_PREFIXES["resp"]), errors="coerce").to_numpy(dtype=float))
        sbp_slope = _slope(times[sbp_mask], pd.to_numeric(group.loc[sbp_mask, "name"].str.removeprefix(VITAL_PREFIXES["sbp"]), errors="coerce").to_numpy(dtype=float))
        spo2_slope = _slope(times[spo2_mask], pd.to_numeric(group.loc[spo2_mask, "name"].str.removeprefix(VITAL_PREFIXES["spo2"]), errors="coerce").to_numpy(dtype=float))

        age = _age_from_group(group)
        first_pulse = _first_numeric_value(group, VITAL_PREFIXES["pulse"])
        first_resp = _first_numeric_value(group, VITAL_PREFIXES["resp"])
        first_spo2 = _first_numeric_value(group, VITAL_PREFIXES["spo2"])
        first_sbp = _first_numeric_value(group, VITAL_PREFIXES["sbp"])

        abnormal_count = sum(
            [
                _is_abnormal(first_pulse, age, "pulse"),
                _is_abnormal(first_resp, age, "resp"),
                _is_abnormal(first_spo2, age, "spo2"),
                _is_abnormal(first_sbp, age, "sbp"),
            ]
        )

        abx_rows = group[group["name"].str.startswith("Med_")]
        abx_time = np.nan
        if not abx_rows.empty:
            med_names = abx_rows["name"].str.removeprefix("Med_").str.lower()
            mask = med_names.apply(lambda x: any(term in x for term in ABX_TERMS))
            if mask.any():
                abx_time = float(abx_rows.loc[mask, "t"].min())

        rows.append(
            {
                "encounter_key": encounter_key,
                "n_tokens": int(len(group)),
                "n_unique_tokens": int(pd.Index(names).nunique()),
                "last_time_min": float(pd.to_numeric(group["t"], errors="coerce").max()),
                "chief_complaint": _encounter_chief_complaint(group),
                "age": age,
                "pulse_slope": pulse_slope,
                "resp_slope": resp_slope,
                "sbp_slope": sbp_slope,
                "spo2_slope": spo2_slope,
                "first_pulse": first_pulse,
                "first_resp": first_resp,
                "first_spo2": first_spo2,
                "first_sbp": first_sbp,
                "abnormal_vital_count": int(abnormal_count),
                "abx_time_min": abx_time,
            }
        )

    features = pd.DataFrame(rows)
    for col, sign in {"pulse_slope": 1.0, "resp_slope": 1.0, "sbp_slope": -1.0, "spo2_slope": -1.0}.items():
        adjusted = features[col] * sign
        mu = adjusted.mean(skipna=True)
        sd = adjusted.std(skipna=True, ddof=0)
        if sd == 0 or pd.isna(sd):
            features[f"{col}_z"] = adjusted - mu
        else:
            features[f"{col}_z"] = (adjusted - mu) / sd
    features["decompensation_score"] = features[["pulse_slope_z", "resp_slope_z", "sbp_slope_z", "spo2_slope_z"]].mean(axis=1, skipna=True)
    features.to_csv(cache, index=False)
    return features


def _load_predictions(outcome: str) -> pd.DataFrame:
    spec = OUTCOME_SPECS[outcome]
    mint = pd.read_csv(spec["mint_path"], usecols=["encounter_key", spec["mint_col"], "labels"])
    xgb = pd.read_csv(spec["xgb_path"], usecols=["encounter_key", spec["xgb_col"], "labels"])
    mint = mint.rename(columns={spec["mint_col"]: "mint_prob", "labels": "label"})
    xgb = xgb.rename(columns={spec["xgb_col"]: "xgb_prob", "labels": "label_xgb"})
    if mint["encounter_key"].duplicated().any():
        mint = mint.groupby("encounter_key", as_index=False).agg({"mint_prob": "max", "label": "max"})
    if xgb["encounter_key"].duplicated().any():
        xgb = xgb.groupby("encounter_key", as_index=False).agg({"xgb_prob": "max", "label_xgb": "max"})
    merged = mint.merge(xgb, on="encounter_key", how="inner", validate="one_to_one")
    merged["label"] = merged["label"].fillna(merged["label_xgb"])
    merged = merged.drop(columns=["label_xgb"])
    merged["label"] = merged["label"].astype(int)
    return merged


def _merged_outcome_frame(outcome: str, features: pd.DataFrame) -> OutcomeFrame:
    cache = CACHE_DIR / f"{outcome}_merged.csv"
    if cache.exists():
        return OutcomeFrame(outcome=outcome, label=OUTCOME_SPECS[outcome]["label"], df=pd.read_csv(cache))
    preds = _load_predictions(outcome)
    merged = preds.merge(features, on="encounter_key", how="left", validate="one_to_one")
    merged.to_csv(cache, index=False)
    return OutcomeFrame(outcome=outcome, label=OUTCOME_SPECS[outcome]["label"], df=merged)
def _load_frames() -> dict[str, OutcomeFrame]:
    features = _compute_encounter_features()
    frames = {}
    for outcome in H1_OUTCOMES:
        frames[outcome] = _merged_outcome_frame(outcome, features)
    return frames


def build_h1(
    frames: dict[str, OutcomeFrame],
    summary_rows: list[dict] | None = None,
    save_dir: Path = SAVE_DIR,
) -> None:
    if summary_rows is None:
        summary_rows = []

    fig, ax = plt.subplots(1, 1, figsize=(6.8, 3.6))
    tertile_names = ["Improving", "Stable", "Deteriorating"]
    palette = ["#56B4E9", "#999999", "#D55E00"]
    x = np.arange(len(H1_OUTCOMES))
    width = 0.24

    for tertile_idx, tertile_name, color, offset in zip(range(3), tertile_names, palette, [-width, 0.0, width]):
        diffs = []
        lo_err = []
        hi_err = []
        for outcome in H1_OUTCOMES:
            df = frames[outcome].df.copy()
            df = df[df["decompensation_score"].notna()].copy()
            df["tertile"] = _tertiles(df["decompensation_score"])
            sub = df[df["tertile"] == tertile_idx]
            if sub.empty or sub["label"].nunique() < 2:
                diffs.append(np.nan)
                lo_err.append(np.nan)
                hi_err.append(np.nan)
                continue
            diff = _bootstrap_diff(sub["label"], sub["mint_prob"], sub["xgb_prob"])
            diffs.append(diff["diff"])
            lo_err.append(diff["diff"] - diff["lo"] if pd.notna(diff["lo"]) else np.nan)
            hi_err.append(diff["hi"] - diff["diff"] if pd.notna(diff["hi"]) else np.nan)
            summary_rows.append(
                {
                    "hypothesis": "H1",
                    "outcome": outcome,
                    "subgroup": tertile_name,
                    "metric": "auroc_diff",
                    "value": diff["diff"],
                    "n": len(sub),
                    "extra": "",
                }
            )
            summary_rows.append(
                {
                    "hypothesis": "H1",
                    "outcome": outcome,
                    "subgroup": tertile_name,
                    "metric": "auroc_diff_lo",
                    "value": diff["lo"],
                    "n": len(sub),
                    "extra": "",
                }
            )
            summary_rows.append(
                {
                    "hypothesis": "H1",
                    "outcome": outcome,
                    "subgroup": tertile_name,
                    "metric": "auroc_diff_hi",
                    "value": diff["hi"],
                    "n": len(sub),
                    "extra": "",
                }
            )
        ax.bar(
            x + offset,
            diffs,
            width=width,
            label=tertile_name,
            color=color,
            alpha=0.92,
            yerr=np.vstack([lo_err, hi_err]),
            error_kw={"ecolor": "#444444", "elinewidth": 1.0, "capsize": 3, "capthick": 1.0},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([ABBREV[o] for o in H1_OUTCOMES], rotation=0)
    ax.axhline(0, color="#888888", lw=0.8)
    ax.set_ylabel("AUROC gap (MINT - XGBoost)")
    ax.set_title("Temporal decompensation")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _subplot_label(ax, "A")
    fig.suptitle("H1  Temporal decompensation", x=0.01, ha="left", fontweight="bold")

    save_dir.mkdir(parents=True, exist_ok=True)
    _save_figure(fig, save_dir / "h1_temporal_decompensation")

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(save_dir / "summary_table.csv", index=False)


def main(argv: list[str] | None = None) -> None:
    _ = argv
    frames = _load_frames()
    build_h1(frames, [], SAVE_DIR)


if __name__ == "__main__":
    main()
