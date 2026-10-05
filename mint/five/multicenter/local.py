"""Local-cohort supplementary table for the manuscript.

Builds a two-column table for the local data split:
  - Training = all encounter keys in train.feather and val.feather
  - Evaluation = all encounter keys in test.feather

Outputs are written to ``artifacts/local_analysis``:
  - local_characteristics.csv
  - local_characteristics.png
  - local_characteristics.pdf
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "output"
CDW_DIR = ROOT / "cdw"
OUT_DIR = ROOT / "artifacts" / "local_analysis"

TITLE = "Supplementary Table E: Characteristics of visits to the local health system"
COLUMN_ORDER = ["Training", "Evaluation"]
RACE_ORDER = ["Asian", "Black", "White", "Other/Unknown"]


def _load_feather(name: str) -> pd.DataFrame:
    return pd.read_feather(DATA_DIR / name, columns=["encounter_key", "name", "t"])


def _load_split_frames() -> dict[str, pd.DataFrame]:
    train = _load_feather("train.feather")
    val = _load_feather("val.feather")
    test = _load_feather("test.feather")
    return {
        "Training": pd.concat([train, val], ignore_index=True),
        "Evaluation": test,
    }


def _fmt_pct_count(n: int, denom: int) -> str:
    if denom <= 0:
        return "0.0% (0)"
    return f"{100.0 * n / denom:.1f}% ({n})"


def _fmt_median_iqr(values: pd.Series) -> str:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if clean.size == 0:
        return ""
    q1, med, q3 = np.percentile(clean, [25, 50, 75])
    return f"{med:g} ({q1:g}-{q3:g})"


def _fmt_year_range(values: pd.Series) -> str:
    dates = pd.to_datetime(values, errors="coerce").dropna()
    if dates.empty:
        return ""
    years = dates.dt.year.astype(int)
    return f"{years.min()}-{years.max()}"


def _collapse_race(value: object) -> str:
    if pd.isna(value):
        return "Other/Unknown"
    race = str(value).strip()
    if race == "Asian":
        return "Asian"
    if race in {"Black", "Black or African American"}:
        return "Black"
    if race == "White":
        return "White"
    return "Other/Unknown"


def _subset_encounter_keys(df: pd.DataFrame) -> set[str]:
    return set(df["encounter_key"].unique())


def _load_demo_frames(encounter_keys: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    demos = pd.read_csv(
        CDW_DIR / "demos_v3.csv",
        usecols=["EncounterKey", "Sex", "FirstRace", "Ethnicity"],
    )
    visits = pd.read_csv(
        CDW_DIR / "all_pediatric_ed_visits_with_note.csv",
        usecols=["EncounterKey", "Age", "ArrivalInstant", "deid_service_date"],
    )

    demos = demos[demos["EncounterKey"].isin(encounter_keys)].copy()
    visits = visits[visits["EncounterKey"].isin(encounter_keys)].copy()

    return demos, visits


def _token_counts_before_disposition(tokens: pd.DataFrame) -> pd.Series:
    keys = tokens["encounter_key"].drop_duplicates().tolist()
    cutoff = (
        tokens.loc[tokens["name"].isin(["Admit", "Discharge"]), ["encounter_key", "t"]]
        .groupby("encounter_key", sort=False)["t"]
        .min()
    )
    annotated = tokens.join(cutoff.rename("cutoff"), on="encounter_key")
    before = annotated["cutoff"].isna() | (annotated["t"] < annotated["cutoff"])
    counts = annotated.loc[before].groupby("encounter_key", sort=False).size()
    return counts.reindex(keys, fill_value=0)


def _split_stats(split_frames: dict[str, pd.DataFrame]) -> dict[str, dict[str, str]]:
    cohort_keys = {split: _subset_encounter_keys(df) for split, df in split_frames.items()}
    all_keys = set().union(*cohort_keys.values())
    demos, visits = _load_demo_frames(all_keys)
    demos = demos.rename(columns={"EncounterKey": "encounter_key"})
    visits = visits.rename(columns={"EncounterKey": "encounter_key"})

    out: dict[str, dict[str, str]] = {}
    for split_name, tokens in split_frames.items():
        key_set = cohort_keys[split_name]
        token_subset = tokens[tokens["encounter_key"].isin(key_set)].copy()
        demo_subset = demos[demos["encounter_key"].isin(key_set)].copy()
        visit_subset = visits[visits["encounter_key"].isin(key_set)].copy()
        race = demo_subset["FirstRace"].map(_collapse_race)

        if demo_subset["encounter_key"].nunique() != len(key_set):
            raise ValueError(f"{split_name} missing demographic rows")
        if visit_subset["encounter_key"].nunique() != len(key_set):
            raise ValueError(f"{split_name} missing visit rows")

        token_counts = _token_counts_before_disposition(token_subset)
        if len(token_counts) != len(key_set):
            missing = key_set.difference(token_counts.index)
            raise ValueError(f"{split_name} missing token trajectories for: {sorted(missing)[:5]}")
        
        out[split_name] = {
            "Total visits": f"{len(key_set):,}",
            "Date range": _fmt_year_range(visit_subset["ArrivalInstant"]),
            "Tokens,\nmedian [IQR]": _fmt_median_iqr(token_counts),
            "Race": "",
            "Asian": _fmt_pct_count(int((race == "Asian").sum()), len(key_set)),
            "Black": _fmt_pct_count(int((race == "Black").sum()), len(key_set)),
            "White": _fmt_pct_count(int((race == "White").sum()), len(key_set)),
            "Other/Unknown": _fmt_pct_count(int((race == "Other/Unknown").sum()), len(key_set)),
            "Ethnicity -\nHispanic or Latino": _fmt_pct_count(int((demo_subset["Ethnicity"] == "Hispanic or Latino").sum()), len(key_set)),
            "Sex - Female": _fmt_pct_count(int((demo_subset["Sex"] == "Female").sum()), len(key_set)),
            "Age, median [IQR]": _fmt_median_iqr(visit_subset["Age"]),
        }

    ordered_rows = [
        "Total visits",
        "Date range",
        "Tokens,\nmedian [IQR]",
        "Race",
        *RACE_ORDER,
        "Ethnicity -\nHispanic or Latino",
        "Sex - Female",
        "Age, median [IQR]",
    ]
    return pd.DataFrame(
        [{"Metric": label, **{split: out[split][label] for split in COLUMN_ORDER}} for label in ordered_rows]
    )


def _save_fig(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _render_table(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12.5, 8.25))
    ax.axis("off")
    fig.text(0.02, 0.965, TITLE, ha="left", va="top", fontsize=18, fontweight="bold")

    body = df[["Metric", *COLUMN_ORDER]].fillna("").values.tolist()
    table = ax.table(
        cellText=body,
        colLabels=["", *COLUMN_ORDER],
        loc="upper left",
        cellLoc="left",
        colLoc="left",
        bbox=[0.0, 0.0, 1.0, 0.92],
        colWidths=[0.44, 0.28, 0.28],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1.0, 1.65)

    section_labels = {
        "Total visits",
        "Date range",
        "Tokens,\nmedian [IQR]",
        "Race",
        "Ethnicity -\nHispanic or Latino",
        "Sex - Female",
        "Age, median [IQR]",
    }

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.9)
        cell.set_facecolor("white")
        cell.PAD = 0.03
        if row == 0:
            cell.set_text_props(weight="bold")
        elif col == 0:
            label = str(df.iloc[row - 1]["Metric"])
            if label in section_labels:
                cell.set_text_props(weight="bold")
        if col == 0:
            cell._loc = "left"

    return fig


def build_local_characteristics() -> pd.DataFrame:
    split_frames = _load_split_frames()
    return _split_stats(split_frames)


def write_outputs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = build_local_characteristics()
    table.to_csv(OUT_DIR / "local_characteristics.csv", index=False)
    fig = _render_table(table)
    _save_fig(fig, "local_characteristics")


def main() -> None:
    plt.style.use(str(Path(__file__).resolve().parents[1] / "design-skill" / "nature.mplstyle"))
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 18,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
        }
    )
    write_outputs()


if __name__ == "__main__":
    main()
