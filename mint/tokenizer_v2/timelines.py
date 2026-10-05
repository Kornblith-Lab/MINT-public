"""Timeline building: percentile resolution, magic timestamps, encounter timelines."""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import GlobalConfig, NumericalStrategy
from .utils import MAGIC_ADMIT


def resolve_percentiles(tokens_df: pd.DataFrame, strategy: NumericalStrategy) -> pd.DataFrame:
    """Replace __PERCENTILE__ placeholders with computed percentile tokens."""
    if strategy.mode not in ("separate_percentile", "combined_percentile"):
        return tokens_df

    mask = tokens_df["name"].str.startswith("__PERCENTILE__")
    if not mask.any():
        return tokens_df

    percentile_rows = tokens_df[mask].copy()
    parts = percentile_rows["name"].str.split("__PERCENTILE__").str[1]
    split_parts = parts.str.rsplit("__", n=1)
    percentile_rows["_base"] = split_parts.str[0]
    percentile_rows["_raw_value"] = pd.to_numeric(split_parts.str[1], errors="coerce")

    rounding = strategy.percentile_rounding
    resolved_names = []

    for base_name, group in percentile_rows.groupby("_base"):
        values = group["_raw_value"].dropna()
        if values.empty:
            resolved_names.extend(["UNKNOWN_PERCENTILE"] * len(group))
            continue

        sorted_values = values.sort_values().values
        for _, row in group.iterrows():
            val = row["_raw_value"]
            if pd.isnull(val):
                resolved_names.append("UNKNOWN_PERCENTILE")
                continue
            pct = (sorted_values < val).sum() / len(sorted_values) * 100
            rounded_pct = int(round(pct / rounding) * rounding)
            rounded_pct = max(rounding, min(100, rounded_pct))

            if strategy.mode == "separate_percentile":
                resolved_names.append(f"Percentile_{rounded_pct}")
            else:
                resolved_names.append(f"{base_name}_Percentile_{rounded_pct}")

    percentile_rows["name"] = resolved_names
    tokens_df = tokens_df.copy()
    tokens_df.loc[mask, "name"] = percentile_rows["name"]
    tokens_df = tokens_df[tokens_df["name"] != "UNKNOWN_PERCENTILE"]

    return tokens_df

def resolve_magic_timestamps(tokens_df: pd.DataFrame, enc_col: str) -> pd.DataFrame:
    """Resolve __MAGIC_ADMIT__ placeholders to actual Admit timestamps per encounter."""
    magic_mask = tokens_df["timestamp"].eq(MAGIC_ADMIT)
    if not magic_mask.any():
        return tokens_df

    # Keep the first Admit timestamp per encounter, matching your current behavior.
    admit_times = (
        tokens_df.loc[tokens_df["name"].eq("Admit"), [enc_col, "timestamp"]]
        .drop_duplicates(subset=[enc_col])
        .set_index(enc_col)["timestamp"]
    )

    tokens_df = tokens_df.copy()

    # Vectorized lookup for all magic rows
    resolved = tokens_df.loc[magic_mask, enc_col].map(admit_times)

    # Fill in resolved timestamps
    tokens_df.loc[magic_mask, "timestamp"] = resolved.to_numpy()

    # Drop magic rows that had no matching Admit (resolved only where map succeeded)
    drop_mask = pd.Series(False, index=tokens_df.index)
    drop_mask.loc[resolved.index[resolved.isna()]] = True
    return tokens_df.loc[~drop_mask].reset_index(drop=True)

def build_timelines(tokens_df: pd.DataFrame, global_config: GlobalConfig) -> pd.DataFrame:
    enc_col = global_config.encounter_key_column
    has_pair_id = "pair_id" in tokens_df.columns

    df = tokens_df.copy()
    total_start = len(df)
    enc_start = df[enc_col].nunique()

    # 1) Filter by min token count
    counts = df["name"].value_counts()
    valid_tokens = counts[counts >= global_config.min_token_count].index
    df = df[df["name"].isin(valid_tokens)].copy()
    print(f"  [1] min_token_count ({global_config.min_token_count}): {total_start:,} -> {len(df):,} tokens "
          f"(dropped {total_start - len(df):,}, {len(counts) - len(valid_tokens):,} rare token types removed)")

    if df.empty:
        cols = ["encounter_key", "name", "t", "year"] + (["pair_id"] if has_pair_id else [])
        return pd.DataFrame(columns=cols)

    # 2) Keep encounters with exactly one of Admit / Discharge
    before2 = len(df)
    enc_before2 = df[enc_col].nunique()
    admit = df["name"].eq("Admit")
    discharge = df["name"].eq("Discharge")

    flags = (
        df.assign(_admit=admit, _discharge=discharge)
          .groupby(enc_col, sort=False)[["_admit", "_discharge"]]
          .any()
    )
    valid_encounters = flags.index[flags["_admit"] ^ flags["_discharge"]]
    df = df[df[enc_col].isin(valid_encounters)].copy()
    print(f"  [2] Admit XOR Discharge: {before2:,} -> {len(df):,} tokens "
          f"(dropped {before2 - len(df):,}, {enc_before2 - df[enc_col].nunique():,} encounters removed)")

    if df.empty:
        cols = ["encounter_key", "name", "t", "year"] + (["pair_id"] if has_pair_id else [])
        return pd.DataFrame(columns=cols)

    # 3) Convert timestamps once
    before3 = len(df)
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")

    df = df.sort_values([enc_col, "timestamp"])
    arrival_prefixes = ("Age_", "Sex_", "Acuity_", "CC_", "Arrival_Method_")
    arrival_mask = df["name"].str.startswith(arrival_prefixes)
    arrival_t0 = df.loc[arrival_mask].groupby(enc_col, sort=False)["timestamp"].min()
    df["_t0"] = df[enc_col].map(arrival_t0)
    df["t"] = ((df["timestamp"] - df["_t0"]).dt.total_seconds() / 60)

    df = df.dropna(subset=["t"])
    before_neg = len(df)
    df = df[df["t"] >= 0]
    print(f"  [3-4] Timestamp conversion + t0 from ArrivalInstant: {before3:,} -> {len(df):,} tokens "
        f"(dropped {before3 - before_neg:,} with unparseable/NaT timestamps, "
        f"{before_neg - len(df):,} with t < 0)")

    # 5) Drop encounters longer than max_encounter_minutes (save sample first)
    before5 = len(df)
    enc_before5 = df[enc_col].nunique()
    if global_config.max_encounter_minutes is not None:
        max_t_per_enc = df.groupby(enc_col, sort=False)["t"].max()
        long_encs = max_t_per_enc[max_t_per_enc > global_config.max_encounter_minutes].index
        if len(long_encs) > 0:
            sample_keys = long_encs[:500]
            sample_df = df[df[enc_col].isin(sample_keys)][[enc_col, "name", "timestamp", "t"]].copy()
            sample_df = sample_df.sort_values([enc_col, "timestamp"])
            sample_df.columns = ["encounter_key", "name", "timestamp", "t"]
            out_dir = Path(global_config.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            # sample_df.to_csv(out_dir / "long_encounters_sample.csv", index=False)
            # print(f"  Saved {len(sample_keys):,} long encounters ({len(sample_df):,} tokens) to long_encounters_sample.csv")

        max_t = df.groupby(enc_col, sort=False)["t"].transform("max")
        df = df[max_t <= global_config.max_encounter_minutes]
        print(f"  [5] max_encounter_minutes ({global_config.max_encounter_minutes}): {before5:,} -> {len(df):,} tokens "
              f"(dropped {before5 - len(df):,}, {enc_before5 - df[enc_col].nunique():,} encounters removed)")
    else:
        print(f"  [5] max_encounter_minutes: disabled (no limit set)")

    if df.empty:
        cols = ["encounter_key", "name", "t", "year"] + (["pair_id"] if has_pair_id else [])
        return pd.DataFrame(columns=cols)

    # 6) Drop encounters with fewer than min_encounter_tokens tokens
    before6 = len(df)
    enc_before6 = df[enc_col].nunique()
    enc_counts = df.groupby(enc_col, sort=False)[enc_col].transform("size")
    df = df[enc_counts >= global_config.min_encounter_tokens]
    print(f"  [6] min_encounter_tokens ({global_config.min_encounter_tokens}): {before6:,} -> {len(df):,} tokens "
          f"(dropped {before6 - len(df):,}, {enc_before6 - df[enc_col].nunique():,} encounters removed)")

    if df.empty:
        cols = ["encounter_key", "name", "t", "year"] + (["pair_id"] if has_pair_id else [])
        return pd.DataFrame(columns=cols)

    # 7) Final formatting
    df["t"] = df["t"].round().astype(np.int64)
    df["encounter_key"] = df[enc_col]
    df["year"] = df["_t0"].dt.year

    # 8) Auto-append Discharge at t_max + 1 for encounters with Admit
    admit_encs = df.loc[df["name"].eq("Admit"), "encounter_key"].unique()
    if len(admit_encs) > 0:
        t_max = df[df["encounter_key"].isin(admit_encs)].groupby("encounter_key", sort=False)["t"].max()
        discharge_rows = pd.DataFrame({
            "encounter_key": t_max.index,
            "name": "Discharge",
            "t": (t_max.values + 1).astype(np.int64),
            "year": df.drop_duplicates("encounter_key").set_index("encounter_key").loc[t_max.index, "year"].values,
        })
        if has_pair_id:
            discharge_rows["pair_id"] = np.nan
        df = pd.concat([df, discharge_rows], ignore_index=True)
        print(f"  [8] Auto-appended Discharge token to {len(admit_encs):,} encounters with Admit")

    cols = ["encounter_key", "name", "t", "year"]
    if has_pair_id:
        cols.append("pair_id")

    final = df[cols].reset_index(drop=True)
    print(f"  [Summary] {total_start:,} -> {len(final):,} tokens "
          f"({total_start - len(final):,} dropped, {(total_start - len(final)) / total_start * 100:.1f}% loss), "
          f"{enc_start:,} -> {final['encounter_key'].nunique():,} encounters")
    return final

# def build_timelines(tokens_df: pd.DataFrame, global_config: GlobalConfig) -> pd.DataFrame:
#     """Convert absolute timestamps to relative minutes per encounter.

#     Applies:
#     - Disposition XOR filter (exactly one of Admit/Discharge per encounter)
#     - Duration filter (max_encounter_minutes)
#     - Cuts timeline at disposition event
#     - Extracts year for train/val/test splitting
#     """
#     enc_col = global_config.encounter_key_column
#     disposition_tokens = set(global_config.disposition_filter)

#     # Filter by min token count
#     counts = tokens_df["name"].value_counts()
#     valid_tokens = counts[counts >= global_config.min_token_count].index
#     tokens_df = tokens_df[tokens_df["name"].isin(valid_tokens)].copy()

#     # Filter encounters: must have exactly one of Admit/Discharge (XOR)
#     def has_valid_disposition(group):
#         names = set(group["name"])
#         has_admit = "Admit" in names
#         has_discharge = "Discharge" in names
#         return has_admit ^ has_discharge

#     tokens_df = tokens_df.groupby(enc_col).filter(has_valid_disposition)

#     # Convert timestamps to datetime
#     if not pd.api.types.is_datetime64_any_dtype(tokens_df["timestamp"]):
#         tokens_df["timestamp"] = pd.to_datetime(tokens_df["timestamp"], format="mixed")

#     tokens_df = tokens_df.sort_values("timestamp")

#     has_pair_id = "pair_id" in tokens_df.columns
#     encounters = []

#     for enc_key, group in tqdm(tokens_df.groupby(enc_col), desc="Building timelines"):
#         try:
#             t0 = group["timestamp"].min()
#             group = group.copy()
#             group["t"] = (group["timestamp"] - t0).dt.total_seconds() / 60
#             group = group.dropna(subset=["t"])

#             if group["t"].max() > global_config.max_encounter_minutes:
#                 continue

#             # Cut at disposition
#             disposition_mask = group["name"].isin(disposition_tokens)
#             if disposition_mask.any():
#                 cutoff = group.loc[disposition_mask, "t"].min()
#                 group = group[group["t"] <= cutoff]

#             group["t"] = group["t"].astype(int)
#             group["encounter_key"] = enc_key
#             group["year"] = t0.year

#             cols = ["encounter_key", "name", "t", "year"]
#             if has_pair_id:
#                 cols.append("pair_id")
#             encounters.append(group[cols])
#         except Exception:
#             continue

#     if not encounters:
#         cols = ["encounter_key", "name", "t", "year"]
#         if has_pair_id:
#             cols.append("pair_id")
#         return pd.DataFrame(columns=cols)

#     return pd.concat(encounters, ignore_index=True)


def save_outputs(timeline_df: pd.DataFrame, global_config: GlobalConfig, output_dir: Path):
    """Save final outputs: feather, vocab, numpy splits, encounter map."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build vocabulary (only tokens meeting min_token_count)
    token_counts = timeline_df["name"].value_counts()
    vocab = token_counts.rename_axis("name").reset_index(name="count")
    vocab = vocab[vocab["count"] >= global_config.min_token_count].reset_index(drop=True)
    vocab.index = vocab.index + 1
    vocab.index.name = "index"
    vocab.to_csv(output_dir / "vocab.csv")

    token_to_id = dict(zip(vocab["name"], vocab.index))

    # Save feather
    feather_cols = ["encounter_key", "name", "t"]
    if "pair_id" in timeline_df.columns:
        feather_cols.append("pair_id")
    feather_df = timeline_df[feather_cols].copy()
    feather_df.to_feather(output_dir / "tokens.feather")

    # Save numpy arrays split by year
    timeline_df = timeline_df.copy()
    timeline_df["token_id"] = timeline_df["name"].map(token_to_id)
    timeline_df = timeline_df[timeline_df["token_id"].notna()]

    enc_to_id = {k: i for i, k in enumerate(timeline_df["encounter_key"].unique(), start=1)}
    timeline_df["patient_id"] = timeline_df["encounter_key"].map(enc_to_id)

    rng = np.random.default_rng(42)

    for split_name, rule in global_config.split.items():
        if timeline_df["year"].isna().all():
            continue
        mask = timeline_df.eval(rule)
        split_df = timeline_df[mask]

        if split_name == "train":
            # Random 90-10 split of train encounters into train/val
            train_encounters = split_df["patient_id"].unique()
            rng.shuffle(train_encounters)
            n_train = int(len(train_encounters) * 0.9)
            train_enc_ids = set(train_encounters[:n_train])
            val_enc_ids = set(train_encounters[n_train:])

            train_df = split_df[split_df["patient_id"].isin(train_enc_ids)]
            val_df = split_df[split_df["patient_id"].isin(val_enc_ids)]

            for sub_name, sub_df in [("train", train_df), ("val", val_df)]:
                sub_data = sub_df[["patient_id", "t", "token_id"]].values.astype(np.uint32)
                np.save(output_dir / f"{sub_name}.npy", sub_data)
                sub_df[feather_cols].reset_index(drop=True).to_feather(output_dir / f"{sub_name}.feather")
                n_enc = len(sub_df["patient_id"].unique())
                print(f"  {sub_name}: {len(sub_data):,} tokens, {n_enc:,} encounters")
        else:
            split_data = split_df[["patient_id", "t", "token_id"]].values.astype(np.uint32)
            np.save(output_dir / f"{split_name}.npy", split_data)
            split_df[feather_cols].reset_index(drop=True).to_feather(output_dir / f"{split_name}.feather")
            n_encounters = len(split_df["patient_id"].unique())
            print(f"  {split_name}: {len(split_data):,} tokens, {n_encounters:,} encounters")

    # Save encounter key mapping
    enc_map = pd.DataFrame(list(enc_to_id.items()), columns=["encounter_key", "assigned_id"])
    enc_map.to_csv(output_dir / "encounter_key_map.csv", index=False)

    print(f"  Vocab size: {len(vocab)}")
    print(f"  Total tokens: {len(timeline_df):,}")
    print(f"  Total encounters: {len(enc_to_id):,}")
