"""Step 4: Main tokenization engine — YAML-driven clinical data tokenizer.

Usage:
    python -m mint.tokenizer_v2.tokenize \
        --config tokenizer_config.yaml \
        --data-dir cdw/ \
        --output-dir output/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import GlobalConfig, SourceConfig, NumericalStrategy, load_config
from .processors import get_processor
from .strategies import emit_numeric_tokens
from .timelines import resolve_percentiles, resolve_magic_timestamps, build_timelines, save_outputs
from .utils import MAGIC_ADMIT, round_value, get_row_timestamp, check_condition, zscore_to_token_suffix


def tokenize_keyed_source(
    df: pd.DataFrame,
    source: SourceConfig,
    global_config: GlobalConfig,
    zscore_stats: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Tokenize a keyed source. Returns a DataFrame with columns: enc_key, name, timestamp, pair_id."""
    tokens = []
    key_col = source.key_column
    value_col = source.value_column
    enc_col = global_config.encounter_key_column
    strategy = global_config.numerical_strategy

    default_prefix = source.defaults.get("prefix", "")
    default_type = source.defaults.get("type", "numeric")
    default_rounding = source.defaults.get("rounding", "int")

    for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"Tokenizing {source.name}"):
        key = getattr(row, key_col)
        if pd.isnull(key):
            continue
        key = str(key)

        if source.variables and key not in source.variables:
            continue

        var_config = source.variables.get(key, {})
        if var_config is None:
            var_config = {}

        if var_config.get("skip"):
            continue

        final_name = var_config.get("rename", key)

        condition = var_config.get("condition")
        if condition and not check_condition(row, condition):
            continue

        ts_override = var_config.get("timestamp_override")
        if ts_override and ts_override.get("magic") == "admit":
            timestamp = MAGIC_ADMIT
        else:
            timestamp = get_row_timestamp(row, source, var_config)
            if timestamp is None or (not isinstance(timestamp, str) and pd.isnull(timestamp)):
                continue

        token_type = var_config.get("type", default_type)
        prefix = var_config.get("prefix", default_prefix)

        token_name = var_config.get("token_name")
        if token_name:
            tokens.append((getattr(row, enc_col), token_name, timestamp, None))
            continue

        if token_type == "custom":
            process_name = var_config.get("process")
            if process_name:
                processor = get_processor(process_name)
                ts_col = var_config.get("timestamp_column", source.timestamp_column)
                custom_tokens = processor.process(row, enc_col, ts_col, prefix)
                for ct in custom_tokens:
                    tokens.append((ct[enc_col], ct["name"], ct["timestamp"], None))
            continue

        if token_type == "categorical":
            # Categorical with value: use per-variable value_column or emit just the name
            var_value_col = var_config.get("value_column")
            if var_value_col:
                raw_val = getattr(row, var_value_col, None)
                if raw_val is None or (not isinstance(raw_val, str) and pd.isnull(raw_val)):
                    continue
                base = f"{prefix}_{raw_val}" if prefix else f"{final_name}_{raw_val}"
            else:
                base = f"{prefix}_{final_name}" if prefix else final_name
            tokens.append((getattr(row, enc_col), base, timestamp, None))
            continue

        if value_col is None:
            base = f"{prefix}_{final_name}" if prefix else final_name
            tokens.append((getattr(row, enc_col), base, timestamp, None))
            continue

        # Use per-variable value_column override if present
        effective_value_col = var_config.get("value_column", value_col)
        raw_value = getattr(row, effective_value_col) if effective_value_col else None
        if raw_value is not None and not isinstance(raw_value, str) and pd.isnull(raw_value):
            raw_value = None

        numeric_fallback = var_config.get("numeric_fallback") or source.defaults.get("numeric_fallback")
        if numeric_fallback:
            use_numeric = var_config.get("numeric", source.defaults.get("numeric", True))
            numeric_value = None
            if use_numeric and raw_value is not None:
                try:
                    numeric_value = float(raw_value)
                except (ValueError, TypeError):
                    numeric_value = None

            if numeric_value is not None:
                rounding = var_config.get("rounding", default_rounding)
                base_name = f"{prefix}_{final_name}" if prefix else final_name
                if rounding == "zscore":
                    if zscore_stats and final_name in zscore_stats:
                        lab_info = zscore_stats[final_name]
                        z = (numeric_value - lab_info["mean"]) / lab_info["std"]
                        suffix = zscore_to_token_suffix(z, step=strategy.zscore_step)
                        tokens.append((getattr(row, enc_col), f"{base_name}_{suffix}", timestamp, None))
                        continue
                    else:
                        numeric_value = None  # fall through to _Abnormal
                else:
                    rounded = round_value(numeric_value, rounding)
                    _emit_numeric(tokens, getattr(row, enc_col), base_name, rounded, timestamp, strategy)
                    continue

            if numeric_value is None:
                fb_condition = numeric_fallback.get("condition")
                if fb_condition and not check_condition(row, fb_condition):
                    continue
                suffix = numeric_fallback.get("suffix", "")
                base_name = f"{prefix}_{final_name}" if prefix else final_name
                tokens.append((getattr(row, enc_col), f"{base_name}{suffix}", timestamp, None))
                continue

        if raw_value is None:
            continue

        numeric_value = None
        try:
            numeric_value = float(raw_value)
        except (ValueError, TypeError):
            pass

        if numeric_value is not None:
            rounding = var_config.get("rounding", default_rounding)
            base_name = f"{prefix}_{final_name}" if prefix else final_name
            if rounding == "zscore" and zscore_stats and final_name in zscore_stats:
                lab_info = zscore_stats[final_name]
                z = (numeric_value - lab_info["mean"]) / lab_info["std"]
                suffix = zscore_to_token_suffix(z, step=strategy.zscore_step)
                tokens.append((getattr(row, enc_col), f"{base_name}_{suffix}", timestamp, None))
            elif rounding == "zscore":
                pass  # no stats available, skip
            else:
                rounded = round_value(numeric_value, rounding)
                _emit_numeric(tokens, getattr(row, enc_col), base_name, rounded, timestamp, strategy)
        else:
            base_name = f"{prefix}_{final_name}" if prefix else final_name
            tokens.append((getattr(row, enc_col), f"{base_name}_{raw_value}", timestamp, None))

    return _tokens_to_df(tokens, enc_col)


def tokenize_multi_column_source(
    df: pd.DataFrame,
    source: SourceConfig,
    global_config: GlobalConfig,
) -> pd.DataFrame:
    """Tokenize a multi-column source. Returns DataFrame with enc_key, name, timestamp, pair_id."""
    tokens = []
    enc_col = global_config.encounter_key_column
    strategy = global_config.numerical_strategy

    for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"Tokenizing {source.name}"):
        for var_name, var_config in source.variables.items():
            if var_config is None:
                var_config = {}
            if var_config.get("skip"):
                continue

            condition = var_config.get("condition")
            if condition and not check_condition(row, condition):
                continue

            rules = var_config.get("rules")
            if rules:
                col_value = getattr(row, var_name, None)
                if col_value is None or (not isinstance(col_value, str) and pd.isnull(col_value)):
                    continue
                for rule in rules:
                    if str(col_value) == str(rule["value"]):
                        ts_col = rule.get("timestamp_column", source.timestamp_column)
                        ts = getattr(row, ts_col, None) if ts_col else None
                        if ts is not None and not (isinstance(ts, float) and pd.isnull(ts)):
                            tokens.append((getattr(row, enc_col), rule["token_name"], ts, None))
                        break
                continue

            col_value = getattr(row, var_name, None)
            if col_value is None or (not isinstance(col_value, str) and pd.isnull(col_value)):
                continue

            ts_override = var_config.get("timestamp_override")
            if ts_override and ts_override.get("magic") == "admit":
                timestamp = MAGIC_ADMIT
            else:
                ts_col = var_config.get("timestamp_column", source.timestamp_column)
                timestamp = getattr(row, ts_col, None) if ts_col else None
                if timestamp is None or (isinstance(timestamp, float) and pd.isnull(timestamp)):
                    continue

            token_name = var_config.get("token_name")
            if token_name:
                tokens.append((getattr(row, enc_col), token_name, timestamp, None))
            else:
                token_type = var_config.get("type", "categorical")
                prefix = var_config.get("prefix", var_name)

                if token_type == "numeric":
                    try:
                        numeric_value = float(col_value)
                        rounding = var_config.get("rounding", "int")
                        rounded = round_value(numeric_value, rounding)
                        _emit_numeric(tokens, getattr(row, enc_col), prefix, rounded, timestamp, strategy)
                    except (ValueError, TypeError):
                        tokens.append((getattr(row, enc_col), f"{prefix}_{col_value}", timestamp, None))
                else:
                    tokens.append((getattr(row, enc_col), f"{prefix}_{col_value}", timestamp, None))

    return _tokens_to_df(tokens, enc_col)


def _emit_numeric(tokens: list, enc_key, base_name: str, rounded_value, timestamp, strategy: NumericalStrategy):
    """Emit numeric tokens as tuples instead of dicts for memory efficiency."""
    import uuid
    mode = strategy.mode

    if mode == "combined_exact":
        tokens.append((enc_key, f"{base_name}_{rounded_value}", timestamp, None))
    elif mode == "combined_percentile":
        tokens.append((enc_key, f"__PERCENTILE__{base_name}__{rounded_value}", timestamp, None))
    elif mode == "separate_exact":
        pair_id = uuid.uuid4().hex[:8]
        tokens.append((enc_key, base_name, timestamp, pair_id))
        tokens.append((enc_key, str(rounded_value), timestamp, pair_id))
    elif mode == "separate_percentile":
        pair_id = uuid.uuid4().hex[:8]
        tokens.append((enc_key, base_name, timestamp, pair_id))
        tokens.append((enc_key, f"__PERCENTILE__{base_name}__{rounded_value}", timestamp, pair_id))
    else:
        raise ValueError(f"Unknown strategy mode: {mode}")


def _tokens_to_df(tokens: list[tuple], enc_col: str) -> pd.DataFrame:
    """Convert list of tuples to DataFrame."""
    if not tokens:
        return pd.DataFrame(columns=[enc_col, "name", "timestamp", "pair_id"])
    return pd.DataFrame(tokens, columns=[enc_col, "name", "timestamp", "pair_id"])


def _filter_labs_by_abnormal(df: pd.DataFrame, source: SourceConfig, global_config: GlobalConfig) -> pd.DataFrame:
    """For non-percentile modes, filter labs to only those with >min_abnormal_fraction abnormal rate."""
    mode = global_config.numerical_strategy.mode
    if mode in ("combined_percentile", "separate_percentile"):
        return df

    min_frac = source.defaults.get("min_abnormal_fraction", 0)
    if min_frac <= 0:
        return df

    key_col = source.key_column
    abnormal_col = "Abnormal"
    if abnormal_col not in df.columns:
        return df

    # Compute abnormal fraction per lab type
    abnormal_mask = df[abnormal_col].astype(str).isin(["1", "True", "true", "1.0"])
    stats = df.groupby(key_col).agg(
        total=(key_col, "size"),
        abnormal=(abnormal_col, lambda x: x.astype(str).isin(["1", "True", "true", "1.0"]).sum()),
    )
    stats["fraction"] = stats["abnormal"] / stats["total"]
    valid_labs = stats[stats["fraction"] > min_frac].index
    before = len(df)
    df = df[df[key_col].isin(valid_labs)]
    print(f"  Labs abnormal filter (>{min_frac*100:.0f}%): {before:,} -> {len(df):,} rows ({len(valid_labs)} lab types kept)")
    return df


def tokenize_source(
    df: pd.DataFrame,
    source: SourceConfig,
    global_config: GlobalConfig,
    zscore_stats: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Tokenize a single source, dispatching between keyed and multi_column modes."""
    if source.pre_filter:
        for col, allowed in source.pre_filter.items():
            df = df[df[col].isin(allowed)]

    # Apply labs abnormal fraction filter for non-percentile modes
    if source.name == "labs" and source.defaults.get("min_abnormal_fraction", 0) > 0:
        df = _filter_labs_by_abnormal(df, source, global_config)

    if source.mode == "keyed":
        return tokenize_keyed_source(df, source, global_config, zscore_stats=zscore_stats)
    elif source.mode == "multi_column":
        return tokenize_multi_column_source(df, source, global_config)
    else:
        raise ValueError(f"Unknown source mode: {source.mode}")


def load_source_data(source: SourceConfig, data_dir: Path) -> pd.DataFrame:
    """Load source data from CSV."""
    filepath = data_dir / source.file
    if not filepath.exists():
        print(f"  WARNING: {filepath} not found, skipping {source.name}")
        return pd.DataFrame()

    print(f"  Reading {filepath}...")
    df = pd.read_csv(filepath, low_memory=False)

    if source.name == "visits":
        demos_path = data_dir / "demos.csv"
        if demos_path.exists():
            demos = pd.read_csv(demos_path, usecols=["EncounterKey", "Sex"], low_memory=False)
            demos = demos.drop_duplicates(subset="EncounterKey", keep="first")
            demos["Sex"] = demos["Sex"].replace({"Nonbinary": "Other", "Unknown": "Other"})
            df = df.merge(demos, on="EncounterKey", how="left")
            print(f"  Joined Sex from demos.csv ({df['Sex'].notna().sum():,} matched)")

    return df


def main():
    parser = argparse.ArgumentParser(description="YAML-driven tokenizer V2 for MINT clinical data")
    parser.add_argument("--config", type=Path, required=True, help="YAML config file")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing source CSVs")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (overrides config)")
    args = parser.parse_args()

    global_config, sources = load_config(args.config)
    if args.output_dir:
        global_config.output_dir = args.output_dir

    data_dir = args.data_dir
    output_dir = global_config.output_dir

    # Load z-score stats if available (optional)
    zscore_stats = None
    zscore_stats_path = data_dir / "lab_zscore_stats.json"
    if zscore_stats_path.exists():
        with open(zscore_stats_path) as f:
            zscore_stats = json.load(f)
        print(f"Loaded z-score stats for {len(zscore_stats)} lab types")

    # Tokenize all sources, collecting DataFrames
    all_dfs = []

    for source in sources:
        print(f"\n--- {source.name} ---")
        df = load_source_data(source, data_dir)
        if df.empty:
            continue

        print(f"  {len(df):,} rows")
        tokens_df = tokenize_source(df, source, global_config, zscore_stats)
        print(f"  {len(tokens_df):,} tokens generated")

        if not tokens_df.empty:
            all_dfs.append(tokens_df)

        # Free source DataFrame memory
        del df

    enc_col = global_config.encounter_key_column

    if not all_dfs:
        print("No tokens generated. Exiting.")
        return

    print("\nConcatenating all tokens...")
    tokens_df = pd.concat(all_dfs, ignore_index=True)
    del all_dfs
    print(f"Total tokens: {len(tokens_df):,}")

    # Resolve percentiles
    print("Resolving percentiles...")
    tokens_df = resolve_percentiles(tokens_df, global_config.numerical_strategy)

    # Resolve magic timestamps
    print("Resolving magic timestamps...")
    tokens_df = resolve_magic_timestamps(tokens_df, enc_col)

    # Build timelines
    print("Building timelines...")
    timeline_df = build_timelines(tokens_df, global_config)
    del tokens_df

    print(f"\nFinal timeline: {len(timeline_df):,} tokens, {timeline_df['encounter_key'].nunique():,} encounters")

    # Save
    print(f"\nSaving to {output_dir}...")
    save_outputs(timeline_df, global_config, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
