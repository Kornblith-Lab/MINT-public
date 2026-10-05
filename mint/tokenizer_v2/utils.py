"""Shared utilities for the tokenizer pipeline."""

import pandas as pd

MAGIC_ADMIT = "__MAGIC_ADMIT__"


def round_to_nearest_n(value, n):
    return n * round(value / n)


def round_value(value, rounding):
    """Round a numeric value according to the rounding spec.

    rounding can be:
      - "int": truncate to integer
      - "int2": round to nearest 2
      - "int5": round to nearest 5
      - "exact": no rounding, keep as-is
      - int/float N: round to nearest multiple of N
    """
    if rounding == "exact" or rounding is None:
        return value
    if rounding == "int":
        return int(value)
    if isinstance(rounding, str) and rounding.startswith("int"):
        n = int(rounding[3:])
        return round_to_nearest_n(int(value), n)
    if isinstance(rounding, (int, float)):
        return round_to_nearest_n(int(value), n=rounding)
    return value


def get_row_timestamp(row, source_config, var_config: dict):
    ts_col = var_config.get("timestamp_column", source_config.timestamp_column)
    if ts_col:
        return getattr(row, ts_col, None)
    if source_config.timestamp_composite:
        date_val = getattr(row, source_config.timestamp_composite["date_column"], None)
        time_val = getattr(row, source_config.timestamp_composite["time_column"], None)
        if pd.isnull(date_val) or pd.isnull(time_val):
            return None
        return f"{date_val} {time_val}"
    return None


def zscore_to_token_suffix(z_raw: float, step: float = 0.2, clip_min: float = -3.0, clip_max: float = 3.0) -> str:
    """Convert a raw z-score to a discretized token suffix like 'Z1.2' or 'Z-0.4'."""
    z = max(clip_min, min(clip_max, z_raw))
    z_rounded = round(z / step) * step
    z_rounded = round(z_rounded, 1)
    if z_rounded == 0:
        z_rounded = 0.0
    return f"Z{z_rounded}"


def check_condition(row, condition: dict) -> bool:
    col = condition["column"]
    expected = condition["equals"]
    actual = getattr(row, col, None)
    if pd.isnull(actual):
        return False
    if isinstance(expected, (int, float)):
        try:
            return float(actual) == float(expected)
        except (ValueError, TypeError):
            return False
    return str(actual) == str(expected)
