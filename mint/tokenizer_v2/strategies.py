"""Numerical strategy implementations for token emission.

NOTE: The main tokenize.py uses an inline _emit_numeric() for performance with tuple-based
token collection. This module provides the dict-based interface for external use and testing.
"""

import uuid

from .config import NumericalStrategy


def emit_numeric_tokens(
    tokens: list,
    enc_col: str,
    enc_key,
    base_name: str,
    rounded_value,
    timestamp,
    strategy: NumericalStrategy,
):
    """Emit token(s) according to the numerical strategy (dict-based interface).

    For 'separate' modes, a pair_id links the Type and Value tokens.
    For 'percentile' modes, a __PERCENTILE__ placeholder is emitted for later resolution.
    """
    mode = strategy.mode

    if mode == "combined_exact":
        tokens.append({
            enc_col: enc_key,
            "name": f"{base_name}_{rounded_value}",
            "timestamp": timestamp,
            "pair_id": None,
        })

    elif mode == "combined_percentile":
        tokens.append({
            enc_col: enc_key,
            "name": f"__PERCENTILE__{base_name}__{rounded_value}",
            "timestamp": timestamp,
            "pair_id": None,
        })

    elif mode == "separate_exact":
        pair_id = uuid.uuid4().hex[:8]
        tokens.append({
            enc_col: enc_key,
            "name": base_name,
            "timestamp": timestamp,
            "pair_id": pair_id,
        })
        tokens.append({
            enc_col: enc_key,
            "name": str(rounded_value),
            "timestamp": timestamp,
            "pair_id": pair_id,
        })

    elif mode == "separate_percentile":
        pair_id = uuid.uuid4().hex[:8]
        tokens.append({
            enc_col: enc_key,
            "name": base_name,
            "timestamp": timestamp,
            "pair_id": pair_id,
        })
        tokens.append({
            enc_col: enc_key,
            "name": f"__PERCENTILE__{base_name}__{rounded_value}",
            "timestamp": timestamp,
            "pair_id": pair_id,
        })

    else:
        raise ValueError(f"Unknown strategy mode: {mode}")
