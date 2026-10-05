"""Custom token processors for special data types."""

from abc import ABC, abstractmethod

import pandas as pd


class TokenProcessor(ABC):
    """Interface for custom token processors.

    Each processor takes a row of data and returns a list of token dicts,
    each with keys: encounter_key, name, timestamp.
    """

    @abstractmethod
    def process(self, row, enc_col: str, timestamp_col: str, prefix: str) -> list[dict]:
        ...


class BPProcessor(TokenProcessor):
    """Parse blood pressure "SYS/DIA" into two tokens: Vital_Systolic_X and Vital_Diastolic_Y, rounded to nearest 2."""

    def process(self, row, enc_col: str, timestamp_col: str, prefix: str) -> list[dict]:
        value = getattr(row, "Value", None)
        if value is None or (not isinstance(value, str) and pd.isnull(value)):
            return []

        value = str(value).strip()
        if "/" not in value:
            return []

        parts = value.split("/")
        if len(parts) != 2:
            return []

        try:
            systolic = 2 * round(float(parts[0]) / 2)
            diastolic = 2 * round(float(parts[1]) / 2)
        except (ValueError, TypeError):
            return []

        enc_key = getattr(row, enc_col)
        timestamp = getattr(row, timestamp_col)

        return [
            {enc_col: enc_key, "name": f"Vital_Systolic_{systolic}", "timestamp": timestamp},
            {enc_col: enc_key, "name": f"Vital_Diastolic_{diastolic}", "timestamp": timestamp},
        ]


class WeightProcessor(TokenProcessor):
    """Convert weight from ounces to kilograms, round to nearest int."""

    def process(self, row, enc_col: str, timestamp_col: str, prefix: str) -> list[dict]:
        value = getattr(row, "NumericValue", None)
        if value is None or (not isinstance(value, str) and pd.isnull(value)):
            return []
        try:
            ounces = float(value)
            kg = int(round(ounces / 35.274))
        except (ValueError, TypeError):
            return []

        enc_key = getattr(row, enc_col)
        timestamp = getattr(row, timestamp_col)
        return [{enc_col: enc_key, "name": f"Vital_Weight_{kg}kg", "timestamp": timestamp}]


class O2DeviceProcessor(TokenProcessor):
    """Emit O2 Device categorical tokens with prefix Vital_O2 Device_."""

    def process(self, row, enc_col: str, timestamp_col: str, prefix: str) -> list[dict]:
        value = getattr(row, "Value", None)
        if value is None or (not isinstance(value, str) and pd.isnull(value)):
            return []

        enc_key = getattr(row, enc_col)
        timestamp = getattr(row, timestamp_col)
        return [{enc_col: enc_key, "name": f"Vital_O2 Device_{value}", "timestamp": timestamp}]


PROCESSOR_REGISTRY: dict[str, TokenProcessor] = {
    "bp": BPProcessor(),
    "weight_oz_to_kg": WeightProcessor(),
    "o2_device": O2DeviceProcessor(),
}


def get_processor(name: str) -> TokenProcessor:
    if name not in PROCESSOR_REGISTRY:
        raise ValueError(f"Unknown processor: {name}. Available: {list(PROCESSOR_REGISTRY.keys())}")
    return PROCESSOR_REGISTRY[name]
