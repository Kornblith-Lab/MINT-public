"""Configuration dataclasses and YAML loading for the tokenizer."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class NumericalStrategy:
    mode: str = "combined_exact"  # combined_exact | combined_percentile | separate_exact | separate_percentile
    percentile_rounding: int = 5
    zscore_step: float = 0.2


@dataclass
class GlobalConfig:
    min_token_count: int = 25
    min_encounter_tokens: int = 12
    max_encounter_minutes: int | None = None
    encounter_key_column: str = "EncounterKey"
    disposition_filter: list[str] = field(default_factory=lambda: ["Discharge", "Admit"])
    numerical_strategy: NumericalStrategy = field(default_factory=NumericalStrategy)
    split: dict[str, str] = field(default_factory=dict)
    output_dir: Path = field(default_factory=lambda: Path("output"))


@dataclass
class SourceConfig:
    name: str
    file: str
    mode: str  # "keyed" | "multi_column"
    key_column: str | None = None
    value_column: str | None = None
    timestamp_column: str | None = None
    timestamp_composite: dict | None = None
    pre_filter: dict | None = None
    defaults: dict = field(default_factory=dict)
    variables: dict = field(default_factory=dict)


def load_config(path: Path) -> tuple[GlobalConfig, list[SourceConfig]]:
    with open(path) as f:
        raw = yaml.safe_load(f)

    g = raw.get("global", {})
    ns = g.get("numerical_strategy", {})
    global_config = GlobalConfig(
        min_token_count=g.get("min_token_count", 25),
        min_encounter_tokens=g.get("min_encounter_tokens", 12),
        max_encounter_minutes=g.get("max_encounter_minutes"),
        encounter_key_column=g.get("encounter_key_column", "EncounterKey"),
        disposition_filter=g.get("disposition_filter", ["Discharge", "Admit"]),
        numerical_strategy=NumericalStrategy(
            mode=ns.get("mode", "combined_exact"),
            percentile_rounding=ns.get("percentile_rounding", 5),
            zscore_step=ns.get("zscore_step", 0.2),
        ),
        split=g.get("split", {}),
        output_dir=Path(g.get("output_dir", "output")),
    )

    sources = []
    for name, src in raw.get("sources", {}).items():
        ts_composite = src.get("timestamp")
        if isinstance(ts_composite, dict) and "date_column" in ts_composite:
            timestamp_composite = ts_composite
        else:
            timestamp_composite = None

        sources.append(SourceConfig(
            name=name,
            file=src["file"],
            mode=src["mode"],
            key_column=src.get("key_column"),
            value_column=src.get("value_column"),
            timestamp_column=src.get("timestamp_column"),
            timestamp_composite=timestamp_composite,
            pre_filter=src.get("pre_filter"),
            defaults=src.get("defaults") or {},
            variables=src.get("variables") or {},
        ))

    return global_config, sources
