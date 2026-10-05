# Tokenizer V2

YAML-driven clinical data tokenizer that converts raw CDW tables (vitals, labs, meds, procedures, visits, ICU) into integer-encoded encounter timelines for model training.

## Usage

```bash
python -m mint.tokenizer_v2.tokenize \
    --config tokenizer_config.yaml \
    --data-dir cdw/ \
    --output-dir output/
```

The YAML config is the single source of truth — no pre-filtering or caching steps required. Only variables listed in the `variables` section of each source are tokenized.

If `lab_zscore_stats.json` exists in `--data-dir`, z-score tokenization is enabled for labs configured with `rounding: zscore`.

## Configuration (tokenizer_config.yaml)

### Global Settings

| Field | Default | Description |
|-------|---------|-------------|
| `min_token_count` | 25 | Drop token types appearing fewer than N times globally |
| `min_encounter_tokens` | 12 | Drop encounters with fewer than N tokens |
| `max_encounter_minutes` | null | Drop encounters spanning more than this many minutes |
| `encounter_key_column` | EncounterKey | Column identifying encounters |
| `disposition_filter` | [Discharge, Admit] | Tokens used for Admit/Discharge XOR filtering |
| `numerical_strategy.mode` | combined_exact | One of: `combined_exact`, `combined_percentile`, `separate_exact`, `separate_percentile` |
| `numerical_strategy.percentile_rounding` | 5 | Rounding step for percentile buckets |
| `numerical_strategy.zscore_step` | 0.2 | Step size for z-score discretization |
| `split` | year-based | Rules for train/val/test splitting by year |

### Sources

Each source defines:
- `file` — CSV filename in the data directory
- `mode` — `keyed` (one key column identifies the variable) or `multi_column` (each column is its own variable)
- `key_column`, `value_column`, `timestamp_column` — column mappings
- `defaults` — default type/rounding/prefix for all variables in this source
- `variables` — per-variable overrides (rename, skip, rounding, type, custom processors). Acts as an allowlist: only listed keys are tokenized.

### Variable Types

- **numeric** — Emits `{prefix}_{name}_{rounded_value}` tokens. Rounding options: `int`, `exact`, `int5`, `int2`, `zscore`
- **categorical** — Emits `{prefix}_{name}` or `{prefix}_{value}` tokens
- **custom** — Delegates to a registered processor (e.g., `bp`, `weight_oz_to_kg`, `o2_device`)

### Special Behaviors

- **visits source**: Automatically joins `demos.csv` (if present in data-dir) to add Sex tokens, remapping Nonbinary/Unknown to Other.
- **labs source**: When `min_abnormal_fraction` is set, filters to lab types exceeding that abnormal rate.

## Outputs

| File | Description |
|------|-------------|
| `vocab.csv` | Token vocabulary (index, name, count) |
| `tokens.feather` | Full timeline: encounter_key, name, t, [pair_id] |
| `train.npy`, `val.npy`, `test.npy` | Arrays of shape (N, 3): [patient_id, token_id, t] as uint32 |
| `train.feather`, `val.feather`, `test.feather` | Split timelines in feather format |
| `encounter_key_map.csv` | Maps encounter_key to assigned integer patient_id |

## Timeline Building Filters

During `build_timelines()`, these filters are applied in order:

1. **min_token_count** — Remove globally rare token types
2. **Admit XOR Discharge** — Keep only encounters with exactly one disposition event
3. **Timestamp conversion** — Parse timestamps, compute relative minutes from arrival
4. **Negative time filter** — Drop tokens occurring before t=0
5. **max_encounter_minutes** — Remove encounters exceeding duration threshold
6. **min_encounter_tokens** — Remove encounters with too few tokens
7. **Auto-append Discharge** — For admitted encounters, append Discharge at t_max + 1
