"""Generate NEJM-style AUROC and AUPRC grids for Fig. 1 outcomes.

════════════════════════════════════════════════════════════════════════════
RUN THIS (canonical, reproduces every panel + CSV with paired 1000x bootstrap):

    python -m mint.five.fig_one.fig1_nejm \
        --final_dir artifacts/fig1-final \
        --operational_dir artifacts/fig1_operational \
        --output_dir artifacts/fig1_nejm

The first run bootstraps all tasks (a few minutes; the intervention tasks are
the slow ones). Every subsequent run is near-instant because results are cached
under ``<output_dir>/cache/``. Outputs land in ``<output_dir>/`` (see below).
════════════════════════════════════════════════════════════════════════════

WHAT IT PRODUCES (in --output_dir)
    fig1_nejm_auroc.png                2x5 vitals/intervention ROC grid
    fig1_nejm_auprc.png                2x5 vitals/intervention PR grid
    fig1_nejm_paired_differences.csv   per-method AUROC/AUPRC + CIs, and the
                                       paired difference vs the target (+ CIs,
                                       significance flags)
    fig1_operational_auroc.png         2x2 operational grid (MINT zero-shot
    fig1_operational_auprc.png         excluded; stars mark significance vs the
                                       MINT linear probe)
    fig1_operational_summary.csv       per-method operational AUROC/AUPRC + CIs
                                       and paired diffs vs the linear probe
    cache/                             pickled bootstrap results (safe to delete
                                       to force a full recompute)

INPUT LAYOUT
    Vitals / intervention curves come from the per-method prediction CSVs under
    ``--final_dir`` (default artifacts/fig1-final), in these subdirs:
        fig1-final-softmax/{task}_softmax.csv        -> mint_softmax
        fig1-final-softmax/{task}_XGBoost.csv        -> xgboost
        fig1-final-softmax-tt/{task}_test_time_softmax.csv  -> mint_tt_mean
        fig1-final-cdf/{task}_CDF.csv                -> mint_cdf
        fig1-final-cdf-tt/{task}_test_time_cdf.csv   -> mint_cdf_tt
        fig1-final-triage/{task}_triage.csv          -> triage
    Every CSV must carry an ``encounter_key`` column; if one is missing the run
    aborts with a clear error (it never silently approximates).
    The operational panel reads ``--operational_dir`` (default
    artifacts/fig1_operational).

HOW THE CONFIDENCE INTERVALS WORK (paired clustered bootstrap)
    All CIs are recomputed here from scratch -- none are read from the input
    files. Within each task a SINGLE resample of encounter_keys (drawn with
    replacement from the UNION of the selected methods' encounters) is shared
    across every method in that iteration, so the comparison is paired. The
    resample is at the ENCOUNTER level: a sampled encounter contributes all of
    its rows (there are multiple time-windows per encounter), and AUROC + AUPRC
    are computed together in one pass. Point estimates use all rows.

    Paired differences (method - target) are computed on the per-iteration
    intersection of the encounters that both the method and target possess.
    A ``*`` next to a method's value in a panel's annotation box means that
    method differs significantly from the target on THAT task for THAT figure's
    metric (the 95% difference CI excludes 0). Seed is fixed (default 42) so
    cached results are deterministic.

COMMON OPTIONS
    --methods mint_softmax xgboost triage   plot only a subset (default: all 6)
    --style bar | curve                     bar plots of AUROC/AUPRC (default) or
                                            ROC/PR curves
    --labels mint_softmax=MINT "mint_tt_mean=Test-time softmax"
                                            per-run legend/label overrides
                                            (key=Label; vitals methods only)
    --target_estimator mint_cdf             reference for paired diffs
                                            (default: mint_softmax; must be one
                                            of the selected --methods)
    --n_boot 1000                           bootstrap iterations (default 1000)
    --seed 42                               resample seed (changes the cache key)
    --skip_operational                      vitals/intervention panels only
    --dpi 600                               output resolution

    Valid method keys: mint_softmax, mint_tt_mean, mint_cdf, mint_cdf_tt,
    xgboost, triage.
    (tap uses underscores, e.g. --final_dir, not --final-dir.)

NOTE
    MINT LR (linear probe) is intentionally NOT included for the vitals /
    intervention tasks: fig1-final has no per-encounter probability CSV for it,
    only point estimates, so it cannot be curve-plotted or bootstrapped here. It
    still appears in the operational panel (from {outcome}_MINT_probes.csv).
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from tap import tapify
from tqdm import tqdm


plt.style.use(str(Path(__file__).resolve().parents[1] / "design-skill" / "nature.mplstyle"))
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    }
)

BASELINE_COLOR = "#B8B8B8"

# Okabe-Ito colorblind-safe palette (see design-skill/nature.mplstyle).
_OKABE = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
}

# Off-palette accents:
#   XGBoost is dark grey (not black) so its darker CI whisker stays visible.
#   MINT (LR) is a darker shade of the MINT zero-shot green, tying the two MINT
#   variants together (was an amber/yellow that read as unrelated).
XGBOOST_COLOR = "#4D4D4D"
MINT_LR_COLOR = "#00664B"

TITLE_FS = 11
AXIS_LABEL_FS = 11
TICK_FS = 11
ANNOTATION_FS = 6.5
# In-panel value+CI box for curve mode (bigger than the bar-mode x-tick labels,
# which stay at ANNOTATION_FS so long method names don't collide).
CURVE_ANNOTATION_FS = 12
LEGEND_FS = 10

AIRWAY_TASKS = ["hypoxia", "tachypnea", "ppv", "resp_rescue", "ventilator"]
CIRCULATION_TASKS = ["periarrest", "tachycardia", "hypotension", "vasopressor", "transfusion"]
TASK_GRID = [AIRWAY_TASKS, CIRCULATION_TASKS]

TASK_LABELS = {
    "hypoxia": "Hypoxia",
    "tachypnea": "Tachypnea",
    "ppv": "Positive pressure ventilation",
    "resp_rescue": "Airway medication",
    "ventilator": "Ventilator",
    "periarrest": "Periarrest",
    "tachycardia": "Tachycardia",
    "hypotension": "Hypotension",
    "vasopressor": "Vasopressor",
    "transfusion": "Blood products",
}


# ─── Method registry ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MethodSpec:
    key: str            # canonical CLI identifier
    label: str          # legend label
    subdir: str         # subdirectory under the final-dir root
    filename: str       # CSV filename template ("{task}")
    prob_col: str       # probability column in the CSV
    color: str
    linestyle: str = "-"
    lw: float = 1.6
    zorder: int = 2
    short: str = ""     # short label for the per-axis annotation


# Draw order matters (later = on top). Test-time variants use the mean only.
VITALS_METHOD_LIST = [
    MethodSpec("mint_softmax", "MINT (zero-shot)", "fig1-final-softmax", "{task}_softmax.csv", "probs",
               _OKABE["green"], "-", 1.9, 6, "MINT"),
    MethodSpec("mint_tt_mean", "MINT test-time (softmax)", "fig1-final-softmax-tt", "{task}_test_time_softmax.csv", "probs_mean",
               _OKABE["orange"], "-", 1.6, 5, "TT-softmax"),
    MethodSpec("mint_cdf", "MINT (CDF)", "fig1-final-cdf", "{task}_CDF.csv", "probs",
               _OKABE["sky"], "-", 1.6, 4, "CDF"),
    MethodSpec("mint_cdf_tt", "MINT test-time (CDF)", "fig1-final-cdf-tt", "{task}_test_time_cdf.csv", "probs_mean",
               _OKABE["blue"], "-", 1.6, 3, "TT-CDF"),
    MethodSpec("xgboost", "XGBoost", "fig1-final-softmax", "{task}_XGBoost.csv", "probs",
               XGBOOST_COLOR, "-", 1.5, 2, "XGBoost"),
    MethodSpec("triage", "Triage", "fig1-final-triage", "{task}_triage.csv", "probs",
               _OKABE["purple"], "-", 1.5, 2, "Triage"),
    # Temporal-fidelity ablation of the softmax model (same {task}_softmax.csv
    # layout, coarser token resolution). Full MINT tokenizes every minute;
    # mint_hour collapses to one token/hour (fig1-60) and mint_day to one/day
    # (fig1-1440). mint_softmax above is the full (minute-resolution) model.
    MethodSpec("mint_hour", "MINT Hourly", "fig1-60-softmax", "{task}_softmax.csv", "probs",
               _OKABE["orange"], "-", 1.6, 5, "Hourly"),
    MethodSpec("mint_day", "MINT Daily", "fig1-1440-softmax", "{task}_softmax.csv", "probs",
               _OKABE["sky"], "-", 1.6, 4, "Daily"),
]
VITALS_METHODS = {m.key: m for m in VITALS_METHOD_LIST}


@dataclass
class Args:
    final_dir: str = "artifacts/fig1-final"
    operational_dir: str = "artifacts/fig1_operational"
    output_dir: str = "artifacts/fig1_nejm"
    methods: list[str] = None  # default: all methods
    tasks: list[str] = None  # subset of vitals/intervention tasks to plot (default: all 10)
    labels: list[str] = None  # per-run label overrides, e.g. mint_softmax=Softmax mint_tt_mean="Test-time softmax"
    target_estimator: str = "mint_softmax"
    style: str = "bar"  # {bar, curve}: bar plots of AUROC/AUPRC, or ROC/PR curves
    n_boot: int = 1000
    seed: int = 42
    dpi: int = 600
    skip_operational: bool = False
    # Composite figure flags (fig1_composite.png).
    # All default False so tap registers them as store_true (pass the flag to enable).
    hide_xgboost: bool = False      # pass --hide_xgboost to remove XGBoost from composite
    show_triage: bool = False       # pass --show_triage to add Triage to composite
    hide_dim_curves: bool = False   # pass --hide_dim_curves to suppress per-outcome background lines
    three_decimals: bool = False    # pass --three_decimals to use 3 dp in mini-panel annotations (default: 2 dp)


def _format_incidence(value: float) -> str:
    pct = value * 100.0
    return f"{pct:.2f}%" if pct < 1 else f"{pct:.1f}%"


# ─── Data loading ────────────────────────────────────────────────────────────

def _load_summary(final_dir: Path) -> dict[str, dict]:
    """Load the softmax summary JSONL (used only for incidence / n in titles)."""
    results_path = final_dir / "fig1-final-softmax" / "results_softmax_test.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing summary file: {results_path}")
    records = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
    return {record["task"]: record for record in records}


def _load_method_csv(path: Path, prob_col: str) -> pd.DataFrame:
    """Load one method's per-encounter predictions.

    Hard error (never approximate) if encounter_key is absent."""
    df = pd.read_csv(path)
    if "encounter_key" not in df.columns:
        raise RuntimeError(
            f"encounter_key column missing from {path}. Cannot pair the bootstrap; "
            f"regenerate this CSV with encounter_key before running."
        )
    if prob_col not in df.columns:
        raise RuntimeError(f"probability column '{prob_col}' missing from {path}")
    return pd.DataFrame({
        "encounter_key": df["encounter_key"].to_numpy(),
        "probs": df[prob_col].to_numpy(dtype=float),
        "labels": df["labels"].to_numpy(dtype=int),
    })


def _load_task_methods(final_dir: Path, task: str, method_keys: list[str]) -> dict[str, pd.DataFrame]:
    """Load every requested method for a task. Raises if a CSV is missing."""
    out: dict[str, pd.DataFrame] = {}
    for key in method_keys:
        spec = VITALS_METHODS[key]
        path = final_dir / spec.subdir / spec.filename.format(task=task)
        if not path.exists():
            raise FileNotFoundError(f"Missing prediction CSV for method '{key}', task '{task}': {path}")
        out[key] = _load_method_csv(path, spec.prob_col)
    return out


# ─── Paired clustered bootstrap ──────────────────────────────────────────────

def _both_metrics(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    """AUROC and AUPRC in one pass; NaN if a single class is present."""
    if labels.min() == labels.max():
        return np.nan, np.nan
    return roc_auc_score(labels, probs), average_precision_score(labels, probs)


class _MethodBoot:
    """Encounter-clustered gather structure for one method over a shared union coding."""

    def __init__(self, codes: np.ndarray, probs: np.ndarray, labels: np.ndarray, n_union: int):
        order = np.argsort(codes, kind="stable")
        self.codes = codes[order]
        self.probs = probs[order]
        self.labels = labels[order]
        self.counts = np.bincount(self.codes, minlength=n_union)
        self.offsets = np.zeros(n_union + 1, dtype=np.int64)
        self.offsets[1:] = np.cumsum(self.counts)
        self.has_key = self.counts > 0

    def gather(self, sampled_codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (probs, labels) for all rows belonging to the sampled encounter codes."""
        sel_counts = self.counts[sampled_codes]
        total = int(sel_counts.sum())
        if total == 0:
            return np.empty(0), np.empty(0, dtype=int)
        starts = self.offsets[sampled_codes]
        base = np.repeat(starts, sel_counts)
        within = np.arange(total) - np.repeat(np.cumsum(sel_counts) - sel_counts, sel_counts)
        idx = base + within
        return self.probs[idx], self.labels[idx]


def _point_metrics(df: pd.DataFrame) -> tuple[float, float]:
    return _both_metrics(df["labels"].to_numpy(), df["probs"].to_numpy())


def _point_diff(target_df: pd.DataFrame, method_df: pd.DataFrame) -> tuple[float, float]:
    """Point-estimate paired difference (method - target) on the intersection of their
    full encounter sets, using all rows for the shared encounters."""
    shared = np.intersect1d(target_df["encounter_key"].to_numpy(), method_df["encounter_key"].to_numpy())
    t = target_df[target_df["encounter_key"].isin(shared)]
    m = method_df[method_df["encounter_key"].isin(shared)]
    t_auroc, t_auprc = _both_metrics(t["labels"].to_numpy(), t["probs"].to_numpy())
    m_auroc, m_auprc = _both_metrics(m["labels"].to_numpy(), m["probs"].to_numpy())
    return m_auroc - t_auroc, m_auprc - t_auprc


def _ci(arr: np.ndarray) -> list[float]:
    arr = arr[~np.isnan(arr)]
    return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]


def _significant(ci: list[float]) -> bool:
    """CI excludes 0 (both endpoints on the same side of 0)."""
    return (ci[0] > 0 and ci[1] > 0) or (ci[0] < 0 and ci[1] < 0)


def paired_bootstrap(
    methods_data: dict[str, pd.DataFrame],
    method_keys: list[str],
    target_key: str | None,
    n_boot: int,
    seed: int,
) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    """Union-sampled, encounter-clustered paired bootstrap.

    A single resample of encounter codes (drawn with replacement from the UNION of
    all methods' encounters) is shared across every method each iteration. Per-method
    AUROC/AUPRC CIs come from that draw. Paired differences (method - target) are
    computed on the per-iteration intersection of the sampled codes that both the
    method and target possess.

    Returns (summary_dict, arrays_dict) where arrays_dict maps method key ->
    {"auroc": array(n_boot,), "auprc": array(n_boot,)}.  The raw arrays are needed
    for macro-average CIs across tasks (see bootstrap_task_cached).
    """
    rng = np.random.default_rng(seed)

    # Build shared union coding.
    union_keys = np.unique(np.concatenate([methods_data[k]["encounter_key"].to_numpy() for k in method_keys]))
    n_union = len(union_keys)

    boot: dict[str, _MethodBoot] = {}
    for k in method_keys:
        df = methods_data[k]
        codes = np.searchsorted(union_keys, df["encounter_key"].to_numpy())
        boot[k] = _MethodBoot(codes, df["probs"].to_numpy(), df["labels"].to_numpy(dtype=int), n_union)

    # Per-method full-sample bootstrap arrays.
    auroc_boot = {k: np.full(n_boot, np.nan) for k in method_keys}
    auprc_boot = {k: np.full(n_boot, np.nan) for k in method_keys}

    # Which non-target methods share the target's exact encounter set (fast diff path).
    diff_methods = [k for k in method_keys if k != target_key] if target_key else []
    aligned = {}
    if target_key:
        t_has = boot[target_key].has_key
        for k in diff_methods:
            aligned[k] = bool(np.array_equal(boot[k].has_key, t_has))
    # Slow-path diff arrays (computed inside the loop only for misaligned methods).
    diff_auroc_boot = {k: np.full(n_boot, np.nan) for k in diff_methods if not aligned.get(k, False)}
    diff_auprc_boot = {k: np.full(n_boot, np.nan) for k in diff_methods if not aligned.get(k, False)}

    for b in tqdm(range(n_boot)):
        sampled = rng.integers(0, n_union, size=n_union)

        for k in method_keys:
            p, l = boot[k].gather(sampled)
            if len(l):
                auroc_boot[k][b], auprc_boot[k][b] = _both_metrics(l, p)

        # Misaligned diffs need a per-iteration intersection recompute.
        if target_key:
            t_has = boot[target_key].has_key
            for k in diff_auroc_boot:  # misaligned only
                both = t_has & boot[k].has_key
                si = sampled[both[sampled]]
                if len(si) == 0:
                    continue
                tp, tl = boot[target_key].gather(si)
                mp, ml = boot[k].gather(si)
                t_ro, t_pr = _both_metrics(tl, tp)
                m_ro, m_pr = _both_metrics(ml, mp)
                diff_auroc_boot[k][b] = m_ro - t_ro
                diff_auprc_boot[k][b] = m_pr - t_pr

    # Assemble per-method point estimates + CIs.
    result: dict = {"methods": {}, "diffs": {}}
    for k in method_keys:
        pe_auroc, pe_auprc = _point_metrics(methods_data[k])
        result["methods"][k] = {
            "auroc": pe_auroc,
            "auprc": pe_auprc,
            "auroc_ci": _ci(auroc_boot[k]),
            "auprc_ci": _ci(auprc_boot[k]),
        }

    # Paired differences vs target.
    for k in diff_methods:
        pd_auroc, pd_auprc = _point_diff(methods_data[target_key], methods_data[k])
        if aligned.get(k, False):
            da = auroc_boot[k] - auroc_boot[target_key]
            dp = auprc_boot[k] - auprc_boot[target_key]
        else:
            da = diff_auroc_boot[k]
            dp = diff_auprc_boot[k]
        auroc_ci = _ci(da)
        auprc_ci = _ci(dp)
        result["diffs"][k] = {
            "auroc_diff": pd_auroc,
            "auroc_diff_ci": auroc_ci,
            "auroc_sig": _significant(auroc_ci),
            "auprc_diff": pd_auprc,
            "auprc_diff_ci": auprc_ci,
            "auprc_sig": _significant(auprc_ci),
        }

    arrays = {k: {"auroc": auroc_boot[k], "auprc": auprc_boot[k]} for k in method_keys}
    return result, arrays


# ─── Cache ───────────────────────────────────────────────────────────────────

def _file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_key(
    final_dir: Path, task: str, method_keys: list[str], target_key: str | None, n_boot: int, seed: int
) -> str:
    files = {}
    for k in method_keys:
        spec = VITALS_METHODS[k]
        files[k] = _file_hash(final_dir / spec.subdir / spec.filename.format(task=task))
    payload = {
        "task": task,
        "methods": sorted(method_keys),
        "target": target_key,
        "n_boot": n_boot,
        "seed": seed,
        "files": files,
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def bootstrap_task_cached(
    final_dir: Path,
    task: str,
    methods_data: dict[str, pd.DataFrame],
    method_keys: list[str],
    target_key: str | None,
    n_boot: int,
    seed: int,
    cache_dir: Path,
) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    """Return (summary, arrays).

    summary is the existing per-task CI dict (backward compatible).
    arrays maps method key -> {"auroc": array(n_boot,), "auprc": array(n_boot,)}.

    Two parallel cache files share the same key:
      {task}_{key}.pkl       — summary (existing, never rewritten if present)
      {task}_{key}_arrays.pkl — raw bootstrap arrays (written on first run)
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(final_dir, task, method_keys, target_key, n_boot, seed)
    summary_path = cache_dir / f"{task}_{key}.pkl"
    arrays_path = cache_dir / f"{task}_{key}_arrays.pkl"

    summary_hit = summary_path.exists()
    arrays_hit = arrays_path.exists()

    if summary_hit and arrays_hit:
        print(f"  [cache hit] {task}")
        return pickle.loads(summary_path.read_bytes()), pickle.loads(arrays_path.read_bytes())

    if summary_hit and not arrays_hit:
        # Existing summary cache is present but raw arrays were never stored.
        # Rerun bootstrap to generate both; summary result will be identical
        # (same seed, same data) but we don't overwrite the existing file.
        print(f"  [bootstrap arrays] {task} ({n_boot} iters) — generating raw arrays cache...")
        result, arrays = paired_bootstrap(methods_data, method_keys, target_key, n_boot, seed)
        arrays_path.write_bytes(pickle.dumps(arrays))
        return pickle.loads(summary_path.read_bytes()), arrays

    print(f"  [bootstrap] {task} ({n_boot} iters, {len(method_keys)} methods)...")
    result, arrays = paired_bootstrap(methods_data, method_keys, target_key, n_boot, seed)
    summary_path.write_bytes(pickle.dumps(result))
    arrays_path.write_bytes(pickle.dumps(arrays))
    return result, arrays


# ─── Plotting ────────────────────────────────────────────────────────────────

def _curve_data(metric: Literal["auroc", "auprc"], labels: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if metric == "auroc":
        x, y, _ = roc_curve(labels, probs)
        return x, y
    precision, recall, _ = precision_recall_curve(labels, probs)
    return recall[::-1], precision[::-1]


def _disp(k: str, attr: str, overrides: dict[str, str] | None) -> str:
    """Display name for a method: a per-run override wins, else the MethodSpec attr."""
    if overrides and k in overrides:
        return overrides[k]
    return getattr(VITALS_METHODS[k], attr)


def _annotation(boot: dict, method_keys: list[str], metric: Literal["auroc", "auprc"],
                label_overrides: dict[str, str] | None = None) -> str:
    """Compact per-axis value+CI lines, sorted highest to lowest.

    A trailing ``*`` marks a method that differs significantly from the target on
    THIS task for this metric (the 95% paired-difference CI excludes 0)."""
    entries = []
    for k in method_keys:
        m = boot["methods"][k]
        val = m[metric]
        ci = m[f"{metric}_ci"]
        if np.isnan(val):
            continue
        star = " *" if boot["diffs"].get(k, {}).get(f"{metric}_sig", False) else ""
        entries.append((val, f"{_disp(k, 'short', label_overrides)}: {val:.3f} ({ci[0]:.3f}-{ci[1]:.3f}){star}"))
    entries.sort(key=lambda x: x[0], reverse=True)
    return "\n".join(line for _, line in entries)


def _legend_handles(method_keys: list[str], label_overrides: dict[str, str] | None = None) -> list[Line2D]:
    """Legend handles (no significance markers; stars live in per-task annotations)."""
    handles = []
    for k in method_keys:
        spec = VITALS_METHODS[k]
        handles.append(
            Line2D([0], [0], color=spec.color, lw=2.0, ls=spec.linestyle,
                   label=_disp(k, "label", label_overrides))
        )
    return handles


def _task_title(summary: dict, task: str) -> str:
    n_total = int(round(summary["n"] / summary["incidence"]))
    return f"{TASK_LABELS[task]}\n(n={n_total:,}; {_format_incidence(float(summary['incidence']))})"


def _draw_curve_panel(ax, summary, boot, task_curves_task, method_keys, metric, label_overrides=None) -> None:
    if metric == "auroc":
        ax.plot([0, 1], [0, 1], color=BASELINE_COLOR, lw=0.8, ls="--", zorder=1)
    else:
        ax.axhline(float(summary["incidence"]), color=BASELINE_COLOR, lw=0.8, ls="--", zorder=1)

    for k in method_keys:
        spec = VITALS_METHODS[k]
        df = task_curves_task[k]
        cx, cy = _curve_data(metric, df["labels"].to_numpy(), df["probs"].to_numpy())
        ax.plot(cx, cy, color=spec.color, lw=spec.lw, ls=spec.linestyle, zorder=spec.zorder)

    if metric == "auprc":
        text_x, text_y, ha, va = 0.98, 0.98, "right", "top"
    else:
        text_x, text_y, ha, va = 0.98, 0.04, "right", "bottom"
    ax.text(
        text_x, text_y, _annotation(boot, method_keys, metric, label_overrides),
        transform=ax.transAxes, ha=ha, va=va, fontsize=CURVE_ANNOTATION_FS, color="#222222",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.88),
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def _draw_bar_panel(ax, summary, boot, method_keys, metric, label_overrides=None) -> None:
    """One bar per method: the metric value, an asymmetric CI error bar, and a
    text label above each bar with the value and 95% CI (plus, for AUPRC, the
    fold-improvement over the incidence baseline). A trailing ``*`` marks bars
    that differ significantly from the target on this task."""
    vals, los, his, colors, sigs, present = [], [], [], [], [], []
    for k in method_keys:
        m = boot["methods"][k]
        v = m[metric]
        ci = m[f"{metric}_ci"]
        present.append(not np.isnan(v))
        vals.append(0.0 if np.isnan(v) else v)
        los.append(0.0 if np.isnan(v) else max(0.0, v - ci[0]))
        his.append(0.0 if np.isnan(v) else max(0.0, ci[1] - v))
        colors.append(VITALS_METHODS[k].color)
        sigs.append(bool(boot["diffs"].get(k, {}).get(f"{metric}_sig", False)))

    x = np.arange(len(method_keys))
    ax.bar(x, vals, width=0.72, color=colors, zorder=2)
    ax.errorbar(x, vals, yerr=[los, his], fmt="none", ecolor="#333333",
                elinewidth=0.8, capsize=2, zorder=3)

    incidence = float(summary["incidence"])
    baseline = 0.5 if metric == "auroc" else incidence
    ax.axhline(baseline, color=BASELINE_COLOR, lw=0.8, ls="--", zorder=1)

    if metric == "auroc":
        # AUROC panels use a fixed 0.5-1.0 axis so tasks are visually comparable.
        pad = 0.02
        ax.set_ylim(0.5, 1.0)
    else:
        lo_lim = min([baseline] + [v - l for v, l in zip(vals, los)])
        hi_lim = max([baseline] + [v + h for v, h in zip(vals, his)])
        pad = 0.04 * (hi_lim - lo_lim) if hi_lim > lo_lim else 0.02
        # Extra headroom so the multi-line value+CI labels clear the tallest bar.
        ax.set_ylim(max(0.0, lo_lim - pad), min(1.0, hi_lim + 12 * pad))

    for xi, v, ci_lo, ci_hi, h, sig, ok in zip(
        x, vals, [v - l for v, l in zip(vals, los)],
        [v + hh for v, hh in zip(vals, his)], his, sigs, present
    ):
        if not ok:
            continue
        star = " *" if sig else ""
        if metric == "auprc":
            # AUPRC autoscales with headroom -> label above the whisker.
            improvement = v / incidence if incidence > 0 else float("nan")
            label = f"{v:.3f}{star}\n({ci_lo:.3f}-{ci_hi:.3f})\n{improvement:.1f}×"
            ax.text(xi, v + h + pad, label, ha="center", va="bottom",
                    fontsize=5.5, color="#222222", zorder=4, linespacing=0.95)
        else:
            # AUROC axis is fixed 0.5-1.0 (no headroom) -> label inside the bar top.
            label = f"{v:.3f}{star}\n({ci_lo:.3f}-{ci_hi:.3f})"
            ax.text(xi, v - pad, label, ha="center", va="top",
                    fontsize=5.5, color="white", zorder=4, linespacing=0.95)

    ax.set_xticks(x)
    ax.set_xticklabels([_disp(k, "short", label_overrides) for k in method_keys],
                       rotation=0, ha="center", fontsize=ANNOTATION_FS)


def _plot_metric(
    summary_by_task: dict[str, dict],
    boot_by_task: dict[str, dict],
    task_curves: dict[str, dict[str, pd.DataFrame]],
    method_keys: list[str],
    target_key: str | None,
    metric: Literal["auroc", "auprc"],
    output_path: Path,
    dpi: int,
    style: Literal["bar", "curve"] = "bar",
    label_overrides: dict[str, str] | None = None,
    task_grid: list[list[str]] | None = None,
) -> None:
    is_bar = style == "bar"
    if is_bar:
        ylabel = "AUROC" if metric == "auroc" else "AUPRC"
        xlabel = ""
    else:
        xlabel = "False positive rate" if metric == "auroc" else "Recall"
        ylabel = "True positive rate" if metric == "auroc" else "Precision"

    # Grid shape follows the (possibly subset) task layout. The per-panel size
    # matches the canonical 2x5 grid (3.3 x 3.25 in each), so the full figure is
    # unchanged and subsets simply have fewer panels.
    grid = task_grid if task_grid is not None else TASK_GRID
    nrows = len(grid)
    ncols = max(len(row) for row in grid)

    # Bars are autoscaled per task, so axes are not shared in bar mode. Rotated
    # x-tick labels also need extra vertical room between the two rows.
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 3.25 * nrows),
                             sharex=not is_bar, sharey=not is_bar,
                             squeeze=False, constrained_layout=False)
    fig.subplots_adjust(left=0.045, right=0.995, top=0.86, bottom=0.16, wspace=0.25,
                        hspace=0.92 if is_bar else 0.42)

    used = set()
    for row_idx, tasks in enumerate(grid):
        for col_idx, task in enumerate(tasks):
            ax = axes[row_idx][col_idx]
            used.add((row_idx, col_idx))
            summary = summary_by_task[task]

            if is_bar:
                _draw_bar_panel(ax, summary, boot_by_task[task], method_keys, metric, label_overrides)
            else:
                _draw_curve_panel(ax, summary, boot_by_task[task], task_curves[task], method_keys, metric, label_overrides)

            ax.set_title(_task_title(summary, task), fontsize=TITLE_FS, pad=5)
            ax.tick_params(axis="y", labelsize=TICK_FS, length=2)
            # In bar mode the (horizontal) method labels are drawn small so long
            # names like "Test-time softmax" don't collide; don't let TICK_FS win.
            ax.tick_params(axis="x", labelsize=ANNOTATION_FS if is_bar else TICK_FS, length=2)
            ax.set_xlabel(xlabel if row_idx == nrows - 1 else "", fontsize=AXIS_LABEL_FS)
            ax.set_ylabel(ylabel if col_idx == 0 else "", fontsize=AXIS_LABEL_FS)

    # Hide any trailing axes on a ragged final row.
    for row_idx in range(nrows):
        for col_idx in range(ncols):
            if (row_idx, col_idx) not in used:
                axes[row_idx][col_idx].axis("off")

    fig.legend(
        handles=_legend_handles(method_keys, label_overrides),
        loc="lower center", ncol=len(method_keys), frameon=False, fontsize=LEGEND_FS,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ─── Summary CSV ─────────────────────────────────────────────────────────────

def _write_summary_csv(boot_by_task: dict[str, dict], target_key: str | None, path: Path) -> None:
    rows = []
    for task, boot in boot_by_task.items():
        for k, m in boot["methods"].items():
            row = {
                "task": task,
                "method": k,
                "target": target_key,
                "is_target": k == target_key,
                "auroc": m["auroc"],
                "auroc_ci_lo": m["auroc_ci"][0],
                "auroc_ci_hi": m["auroc_ci"][1],
                "auprc": m["auprc"],
                "auprc_ci_lo": m["auprc_ci"][0],
                "auprc_ci_hi": m["auprc_ci"][1],
            }
            d = boot["diffs"].get(k)
            if d is not None:
                row.update({
                    "auroc_diff_vs_target": d["auroc_diff"],
                    "auroc_diff_ci_lo": d["auroc_diff_ci"][0],
                    "auroc_diff_ci_hi": d["auroc_diff_ci"][1],
                    "auroc_diff_sig": d["auroc_sig"],
                    "auprc_diff_vs_target": d["auprc_diff"],
                    "auprc_diff_ci_lo": d["auprc_diff_ci"][0],
                    "auprc_diff_ci_hi": d["auprc_diff_ci"][1],
                    "auprc_diff_sig": d["auprc_sig"],
                })
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Wrote paired-difference summary: {path}")


# ─── Operational panel ───────────────────────────────────────────────────────

OPERATIONAL_GRID = [["admit", "icu"], ["sepsis", "septic_shock"]]
OPERATIONAL_LABELS = {"admit": "Admit", "icu": "ICU", "sepsis": "Sepsis", "septic_shock": "Septic shock"}

# (filename template, prob column, legend label, color)
OPERATIONAL_METHOD_SPECS = {
    "MINT_softmax": ("{outcome}_MINT_softmax.csv", "probs", "MINT (zero-shot)", _OKABE["green"]),
    "MINT_LR": ("{outcome}_MINT_probes.csv", "probs_lr", "MINT (LR)", MINT_LR_COLOR),
    "XGBoost_baseline": ("{outcome}_XGBoost_baseline.csv", "probs", "XGBoost", XGBOOST_COLOR),
    "ESI": ("{outcome}_ESI.csv", "probs", "ESI", _OKABE["vermillion"]),
    "ED-PEWS": ("{outcome}_ED-PEWS.csv", "probs", "ED-PEWS", _OKABE["purple"]),
}
# MINT zero-shot is intentionally excluded here; the operational panel compares
# MINT's linear probe against the clinical/ML baselines. Significance stars are
# computed vs OPERATIONAL_TARGET (the linear probe), which appears in every outcome.
OPERATIONAL_TARGET = "MINT_LR"
OPERATIONAL_METHODS = {
    "admit": ["MINT_LR", "XGBoost_baseline", "ESI", "ED-PEWS"],
    "icu": ["MINT_LR", "XGBoost_baseline", "ESI", "ED-PEWS"],
    "sepsis": ["MINT_LR", "XGBoost_baseline"],
    "septic_shock": ["MINT_LR", "XGBoost_baseline"],
}


def _load_operational_methods(operational_dir: Path, outcome: str, method_keys: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for k in method_keys:
        fname, prob_col, _, _ = OPERATIONAL_METHOD_SPECS[k]
        path = operational_dir / fname.format(outcome=outcome)
        if not path.exists():
            print(f"  [op skip] {outcome}/{k}: missing {path.name}")
            continue
        out[k] = _load_method_csv(path, prob_col)
    return out


def _op_bootstrap_cached(
    operational_dir: Path, outcome: str, methods_data: dict[str, pd.DataFrame],
    target_key: str | None, n_boot: int, seed: int, cache_dir: Path,
) -> dict:
    """Paired union bootstrap for an operational outcome, incl. paired diffs vs target."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    keys = list(methods_data.keys())
    files = {}
    for k in keys:
        fname, _, _, _ = OPERATIONAL_METHOD_SPECS[k]
        files[k] = _file_hash(operational_dir / fname.format(outcome=outcome))
    tgt = target_key if target_key in keys else None
    ck = hashlib.md5(json.dumps(
        {"outcome": outcome, "methods": sorted(keys), "target": tgt,
         "n_boot": n_boot, "seed": seed, "files": files},
        sort_keys=True).encode()).hexdigest()
    cache_path = cache_dir / f"op_{outcome}_{ck}.pkl"
    if cache_path.exists():
        print(f"  [cache hit] operational {outcome}")
        return pickle.loads(cache_path.read_bytes())
    print(f"  [bootstrap] operational {outcome} ({n_boot} iters, {len(keys)} methods)...")
    result = paired_bootstrap(methods_data, keys, tgt, n_boot, seed)
    cache_path.write_bytes(pickle.dumps(result))
    return result


def _load_operational_incidence(operational_dir: Path) -> dict[str, dict]:
    results_path = operational_dir / "results_operational.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing operational summary: {results_path}")
    records = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
    out: dict[str, dict] = {}
    for rec in records:
        out.setdefault(rec["outcome"], rec)  # any method row carries incidence / n_total
    return out


def _draw_op_curve_panel(ax, boot, curves, methods, metric, incidence) -> None:
    if metric == "auroc":
        ax.plot([0, 1], [0, 1], color=BASELINE_COLOR, lw=0.8, ls="--", zorder=1)
    else:
        ax.axhline(incidence, color=BASELINE_COLOR, lw=0.8, ls="--", zorder=1)

    entries = []
    for k in methods:
        _, _, label, color = OPERATIONAL_METHOD_SPECS[k]
        df = curves[k]
        cx, cy = _curve_data(metric, df["labels"].to_numpy(), df["probs"].to_numpy())
        ax.plot(cx, cy, color=color, lw=1.6, zorder=2)
        m = boot["methods"][k]
        if not np.isnan(m[metric]):
            star = " *" if boot["diffs"].get(k, {}).get(f"{metric}_sig", False) else ""
            entries.append((m[metric], f"{label}: {m[metric]:.3f} ({m[metric+'_ci'][0]:.3f}-{m[metric+'_ci'][1]:.3f}){star}"))
    entries.sort(key=lambda x: x[0], reverse=True)

    if metric == "auprc":
        text_x, text_y, ha, va = 0.98, 0.98, "right", "top"
    else:
        text_x, text_y, ha, va = 0.98, 0.04, "right", "bottom"
    ax.text(text_x, text_y, "\n".join(e for _, e in entries), transform=ax.transAxes,
            ha=ha, va=va, fontsize=CURVE_ANNOTATION_FS, color="#222222",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.88))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def _draw_op_bar_panel(ax, boot, methods, metric, incidence) -> None:
    vals, los, his, colors, sigs = [], [], [], [], []
    for k in methods:
        m = boot["methods"][k]
        v = m[metric]
        ci = m[f"{metric}_ci"]
        vals.append(0.0 if np.isnan(v) else v)
        los.append(0.0 if np.isnan(v) else max(0.0, v - ci[0]))
        his.append(0.0 if np.isnan(v) else max(0.0, ci[1] - v))
        colors.append(OPERATIONAL_METHOD_SPECS[k][3])
        sigs.append(bool(boot["diffs"].get(k, {}).get(f"{metric}_sig", False)))

    x = np.arange(len(methods))
    ax.bar(x, vals, width=0.68, color=colors, zorder=2)
    ax.errorbar(x, vals, yerr=[los, his], fmt="none", ecolor="#333333",
                elinewidth=0.8, capsize=2, zorder=3)

    baseline = 0.5 if metric == "auroc" else incidence
    ax.axhline(baseline, color=BASELINE_COLOR, lw=0.8, ls="--", zorder=1)

    if metric == "auroc":
        # AUROC panels use a fixed 0.5-1.0 axis so outcomes are visually comparable.
        pad = 0.02
        ax.set_ylim(0.5, 1.0)
    else:
        lo_lim = min([baseline] + [v - l for v, l in zip(vals, los)])
        hi_lim = max([baseline] + [v + h for v, h in zip(vals, his)])
        pad = 0.04 * (hi_lim - lo_lim) if hi_lim > lo_lim else 0.02
        ax.set_ylim(max(0.0, lo_lim - pad), min(1.0, hi_lim + 3 * pad))

    for xi, v, h, sig in zip(x, vals, his, sigs):
        if sig:
            ax.text(xi, v + h + pad, "*", ha="center", va="bottom",
                    fontsize=LEGEND_FS, color="#222222", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([OPERATIONAL_METHOD_SPECS[k][2] for k in methods],
                       rotation=0, ha="center", fontsize=ANNOTATION_FS)


def _plot_operational(
    op_incidence: dict[str, dict],
    op_boot: dict[str, dict],
    op_curves: dict[str, dict[str, pd.DataFrame]],
    metric: Literal["auroc", "auprc"],
    output_path: Path,
    dpi: int,
    style: Literal["bar", "curve"] = "bar",
) -> None:
    is_bar = style == "bar"
    if is_bar:
        ylabel = "AUROC" if metric == "auroc" else "AUPRC"
        xlabel = ""
    else:
        xlabel = "False positive rate" if metric == "auroc" else "Recall"
        ylabel = "True positive rate" if metric == "auroc" else "Precision"

    fig, axes = plt.subplots(2, 2, figsize=(7, 7.5) if is_bar else (7, 6.5),
                             sharex=not is_bar, sharey=not is_bar, constrained_layout=False)
    if is_bar:
        fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.16, wspace=0.30, hspace=1.05)
    else:
        fig.subplots_adjust(left=0.10, right=0.98, top=0.86, bottom=0.14, wspace=0.30, hspace=0.42)

    for row_idx, outcomes in enumerate(OPERATIONAL_GRID):
        for col_idx, outcome in enumerate(outcomes):
            ax = axes[row_idx, col_idx]
            methods = [m for m in OPERATIONAL_METHODS[outcome] if m in op_curves[outcome]]
            incidence = float(op_incidence[outcome]["incidence"])
            n_total = int(op_incidence[outcome]["n_total"])

            if is_bar:
                _draw_op_bar_panel(ax, op_boot[outcome], methods, metric, incidence)
            else:
                _draw_op_curve_panel(ax, op_boot[outcome], op_curves[outcome], methods, metric, incidence)

            ax.set_title(f"{OPERATIONAL_LABELS[outcome]}\n(n={n_total:,}; {_format_incidence(incidence)})",
                         fontsize=TITLE_FS, pad=5)
            ax.tick_params(axis="y", labelsize=TICK_FS, length=2)
            # Keep the (horizontal) x-tick method labels small enough that the
            # four operational labels don't collide; don't let TICK_FS override.
            ax.tick_params(axis="x", labelsize=ANNOTATION_FS, length=2)
            ax.set_xlabel(xlabel if row_idx == 1 else "", fontsize=AXIS_LABEL_FS)
            ax.set_ylabel(ylabel if col_idx == 0 else "", fontsize=AXIS_LABEL_FS)

    used = []
    for row in OPERATIONAL_GRID:
        for outcome in row:
            for k in OPERATIONAL_METHODS[outcome]:
                if k in op_curves[outcome] and k not in used:
                    used.append(k)
    handles = [Line2D([0], [0], color=OPERATIONAL_METHOD_SPECS[k][3], lw=1.8, label=OPERATIONAL_METHOD_SPECS[k][2]) for k in used]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=LEGEND_FS, bbox_to_anchor=(0.5, 0.01))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _write_operational_summary_csv(op_boot: dict[str, dict], target_key: str | None, path: Path) -> None:
    rows = []
    for outcome, boot in op_boot.items():
        for k, m in boot["methods"].items():
            row = {
                "outcome": outcome, "method": k, "target": target_key, "is_target": k == target_key,
                "auroc": m["auroc"], "auroc_ci_lo": m["auroc_ci"][0], "auroc_ci_hi": m["auroc_ci"][1],
                "auprc": m["auprc"], "auprc_ci_lo": m["auprc_ci"][0], "auprc_ci_hi": m["auprc_ci"][1],
            }
            d = boot["diffs"].get(k)
            if d is not None:
                row.update({
                    "auroc_diff_vs_target": d["auroc_diff"],
                    "auroc_diff_ci_lo": d["auroc_diff_ci"][0],
                    "auroc_diff_ci_hi": d["auroc_diff_ci"][1],
                    "auroc_diff_sig": d["auroc_sig"],
                    "auprc_diff_vs_target": d["auprc_diff"],
                    "auprc_diff_ci_lo": d["auprc_diff_ci"][0],
                    "auprc_diff_ci_hi": d["auprc_diff_ci"][1],
                    "auprc_diff_sig": d["auprc_sig"],
                })
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Wrote operational summary: {path}")


# ─── Composite figure ────────────────────────────────────────────────────────

_GRID_X = 200  # resolution of the shared x-grid for mean curve interpolation


def _composite_methods(method_keys: list[str], show_xgboost: bool, show_triage: bool) -> list[str]:
    """Filter method_keys to those shown in the composite figure."""
    excluded = set()
    if not show_xgboost:
        excluded.add("xgboost")
    if not show_triage:
        excluded.add("triage")
    # Keep only methods that are in the vitals method registry and were selected.
    keep = {"mint_softmax", "xgboost", "triage"}
    return [k for k in method_keys if k in keep and k not in excluded]


def _mean_curve(
    task_curves: dict[str, dict[str, pd.DataFrame]],
    display_keys: list[str],
    tasks: list[str],
    metric: Literal["auroc", "auprc"],
) -> dict[str, tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]]:
    """Compute macro-average curve for each method by interpolating onto a shared x-grid.

    Returns {method: (x_grid, mean_y, per_task_curves)} where per_task_curves is a list
    of (x, y) tuples (one per task) used for the dim background lines.
    """
    x_grid = np.linspace(0, 1, _GRID_X)
    out = {}
    for k in display_keys:
        ys = []
        per_task = []
        for task in tasks:
            df = task_curves[task].get(k)
            if df is None:
                continue
            cx, cy = _curve_data(metric, df["labels"].to_numpy(), df["probs"].to_numpy())
            per_task.append((cx, cy))
            # For AUROC: cx is fpr (monotone increasing); for AUPRC: cx is recall
            # (monotone increasing after the [::-1] flip in _curve_data).
            ys.append(np.interp(x_grid, cx, cy))
        if not ys:
            continue
        mean_y = np.mean(np.stack(ys, axis=0), axis=0)
        out[k] = (x_grid, mean_y, per_task)
    return out


def _mean_metric_ci(
    boot_by_task: dict[str, dict],
    arrays_by_task: dict[str, dict[str, dict[str, np.ndarray]]],
    display_keys: list[str],
    tasks: list[str],
    metric: Literal["auroc", "auprc"],
) -> dict[str, tuple[float, float, float]]:
    """Macro-average point estimate and 95% CI for each method.

    For each bootstrap iteration b, average the per-task metric[b] values across
    all tasks (element-wise mean of the (n_tasks, n_boot) matrix).  The CI is the
    2.5/97.5th percentile of the resulting (n_boot,) array — no distributional
    assumptions required.

    Returns {method: (mean_point_est, ci_lo, ci_hi)}.
    """
    out = {}
    for k in display_keys:
        # Point estimate: mean of per-task point estimates.
        point_ests = [boot_by_task[task]["methods"][k][metric]
                      for task in tasks if k in boot_by_task[task]["methods"]]
        if not point_ests:
            continue
        mean_pe = float(np.mean(point_ests))

        # CI: stack per-task raw bootstrap arrays, average column-wise.
        arrs = [arrays_by_task[task][k][metric]
                for task in tasks if k in arrays_by_task.get(task, {})]
        if not arrs:
            out[k] = (mean_pe, float("nan"), float("nan"))
            continue
        stacked = np.stack(arrs, axis=0)  # (n_tasks, n_boot)
        mean_boot = np.nanmean(stacked, axis=0)  # (n_boot,)
        ci_lo, ci_hi = float(np.percentile(mean_boot, 2.5)), float(np.percentile(mean_boot, 97.5))
        out[k] = (mean_pe, ci_lo, ci_hi)
    return out


def _draw_avg_panel(
    ax,
    mean_curves: dict[str, tuple[np.ndarray, np.ndarray, list]],
    mean_ci: dict[str, tuple[float, float, float]],
    display_keys: list[str],
    metric: Literal["auroc", "auprc"],
    show_dim: bool,
    avg_incidence: float | None = None,
    annotation_fs: float = 9,
) -> None:
    """Draw the left-side average AUROC or AUPRC panel."""
    if metric == "auroc":
        ax.plot([0, 1], [0, 1], color=BASELINE_COLOR, lw=0.8, ls="--", zorder=1)
    else:
        if avg_incidence is not None:
            ax.axhline(avg_incidence, color=BASELINE_COLOR, lw=0.8, ls="--", zorder=1)

    for k in display_keys:
        if k not in mean_curves:
            continue
        spec = VITALS_METHODS[k]
        x_grid, mean_y, per_task = mean_curves[k]

        if show_dim:
            for cx, cy in per_task:
                ax.plot(cx, cy, color=spec.color, lw=0.5, alpha=0.25, zorder=2)

        ax.plot(x_grid, mean_y, color=spec.color, lw=spec.lw + 0.4, ls=spec.linestyle, zorder=3)

    entries = []
    for k in display_keys:
        if k not in mean_ci:
            continue
        spec = VITALS_METHODS[k]
        mean_val, lo, hi = mean_ci[k]
        entries.append((mean_val, f"{spec.short}: {mean_val:.3f} ({lo:.3f}–{hi:.3f})"))
    entries.sort(key=lambda x: x[0], reverse=True)

    if metric == "auprc":
        text_x, text_y, ha, va = 0.98, 0.98, "right", "top"
    else:
        text_x, text_y, ha, va = 0.98, 0.04, "right", "bottom"
    if entries:
        ax.text(
            text_x, text_y, "\n".join(line for _, line in entries),
            transform=ax.transAxes, ha=ha, va=va, fontsize=annotation_fs,
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.88),
        )

    xlabel = "False positive rate" if metric == "auroc" else "Recall"
    ylabel = "True positive rate" if metric == "auroc" else "Precision"
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FS)
    title = "Average AUROC" if metric == "auroc" else "Average AUPRC"
    ax.set_title(title, fontsize=TITLE_FS, pad=5, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=TICK_FS, length=2)


def _draw_individual_panel(
    ax,
    task: str,
    summary: dict,
    task_curves_task: dict[str, pd.DataFrame],
    boot: dict,
    display_keys: list[str],
    two_decimals: bool = True,
) -> None:
    """Draw one individual AUROC panel for the right-side grid.

    If MINT is significantly better than XGBoost (paired-diff CI excludes 0),
    the panel spine is drawn in MINT green to signal significance.
    """
    ax.plot([0, 1], [0, 1], color=BASELINE_COLOR, lw=0.6, ls="--", zorder=1)
    for k in display_keys:
        if k not in task_curves_task:
            continue
        spec = VITALS_METHODS[k]
        df = task_curves_task[k]
        cx, cy = _curve_data("auroc", df["labels"].to_numpy(), df["probs"].to_numpy())
        ax.plot(cx, cy, color=spec.color, lw=spec.lw, ls=spec.linestyle, zorder=spec.zorder)

    # Significance: MINT significantly better than XGBoost (diff CI excludes 0, MINT > XGB).
    mint_sig = (
        "mint_softmax" in display_keys
        and "xgboost" in display_keys
        and boot["diffs"].get("xgboost", {}).get("auroc_sig", False)
        and boot["methods"].get("mint_softmax", {}).get("auroc", float("nan"))
        > boot["methods"].get("xgboost", {}).get("auroc", float("nan"))
    )

    # Light green panel background for significant panels.
    if mint_sig:
        ax.patch.set_facecolor(_OKABE["green"])
        ax.patch.set_alpha(0.06)

    # Annotation: each line rendered separately so MINT can be bold when significant.
    # line_h is the axes-fraction height of one 7pt line in a ~1.55 in panel.
    entries = []
    for k in display_keys:
        if k not in boot["methods"]:
            continue
        m = boot["methods"][k]
        val = m["auroc"]
        ci = m["auroc_ci"]
        if np.isnan(val):
            continue
        is_bold = mint_sig and k == "mint_softmax"
        fmt = ".2f" if two_decimals else ".3f"
        entries.append((val, f"{VITALS_METHODS[k].short}: {val:{fmt}} ({ci[0]:{fmt}}-{ci[1]:{fmt}})", is_bold))
    entries.sort(key=lambda x: x[0], reverse=True)

    fs = 8.5
    if not mint_sig:
        # Let matplotlib handle line spacing naturally, identical to the big panels.
        ax.text(
            0.98, 0.04, "\n".join(line for _, line, _ in reversed(entries)),
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=fs, color="#222222",
        )
    else:
        # Manual stack so the MINT line can be bold independently.
        # 1.2× leading matches matplotlib's default \n spacing.
        line_h = (fs / 72) / 1.55 * 1.2
        for i, (_, line, bold) in enumerate(reversed(entries)):
            ax.text(
                0.98, 0.04 + i * line_h, line,
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=fs, fontweight="bold" if bold else "normal",
                color="#222222",
            )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    n_total = int(round(summary["n"] / summary["incidence"]))
    ax.set_title(
        f"{TASK_LABELS[task]}\n(n={n_total:,}; {_format_incidence(float(summary['incidence']))})",
        fontsize=8, pad=3,
    )
    ax.tick_params(labelsize=7, length=2)


def _plot_composite(
    task_curves: dict[str, dict[str, pd.DataFrame]],
    boot_by_task: dict[str, dict],
    arrays_by_task: dict[str, dict[str, dict[str, np.ndarray]]],
    summary_by_task: dict[str, dict],
    method_keys: list[str],
    all_tasks: list[str],
    output_path: Path,
    dpi: int,
    show_xgboost: bool = True,
    show_triage: bool = False,
    show_dim_curves: bool = True,
    two_decimals: bool = True,
) -> None:
    """Composite figure: avg AUROC + avg AUPRC on the left; individual AUROC grid on the right.

    All axes are placed with add_axes using exact inch measurements converted to figure
    fractions. This guarantees every axes is exactly square: no GridSpec hspace/wspace
    approximations. Geometry:
      s  = individual panel side (in)
      L  = left panel side, derived so both sections span the same total height
      right_h = n_rows * s + (n_rows-1) * hs  =>  L = (right_h - lg) / 2
    """
    display_keys = _composite_methods(method_keys, show_xgboost, show_triage)
    if not display_keys:
        print("  [composite] no methods to display; skipping fig1_composite.png")
        return

    airway = [t for t in AIRWAY_TASKS if t in all_tasks]
    circ   = [t for t in CIRCULATION_TASKS if t in all_tasks]
    n_rows = max(len(airway), len(circ))

    avg_incidence = float(np.mean([summary_by_task[t]["incidence"] for t in all_tasks]))
    mean_curves_auroc = _mean_curve(task_curves, display_keys, all_tasks, "auroc")
    mean_curves_auprc = _mean_curve(task_curves, display_keys, all_tasks, "auprc")
    mean_ci_auroc = _mean_metric_ci(boot_by_task, arrays_by_task, display_keys, all_tasks, "auroc")
    mean_ci_auprc = _mean_metric_ci(boot_by_task, arrays_by_task, display_keys, all_tasks, "auprc")

    # ── Geometry (all in inches) ──────────────────────────────────────────────
    s   = 1.55   # individual panel side (square)
    hs  = 0.48   # vertical gap between individual panels (room for FPR/TPR labels)
    ws  = 0.50   # horizontal gap between the two individual columns
    lg  = 0.70   # vertical gap between the two left (avg) panels (room for x-label + title)
    lm  = 0.60   # left margin  (room for y-axis labels on avg panels)
    rm  = 0.20   # right margin
    tm  = 0.65   # top margin   (room for column headers above top individual panels)
    bm  = 0.62   # bottom margin (legend + FPR label on bottom individual panels)
    mg  = 0.55   # gap between left section and right section

    right_h = n_rows * s + (n_rows - 1) * hs
    right_w = 2 * s + ws
    L = (right_h - lg) / 2   # left panel side: equals square when right_h matches

    fig_w = lm + L + mg + right_w + rm
    fig_h = bm + right_h + tm

    fig = plt.figure(figsize=(fig_w, fig_h))

    def _rect(x0: float, y0: float, w: float, h: float) -> list:
        return [x0 / fig_w, y0 / fig_h, w / fig_w, h / fig_h]

    # ── Left panels ──────────────────────────────────────────────────────────
    ax_auprc = fig.add_axes(_rect(lm, bm,          L, L))
    ax_auroc = fig.add_axes(_rect(lm, bm + L + lg, L, L))

    _draw_avg_panel(ax_auroc, mean_curves_auroc, mean_ci_auroc, display_keys,
                    "auroc", show_dim_curves, annotation_fs=14)
    ax_auroc.text(-0.10, 1.04, "a", transform=ax_auroc.transAxes,
                  fontsize=14, fontweight="bold", va="bottom", ha="left", clip_on=False)

    _draw_avg_panel(ax_auprc, mean_curves_auprc, mean_ci_auprc, display_keys,
                    "auprc", show_dim_curves, avg_incidence=avg_incidence, annotation_fs=14)
    ax_auprc.text(-0.10, 1.04, "b", transform=ax_auprc.transAxes,
                  fontsize=14, fontweight="bold", va="bottom", ha="left", clip_on=False)

    # ── Right individual panels ───────────────────────────────────────────────
    rx0_a = lm + L + mg           # airway column left edge
    rx0_c = rx0_a + s + ws        # circulation column left edge

    # Column headers in figure coords, centred above each column.
    header_y = (bm + right_h + tm * 0.55) / fig_h
    fig.text((rx0_a + s / 2) / fig_w, header_y, "Airway/Breathing",
             ha="center", va="center", fontsize=9, fontweight="bold")
    fig.text((rx0_c + s / 2) / fig_w, header_y, "Circulation",
             ha="center", va="center", fontsize=9, fontweight="bold")

    def _row_y0(row_i: int) -> float:
        """Bottom edge in inches for row row_i (0 = topmost)."""
        return bm + (n_rows - 1 - row_i) * (s + hs)

    for row_i, task in enumerate(airway):
        ax = fig.add_axes(_rect(rx0_a, _row_y0(row_i), s, s))
        _draw_individual_panel(ax, task, summary_by_task[task],
                               task_curves[task], boot_by_task[task], display_keys, two_decimals)
        ax.set_ylabel("TPR", fontsize=7)
        if row_i == 0:
            ax.text(-0.22, 1.08, "c", transform=ax.transAxes,
                    fontsize=14, fontweight="bold", va="bottom", ha="left", clip_on=False)
        if row_i == len(airway) - 1:
            ax.set_xlabel("FPR", fontsize=7)
        else:
            ax.tick_params(labelbottom=False)

    for row_i, task in enumerate(circ):
        ax = fig.add_axes(_rect(rx0_c, _row_y0(row_i), s, s))
        _draw_individual_panel(ax, task, summary_by_task[task],
                               task_curves[task], boot_by_task[task], display_keys, two_decimals)
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
        if row_i == len(circ) - 1:
            ax.set_xlabel("FPR", fontsize=7)
        else:
            ax.tick_params(labelbottom=False)

    # Shared legend centred at the bottom.
    handles = _legend_handles(display_keys)
    fig.legend(handles=handles, loc="lower center", ncol=len(display_keys),
               frameon=False, fontsize=LEGEND_FS,
               bbox_to_anchor=(0.5, 0.0), bbox_transform=fig.transFigure)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote composite figure: {output_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    args = tapify(Args)
    final_dir = Path(args.final_dir)
    operational_dir = Path(args.operational_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"

    if args.style not in ("bar", "curve"):
        raise ValueError(f"--style must be 'bar' or 'curve', got '{args.style}'")

    method_keys = list(args.methods) if args.methods else list(VITALS_METHODS.keys())
    unknown = [k for k in method_keys if k not in VITALS_METHODS]
    if unknown:
        raise ValueError(f"Unknown method(s): {unknown}. Valid: {list(VITALS_METHODS)}")

    label_overrides: dict[str, str] = {}
    for item in (args.labels or []):
        if "=" not in item:
            raise ValueError(f"--labels entries must be key=Label, got '{item}'")
        k, lbl = item.split("=", 1)
        if k not in VITALS_METHODS:
            raise ValueError(f"--labels references unknown method '{k}'. Valid: {list(VITALS_METHODS)}")
        label_overrides[k] = lbl
    target_key = args.target_estimator
    if target_key not in method_keys:
        raise ValueError(f"--target-estimator '{target_key}' must be one of the selected --methods {method_keys}")

    print(f"Methods: {method_keys}")
    print(f"Target estimator: {target_key}")

    # ── Vitals / interventions ──
    # Optional task subset: keep the two-row airway/circulation layout, dropping
    # any tasks not requested (and any row that ends up empty).
    if args.tasks:
        valid = {t for row in TASK_GRID for t in row}
        unknown_tasks = [t for t in args.tasks if t not in valid]
        if unknown_tasks:
            raise ValueError(f"Unknown task(s): {unknown_tasks}. Valid: {sorted(valid)}")
        selected = set(args.tasks)
        task_grid = [[t for t in row if t in selected] for row in TASK_GRID]
        task_grid = [row for row in task_grid if row]
    else:
        task_grid = TASK_GRID

    summary_by_task = _load_summary(final_dir)
    all_tasks = [t for row in task_grid for t in row]
    missing = [t for t in all_tasks if t not in summary_by_task]
    if missing:
        raise KeyError(f"Missing summary records for: {', '.join(missing)}")

    task_curves: dict[str, dict[str, pd.DataFrame]] = {}
    boot_by_task: dict[str, dict] = {}
    arrays_by_task: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for task in all_tasks:
        methods_data = _load_task_methods(final_dir, task, method_keys)
        task_curves[task] = methods_data
        boot_by_task[task], arrays_by_task[task] = bootstrap_task_cached(
            final_dir, task, methods_data, method_keys, target_key, args.n_boot, args.seed, cache_dir,
        )

    _write_summary_csv(boot_by_task, target_key, output_dir / "fig1_nejm_paired_differences.csv")

    _plot_metric(summary_by_task, boot_by_task, task_curves, method_keys, target_key,
                 "auroc", output_dir / "fig1_nejm_auroc.png", args.dpi, args.style, label_overrides, task_grid)
    _plot_metric(summary_by_task, boot_by_task, task_curves, method_keys, target_key,
                 "auprc", output_dir / "fig1_nejm_auprc.png", args.dpi, args.style, label_overrides, task_grid)

    _plot_composite(
        task_curves, boot_by_task, arrays_by_task, summary_by_task,
        method_keys, all_tasks,
        output_dir / "fig1_composite.png", args.dpi,
        show_xgboost=not args.hide_xgboost,
        show_triage=args.show_triage,
        show_dim_curves=not args.hide_dim_curves,
        two_decimals=not args.three_decimals,
    )

    # ── Operational (paired bootstrap CIs + paired diffs vs the linear probe) ──
    if not args.skip_operational:
        op_incidence = _load_operational_incidence(operational_dir)
        op_curves: dict[str, dict[str, pd.DataFrame]] = {}
        op_boot: dict[str, dict] = {}
        for row in OPERATIONAL_GRID:
            for outcome in row:
                methods_data = _load_operational_methods(operational_dir, outcome, OPERATIONAL_METHODS[outcome])
                op_curves[outcome] = methods_data
                op_boot[outcome] = _op_bootstrap_cached(
                    operational_dir, outcome, methods_data, OPERATIONAL_TARGET, args.n_boot, args.seed, cache_dir,
                )
        _write_operational_summary_csv(op_boot, OPERATIONAL_TARGET, output_dir / "fig1_operational_summary.csv")
        _plot_operational(op_incidence, op_boot, op_curves, "auroc",
                          output_dir / "fig1_operational_auroc.png", args.dpi, args.style)
        _plot_operational(op_incidence, op_boot, op_curves, "auprc",
                          output_dir / "fig1_operational_auprc.png", args.dpi, args.style)


if __name__ == "__main__":
    main()
