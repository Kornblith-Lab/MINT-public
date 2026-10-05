"""Supplementary token-ablation figure: AUROC/AUPRC change vs the full model.

How much does each input token type (and each temporal token resolution) matter?
For every ablation we recompute the paired clustered bootstrap of

    (ablation - full MINT)

per task, for AUROC and AUPRC together, and render the differences as heatmaps.
Negative (blue) means the ablated model is WORSE than the full model, i.e. the
removed information helped. A ``*`` marks cells whose 95% paired-difference CI
excludes 0 (significant on that task).

The figure has two stacked blocks, each a pair of AUROC | AUPRC heatmaps sharing
one diverging color scale per metric:

    Top block  -- token-type ablations   (rows: Lab, Med, O2Device, Procedure, Vital)
    Bottom block -- temporal fidelity     (rows: Hourly, Daily)

Temporal fidelity is a SEPARATE block (different axes), not extra rows: the full
model tokenizes every minute; "Hourly" collapses to one token/hour and "Daily"
to one/day, so these ablate temporal resolution rather than a token type.

The pairing/CI method is identical to fig1_nejm.py (a single encounter-level
resample shared across the two methods each iteration; diffs on the per-iteration
intersection). Results are cached under <output_dir>/cache/.

RUN (from repo root):
    python -m mint.five.fig_one.fig_token_ablation \
        --final_dir artifacts/fig1-final \
        --output_dir artifacts/fig1_nejm_token_ablation

OUTPUT (in --output_dir)
    fig_token_ablation.png           heatmap blocks (AUPRC by default; see --metrics)
    fig_token_ablation_summary.csv   per-ablation/task diffs + CIs + significance

    --metrics {auprc, auroc, both}   which metric column(s) to render (default auprc)

INPUT LAYOUT (under --final_dir)
    fig1-final-softmax/{task}_softmax.csv                         full model (baseline)
    fig1-final-token-ablation/fig1-ablate-{X}/{task}_softmax.csv  token-type ablations
    fig1-60-softmax/{task}_softmax.csv                            hourly tokens
    fig1-1440-softmax/{task}_softmax.csv                          daily tokens
    Every CSV must carry probs / labels / encounter_key.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tap import tapify

from mint.five.fig_one.fig1_nejm import (
    TASK_LABELS,
    _file_hash,
    _load_method_csv,
    paired_bootstrap,
)


plt.style.use(str(Path(__file__).resolve().parents[1] / "design-skill" / "nature.mplstyle"))
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 10,
    }
)

TITLE_FS = 12
CELL_FS = 7.5
STAR_FS = 9

# Column order: all vitals (physiologic signs) first, then interventions. A
# separator is drawn between the two groups (after VITALS_COUNT columns).
VITALS_TASKS = ["hypoxia", "tachypnea", "tachycardia", "hypotension", "periarrest"]
INTERVENTION_TASKS = ["ppv", "resp_rescue", "ventilator", "vasopressor", "transfusion"]
TASKS = VITALS_TASKS + INTERVENTION_TASKS
VITALS_COUNT = len(VITALS_TASKS)

# ── Ablation registry ────────────────────────────────────────────────────────
# key -> (row label, subdir under --final_dir, {task}-templated filename)

TOKEN_ABLATIONS = [
    ("Vital",     "fig1-final-token-ablation/fig1-ablate-Vital",     "{task}_softmax.csv"),
    ("O2 device", "fig1-final-token-ablation/fig1-ablate-O2Device",  "{task}_softmax.csv"),
    ("Medication","fig1-final-token-ablation/fig1-ablate-Med",       "{task}_softmax.csv"),
    ("Procedure", "fig1-final-token-ablation/fig1-ablate-Procedure", "{task}_softmax.csv"),
    ("Lab",       "fig1-final-token-ablation/fig1-ablate-Lab",       "{task}_softmax.csv"),
]
FIDELITY_ABLATIONS = [
    ("Hourly", "fig1-60-softmax",   "{task}_softmax.csv"),
    ("Daily",  "fig1-1440-softmax", "{task}_softmax.csv"),
]
BASELINE_SUBDIR = "fig1-final-softmax"


@dataclass
class Args:
    final_dir: str = "artifacts/fig1-final"
    output_dir: str = "artifacts/fig1_nejm_token_ablation"
    metrics: str = "auprc"  # {auprc, auroc, both}: which metric panel(s) to show
    n_boot: int = 1000
    seed: int = 42
    dpi: int = 600


# ── Diff computation (paired bootstrap, cached) ───────────────────────────────

def _diff_cached(
    final_dir: Path, baseline_df: pd.DataFrame, abl_subdir: str, filename: str,
    task: str, n_boot: int, seed: int, cache_dir: Path,
) -> dict:
    """Paired (ablation - full) AUROC/AUPRC diff + CIs for one task, cached.

    Uses paired_bootstrap with two methods ("full" as target, "abl" as the other)
    so the returned diff is abl - full on the shared encounter intersection."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = final_dir / BASELINE_SUBDIR / f"{task}_softmax.csv"
    abl_path = final_dir / abl_subdir / filename.format(task=task)
    ck = hashlib.md5(json.dumps({
        "task": task, "abl": abl_subdir, "n_boot": n_boot, "seed": seed,
        "base": _file_hash(baseline_path), "ablf": _file_hash(abl_path),
    }, sort_keys=True).encode()).hexdigest()
    cache_path = cache_dir / f"abl_{task}_{ck}.pkl"
    if cache_path.exists():
        return pickle.loads(cache_path.read_bytes())
    abl_df = _load_method_csv(abl_path, "probs")
    boot = paired_bootstrap(
        {"full": baseline_df, "abl": abl_df}, ["full", "abl"], "full", n_boot, seed,
    )
    d = boot["diffs"]["abl"]
    result = {
        "auroc_diff": d["auroc_diff"], "auroc_sig": d["auroc_sig"], "auroc_ci": d["auroc_diff_ci"],
        "auprc_diff": d["auprc_diff"], "auprc_sig": d["auprc_sig"], "auprc_ci": d["auprc_diff_ci"],
    }
    cache_path.write_bytes(pickle.dumps(result))
    return result


def _build_matrices(
    final_dir: Path, ablations: list[tuple[str, str, str]],
    n_boot: int, seed: int, cache_dir: Path,
) -> dict:
    """Return diff/sig matrices [rows=ablations, cols=TASKS] for both metrics."""
    baselines = {t: _load_method_csv(final_dir / BASELINE_SUBDIR / f"{t}_softmax.csv", "probs") for t in TASKS}
    shape = (len(ablations), len(TASKS))
    out = {m: {"diff": np.full(shape, np.nan), "sig": np.zeros(shape, dtype=bool)}
           for m in ("auroc", "auprc")}
    rows = []
    for i, (label, subdir, fname) in enumerate(ablations):
        print(f"  [{label}]")
        for j, task in enumerate(TASKS):
            r = _diff_cached(final_dir, baselines[task], subdir, fname, task, n_boot, seed, cache_dir)
            for m in ("auroc", "auprc"):
                out[m]["diff"][i, j] = r[f"{m}_diff"]
                out[m]["sig"][i, j] = r[f"{m}_sig"]
            rows.append({
                "ablation": label, "task": task,
                "auroc_diff": r["auroc_diff"], "auroc_ci_lo": r["auroc_ci"][0],
                "auroc_ci_hi": r["auroc_ci"][1], "auroc_sig": r["auroc_sig"],
                "auprc_diff": r["auprc_diff"], "auprc_ci_lo": r["auprc_ci"][0],
                "auprc_ci_hi": r["auprc_ci"][1], "auprc_sig": r["auprc_sig"],
            })
    out["rows"] = [a[0] for a in ablations]
    out["records"] = rows
    return out


# ── Plotting ──────────────────────────────────────────────────────────────────

def _vmax(*mats: dict, metric: str) -> float:
    """Symmetric color limit shared across the blocks of one metric."""
    vals = np.concatenate([np.abs(m[metric]["diff"][~np.isnan(m[metric]["diff"])].ravel()) for m in mats])
    return float(np.nanmax(vals)) if vals.size else 0.01


def _draw_heatmap(ax, mat: dict, metric: str, vmax: float, show_xticks: bool):
    """Draw one heatmap block; return the image (for a shared colorbar)."""
    diff = mat[metric]["diff"]
    sig = mat[metric]["sig"]

    # Prepend a mean-across-tasks column as the leftmost summary column.
    mean_col = np.nanmean(diff, axis=1, keepdims=True)
    diff = np.concatenate([mean_col, diff], axis=1)
    sig = np.concatenate([np.zeros((sig.shape[0], 1), dtype=bool), sig], axis=1)
    n_cols = diff.shape[1]

    # RdBu: negative (ablated worse than full) -> red, positive -> blue.
    im = ax.imshow(diff, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")

    for i in range(diff.shape[0]):
        for j in range(n_cols):
            v = diff[i, j]
            if np.isnan(v):
                continue
            star = "*" if sig[i, j] else ""
            # White text on saturated cells, dark on pale ones.
            txt_color = "white" if abs(v) > 0.55 * vmax else "#222222"
            fw = "bold" if j == 0 else "normal"
            ax.text(j, i, f"{v*100:+.1f}{star}", ha="center", va="center",
                    fontsize=CELL_FS, color=txt_color, fontweight=fw)

    ax.set_xticks(np.arange(n_cols))
    all_labels = ["Average"] + [TASK_LABELS[t] for t in TASKS]
    if show_xticks:
        ax.set_xticklabels(all_labels, rotation=40, ha="right", fontsize=8)
        ax.get_xticklabels()[0].set_fontweight("bold")
    else:
        ax.set_xticklabels([])
    ax.set_yticks(np.arange(len(mat["rows"])))
    ax.set_yticklabels(mat["rows"])
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(mat["rows"]), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)

    # Thick separator after the mean summary column.
    ax.axvline(0.5, color="black", linewidth=2.5, zorder=5)
    # Separator between vitals and intervention groups (shifted right by 1).
    ax.axvline(VITALS_COUNT + 0.5, color="black", linewidth=1.4, zorder=5)
    return im


def _plot(token_mat: dict, fid_mat: dict, metrics: list[str], output_path: Path, dpi: int) -> None:
    # One column of stacked blocks per requested metric. Block heights are
    # proportional to row counts so cells stay square-ish.
    n_top, n_bot = len(token_mat["rows"]), len(fid_mat["rows"])
    ncols = len(metrics)
    fig, axes = plt.subplots(
        2, ncols, figsize=(7.5 * ncols, 6.4), squeeze=False,
        gridspec_kw={"height_ratios": [n_top, n_bot], "hspace": 0.12, "wspace": 0.14},
    )
    fig.subplots_adjust(left=0.075 if ncols == 2 else 0.14,
                        right=0.93, top=0.92, bottom=0.24)

    for col, metric in enumerate(metrics):
        vmax = _vmax(token_mat, fid_mat, metric=metric)
        # Top block: token-type ablations (x-ticks hidden; shared with bottom).
        _draw_heatmap(axes[0, col], token_mat, metric, vmax, show_xticks=False)
        # Bottom block: temporal fidelity (shares the metric's color scale).
        im = _draw_heatmap(axes[1, col], fid_mat, metric, vmax, show_xticks=True)
        axes[0, col].set_title(metric.upper(), fontsize=TITLE_FS, pad=8)
        # One colorbar per metric, spanning both blocks of that column.
        cbar = fig.colorbar(im, ax=[axes[0, col], axes[1, col]], fraction=0.03, pad=0.02)
        cbar.set_label(f"Δ {metric.upper()} (×100) vs full", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    axes[0, 0].set_ylabel("Token type ablated", fontsize=10)
    axes[1, 0].set_ylabel("Temporal\nfidelity", fontsize=10)

    # Panel labels "a" and "b" on the leftmost axes of each block.
    for label, ax in zip(("a", "b"), (axes[0, 0], axes[1, 0])):
        ax.text(-0.18, 1.06, label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="left")

    # Suggested caption (kept out of the figure so it can live in the manuscript):
    #   Ablation effect on discrimination (ablation − full MINT). Cells show Δ×100;
    #   red = ablated model worse than full, blue = better. * marks a 95%
    #   paired-difference CI excluding 0.
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--final_dir", default="artifacts/fig1-final")
    p.add_argument("--output_dir", default="artifacts/fig1_nejm_token_ablation")
    p.add_argument("--metrics", default="auprc")
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dpi", type=int, default=600)
    ns = p.parse_args()
    args = Args(
        final_dir=ns.final_dir, output_dir=ns.output_dir, metrics=ns.metrics,
        n_boot=ns.n_boot, seed=ns.seed, dpi=ns.dpi,
    )
    final_dir = Path(args.final_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"

    metric_map = {"auprc": ["auprc"], "auroc": ["auroc"], "both": ["auroc", "auprc"]}
    if args.metrics not in metric_map:
        raise ValueError(f"--metrics must be one of {list(metric_map)}, got '{args.metrics}'")
    metrics = metric_map[args.metrics]

    print("Token-type ablations:")
    token_mat = _build_matrices(final_dir, TOKEN_ABLATIONS, args.n_boot, args.seed, cache_dir)
    print("Temporal-fidelity ablations:")
    fid_mat = _build_matrices(final_dir, FIDELITY_ABLATIONS, args.n_boot, args.seed, cache_dir)

    pd.DataFrame(token_mat["records"] + fid_mat["records"]).to_csv(
        output_dir / "fig_token_ablation_summary.csv", index=False)
    print(f"Wrote {output_dir / 'fig_token_ablation_summary.csv'}")

    out_path = output_dir / "fig_token_ablation.png"
    _plot(token_mat, fid_mat, metrics, out_path, args.dpi)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()