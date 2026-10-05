"""Plotting for fig_atlas: Panels A, B, and D.

Panel A: UMAP colored by cluster (development set, 2024)
Panel B: ED-PEWS vs admission rate, colored by miscalibration gradient (development set)
Panel D: AUROC/AUPRC + operating point metrics (evaluation set, 2025)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
STYLE_PATH = Path(__file__).resolve().parent.parent / "design-skill" / "nature.mplstyle"
SAVE_DIR = REPO_ROOT / "artifacts/fig_atlas"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def _build_cluster_colormap(n_clusters):
    """Build a color array for up to ~80 clusters using tab20 + tab20b + twilight + terrain."""
    colors = []
    for cmap_name in ["tab20", "tab20b"]:
        cmap = plt.colormaps.get_cmap(cmap_name)
        colors.extend([cmap(i / 20) for i in range(20)])
    for cmap_name in ["twilight", "terrain"]:
        cmap = plt.colormaps.get_cmap(cmap_name)
        colors.extend([cmap(i / 20) for i in range(1, 20)])
    return colors[:n_clusters]


# ──────────────────────────────────────────────────────────────────────────────
# Panel A: UMAP colored by cluster
# ──────────────────────────────────────────────────────────────────────────────

LABELED_CLUSTERS = {
    0: "Complex high-acuity\ninpatient transfer",
    47: "Lower extremity\nsports injuries",
    37: "Adolescent surgical\nand trauma admissions",
    49: "Toddler croup and\nviral wheezing",
    # Additional clusters near key neighborhoods
    48: "Ankle/foot fractures\nand sprains",
    46: "Upper extremity\nfractures",
    44: "Febrile infant\nworkup",
    41: "Bronchiolitis and\nrespiratory distress",
}

# Offset directions for leader lines (dx, dy in points) to avoid overlap
_LABEL_OFFSETS = {
    0: (40, 30),
    47: (-45, -25),
    37: (40, -25),
    49: (-40, 30),
    48: (-45, -35),
    46: (45, -20),
    44: (40, 25),
    41: (-40, -30),
}


def plot_panel_a():
    """UMAP of representations colored by cluster assignment (development set)."""
    plt.style.use(str(STYLE_PATH))

    clusters_data = np.load(SAVE_DIR / "clusters.npz", allow_pickle=True)
    umap_data = np.load(SAVE_DIR / "umap.npz")

    labels = clusters_data["labels"]
    embedding = umap_data["embedding"]
    n_clusters = labels.max() + 1

    colors = _build_cluster_colormap(n_clusters)

    # Load cluster stats for N and ICU rate
    cluster_stats = pd.read_csv(SAVE_DIR / "cluster_stats.csv")
    stats_lookup = cluster_stats.set_index("cluster_id")

    fig, ax = plt.subplots(figsize=(4.5, 4))

    for cid in range(n_clusters):
        mask = labels == cid
        if not mask.any():
            continue
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=[colors[cid]], s=2, alpha=0.5,
            edgecolors="none", rasterized=True,
        )

    # Annotate selected clusters with leader lines
    for cid, label_text in LABELED_CLUSTERS.items():
        mask = labels == cid
        if not mask.any():
            continue
        cx = embedding[mask, 0].mean()
        cy = embedding[mask, 1].mean()

        # Build label with stats
        if cid in stats_lookup.index:
            n = int(stats_lookup.loc[cid, "N"])
            icu_pct = stats_lookup.loc[cid, "icu_rate"] * 100
            full_label = f"{label_text}\n(N={n:,}; ICU {icu_pct:.0f}%)"
        else:
            full_label = label_text

        offset = _LABEL_OFFSETS.get(cid, (30, 20))
        ax.annotate(
            full_label,
            xy=(cx, cy),
            xytext=offset,
            textcoords="offset points",
            fontsize=5,
            fontweight="medium",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.3",
                fc="white",
                ec="#4a4a4a",
                alpha=0.92,
                lw=0.4,
            ),
            arrowprops=dict(
                arrowstyle="-",
                color="#4a4a4a",
                lw=0.5,
                connectionstyle="arc3,rad=0.1",
            ),
            zorder=10,
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(-0.08, 1.05, "a", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")

    out = SAVE_DIR / "panel_a_umap.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(SAVE_DIR / "panel_a_umap.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Panel B: ED-PEWS vs admission rate, colored by miscalibration
# ──────────────────────────────────────────────────────────────────────────────

def _miscalibration_color(mean_pews, admit_rate, pews_all, admit_all):
    """Compute residual from trend line as miscalibration metric.

    Positive residual = higher admission than PEWS predicts (underestimated risk)
    Negative residual = lower admission than PEWS predicts (overestimated risk)
    """
    coeffs = np.polyfit(pews_all, admit_all, 1)
    predicted_admit = np.polyval(coeffs, mean_pews)
    residual = admit_rate - predicted_admit
    return residual


def plot_panel_b():
    """Scatter of mean ED-PEWS vs admission rate, colored by miscalibration."""
    plt.style.use(str(STYLE_PATH))

    cluster_df = pd.read_csv(SAVE_DIR / "cluster_stats.csv")
    cluster_df = cluster_df.dropna(subset=["mean_pews"])

    # Compute residuals (miscalibration)
    coeffs = np.polyfit(cluster_df["mean_pews"], cluster_df["admit_rate"], 1)
    predicted = np.polyval(coeffs, cluster_df["mean_pews"])
    residuals = cluster_df["admit_rate"].values - predicted
    r_val = np.corrcoef(cluster_df["mean_pews"], cluster_df["admit_rate"])[0, 1]

    # Diverging colormap: blue = overestimate (low admit for PEWS), red = underestimate
    norm = mcolors.TwoSlopeNorm(vmin=residuals.min(), vcenter=0, vmax=residuals.max())
    cmap = plt.colormaps.get_cmap("RdBu_r")

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    sizes = 30 + (cluster_df["N"] / cluster_df["N"].max()) * 120

    for i, (_, row) in enumerate(cluster_df.iterrows()):
        cid = int(row["cluster_id"])
        color = cmap(norm(residuals[i]))
        ax.scatter(
            row["mean_pews"], row["admit_rate"],
            s=sizes.iloc[i],
            c=[color], edgecolors="black", linewidths=0.3,
            alpha=0.85, zorder=3,
        )
        ax.annotate(
            str(cid),
            (row["mean_pews"], row["admit_rate"]),
            fontsize=5, ha="center", va="bottom",
            xytext=(0, 3), textcoords="offset points",
        )

    # Trend line
    x_fit = np.linspace(cluster_df["mean_pews"].min(), cluster_df["mean_pews"].max(), 100)
    y_fit = np.polyval(coeffs, x_fit)
    ax.plot(x_fit, y_fit, "k--", lw=0.8, alpha=0.5, label=f"r = {r_val:.2f}")

    ax.set_xlabel("Mean ED-PEWS Score")
    ax.set_ylabel("Hospital Admission Rate")
    ax.legend(loc="upper left", fontsize=6)
    ax.text(-0.12, 1.05, "b", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Miscalibration\n(admission residual)", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    out = SAVE_DIR / "panel_b_pews_vs_admit.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(SAVE_DIR / "panel_b_pews_vs_admit.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out} (r={r_val:.3f})")


# ──────────────────────────────────────────────────────────────────────────────
# Panel D: AUROC/AUPRC + operating point metrics (evaluation set, 2025)
# ──────────────────────────────────────────────────────────────────────────────

def bootstrap_auroc(y_true, y_score, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(roc_auc_score(y_true[idx], y_score[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bootstrap_auprc(y_true, y_score, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(average_precision_score(y_true[idx], y_score[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bootstrap_metric(y_true, y_score, metric_fn, n_boot=1000, seed=42):
    """Bootstrap any metric function that takes (y_true, y_score) -> float."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(metric_fn(y_true[idx], y_score[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def sensitivity_at_threshold(y_true, y_score, threshold):
    """Sensitivity (recall) for y_score >= threshold predicting positive."""
    predicted_pos = y_score >= threshold
    tp = (predicted_pos & (y_true == 1)).sum()
    fn = (~predicted_pos & (y_true == 1)).sum()
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def specificity_at_threshold(y_true, y_score, threshold):
    """Specificity for y_score >= threshold predicting positive."""
    predicted_pos = y_score >= threshold
    tn = (~predicted_pos & (y_true == 0)).sum()
    fp = (predicted_pos & (y_true == 0)).sum()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def compute_operating_points(y_admit, y_icu, y_score, label):
    """Compute sensitivity@<6 for ICU ruleout and specificity@>15 for admission."""
    # Sensitivity at >=6 for ICU (rule-out: score <6 means low risk)
    # "sensitivity for inverse of kids who go to ICU" = among ICU kids, what fraction score >=6?
    sens_icu = sensitivity_at_threshold(y_icu, y_score, 6)
    sens_icu_ci = bootstrap_metric(
        y_icu, y_score,
        lambda yt, ys: sensitivity_at_threshold(yt, ys, 6))

    # Specificity at >15 for admission (rule-in: score >15 means high risk)
    spec_admit = specificity_at_threshold(y_admit, y_score, 16)
    spec_admit_ci = bootstrap_metric(
        y_admit, y_score,
        lambda yt, ys: specificity_at_threshold(yt, ys, 16))

    # Also compute sensitivity at >=16 for admission (rule-in PPV sense)
    sens_admit_16 = sensitivity_at_threshold(y_admit, y_score, 16)
    sens_admit_16_ci = bootstrap_metric(
        y_admit, y_score,
        lambda yt, ys: sensitivity_at_threshold(yt, ys, 16))

    print(f"\n  Operating points ({label}):")
    print(f"    ICU rule-out (score >=6): sensitivity = {sens_icu:.3f} "
          f"[{sens_icu_ci[0]:.3f}-{sens_icu_ci[1]:.3f}]")
    print(f"    Admission rule-in (score >15): specificity = {spec_admit:.3f} "
          f"[{spec_admit_ci[0]:.3f}-{spec_admit_ci[1]:.3f}]")
    print(f"    Admission rule-in (score >15): sensitivity = {sens_admit_16:.3f} "
          f"[{sens_admit_16_ci[0]:.3f}-{sens_admit_16_ci[1]:.3f}]")

    return {
        "icu_sensitivity_at_6": (sens_icu, sens_icu_ci),
        "admit_specificity_at_16": (spec_admit, spec_admit_ci),
        "admit_sensitivity_at_16": (sens_admit_16, sens_admit_16_ci),
    }


def _bootstrap_roc_curves(y_true, y_score, n_boot=200, seed=42):
    """Bootstrap ROC curves and return pointwise 95% CI bands."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        fpr_b, tpr_b, _ = roc_curve(y_true[idx], y_score[idx])
        tpr_interp = np.interp(mean_fpr, fpr_b, tpr_b)
        tprs.append(tpr_interp)
    tprs = np.array(tprs)
    return mean_fpr, np.percentile(tprs, 2.5, axis=0), np.percentile(tprs, 97.5, axis=0)


def _bootstrap_prc_curves(y_true, y_score, n_boot=200, seed=42):
    """Bootstrap precision-recall curves and return pointwise 95% CI bands."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    mean_rec = np.linspace(0, 1, 100)
    precs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        prec_b, rec_b, _ = precision_recall_curve(y_true[idx], y_score[idx])
        # precision_recall_curve returns descending recall; flip for interp
        prec_interp = np.interp(mean_rec, rec_b[::-1], prec_b[::-1])
        precs.append(prec_interp)
    precs = np.array(precs)
    return mean_rec, np.percentile(precs, 2.5, axis=0), np.percentile(precs, 97.5, axis=0)


def plot_panel_d():
    """AUROC and AUPRC for original vs adjusted ED-PEWS on 2025 evaluation set."""
    plt.style.use(str(STYLE_PATH))

    enc_df = pd.read_parquet(SAVE_DIR / "encounters_eval.parquet")
    enc_df = enc_df[enc_df["has_complete_vitals"]].copy()
    enc_df = enc_df.dropna(subset=["pews_original", "pews_adjusted"])

    y_admit = enc_df["has_admit"].astype(int).values
    y_icu = enc_df["has_icu"].astype(int).values
    y_orig = enc_df["pews_original"].values
    y_adj = enc_df["pews_adjusted"].values

    print(f"  Evaluation set: {len(enc_df)} encounters with complete data")
    print(f"  Admissions: {y_admit.sum()} ({100*y_admit.mean():.1f}%)")
    print(f"  ICU: {y_icu.sum()} ({100*y_icu.mean():.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))

    colors = {"Original ED-PEWS": "#d62728", "Adjusted ED-PEWS": "#1f77b4"}

    # --- AUROC ---
    ax = axes[0]
    for name, scores in [("Original ED-PEWS", y_orig), ("Adjusted ED-PEWS", y_adj)]:
        fpr, tpr, _ = roc_curve(y_admit, scores)
        auroc = roc_auc_score(y_admit, scores)
        ci_lo, ci_hi = bootstrap_auroc(y_admit, scores)
        label = f"{name} ({auroc:.3f} [{ci_lo:.3f}-{ci_hi:.3f}])"
        ax.plot(fpr, tpr, color=colors[name], lw=1.5, label=label)

        # CI band
        mean_fpr, tpr_lo, tpr_hi = _bootstrap_roc_curves(y_admit, scores)
        ax.fill_between(mean_fpr, tpr_lo, tpr_hi,
                        color=colors[name], alpha=0.15, linewidth=0)

    ax.plot([0, 1], [0, 1], "k--", lw=0.5, alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Hospital Admission — AUROC")
    ax.legend(loc="lower right", fontsize=5.5)

    # --- AUPRC ---
    ax = axes[1]
    for name, scores in [("Original ED-PEWS", y_orig), ("Adjusted ED-PEWS", y_adj)]:
        prec, rec, _ = precision_recall_curve(y_admit, scores)
        auprc = average_precision_score(y_admit, scores)
        ci_lo, ci_hi = bootstrap_auprc(y_admit, scores)
        label = f"{name} ({auprc:.3f} [{ci_lo:.3f}-{ci_hi:.3f}])"
        ax.plot(rec, prec, color=colors[name], lw=1.5, label=label)

        # CI band
        mean_rec, prec_lo, prec_hi = _bootstrap_prc_curves(y_admit, scores)
        ax.fill_between(mean_rec, prec_lo, prec_hi,
                        color=colors[name], alpha=0.15, linewidth=0)

    ax.axhline(y_admit.mean(), color="gray", ls=":", lw=0.5, alpha=0.5)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Hospital Admission — AUPRC")
    ax.legend(loc="upper right", fontsize=5.5)

    axes[0].text(-0.12, 1.05, "d", transform=axes[0].transAxes,
                 fontsize=11, fontweight="bold", va="top")

    out = SAVE_DIR / "panel_d_auroc_auprc.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(SAVE_DIR / "panel_d_auroc_auprc.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

    # Operating points
    ops_orig = compute_operating_points(y_admit, y_icu, y_orig, "Original")
    ops_adj = compute_operating_points(y_admit, y_icu, y_adj, "Adjusted")

    # Save metrics
    metrics = {
        "n_eval": len(enc_df),
        "n_admit": int(y_admit.sum()),
        "n_icu": int(y_icu.sum()),
        "original": ops_orig,
        "adjusted": ops_adj,
    }
    import json
    with open(SAVE_DIR / "panel_d_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Main plot entry
# ──────────────────────────────────────────────────────────────────────────────

def run_plot(panel=None):
    print("=" * 60)
    print("Fig Atlas: Plotting")
    print("=" * 60)

    if panel is None or panel == "a":
        print("\nPanel A: UMAP by cluster...")
        plot_panel_a()

    if panel is None or panel == "b":
        print("\nPanel B: ED-PEWS vs admission rate (miscalibration gradient)...")
        plot_panel_b()

    if panel is None or panel == "d":
        print("\nPanel D: AUROC/AUPRC comparison (2025 evaluation set)...")
        plot_panel_d()

    print("\nDone.")
