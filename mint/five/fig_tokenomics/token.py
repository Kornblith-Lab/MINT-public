# python -m mint.five.fig_tokenomics.token

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb
from matplotlib.lines import Line2D
import plotly.express as px
import umap

STYLE_PATH = Path(__file__).parent.parent / "design-skill" / "nature.mplstyle"
SAVE_DIR = Path("artifacts/fig_umap")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = "output/mint/ckpt.pt"
VOCAB_PATH = "output/vocab.csv"
MIN_COUNT = 300

CATEGORY_ORDER = [
    "Vitals and Respiratory Support", "Medication", "Chief Complaint", "Procedure",
    "Lab", "Age", "Sex", "Arrival", "Acuity", "Disposition",
]
CATEGORY_COLORS = {
    "Vitals and Respiratory Support": "#0072B2",
    "Medication": "#E69F00",
    "Chief Complaint": "#009E73",
    "Procedure": "#CC79A7",
    "Lab": "#D55E00",
    "Age": "#56B4E9",
    "Arrival": "#F0E442",
    "Acuity": "#d62728",
    "Disposition": "#000000",
    "Sex": "#6A3D9A",
}


# Vital types that carry a numeric value and should be shaded by that value.
# "O2 Device" is intentionally excluded (categorical values, not numeric).
GRADIENT_VITAL_EXCLUDE = {"O2 Device"}
# Fraction of the light->dark ramp actually used, so the lightest points stay
# visible (not near-white) and the darkest stay distinct from black.
GRADIENT_LIGHT = 0.35
GRADIENT_DARK = 1.0


def _value_ramp(base_hex: str) -> LinearSegmentedColormap:
    """Light-tinted -> full base color ramp, so low values are lighter."""
    base = np.array(to_rgb(base_hex))
    light = base + (1.0 - base) * 0.85  # pale tint of the base color
    return LinearSegmentedColormap.from_list("ramp", [light, base])


def vital_type(name: str) -> str | None:
    """Return the vital sub-type, e.g. 'Systolic' for 'Vital_Systolic_120'."""
    if not name.startswith("Vital_"):
        return None
    parts = name.split("_")
    if len(parts) < 3:
        return None
    return "_".join(parts[1:-1])


def token_value(name: str) -> float:
    """Numeric value from the trailing '_{X}' field, or NaN if not numeric."""
    try:
        return float(name.rsplit("_", 1)[-1])
    except ValueError:
        return np.nan


def assign_colors(df: pd.DataFrame) -> pd.Series:
    """Per-token colors: flat category color, or a within-group value gradient.

    Gradients apply to Age (all tokens) and each numeric Vital sub-type,
    normalizing the value within the group so low->light, high->dark.
    Vital 'O2 Device' and any non-numeric tokens keep the flat category color.
    """
    colors = df["category"].map(CATEGORY_COLORS).copy()

    def shade(mask: pd.Series, base_hex: str):
        good = mask & df["value"].notna()
        if good.sum() == 0:
            return
        v = df.loc[good, "value"]
        lo, hi = v.min(), v.max()
        norm = (v - lo) / (hi - lo) if hi > lo else pd.Series(0.5, index=v.index)
        cmap = _value_ramp(base_hex)
        scaled = GRADIENT_LIGHT + (GRADIENT_DARK - GRADIENT_LIGHT) * norm
        colors.loc[good] = [to_hex(cmap(t)) for t in scaled]

    # Age: single numeric group.
    shade(df["category"] == "Age", CATEGORY_COLORS["Age"])

    # Vitals: one gradient per numeric sub-type.
    vital = df["category"] == "Vitals and Respiratory Support"
    for vt in df.loc[vital, "vital_type"].dropna().unique():
        if vt in GRADIENT_VITAL_EXCLUDE:
            continue
        shade(vital & (df["vital_type"] == vt), CATEGORY_COLORS["Vitals and Respiratory Support"])

    return colors


def categorize_token(name: str) -> str:
    if name.startswith("Vital_"):
        return "Vitals and Respiratory Support"
    if name.startswith("Med_"):
        return "Medication"
    if name.startswith("CC_"):
        return "Chief Complaint"
    if name.startswith("Procedure_"):
        return "Procedure"
    if name.startswith("Lab_"):
        return "Lab"
    if name.startswith("Age_"):
        return "Age"
    if name.startswith("Arrival_"):
        return "Arrival"
    if name.startswith("Acuity_"):
        return "Acuity"
    if name.startswith("Sex_"):
        return "Sex"
    print(name)
    return "Disposition"


def load_embeddings() -> pd.DataFrame:
    vocab = pd.read_csv(VOCAB_PATH)
    high = vocab[vocab["count"] >= MIN_COUNT].copy()

    model_ids = (high["index"].values + 1).astype(int)
    state = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    wte = state["model"]["transformer.wte.weight"].numpy()
    embeddings = wte[model_ids]

    high = high.reset_index(drop=True)
    high["category"] = high["name"].apply(categorize_token)
    high["vital_type"] = high["name"].apply(vital_type)
    high["value"] = high["name"].apply(token_value)
    high["color"] = assign_colors(high)
    high["embedding"] = list(embeddings)
    return high


def run_umap(embeddings: np.ndarray) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=2, random_state=42, n_neighbors=50, min_dist=0.3, spread=1.0
    )
    return reducer.fit_transform(embeddings)


def plot_static(df: pd.DataFrame):
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))

    fig, ax = plt.subplots(figsize=(3.5, 3.5), layout="none")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

    legend_handles = []
    for cat in CATEGORY_ORDER:
        mask = df["category"] == cat
        if not mask.any():
            continue
        sub = df[mask]
        ax.scatter(
            sub["umap_1"], sub["umap_2"],
            c=sub["color"].tolist(), s=6, alpha=0.9,
            edgecolors="white", linewidths=0.2,
            rasterized=True,
        )
        legend_handles.append(Line2D(
            [], [], marker="o", linestyle="none", markersize=3,
            markerfacecolor=CATEGORY_COLORS[cat], markeredgecolor="white",
            markeredgewidth=0.2, label=f"{cat}", # f"{cat} ({mask.sum()})",
        ))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    pad = 0.02
    xrange = df["umap_1"].max() - df["umap_1"].min()
    yrange = df["umap_2"].max() - df["umap_2"].min()
    ax.set_xlim(df["umap_1"].min() - pad * xrange, df["umap_1"].max() + pad * xrange)
    ax.set_ylim(df["umap_2"].min() - pad * yrange, df["umap_2"].max() + pad * yrange)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(handles=legend_handles, markerscale=1.5, fontsize=4,
              loc="upper left", framealpha=0.95, handletextpad=0.3,
              borderpad=0.3, edgecolor="0.8", labelspacing=0.3)

    plt.savefig(SAVE_DIR / "token_embeddings.pdf", dpi=1200)
    plt.savefig(SAVE_DIR / "token_embeddings.png", dpi=1200)
    plt.close()
    print(f"Saved static figure to {SAVE_DIR / 'token_embeddings.png'}")


def plot_interactive(df: pd.DataFrame):
    fig = px.scatter(
        df, x="umap_1", y="umap_2",
        color="category",
        hover_name="name",
        hover_data={"count": True, "category": True, "umap_1": False, "umap_2": False},
        category_orders={"category": CATEGORY_ORDER},
        color_discrete_map=CATEGORY_COLORS,
        opacity=0.7,
    )
    # Recolor each category's points by their precomputed value gradient,
    # keeping one legend entry per category (base color).
    for trace in fig.data:
        sub = df[df["category"] == trace.name]
        trace.marker.color = sub["color"].tolist()
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(
        template="plotly_white",
        width=900, height=750,
        legend=dict(title="Token Category", itemsizing="constant"),
        xaxis=dict(showticklabels=False, title="UMAP 1"),
        yaxis=dict(showticklabels=False, title="UMAP 2"),
    )
    out = SAVE_DIR / "token_embeddings.html"
    fig.write_html(str(out))
    print(f"Saved interactive figure to {out}")


def main():
    print("Loading embeddings...")
    df = load_embeddings()
    print(f"{len(df)} tokens with count >= {MIN_COUNT}")

    print("Computing UMAP...")
    X = np.stack(df["embedding"].values)
    coords = run_umap(X)
    df["umap_1"] = coords[:, 0]
    df["umap_2"] = coords[:, 1]

    plot_static(df)
    plot_interactive(df[["name", "count", "category", "color", "umap_1", "umap_2"]])


if __name__ == "__main__":
    main()