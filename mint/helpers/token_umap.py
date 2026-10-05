"""2D UMAP visualization of high-frequency token embeddings, colored by prefix."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import umap
from pathlib import Path

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
mpl.rcParams["font.size"] = 10


def get_prefix(label: str) -> str:
    """Extract prefix: double prefix for Vitals, single prefix otherwise."""
    if label.startswith("Vital_"):
        parts = label.split("_", 2)
        if len(parts) >= 3:
            return f"{parts[0]}_{parts[1]}"
    parts = label.split("_", 1)
    if len(parts) >= 2:
        return parts[0]
    return label


def main():
    viz_dir = Path("artifacts/viz")

    embeddings = np.loadtxt(viz_dir / "embeddings_high.tsv", delimiter="\t")
    labels = (viz_dir / "labels_high.tsv").read_text().strip().split("\n")
    counts = np.loadtxt(viz_dir / "counts_high.tsv")

    prefixes = [get_prefix(l) for l in labels]
    unique_prefixes = sorted(set(prefixes))

    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=40, min_dist=0.05, spread=0.3)
    coords = reducer.fit_transform(embeddings)

    # Separate vital and non-vital prefixes
    vital_prefixes = sorted([p for p in unique_prefixes if p.startswith("Vital")])
    non_vital_prefixes = sorted([p for p in unique_prefixes if not p.startswith("Vital")])

    # Bright non-vital colors
    non_vital_colors = [
        "#DE0D92",  # Electric Rose
        "#FFB30F",  # Amber Flame
        "#5DD9C1",  # Turquoise
        "#D64045",  # Scarlet Rush
        "#849324",  # Olive
        "#241023",  # Midnight Violet
        "#E9FFF9",  # Frozen Water
        "#FF6B6B",  # coral
        "#FFA94D",  # orange
        "#9B59B6",  # amethyst
        "#6BCB77",  # green
        "#F9F871",  # lime
    ]
    prefix_to_color = {}
    for i, p in enumerate(non_vital_prefixes):
        prefix_to_color[p] = non_vital_colors[i % len(non_vital_colors)]

    # Overrides
    for p in ["Admit", "Discharge", "ICU Start"]:
        if p in prefix_to_color:
            prefix_to_color[p] = "#000000"
    if "Arrival" in prefix_to_color:
        prefix_to_color["Arrival"] = "#2ECC40"
    if "Vital_Glasgow Coma Scale Score" in unique_prefixes:
        prefix_to_color["Vital_Glasgow Coma Scale Score"] = "#514663"

    # Vitals get a blue gradient (Frosted Blue → Rich Cerulean → Twilight Indigo)
    vital_cmap = mcolors.LinearSegmentedColormap.from_list(
        "vital_blues", ["#9ED8DB", "#467599", "#1D3354"]
    )
    for i, p in enumerate(vital_prefixes):
        if p == "Vital_Glasgow Coma Scale Score":
            continue
        prefix_to_color[p] = vital_cmap(i / max(len(vital_prefixes) - 1, 1))

    # Log-scale dot sizes so high-frequency tokens don't dominate
    log_counts = np.log10(counts)
    sizes = 20 + 80 * (log_counts - log_counts.min()) / (log_counts.max() - log_counts.min())

    # --- Static matplotlib figure ---
    fig, ax = plt.subplots(figsize=(14, 14), dpi=600)

    for prefix in unique_prefixes:
        mask = [i for i, p in enumerate(prefixes) if p == prefix]
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[prefix_to_color[prefix]],
            label=prefix,
            s=sizes[mask],
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
        )

    # Color legend (upper right)
    color_handles = [
        ax.scatter([], [], c=[prefix_to_color[p]], s=50, edgecolors="white", linewidths=0.5)
        for p in unique_prefixes
    ]
    color_legend = ax.legend(
        color_handles, unique_prefixes,
        loc="upper right", fontsize=8, markerscale=1.5, frameon=False,
    )
    ax.add_artist(color_legend)

    # Size legend (lower right) — match font to color legend
    def count_to_size(c):
        lc = np.log10(c)
        return 20 + 80 * (lc - log_counts.min()) / (log_counts.max() - log_counts.min())

    size_examples = [1000, 10000, 100000]
    size_handles = [
        ax.scatter([], [], s=count_to_size(c), c="black", edgecolors="white",
                   linewidths=0.5, alpha=0.85)
        for c in size_examples
    ]
    size_legend = ax.legend(
        size_handles, [f"{c:,}" for c in size_examples],
        loc="lower right", fontsize=8, frameon=False,
        title="Incidence", title_fontsize=9, labelspacing=1.2,
        handletextpad=1.2,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)

    out_path = viz_dir / "token_umap.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"Saved static UMAP to {out_path}")

    # --- Interactive Plotly figure ---
    import plotly.graph_objects as go

    # Convert colors to hex strings for plotly
    def to_hex(c):
        if isinstance(c, str):
            return c
        return mcolors.to_hex(c)

    point_colors = [to_hex(prefix_to_color[p]) for p in prefixes]

    fig_plotly = go.Figure()

    for prefix in unique_prefixes:
        mask = [i for i, p in enumerate(prefixes) if p == prefix]
        fig_plotly.add_trace(go.Scatter(
            x=coords[mask, 0],
            y=coords[mask, 1],
            mode="markers",
            name=prefix,
            marker=dict(
                size=np.sqrt(sizes[mask]) * 1.5,
                color=to_hex(prefix_to_color[prefix]),
                line=dict(width=0.5, color="white"),
                opacity=0.85,
            ),
            text=[labels[i] for i in mask],
            customdata=np.array([[labels[i], int(counts[i])] for i in mask]),
            hovertemplate="<b>%{customdata[0]}</b><br>Count: %{customdata[1]}<extra></extra>",
        ))

    fig_plotly.update_layout(
        width=1000, height=1000,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x"),
        plot_bgcolor="white",
        legend=dict(font=dict(size=10)),
        margin=dict(l=20, r=20, t=20, b=20),
    )

    plotly_path = viz_dir / "token_umap.html"
    fig_plotly.write_html(str(plotly_path))
    print(f"Saved interactive UMAP to {plotly_path}")


if __name__ == "__main__":
    main()
