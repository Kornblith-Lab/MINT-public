"""Export token embeddings and labels for TensorFlow Projector."""

import argparse
import csv
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default="output/mint/ckpt_100000.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--vocab", default="output/vocab.csv",
        help="Path to vocab.csv",
    )
    parser.add_argument(
        "--out-dir", default="artifacts/viz",
        help="Output directory for .tsv files",
    )
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    wte = state["transformer.wte.weight"]

    with open(args.vocab) as f:
        reader = csv.DictReader(f)
        vocab_rows = list(reader)
    labels = [row["name"] for row in vocab_rows]
    counts = [int(row["count"]) for row in vocab_rows]

    num_embeddings = wte.shape[0]
    num_labels = len(labels)

    print(f"Embedding rows (model.transformer.wte.weight): {num_embeddings}")
    print(f"Labels (vocab.csv):                            {num_labels}")

    # Index 0 = PAD, Index 1 = NO_EVENT, Indices 2+ = vocab tokens (shifted +1)
    labels = ["<PAD>", "<NO_EVENT>"] + labels
    num_labels = len(labels)
    print(f"After prepending special tokens:               {num_labels}")

    assert num_embeddings == num_labels, (
        f"Still mismatched: {num_embeddings} embeddings vs {num_labels} labels"
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "embeddings.tsv", "w") as f:
        for row in wte:
            f.write("\t".join(f"{v:.6f}" for v in row.tolist()) + "\n")

    with open(out / "labels.tsv", "w") as f:
        for label in labels:
            f.write(label + "\n")

    print(f"\nSaved {num_embeddings} embeddings to {out / 'embeddings.tsv'}")
    print(f"Saved {num_labels} labels to {out / 'labels.tsv'}")

    # High-frequency subset (tokens with count >= 300)
    # Special tokens (PAD, NO_EVENT) have no count — exclude them
    high_indices = [i + 2 for i, c in enumerate(counts) if c >= 300]
    high_labels = [labels[i] for i in high_indices]
    high_wte = wte[high_indices]

    with open(out / "embeddings_high.tsv", "w") as f:
        for row in high_wte:
            f.write("\t".join(f"{v:.6f}" for v in row.tolist()) + "\n")

    with open(out / "labels_high.tsv", "w") as f:
        for label in high_labels:
            f.write(label + "\n")

    high_counts = [counts[i] for i in range(len(counts)) if counts[i] >= 300]
    with open(out / "counts_high.tsv", "w") as f:
        for c in high_counts:
            f.write(f"{c}\n")

    print(f"Saved {len(high_indices)} high-frequency embeddings to {out / 'embeddings_high.tsv'}")
    print(f"Saved {len(high_indices)} high-frequency labels to {out / 'labels_high.tsv'}")


if __name__ == "__main__":
    main()
