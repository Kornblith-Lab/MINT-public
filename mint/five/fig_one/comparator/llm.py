"""
llm.py — Run LLM comparator on human review Excel trajectories.

Sends each trajectory to an LLM asking for P(positive pressure ventilation in next 60 min),
then computes AUROC/AUPRC and compares to mint and human columns.

Usage:
    python -m mint.five.fig_one.comparator.llm --excel artifacts/fig1/human_review/ppv.xlsx
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from mint.helpers.versa import client

PROMPT = ("REDACTED")


def query_llm(trajectory: str, model: str) -> tuple[float, int, int]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": trajectory},
        ],
    )
    text = response.choices[0].message.content.strip()
    usage = response.usage
    return float(text), usage.prompt_tokens, usage.completion_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", type=str, required=True, help="Path to the Excel file")
    parser.add_argument("--llm", type=str, default="gpt-5.5-2026-04-24") # gpt-5-mini-2025-08-07
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    df = pd.read_excel(args.excel, sheet_name="model")
    labels = df["label"].values
    trajectories = df["trajectory"].tolist()

    print(f"Running LLM ({args.llm}) on {len(trajectories)} trajectories...")
    llm_probs = [None] * len(trajectories)

    total_in, total_out = 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(query_llm, traj, args.llm): i for i, traj in enumerate(trajectories)}
        with tqdm(as_completed(futures), total=len(futures)) as pbar:
            for future in pbar:
                idx = futures[future]
                try:
                    prob, tok_in, tok_out = future.result()
                    llm_probs[idx] = prob
                    total_in += tok_in
                    total_out += tok_out
                except Exception as e:
                    print(f"  Error on row {idx}: {e}")
                    llm_probs[idx] = 0.5
                pbar.set_description(f"in={total_in:,} out={total_out:,}")

    llm_probs = np.array(llm_probs, dtype=float)

    # Compute metrics
    print("\n--- Results ---")
    print(f"{'Method':>8s}  {'AUROC':>7s}  {'AUPRC':>7s}  {'n':>4s}")
    print("-" * 36)
    for method, probs in [("mint", df["mint"].values), ("llm", llm_probs)]:
        auroc = roc_auc_score(labels, probs)
        auprc = average_precision_score(labels, probs)
        print(f"{method:>8s}  {auroc:.4f}   {auprc:.4f}  {len(labels):>4d}")

    if "human" in df.columns:
        mask = df["human"].notna()
        h_labels = labels[mask]
        h_probs = df["human"].values[mask]
        auroc = roc_auc_score(h_labels, h_probs)
        auprc = average_precision_score(h_labels, h_probs)
        print(f"{'human':>8s}  {auroc:.4f}   {auprc:.4f}  {int(mask.sum()):>4d}")

    # Save LLM predictions back
    df["gpt5_5"] = llm_probs
    out_path = args.excel.replace(".xlsx", "_llm_updated.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="model", index=False)
    print(f"\nSaved updated predictions to {out_path}")


if __name__ == "__main__":
    main()
