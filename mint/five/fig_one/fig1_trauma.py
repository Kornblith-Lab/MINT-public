"""
Sensitivity analysis: MINT (softmax) vs XGBoost performance stratified by injury vs non-injury
chief complaint, restricted to the allowed CC list.

Reads pre-computed prediction CSVs from artifacts/fig1_operational (admit outcome only)
and the test feather to look up chief complaints. Outputs a 2x2 CSV:

    group, AUROC_MINT, AUPRC_MINT, AUROC_XGBoost, AUPRC_XGBoost
    injury, ...
    non-injury, ...
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

ARTIFACTS = Path("artifacts/fig1_operational")
DATA_DIR = Path("output")

ALLOWED_CC = {
    'FEVER','COUGH','ABDOMINAL PAIN','EMESIS','RASH','RESPIRATORY DISTRESS','ARM INJURY','FALL',
    'OTALGIA','HEADACHE','HEAD INJURY','SEIZURES','DIARRHEA','SORE THROAT','ALLERGIC REACTION',
    'SHORTNESS OF BREATH','FACIAL LACERATION','CHEST PAIN','NASAL CONGESTION','FUSSY','EYE PROBLEM',
    'EAR PAIN','CONSTIPATION','FINGER INJURY','ANKLE PAIN','HEAD LACERATION','CROUP','DYSURIA','URI',
    'LACERATION','LEG INJURY','EYE DRAINAGE','TRAUMA','INGESTION','LEG PAIN','FACIAL SWELLING',
    'WHEEZING','FEBRILE SEIZURE','HAND PAIN','ASTHMA','PSYCHIATRIC EVALUATION','WRIST PAIN',
    'HAND INJURY','SWALLOWED FOREIGN BODY','FOOT PAIN','EXTREMITY LACERATION','MOTOR VEHICLE CRASH',
    'KNEE PAIN','BACK PAIN','ANKLE INJURY','DENTAL PAIN','REFERRAL','SYNCOPE','EPISTAXIS',
    'FOOT INJURY','KNEE INJURY','SUICIDAL','ANIMAL BITE','FEEDING TUBE PROBLEM','ABSCESS',
    'GROIN SWELLING','INSECT BITE','FOREIGN BODY IN NOSE','ARM PAIN','DIZZINESS','MOUTH LESIONS',
    'LIP LACERATION','FACIAL INJURY','ALTERED MENTAL STATUS','URTICARIA','EYE INJURY','NECK PAIN',
    'WRIST INJURY','TOE INJURY','JAUNDICE','EYE PAIN','SHOULDER INJURY','BURN','NAUSEA',
    'FOREIGN BODY IN EAR','RECTAL BLEEDING','POST-OP PROBLEM','ASSAULT VICTIM','HIP PAIN',
    'CHOKING','TESTICLE PAIN','DEHYDRATION','MOUTH INJURY','LETHARGY','HYPERGLYCEMIA','TOE PAIN',
    'CELLULITIS','SUTURE / STAPLE REMOVAL','WOUND CHECK','MASS','FATIGUE','SICKLE CELL PAIN CRISIS',
    'ABNORMAL LAB',
}

INJURY_CC = {
    'ARM INJURY','FALL','HEAD INJURY','FACIAL LACERATION','FINGER INJURY','HEAD LACERATION',
    'LACERATION','LEG INJURY','TRAUMA','HAND INJURY','EXTREMITY LACERATION','MOTOR VEHICLE CRASH',
    'ANKLE INJURY','FOOT INJURY','KNEE INJURY','ANIMAL BITE','LIP LACERATION','FACIAL INJURY',
    'EYE INJURY','WRIST INJURY','TOE INJURY','SHOULDER INJURY','BURN','ASSAULT VICTIM','MOUTH INJURY',
}


def compute_metrics(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    auroc = roc_auc_score(labels, probs)
    auprc = average_precision_score(labels, probs)
    return auroc, auprc


def load_cc_map(test_feather: Path) -> dict[str, str]:
    """Return {encounter_key: chief_complaint} for all encounters with a CC_ token."""
    df = pd.read_feather(str(test_feather))
    cc_rows = df[df["name"].str.startswith("CC_")][["encounter_key", "name"]].copy()
    cc_rows["cc"] = cc_rows["name"].str.removeprefix("CC_")
    # Keep first CC per encounter (they're at t=0)
    cc_rows = cc_rows.drop_duplicates("encounter_key")
    return dict(zip(cc_rows["encounter_key"], cc_rows["cc"]))


def main():
    print("Loading chief complaints from test.feather...")
    cc_map = load_cc_map(DATA_DIR / "test.feather")

    print("Loading prediction CSVs...")
    mint_df = pd.read_csv(ARTIFACTS / "admit_MINT_softmax.csv")
    xgb_df = pd.read_csv(ARTIFACTS / "admit_XGBoost_baseline.csv")

    # Merge predictions on encounter_key
    merged = mint_df[["encounter_key", "probs", "labels"]].rename(columns={"probs": "mint_prob"})
    merged = merged.merge(
        xgb_df[["encounter_key", "probs"]].rename(columns={"probs": "xgb_prob"}),
        on="encounter_key",
        how="inner",
    )

    # Attach CC and filter to allowed set
    merged["cc"] = merged["encounter_key"].map(cc_map)
    merged = merged[merged["cc"].isin(ALLOWED_CC)].copy()
    print(f"Encounters after CC filter: {len(merged)}")

    # Label injury vs non-injury
    merged["is_injury"] = merged["cc"].isin(INJURY_CC)

    rows = []
    for group_name, group_flag in [("injury", True), ("non-injury", False)]:
        subset = merged[merged["is_injury"] == group_flag]
        if len(subset) == 0 or subset["labels"].nunique() < 2:
            print(f"Skipping {group_name}: insufficient data")
            continue
        auroc_mint, auprc_mint = compute_metrics(subset["mint_prob"].values, subset["labels"].values)
        auroc_xgb, auprc_xgb = compute_metrics(subset["xgb_prob"].values, subset["labels"].values)
        print(f"{group_name}: n={len(subset)}, MINT AUROC={auroc_mint:.4f}, XGB AUROC={auroc_xgb:.4f}")
        rows.append({
            "group": group_name,
            "N": len(subset),
            "AUROC_MINT": round(auroc_mint, 4),
            "AUPRC_MINT": round(auprc_mint, 4),
            "AUROC_XGBoost": round(auroc_xgb, 4),
            "AUPRC_XGBoost": round(auprc_xgb, 4),
        })

    out_df = pd.DataFrame(rows)
    out_path = ARTIFACTS / "fig1_trauma_sensitivity.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
