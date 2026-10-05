"""Sample-efficiency of XGBoost vs a supervised linear probe on MINT embeddings.

resp_rescue is the one main-text outcome where XGBoost beats MINT's frozen zero-shot
readout. This script shows why: swept over training-set size, XGBoost (bag-of-words) and
a logistic probe on MINT's *frozen* embeddings diverge sharply in sample efficiency, and
both eventually clear MINT's flat frozen readout. The probe reaches that line with a
fraction of the labels XGBoost needs.

Both estimators use the exact fig1.py hyperparameters and train on the *same* cases at
each size, so the comparison is apples-to-apples. Features are cached on first run.

  python -m mint.five.fig_one.sample_eff --save_path <dir>
  python -m mint.five.fig_one.sample_eff --save_path artifacts/sample_eff --plot_only
"""
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from mint.five.data import CaseDataset, case_collate_fn
from mint.five.fig_one.fig1 import (
    Definition, load_model, softmax_probe, temporal_cdf_probe_new,
)

torch.set_num_threads(4)
DATA_DIR = Path("output")
N_SIZES = 20
N_SEEDS = 10

# Community-hospital-years top axis: a training-case count N is converted to encounters
# (N / cases-per-encounter, since --first lets one encounter yield up to a positive + a
# negative case) then to years (encounters / COMMUNITY_VISITS_PER_YEAR).
# https://pmc.ncbi.nlm.nih.gov/articles/PMC3639791/
COMMUNITY_VISITS_PER_YEAR = 3_870

# fig1_nejm.py method colors
XGBOOST_COLOR = "#4D4D4D"   # grey
MINT_LR_COLOR = "#00664B"   # dark green
MINT_ZS_COLOR = "#0072B2"   # blue (frozen zero-shot readout)


def make_xgb(spw, seed=42):
    """XGBoost classifier with the fig1.py hyperparameters (n_jobs only affects speed)."""
    return xgb.XGBClassifier(
        random_state=seed, n_jobs=4, eval_metric="aucpr",
        objective="binary:logistic", scale_pos_weight=spw,
    )


def make_lr():
    """MINT linear probe with the fig1.py hyperparameters."""
    return LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42, solver="lbfgs",
    )


def build_features(save, ckpt, lookahead, max_len):
    """Build (and cache) BoW + MINT embeddings for the SAME cases, row-aligned."""
    if (save / "Xtr_bow.npz").exists():
        return
    train = pd.read_feather(DATA_DIR / "train.feather")
    test = pd.read_feather(DATA_DIR / "test.feather")
    d = Definition.get("resp_rescue", lookahead)
    tr, te = d.build_cases(
        train, test, use_only_first_event=True, use_ed_data_only=True,
        use_negative_exclusion=False, do_exclusion=False,
    )
    del train, test

    model, cfg = load_model(ckpt)
    age_pos, age_neg = d.label.get_age_to_token_ids()
    vocab_dim = len(d.label.vocab) + 2
    for name, cs in [("tr", tr), ("te", te)]:
        cases = cs["positive"] + cs["negative"]  # same order for BoW and embeddings
        cs["positive"].clear(); cs["negative"].clear()  # free the source lists
        if name == "tr":  # persist keys so the cases-per-encounter ratio is cacheable
            pd.Series([c.encounter_key for c in cases], name="encounter_key").to_csv(
                save / "tr_keys.csv", index=False)
        # sparse bag-of-words over the last max_len tokens (matches build_bag_of_words)
        rows, cols, data = [], [], []
        for i, c in enumerate(cases):
            ev = c.events[-max_len:] if max_len else c.events
            vals, cnts = np.unique(ev, return_counts=True)
            rows += [i] * len(vals); cols += vals.tolist(); data += cnts.tolist()
        X = sp.csr_matrix((data, (rows, cols)), shape=(len(cases), vocab_dim), dtype=np.float32)
        sp.save_npz(save / f"X{name}_bow.npz", X)
        del rows, cols, data, X  # drop before the memory-heavier forward pass
        # MINT frozen embeddings via a shuffle=False loader (row-aligned to BoW)
        loader = DataLoader(CaseDataset(cases, age_pos, age_neg, max_len=max_len),
                            batch_size=32, shuffle=False, collate_fn=case_collate_fn)
        _, y, _, emb = temporal_cdf_probe_new(
            model, cfg, loader, horizon_minutes=d.lookahead_min, extract_embeddings=True)
        np.save(save / f"emb_{name}.npy", emb)
        np.save(save / f"y{name}.npy", y)
        if name == "te":  # frozen zero-shot readout (softmax) for the reference line
            probs, labels, _ = softmax_probe(model, cfg, loader)
            np.save(save / "mint_readout.npy", np.array([roc_auc_score(labels, probs)]))
        print(f"{name}: emb {emb.shape} inc {y.mean():.4f}")
        cases.clear()


def sweep(save):
    Xtr = sp.load_npz(save / "Xtr_bow.npz"); Etr = np.load(save / "emb_tr.npy")
    ytr = np.load(save / "ytr.npy")
    Xte = sp.load_npz(save / "Xte_bow.npz"); Ete = np.load(save / "emb_te.npy")
    yte = np.load(save / "yte.npy")
    n = len(ytr)
    sizes = np.unique(np.geomspace(500, n, N_SIZES).astype(int))
    rows = []
    for k in sizes:
        seeds = range(1 if k >= n else N_SEEDS)
        xg, lr = [], []
        for s in seeds:
            idx = np.random.default_rng(s).choice(n, size=min(k, n), replace=False)
            yk = ytr[idx]
            spw = (yk == 0).sum() / max((yk == 1).sum(), 1)
            xc = make_xgb(spw).fit(Xtr[idx], yk)
            xg.append(roc_auc_score(yte, xc.predict_proba(Xte)[:, 1]))
            sc = StandardScaler().fit(Etr[idx])
            lc = make_lr().fit(sc.transform(Etr[idx]), yk)
            lr.append(roc_auc_score(yte, lc.predict_proba(sc.transform(Ete))[:, 1]))
        rows.append(dict(n=int(min(k, n)),
                         xgb=np.mean(xg), xgb_std=np.std(xg),
                         probe=np.mean(lr), probe_std=np.std(lr)))
        print(f"n={min(k,n):>6}  xgb={np.mean(xg):.4f}  probe={np.mean(lr):.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(save / "sample_eff.csv", index=False)
    return df


def cases_per_encounter(save, lookahead=5):
    """Cases per training encounter (>1 because --first yields <=1 pos + <=1 neg each).

    Regenerates tr_keys.csv from build_cases if a stale cache lacks it (cheap: no model,
    no embedding extraction)."""
    keys_path = save / "tr_keys.csv"
    if not keys_path.exists():
        d = Definition.get("resp_rescue", lookahead)
        tr = d.build_cases(pd.read_feather(DATA_DIR / "train.feather"),
                           use_only_first_event=True, use_ed_data_only=True,
                           use_negative_exclusion=False, do_exclusion=False)[0]
        cases = tr["positive"] + tr["negative"]
        pd.Series([c.encounter_key for c in cases], name="encounter_key").to_csv(
            keys_path, index=False)
    keys = pd.read_csv(keys_path)["encounter_key"]
    return len(keys) / keys.nunique()


def plot(save, df, lookahead=5):
    import matplotlib.pyplot as plt
    style = Path(__file__).resolve().parents[1] / "design-skill" / "nature.mplstyle"
    if style.exists():
        plt.style.use(str(style))
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7})
    mint = float(np.load(save / "mint_readout.npy")[0])

    # cases -> encounters -> years: years = cases / (cases_per_encounter * visits_per_year)
    cpe = cases_per_encounter(save, lookahead)
    per_year = cpe * COMMUNITY_VISITS_PER_YEAR
    print(f"cases/encounter={cpe:.4f} -> {per_year:.1f} cases per community-hospital-year")

    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.axhline(mint, color=MINT_ZS_COLOR, ls="--", lw=1,
               label=f"MINT zero-shot ({mint:.3f})")
    for col, std, c, lab in [("probe", "probe_std", MINT_LR_COLOR, "MINT (LR)"),
                             ("xgb", "xgb_std", XGBOOST_COLOR, "XGBoost")]:
        ax.plot(df.n, df[col], color=c, marker="o", ms=3, lw=1, label=lab)
        ax.fill_between(df.n, df[col] - df[std], df[col] + df[std], color=c, alpha=0.18, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel("Training cases (log scale)")
    ax.set_ylabel("AUROC")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    # top axis: community hospital-years (linear rescale of the same log axis), with
    # explicit whole-number ticks at 1, 3, 10 years instead of powers of ten
    from matplotlib.ticker import FixedLocator, FixedFormatter
    secax = ax.secondary_xaxis("top", functions=(lambda n: n / per_year,
                                                 lambda yr: yr * per_year))
    year_ticks = [1, 3, 10]
    minor_years = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                   2, 4, 5, 6, 7, 8, 9, 20, 30, 40]
    secax.xaxis.set_major_locator(FixedLocator(year_ticks))
    secax.xaxis.set_major_formatter(FixedFormatter([str(y) for y in year_ticks]))
    secax.xaxis.set_minor_locator(FixedLocator(minor_years))
    secax.set_xlabel("Estimated years of data at a median-volume ED")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(save / f"sample_eff.{ext}", dpi=600, bbox_inches="tight")
    print("saved", save / "sample_eff.png")


def main(save_path: str = "artifacts/sample_eff", checkpoint_path: str = "output/mint/ckpt.pt",
         lookahead: int = 5, max_len: int = 512, plot_only: bool = False):
    save = Path(save_path); save.mkdir(parents=True, exist_ok=True)
    if plot_only:  # reuse the cached sweep; just (re)draw the figure
        df = pd.read_csv(save / "sample_eff.csv")
    else:
        build_features(save, checkpoint_path, lookahead, max_len)
        df = sweep(save)
    plot(save, df, lookahead)


if __name__ == "__main__":
    from tap import tapify
    tapify(main)
