"""
fig1_excel.py — Generate Excel files for human review of MINT predictions.

Samples K cases from the test set for a given outcome task, with a configurable
positive-class ratio (default 25% positive, 75% negative). Produces an Excel
workbook with two sheets:

  1. "human" — trajectory column only (for blinded human review)
  2. "model" — trajectory + label (0/1) + mint predicted probability

Trajectories are formatted as: Token1 (t=X min) -> Token2 (t=Y min) -> ...

Before writing the workbook, a calibration study fits isotonic regression on the
train-set softmax probabilities and produces a Brier-scored reliability diagram
on the test set (saved as a PNG under --cache_dir). Train/test predictions are
cached to disk so re-runs skip inference.

Usage:
    python -m mint.five.fig_one.comparator.fig1_excel
"""

import hashlib

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from tap import tapify
from torch.utils.data import DataLoader
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from mint.five.data import Case, CaseDataset, Task, case_collate_fn
from mint.model.respiratory import RespiratoryEscalation
from mint.five.fig_one.fig1 import (
    DATA_DIR,
    VOCAB_CSV,
    VOCAB,
    DEVICE,
    Definition,
    load_model,
    softmax_probe,
)

STYLE_PATH = Path(__file__).resolve().parents[2] / "design-skill" / "nature.mplstyle"
# Okabe-Ito colorblind-safe palette (see design-skill/nature.mplstyle).
_OKABE = {"orange": "#E69F00", "green": "#009E73", "grey": "#999999"}

RESP_CC = ["CC_RESPIRATORY DISTRESS","CC_ALLERGIC REACTION","CC_SHORTNESS OF BREATH","CC_CROUP","CC_WHEEZING","CC_ASTHMA"]

# After max_len truncation, the final token must be within this many minutes of outcome
MAX_GAP_MIN = 60

def filter_cases(
    cases: dict[str, list[Case]],
    max_len: Optional[int],
    override_resp_only: bool,
    resp_ids: set[int],
) -> dict[str, list[Case]]:
    """Apply eligibility filters to a {'positive': [...], 'negative': [...]} case dict.

    Filters:
      - max-gap: after max_len truncation, the final token must be within
        MAX_GAP_MIN minutes of the outcome time.
      - respiratory override: unless override_resp_only is set, keep only cases
        that contain a respiratory chief-complaint token.
    """
    def is_eligible(case: Case) -> bool:
        times = case.times
        if max_len:
            times = times[-max_len:]
        return case.outcome_time - times[-1] <= MAX_GAP_MIN

    def has_resp_cc(case: Case) -> bool:
        return bool(resp_ids & set(case.events.tolist()))

    def keep(case: Case) -> bool:
        if not is_eligible(case):
            return False
        if not override_resp_only and not has_resp_cc(case):
            return False
        return True

    return {
        "positive": [c for c in cases["positive"] if keep(c)],
        "negative": [c for c in cases["negative"] if keep(c)],
    }

@dataclass
class Args:
    mode: str = "ppv"
    lookahead: int = 5
    k: int = 100
    pos_ratio: float = 0.28 # 7 out of 25; alternatively, 0.32 for 8 out of 25
    checkpoint_path: str = "output/mint/ckpt.pt"
    max_len: Optional[int] = 512
    seed: int = 42
    override_resp_only: bool = False
    cache_dir: str = "artifacts/fig1/calibration"
    shap_dir: str = "artifacts/fig1/shap"
    shap_max_evals: int = 2000  # PermutationExplainer evaluations per case (higher = more accurate)
    shap_top_k: int = 8  # show only the top-K contributors in the force plot
    o2_group_by_severity: bool = False  # group O2 devices by RespiratoryEscalation severity tier

def format_trajectory(events: np.ndarray, times: np.ndarray, vocab: pd.DataFrame) -> str:
    """Format a case's token sequence as 'Token (t=X min) -> Token (t=Y min) -> ...'"""
    id_to_name = {row["index"] + 1: row["name"] for _, row in vocab.iterrows()}
    parts = []
    for token_id, t in zip(events, times):
        name = id_to_name.get(int(token_id), f"UNK_{token_id}")
        parts.append(f"{name} (t={t:.0f} min)")
    return " -> ".join(parts)

def trajectory_id(trajectory: str) -> str:
    """Stable short identifier for a case, derived from its trajectory string."""
    return hashlib.sha1(trajectory.encode("utf-8")).hexdigest()[:12]


def sample_cases(
    positive_cases: list[Case],
    negative_cases: list[Case],
    k: int,
    pos_ratio: float,
    rng: np.random.Generator,
) -> list[Case]:
    """Sample k cases with the desired positive ratio."""
    n_pos = int(round(k * pos_ratio))
    n_neg = k - n_pos

    n_pos = min(n_pos, len(positive_cases))
    n_neg = min(n_neg, len(negative_cases))

    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"Not enough cases: {len(positive_cases)} positive, "
            f"{len(negative_cases)} negative (requested {n_pos} pos, {n_neg} neg)"
        )

    pos_idx = rng.choice(len(positive_cases), size=n_pos, replace=False)
    neg_idx = rng.choice(len(negative_cases), size=n_neg, replace=False)

    sampled = [positive_cases[i] for i in pos_idx] + [negative_cases[i] for i in neg_idx]
    return sampled


def run_inference(cases, model, model_config, age_to_pos_id, age_to_neg_id, max_len):
    """Run the softmax probe over a list of cases -> (probs, labels)."""
    ds = CaseDataset(cases, age_to_pos_id, age_to_neg_id, max_len=max_len)
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=case_collate_fn)
    probs, labels, _ = softmax_probe(model, model_config, loader)
    return probs, labels.astype(int)


def cached_probe(
    tag, cases, model, model_config, age_to_pos_id, age_to_neg_id, max_len, cache_dir
):
    """softmax probe with an on-disk npz cache so re-runs skip inference."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{tag}_probs.npz"
    if cache_path.exists():
        d = np.load(cache_path)
        print(f"  loaded cached {tag} predictions from {cache_path}")
        return d["probs"], d["labels"]
    print(f"  running inference on {tag} ({len(cases)} cases)...")
    probs, labels = run_inference(
        cases, model, model_config, age_to_pos_id, age_to_neg_id, max_len
    )
    np.savez(cache_path, probs=probs, labels=labels)
    return probs, labels


def calibration_study(
    train_cases, test_cases, model, model_config,
    age_to_pos_id, age_to_neg_id, max_len, cache_dir, mode,
):
    """Fit isotonic regression on train softmax probabilities, then produce a
    Brier-scored reliability diagram on the test set (saved as PNG). Returns the
    fitted isotonic calibrator so callers can calibrate other predictions.

    Predictions are cached to disk so re-runs reuse them without redoing inference.
    """
    print("Calibration study: obtaining train/test predictions...")
    train_probs, train_labels = cached_probe(
        f"{mode}_train", train_cases, model, model_config,
        age_to_pos_id, age_to_neg_id, max_len, cache_dir,
    )
    test_probs, test_labels = cached_probe(
        f"{mode}_test", test_cases, model, model_config,
        age_to_pos_id, age_to_neg_id, max_len, cache_dir,
    )

    # Fit the isotonic calibrator on the train set, apply to the test set.
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(train_probs, train_labels)
    test_iso = isotonic.predict(test_probs)

    methods = [
        ("Uncalibrated", test_probs, _OKABE["grey"]),
        ("Isotonic", test_iso, _OKABE["green"]),
    ]

    # Brier-scored reliability diagram on the test set.
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.plot([0, 1], [0, 1], ls=":", color="black", lw=0.8, label="Perfect")
    for name, p, color in methods:
        brier = brier_score_loss(test_labels, p)
        frac_pos, mean_pred = calibration_curve(test_labels, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", ms=3, lw=1, color=color,
                label=f"{name} (Brier={brier:.3f})")
        print(f"  {name:13s} Brier={brier:.4f}")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    save_path = Path(cache_dir) / f"{mode}_calibration.png"
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved calibration plot to {save_path}")

    return isotonic


def assign_subgroups(labels, rng, n_groups=4, names="ABCD"):
    """Assign each case to one of n_groups subgroups, stratified by label so
    positives (and negatives) are spread as evenly as possible across groups.

    Returns an array of subgroup names aligned with `labels`.
    """
    subgroup = np.empty(len(labels), dtype=object)
    # A single round-robin counter shared across strata keeps both the per-group
    # sizes and the positive counts as even as possible (exactly equal when the
    # totals divide by n_groups).
    offset = 0
    for value in np.unique(labels):
        idx = rng.permutation(np.where(labels == value)[0])
        for i in idx:
            subgroup[i] = names[offset % n_groups]
            offset += 1
    return subgroup


def _token_group_key(name: str, o2_by_severity: bool) -> str:
    """Map a token name to the SHAP feature it belongs to.

    Grouping rules (see design discussion):
      - O2 devices ("Vital_O2 Device_*") collapse to one feature, or to a
        severity tier when o2_by_severity is set.
      - Other vitals ("Vital_<Type>_<Value>") collapse by type (all SpO2
        readings -> one feature, all Resp -> one, etc.).
      - Everything else (Med_*, Procedure_*, CC_*, demographics, ...) keys on
        the full token name, so identical tokens merge but distinct ones stay
        separate.
    """
    if name.startswith("Vital_O2 Device_"):
        if o2_by_severity:
            sev = RespiratoryEscalation.SEVERITY.get(name)
            return f"O2 device (severity {sev})" if sev is not None else "O2 device"
        return "O2 device"
    if name.startswith("Vital_"):
        # "Vital_<Type>_<Value>" -> "Vital_<Type>"
        return name.rsplit("_", 1)[0]
    return name


def _vital_value(name: str):
    """Parse the numeric reading off a vital token, e.g. 'Vital_Pulse_110' -> 110.0."""
    try:
        return float(name.rsplit("_", 1)[1])
    except (ValueError, IndexError):
        return None


def _group_display_label(group_key: str, member_names: list[str]):
    """Human-facing label for a SHAP feature group, using the tokens present in
    the case. Returns None to HIDE the group from the plot (its contribution is
    folded into 'Other' so the force plot still sums to the prediction).

    Labels: vitals drop the 'Vital_' prefix and show their min-max range
    ('Pulse (100-120)'); O2 devices become 'Respiratory support (dev, ...)';
    'Age_N' -> 'Age (N years old)'; 'Acuity_X' -> 'ESI: X'; medications and
    procedures drop their prefixes; chief complaints are hidden.
    """
    if group_key.startswith("CC_"):
        return None  # kept in trajectory + SHAP math, just not shown as a bar
    if group_key.startswith("O2 device"):
        devices = list(dict.fromkeys(
            n[len("Vital_O2 Device_"):] for n in member_names
            if n.startswith("Vital_O2 Device_")
        ))
        return f"Respiratory support (oxygen)" if devices else group_key
    if group_key.startswith("Vital_"):
        vtype = group_key[len("Vital_"):]
        vals = [v for v in (_vital_value(n) for n in member_names) if v is not None]
        if not vals:
            return vtype
        lo, hi = min(vals), max(vals)
        rng = f"{lo:g}" if lo == hi else f"{lo:g}-{hi:g}"
        if vtype == "O2 Flow Rate (l/min)":
            return f"O2 Flow rate ({rng} lpm)"
        return f"{vtype} ({rng})"
    if group_key.startswith("Age_"):
        return f"Age ({group_key[len('Age_'):]} years old)"
    if group_key.startswith("Acuity_"):
        return f"ESI: {group_key[len('Acuity_'):]}"
    if group_key.startswith("Med_"):
        return group_key[len("Med_"):]
    if group_key.startswith("Procedure_"):
        return group_key[len("Procedure_"):]
    return group_key


def _group_members(case, max_len, id_to_name, o2_by_severity):
    """Map each group key present in the case to the token names it contains,
    so render-time labels can show vital ranges / device lists without any
    model inference (the npz cache only stores group KEYS, not token values)."""
    events, _, group_keys, _ = _case_group_features(
        case, max_len, id_to_name, o2_by_severity
    )
    members: dict[str, list[str]] = {}
    for tid, gk in zip(events, group_keys):
        members.setdefault(gk, []).append(id_to_name.get(int(tid), f"UNK_{tid}"))
    return members


def _case_group_features(case, max_len, id_to_name, o2_by_severity):
    """For one case, return (events, times, group_keys, unique_groups).

    group_keys[i] is the feature key for token position i; unique_groups is the
    ordered list of distinct feature keys present (the SHAP feature set).
    """
    events = case.events
    times = case.times
    if max_len:
        events = events[-max_len:]
        times = times[-max_len:]
    group_keys = [
        _token_group_key(id_to_name.get(int(tid), f"UNK_{tid}"), o2_by_severity)
        for tid in events
    ]
    unique_groups = list(dict.fromkeys(group_keys))  # order-preserving
    return events, times, group_keys, unique_groups


def _predict_ppv_masked(mask_rows, events, times, group_keys, unique_groups,
                        pos_ids, model, device):
    """SHAP prediction function: P(PPV) for each row of a binary group-mask matrix.

    mask_rows[r, g] == 1 keeps group unique_groups[g]; 0 replaces every token in
    that group with padding (id 0), which the model's attention mask ignores.
    Returns the final-position summed-positive softmax probability per row.
    """
    events = torch.as_tensor(np.asarray(events).copy(), dtype=torch.long)
    times = torch.as_tensor(np.asarray(times).copy(), dtype=torch.float)
    group_idx = np.array([unique_groups.index(g) for g in group_keys])
    pos_idx = torch.as_tensor(pos_ids, dtype=torch.long, device=device)

    out = np.empty(len(mask_rows), dtype=np.float32)
    with torch.no_grad():
        for r, mask in enumerate(mask_rows):
            keep = mask[group_idx] > 0.5  # per-token keep flag
            ev = events.clone()
            ev[~torch.as_tensor(keep)] = 0  # mask -> padding
            ev = ev.unsqueeze(0).to(device)
            tm = times.unsqueeze(0).to(device)
            logits, _, _ = model(ev, tm)
            last = logits[0, -1, :]
            p = torch.softmax(last, dim=0).index_select(0, pos_idx).sum()
            out[r] = p.item()
    return out


def compute_case_shap(case, model, age_to_pos_id, args, id_to_name):
    """Compute group-level SHAP values for one case's P(PPV), with on-disk cache.

    Returns (shap_values, base_value, unique_groups, prediction).
    """
    import shap

    events, times, group_keys, unique_groups = _case_group_features(
        case, args.max_len, id_to_name, args.o2_group_by_severity
    )
    pos_ids = age_to_pos_id[case.age]
    n = len(unique_groups)

    f = lambda m: _predict_ppv_masked(
        m, events, times, group_keys, unique_groups, pos_ids, model, DEVICE
    )

    # Background = all groups masked (no-evidence prior); explain the all-present row.
    masker = np.zeros((1, n))
    present = np.ones((1, n))
    explainer = shap.PermutationExplainer(f, masker)
    expl = explainer(present, max_evals=max(2 * n + 1, args.shap_max_evals), silent=True)

    shap_values = np.asarray(expl.values[0], dtype=np.float32)
    base_value = float(np.asarray(expl.base_values).ravel()[0])
    prediction = float(f(present)[0])
    return shap_values, base_value, unique_groups, prediction


def _top_k_contributors(shap_values, labels, k):
    """Keep the top-k *displayable* features by |SHAP value|; collapse the rest
    (and any hidden features, where labels[i] is None) into a single 'Other' bar.

    Collapsing (rather than dropping) preserves the additive property so the
    force plot's contributions still sum to the prediction.
    """
    shap_values = np.asarray(shap_values, dtype=np.float32)
    visible = [i for i, lab in enumerate(labels) if lab is not None]
    hidden = [i for i, lab in enumerate(labels) if lab is None]

    order = sorted(visible, key=lambda i: abs(shap_values[i]), reverse=True)
    keep = order if (k <= 0 or len(order) <= k) else order[:k]
    rest = ([] if (k <= 0 or len(order) <= k) else order[k:]) + hidden

    values = [float(shap_values[i]) for i in keep]
    names = [labels[i] for i in keep]
    if rest:
        values.append(float(shap_values[rest].sum()))
        names.append(f"Other ({len(rest)} features)")
    return np.asarray(values, dtype=np.float32), names


def save_force_plots(case_id, shap_values, base_value, groups, prediction,
                     shap_dir, top_k, members):
    """Render shap.plots.force for one case and save HTML + PNG. Returns (html, png) paths.

    `members` maps each group key to the token names present in the case, used
    to build human-facing labels (vital ranges, device lists) at render time.
    """
    import shap

    shap_dir = Path(shap_dir)
    shap_dir.mkdir(parents=True, exist_ok=True)
    html_path = shap_dir / f"{case_id}.html"
    png_path = shap_dir / f"{case_id}.png"

    labels = [_group_display_label(g, members.get(g, [])) for g in groups]
    values, names = _top_k_contributors(shap_values, labels, top_k)

    # feature_names only (no dummy feature values) -> clean presence labels.
    fp = shap.plots.force(base_value, values, feature_names=names, show=False)
    shap.save_html(str(html_path), fp)

    # Static PNG (matplotlib mode).
    fig = shap.plots.force(
        base_value, values, feature_names=names, matplotlib=True, show=False,
    )
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return html_path, png_path


def build_shap_explanations(sampled, case_ids, model, age_to_pos_id, args):
    """Compute + render SHAP force plots for every sampled case, with caching.

    SHAP values are cached to npz per case_id so re-runs skip recomputation and
    only re-render the plots. Returns dicts case_id -> html/png relative path.
    """
    print("Building SHAP force plots...")
    id_to_name = {row["index"] + 1: row["name"] for _, row in VOCAB.iterrows()}
    shap_dir = Path(args.shap_dir)
    shap_dir.mkdir(parents=True, exist_ok=True)

    html_paths, png_paths = {}, {}
    for case, case_id in tqdm(zip(sampled, case_ids), total=len(sampled), desc="SHAP force plots"):
        cache_path = shap_dir / f"{case_id}_shap.npz"
        if cache_path.exists():
            d = np.load(cache_path, allow_pickle=True)
            shap_values = d["shap_values"]
            base_value = float(d["base_value"])
            groups = list(d["groups"])
            prediction = float(d["prediction"])
        else:
            shap_values, base_value, groups, prediction = compute_case_shap(
                case, model, age_to_pos_id, args, id_to_name
            )
            np.savez(
                cache_path, shap_values=shap_values, base_value=base_value,
                groups=np.array(groups, dtype=object), prediction=prediction,
            )
        members = _group_members(case, args.max_len, id_to_name, args.o2_group_by_severity)
        html_path, png_path = save_force_plots(
            case_id, shap_values, base_value, groups, prediction,
            shap_dir, args.shap_top_k, members,
        )
        html_paths[case_id] = str(html_path)
        png_paths[case_id] = str(png_path)
    print(f"  saved {len(sampled)} force plots to {shap_dir}")
    return html_paths, png_paths


def build_excel(sampled, model, model_config, age_to_pos_id, age_to_neg_id, args, rng, calibrator):
    """Run inference on the sampled test cases and write the human/model Excel workbook."""
    print("Running model inference...")
    probs, labels = run_inference(
        sampled, model, model_config, age_to_pos_id, age_to_neg_id, args.max_len
    )

    print("Formatting trajectories...")
    trajectories = []
    for case in sampled:
        events = case.events
        times = case.times
        if args.max_len:
            events = events[-args.max_len:]
            times = times[-args.max_len:]
        trajectories.append(format_trajectory(events, times, VOCAB))

    df = pd.DataFrame({
        "case_id": [trajectory_id(t) for t in trajectories],
        "trajectory": trajectories,
        "label": labels,
        "mint": probs,
        "calibrated_mint": calibrator.predict(probs),
    })

    n_unique = df["case_id"].nunique()
    assert n_unique == len(df), f"case_id collision: {n_unique} unique ids for {len(df)} cases"

    # SHAP force plots (one file per case); paths keyed by case_id, so shuffle-safe.
    html_paths, png_paths = build_shap_explanations(
        sampled, df["case_id"].tolist(), model, age_to_pos_id, args
    )
    df["shap_html_path"] = df["case_id"].map(html_paths)
    df["shap_png_path"] = df["case_id"].map(png_paths)

    shuffle_idx = rng.permutation(len(df))
    df = df.iloc[shuffle_idx].reset_index(drop=True)

    # Randomly split into 4 subgroups A/B/C/D, stratified by outcome so positives
    # are spread as evenly as possible across groups while keeping equal group sizes.
    df["subgroup"] = assign_subgroups(df["label"].to_numpy(), rng)
    df = df.sort_values("subgroup", kind="stable").reset_index(drop=True)

    for g, grp in df.groupby("subgroup"):
        print(f"  subgroup {g}: {len(grp)} cases, {int(grp['label'].sum())} positive")

    save_dir = Path("artifacts/fig1/human_review")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{args.mode}.xlsx"

    human_cols = ["case_id", "subgroup", "trajectory"]
    model_cols = ["case_id", "subgroup", "trajectory", "label", "mint",
                  "calibrated_mint", "shap_html_path", "shap_png_path"]
    with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
        df[human_cols].to_excel(writer, sheet_name="human", index=False)
        df[model_cols].to_excel(writer, sheet_name="model", index=False)

    print(f"Saved {len(df)} cases to {save_path}")
    print(f"  Positive: {df['label'].sum()} ({df['label'].mean():.0%})")
    print(f"  Negative: {len(df) - df['label'].sum()}")


def main():
    args = tapify(Args)
    rng = np.random.default_rng(args.seed)

    task: Task = Definition.get(args.mode, args.lookahead)

    train_path = str(DATA_DIR / "train.feather")
    train = pd.read_feather(train_path)

    train = pd.read_feather(DATA_DIR / "train.feather")
    test = pd.read_feather(DATA_DIR / "test.feather")

    print(f"Building cases for {args.mode}@{args.lookahead}min...")
    train_cases, test_cases = task.build_cases(
        train, test,
        use_only_first_event=True,
        use_ed_data_only=True,
        use_negative_exclusion=True, # enabled to simplfy task definition for humans as pure "initation"
        limit_pos_patients=False,
        do_exclusion=False,
        high_acuity_only=False,
        return_outcome_time=True,
    )

    name_to_id = {row["name"]: row["index"] + 1 for _, row in VOCAB.iterrows()}
    resp_ids = set(name_to_id[n] for n in RESP_CC if n in name_to_id)

    for split_name, cases in [("train", train_cases), ("test", test_cases)]:
        before = (len(cases["positive"]), len(cases["negative"]))
        filtered = filter_cases(cases, args.max_len, args.override_resp_only, resp_ids)
        cases["positive"], cases["negative"] = filtered["positive"], filtered["negative"]
        print(
            f"{split_name}: {before[0]} pos, {before[1]} neg -> "
            f"{len(cases['positive'])} pos, {len(cases['negative'])} neg (after filters)"
        )

    sampled = sample_cases(
        test_cases["positive"], test_cases["negative"], args.k, args.pos_ratio, rng
    )

    print(f"Loading model from {args.checkpoint_path}...")
    model, model_config = load_model(args.checkpoint_path)

    age_to_pos_id, age_to_neg_id = task.label.get_age_to_token_ids()

    calibrator = calibration_study(
        train_cases["positive"] + train_cases["negative"],
        test_cases["positive"] + test_cases["negative"],
        model, model_config, age_to_pos_id, age_to_neg_id,
        args.max_len, args.cache_dir, args.mode,
    )

    build_excel(sampled, model, model_config, age_to_pos_id, age_to_neg_id, args, rng, calibrator)


if __name__ == "__main__":
    main()
