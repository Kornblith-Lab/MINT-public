# python -m mint.five.fig_tokenomics.token_figures
# Run from project root (needs output/mint/ckpt.pt and output/vocab.csv)
#
# Self-contained script for a single one-page supplement figure
# (token_semantics.pdf/.png) with two stacked blocks:
#   a) nearest-neighbor retrieval for representative clinical tokens
#      (top-4, salt-form deduped)
#   b) clinical vector arithmetic -- category-restricted analogies plus two
#      cross-category treatment-blend -> chief-complaint panels
#
# Bars are colored by clinical token category (matching token.py's
# CATEGORY_COLORS) and styled with the mint design skill. The neighbor list is
# also written to
#   artifacts/fig_tokenomics/token_semantics_metrics/analysis1_neighbors.csv

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

STYLE_PATH = Path(__file__).parent.parent / "design-skill" / "nature.mplstyle"
SAVE_DIR = Path("artifacts/fig_tokenomics")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR = SAVE_DIR / "token_semantics_metrics"
METRIC_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = "output/mint/ckpt.pt"
VOCAB_PATH = "output/vocab.csv"

# Okabe-Ito colorblind-safe palette (from the design skill).
OKABE_ITO = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "pink": "#CC79A7", "vermillion": "#D55E00", "skyblue": "#56B4E9",
    "yellow": "#F0E442", "black": "#000000",
}


# ============================================================
# Embedding loader + similarity helpers
# ============================================================

def load_embeddings() -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    vocab = pd.read_csv(VOCAB_PATH)
    state = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    wte = state["model"]["transformer.wte.weight"].numpy()

    token_to_emb = {}
    for _, row in vocab.iterrows():
        idx = int(row["index"]) + 1
        if idx < wte.shape[0]:
            token_to_emb[row["name"]] = wte[idx]
    return token_to_emb, vocab


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine(a, b)


def find_nearest(query: np.ndarray, token_to_emb: dict, exclude: set | None = None,
                 top_k: int = 5) -> list[tuple[str, float]]:
    exclude = exclude or set()
    results = [(name, cosine_sim(query, emb))
               for name, emb in token_to_emb.items() if name not in exclude]
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def find_nearest_filtered(query: np.ndarray, token_to_emb: dict, prefix: str,
                          exclude: set | None = None, top_k: int = 3) -> list[tuple[str, float]]:
    exclude = exclude or set()
    results = [(name, cosine_sim(query, emb)) for name, emb in token_to_emb.items()
               if name not in exclude and name.startswith(prefix)]
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


# ============================================================
# Salt-form / formulation dedup (for neighbor lists)
# ============================================================

_SALT_SUFFIXES = [
    "(pf)", "hcl", "sulfate", "bitartrate", "maleate", "edisylate",
    "sodium", "bromide", "concentrate", "hfa", "besylate", "tartrate",
]


def med_base(name: str) -> str:
    """Collapse a Med_ token to its base drug, dropping salt/formulation tags."""
    if not name.startswith("Med_"):
        return name
    body = name[len("Med_"):].split(";")[0].strip().lower()
    body = re.sub(r"\([^)]*\)", "", body)
    tokens = [t for t in body.replace("-", " ").split() if t]
    kept = [t for t in tokens if t not in _SALT_SUFFIXES]
    kept = kept or tokens
    return "Med_" + " ".join(kept)


def find_nearest_dedup(query: np.ndarray, token_to_emb: dict, exclude: set | None = None,
                       top_k: int = 4, exclude_bases: set | None = None,
                       prefix: str | None = None) -> list[tuple[str, float]]:
    """Nearest neighbors keeping at most one token per medication base.

    If `prefix` is given, only tokens with that prefix are considered.
    """
    exclude = set(exclude or set())
    seen_bases = set(exclude_bases or set())
    ranked = find_nearest(query, token_to_emb, exclude=exclude, top_k=len(token_to_emb))
    out: list[tuple[str, float]] = []
    for name, sim in ranked:
        if prefix is not None and not name.startswith(prefix):
            continue
        base = med_base(name)
        if base in seen_bases:
            continue
        seen_bases.add(base)
        out.append((name, sim))
        if len(out) >= top_k:
            break
    return out


# ============================================================
# Category-restricted analogies (bottom-row vector arithmetic)
# ============================================================

# Prefix that a candidate must share with the analogy's anchor token `a`.
_CATEGORY_PREFIXES = ["Med_", "CC_", "Procedure_", "Lab_", "Acuity_", "Age_", "Arrival_", "Vital_"]


def category_prefix(token: str) -> str | None:
    for p in _CATEGORY_PREFIXES:
        if token.startswith(p):
            return p
    return None


def run_analogy_filtered(a: str, b: str, c: str, token_to_emb: dict,
                         top_k: int = 3) -> list[tuple[str, float]]:
    """a - b + c = ?, restricting matches to a's own token category."""
    query = token_to_emb[a] - token_to_emb[b] + token_to_emb[c]
    prefix = category_prefix(a)
    return find_nearest_filtered(query, token_to_emb, prefix,
                                 exclude={a, b, c}, top_k=top_k)


def intervention_pairs(token_to_emb: dict) -> list[dict]:
    experiments = [
        {"label": "ketamine − reduction\n+ intubation = ?",
         "a": "Med_ketamine", "b": "Procedure_REDUCTION", "c": "Procedure_INTUBATION",
         "expected": "Med_propofol"},
        {"label": "albuterol - WHEEZING\n+ ALLERGIC REACTION = ?",
         "a": "Med_albuterol sulfate", "b": "CC_WHEEZING", "c": "CC_ALLERGIC REACTION",
         "expected": "Med_epinephrine"},
        {"label": "ondansetron - EMESIS\n+ ABD PAIN = ?",
         "a": "Med_ondansetron hcl (pf)", "b": "CC_EMESIS", "c": "CC_ABDOMINAL PAIN",
         "expected": "Med_ketorolac"},
        {"label": "acetaminophen − fever\n+ seizures = ?",
         "a": "Med_acetaminophen", "b": "CC_FEVER", "c": "CC_SEIZURES",
         "expected": "Med_midazolam (pf)"},
    ]
    return _run_experiments(experiments, token_to_emb)


def age_transformations(token_to_emb: dict) -> list[dict]:
    experiments = [
        {"label": "febrile seizure\n− age 1 + age 12 = ?",
         "a": "CC_FEBRILE SEIZURE", "b": "Age_1", "c": "Age_12", "expected": "CC_SEIZURES"},
        {"label": "CROUP - Age_1\n+ Age_15 = ?",
         "a": "CC_CROUP", "b": "Age_1", "c": "Age_15", "expected": "CC_SHORTNESS OF BREATH"},
        {"label": "FUSSY - Age_0\n+ Age_15 = ?",
         "a": "CC_FUSSY", "b": "Age_0", "c": "Age_15", "expected": "CC_ANXIETY"},
        {"label": "amoxicillin - Age_2\n+ Age_16 = ?",
         "a": "Med_amoxicillin", "b": "Age_2", "c": "Age_16", "expected": "Med_azithromycin"},
    ]
    return _run_experiments(experiments, token_to_emb)


def diagnostic_workup(token_to_emb: dict) -> list[dict]:
    experiments = [
        {"label": "XR CHEST - RESP\nDISTRESS + ABD PAIN = ?",
         "a": "Procedure_XR CHEST", "b": "CC_RESPIRATORY DISTRESS", "c": "CC_ABDOMINAL PAIN",
         "expected": "Procedure_XR ABDOMEN"},
        {"label": "XR ANKLE - ANKLE\nINJURY + WRIST INJURY = ?",
         "a": "Procedure_XR ANKLE", "b": "CC_ANKLE INJURY", "c": "CC_WRIST INJURY",
         "expected": "Procedure_XR WRIST"},
        {"label": "CT BRAIN - SEIZURES\n+ ABD PAIN = ?",
         "a": "Procedure_CT BRAIN", "b": "CC_SEIZURES", "c": "CC_ABDOMINAL PAIN",
         "expected": "Procedure_US ABDOMEN"},
        {"label": "US APPENDIX - ABD\nPAIN + TESTICLE PAIN = ?",
         "a": "Procedure_US APPENDIX", "b": "CC_ABDOMINAL PAIN", "c": "CC_TESTICLE PAIN",
         "expected": "Procedure_US SCROTUM"},
    ]
    return _run_experiments(experiments, token_to_emb)


def _run_experiments(experiments: list[dict], token_to_emb: dict) -> list[dict]:
    results = []
    for exp in experiments:
        if all(k in token_to_emb for k in [exp["a"], exp["b"], exp["c"]]):
            nearest = run_analogy_filtered(exp["a"], exp["b"], exp["c"], token_to_emb, top_k=4)
            results.append({**exp, "results": nearest})
    return results


# ============================================================
# Label formatting
# ============================================================

# Medical acronyms kept uppercase when sentence-casing chief complaints / procedures.
_ACRONYMS = {"XR", "CT", "US", "MRI", "ECG", "EKG", "ECHO", "IV", "GCS",
             "CPAP", "BIPAP", "HFNC", "CSPINE", "ENT", "GI", "UTI", "IUD"}
_ACRONYM_DISPLAY = {"BIPAP": "BiPAP", "HFNC": "HFNC", "CSPINE": "C-spine"}


def _sentence_case(text: str) -> str:
    """Sentence-case free text, preserving known medical acronyms."""
    words = []
    for i, w in enumerate(text.split()):
        up = w.upper()
        if up in _ACRONYMS:
            words.append(_ACRONYM_DISPLAY.get(up, up))
        elif i == 0:
            words.append(w[:1].upper() + w[1:].lower())
        else:
            words.append(w.lower())
    return " ".join(words)


def pretty_label(token: str) -> str:
    """Human-readable label for a vocab token (used in neighbor lists)."""
    if token.startswith("Med_"):
        return token[len("Med_"):]
    if token.startswith("CC_"):
        return _sentence_case(token[len("CC_"):])
    if token.startswith("Procedure_"):
        return _sentence_case(token[len("Procedure_"):])
    if token.startswith("Lab_"):
        body = token[len("Lab_"):]
        if body.endswith("_Abnormal"):
            body = body[:-len("_Abnormal")]
        analyte = body.split(",")[0].strip()
        return f"Abnormal {analyte}"
    if token.startswith("Vital_SpO2_"):
        return f"Oxygen saturation, {token[len('Vital_SpO2_'):]}%"
    if token.startswith("Vital_O2 Device_"):
        return token[len("Vital_O2 Device_"):]
    if token.startswith("Vital_"):
        return token[len("Vital_"):]
    if token.startswith("Acuity_"):
        return token[len("Acuity_"):].lstrip("*")
    if token.startswith("Age_"):
        yrs = token[len("Age_"):]
        unit = "year" if yrs == "1" else "years"
        return f"{yrs} {unit} old"
    return token


# Category descriptor appended to each probe's panel title.
def probe_title(token: str) -> str:
    label = pretty_label(token)
    if token.startswith("Med_"):
        return f"{label} (medication)"
    if token.startswith("CC_"):
        return f"{label} (chief complaint)"
    if token.startswith("Procedure_"):
        return f"{label} (procedure)"
    if token.startswith("Lab_"):
        return f"{label} (lab)"
    if token.startswith("Vital_O2 Device_"):
        return f"{label} (respiratory support)"
    if token.startswith("Vital_SpO2_"):
        return label  # already self-describing, e.g. "Oxygen saturation, 90%"
    if token.startswith("Acuity_"):
        return f"{label} (triage acuity)"
    if token.startswith("Age_"):
        return f"{label} (age)"
    return label


# ============================================================
# Category coloring (shared across both figure panels)
# ============================================================

# Per-token-category palette, matching token.py's CATEGORY_COLORS so the whole
# supplement colors each clinical group consistently (Okabe-Ito based).
CATEGORY_COLORS = {
    "Vital": "#0072B2", "Medication": "#E69F00", "Chief Complaint": "#009E73",
    "Procedure": "#CC79A7", "Lab": "#D55E00", "Age": "#56B4E9",
    "Arrival": "#F0E442", "Acuity": "#d62728", "Disposition": "#000000",
}

# Vocab prefix -> CATEGORY_COLORS key.
_PREFIX_TO_CATEGORY = {
    "Med_": "Medication", "CC_": "Chief Complaint", "Procedure_": "Procedure",
    "Lab_": "Lab", "Vital_": "Vital", "Acuity_": "Acuity", "Age_": "Age",
    "Arrival_": "Arrival", "Disposition_": "Disposition",
}


def token_color(token: str) -> str:
    """Color for a token, keyed on its clinical category."""
    for prefix, cat in _PREFIX_TO_CATEGORY.items():
        if token.startswith(prefix):
            return CATEGORY_COLORS[cat]
    return CATEGORY_COLORS["Disposition"]


def _short_math(token: str, max_len: int = 22) -> str:
    """Compact, human-readable label for the analogy bars."""
    label = pretty_label(token)
    if len(label) > max_len:
        label = label[:max_len - 1] + "…"
    return label


def _draw_bar_panel(ax, names, sims, color, title):
    """Unified horizontal-bar panel shared by every panel in the figure.

    Same bar height, colors, inline cosine labels, fonts and spines throughout,
    so the neighbor, analogy and combo panels all read identically.
    """
    y = range(len(names))
    ax.barh(y, sims, color=color, edgecolor="white", linewidth=0.4,
            height=0.72, zorder=3)
    ax.set_yticks(list(y))

    mapper = {
        "XR chest and abdomen": "XR chest/abd",
        "piperacillin-tazobactam": "Zosyn"
    }

    names = [mapper.get(n,n) for n in names]

    ax.set_yticklabels(names, fontsize=5.5)
    ax.invert_yaxis()

    # Inline cosine value at the tip of each bar.
    for i, s in enumerate(sims):
        ax.text(s - 0.004, i, f"{s:.2f}", va="center", ha="right",
                fontsize=4.6, color="white", fontweight="bold", zorder=4)

    ax.set_xlim(min(sims) - 0.045, min(max(sims) + 0.02, 1.0))
    ax.set_xlabel("Cosine similarity", fontsize=5.5)
    ax.tick_params(axis="x", labelsize=5)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.set_title(title, fontsize=6, pad=4, loc="left", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.4)
    ax.spines["bottom"].set_linewidth(0.4)


def _draw_analogy_panel(ax, exp: dict):
    """One analogy panel: top-k candidate bars colored by the query category."""
    results = exp["results"]
    names = [r[0] for r in results]
    sims = [r[1] for r in results]
    short_names = [_short_math(n) for n in names]
    color = token_color(names[0]) if names else CATEGORY_COLORS["Disposition"]
    _draw_bar_panel(ax, short_names, sims, color, exp["label"])


# Brainstorm: CROSS-CATEGORY treatment blends -> chief complaints (temporary).
#
# Question: "what presenting complaint is associated with a *combination* of
# treatments?" We average two treatment embeddings (meds and/or procedures) and
# retrieve the nearest chief-complaint tokens. Raw cross-category cosine is
# dominated by the constant category offset (every treatment is roughly equ"far"
# from the CC cluster) plus token frequency, so we first project the blend into
# complaint space by subtracting the mean treatment->complaint offset learned
# from clinically-valid pairs (the same offset the parallelogram analysis
# validated). The projected point then lands on the shared clinical context.
#
# Lead example (transfusion + tranexamic acid) localizes to a hemorrhage
# picture: hematemesis / airway obstruction / dysphagia.
TX_CC_OFFSET_PAIRS = [
    ("Med_albuterol sulfate", "CC_WHEEZING"),
    ("Med_epinephrine", "CC_ALLERGIC REACTION"),
    ("Med_ondansetron", "CC_EMESIS"),
    ("Med_levetiracetam", "CC_SEIZURES"),
    ("Med_acetaminophen", "CC_FEVER"),
    ("Med_naloxone", "CC_DRUG OVERDOSE"),
    ("Med_morphine", "CC_CHEST PAIN"),
    ("Med_ketorolac", "CC_ABDOMINAL PAIN"),
    ("Procedure_XR CHEST", "CC_RESPIRATORY DISTRESS"),
    ("Procedure_REDUCTION", "CC_ARM INJURY"),
]

BRAINSTORM_COMBOS = [
    (("Procedure_BLOOD TRANSFUSION ORDERABLES", "Med_tranexamic acid"),
     "transfusion\n+ tranexamic acid = ?"),
    (("Med_albuterol sulfate", "Med_ipratropium"),
     "albuterol\n+ ipratropium = ?"),
    (("Med_epinephrine", "Med_diphenhydramine"),
     "epinephrine\n+ diphenhydramine = ?"),
    (("Med_ceftriaxone", "Med_acyclovir"),
     "ceftriaxone\n+ acyclovir = ?"),
]


def _treatment_cc_offset(token_to_emb: dict) -> np.ndarray:
    """Mean (treatment - complaint) offset from clinically-valid pairs."""
    offs = [token_to_emb[m] - token_to_emb[c]
            for m, c in TX_CC_OFFSET_PAIRS
            if m in token_to_emb and c in token_to_emb]
    return np.mean(offs, axis=0)


def compute_combo_complaints(token_to_emb: dict, combos=BRAINSTORM_COMBOS,
                             top_k: int = 5) -> list[dict]:
    """For each treatment pair, chief complaints nearest to the blend projected
    into complaint space (blend - mean treatment->complaint offset)."""
    offset = _treatment_cc_offset(token_to_emb)
    cc = {n: e for n, e in token_to_emb.items() if n.startswith("CC_")}
    results = []
    for (a, b), label in combos:
        if a not in token_to_emb or b not in token_to_emb:
            print(f"  [combo] skip missing: {a} / {b}")
            continue
        query = (token_to_emb[a] + token_to_emb[b]) / 2 - offset
        scored = [(n, cosine_sim(query, e)) for n, e in cc.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        results.append({"label": label, "pair": (a, b), "results": scored[:top_k]})
    return results


def _draw_combo_panel(ax, combo: dict):
    """One panel: chief complaints associated with a treatment combination.
    Bars use the chief-complaint category color (the retrieved token type)."""
    names = [pretty_label(n) for n, _ in combo["results"]]
    sims = [s for _, s in combo["results"]]
    _draw_bar_panel(ax, names, sims, CATEGORY_COLORS["Chief Complaint"],
                    combo["label"])


# ============================================================
# Nearest-neighbor retrieval (top block of the combined figure)
# ============================================================

PROBE_TOKENS = [
    "Med_ceftriaxone",
    "Med_albuterol sulfate",
    "CC_CHEST PAIN",
    "CC_SEIZURES",
    "Procedure_XR CHEST",
    "Lab_Potassium, Serum / Plasma_Abnormal",
    "Vital_O2 Device_High flow nasal cannula",
    "Acuity_Immediate",
    "Age_2",
    "Vital_SpO2_90",
]


def compute_neighbors(token_to_emb: dict, top_k: int = 4) -> list[dict]:
    records = []
    rows = []
    for probe in PROBE_TOKENS:
        if probe not in token_to_emb:
            print(f"  [neighbors] skip missing probe: {probe}")
            continue
        neigh = find_nearest_dedup(
            token_to_emb[probe], token_to_emb, exclude={probe},
            top_k=top_k, exclude_bases={med_base(probe)})
        records.append({"probe": probe, "neighbors": neigh})
        for rank, (name, sim) in enumerate(neigh, 1):
            rows.append({"probe": probe, "rank": rank, "neighbor": name,
                         "cosine": round(sim, 4)})
    pd.DataFrame(rows).to_csv(METRIC_DIR / "analysis1_neighbors.csv", index=False)
    return records


def _draw_neighbor_panel(ax, rec: dict):
    """One nearest-neighbor panel, colored by the probe token's category."""
    neigh = rec["neighbors"]
    names = [pretty_label(nm) for nm, _ in neigh]
    sims = [s for _, s in neigh]
    _draw_bar_panel(ax, names, sims, token_color(rec["probe"]),
                    probe_title(rec["probe"]))


# ============================================================
# COMBINED FIGURE (one supplement page)
# ============================================================

# Legend entries: which token categories appear in the figure.
_LEGEND_CATEGORIES = [
    ("Medication", "Medication"),
    ("Chief Complaint", "Chief complaint"),
    ("Procedure", "Procedure"),
    ("Lab", "Lab"),
    ("Vital", "Vital sign"),
    ("Acuity", "Triage acuity"),
    ("Age", "Age"),
]


def plot_combined_figure(neighbor_records: list[dict], analogy_panels: list[dict],
                         combo_panels: list[dict]):
    """Single supplement-page figure.

    Top block (a): nearest-neighbor retrieval for representative tokens (5x2).
    Bottom block (b): clinical vector arithmetic -- three analogy panels plus two
    cross-category treatment-blend -> chief-complaint panels (1x5). Every panel
    shows the top-4 matches so bar heights are identical throughout.
    Bars are colored by clinical token category (shared with token.py).
    """
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))
    plt.rcParams["figure.constrained_layout.use"] = False

    n_probe = len(neighbor_records)
    n_prow = int(np.ceil(n_probe / 2))          # neighbor rows (2 columns)
    fig = plt.figure(figsize=(7.2, 0.92 * n_prow + 2.1))

    # Two stacked blocks: neighbors on top, analogies on the bottom.
    outer = fig.add_gridspec(
        2, 1, height_ratios=[n_prow, 1.7], hspace=0.22,
        left=0.040, right=0.985, top=0.955, bottom=0.075)

    # --- Block a: nearest neighbors (5x2) ---
    gs_top = outer[0].subgridspec(n_prow, 2, hspace=1.20, wspace=0.75)
    top_axes = [fig.add_subplot(gs_top[i // 2, i % 2]) for i in range(n_probe)]
    for ax, rec in zip(top_axes, neighbor_records):
        _draw_neighbor_panel(ax, rec)

    # --- Block b: vector arithmetic (1x5) ---
    n_bot = len(analogy_panels) + len(combo_panels)
    gs_bot = outer[1].subgridspec(1, n_bot, wspace=0.9)
    bot_axes = [fig.add_subplot(gs_bot[0, i]) for i in range(n_bot)]
    for ax, exp in zip(bot_axes[:len(analogy_panels)], analogy_panels):
        _draw_analogy_panel(ax, exp)
    for ax, combo in zip(bot_axes[len(analogy_panels):], combo_panels):
        _draw_combo_panel(ax, combo)

    # Panel labels (lowercase, Nature convention), placed in figure space so
    # both letters share the same x coordinate even though the panels differ in
    # width.
    label_x = -0.04
    top_label_y = top_axes[0].get_position().y1 + 0.012
    bot_label_y = bot_axes[0].get_position().y1 + 0.012
    fig.text(label_x, top_label_y, "a", fontsize=10, fontweight="bold",
             va="top", ha="right")
    fig.text(label_x, bot_label_y, "b", fontsize=10, fontweight="bold",
             va="top", ha="right")

    # Shared category legend.
    handles = [plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLORS[key])
               for key, _ in _LEGEND_CATEGORIES]
    labels = [lab for _, lab in _LEGEND_CATEGORIES]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               fontsize=5.8, frameon=False, handlelength=1.0, handleheight=1.0,
               columnspacing=1.3, handletextpad=0.4,
               bbox_to_anchor=(0.5, 0.0))

    plt.savefig(SAVE_DIR / "token_semantics.pdf", dpi=600)
    plt.savefig(SAVE_DIR / "token_semantics.png", dpi=600)
    plt.close()
    print(f"Saved combined figure to {SAVE_DIR / 'token_semantics.png'}")


# ============================================================
# Console output
# ============================================================

def print_analogies(all_results: dict[str, list[dict]]):
    for cat_name, experiments in all_results.items():
        print(f"\n{'='*54}\n  {cat_name}\n{'='*54}")
        for exp in experiments:
            print(f"\n  {exp['label'].replace(chr(10), ' ')}")
            print(f"  Expected: {exp.get('expected', '?')}")
            for name, sim in exp["results"]:
                marker = " <--" if name == exp.get("expected") else ""
                print(f"    {sim:.4f}  {name}{marker}")


# ============================================================
# Main
# ============================================================

def main():
    print("Loading embeddings...")
    token_to_emb, _ = load_embeddings()
    dim = next(iter(token_to_emb.values())).shape[0]
    print(f"Loaded {len(token_to_emb)} token embeddings (dim={dim})")

    print("\n--- Vector-arithmetic analogies (top-3, category-restricted) ---")
    all_results = {
        "Intervention Pairs": intervention_pairs(token_to_emb),
        "Age Transformations": age_transformations(token_to_emb),
        "Diagnostic Workup": diagnostic_workup(token_to_emb),
    }
    # Bottom-row analogy panels: ketamine substitution, acetaminophen
    # fever->seizures substitution, and the febrile-seizure age transformation.
    ketamine = all_results["Intervention Pairs"][0]
    acetaminophen_sub = all_results["Intervention Pairs"][3]
    febrile_age = all_results["Age Transformations"][0]
    analogy_panels = [ketamine, acetaminophen_sub, febrile_age]
    print_analogies({"Bottom-row analogies": analogy_panels})

    print("\n--- Cross-category treatment blends -> chief complaints ---")
    # The two clean cross-category panels: hemorrhage and obstructive airway.
    combos = compute_combo_complaints(token_to_emb, combos=BRAINSTORM_COMBOS[:2], top_k=4)
    for combo in combos:
        tops = ", ".join(f"{pretty_label(n)} ({s:.2f})" for n, s in combo["results"])
        print(f"  {combo['label'].replace(chr(10), ' ')}: {tops}")

    print("\n--- Nearest-neighbor retrieval (top-4, deduped) ---")
    records = compute_neighbors(token_to_emb, top_k=4)
    for rec in records:
        top = ", ".join(pretty_label(n) for n, _ in rec["neighbors"])
        print(f"  {probe_title(rec['probe']):40s} -> {top}")

    plot_combined_figure(records, analogy_panels, combos)


if __name__ == "__main__":
    main()
