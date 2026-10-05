# =============================================================================
# BLOCK 4: STATISTICAL COMPARISONS (execute after Block 3)
# =============================================================================
# Notebook cell — runs after the run block. All names from prior blocks are
# already in scope. Do NOT add imports for symbols defined in other blocks.
#
# Computes paired bootstrap CIs for AUROC and AUPRC for each method, plus
# paired differences (method - target) with significance testing, against one
# or more reference methods. The bootstrap is encounter-clustered: each
# iteration resamples encounter_keys with replacement from the union of all
# methods, and all rows for a sampled encounter are included together.
#
# The per-method AUROC/AUPRC bootstrap is computed ONCE and reused for every
# reference method, so adding a second target (e.g. XGBoost) costs only a few
# extra percentile calls for methods aligned with that target (same encounter
# set), plus a slow-path recompute inside the loop for any misaligned methods.
#
# Outputs (one diffs file per reference method; primary target keeps the legacy
# filename so downstream consumers/tests are unaffected):
#   output/<hospital>/bootstrap_stats.csv            per-method AUROC/AUPRC + 95% CIs
#   output/<hospital>/bootstrap_diffs.csv            paired diffs vs primary target (Softmax)
#   output/<hospital>/bootstrap_diffs_xgboost.csv    paired diffs vs XGBoost
#   output/bootstrap_summary.csv                     combined per-method summary
#   output/bootstrap_diffs_summary.csv               combined diffs vs primary target
#   output/bootstrap_diffs_xgboost_summary.csv       combined diffs vs XGBoost
#
# Configuration (set before running this block):
#   STATS_TARGET_METHODS = ["Softmax", "XGBoost"]  # reference methods for paired diffs
#   STATS_N_BOOTSTRAP = 1000                         # bootstrap iterations
#   STATS_SEED = 42                                  # random seed for reproducibility
#
# Testing:
#   Run the integration test after any changes:
#     TEST_OUTCOMES=tachypnea python -m mint.five.fig_one.notebook.test_fig1_notebook
# =============================================================================

# ─── Configuration ───────────────────────────────────────────────────────────

# Reference methods for paired differences. The first entry is the "primary"
# target and its diffs are written to the legacy bootstrap_diffs.csv filename;
# each additional target gets its own suffixed file (e.g. XGBoost -> _xgboost).
STATS_TARGET_METHODS = ["Softmax", "XGBoost"]
STATS_N_BOOTSTRAP = 1000         # bootstrap iterations (1000 is standard)
STATS_SEED = 42                  # random seed

# Methods to include in the comparison (CSVs that don't exist are skipped)
# Includes all standard methods plus fine-tuned variants (PEFT_BACKBONE outputs)
STATS_METHODS = [
    "Softmax", "Softmax_FT",
    "XGBoost", "GlobalXGBoost",
    "Triage",
    "MINT_SVM", "MINT_SVM_FT",
    "MINT_LR", "MINT_LR_FT",
    "ClassHead",  # PEFT_CLASSIFICATION output
]

# ─── Bootstrap helpers (adapted from fig1_nejm.py) ───────────────────────────

def _both_metrics_boot(labels, probs):
    """Compute AUROC and AUPRC; return (nan, nan) if only one class."""
    if len(labels) == 0 or labels.min() == labels.max():
        return np.nan, np.nan
    return roc_auc_score(labels, probs), average_precision_score(labels, probs)


class _MethodBootstrap:
    """Encounter-clustered gather structure for one method.

    Enables fast resampling: given a list of sampled encounter codes, returns
    all (probs, labels) rows belonging to those encounters.
    """

    def __init__(self, codes, probs, labels, n_union):
        order = np.argsort(codes, kind="stable")
        self.codes = codes[order]
        self.probs = probs[order]
        self.labels = labels[order]
        self.counts = np.bincount(self.codes, minlength=n_union)
        self.offsets = np.zeros(n_union + 1, dtype=np.int64)
        self.offsets[1:] = np.cumsum(self.counts)
        self.has_key = self.counts > 0

    def gather(self, sampled_codes):
        """Return (probs, labels) for all rows belonging to sampled encounters."""
        sel_counts = self.counts[sampled_codes]
        total = int(sel_counts.sum())
        if total == 0:
            return np.empty(0), np.empty(0, dtype=int)
        starts = self.offsets[sampled_codes]
        base = np.repeat(starts, sel_counts)
        within = np.arange(total) - np.repeat(np.cumsum(sel_counts) - sel_counts, sel_counts)
        idx = base + within
        return self.probs[idx], self.labels[idx]


def _compute_ci(arr, alpha=0.05):
    """Compute confidence interval from bootstrap samples."""
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    lo = float(np.percentile(arr, 100 * alpha / 2))
    hi = float(np.percentile(arr, 100 * (1 - alpha / 2)))
    return lo, hi


def _is_significant(ci_lo, ci_hi):
    """Check if CI excludes 0 (both endpoints on same side)."""
    if np.isnan(ci_lo) or np.isnan(ci_hi):
        return False
    return (ci_lo > 0 and ci_hi > 0) or (ci_lo < 0 and ci_hi < 0)


def paired_encounter_bootstrap(methods_data, method_keys, target_keys, n_boot, seed):
    """Paired encounter-clustered bootstrap for multiple methods.

    The per-method AUROC/AUPRC bootstrap is computed once and reused for every
    reference method, so comparing against additional targets adds only a few
    percentile calls (for methods aligned with a target) plus an in-loop
    slow-path recompute for any methods misaligned with a target.

    Args:
        methods_data: dict[method_name -> DataFrame with encounter_key, probs, labels]
        method_keys: list of method names to include
        target_keys: list of reference methods for paired differences. A str or
            None is accepted for backward compatibility (None -> no diffs).
        n_boot: number of bootstrap iterations
        seed: random seed

    Returns:
        dict with keys:
            'methods': {method -> {auroc, auprc, auroc_ci, auprc_ci}}
            'diffs':   {target -> {method -> {auroc_diff, auprc_diff,
                        auroc_diff_ci, auprc_diff_ci, auroc_sig, auprc_sig}}}
    """
    # Normalize target_keys to a list of strings
    if target_keys is None:
        target_keys = []
    elif isinstance(target_keys, str):
        target_keys = [target_keys]

    rng = np.random.default_rng(seed)

    # Build union of all encounter keys across methods
    all_keys = [methods_data[k]["encounter_key"].to_numpy() for k in method_keys if k in methods_data]
    if not all_keys:
        return {"methods": {}, "diffs": {}}
    union_keys = np.unique(np.concatenate(all_keys))
    n_union = len(union_keys)

    # Build gather structures for each method
    boot_structs = {}
    for k in method_keys:
        if k not in methods_data:
            continue
        df = methods_data[k]
        codes = np.searchsorted(union_keys, df["encounter_key"].to_numpy())
        boot_structs[k] = _MethodBootstrap(
            codes,
            df["probs"].to_numpy(),
            df["labels"].to_numpy(dtype=int),
            n_union
        )

    available_methods = list(boot_structs.keys())
    active_targets = [t for t in target_keys if t in boot_structs]

    # Bootstrap arrays for per-method metrics (computed once, shared by all targets)
    auroc_boot = {k: np.full(n_boot, np.nan) for k in available_methods}
    auprc_boot = {k: np.full(n_boot, np.nan) for k in available_methods}

    # For each target: which non-target methods are aligned (same encounter set)?
    diff_methods = {t: [k for k in available_methods if k != t] for t in active_targets}
    aligned = {t: {} for t in active_targets}
    for t in active_targets:
        t_has = boot_structs[t].has_key
        for k in diff_methods[t]:
            aligned[t][k] = bool(np.array_equal(boot_structs[k].has_key, t_has))

    # Slow-path diff arrays for (target, misaligned-method) pairs
    diff_auroc_boot = {t: {k: np.full(n_boot, np.nan) for k in diff_methods[t]
                           if not aligned[t].get(k, False)} for t in active_targets}
    diff_auprc_boot = {t: {k: np.full(n_boot, np.nan) for k in diff_methods[t]
                           if not aligned[t].get(k, False)} for t in active_targets}

    # Run bootstrap
    for b in range(n_boot):
        sampled = rng.integers(0, n_union, size=n_union)

        for k in available_methods:
            p, l = boot_structs[k].gather(sampled)
            if len(l) > 0:
                auroc_boot[k][b], auprc_boot[k][b] = _both_metrics_boot(l, p)

        # Slow-path diffs for methods misaligned with each target
        for t in active_targets:
            t_has = boot_structs[t].has_key
            for k in diff_auroc_boot[t]:
                both = t_has & boot_structs[k].has_key
                si = sampled[both[sampled]]
                if len(si) == 0:
                    continue
                tp, tl = boot_structs[t].gather(si)
                mp, ml = boot_structs[k].gather(si)
                t_auroc, t_auprc = _both_metrics_boot(tl, tp)
                m_auroc, m_auprc = _both_metrics_boot(ml, mp)
                diff_auroc_boot[t][k][b] = m_auroc - t_auroc
                diff_auprc_boot[t][k][b] = m_auprc - t_auprc

    # Assemble results
    result = {"methods": {}, "diffs": {t: {} for t in active_targets}}

    for k in available_methods:
        df = methods_data[k]
        pe_auroc, pe_auprc = _both_metrics_boot(df["labels"].to_numpy(), df["probs"].to_numpy())
        auroc_ci = _compute_ci(auroc_boot[k])
        auprc_ci = _compute_ci(auprc_boot[k])
        result["methods"][k] = {
            "auroc": pe_auroc,
            "auprc": pe_auprc,
            "auroc_ci_lo": auroc_ci[0],
            "auroc_ci_hi": auroc_ci[1],
            "auprc_ci_lo": auprc_ci[0],
            "auprc_ci_hi": auprc_ci[1],
        }

    # Paired differences vs each target
    for t in active_targets:
        t_df = methods_data[t]
        for k in diff_methods[t]:
            # Point estimate on intersection
            m_df = methods_data[k]
            shared = np.intersect1d(t_df["encounter_key"].to_numpy(), m_df["encounter_key"].to_numpy())
            t_sub = t_df[t_df["encounter_key"].isin(shared)]
            m_sub = m_df[m_df["encounter_key"].isin(shared)]
            t_auroc, t_auprc = _both_metrics_boot(t_sub["labels"].to_numpy(), t_sub["probs"].to_numpy())
            m_auroc, m_auprc = _both_metrics_boot(m_sub["labels"].to_numpy(), m_sub["probs"].to_numpy())

            # CI from bootstrap
            if aligned[t].get(k, False):
                da = auroc_boot[k] - auroc_boot[t]
                dp = auprc_boot[k] - auprc_boot[t]
            else:
                da = diff_auroc_boot[t][k]
                dp = diff_auprc_boot[t][k]

            auroc_diff_ci = _compute_ci(da)
            auprc_diff_ci = _compute_ci(dp)

            result["diffs"][t][k] = {
                "auroc_diff": m_auroc - t_auroc,
                "auprc_diff": m_auprc - t_auprc,
                "auroc_diff_ci_lo": auroc_diff_ci[0],
                "auroc_diff_ci_hi": auroc_diff_ci[1],
                "auprc_diff_ci_lo": auprc_diff_ci[0],
                "auprc_diff_ci_hi": auprc_diff_ci[1],
                "auroc_sig": _is_significant(auroc_diff_ci[0], auroc_diff_ci[1]),
                "auprc_sig": _is_significant(auprc_diff_ci[0], auprc_diff_ci[1]),
            }

    return result


def load_method_predictions(hosp_dir, outcome, methods):
    """Load prediction CSVs for all available methods for an outcome.

    Returns dict[method_name -> DataFrame with encounter_key, probs, labels].
    Methods without CSVs are silently skipped.
    """
    data = {}
    for method in methods:
        csv_path = hosp_dir / f"{outcome}_{method}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if "encounter_key" in df.columns and "probs" in df.columns and "labels" in df.columns:
                data[method] = df[["encounter_key", "probs", "labels"]].copy()
    return data


# ─── Main stats computation ──────────────────────────────────────────────────

# Map each target to the diffs filename suffix. The primary (first) target
# keeps the legacy unsuffixed filename so existing consumers/tests are unaffected.
def _diffs_suffix(target, is_primary):
    return "" if is_primary else f"_{target.lower()}"

logger.info("=" * 60)
logger.info("BLOCK 4: PAIRED BOOTSTRAP STATISTICS")
logger.info("=" * 60)
logger.info(f"Target methods: {STATS_TARGET_METHODS}")
logger.info(f"Bootstrap iterations: {STATS_N_BOOTSTRAP}")
logger.info(f"Methods: {STATS_METHODS}")

all_stats = []
# Diffs accumulated per target: {target -> [rows]}
all_diffs = {t: [] for t in STATS_TARGET_METHODS}

for hospital_name in hospital_splits:
    hosp_dir = OUTPUT_DIR / hospital_name
    if not hosp_dir.exists():
        logger.warning(f"  [{hospital_name}] Output directory not found, skipping")
        continue

    logger.info(f"  [{hospital_name}] Running paired bootstrap...")

    hosp_stats_rows = []
    hosp_diffs_rows = {t: [] for t in STATS_TARGET_METHODS}

    for outcome in OUTCOMES:
        # Load predictions for all methods
        methods_data = load_method_predictions(hosp_dir, outcome, STATS_METHODS)

        if len(methods_data) < 2:
            logger.warning(f"    [{hospital_name}] {outcome}: <2 methods available, skipping bootstrap")
            continue

        available = list(methods_data.keys())
        targets = [t for t in STATS_TARGET_METHODS if t in methods_data]

        for missing in [t for t in STATS_TARGET_METHODS if t not in methods_data]:
            logger.warning(f"    [{hospital_name}] {outcome}: target '{missing}' not available")

        logger.info(f"    [{hospital_name}] {outcome}: bootstrapping {len(available)} methods "
                    f"(targets={targets}, n_boot={STATS_N_BOOTSTRAP})...")

        # Run paired bootstrap (per-method bootstrap computed once, reused per target)
        boot_result = paired_encounter_bootstrap(
            methods_data, available, targets, STATS_N_BOOTSTRAP, STATS_SEED
        )

        # Collect per-method stats
        for method, m in boot_result["methods"].items():
            row = {
                "hospital": hospital_name,
                "outcome": outcome,
                "method": method,
                "n_encounters": len(methods_data[method]["encounter_key"].unique()),
                "n_total": len(methods_data[method]),
                "n_pos": int(methods_data[method]["labels"].sum()),
                "auroc": m["auroc"],
                "auroc_ci_lo": m["auroc_ci_lo"],
                "auroc_ci_hi": m["auroc_ci_hi"],
                "auprc": m["auprc"],
                "auprc_ci_lo": m["auprc_ci_lo"],
                "auprc_ci_hi": m["auprc_ci_hi"],
            }
            hosp_stats_rows.append(row)
            all_stats.append(row)

        # Collect paired diffs (one bucket per target)
        for target, diffs in boot_result["diffs"].items():
            for method, d in diffs.items():
                row = {
                    "hospital": hospital_name,
                    "outcome": outcome,
                    "method": method,
                    "target": target,
                    "auroc_diff": d["auroc_diff"],
                    "auroc_diff_ci_lo": d["auroc_diff_ci_lo"],
                    "auroc_diff_ci_hi": d["auroc_diff_ci_hi"],
                    "auroc_sig": d["auroc_sig"],
                    "auprc_diff": d["auprc_diff"],
                    "auprc_diff_ci_lo": d["auprc_diff_ci_lo"],
                    "auprc_diff_ci_hi": d["auprc_diff_ci_hi"],
                    "auprc_sig": d["auprc_sig"],
                }
                hosp_diffs_rows[target].append(row)
                all_diffs[target].append(row)

    # Save per-hospital CSVs
    if hosp_stats_rows:
        stats_df = pd.DataFrame(hosp_stats_rows)
        stats_path = hosp_dir / "bootstrap_stats.csv"
        stats_df.to_csv(stats_path, index=False)
        logger.info(f"    [{hospital_name}] Saved {stats_path.name}")

    for i, target in enumerate(STATS_TARGET_METHODS):
        rows = hosp_diffs_rows[target]
        if not rows:
            continue
        suffix = _diffs_suffix(target, is_primary=(i == 0))
        diffs_df = pd.DataFrame(rows)
        diffs_path = hosp_dir / f"bootstrap_diffs{suffix}.csv"
        diffs_df.to_csv(diffs_path, index=False)
        logger.info(f"    [{hospital_name}] Saved {diffs_path.name} (vs {target})")

# ─── Combined summary ────────────────────────────────────────────────────────

if all_stats:
    summary_stats_df = pd.DataFrame(all_stats)
    summary_stats_path = OUTPUT_DIR / "bootstrap_summary.csv"
    summary_stats_df.to_csv(summary_stats_path, index=False)
    logger.info(f"Saved combined stats: {summary_stats_path}")

for i, target in enumerate(STATS_TARGET_METHODS):
    rows = all_diffs[target]
    if not rows:
        continue
    suffix = _diffs_suffix(target, is_primary=(i == 0))
    summary_diffs_df = pd.DataFrame(rows)
    summary_diffs_path = OUTPUT_DIR / f"bootstrap_diffs{suffix}_summary.csv"
    summary_diffs_df.to_csv(summary_diffs_path, index=False)
    logger.info(f"Saved combined diffs (vs {target}): {summary_diffs_path}")

# ─── Print summary table ─────────────────────────────────────────────────────

logger.info("=" * 60)
logger.info("BOOTSTRAP STATISTICS COMPLETE")
logger.info("=" * 60)

if all_stats:
    print("\n" + "=" * 140)
    print(f"{'Hospital':<15} {'Outcome':<15} {'Method':<12} {'AUROC':<22} {'AUPRC':<22} {'n_enc':<8} {'n_pos'}")
    print("-" * 140)
    for r in all_stats:
        auroc_str = f"{r['auroc']:.4f} ({r['auroc_ci_lo']:.4f}-{r['auroc_ci_hi']:.4f})"
        auprc_str = f"{r['auprc']:.4f} ({r['auprc_ci_lo']:.4f}-{r['auprc_ci_hi']:.4f})"
        print(f"{r['hospital']:<15} {r['outcome']:<15} {r['method']:<12} {auroc_str:<22} {auprc_str:<22} {r['n_encounters']:<8} {r['n_pos']}")
    print("=" * 140)

diffs_flat = [r for target in STATS_TARGET_METHODS for r in all_diffs[target]]
if diffs_flat:
    print("\n" + "=" * 160)
    print(f"{'Hospital':<15} {'Outcome':<15} {'Method':<12} {'vs':<10} {'ΔAUROC':<28} {'sig':<5} {'ΔAUPRC':<28} {'sig'}")
    print("-" * 160)
    for r in diffs_flat:
        auroc_str = f"{r['auroc_diff']:+.4f} ({r['auroc_diff_ci_lo']:.4f} to {r['auroc_diff_ci_hi']:.4f})"
        auprc_str = f"{r['auprc_diff']:+.4f} ({r['auprc_diff_ci_lo']:.4f} to {r['auprc_diff_ci_hi']:.4f})"
        auroc_sig = "*" if r['auroc_sig'] else ""
        auprc_sig = "*" if r['auprc_sig'] else ""
        print(f"{r['hospital']:<15} {r['outcome']:<15} {r['method']:<12} {r['target']:<10} {auroc_str:<28} {auroc_sig:<5} {auprc_str:<28} {auprc_sig}")
    print("=" * 160)
    print("* = 95% CI excludes 0 (statistically significant difference)")
