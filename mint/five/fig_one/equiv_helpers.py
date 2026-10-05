import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def deterministic_prefix_sample(n_total: int, size: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    return perm[:size]


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train an XGBoost classifier with standard hyperparameters."""
    from xgboost import XGBClassifier

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    clf = XGBClassifier(
        random_state=42,
        n_jobs=1,
        eval_metric="aucpr",
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return clf

def train_xgb_for_size(
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_size: int,
    seed: int = 42,
) -> tuple[object, np.ndarray]:
    idx = deterministic_prefix_sample(len(y_train_full), min(n_size, len(y_train_full)), seed=seed)
    X_sub = X_train_full[idx]
    y_sub = y_train_full[idx]
    if y_sub.sum() == 0 or y_sub.sum() == len(y_sub):
        return None, y_sub
    clf = train_xgboost(X_sub, y_sub, X_val, y_val)
    return clf, y_sub


def _eval_size(X_train_full, y_train_full, X_val, y_val, X_test, y_test, size, seed):
    clf, _ = train_xgb_for_size(X_train_full, y_train_full, X_val, y_val, size, seed=seed)
    if clf is None:
        return None
    score = clf.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(y_test, score))


def equivalence_search(
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    mint_auroc: float,
    task_key: str,
    verbose: bool = True,
    precision: int = 100,
    seed: int = 42,
) -> int | None:
    """Binary search for the smallest training set size where XGBoost outperforms MINT by 0.5% AUROC or more.

    Returns the exact sample count (within `precision`) or None if not reached.
    """
    if y_test.sum() == 0 or y_test.sum() == len(y_test):
        return None

    n_total = len(y_train_full)
    target_auroc = mint_auroc + 0.005  # 0.5% improvement

    # Coarse scan to find upper bound
    coarse_fractions = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    coarse_sizes = sorted(set(max(precision, int(f * n_total)) for f in coarse_fractions))

    hi: int | None = None
    lo: int = 0

    for size in coarse_sizes:
        if size > n_total:
            continue
        auroc = _eval_size(X_train_full, y_train_full, X_val, y_val, X_test, y_test, size, seed)
        if auroc is None:
            continue
        if verbose:
            print(f"    [eq {task_key}] n={size:,} AUROC={auroc:.4f} (target={target_auroc:.4f})")
        if auroc >= target_auroc:
            hi = size
            break
        lo = size

    if hi is None:
        return None

    # Binary search between lo and hi
    while hi - lo > precision:
        mid = (lo + hi) // 2
        auroc = _eval_size(X_train_full, y_train_full, X_val, y_val, X_test, y_test, mid, seed)
        if verbose:
            print(f"    [eq {task_key}] mid={mid:,} AUROC={auroc:.4f}" if auroc else f"    [eq {task_key}] mid={mid:,} (degenerate)")
        if auroc is not None and auroc >= target_auroc:
            hi = mid
        else:
            lo = mid

    return hi
