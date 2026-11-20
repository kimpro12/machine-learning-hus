"""Metrics: accuracy, F1 (phishing=1), ROC/PR-AUC, bootstrap CI, threshold tuning."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


def scores_from_model(pipe, X):
    # prefer predict_proba, else decision_function
    if hasattr(pipe, "predict_proba"):
        try:
            return np.asarray(pipe.predict_proba(X)[:, 1], dtype=float)
        except Exception:
            pass
    if hasattr(pipe, "decision_function"):
        try:
            return np.asarray(pipe.decision_function(X), dtype=float)
        except Exception:
            pass
    return np.asarray(pipe.predict(X), dtype=float)


def compute_metrics(y_true, y_pred, y_score=None) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    out = {
        "n": int(len(y_true)),
        "prevalence_phishing": float(y_true.mean()) if len(y_true) else 0.0,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_phishing": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_phishing": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_phishing": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }
    if y_score is not None:
        y_score = np.asarray(y_score, dtype=float)
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            out["roc_auc"] = float("nan")
        try:
            out["pr_auc"] = float(average_precision_score(y_true, y_score))
        except Exception:
            out["pr_auc"] = float("nan")
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def tune_threshold(y_true, y_score, rule="f1"):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    thrs = np.unique(np.quantile(y_score, np.linspace(0, 1, 101)))
    best = {"threshold": 0.5 if y_score.max() <= 1.0 and y_score.min() >= 0.0 else 0.0}
    # default metric
    def_thr = best["threshold"]
    ypd = (y_score >= def_thr).astype(int)
    best["default"] = compute_metrics(y_true, ypd, y_score)
    best_f1, best_thr = -1.0, def_thr
    best_prec_at_r95, thr_r95 = -1.0, None
    for t in thrs:
        yp = (y_score >= t).astype(int)
        f1 = f1_score(y_true, yp, zero_division=0)
        rec = recall_score(y_true, yp, zero_division=0)
        prec = precision_score(y_true, yp, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(t)
        if rec >= 0.95 and prec > best_prec_at_r95:
            best_prec_at_r95, thr_r95 = prec, float(t)
    yp_best = (y_score >= best_thr).astype(int)
    best["tuned_threshold_f1"] = best_thr
    best["tuned"] = compute_metrics(y_true, yp_best, y_score)
    best["recall95_threshold"] = thr_r95
    if thr_r95 is not None:
        best["recall95"] = compute_metrics(y_true, (y_score >= thr_r95).astype(int), y_score)
    if rule == "recall95" and thr_r95 is not None:
        best["selected_threshold"] = thr_r95
        best["selected"] = best["recall95"]
    else:
        best["selected_threshold"] = best_thr
        best["selected"] = best["tuned"]
    return best


def bootstrap_ci(y_true, y_pred, n_boot=500, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    n = len(y_true)
    accs, f1s = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        # skip degenerate (single class) samples
        if len(np.unique(yt)) < 2:
            continue
        accs.append(accuracy_score(yt, yp))
        f1s.append(f1_score(yt, yp, zero_division=0))
    def ci(a):
        a = np.asarray(a)
        if len(a) == 0:
            return (float("nan"), float("nan"))
        return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    out = {"acc_ci95": ci(accs), "f1_ci95": ci(f1s), "n_boot": int(len(accs))}
    # Corrected exact binomial (Clopper-Pearson) CI for accuracy.
    # Percentile bootstrap is degenerate when accuracy==1.0 (all resamples 1.0 -> [1,1]),
    # which understates uncertainty. Exact CI gives proper lower bound via Rule-of-Three.
    try:
        k = int((y_true == y_pred).sum())
        out["acc_exact_ci95"] = clopper_pearson_ci(k, n)
        # Flag degenerate bootstrap
        if k == n:
            out["acc_ci95_note"] = "percentile bootstrap degenerate [1,1]; use acc_exact_ci95"
        # F1 percentile is likewise degenerate when no errors; flag it
        if k == n:
            out["f1_ci95_note"] = "percentile bootstrap degenerate [1,1] when zero errors; do not interpret as zero uncertainty"
    except Exception:
        pass
    return out


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05):
    """Exact Clopper-Pearson 95% CI for binomial proportion k/n.

    For perfect accuracy k==n, lower = (alpha/2)**(1/n) ~= 1 - 3/n (Rule of Three).
    Returns (lo, hi). Uses scipy if available, else closed-form for edges.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    try:
        from scipy.stats import beta as beta_dist

        lo = float(beta_dist.ppf(alpha / 2, k, n - k + 1)) if k > 0 else 0.0
        hi = float(beta_dist.ppf(1 - alpha / 2, k + 1, n - k)) if k < n else 1.0
        return (lo, hi)
    except Exception:
        import math

        if k == n:
            lo = float((alpha / 2) ** (1.0 / n))
            return (lo, 1.0)
        if k == 0:
            hi = float(1.0 - (alpha / 2) ** (1.0 / n))
            return (0.0, hi)
        # Wilson fallback
        z = 1.96
        p = k / n
        denom = 1 + z * z / n
        centre = p + z * z / (2 * n)
        sd = math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return (max(0.0, (centre - z * sd) / denom), min(1.0, (centre + z * sd) / denom))
