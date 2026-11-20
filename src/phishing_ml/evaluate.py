"""Metrics: accuracy, F1 (phishing=1), ROC/PR-AUC, exact Clopper-Pearson CI."""
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
