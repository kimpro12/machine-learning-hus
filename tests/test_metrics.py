from phishing_ml.evaluate import compute_metrics, tune_threshold, bootstrap_ci

def test_confusion_four_cases():
    # TN, FP, FN, TP each 1: y_true=[0,0,1,1], y_pred=[0,1,0,1]
    m = compute_metrics([0, 0, 1, 1], [0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert (m["TN"], m["FP"], m["FN"], m["TP"]) == (1, 1, 1, 1)
    assert abs(m["accuracy"] - 0.5) < 1e-9
    assert abs(m["precision_phishing"] - 0.5) < 1e-9
    assert abs(m["recall_phishing"] - 0.5) < 1e-9
    assert m["roc_auc"] > 0.4

def test_threshold_val_only():
    import numpy as np
    yt = np.array([0, 0, 1, 1, 1, 0])
    ys = np.array([0.1, 0.4, 0.6, 0.8, 0.9, 0.2])
    out = tune_threshold(yt, ys, rule="f1")
    assert "selected_threshold" in out
    assert 0.0 <= out["selected_threshold"] <= 1.0

def test_bootstrap():
    ci = bootstrap_ci([0, 0, 1, 1, 0, 1, 0, 1], [0, 1, 1, 1, 0, 0, 0, 1], n_boot=50, seed=42)
    assert "acc_ci95" in ci and "f1_ci95" in ci
