from phishing_ml.evaluate import compute_metrics

def test_confusion_four_cases():
    # TN, FP, FN, TP each 1: y_true=[0,0,1,1], y_pred=[0,1,0,1]
    m = compute_metrics([0, 0, 1, 1], [0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert (m["TN"], m["FP"], m["FN"], m["TP"]) == (1, 1, 1, 1)
    assert abs(m["accuracy"] - 0.5) < 1e-9
    assert abs(m["precision_phishing"] - 0.5) < 1e-9
    assert abs(m["recall_phishing"] - 0.5) < 1e-9
    assert m["roc_auc"] > 0.4
