"""Model definitions for quick benchmark."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


def build_configs(cfg: dict):
    m = cfg.get("models", {})
    out = []
    if m.get("dummy", {}).get("enabled", True):
        out.append(("dummy_most_frequent", __import__("sklearn.dummy", fromlist=["DummyClassifier"]).DummyClassifier(strategy="most_frequent"), False, {}))
    lr = m.get("logreg", {})
    if lr.get("enabled", True):
        for C in lr.get("C", [0.1, 1.0, 10.0]):
            out.append((f"logreg_C{C}", LogisticRegression(max_iter=int(lr.get("max_iter", 2000)), solver="lbfgs", C=float(C)), True, {"C": C}))
    rf = m.get("rf", {})
    if rf.get("enabled", True):
        for md in rf.get("max_depth", [None, 16]):
            for msl in rf.get("min_samples_leaf", [1, 3]):
                out.append((f"rf_md{md}_msl{msl}", RandomForestClassifier(n_estimators=int(rf.get("n_estimators", 200)), max_depth=md, min_samples_leaf=int(msl), random_state=42, n_jobs=4), False, {"max_depth": md, "min_samples_leaf": msl}))
    sv = m.get("linsvc", {})
    if sv.get("enabled", True):
        for C in sv.get("C", [0.1, 1.0, 10.0]):
            out.append((f"linsvc_C{C}", LinearSVC(C=float(C), dual="auto", max_iter=int(sv.get("max_iter", 5000))), True, {"C": C}))
    ts = m.get("tree_stump", {})
    if ts.get("enabled", True):
        for d in ts.get("max_depth", [1, 3]):
            out.append((f"tree_d{d}", DecisionTreeClassifier(max_depth=int(d), random_state=42), False, {"max_depth": d}))
    return out
