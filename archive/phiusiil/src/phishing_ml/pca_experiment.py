"""PCA experiments (E3)."""
from __future__ import annotations
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from .evaluate import compute_metrics, scores_from_model
from .preprocess import get_xy


def _build_pca_pipe(model, n_comp=None, variance=None):
    steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    if n_comp is not None or variance is not None:
        if variance is not None:
            steps.append(("pca", PCA(n_components=variance, random_state=42)))
        else:
            steps.append(("pca", PCA(n_components=n_comp, random_state=42)))
    steps.append(("clf", model))
    return Pipeline(steps)


def run_pca(df, feature_sets: dict, splits_random: dict, cfg: dict, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tr, va = np.array(splits_random["train"]), np.array(splits_random["val"])
    pca_cfg = cfg.get("pca", {})
    var_targets = pca_cfg.get("variance_targets", [0.90, 0.95, 0.99])
    fixed = pca_cfg.get("fixed_components", [5, 10, 20])
    # limit to 2 feature sets + 2 models for CPU sanity
    fs_list = [f for f in ["URL_LEXICAL_STRICT", "NO_SUSPECT_DERIVED"] if f in feature_sets]
    if not fs_list:
        fs_list = list(feature_sets.keys())[:2]
    models = [
        ("logreg_C1", LogisticRegression(max_iter=2000, C=1.0)),
        ("linsvc_C1", LinearSVC(C=1.0, dual="auto", max_iter=5000)),
    ]
    rows = []
    scree_info = {}
    for fs in fs_list:
        feats = [f for f in feature_sets[fs] if f in df.columns]
        Xtr, ytr = get_xy(df.iloc[tr], feats)
        Xva, yva = get_xy(df.iloc[va], feats)
        n_feat = len(feats)
        fixed_valid = [k for k in fixed if k is not None and k < n_feat]
        for mname, model in models:
            # no-PCA baseline
            for cond, kw in [("no-PCA", {})] + [(f"var{int(v*100)}", {"variance": float(v)}) for v in var_targets] + [(f"k{k}", {"n_comp": int(k)}) for k in fixed_valid]:
                try:
                    pipe = _build_pca_pipe(model, **kw) if kw else _build_pca_pipe(model)
                    t0 = time.time()
                    pipe.fit(Xtr, ytr)
                    ft = time.time() - t0
                    t1 = time.time()
                    pred = pipe.predict(Xva)
                    score = scores_from_model(pipe, Xva)
                    pt = time.time() - t1
                    m = compute_metrics(yva, pred, score)
                    n_comp = None
                    exp_var = None
                    if "pca" in pipe.named_steps:
                        pca = pipe.named_steps["pca"]
                        n_comp = int(pca.n_components_)
                        exp_var = float(np.sum(pca.explained_variance_ratio_))
                        # store full curve once per fs
                        if fs not in scree_info:
                            # fit a full PCA to get curve (train only)
                            from sklearn.impute import SimpleImputer as SI
                            from sklearn.preprocessing import StandardScaler as SS
                            Xi = SI(strategy="median").fit_transform(Xtr)
                            Xs = SS().fit_transform(Xi)
                            pf = PCA(random_state=42)
                            pf.fit(Xs)
                            scree_info[fs] = (pf.explained_variance_ratio_, np.cumsum(pf.explained_variance_ratio_))
                    rows.append({"feature_set": fs, "model": mname, "condition": cond, "n_components": n_comp, "explained_variance": exp_var, "n_features_orig": n_feat, "accuracy": round(m["accuracy"], 6), "f1_phishing": round(m["f1_phishing"], 6), "roc_auc": round(float(m.get("roc_auc", float("nan"))), 6), "fit_s": round(ft, 3), "predict_s": round(pt, 3), "status": "OK"})
                except Exception as e:
                    rows.append({"feature_set": fs, "model": mname, "condition": cond, "status": f"FAIL: {e}"})
    rdf = pd.DataFrame(rows)
    rdf.to_csv(out_dir / "pca_results.csv", index=False)
    # scree + tradeoff plots
    try:
        for fs, (evr, cum) in scree_info.items():
            plt.figure(figsize=(7, 4))
            plt.plot(np.arange(1, len(evr) + 1), evr, marker="o", label="individual")
            plt.plot(np.arange(1, len(cum) + 1), cum, marker="s", label="cumulative")
            plt.xlabel("PC index")
            plt.ylabel("explained variance ratio")
            plt.title(f"PCA scree ({fs}, n={len(feats)} feats, train-only fit)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_dir / "pca_scree.png", dpi=150)
            plt.close()
            break  # one scree plot is enough
        ok = rdf[rdf["status"] == "OK"]
        if len(ok):
            piv = ok.pivot_table(index="condition", columns="model", values="accuracy", aggfunc="max")
            piv.plot(kind="bar", figsize=(10, 4))
            plt.title("PCA accuracy tradeoff (validation)")
            plt.ylabel("accuracy")
            plt.tight_layout()
            plt.savefig(fig_dir / "pca_accuracy_tradeoff.png", dpi=150)
            plt.close()
    except Exception:
        pass
    return rdf
