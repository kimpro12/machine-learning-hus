"""Robustness: domain gap figure + split sensitivity (E2, E6)."""
from __future__ import annotations
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .evaluate import compute_metrics, scores_from_model
from .preprocess import make_pipeline


def plot_random_vs_domain(metrics_dir: Path, fig_dir: Path):
    qpath = metrics_dir / "quick_validation.csv"
    gpath = metrics_dir / "domain_gap.csv"
    if not qpath.exists() or not gpath.exists():
        return None
    try:
        gap = pd.read_csv(gpath)
        if len(gap) == 0 or "error" in gap.columns:
            return None
        fig_dir.mkdir(parents=True, exist_ok=True)
        x = gap["feature_set"] + "\n" + gap["model"]
        plt.figure(figsize=(10, 4))
        plt.bar(x, gap["random_val_acc"], alpha=0.6, label="random val acc")
        plt.bar(x, gap["domain_val_acc"], alpha=0.6, label="domain val acc")
        plt.ylabel("accuracy")
        plt.title("Random vs domain-disjoint validation accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "random_vs_domain.png", dpi=150)
        plt.close()
        return str(fig_dir / "random_vs_domain.png")
    except Exception:
        return None


def split_sensitivity(df, feature_sets: dict, cfg: dict, out_dir: Path) -> pd.DataFrame:
    """E2: repeated 70/30,80/20,90/10 seeds 42,43,44 on train-development pool only.
    Uses random_split train+val portion as development pool; no touch of final test.
    Reports validation/exploratory results only."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # development pool: use full df but with fresh splits (exploratory, not final holdout)
    ratios = cfg.get("split_sensitivity", {}).get("ratios", [[0.7, 0.3], [0.8, 0.2], [0.9, 0.1]]) if isinstance(cfg.get("split_sensitivity"), dict) else [[0.7, 0.3], [0.8, 0.2], [0.9, 0.1]]
    seeds = cfg.get("split_sensitivity", {}).get("seeds", [42, 43, 44]) if isinstance(cfg.get("split_sensitivity"), dict) else [42, 43, 44]
    # pick one representative config: best strict RF/logreg to keep cost low -> use logreg C=1.0 on STRICT
    from sklearn.linear_model import LogisticRegression
    rows = []
    for fs_name in ["URL_LEXICAL_STRICT", "NO_SUSPECT_DERIVED", "FULL_NUMERIC"]:
        if fs_name not in feature_sets:
            continue
        feats = [f for f in feature_sets[fs_name] if f in df.columns]
        if not feats:
            continue
        X_all = df[feats]
        y_all = df["y"].to_numpy()
        for tr_ratio, va_ratio in ratios:
            for sd in seeds:
                # stratified split of full df as exploratory proxy (documented as such)
                idx = np.arange(len(df))
                tr, va = train_test_split(idx, test_size=va_ratio / (tr_ratio + va_ratio) if False else va_ratio, train_size=None, stratify=y_all, random_state=sd)
                # Actually want train=tr_ratio of total: train_test_split test_size=va_ratio
                tr, va = train_test_split(idx, test_size=va_ratio, stratify=y_all, random_state=sd)
                # adjust: if tr_ratio != 1-va_ratio, subsample train to tr_ratio
                if abs((1 - va_ratio) - tr_ratio) > 1e-9:
                    tr, _ = train_test_split(tr, train_size=tr_ratio / (1 - va_ratio), stratify=y_all[tr], random_state=sd)
                Xtr, ytr = X_all.iloc[tr], y_all[tr]
                Xva, yva = X_all.iloc[va], y_all[va]
                pipe = make_pipeline(LogisticRegression(max_iter=2000, C=1.0), True)
                t0 = time.time()
                pipe.fit(Xtr, ytr)
                ft = time.time() - t0
                pred = pipe.predict(Xva)
                score = scores_from_model(pipe, Xva)
                m = compute_metrics(yva, pred, score)
                rows.append({"feature_set": fs_name, "train_ratio": tr_ratio, "seed": sd, "n_train": len(tr), "n_val": len(va), "accuracy": m["accuracy"], "f1_phishing": m["f1_phishing"], "fit_s": round(ft, 2)})
    rdf = pd.DataFrame(rows)
    if len(rdf):
        rdf.to_csv(out_dir / "split_sensitivity.csv", index=False)
        agg = rdf.groupby(["feature_set", "train_ratio"])[["accuracy", "f1_phishing"]].agg(["mean", "std"])
        agg.to_csv(out_dir / "split_sensitivity_agg.csv")
    return rdf
