"""Quick benchmark: train on random split, validate on val; then domain-disjoint val gap."""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluate import compute_metrics, scores_from_model, tune_threshold
from .models import build_configs
from .preprocess import get_xy, make_pipeline

NEEDS_SCALE = {"logreg": True, "linsvc": True, "dummy": False, "rf": False, "tree": False}


def _needs_scaling(name: str) -> bool:
    for k, v in NEEDS_SCALE.items():
        if name.startswith(k):
            return v
    return False


def run_quick(df, feature_sets: dict, splits_random: dict, splits_domain: dict, cfg: dict, out_dir: Path, train_subset_n=None) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    model_cfgs = build_configs(cfg)
    rows = []
    # Q1: random train/val
    tr, va = np.array(splits_random["train"]), np.array(splits_random["val"])
    if train_subset_n:
        rng = np.random.default_rng(cfg.get("seed", 42))
        # stratified subsample of train
        ytr = df.iloc[tr]["y"].to_numpy()
        keep = []
        for cls in [0, 1]:
            idx = tr[ytr == cls]
            n_cls = max(1, int(round(len(idx) * train_subset_n / len(tr))))
            keep.append(rng.choice(idx, size=min(len(idx), n_cls), replace=False))
        tr = np.concatenate(keep)
    for fs_name, feats in feature_sets.items():
        feats = [f for f in feats if f in df.columns]
        if not feats:
            continue
        Xtr, ytr = get_xy(df.iloc[tr], feats)
        Xva, yva = get_xy(df.iloc[va], feats)
        for mname, model, _scale_ignored, params in model_cfgs:
            pipe = make_pipeline(model, _needs_scaling(mname))
            t0 = time.time()
            try:
                pipe.fit(Xtr, ytr)
            except Exception as e:
                rows.append({"feature_set": fs_name, "model": mname, "params": json.dumps(params), "status": f"FIT_FAIL: {e}", "n_train": len(Xtr), "n_val": len(Xva), "n_features": len(feats)})
                continue
            fit_t = time.time() - t0
            t1 = time.time()
            try:
                pred = pipe.predict(Xva)
                score = scores_from_model(pipe, Xva)
            except Exception as e:
                rows.append({"feature_set": fs_name, "model": mname, "params": json.dumps(params), "status": f"PRED_FAIL: {e}", "n_train": len(Xtr), "n_val": len(Xva), "n_features": len(feats)})
                continue
            pred_t = time.time() - t1
            m = compute_metrics(yva, pred, score)
            thr = tune_threshold(yva, score, rule=cfg.get("val_threshold_rule", "f1"))
            rows.append({
                "feature_set": fs_name, "model": mname, "params": json.dumps(params), "status": "OK",
                "n_train": len(Xtr), "n_val": len(Xva), "n_features": len(feats),
                "fit_s": round(fit_t, 3), "predict_s": round(pred_t, 3),
                **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items()},
                "tuned_threshold": thr["selected_threshold"],
                "tuned_f1": thr["selected"]["f1_phishing"],
                "tuned_acc": thr["selected"]["accuracy"],
            })
    qdf = pd.DataFrame(rows)
    qdf.to_csv(out_dir / "quick_validation.csv", index=False)
    qdf.to_json(out_dir / "quick_validation.json", orient="records", indent=2)
    # figures
    try:
        ok = qdf[qdf["status"] == "OK"]
        if len(ok):
            for metric, fn in [("accuracy", "quick_accuracy_by_feature_set.png"), ("f1_phishing", "quick_f1_by_feature_set.png")]:
                piv = ok.pivot_table(index="feature_set", columns="model", values=metric, aggfunc="max")
                piv.plot(kind="bar", figsize=(10, 4))
                plt.title(f"Quick validation {metric} by feature set (random split)")
                plt.ylabel(metric)
                plt.tight_layout()
                plt.savefig(fig_dir / fn, dpi=150)
                plt.close()
    except Exception:
        pass
    # Q2: domain-disjoint val gap for best 1-2 models (pick best RF/logreg per FULL and STRICT by val F1)
    try:
        gap_rows = domain_gap(df, feature_sets, splits_domain, cfg, out_dir, qdf)
    except Exception as e:
        gap_rows = [{"error": str(e)}]
        pd.DataFrame(gap_rows).to_csv(out_dir / "domain_gap.csv", index=False)
    return qdf


def domain_gap(df, feature_sets, splits_domain, cfg, out_dir, qdf) -> list:
    ok = qdf[qdf["status"] == "OK"]
    picks = []
    for fs in ["FULL_NUMERIC", "URL_LEXICAL_STRICT"]:
        sub = ok[ok["feature_set"] == fs].sort_values("f1_phishing", ascending=False)
        if len(sub):
            picks.append((fs, sub.iloc[0]["model"]))
    if not picks:
        return []
    model_map = {m[0]: m[1] for m in build_configs(cfg)}
    tr, va = np.array(splits_domain["train"]), np.array(splits_domain["val"])
    rows = []
    for fs, mname in picks:
        feats = [f for f in feature_sets[fs] if f in df.columns]
        Xtr, ytr = get_xy(df.iloc[tr], feats)
        Xva, yva = get_xy(df.iloc[va], feats)
        pipe = make_pipeline(model_map[mname], _needs_scaling(mname))
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xva)
        score = scores_from_model(pipe, Xva)
        m = compute_metrics(yva, pred, score)
        # random val reference
        ref = ok[(ok["feature_set"] == fs) & (ok["model"] == mname)].iloc[0]
        rows.append({
            "feature_set": fs, "model": mname,
            "random_val_acc": float(ref["accuracy"]), "random_val_f1": float(ref["f1_phishing"]),
            "domain_val_acc": float(m["accuracy"]), "domain_val_f1": float(m["f1_phishing"]),
            "gap_acc_pp": float((ref["accuracy"] - m["accuracy"]) * 100),
            "gap_f1_pp": float((ref["f1_phishing"] - m["f1_phishing"]) * 100),
            "domain_val_n": len(Xva), "domain_val_phish_share": float(np.mean(yva)),
        })
    pd.DataFrame(rows).to_csv(out_dir / "domain_gap.csv", index=False)
    return rows
