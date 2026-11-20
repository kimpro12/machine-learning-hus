"""Collection-bias + data-quality audit (train-aware where needed)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from .schema import SUSPECT_DERIVED
from .url_features import STRICT_COLS


def run_audit(df, url_feat: pd.DataFrame, train_idx, val_idx, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res: dict = {}
    # 1. basics
    res["n"] = int(len(df))
    res["phishing_share_internal"] = float(df["y"].mean())
    res["n_cols"] = int(df.shape[1])
    # 2. label-conditional rates for collection cues
    # derive proxies from URL lexical + raw IsHTTPS if present
    # use url_feat columns if available else fallback
    merged = pd.DataFrame({"y": df["y"].to_numpy()})
    for c in ["is_https", "has_www", "path_present", "query_present"]:
        if c in url_feat.columns:
            merged[c] = url_feat[c].to_numpy()
    if "IsHTTPS" in df.columns:
        merged["IsHTTPS_raw"] = df["IsHTTPS"].to_numpy()
    # similarity flag
    if "URLSimilarityIndex" in df.columns:
        merged["sim_ge_999"] = (df["URLSimilarityIndex"].to_numpy() >= 99.9).astype(int)
    # per-class rates
    rates = {}
    for c in [x for x in merged.columns if x != "y"]:
        g = merged.groupby("y")[c].agg(["mean", "sum", "count"])
        rates[c] = {str(k): {"mean": float(v["mean"]), "num": int(v["sum"]), "den": int(v["count"])} for k, v in g.iterrows()}
    res["rates_by_label"] = rates
    pd.DataFrame(
        [{"feature": k, **{f"y{yy}_{m}": vv for yy, dd in d.items() for m, vv in dd.items()}} for k, d in rates.items()]
    ).to_csv(out_dir / "label_conditional_stats.csv", index=False)
    # severe collection bias flag
    flag = False
    try:
        lp = rates["path_present"]["0"]["mean"] if "path_present" in rates else None
        pp = rates["path_present"]["1"]["mean"] if "path_present" in rates else None
        if lp is not None and pp is not None and lp < 0.01 and pp > 0.10:
            flag = True
    except Exception:
        pass
    res["SEVERE_COLLECTION_BIAS"] = bool(flag)
    # 3. single-feature stumps (train-only fit, val eval)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in ("y", "label", "row_id")]
    # restrict to plausible numeric (exclude row_id etc). cap at all.
    stump_rows = []
    trX = df.iloc[train_idx]
    vaX = df.iloc[val_idx]
    for c in num_cols:
        try:
            x_tr = trX[[c]].fillna(trX[c].median())
            x_va = vaX[[c]].fillna(trX[c].median())
            if x_tr[c].nunique() <= 1:
                continue
            clf = DecisionTreeClassifier(max_depth=1, random_state=42)
            clf.fit(x_tr, trX["y"])
            pred = clf.predict(x_va)
            stump_rows.append({"feature": c, "val_acc": float(accuracy_score(vaX["y"], pred)), "val_bal_acc": float(balanced_accuracy_score(vaX["y"], pred)), "threshold": float(clf.tree_.threshold[0])})
        except Exception:
            continue
    stump = pd.DataFrame(stump_rows).sort_values("val_acc", ascending=False) if stump_rows else pd.DataFrame()
    if len(stump):
        stump.to_csv(out_dir / "single_feature_stumps.csv", index=False)
    res["top_stump"] = stump.head(15).to_dict(orient="records") if len(stump) else []
    res["SUSPECT_LABEL_PROXY"] = bool(len(stump) and float(stump.iloc[0]["val_acc"]) >= 0.99)
    if len(stump):
        res["top_stump_feature"] = str(stump.iloc[0]["feature"])
        res["top_stump_acc"] = float(stump.iloc[0]["val_acc"])
    # 4. correlation / MI on train only
    try:
        corrs = []
        ytr = trX["y"].to_numpy()
        for c in num_cols:
            x = trX[c].fillna(trX[c].median()).to_numpy()
            if np.std(x) == 0:
                continue
            r = float(np.corrcoef(x, ytr)[0, 1])
            corrs.append({"feature": c, "pearson_r": r, "abs_r": abs(r)})
        cordf = pd.DataFrame(corrs).sort_values("abs_r", ascending=False).head(15)
        cordf.to_csv(out_dir / "feature_correlation_top.csv", index=False)
        res["top_corr"] = cordf.to_dict(orient="records")
    except Exception as e:
        res["corr_error"] = str(e)
    # 5. provenance
    prov_rows = []
    for c in df.columns:
        if c in ("y", "label", "row_id", "normalized_url", "url_orig", "group"):
            kind = "target/id/meta"
        elif c in ("FILENAME", "URL", "Domain", "TLD", "Title"):
            kind = "identifier/text (must not feed sklearn directly)"
        elif c in SUSPECT_DERIVED:
            kind = "HIGH_RISK_DERIVED (corpus/label-dependent suspected)"
        elif c in STRICT_COLS:
            kind = "url_lexical_strict (recomputed offline)"
        elif c in ("IsHTTPS", "URLLength", "DomainLength", "IsDomainIP", "TLDLength", "NoOfSubDomain"):
            kind = "url_derived (plausibly extractable, verify)"
        elif c.startswith(("NoOf", "Has", "Is", "Bank", "Pay", "Crypto", "Line", "Largest", "Robots", "Spacial", "Letter", "Degit", "Obfus", "Char", "TLD")):
            kind = "webpage/html-derived or mixed (needs crawler; REVIEW_REQUIRED)"
        else:
            kind = "REVIEW_REQUIRED"
        prov_rows.append({"column": c, "provenance": kind})
    pd.DataFrame(prov_rows).to_csv(out_dir / "feature_provenance.csv", index=False)
    # figures
    try:
        _plot_class_dist(df, fig_dir)
        _plot_rates(rates, fig_dir)
        _plot_stump(stump.head(15) if len(stump) else stump, fig_dir)
        # url length hist needs URLLength or url_length
        _plot_url_length(df, url_feat, fig_dir)
    except Exception as e:
        res["fig_error"] = str(e)
    # data quality md
    dq = out_dir / "data_quality.md"
    dq.write_text(_dq_markdown(df, res), encoding="utf-8")
    (out_dir / "audit_summary.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    return res


def _dq_markdown(df, res) -> str:
    lines = ["# Data quality audit", ""]
    lines.append(f"- n={len(df)}, cols={df.shape[1]}, phishing_share(internal y=1)={df['y'].mean():.4f}")
    lines.append(f"- SEVERE_COLLECTION_BIAS={res.get('SEVERE_COLLECTION_BIAS')}, SUSPECT_LABEL_PROXY={res.get('SUSPECT_LABEL_PROXY')}")
    if res.get("top_stump_feature"):
        lines.append(f"- top single-feature stump: {res['top_stump_feature']} val_acc={res['top_stump_acc']:.4f}")
    lines.append("")
    lines.append("Label convention: original label=0 phishing / 1 legitimate; internal y=1 phishing / 0 legitimate.")
    lines.append("")
    lines.append("See label_conditional_stats.csv, single_feature_stumps.csv, feature_correlation_top.csv, feature_provenance.csv.")
    return "\n".join(lines)


def _plot_class_dist(df, fig_dir):

    vc = df["y"].value_counts().sort_index()
    plt.figure(figsize=(5, 4))
    plt.bar(["legitimate (y=0)", "phishing (y=1)"], [vc.get(0, 0), vc.get(1, 0)])
    plt.title(f"Class distribution (n={len(df)})")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(fig_dir / "class_distribution.png", dpi=150)
    plt.close()


def _plot_rates(rates, fig_dir):

    keys = [k for k in ["path_present", "query_present", "is_https", "has_www", "sim_ge_999", "IsHTTPS_raw"] if k in rates]
    if not keys:
        return
    legit = [rates[k]["0"]["mean"] for k in keys]
    phish = [rates[k]["1"]["mean"] for k in keys]
    x = range(len(keys))
    plt.figure(figsize=(8, 4))
    plt.bar([i - 0.2 for i in x], legit, width=0.4, label="legitimate")
    plt.bar([i + 0.2 for i in x], phish, width=0.4, label="phishing")
    plt.xticks(list(x), keys, rotation=20)
    plt.ylabel("rate within class")
    plt.title("Collection-cue rates by label (within-class denominators)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "path_https_www_by_label.png", dpi=150)
    plt.close()


def _plot_stump(stump, fig_dir):

    if stump is None or len(stump) == 0:
        return
    plt.figure(figsize=(8, 4))
    plt.barh(stump["feature"][::-1], stump["val_acc"][::-1])
    plt.xlabel("single-feature depth-1 val accuracy")
    plt.title("Top single-feature stumps (train-fit, val-eval)")
    plt.tight_layout()
    plt.savefig(fig_dir / "single_feature_stump.png", dpi=150)
    plt.close()


def _plot_url_length(df, url_feat, fig_dir):

    col = "URLLength" if "URLLength" in df.columns else ("url_length" if "url_length" in url_feat.columns else None)
    if col is None:
        return
    plt.figure(figsize=(7, 4))
    for yv, lab in [(0, "legitimate"), (1, "phishing")]:
        vals = (df if col in df.columns else url_feat)
        m = df["y"].to_numpy() == yv
        v = vals[col].to_numpy()[m]
        v = v[np.isfinite(v)]
        plt.hist(v, bins=60, alpha=0.5, label=lab, density=True)
    plt.xlabel(col)
    plt.title("URL length by label (density)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "url_length_by_label.png", dpi=150)
    plt.close()
