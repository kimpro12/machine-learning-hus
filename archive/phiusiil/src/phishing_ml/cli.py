"""CLI: fetch/audit/quick/finalize/full/report/run. Single source of truth."""
from __future__ import annotations
import argparse
import hashlib
import json
import pickle
import shutil
import sys
import time
import zipfile
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import sklearn.base

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from phishing_ml.fetch import fetch_data
from phishing_ml.schema import validate_schema, recode_label, numeric_feature_columns, feature_sets
from phishing_ml.url_features import extract_url_features, STRICT_COLS, NO_CUES_COLS
from phishing_ml.split import dedup_and_group, random_split_ids, domain_split_ids, split_hash
from phishing_ml.audit import run_audit
from phishing_ml.quick import run_quick, _needs_scaling
from phishing_ml.evaluate import compute_metrics, scores_from_model, bootstrap_ci
from phishing_ml.models import build_configs
from phishing_ml.preprocess import get_xy, make_pipeline
from phishing_ml.robustness import plot_random_vs_domain, split_sensitivity
from phishing_ml.pca_experiment import run_pca
from phishing_ml.clustering import run_clustering
from phishing_ml.reporting import build_report, _ensure_missing_fig, _ensure_duplicate_fig, _ensure_corr_fig, _confusion_and_roc
from phishing_ml.decision import decide

CSV_NAME = "PhiUSIIL_Phishing_URL_Dataset.csv"
SPLIT_KEYS = ("train", "val", "test")
SPLIT_PAIRS = [("train", "val"), ("train", "test"), ("val", "test")]


def load_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}


def get_paths():
    return {
        "raw_dir": ROOT / "data" / "raw",
        "proc_dir": ROOT / "data" / "processed",
        "outputs": ROOT / "outputs",
        "audit": ROOT / "outputs" / "audit",
        "metrics": ROOT / "outputs" / "metrics",
        "figures": ROOT / "outputs" / "figures",
        "models": ROOT / "outputs" / "models",
        "manifests": ROOT / "outputs" / "manifests",
    }


def cmd_fetch(args):
    P = get_paths()
    P["manifests"].mkdir(parents=True, exist_ok=True)
    dest = fetch_data(P["raw_dir"], P["manifests"] / "data_manifest.json")
    print(f"FETCH OK -> {dest}")
    return 0


def ensure_data():
    """Locate the raw CSV (data/raw, raw_dl/, or inside a local ZIP) and copy it to data/raw."""
    P = get_paths()
    target = P["raw_dir"] / CSV_NAME
    csv = next((c for c in [target, ROOT / "raw_dl" / CSV_NAME] if c.is_file()), None)
    if csv is None:
        for z in [P["raw_dir"] / "phiusiil+phishing+url+dataset.zip", ROOT / "phiusiil.zip"]:
            if not z.is_file():
                continue
            with zipfile.ZipFile(z) as zz:
                name = next((n for n in zz.namelist() if n.lower().endswith(".csv")), None)
                if name:
                    P["raw_dir"].mkdir(parents=True, exist_ok=True)
                    with zz.open(name) as src, open(target, "wb") as out:
                        shutil.copyfileobj(src, out)
                    csv = target
                    break
    if csv is None:
        print(f"BLOCKED: no raw CSV/ZIP found. Place official CSV at data/raw/{CSV_NAME}", file=sys.stderr)
        return None
    if csv.resolve() != target.resolve():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(csv, target)
    return target


def build_splits_and_feature_sets(kept: pd.DataFrame, cfg: dict, proc_dir: Path) -> None:
    """Random + domain-disjoint splits (asserted leak-free) and feature-set lists -> data/processed."""
    seed = int(cfg.get("seed", 42))
    rs = cfg.get("random_split", {"train": 0.64, "val": 0.16, "test": 0.20})
    y = kept["y"]
    rids = random_split_ids(y, seed=seed, train=float(rs.get("train", 0.64)), val=float(rs.get("val", 0.16)), test=float(rs.get("test", 0.20)))
    dids = domain_split_ids(kept["group"], y, seed=seed)
    for ids, col, kind in [(rids, "normalized_url", "random split URL"), (dids, "group", "domain split group")]:
        for a, b in SPLIT_PAIRS:
            overlap = set(kept.iloc[ids[a]][col]) & set(kept.iloc[ids[b]][col])
            assert not overlap, f"{kind} overlap {a}-{b}"
    summary = {
        "random_hash": split_hash(rids),
        "domain_hash": split_hash(dids),
        "random_counts": {k: len(rids[k]) for k in SPLIT_KEYS},
        "domain_counts": {k: len(dids[k]) for k in SPLIT_KEYS},
    }
    (proc_dir / "splits.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with open(proc_dir / "splits.pkl", "wb") as f:
        pickle.dump({"random": rids, "domain": dids}, f)
    fsets = feature_sets(numeric_feature_columns(kept), STRICT_COLS, NO_CUES_COLS)
    (proc_dir / "feature_sets.json").write_text(json.dumps(fsets, indent=2), encoding="utf-8")


def load_or_build_processed(cfg):
    """Return cleaned dataframe; build parquet/splits/feature sets from raw CSV if missing."""
    P = get_paths()
    P["proc_dir"].mkdir(parents=True, exist_ok=True)
    pq = P["proc_dir"] / "clean.parquet"
    have_splits = (P["proc_dir"] / "splits.pkl").exists() and (P["proc_dir"] / "feature_sets.json").exists()
    if pq.exists():
        kept = pd.read_parquet(pq)
        if not have_splits:
            print("Processed parquet exists but splits/feature_sets missing — rebuilding splits only ...")
            build_splits_and_feature_sets(kept, cfg, P["proc_dir"])
        return kept
    csv = ensure_data()
    if csv is None:
        return None
    print(f"Loading raw CSV {csv} ...")
    df = pd.read_csv(csv, low_memory=False)
    print(f"Raw shape {df.shape}")
    info = validate_schema(df)
    if info.get("errors"):
        print(f"SCHEMA ERRORS: {info['errors']}", file=sys.stderr)
        return None
    df["y"] = recode_label(df["label"])
    print("Computing URL lexical features ...")  # row-wise; ~30-60s for 235k
    t0 = time.time()
    uf = pd.DataFrame([extract_url_features(u) for u in df["URL"].astype(str).tolist()], index=df.index)
    print(f"Lexical done in {time.time()-t0:.1f}s")
    for c in uf.columns:
        df[c] = uf[c].to_numpy()
    kept, quar, stats = dedup_and_group(df)
    print(f"Dedup stats: {stats}")
    kept.to_parquet(pq, index=False)
    quar.to_parquet(P["proc_dir"] / "quarantine.parquet", index=False)
    P["manifests"].mkdir(parents=True, exist_ok=True)
    (P["manifests"] / "dedup_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    build_splits_and_feature_sets(kept, cfg, P["proc_dir"])
    return kept


def load_splits(proc_dir: Path):
    with open(proc_dir / "splits.pkl", "rb") as f:
        return pickle.load(f)


def cmd_audit(args):
    cfg = load_yaml(ROOT / "configs" / "quick.yaml")
    P = get_paths()
    df = load_or_build_processed(cfg)
    if df is None:
        print("BLOCKED: no data for audit", file=sys.stderr)
        return 2
    splits = load_splits(P["proc_dir"])
    rids = splits["random"]
    # url feat frame
    uf = df[[c for c in STRICT_COLS if c in df.columns]].copy()
    audit_res = run_audit(df, uf, np.array(rids["train"]), np.array(rids["val"]), P["audit"])
    # extra figs
    _ensure_missing_fig(df, P["figures"])
    stats = json.loads((P["manifests"] / "dedup_stats.json").read_text()) if (P["manifests"] / "dedup_stats.json").exists() else {}
    _ensure_duplicate_fig(stats, P["figures"])
    _ensure_corr_fig(P["audit"], P["figures"])
    print(f"AUDIT OK severe_bias={audit_res.get('SEVERE_COLLECTION_BIAS')} proxy={audit_res.get('SUSPECT_LABEL_PROXY')}")
    return 0


def cmd_quick(args):
    cfg = load_yaml(Path(args.config) if args.config else ROOT / "configs" / "quick.yaml")
    P = get_paths()
    df = load_or_build_processed(cfg)
    if df is None:
        print("BLOCKED", file=sys.stderr)
        return 2
    splits = load_splits(P["proc_dir"])
    fsets = json.loads((P["proc_dir"] / "feature_sets.json").read_text())
    # filter to cfg list
    want = cfg.get("feature_sets", list(fsets.keys()))
    fsets = {k: v for k, v in fsets.items() if k in want}
    if args.dry_run:
        print(f"DRY-RUN quick: n={len(df)} feature_sets={list(fsets.keys())} splits random={ {k: len(v) for k,v in splits['random'].items() if k in ('train','val','test')}}")
        return 0
    subset = cfg.get("train_subset_n")
    qdf = run_quick(df, fsets, splits["random"], splits["domain"], cfg, P["metrics"], train_subset_n=subset)
    plot_random_vs_domain(P["metrics"], P["figures"])
    # update SUMMARY
    build_report(P["outputs"], None)
    print(f"QUICK OK rows={len(qdf)} -> {P['metrics'] / 'quick_validation.csv'}")
    print(qdf.sort_values("f1_phishing", ascending=False).head(10).to_string() if len(qdf) else "no rows")
    return 0


def cmd_finalize(args):
    cfg = load_yaml(Path(args.config) if args.config else ROOT / "configs" / "quick.yaml")
    P = get_paths()
    df = load_or_build_processed(cfg)
    if df is None:
        print("BLOCKED", file=sys.stderr)
        return 2
    splits = load_splits(P["proc_dir"])
    fsets = json.loads((P["proc_dir"] / "feature_sets.json").read_text())
    want = cfg.get("feature_sets", list(fsets.keys()))
    fsets = {k: v for k, v in fsets.items() if k in want}
    if args.dry_run:
        print(f"DRY-RUN finalize: feature_sets={list(fsets.keys())} (test NOT opened)")
        return 0
    # frozen plan BEFORE opening test
    frozen = {"config_sha": hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16], "feature_sets": {k: v for k, v in fsets.items()}, "seed": cfg.get("seed", 42), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        frozen["data_sha"] = json.loads((P["manifests"] / "data_manifest.json").read_text()).get("sha256", "")
    except Exception:
        frozen["data_sha"] = ""
    (P["manifests"] / "frozen_plan.json").write_text(json.dumps(frozen, indent=2, default=str), encoding="utf-8")
    # pick best model per feature set from quick_validation (train-fit, val-selected)
    qpath = P["metrics"] / "quick_validation.csv"
    if not qpath.exists():
        print("Run quick before finalize", file=sys.stderr)
        return 2
    q = pd.read_csv(qpath)
    ok = q[q["status"] == "OK"]
    model_map = {m[0]: m[1] for m in build_configs(cfg)}
    # finalize both random test and domain test
    for split_name, test_key in [("random", "random"), ("domain", "domain")]:
        sids = splits[split_name]
        tr, te = np.array(sids["train"]), np.array(sids["test"])
        rows = []
        for fs, feats in fsets.items():
            feats = [f for f in feats if f in df.columns]
            sub = ok[ok["feature_set"] == fs].sort_values("f1_phishing", ascending=False) if len(ok[ok["feature_set"] == fs]) else pd.DataFrame()
            if len(sub) == 0:
                continue
            mname = str(sub.iloc[0]["model"])
            model = model_map.get(mname)
            if model is None:
                continue
            model_fresh = sklearn.base.clone(model)
            pipe = make_pipeline(model_fresh, _needs_scaling(mname))
            Xtr, ytr = get_xy(df.iloc[tr], feats)
            Xte, yte = get_xy(df.iloc[te], feats)
            t0 = time.time()
            pipe.fit(Xtr, ytr)
            ft = time.time() - t0
            pred = pipe.predict(Xte)
            score = scores_from_model(pipe, Xte)
            m = compute_metrics(yte, pred, score)
            ci = bootstrap_ci(yte, pred, n_boot=int(cfg.get("bootstrap_n", 500)), seed=42)
            rows.append({"feature_set": fs, "model": mname, "split": split_name, "status": "OK", "n_train": len(Xtr), "n_test": len(Xte), "n_features": len(feats), "fit_s": round(ft, 2), **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items()}, "acc_ci95_lo": round(ci["acc_ci95"][0], 4), "acc_ci95_hi": round(ci["acc_ci95"][1], 4), "f1_ci95_lo": round(ci["f1_ci95"][0], 4), "f1_ci95_hi": round(ci["f1_ci95"][1], 4)})
        rdf = pd.DataFrame(rows)
        out = P["metrics"] / ("final_random_test.csv" if split_name == "random" else "final_group_test.csv")
        rdf.to_csv(out, index=False)
        print(f"FINALIZE {split_name} -> {out} rows={len(rdf)}")
        if len(rdf):
            print(rdf.sort_values("f1_phishing", ascending=False).to_string())
    # figures + decision inputs
    _confusion_and_roc(P["metrics"], P["figures"])
    # decision
    try:
        audit_res = json.loads((P["audit"] / "audit_summary.json").read_text())
    except Exception:
        audit_res = {}
    dec = decide(P["metrics"], P["manifests"], cfg, audit_res, P["outputs"] / "decision.json")
    build_report(P["outputs"], dec)
    # mark test evaluated
    (P["manifests"] / "test_evaluated.json").write_text(json.dumps({"evaluated": True, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))
    print(f"DECISION: {dec['status']} — {dec['decision_reason']}")
    return 0


def cmd_full(args):
    cfg = load_yaml(Path(args.config) if args.config else ROOT / "configs" / "full.yaml")
    P = get_paths()
    df = load_or_build_processed(cfg)
    if df is None:
        return 2
    splits = load_splits(P["proc_dir"])
    fsets = json.loads((P["proc_dir"] / "feature_sets.json").read_text())
    if args.dry_run:
        print(f"DRY-RUN full: n={len(df)}")
        return 0
    # E3 PCA
    try:
        print("E3 PCA ...")
        run_pca(df, fsets, splits["random"], cfg, P["metrics"])
    except Exception as e:
        print(f"PCA fail: {e}")
    # E5 clustering
    try:
        print("E5 clustering ...")
        run_clustering(df, fsets, splits["random"], cfg, P["metrics"])
    except Exception as e:
        print(f"Clustering fail: {e}")
    # E2 split sensitivity + E4 ablation is already covered by quick feature sets; add sensitivity
    try:
        print("E2 split sensitivity ...")
        split_sensitivity(df, fsets, cfg, P["metrics"])
    except Exception as e:
        print(f"Sensitivity fail: {e}")
    # refresh report
    try:
        dec = json.loads((P["outputs"] / "decision.json").read_text()) if (P["outputs"] / "decision.json").exists() else None
    except Exception:
        dec = None
    build_report(P["outputs"], dec)
    print("FULL OK")
    return 0


def cmd_report(args):
    P = get_paths()
    try:
        dec = json.loads((P["outputs"] / "decision.json").read_text())
    except Exception:
        dec = None
    # ensure figs
    try:
        _confusion_and_roc(P["metrics"], P["figures"])
        _ensure_corr_fig(P["audit"], P["figures"])
    except Exception:
        pass
    out = build_report(P["outputs"], dec)
    print(f"REPORT OK -> {out}")
    return 0


def cmd_run(args):
    # run = fetch + audit + quick + (robustness inside quick) + finalize + decision; --mode full adds E1-E7
    rc = cmd_fetch(args)
    if rc != 0:
        # try continue offline if data exists
        if ensure_data() is None:
            return rc
    rc = cmd_audit(args)
    if rc != 0:
        return rc
    rc = cmd_quick(args)
    if rc != 0:
        return rc
    rc = cmd_finalize(args)
    if rc != 0:
        return rc
    if getattr(args, "mode", "quick") == "full":
        rc = cmd_full(args)
    else:
        cmd_report(args)
    return rc


def main():
    ap = argparse.ArgumentParser(prog="phishing_ml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fetch")
    p.set_defaults(fn=cmd_fetch)
    p = sub.add_parser("audit")
    p.set_defaults(fn=cmd_audit)
    p = sub.add_parser("quick")
    p.add_argument("--config", default=str(ROOT / "configs" / "quick.yaml"))
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_quick)
    p = sub.add_parser("finalize")
    p.add_argument("--config", default=str(ROOT / "configs" / "quick.yaml"))
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_finalize)
    p = sub.add_parser("full")
    p.add_argument("--config", default=str(ROOT / "configs" / "full.yaml"))
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_full)
    p = sub.add_parser("report")
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("run")
    p.add_argument("--mode", default="quick", choices=["quick", "full"])
    p.add_argument("--config", default=str(ROOT / "configs" / "quick.yaml"))
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_run)
    args = ap.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
