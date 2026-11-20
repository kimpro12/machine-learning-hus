"""Split utilities: dedup, random stratified, domain-disjoint. Deterministic."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .url_features import normalize_url, registrable_domain, row_id_for


def dedup_and_group(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    d = df.copy()
    d["url_orig"] = d["URL"].astype(str)
    d["normalized_url"] = d["url_orig"].map(normalize_url)
    d["group"] = [registrable_domain(u, n) for u, n in zip(d["url_orig"], d["normalized_url"])]
    # exact-row duplicates
    n_exact_dup = int(d.duplicated(keep="first").sum())
    # normalized URL dup stats
    vc = d["normalized_url"].value_counts()
    n_dup_norm_urls = int((vc > 1).sum())
    # label conflicts per normalized URL
    conflict_groups = d.groupby("normalized_url")["y"].nunique()
    conflict_urls = conflict_groups[conflict_groups > 1]
    quarantine_mask = d["normalized_url"].isin(set(conflict_urls.index))
    quarantine = d[quarantine_mask].copy()
    kept = d[~quarantine_mask].copy()
    # deterministic keep-first by sorted normalized_url? keep first occurrence order
    kept = kept.drop_duplicates(subset=["normalized_url"], keep="first").copy()
    # stable row_id
    kept["row_id"] = [row_id_for(n, str(i)) for i, n in enumerate(kept["normalized_url"])]
    stats = {
        "n_in": int(len(d)),
        "n_exact_dup_rows": n_exact_dup,
        "n_dup_norm_urls": n_dup_norm_urls,
        "n_conflict_urls": int(len(conflict_urls)),
        "n_quarantined_rows": int(len(quarantine)),
        "n_kept": int(len(kept)),
    }
    return kept, quarantine, stats


def random_split_ids(y: pd.Series, seed: int = 42, train=0.64, val=0.16, test=0.20):
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=val + test, stratify=y, random_state=seed)
    # tmp fraction val/(val+test)
    vfrac = val / (val + test)
    va, te = train_test_split(tmp, test_size=1 - vfrac, stratify=y.iloc[tmp] if hasattr(y, "iloc") else y[tmp], random_state=seed + 1)
    return {"train": tr.tolist(), "val": va.tolist(), "test": te.tolist()}


def domain_split_ids(groups: pd.Series, y: pd.Series, seed: int = 42, train=0.64, val=0.16, test=0.20):
    """Deterministic balanced group split: split unique groups stratified by majority label."""
    groups = groups.reset_index(drop=True)
    y = y.reset_index(drop=True) if hasattr(y, "reset_index") else y
    gdf = pd.DataFrame({"group": groups.values, "y": np.asarray(y)}).copy()
    # group majority label
    gm = gdf.groupby("group")["y"].agg(["mean", "size"]).reset_index()
    gm["glabel"] = (gm["mean"] >= 0.5).astype(int)
    uniq = gm["group"].to_numpy()
    glab = gm["glabel"].to_numpy()
    # stratify groups; fallback if single class or too few
    try:
        g_train, g_tmp = train_test_split(uniq, test_size=val + test, stratify=glab, random_state=seed)
    except ValueError:
        g_train, g_tmp = train_test_split(uniq, test_size=val + test, random_state=seed)
    # label tmp groups for second split via map (fast, no large .loc list)
    lab_map = dict(zip(gm["group"].tolist(), gm["glabel"].tolist()))
    tmp_lab = np.array([lab_map[g] for g in g_tmp])
    vfrac = val / (val + test)
    try:
        g_val, g_test = train_test_split(list(g_tmp), test_size=1 - vfrac, stratify=tmp_lab, random_state=seed + 1)
    except ValueError:
        g_val, g_test = train_test_split(list(g_tmp), test_size=1 - vfrac, random_state=seed + 1)
    g_train_s, g_val_s, g_test_s = set(g_train), set(g_val), set(g_test)
    assert not (g_train_s & g_val_s) and not (g_train_s & g_test_s) and not (g_val_s & g_test_s)
    # pandas isin uses hashtable -> fast for 200k groups
    train_mask = groups.isin(g_train_s).to_numpy()
    val_mask = groups.isin(g_val_s).to_numpy()
    idx = np.arange(len(groups))
    train_idx = idx[train_mask].tolist()
    val_idx = idx[val_mask].tolist()
    # test = remainder (avoids third isin pass and guarantees partition)
    test_mask = ~(train_mask | val_mask)
    test_idx = idx[test_mask].tolist()
    return {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
        "groups": {"n_train": len(g_train_s), "n_val": len(g_val_s), "n_test": len(g_test_s)},
    }


def split_hash(ids: dict) -> str:
    s = json.dumps({k: sorted(v)[:10] + [len(v)] for k, v in ids.items() if k in ("train", "val", "test")}, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:16]
