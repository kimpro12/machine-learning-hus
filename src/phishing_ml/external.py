"""Independent external evaluation using URL strings only (no test-label tuning).

Primary: Hassan-Albattra balanced_urls.csv (340k, PhishTank/OpenPhish + Alexa/CommonCrawl, Jan-Dec 2023).
Secondary (phishing-only): JPCERTCC phishurl-list 2025 monthly CSVs (CERT-verified).

All features are recomputed from URL strings with phishing_ml.url_features.extract_url_features.
Frozen PhiUSIIL-trained pipelines are applied as-is; thresholds are NOT retuned on external labels.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import compute_metrics, scores_from_model, clopper_pearson_ci
from .url_features import extract_url_features


HASSAN_URL = "https://raw.githubusercontent.com/Hassan-Albattra/URL_Phishing_Detection_Dataset/main/balanced_urls.csv"
HASSAN_PROVENANCE = (
    "Hassan-Albattra/URL_Phishing_Detection_Dataset balanced_urls.csv; "
    "phishing from PhishTank + OpenPhish, legitimate from Alexa Top Sites + Common Crawl, "
    "collected Jan-Dec 2023; MIT; label column in {legitimate,phishing}, url column."
)
JPCERT_PROVENANCE = (
    "JPCERTCC/phishurl-list monthly CSVs (CERT-verified phishing URLs, 2025); "
    "columns: date,URL,description; phishing-only, so recall-only evaluation."
)


def sha256_file(p: Path) -> str:
    import hashlib as hl

    h = hl.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def download_hassan(dest: Path) -> Path:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    with requests.get(HASSAN_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    return dest


def load_hassan(csv_path: Path, max_n: int | None = None, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    # normalize columns
    cols = {c.lower(): c for c in df.columns}
    url_col = cols.get("url", df.columns[1])
    lab_col = cols.get("label", df.columns[0])
    out = pd.DataFrame({"ext_url": df[url_col].astype(str), "ext_label_raw": df[lab_col].astype(str)})
    out["y"] = out["ext_label_raw"].str.lower().map({"phishing": 1, "legitimate": 0, "bad": 1, "good": 0, "1": 1, "0": 0})
    out = out.dropna(subset=["y"])
    out["y"] = out["y"].astype(int)
    out = out[out["ext_url"].str.len() > 0].reset_index(drop=True)
    if max_n is not None and len(out) > max_n:
        out = out.sample(n=max_n, random_state=seed).reset_index(drop=True)
    return out


def download_jpcert(dest_dir: Path, months: list[str] | None = None) -> list[Path]:
    import requests

    dest_dir.mkdir(parents=True, exist_ok=True)
    if months is None:
        months = ["202501", "202502", "202503", "202504", "202505", "202506"]
    paths = []
    for m in months:
        url = f"https://raw.githubusercontent.com/JPCERTCC/phishurl-list/main/{m[:4]}/{m}.csv"
        dest = dest_dir / f"jpcert-{m}.csv"
        if dest.exists() and dest.stat().st_size > 0:
            paths.append(dest)
            continue
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                if r.status_code != 200:
                    continue
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
            paths.append(dest)
        except Exception:
            continue
    return paths


def load_jpcert(paths: list[Path], max_n: int | None = None, seed: int = 42) -> pd.DataFrame:
    frames = []
    for p in paths:
        try:
            d = pd.read_csv(p, low_memory=False)
            # find URL column (usually 'URL' or second col)
            ucol = None
            for c in d.columns:
                if c.strip().lower() == "url":
                    ucol = c
                    break
            if ucol is None:
                ucol = d.columns[1] if len(d.columns) > 1 else d.columns[0]
            frames.append(pd.DataFrame({"ext_url": d[ucol].astype(str)}))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["ext_url", "y"])
    out = pd.concat(frames, ignore_index=True)
    out["y"] = 1  # phishing-only
    out = out[out["ext_url"].str.len() > 0].reset_index(drop=True)
    out = out.drop_duplicates(subset=["ext_url"]).reset_index(drop=True)
    if max_n is not None and len(out) > max_n:
        out = out.sample(n=max_n, random_state=seed).reset_index(drop=True)
    return out


def add_lexical(df: pd.DataFrame, url_col: str = "ext_url") -> pd.DataFrame:
    feats = [extract_url_features(u) for u in df[url_col].astype(str).tolist()]
    uf = pd.DataFrame(feats, index=df.index)
    for c in uf.columns:
        df[c] = uf[c].to_numpy()
    return df


def wilson_ci(k: int, n: int, alpha: float = 0.05):
    return clopper_pearson_ci(k, n, alpha)


def eval_frozen(pipe, df: pd.DataFrame, feats: list[str], label: str = "external") -> dict:
    import time as _t

    feats = [f for f in feats if f in df.columns]
    X = df[feats]
    y = df["y"].to_numpy().astype(int)
    t0 = _t.time()
    pred = pipe.predict(X)
    pt = _t.time() - t0
    try:
        score = scores_from_model(pipe, X)
    except Exception:
        score = pred.astype(float)
    m = compute_metrics(y, pred, score)
    k = int((y == pred).sum())
    m["acc_exact_ci95_lo"], m["acc_exact_ci95_hi"] = clopper_pearson_ci(k, len(y))
    m["n_features"] = len(feats)
    m["predict_s"] = round(pt, 2)
    m["eval_label"] = label
    return m


def subgroup_eval(y_true, y_pred, y_score, mask, name: str) -> dict:
    yt = np.asarray(y_true)[np.asarray(mask)]
    yp = np.asarray(y_pred)[np.asarray(mask)]
    ys = np.asarray(y_score)[np.asarray(mask)] if y_score is not None else None
    if len(yt) == 0:
        return {"subgroup": name, "n": 0, "note": "empty slice"}
    m = compute_metrics(yt, yp, ys)
    k = int((yt == yp).sum())
    lo, hi = clopper_pearson_ci(k, len(yt))
    m.update({"subgroup": name, "n": int(len(yt)), "acc_exact_ci95_lo": lo, "acc_exact_ci95_hi": hi})
    return m
