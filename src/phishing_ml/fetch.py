"""Fetch / ingest official PhiUSIIL data."""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

UCI_ID = 967
UCI_URL = "https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset"
ZIP_URL = "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"
CSV_NAME = "PhiUSIIL_Phishing_URL_Dataset.csv"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _extract_csv_from_zip(zpath: Path, dest: Path) -> Path:
    import zipfile

    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        csv_member = next((n for n in names if n.lower().endswith(".csv")), names[0])
        with z.open(csv_member) as src, open(dest, "wb") as out:
            shutil.copyfileobj(src, out)
    return dest


def _download_zip_streaming(zpath: Path) -> Path:
    import requests

    with requests.get(ZIP_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(zpath, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    return zpath


def fetch_data(raw_dir: Path, manifest_path: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / CSV_NAME
    if dest.is_file():
        src_csv = dest
    else:
        # search common local candidates (absolute, relative to CWD and to project root)
        root = Path(__file__).resolve().parents[2]
        candidates_csv = [
            dest,
            Path.cwd() / "raw_dl" / CSV_NAME,
            root / "raw_dl" / CSV_NAME,
            Path.cwd() / CSV_NAME,
        ]
        candidates_zip = [
            raw_dir / "phiusiil+phishing+url+dataset.zip",
            Path.cwd() / "phiusiil.zip",
            root / "phiusiil.zip",
        ]
        src_csv = next((c for c in candidates_csv if c.is_file()), None)
        if src_csv is None:
            # try extracting from any local zip
            zlocal = next((z for z in candidates_zip if z.is_file()), None)
            if zlocal is not None:
                _extract_csv_from_zip(zlocal, dest)
                src_csv = dest
        if src_csv is None:
            # try ucimlrepo (metadata); may fail on large CSV -> fallback to streaming ZIP
            try:
                from ucimlrepo import fetch_ucirepo

                ds = fetch_ucirepo(id=UCI_ID)
                df_u = pd.concat([ds.data.features, ds.data.targets], axis=1)
                if "URL" not in df_u.columns or "label" not in df_u.columns:
                    raise RuntimeError("ucimlrepo frame missing URL/label, falling back to ZIP")
                df_u.to_csv(dest, index=False)
                src_csv = dest
            except Exception as e:
                try:
                    zpath = raw_dir / "phiusiil+phishing+url+dataset.zip"
                    if not zpath.exists():
                        _download_zip_streaming(zpath)
                    _extract_csv_from_zip(zpath, dest)
                    src_csv = dest
                except Exception as e2:
                    raise RuntimeError(
                        f"Could not obtain dataset. Place official CSV at {dest}. "
                        f"ucimlrepo error: {e}; zip error: {e2}"
                    )
    if Path(src_csv) != dest:
        shutil.copyfile(src_csv, dest)
    df = pd.read_csv(dest, nrows=5)
    manifest = {
        "source_url": UCI_URL,
        "zip_url": ZIP_URL,
        "csv": str(dest),
        "sha256": sha256_file(dest),
        "ingest_timestamp": datetime.now(timezone.utc).isoformat(),
        "license": "CC BY 4.0",
        "columns_sample": list(df.columns),
    }
    with open(dest, "rb") as f:
        n = sum(1 for _ in f) - 1
    manifest["n_rows"] = n
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return dest
