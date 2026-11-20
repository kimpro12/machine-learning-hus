"""Download the independent external evaluation sets (used for scoring only, never tuning).

- Hassan-Albattra balanced_urls.csv: 340k URLs, balanced phishing/legitimate, 2023, MIT.
- JPCERTCC phishurl-list 2025-01..06: CERT-verified phishing URLs (phishing-only).
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

HASSAN_URL = "https://raw.githubusercontent.com/Hassan-Albattra/URL_Phishing_Detection_Dataset/main/balanced_urls.csv"
JPCERT_URL = "https://raw.githubusercontent.com/JPCERTCC/phishurl-list/main/{year}/{month}.csv"
JPCERT_MONTHS = ("202501", "202502", "202503", "202504", "202505", "202506")


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> Path:
    """Download url -> dest (skips if dest already exists and is non-empty)."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip (exists) {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "phishing-ml-course-project"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)
    print(f"  ok {dest.name} ({dest.stat().st_size / 1e6:.1f} MB, sha256 {_sha256(dest)[:16]})")
    return dest


def fetch_external(external_dir: Path) -> int:
    print(f"Downloading external evaluation sets -> {external_dir}")
    try:
        _download(HASSAN_URL, external_dir / "hassan_balanced_urls.csv")
        for m in JPCERT_MONTHS:
            _download(JPCERT_URL.format(year=m[:4], month=m), external_dir / "jpcert" / f"jpcert-{m}.csv")
    except OSError as e:
        print(f"DOWNLOAD FAILED: {e}")
        return 1
    return 0
