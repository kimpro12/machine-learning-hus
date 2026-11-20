# Dedup reconciliation — 235795 raw vs 235150 kept

- Raw rows (data/raw/PhiUSIIL_Phishing_URL_Dataset.csv): **235795**.
- Unique normalized URLs (lowercase scheme/host, drop default port, keep path/query order, drop fragment): **235150**.
- Removed: **645** rows. Kept: 235150 = 235795 − 645.
- Distinct normalized URLs occurring >1 (groups): **623** (`n_dup_norm_urls`).
- Surplus-row decomposition `sum(count−1)`: count=2: 605 groups → 605 rows; count=3: 15 groups → 30 rows; count=4: 2 groups → 6 rows; count=5: 1 group → 4 rows. Total 605+30+6+4 = **645**.
- Exact full-row duplicates (pandas duplicated on all columns): **425**. This is a subset of the 645 normalized dups; remaining 220 share normalized URL but differ in FILENAME/Title/numeric crawl fields, so they are not byte-identical rows.
- Label conflicts per normalized URL: **0**; quarantined rows: **0**. All duplicates were same-label, kept deterministically (first occurrence).
- Top duplicated normalized URLs are phishing deep links / IPFS gateways (e.g. pinata.cloud/ipfs/… ×5, panelweb-4cae2.web.app ×4), consistent with phishing URL reuse.

Why the report mentions 623: 623 counts *groups* (distinct URLs that are duplicated); 645 counts *rows removed*. Both are correct; they measure different things. `outputs/manifests/dedup_stats.json` stores both; `outputs/external/dedup_reconciliation.json` stores the decomposition.

Verification: see `outputs/external/dedup_reconciliation.json`; reproduce with `normalize_url` in `src/phishing_ml/url_features.py` + `split.dedup_and_group`.
