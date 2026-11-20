# PhishVN — Quick Results

Label convention (PhishVN): `label=1` phishing / `label=0` benign. Positive class = phishing (1) in all metrics.
Dataset: Mendeley DOI `10.17632/b97hxbxtpd.4` (v3.1.0 open tier), zip SHA-256 `308351e9…5d0ab` verified; all `MANIFEST.txt` file hashes OK (14/14).
Official group-aware temporal split preserved: train 36909 / val 7266 / test 8941. Test opened once, after val selection.

## 1. Audit snapshot
- 53116 rows: phishing 36706 / benign 16410. Verified gold+silver core: 18997.
- Exact URL dups: 0; normalized-URL dup groups: 0; registrable-domain overlap train/val/test: **0/0/0**.
- Collection artefact confirmed: `is_https` benign mean 0.1230 vs phish 0.0055 — **excluded from all modelling**.
- Modelling features (21, constant `PHISHVN_FEATURES`): url_len, dom_len, is_ip, tld_len, subdom_cnt, letter_cnt, digit_cnt, special_cnt, eq_cnt, qm_cnt, amp_cnt, dot_cnt, dash_cnt, under_cnt, letter_ratio, digit_ratio, spec_ratio, slash_cnt, entropy, path_len, query_len. IDs, source, tier, timestamps, scenario/impersonated_org, metadata excluded. See `feature_audit.csv`.
- Extractor validated 53114/53116 integer-exact; 2 residual release-quirk rows documented (see DECISION.md). Covered by `tests/test_features.py`.

## 2. Validation (official val, selection only)
FULL: best **logreg_C10** acc 0.8713, F1 0.9215, ROC 0.8970. See `quick_val_FULL.csv`.
GOLDSILVER (only 24 phishing in val (train 2076 / test 487)): best **rf_mdNone_msl3** acc 0.9666, F1 0.3291, ROC 0.9146. RF wins F1 but trails LinSVC on val ROC; ±2–3 TP moves F1 at n=24 — selection unstable by construction. See `quick_val_GOLDSILVER.csv`.
- Accuracy-paradox check (majority baseline `dummy_most_frequent`): FULL dummy acc 0.7852, F1 0.8797, macro-F1 0.4398 vs best macro-F1 0.7823; GOLDSILVER dummy acc 0.9849, F1 0.0000, macro-F1 0.4962 vs best macro-F1 0.6560. The always-benign model "wins" on accuracy but scores F1=0 on phishing — macro-F1 (equal class weights) exposes it. See §"Vì sao macro-F1" trong DECISION.md.

## 3. Official test (frozen, once)
- FULL logreg_C10 (train 36909 → test 8941): acc **0.9026** CI [0.89625,0.90865], F1 **0.9194**, ROC 0.9584, CM [TN3105 FP740 FN131 TP4965].
- GOLDSILVER rf_mdNone_msl3: acc **0.9114** CI [0.90250,0.91966], F1 **0.4467**, ROC 0.9247, CM [TN3793 FP52 FN332 TP155].
- See `classification_metrics.csv`.

## 4. Bias slices (test, frozen FULL)
- Source: worst **tranco 0.2973** (n=888); full table in `test_by_source_FULL.csv`.
- Tier: weakest **silver 0.6010** (n=1659); scenarios range other 0.8545 → ecommerce 0.9939 (n>=100).
- URL slices carry `tiny_n_flag` (n<30): has_query, has_ip flagged, not interpreted. See `test_by_*.csv`.

## 5. External (frozen, no retraining/tuning)
- Hassan 340k: FULL acc **0.4393** (gap **-46.32pp**), F1 0.5998, ROC 0.4298. GOLDSILVER acc 0.4800, F1 0.5392.
- JPCERT 2025-H1 phishing-only: FULL **recall 0.9222**; GOLDSILVER 0.5586 (`recall_phishonly`, `single_class=True`; accuracy left empty).
- Features recomputed from URL strings only; threshold 0.5 frozen. See `external_metrics.csv`.

## 6. PCA / clustering (after benchmark, exploratory)
- PCA (train-only fit, 21 feats): no-PCA logreg val acc 0.871/F1 0.921; var90 (8 PCs) 0.773/0.863. Reduction hurts — keep full features. See `pca_results.csv`, `pca_scree.png`.
- KMeans/GMM on PCA(10) subsample n=15000: k=2 ARI -0.0005; best ARI 0.2089 (gmm_diag k=4). No label-aligned clusters — expected, descriptive only. See `clustering_results.csv`.

## Files
- `outputs/{feature_audit.csv,classification_metrics.csv,external_metrics.csv,quick_val_*.csv,test_by_*.csv,pca_results.csv,clustering_results.csv,frozen_*.joblib}`
- `outputs/figures/` confusion, gap, source-bias, PCA scree, cluster scatter plots.
- PhiUSIIL negative benchmark archived in `archive/phiusiil/` (git tag `phiusiil-baseline`).
