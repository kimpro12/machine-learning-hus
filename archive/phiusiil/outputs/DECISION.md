# PhiUSIIL Decision — GO / NO-GO (with independent external validation)

Label convention: original label=0 phishing / 1 legitimate; internal y=1 phishing / 0 legitimate.

(1) Accuracy best (random internal)? Accuracy=1.0000, F1_phishing=1.0000 (FULL_NUMERIC/rf_mdNone_msl1) [see metrics/final_random_test.csv; exact CI95 [0.99992,1.0], not [1.0,1.0]]
(2) F1 phishing (random best)? Accuracy=1.0000, F1_phishing=1.0000 (FULL_NUMERIC/rf_mdNone_msl1) [see metrics/final_random_test.csv]
(3) Strict URL-only (internal)? Accuracy=0.9981, F1_phishing=0.9978 (URL_LEXICAL_STRICT/rf_mdNone_msl3, n=47030, TN26968 FP2 FN88 TP19972, ROC-AUC 0.9986) [see metrics/final_random_test.csv]
(4) Domain-disjoint / external? Domain STRICT Accuracy=0.9978 F1=0.9972 [see metrics/final_group_test.csv]; Hassan external STRICT Accuracy=0.5001 F1=0.6667 ROC-AUC=0.5798 (TN30 FP169970 FN1 TP169999) [see external/external_vs_internal_metrics.csv] — gap −49.80pp; JPCERT 2025 phishing recall 0.9984 (n=14775)
(5) GIỮ hay ĐỔI và tại sao? KEEP_AS_BIAS_STUDY: internal 99–100% collapses to chance (50.01%) on 340k independent URLs; legit-with-path accuracy 0.0% (0/165085). Keep ONLY as leakage/bias case study; CHANGE_DATASET for any deployment claim.

> **BENCHMARK_NOT_REAL_WORLD_VALIDATED** — severe collection bias + direct leakage confirmed and now externally reproduced. Do not claim real-world performance even though internal accuracy ~100%.

- Status: `KEEP_AS_BIAS_STUDY` | academic_viability=True (techniques runnable) | real_world_validity=False
- Recommendation: Retain PhiUSIIL only to teach leakage/ablation/domain-shift (RQ2–RQ4) with reframed RQs; CHANGE_DATASET for real phishing detection.
- Dataset hash (PhiUSIIL): `a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6`
- Single-feature proxy: URLSimilarityIndex stump val_acc=0.9968 (threshold 98.44); legit ALL 100.0 vs phish 0.78% >=99.9.

## 1. Dedup reconciliation (235795 vs 235150)
- 623 = distinct normalized URLs occurring >1 (groups); 645 = surplus rows removed = 605×1+15×2+2×3+1×4.
- Kept 235150 = 235795−645. Exact full-row dups 425 (subset); label conflicts 0, quarantine 0.
- See `outputs/audit/dedup_reconciliation.md`, `outputs/external/dedup_reconciliation.json`.

## 2. Leakage vs bias audit
- Direct leakage (needs labels/corpus/HTML, not URL-only): URLSimilarityIndex (constant 100 for legit), TLDLegitimateProb (P(TLD|legit)), URLCharProb (legit-corpus probs), Domain/URLTitleMatchScore (fetched HTML Title).
- Collection bias (URL-computable but sampled): legit 100% https/100% www/0% path/0% query vs phish 48.5% https/41.6% www/27.0% path/6.0% query.
- See `outputs/audit/suspicious_features.md`, `outputs/external/suspicious_feature_summary.csv`.

## 3. External validation (frozen models, no tuning on external labels)
- Primary: Hassan-Albattra `balanced_urls.csv` (340k balanced; phishing PhishTank/OpenPhish, legit Alexa/CommonCrawl Jan–Dec 2023; MIT; sha 8a864a7fb867cce6; overlap with PhiUSIIL 10 URLs). Features recomputed from URL strings only via `extract_url_features` (STRICT 28, NOCUES 18). Thresholds frozen (0.5).
- Secondary: JPCERTCC `phishurl-list` 2025-01..06 (14775 phishing, CERT-verified, overlap 0) — recall-only.
- STRICT RF (PhiUSIIL-train, min_samples_leaf=3): Hassan acc 0.500085 CI95 [0.49840,0.50177], prec 0.50004, rec 0.99999, F1 0.66670, ROC-AUC 0.57982, PR-AUC 0.54398.
- NOCUES RF (leaf=1): Hassan acc 0.499694 CI95 [0.49801,0.50138], prec 0.49985, rec 0.99921, F1 0.66635, ROC-AUC 0.47811.
- Gaps vs internal: STRICT −49.80pp (0.998086→0.500085); NOCUES −49.10pp (0.990687→0.499694).
- JPCERT: STRICT recall 0.9984 CI [0.9976,0.9990] (TP14751 FN24); NOCUES recall 0.9952 CI [0.9939,0.9962]. High phishing recall is consistent with “predict phishing for any path”, not with discrimination.
- See `outputs/external/external_vs_internal_metrics.csv`, `outputs/external/external_manifest.json`, `outputs/figures/external_confusion_hassan_strict.png`, `internal_vs_external_gap.png`.

## 4. Counter-bias subgroups (exact CIs, frozen STRICT RF)
- Internal random test: legit_with_path n=0 (EMPTY — never tested); legit_no_path n=26970 acc 0.99993 CI [0.99973,0.99999]; phish_with_https n=9727 acc 0.99095 CI [0.98887,0.99274] (all 88 errors here); phish_no_https n=10333 acc 1.0 CI [0.99964,1.0]; phish_with_www n=8294 acc 0.98939 CI [0.98694,0.99148].
- Hassan: legit_with_path n=165085 acc 0.0 CI [0.0,0.000022] (0 correct); legit_no_path n=4915 acc 0.00610 CI [0.00412,0.00870] (30 correct); phish_with_https n=3684 acc 0.99973 CI [0.99849,0.99999]; phish_with_www n=47259 acc 0.99998 CI [0.99988,1.0]; legit_with_https n=113215 acc 0.000265; legit_with_www n=122399 acc 0.000245.
- NOCUES shows the same pattern (hassan legit_with_path 1/165085 correct).
- See `outputs/external/hassan_subgroups_STRICT.csv`, `internal_test_subgroups_STRICT.csv`, `outputs/figures/external_subgroup_slices.png`. Single-class slices report ROC-AUC as nan (undefined) — correctly, not as 1.0.

## 5. CI correction for perfect accuracy
- Old `final_random_test.csv` percentile-bootstrap CIs of [1.0,1.0] for FULL 1.0 are degenerate (all resamples perfect). Exact Clopper-Pearson 95%: FULL (k=47030/n=47030) → [0.99992,1.0] (Rule of Three ≈1−3/n). Code fixed in `src/phishing_ml/evaluate.py:clopper_pearson_ci`; all external/subgroup tables use exact CIs. Frozen internal CSV preserved for provenance; use exact interval above for claims.

## Evidence files
- outputs/metrics/final_random_test.csv, final_group_test.csv, domain_gap.csv
- outputs/external/external_vs_internal_metrics.csv, hassan_subgroups_STRICT.csv, internal_test_subgroups_STRICT.csv, external_manifest.json, dedup_reconciliation.json
- outputs/audit/dedup_reconciliation.md, suspicious_features.md, label_conditional_stats.csv, single_feature_stumps.csv
- outputs/figures/external_confusion_hassan_strict.png, internal_vs_external_gap.png, external_subgroup_slices.png, random_vs_domain.png

## Limitations
- FULL_NUMERIC is optimistic (derived/crawled, corpus-dependent); STRICT still encodes homepage-vs-path bias.
- Hassan mirror is low-profile GitHub (claims reputable upstreams; independence verified by 10-URL overlap + 97%-paths legit distribution shift); JPCERTCC corroborates phishing side with high-trust CERT feed.
- Internal legit-with-path slice empty, so internal testing never covers the dominant real-world case.
- No threshold tuning on external (by design); test holdout opened once post-freeze.

## Next steps
- KEEP_AS_BIAS_STUDY: reframe report around leakage/ablation/counter-bias slices (RQ2–RQ4); keep BENCHMARK_NOT_REAL_WORLD_VALIDATED banner.
- CHANGE_DATASET for deployment: time-split, vendor-diverse URLs with legitimate deep links + independent validation.
