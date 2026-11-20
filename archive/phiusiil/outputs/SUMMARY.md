# PhiUSIIL Phishing Detection — Summary

Label convention: original label=0 phishing / 1 legitimate; internal y=1 phishing / 0 legitimate.

- Quick validation runs: 52/52 OK.
  - FULL_NUMERIC: best rf_mdNone_msl1 acc=1.0000 f1=1.0000 roc_auc=1.0000
  - NO_SUSPECT_DERIVED: best rf_mdNone_msl1 acc=1.0000 f1=1.0000 roc_auc=1.0000
  - STRICT_NO_COLLECTION_CUES: best rf_mdNone_msl1 acc=0.9899 f1=0.9881 roc_auc=0.9968
  - URL_LEXICAL_STRICT: best rf_mdNone_msl3 acc=0.9976 f1=0.9972 roc_auc=0.9983

Preliminary: see outputs/metrics/quick_validation.csv and figures quick_accuracy_by_feature_set.png / quick_f1_by_feature_set.png.
- final_random_test.csv: 4 rows.
- final_group_test.csv: 4 rows.
- domain_gap.csv: 2 rows.
- pca_results.csv: 28 rows.
- clustering_metrics.csv: 10 rows.
- split_sensitivity.csv: 27 rows.

Decision: `KEEP_AS_BIAS_STUDY` — Internal 99-100% does not generalize: frozen STRICT RF drops 99.81%->50.01% (-49.80pp) on independent Hassan 340k (340k URLs, 0.78% legit homepage overlap with PhiUSIIL=10 URLs); legit-with-path accuracy 0.0% (0/165085, CI [0,0.000022]); JPCERT 2025 phishing recall stays 99.84% only because model predicts phishing for any path. Direct leakage (USI constant 100 for legit; TLDLegitimateProb/URLCharProb corpus-dependent) + severe homepage-vs-deeplink sampling bias. Keep ONLY as bias/leakage case study with reframed RQs; CHANGE_DATASET for any deployment claim.
> BENCHMARK_NOT_REAL_WORLD_VALIDATED

## Artifacts
- outputs/audit/, outputs/metrics/, outputs/figures/, outputs/manifests/, outputs/decision.json, outputs/DECISION.md