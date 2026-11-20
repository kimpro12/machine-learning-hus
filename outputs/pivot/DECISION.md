# PhishVN Pivot — Decision

Label convention: PhishVN `label=1` phishing / `label=0` benign (positive = phishing).

## Verdict
**ADOPT PhishVN as the primary dataset for the ML course project (suitable, with documented caveats). Retain PhiUSIIL only as a separate negative benchmark case study. PhishVN is NOT validated for real-world deployment.**

- Internal classification (official temporal test): SUITABLE. FULL logreg_C10 acc 0.9026 CI [0.89625,0.90865], F1 0.9194, ROC 0.9584 — honest ~90% under a leakage-controlled temporal split with `is_https` excluded, vs PhiUSIIL's artefactual 99–100%.
- Cross-dataset generalization: NOT demonstrated. Frozen FULL drops to Hassan 0.4393 (-46.32pp, ROC 0.4298, specificity 0.0384); JPCERT phishing recall stays 0.9222 but benign collapses. Same failure mode as PhiUSIIL (benign side), so no deployment claim.
- The course value is precisely this contrast: a clean temporal benchmark (~90%) plus measurable source/tier/external gaps to analyse (RQ2–RQ4, PCA/clustering already runnable).

## Why suitable for the course (internal)
1. Provenance + protocol are sound: CC BY 4.0, MANIFEST SHA-256 verified (14/14), 0 URL dups, 0 cross-split domain overlap, official train/val/test honoured, HP selection on val only, test opened once.
2. Leakage controls present by design: CompPhish-aligned scheme-stripped lexical features, `is_https` excluded per authors, no IDs/sources/tiers/timestamps/scenarios in features (21 numeric only).
3. Headroom for ML coursework: LR vs RF vs LinearSVC compared on identical splits (FULL best logreg_C10 F1 0.9215 val → 0.9194 test); PCA hurts and clustering ARI ≈ 0 — reportable negative results, not failures.
4. Documented noise, not hidden: inter-annotator κ 0.609 (0.725 abuse-vs-legit), positive-arm noise 12.1% (bronze-concentrated), benign 4.0% — explains GOLDSILVER recall 0.3183 and silver-tier weakness without invalidating the benchmark.

## Why not deployable (generalization)
1. Source bias is large and measured: tranco global benign test acc lowest (see `test_by_source_FULL.csv`); silver tier weakest. The model learns ".vn/trusted = benign".
2. External benign collapses: Hassan specificity 0.0384. Internal 90% → external 44–48%.
3. JPCERT recall does not rescue this: phishing-only feeds cannot measure false positives (reported as `recall_phishonly` with `single_class=True`, accuracy left empty), which is where both PhiUSIIL and PhishVN fail.

## GOLDSILVER caveat
Gold+silver-only looks accurate (0.9114) but F1 is 0.4467 with phishing recall 0.3183 (155/487) on the official test — the verified core's newest phishing (temporal test = newest groups) is the hardest slice. Val positives are only 24, so the RF selection itself is unstable (LinSVC led val ROC). Use FULL as the primary benchmark; use GOLDSILVER to study label noise, drift, and class-imbalance handling — not as the headline number.
- Extractor audit note: `PHISHVN_FEATURES` (21 items; legacy alias `PHISHVN_NUMERIC_20` kept) validated 53114/53116 exact on integers; 2 residual release-quirk rows documented. Slice tables carry `tiny_n_flag` (n<30).

## What to do
- Course report: benchmark FULL (LR/RF/SVC), ablate tiers (gold/silver vs bronze), sources (tranco global vs .vn), URL slices (tiny-n flagged, don't overclaim), PCA/clustering as completed.
- Keep PhiUSIIL artifacts untouched as the catastrophic-generalization baseline.
- Deployment: CHANGE_DATASET (collect time-split, vendor-diverse URLs with legitimate deep links + independent validation) — out of scope for this course.

## Vì sao macro-F1 (và baseline dummy)
- Bài toán accuracy paradox: GOLDSILVER val chỉ có 24 phishing / 1585 mẫu. Model luôn đoán benign đạt acc 0.9849 — cao hơn RF 0.9666 — nhưng F1 phishing = 0.0000 (recall = 0, không bắt được ca phishing nào).
- Accuracy đánh trọng số theo prevalence (~98.5% trọng số dồn vào benign) nên bị đánh lừa. Macro-F1 cho mỗi lớp trọng số bằng nhau (50/50): dummy đạt macro-F1 0.4962 (trung bình của F1_benign≈0.99 và F1_phish=0), trong khi RF đạt 0.6560 — đảo ngược bảng xếp hạng và phản ánh đúng giá trị thực (bắt được 54.2% phishing).
- Quy tắc đọc báo cáo này: so accuracy với dummy trước (nếu model thua dummy về accuracy mà hơn về F1/macro-F1 thì đó là tín hiệu class-imbalance, không phải model tệ); chốt bằng F1 phishing + macro-F1, không chốt bằng accuracy.

## Evidence
- `outputs/pivot/QUICK_RESULTS.md`, `feature_audit.csv`, `classification_metrics.csv`, `external_metrics.csv`, `test_by_{source,tier,scenario,urlslice}_{FULL,GOLDSILVER}.csv`, `pca_results.csv`, `clustering_results.csv`
- `outputs/pivot_figures/pivot_confusion_internal_full.png`, `pivot_confusion_hassan_full.png`, `pivot_internal_vs_external_gap.png`, `pivot_source_bias.png`, `pivot_pca_scree.png`, `pivot_cluster_pca_scatter.png`
- Downloads: `data/pivot/PhishVN_open.zip` (sha verified), extracted `data/pivot/data/*`, `docs/*`, `MANIFEST.txt`
