# Suspicious-feature audit — leakage vs collection bias

Label convention: original label=0 phishing / 1 legitimate; internal y=1 phishing / 0 legitimate.

## URLSimilarityIndex (HIGH_RISK — effective label proxy, excluded from deployable sets)
- Legitimate: ALL exactly 100.0 (std 0, n=134850, P(USI>=99.9|legit)=1.0).
- Phishing: mean 49.62, median 51.42, only 786/100945 (0.78%) >=99.9.
- Single depth-1 stump (train-fit, val-eval): threshold 98.44 → val_acc 0.9968, bal_acc 0.9962.
- Per PhiUSIIL paper (Prasad & Chandra, Computers & Security 2024), USI is a Phase-1 detector for visual-similarity attacks (homograph/punycode/combosquat), blocking URLs with 80<=USI<100. Its construction compares a URL against a reference (whitelist/corpus + crawled context), i.e. NOT computable from the URL string alone at inference.
- Verdict: whether or not the formula literally ingests the label, the empirical distribution (constant 100 for one class) makes it a near-perfect class proxy. Treat as **direct leakage in effect**; never in URL-only sets. Removed in NO_SUSPECT_DERIVED and strict sets.

## TLDLegitimateProb (DIRECT LEAKAGE — corpus/label-dependent)
- Legit median 0.522907 (=global max), phish median 0.018013. This is P(TLD | legitimate) estimated from the dataset's own label distribution. Requires labels to compute; unavailable at inference for unseen TLDs without the training corpus. Direct leakage.

## URLCharProb (DIRECT LEAKAGE — corpus-dependent)
- Legit mean 0.06005 vs phish 0.049999. Character probability under the legitimate corpus. Same corpus-dependence as above. Direct leakage.

## DomainTitleMatchScore / URLTitleMatchScore (WEBPAGE-DERIVED + bias-amplified, REVIEW_REQUIRED)
- Legit median 100 vs phish median 0 for both. Require fetching HTML Title and matching to domain/URL — unavailable from URL alone, fragile, and confounded by homepage-only legit sampling (homepage title matches domain by construction). Excluded from strict sets.

## CharContinuationRate, URLLength, DomainLength, IsHTTPS/has_www/path/query (COLLECTION BIAS, not formula leakage)
- URL-derived (computable from string) but sampling artifact: legit 100% https, 100% www, 0% paths, 0% queries, URL length 26.2±4.8 (max 57); phish 49.2% https, 41.6% www, 27.0% paths, 6.0% queries, length 45.7±61.1 (max 6097).
- Formula does not use labels, but the sampling process separates classes, so even strict URL-only models inherit the bias (see external collapse). Kept in STRICT to measure bias; removed as a family in STRICT_NO_COLLECTION_CUES — which still reaches 99.07% internal because residual cues (length/entropy/dots) encode the same homepage-vs-deeplink split.

Rule: correlation ≠ proof of formula leakage. The three corpus-dependent features + USI are leakage (need labels/corpus/HTML at build time); the URL-structure cues are bias (computable but sampled adversarially). Evidence: `outputs/external/suspicious_feature_summary.csv`, `outputs/audit/label_conditional_stats.csv`, `outputs/audit/single_feature_stumps.csv`.
