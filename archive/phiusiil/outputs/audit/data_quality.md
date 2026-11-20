# Data quality audit

- n=235150, cols=89, phishing_share(internal y=1)=0.4265
- SEVERE_COLLECTION_BIAS=True, SUSPECT_LABEL_PROXY=True
- top single-feature stump: URLSimilarityIndex val_acc=0.9968

Label convention: original label=0 phishing / 1 legitimate; internal y=1 phishing / 0 legitimate.

See label_conditional_stats.csv, single_feature_stumps.csv, feature_correlation_top.csv, feature_provenance.csv.