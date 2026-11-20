# PHIUSIIL PHISHING DETECTION — AI CODING AGENT TECH SPEC (v1.0)

**Vai trò:** Senior ML Engineer + Research Engineer. **Ngôn ngữ code:** Python 3.11+. **Mục tiêu số 1:** thu được **Accuracy/F1 đo đúng**, đủ bằng chứng để quyết định **GIỮ / GIỮ NHƯNG ĐỔI GÓC NGHIÊN CỨU / ĐỔI DATASET**. **Mục tiêu số 2:** nếu qua vòng sàng lọc, chạy đủ thí nghiệm phù hợp môn Học máy. **Không xây website, API, GUI, Docker, deep learning hay sản phẩm production trong vòng đầu.**

> **Chỉ thị bắt buộc cho agent:** Đây là yêu cầu triển khai và CHẠY thực tế, không chỉ viết pseudocode. Tự xây repository hoàn chỉnh, cài thư viện, tải dữ liệu chính thức (khi mạng cho phép), chạy kiểm thử, chạy benchmark, lưu artifacts, đọc số liệu và viết kết luận. Không bịa kết quả hoặc tuyên bố đã chạy khi chưa chạy. Nếu môi trường không có mạng/dữ liệu, hoàn thiện code, chạy unit tests trên dữ liệu tổng hợp và ghi rõ bước nào đang bị chặn. Đừng dừng sau khi viết code. Không tự ý tối ưu trên test set.

## 0. Context và câu hỏi nghiên cứu

- Đề tài: **Phát hiện website phishing bằng các thuật toán học máy: phân loại, giảm chiều và phân cụm**.
- Dataset chính: **PhiUSIIL Phishing URL (Website), UCI ID 967**.
- UCI: https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
- Tên CSV: `PhiUSIIL_Phishing_URL_Dataset.csv`. Dữ liệu gốc: 235,795 dòng; 54 feature được UCI khai báo (bao gồm cả feature chuỗi), cùng trường `FILENAME` và nhãn `label`; kiểm tra schema thực thay vì hardcode tổng cột. UCI công bố tháng 03/2024; bản dữ liệu trên Mendeley xuất hiện từ 2023, nên phải báo cáo chính xác độ mới. Dữ liệu có 134,850 legitimate, 100,945 phishing; UCI ghi nhận không có missing values.
- **Nhãn gốc:** `label=0` phishing, `label=1` legitimate. **Trong nội bộ dự án** đổi thành `y=1` phishing, `y=0` legitimate. Bất cứ report nào cũng phải nêu rõ quy ước này.
- Thuật toán ưu tiên (trong phạm vi thầy dạy): DummyClassifier, Logistic Regression, Linear SVM (LinearSVC), Random Forest; Decision Tree dùng cho feature audit. Giai đoạn sau: PCA, K-Means, GMM, và tùy chọn KNN / RBF SVM nếu tài nguyên cho phép.
- Câu hỏi RQ1: Ba mô hình phân loại so sánh thế nào trên cùng bộ dữ liệu và phép chia?
- RQ2: Hiệu năng có giảm khi loại các feature nghi rò rỉ nhãn, chỉ sử dụng feature có thể trích xuất từ URL, hoặc giữ domain chưa thấy làm test không?
- RQ3: Giảm chiều bằng PCA ảnh hưởng ra sao đến F1, Accuracy và chi phí chạy?
- RQ4: K-Means và GMM có phát hiện được cấu trúc cụm và quan hệ với nhãn không?

## 1. Nguyên tắc ưu tiên / definition of done

**Thứ tự thực thi cứng:** `bootstrap -> fetch -> audit -> split -> quick benchmark -> robustness -> decision` **trước** `PCA -> clustering -> detailed report`. Bắt buộc in kết quả sớm từ quick benchmark; nếu kém hoặc có vấn đề nghiêm trọng thì vẫn xuất quyết định rõ, không phí công dựng phần mở rộng.

Mỗi lệnh có thể chạy lại độc lập, idempotent, seed rõ ràng. Chỉ dùng train để fit scaler/imputer/encoder/PCA/feature selection. Validation dùng để chọn model/siêu tham số/ngưỡng. **Test holdout bị khóa; chỉ mở khi model và quy trình đã được cố định**. Diagnostic được đăng ký trước thì có thể đánh giá trên holdout một lần, nhưng tuyệt đối không sửa mô hình dựa trên kết quả test rồi báo cáo lại cùng test là unbiased.

Không tính gần 100% trên random split là bằng chứng hoạt động ngoài đời. PhiUSIIL có dấu hiệu *collection bias*: các phân tích độc lập cho thấy URL legitimate chủ yếu là homepage dạng `https://www...`, trong khi URL phishing có đường dẫn; `URLSimilarityIndex` có thể gần như làm lộ nhãn. Bắt buộc có audit định lượng và ablation. Mục tiêu là nhận diện chính hạn chế này, không che đi.

## 2. Technology / runtime

- Python >=3.11, venv, `pyproject.toml` với phiên bản thư viện có pin hoặc constraints tái lập được.
- Core: pandas, numpy, scipy, scikit-learn, matplotlib, seaborn (nếu cần), PyYAML, joblib, ucimlrepo, tldextract (PSL snapshot offline), pytest, ruff.
- Chỉ CPU; không đòi GPU. Giới hạn thread qua cấu hình (`n_jobs`, OMP/MKL); không tạo dense n×n matrix; không dùng RBF SVM full 235k mẫu hoặc DBSCAN full nếu chưa tính tài nguyên.
- Đặt `RANDOM_SEED=42`, `PYTHONHASHSEED` nếu phù hợp; ghi các phiên bản package, OS, CPU, commit hash, cấu hình và thời điểm chạy.
- Ngắt an toàn / timeout các mô hình nặng, ghi status `SKIPPED_RESOURCE_LIMIT` chứ không tuyên bố fail dữ liệu.

## 3. Repository structure yêu cầu

```text
phishing-ml/
├── README.md
├── pyproject.toml
├── configs/
│   ├── quick.yaml
│   ├── full.yaml
│   └── feature_sets.yaml
├── data/
│   ├── raw/                 # original ZIP/CSV; gitignore
│   ├── processed/           # cleaned parquet, row IDs, split IDs; gitignore
│   └── external/            # optional independent labeled data; gitignore
├── src/phishing_ml/
│   ├── __init__.py
│   ├── cli.py
│   ├── fetch.py
│   ├── schema.py
│   ├── audit.py
│   ├── preprocess.py
│   ├── url_features.py
│   ├── split.py
│   ├── models.py
│   ├── evaluate.py
│   ├── quick.py
│   ├── robustness.py
│   ├── pca_experiment.py
│   ├── clustering.py
│   ├── reporting.py
│   └── decision.py
├── tests/
│   ├── test_label_mapping.py
│   ├── test_schema.py
│   ├── test_split_no_leakage.py
│   ├── test_preprocess_train_only.py
│   ├── test_url_features.py
│   ├── test_metrics.py
│   └── test_determinism.py
├── outputs/
│   ├── audit/
│   ├── metrics/
│   ├── figures/
│   ├── models/
│   ├── manifests/
│   ├── decision.json
│   ├── DECISION.md
│   └── SUMMARY.md
└── .gitignore
```

Không commit dataset thô, đường dẫn URL cá nhân, model binary lớn hoặc logs chứa URL nguyên văn nếu không cần. Đưa `outputs/` vào `.gitignore` (ngoại trừ README / sample outputs nếu hữu ích). Không bắt buộc notebook; CLI là nguồn chân lý, có thể thêm notebook trình bày nhưng không đặt logic chính ở notebook.

## 4. Data ingestion + schema

### 4.1 Fetch

- CLI: `python -m phishing_ml.cli fetch`.
- Ưu tiên gọi `ucimlrepo.fetch_ucirepo(id=967)` cho lấy data/metadata; nếu khó giữ raw `URL`, dùng ZIP/CSV từ link tải chính thức của UCI; fallback duy nhất là CSV user đặt vào `data/raw/PhiUSIIL_Phishing_URL_Dataset.csv`.
- Không dùng bản cleaned Hugging Face mặc định vì đó là *dataset đã biến đổi*; chỉ tham khảo để đối chiếu chất lượng.
- Lưu raw file bất biến; tính SHA256, row count, column list/types, source URL, license, ingest timestamp vào `outputs/manifests/data_manifest.json`.
- Nếu download lỗi thì thoát có hướng dẫn manual, tuyệt đối không tạo dữ liệu giả để báo accuracy.

### 4.2 Validate schema

- Fail fast nếu thiếu `URL` hoặc `label`. Audit toàn bộ schema thực tế; không suy luận nhãn từ tên tập tin/nhóm URL.
- Original target chỉ nhận 0/1; assert có cả hai lớp. Recode nội bộ `y=(label==0).astype(int)`; unit tests kiểm tra mapping.
- `FILENAME` hoàn toàn là metadata, **không phải feature**. `URL`, `Domain`, `TLD`, `Title` phải là metadata hoặc dùng cách encode được định nghĩa rõ, không đem thẳng chuỗi vào sklearn.
- `URLSimilarityIndex`, `TLDLegitimateProb`, `URLCharProb`, `DomainTitleMatchScore`, `URLTitleMatchScore` mặc định đánh dấu `HIGH_RISK_DERIVED` hoặc `REVIEW_REQUIRED` theo provenance; không tự khẳng định feature nào dùng trực tiếp nhãn nếu chưa kiểm chứng.
- Kiểm tra NaN, blank, Inf, duplicated names, dtype bất thường, constant features, outliers; report số đếm kể cả khi là 0.

### 4.3 Duplicate handling

- Chuẩn hóa URL *chỉ phục vụ dedup và grouping* (lowercase scheme/host, bỏ default port, chuẩn hóa slash hợp lý; **không** hợp nhất query/path tùy tiện; không đảo query params). Lưu cả URL gốc và normalized URL.
- Báo số duplicate exact rows, duplicate normalized URL, label-conflicting URLs. Với URL trùng **và cùng nhãn**: giữ một hàng deterministic. Với nhãn mâu thuẫn: quarantine toàn bộ group để tránh train/test conflict, ghi lý do; không chọn nhãn theo đa số.
- Không dedup dựa trên feature vector đơn thuần: hai URL khác nhau có thể trùng bộ feature hợp lệ.
- `row_id` ổn định, hash của normalized URL cộng dấu phân biệt nếu cần; tạo split assignments chỉ sau audit/dedup.

## 5. Feature sets: phải chạy tối thiểu 3 cấu hình

| Set | Mục tiêu | Quy tắc |
|---|---|---|
| `FULL_NUMERIC` | Baseline dễ so với bài báo | Toàn bộ feature số có sẵn sau khi bỏ metadata, target, ID. Có thể chứa feature derived/có bias; gắn nhãn **optimistic**. Không claim deployable. |
| `NO_SUSPECT_DERIVED` | Ablation | Từ FULL_NUMERIC trừ ít nhất `URLSimilarityIndex`, `TLDLegitimateProb`, `URLCharProb`, `DomainTitleMatchScore`, `URLTitleMatchScore`; thêm feature chỉ khi audit xác nhận rủi ro. |
| `URL_LEXICAL_STRICT` | Benchmark bảo thủ hơn | Chỉ tự tính feature từ chuỗi URL thô bằng hàm deterministic, **không dựa vào nhãn, tập test, HTML, WHOIS, external database hoặc thống kê toàn dataset**. |

### 5.1 URL_LEXICAL_STRICT implementation

Dùng `urllib.parse.urlsplit`, `ipaddress`, `re` và parser eTLD+1 offline (tldextract với snapshot PSL, không network request). Không gửi HTTP request tới website trong dataset (URL phishing có thể gây hại).

Tối thiểu extract: `url_length`, `host_length`, `path_length`, `query_length`, `fragment_length`, `num_dots_host`, `num_hyphens_host`, `num_subdomain_levels`, `num_digits_url`, `digits_ratio`, `num_special_chars_url`, `num_at`, `num_percent`, `num_equal`, `num_question`, `num_ampersand`, `num_slashes_path`, `num_query_params`, `is_https`, `host_is_ip`, `has_port`, `has_punycode`, `url_entropy`, `path_present`, `query_present`. Tùy chọn `tld_length`, `has_www` để *audit*, nhưng chạy thêm `STRICT_NO_COLLECTION_CUES` phải loại **cả họ proxy** cho cách lấy mẫu: `has_www`, `is_https`, `path_present`, `query_present`, `path_length`, `query_length`, `num_slashes_path`, `num_query_params`, `num_question`, `num_ampersand` (và proxy tương đương phát hiện trong audit); ghi chính xác danh sách feature bị loại, tránh loại flag nhưng vẫn để biến liên tục mã hóa y hệt tín hiệu đó.

Xử lý URL malformed bằng giá trị có định nghĩa + `parse_error_flag`, không âm thầm drop mẫu. Định nghĩa entropy Shannon chính xác, unit tests. Ghi rõ normalization chỉ phục vụ split, không vô tình sửa URL đầu vào feature extraction.

### 5.2 Grouping risk

`FULL_NUMERIC` có thể đạt Accuracy rất cao vì feature đã được tính bằng công cụ crawl và có thống kê/metadata phụ thuộc corpus. `URL_LEXICAL_STRICT` vẫn có thể đạt rất cao nhờ **collection bias** (ví dụ tất cả legitimate đều là homepage). Vì vậy phải có sanity audit thống kê theo lớp và một tập thử nghiệm khác nguồn nếu có.

## 6. EDA và collection-bias audit — bắt buộc trước khi huấn luyện

Xuất `outputs/audit/data_quality.md`, `outputs/audit/feature_provenance.csv`, `outputs/audit/label_conditional_stats.csv`, cùng PNG/CSV:

1. Tổng số mẫu, nhãn gốc và nhãn nội bộ, tỷ lệ phishing, cột, missing, duplicates.
2. Histogram của URLLength, DomainLength, numerical top features theo nhãn; thống kê mean, median, std, IQR, percentiles.
3. **Tỷ lệ theo nhãn** của `is_https`, `has_www`, `path_present`, `query_present`, `URLSimilarityIndex >= 99.9` và, nếu có, `is_http` hoặc tín hiệu HTML load failure. Ghi cả numerator/denominator; sử dụng tỷ lệ trong từng lớp.
4. Cây quyết định depth=1 trên mỗi feature số, fit trên *train only*, evaluate trên validation, xếp hạng single-feature balanced accuracy / accuracy; không dùng test để chọn.
5. Chỉ kiểm tra correlation / mutual information của feature–label trên train. Liệt kê top 15 và cảnh báo nếu một feature phân loại gần hoàn hảo hoặc có giá trị hằng trong một lớp.
6. Kiểm tra domain count, share của các domain giữa train/test, URL exact duplicates cross-split (phải bằng 0), feature distribution shift giữa random và domain holdout.
7. Báo schema provenance: URL derived, webpage derived, label-/corpus-dependent suspected, identifier/text. Không đánh đồng 'tương quan cao' với bằng chứng rò rỉ nhãn.

Cờ `SEVERE_COLLECTION_BIAS` nếu ví dụ có `P(path_present | legitimate) < 0.01` và `P(path_present | phishing) > 0.10`, hoặc cơ chế tương tự phân tách hai lớp do cách lấy mẫu; đây là diagnostic threshold do dự án đặt, không phải chân lý. Cờ `SUSPECT_LABEL_PROXY` nếu một feature đơn lẻ cho Accuracy validation >=0.99. Cả hai được ghi vào decision, **không được che**.

## 7. Splits và phòng chống leakage

Tạo, serialize và reuse split IDs; mọi mô hình/feature set cùng split IDs.

### 7.1 Random stratified (benchmark phổ thông)

- Seed 42, chia 64% train / 16% val / 20% test; stratify theo `y`.
- Split trên row IDs đã dedup; kiểm tra URL normalized không giao giữa splits.
- Dùng val cho chọn hyperparams, threshold, model. Chỉ mở test một lần sau freeze.

### 7.2 Domain-disjoint (robustness)

- Group theo eTLD+1 (registrable domain); các URL chung domain tuyệt đối không đi qua train/test.
- Dùng `StratifiedGroupKFold` 5 folds seed 42 lấy một fold làm test, áp dụng tiếp trong train-remaining với seed khác để lấy val khoảng 16% tổng; log class distribution vì group sizes có thể lệch. Nếu không khả thi, triển khai deterministic balanced group split có unit tests; không lén fallback random split mà vẫn gọi domain split.
- Domain nhóm không xác định hoặc IP: group theo host canonical hoặc normalized URL, ghi rule; tránh gom tất cả `UNKNOWN` thành một siêu group.
- Validate non-overlap domain, URL, IDs. Threshold và hyperparameter selection riêng trên val của domain split, không lấy kết quả random test để tune.

### 7.3 Holdout protocol

- **Quick benchmark trả về chỉ TRAIN/VALIDATION accuracy**, không phải final-test result.
- `finalize` lock config/model/threshold trước, sau đó chạy một lượt eval trên 2 holdouts và đánh dấu test evaluated. Nếu rerun vì bug thì ghi rõ invalidation và revision. Không chạy nhiều lần rồi lấy kết quả test đẹp nhất.
- Repeated 70/30, 80/20, 90/10 ở stage nghiên cứu phụ chỉ báo cáo *validation/repeated-split exploratory results*, không dùng để chọn mô hình dựa vào holdout cuối.

## 8. Quick benchmark — chạy trước tiên

CLI: `python -m phishing_ml.cli quick --config configs/quick.yaml`.

**Phase Q1 (mặc định):** train trên random split, đánh giá validation cho từng `FULL_NUMERIC`, `NO_SUSPECT_DERIVED`, `URL_LEXICAL_STRICT`, `STRICT_NO_COLLECTION_CUES`. Mỗi set ít nhất:

- `DummyClassifier(strategy='most_frequent')` — baseline class majority.
- `LogisticRegression(max_iter=2000, solver='lbfgs', class_weight=None)` kết hợp median imputer + `StandardScaler` đặt **trong Pipeline**; grid `C in {0.1,1,10}`.
- `RandomForestClassifier(n_estimators=200, max_depth in {None,16}, min_samples_leaf in {1,3}, random_state=42, n_jobs=bounded)`; median impute trong Pipeline; không cần scale.
- `LinearSVC(C in {0.1,1,10}, dual='auto', max_iter=5000)` với imputer + scaler trong Pipeline. ROC-AUC tính bằng `decision_function`; không gọi `.predict_proba()` nếu chưa calibration.
- Optional: single `DecisionTree(max_depth=1/3)` phục vụ kiểm tra label proxy, không dùng làm mô hình chính nếu không muốn.

**Không cần grid-search rộng ở quick.** Dùng val để chọn. Nếu cả LR/RF/SVM trên URL strict thấp, vẫn chạy audit trước khi quyết định đổi dataset. Nếu environment thiếu RAM/CPU, training subset của train (`50,000` stratified) để có tín hiệu sơ bộ; log rõ `train_subset_n` và không gọi kết quả đó là full benchmark; full run train toàn bộ khi khả thi.

Metrics từng run: accuracy, balanced_accuracy, precision/recall/F1 (positive=phishing), specificity, ROC-AUC, PR-AUC, confusion matrix `[[TN,FP],[FN,TP]]` với `y=1 phishing`, train duration, predict duration, sample count, feature count. Baseline tính majority accuracy khoảng 57.2% (trước dedup), phải kiểm tra từ actual data.

Tuning threshold (nếu dùng xác suất hoặc decision score): chỉ trên validation, tối ưu F1_phishing hoặc ràng buộc recall >=0.95 rồi tối ưu precision; **ghi cả metric threshold=0.5 (nếu probability) / 0 (nếu decision_function) và tuned threshold**. Không tối ưu threshold bằng test.

Outputs: `outputs/metrics/quick_validation.csv`, `outputs/metrics/quick_validation.json`, `outputs/figures/quick_accuracy_by_feature_set.png`, `outputs/figures/quick_f1_by_feature_set.png` và cập nhật `outputs/SUMMARY.md` với kết luận sơ bộ.

**Phase Q2:** Chọn 1–2 mô hình tốt nhất trên val từ URL strict và từ full numeric; chạy domain-disjoint train/val với config/feature set tương đương, xuất gap giữa random validation và domain validation. Viết warning nếu class balance giữa hai splits quá khác.

## 9. Final benchmark và thống kê

CLI `python -m phishing_ml.cli finalize --config configs/quick.yaml`.

Trước finalize: ghi `outputs/manifests/frozen_plan.json` (config SHA, data SHA, feature list, model params, thresholds, split hashes). Đánh giá test **chỉ sau freeze**. Không gọi `fit` trên test; không fit preprocessing trên train+test. Có thể fit lại train+val *sau khi* chọn tham số và threshold, nhưng nếu threshold calibration thay đổi phải có validation/calibration carve-out; mặc định an toàn nhất giữ model train-fit + val-calibrated và test nguyên vẹn.

Xuất `outputs/metrics/final_random_test.csv`, `outputs/metrics/final_group_test.csv`, confusion matrices, ROC/PR plots; 500–1000 bootstrap stratified resamples trên test predictions (không refit) cho 95% CI của Accuracy và F1. Báo N/phishing prevalence mỗi split. So sánh các feature sets và group-vs-random theo cùng quy tắc, dùng hiệu số percentage points, không giấu những run bị fail.

## 10. Go / No-Go policy (vì user chủ yếu muốn biết có đáng giữ đề tài không)

**Quyết định được chia thành hai khái niệm:** (A) đủ tốt để làm *course project* và (B) đủ tin để nói về *URL phishing thực tế*. Không đánh đồng hai việc.

Default thresholds cấu hình được tại `configs/quick.yaml` (heuristic dự án, không phải chuẩn khoa học):

- `academic_accuracy_target = 0.95` (random holdout), `academic_f1_target = 0.95` (phishing F1).
- `robust_group_accuracy_target = 0.90`, `robust_group_f1_target = 0.90` đối với **strict URL features**; `max_random_group_gap_pp = 5.0` là cảnh báo.
- `label_proxy_warning = 0.99` nếu 1 feature đơn độc đạt Accuracy rất cao.
- `severe_path_bias`: tỷ lệ có path ở legitimate <1% nhưng phishing >10% (kiểm tra thêm các dấu khác).
- `independent_external_test` là OPTIONAL; nếu chưa có dữ liệu ngoài độc lập gán nhãn, không được kết luận production quality.

Thứ tự ưu tiên áp dụng status: `BLOCKED` > `CHANGE_DATASET_RECOMMENDED` > `BORDERLINE_REVIEW` > `KEEP_WITH_WARNINGS` > `KEEP_ACADEMIC`; khi vừa đạt nhiều điều kiện, chọn status có mức cảnh báo cao hơn. Vẫn tách hai trục `academic_viability` và `real_world_validity` trong JSON.

Các status:

1. **`KEEP_ACADEMIC`**: đạt target 95% Accuracy + F1 trên random holdout; hoàn thành cả báo cáo audit và domain holdout. Nếu có collection bias, vẫn có thể làm môn học **nhưng đổi framing sang 'Ảnh hưởng của feature selection, data leakage và cách chia tập lên đánh giá phishing'**; không quảng cáo khả năng ngoài đời.
2. **`KEEP_WITH_WARNINGS`**: random >=95% nhưng domain strict <90%, gap >5pp hoặc audit phát hiện label proxy/collection bias nặng. Quyết định còn lại là: dùng dataset làm *case study phân tích tính hợp lệ* hoặc đổi dataset nếu muốn kết quả thực tế. Nêu chính xác warning nào được kích hoạt.
3. **`CHANGE_DATASET_RECOMMENDED`**: best random clean/strict Accuracy <90% **và** F1 <0.90 sau khi kiểm tra bug, mapping, split, class imbalance; hoặc dữ liệu vi phạm điều kiện môn học; hoặc không có mục đích học thuật hợp lệ nếu bias quá nặng. Accuracy 90–95% phải báo `BORDERLINE_REVIEW` thay vì kết luận fail ngay.
4. **`BLOCKED`**: thiếu dữ liệu, schema sai, split không thể xây hoặc test chưa đủ; không bịa decision từ một model chưa chạy.

Các điều kiện không được kết luận `KEEP` chỉ vì FULL_NUMERIC accuracy >=0.99. Nếu strict-only hay domain test kém, giải thích sai lệch và chỉ rõ model/feature/split cụ thể. Với `SEVERE_COLLECTION_BIAS`, luôn in banner **`BENCHMARK_NOT_REAL_WORLD_VALIDATED`** ngay cả khi Accuracy 100%.

`outputs/decision.json` bắt buộc chứa: `status`, `academic_viability`, `real_world_validity`, `decision_reason`, `dataset_hash`, `best_model_full`, `best_model_clean`, `best_model_strict`, `random_test_metrics`, `domain_test_metrics`, `single_feature_proxy`, `collection_bias_flags`, `recommendation`, `limitations`, `next_steps`.

`outputs/DECISION.md` mở đầu bằng **5 dòng** trả lời: (1) Accuracy best? (2) F1 phishing? (3) Strict URL-only ra sao? (4) domain-disjoint ra sao? (5) GIỮ hay ĐỔI và tại sao? Mọi giá trị phải có đường dẫn file metrics dẫn chứng.

## 11. Full academic experiments — CHỈ khi đã có decision sơ bộ

CLI: `python -m phishing_ml.cli full --config configs/full.yaml`.

**E1. So sánh phân loại:** Logistic Regression vs LinearSVC vs Random Forest trên feature sets, một split và tiền xử lý nhất quán. Learning curves / train-vs-val gaps nếu có thể.

**E2. Split sensitivity:** 70/30, 80/20, 90/10 với seeds 42/43/44 trên train-development pool, không đụng final holdout; báo mean ± std Accuracy/F1. Cố định preprocessing fit-only-train.

**E3. PCA:** Logistic Regression + Linear SVM (và RF nếu hợp lý) trên features numerical. Đưa PCA sau imputer+scaler trong Pipeline. Conditions: no-PCA; PCA giữ 90%, 95%, 99% explained variance hoặc số thành phần 5/10/20 nếu hợp lệ. Báo `n_components_`, explained variance, Accuracy, F1, fit/predict time. **Không fit PCA trên toàn bộ dataset.** Vẽ scree plot và cumulative explained variance.

**E4. Feature-group ablation:** FULL_NUMERIC vs NO_SUSPECT_DERIVED vs URL_LEXICAL_STRICT vs STRICT_NO_COLLECTION_CUES; từ URL đến HTML chỉ với các cột xác định được provenance. Phải có bảng mô tả trường nào thuộc nhóm nào. Không dùng tên `URL_ONLY` nếu chứa TLDLegitimateProb/label-derived corpus statistics mà chưa chứng minh có sẵn khi inference.

**E5. Clustering (phụ):** trên train hoặc independent stratified subsample tối đa 25,000 điểm; chỉ numerical clean, impute/scale/PCA 2-10 dims fit trên train. `KMeans(k=2..6, random_state=42, n_init=10)` và `GaussianMixture(n_components=2..6, covariance_type='diag', random_state=42)`; ghi Silhouette (sample max 5,000), Davies–Bouldin, Calinski–Harabasz, GMM BIC/AIC. Dùng `adjusted_rand_score` giữa clusters và labels **chỉ để mô tả hậu nghiệm**, không dùng nhãn train clusters. Visualize PCA scatter 2D (sample), cluster profile và tỷ lệ phishing trong từng cluster. Không diễn giải cluster ID 0 luôn là legitimate.

**E6. Bias/domain diagnostic:** báo `URLSimilarityIndex` stump validation performance; bảng tỷ lệ path/https/www/query theo nhãn; so sánh gap random/domain; confusion matrix riêng strict URL model; báo số lỗi trên legitimate deep-link nếu có independent labeled collection.

**E7 (optional). Independent external set:** nếu có nguồn URLs độc lập *đã được gán nhãn có nguồn*, chỉ áp dụng URL_LEXICAL_STRICT để infer offline, tuyệt đối không fetch URL hay render trang. Không reuse dataset khi tự cho là độc lập, không gán nhãn bằng chính model. Nếu chưa có dữ liệu độc lập, ghi `NOT_AVAILABLE`, không thay bằng synthetic test rồi xem như accuracy ngoài đời.

## 12. Tính khoa học và cách kể chuyện trong report

Báo cáo môn học phải có: đặt vấn đề → nguồn và giới hạn dữ liệu → EDA/thống kê → cleaning/feature engineering/normalization → train/val/test protocol → >=3 mô hình → PCA → clustering K-Means/GMM → so sánh kịch bản → giải thích kết quả → hạn chế. Phân cụm không phải bài toán chính; không buộc cluster cùng label. Không cần SMOTE/oversampling vì hai lớp tương đối cân bằng trừ khi thật sự chứng minh nó hữu ích; nếu có, chỉ resample trên train của từng fold. Không 'tăng cường' URL bằng thay chuỗi tùy tiện nếu nhãn mới không chắc chắn.

Không bỏ những kết quả kém. Đặc biệt, nếu random accuracy cao và domain/external thấp thì đó **là kết quả nghiên cứu đáng báo cáo**, không phải thất bại coding.

## 13. Visualizations bắt buộc

- `class_distribution.png`, `missing_by_column.png` (hoặc bảng xác nhận không missing), `duplicate_audit.png`.
- `url_length_by_label.png`, `feature_correlation_top.png`, `path_https_www_by_label.png`, `single_feature_stump.png`.
- `quick_accuracy_by_feature_set.png`, `quick_f1_by_feature_set.png`, `random_vs_domain.png`, `confusion_matrix_best_strict.png`, `roc_pr_best_models.png`.
- `pca_scree.png`, `pca_accuracy_tradeoff.png`, `cluster_pca_scatter.png`, `cluster_class_composition.png`, `cluster_selection_metrics.png`.
- Tất cả axes, units, legends, sample sizes rõ; các biểu đồ hiển thị tỷ lệ phải normalize theo mẫu số đúng. Không tự vẽ hoặc hardcode số liệu không lấy từ artifacts.

## 14. CLI contract và exit behavior

```bash
python -m venv .venv
# activate .venv, then:
pip install -e '.[dev]'
python -m phishing_ml.cli fetch
python -m phishing_ml.cli audit
python -m phishing_ml.cli quick --config configs/quick.yaml
python -m phishing_ml.cli finalize --config configs/quick.yaml
python -m phishing_ml.cli full --config configs/full.yaml
python -m phishing_ml.cli report
pytest -q
ruff check .
```

Thêm lệnh orchestration `python -m phishing_ml.cli run --mode quick` = fetch + audit + quick + robustness + freeze + finalize + decision; `--mode full` chạy thêm E1–E7 sau các bước trên. Nếu chưa có internet, `--offline` load data/raw. Lệnh nào cũng ghi structured logs, lỗi meaningful, không im lặng bắt exceptions. `--dry-run` in thông số dự tính và paths mà không bắt buộc fit; `--seed` override config chỉ khi toàn bộ manifests thay đổi tương ứng.

## 15. Test cases bắt buộc (tự tạo synthetic fixtures, tuyệt đối không coi đây là accuracy thật)

- `label=0 -> phishing (internal 1)` và `label=1 -> legitimate (internal 0)`.
- Test metric confusion matrix với 4 cases có TN/FP/FN/TP kiểm tra bằng tay; Precision/Recall dùng positive class internal 1.
- FILENAME, raw URL, Domain, Title và target không lọt vào numeric ML feature matrix.
- Preprocessor/PCA/encoder fit train only; thử thay đổi test distribution phải không làm thay fitted scaler parameters.
- Không có overlap normalized URL hoặc eTLD+1 giữa domain train/val/test.
- Duplicate URL conflicting label không lọt hai splits; được quarantine.
- Test parser với `https://www.example.org/a?x=1`, IP host, punycode, relative/malformed URL, percent encoding, port; feature output deterministic/finite.
- Test data source / schema invalid phát hiện chính xác; no fake dataset fallback; train/val/test sample counts sum correctly.
- Two reruns cùng version/seed/data tạo split SHA và metrics bằng nhau trong tolerance.
- Test bootstrap CI, threshold selection validation-only, `run` báo trạng thái BLOCKED đúng khi data vắng.

## 16. Bảng output và yêu cầu kết thúc của coding agent

Agent **không được dừng sau code generation**. Kết quả trả lời cuối cùng bắt buộc theo format:

```text
IMPLEMENTATION STATUS: COMPLETE / PARTIAL / BLOCKED
DATA: n=..., columns=..., phishing share=..., sha256=...
VALIDATION (quick): best FULL_NUMERIC Accuracy=... F1=...
VALIDATION (quick): best URL_LEXICAL_STRICT Accuracy=... F1=...
FINAL random test: best Accuracy=... F1=... [CI]
FINAL domain-disjoint test: Accuracy=... F1=... [CI]
AUDIT: suspect proxy=...; collection bias=...; split leakage=...
DECISION: KEEP_ACADEMIC / KEEP_WITH_WARNINGS / BORDERLINE_REVIEW / CHANGE_DATASET_RECOMMENDED / BLOCKED
WHY: <=5 bullet points, grounded in actual metrics
FILES: outputs/SUMMARY.md, outputs/DECISION.md, outputs/decision.json, CSVs, PNGs
COMMANDS RUN: ...
TESTS: passed/failed/skipped and explanations
```

**Không in giá trị metrics nếu chưa chạy.** Nếu chỉ quick chạy được: ghi rõ Validation, `FINAL TEST = NOT RUN`; không suy diễn. Nếu accuracy rất cao, dành ưu tiên kiểm tra label proxy + collection bias trước khi nâng cao mô hình. Nếu full numeric gần 100% nhưng strict hay external thấp, không được ghi 'thành công 100% trong thực tế'.

## 17. Tài liệu tham khảo + cảnh báo có nguồn

1. UCI official, PhiUSIIL ID 967: https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
2. Paper gốc Prasad & Chandra, Computers & Security: https://www.sciencedirect.com/science/article/pii/S0167404823004558
3. Mendeley Data v2 (November 2023): https://data.mendeley.com/datasets/shwpxscxy2/2
4. Independent data-quality audit (tư liệu tham khảo, KHÔNG coi là kết quả đã tái lập): https://github.com/Yoyagm/phishing-url-detector
5. Independent project documenting homepage-only bias: https://github.com/Muneeraothman/phishing-detection-ml
6. 2026 dataset-validity analysis (external literature, not independently reproduced by this agent): https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7431520

**End of spec. Implement and execute; do not stop at planning.**
