# PhishVN — Thí nghiệm bổ sung: tiền xử lý + leave-one-source-out

Logistic Regression C=10, tập FULL. Nhãn: `1` phishing / `0` benign, lớp dương = phishing.
**Đây là thí nghiệm thứ hai, chạy sau benchmark chính** (test chính đã mở một lần cho `logreg_C10`). Mọi lựa chọn (ngưỡng, biến thể tốt nhất) dựa trên **val**; test, Hassan, JPCERT chỉ dùng để chấm. Kết quả này không thay thế model chính trong `DECISION.md`.

## 0. Phát hiện chính: tiền tố `www.` là dấu vết thu thập dữ liệu

| Tập | Nhóm | n | tỷ lệ URL bắt đầu bằng `www.` |
|---|---|---|---|
| phishvn_train | benign | 11004 | 0.6% |
| phishvn_train | phishing | 25905 | 45.7% |
| phishvn_train_source | chongluadao | 23621 | 49.3% |
| phishvn_train_source | openphish | 208 | 19.2% |
| phishvn_train_source | tinnhiem_org | 1659 | 3.7% |
| phishvn_train_source | tinnhiem_web | 3362 | 0.1% |
| phishvn_train_source | tinnhiemmang | 2076 | 7.5% |
| phishvn_train_source | tranco | 4170 | 0.0% |
| phishvn_train_source | tranco_vn | 1813 | 0.0% |
| hassan | benign | 170000 | 72.0% |
| hassan | phishing | 170000 | 27.8% |
| jpcert | phishing | 14775 | 8.3% |

Trong PhishVN, URL benign gần như được lưu dưới dạng domain trần (train: 0.6% có `www.`), còn phishing thì 45.7%. Model vì vậy học luôn quy tắc "`www.` = phishing" (n-gram `www.` đứng đầu danh sách đẩy về phishing ở mục 1b). Trên Hassan, 72.0% URL hợp lệ có `www.`, nên bị gán nhãn phishing hàng loạt. Đây là cơ chế cụ thể đằng sau specificity rất thấp trên Hassan. Cách xử lý: **chuẩn hóa host bằng cách bỏ `www.` đứng đầu** trước khi trích feature (các biến thể `*_nowww`). Đây là bước làm sạch hợp lý trong mọi trường hợp: `www.example.com` và `example.com` là cùng một site.

## 1. So sánh cách tiền xử lý

| Variant | thr | Val acc | Val F1 | Val macro-F1 | Test acc [CI95] | Test F1 | Test macro-F1 | Test ROC | Hassan acc | Hassan spec | Hassan macro-F1 | JPCERT rec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `num_standard` | 0.50 | 0.8713 | 0.9215 | 0.7823 | 0.9026 [0.896, 0.909] | 0.9194 | 0.8982 | 0.9584 | 0.4393 | 0.0384 | 0.3320 | 0.9222 |
| `num_log1p` | 0.50 | 0.8688 | 0.9198 | 0.7804 | 0.9033 [0.897, 0.909] | 0.9199 | 0.8989 | 0.9587 | 0.4183 | 0.0360 | 0.3187 | 0.9350 |
| `num_quantile` | 0.50 | 0.8633 | 0.9160 | 0.7751 | 0.8886 [0.882, 0.895] | 0.9077 | 0.8836 | 0.9416 | 0.4414 | 0.1093 | 0.3722 | 0.9343 |
| `num_log1p_balanced` | 0.50 | 0.8043 | 0.8637 | 0.7585 | 0.8544 [0.847, 0.862] | 0.8610 | 0.8541 | 0.9605 | 0.3931 | 0.0974 | 0.3349 | 0.9059 |
| `num_log1p_balanced_valthr` | 0.26 | 0.8709 | 0.9212 | 0.7819 | 0.9027 [0.896, 0.909] | 0.9195 | 0.8983 | 0.9605 | 0.3968 | 0.0780 | 0.3286 | 0.9241 |
| `url_tfidf` | 0.50 | 0.8819 | 0.9222 | 0.8388 | 0.9389 [0.934, 0.944] | 0.9462 | 0.9378 | 0.9856 | 0.4967 | 0.2221 | 0.4557 | 0.8585 |
| `url_tfidf+num_log1p` | 0.50 | 0.8953 | 0.9316 | 0.8542 | 0.9388 [0.934, 0.944] | 0.9462 | 0.9376 | 0.9853 | 0.4707 | 0.0283 | 0.3419 | 0.9604 |
| `num_standard_nowww` | 0.50 | 0.8358 | 0.9010 | 0.7099 | 0.8799 [0.873, 0.887] | 0.9016 | 0.8737 | 0.9380 | 0.4287 | 0.0469 | 0.3312 | 0.9081 |
| `url_tfidf_nowww` | 0.50 | 0.8395 | 0.8925 | 0.7882 | 0.9189 [0.913, 0.924] | 0.9282 | 0.9175 | 0.9766 | 0.5923 | 0.4537 | 0.5843 | 0.8569 |
| `url_tfidf+num_log1p_nowww` | 0.50 | 0.8390 | 0.8921 | 0.7873 | 0.9195 [0.914, 0.925] | 0.9288 | 0.9181 | 0.9754 | 0.4786 | 0.0659 | 0.3715 | 0.9609 |
| `num_pca95` | 0.50 | 0.7726 | 0.8621 | 0.6071 | 0.8150 [0.807, 0.823] | 0.8472 | 0.8064 | 0.8717 | 0.4782 | 0.1689 | 0.4230 | 0.9496 |
| `num_undersample` | 0.50 | 0.8080 | 0.8670 | 0.7611 | 0.8574 [0.850, 0.865] | 0.8647 | 0.8570 | 0.9588 | 0.4109 | 0.1110 | 0.3527 | 0.8758 |
| `num_kmeans_onehot` | 0.50 | 0.8724 | 0.9220 | 0.7858 | 0.9026 [0.896, 0.909] | 0.9192 | 0.8983 | 0.9603 | 0.4395 | 0.0459 | 0.3367 | 0.9141 |

Mô tả biến thể:
- `num_standard`: baseline: median impute + StandardScaler (= main benchmark logreg_C10)
- `num_log1p`: log1p before scaling (counts are heavily right-skewed)
- `num_quantile`: QuantileTransformer -> normal (rank-based, outlier-proof)
- `num_log1p_balanced`: log1p + class_weight=balanced (threshold 0.5)
- `num_log1p_balanced_valthr`: log1p + class_weight=balanced + threshold chosen on val
- `url_tfidf`: char 2-5-gram TF-IDF on the scheme-stripped URL string only
- `url_tfidf+num_log1p`: char TF-IDF + 21 numeric features (log1p + scaled)
- `num_standard_nowww`: baseline on www-normalised URLs (leading 'www.' stripped, features recomputed)
- `url_tfidf_nowww`: char TF-IDF on www-normalised URLs
- `url_tfidf+num_log1p_nowww`: char TF-IDF + numeric (log1p), both on www-normalised URLs
- `num_pca95`: dimensionality reduction: PCA keeping 95% of variance (fit on train), then LogReg
- `num_undersample`: imbalance: randomly drop phishing rows in train to a 50/50 balance
- `num_kmeans_onehot`: clustering as a feature: append one-hot K-Means (k=8, fit on train) cluster id

### Kiểm định so với baseline (`num_standard`) trên test (n=8941)
| Variant | Δ test macro-F1 vs baseline [paired bootstrap CI95] | only new right / only baseline right | McNemar p |
|---|---|---|---|
| `num_log1p` | +0.0007 [-0.0009, +0.0020] | 21 / 15 | 0.41 |
| `num_quantile` | -0.0146 [-0.0179, -0.0114] | 39 / 164 | 2e-19 * |
| `num_log1p_balanced` | -0.0441 [-0.0525, -0.0355] | 503 / 934 | 3e-30 * |
| `num_log1p_balanced_valthr` | +0.0001 [-0.0009, +0.0012] | 10 / 9 | 1 |
| `url_tfidf` | +0.0396 [+0.0329, +0.0459] | 603 / 278 | 2.3e-28 * |
| `url_tfidf+num_log1p` | +0.0395 [+0.0329, +0.0462] | 604 / 280 | 4.3e-28 * |
| `num_standard_nowww` | -0.0244 [-0.0285, -0.0205] | 52 / 255 | 2.6e-33 * |
| `url_tfidf_nowww` | +0.0193 [+0.0122, +0.0260] | 545 / 399 | 2.3e-06 * |
| `url_tfidf+num_log1p_nowww` | +0.0199 [+0.0132, +0.0271] | 551 / 400 | 1.1e-06 * |
| `num_pca95` | -0.0918 [-0.0991, -0.0842] | 197 / 980 | 3e-125 * |
| `num_undersample` | -0.0412 [-0.0492, -0.0328] | 487 / 891 | 7.8e-28 * |
| `num_kmeans_onehot` | +0.0001 [-0.0012, +0.0014] | 16 / 16 | 1 |

`*` = p < 0.05 (McNemar chính xác, trên các mẫu hai model dự đoán khác nhau). CI bootstrap ghép cặp 1000 lần, cùng các dòng test.

### Đọc kết quả
- **Biến thể tốt nhất theo val macro-F1: `url_tfidf+num_log1p`** (val 0.8542 so với baseline 0.7823; test 0.9376 so với 0.8982, Δ +0.0395 [+0.0329, +0.0462], McNemar p=4.3e-28).
- **Đổi cách scale feature số** (val macro-F1: `num_standard` 0.7823, `num_log1p` 0.7804, `num_quantile` 0.7751, `num_log1p_balanced` 0.7585, `num_standard_nowww` 0.7099, `num_pca95` 0.6071, `num_undersample` 0.7611, `num_kmeans_onehot` 0.7858). Feature lệch rất mạnh (skew trên train: `amp_cnt` 86, `under_cnt` 70, `query_len` 64, `eq_cnt` 56), vậy mà log1p không giúp gì; còn QuantileTransformer làm kết quả **kém hơn** (`num_log1p` Δ test macro-F1 +0.0007, McNemar p=0.41 (không khác có ý nghĩa); `num_quantile` Δ test macro-F1 -0.0146, McNemar p=2e-19 (khác có ý nghĩa)). Với LogReg ở đây, thông tin nằm ở việc feature có hay không có và ở thứ tự thô của giá trị; ép về phân phối chuẩn theo hạng làm mất khoảng cách giữa các giá trị cực trị (URL rất dài, query rất nhiều tham số), mà chính các giá trị đó mang tín hiệu.
- **class_weight / ngưỡng**: `num_log1p_balanced` ở ngưỡng 0.5 có test F1 0.8610; khi chọn lại ngưỡng trên val (thr=0.26) F1 về 0.9195. Trọng số lớp chủ yếu chỉ dịch ngưỡng; chọn ngưỡng trên val là đủ.
- **TF-IDF n-gram ký tự** học trực tiếp các chuỗi con của URL mà 21 feature đếm không thấy (xem mục 1b). Trên Hassan, biến thể tốt nhất theo val (`url_tfidf+num_log1p`) có macro-F1 0.3419, specificity 0.0283 so với baseline 0.3320 / 0.0384. Biến thể tổng quát hóa tốt nhất sang Hassan lại là `url_tfidf_nowww` (macro-F1 0.5843, specificity 0.4537). Cải thiện nội bộ **không** tự động chuyển thành khả năng tổng quát hóa, và chọn model theo val không chọn ra model tổng quát hóa tốt nhất.

### 1b. N-gram có trọng số lớn nhất (`url_tfidf`)
- Đẩy về **phishing**: `ww`, `www.`, ` www.`, ` vn`, `www`, `ww.`, `vn.`, ` www`, `vay`, ` ww`, `w.`, ` nap`, `vip`, `nap`, `vn-`
- Đẩy về **benign**: `.vn `, `vn `, `.vn`, `n `, `r `, `.de `, `l `, `u `, `.gov`, ` 1`, `jp`, `.jp`, `.go`, `.gov.`, `jp `

Phía phishing bị chi phối bởi `www.` (mục 0) và các từ khóa lừa đảo tiếng Việt (ví dụ `vay`, `nap`, `vip`). Phía benign là đuôi domain (`.vn` ở cuối host, `.gov`, `.jp`, `.de`): đó là chữ ký của danh sách tổ chức tin cậy (tinnhiem) và Tranco, đúng loại phụ thuộc nguồn mà mục 2 đo được. Danh sách đầy đủ ở `tfidf_top_ngrams.csv`.

### 1c. Bỏ `www.` thay đổi được gì
| Raw → www-normalised | Val macro-F1 | Test macro-F1 | Hassan specificity | Hassan macro-F1 |
|---|---|---|---|---|
| `num_standard` → `num_standard_nowww` | 0.7823 → 0.7099 | 0.8982 → 0.8737 | 0.0384 → **0.0469** | 0.3320 → **0.3312** |
| `url_tfidf` → `url_tfidf_nowww` | 0.8388 → 0.7882 | 0.9378 → 0.9175 | 0.2221 → **0.4537** | 0.4557 → **0.5843** |
| `url_tfidf+num_log1p` → `url_tfidf+num_log1p_nowww` | 0.8542 → 0.7873 | 0.9376 → 0.9181 | 0.0283 → **0.0659** | 0.3419 → **0.3715** |

**Điểm nội bộ giảm** khi bỏ `www.` (Δ val macro-F1 từ -0.0724 đến -0.0506). Một phần thành tích nội bộ đến từ lối tắt `www.`; con số sau khi bỏ trung thực hơn, dù thấp hơn. Trên Hassan, cải thiện lớn nhất ở `url_tfidf_nowww`: specificity +0.232, macro-F1 0.5843, accuracy 0.5923. Như vậy với model chỉ đọc chuỗi URL, `www.` là nguyên nhân lớn của lỗi benign trên Hassan. Ngược lại `num_standard_nowww` gần như không đổi (Δ specificity < 0.02): các feature số còn chứa lối tắt khác (độ dài domain, số subdomain, đuôi domain…) mà việc bỏ `www.` không gỡ được.

### 1d. PCA, undersampling và cụm K-Means trên 21 feature số
Cùng LogReg C=10, mỗi dòng chỉ đổi **một** bước xử lý so với baseline (`num_standard`). Mọi tham số của bước xử lý (các thành phần PCA, tâm cụm K-Means, mẫu được giữ khi undersampling) chỉ học trên train.

| Nhóm kỹ thuật | Biến thể | Làm gì | Val macro-F1 | Test macro-F1 | Δ test vs baseline [CI95] | McNemar p | Hassan macro-F1 |
|---|---|---|---|---|---|---|---|
| dimensionality reduction | `num_pca95` | dimensionality reduction: PCA keeping 95% of variance (fit on train), then LogReg | 0.6071 | 0.8064 | -0.0918 [-0.0991, -0.0842] | 3e-125 | 0.4230 |
| class imbalance | `num_undersample` | imbalance: randomly drop phishing rows in train to a 50/50 balance | 0.7611 | 0.8570 | -0.0412 [-0.0492, -0.0328] | 7.8e-28 | 0.3527 |
| clustering as feature | `num_kmeans_onehot` | clustering as a feature: append one-hot K-Means (k=8, fit on train) cluster id | 0.7858 | 0.8983 | +0.0001 [-0.0012, +0.0014] | 1 | 0.3367 |

- Tốt nhất theo val trong nhóm này: **`num_kmeans_onehot`** (val macro-F1 0.7858 so với baseline 0.7823).
- **Tốt hơn baseline có ý nghĩa** (cả khoảng tin cậy > 0): không có.
- **Không khác biệt** (khoảng tin cậy chứa 0): `num_kmeans_onehot` (+0.0001).
- **Kém hơn có ý nghĩa** (cả khoảng tin cậy < 0): `num_pca95` (-0.0918), `num_undersample` (-0.0412).
- Trên Hassan, tốt nhất trong nhóm này là `num_pca95` (macro-F1 0.4230 so với baseline 0.3320).

## 2. Leave-one-source-out

Mỗi nguồn trong PhishVN chỉ chứa **một nhãn** (ví dụ chongluadao = 100% phishing, tranco = 100% benign), nên model có thể học "URL này giống nguồn nào" thay vì "URL này có phải phishing không". Cách đo: bỏ hẳn một họ nguồn khỏi train, rồi chấm trên các dòng val+test của chính họ đó. Accuracy ở đây bằng recall (họ phishing) hoặc specificity (họ benign).

| Variant | Source family (label) | n eval | acc, source in train | acc, source held out [CI95] | gap (pp) |
|---|---|---|---|---|---|
| `num_standard` | chongluadao (phishing) | 10202 | 0.9690 | 0.6163 [0.607, 0.626] | -35.3 |
| `num_standard` | openphish (phishing) | 88 | 0.9432 | 0.4659 [0.359, 0.575] | -47.7 |
| `num_standard` | tinnhiem (org+web) (benign) | 2884 | 0.9442 | 0.0988 [0.088, 0.110] | -84.5 |
| `num_standard` | tinnhiemmang (phishing) | 511 | 0.9491 | 0.9393 [0.915, 0.958] | -1.0 |
| `num_standard` | tranco (global+vn) (benign) | 2522 | 0.4853 | 0.1538 [0.140, 0.169] | -33.1 |
| `num_standard_nowww` | chongluadao (phishing) | 10202 | 0.9604 | 0.2772 [0.269, 0.286] | -68.3 |
| `num_standard_nowww` | openphish (phishing) | 88 | 0.9205 | 0.6023 [0.492, 0.705] | -31.8 |
| `num_standard_nowww` | tinnhiem (org+web) (benign) | 2884 | 0.9282 | 0.0652 [0.056, 0.075] | -86.3 |
| `num_standard_nowww` | tinnhiemmang (phishing) | 511 | 0.9198 | 0.9178 [0.891, 0.940] | -0.2 |
| `num_standard_nowww` | tranco (global+vn) (benign) | 2522 | 0.3624 | 0.1519 [0.138, 0.166] | -21.1 |
| `url_tfidf+num_log1p_nowww` | chongluadao (phishing) | 10202 | 0.8808 | 0.4529 [0.443, 0.463] | -42.8 |
| `url_tfidf+num_log1p_nowww` | openphish (phishing) | 88 | 0.9773 | 0.6023 [0.492, 0.705] | -37.5 |
| `url_tfidf+num_log1p_nowww` | tinnhiem (org+web) (benign) | 2884 | 0.9719 | 0.1172 [0.106, 0.130] | -85.5 |
| `url_tfidf+num_log1p_nowww` | tinnhiemmang (phishing) | 511 | 0.9061 | 0.9002 [0.871, 0.925] | -0.6 |
| `url_tfidf+num_log1p_nowww` | tranco (global+vn) (benign) | 2522 | 0.7847 | 0.2280 [0.212, 0.245] | -55.7 |

### Đọc kết quả
- Gap lớn nhất: **tinnhiem (org+web) (benign)** với `num_standard_nowww`, -86.3pp (0.9282 → 0.0652).
- Với baseline, accuracy trên họ **benign** khi bị loại khỏi train: tinnhiem (org+web) 0.0988, tranco (global+vn) 0.1538. Gap âm lớn nghĩa là model nhận ra benign nhờ đã thấy phong cách của nguồn đó, không phải nhờ quy luật chung. Đây là cùng cơ chế khiến model sụp trên Hassan.
- Khi bỏ `chongluadao` (chiếm phần lớn phishing của train), số phishing còn lại để học giảm mạnh (xem `leave_one_source_out.csv`, cột `train_pos_left`). Gap ở đây phản ánh cả việc thiếu dữ liệu lẫn việc lệch nguồn.
- Bỏ `www.` (baseline, accuracy khi họ bị loại khỏi train): chongluadao: 0.6163 → 0.2772; openphish: 0.4659 → 0.6023; tinnhiem (org+web): 0.0988 → 0.0652; tinnhiemmang: 0.9393 → 0.9178; tranco (global+vn): 0.1538 → 0.1519. Với chongluadao (nguồn phishing lớn nhất) bị loại khỏi train, recall rơi 0.6163 → 0.2772 khi bỏ `www.`: `www.` là tín hiệu chính model dùng để nhận ra phishing kiểu chongluadao. Các họ benign chưa thấy vẫn chỉ đạt ≤ 0.1519, nên chữ ký nguồn không chỉ nằm ở `www.` (đuôi domain, độ dài, kiểu đặt tên của từng danh sách).
- Họ có n<30 (⚠) chỉ để tham khảo, không diễn giải.

## Files
- `outputs/preprocessing_ablation.csv`, `outputs/leave_one_source_out.csv`, `outputs/tfidf_top_ngrams.csv`, `outputs/www_prefix_rates.csv`
- `outputs/figures/preprocessing_ablation.png`, `outputs/figures/leave_one_source_out.png`
- Code: `src/phishing_ml/experiments.py`
