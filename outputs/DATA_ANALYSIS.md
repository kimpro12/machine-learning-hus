# PhishVN — Phân tích dữ liệu (chọn feature, PCA, phân cụm)

Mọi phân tích dùng **tập train** (36,909 URL) của PhishVN. Tập val chỉ dùng để chấm các thí nghiệm có giám sát (đường cong top-k, đường L1, AUC từng thành phần PCA). **Tập test không được dùng ở đây.**
Trừ khi ghi khác, dữ liệu được tiền xử lý bằng: điền median → `log1p` → `StandardScaler`. Lý do dùng `log1p`: các feature đếm lệch rất mạnh (12/21 feature có hơn 50% giá trị bằng 0, vài URL có giá trị rất lớn); `log1p(x) = log(1 + x)` giữ 0 là 0 và kéo các giá trị cực lớn lại gần, để chúng không lấn át khoảng cách trong PCA và K-Means.

---

## A1. Xếp hạng và chọn feature

**Làm gì:** xếp hạng feature theo **mutual information (MI)** với nhãn: biết giá trị feature thì giảm được bao nhiêu độ bất định về nhãn (0 = feature và nhãn độc lập). MI bắt được cả quan hệ phi tuyến.

Sau đó thử hai cách **chọn feature**: (1) giữ k feature có MI cao nhất, k = 1..21; (2) **L1 / Lasso**: phạt tổng |trọng số| khiến các feature yếu bị đưa trọng số về đúng 0.
**Tại sao:** ít feature hơn thì model đơn giản và dễ giải thích hơn, và cho biết thông tin thật sự nằm ở đâu.

| mi_rank | feature | mutual_info | l1_entry_C |
|---|---|---|---|
| 1 | tld_len | 0.206 | 0.001 |
| 2 | dot_cnt | 0.145 | 0.001 |
| 3 | letter_ratio | 0.123 | 0.00464 |
| 4 | entropy | 0.099 | 0.00215 |
| 5 | digit_ratio | 0.098 | 0.0464 |
| 6 | digit_cnt | 0.097 | 0.001 |
| 7 | special_cnt | 0.083 | 0.0464 |
| 8 | spec_ratio | 0.059 | 0.00464 |
| 9 | subdom_cnt | 0.046 | 0.001 |
| 10 | url_len | 0.046 | 0.0215 |
| 11 | dom_len | 0.045 | 0.00215 |
| 12 | letter_cnt | 0.029 | 0.01 |
| 13 | is_ip | 0.004 | không vào |
| 14 | path_len | 0.002 | 0.00464 |
| 15 | qm_cnt | 0.001 | 0.1 |
| 16 | slash_cnt | 0.000 | 0.00464 |
| 17 | dash_cnt | 0.000 | 0.001 |
| 18 | amp_cnt | 0.000 | không vào |
| 19 | eq_cnt | 0.000 | không vào |
| 20 | under_cnt | 0.000 | 0.01 |
| 21 | query_len | 0.000 | không vào |

`l1_entry_C`: giá trị C nhỏ nhất (phạt mạnh nhất) mà feature vẫn có trọng số khác 0 trên đường L1. C càng nhỏ mà feature vẫn được giữ thì feature càng quan trọng.

**Đọc kết quả:**
- 5 feature có MI cao nhất: `tld_len`, `dot_cnt`, `letter_ratio`, `entropy`, `digit_ratio`. Các feature đếm tham số truy vấn (`query_len`, `eq_cnt`, `amp_cnt`…) có MI ≈ 0 vì gần như mọi URL đều không có tham số.
- **Đường cong top-k:** dùng đủ 21 feature đạt val macro-F1 0.780; tốt nhất 0.783 tại k=15. Chỉ cần **9 feature** đầu là đã cách mức tốt nhất không quá 0.005. Bước nhảy lớn nhất (+0.181 macro-F1) xảy ra ở k=9, khi thêm **`subdom_cnt`**. Đây chính là feature chứa tiền tố `www.` (www được tính là một subdomain), tức dấu vết thu thập dữ liệu đã nói ở `EXPERIMENTS.md` mục 0: phần lớn khả năng phân loại của 21 feature số đến từ nó.
- **Đường L1:** với C=0.0215, model chỉ giữ **12 feature** mà val macro-F1 = 0.782, cách mức tốt nhất của đường L1 (0.783) không quá 0.005. Các feature được giữ: `url_len`, `dom_len`, `tld_len`, `subdom_cnt`, `letter_cnt`, `digit_cnt`, `dot_cnt`, `under_cnt`, `spec_ratio`, `slash_cnt`, `entropy`, `path_len`.

Hình: `figures/feature_selection.png`. Dữ liệu: `feature_ranking.csv`, `feature_selection_curve.csv`, `l1_path.csv`.

---

## A2. PCA (phân tích thành phần chính)

**Làm gì:** PCA tìm các hướng (thành phần chính, PC) mà dữ liệu biến thiên nhiều nhất. Mỗi PC là tổ hợp tuyến tính của 21 feature; trọng số của từng feature trong một PC gọi là **loading**.
**Tại sao:** giảm chiều, khử các feature trùng thông tin, và vẽ dữ liệu 21 chiều lên mặt phẳng 2D.

### A2.1 Vì sao phải chuẩn hóa trước khi làm PCA
| preprocessing | pc1_var | pc2_var | n_for_90pct | n_for_95pct | pc1_top_features |
|---|---|---|---|---|---|
| raw (no scaling) | 81.6% | 9.2% | 2 | 3 | url_len (+0.69), letter_cnt (+0.64), dom_len (+0.31) |
| StandardScaler | 28.6% | 21.4% | 8 | 10 | url_len (+0.38), special_cnt (+0.35), letter_cnt (+0.34) |
| log1p + StandardScaler | 27.4% | 20.8% | 7 | 9 | letter_cnt (+0.39), url_len (+0.39), special_cnt (+0.33) |

Không chuẩn hóa thì PC1 chiếm 81.6% phương sai và bị chi phối bởi url_len (+0.69): PCA chỉ "thấy" feature có đơn vị lớn nhất. Sau `log1p + StandardScaler`, phương sai trải đều hơn (PC1 27.4%) và cần 7 thành phần để giữ 90% thông tin. **Đây là lý do luôn chuẩn hóa trước PCA.**

### A2.2 Các thành phần chính có ý nghĩa gì (loadings)
- **PC1**: `letter_cnt` (+0.39), `url_len` (+0.39), `special_cnt` (+0.33), `dot_cnt` (+0.30)
- **PC2**: `query_len` (+0.41), `eq_cnt` (+0.40), `qm_cnt` (+0.36), `under_cnt` (+0.34)
- **PC3**: `spec_ratio` (+0.42), `letter_ratio` (-0.37), `subdom_cnt` (+0.37), `dot_cnt` (+0.36)

### A2.3 Phương sai lớn có đồng nghĩa với hữu ích cho việc phân loại không?
| component | var_ratio | single_pc_val_auc |
|---|---|---|
| PC1 | 27.4% | 0.584 |
| PC2 | 20.8% | 0.597 |
| PC3 | 15.1% | 0.704 |
| PC4 | 9.8% | 0.640 |
| PC5 | 7.2% | 0.579 |
| PC6 | 5.5% | 0.510 |
| PC7 | 4.5% | 0.507 |
| PC8 | 3.7% | 0.605 |
| PC9 | 1.5% | 0.536 |
| PC10 | 1.2% | 0.509 |

PC phân biệt phishing tốt nhất khi đứng một mình là **PC3** (AUC 0.704), dù nó chỉ chiếm 15.1% phương sai. PCA chọn hướng theo **phương sai**, không theo **nhãn**; hướng biến thiên nhiều nhất không nhất thiết là hướng tách phishing khỏi benign. Điều này giải thích vì sao giảm chiều bằng PCA làm accuracy giảm (RQ3 trong `QUICK_RESULTS.md`).

Hình: `figures/pca_loadings.png`, `figures/pca_2d_label_source.png` (chiếu lên PC1–PC2, tô màu theo nhãn và theo nguồn). Dữ liệu: `pca_scaling_comparison.csv`, `pca_loadings.csv`, `pca_component_auc.csv`.

---

## A3. Phân cụm (K-Means, GMM)

**Làm gì:** gom URL thành k nhóm mà **không dùng nhãn**, với k = 2..10, bằng hai thuật toán:
- **K-Means:** mỗi cụm có một tâm; mỗi URL thuộc về tâm gần nhất. Lặp lại cho tới khi tâm không đổi.
- **GMM (Gaussian Mixture Model):** mỗi cụm là một phân phối chuẩn nhiều chiều (hình elip, ma trận hiệp phương sai đầy đủ); mỗi URL có xác suất thuộc từng cụm.

Cách chọn k (không cần nhãn):
- **Elbow:** inertia (tổng bình phương khoảng cách tới tâm) giảm khi k tăng; chọn chỗ đường cong "gãy khúc".
- **Silhouette** (−1..1, càng cao càng tốt): mỗi điểm gần cụm của mình hơn cụm khác bao nhiêu. Tính trên 5,000 URL lấy mẫu.
- **Davies–Bouldin** (càng thấp càng tốt), **Calinski–Harabasz** (càng cao càng tốt).
- **BIC** cho GMM (càng thấp càng tốt): cân bằng giữa độ khớp dữ liệu và số tham số.

Cách đối chiếu với nhãn (chỉ để đánh giá, không dùng khi phân cụm): **ARI** (Adjusted Rand Index, 0 = như ngẫu nhiên, 1 = khớp hoàn toàn) và **NMI** (Normalized Mutual Information, 0..1). Đối chiếu với **cả nhãn lẫn họ nguồn** dữ liệu.

### A3.1 Chọn k
| k | inertia | silhouette | davies_bouldin | calinski_harabasz | ari_label | ari_source | nmi_label | nmi_source |
|---|---|---|---|---|---|---|---|---|
| 2 | 609,061 | 0.255 | 1.612 | 7,824 | 0.022 | 0.060 | 0.010 | 0.095 |
| 3 | 499,260 | 0.267 | 1.312 | 8,831 | 0.001 | 0.047 | 0.001 | 0.097 |
| 4 | 413,703 | 0.298 | 1.134 | 9,648 | 0.008 | 0.035 | 0.068 | 0.142 |
| 5 | 357,694 | 0.300 | 0.906 | 9,814 | 0.008 | 0.035 | 0.068 | 0.142 |
| 6 | 310,286 | 0.303 | 0.880 | 10,178 | 0.007 | 0.039 | 0.068 | 0.152 |
| 7 | 273,272 | 0.272 | 0.945 | 10,463 | 0.024 | 0.059 | 0.064 | 0.149 |
| 8 | 244,831 | 0.286 | 1.015 | 10,623 | 0.027 | 0.061 | 0.062 | 0.136 |
| 9 | 223,344 | 0.291 | 1.018 | 10,632 | 0.022 | 0.046 | 0.083 | 0.161 |
| 10 | 199,156 | 0.325 | 0.961 | 11,097 | 0.073 | 0.104 | 0.159 | 0.237 |

GMM (ma trận hiệp phương sai đầy đủ):

| k | bic | silhouette | ari_label | ari_source |
|---|---|---|---|---|
| 2 | -1,945,338 | 0.435 | -0.015 | 0.009 |
| 3 | -2,254,189 | 0.216 | 0.193 | 0.225 |
| 4 | -3,029,210 | 0.267 | 0.071 | 0.112 |
| 5 | -2,696,732 | 0.227 | 0.178 | 0.225 |
| 6 | -3,308,451 | 0.225 | 0.098 | 0.141 |
| 7 | -3,271,383 | 0.261 | 0.136 | 0.185 |
| 8 | -3,445,230 | 0.268 | 0.025 | 0.056 |
| 9 | -3,404,620 | 0.275 | 0.095 | 0.124 |
| 10 | -3,595,337 | 0.275 | 0.056 | 0.088 |

- Silhouette cao nhất ở **k=10** (0.325); GMM có BIC thấp nhất ở **k=10**.
- **Cụm khớp với nguồn nhiều hơn khớp với nhãn ở 9/9 giá trị k** (ARI vs nguồn so với ARI vs nhãn, K-Means). Cấu trúc tự nhiên của dữ liệu chủ yếu phản ánh **URL đến từ danh sách nào**. Đây là cùng hiện tượng phụ thuộc nguồn đo được ở `EXPERIMENTS.md`.

### A3.2 Hồ sơ từng cụm (K-Means, k=6)
k được chọn là k có silhouette cao nhất trong khoảng 4..8, đủ chi tiết để diễn giải. Cột "Mô tả tự động" được sinh từ trung vị và tỷ lệ của cụm so với toàn bộ train.

| Cụm | Số URL (tỷ lệ) | % phishing | Nguồn chính (tỷ lệ) | `www.` | .vn | Mô tả tự động |
|---|---|---|---|---|---|---|
| 1 | 15,380 (41.7%) | 63.9% | chongluadao (59.1%) | 52.8% | 35.6% | có `www.`, chỉ domain |
| 0 | 13,241 (35.9%) | 61.5% | chongluadao (52.8%) | 0.1% | 12.8% | không `www.`, chỉ domain |
| 3 | 8,154 (22.1%) | 95.7% | chongluadao (92.5%) | 46.1% | 1.7% | không phải .vn, chỉ domain, nhiều chữ số |
| 2 | 102 (0.3%) | 92.2% | openphish (83.3%) | 6.9% | 4.9% | không `www.`, không phải .vn, có đường dẫn, URL dài |
| 5 | 27 (0.1%) | 100.0% | openphish (100.0%) | 29.6% | 0.0% | không phải .vn, chỉ domain, có tham số truy vấn, nhiều chữ số, nhiều dấu gạch ngang, URL dài |
| 4 | 5 (0.0%) | 100.0% | openphish (100.0%) | 100.0% | 0.0% | có `www.`, không phải .vn, có đường dẫn, có tham số truy vấn, nhiều chữ số, nhiều dấu gạch ngang, URL dài |

- Hai cụm lớn nhất (cụm 1 và 0, cộng lại 77.5% dữ liệu) khác nhau rõ nhất ở **tỷ lệ `www.`** (52.8% so với 0.1%). Khi không có nhãn, K-Means vẫn tự tìm ra đúng dấu vết thu thập mà `EXPERIMENTS.md` mục 0 đã chỉ ra.
- Trong 3 cụm chiếm ≥ 1% dữ liệu: **1 cụm gần như thuần một nhãn** (>95% hoặc <5% phishing), **1 cụm có hơn 80% URL từ cùng một họ nguồn**. Ngoài ra có 3 cụm tí hon (134 URL, chủ yếu từ openphish): K-Means tách riêng các URL ngoại lai này; xem cột mô tả trong bảng.
- Trung vị chi tiết từng feature theo cụm: `cluster_profiles.csv`.

Hình: `figures/clustering_selection.png`, `figures/cluster_profiles.png`. Dữ liệu: `clustering_selection.csv`, `cluster_profiles.csv`.

---

## Liên hệ với thí nghiệm tiền xử lý
Ảnh hưởng thật của từng kỹ thuật (cắt ngoại lai, rời rạc hóa, feature đa thức, lọc tương quan, SelectKBest, L1, PCA, under/oversampling, cụm K-Means làm feature) lên Logistic Regression, đo trên val/test/Hassan kèm kiểm định thống kê, nằm ở `EXPERIMENTS.md` mục 1d.
