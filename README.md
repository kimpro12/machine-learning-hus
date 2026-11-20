# Phát hiện URL phishing bằng Học máy: phân loại, PCA, phân cụm

Đồ án môn Học máy. Câu hỏi chính của repo **không phải** "đạt accuracy bao nhiêu" mà là:

> Con số accuracy đo được có đúng không? Dataset có đáng tin để làm đề tài không?

Repo có **hai pipeline** chạy trên hai dataset, và kết luận nằm ở chỗ so sánh hai cái:

| | PhiUSIIL (UCI 967) | PhishVN v3.1.0 |
|---|---|---|
| Vai trò | **Ví dụ phản diện** (negative benchmark): ví dụ về leakage và collection bias | **Dataset chính** của đồ án |
| Accuracy nội bộ | 99–100% (là ảo, do leakage và bias) | ~90% (temporal split sạch) |
| Trên dữ liệu ngoài (Hassan 340k) | 50.0%, ngang đoán bừa | 43.9%, vẫn sụp ở lớp benign |
| Kết luận | `KEEP_AS_BIAS_STUDY`: chỉ giữ để minh họa leakage | `ADOPT`: dùng cho đồ án, **không** tuyên bố dùng được ngoài đời thực |

> ⚠️ **BENCHMARK_NOT_REAL_WORLD_VALIDATED**: không model nào trong repo được kiểm chứng để triển khai thực tế.

---

## 1. Người mới: đọc theo thứ tự này

Đừng đọc code trước. Hãy đọc **kết quả → lý do → code**.

### Bước 1: Hiểu đề bài và luật chơi (≈15 phút)
1. **`README.md`** (file này): bức tranh tổng.
2. **`PHIUSIIL_AGENT_TECH_SPEC.md`**: spec gốc của dự án (tiếng Việt). Nên đọc kỹ:
   - §0: các câu hỏi nghiên cứu RQ1–RQ4 và quy ước nhãn.
   - §1: luật chống leakage (chỉ fit trên train, test chỉ mở một lần).
   - §5–§7: các bộ feature, audit bias, cách chia dữ liệu.
   - §10: chính sách GO / NO-GO.
   - Các mục còn lại (§2–3, §13–16) là yêu cầu kỹ thuật, khi cần thì tra.

### Bước 2: Đọc kết luận (≈20 phút)
3. **`outputs/pivot/DECISION.md`**: kết luận cho **PhishVN** (dataset chính). Đọc kỹ mục *"Vì sao macro-F1"* để hiểu vì sao không chốt bằng accuracy.
4. **`outputs/pivot/QUICK_RESULTS.md`**: số liệu PhishVN theo từng bước: audit → val → test → slice → external → PCA/cluster.
5. **`outputs/DECISION.md`**: kết luận cho **PhiUSIIL**: vì sao 100% là ảo, kết quả audit leakage, validation ngoài, subgroup.
6. **`outputs/SUMMARY.md`**: tóm tắt ngắn quick benchmark của PhiUSIIL.

### Bước 3: Xem hình (≈10 phút)
Xem các hình kể câu chuyện chính trước:
- `outputs/figures/path_https_www_by_label.png`: collection bias của PhiUSIIL (legit 100% https + www + không có path).
- `outputs/figures/single_feature_stump.png`: chỉ một feature `URLSimilarityIndex` đã đạt 99.7%, tức là lộ nhãn.
- `outputs/figures/internal_vs_external_gap.png` và `external_subgroup_slices.png`: 100% → 50% khi gặp dữ liệu thật.
- `outputs/pivot_figures/pivot_internal_vs_external_gap.png`, `pivot_source_bias.png`: cùng bệnh trên PhishVN nhưng nhẹ hơn.
- `outputs/pivot_figures/pivot_pca_scree.png`, `pivot_cluster_pca_scatter.png`: PCA làm giảm hiệu năng, cluster không khớp nhãn.

### Bước 4: Hiểu dữ liệu (≈10 phút)
7. **`data/pivot/README.md`**, **`data/pivot/docs/datasheet.md`**, **`data/pivot/docs/schema.md`**: PhishVN gồm những gì, ý nghĩa các tier gold/silver/bronze, vì sao loại `is_https`.
8. **`configs/feature_sets.yaml`**: danh sách feature nghi rò rỉ / proxy thu thập của PhiUSIIL.
9. `outputs/audit/suspicious_features.md`, `outputs/audit/dedup_reconciliation.md`: chi tiết audit PhiUSIIL.

### Bước 5: Đọc code (theo luồng dữ liệu, xem mục 4)
- Muốn hiểu **PhishVN**: `pivot.py` → `pivot_train.py` (hàm `run_pivot` ở cuối file là entry point, đọc từ dưới lên).
- Muốn hiểu **PhiUSIIL**: `cli.py` (bắt đầu từ `main()` và `cmd_run`) → `schema.py` → `url_features.py` → `split.py` → `audit.py` → `quick.py` → `evaluate.py` → `decision.py`.
- Dùng chung: `evaluate.py` (metrics, CI), `preprocess.py` (pipeline chỉ fit trên train).

### Bước 6: Đọc test để thấy các bất biến được kiểm tra
`tests/test_split_no_leakage.py`, `test_preprocess_train_only.py`, `test_label_mapping.py`, `test_exact_ci.py`, `test_pivot.py`. Các file này ngắn (dưới 65 dòng) và là cách nhanh nhất để biết repo cam kết những gì.

---

## 2. Quy ước nhãn (quan trọng, dễ nhầm)

| Dataset | Nhãn gốc | Nhãn nội bộ dùng khi train |
|---|---|---|
| PhiUSIIL | `label=0` phishing, `label=1` legitimate (**ngược**) | `y=1` phishing, `y=0` legitimate (đổi trong `schema.recode_label`) |
| PhishVN | `label=1` phishing, `label=0` benign | giữ nguyên |

**Lớp dương (positive) luôn là phishing.** Mọi F1, precision, recall đều tính cho lớp phishing.

---

## 3. Cấu trúc thư mục

```text
Project/
├── README.md                       ← bạn đang ở đây
├── PHIUSIIL_AGENT_TECH_SPEC.md     ← spec gốc (RQ, luật leakage, GO/NO-GO)
├── pyproject.toml                  ← dependencies có pin version, cấu hình pytest/ruff
├── configs/
│   ├── quick.yaml                  ← seed, tỷ lệ split, lưới siêu tham số, ngưỡng quyết định
│   ├── full.yaml                   ← như quick + PCA, clustering, split sensitivity
│   └── feature_sets.yaml           ← SUSPECT_DERIVED / TEXT_META / COLLECTION_PROXY
├── src/phishing_ml/                ← toàn bộ code (xem mục 4)
├── tests/                          ← pytest, dùng dữ liệu tổng hợp
├── data/                           ← (gitignored) dữ liệu
│   ├── raw/                        ← CSV gốc PhiUSIIL
│   ├── processed/                  ← clean.parquet, splits.pkl, feature_sets.json (sinh tự động)
│   ├── external/                   ← Hassan 340k + JPCERT 2025 (chỉ dùng để đánh giá ngoài)
│   └── pivot/                      ← bản phát hành PhishVN (CSV, splits, docs, MANIFEST)
├── raw_dl/, phiusiil.zip           ← bản tải về dự phòng của PhiUSIIL
└── outputs/                        ← (gitignored trừ *.md) mọi kết quả
    ├── DECISION.md, SUMMARY.md, decision.json   ← kết luận PhiUSIIL
    ├── audit/                      ← audit chất lượng dữ liệu và bias
    ├── metrics/                    ← quick_validation, final_*_test, pca, clustering, ...
    ├── external/                   ← PhiUSIIL-model trên Hassan/JPCERT + subgroup
    ├── figures/                    ← hình PhiUSIIL
    ├── manifests/                  ← hash dữ liệu, frozen_plan, test_evaluated (bằng chứng test chỉ mở 1 lần)
    ├── models/                     ← model đã đóng băng (.joblib)
    ├── pivot/                      ← toàn bộ kết quả PhishVN + DECISION.md, QUICK_RESULTS.md
    └── pivot_figures/              ← hình PhishVN
```

---

## 4. Bản đồ code: `src/phishing_ml/`

### Dùng chung
| File | Làm gì | Hàm chính |
|---|---|---|
| `cli.py` | Entry point duy nhất, gồm mọi lệnh `fetch/audit/quick/finalize/full/report/run/pivot` | `main`, `cmd_*`, `load_or_build_processed` |
| `evaluate.py` | Metrics (acc, F1 phishing, ROC/PR-AUC, balanced acc), bootstrap CI, **Clopper–Pearson exact CI**, tune ngưỡng | `compute_metrics`, `bootstrap_ci`, `clopper_pearson_ci` |
| `preprocess.py` | Pipeline sklearn (imputer + scaler) để **bảo đảm chỉ fit trên train** | `make_pipeline`, `get_xy` |
| `models.py` | Sinh danh sách model + lưới siêu tham số từ config | `build_configs` |

### Pipeline PhiUSIIL (bài học về leakage)
| File | Làm gì |
|---|---|
| `fetch.py` | Tải ZIP từ UCI, giải nén, ghi SHA-256 vào manifest |
| `schema.py` | Kiểm tra schema, đổi nhãn, định nghĩa `SUSPECT_DERIVED` (5 feature lộ nhãn) |
| `url_features.py` | Tính lại 28 feature **chỉ từ chuỗi URL** (`STRICT_COLS`); `NO_CUES_COLS` = bỏ các proxy thu thập; chuẩn hóa URL, lấy eTLD+1 |
| `split.py` | Khử trùng lặp, chia random stratified 64/16/20, chia **domain-disjoint** theo eTLD+1 |
| `audit.py` | Audit bias: tỷ lệ https/www/path theo nhãn, decision stump một feature, tương quan |
| `quick.py` | Quick benchmark: train → val cho 4 feature set × các model; đo gap random vs domain |
| `robustness.py` | Hình random vs domain, độ nhạy theo tỷ lệ split và seed |
| `pca_experiment.py` | RQ3: PCA theo % phương sai / số thành phần cố định |
| `clustering.py` | RQ4: K-Means + GMM, silhouette/DB/CH/ARI |
| `external.py` | Tải và đánh giá model đóng băng trên Hassan và JPCERT (không tune trên nhãn ngoài) |
| `decision.py` | Logic GO/NO-GO theo hai trục: học thuật và thực tế → `decision.json`, `DECISION.md` |
| `reporting.py` | Sinh `SUMMARY.md` và các hình còn thiếu (confusion, ROC/PR, ...) |

### Pipeline PhishVN (dataset chính)
| File | Làm gì |
|---|---|
| `pivot.py` | Bộ trích 21 feature **khớp đúng** schema PhishVN/CompPhish (URL bỏ scheme, quy tắc TLD `.vn`, loại `is_https`). Hằng `PHISHVN_FEATURES` |
| `pivot_train.py` | Toàn bộ pipeline: `load_frames` → `run_audit` → `run_quick` (chọn trên val) → `run_final_test` (mở test 1 lần) → `run_external` → `run_pca_cluster` → `run_figures` → `write_reports`. Entry: `run_pivot` |

> Lưu ý: `outputs/external/` của PhiUSIIL được sinh bằng các hàm trong `external.py`; chưa có lệnh CLI riêng cho bước này. Còn `pivot` tự đọc `data/external/hassan_balanced_urls.csv` và JPCERT nếu các file đó có sẵn.

---

## 5. Cài đặt và chạy

Yêu cầu: Python ≥ 3.11, chỉ cần CPU. Trên Windows nên dùng PowerShell.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\Activate.ps1      Linux/macOS:  source .venv/bin/activate
pip install -e ".[dev]"
```

### 5.1 PhishVN (dataset chính), nên chạy cái này trước
Cần sẵn `data/pivot/data/*.csv` (giải nén từ `data/pivot/PhishVN_open.zip`; kiểm tra hash bằng `MANIFEST.txt`).
```bash
python -m phishing_ml.cli pivot              # đầy đủ: audit → quick → test → external → PCA/cluster → hình → báo cáo
python -m phishing_ml.cli pivot --mode quick # bỏ qua PCA/clustering
```
Kết quả nằm ở `outputs/pivot/` và `outputs/pivot_figures/`.

### 5.2 PhiUSIIL (benchmark phản diện)
Chạy từng bước:
```bash
python -m phishing_ml.cli fetch                               # tải từ UCI (hoặc dùng data/raw, raw_dl/, phiusiil.zip nếu có)
python -m phishing_ml.cli audit                               # dedup, tính feature URL, split, audit bias
python -m phishing_ml.cli quick    --config configs/quick.yaml # train → val (chưa đụng test)
python -m phishing_ml.cli finalize --config configs/quick.yaml # đóng băng plan, mở test MỘT lần, ra quyết định
python -m phishing_ml.cli full     --config configs/full.yaml  # PCA, clustering, split sensitivity
python -m phishing_ml.cli report                              # dựng lại SUMMARY.md và hình
```
Hoặc chạy gộp:
```bash
python -m phishing_ml.cli run --mode quick   # fetch → audit → quick → finalize → report
python -m phishing_ml.cli run --mode full    # như trên + full
```
Có cờ `--dry-run` cho `quick/finalize/full/run` để kiểm tra dữ liệu và split mà không train.

### 5.3 Kiểm thử và lint
```bash
pytest -q
ruff check .
```

**Nếu RAM thấp:** đặt `train_subset_n: 50000` trong `configs/quick.yaml`. Giảm `n_jobs` nếu máy bị treo.

---

## 6. Phương pháp (tóm tắt để đọc báo cáo cho đúng)

### Luật chống leakage (áp dụng cho cả hai pipeline)
- Imputer, scaler, PCA **chỉ fit trên train** (bọc trong `Pipeline`, có test kiểm tra).
- **Validation** dùng để chọn model, siêu tham số và ngưỡng.
- **Test bị khóa**: chỉ mở một lần sau khi đã đóng băng plan (`outputs/manifests/frozen_plan.json`, `test_evaluated.json`).
- Dữ liệu ngoài (Hassan, JPCERT): dùng model đóng băng, ngưỡng cố định 0.5, **không bao giờ tune** trên nhãn ngoài.

### PhiUSIIL: 4 bộ feature (ablation cho RQ2)
| Feature set | Nội dung | Ý nghĩa |
|---|---|---|
| `FULL_NUMERIC` | Mọi cột số | Lạc quan quá mức, chứa feature lộ nhãn |
| `NO_SUSPECT_DERIVED` | FULL bỏ `URLSimilarityIndex`, `TLDLegitimateProb`, `URLCharProb`, `DomainTitleMatchScore`, `URLTitleMatchScore` | Bỏ leakage trực tiếp |
| `URL_LEXICAL_STRICT` | 28 feature tính lại từ chuỗi URL, không dùng nhãn/HTML/WHOIS | Thứ có thể trích ra lúc triển khai |
| `STRICT_NO_COLLECTION_CUES` | STRICT bỏ `has_www`, `is_https`, `path_*`, `query_*`, ... | Bỏ thêm các dấu vết của cách thu thập dữ liệu |

Splits: random stratified 64/16/20 (seed 42) **và** domain-disjoint theo eTLD+1.

### PhishVN
- 21 feature lexical (`PHISHVN_FEATURES`), loại `is_https` và mọi metadata (source, tier, timestamp, scenario).
- Dùng **split temporal chính thức** của tác giả (train 36909 / val 7266 / test 8941), không có domain nào trùng giữa các split.
- Hai tập con: `FULL` (mọi tier, là headline) và `GOLDSILVER` (chỉ nhãn tin cậy cao, dùng để nghiên cứu nhiễu nhãn).
- Có slice theo source / tier / scenario / URL (`test_by_*.csv`). Slice có `tiny_n_flag` (n<30) thì **không diễn giải**.

### Đọc metric thế nào
- So với **dummy baseline** trước. Ví dụ GOLDSILVER: dummy acc 0.985 > RF 0.967, nhưng F1 phishing của dummy = 0.
- Chốt bằng **F1 phishing + macro-F1**, không chốt bằng accuracy khi lệch lớp.
- Accuracy 100% thì dùng **CI chính xác Clopper–Pearson** (ví dụ [0.99992, 1.0]), không dùng bootstrap [1.0, 1.0].
- JPCERT chỉ có phishing nên chỉ báo `recall_phishonly`. Nó **không đo được false positive**, mà đó chính là chỗ cả hai dataset hỏng.

---

## 7. Kết quả chính (trả lời RQ1–RQ4)

| RQ | PhiUSIIL | PhishVN |
|---|---|---|
| **RQ1** So sánh model | RF thắng, acc 1.000 (FULL) / 0.998 (STRICT) | LogReg C=10 thắng: test acc **0.9026**, F1 **0.9194**, ROC 0.958 |
| **RQ2** Bỏ feature nghi vấn / domain chưa gặp / dữ liệu ngoài | Nội bộ gần như không giảm, nhưng Hassan **0.500** (−49.8pp); legit có path đúng 0/165085 | Hassan **0.439** (−46.3pp), specificity 0.038; nguồn tranco chỉ đạt 0.297 |
| **RQ3** PCA | `outputs/metrics/pca_results.csv` | PCA làm giảm: 8 PCs (90% var) acc 0.773 so với 0.871 khi không dùng PCA |
| **RQ4** Clustering | `outputs/metrics/clustering_metrics.csv` | ARI ≈ 0 với k=2, tốt nhất 0.209 (GMM k=4); không có cụm khớp nhãn |

Số liệu và giải thích đầy đủ nằm trong các file `DECISION.md` ở mục 1.

---

## 8. Câu hỏi thường gặp

- **Sao không dùng PhiUSIIL khi nó được 100%?** Vì 100% đến từ leakage (`URLSimilarityIndex` = 100 với mọi URL legit) và collection bias (legit luôn là homepage `https://www...`). Gặp URL thật có path, model đoán phishing hết.
- **Sao bỏ `is_https`?** Đây là dấu vết của cách thu thập dữ liệu, không phải tín hiệu phishing. Chính tác giả PhishVN khuyến nghị bỏ.
- **Muốn thêm model mới?** Với PhiUSIIL: thêm vào `models.py` + `configs/*.yaml`. Với PhishVN: thêm vào `grids()` trong `pivot_train.py`. Chỉ chọn trên val, không chỉnh sau khi đã xem test.
- **Có được chạy lại test không?** Chạy lại để tái lập kết quả thì được. Nhưng **không được sửa model dựa trên kết quả test** rồi báo cáo lại trên cùng test đó như kết quả không thiên lệch.

---

## 9. Nguồn dữ liệu
- PhiUSIIL: UCI ID 967, <https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset>
- PhishVN v3.1.0 (CC BY 4.0): Mendeley DOI `10.17632/b97hxbxtpd.4`, xem `data/pivot/CITATION.cff`
- Hassan-Albattra `balanced_urls.csv` (MIT, 340k URL, 2023): chỉ dùng để đánh giá ngoài
- JPCERTCC `phishurl-list` 2025-01..06: phishing do CERT xác minh, chỉ dùng để đo recall
