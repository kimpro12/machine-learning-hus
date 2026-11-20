# Phát hiện URL phishing bằng Học máy: phân loại, PCA, phân cụm (PhishVN)

Đồ án môn Học máy, dùng dataset **PhishVN v3.1.0**: URL phishing tiếng Việt có gắn thời gian, Mendeley DOI [`10.17632/b97hxbxtpd.4`](https://doi.org/10.17632/b97hxbxtpd.4), giấy phép CC BY 4.0.

Câu hỏi chính của repo **không phải** "đạt accuracy bao nhiêu" mà là:

> Con số accuracy đo được có đúng không? Model học được gì, và hỏng ở đâu?

| | Kết quả |
|---|---|
| Model tốt nhất | Logistic Regression (C=10), 21 feature lexical |
| Test chính thức (temporal split, mở **một lần**) | **Acc 0.9026** (CI95 [0.896, 0.909]), **F1 phishing 0.9194**, ROC-AUC 0.958 |
| Dữ liệu ngoài, Hassan 340k (model đóng băng) | Acc 0.439 (−46.3pp); lớp benign sụp, specificity 0.038 |
| PCA | Làm giảm hiệu năng: 8 PCs (90% phương sai) acc 0.773 so với 0.871 khi không dùng PCA |
| Phân cụm (K-Means/GMM) | Không có cụm khớp nhãn: ARI ≈ 0 với k=2, tốt nhất 0.209 |
| Kết luận | Phù hợp làm benchmark cho đồ án; **chưa** kiểm chứng để triển khai thực tế |

> ⚠️ **BENCHMARK_NOT_REAL_WORLD_VALIDATED**: không model nào trong repo được kiểm chứng để dùng ngoài đời thực.

---

## 1. Người mới: đọc theo thứ tự này

Đọc **kết quả → dữ liệu → code**, đừng mở code trước.

### Bước 1: Kết luận (≈20 phút)
1. **`README.md`** (file này): bức tranh tổng.
2. **`outputs/DECISION.md`**: kết luận và lập luận: vì sao dataset phù hợp cho đồ án nhưng không dùng để triển khai được, lưu ý về GOLDSILVER. Đọc kỹ mục **"Vì sao macro-F1"**: nó giải thích vì sao **không chốt bằng accuracy**.
3. **`outputs/QUICK_RESULTS.md`**: số liệu theo từng bước: audit → validation → test → slice bias → external → PCA/cluster.
4. **`outputs/EXPERIMENTS.md`**: thí nghiệm bổ sung (chỉ Logistic Regression): so sánh 11 cách tiền xử lý và leave-one-source-out. **Đọc mục 0 trước**: tiền tố `www.` là dấu vết thu thập dữ liệu và là nguyên nhân cụ thể khiến model sụp trên Hassan.

### Bước 2: Hình (≈5 phút), trong `outputs/figures/`
- `confusion_internal_full.png` / `confusion_hassan_full.png`: model trên test nội bộ so với dữ liệu ngoài.
- `internal_vs_external_gap.png`: khoảng rơi từ 90% xuống 44%.
- `source_bias.png`: accuracy theo nguồn dữ liệu; nguồn tranco (benign toàn cầu) kém nhất.
- `preprocessing_ablation.png`: macro-F1 trên val / test / Hassan của từng cách tiền xử lý.
- `leave_one_source_out.png`: accuracy trên một nguồn khi nguồn đó có trong train và khi bị loại khỏi train.
- `pca_scree.png`, `cluster_pca_scatter.png`: RQ3 và RQ4.

### Bước 3: Dữ liệu (≈10 phút)
4. **`data/phishvn/README.md`**: thành phần dataset, các tier gold/silver/bronze, split chính thức.
5. **`data/phishvn/docs/schema.md`**: ý nghĩa 22 cột feature, vì sao loại `is_https`.
6. **`data/phishvn/docs/datasheet.md`**: cách gán nhãn, độ đồng thuận giữa người gán (κ), tỷ lệ nhiễu nhãn.
7. `outputs/feature_audit.csv`: cột nào được dùng để train, cột nào bị loại và lý do.

### Bước 4: Code (≈30 phút). Chỉ có 7 file, đọc theo thứ tự:
1. `src/phishing_ml/features.py`: trích 21 feature từ chuỗi URL.
2. `src/phishing_ml/pipeline.py`: **đọc hàm `run_pipeline` ở cuối file trước**. Nó gọi lần lượt từng bước, rồi bạn nhảy vào từng hàm (xem mục 4).
3. `src/phishing_ml/evaluate.py`: cách tính metric và CI.
4. `src/phishing_ml/preprocess.py`: pipeline imputer → scaler → model.
5. `src/phishing_ml/cli.py`: entry point, chỉ khoảng 30 dòng.
6. `src/phishing_ml/download.py`: tải dữ liệu ngoài (Hassan, JPCERT).
7. `src/phishing_ml/experiments.py`: thí nghiệm bổ sung. `build_variants()` liệt kê 11 cách tiền xử lý; thống kê nằm ở `mcnemar_exact`, `paired_bootstrap_macro_f1_diff`.

### Bước 5: Test (≈5 phút)
`tests/`: file ngắn, cho biết repo cam kết những gì: không leakage khi tiền xử lý, feature khớp đúng dataset, CI chính xác, grid model ổn định.

> Muốn biết vì sao **không** dùng dataset UCI PhiUSIIL (accuracy 100% nhưng là ảo)? Xem `archive/phiusiil/README.md`.

---

## 2. Quy ước nhãn

PhishVN: **`label=1` phishing, `label=0` benign**. Lớp dương luôn là **phishing**: F1, precision, recall đều tính cho lớp phishing.

---

## 3. Cấu trúc thư mục

```text
Project/
├── README.md
├── pyproject.toml              ← dependencies có pin version, cấu hình pytest/ruff
├── src/phishing_ml/
│   ├── cli.py                  ← entry point: `run [--mode quick|full]`, `fetch-external`
│   ├── pipeline.py             ← toàn bộ pipeline thực nghiệm
│   ├── features.py             ← trích 21 feature lexical (khớp đúng schema PhishVN)
│   ├── evaluate.py             ← metrics + Clopper–Pearson CI
│   ├── download.py             ← tải Hassan + JPCERT vào data/external/
│   ├── experiments.py          ← thí nghiệm bổ sung: tiền xử lý + leave-one-source-out (LogReg)
│   └── preprocess.py           ← sklearn Pipeline (chỉ fit trên train)
├── tests/
├── data/                       ← (CSV bị gitignore)
│   ├── phishvn/                ← bản phát hành PhishVN: README, docs/, MANIFEST.txt, LICENSE, data/*.csv
│   └── external/               ← hassan_balanced_urls.csv + jpcert/jpcert-2025MM.csv (chỉ để đánh giá ngoài)
├── outputs/                    ← mọi kết quả (model .joblib bị gitignore)
│   ├── DECISION.md, QUICK_RESULTS.md
│   ├── feature_audit.csv, quick_val_*.csv, classification_metrics.csv, external_metrics.csv
│   ├── test_by_{source,tier,scenario,urlslice}_*.csv
│   ├── pca_results.csv, clustering_results.csv, frozen_models.json
│   ├── EXPERIMENTS.md, preprocessing_ablation.csv, leave_one_source_out.csv,
│   │   tfidf_top_ngrams.csv, www_prefix_rates.csv
│   └── figures/
└── archive/phiusiil/           ← pipeline cũ trên PhiUSIIL (benchmark phản diện, không thuộc đề tài chính)
```

---

## 4. Pipeline: `pipeline.py` theo từng bước

`run_pipeline(mode)` trước hết gọi `missing_inputs()`. Nếu thiếu file đầu vào, nó in ra danh sách file thiếu và dừng với mã 2. Đủ dữ liệu thì chạy lần lượt:

| # | Hàm | Làm gì | Ghi ra |
|---|---|---|---|
| 1 | `load_frames` | Đọc `vn_compphish.csv` (feature) và ghép `source`/`scenario` từ `dataset_url.csv` | — |
| 2 | `run_audit` | Kiểm tra trùng URL, trùng domain giữa các split, `is_https` theo nhãn, danh sách feature được dùng | `feature_audit.csv` |
| 3 | `run_quick` | Train trên **train**, chọn model trên **val** (10 cấu hình: dummy, LogReg, RF, LinearSVC) cho 2 tập: `FULL` và `GOLDSILVER` | `quick_val_*.csv`, `frozen_models.json` |
| 4 | `run_final_test` | Fit lại model đã chọn, đánh giá **test một lần**, đóng băng model | `classification_metrics.csv`, `frozen_*.joblib`, `test_by_*.csv` |
| 5 | `run_external` | Đưa model đóng băng vào Hassan (340k, cân bằng) và JPCERT (chỉ phishing; URL đọc bằng `load_jpcert_urls`). Ngưỡng cố định 0.5, **không tune** | `external_metrics.csv` |
| 6 | `run_pca_cluster` | PCA chỉ fit trên train (RQ3); K-Means/GMM trên PCA(10) (RQ4). Bỏ qua khi `--mode quick` | `pca_results.csv`, `clustering_results.csv` |
| 7 | `run_figures` | Vẽ hình | `figures/*.png` |
| 8 | `write_reports` | Sinh báo cáo từ các CSV ở trên | `DECISION.md`, `QUICK_RESULTS.md` |
| 9 | `run_supplementary` → `experiments.py` | Thí nghiệm bổ sung chỉ dùng LogReg (xem mục 6). Bỏ qua khi `--mode quick` | `EXPERIMENTS.md` + 4 CSV + 2 hình |

Hàm phụ đáng biết: `grids()` (danh sách model + siêu tham số), `norm_key()` (chuẩn hóa URL để ghép bảng), `_slice_frame()` (metric cho từng slice, kèm `tiny_n_flag`).

---

## 5. Cài đặt và chạy

Yêu cầu: Python ≥ 3.11, chỉ cần CPU. `run --mode quick` mất khoảng 30–60 giây; `run` đầy đủ mất khoảng 6 phút, phần lớn cho thí nghiệm bổ sung (TF-IDF trên 340k URL Hassan).

```bash
python -m venv .venv
# Windows:  .venv\Scripts\Activate.ps1      Linux/macOS:  source .venv/bin/activate
pip install -e ".[dev]"
```

### 5.1 Chuẩn bị dữ liệu (không có trong git)
1. **PhishVN**: tải bản open tier từ Mendeley (DOI `10.17632/b97hxbxtpd.4`), giải nén sao cho có `data/phishvn/data/vn_compphish.csv`, `dataset_url.csv`, `splits/`, ... Kiểm tra hash bằng `data/phishvn/MANIFEST.txt`.
2. **Dữ liệu ngoài**: tải tự động bằng
   ```bash
   python -m phishing_ml.cli fetch-external
   ```
   Lệnh này tải `data/external/hassan_balanced_urls.csv` (từ [Hassan-Albattra/URL_Phishing_Detection_Dataset](https://github.com/Hassan-Albattra/URL_Phishing_Detection_Dataset), MIT, khoảng 31 MB, sha256 bắt đầu `8a864a7f`) và `data/external/jpcert/jpcert-202501..202506.csv` (từ [JPCERTCC/phishurl-list](https://github.com/JPCERTCC/phishurl-list)). File nào đã có thì bỏ qua.

### 5.2 Chạy
```bash
python -m phishing_ml.cli run               # đầy đủ (gồm PCA + clustering + thí nghiệm bổ sung)
python -m phishing_ml.cli run --mode quick  # bỏ PCA/clustering và thí nghiệm bổ sung
python -m phishing_ml.cli experiments       # chỉ chạy thí nghiệm bổ sung (≈4–5 phút)
```
Mọi kết quả được ghi đè vào `outputs/`. Pipeline tất định (seed 42), nên chạy lại sẽ ra cùng số liệu (chỉ cột thời gian `fit_s` thay đổi). Nếu thiếu dữ liệu, lệnh sẽ in `BLOCKED` kèm danh sách file thiếu.

### 5.3 Kiểm thử và lint
```bash
pytest -q
ruff check .
```

---

## 6. Phương pháp

### Feature
- **21 feature lexical** (`PHISHVN_FEATURES`): độ dài URL/domain/path/query, số chữ/số/ký tự đặc biệt, số dấu `. - _ / = ? &`, tỷ lệ ký tự, entropy, số subdomain, độ dài TLD, URL có phải IP không.
- Tính trên URL **đã bỏ scheme** (`http://`, `https://`), theo đúng schema CompPhish.
- **Loại `is_https`**: đây là dấu vết của cách thu thập (benign 12.3% so với phishing 0.55%), không phải tín hiệu phishing. Tác giả dataset cũng khuyến nghị bỏ.
- **Loại mọi metadata**: ID, source, tier, timestamp, scenario. Đưa vào là leakage.

### Chia dữ liệu và luật chống leakage
- Dùng **split temporal chính thức** của tác giả (train 36,909 / val 7,266 / test 8,941). Split chia theo nhóm, không domain nào xuất hiện ở hai split; test là dữ liệu mới nhất.
- Imputer và scaler nằm trong `Pipeline` nên **chỉ fit trên train**.
- **Validation** dùng để chọn model và siêu tham số. **Test chỉ mở một lần** sau khi đã chọn xong.
- Dữ liệu ngoài: model đóng băng, ngưỡng 0.5, nhãn ngoài chỉ dùng để chấm điểm.

### Hai tập con
- **`FULL`**: mọi tier, là **kết quả chính**.
- **`GOLDSILVER`**: chỉ nhãn tin cậy cao. Val chỉ có 24 phishing nên việc chọn model không ổn định. Dùng tập này để nghiên cứu nhiễu nhãn và drift, không dùng làm con số chính.

### Đọc metric thế nào
- **So với dummy baseline trước.** Ví dụ GOLDSILVER: dummy (luôn đoán benign) đạt acc 0.985, cao hơn RF 0.967, nhưng F1 phishing = 0.
- **Chốt bằng F1 phishing + macro-F1**, không chốt bằng accuracy khi lệch lớp.
- CI của accuracy dùng **Clopper–Pearson** (chính xác theo phân phối nhị thức).
- Slice có `tiny_n_flag=True` (n < 30) thì **không diễn giải**.
- JPCERT chỉ có phishing nên chỉ báo `recall_phishonly`. Nó **không đo được false positive**, mà false positive lại chính là chỗ model hỏng.

### Thí nghiệm bổ sung (`experiments.py`, chỉ Logistic Regression C=10, tập FULL)
Chạy **sau** benchmark chính và trình bày như một thí nghiệm thứ hai: chọn trên val, test và dữ liệu ngoài chỉ để chấm, không thay model chính.
- **So sánh tiền xử lý** (11 biến thể): StandardScaler (baseline), log1p, QuantileTransformer, `class_weight=balanced`, ngưỡng chọn trên val, TF-IDF n-gram ký tự (2–5) trên chuỗi URL, TF-IDF + feature số, và các bản **bỏ tiền tố `www.`** (`*_nowww`). Thống kê so với baseline trên test: **McNemar chính xác** và **CI bootstrap ghép cặp** (1000 lần) cho Δ macro-F1.
- **Leave-one-source-out**: mỗi nguồn PhishVN chỉ có một nhãn. Gom thành 5 họ nguồn, bỏ từng họ khỏi train rồi đo trên val+test của chính họ đó.

Kết quả chính:

| Phát hiện | Số liệu |
|---|---|
| `www.` là dấu vết thu thập | Train PhishVN: benign **0.6%** có `www.`, phishing **45.7%**. Hassan: URL hợp lệ **72%** có `www.` |
| TF-IDF n-gram mạnh nhất trong nội bộ | Test macro-F1 0.898 → **0.938** (Δ +0.040 [+0.033, +0.046], McNemar p ≈ 1e-28) |
| Bỏ `www.` làm điểm nội bộ giảm nhưng tổng quát hóa tăng | `url_tfidf` → `url_tfidf_nowww`: Hassan specificity 0.22 → **0.45**, macro-F1 0.46 → **0.58** (tốt nhất trong các biến thể) |
| Đổi cách scale không giúp gì | log1p: Δ ≈ 0 (p=0.41); QuantileTransformer: kém hơn (Δ −0.015, p ≈ 2e-19) |
| `class_weight` chỉ dịch ngưỡng | Chọn lại ngưỡng trên val thì bằng baseline |
| Model phụ thuộc nguồn | Benign từ nguồn chưa thấy chỉ nhận đúng 7–23%; bỏ `www.` thì recall chongluadao (khi bị loại khỏi train) rơi 0.62 → 0.28 |

---

## 7. Trả lời các câu hỏi nghiên cứu

| RQ | Câu hỏi | Kết quả |
|---|---|---|
| **RQ1** | So sánh LogReg / RF / LinearSVC trên cùng split | LogReg C=10 tốt nhất trên val (F1 0.9215) → test F1 **0.9194**, acc **0.9026** |
| **RQ2** | Hiệu năng khi gặp nguồn/tier/dữ liệu chưa thấy | Nguồn tranco chỉ đạt 0.297, tier silver 0.601; Hassan **0.439** (−46.3pp). Nguyên nhân đo được: `www.` (benign PhishVN 0.6% so với Hassan 72%) và chữ ký nguồn (leave-one-source-out: benign từ nguồn chưa thấy chỉ đúng 7–23%). Xem `EXPERIMENTS.md` |
| **RQ3** | PCA ảnh hưởng thế nào | Giảm hiệu năng: var90 (8 PCs) acc 0.773 / F1 0.863 so với 0.871 / 0.921 khi không dùng PCA. Nên giữ đủ feature |
| **RQ4** | K-Means/GMM có cụm khớp nhãn không | Không: ARI ≈ 0 với k=2, tốt nhất 0.209 (GMM diag k=4). Chỉ mang tính mô tả |

Kết quả âm (PCA làm giảm hiệu năng, cluster không khớp nhãn, sụp trên dữ liệu ngoài, scale không giúp gì) **là kết quả đáng báo cáo**, không phải lỗi.

---

## 8. Câu hỏi thường gặp

- **Muốn thêm model?** Thêm vào `grids()` trong `pipeline.py`. Chỉ chọn trên val, không chỉnh model sau khi đã xem test.
- **Có được chạy lại test không?** Chạy lại để tái lập kết quả thì được (kết quả tất định). Nhưng **không được sửa model dựa trên kết quả test** rồi báo cáo lại trên cùng test đó như kết quả không thiên lệch.
- **Sao accuracy chỉ 90% mà vẫn "tốt"?** Vì đó là con số trung thực: split theo thời gian, không trùng domain, đã bỏ `is_https`. Dataset UCI PhiUSIIL cho 100% nhưng ra dữ liệu thật chỉ còn 50% (xem `archive/phiusiil/`).
- **Sao không dùng PhiUSIIL?** Vì có leakage (`URLSimilarityIndex`) và collection bias (legit luôn là homepage `https://www...`). Chi tiết ở `archive/phiusiil/README.md`; code gốc ở git tag `phiusiil-baseline`.

---

## 9. Nguồn dữ liệu
- **PhishVN v3.1.0** (CC BY 4.0): Mendeley DOI `10.17632/b97hxbxtpd.4`; trích dẫn theo `data/phishvn/CITATION.cff`.
- **Hassan-Albattra `balanced_urls.csv`** (MIT, 340k URL, 2023): chỉ dùng để đánh giá ngoài.
- **JPCERTCC `phishurl-list`** 2025-01..06: phishing do CERT xác minh, chỉ dùng để đo recall.
