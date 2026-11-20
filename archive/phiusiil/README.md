# [Archive] PhiUSIIL (UCI 967): benchmark phản diện

> Đây là **phần lưu trữ**, không thuộc đề tài chính. Đề tài đã đăng ký dùng **PhishVN** (xem `README.md` ở thư mục gốc).

Đây là pipeline đầu tiên của dự án, chạy trên PhiUSIIL Phishing URL Dataset (UCI ID 967). Kết luận của nó là lý do dự án chuyển sang PhishVN:

| | Kết quả |
|---|---|
| Accuracy nội bộ (random split) | **100%** (FULL_NUMERIC), 99.8% (chỉ dùng feature từ URL) |
| Trên dữ liệu ngoài Hassan 340k | **50.0%**, ngang đoán bừa (−49.8pp) |
| URL legit có path trên Hassan | đúng **0 / 165,085** |
| Nguyên nhân | **Leakage**: `URLSimilarityIndex` = 100 với mọi URL legit, một decision stump đã đạt 99.7%. **Collection bias**: legit luôn là homepage `https://www...` không có path |
| Quyết định | `KEEP_AS_BIAS_STUDY` → chỉ dùng để minh họa leakage, `CHANGE_DATASET` |

Dùng phần này khi bị hỏi *"Sao không dùng dataset UCI phổ biến?"*.

## Nên đọc gì
1. `outputs/DECISION.md`: toàn bộ lập luận, số liệu, CI chính xác, subgroup.
2. Hình: `outputs/figures/path_https_www_by_label.png`, `single_feature_stump.png`, `external_subgroup_slices.png`, `random_vs_domain.png`.
3. `PHIUSIIL_AGENT_TECH_SPEC.md`: spec gốc (RQ1–RQ4, luật chống leakage, chính sách GO/NO-GO). Phần lớn nguyên tắc trong spec vẫn áp dụng cho pipeline PhishVN.

Lưu ý: `outputs/figures/` ở đây là hình của PhiUSIIL. Các file `pca_scree.png`, `cluster_pca_scatter.png`, `internal_vs_external_gap.png` trùng tên với hình PhishVN ở thư mục gốc nhưng là hai hình khác nhau.

## Chạy lại (nếu cần)
Cách sạch nhất là checkout tag chứa nguyên trạng repo trước khi lưu trữ:
```bash
git checkout phiusiil-baseline
```
Hoặc chạy ngay trong thư mục này. Code ở đây tự chứa đủ, kể cả bản sao `evaluate.py`/`preprocess.py`; dữ liệu đặt ở `archive/phiusiil/data/` và đã gitignore:
```bash
cd archive/phiusiil
pip install -r requirements.txt
PYTHONPATH=src python -m phishing_ml.cli run --mode full    # PowerShell: $env:PYTHONPATH="src"; python -m phishing_ml.cli run --mode full
PYTHONPATH=src python -m pytest -q tests
```
Các lệnh: `fetch`, `audit`, `quick`, `finalize`, `full`, `report`, `run` (có `--dry-run`). Kết quả đánh giá ngoài trong `outputs/external/` được sinh bằng các hàm trong `src/phishing_ml/external.py`; CLI không có lệnh riêng cho bước này.

## Cấu trúc
```text
archive/phiusiil/
├── PHIUSIIL_AGENT_TECH_SPEC.md
├── configs/          quick.yaml, full.yaml, feature_sets.yaml
├── src/phishing_ml/  fetch, schema, url_features, split, audit, quick, robustness,
│                     pca_experiment, clustering, external, decision, reporting, models, cli
├── tests/
└── outputs/          DECISION.md, SUMMARY.md, audit/, metrics/, external/, figures/, manifests/
```
Quy ước nhãn PhiUSIIL (ngược với PhishVN): gốc `label=0` phishing / `1` legitimate → nội bộ `y=1` phishing.
