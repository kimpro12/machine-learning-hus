# Phát hiện URL phishing bằng Học máy (dataset PhishVN)

Đồ án môn Học máy. Câu hỏi của đồ án: **chỉ nhìn vào chuỗi URL, máy có đoán được một đường link là lừa đảo (phishing) hay không?** Đồ án trả lời bằng ba nhóm kỹ thuật được dạy trong môn: **phân loại** (Logistic Regression, Random Forest, Linear SVM), **giảm chiều** (PCA) và **phân cụm** (K-Means, GMM).

README này viết cho người **chưa biết gì về repo**. Đọc từ trên xuống, bạn sẽ hiểu dữ liệu trông thế nào, mỗi bước xử lý làm gì và tại sao, model hoạt động ra sao, và nên tin con số nào. Các URL ví dụ ở [mục 2.5](#25-bộ-ví-dụ-dùng-xuyên-suốt-readme) được dùng lại ở mọi bước.

> ⚠️ Đây là bài tập nghiên cứu, **không phải công cụ chặn phishing để dùng thật**. Model chưa được kiểm chứng ngoài đời thực (xem [mục 7](#7-kết-quả)).

---

## Mục lục
1. [Bài toán](#1-bài-toán)
2. [Dữ liệu](#2-dữ-liệu)
3. [Xử lý dữ liệu](#3-xử-lý-dữ-liệu-từng-bước)
4. [Phân tích dữ liệu: chọn feature, PCA, phân cụm](#4-phân-tích-dữ-liệu-chọn-feature-pca-phân-cụm)
5. [Mô hình](#5-mô-hình)
6. [Đánh giá thế nào cho đúng](#6-đánh-giá-thế-nào-cho-đúng)
7. [Kết quả](#7-kết-quả)
8. [Cấu trúc repo và bản đồ code](#8-cấu-trúc-repo-và-bản-đồ-code)
9. [Cài đặt và chạy](#9-cài-đặt-và-chạy)
10. [Đọc tiếp ở đâu](#10-đọc-tiếp-ở-đâu)
11. [Câu hỏi thường gặp](#11-câu-hỏi-thường-gặp)

---

## 1. Bài toán

**Phishing** là trang web giả mạo (ngân hàng, cơ quan nhà nước, sàn thương mại…) dụ người dùng nhập mật khẩu, mã OTP hoặc chuyển tiền. Kẻ gian thường gửi link qua SMS, Zalo hay Facebook.

Đồ án coi đây là bài toán **phân loại nhị phân**:

| | |
|---|---|
| **Đầu vào** | Một chuỗi URL, ví dụ `https://www.abcbank-xacthuc.com/dang-nhap?id=12345` |
| **Đầu ra** | `1` = phishing, `0` = benign (an toàn) |
| **Cách làm** | Biến URL thành một dãy số (feature), rồi cho model học từ hàng chục nghìn URL đã gán nhãn |

Tại sao chỉ dùng URL mà không mở trang web ra xem? Vì URL có sẵn ngay lập tức, không cần truy cập trang lừa đảo (vừa nguy hiểm vừa chậm), và trang phishing thường bị gỡ chỉ sau vài giờ.

---

## 2. Dữ liệu

### 2.1 PhishVN là gì
**PhishVN v3.1.0** là bộ URL phishing của Việt Nam, công bố trên Mendeley ([DOI `10.17632/b97hxbxtpd.4`](https://doi.org/10.17632/b97hxbxtpd.4), giấy phép CC BY 4.0). Đây là dataset đã đăng ký cho đồ án.

| | Số lượng |
|---|---|
| Tổng số URL | **53,116** |
| Phishing (`label=1`) | 36,706 (69%) |
| Benign (`label=0`) | 16,410 (31%) |
| Giá trị bị thiếu trong 21 feature | 0 |

Dữ liệu **lệch lớp**: phishing nhiều gấp đôi benign. Điều này ảnh hưởng tới việc chọn metric (xem [6.2](#62-các-metric-và-ví-dụ-về-nghịch-lý-accuracy)).

### 2.2 URL đến từ đâu (rất quan trọng, sẽ quay lại nhiều lần)

| Nguồn | Là gì | Nhãn | Số URL |
|---|---|---|---|
| `tinnhiemmang` | Danh sách đen của Cục An toàn thông tin (NCSC) | phishing | 2,587 |
| `chongluadao` | Danh sách đen do cộng đồng ChongLuaDao báo cáo | phishing | 33,823 |
| `openphish` | Feed phishing quốc tế | phishing | 296 |
| `tinnhiem_org`, `tinnhiem_web` | Danh sách tổ chức, website được NCSC xác nhận tin cậy | benign | ≈7,900 |
| `tranco`, `tranco_vn` | Các website phổ biến nhất thế giới / ở Việt Nam (bảng xếp hạng Tranco) | benign | ≈8,500 |

Để ý: **mỗi nguồn chỉ chứa đúng một loại nhãn**. Nếu model học được "URL này trông giống nguồn nào", nó sẽ đoán đúng nhãn mà không cần hiểu phishing là gì. Đây là cái bẫy chính của dataset; [mục 3.8](#38-chuẩn-hóa-bỏ-www-phát-hiện-quan-trọng-nhất) và [6.5](#65-kiểm-tra-model-có-học-vẹt-nguồn-không-leave-one-source-out) sẽ đo nó.

### 2.3 Độ tin cậy của nhãn (tier)
Mỗi URL có một mức tin cậy:
- **gold** (9,593) và **silver** (9,404): đã được NCSC xác nhận, hoặc là benign từ danh sách tuyển chọn.
- **bronze** (34,119): toàn bộ là phishing do cộng đồng báo cáo (chongluadao, openphish), chưa được xác minh.

Tác giả dataset kiểm tra lại bằng tay và ước tính khoảng **12% URL phishing có thể bị gán nhãn sai** (chủ yếu ở bronze), và 4% URL benign đáng ngờ. Vì vậy đồ án chạy trên hai tập:
- **`FULL`**: mọi tier. Đây là kết quả chính.
- **`GOLDSILVER`**: chỉ tier gold và silver, dùng để nghiên cứu ảnh hưởng của nhiễu nhãn.

### 2.4 Hai file dữ liệu và các cột
Bản phát hành có hai file CSV quan trọng (đặt trong `data/phishvn/data/`):

| File | Nội dung | Dùng để |
|---|---|---|
| `vn_compphish.csv` | URL + **21 feature số** đã tính sẵn + `is_https` + `label` + `split` + `tier` | Train model |
| `dataset_url.csv` | URL + metadata: `source`, `scenario` (giả mạo ngân hàng, cơ quan nhà nước…), ngày thu thập, … | Phân tích lỗi theo nguồn và loại giả mạo. **Không đưa vào model** |

Dữ liệu đã được tác giả **chia sẵn** thành train / val / test (cột `split`):

| Split | Phishing | Benign | Tổng | Dùng để |
|---|---|---|---|---|
| train | 25,905 | 11,004 | 36,909 | Học tham số model |
| val | 5,705 | 1,561 | 7,266 | Chọn model và siêu tham số |
| test | 5,096 | 3,845 | 8,941 | Chấm điểm cuối cùng, **mở một lần** |

Cách chia này có hai tính chất quan trọng:
- **Chia theo nhóm domain.** Mọi URL cùng một domain (ví dụ `abc.com/a` và `abc.com/b`) nằm chung một split. Nếu không, model chỉ cần nhớ tên domain là đoán đúng trên test.
- **Chia theo thời gian.** Train là phishing cũ nhất, test là phishing mới nhất. Điều này mô phỏng thực tế: model học từ quá khứ và phải đoán các chiêu lừa mới.

### 2.5 Bộ ví dụ dùng xuyên suốt README
Ba ví dụ sau xuất hiện ở mọi bước. Các con số đi kèm được tính bằng chính code trong repo.

| Ký hiệu | URL | Thực tế |
|---|---|---|
| **A** | `https://www.abcbank-xacthuc.com/dang-nhap?id=12345` | Phishing (giả lập: domain lạ giả mạo ngân hàng, có trang "đăng nhập") |
| **B** | `https://dichvucong.gov.vn/p/home/dvc-trang-chu.html` | Benign (Cổng dịch vụ công quốc gia) |
| **C** | `https://www.shopee.vn` và `https://shopee.vn` | Benign (cùng một trang thật, có và không có `www.`) |

### 2.6 Dữ liệu ngoài (chỉ để kiểm tra, không dùng để train)
Đạt điểm cao trên test của chính PhishVN chưa chứng minh model dùng được ở nơi khác. Vì vậy đồ án thử model đã train xong trên hai bộ dữ liệu **hoàn toàn độc lập**:
- **Hassan**: 340,000 URL quốc tế, 50% phishing / 50% hợp lệ (năm 2023).
- **JPCERT**: 14,775 URL phishing do CERT Nhật Bản xác minh, nửa đầu 2025. Bộ này chỉ có phishing nên chỉ đo được recall.

Model không bao giờ được train hay tinh chỉnh trên hai bộ này.

---

## 3. Xử lý dữ liệu (từng bước)

Luồng xử lý tổng quát:

```text
URL thô ──► kiểm tra dữ liệu ──► bỏ "http(s)://" ──► trích 21 feature số ──► bỏ cột gây leakage
        ──► điền giá trị thiếu ──► chuẩn hóa (StandardScaler) ──► model
```

Mỗi bước dưới đây trả lời ba câu: **làm gì**, **tại sao**, và **ví dụ**. Sau khi có 21 feature, [mục 4](#4-phân-tích-dữ-liệu-chọn-feature-pca-phân-cụm) phân tích kỹ chính các feature này (chọn feature, PCA, phân cụm).

### 3.1 Kiểm tra dữ liệu (audit)
- **Làm gì:** đếm URL trùng lặp, kiểm tra không domain nào xuất hiện ở hai split, kiểm tra giá trị thiếu, so sánh từng cột giữa hai nhãn.
- **Tại sao:** nếu cùng một URL nằm ở cả train lẫn test, model "nhìn thấy đáp án" và điểm số bị thổi phồng.
- **Kết quả:** 0 URL trùng, 0 domain trùng giữa các split, 0 giá trị thiếu. Có một cột bất thường là `is_https` (xem bước 3.2).
- Code: `run_audit()` trong `pipeline.py`. Kết quả: `outputs/feature_audit.csv`.

### 3.2 Bỏ scheme `http://` / `https://` và loại cột `is_https`
- **Làm gì:** mọi feature được tính trên URL **đã bỏ phần `http://` hoặc `https://`**. Cột `is_https` (URL có dùng https không) bị loại khỏi model.
- **Tại sao:** nghe thì có vẻ "có https = an toàn hơn", nhưng trong dataset này https chỉ phản ánh **cách thu thập dữ liệu**. 12.3% URL benign có ghi https, trong khi phishing chỉ 0.55%, do các nguồn lưu URL theo cách khác nhau. Nếu giữ lại, model sẽ học "có https = benign". Quy tắc đó sai ngoài đời: hầu hết trang phishing hiện nay đều dùng https. Tác giả dataset cũng khuyến nghị bỏ cột này.
- **Ví dụ:** A `https://www.abcbank-xacthuc.com/dang-nhap?id=12345` → `www.abcbank-xacthuc.com/dang-nhap?id=12345`

> Thuật ngữ: một feature chỉ phản ánh *cách dữ liệu được thu thập* chứ không phản ánh bản chất phishing gọi là **collection artefact** (dấu vết thu thập). Model học từ nó sẽ rất giỏi trên dataset nhưng thất bại ngoài đời.

### 3.3 Trích 21 feature số từ URL
- **Làm gì:** máy không đọc được chữ, nên mỗi URL được biến thành 21 con số mô tả "hình dạng" của nó.
- **Tại sao:** URL phishing thường có dấu hiệu nhận ra được: domain dài và lạ, nhiều dấu gạch ngang, nhiều chữ số, có tham số truy vấn, cố nhét tên thương hiệu vào subdomain…
- Code: `phishvn_features()` trong `src/phishing_ml/features.py`. Code này tái tạo đúng bộ số trong `vn_compphish.csv`: kiểm tra trên 53,116 URL, khớp chính xác 53,114 URL.

Giá trị thật của từng feature cho hai ví dụ A và B:

| Feature | Ý nghĩa | A (phishing) | B (benign) |
|---|---|---|---|
| `url_len` | Độ dài URL (đã bỏ scheme) | 42 | 43 |
| `dom_len` | Độ dài domain đăng ký (`abcbank-xacthuc.com`) | 19 | 17 |
| `tld_len` | Độ dài đuôi domain (`com` = 3, `gov.vn` = 6) | 3 | 6 |
| `subdom_cnt` | Số subdomain đứng trước domain (`www` tính là 1) | 1 | 0 |
| `is_ip` | Host có phải là địa chỉ IP không (0/1) | 0 | 0 |
| `letter_cnt` / `digit_cnt` / `special_cnt` | Số chữ cái / chữ số / ký tự đặc biệt | 30 / 5 / 7 | 35 / 0 / 8 |
| `letter_ratio` / `digit_ratio` / `spec_ratio` | Các số trên chia cho `url_len` | 0.71 / 0.12 / 0.17 | 0.81 / 0.00 / 0.19 |
| `dot_cnt` / `dash_cnt` / `under_cnt` / `slash_cnt` | Số dấu `.` `-` `_` `/` | 2 / 2 / 0 / 1 | 3 / 2 / 0 / 3 |
| `eq_cnt` / `qm_cnt` / `amp_cnt` | Số dấu `=` `?` `&` (dấu hiệu có tham số truy vấn) | 1 / 1 / 0 | 0 / 0 / 0 |
| `path_len` | Độ dài phần đường dẫn sau domain (`dang-nhap`) | 9 | 25 |
| `query_len` | Độ dài phần sau dấu `?` (`id=12345`) | 8 | 0 |
| `entropy` | Độ "hỗn loạn" của ký tự (Shannon entropy); chuỗi ngẫu nhiên có entropy cao | 4.46 | 4.08 |

Có thể thấy A và B **gần như dài bằng nhau**. Khác biệt nằm ở chi tiết: A có subdomain, có chữ số, có tham số truy vấn, có đuôi `.com`; B có đuôi `.gov.vn` và không có chữ số. Model phải học được những khác biệt nhỏ như vậy.

### 3.4 Loại các cột có thể gây leakage
- **Làm gì:** không đưa vào model các cột `source`, `tier`, `scenario`, ngày thu thập, ID.
- **Tại sao:** **leakage** (rò rỉ nhãn) là khi model được xem một thông tin mà lúc dùng thật sẽ không có, hoặc thông tin đó gần như chính là đáp án. Ví dụ `source = chongluadao` thì 100% là phishing ([2.2](#22-url-đến-từ-đâu-rất-quan-trọng-sẽ-quay-lại-nhiều-lần)), nên đưa cột này vào thì accuracy gần 100% nhưng vô nghĩa. Khi gặp một URL mới ngoài đời, ta không biết nó "đến từ danh sách nào".
- Danh sách cột được dùng và bị loại (kèm lý do) nằm ở `outputs/feature_audit.csv`.

### 3.5 Điền giá trị thiếu (median imputation)
- **Làm gì:** nếu một feature bị thiếu thì điền bằng **trung vị** (median) của cột đó trên tập train.
- **Tại sao:** đa số model không chạy được khi có ô trống. Dùng median thay vì trung bình vì median không bị kéo lệch bởi vài URL cực dài.
- PhishVN không có giá trị thiếu, nên bước này chỉ là lớp bảo vệ khi dữ liệu mới bị lỗi.

### 3.6 Chuẩn hóa (StandardScaler)
- **Làm gì:** đưa mọi feature về cùng thang đo bằng công thức `z = (x − trung bình) / độ lệch chuẩn`.
- **Tại sao:** `url_len` có thể lên tới hàng trăm, còn `is_ip` chỉ là 0 hoặc 1. Logistic Regression và SVM so sánh trọng số giữa các feature, nên nếu không chuẩn hóa thì feature có giá trị lớn sẽ lấn át. Random Forest chỉ so sánh "lớn hơn hay nhỏ hơn ngưỡng" nên không cần bước này.
- **Ví dụ:** trên train, `url_len` có trung bình 16.09 và độ lệch chuẩn 6.79. URL A có `url_len = 42` → z = (42 − 16.09) / 6.79 ≈ **3.8**, tức dài hơn mức bình thường 3.8 lần độ lệch chuẩn. (Đa số URL trong dataset chỉ là domain trần nên rất ngắn.)

> **Quy tắc "chỉ fit trên train":** trung bình, độ lệch chuẩn và median đều **chỉ được tính trên tập train**, rồi áp nguyên các con số đó cho val và test. Nếu tính trên toàn bộ dữ liệu, thông tin của test đã lọt vào lúc huấn luyện. Repo bọc imputer, scaler và model trong một `sklearn.Pipeline` để quy tắc này luôn được giữ, và có test tự động kiểm tra (`tests/test_preprocess_train_only.py`).

Code: `make_pipeline()` trong `src/phishing_ml/preprocess.py`.

### 3.7 Các cách xử lý khác đã thử (thí nghiệm bổ sung)
Bước 3.5–3.6 là cách xử lý chính. Ngoài ra đồ án thử thêm các cách sau, đều với cùng Logistic Regression, để xem cách xử lý dữ liệu ảnh hưởng thế nào. Kết quả nằm ở [mục 7.2](#72-thí-nghiệm-bổ-sung-cách-xử-lý-dữ-liệu-ảnh-hưởng-thế-nào), chi tiết trong `outputs/EXPERIMENTS.md`.

| Kỹ thuật | Làm gì | Tại sao thử | Ví dụ |
|---|---|---|---|
| **log1p** | Thay `x` bằng `log(1 + x)` trước khi chuẩn hóa | Các feature đếm bị lệch rất mạnh: đa số bằng 0, một số ít rất lớn. Log "nén" giá trị lớn lại | `amp_cnt`: 0 → 0; 10 → 2.4; 100 → 4.6 |
| **QuantileTransformer** | Thay mỗi giá trị bằng **thứ hạng** của nó, rồi ép thành phân phối chuẩn | Hoàn toàn miễn nhiễm với giá trị ngoại lai | URL dài nhất và URL dài thứ nhì có giá trị gần nhau, dù độ dài chênh nhau nhiều |
| **class_weight = balanced** | Phạt lỗi trên lớp ít mẫu (benign) nặng hơn khi train | Dữ liệu lệch lớp 69/31 | Đoán sai 1 URL benign bị tính nặng ≈2.4 lần đoán sai 1 URL phishing |
| **Chỉnh ngưỡng trên val** | Mặc định model báo phishing khi P(phishing) ≥ 0.5; ở đây chọn ngưỡng cho F1 cao nhất trên **val** | 0.5 không phải lúc nào cũng là ngưỡng tốt nhất | Ngưỡng 0.26 thay vì 0.5 |
| **TF-IDF n-gram ký tự** | Cắt URL thành mọi đoạn 2–5 ký tự liên tiếp, đếm và đánh trọng số (đoạn hiếm thì trọng số cao), rồi dùng làm feature | 21 feature đếm không "đọc" được nội dung. N-gram bắt được từ khóa như `xacth`, `nhap`, `vay`, `vip` | A sinh ra 149 n-gram như `xac`, `xacth`, `-nhap`, `nhap?`… |
| **Bỏ tiền tố `www.`** | Xóa `www.` ở đầu host trước khi trích feature | Xem ngay mục 3.8 | `www.shopee.vn` → `shopee.vn` |
| **Cắt ngoại lai (winsorize)** | Mỗi feature bị kẹp trong khoảng [phân vị 1%, phân vị 99%] của train | Vài URL cực dài có thể kéo lệch trọng số của model tuyến tính | `url_len` train: p1 = 7, p99 = 36, max = 184. URL A (`url_len` 42) bị coi là 36 |
| **Rời rạc hóa (KBinsDiscretizer)** | Chia mỗi feature thành 10 khoảng theo phân vị, rồi one-hot (mỗi khoảng là một cột 0/1) | Cho model tuyến tính học được quan hệ "cong": ví dụ quá ngắn và quá dài đều đáng ngờ | Ranh giới `url_len`: 4, 10, 11, 12, 14, 15, 16, 18, 20, 25, 184. A (42) rơi vào khoảng cuối |
| **Feature đa thức bậc 2** | Thêm mọi tích từng cặp feature (`url_len × digit_cnt`, `subdom_cnt²`…): 21 → 252 cột | Cho model tuyến tính học tương tác giữa các feature | "URL dài **và** nhiều chữ số" thành một feature riêng |
| **Lọc tương quan** | Bỏ feature có \|r\| > 0.9 với một feature đã giữ (tính trên train) | Hai feature gần như trùng thông tin là thừa | Bỏ `letter_cnt` (r = 0.95 với `url_len`) và `digit_ratio` (r = −0.96 với `letter_ratio`) |
| **SelectKBest (mutual information)** | Chỉ giữ 10 feature có mutual information với nhãn cao nhất | Model gọn hơn, xem bỏ bớt có mất gì không | Giữ `tld_len`, `dot_cnt`, `letter_ratio`, `entropy`, … |
| **L1 / Lasso** | Phạt tổng \|trọng số\|, khiến trọng số của feature yếu về đúng 0 | Chọn feature ngay trong lúc train (embedded selection) | Với C = 0.02 chỉ còn 12/21 feature |
| **PCA 95%** | Thay 21 feature bằng các thành phần chính giữ 95% phương sai | Giảm chiều, khử tương quan | 21 → 10 thành phần |
| **Undersampling / oversampling** | Train trên dữ liệu cân bằng 50/50: bỏ bớt ngẫu nhiên URL phishing, hoặc nhân bản ngẫu nhiên URL benign | Một cách khác để xử lý lệch lớp, ngoài `class_weight` | Train 11,004 benign / 25,905 phishing → 11,004 / 11,004 (under) hoặc 25,905 / 25,905 (over) |
| **Cụm K-Means làm feature** | Phân cụm train thành 8 cụm (không dùng nhãn), thêm 8 cột 0/1 "URL thuộc cụm nào" | Dùng kết quả học không giám sát để hỗ trợ học có giám sát | Mỗi URL có thêm 8 cột, đúng một cột bằng 1 |

### 3.8 Chuẩn hóa bỏ `www.` (phát hiện quan trọng nhất)
Khi xem các n-gram mà model TF-IDF dựa vào nhiều nhất để kết luận "phishing", n-gram đứng đầu là **`www.`**. Mình đếm lại trên dữ liệu:

| Tập dữ liệu | Tỷ lệ URL bắt đầu bằng `www.` |
|---|---|
| PhishVN, benign (train) | **0.6%** (Tranco: 0%) |
| PhishVN, phishing (train) | **45.7%** (chongluadao: 49%) |
| Hassan, URL hợp lệ | **72%** |

Lý do: các danh sách benign (Tranco, tinnhiem) lưu website dưới dạng domain trần như `shopee.vn`, còn danh sách đen nhiều khả năng lưu nguyên URL như người dùng báo cáo. Model vì thế học quy tắc **"có `www.` → phishing"**. Đây là một collection artefact giống `is_https` ở 3.2.

**Ví dụ C** với model chính (Logistic Regression, đã chuẩn hóa):

| URL | P(phishing) | Model kết luận |
|---|---|---|
| `https://shopee.vn` | 0.35 | benign ✅ |
| `https://www.shopee.vn` | **0.90** | **phishing** ❌ |

Cùng một trang Shopee thật, chỉ thêm `www.` mà bị coi là lừa đảo. Ở Hassan, 72% URL hợp lệ có `www.`, nên model sụp trên Hassan là điều dễ hiểu. Bỏ `www.` trước khi trích feature (`www.example.com` và `example.com` là cùng một trang) giúp model TF-IDF tổng quát hóa tốt hơn nhiều (số liệu ở [7.2](#72-thí-nghiệm-bổ-sung-cách-xử-lý-dữ-liệu-ảnh-hưởng-thế-nào)).

---

## 4. Phân tích dữ liệu (chọn feature, PCA, phân cụm)

Mục 3 biến URL thành 21 con số. Trước khi đưa vào model, ta cần **hiểu chính 21 con số đó**: feature nào thật sự khác nhau giữa phishing và benign, feature nào thừa, dữ liệu có cấu trúc nhóm tự nhiên không, và cấu trúc đó phản ánh phishing hay phản ánh thứ khác. Mục này trả lời các câu hỏi đó bằng các kỹ thuật thống kê và học không giám sát.

Quy ước cho cả mục:
- **Chỉ dùng tập train** (36,909 URL) để tính toán. Val chỉ dùng để chấm điểm ở các thí nghiệm có nhãn (4.1, và AUC từng thành phần ở 4.2). **Test không được đụng tới.**
- Các phương pháp dựa trên khoảng cách (PCA, phân cụm) dùng dữ liệu đã qua `log1p` rồi `StandardScaler`. Lý do: các feature đếm lệch rất mạnh. 11/21 feature có hơn 50% giá trị bằng 0 (ví dụ `query_len` bằng 0 ở 99.9% URL), nhưng vài URL có giá trị rất lớn. Nếu để nguyên, vài giá trị cực lớn sẽ lấn át khoảng cách. `log1p(x) = log(1 + x)` giữ 0 là 0 và nén giá trị lớn lại (10 → 2.4, 100 → 4.6). Ví dụ với URL A: `url_len` = 42 → log1p(42) = 3.76. Trên train, log1p(`url_len`) có trung bình 2.78 và độ lệch chuẩn 0.34, nên A có z = (3.76 − 2.78) / 0.34 ≈ **2.9**.
- Code: `src/phishing_ml/analysis.py`. Chạy riêng: `python -m phishing_ml.cli analysis` (≈2 phút). Báo cáo đầy đủ với mọi bảng số: `outputs/DATA_ANALYSIS.md`.

### 4.1 Xếp hạng và chọn feature: feature nào quan trọng, cần bao nhiêu là đủ?
- **Làm gì:**
  1. **Xếp hạng bằng mutual information (MI):** với mỗi feature, đo xem biết giá trị của nó thì bớt được bao nhiêu "không chắc chắn" về nhãn. MI = 0 nghĩa là feature và nhãn độc lập (feature vô dụng); MI càng lớn thì feature càng mang nhiều thông tin về nhãn. MI bắt được mọi kiểu quan hệ, kể cả phi tuyến.
  2. **Top-k theo MI:** lần lượt train Logistic Regression với 1, 2, …, 21 feature có MI cao nhất, đo macro-F1 trên val.
  3. **L1 / Lasso:** thêm vào hàm mất mát một khoản phạt bằng tổng |trọng số|. Khoản phạt này đẩy trọng số của feature yếu về **đúng 0**, tức feature bị loại ngay trong lúc train. Hệ số `C` càng nhỏ thì phạt càng nặng và càng ít feature sống sót. Thử 13 giá trị C từ 0.001 đến 10.
- **Tại sao:** model ít feature hơn thì đơn giản hơn, dễ giải thích hơn, và cho biết thông tin thật sự nằm ở đâu.

Kết quả (hình `outputs/figures/feature_selection.png`; số liệu `outputs/feature_ranking.csv`, `outputs/feature_selection_curve.csv`, `outputs/l1_path.csv`):
- **Top 5 theo MI:** `tld_len`, `dot_cnt`, `letter_ratio`, `entropy`, `digit_ratio`. Các feature đếm tham số truy vấn (`query_len`, `eq_cnt`, `amp_cnt`, `under_cnt`) có MI ≈ 0, vì 99.9% URL không có tham số. URL A (có `?id=12345`) thuộc số rất ít ngoại lệ.
- **Đường cong top-k:** dùng đủ 21 feature đạt val macro-F1 = 0.780. Chỉ **9 feature** đầu đã cách mức tốt nhất không quá 0.005.
- **Phát hiện quan trọng:** đường cong gần như phẳng ở mức 0.60 với 1–8 feature, rồi **nhảy +0.18 lên 0.78 đúng lúc thêm feature thứ 9 là `subdom_cnt`**. Như đã nói ở 3.3, `www` được tính là 1 subdomain. Đếm trên train:

  | `subdom_cnt` | Benign | Phishing |
  |---|---|---|
  | 0 | 6,995 | 12,685 |
  | 1 | 2,972 | **13,085** |
  | ≥ 2 | 1,037 | 135 |

  Khi có đúng 1 subdomain (đa số là `www.`), tỷ lệ phishing là 81%. Phần lớn sức mạnh của 21 feature số đến từ dấu vết thu thập `www.` ở mục 3.8, chứ không phải từ "hình dạng" của URL lừa đảo. Ví dụ C cho thấy điều này: `www.shopee.vn` có `subdom_cnt` = 1, còn `shopee.vn` có `subdom_cnt` = 0.
- **L1:** với C = 0.02, model chỉ giữ **12/21 feature** mà val macro-F1 = 0.782, ngang mức tốt nhất. Có thể bỏ gần một nửa số feature mà không mất gì.

### 4.2 PCA (phân tích thành phần chính)
- **Làm gì:** PCA "xoay" hệ trục tọa độ 21 chiều sao cho trục thứ nhất (PC1) nằm theo hướng dữ liệu trải rộng nhất, PC2 theo hướng trải rộng nhất còn lại (vuông góc với PC1), và cứ thế tiếp. Mỗi PC là một tổ hợp tuyến tính của 21 feature, ví dụ `PC1 = 0.39·letter_cnt + 0.39·url_len + 0.33·special_cnt + …`. Các hệ số này gọi là **loading**: loading lớn nghĩa là feature đó đóng góp nhiều vào PC.
- **Tại sao:** (1) giảm chiều: nếu vài PC đầu giữ được gần hết phương sai thì bỏ được các PC sau; (2) gộp các feature trùng thông tin (ví dụ `url_len` và `letter_cnt` tương quan 0.95); (3) vẽ dữ liệu 21 chiều lên mặt phẳng 2D.

**Phải chuẩn hóa trước PCA.** So sánh ba cách tiền xử lý:

| Tiền xử lý | PC1 chiếm | Số PC để giữ 90% phương sai | Feature chi phối PC1 |
|---|---|---|---|
| Không chuẩn hóa | **81.6%** | 2 | `url_len` (+0.69), `letter_cnt` (+0.64) |
| StandardScaler | 28.6% | 8 | `url_len`, `special_cnt`, `letter_cnt` |
| log1p + StandardScaler | 27.4% | 7 | `letter_cnt`, `url_len`, `special_cnt` |

Không chuẩn hóa thì PCA chỉ "thấy" `url_len` và `letter_cnt`. Lý do không phải vì chúng quan trọng, mà vì chúng có **đơn vị lớn nhất** (hàng chục), trong khi các tỷ lệ chỉ nằm trong 0–1. Kết quả "2 PC giữ 90% phương sai" trông rất đẹp nhưng sai: nó chỉ nói rằng độ dài URL dao động mạnh.

**Ý nghĩa của các PC** (sau log1p + StandardScaler, hình `outputs/figures/pca_loadings.png`):
- **PC1 ≈ "URL dài cỡ nào":** `letter_cnt`, `url_len`, `special_cnt`, `dot_cnt` đều có loading dương.
- **PC2 ≈ "có tham số truy vấn không":** `query_len`, `eq_cnt`, `qm_cnt`, `under_cnt`.
- **PC3 ≈ "nhiều subdomain/dấu chấm, ít chữ cái":** `spec_ratio` (+), `letter_ratio` (−), `subdom_cnt` (+), `dot_cnt` (+).

*Ví dụ:* tọa độ của URL A trên (PC1, PC2) là (17.8, 40.4), còn B là (10.4, 9.5). A nằm rất xa mọi URL khác theo PC2, vì A có `?id=12345` mà 99.9% URL trong train không có tham số. Sau chuẩn hóa, một giá trị hiếm như vậy thành một con số rất lớn.

**Phương sai lớn có đồng nghĩa với hữu ích cho phân loại không?** Không. Train Logistic Regression chỉ với đúng một PC và đo AUC trên val:

| PC | % phương sai | AUC khi dùng riêng PC đó |
|---|---|---|
| PC1 | 27.4% | 0.58 |
| PC2 | 20.8% | 0.60 |
| **PC3** | 15.1% | **0.70** |
| PC4 | 9.8% | 0.64 |
| PC8 | 3.7% | 0.61 |

PC hữu ích nhất là PC3, không phải PC1. PCA chọn hướng theo **phương sai** mà không nhìn nhãn, nên hướng "trải rộng nhất" (độ dài URL) không trùng với hướng "tách phishing khỏi benign" (PC3 chứa `subdom_cnt`, tức là `www.`). Vì vậy giữ ít PC sẽ mất thông tin: giữ 95% phương sai (10 PC) làm test macro-F1 giảm từ 0.898 xuống 0.806 ([mục 7.2](#72-thí-nghiệm-bổ-sung-cách-xử-lý-dữ-liệu-ảnh-hưởng-thế-nào)).

Hình 2D: `outputs/figures/pca_2d_label_source.png`, chiếu train lên PC1–PC2, một bên tô màu theo nhãn, một bên theo nguồn. Số liệu: `outputs/pca_scaling_comparison.csv`, `outputs/pca_loadings.csv`, `outputs/pca_component_auc.csv`.

### 4.3 Phân cụm (K-Means, GMM)
- **Làm gì:** gom URL thành k nhóm **mà không dùng nhãn**, rồi xem các nhóm đó là gì.
- **Tại sao:** nếu URL phishing và benign tự tách thành hai nhóm, bài toán sẽ dễ. Nếu không, ta muốn biết dữ liệu tự chia theo tiêu chí nào.

**Hai thuật toán:**
- **K-Means:** (1) chọn ngẫu nhiên k điểm làm tâm; (2) gán mỗi URL vào tâm gần nhất; (3) dời mỗi tâm về trung bình các URL của nó; lặp (2)–(3) đến khi tâm không đổi. Kết quả là các cụm "tròn" quanh tâm. Chạy 10 lần với khởi tạo khác nhau, giữ lần tốt nhất.
- **GMM (Gaussian Mixture Model):** mỗi cụm là một phân phối chuẩn nhiều chiều có hình **elip** tùy ý (ma trận hiệp phương sai đầy đủ). Mỗi URL có xác suất thuộc từng cụm, nên các cụm được phép chồng lên nhau.

**Chọn k thế nào khi không có nhãn?** Thử k = 2..10 và dùng các chỉ số sau:

| Chỉ số | Đo cái gì | Tốt khi |
|---|---|---|
| **Inertia (elbow)** | Tổng bình phương khoảng cách từ mỗi URL tới tâm cụm. Luôn giảm khi k tăng; chọn k ở chỗ đường cong "gãy khúc" | Có khúc gãy rõ |
| **Silhouette** | Với mỗi URL: (khoảng cách trung bình tới cụm gần nhất khác − tới cụm của mình) / max của hai số. Từ −1 đến 1 | Cao |
| **Davies–Bouldin** | Trung bình độ "giống nhau" giữa mỗi cụm và cụm giống nó nhất | Thấp |
| **Calinski–Harabasz** | Tỷ lệ phương sai giữa các cụm so với phương sai trong cụm | Cao |
| **BIC** (cho GMM) | Độ khớp dữ liệu trừ đi một khoản phạt theo số tham số, để không chọn k quá lớn | Thấp |

**Đối chiếu với nhãn (chỉ để đánh giá, không dùng khi phân cụm):**
- **ARI (Adjusted Rand Index):** xét mọi cặp URL, hai cách chia có đồng ý "cặp này cùng nhóm / khác nhóm" không, đã trừ phần trùng do ngẫu nhiên. 0 = như chia ngẫu nhiên, 1 = khớp hoàn toàn.
- **NMI (Normalized Mutual Information):** cụm chứa bao nhiêu thông tin về nhãn, từ 0 đến 1.

Mình đối chiếu cụm với **hai thứ**: nhãn (phishing/benign) và **họ nguồn** của URL (chongluadao, openphish, tinnhiemmang, tinnhiem, tranco).

**Kết quả** (hình `outputs/figures/clustering_selection.png`):
- Silhouette của K-Means chỉ từ 0.26 đến 0.33 (tối đa là 1): dữ liệu có cấu trúc cụm, nhưng các cụm không tách nhau rõ.
- ARI với nhãn cao nhất chỉ **0.07** (K-Means) và 0.19 (GMM): **cụm không trùng với phishing/benign**.
- Ở **cả 9 giá trị k**, ARI với **nguồn** cao hơn ARI với **nhãn**. Dữ liệu tự nhóm theo "URL đến từ danh sách nào" nhiều hơn là theo "URL có lừa đảo không".

**Hồ sơ cụm** (K-Means, k = 6: k có silhouette cao nhất trong 4..8; hình `outputs/figures/cluster_profiles.png`):

| Cụm | Số URL | % phishing | `www.` | Đuôi .vn | Đặc điểm |
|---|---|---|---|---|---|
| 1 | 15,380 (42%) | 64% | **53%** | 36% | Domain trần, có `www.` |
| 0 | 13,241 (36%) | 62% | **0.1%** | 13% | Domain trần, không `www.` |
| 3 | 8,154 (22%) | **96%** | 46% | 2% | Không phải .vn, nhiều chữ số |
| 2 | 102 | 92% | 7% | 5% | Có đường dẫn, URL dài |
| 5 | 27 | 100% | 30% | 0% | Có tham số truy vấn, nhiều chữ số, nhiều dấu gạch ngang |
| 4 | 5 | 100% | 100% | 0% | URL cực dài (trung vị 154 ký tự) |

Đọc bảng:
- **Hai cụm lớn nhất (78% dữ liệu) khác nhau chủ yếu ở việc có `www.` hay không** (53% so với 0.1%), còn tỷ lệ phishing gần như bằng nhau (64% và 62%). Không cần nhãn, K-Means vẫn tự tìm ra đúng dấu vết thu thập `www.` của mục 3.8.
- Cụm 3 gần như thuần phishing: URL nước ngoài (không phải .vn) với nhiều chữ số.
- Ba cụm tí hon (134 URL, chủ yếu từ openphish) là các URL bất thường: dài, có đường dẫn, có tham số. K-Means tách chúng ra như ngoại lai.

*Ví dụ với A, B, C:*

| URL | Rơi vào cụm |
|---|---|
| A `www.abcbank-xacthuc.com/dang-nhap?id=12345` | Cụm 5 (có tham số truy vấn, nhiều chữ số) |
| B `dichvucong.gov.vn/p/home/dvc-trang-chu.html` | Cụm 2 (có đường dẫn, URL dài) |
| C `www.shopee.vn` | **Cụm 1** (có `www.`) |
| C `shopee.vn` | **Cụm 0** (không `www.`) |

Cùng một trang Shopee mà nằm ở hai cụm khác nhau, chỉ vì có hay không có `www.`.

Số liệu: `outputs/clustering_selection.csv`, `outputs/clustering_ward.csv`, `outputs/cluster_profiles.csv`.

### 4.4 Tóm tắt: phân tích dữ liệu dẫn tới quyết định gì

| Phát hiện | Hệ quả |
|---|---|
| Feature lệch mạnh, nhiều số 0 (phần mở đầu mục 4) | Dùng `log1p` trước PCA và phân cụm; thử `log1p`, winsorize, QuantileTransformer trong thí nghiệm ([7.2](#72-thí-nghiệm-bổ-sung-cách-xử-lý-dữ-liệu-ảnh-hưởng-thế-nào)) |
| Sức mạnh của feature số phần lớn đến từ `subdom_cnt`, tức `www.` (4.1) | Thử bỏ `www.` trước khi trích feature (3.8) |
| PC hữu ích nhất không phải PC1 (4.2) | PCA không phù hợp để giảm chiều cho bài toán này |
| Cụm khớp với nguồn hơn với nhãn (4.3) | Phải kiểm tra leave-one-source-out (6.5) và dữ liệu ngoài (6.4) |

---

## 5. Mô hình

### 5.1 Dummy (mốc so sánh tối thiểu)
- **Làm gì:** luôn đoán lớp xuất hiện nhiều nhất trong train, không nhìn vào dữ liệu.
- **Tại sao cần:** để biết một model "thật" có hơn đoán mò không. Trên tập FULL, dummy luôn đoán phishing và đã đạt accuracy 78.5% trên val. Vậy model đạt 80% chưa chắc đã giỏi.

### 5.2 Logistic Regression (model chính)
- **Cách hoạt động:** gán cho mỗi feature một trọng số `w`, tính điểm `s = b + w₁·x₁ + … + w₂₁·x₂₁`, rồi đưa qua hàm sigmoid `P = 1 / (1 + e^(−s))` để được xác suất phishing từ 0 đến 1. Nếu `P ≥ 0.5` thì kết luận phishing.
- **Siêu tham số `C`:** điều chỉnh mức "phạt" khi trọng số quá lớn (regularization). C nhỏ thì model đơn giản, ít học thuộc; C lớn thì model bám sát dữ liệu hơn. Đồ án thử C = 0.1, 1, 10.
- **Ví dụ:** model được chọn (C = 10) cho **A: P = 1.00 → phishing ✅**, **B: P = 0.12 → benign ✅**.
- **Dễ giải thích:** trọng số cho biết feature nào đẩy về phía nào. Ví dụ `subdom_cnt` có trọng số +6.9 (nhiều subdomain → phishing hơn), `dot_cnt` −4.7. Chính `subdom_cnt` là nơi `www` được tính vào (www = 1 subdomain), nên đây là cửa ngõ của artefact ở mục 3.8.

### 5.3 Random Forest
- **Cách hoạt động:** xây 200 **cây quyết định**. Mỗi cây là chuỗi câu hỏi có/không kiểu "`digit_cnt` > 3?" → "`tld_len` ≤ 2?" → … Mỗi cây học trên một phần dữ liệu ngẫu nhiên, và kết quả cuối là cây nào đông phiếu hơn thì thắng.
- **Ưu điểm:** bắt được quan hệ phi tuyến, không cần chuẩn hóa.
- **Siêu tham số:** độ sâu tối đa của cây (`max_depth`), số mẫu tối thiểu ở một lá (`min_samples_leaf`).

### 5.4 Linear SVM
- **Cách hoạt động:** tìm một mặt phẳng chia hai lớp sao cho **khoảng cách từ mặt phẳng tới các điểm gần nhất** là lớn nhất. Giống Logistic Regression ở chỗ là model tuyến tính, nhưng tối ưu theo tiêu chí khác.
- **Siêu tham số:** `C`, ý nghĩa tương tự như ở Logistic Regression.

### 5.5 Chọn model
Có tổng cộng 10 cấu hình: dummy, 3 Logistic Regression, 3 Random Forest, 3 Linear SVM. Tất cả **học trên train, so điểm trên val**. Cấu hình có F1 cao nhất trên val được chọn, rồi đóng băng (lưu ra file) và chấm trên test đúng một lần.

Kết quả chọn trên FULL: **Logistic Regression C = 10**. Code: `grids()`, `run_quick()`, `run_final_test()` trong `pipeline.py`.

### 5.6 PCA (giảm chiều)
- **Làm gì:** gộp 21 feature thành ít "thành phần chính" hơn, mỗi thành phần là tổ hợp tuyến tính của các feature cũ, chọn theo hướng dữ liệu biến thiên nhiều nhất.
- **Tại sao thử:** nhiều feature trùng thông tin nhau (`letter_cnt` và `url_len` tương quan 0.95). Ít chiều hơn thì model nhanh hơn và dễ vẽ hình.
- **Câu hỏi:** giảm chiều có làm mất thông tin phân biệt phishing không?
- Cách PCA hoạt động, loadings và vì sao PC1 không phải PC hữu ích nhất: xem [4.2](#42-pca-phân-tích-thành-phần-chính).

### 5.7 K-Means và GMM (phân cụm)
- **Làm gì:** gom URL thành nhóm theo độ giống nhau, **không dùng nhãn**. K-Means chia thành k cụm dựa trên khoảng cách tới tâm cụm; GMM (Gaussian Mixture) coi mỗi cụm là một "đám mây" hình elip và cho phép cụm chồng lấn.
- **Tại sao thử:** xem dữ liệu có tự chia thành "nhóm phishing" và "nhóm benign" không.
- **Đo bằng ARI** (Adjusted Rand Index): bằng 1 nếu các cụm khớp hoàn toàn với nhãn, bằng 0 nếu chỉ khớp như chia ngẫu nhiên.
- Cách chọn k, hồ sơ từng cụm và đối chiếu cụm với nguồn dữ liệu: xem [4.3](#43-phân-cụm-k-means-gmm).

---

## 6. Đánh giá thế nào cho đúng

### 6.1 Ba tập dữ liệu, ba vai trò
```text
train ──► model học tham số
val   ──► so sánh các model / siêu tham số / ngưỡng ──► chọn 1 model, đóng băng
test  ──► chấm điểm model đã đóng băng, MỘT lần
```
Tại sao test chỉ được mở một lần? Nếu nhìn điểm test, sửa model, rồi nhìn lại, ta đang gián tiếp "học" trên test và con số cuối sẽ đẹp hơn thực tế. Repo tuân thủ quy tắc này cho model chính.

### 6.2 Các metric và ví dụ về "nghịch lý accuracy"
Mọi metric đều coi **phishing là lớp dương**. Lấy kết quả thật của model chính trên test làm ví dụ:

| | Đoán benign | Đoán phishing |
|---|---|---|
| **Thật là benign** (3,845) | TN = 3,105 | FP = 740 (báo nhầm) |
| **Thật là phishing** (5,096) | FN = 131 (bỏ sót) | TP = 4,965 |

| Metric | Công thức | Ý nghĩa | Giá trị |
|---|---|---|---|
| Accuracy | (TP+TN) / tổng | Tỷ lệ đoán đúng | 0.903 |
| Precision | TP / (TP+FP) | Trong các URL bị báo phishing, bao nhiêu là phishing thật | 0.870 |
| Recall | TP / (TP+FN) | Trong các URL phishing, bắt được bao nhiêu | 0.974 |
| Specificity | TN / (TN+FP) | Trong các URL benign, bao nhiêu được cho qua đúng | 0.808 |
| F1 | Trung bình điều hòa của precision và recall | Cân bằng giữa báo nhầm và bỏ sót | 0.919 |
| **macro-F1** | Trung bình F1 của lớp phishing và F1 của lớp benign | **Mỗi lớp quan trọng như nhau**, không phụ thuộc lớp nào đông hơn | 0.898 |
| ROC-AUC | Xác suất model chấm điểm một URL phishing cao hơn một URL benign ngẫu nhiên | Chất lượng xếp hạng, không phụ thuộc ngưỡng | 0.958 |

**Nghịch lý accuracy:** trên tập GOLDSILVER, val chỉ có 24 URL phishing trong 1,585 URL. Model "luôn đoán benign" đạt accuracy **98.5%**, cao hơn Random Forest (96.7%), nhưng **không bắt được URL phishing nào** (F1 = 0). Vì vậy đồ án luôn so với dummy trước, và chốt kết quả bằng **F1 và macro-F1**, không chốt bằng accuracy.

### 6.3 Độ tin cậy của con số
- **Khoảng tin cậy Clopper–Pearson** cho accuracy. Ví dụ test accuracy 0.903 có khoảng tin cậy 95% là [0.896, 0.909]. Nếu chấm trên một bộ test khác cùng kích thước, kết quả nhiều khả năng rơi trong khoảng này.
- **Kiểm định McNemar:** khi so hai model trên cùng bộ test, chỉ xét những URL mà hai model đoán **khác nhau**. Nếu model mới đúng ở 604 URL mà model cũ sai, còn model cũ chỉ đúng ở 280 URL mà model mới sai, thì p-value rất nhỏ, nghĩa là chênh lệch không phải do may rủi.
- **Bootstrap ghép cặp:** lấy mẫu lại bộ test 1,000 lần, mỗi lần tính chênh lệch macro-F1 giữa hai model. Ta được khoảng tin cậy của chênh lệch. Nếu cả khoảng nằm trên 0 thì model mới tốt hơn thật.

### 6.4 Kiểm tra trên dữ liệu ngoài
Model đã đóng băng được chạy nguyên trạng (ngưỡng 0.5, không chỉnh gì) trên Hassan và JPCERT ([2.6](#26-dữ-liệu-ngoài-chỉ-để-kiểm-tra-không-dùng-để-train)). Đây là phép thử "model có dùng được ở chỗ khác không".

### 6.5 Kiểm tra model có học vẹt nguồn không (leave-one-source-out)
Vì mỗi nguồn chỉ có một nhãn ([2.2](#22-url-đến-từ-đâu-rất-quan-trọng-sẽ-quay-lại-nhiều-lần)), ta làm như sau: **bỏ hẳn một nguồn ra khỏi train**, train lại, rồi đo xem model có nhận ra URL của nguồn đó không. Nếu model thật sự hiểu phishing, điểm sẽ không giảm nhiều. Nếu điểm rơi mạnh, model đang dựa vào "phong cách" của nguồn.

---

## 7. Kết quả

### 7.1 Benchmark chính (10 cấu hình, chọn trên val, test mở một lần)

| Câu hỏi | Kết quả |
|---|---|
| **RQ1** Model nào tốt nhất? | **Logistic Regression C=10**. Test: accuracy **0.903** [0.896, 0.909], F1 **0.919**, macro-F1 0.898, ROC-AUC 0.958 |
| **RQ2** Model có tổng quát hóa không? | **Không.** Trên Hassan accuracy chỉ **0.439** (giảm 46 điểm %), specificity 0.038: gần như mọi URL hợp lệ bị báo phishing. Nguồn Tranco (benign quốc tế) chỉ đúng 30% ngay trên test nội bộ |
| **RQ3** PCA có giúp không? | **Không.** Giữ 90% phương sai (8 thành phần): accuracy trên val 0.773 so với 0.871 khi dùng đủ 21 feature. Hướng biến thiên nhiều nhất không phải hướng phân biệt phishing |
| **RQ4** Phân cụm có tách được phishing không? | **Không.** ARI ≈ 0 với k=2; tốt nhất 0.21 (GMM, k=4). Dữ liệu có cấu trúc cụm, nhưng các cụm không trùng với ranh giới phishing/benign |

Các kết quả "không" ở trên **là kết quả có giá trị**, không phải thất bại: chúng chỉ ra chính xác dữ liệu và model hỏng ở đâu.

### 7.2 Thí nghiệm bổ sung: cách xử lý dữ liệu ảnh hưởng thế nào
Cùng Logistic Regression C=10 trên tập FULL, chỉ thay cách xử lý dữ liệu. Δ là chênh lệch macro-F1 so với baseline trên test (khoảng tin cậy 95% bootstrap ghép cặp, p-value McNemar).

| Cách xử lý | Test macro-F1 | Δ so với baseline | Hassan macro-F1 |
|---|---|---|---|
| Baseline: median + StandardScaler (= model chính) | 0.898 | — | 0.332 |
| log1p | 0.899 | +0.001, không có ý nghĩa (p = 0.41) | 0.319 |
| QuantileTransformer | 0.884 | **−0.015**, kém hơn (p ≈ 2e-19) | 0.372 |
| class_weight = balanced | 0.854 | −0.044, kém hơn; chỉnh lại ngưỡng trên val thì bằng baseline | 0.335 |
| **TF-IDF n-gram ký tự** | **0.938** | **+0.040** [+0.033, +0.046], p ≈ 2e-28 | 0.456 |
| **TF-IDF n-gram, bỏ `www.`** | 0.918 | +0.019 | **0.584** (tốt nhất) |

Các kỹ thuật xử lý dữ liệu kinh điển trên 21 feature số (mỗi dòng chỉ đổi một bước so với baseline):

| Kỹ thuật | Test macro-F1 | Δ so với baseline [CI95] | Kết luận |
|---|---|---|---|
| Cắt ngoại lai (winsorize 1–99%) | 0.898 | −0.000 [−0.002, +0.001] | Không khác biệt |
| Lọc tương quan (\|r\| > 0.9) | 0.898 | −0.000 [−0.000, +0.000] | Không khác biệt; bỏ 2 feature không mất gì |
| SelectKBest (10 feature, MI) | 0.899 | +0.001 [−0.001, +0.002] | Không khác biệt; một nửa số feature là đủ |
| Cụm K-Means làm feature (k=8) | 0.898 | +0.000 [−0.001, +0.001] | Không khác biệt |
| Feature đa thức bậc 2 | **0.902** | +0.003 [+0.000, +0.006] | Nhỉnh hơn chút; McNemar p = 0.07 nên chưa chắc chắn |
| L1 / Lasso (C = 0.05) | 0.897 | −0.001 [−0.002, −0.001] | Kém hơn rất ít |
| Rời rạc hóa (10 bin + one-hot) | 0.894 | −0.004 [−0.007, −0.001] | Kém hơn rất ít |
| Undersampling / oversampling | 0.857 | −0.041 | Kém hơn ở ngưỡng 0.5 (giống `class_weight`: chỉ dịch ngưỡng) |
| PCA 95% (10 thành phần) | 0.806 | **−0.092** | Kém hơn rõ rệt (hướng hữu ích có phương sai nhỏ, xem 4.2) |

Rút ra:
1. **Đọc trực tiếp chuỗi URL (TF-IDF) tốt hơn hẳn 21 feature đếm.**
2. **Đổi cách scale hầu như không giúp gì.** Các feature lệch mạnh nhưng thông tin nằm ở chỗ "có hay không", không ở độ lớn. QuantileTransformer còn làm mất khoảng cách giữa các giá trị cực trị nên kém hơn.
3. **class_weight chỉ dịch ngưỡng quyết định.** Chọn ngưỡng trên val cho cùng kết quả.
4. **Bỏ `www.` làm điểm nội bộ giảm nhưng tổng quát hóa tăng** (specificity trên Hassan 0.22 → 0.45). Nghĩa là một phần điểm nội bộ đến từ lối tắt `www.`, và con số sau khi bỏ trung thực hơn.
5. **Với 21 feature số, các kỹ thuật kinh điển (cắt ngoại lai, chọn feature, rời rạc hóa, đa thức, cụm làm feature) chỉ thay đổi macro-F1 dưới ±0.005.** Giới hạn nằm ở thông tin có trong feature (và dấu vết thu thập), không nằm ở cách xử lý. Chọn feature cho thấy có thể bỏ một nửa số feature mà không mất gì.
6. **Leave-one-source-out:** URL benign của tinnhiem được nhận đúng 93–97% khi nguồn này có trong train, nhưng chỉ **7–12%** khi bị loại khỏi train; với Tranco là 36–79% → **15–23%**. Model phụ thuộc rất nặng vào nguồn, và đó là giới hạn của chính dataset.

Thí nghiệm này chạy **sau khi** test đã mở cho model chính, nên được trình bày như một thí nghiệm thứ hai và không thay model chính.

---

## 8. Cấu trúc repo và bản đồ code

```text
Project/
├── README.md
├── pyproject.toml                ← thư viện cần cài (có pin version), cấu hình pytest/ruff
├── src/phishing_ml/
│   ├── cli.py                    ← lệnh chạy: run, analysis, experiments, fetch-external
│   ├── features.py               ← URL → 21 feature            (mục 3.2–3.3)
│   ├── preprocess.py             ← imputer → scaler → model     (mục 3.5–3.6)
│   ├── pipeline.py               ← toàn bộ benchmark chính      (mục 5–7.1)
│   ├── experiments.py            ← thí nghiệm bổ sung          (mục 3.7–3.8, 7.2)
│   ├── analysis.py               ← chọn feature, PCA, phân cụm (mục 4)
│   ├── transformers.py           ← cắt ngoại lai, lọc tương quan, cụm làm feature, resampling
│   ├── evaluate.py               ← metric + khoảng tin cậy      (mục 6.2–6.3)
│   └── download.py               ← tải dữ liệu Hassan + JPCERT
├── tests/                        ← test tự động (pytest)
├── data/                         ← dữ liệu (CSV không có trong git)
│   ├── phishvn/                  ← PhishVN: README, docs/, MANIFEST.txt, data/*.csv
│   └── external/                 ← Hassan + JPCERT
├── outputs/                      ← mọi kết quả (sinh tự động, đừng sửa tay)
│   ├── DECISION.md               ← kết luận benchmark chính
│   ├── QUICK_RESULTS.md          ← số liệu theo từng bước
│   ├── EXPERIMENTS.md            ← thí nghiệm bổ sung
│   ├── DATA_ANALYSIS.md          ← phân tích dữ liệu (chọn feature, PCA, phân cụm)
│   ├── *.csv                     ← bảng số liệu
│   └── figures/                  ← hình
└── archive/phiusiil/             ← thí nghiệm cũ trên dataset UCI PhiUSIIL (không thuộc đề tài chính)
```

`run_pipeline()` ở cuối `pipeline.py` là nơi nối mọi bước lại. Đọc hàm này trước, rồi nhảy vào từng hàm nó gọi:

| # | Hàm | Làm gì | Ghi ra (`outputs/`) |
|---|---|---|---|
| 0 | `missing_inputs` | Kiểm tra đủ file dữ liệu; thiếu thì dừng và in ra file nào thiếu | — |
| 1 | `load_frames` | Đọc 2 file CSV, ghép `source`/`scenario` vào | — |
| 2 | `run_audit` | Kiểm tra dữ liệu (mục 3.1) | `feature_audit.csv` |
| 3 | `run_quick` | Train 10 cấu hình, chọn trên val (5.5) | `quick_val_*.csv` |
| 4 | `run_final_test` | Chấm model đã chọn trên test một lần; phân tích theo nguồn, tier, loại giả mạo | `classification_metrics.csv`, `test_by_*.csv` |
| 5 | `run_external` | Chấm trên Hassan, JPCERT (6.4) | `external_metrics.csv` |
| 6 | `run_pca_cluster` | PCA, K-Means, GMM (5.6–5.7) | `pca_results.csv`, `clustering_results.csv` |
| 7 | `run_figures`, `write_reports` | Vẽ hình, viết báo cáo | `figures/`, `DECISION.md`, `QUICK_RESULTS.md` |
| 8 | `run_data_analysis` → `analysis.py` | Phân tích dữ liệu (mục 4). Bỏ qua khi `--mode quick` | `DATA_ANALYSIS.md`, `feature_*.csv`, `pca_*.csv`, `clustering_*.csv`, `cluster_profiles.csv` |
| 9 | `run_supplementary` → `experiments.py` | Thí nghiệm bổ sung (7.2). Bỏ qua khi `--mode quick` | `EXPERIMENTS.md` và các CSV, hình đi kèm |

---

## 9. Cài đặt và chạy

Cần Python ≥ 3.11, chỉ cần CPU.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\Activate.ps1      Linux/macOS:  source .venv/bin/activate
pip install -e ".[dev]"
```

**Chuẩn bị dữ liệu** (dữ liệu không có trong git):
1. Tải PhishVN (open tier) từ Mendeley, DOI `10.17632/b97hxbxtpd.4`. Giải nén sao cho có `data/phishvn/data/vn_compphish.csv` và `data/phishvn/data/dataset_url.csv`. Có thể kiểm tra mã hash bằng `data/phishvn/MANIFEST.txt`.
2. Tải dữ liệu ngoài: `python -m phishing_ml.cli fetch-external`

**Chạy:**
```bash
python -m phishing_ml.cli run --mode quick   # benchmark chính, ≈1 phút
python -m phishing_ml.cli run                # + PCA, phân cụm, phân tích dữ liệu, thí nghiệm bổ sung, ≈8 phút
python -m phishing_ml.cli analysis           # chỉ phân tích dữ liệu (chọn feature, PCA, phân cụm), ≈2 phút
python -m phishing_ml.cli experiments        # chỉ thí nghiệm bổ sung (21 cách xử lý), ≈6 phút
```
Kết quả ghi đè vào `outputs/`. Seed cố định (42), nên chạy lại ra cùng số liệu (chỉ cột thời gian chạy `fit_s` thay đổi). Nếu thiếu dữ liệu, lệnh in `BLOCKED` kèm danh sách file thiếu.

**Kiểm thử:** `pytest -q` và `ruff check .`

---

## 10. Đọc tiếp ở đâu

| Muốn biết | Đọc |
|---|---|
| Kết luận đầy đủ của benchmark chính | `outputs/DECISION.md` |
| Số liệu từng bước, phân tích theo nguồn và tier | `outputs/QUICK_RESULTS.md`, `outputs/test_by_*.csv` |
| Phân tích dữ liệu: chọn feature, PCA, phân cụm | `outputs/DATA_ANALYSIS.md` |
| Chi tiết thí nghiệm tiền xử lý, `www.`, leave-one-source-out | `outputs/EXPERIMENTS.md` |
| Hình | `outputs/figures/`, nên xem trước: `internal_vs_external_gap.png`, `source_bias.png`, `preprocessing_ablation.png`, `leave_one_source_out.png` |
| Dataset được xây dựng thế nào, nhiễu nhãn ra sao | `data/phishvn/README.md`, `data/phishvn/docs/datasheet.md` |
| Vì sao không dùng dataset UCI PhiUSIIL | `archive/phiusiil/README.md` |

---

## 11. Câu hỏi thường gặp

**Accuracy 90% có thấp không?** Không, đó là con số trung thực: chia theo thời gian, không trùng domain, đã bỏ `is_https`. Dataset UCI PhiUSIIL cho 100%, nhưng con số đó đến từ leakage, và ra dữ liệu thật chỉ còn 50% (xem `archive/phiusiil/`).

**Vì sao không dùng luôn TF-IDF làm model chính?** Vì nó được phát hiện **sau khi** test đã mở cho model chính. Đổi model chính dựa trên điểm test là vi phạm quy tắc ở 6.1. TF-IDF được báo cáo riêng như một thí nghiệm thứ hai.

**Muốn thêm model hoặc cách xử lý mới?** Model: thêm vào `grids()` trong `pipeline.py`. Cách xử lý: thêm một mục vào `build_variants()` trong `experiments.py`. Luôn chọn trên val.

**Nhãn là gì?** `1` = phishing, `0` = benign. Lớp dương (dùng để tính precision, recall, F1) luôn là phishing.

---

## Nguồn dữ liệu
- **PhishVN v3.1.0** (CC BY 4.0): Mendeley DOI `10.17632/b97hxbxtpd.4`. Cách trích dẫn: `data/phishvn/CITATION.cff`.
- **Hassan-Albattra `balanced_urls.csv`** (MIT): chỉ dùng để đánh giá ngoài.
- **JPCERTCC `phishurl-list`** (2025-01 đến 2025-06): chỉ dùng để đo recall.
