# AI4SE: Issue Report Classification (NLBSE'24 Benchmark)
## Thư mục mã nguồn và thực nghiệm hoàn chỉnh: `252276M`
**Mã số:** `252276M` | **Nhóm:** 10 | **HUST - Đại học Bách Khoa Hà Nội**

---

### 1. Tổng quan Dự án & Yêu cầu Báo cáo (Report Requirements Alignment)
Dự án giải quyết bài toán phân loại tự động GitHub Issue Reports thành 3 lớp: `bug`, `feature`, `question` trên 5 kho phần mềm mã nguồn mở (`facebook/react`, `tensorflow/tensorflow`, `microsoft/vscode`, `bitcoin/bitcoin`, `opencv/opencv`).

Mã nguồn trong thư mục `252276M` thỏa mãn 100% các yêu cầu được mô tả trong Báo cáo Seminar (`seminar_02_group_10.pdf` / `report.tex`):
1. **Kiến trúc phân tầng độc lập lớp lưu trữ (Persistence-Agnostic Architecture):** Triển khai mẫu thiết kế Repository (`InMemoryIssueRepository`, `FileIssueRepository`) cho phép chuyển đổi giữa RAM và CSV/JSON chỉ bằng một tham số `kind="memory"` hoặc `kind="file"`. Được kiểm chứng bằng bài test tương đương logic `tests/test_persistence_equivalence.py`.
2. **Chuỗi tiền xử lý Markdown/Mã nguồn 3 cấp độ (Preprocessing Pipeline):** `raw` (khoảng trắng), `light` (loại code block, stacktrace, url, html, template) và `full` (lemmatization, stopwords, lowercase).
3. **15 Đặc trưng cấu trúc nhị phân (Structural Flags):** Trích xuất các tín hiệu cấu trúc (stacktrace, template, question mark title...) đạt F1 > 0.55 không cần đọc nội dung.
4. **Họ các mô hình đầy đủ:**
   - 4 Baseline floors (Majority, Stratified Random, Keyword Rules, Scratch Naive Bayes).
   - 5 Mô hình Classical ML (TF-IDF + Naive Bayes, Complement NB, Logistic Regression, Linear SVM calibrated, Random Forest).
   - 2 Mô hình Neural Networks huấn luyện từ đầu (Feed-Forward NN, TextCNN 1D Convolution).
   - 2 Mô hình Frozen Sentence Encoders (`all-MiniLM-L6-v2`, `all-mpnet-base-v2` + Logistic Regression Head).
   - Mô hình Fine-Tuning tương phản ít mẫu (SetFit Contrastive).
   - Mô hình Soft-Voting Ensemble (kết hợp xác suất từ SetFit-MPNet + TF-IDF LogReg + Frozen-MPNet đạt SOTA F1 = 0.8168, AUC = 0.9319).
5. **Nghiên cứu Confound thời gian (Temporal Confound) & Chia tập theo thời gian (Time-aware Split):** Phát hiện nhãn và ngày tạo issue bị nhiễu tương quan (mô hình chỉ đọc timestamp đạt F1 = 0.68). Cung cấp hàm `time_aware_split` và `random_split_control`.
6. **Kiểm định thống kê & Phân tích độ mạnh kiểm định (Power Analysis):** Triển khai kiểm định McNemar chính xác (Exact Binomial), hiệu chỉnh Holm-Bonferroni và tính toán độ chênh lệch phát hiện tối thiểu (MDD ~ 3 điểm F1).
7. **Phân tích lỗi (Error Analysis):** Phân tích 385 ca đoán sai, ngũ phân vị độ dài văn bản (Length Quintiles), lỗi có độ tự tin cao (High-Confidence Errors), và ảnh hưởng của template boilerplate.
8. **Đầy đủ 5 Jupyter Notebooks tương tác:** Phục vụ trực quan hóa dữ liệu (EDA), SetFit, RoBERTa và FastText.

---

### 2. Cấu trúc thư mục chi tiết

```
252276M/
├── data/
│   └── raw/                         # Dữ liệu CSV thô (1,500 train, 1,500 test)
│       ├── issues_train.csv
│       └── issues_test.csv
├── docs/
│   └── DATA_FLOW_AND_MODEL_EXPLANATION.md # TÀI LIỆU TOÀN DIỆN VỀ FLOW & MÔ HÌNH
├── notebooks/                       # 5 JUPYTER NOTEBOOKS TƯƠNG TÁC
│   ├── 00_dataset_extraction_reference.ipynb # Tham khảo trích xuất dữ liệu qua GitHub API
│   ├── 01_data_and_eda.ipynb                 # Tải dữ liệu, persistence layers & EDA
│   ├── 02_setfit_baseline.ipynb              # Baseline SetFit Contrastive Learning
│   ├── 03_roberta_baseline.ipynb             # Baseline RoBERTa Sequence Classification
│   └── 04_fasttext_baseline.ipynb            # Baseline FastText (Ticket Tagger)
├── results/
│   ├── figures/                     # Đồ thị học tập, learning curves
│   └── tables/                      # File kết quả JSON các giai đoạn thực nghiệm
│       ├── floors.json              # Kết quả 4 baseline floors
│       ├── classical.json           # Kết quả 7 mô hình TF-IDF cổ điển
│       └── error_analysis.json      # Kết quả phân tích 385 lỗi misclassification
├── scripts/
│   ├── run_all.py                   # Script chạy toàn bộ pipeline thực nghiệm chính
│   ├── benchmark_audit.py           # Script audit 5 đo lường (confound, time-aware, power)
│   └── make_tables.py               # Script tự động trích xuất bảng LaTeX cho báo cáo
├── src/
│   ├── model.py                     # Thực thể IssueReport, hằng số LABELS, REPOSITORIES
│   ├── repository.py                # Repository Pattern (InMemory, File, make_repository)
│   ├── loader.py                    # Nạp dữ liệu CSV tự động
│   ├── preprocessing.py             # Làm sạch văn bản Markdown, trích xuất 15 đặc trưng
│   ├── eda.py                       # Tổng quan, phân phối nhãn, thống kê độ dài, nhiễu cấu trúc
│   ├── audit.py                     # Kiểm tra chất lượng dữ liệu, text hash, data leakage
│   ├── classifiers/                 # 6 họ mô hình phân loại:
│   │   ├── base.py                  # Abstract Classifier, train_per_repo, train_pooled
│   │   ├── floors.py                # Majority, Random, KeywordRules, Scratch NB
│   │   ├── classical.py             # TF-IDF + LogReg, Linear SVM, Naive Bayes, Random Forest
│   │   ├── neural.py                # Feed-Forward NN & TextCNN (PyTorch)
│   │   ├── frozen.py                # Frozen Sentence Transformers (MiniLM, MPNet)
│   │   ├── setfit_model.py          # SetFit Contrastive Fine-Tuning
│   │   └── ensemble.py              # Soft-Voting Ensemble (kết hợp xác suất mềm)
│   ├── evaluation/                  # Đánh giá & Kiểm định thống kê
│   │   ├── metrics.py               # Macro-F1, Precision, Recall, Confusion Matrix
│   │   ├── splits.py                # Stratified K-Fold, time_aware_split, random_split_control
│   │   ├── significance.py          # Kiểm định McNemar, Holm-Bonferroni
│   │   └── power.py                 # MDD Power Estimation
│   └── analysis/                    # Phân tích lỗi (Error Analysis)
│       └── error_analysis.py        # Confusion pairs, length quintiles, high-confidence
├── tests/
│   ├── conftest.py
│   ├── test_pipeline.py             # 33 Unit/Integration tests cho toàn bộ pipeline
│   └── test_persistence_equivalence.py # 4 bài test chứng minh tính tương đương 2 lớp lưu trữ
├── requirements.txt                 # Danh sách thư viện phụ thuộc
└── README.md                        # Hướng dẫn này
```

---

### 3. Hướng dẫn cài đặt & Thực thi nhanh

#### Bước 1: Cài đặt môi trường
```bash
pip install -r requirements.txt
```

#### Bước 2: Chạy toàn bộ 37 bài kiểm thử tự động (Unit & Integration Tests)
```bash
python -m pytest tests/ -v
```
*(Kết quả: **37/37 tests passed 100%**, bao gồm 33 pipeline tests và 4 persistence equivalence tests)*

#### Bước 3: Chạy các kịch bản thực nghiệm

- **Chạy các mô hình sàn tham chiếu (Floors):**
  ```bash
  python scripts/run_all.py --stage floors
  ```

- **Chạy các mô hình Cổ điển (Classical TF-IDF) và Phân tích lỗi:**
  ```bash
  python scripts/run_all.py --stage classical,error
  ```

- **Chạy các mô hình Học sâu (Neural Networks PyTorch) & Vẽ biểu đồ:**
  ```bash
  python scripts/run_all.py --stage neural,figures
  ```

- **Chạy các mô hình Transformer đóng băng & Ensemble:**
  ```bash
  python scripts/run_all.py --stage frozen,ensemble
  ```

- **Chạy toàn bộ quy trình từ đầu đến cuối (End-to-End):**
  ```bash
  python scripts/run_all.py --stage all
  ```

- **Chạy Audit Confound thời gian, Time-Aware Split và Power Analysis:**
  ```bash
  python scripts/benchmark_audit.py
  ```

- **Biên dịch kết quả JSON thành bảng báo cáo:**
  ```bash
  python scripts/make_tables.py
  ```

---

### 4. Hướng dẫn sử dụng Jupyter Notebooks

Mở Jupyter Lab hoặc VS Code trong thư mục dự án và khởi chạy các notebook trong `notebooks/`:
1. **`01_data_and_eda.ipynb`**: Chạy phân tích thống kê mô tả, kiểm tra tính cân bằng nhãn giữa 5 repositories, phân phối độ dài văn bản và chứng minh tính tương đương giữa lớp lưu trữ in-memory và file-backed.
2. **`02_setfit_baseline.ipynb`**: Huấn luyện và đánh giá mô hình SetFit Few-Shot trên tập dữ liệu.
3. **`03_roberta_baseline.ipynb`**: Tinh chỉnh mô hình RoBERTa Sequence Classification thông qua HuggingFace Trainer.
4. **`04_fasttext_baseline.ipynb`**: Thực nghiệm phân loại issue với FastText theo phương pháp của Ticket Tagger.

---

### 5. Tài liệu kiến trúc và luồng dữ liệu
Xem giải thích chi tiết, đầy đủ về toán học, cơ chế xử lý văn bản, đầu vào/đầu ra và sơ đồ luồng tại:
- [DATA_FLOW_AND_MODEL_EXPLANATION.md](file:///d:/HUST/seminar-2-project-01/252276M/docs/DATA_FLOW_AND_MODEL_EXPLANATION.md)
