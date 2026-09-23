# BÁO CÁO TOÀN DIỆN VỀ QUY TRÌNH XỬ LÝ DỮ LIỆU VÀ KIẾN TRÚC MÔ HÌNH
## Dự án: Phân loại Issue Report trong Kỹ nghệ Phần mềm (NLBSE'24 / AI4SE)
**Mã nguồn sinh viên:** `252276M` | **Nhóm:** 10 | **HUST - Đại học Bách Khoa Hà Nội**

---

## MỤC LỤC
1. [TỔNG QUAN HỆ THỐNG VÀ BÀI TOÁN](#1-tổng-quan-hệ-thống-và-bài-toán)
2. [FLOW QUÁ TRÌNH XỬ LÝ DỮ LIỆU (END-TO-END DATA PIPELINE)](#2-flow-quá-trình-xử-lý-dữ-liệu-end-to-end-data-pipeline)
   - 2.1. Đọc và nạp dữ liệu thô (Raw Data Ingestion)
   - 2.2. Biểu diễn thực thể và Mô hình Repository Pattern
   - 2.3. Chuỗi tiền xử lý Markdown/Mã nguồn chuyên sâu (3 cấp độ)
   - 2.4. Trích xuất đặc trưng cấu trúc (Structural Features)
   - 2.5. Cơ chế nhân trọng số tiêu đề (Title Weighting & Truncation)
   - 2.6. Biểu diễn vector hóa đặc trưng (Vectorization)
3. [KIẾN TRÚC VÀ CƠ CHẾ HOẠT ĐỘNG CỦA CÁC MÔ HÌNH](#3-kiến-trúc-và-cơ-chế-hoạt-động-của-các-mô-hình)
   - 3.1. Nhóm mô hình mốc tham chiếu (Reference Floors)
   - 3.2. Nhóm mô hình Học máy cổ điển (Classical ML - TF-IDF)
   - 3.3. Nhóm mô hình Học sâu (Neural Networks - PyTorch)
   - 3.4. Nhóm mô hình Transformer đóng băng (Frozen Sentence Encoders)
   - 3.5. Nhóm mô hình Tinh chỉnh tương phản (SetFit Few-Shot)
   - 3.6. Mô hình Ensemble kết hợp xác suất mềm (Soft-Voting Ensemble)
4. [QUY CHUẨN HUẤN LUYỆN VÀ ĐÁNH GIÁ (EVALUATION PROTOCOL)](#4-quy-chuẩn-huấn-luyện-và-đánh-giá-evaluation-protocol)
5. [PHÂN TÍCH LỖI (ERROR ANALYSIS)](#5-phân-tích-lỗi-error-analysis)
6. [HƯỚNG DẪN CÀI ĐẶT VÀ VẬN HÀNH TOÀN BỘ MÃ NGUỒN](#6-hướng-dẫn-cài-đặt-và-vận-hành-toàn-bộ-mã-nguồn)

---

## 1. TỔNG QUAN HỆ THỐNG VÀ BÀI TOÁN

### 1.1. Bài toán
Trong các dự án phần mềm mã nguồn mở trên GitHub, hàng trăm issue được mở ra mỗi ngày. Việc gán nhãn thủ công tốn nhiều công sức của maintainer. Bài toán đặt ra là: **Tự động phân loại một Issue Report thành 1 trong 3 nhãn định trước:**
1. **`bug`**: Báo cáo lỗi kỹ thuật, sự cố phần mềm (crash, exception, sai logic).
2. **`feature`**: Yêu cầu tính năng mới, đề xuất cải tiến giao diện/chức năng (enhancement, new feature).
3. **`question`**: Câu hỏi thắc mắc về cách sử dụng, tài liệu, cấu hình cài đặt.

### 1.2. Tập dữ liệu (NLBSE'24 Benchmark)
- Dữ liệu thu thập từ **5 kho mã nguồn mở hàng đầu**:
  - `facebook/react` (Frontend JavaScript framework)
  - `tensorflow/tensorflow` (Machine Learning library)
  - `opencv/opencv` (Computer Vision library)
  - `bitcoin/bitcoin` (Cryptocurrency node)
  - `scikit-learn/scikit-learn` (Python ML library)
- **Tập Train (1,500 mẫu)**: 5 repos × 3 nhãn × 100 mẫu/nhãn (cân bằng hoàn hảo).
- **Tập Test (1,500 mẫu)**: 5 repos × 3 nhãn × 100 mẫu/nhãn (cân bằng hoàn hảo).
- **Định dạng dữ liệu**: CSV gồm các trường:
  - `repo`: Tên repository (ví dụ `facebook/react`).
  - `created_at`: Thời gian tạo issue.
  - `title`: Tiêu đề của issue.
  - `body`: Nội dung chi tiết của issue (thường chứa Markdown, log lỗi, code block, checklist).
  - `label`: Nhãn thực tế (`bug`, `feature`, hoặc `question`).

---

## 2. FLOW QUÁ TRÌNH XỬ LÝ DỮ LIỆU (END-TO-END DATA PIPELINE)

Quy trình xử lý dữ liệu được thiết kế thành một chuỗi khép kín, từ dạng chuỗi thô đến dạng ma trận đặc trưng sẵn sàng cho các thuật toán học máy:

```
[Raw CSV File] 
      │
      ▼ (Step 1: Loader & Schema Validation)
[IssueReport Dataclass & Repository Engine]
      │
      ▼ (Step 2: Markdown & Domain-Aware Text Cleaning)
   ├── Level 1: RAW (chuẩn hóa khoảng trắng)
   ├── Level 2: LIGHT (bóc Markdown, tracebacks, URLs, HTML)
   └── Level 3: FULL (lemmatization, stopwords, lowercase, code strip)
      │
      ▼ (Step 3: Title Weighting & Truncation)
   Ghép: (Title * k_weight) + " " + Cleaned_Body -> Max_Words Tokens
      │
      ▼ (Step 4: Vectorization / Embedding)
   ├── Classical: TF-IDF (1-2 grams, sublinear TF) -> Ma trận thưa (N x 10,000)
   ├── Neural: PyTorch Tensor Vocabulary (TextCNN) hoặc TF-IDF (FFNN)
   └── Transformer: Dense Embedding (all-MiniLM 384d / all-mpnet 768d)
      │
      ▼
[Mô hình huấn luyện & Dự đoán (Per-Repository Protocol)]
```

---

### 2.1. Đọc và nạp dữ liệu thô (Raw Data Ingestion)
- **File thực thi:** `src/loader.py`
- **Thách thức:** Nội dung `body` của các issue kỹ thuật thường rất dài (chứa full stacktrace hoặc log file có thể lên tới 20,000 từ). Giới hạn đọc CSV mặc định của Python (`sys.maxsize`) có thể bị tràn bộ nhớ nếu không tăng `csv.field_size_limit`.
- **Giải pháp:** `loader.py` tự động mở rộng `csv.field_size_limit` động, phân tích trường thời gian `created_at` (hỗ trợ cả định dạng ISO có timezone lẫn dấu cách), đồng thời tự sinh `issue_id` khi CSV không cung cấp cột định danh riêng.

### 2.2. Biểu diễn thực thể và Mô hình Repository Pattern
- **File thực thi:** `src/ai4se/model.py` và `src/ai4se/repository.py`
- Thay vì sử dụng DataFrame phân mảnh dễ gây rò rỉ dữ liệu (data leakage), toàn bộ logic quản lý dữ liệu tuân theo **Repository Pattern** chuẩn kiến trúc phần mềm:
  - `IssueReport`: Dataclass bất biến chứa thông tin một issue (`repo`, `title`, `body`, `label`, `created_at`). Cung cấp thuộc tính tổng hợp `.text = title + " " + body`.
  - `InMemoryIssueRepository`: Lưu trữ danh sách issue trên RAM, cung cấp các hàm nghiệp vụ: lọc theo repo (`by_repo`), lọc theo nhãn (`by_label`), trích xuất cặp văn bản-nhãn (`texts_and_labels`), biến đổi hàng loạt (`apply`).
  - `FileIssueRepository`: Đọc và ghi trực tiếp từ file CSV/JSON trên ổ đĩa.

---

### 2.3. Chuỗi tiền xử lý Markdown/Mã nguồn chuyên sâu (3 cấp độ)
- **File thực thi:** `src/preprocessing.py`
- Do issue report là văn bản kỹ thuật pha trộn ngôn ngữ tự nhiên và mã lệnh, hệ thống triển khai 3 cấp độ làm sạch phù hợp với từng loại mô hình:

#### Cấp độ 1: `raw`
- Chỉ chuẩn hóa ký tự khoảng trắng dư thừa (`\r\n`, tabs, space liên tiếp).
- Giữ nguyên cấu trúc câu, cú pháp viết hoa/thường, code blocks. Thích hợp cho các mô hình Transformer sâu vốn học được cả ký tự phân cách.

#### Cấp độ 2: `light` (Khuyên dùng cho Neural & Frozen Transformer)
1. **Loại bỏ khối mã (Fenced Code Blocks):** Cắt bỏ các đoạn nằm trong ``` ``` (chứa log hệ thống dài dòng gây nhiễu từ vựng).
2. **Loại bỏ mã nội dòng (Inline Code):** Chuyển các đoạn \`code\` thành từ ngữ thông thường.
3. **Loại bỏ liên kết (URLs & Links):** Thay thế đường dẫn `http://...` và markdown link `[text](url)` bằng khoảng trắng để tránh ghi nhớ các hash ngẫu nhiên.
4. **Loại bỏ thẻ HTML:** Làm sạch các thẻ `<details>`, `<img>`, `<b>` thường dùng trong mẫu GitHub.
5. **Khử mẫu định dạng GitHub (Issue Templates):** Lọc bỏ tiêu đề mặc định như `### Steps to reproduce`, `### Expected behavior`, `### Actual behavior`.

#### Cấp độ 3: `full` (Tối ưu cho Classical TF-IDF)
Kế thừa toàn bộ cấp độ `light`, bổ sung:
- Chuyển toàn bộ về chữ thường (lowercase).
- Lọc bỏ dấu câu và ký tự đặc biệt (chỉ giữ lại ký tự chữ và số).
- Lọc bỏ Stopwords tiếng Anh (thư viện NLTK: *the, is, at, which, ...*).
- Đưa từ về nguyên mẫu (WordNet Lemmatization: *running -> run, crashed -> crash*).

---

### 2.4. Trích xuất đặc trưng cấu trúc (Structural Features)
- **File thực thi:** `src/preprocessing.py:extract_structural_features`
- Để không bỏ phí tín hiệu kỹ thuật khi đã lọc code, hệ thống trích xuất **15 đặc trưng nhị phân (0 hoặc 1)** bổ trợ:
  1. `has_code_block`: Văn bản có chứa khối code hay không (thường xuất hiện ở `bug`).
  2. `has_stack_trace`: Có chứa từ khóa lỗi như `traceback`, `exception`, `error at line`.
  3. `has_github_template`: Có chứa định dạng tiêu đề mẫu báo cáo GitHub không.
  4. `has_question_mark_title`: Tiêu đề có kết thúc bằng dấu hỏi `?` (tín hiệu mạnh của `question`).
  5. `has_log_output`: Có chứa tiền tố log dạng timestamp hoặc `[INFO]`, `[ERROR]`.
  6. `has_repro_steps`: Có chứa các bước tái hiện (Repro steps).
  7. `has_expected_vs_actual`: Có so sánh hành vi mong muốn và thực tế.
  8. `has_checkboxes`: Có chứa markdown checklist `- [x]` hoặc `- [ ]`.
  9. `has_version_info`: Có thông tin phiên bản OS/phần mềm.
  10. ... và các cờ nhận diện từ khóa đề xuất (`feature`, `enhancement`, `suggest`).

---

### 2.5. Cơ chế nhân trọng số tiêu đề (Title Weighting & Truncation)
- **Ý nghĩa:** Tiêu đề (`title`) của một issue là phần cô đọng súc tích nhất do người dùng viết tóm tắt mục tiêu (ví dụ: *"Add dark mode support"* hoặc *"NullPointerException on startup"*). Nếu trộn trực tiếp `title + body` với tỉ lệ 1:1, những từ khóa then chốt trong tiêu đề sẽ bị lấn át bởi hàng nghìn từ trong phần thân.
- **Giải pháp:** Nhân bản tiêu đề $k$ lần trước khi nối với phần thân:
  $$\text{Text}_{\text{processed}} = (\text{Title} \times k) + \text{ " " } + \text{Body}_{\text{cleaned}}$$
  - Trong các mô hình Classical ML: $k = 3$, cắt tối đa 400 từ (`max_words=400`).
  - Trong các mô hình Neural/Transformer: $k = 2$, cắt tối đa 256–300 từ.

---

### 2.6. Biểu diễn vector hóa đặc trưng (Vectorization)
1. **TF-IDF N-grams (Classical & FFNN):**
   - N-gram range: $(1, 2)$ (từ đơn và cụm 2 từ liền kề).
   - `sublinear_tf=True`: Áp dụng công thức $1 + \log(\text{tf})$ nhằm giảm tầm ảnh hưởng của các từ xuất hiện quá nhiều lần liên tiếp.
   - `min_df=2`: Loại bỏ từ chỉ xuất hiện duy nhất 1 lần (từ hiếm, lỗi chính tả).
   - `max_features=10000`: Giữ lại 10,000 đặc trưng từ vựng có độ phân tách cao nhất.
2. **Dense Sentence Embeddings (Frozen & SetFit):**
   - `all-MiniLM-L6-v2`: Chuyển đổi văn bản thành vector đậm đặc kích thước $d = 384$.
   - `all-mpnet-base-v2`: Mô hình SOTA sentence embedding, chuyển đổi thành vector kích thước $d = 768$.

---

## 3. KIẾN TRÚC VÀ CƠ CHẾ HOẠT ĐỘNG CỦA CÁC MÔ HÌNH

Hệ thống triển khai 6 họ mô hình tương ứng với các chương phân tích trong báo cáo:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        HỌ CÁC MÔ HÌNH TRONG DỰ ÁN                      │
├──────────────────┬──────────────────┬─────────────────┬────────────────┤
│ Nhóm             │ Tên mô hình      │ Đầu vào (Input) │ Đầu ra (Output)│
├──────────────────┼──────────────────┼─────────────────┼────────────────┤
│ 1. Floors        │ Majority         │ Text (bỏ qua)   │ Nhãn đa số     │
│                  │ StratifiedRandom │ Text (bỏ qua)   │ Nhãn ngẫu nhiên│
│                  │ KeywordRules     │ Text thô        │ Nhãn theo luật │
│                  │ ScratchNB        │ Bag-of-Words    │ Xác suất 3 lớp │
├──────────────────┼──────────────────┼─────────────────┼────────────────┤
│ 2. Classical     │ Naive Bayes      │ TF-IDF (10,000d)│ Nhãn & Xác suất│
│                  │ Complement NB    │ TF-IDF (10,000d)│ Nhãn & Xác suất│
│                  │ Tuned LogReg     │ TF-IDF (10,000d)│ Nhãn & Xác suất│
│                  │ Tuned Linear SVM │ TF-IDF (10,000d)│ Nhãn & Xác suất│
│                  │ Random Forest    │ TF-IDF (10,000d)│ Nhãn & Xác suất│
├──────────────────┼──────────────────┼─────────────────┼────────────────┤
│ 3. Neural Net    │ FFNN (PyTorch)   │ TF-IDF (10,000d)│ Softmax 3 lớp  │
│                  │ TextCNN (PyTorch)│ Token Sequence  │ Softmax 3 lớp  │
├──────────────────┼──────────────────┼─────────────────┼────────────────┤
│ 4. Frozen Enc    │ MiniLM + LogReg  │ Vector 384d     │ Nhãn & Xác suất│
│                  │ MPNet + LogReg   │ Vector 768d     │ Nhãn & Xác suất│
├──────────────────┼──────────────────┼─────────────────┼────────────────┤
│ 5. Contrastive   │ SetFit MPNet     │ Text câu        │ Nhãn & Xác suất│
├──────────────────┼──────────────────┼─────────────────┼────────────────┤
│ 6. Ensemble      │ Soft-Voting      │ 3 Model Proba   │ Trung bình cộng│
└──────────────────┴──────────────────┴─────────────────┴────────────────┘
```

---

### 3.1. Nhóm mô hình mốc tham chiếu (Reference Floors)
- **File thực thi:** `src/ai4se/classifiers/floors.py`
- Mục đích: Đóng vai trò là "sàn hiệu năng" (lower bounds) để kiểm chứng xem các thuật toán ML thực sự học được tri thức hay chỉ đoán mò.
1. **MajorityClassifier:** Luôn dự đoán nhãn chiếm số lượng nhiều nhất trong tập huấn luyện. Với bài toán 3 lớp cân bằng (100 mẫu mỗi lớp), F1 đạt đúng $\approx 0.1667$.
2. **StratifiedRandomClassifier:** Sinh nhãn ngẫu nhiên theo phân phối tiên nghiệm của tập train. Kết quả F1 đạt $\approx 0.33$.
3. **KeywordRulesClassifier:** Phân loại hoàn toàn bằng bộ luật từ khóa chuyên biệt trong ngành công nghệ thông tin (ví dụ: nếu có `traceback`, `fail`, `segfault` $\rightarrow$ `bug`; nếu có `would like`, `please add`, `feat` $\rightarrow$ `feature`; nếu có `how do i`, `how to`, `where can` $\rightarrow$ `question`). Mô hình này không cần huấn luyện mà đạt F1 ấn tượng $\approx 0.5976$.
4. **ScratchNaiveBayesClassifier:** Cài đặt thuật toán Multinomial Naive Bayes hoàn toàn bằng Python thuần và NumPy (không dùng scikit-learn). Tính xác suất hậu nghiệm dựa trên định lý Bayes với Laplace Smoothing ($\alpha = 1.0$):
   $$P(c|d) \propto P(c) \prod_{w \in d} \frac{\text{count}(w, c) + 1}{\sum_{w'} \text{count}(w', c) + |V|}$$
   Đạt F1 $\approx 0.6852$.

---

### 3.2. Nhóm mô hình Học máy cổ điển (Classical ML - TF-IDF)
- **File thực thi:** `src/ai4se/classifiers/classical.py`
- Kết hợp ma trận TF-IDF 10,000 chiều với các thuật toán học máy giám sát chuẩn:
1. **MultinomialNB & ComplementNB:** Giả định các từ xuất hiện độc lập có điều kiện theo lớp. ComplementNB được tối ưu đặc biệt để khắc phục sự mất cân bằng giữa các lớp.
2. **Tuned Logistic Regression:** Phân loại tuyến tính với hàm mất mát Cross-Entropy. Điều chuẩn $L_2$ với siêu tham số tối ưu $C = 2.0$, sử dụng thuật toán tối ưu `lbfgs`. Đây là một trong những mô hình hoạt động hiệu quả và ổn định nhất trên văn bản thưa.
3. **Tuned Linear SVM:** Tìm siêu phẳng phân tách có lề (margin) cực đại. Do SVM không xuất ra xác suất trực tiếp, hệ thống bọc ngoài bằng `CalibratedClassifierCV(cv=3)` (Platt Scaling) để chuẩn hóa khoảng cách tới siêu phẳng thành ma trận xác suất hợp lệ có tổng bằng 1.
4. **Random Forest:** Tập hợp 100 cây quyết định ngẫu nhiên với độ sâu cực đại `max_depth=30`.

---

### 3.3. Nhóm mô hình Học sâu (Neural Networks - PyTorch)
- **File thực thi:** `src/ai4se/classifiers/neural.py`
- Huấn luyện với cơ chế **Early Stopping** (dừng sớm khi loss trên tập validation không giảm sau 4 epochs liên tiếp) để ngăn chặn hiện tượng quá khớp (overfitting) do tập dữ liệu per-repo chỉ có 300 mẫu:

#### 1. Feed-Forward Neural Network (FFNN)
- **Đầu vào:** Vector TF-IDF 10,000 chiều.
- **Kiến trúc mạng:**
  $$\text{Input (10,000)} \xrightarrow{\text{Linear}} \text{Hidden 1 (256)} \xrightarrow{\text{BatchNorm + ReLU}} \xrightarrow{\text{Dropout (0.4)}} \text{Hidden 2 (64)} \xrightarrow{\text{ReLU}} \xrightarrow{\text{Dropout (0.3)}} \text{Output (3)}$$
- **Hàm mất mát & Tối ưu:** `CrossEntropyLoss`, `AdamW(lr=1e-3, weight_decay=1e-4)`.
- **Đầu ra:** 3 giá trị logit $\rightarrow$ Hàm `Softmax` để chuyển thành xác suất 3 lớp: $[P(\text{bug}), P(\text{feature}), P(\text{question})]$.

#### 2. TextCNN (Mạng nơ-ron tích chập 1D cho văn bản - Kim, 2014)
- **Đầu vào:** Chuỗi chỉ số từ (Tokenized indices) với độ dài cố định 256 từ.
- **Kiến trúc mạng:**
  1. *Embedding Layer:* Nhúng từ thành vector 128 chiều: Tensor $(\text{Batch}, 256, 128)$.
  2. *Multi-Scale 1D Convolutions:* Áp dụng song song 3 bộ lọc tích chập với kích thước cửa sổ khác nhau:
     - Kernel size = 2 (bắt cụm 2 từ liền kề), 64 bộ lọc $\rightarrow$ Feature map 1.
     - Kernel size = 3 (bắt cụm 3 từ liền kề), 64 bộ lọc $\rightarrow$ Feature map 2.
     - Kernel size = 4 (bắt cụm 4 từ liền kề), 64 bộ lọc $\rightarrow$ Feature map 3.
  3. *Global Max-over-Time Pooling:* Lấy đặc trưng nổi trội nhất trên mỗi bộ lọc $\rightarrow$ mỗi nhánh cho ra vector 64 chiều.
  4. *Ghép nối (Concatenation):* Gộp 3 nhánh lại thành 1 vector đặc trưng thống nhất kích thước $64 \times 3 = 192$ chiều.
  5. *Phân loại:* `Dropout(0.5)` $\rightarrow$ `Linear(192, 3)` $\rightarrow$ `Softmax`.

---

### 3.4. Nhóm mô hình Transformer đóng băng (Frozen Sentence Encoders)
- **File thực thi:** `src/ai4se/classifiers/frozen.py`
- **Nguyên lý:** Thay vì huấn luyện lại toàn bộ hàng trăm triệu trọng số của Transformer (rất dễ overfit trên 300 mẫu), ta **đóng băng (freeze)** toàn bộ tham số của mô hình ngôn ngữ lớn đã được huấn luyện trước trên hàng tỷ câu, chỉ dùng nó như một bộ trích xuất đặc trưng ngữ nghĩa (Feature Extractor):
  $$\text{Text} \xrightarrow{\text{SentenceTransformer (Frozen)}} \mathbf{z} \in \mathbb{R}^d \xrightarrow{\text{Logistic Regression Classifier Head}} \hat{y}$$
- **Mô hình triển khai:**
  1. `all-MiniLM-L6-v2`: Trích xuất vector ngữ nghĩa đậm đặc $d = 384$.
  2. `all-mpnet-base-v2`: Trích xuất vector ngữ nghĩa đậm đặc $d = 768$.
- **Ưu điểm:** Khắc phục triệt để nhược điểm từ vựng đồng nghĩa/trái nghĩa mà TF-IDF gặp phải, đồng thời tốc độ huấn luyện phần đầu phân loại chỉ mất chưa tới 1 giây.

---

### 3.5. Nhóm mô hình Tinh chỉnh tương phản (SetFit Few-Shot)
- **File thực thi:** `src/ai4se/classifiers/setfit_model.py`
- **Nguyên lý:** SetFit (Sentence Transformer Fine-tuning) là phương pháp SOTA cho bài toán ít dữ liệu (few-shot).
  - Giai đoạn 1: Sinh các cặp câu (positive pairs cùng nhãn, negative pairs khác nhãn). Tinh chỉnh Sentence Transformer thông qua hàm mất mát tương phản Cosine Similarity Loss.
  - Giai đoạn 2: Trích xuất embedding từ mô hình đã được tinh chỉnh tương phản và huấn luyện bộ phân loại Logistic Regression.
- Cho độ chính xác vượt trội khi số lượng mẫu của mỗi repository chỉ có 100 mẫu/nhãn.

---

### 3.6. Mô hình Ensemble kết hợp xác suất mềm (Soft-Voting Ensemble)
- **File thực thi:** `src/ai4se/classifiers/ensemble.py`
- **Cơ chế Soft-Voting:**
  Thay vì bỏ phiếu cứng (Hard Voting: chỉ lấy nhãn có số phiếu cao nhất), hệ thống sử dụng phương pháp trung bình cộng trọng số của phân phối xác suất dự báo từ các mô hình thành phần:
  $$P_{\text{ensemble}}(c \mid x) = \sum_{m=1}^{M} w_m \cdot P_m(c \mid x)$$
  Trong đó $\sum w_m = 1$. Nhãn cuối cùng được chọn là lớp có xác suất tổng hợp cao nhất:
  $$\hat{y} = \arg\max_{c \in \{\text{bug}, \text{feature}, \text{question}\}} P_{\text{ensemble}}(c \mid x)$$
- **Tập hợp tốt nhất (Best Ensemble - Bảng 5.9 trong Báo cáo):**
  Kết hợp 3 nguồn tri thức đa dạng:
  1. **`SetFit-MPNet`** (Tri thức tương phản sâu ít mẫu).
  2. **`Tfidf-LogReg-Tuned`** (Tri thức từ vựng bề mặt và từ khóa kỹ thuật cụ thể).
  3. **`Frozen-MPNet`** (Tri thức biểu diễn câu khái quát toàn cục).
  - Kết quả đạt **Cross-repo Macro-F1 = 0.8168** và **AUC = 0.9319** (vượt xa tất cả các mô hình đơn lẻ).

---

## 4. QUY CHUẨN HUẤN LUYỆN VÀ ĐÁNH GIÁ (EVALUATION PROTOCOL)

### 4.1. Giao thức huấn luyện chuẩn NLBSE'24
- Cuộc thi quy định: Với mỗi phương pháp, ta phải huấn luyện **5 mô hình hoàn toàn độc lập cho 5 repository**:
  - Model 1: Train trên 300 mẫu của `facebook/react` $\rightarrow$ Test trên 300 mẫu của `react`.
  - Model 2: Train trên 300 mẫu của `tensorflow/tensorflow` $\rightarrow$ Test trên 300 mẫu của `tensorflow`.
  - Model 3: Train trên 300 mẫu của `opencv/opencv` $\rightarrow$ Test trên 300 mẫu của `opencv`.
  - Model 4: Train trên 300 mẫu của `bitcoin/bitcoin` $\rightarrow$ Test trên 300 mẫu của `bitcoin`.
  - Model 5: Train trên 300 mẫu của `scikit-learn/scikit-learn` $\rightarrow$ Test trên 300 mẫu của `scikit-learn`.
- Hàm thực thi: `classifiers/base.py:train_per_repo`.

### 4.2. Các chỉ số đo lường (Metrics)
- **File thực thi:** `src/ai4se/evaluation/metrics.py`
1. **Per-class Precision, Recall, F1:**
   $$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}, \quad F1_c = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$
2. **Repository Macro-F1:**
   $$F1_{\text{macro}}^{(r)} = \frac{1}{3} \left( F1_{\text{bug}}^{(r)} + F1_{\text{feature}}^{(r)} + F1_{\text{question}}^{(r)} \right)$$
3. **Cross-Repository Macro-F1 (Điểm xếp hạng chính):**
   $$\text{Cross-Repo } F1 = \frac{1}{5} \sum_{r=1}^{5} F1_{\text{macro}}^{(r)}$$
4. **Đa hạt giống (Multi-Seed Aggregation):** Chạy lặp trên 5 hạt giống ngẫu nhiên: `SEEDS = [41, 42, 43, 44, 45]` để lấy giá trị trung bình $\mu$ và độ lệch chuẩn $\sigma$, đảm bảo tính khách quan khoa học.

### 4.3. Kiểm định thống kê (Statistical Significance)
- **File thực thi:** `src/ai4se/evaluation/significance.py`
- **Kiểm định McNemar chính xác (Exact McNemar Test):** So sánh 2 mô hình dựa trên bảng ngẫu nhiên phân loại đúng/sai giữa 2 thuật toán:
  $$b = \text{Model 1 đúng & Model 2 sai}, \quad c = \text{Model 1 sai & Model 2 đúng}$$
  Tính p-value dựa trên phân phối nhị thức chính xác (Binomial Distribution).
- **Hiệu chỉnh Holm-Bonferroni:** Kiểm soát tỷ lệ lỗi Family-Wise Error Rate (FWER) khi so sánh cặp nhiều mô hình đồng thời.

---

## 5. PHÂN TÍCH LỖI (ERROR ANALYSIS)
- **File thực thi:** `src/ai4se/analysis/error_analysis.py`
- Hệ thống tự động phân tích sâu các trường hợp dự đoán sai:
1. **Ma trận nhầm lẫn (Confusion Patterns):** Cặp nhầm lẫn lớn nhất thường diễn ra giữa `feature` và `question` (khi người dùng hỏi: *"Làm thế nào để làm X?"* nhưng thực tế thư viện chưa hỗ trợ và mang ý đề xuất *"Hãy thêm X"*), hoặc giữa `bug` và `question` (khi người dùng không chắc code mình viết sai hay do thư viện bị lỗi).
2. **Phân tích theo ngũ phân vị độ dài (Length Quintiles):** Chia tập dữ liệu thành 5 nhóm độ dài từ ngắn nhất đến dài nhất. Kết quả chỉ ra rằng các issue quá ngắn (< 30 từ) có độ chính xác thấp nhất do thiếu ngữ cảnh, trong khi các issue có độ dài vừa phải (100 - 300 từ) đạt độ chính xác cao nhất.
3. **Lỗi có độ tự tin cao (High-Confidence Errors):** Trích xuất danh sách các issue mà mô hình dự đoán sai nhưng xác suất gán nhãn lại rất cao (> 90%). Khi đối chiếu thủ công, nhiều trường hợp thực chất là do người dùng gán sai nhãn gốc trong ground truth (Label Noise).
4. **Ảnh hưởng của Issue Template (Template Lift):** Khảo sát tỷ lệ xuất hiện của các phần tiêu đề mẫu GitHub. Các issue chứa template có xu hướng nghiêng hẳn về nhãn `bug` với tỷ lệ áp đảo.

---

## 6. HƯỚNG DẪN CÀI ĐẶT VÀ VẬN HÀNH TOÀN BỘ MÃ NGUỒN

### Cấu trúc thư mục `252276M`:
```
252276M/
├── data/
│   └── raw/
│       ├── issues_train.csv         # 1,500 issue tập huấn luyện
│       └── issues_test.csv          # 1,500 issue tập kiểm thử
├── docs/
│   └── DATA_FLOW_AND_MODEL_EXPLANATION.md  # Tài liệu giải thích chi tiết này
├── results/
│   ├── figures/                     # Đồ thị học tập (learning curves), biểu đồ
│   └── tables/                      # Kết quả JSON các giai đoạn (floors, classical...)
├── scripts/
│   └── run_all.py                   # Script chạy toàn bộ pipeline từ A đến Z
├── src/
│   ├── __init__.py
│   ├── model.py                     # Thực thể IssueReport, hằng số LABELS, REPOS
│   ├── repository.py                # Repository Pattern (InMemory, File)
│   ├── loader.py                    # Trình nạp dữ liệu từ CSV
│   ├── preprocessing.py             # Làm sạch văn bản Markdown, trích xuất đặc trưng
│   ├── classifiers/                 # Toàn bộ 6 họ mô hình phân loại
│   │   ├── base.py                  # Abstract Classifier, train_per_repo, train_pooled
│   │   ├── floors.py                # Majority, Random, Keyword, Scratch NB
│   │   ├── classical.py             # TF-IDF + LogReg, SVM, NB, RF
│   │   ├── neural.py                # FFNN & TextCNN (PyTorch)
│   │   ├── frozen.py                # Frozen Sentence Transformers (MiniLM, MPNet)
│   │   ├── setfit_model.py          # SetFit Contrastive Fine-Tuning
│   │   └── ensemble.py              # Soft-Voting Ensemble (kết hợp xác suất)
│   ├── evaluation/                  # Đánh giá & Thống kê
│   │   ├── metrics.py               # Macro-F1, Precision, Recall, Confusion Matrix
│   │   ├── splits.py                # Stratified K-Fold, Train/Val split
│   │   ├── significance.py          # McNemar Test, Holm-Bonferroni
│   │   └── power.py                 # MDD Power Estimation
│   └── analysis/                    # Phân tích lỗi (Error Analysis)
│       └── error_analysis.py        # Confusion pairs, length quintiles, high-conf
├── tests/
│   ├── conftest.py
│   └── test_pipeline.py             # 33 Unit/Integration tests tự động
├── requirements.txt                 # Danh sách thư viện phụ thuộc
└── README.md                        # Giới thiệu nhanh và lệnh thực thi
```

### Các lệnh vận hành:

1. **Chạy toàn bộ 33 Unit Tests tự động:**
   ```bash
   python -m pytest 252276M/tests/test_pipeline.py -v
   ```

2. **Chạy các mô hình mốc tham chiếu (Floors):**
   ```bash
   python 252276M/scripts/run_all.py --stage floors
   ```

3. **Chạy các mô hình Cổ điển (Classical TF-IDF) & Phân tích lỗi:**
   ```bash
   python 252276M/scripts/run_all.py --stage classical,error
   ```

4. **Chạy các mô hình Học sâu (Neural Networks PyTorch):**
   ```bash
   python 252276M/scripts/run_all.py --stage neural,figures
   ```

5. **Chạy các mô hình Transformer đóng băng & Ensemble:**
   ```bash
   python 252276M/scripts/run_all.py --stage frozen,ensemble
   ```

6. **Chạy toàn bộ quy trình (End-to-End full pipeline):**
   ```bash
   python 252276M/scripts/run_all.py --stage all
   ```
