# Túi Từ, TF-IDF và Biểu Diễn Văn Bản (Bag of Words, TF-IDF, and Text Representation)

> Đếm trước, suy ngẫm sau. Vào năm 2026, TF-IDF vẫn đánh bại các mô hình vector nhúng (embeddings) trên nhiều bài toán có mục tiêu xác định rõ ràng.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 02 (Linear Regression from Scratch)
**Time:** ~75 phút

## The Problem

Mô hình học máy cần những con số. Trong khi bạn chỉ có các chuỗi ký tự.

Mọi quy trình NLP đều phải giải quyết cùng một câu hỏi then chốt: Làm thế nào để biến đổi một luồng token có độ dài thay đổi thành một vector kích thước cố định mà một bộ phân loại có thể tiếp nhận? Câu trả lời đầu tiên mà ngành trí tuệ nhân tạo tìm ra là giải pháp đơn giản nhất nhưng lại hiệu quả bất ngờ: Đếm số lần xuất hiện của các từ. Tạo thành một vector.

Vector đơn sơ đó đã gánh vác nhiều hệ thống NLP trong sản xuất hơn bất kỳ mô hình embedding nào: các bộ lọc thư rác, phân loại chủ đề, phát hiện bất thường trong nhật ký log, xếp hạng tìm kiếm (trước khi có BM25), làn sóng phân tích cảm xúc đầu tiên, và suốt cả thập kỷ trên các bảng đo chuẩn học thuật. Đến năm 2026, các kỹ sư giàu kinh nghiệm vẫn chọn nó làm giải pháp đầu tiên cho các tác vụ phân loại hẹp. Nó chạy cực nhanh, có khả năng giải thích tường minh, và thường cho kết quả không hề thua kém một mô hình embedding 400 triệu tham số trên những tác vụ mà sự hiện diện của từ ngữ là yếu tố mang tính quyết định.

Bài học này xây dựng mô hình Túi từ (Bag of Words - BoW), sau đó là TF-IDF từ con số không. Tiếp theo là cách scikit-learn thực hiện tương tự chỉ trong ba dòng lệnh, và cuối cùng chỉ ra chính xác trạng thái lỗi khiến bạn buộc phải chuyển sang dùng vector nhúng.

## The Concept

**Túi từ (Bag of Words - BoW)** vứt bỏ hoàn toàn thứ tự từ ngữ. Đối với mỗi tài liệu, ta đếm xem mỗi từ trong từ vựng xuất hiện bao nhiêu lần. Độ dài của vector chính là kích thước của tập từ vựng. Vị trí thứ `i` thể hiện số lần xuất hiện của từ thứ `i`.

**TF-IDF** điều chỉnh lại trọng số cho BoW. Một từ xuất hiện trong mọi tài liệu thì không mang nhiều thông tin phân loại, vì vậy hãy giảm trọng số của nó xuống. Ngược lại, một từ hiếm gặp trên toàn bộ kho ngữ liệu nhưng lại xuất hiện dày đặc trong một tài liệu cụ thể chính là một tín hiệu đặc trưng mạnh, vì vậy hãy tăng trọng số của nó lên.

```math
\begin{aligned}
\text{TF-IDF}(w, d) &= \text{TF}(w, d) \cdot \text{IDF}(w) \\
&= \frac{\mathop{\mathrm{count}}(w \text{ in } d)}{|d|} \cdot \log\frac{N}{\mathop{\mathrm{df}}(w)}
\end{aligned}
```

Trong đó `TF` là tần suất từ trong tài liệu, `df` là tần suất tài liệu (có bao nhiêu tài liệu chứa từ đó), `N` là tổng số tài liệu. Hàm `log` giữ cho trọng số không bị bùng nổ đối với các từ quá phổ biến.

Thuộc tính then chốt: Cả hai phương pháp đều tạo ra các **vector thưa (sparse vectors)** với các trục toạ độ hoàn toàn diễn giải được. Bạn có thể nhìn thẳng vào các trọng số của một bộ phân loại tuyến tính đã huấn luyện để đọc vanh vách những từ nào đang đẩy tài liệu về từng nhãn lớp tương ứng. Bạn không thể làm được điều này với một vector nhúng BERT 768 chiều vô danh.

```figure
bow-tfidf
```

## Build It

### Bước 1: Xây dựng tập từ vựng (Vocabulary)

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

Đầu vào: danh sách các tài liệu đã tách từ (bất kỳ bộ tokenizer cấp từ nào; file `code/main.py` của bài học dùng biến thể chữ thường đơn giản). Đầu ra: từ điển `{từ: chỉ_số}`. Thứ tự chèn ổn định đồng nghĩa với việc từ có chỉ số 0 là từ đầu tiên bắt gặp trong tài liệu đầu tiên. Quy ước này có thể linh hoạt; scikit-learn thì sắp xếp theo bảng chữ cái.

### Bước 2: Túi từ (Bag of Words)

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

Mỗi hàng đại diện cho một tài liệu. Mỗi cột đại diện cho một chỉ số trong từ vựng. Phần tử `[i][j]` trả lời câu hỏi: "từ `j` xuất hiện bao nhiêu lần trong tài liệu `i`?" Tài liệu 1 có từ `cat` 2 lần. Tài liệu 0 có từ `ran` 0 lần vì nó không xuất hiện.

### Bước 3: Tần suất từ (TF) và Tần suất tài liệu nghịch đảo (IDF)

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

Hai kỹ thuật làm mượt (smoothing) đáng chú ý: Phép chia `(n+1)/(d+1)` giúp tránh chia cho 0 khi tính `log`. Phép cộng `+1` ở cuối đảm bảo rằng một từ xuất hiện trong mọi tài liệu vẫn có giá trị IDF bằng 1 (chứ không phải 0), hoàn toàn tương thích với mặc định của scikit-learn. Một số tài liệu khác dùng công thức gốc `log(N/df)`. Cả hai đều hoạt động tốt; phiên bản làm mượt thân thiện hơn khi xử lý thực tế.

### Bước 4: TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

Ba tài liệu, năm từ vựng (`the`, `cat`, `sat`, `dog`, `ran`). Từ `the` xuất hiện trong cả ba tài liệu nên có giá trị IDF thấp. Từ `dog` chỉ xuất hiện trong một tài liệu nên có IDF rất cao. Với một tập từ vựng thực tế gồm hàng chục nghìn từ, các vector này cực kỳ thưa (phần lớn phần tử bằng 0). Ngay cả trong ví dụ nhỏ này, những từ mang tính phân loại cao đã nổi bật lên rõ rệt.

### Bước 5: Chuẩn hoá L2 theo hàng (L2-normalize)

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

Nếu không chuẩn hoá, một tài liệu dài sẽ tạo ra một vector có độ lớn vượt trội và lấn át hoàn toàn điểm số tương đồng. Chuẩn hoá L2 đưa mọi vector tài liệu về cùng mặt cầu đơn vị. Khi đó, độ tương đồng cosine giữa các hàng đơn giản chỉ là phép nhân vô hướng (dot product).

## Use It

scikit-learn cung cấp bản triển khai công nghiệp chuẩn mực:

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer` tự động thực hiện tách từ, lập từ vựng và tạo ma trận BoW trong một lệnh duy nhất. `TfidfVectorizer` bổ sung thêm việc đánh trọng số IDF và chuẩn hoá L2. Cả hai đều trả về ma trận thưa (sparse matrix). Với 100k tài liệu, ma trận dày đặc (dense) sẽ làm tràn bộ nhớ RAM; hãy luôn duy trì định dạng thưa cho đến khi mô hình phân loại yêu cầu ma trận đặc.

Các tham số cấu hình cốt lõi:

| Tham số | Tác dụng |
|---------|----------|
| `ngram_range=(1, 2)` | Bao gồm cả cụm 2 từ (bigrams). Thường giúp tăng độ chính xác phân loại rõ rệt. |
| `min_df=2` | Loại bỏ các từ xuất hiện trong ít hơn 2 tài liệu. Giúp cắt tỉa từ vựng trên dữ liệu nhiều nhiễu. |
| `max_df=0.95` | Loại bỏ các từ xuất hiện trong hơn 95% số tài liệu. Hoạt động như một bộ lọc stopwords tự động mà không cần danh sách cứng. |
| `stop_words="english"` | Danh sách từ dừng dựng sẵn của scikit-learn. Cần lưu ý theo tác vụ — phân tích cảm xúc *không* nên xoá các từ phủ định. |
| `sublinear_tf=True` | Áp dụng `1 + log(tf)` thay vì `tf` thô. Rất hữu ích khi một từ lặp lại quá nhiều lần trong một tài liệu dài. |

### Khi nào TF-IDF vẫn chiếm ưu thế (tính đến năm 2026)

- Phát hiện thư rác, gán nhãn chủ đề, phát hiện bất thường trong nhật ký log. Sự xuất hiện của từ khoá là yếu tố cốt lõi; sắc thái ngữ nghĩa tinh tế không ảnh hưởng nhiều.
- Tập dữ liệu ít mẫu có nhãn (vài trăm ví dụ gán nhãn). TF-IDF kết hợp Hồi quy Logistic không tốn chi phí tiền huấn luyện.
- Bất kỳ hệ thống nào đòi hỏi độ trễ siêu thấp. TF-IDF kết hợp mô hình tuyến tính trả lời trong vài micro giây. Trong khi đưa văn bản qua mô hình Transformer mất từ 10-100 mili giây.
- Các hệ thống yêu cầu tính giải trình cao (explainability). Soi xét trực tiếp hệ số của bộ phân loại: các từ có trọng số dương cao nhất chính là nguyên nhân dẫn đến quyết định phân loại.

### Khi nào TF-IDF thất bại?

**Hạn chế về mặt ngữ nghĩa (Semantic blindness):** Hãy xem xét hai tài liệu sau:

- "The movie was not good at all." (Bộ phim hoàn toàn không hay)
- "The movie was excellent." (Bộ phim xuất sắc)

Một câu mang cảm xúc tiêu cực, một câu mang cảm xúc tích cực. Tuy nhiên, sự trùng lặp TF-IDF của chúng chỉ là `{the, movie, was}`. Một bộ phân loại túi từ phải học thuộc lòng rằng từ `not` đứng cạnh từ `good` sẽ đảo ngược nhãn. Nó có thể học được điều này nếu có lượng dữ liệu đủ lớn, nhưng không bao giờ mượt mà và tự nhiên như một mô hình hiểu được cú pháp.

Điểm hạn chế thứ hai: **Các từ nằm ngoài tập từ vựng (Out-of-vocabulary - OOV)** tại thời điểm suy luận. Một mô hình BoW được huấn luyện trên các bài đánh giá phim IMDb sẽ hoàn toàn bối rối trước từ `Zoomer-approved` nếu token đó chưa từng xuất hiện khi huấn luyện. Các vector nhúng cấp độ từ phụ (subword embeddings - Bài 04) xử lý tốt vấn đề này; còn TF-IDF thì bất lực.

### Giải pháp lai: Vector nhúng đánh trọng số theo TF-IDF

Lựa chọn thực dụng hàng đầu năm 2026 cho bài toán phân loại dữ liệu quy mô trung bình: sử dụng trọng số TF-IDF như một cơ chế chú ý (attention) đặt lên các vector nhúng từ.

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

Bạn vừa nhận được năng lực ngữ nghĩa từ các vector nhúng, vừa tận dụng được sự nhấn mạnh vào các từ hiếm từ TF-IDF. Bộ phân loại sẽ học trên vector gộp cuối cùng này. Phương pháp này thường đánh bại cả TF-IDF thuần lẫn vector nhúng trung bình đơn giản trên các tác vụ phân loại cảm xúc, chủ đề và ý định người dùng dưới 50.000 mẫu gán nhãn.

## Ship It

Lưu tại `outputs/prompt-vectorization-picker.md`:

```markdown
---
name: vectorization-picker
description: Given a text-classification task, recommend BoW, TF-IDF, embeddings, or a hybrid.
phase: 5
lesson: 02
---

You recommend a text-vectorization strategy. Given a task description, output:

1. Representation (BoW, TF-IDF, transformer embeddings, or a hybrid). Explain why in one sentence.
2. Specific vectorizer configuration. Name the library. Quote the arguments (`ngram_range`, `min_df`, `max_df`, `sublinear_tf`, `stop_words`).
3. One failure mode to test before shipping.

Refuse to recommend embeddings when the user has under 500 labeled examples unless they show evidence of semantic failure in a TF-IDF baseline. Refuse to remove stopwords for sentiment analysis (negations carry signal). Flag class imbalance as needing more than a vectorizer change.

Example input: "Classifying 30k customer support tickets into 12 categories. Most tickets are 2-3 sentences. English only. Need explainability for audit logs."

Example output:

- Representation: TF-IDF. 30k examples is not small; explainability requirement rules out dense embeddings.
- Config: `TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`. Keep stopwords because category keywords sometimes are stopwords ("not working" vs "working").
- Failure to test: verify `min_df=3` does not drop rare category keywords. Run `get_feature_names_out` filtered by class and eyeball.
```

## Exercises

1. **(Dễ)** Triển khai hàm `cosine_similarity(doc_vec_a, doc_vec_b)` trên kết quả TF-IDF đã chuẩn hoá L2. Kiểm tra rằng hai tài liệu giống hệt nhau có điểm số là 1.0 và hai tài liệu có từ vựng hoàn toàn tách rời nhau có điểm số là 0.0.
2. **(Trung bình)** Bổ sung tính năng hỗ trợ `n-gram` vào hàm `bag_of_words`. Tham số `n` tạo ra tần suất đếm trên các `n`-gram. Kiểm tra rằng `n=2` trên `["the", "cat", "sat"]` tạo ra các đếm bigram chính xác cho `["the cat", "cat sat"]`.
3. **(Khó)** Xây dựng giải pháp lai TF-IDF-weighted-embedding ở trên sử dụng các vector GloVe 100d. So sánh độ chính xác phân loại với TF-IDF thuần và vector nhúng tính trung bình đơn giản trên tập dữ liệu 20 Newsgroups. Báo cáo trường hợp nào chiếm ưu thế ở từng phân nhóm dữ liệu.

## Key Terms

| Thuật ngữ | Cách gọi thông thường | Bản chất kỹ thuật thực sự |
|-----------|----------------------|---------------------------|
| BoW | Vector tần suất từ | Tần suất xuất hiện của các từ trong từ vựng trong một tài liệu. Bỏ qua hoàn toàn thứ tự từ |
| TF | Tần suất từ | Số lần xuất hiện của một từ trong tài liệu, có thể chuẩn hoá theo độ dài tài liệu |
| DF | Tần suất tài liệu | Số lượng tài liệu chứa từ đó ít nhất một lần |
| IDF | Tần suất tài liệu nghịch đảo | `log(N / df)` có làm mượt. Giảm trọng số của các từ xuất hiện tràn lan ở khắp mọi nơi |
| Sparse vector | Vector thưa | Tập từ vựng thường có từ 10k-100k từ; phần lớn các từ này vắng mặt trong một tài liệu cụ thể |
| Cosine similarity | Góc giữa hai vector | Tích vô hướng của các vector đã chuẩn hoá L2. Bằng 1 là giống hệt, bằng 0 là trực giao |

## Further Reading

- [scikit-learn — trích xuất đặc trưng từ văn bản](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — Tài liệu tham chiếu API chuẩn mực kèm ghi chú chi tiết cho từng tham số
- [Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval](https://www.sciencedirect.com/science/article/pii/0306457388900210) — Bài báo kinh điển đặt nền móng biến TF-IDF thành tiêu chuẩn suốt cả thập kỷ
- ["Why TF-IDF Still Beats Embeddings" — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) — Góc nhìn thực tế năm 2026 giải thích tại sao phương pháp cổ điển vẫn thắng thế

