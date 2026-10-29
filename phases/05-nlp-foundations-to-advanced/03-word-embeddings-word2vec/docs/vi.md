# Vector Nhúng Từ — Word2Vec Từ Con Số Không (Word Embeddings — Word2Vec from Scratch)

> Bạn sẽ hiểu bản chất của một từ qua những từ thường đi kèm với nó. Huấn luyện một mạng nơ-ron nông trên ý tưởng đó, và các tính chất hình học không gian kỳ diệu sẽ tự khắc xuất hiện.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 3 · 03 (Backpropagation from Scratch)
**Time:** ~75 phút

## The Problem

Mô hình TF-IDF chỉ biết `dog` (chó) và `puppy` (chó con) là hai từ khác nhau. Nó hoàn toàn không hiểu rằng chúng mang ngữ nghĩa gần như tương đồng. Một bộ phân loại được huấn luyện trên từ `dog` sẽ không thể khái quát hoá để hiểu một bài đánh giá chứa từ `puppy`. Bạn có thể chữa cháy bằng cách liệt kê danh sách từ đồng nghĩa thủ công, nhưng cách này sẽ phá sản hoàn toàn trước các từ hiếm, thuật ngữ chuyên ngành và bất kỳ ngôn ngữ nào mà bạn chưa kịp dự đoán trước.

Bạn cần một biểu diễn không gian nơi `dog` và `puppy` nằm cạnh nhau. Nơi phép tính vector `king - man + woman` sẽ đáp xuống vị trí cận kề `queen`. Nơi một mô hình học trên `dog` sẽ tự động chuyển giao một phần tri thức sang `puppy` hoàn toàn miễn phí.

Word2Vec đã mang lại cho chúng ta không gian đó. Một mạng nơ-ron hai tầng đơn giản, được huấn luyện trên các tập dữ liệu nghìn tỷ token, xuất bản năm 2013. Kiến trúc của nó đơn giản đến mức đáng kinh ngạc, nhưng kết quả tạo ra đã định hình lại toàn bộ ngành xử lý ngôn ngữ tự nhiên trong suốt cả một thập kỷ.

## The Concept

**Giả thuyết phân phối (Distributional hypothesis)** (Firth, 1957): "Bạn sẽ hiểu một từ qua những từ đi cùng nó." Nếu hai từ thường xuyên xuất hiện trong các ngữ cảnh tương tự nhau, rất có khả năng chúng mang ý nghĩa tương tự nhau.

Word2Vec có hai biến thể chính khai thác triệt để ý tưởng này:

- **Skip-gram:** Cho trước từ trung tâm, dự đoán các từ ngữ cảnh xung quanh. Ví dụ: `cat -> (the, sat, on)` với kích thước cửa sổ là 2.
- **CBOW (Continuous Bag of Words):** Cho trước các từ ngữ cảnh xung quanh, dự đoán từ trung tâm. Ví dụ: `(the, sat, on) -> cat`.

Skip-gram huấn luyện chậm hơn nhưng xử lý các từ hiếm tốt hơn nhiều, vì vậy nó đã trở thành lựa chọn mặc định phổ biến nhất.

Mạng nơ-ron chỉ gồm một tầng ẩn duy nhất không có hàm kích hoạt phi tuyến tính. Đầu vào là vector one-hot trên toàn bộ từ vựng. Đầu ra là hàm softmax trên toàn bộ từ vựng. Sau khi huấn luyện xong, bạn vứt bỏ tầng đầu ra. Các trọng số của tầng ẩn chính là các vector nhúng (embeddings) bạn cần.

```text
one-hot(từ_trung_tâm) ── W ──▶ tầng_ẩn (d-chiều) ── W' ──▶ softmax(từ_vựng)
                                  ^
                                  đây chính là vector nhúng
```

Bí quyết then chốt: Việc tính hàm softmax trên một tập từ vựng 100k từ cực kỳ tốn kém tài nguyên tính toán. Word2Vec sử dụng kỹ thuật **lấy mẫu âm (negative sampling)** để biến bài toán dự đoán đa lớp thành bài toán phân loại nhị phân: Dự đoán xem "từ ngữ cảnh này có thực sự xuất hiện gần từ trung tâm này hay không, có hoặc không". Lấy mẫu ngẫu nhiên một vài từ âm tính (không đồng xuất hiện) cho mỗi cặp huấn luyện thay vì phải tính toán phân phối xác suất trên toàn bộ từ vựng.

```figure
word-vector-arithmetic
```

## Build It

### Bước 1: Tạo các cặp huấn luyện từ kho ngữ liệu

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

Mỗi cặp (từ trung tâm, từ ngữ cảnh) xuất hiện trong phạm vi cửa sổ được xem là một mẫu huấn luyện dương tính (positive example).

### Bước 2: Bảng trọng số nhúng (Embedding tables)

Gồm hai ma trận: `W` là bảng nhúng của từ trung tâm (ma trận bạn sẽ giữ lại làm vector nhúng cuối cùng). `W'` là bảng nhúng của từ ngữ cảnh (thường bị bỏ đi, hoặc đôi khi được tính trung bình với `W`).

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

Khởi tạo ngẫu nhiên với giá trị nhỏ. Trong thực tế, từ vựng 10k từ và số chiều 100 là phổ biến; ở đây với mục đích giảng dạy, từ vựng 50 từ và 16 chiều là đủ để quan sát các tính chất hình học.

### Bước 3: Hàm mục tiêu lấy mẫu âm (Negative Sampling Objective)

Đối với mỗi cặp dương tính `(từ_trung_tâm, từ_ngữ_cảnh)`, ta lấy mẫu ngẫu nhiên `k` từ trong từ vựng làm các mẫu âm tính. Huấn luyện mô hình sao cho tích vô hướng `W[từ_trung_tâm] · W'[từ_ngữ_cảnh]` đạt giá trị cao cho cặp dương tính và đạt giá trị thấp cho các cặp âm tính.

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

Công thức toán học cốt lõi: hàm mất mát logistic trên cặp dương tính (muốn hàm sigmoid tiến về 1) cộng với hàm mất mát logistic trên các cặp âm tính (muốn hàm sigmoid tiến về 0). Gradient được lan truyền cập nhật cho cả hai ma trận bảng nhúng.

### Bước 4: Huấn luyện trên kho ngữ liệu đồ chơi

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

Sau đủ số vòng lặp trên một kho ngữ liệu lớn, những từ chia sẻ chung ngữ cảnh xuất hiện sẽ có các vector nhúng trung tâm gần nhau trong không gian. Trên kho ngữ liệu nhỏ mô phỏng này, bạn có thể bắt đầu nhìn thấy xu hướng đó. Trên hàng tỷ token, hiện tượng này thể hiện cực kỳ rõ rệt.

### Bước 5: Phép loại suy (The Analogy Trick)

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

Trên các vector Google News 300d được huấn luyện sẵn:

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`. Kết quả này không phải vì mô hình hiểu khái niệm hoàng gia hay giới tính. Mà bởi vì vector hiệu `(king - man)` đã nắm bắt được hướng không gian tương ứng với khái niệm "hoàng gia", và việc cộng vector đó vào `woman` sẽ đưa ta đến vùng không gian của phụ nữ thuộc hoàng tộc.

## Use It

Tự viết Word2Vec từ con số không giúp bạn thấu hiểu bản chất giải thuật. Trong sản xuất, người ta sử dụng thư viện `gensim`:

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

Trong thực tế công việc, hiếm khi bạn phải tự huấn luyện Word2Vec từ đầu. Thay vào đó, bạn thường tải các vector đã được huấn luyện sẵn:

- **GloVe** — Phương pháp phân tích ma trận đồng xuất hiện của Stanford. Có sẵn các bản 50d, 100d, 200d, 300d với độ bao phủ từ vựng rộng (Bài 04 sẽ đi sâu vào GloVe).
- **fastText** — Bản mở rộng Word2Vec của Facebook nhúng cả các n-gram ký tự. Xử lý xuất sắc các từ chưa từng thấy trong từ vựng bằng cách ghép nối các từ phụ (Bài 04).
- **Word2Vec trên Google News** — 300d, 3 triệu từ vựng, phát hành năm 2013, vẫn được tải về hàng ngày.

### Khi nào Word2Vec vẫn phát huy tác dụng năm 2026?

- Truy vấn gọn nhẹ cho các lĩnh vực chuyên biệt: Huấn luyện trên các bài báo y khoa chỉ trong một giờ trên laptop, thu được các vector chuyên ngành mà mô hình tổng quát không thể có được.
- Kỹ nghệ đặc trưng theo kiểu loại suy (analogy-style feature engineering): `gender_vector = mean(man - woman)`. Trừ vector này khỏi các từ nghề nghiệp để tạo trục trung tính về giới tính trong nghiên cứu công bằng AI.
- Khả năng diễn giải trực quan: 100 chiều đủ nhỏ để vẽ đồ thị qua PCA hoặc t-SNE và quan sát trực tiếp các cụm ngữ nghĩa hình thành.
- Môi trường chạy trên thiết bị biên (on-device) không có GPU: Tra cứu Word2Vec chỉ đơn giản là đọc một hàng trong ma trận bộ nhớ.

### Điểm yếu của Word2Vec

**Bức tường đa nghĩa (The polysemy wall):** Từ `bank` chỉ có một vector duy nhất. Cả nghĩa `river bank` (bờ sông) lẫn `financial bank` (ngân hàng) đều dùng chung vector đó. Từ `table` (bảng số liệu vs. cái bàn) cũng chia sẻ chung một vector. Một bộ phân loại ở tầng hạ nguồn không thể nào phân biệt được ngữ nghĩa nếu chỉ nhìn vào vector tĩnh đó.

Các vector nhúng phụ thuộc ngữ cảnh (Contextual embeddings như ELMo, BERT và mọi mô hình Transformer sau này) đã giải quyết triệt để điều này bằng cách sinh ra một vector khác nhau cho mỗi lần xuất hiện của từ dựa trên ngữ cảnh xung quanh. Đó là bước nhảy vọt lịch sử từ Word2Vec lên BERT: từ tĩnh sang động (Phase 7 sẽ phân tích sâu về Transformer).

Điểm yếu thứ hai là bài toán từ ngoài từ vựng (OOV). Word2Vec hoàn toàn bó tay trước từ `Zoomer-approved` nếu nó không có trong tập huấn luyện. fastText đã khắc phục điều này bằng cách phân rã thành các từ phụ (Bài 04).

## Ship It

Lưu tại `outputs/skill-embedding-probe.md`:

```markdown
---
name: embedding-probe
description: Inspect a word2vec model. Run analogies, find neighbors, diagnose quality.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

You probe trained word embeddings to verify they are working. Given a `gensim.models.KeyedVectors` object and a vocabulary, you run:

1. Three canonical analogy tests. `king : man :: queen : woman`. `paris : france :: tokyo : japan`. `walking : walked :: swimming : ?`. Report the top-1 result and its cosine.
2. Five nearest-neighbor tests on domain-specific words the user supplies. Print top-5 neighbors with cosines.
3. One symmetry check. `similarity(a, b) == similarity(b, a)` to within float precision.
4. One degenerate check. If any embedding has a norm below 0.01 or above 100, the model has a training bug. Flag it.

Refuse to declare a model good on analogy accuracy alone. Analogy benchmarks are gameable and do not transfer to downstream tasks. Recommend intrinsic + downstream evaluation together.
```

## Exercises

1. **(Dễ)** Chạy vòng lặp huấn luyện trên một kho ngữ liệu nhỏ (20 câu về mèo và chó). Sau 200 epoch, kiểm tra xem `nearest(vocab, W, W[vocab["cat"]])` có trả về `dog` trong top 3 kết quả gần nhất hay không. Nếu không, hãy tăng số epoch hoặc làm giàu thêm câu văn.
2. **(Trung bình)** Bổ sung kỹ thuật giảm tỷ lệ lấy mẫu các từ quá phổ biến (subsampling). Các từ có tần suất trên `10^-5` sẽ bị loại bỏ khỏi các cặp huấn luyện với xác suất tỷ lệ thuận với tần suất của chúng. Đo lường mức độ cải thiện độ tương đồng của các từ hiếm.
3. **(Khó)** Huấn luyện một mô hình trên kho ngữ liệu 20 Newsgroups. Tính hai trục định kiến không gian: `he - she` và `doctor - nurse`. Chiếu các từ chỉ nghề nghiệp lên cả hai trục. Báo cáo các nghề nghiệp có khoảng cách thiên kiến lớn nhất. Đây chính là phương pháp mà các nhà nghiên cứu công bằng thuật toán thường dùng.

## Key Terms

| Thuật ngữ | Cách gọi thông thường | Bản chất kỹ thuật thực sự |
|-----------|----------------------|---------------------------|
| Word embedding | Biểu diễn từ bằng vector | Vector biểu diễn dày đặc, số chiều thấp (100-300 chiều) học được từ ngữ cảnh |
| Skip-gram | Kỹ thuật Word2Vec | Dự đoán các từ ngữ cảnh từ một từ trung tâm; chậm hơn CBOW nhưng tốt cho từ hiếm |
| Negative sampling | Lối tắt huấn luyện | Thay thế hàm softmax toàn bộ từ vựng bằng phân loại nhị phân trên `k` từ ngẫu nhiên |
| Static embedding | Vector nhúng tĩnh | Mỗi từ có duy nhất một vector cố định bất kể ngữ cảnh; thất bại trước từ đa nghĩa |
| Contextual embedding | Vector nhúng theo ngữ cảnh | Sinh vector khác nhau cho từng lần từ xuất hiện tuỳ thuộc vào các từ xung quanh |
| OOV | Từ ngoài từ vựng | Từ chưa từng xuất hiện khi huấn luyện; Word2Vec không thể sinh vector cho từ này |

## Further Reading

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) — Bài báo gốc về negative sampling; ngắn gọn và dễ hiểu
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) — Bản giải thích đạo hàm và toán học rõ ràng nhất về quá trình học tham số
- [Hướng dẫn gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html) — Các thiết lập huấn luyện thực tế đem lại hiệu năng cao

