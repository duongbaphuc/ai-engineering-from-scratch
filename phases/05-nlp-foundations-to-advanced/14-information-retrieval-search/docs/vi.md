# Truy xuất Thông tin và Tìm kiếm (Information Retrieval and Search)

> BM25 chuẩn xác nhưng cứng nhắc. Vector dày đặc (Dense) bao quát rộng nhưng bỏ sót từ khóa. Truy xuất lai (Hybrid) là tiêu chuẩn năm 2026. Mọi thứ còn lại chỉ là tinh chỉnh.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 04 (GloVe, FastText, Subword)
**Time:** ~75 phút

## Vấn đề đặt ra (The Problem)

Người dùng gõ: "chuyện gì sẽ xảy ra nếu ai đó nói dối để chiếm đoạt tiền" và kỳ vọng tìm thấy đúng điều luật quy định về hành vi đó: "Điều 420 Bộ luật Hình sự." Một công cụ tìm kiếm từ khóa thông thường sẽ bỏ sót hoàn toàn (không có từ vựng chung nào giữa câu hỏi và điều luật). Một công cụ tìm kiếm ngữ nghĩa sẽ bỏ sót nếu các vector nhúng không được huấn luyện trên ngữ liệu pháp lý. Một hệ thống tìm kiếm thực thụ phải xử lý trơn tru cả hai tình huống.

Truy xuất thông tin (Information Retrieval - IR) là đường ống nền tảng bên dưới mọi hệ thống RAG, mọi thanh tìm kiếm, mọi tính năng tra cứu mờ (fuzzy lookup) của các trang tài liệu kỹ thuật. Kiến trúc sản xuất hoạt động hiệu quả nhất năm 2026 không phải là một phương pháp đơn độc; nó là một chuỗi các phương pháp bổ trợ lẫn nhau, trong đó mỗi mắt xích khắc phục điểm yếu của mắt xích phía trước.

Bài học này xây dựng từng thành phần và chỉ rõ mỗi mắt xích giải quyết những trường hợp thất bại nào.

## Khái niệm (The Concept)

![Truy xuất lai: BM25 + dense + RRF + xếp hạng lại bằng cross-encoder](../assets/retrieval.svg)

Bốn tầng kiến trúc. Hãy lựa chọn các tầng bạn thực sự cần:

1. **Truy xuất thưa thớt (Sparse retrieval - BM25).** Nhanh, cực kỳ chuẩn xác trên các phép khớp từ khóa chính xác, nhưng kém về mặt ngữ nghĩa. Chạy trên chỉ mục đảo (inverted index). Tốc độ dưới 10ms cho mỗi truy vấn trên hàng triệu tài liệu. Giúp bạn tìm đúng các trích dẫn điều luật, mã sản phẩm, thông báo lỗi hệ thống, tên riêng thực thể.
2. **Truy xuất dày đặc (Dense retrieval).** Mã hóa câu truy vấn và tài liệu thành các vector. Tìm kiếm láng giềng gần nhất (nearest neighbor search). Bắt trọn các cách diễn đạt tương đương và sự tương đồng về ngữ nghĩa. Nhưng dễ bỏ sót các từ khóa chính xác chỉ khác nhau một ký tự. Tốc độ từ 50-200ms mỗi truy vấn với FAISS hoặc cơ sở dữ liệu vector.
3. **Hợp nhất (Fusion).** Kết hợp các danh sách kết quả đã xếp hạng từ cả hai phương pháp thưa và dày. Thuật toán Reciprocal Rank Fusion (RRF) là lựa chọn mặc định lý tưởng vì nó bỏ qua điểm số thô (vốn nằm ở các thang đo khác nhau) và chỉ thuần túy dựa vào vị trí thứ hạng. Hợp nhất có trọng số (weighted fusion) là phương án thay thế khi bạn biết chắc chắn một tín hiệu luôn chiếm ưu thế trong miền nghiệp vụ của mình.
4. **Xếp hạng lại bằng Cross-Encoder (Cross-encoder rerank).** Lấy top-30 kết quả từ bước hợp nhất. Chạy một mô hình cross-encoder (đưa cả câu truy vấn và tài liệu vào cùng nhau để chấm điểm từng cặp). Giữ lại top-5 tốt nhất. Cross-encoder tính toán chậm hơn nhiều so với bi-encoder nhưng độ chính xác vượt trội hơn hẳn. Chi phí tính toán được tối ưu hóa vì ta chỉ chạy trên 30 tài liệu ứng viên tốt nhất.

Truy xuất ba hướng (BM25 + dense + learned-sparse như SPLADE) đạt hiệu năng cao hơn truy xuất hai hướng trên các bài đo benchmark năm 2026, nhưng đòi hỏi hạ tầng phức tạp hơn cho chỉ mục learned-sparse. Đối với hầu hết các đội ngũ kỹ thuật, truy xuất hai hướng kết hợp xếp hạng lại bằng cross-encoder là điểm cân bằng hoàn hảo nhất.

```figure
gx-hybrid-retrieval
```

## Xây dựng thực tế (Build It)

### Bước 1: BM25 từ đầu

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

Hai siêu tham số cốt lõi: `k1=1.5` kiểm soát mức độ bão hòa tần suất xuất hiện của từ (term-frequency saturation); giá trị càng cao càng tăng trọng số cho việc lặp từ. `b=0.75` kiểm soát mức độ chuẩn hóa theo độ dài tài liệu; 0 là bỏ qua độ dài tài liệu, 1 là chuẩn hóa hoàn toàn. Các giá trị mặc định này là khuyến nghị của Robertson trong bài báo gốc và rất hiếm khi cần tinh chỉnh.

### Bước 2: Truy xuất dày đặc với Bi-Encoder

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

Chuẩn hóa L2 cho các vector nhúng để tích vô hướng tương đương với độ tương đồng cosine. Mô hình `all-MiniLM-L6-v2` có 384 chiều, tốc độ nhanh, và đủ mạnh mẽ cho hầu hết các bài toán truy xuất tiếng Anh. Với đa ngôn ngữ, dùng `paraphrase-multilingual-MiniLM-L12-v2`. Để có độ chính xác cao nhất, dùng `bge-large-en-v1.5` hoặc `e5-large-v2`.

### Bước 3: Reciprocal Rank Fusion (RRF)

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

Hằng số `k=60` được đề xuất từ bài báo gốc về RRF. Giá trị `k` lớn hơn sẽ làm phẳng đóng góp của các khác biệt về thứ hạng; `k` nhỏ hơn sẽ khiến các thứ hạng đầu tiên chi phối áp đảo. Con số 60 là chuẩn mặc định đã được công bố rộng rãi và hiếm khi cần thay đổi.

### Bước 4: Tìm kiếm lai (Hybrid Search) kết hợp Xếp hạng lại (Rerank)

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

Ba giai đoạn phối hợp nhịp nhàng: BM25 tìm kiếm các so khớp từ vựng chính xác. Dense tìm kiếm các tương đồng ngữ nghĩa sâu. RRF hợp nhất hai bảng xếp hạng mà không cần phải chuẩn hóa điểm số phức tạp. Cross-encoder chấm điểm lại top-30 ứng viên bằng cách xử lý đồng thời cặp truy vấn - tài liệu, nắm bắt được các sắc thái liên quan tinh tế mà bi-encoder đã bỏ lỡ. Cuối cùng giữ lại top-5 xuất sắc nhất.

### Bước 5: Đánh giá chất lượng truy xuất

| Chỉ số | Ý nghĩa |
|--------|---------|
| Recall@k | Trong số các truy vấn có tài liệu liên quan tồn tại, tần suất tài liệu đó nằm trong top-k là bao nhiêu? |
| MRR (Mean Reciprocal Rank) | Giá trị trung bình của nghịch đảo thứ hạng (1/rank) của tài liệu liên quan đầu tiên. |
| nDCG@k | Đánh giá có tính đến các cấp độ liên quan phân tầng, không chỉ là nhị phân (liên quan / không liên quan). |

Riêng đối với RAG, **Recall@k** của bộ truy xuất là chỉ số quan trọng sống còn nhất. Bộ đọc của bạn không thể đưa ra câu trả lời đúng nếu đoạn văn bản cần thiết không nằm trong tập hợp được truy xuất về.

Mẹo gỡ lỗi: Đối với các truy vấn thất bại, hãy so sánh sự khác biệt giữa bảng xếp hạng của sparse và dense. Nếu một bên tìm ra đúng tài liệu còn bên kia thì không, bạn đang gặp vấn đề không khớp từ vựng (khắc phục: bổ sung nửa phần còn thiếu) hoặc nhập nhằng ngữ nghĩa (khắc phục: dùng vector nhúng tốt hơn hoặc thêm bộ reranker).

## Sử dụng trong thực tế (Use It)

Ngăn xếp công nghệ năm 2026:

| Quy mô | Ngăn xếp đề xuất |
|--------|------------------|
| 1k - 100k tài liệu | BM25 trong bộ nhớ RAM + vector nhúng `all-MiniLM-L6-v2` + RRF. Không cần cơ sở dữ liệu riêng. |
| 100k - 10M tài liệu | FAISS hoặc pgvector cho dense + Elasticsearch / OpenSearch cho BM25. Chạy song song. |
| Trên 10M tài liệu | Qdrant / Weaviate / Vespa / Milvus có hỗ trợ hybrid. Cross-encoder rerank trên top-30. |
| Đỉnh cao chất lượng | Truy xuất 3 hướng (BM25 + dense + SPLADE) + ColBERT late-interaction reranking. |

Bất kể bạn chọn ngăn xếp nào, hãy dành ngân sách cho việc đánh giá: Đo lường chuẩn hóa recall của bộ truy xuất trước khi đo độ chính xác toàn diện của RAG end-to-end. Bộ đọc không thể bù đắp cho những gì mà bộ truy xuất đã bỏ sót.

### Những bài học xương máu từ RAG thực tế năm 2026

- **80% thất bại của RAG bắt nguồn từ khâu nạp dữ liệu (ingestion) và phân đoạn (chunking), không phải do mô hình.** Các đội ngũ kỹ sư thường lãng phí hàng tuần để đổi LLM và tinh chỉnh prompt trong khi bộ truy xuất cứ ba lượt lại âm thầm trả về sai ngữ cảnh một lần. Hãy tập trung sửa khâu chunking trước tiên.
- **Chiến lược phân đoạn quan trọng hơn kích thước đoạn văn.** Cắt xén kích thước cố định (fixed-size splits) sẽ phá hỏng các bảng biểu, đoạn mã nguồn và tiêu đề lồng nhau. Phân đoạn theo ranh giới câu (sentence-aware) là chuẩn mặc định; phân đoạn theo ngữ nghĩa (semantic) hoặc dựa trên LLM mang lại giá trị vượt trội cho tài liệu kỹ thuật và sách hướng dẫn sản phẩm.
- **Mẫu tài liệu cha - con (Parent-doc pattern).** Truy xuất các đoạn nhỏ "con" (child chunks) để đạt độ chính xác từ khóa tối đa. Khi có nhiều đoạn con cùng thuộc một phần tài liệu xuất hiện, hãy thay thế bằng khối tài liệu "cha" (parent block) lớn hơn để giữ trọn ngữ cảnh. Kỹ thuật này luôn nâng cao chất lượng câu trả lời mà không cần huấn luyện lại bất kỳ mô hình nào.
- **k_rerank=3 thường là tối ưu nhất.** Mỗi đoạn bổ sung sau ngưỡng đó chỉ làm tăng chi phí token và độ trễ sinh từ mà không hề cải thiện chất lượng câu trả lời. Nếu k=8 vẫn cho kết quả tốt hơn k=3 trong hệ thống của bạn, điều đó chứng tỏ bộ reranker của bạn đang hoạt động kém hiệu quả.
- **HyDE / Mở rộng truy vấn (Query expansion).** Sinh ra một câu trả lời giả định từ câu hỏi, chuyển đổi câu trả lời đó thành vector, rồi mới truy xuất. Điều này thu hẹp khoảng cách diễn đạt giữa câu hỏi ngắn và tài liệu dài. Mang lại mức tăng độ chính xác miễn phí mà không cần huấn luyện thêm.
- **Giữ ngân sách ngữ cảnh dưới 8K token.** Nếu liên tục chạm ngưỡng giới hạn đó, nghĩa là ngưỡng lọc của bộ reranker đang quá lỏng lẻo.
- **Quản lý phiên bản mọi thứ (Version everything).** Prompt, quy tắc phân đoạn, mô hình nhúng, bộ reranker. Bất kỳ sự trôi dạt nào cũng âm thầm làm gãy đổ chất lượng câu trả lời. Thiết lập cổng kiểm thử CI trên độ trung thực, độ chính xác ngữ cảnh và tỷ lệ câu hỏi không trả lời được để chặn lỗi trước khi người dùng kịp phát hiện.
- **Truy xuất ba hướng (BM25 + dense + SPLADE) vượt trội hơn hai hướng** trên các bài đo benchmark năm 2026, đặc biệt với các truy vấn kết hợp lẫn lộn tên riêng thực thể với ngữ nghĩa trừu tượng. Hãy triển khai khi hạ tầng của bạn hỗ trợ chỉ mục SPLADE.

Theo các số liệu đo đạc thực tế năm 2026, việc thiết kế bộ truy xuất chuẩn mực giúp giảm ảo giác từ 70% đến 90%. Phần lớn bước nhảy vọt về hiệu năng RAG đến từ việc cải thiện bộ truy xuất, chứ không phải từ việc fine-tune mô hình ngôn ngữ.

## Đóng gói sản phẩm (Ship It)

Lưu thành `outputs/skill-retrieval-picker.md`:

```markdown
---
name: retrieval-picker
description: Lựa chọn ngăn xếp truy xuất phù hợp cho một kho ngữ liệu và mẫu hình truy vấn cụ thể.
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

Dựa trên các yêu cầu (kích thước kho dữ liệu, mẫu hình truy vấn, ngân sách độ trễ, tiêu chuẩn chất lượng, ràng buộc hạ tầng), bạn đưa ra:

1. Ngăn xếp công nghệ: Chỉ BM25, chỉ dense, hybrid (BM25 + dense + RRF), hybrid + cross-encoder rerank, hoặc 3 hướng (BM25 + dense + learned-sparse).
2. Bộ mã hóa Dense (Dense encoder): Nêu tên mô hình cụ thể. Khớp với ngôn ngữ, miền nghiệp vụ và độ dài ngữ cảnh.
3. Bộ xếp hạng lại (Reranker): Nêu tên mô hình cross-encoder cụ thể nếu sử dụng. Cảnh báo rằng rerank sẽ tăng thêm 30-100ms độ trễ trên top-30 ứng viên.
4. Kế hoạch đánh giá: Recall@10 là chỉ số chính cho bộ truy xuất. MRR cho các truy vấn có nhiều đáp án. Thiết lập baseline trước, đo lường các cải tiến tăng dần dựa trên baseline đó.

Từ chối đề xuất chỉ dùng dense-only cho các kho dữ liệu có chứa thực thể tên riêng, mã lỗi kỹ thuật hoặc mã SKU sản phẩm trừ khi người dùng có bằng chứng chứng minh dense xử lý tốt các so khớp chính xác này. Từ chối bỏ qua bước rerank đối với các bài toán truy xuất quan trọng sống còn (pháp lý, y tế) nơi top-5 quyết định trực tiếp câu trả lời cho người dùng.
```

## Bài tập (Exercises)

1. **Dễ.** Cài đặt `hybrid_search` ở trên trên một kho ngữ liệu gồm 500 tài liệu. Kiểm thử 20 câu truy vấn. So sánh Recall@5 giữa ba phương án: chỉ dùng BM25, chỉ dùng dense, và hybrid kết hợp.
2. **Trung bình.** Bổ sung phép tính chỉ số MRR. Với mỗi câu truy vấn kiểm thử có một tài liệu đúng đã biết, tìm thứ hạng của tài liệu đúng trong các danh sách xếp hạng của BM25, dense và hybrid. Báo cáo điểm MRR cho từng phương pháp.
3. **Khó.** Tinh chỉnh một bộ mã hóa dense trên miền nghiệp vụ của bạn bằng `MultipleNegativesRankingLoss` (Sentence Transformers). Xây dựng tập dữ liệu huấn luyện từ 500 cặp truy vấn - tài liệu. So sánh độ nhạy Recall trước và sau khi tinh chỉnh.

## Thuật ngữ cốt lõi (Key Terms)

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|-----------|-------------------|-----------------|
| BM25 | Tìm kiếm từ khóa | Thuật toán Okapi BM25. Chấm điểm tài liệu dựa trên tần suất từ (TF), tần suất nghịch đảo tài liệu (IDF) và độ dài văn bản. |
| Truy xuất dày đặc (Dense retrieval) | Tìm kiếm vector | Biến đổi câu truy vấn và tài liệu thành vector, tìm các láng giềng gần nhất. |
| Bi-encoder | Mô hình embedding | Mã hóa câu truy vấn và tài liệu một cách độc lập. Cho tốc độ tìm kiếm cực nhanh tại thời điểm truy vấn. |
| Cross-encoder | Mô hình reranker | Mã hóa đồng thời cả truy vấn và tài liệu cùng lúc. Tốc độ chậm nhưng độ chính xác rất cao. |
| RRF | Hợp nhất thứ hạng | Kết hợp hai bảng xếp hạng bằng cách lấy tổng các phân số `1/(k + rank)`. |
| Recall@k | Chỉ số truy xuất | Tỷ lệ phần trăm các câu truy vấn mà tài liệu liên quan xuất hiện trong top-k kết quả. |

## Đọc thêm (Further Reading)

- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — tài liệu khảo cứu kinh điển toàn diện nhất về BM25.
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR, bài báo kinh điển về bi-encoder.
- [Formal et al. (2021). SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720) — bộ truy xuất learned-sparse thu hẹp khoảng cách với dense retrieval.
- [Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — bài báo gốc đặt nền móng cho RRF.
- [Khattab and Zaharia (2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832) — cơ chế truy xuất tương tác muộn (late-interaction).

