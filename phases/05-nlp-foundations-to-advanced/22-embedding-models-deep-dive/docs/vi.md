# Mô hình nhúng — Nghiên cứu chuyên sâu 2026

> Word2Vec mang lại cho bạn một vector trên mỗi từ. Các mô hình nhúng (embedding models) hiện đại mang lại cho bạn một vector trên mỗi đoạn văn (passage), hỗ trợ xuyên ngôn ngữ (cross-lingual), với các biểu diễn thưa (sparse), dày đặc (dense) và đa vector (multi-vector), được định cỡ tối ưu cho chỉ mục của bạn. Chọn sai mô hình và hệ thống RAG của bạn sẽ truy xuất hoàn toàn nhầm dữ liệu.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Giai đoạn 5 · 03 (Word2Vec), Giai đoạn 5 · 14 (Truy xuất thông tin)
**Time:** ~60 phút

## Vấn đề

Hệ thống RAG của bạn truy xuất sai đoạn văn tới 40% thời gian thực thi. Thách thức cốt lõi hiếm khi nằm ở cơ sở dữ liệu vector hay prompt. Thủ phạm chính là mô hình nhúng (embedding model).

Việc lựa chọn một mô hình nhúng vào năm 2026 đồng nghĩa với việc đưa ra quyết định cân nhắc trên 5 trục chính:

1. **Dày đặc vs Thưa vs Đa vector (Dense vs sparse vs multi-vector).** Một vector cho mỗi đoạn văn, hay một vector cho từng token, hay một túi từ có trọng số thưa thớt (sparse weighted bag of words).
2. **Độ phủ ngôn ngữ (Language coverage).** Các mô hình đơn ngữ tiếng Anh vẫn chiếm ưu thế trên các tác vụ thuần tiếng Anh. Các mô hình đa ngôn ngữ giành chiến thắng khi tập ngữ liệu bị pha trộn nhiều thứ tiếng.
3. **Độ dài ngữ cảnh (Context length).** 512 token so với 8.192 hay 32.768 — và dung lượng hiệu dụng thực tế thường chỉ đạt 60-70% con số tối đa được quảng cáo.
4. **Ngân sách chiều không gian (Dimension budget).** 3.072 số thực dấu phẩy động ở độ chính xác đầy đủ (full precision) = 12 KB cho mỗi vector. Với 100 triệu vector, chi phí lưu trữ lên đến 1.300 USD/tháng. Kỹ thuật cắt tỉa Matryoshka giúp giảm con số này đi 4 lần.
5. **Mã nguồn mở vs Dịch vụ lưu trữ sẵn (Open vs hosted).** Trọng số mở (open-weight) giúp bạn làm chủ toàn bộ hạ tầng và dữ liệu. Dịch vụ lưu trữ (hosted API) đánh đổi quyền kiểm soát để luôn nhận được bản cập nhật mới nhất.

Bài học này định danh rõ các sự đánh đổi để bạn đưa ra lựa chọn dựa trên bằng chứng đo lường thực tế, thay vì chỉ chạy theo những gì thịnh hành trong quý trước.

## Khái niệm

![Nhúng dày đặc, thưa và đa vector](../assets/embedding-modes.svg)

**Nhúng dày đặc (Dense embeddings).** Một vector duy nhất cho mỗi đoạn văn (thường từ 384 đến 3.072 chiều). Độ tương đồng cosine (cosine similarity) xếp hạng các đoạn văn dựa trên khoảng cách ngữ nghĩa. Ví dụ: OpenAI `text-embedding-3-large`, chế độ dense của BGE-M3, Voyage-3. Đây là lựa chọn mặc định phổ biến nhất.

**Nhúng thưa (Sparse embeddings).** Theo phong cách SPLADE. Một transformer dự đoán trọng số cho mọi token trong từ vựng, sau đó triệt tiêu (zero out) phần lớn chúng. Kết quả là một vector thưa có kích thước bằng $|vocab|$. Cơ chế này nắm bắt sự khớp từ vựng chính xác (như BM25) nhưng kết hợp trọng số từ vựng được học qua mạng nơ-ron. Rất mạnh đối với các truy vấn giàu từ khóa kỹ thuật.

**Đa vector / Tương tác muộn (Multi-vector / late interaction).** ColBERTv2, Jina-ColBERT. Một vector cho mỗi token riêng lẻ. Chấm điểm bằng hàm MaxSim: với mỗi token trong truy vấn, tìm token tài liệu tương đồng nhất, sau đó tính tổng điểm. Tốn kém hơn về dung lượng lưu trữ và tính toán chấm điểm, nhưng vượt trội đối với các câu truy vấn dài và các ngữ liệu mang tính chuyên ngành sâu.

**BGE-M3: tích hợp cả ba cơ chế cùng lúc.** Mô hình duy nhất xuất ra đồng thời cả ba dạng biểu diễn dense, sparse và multi-vector. Mỗi dạng có thể được truy vấn độc lập; điểm số được dung hợp thông qua tổng có trọng số (weighted sum). Đây là tiêu chuẩn mặc định năm 2026 khi bạn muốn tính linh hoạt tối đa từ một checkpoint duy nhất.

**Học biểu diễn Matryoshka (Matryoshka Representation Learning - MRL).** Được huấn luyện sao cho $N$ chiều đầu tiên của vector tự thân nó đã tạo thành một vector nhúng độc lập hữu ích. Bạn có thể cắt tỉa (truncate) vector 1.536 chiều xuống còn 256 chiều, chỉ mất khoảng ~1% độ chính xác nhưng tiết kiệm dung lượng lưu trữ tới 6 lần. Được hỗ trợ bởi OpenAI text-3, Cohere v4, Voyage-4, Jina v5, Gemini Embedding 2, Nomic v1.5+.

### Bảng xếp hạng MTEB chỉ phản ánh một phần câu chuyện

Massive Text Embedding Benchmark (MTEB) — gồm 56 tác vụ trên 8 loại nhiệm vụ lúc khởi chạy (2022), mở rộng lên hơn 100 tác vụ trong MTEB v2. Đầu năm 2026, Gemini Embedding 2 dẫn đầu mảng truy xuất (67.71 MTEB-R). Cohere embed-v4 dẫn đầu điểm tổng quát (65.2 MTEB). BGE-M3 dẫn đầu các mô hình đa ngôn ngữ trọng số mở (63.0). Bảng xếp hạng là tài liệu tham khảo cần thiết nhưng chưa đủ — hãy luôn đo kiểm điểm chuẩn (benchmark) trực tiếp trên miền dữ liệu của riêng bạn.

### Mô hình kiến trúc ba tầng (The three-tier pattern)

| Tình huống sử dụng | Mô hình kiến trúc |
|--------------------|-------------------|
| Vòng lọc thô tốc độ cao (Fast first-pass) | Dense bi-encoder (BGE-M3, text-3-small) |
| Gia tăng độ phủ (Recall boost) | Sparse (SPLADE, BGE-M3 sparse) + Dung hợp RRF |
| Độ chính xác trên top-50 (Precision on top-50) | Multi-vector (ColBERTv2) hoặc cross-encoder reranker |

Hầu hết các hệ thống sản xuất (production) đều kết hợp cả ba tầng này.

```figure
gx-matryoshka
```

## Thực hành xây dựng

### Bước 1: Đường cơ sở — nhúng dày đặc với Sentence-BERT

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True` biến tích vô hướng (dot product) thành độ tương đồng cosine chuẩn xác. Hãy luôn bật tùy chọn này.

### Bước 2: Cắt tỉa Matryoshka

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

Luôn chuẩn hóa lại (re-normalize) sau khi cắt tỉa. Nomic v1.5, OpenAI text-3 và Voyage-4 được huấn luyện để bảo toàn gần như nguyên vẹn thông tin ở các cấp độ cắt đầu tiên. Các mô hình không hỗ trợ Matryoshka (như Sentence-BERT nguyên bản) sẽ suy giảm độ chính xác nghiêm trọng nếu bị cắt tỉa.

### Bước 3: Đa chức năng cùng BGE-M3

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
```

Ba chỉ mục riêng biệt chỉ từ một lần gọi suy luận (inference). Dung hợp điểm số:

```python
dense_score = ... # cosine over dense_vecs
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

Tinh chỉnh các trọng số này dựa trên tập đánh giá của miền dữ liệu bạn xử lý.

### Bước 4: Đánh giá MTEB trên tác vụ tùy chỉnh

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

Chạy các mô hình ứng viên trên một tập con *mang tính đại diện cao*. Đừng bao giờ chỉ tin tưởng tuyệt đối vào thứ hạng leaderboard — bản chất miền dữ liệu của bạn mới là yếu tố quyết định.

### Bước 5: Tự xây dựng hàm cosine từ đầu

Xem chi tiết tại `code/main.py`. Thuật toán Averaged Hashing Trick embeddings (chỉ sử dụng thư viện chuẩn stdlib). Dù không thể cạnh tranh về độ chính xác với các transformer embedding hiện đại, nó phơi bày toàn bộ cấu trúc nền tảng: tách từ (tokenize) → chuyển thành vector → chuẩn hóa vector → tính tích vô hướng.

## Cạm bẫy

- **Dùng chung một mô hình cho cả truy vấn và tài liệu.** Một số mô hình (Voyage, Jina-ColBERT) sử dụng mã hóa bất đối xứng (asymmetric encoding) — câu truy vấn và văn bản tài liệu đi qua hai nhánh xử lý khác nhau. Hãy luôn đọc kỹ thẻ mô hình (model card).
- **Thiếu tiền tố bắt buộc.** Các mô hình họ `bge-*` yêu cầu phải gắn thêm chuỗi tiền tố `"Represent this sentence for searching relevant passages: "` vào trước các câu truy vấn. Bỏ quên tiền tố này có thể làm sụt giảm 3-5 điểm recall.
- **Cắt tỉa Matryoshka quá sâu.** Cắt từ 1.536 xuống 256 chiều thường an toàn. Nhưng cắt từ 1.536 xuống 64 chiều sẽ mất mát rất nhiều thông tin. Cần kiểm chứng cẩn thận trên tập đánh giá (eval set).
- **Cắt xén ngữ cảnh âm thầm (Context truncation).** Đa số mô hình sẽ âm thầm cắt bỏ những phần văn bản vượt quá độ dài tối đa mà không báo lỗi. Các tài liệu dài bắt buộc phải được chia đoạn (xem bài 23).
- **Bỏ qua độ trễ đuôi (latency tail).** Điểm số MTEB che giấu độ trễ p99. Một mô hình 600M tham số có thể nhỉnh hơn mô hình 335M tham số 2 điểm đánh giá, nhưng tốn chi phí và thời gian gấp 3 lần cho mỗi truy vấn.

## Sử dụng trong thực tế

Ngăn xếp công nghệ năm 2026:

| Tình huống | Lựa chọn đề xuất |
|------------|-------------------|
| Thuần tiếng Anh, tốc độ nhanh, dùng API | `text-embedding-3-large` hoặc `voyage-3-large` |
| Trọng số mở, thuần tiếng Anh | `BAAI/bge-large-en-v1.5` |
| Trọng số mở, đa ngôn ngữ | `BAAI/bge-m3` hoặc `Qwen3-Embedding-8B` |
| Ngữ cảnh dài (32k+ token) | Voyage-3-large, Cohere embed-v4, Qwen3-Embedding-8B |
| Triển khai chỉ chạy trên CPU | Nomic Embed v2 (137M tham số, cấu trúc MoE) |
| Bị giới hạn nghiêm ngặt về dung lượng lưu trữ | Matryoshka cắt tỉa + lượng tử hóa int8 |
| Truy vấn chứa nhiều từ khóa đặc thù | Thêm biểu diễn thưa SPLADE, dung hợp RRF với dense |

Mô hình chuẩn của năm 2026: Bắt đầu với BGE-M3 hoặc text-3-large, đánh giá trực tiếp trên miền dữ liệu của bạn bằng MTEB, và chỉ thay thế nếu một mô hình chuyên biệt cho miền đó vượt trội hơn ít nhất 3 điểm.

## Đóng gói sản phẩm

Lưu tệp dưới dạng `outputs/skill-embedding-picker.md`:

```markdown
---
name: embedding-picker
description: Pick embedding model, dimension, and retrieval mode for a given corpus and deployment.
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

Given a corpus (size, languages, domain, avg length), deployment target (cloud / edge / on-prem), latency budget, and storage budget, output:

1. Model. Named checkpoint or API. One-sentence reason.
2. Dimension. Full / Matryoshka-truncated / int8-quantized. Reason tied to storage budget.
3. Mode. Dense / sparse / multi-vector / hybrid. Reason.
4. Query prefix / template if required by the model card.
5. Evaluation plan. MTEB tasks relevant to domain + held-out domain eval with nDCG@10.

Refuse recommendations that truncate Matryoshka to <64 dims without domain validation. Refuse ColBERTv2 for corpora under 10k passages (overhead not justified). Flag long-document corpora (>8k tokens) routed to models with 512-token windows.
```

## Bài tập

1. **Dễ.** Mã hóa 100 câu bằng `bge-small-en-v1.5` ở kích thước đầy đủ (384 chiều), sau đó cắt tỉa Matryoshka xuống 128 chiều. Đo lường mức độ suy giảm của chỉ số MRR trên 10 câu truy vấn.
2. **Trung bình.** So sánh ba chế độ dense, sparse và colbert của BGE-M3 trên 500 đoạn văn thuộc miền dữ liệu của bạn. Phương pháp nào giành chiến thắng ở chỉ số recall@10? Kỹ thuật dung hợp RRF có đánh bại chế độ đơn lẻ tốt nhất không?
3. **Khó.** Chạy đánh giá MTEB trên ba mô hình ứng viên thông qua 2 tác vụ quan trọng nhất trong miền dữ liệu của bạn. Báo cáo điểm MTEB, độ trễ p99 trên một lô gồm 100 truy vấn, và chi phí trên mỗi 1 triệu truy vấn. Chọn ra mô hình tối ưu theo đường biên Pareto (Pareto-optimal).

## Thuật ngữ then chốt

| Thuật ngữ | Cách gọi thông thường | Bản chất kỹ thuật thực sự |
|-----------|------------------------|---------------------------|
| Nhúng dày đặc (Dense embedding) | Vector nhúng | Một vector kích thước cố định cho mỗi văn bản. Dùng khoảng cách cosine để xếp hạng. |
| Nhúng thưa (Sparse embedding) | BM25 được học | Một trọng số cho mỗi từ trong từ điển; đa số là số 0; huấn luyện end-to-end. |
| Đa vector (Multi-vector) | Phong cách ColBERT | Một vector cho mỗi token; chấm điểm bằng MaxSim; chỉ mục lớn hơn, độ phủ tốt hơn. |
| Matryoshka | Kỹ thuật búp bê Nga | $N$ chiều đầu tiên tự nó tạo thành một vector nhúng nhỏ hơn nhưng hợp lệ. |
| MTEB | Bộ benchmark chuẩn | Massive Text Embedding Benchmark — 56 tác vụ lúc ban đầu, hơn 100 tác vụ ở v2. |
| BEIR | Bộ benchmark truy xuất | 18 tác vụ truy xuất zero-shot; thường dùng để đo độ bền vững xuyên miền. |
| Mã hóa bất đối xứng (Asymmetric encoding) | Query khác Document | Mô hình dùng hai phép chiếu (projection) khác nhau cho câu truy vấn và văn bản tài liệu. |

## Đọc thêm

- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084) — bài báo nền tảng về bi-encoder.
- [Muennighoff et al. (2022). MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) — bài báo công bố bảng xếp hạng MTEB.
- [Chen et al. (2024). BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity](https://arxiv.org/abs/2402.03216) — mô hình hợp nhất ba chế độ nhúng.
- [Kusupati et al. (2022). Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) — mục tiêu huấn luyện theo bậc thang số chiều.
- [Santhanam et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) — cơ chế tương tác muộn trong môi trường sản xuất.
- [Bảng xếp hạng MTEB trên Hugging Face](https://huggingface.co/spaces/mteb/leaderboard) — bảng xếp hạng trực tiếp.

