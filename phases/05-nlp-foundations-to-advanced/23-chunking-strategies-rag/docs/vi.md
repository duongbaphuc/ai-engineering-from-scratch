# Chiến lược phân đoạn cho RAG

> Cấu hình phân đoạn (chunking) ảnh hưởng đến chất lượng truy xuất ngang ngửa với việc lựa chọn mô hình nhúng (Vectara NAACL 2025). Nếu làm sai bước phân đoạn, không một bộ tái xếp hạng (reranker) nào có thể cứu vãn được hệ thống của bạn.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 5 · 14 (Truy xuất thông tin), Giai đoạn 5 · 22 (Mô hình nhúng)
**Time:** ~60 phút

## Vấn đề

Bạn nạp một bản hợp đồng dài 50 trang vào hệ thống RAG. Người dùng hỏi: "Điều khoản chấm dứt hợp đồng quy định như thế nào?" Bộ truy xuất lại trả về... trang bìa. Tại sao? Bởi vì mô hình nhúng được huấn luyện trên các đoạn văn bản dài 512 token, trong khi điều khoản chấm dứt hợp đồng nằm tận trang 20, bị ngắt quãng giữa chừng qua một dấu ngắt trang (page break), và không hề có từ khóa cục bộ nào khớp trực tiếp với câu truy vấn.

Cách khắc phục không phải là "mua một mô hình nhúng xịn hơn". Lời giải nằm ở kỹ thuật phân đoạn (chunking). Kích thước đoạn bao nhiêu là tối ưu? Tỷ lệ gối đầu (overlap) ra sao? Điểm cắt ở đâu? Có nên giữ ngữ cảnh bao quanh không?

Các nghiên cứu đối chuẩn (benchmark) vào tháng 2 năm 2026 cho thấy những kết quả rất bất ngờ:

- Nghiên cứu năm 2026 của Vectara: Phân đoạn đệ quy (recursive chunking) kích thước 512 token đánh bại phân đoạn ngữ nghĩa (semantic chunking) với tỷ lệ độ chính xác 69% so với 54%.
- SPLADE + Mistral-8B trên tập dữ liệu Natural Questions: Việc cho các đoạn gối đầu nhau (overlap) hoàn toàn không mang lại lợi ích đo lường được nào.
- Vực thẳm ngữ cảnh (Context cliff): Chất lượng phản hồi sụt giảm mạnh xung quanh mốc 2.500 token ngữ cảnh.

Câu trả lời tưởng như "hiển nhiên" (phân đoạn ngữ nghĩa, gối đầu 20%, kích thước 1.000 token) thực tế thường là sai lầm. Bài học này xây dựng trực giác cho 6 chiến lược phân đoạn và hướng dẫn bạn chính xác khi nào nên áp dụng chiến lược nào.

## Khái niệm

![Minh họa sáu chiến lược phân đoạn trên cùng một đoạn văn](../assets/chunking.svg)

**Phân đoạn cố định (Fixed chunking).** Cắt văn bản sau mỗi $N$ ký tự hoặc token. Đây là baseline đơn giản nhất. Dễ bị ngắt gãy giữa câu. Nén dữ liệu tốt nhưng tính mạch lạc rất kém.

**Phân đoạn đệ quy (Recursive chunking).** Lấy cảm hứng từ `RecursiveCharacterTextSplitter` của LangChain. Thử ngắt dòng theo `\n\n` trước tiên, sau đó tới `\n`, dấu chấm `.`, và cuối cùng là khoảng trắng. Rơi về mức thấp hơn một cách mượt mà. Đây là tiêu chuẩn mặc định của năm 2026.

**Phân đoạn ngữ nghĩa (Semantic chunking).** Nhúng từng câu đơn lẻ thành vector. Tính toán độ tương đồng cosine giữa các câu liền kề nhau. Ngắt đoạn tại vị trí độ tương đồng rớt xuống dưới một ngưỡng (threshold) nhất định. Bảo toàn tính mạch lạc theo chủ đề. Tốc độ xử lý chậm hơn; đôi khi tạo ra các mẩu vụn chỉ vỏn vẹn 40 token gây tổn hại đến chất lượng truy xuất.

**Phân đoạn theo câu (Sentence chunking).** Cắt tại ranh giới kết thúc câu. Một câu trên mỗi đoạn hoặc một cửa sổ trượt gồm $N$ câu. Đạt hiệu năng tương đương phân đoạn ngữ nghĩa lên tới ~5k token với chi phí tính toán rẻ hơn rất nhiều.

**Phân đoạn cha-con (Parent-document).** Lưu trữ các đoạn con (child chunks) kích thước nhỏ để phục vụ truy xuất vector *đồng thời* lưu trữ đoạn cha (parent chunk) lớn hơn để cung cấp ngữ cảnh đầy đủ cho LLM. Truy xuất theo đoạn con, nhưng trả về đoạn cha. Cơ chế này suy giảm rất êm ái: đoạn con dẫu chưa hoàn hảo vẫn kéo về được đoạn cha chứa thông tin thỏa đáng.

**Phân đoạn muộn (Late chunking - 2024).** Nhúng toàn bộ tài liệu ở cấp độ token trước tiên, sau đó gom nhóm (pooling) các token embedding thành các chunk embedding tương ứng. Giữ trọn vẹn ngữ cảnh xuyên suốt giữa các đoạn. Hoạt động với các mô hình nhúng ngữ cảnh dài (BGE-M3, Jina v3). Đòi hỏi năng lực tính toán cao hơn.

**Truy xuất theo ngữ cảnh (Contextual retrieval - Anthropic, 2024).** Gắn vào đầu mỗi đoạn văn bản một tóm tắt ngắn do LLM sinh ra mô tả vị trí của đoạn đó trong toàn bộ tài liệu ("Đoạn này thuộc mục 3.2 quy định về điều khoản chấm dứt..."). Giúp cải thiện chất lượng truy xuất từ 35-50% theo báo cáo của Anthropic. Tốn kém chi phí trong giai đoạn đánh chỉ mục (indexing).

### Quy tắc vượt trội hơn mọi giá trị mặc định

Hãy luôn khớp kích thước đoạn (chunk size) với loại câu truy vấn:

| Loại câu truy vấn | Kích thước đoạn tối ưu |
|-------------------|-------------------------|
| Truy vấn dữ kiện ngắn (Factoid, "Ai là CEO?") | 256-512 token |
| Phân tích / Đa bước suy luận (Multi-hop) | 512-1024 token |
| Hiểu toàn bộ đề mục / phần lớn | 1024-2048 token |

Đo chuẩn năm 2026 của NVIDIA: Đoạn văn bản phải đủ lớn để chứa câu trả lời kèm ngữ cảnh cục bộ, nhưng cũng phải đủ nhỏ để kết quả top-K của bộ truy xuất tập trung thẳng vào câu trả lời thay vì bị nhiễu bởi các thông tin râu ria xung quanh.

```figure
n5-chunk-cuts
```

## Thực hành xây dựng

### Bước 1: Phân đoạn cố định và phân đoạn đệ quy

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### Bước 2: Phân đoạn ngữ nghĩa

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

Hãy tinh chỉnh `threshold` trên miền dữ liệu của bạn. Quá cao → văn bản bị xé nhỏ thành các mẩu vụn. Quá thấp → dồn thành một khối văn bản khổng lồ.

### Bước 3: Phân đoạn cha-con (Parent-document)

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

Điểm mấu chốt: Luôn loại bỏ các đoạn cha trùng lặp (deduplicate). Nhiều đoạn con có thể cùng trỏ về một đoạn cha duy nhất; việc trả về lặp lại cùng một đoạn cha sẽ làm lãng phí dung lượng cửa sổ ngữ cảnh.

### Bước 4: Truy xuất theo ngữ cảnh (Contextual retrieval theo mô hình Anthropic)

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

Đánh chỉ mục các đoạn văn bản đã được bổ sung ngữ cảnh này. Vào thời điểm truy vấn, quá trình tìm kiếm sẽ được hưởng lợi từ các tín hiệu bao quát xung quanh.

### Bước 5: Đánh giá chất lượng

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

Hãy luôn chạy thử nghiệm đo lường thực tế. Chiến lược "tốt nhất" cho tập tài liệu của bạn có thể không hề trùng khớp với các bài viết hướng dẫn trên mạng.

## Cạm bẫy

- **Chỉ đánh giá phân đoạn trên các câu hỏi dữ kiện ngắn.** Các câu truy vấn đa bước (multi-hop) sẽ làm lộ diện những phương pháp chiến thắng hoàn toàn khác biệt. Hãy sử dụng tập dữ liệu đánh giá được phân tầng theo loại truy vấn.
- **Dùng phân đoạn ngữ nghĩa mà không chặn kích thước tối thiểu.** Việc này tạo ra những mẩu văn bản chỉ 40 token gây suy giảm nghiêm trọng độ chính xác truy xuất. Phải luôn thiết lập ngưỡng chặn dưới `min_tokens`.
- **Áp dụng tỷ lệ gối đầu (overlap) một cách mù quáng.** Các nghiên cứu năm 2026 chỉ ra rằng overlap thường không mang lại lợi ích gì mà còn làm tăng gấp đôi chi phí lưu trữ chỉ mục. Hãy đo lường thực tế chứ đừng mặc định.
- **Không kiểm soát cận trên và cận dưới (min/max).** Các đoạn chỉ 5 token hay dài tới 5.000 token đều phá hỏng hệ thống truy xuất. Bắt buộc phải kẹp giá trị (clamp).
- **Phân đoạn bắc cầu xuyên tài liệu (Cross-doc chunking).** Tuyệt đối không để một đoạn văn bản bị kéo dài sang ranh giới của hai tài liệu khác nhau. Phải luôn phân đoạn trên từng tài liệu riêng rẽ, sau đó mới tổng hợp lại.

## Sử dụng trong thực tế

Ngăn xếp công nghệ năm 2026:

| Tình huống | Chiến lược đề xuất |
|------------|---------------------|
| Mới xây dựng ban đầu, chưa rõ đặc thù ngữ liệu | Đệ quy (Recursive), 512 token, không gối đầu (no overlap) |
| Hỏi đáp dữ kiện ngắn (Factoid QA) | Đệ quy, 256-512 token |
| Phân tích chuyên sâu / Đa bước suy luận | Đệ quy, 512-1024 token + cấu trúc cha-con (parent-document) |
| Tài liệu nhiều tham chiếu chéo (Hợp đồng, bài báo khoa học) | Phân đoạn muộn (Late chunking) hoặc Truy xuất theo ngữ cảnh (Contextual retrieval) |
| Ngữ liệu hội thoại / Chat | Phân đoạn theo lượt thoại (turn-level) + metadata người nói |
| Phát ngôn ngắn (Tweet, nhận xét đánh giá) | Một tài liệu = Một đoạn (One document = one chunk) |

Bắt đầu với phân đoạn đệ quy 512 token. Đo lường chỉ số recall@5 trên tập đánh giá gồm 50 câu truy vấn. Sau đó tinh chỉnh dần.

## Đóng gói sản phẩm

Lưu tệp dưới dạng `outputs/skill-chunker.md`:

```markdown
---
name: chunker
description: Pick a chunking strategy, size, and overlap for a given corpus and query distribution.
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

Given a corpus (document types, avg length, domain) and query distribution (factoid / analytical / multi-hop), output:

1. Strategy. Recursive / sentence / semantic / parent-document / late / contextual. Reason.
2. Chunk size. Token count. Reason tied to query type.
3. Overlap. Default 0; justify if >0.
4. Min/max enforcement. `min_tokens`, `max_tokens` guards.
5. Evaluation plan. Recall@5 on 50-query stratified eval set (factoid, analytical, multi-hop).

Refuse any chunking strategy without min/max chunk size enforcement. Refuse overlap above 20% without an ablation showing it helps. Flag semantic chunking recommendations without a min-token floor.
```

## Bài tập

1. **Dễ.** Phân đoạn một tài liệu dài 20 trang bằng fixed(512, 0), recursive(512, 0), và recursive(512, 100). So sánh số lượng đoạn tạo thành và chất lượng các điểm cắt ranh giới câu.
2. **Trung bình.** Xây dựng một tập đánh giá gồm 30 câu truy vấn trên 5 tài liệu. Đo lường chỉ số recall@5 cho ba phương pháp: đệ quy, ngữ nghĩa và cha-con. Phương pháp nào chiến thắng? Kết quả có khớp với các khuyến nghị lý thuyết không?
3. **Khó.** Triển khai kỹ thuật truy xuất theo ngữ cảnh (contextual retrieval). Đo lường mức cải thiện của chỉ số MRR so với phân đoạn đệ quy cơ sở. Báo cáo chi phí đánh chỉ mục (số lượt gọi LLM) so với phần trăm độ chính xác gia tăng.

## Thuật ngữ then chốt

| Thuật ngữ | Cách gọi thông thường | Bản chất kỹ thuật thực sự |
|-----------|------------------------|---------------------------|
| Đoạn (Chunk) | Một mẩu tài liệu | Đơn vị văn bản nhỏ được nhúng thành vector, lập chỉ mục và truy xuất. |
| Gối đầu (Overlap) | Vùng biên an toàn | $N$ token được chia sẻ chung giữa hai đoạn liền kề; thường vô tác dụng trong các benchmark 2026. |
| Phân đoạn ngữ nghĩa (Semantic chunking) | Cắt đoạn thông minh | Cắt đoạn tại vị trí độ tương đồng embedding giữa các câu liền kề bị giảm sút. |
| Cha-con (Parent-document) | Truy xuất hai cấp | Truy xuất theo các mẩu con nhỏ gọn, nhưng trả về đoạn cha rộng lớn hơn. |
| Phân đoạn muộn (Late chunking) | Phân đoạn sau nhúng | Nhúng toàn bộ tài liệu ở cấp độ token, sau đó gom nhóm thành vector của từng đoạn. |
| Truy xuất theo ngữ cảnh (Contextual retrieval) | Tuyệt chiêu của Anthropic | Đính kèm bản tóm tắt do LLM tạo ra vào đầu mỗi đoạn văn trước khi đánh chỉ mục. |
| Vực thẳm ngữ cảnh (Context cliff) | Bức tường 2.500 token | Sự sụt giảm chất lượng quan sát thấy quanh mốc 2.5k token ngữ cảnh trong RAG (tháng 1/2026). |

## Đọc thêm

- [Yepes et al. / LangChain — Tài liệu tách văn bản đệ quy](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — cấu hình mặc định trong sản xuất.
- [Vectara (2024, NAACL 2025). Chunking configurations analysis](https://arxiv.org/abs/2410.13070) — phân đoạn quan trọng tương đương lựa chọn mô hình nhúng.
- [Jina AI — Late Chunking in Long-Context Embedding Models (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) — bài báo giới thiệu late chunking.
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — cải thiện 35-50% chất lượng truy xuất nhờ tiền tố ngữ cảnh do LLM sinh ra.
- [Đo chuẩn kích thước đoạn năm 2026 của NVIDIA — Tóm tắt bởi Premai](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) — hướng dẫn chọn kích thước đoạn theo loại truy vấn.

