# Trích xuất quan hệ & Xây dựng đồ thị tri thức (Relation Extraction & Knowledge Graph Construction)

> NER tìm ra các thực thể. Liên kết thực thể neo chúng vào thực tại. Trích xuất quan hệ (relation extraction) tìm ra các cạnh liên kết giữa chúng. Một đồ thị tri thức (knowledge graph) chính là tổng hòa của các nút, các cạnh liên kết và nguồn gốc chứng minh (provenance) của chúng.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 5 · 06 (Nhận dạng thực thể có tên - NER), Giai đoạn 5 · 25 (Liên kết thực thể)
**Time:** ~60 phút

## Vấn đề

Một chuyên viên phân tích đọc câu: "Tim Cook became CEO of Apple in 2011." Câu này chứa đựng 4 sự kiện thực tế:

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

Trích xuất quan hệ (Relation Extraction - RE) chuyển đổi văn bản tự do thành các bộ ba cấu trúc (structured triples) theo dạng `(chủ thể, quan hệ, đối tượng)` — `(subject, relation, object)`. Khi tổng hợp xuyên suốt một tập ngữ liệu lớn, bạn sẽ có được một đồ thị tri thức (knowledge graph). Khi tổng hợp và truy vấn, bạn sẽ sở hữu một nền tảng suy luận (reasoning substrate) vững chắc phục vụ Graph RAG, phân tích dữ liệu chuyên sâu hay kiểm toán tuân thủ.

Thách thức năm 2026: Các LLM trích xuất quan hệ rất nhiệt tình. Thậm chí quá nhiệt tình. Chúng ảo giác (hallucinate) ra những bộ ba mà văn bản nguồn hoàn toàn không hề đề cập. Nếu không có cơ chế truy nguyên nguồn gốc (provenance), bạn không thể nào phân biệt được đâu là sự thật và đâu là câu chuyện hư cấu hợp lý. Giải pháp năm 2026 là áp dụng kiến trúc đường ống kiểm chứng và neo vị trí theo tiêu chuẩn AEVS.

## Khái niệm

![Văn bản → bộ ba quan hệ → đồ thị tri thức](../assets/relation-extraction.svg)

**Định dạng bộ ba (Triple form).** `(subject_entity, relation_type, object_entity)`. Các quan hệ có thể bắt nguồn từ một bản thể luận đóng (closed ontology - như các thuộc tính Wikidata, FIBO, UMLS) hoặc một tập mở tự do (OpenIE-style, bất kỳ cụm từ nào).

**Ba phương pháp trích xuất chính.**

1. **Dựa trên luật / mẫu quy tắc (Rule / pattern-based).** Mẫu hình Hearst (Hearst patterns): "X chẳng hạn như Y" → `(Y, isA, X)`. Kết hợp với biểu thức chính quy (regex) thủ công. Cứng nhắc, độ chính xác cao, tính giải thích tường minh.
2. **Bộ phân loại có giám sát (Supervised classifier).** Cho trước hai biểu thức thực thể trong câu, mô hình dự đoán mối quan hệ giữa chúng từ một tập nhãn cố định. Huấn luyện trên TACRED, ACE, KBP. Tiêu chuẩn chủ đạo giai đoạn 2015–2022.
3. **Mô hình sinh LLM (Generative LLM).** Viết prompt yêu cầu LLM trích xuất các bộ ba. Hoạt động ngay mà không cần huấn luyện. Nhưng bắt buộc phải gắn nguồn gốc trích dẫn, nếu không sẽ sinh ra rất nhiều thông tin rác ngụy tạo.

**Khung làm việc AEVS (Anchor-Extraction-Verification-Supplement, 2026).** Giải pháp giảm thiểu ảo giác tiên tiến nhất:

- **Neo vị trí (Anchor):** Xác định chính xác vị trí chỉ mục (char-span) của mọi thực thể và cụm từ chỉ quan hệ trong văn bản gốc.
- **Trích xuất (Extract):** Tạo ra các bộ ba được liên kết chặt chẽ với các đoạn neo.
- **Kiểm chứng (Verify):** Đối soát từng thành phần của bộ ba ngược lại văn bản nguồn; loại bỏ mọi bộ ba không có bằng chứng văn bản trực tiếp hỗ trợ.
- **Bổ sung (Supplement):** Quét kiểm tra độ phủ để đảm bảo không có đoạn neo hợp lệ nào bị bỏ sót.

Ảo giác giảm sút rõ rệt. Đòi hỏi nhiều năng lực tính toán hơn nhưng mang lại tính kiểm toán (auditability) tuyệt đối.

**Sự đánh đổi giữa bản thể luận Mở vs Đóng (Open-vs-closed tradeoff).**

- **Bản thể luận đóng (Closed ontology):** Danh sách quan hệ cố định (ví dụ hơn 11.000 thuộc tính của Wikidata). Dễ dự đoán. Dễ truy vấn. Khó tự động mở rộng sang các mối quan hệ mới lạ.
- **Trích xuất thông tin mở (Open IE):** Mọi cụm động từ đều có thể trở thành một quan hệ. Độ phủ (recall) rất cao. Độ chính xác (precision) thấp. Rất hỗn loạn khi cần truy vấn cơ sở dữ liệu.

Các đồ thị tri thức trong môi trường sản xuất thường kết hợp cả hai: dùng Open IE để khám phá các mối quan hệ mới, sau đó chuẩn tắc hóa (canonicalize) chúng về bản thể luận đóng trước khi hợp nhất vào đồ thị chính.

```figure
relation-triples
```

## Thực hành xây dựng

### Bước 1: Trích xuất dựa trên mẫu quy tắc

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

Xem toàn bộ mã nguồn trích xuất minh họa trong `code/main.py`. Các mẫu hình Hearst vẫn thường xuyên được vận hành trong các đường ống xử lý nghiệp vụ hẹp vì chúng rất dễ gỡ lỗi (debug).

### Bước 2: Trích xuất quan hệ seq2seq có giám sát với REBEL

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
decoded = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL là mô hình trích xuất quan hệ dạng seq2seq dựa trên BART: nó sinh trực tiếp các bộ ba thay vì phân loại từng cặp thực thể rời rạc. Đưa văn bản vào, nhận bộ ba ra, với các tên quan hệ lấy từ các thuộc tính của Wikidata. Các chuỗi được giải mã vẫn mang các thẻ đánh dấu `<triplet>`, `<subj>` và `<obj>` của REBEL, do đó chỉ cần tách theo các thẻ này để lấy bản ghi `(subject, relation, object)`. Được tinh chỉnh trên dữ liệu giám sát từ xa (distant supervision). Đây là baseline trọng số mở tiêu chuẩn.

### Bước 3: Trích xuất bằng Prompt LLM kết hợp neo vị trí (Anchoring)

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

Kiểm chứng từng đoạn span trả về đối chiếu với văn bản gốc. Loại bỏ bất kỳ bộ ba nào mà `text[start:end] != triple_entity`. Đây chính là bước "Verify" tối giản trong quy trình AEVS.

### Bước 4: Chuẩn tắc hóa về bản thể luận đóng

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (đảo chiều subject/object)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # loại bỏ quan hệ mở chưa ánh xạ hoặc chuyển sang duyệt thủ công
```

Công việc chuẩn tắc hóa thường chiếm từ 60-80% khối lượng kỹ thuật dự án. Hãy lên kế hoạch phân bổ thời gian tương xứng.

### Bước 5: Xây dựng đồ thị tri thức nhỏ và truy vấn

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

Đây là hạt nhân nguyên tử của mọi hệ thống RAG trên đồ thị tri thức (KG-RAG). Mở rộng quy mô bằng các cơ sở dữ liệu bộ ba RDF (Blazegraph, Virtuoso), đồ thị thuộc tính (Neo4j), hoặc các kho lưu trữ đồ thị tăng cường vector nhúng.

## Cạm bẫy

- **Quên giải quyết đồng tham chiếu trước khi trích xuất quan hệ.** "He founded Apple" — bộ RE cần biết "He" là ai. Bắt buộc phải chạy coref trước (bài 24).
- **Chuẩn tắc hóa thực thể.** "Apple Inc" và "Apple" phải được ánh xạ về cùng một nút đồ thị. Phải chạy liên kết thực thể (EL) trước (bài 25).
- **Bộ ba bị ảo giác.** LLM sinh ra những bộ ba không hề có căn cứ trong tài liệu. Bắt buộc phải áp dụng bước xác thực vị trí đoạn (span verification).
- **Trôi dạt chuẩn tắc hóa quan hệ.** Các quan hệ trong Open IE rất thiếu nhất quán ("was born in", "came from", "is a native of"). Bắt buộc phải quy về các mã ID chuẩn tắc, nếu không đồ thị sẽ không thể truy vấn được.
- **Sai lệch về yếu tố thời gian.** "Tim Cook is CEO of Apple" — đúng ở thời điểm hiện tại, nhưng sai vào năm 2005. Rất nhiều quan hệ có tính ràng buộc về thời gian. Cần sử dụng các định từ bổ trợ (qualifiers: `P580` thời điểm bắt đầu, `P582` thời điểm kết thúc trong Wikidata).
- **Lệch miền dữ liệu.** REBEL được huấn luyện trên Wikipedia. Văn bản pháp lý, y sinh học và khoa học đòi hỏi các mô hình RE được tinh chỉnh chuyên biệt cho miền đó.

## Sử dụng trong thực tế

Ngăn xếp công nghệ năm 2026:

| Tình huống | Lựa chọn đề xuất |
|------------|-------------------|
| Sản xuất nhanh, miền tổng quát | REBEL hoặc LlamaPred kết hợp chuẩn tắc hóa theo Wikidata |
| Miền chuyên biệt (Y sinh, Pháp lý) | Tinh chỉnh mô hình dạng SciREX + bản thể luận riêng |
| Dùng LLM, đầu ra có kiểm toán nghiêm ngặt | Đường ống AEVS: neo vị trí → trích xuất → kiểm chứng → bổ sung |
| Trích xuất tin tức khối lượng lớn | Lai tạp giữa mẫu hình quy tắc + mô hình phân loại có giám sát |
| Xây dựng KG hoàn toàn mới từ đầu | Open IE kết hợp vòng duyệt chuẩn tắc hóa bán thủ công |
| Đồ thị tri thức biến thiên theo thời gian (Temporal KG) | Trích xuất kèm các định từ thời gian (bắt đầu/kết thúc, thời điểm) |

Mô hình tích hợp chuẩn: NER → Giải quyết đồng tham chiếu (coref) → Liên kết thực thể (EL) → Trích xuất quan hệ (RE) → Ánh xạ bản thể luận → Nạp vào đồ thị. Mỗi giai đoạn đều là một cổng kiểm soát chất lượng (quality gate).

## Đóng gói sản phẩm

Lưu tệp dưới dạng `outputs/skill-re-designer.md`:

```markdown
---
name: re-designer
description: Design a relation extraction pipeline with provenance and canonicalization.
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

Given a corpus (domain, language, volume) and downstream use (KG-RAG, analytics, compliance), output:

1. Extractor. Pattern-based / supervised / LLM / AEVS hybrid. Reason tied to precision vs recall target.
2. Ontology. Closed property list (Wikidata / domain) or open IE with canonicalization pass.
3. Provenance. Every triple carries source char-span + doc id. Non-negotiable for audit.
4. Merge strategy. Canonical entity id + relation id + temporal qualifiers; dedup policy.
5. Evaluation. Precision / recall on 200 hand-labelled triples + hallucination-rate on LLM-extracted sample.

Refuse any LLM-based RE pipeline without span verification (source provenance). Refuse open-IE output flowing into a production graph without canonicalization. Flag pipelines with no temporal qualifier on time-bounded relations (employer, spouse, position).
```

## Bài tập

1. **Dễ.** Chạy bộ trích xuất theo mẫu trong `code/main.py` trên 5 câu báo chí thời sự. Tự kiểm tra độ chính xác (precision).
2. **Trung bình.** Sử dụng REBEL (hoặc một LLM nhỏ) trên chính các câu văn đó. So sánh các bộ ba thu được. Bộ trích xuất nào có độ chính xác cao hơn? Bộ nào có độ phủ (recall) tốt hơn?
3. **Khó.** Xây dựng đường ống AEVS: trích xuất bằng LLM + kiểm chứng vị trí span đối soát văn bản nguồn. Đo lường tỷ lệ ảo giác trước và sau bước kiểm chứng trên 50 câu văn phong cách Wikipedia.

## Thuật ngữ then chốt

| Thuật ngữ | Cách gọi thông thường | Bản chất kỹ thuật thực sự |
|-----------|------------------------|---------------------------|
| Bộ ba (Triple) | Chủ thể-quan hệ-đối tượng | Bộ dữ liệu `(s, r, o)` là đơn vị nguyên tử cơ bản nhất của một đồ thị tri thức. |
| Open IE | Trích xuất tự do | Các cụm từ chỉ quan hệ dùng từ vựng mở; độ phủ cao, độ chính xác thấp. |
| Bản thể luận đóng (Closed ontology) | Lược đồ cố định | Tập hợp hữu hạn các loại quan hệ xác định trước (Wikidata, UMLS, FIBO). |
| Chuẩn tắc hóa (Canonicalization) | Chuẩn hóa tất cả | Ánh xạ các tên bề mặt / quan hệ về các mã định danh chuẩn tắc duy nhất. |
| AEVS | Trích xuất có neo căn cứ | Quy trình Anchor-Extraction-Verification-Supplement (2026). |
| Truy nguyên nguồn gốc (Provenance) | Liên kết bằng chứng nguồn | Mỗi bộ ba đều mang theo mã tài liệu + chỉ mục đoạn ký tự trích xuất từ văn bản gốc. |
| Giám sát từ xa (Distant supervision) | Gán nhãn giá rẻ | Gióng hàng văn bản với một đồ thị tri thức có sẵn để tự động tạo dữ liệu huấn luyện. |

## Đọc thêm

- [Mintz et al. (2009). Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113.pdf) — bài báo nền tảng về phương pháp giám sát từ xa.
- [Huguet Cabot, Navigli (2021). REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204.pdf) — mô hình seq2seq RE xuất sắc.
- [Wadden et al. (2019). Entity, Relation, and Event Extraction with Contextualized Span Representations (DyGIE++)](https://arxiv.org/abs/1909.03546) — trích xuất thực thể và quan hệ đồng thời.
- [Khung làm việc AEVS — Anchor-Extraction-Verification-Supplement](https://www.mdpi.com/2073-431X/15/3/178) — thiết kế triệt tiêu ảo giác năm 2026.
- [Hướng dẫn truy vấn Wikidata SPARQL](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) — chuẩn mực truy vấn đồ thị tri thức.

