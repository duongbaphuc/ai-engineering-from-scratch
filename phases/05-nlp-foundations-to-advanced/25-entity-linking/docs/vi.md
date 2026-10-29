# Liên kết thực thể & Khử nhập nhằng (Entity Linking & Disambiguation)

> NER tìm thấy từ "Paris". Kỹ thuật liên kết thực thể (entity linking) đưa ra phán quyết: Paris thủ đô nước Pháp? Paris Hilton? Thị trấn Paris ở bang Texas? Hay hoàng tử Paris thành Troy? Nếu thiếu bước liên kết thực thể, đồ thị tri thức của bạn sẽ mãi mãi chìm trong sự mơ hồ.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 5 · 06 (Nhận dạng thực thể có tên - NER), Giai đoạn 5 · 24 (Giải quyết đồng tham chiếu)
**Time:** ~60 phút

## Vấn đề

Một câu văn viết: "Jordan beat the press." Mô hình NER gắn nhãn "Jordan" là PERSON (người). Rất tốt. Nhưng đó là *Jordan nào*?

- Michael Jordan (huyền thoại bóng rổ)?
- Michael B. Jordan (nam diễn viên điện ảnh)?
- Michael I. Jordan (giáo sư học máy tại UC Berkeley — sự nhầm lẫn này xuất hiện rất thật trong các bài báo khoa học ML)?
- Jordan (quốc gia Jordan ở Trung Đông)?
- Jordan (tên riêng trong tiếng Do Thái)?

Liên kết thực thể (Entity Linking - EL) ánh xạ mỗi biểu thức đề cập (mention) về một mục duy nhất trong cơ sở tri thức (Knowledge Base - KB): Wikidata, Wikipedia, DBpedia hoặc cơ sở tri thức nghiệp vụ riêng của bạn. Quy trình gồm hai tác vụ con:

1. **Sinh ứng viên (Candidate generation).** Với từ "Jordan", những thực thể nào trong KB là khả dĩ?
2. **Khử nhập nhằng (Disambiguation).** Dựa vào ngữ cảnh xung quanh, ứng viên nào là chính xác nhất?

Cả hai bước đều có thể huấn luyện bằng mô hình học máy. Cả hai đều có chuẩn đánh giá (benchmark) riêng biệt. Đường ống tích hợp này đã ổn định suốt một thập kỷ qua — điều thay đổi chính là chất lượng vượt bậc của các bộ khử nhập nhằng hiện đại.

## Khái niệm

![Đường ống liên kết thực thể: biểu thức đề cập → các ứng viên → thực thể đã khử nhập nhằng](../assets/entity-linking.svg)

**Sinh ứng viên (Candidate generation).** Dựa vào bề mặt từ ngữ của biểu thức ("Jordan"), tra cứu các ứng viên trong chỉ mục bí danh (alias index). Từ điển bí danh của Wikipedia bao phủ hầu hết các thực thể có tên: "JFK" → John F. Kennedy, Jacqueline Kennedy, Sân bay JFK, Bộ phim JFK. Một chỉ mục điển hình thường trả về từ 10 đến 30 ứng viên tiềm năng cho mỗi biểu thức.

**Khử nhập nhằng: Ba hướng tiếp cận chính.**

1. **Xác suất tiên nghiệm + Ngữ cảnh (Prior + context - Milne & Witten, 2008).** $P(\text{entity} \mid \text{mention}) \times \text{context-similarity}(\text{entity}, \text{text})$. Hoạt động hiệu quả, tốc độ rất nhanh, không đòi hỏi huấn luyện phức tạp.
2. **Dựa trên phép nhúng (Embedding-based - ESS / REL / BLINK).** Nhúng vector cho biểu thức kèm ngữ cảnh. Nhúng mô tả của từng thực thể ứng viên. Chọn ứng viên có độ tương đồng cosine lớn nhất. Tiêu chuẩn mặc định giai đoạn 2020–2024.
3. **Mô hình sinh (Generative - GENRE 2021; dựa trên LLM 2023+).** Sinh mã định danh hoặc tên chuẩn tắc của thực thể theo từng token. Được ràng buộc bởi cây tiền tố (trie) của các tên thực thể hợp lệ, đảm bảo đầu ra luôn là một ID hợp lệ trong KB.

**Đầu-cuối so với Đường ống (End-to-end vs pipeline).** Các mô hình hiện đại (ELQ, BLINK, ExtEnD, GENRE) chạy NER + sinh ứng viên + khử nhập nhằng chỉ trong một lượt duyệt duy nhất. Tuy nhiên, hệ thống dạng đường ống (pipeline) vẫn chiếm ưu thế trong môi trường sản xuất nhờ tính module hóa linh hoạt cho phép thay thế từng thành phần.

### Hai phép đo then chốt

- **Độ phủ biểu thức (Mention recall - bước sinh ứng viên):** Tỷ lệ các biểu thức chuẩn mà thực thể đích thực tế nằm trong danh sách ứng viên được tạo ra. Đây là trần hiệu năng (floor) của toàn bộ đường ống.
- **Độ chính xác khử nhập nhằng (Disambiguation accuracy / F1):** Trong điều kiện ứng viên đúng đã có mặt, tỷ lệ thuật toán chọn đúng thực thể ở vị trí top-1 là bao nhiêu.

Hãy luôn báo cáo đồng thời cả hai chỉ số. Một hệ thống có độ chính xác khử nhập nhằng 99% nhưng độ phủ ứng viên chỉ đạt 80% thì hiệu năng tổng thể của toàn bộ đường ống cũng chỉ dừng lại ở mức 80%.

```figure
gx-entity-linking
```

## Thực hành xây dựng

### Bước 1: Xây dựng chỉ mục bí danh từ các trang đổi hướng Wikipedia

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

Dữ liệu bí danh Wikipedia chứa khoảng ~18 triệu cặp (alias, entity). Tải về từ các bản kết xuất (dump) của Wikidata và lưu trữ dưới dạng chỉ mục đảo (inverted index).

### Bước 2: Khử nhập nhằng dựa trên ngữ cảnh

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

Phép đo trùng lặp từ Jaccard ở trên là phiên bản mô phỏng trực giác. Trong thực tế, bạn sẽ thay thế bằng độ tương đồng cosine trên các vector nhúng (xem bước 2 trong `code/main.py` để xem phiên bản sử dụng Transformer).

### Bước 3: Khử nhập nhằng dựa trên vector nhúng (Phong cách BLINK)

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

Tại thời điểm đánh chỉ mục, nhúng sẵn tất cả thực thể trong KB một lần duy nhất. Tại thời điểm truy vấn, nhúng biểu thức kèm ngữ cảnh, tính tích vô hướng với kho ứng viên và chọn thực thể có điểm cao nhất.

### Bước 4: Liên kết thực thể dạng mô hình sinh (Khái niệm)

GENRE giải mã tiêu đề trang Wikipedia của thực thể theo từng ký tự. Quá trình giải mã có ràng buộc (constrained decoding - xem bài 20) đảm bảo chỉ những tiêu đề thực sự tồn tại mới được phép sinh ra. Con cháu hiện đại của cơ chế này là REL-GEN và các kỹ thuật LLM-prompted EL với định dạng xuất cấu trúc nghiêm ngặt.

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

Kết hợp với danh sách trắng (whitelist - ví dụ Outlines `choice`), đây là đường ống liên kết thực thể tinh gọn và dễ triển khai nhất trong năm 2026.

### Bước 5: Đánh giá trên tập AIDA-CoNLL

AIDA-CoNLL là bộ đo chuẩn quốc tế cho bài toán EL: gồm 1.393 bài báo Reuters, 34.000 lượt đề cập được liên kết tới thực thể Wikipedia. Báo cáo độ chính xác trong tập KB (`P@1`) và tỷ lệ phát hiện thực thể ngoài KB (NIL-detection rate).

## Cạm bẫy

- **Xử lý thực thể rỗng (NIL handling).** Một số biểu thức không hề tồn tại trong cơ sở tri thức (thực thể mới xuất hiện, nhân vật ít người biết). Hệ thống bắt buộc phải dự đoán nhãn NIL thay vì cố gắng đoán bừa một thực thể sai lệch. Cần đánh giá tác vụ này độc lập.
- **Lỗi ranh giới biểu thức (Mention boundary errors).** Mô hình NER phía trước nhận diện thiếu từ (ví dụ "Bank of America" bị cắt ngắn thành "Bank"). Khiến độ phủ của EL giảm mạnh.
- **Thiên lệch mức độ phổ biến (Popularity bias).** Các mô hình đã huấn luyện thường có xu hướng dự đoán thiên vị về các thực thể quá nổi tiếng. Biểu thức "Michael I. Jordan" trong một bài báo về học máy rất dễ bị liên kết sai sang huyền thoại bóng rổ Jordan.
- **EL xuyên ngôn ngữ (Cross-lingual EL).** Ánh xạ các biểu thức trong văn bản tiếng Việt sang thực thể trên Wikipedia tiếng Anh. Đòi hỏi mô hình nhúng đa ngôn ngữ hoặc bước dịch thuật ngữ cảnh.
- **Sự lỗi thời của cơ sở tri thức (KB staleness).** Các công ty mới thành lập, sự kiện mới diễn ra sẽ không có mặt trong bản dump Wikipedia của năm ngoái. Hệ thống sản xuất bắt buộc phải có chu kỳ cập nhật tri thức định kỳ.

## Sử dụng trong thực tế

Ngăn xếp công nghệ năm 2026:

| Tình huống | Lựa chọn đề xuất |
|------------|-------------------|
| Mục đích tổng quát tiếng Anh + Wikipedia | BLINK hoặc REL |
| Đa ngôn ngữ, KB là Wikipedia | mGENRE |
| Tích hợp LLM, lượng đề cập thấp/ngày | Prompt Claude/GPT-4 kèm danh sách ứng viên + ràng buộc JSON |
| KB chuyên biệt (Y tế, Pháp lý) | BERT tùy biến kết hợp truy xuất tri thức + tinh chỉnh trên tập dữ liệu chuẩn miền AIDA |
| Yêu cầu độ trễ cực thấp | Chỉ dùng xác suất tiên nghiệm khớp chính xác (Milne-Witten baseline) |
| Nghiên cứu đỉnh cao (Research SOTA) | GENRE / ExtEnD / LLM-EL sinh trực tiếp |

Mô hình chuẩn triển khai sản xuất năm 2026: NER → Giải quyết đồng tham chiếu (coref) → Liên kết thực thể (EL) trên từng biểu thức → Hợp nhất cụm đồng tham chiếu thành một thực thể chuẩn tắc duy nhất. Đầu ra: Một ID cơ sở tri thức cho mỗi thực thể trong tài liệu, thay vì xử lý vụn vặt từng lượt đề cập riêng rẽ.

## Đóng gói sản phẩm

Lưu tệp dưới dạng `outputs/skill-entity-linker.md`:

```markdown
---
name: entity-linker
description: Design an entity linking pipeline — KB, candidate generator, disambiguator, evaluation.
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

Given a use case (domain KB, language, volume, latency budget), output:

1. Knowledge base. Wikidata / Wikipedia / custom KB. Version date. Refresh cadence.
2. Candidate generator. Alias-index, embedding, or hybrid. Target mention recall @ K.
3. Disambiguator. Prior + context, embedding-based, generative, or LLM-prompted.
4. NIL strategy. Threshold on top score, classifier, or explicit NIL candidate.
5. Evaluation. Mention recall @ 30, top-1 accuracy, NIL-detection F1 on held-out set.

Refuse any EL pipeline without a mention-recall baseline (you cannot evaluate a disambiguator without knowing candidate gen surfaced the right entity). Refuse any pipeline using LLM-prompted EL without constrained output to valid KB ids. Flag systems where popularity bias affects minority entities (e.g. name-clashes) without domain fine-tuning.
```

## Bài tập

1. **Dễ.** Hiện thực hóa bộ khử nhập nhằng prior+context trong `code/main.py` trên 10 biểu thức đa nghĩa (Paris, Jordan, Apple). Tự gán nhãn thực thể chính xác. Đo lường tỷ lệ chính xác.
2. **Trung bình.** Nhúng 50 biểu thức đa nghĩa bằng một sentence transformer. Nhúng phần văn bản mô tả của từng ứng viên. So sánh kết quả khử nhập nhằng bằng vector nhúng với phương pháp so khớp từ vựng Jaccard.
3. **Khó.** Xây dựng một KB chuyên biệt gồm 1.000 thực thể (ví dụ: nhân viên + sản phẩm trong doanh nghiệp của bạn). Xây dựng trọn vẹn quy trình NER + EL từ đầu đến cuối. Đo lường độ chính xác và độ phủ trên 100 câu kiểm thử độc lập.

## Thuật ngữ then chốt

| Thuật ngữ | Cách gọi thông thường | Bản chất kỹ thuật thực sự |
|-----------|------------------------|---------------------------|
| Liên kết thực thể (EL) | Trỏ link tới Wikipedia | Ánh xạ một biểu thức đề cập về một mục thực thể duy nhất trong KB. |
| Sinh ứng viên (Candidate generation) | Có thể là những ai? | Trả về một danh sách rút gọn các thực thể KB khả dĩ cho một biểu thức. |
| Khử nhập nhằng (Disambiguation) | Chọn thực thể đúng | Chấm điểm các ứng viên dựa trên ngữ cảnh và chọn ứng viên chiến thắng. |
| Chỉ mục bí danh (Alias index) | Bảng tra cứu tên | Ánh xạ từ bề mặt từ ngữ sang danh sách các thực thể ứng viên. |
| NIL | Không có trong KB | Dự đoán tường minh rằng không có thực thể nào trong KB phù hợp. |
| Cơ sở tri thức (KB) | Knowledge base | Wikidata, Wikipedia, DBpedia hoặc cơ sở tri thức nội bộ. |
| AIDA-CoNLL | Bộ đo chuẩn quốc tế | Tập 1.393 bài báo Reuters kèm nhãn liên kết thực thể chuẩn xác. |

## Đọc thêm

- [Milne, Witten (2008). Learning to Link with Wikipedia](https://researchcommons.waikato.ac.nz/entities/publication/b9a0b520-abc5-47c5-a86a-da6c579893ab) — bài báo kinh điển về phương pháp prior+context.
- [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) — kiến trúc nhúng nền tảng phổ biến nhất.
- [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) — liên kết thực thể dạng mô hình sinh với giải mã có ràng buộc.
- [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) — bài báo công bố tập dữ liệu chuẩn AIDA.
- [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) — ngăn xếp mã nguồn mở trong môi trường sản xuất.

