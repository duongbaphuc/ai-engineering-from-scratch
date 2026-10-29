# Giải quyết đồng tham chiếu (Coreference Resolution)

> "Cô ấy gọi cho anh ấy. Anh ấy không bắt máy. Vị bác sĩ lúc đó đang bận ăn trưa." Ba cụm từ tham chiếu đến hai người nhưng không có ai được gọi đích danh bằng tên thật. Kỹ thuật giải quyết đồng tham chiếu (coreference resolution) xác định chính xác ai là ai.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Giai đoạn 5 · 06 (Nhận dạng thực thể có tên - NER), Giai đoạn 5 · 07 (Gán nhãn từ loại & Phân tích cú pháp)
**Time:** ~60 phút

## Vấn đề

Trích xuất mọi lượt đề cập đến Apple Inc. từ một bài báo dài 300 từ. Việc này rất dễ dàng khi bài báo ghi rõ "Apple". Nhưng sẽ cực kỳ khó khăn khi bài viết dùng các từ ngữ như "công ty này", "họ", "gã khổng lồ công nghệ vùng Cupertino", hay "doanh nghiệp của Steve Jobs". Nếu không liên kết được các biểu thức đề cập (mentions) này về cùng một thực thể, đường ống trích xuất thực thể (NER) của bạn sẽ bỏ sót từ 60-80% số lượt xuất hiện.

Giải quyết đồng tham chiếu (coreference resolution) liên kết mọi biểu thức ngôn ngữ cùng chỉ về một thực thể trong thế giới thực vào một cụm duy nhất (cluster). Đây là mắt xích kết nối giữa xử lý NLP cấp bề mặt (NER, phân tích cú pháp) và các tác vụ ngữ nghĩa hạ nguồn (trích xuất thông tin - IE, hỏi đáp - QA, tóm tắt văn bản, đồ thị tri thức - KG).

Tầm quan trọng trong bức tranh công nghệ năm 2026:

- Tóm tắt văn bản (Summarization): "Vị CEO thông báo..." so với "Tim Cook thông báo..." — bản tóm tắt chất lượng cần phải nêu đích danh vị CEO đó là ai.
- Hỏi đáp (Question answering): "Cô ấy đã gọi cho ai?" đòi hỏi hệ thống phải giải mã được "cô ấy" là người nào.
- Trích xuất thông tin (Information extraction): Một đồ thị tri thức ghi nhận "PER1 thành lập Apple" và "Jobs thành lập Apple" thành hai nút riêng biệt là hoàn toàn sai.
- Trích xuất thông tin đa tài liệu (Multi-document IE): Hợp nhất các biểu thức đề cập xuyên suốt nhiều bài báo khác nhau viết về cùng một sự kiện (cross-document coreference).

## Khái niệm

![Gom cụm đồng tham chiếu: biểu thức tham chiếu → thực thể](../assets/coref.svg)

**Nhiệm vụ.** Đầu vào: một tài liệu văn bản. Đầu ra: một tập hợp các cụm biểu thức đề cập (spans/mentions) sao cho mỗi cụm trỏ tới một thực thể duy nhất ngoài đời thực.

**Các loại biểu thức đề cập (Mention types).**

- **Thực thể có tên (Named entity):** "Tim Cook"
- **Danh ngữ miêu tả (Nominal):** "vị CEO" (the CEO), "công ty" (the company)
- **Đại từ (Pronominal):** "ông ấy" (he), "bà ấy" (she), "họ" (they), "nó" (it)
- **Đồng vị ngữ (Appositive):** "Tim Cook, CEO của Apple,"

**Các kiến trúc thuật toán.**

1. **Dựa trên tập luật (Rule-based - Hobbs, 1978).** Giải quyết đại từ dựa trên cây phân tích cú pháp (syntactic tree) bằng các luật ngữ pháp hình thức. Là một baseline vững chắc và khó bị đánh bại đối với các đại từ thông dụng.
2. **Bộ phân loại cặp đề cập (Mention-pair classifier).** Với mỗi cặp biểu thức $(m_i, m_j)$, mô hình dự đoán liệu chúng có đồng tham chiếu hay không. Sau đó gom cụm bằng bao đóng bắc cầu (transitive closure). Tiêu chuẩn phổ biến trước năm 2016.
3. **Xếp hạng biểu thức (Mention-ranking).** Với mỗi biểu thức, xếp hạng các tiền ngữ ứng viên (candidate antecedents, bao gồm cả lựa chọn "không có tiền ngữ"). Chọn ứng viên có điểm cao nhất.
4. **Đầu-cuối dựa trên đoạn ngữ (Span-based end-to-end - Lee et al., 2017).** Bộ mã hóa Transformer. Liệt kê toàn bộ các đoạn ứng viên (candidate spans) tới một giới hạn độ dài. Dự đoán điểm cho từng biểu thức. Dự đoán xác suất tiền ngữ cho mỗi đoạn. Gom cụm tham lam (greedy clustering). Đây là tiêu chuẩn hiện đại mặc định.
5. **Mô hình sinh (Generative - 2024+).** Đưa câu lệnh prompt cho LLM: "Hãy liệt kê mọi đại từ trong đoạn văn này cùng tiền ngữ tương ứng của nó." Hoạt động tốt trên các văn bản ngắn, nhưng gặp khó khăn trên tài liệu dài hoặc các thực thể hiếm.

**Chỉ số đánh giá.** Sử dụng 5 chỉ số đo lường tiêu chuẩn (MUC, B³, CEAF, BLANC, LEA) vì không có một chỉ số đơn lẻ nào phản ánh trọn vẹn chất lượng gom cụm. Điểm CoNLL F1 được tính bằng trung bình cộng của 3 chỉ số đầu tiên (MUC, B³, CEAF). Hiệu năng tiên tiến (SOTA) năm 2026 trên tập CoNLL-2012 đạt khoảng ~83 F1.

**Các trường hợp nan giải kinh điển.**

- Danh ngữ xác định (definite descriptions) trỏ về các thực thể đã được giới thiệu từ nhiều trang trước đó.
- Bắc cầu hồi chỉ (Bridging anaphora): "các bánh xe" → ngầm chỉ chiếc xe ô tô đã được nhắc đến trước đó.
- Khuyết đại từ (Zero anaphora) trong các ngôn ngữ lược bỏ chủ ngữ như tiếng Việt, tiếng Trung, tiếng Nhật.
- Khứ chỉ / Hồi chỉ xuôi (Cataphora - đại từ xuất hiện trước thực thể được nhắc tới): "Khi **cô ấy** bước vào, Mary mỉm cười."

```figure
coref-links
```

## Thực hành xây dựng

### Bước 1: Mô hình đồng tham chiếu nơ-ron tiền huấn luyện (spaCy-experimental)

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # mô hình thử nghiệm
doc = nlp("Apple announced new products. The company said it would ship them soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

Trên tài liệu dài, kết quả sẽ có dạng tương tự như:
- Cụm 1: [Apple, The company, it]
- Cụm 2: [new products, them]

### Bước 2: Bộ giải quyết đại từ dựa trên luật (Mục đích sư phạm)

Xem chi tiết trong `code/main.py` để nắm bắt bản hiện thực chỉ dùng thư viện chuẩn Python (stdlib-only):

1. Trích xuất các biểu thức: thực thể có tên (các cụm viết hoa), đại từ (tra cứu từ điển), danh ngữ xác định ("the X").
2. Với mỗi đại từ, quét lại $K$ biểu thức đề cập trước đó và chấm điểm dựa trên:
   - Sự tương thích về giống và số (heuristics)
   - Khoảng cách gần (biểu thức đứng càng gần càng được ưu tiên)
   - Vai trò cú pháp (ưu tiên các chủ ngữ)
3. Liên kết với tiền ngữ có điểm số cao nhất.

Phương pháp này không thể cạnh tranh về độ chính xác với mô hình nơ-ron, nhưng nó minh họa rõ nét không gian tìm kiếm và các quyết định logic mà một mô hình end-to-end phải giải quyết.

### Bước 3: Sử dụng LLM cho giải quyết đồng tham chiếu

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

Hai dạng lỗi phổ biến cần lưu ý: Thứ nhất, LLM dễ bị gộp nhầm (over-merge, ví dụ gom "anh ấy" và "cô ấy" chỉ hai người khác nhau vào cùng một cụm). Thứ hai, LLM thường âm thầm bỏ sót các lượt đề cập nằm sâu trong tài liệu dài. Hãy luôn kiểm tra lại bằng vị trí chỉ mục đoạn văn (span-offset checks).

### Bước 4: Đánh giá chất lượng

Script chuẩn `conll-2012` tính toán điểm số MUC, B³, CEAF-φ4 và báo cáo giá trị trung bình cộng. Để đánh giá nội bộ trong doanh nghiệp, hãy bắt đầu bằng độ chính xác (precision) và độ phủ (recall) ở cấp độ đoạn biểu thức trên tập kiểm thử được gán nhãn thủ công, sau đó bổ sung thêm điểm F1 liên kết đề cập.

## Cạm bẫy

- **Bùng nổ cụm đơn lẻ (Singleton explosion).** Một số hệ thống gán mỗi biểu thức thành một cụm riêng của chính nó. Chỉ số B³ sẽ chấm điểm khá nới tay cho trường hợp này, trong khi MUC sẽ phạt rất nặng. Hãy luôn đối chiếu cả ba chỉ số.
- **Đại từ trong ngữ cảnh dài.** Điểm F1 sụt giảm khoảng ~15 điểm trên các tài liệu vượt quá 2.000 token. Cần chia đoạn văn bản một cách cẩn trọng.
- **Giả định về giới tính.** Các luật cố định về giới tính sẽ thất bại trước các thực thể phi nhị nguyên, tên tổ chức doanh nghiệp hoặc loài vật. Hãy sử dụng mô hình học máy hoặc cơ chế chấm điểm trung tính.
- **LLM bị trôi dạt ngữ cảnh trên tài liệu dài.** Một lượt gọi API đơn lẻ không thể gom cụm nhất quán các biểu thức xuyên suốt 50+ đoạn văn bản. Cần áp dụng kỹ thuật cửa sổ trượt kết hợp thuật toán hợp nhất (sliding-window + merge).

## Sử dụng trong thực tế

Ngăn xếp công nghệ năm 2026:

| Tình huống | Lựa chọn đề xuất |
|------------|-------------------|
| Tiếng Anh, đơn tài liệu | `en_coreference_web_trf` (spaCy-experimental) hoặc AllenNLP neural coref |
| Đa ngôn ngữ | SpanBERT / XLM-R huấn luyện trên OntoNotes hoặc Multilingual CoNLL |
| Đồng tham chiếu sự kiện xuyên tài liệu | Các mô hình chuyên biệt end-to-end (SOTA 2025–2026) |
| Đường cơ sở LLM nhanh gọn | GPT-4o / Claude với prompt xuất cấu trúc JSON nghiêm ngặt |
| Hệ thống hội thoại production | Dự phòng dựa trên luật + mô hình nơ-ron chính + duyệt thủ công cho các khe then chốt |

Quy trình tích hợp chuẩn năm 2026: Chạy NER trước tiên, chạy giải quyết đồng tham chiếu (coref), sau đó hợp nhất các cụm đồng tham chiếu vào các thực thể NER. Các tác vụ hạ nguồn chỉ cần xử lý một thực thể đại diện cho mỗi cụm, thay vì phải xử lý rời rạc từng biểu thức đề cập.

## Đóng gói sản phẩm

Lưu tệp dưới dạng `outputs/skill-coref-picker.md`:

```markdown
---
name: coref-picker
description: Pick a coreference approach, evaluation plan, and integration strategy.
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

Given a use case (single-doc / multi-doc, domain, language), output:

1. Approach. Rule-based / neural span-based / LLM-prompted / hybrid. One-sentence reason.
2. Model. Named checkpoint if neural.
3. Integration. Order of operations: tokenize → NER → coref → downstream task.
4. Evaluation. CoNLL F1 (MUC + B³ + CEAF-φ4 average) on held-out set + manual cluster review on 20 documents.

Refuse LLM-only coref for documents over 2,000 tokens without sliding-window merge. Refuse any pipeline that runs coref without a mention-level precision-recall report. Flag gender-heuristic systems deployed in demographically diverse text.
```

## Bài tập

1. **Dễ.** Chạy bộ giải quyết đại từ dựa trên luật trong `code/main.py` trên 5 đoạn văn tự soạn thảo. Đo lường độ chính xác liên kết biểu thức so với nhãn chuẩn thực tế (ground truth).
2. **Trung bình.** Sử dụng mô hình coref nơ-ron tiền huấn luyện trên một bài báo thời sự. So sánh các cụm sinh ra với kết quả gán nhãn thủ công của bạn. Mô hình đã mắc lỗi ở những trường hợp nào?
3. **Khó.** Xây dựng một đường ống NER được tăng cường bởi coref: chạy NER trước, sau đó hợp nhất thông qua các cụm đồng tham chiếu. Đo lường mức độ mở rộng độ phủ thực thể (entity-coverage) so với chỉ dùng NER đơn thuần trên 100 bài báo.

## Thuật ngữ then chốt

| Thuật ngữ | Cách gọi thông thường | Bản chất kỹ thuật thực sự |
|-----------|------------------------|---------------------------|
| Biểu thức đề cập (Mention) | Một tham chiếu | Một đoạn văn bản đề cập đến một thực thể (tên riêng, đại từ, cụm danh từ). |
| Tiền ngữ (Antecedent) | Đối tượng mà "nó" trỏ tới | Biểu thức đề cập xuất hiện trước mà biểu thức phía sau đồng tham chiếu tới. |
| Cụm (Cluster) | Tập hợp các lượt đề cập | Tập hợp tất cả các biểu thức cùng trỏ về một thực thể duy nhất ngoài đời thực. |
| Hồi chỉ ngược (Anaphora) | Tham chiếu lùi | Biểu thức phía sau trỏ về đối tượng phía trước ("anh ấy" → "John"). |
| Hồi chỉ xuôi (Cataphora) | Tham chiếu tiến | Biểu thức phía trước trỏ về đối tượng xuất hiện phía sau ("Khi anh ấy tới, John..."). |
| Tham chiếu bắc cầu (Bridging) | Tham chiếu ngầm | "Tôi mua một chiếc xe hơi. Bánh xe bị xẹp." (bánh xe của CHÍNH chiếc xe đó). |
| CoNLL F1 | Điểm chuẩn trên bảng xếp hạng | Điểm F1 trung bình cộng của ba phương pháp đo MUC, B³, và CEAF-φ4. |

## Đọc thêm

- [Jurafsky & Martin, SLP3 Ch. 26 — Coreference Resolution and Entity Linking](https://web.stanford.edu/~jurafsky/slp3/26.pdf) — chương giáo trình kinh điển chuẩn mực.
- [Lee et al. (2017). End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045) — bài báo nền tảng về mô hình dựa trên span end-to-end.
- [Joshi et al. (2020). SpanBERT](https://arxiv.org/abs/1907.10529) — phương pháp tiền huấn luyện giúp cải thiện vượt bậc tác vụ coref.
- [Pradhan et al. (2012). CoNLL-2012 Shared Task](https://aclanthology.org/W12-4501/) — tập dữ liệu đo chuẩn quốc tế.
- [Hobbs (1978). Resolving Pronoun References](https://www.sciencedirect.com/science/article/pii/0024384178900064) — thuật toán kinh điển dựa trên luật ngữ pháp.

