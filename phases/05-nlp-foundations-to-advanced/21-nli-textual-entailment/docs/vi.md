# Suy diễn Ngôn ngữ Tự nhiên — Kéo theo Văn bản (Natural Language Inference — Textual Entailment)

> "t kéo theo h" có nghĩa là một người bình thường khi đọc t sẽ kết luận h là đúng. NLI là tác vụ dự đoán mối quan hệ kéo theo (entailment) / mâu thuẫn (contradiction) / trung tính (neutral). Thoạt nhìn có vẻ tẻ nhạt, nhưng lại là trụ cột chịu lực trong môi trường sản xuất.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 05 (Phân tích Cảm xúc), Phase 5 · 13 (Hệ thống Hỏi Đáp)
**Time:** ~60 phút

## Vấn đề đặt ra (The Problem)

Bạn vừa xây dựng một hệ thống tóm tắt văn bản. Nó tạo ra một bản tóm tắt. Làm thế nào bạn biết chắc chắn bản tóm tắt đó không chứa chi tiết ảo giác?

Bạn vừa xây dựng một chatbot. Nó trả lời "có." Làm thế nào bạn biết câu trả lời đó thực sự được chứng thực bởi đoạn văn bản đã truy xuất?

Bạn cần phân loại 10.000 bài báo tin tức theo chủ đề. Bạn không có bất kỳ nhãn huấn luyện nào. Liệu bạn có thể tái sử dụng một mô hình sẵn có?

Cả ba bài toán trên đều quy về **Suy diễn Ngôn ngữ Tự nhiên (Natural Language Inference - NLI)**. NLI đặt ra câu hỏi: Cho trước một tiền đề (premise) `t` và một giả thuyết (hypothesis) `h`, liệu `h` có được kéo theo bởi `t`, bị mâu thuẫn với `t`, hay là trung tính (không liên quan)?

- **Kiểm tra ảo giác (Hallucination check):** `t` = tài liệu nguồn, `h` = luận điểm trong tóm tắt. Không kéo theo = có ảo giác.
- **Hỏi đáp có căn cứ (Grounded QA):** `t` = đoạn văn truy xuất, `h` = câu trả lời sinh ra. Không kéo theo = câu trả lời bịa đặt.
- **Phân loại Zero-shot:** `t` = tài liệu, `h` = nhãn được diễn giải thành câu ("Văn bản này nói về thể thao"). Kéo theo = gán nhãn đó.

Một tác vụ duy nhất, ba ứng dụng sống còn trong sản xuất. Đó là lý do mọi khung đánh giá RAG hiện đại đều tích hợp một mô hình NLI ở tầng lõi bên dưới.

## Khái niệm (The Concept)

![NLI: phân loại 3 lớp, tiền đề so với giả thuyết](../assets/nli.svg)

**Ba nhãn phân loại:**

- **Kéo theo (Entailment).** `t` → `h`. "Con mèo đang ngủ trên tấm thảm" kéo theo "Có một con mèo."
- **Mâu thuẫn (Contradiction).** `t` → ¬`h`. "Con mèo đang ngủ trên tấm thảm" mâu thuẫn với "Không có con mèo nào ở đây cả."
- **Trung tính (Neutral).** Không thể suy diễn theo hướng nào. "Con mèo đang ngủ trên tấm thảm" là trung tính đối với "Con mèo đang bị đói."

**Không phải là kéo theo logic hình thức.** NLI là suy diễn ngôn ngữ *tự nhiên* — những gì mà một người đọc thông thường sẽ suy luận ra dựa trên lẽ thường, không phải logic toán học thuần túy. Câu "John dắt chó của anh ấy đi dạo" trong NLI kéo theo "John có một con chó", mặc dù trong logic vị từ bậc một nghiêm ngặt, bạn chỉ thừa nhận điều đó nếu có tiên đề rõ ràng về quyền sở hữu.

**Các tập ngữ liệu tiêu biểu:**

- **SNLI** (2015). 570.000 cặp câu được con người gán nhãn, tiền đề là chú thích hình ảnh. Miền ngữ cảnh hẹp.
- **MultiNLI** (2017). 433.000 cặp câu trải rộng trên 10 thể loại văn bản khác nhau. Ngữ liệu huấn luyện chuẩn mực năm 2026.
- **ANLI** (2019). Adversarial NLI. Các cặp câu được con người thiết kế có chủ đích nhằm "bẫy" và phá vỡ các mô hình hiện có. Độ khó cao hơn rất nhiều.
- **DocNLI, ConTRoL** (2020–2021). Tiền đề là các tài liệu dài trọn vẹn. Kiểm thử khả năng suy luận đa bước (multi-hop) và ngữ cảnh dài.

**Kiến trúc mạng nơ-ron.** Một bộ mã hóa Transformer (BERT, RoBERTa, DeBERTa) đọc đầu vào có định dạng `[CLS] premise [SEP] hypothesis [SEP]`. Biểu diễn của token `[CLS]` được đưa qua một tầng Softmax phân loại 3 lớp. Huấn luyện trên MNLI, đánh giá trên các tập benchmark tách biệt, đạt độ chính xác trên 90%+ trên các cặp câu cùng phân phối.

**Phân loại Zero-shot qua NLI.** Cho trước một tài liệu và danh sách các nhãn ứng viên, biến mỗi nhãn thành một câu giả thuyết ("Văn bản này nói về thể thao"). Tính toán xác suất kéo theo (entailment probability) cho từng giả thuyết. Chọn nhãn có xác suất cao nhất. Đây chính là cơ chế vận hành bên dưới pipeline `zero-shot-classification` của Hugging Face.

```figure
nli-router
```

## Xây dựng thực tế (Build It)

### Bước 1: Chạy mô hình NLI huấn luyện sẵn

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # trả về tất cả các nhãn; thay thế return_all_scores=True đã lỗi thời

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

Trong môi trường sản xuất, `facebook/bart-large-mnli` và `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` là các lựa chọn mã nguồn mở mặc định hàng đầu. DeBERTa-v3 thường đứng đầu các bảng xếp hạng uy tín.

### Bước 2: Phân loại văn bản Zero-Shot

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

Khuôn mẫu mặc định là "This example is about {label}." Bạn có thể tùy biến mẫu câu thông qua tham số `hypothesis_template`. Không cần bất kỳ dữ liệu huấn luyện nào. Không cần fine-tune. Hoạt động hiệu quả ngay khi cài đặt.

### Bước 3: Kiểm tra độ trung thực cho RAG (Faithfulness check)

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

Đây chính là cơ chế cốt lõi của chỉ số đo lường độ trung thực trong thư viện RAGAS: Tách câu trả lời sinh ra thành các khẳng định nguyên tử (atomic claims). Kiểm tra từng khẳng định độc lập đối chiếu với ngữ cảnh đã truy xuất. Báo cáo tỷ lệ phần trăm các khẳng định đạt quan hệ kéo theo (entailment).

### Bước 4: Bộ phân loại NLI thủ công (Khái niệm)

Xem `code/main.py` để tìm hiểu bản cài đặt đồ chơi thuần thư viện chuẩn: tiền đề và giả thuyết được so sánh dựa trên sự trùng lặp từ vựng kết hợp phát hiện phủ định. Tuy không thể cạnh tranh với Transformer, nó làm sáng tỏ bản chất của bài toán: hai văn bản đầu vào, nhãn phân loại 3 lớp đầu ra, hàm mất mát là cross-entropy trên tập `{entail, contradict, neutral}`.

## Các cạm bẫy cần lưu ý

- **Đường tắt chỉ dựa vào giả thuyết (Hypothesis-only shortcuts).** Các mô hình có thể đoán trúng nhãn chỉ dựa vào câu giả thuyết đơn lẻ với độ chính xác ~60% trên SNLI vì các từ như "not", "nobody", "never" thường có tương quan ngẫu nhiên cao với nhãn mâu thuẫn (contradiction). Đây là baseline quan trọng để phát hiện hiện tượng rò rỉ nhãn (label leakage).
- **Quy tắc phỏng đoán trùng lặp từ vựng (Lexical overlap heuristic).** Quy tắc ngây thơ "mọi chuỗi con đều được kéo theo" vượt qua SNLI nhưng thất bại thảm hại trên các bộ dữ liệu đối kháng HANS/ANLI. Luôn đánh giá trên các benchmark đối kháng.
- **Suy thoái hiệu năng trên tài liệu dài.** Các mô hình NLI cấp độ câu đơn sẽ bị tụt hơn 20 điểm F1 khi tiền đề là một tài liệu dài hoàn chỉnh. Hãy dùng các mô hình huấn luyện chuyên cho DocNLI khi làm việc với ngữ cảnh dài.
- **Độ nhạy cảm của khuôn mẫu Zero-shot.** Thay đổi mẫu câu từ "This example is about {label}" sang "The topic is {label}" hoặc chỉ trơ trọi "{label}" có thể làm thay đổi độ chính xác hơn 10 điểm phần trăm. Hãy chủ động tinh chỉnh và kiểm nghiệm khuôn mẫu.
- **Lệch miền dữ liệu (Domain mismatch).** MultiNLI được huấn luyện trên tiếng Anh thông dụng. Văn bản pháp lý, y tế và khoa học đòi hỏi các mô hình NLI chuyên biệt (ví dụ: SciNLI, MedNLI).

## Sử dụng trong thực tế (Use It)

Ngăn xếp công nghệ năm 2026:

| Trường hợp sử dụng | Mô hình khuyến nghị |
|-------------------|---------------------|
| NLI đa mục đích chất lượng cao | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| Tốc độ cao / Thiết bị biên | `cross-encoder/nli-deberta-v3-base` |
| Phân loại Zero-shot gọn nhẹ | `facebook/bart-large-mnli` |
| NLI cấp độ tài liệu dài | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| Đa ngôn ngữ | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| Phát hiện ảo giác trong RAG | Tầng NLI tích hợp bên trong RAGAS / DeepEval |

Quy luật tổng quát năm 2026: NLI là "băng keo vạn năng" của bài toán thấu hiểu văn bản. Bất cứ khi nào bạn cần kiểm tra "A có chứng minh cho B hay không?" hoặc "A có mâu thuẫn với B hay không?" — hãy ưu tiên dùng mô hình NLI trước khi quyết định tốn chi phí gọi thêm một lượt LLM tạo sinh.

## Đóng gói sản phẩm (Ship It)

Lưu thành `outputs/skill-nli-picker.md`:

```markdown
---
name: nli-picker
description: Lựa chọn mô hình NLI, khuôn mẫu nhãn và thiết lập đánh giá cho bài toán phân loại / kiểm tra độ trung thực / zero-shot.
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

Dựa trên trường hợp sử dụng (kiểm tra độ trung thực faithfulness, phân loại zero-shot, suy luận cấp độ tài liệu), bạn đưa ra:

1. Mô hình: Tên checkpoint NLI cụ thể. Nêu rõ lý do gắn liền với miền dữ liệu, độ dài văn bản và ngôn ngữ.
2. Khuôn mẫu Template (nếu dùng zero-shot): Mẫu câu diễn giải nhãn thành giả thuyết. Cung cấp ví dụ trực quan.
3. Ngưỡng phân loại (Threshold): Điểm cắt xác suất kéo theo để đưa ra quyết định nhị phân. Giải thích dựa trên việc chuẩn hóa xác suất (calibration).
4. Kế hoạch đánh giá: Độ chính xác trên tập dữ liệu gán nhãn riêng biệt, baseline chỉ dùng giả thuyết (hypothesis-only), tập kiểm thử đối kháng (adversarial subset).

Từ chối phát hành hệ thống phân loại zero-shot mà thiếu bài kiểm tra độ tin cậy cơ bản trên 100 ví dụ gán nhãn thực tế. Từ chối sử dụng mô hình NLI cấp độ câu đơn trên các tiền đề có độ dài cả tài liệu. Cảnh báo rõ ràng rằng NLI chỉ giúp giảm thiểu ảo giác, không thể triệt tiêu hoàn toàn 100% ảo giác.
```

## Bài tập (Exercises)

1. **Dễ.** Chạy mô hình `facebook/bart-large-mnli` trên 20 bộ ba (tiền đề, giả thuyết, nhãn) tự tạo bao quát cả ba lớp. Đo độ chính xác. Bổ sung các "cạm bẫy trùng lặp từ vựng" ("Tôi không ăn chiếc bánh" so với "Tôi đã ăn chiếc bánh") và quan sát xem mô hình có bị đánh lừa hay không.
2. **Trung bình.** So sánh khuôn mẫu zero-shot `"This text is about {label}"` với `"The topic is {label}"` và `"{label}"` trên 100 tiêu đề tin tức AG News. Báo cáo mức độ dao động về độ chính xác giữa các mẫu câu.
3. **Khó.** Xây dựng bộ kiểm tra độ trung thực cho RAG: phân rã khẳng định nguyên tử + NLI cho từng khẳng định. Đánh giá trên 50 câu trả lời RAG kèm ngữ cảnh chuẩn. Đo lường tỷ lệ dương tính giả (false-positive) và âm tính giả (false-negative) so với nhãn đánh giá thủ công của con người.

## Thuật ngữ cốt lõi (Key Terms)

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|-----------|-------------------|-----------------|
| NLI | Suy diễn ngôn ngữ tự nhiên | Phân loại 3 lớp mối quan hệ giữa tiền đề và giả thuyết. |
| RTE | Nhận dạng kéo theo văn bản | Tên gọi cổ điển của NLI; cùng một tác vụ bản chất. |
| Kéo theo (Entailment) | "t ngụ ý h" | Một người đọc thông thường sẽ kết luận h là đúng khi biết t. |
| Mâu thuẫn (Contradiction) | "t bác bỏ h" | Một người đọc thông thường sẽ kết luận h là sai khi biết t. |
| Trung tính (Neutral) | "chưa xác định" | Không thể rút ra kết luận khẳng định hay phủ định h từ t. |
| Phân loại Zero-shot | NLI đóng vai bộ phân loại | Diễn giải các nhãn thành các câu giả thuyết, chọn nhãn có xác suất kéo theo cao nhất. |
| Độ trung thực (Faithfulness) | Câu trả lời có căn cứ không? | Chạy NLI trên cặp (ngữ cảnh đã truy xuất, câu trả lời sinh ra). |

## Đọc thêm (Further Reading)

- [Bowman et al. (2015). A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326) — bài báo gốc giới thiệu bộ dữ liệu SNLI.
- [Williams, Nangia, Bowman (2017). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426) — bài báo giới thiệu MultiNLI.
- [Nie et al. (2019). Adversarial NLI](https://arxiv.org/abs/1910.14599) — benchmark ANLI đối kháng.
- [Yin, Hay, Roth (2019). Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161) — công trình tiên phong biến NLI thành bộ phân loại Zero-shot.
- [He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) — cỗ máy nơ-ron nền tảng cho NLI hiện đại năm 2026.

