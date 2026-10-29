# Tóm tắt Văn bản (Text Summarization)

> Các hệ thống trích xuất cho bạn biết tài liệu đã nói những gì. Các hệ thống trừu tượng cho bạn biết tác giả thực sự có ý định gì. Hai tác vụ khác biệt, hai cạm bẫy hoàn toàn khác nhau.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 11 (Dịch máy)
**Time:** ~75 phút

## Vấn đề đặt ra (The Problem)

Một bài báo dài 2.000 từ xuất hiện trên bảng tin của bạn. Bạn cần một bản tóm tắt ngắn gọn 120 từ để nắm bắt nội dung chính. Bạn có hai lựa chọn: hoặc chọn ra ba câu quan trọng nhất từ bài viết (tóm tắt trích xuất - extractive), hoặc viết lại nội dung bằng lời văn của chính mình (tóm tắt trừu tượng - abstractive). Cả hai đều được gọi là tóm tắt văn bản, nhưng chúng là hai bài toán hoàn toàn khác nhau.

Tóm tắt trích xuất là một bài toán xếp hạng (ranking). Chấm điểm từng câu trong tài liệu và trả về top-`k` câu có điểm số cao nhất. Đầu ra luôn đúng ngữ pháp vì nó được trích xuất nguyên văn từng chữ. Rủi ro lớn nhất là bỏ sót các nội dung quan trọng phân tán rải rác trên toàn bộ bài viết.

Tóm tắt trừu tượng là một bài toán tạo sinh (generation). Một mô hình Transformer sinh ra văn bản mới dựa trên điều kiện của văn bản đầu vào. Đầu ra rất trôi chảy, súc tích và có tính cô đọng cao, nhưng có nguy cơ tạo ra các chi tiết ảo giác (hallucination) không hề có trong nguồn. Rủi ro lớn nhất là sự bịa đặt một cách đầy tự tin.

Bài học này xây dựng cả hai phương pháp, đồng thời chỉ rõ phương thức lỗi cố hữu mà mỗi phương pháp mang theo.

## Khái niệm (The Concept)

![TextRank trích xuất so với Transformer tóm tắt trừu tượng](../assets/summarization.svg)

**Tóm tắt trích xuất (Extractive).** Coi bài viết như một đồ thị trong đó các nút là các câu và các cạnh đại diện cho độ tương đồng giữa các câu. Chạy thuật toán PageRank (hoặc thuật toán tương tự) trên đồ thị để chấm điểm các câu dựa trên mức độ kết nối của chúng với phần còn lại của văn bản. Những câu có điểm số cao nhất sẽ tạo thành bản tóm tắt. Bản cài đặt kinh điển là **TextRank** (Mihalcea và Tarau, 2004).

**Tóm tắt trừu tượng (Abstractive).** Tinh chỉnh (fine-tune) một mô hình Transformer encoder-decoder (BART, T5, Pegasus) trên các cặp văn bản - bản tóm tắt. Khi suy luận, mô hình đọc tài liệu và sinh ra bản tóm tắt từng token một thông qua cơ chế chú ý chéo (cross-attention). Đặc biệt, Pegasus sử dụng mục tiêu tiền huấn luyện khuyết câu (gap-sentence pretraining objective) giúp nó đạt khả năng tóm tắt xuất sắc mà không cần nhiều dữ liệu tinh chỉnh.

Đánh giá bằng **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation). ROUGE-1 và ROUGE-2 chấm điểm sự trùng lặp unigram và bigram. ROUGE-L chấm điểm chuỗi con chung dài nhất (longest common subsequence). Điểm càng cao càng tốt, nhưng 40 ROUGE-L đã được coi là "tốt" và 50 là "xuất sắc". Mọi bài báo khoa học đều báo cáo cả ba chỉ số này. Hãy sử dụng thư viện `rouge-score`.

```figure
summarize-collapse
```

## Xây dựng thực tế (Build It)

### Bước 1: TextRank (Tóm tắt trích xuất)

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

Hai điểm cần lưu ý: Hàm tương đồng sử dụng mức độ trùng lặp từ được chuẩn hóa theo logarit, đây là biến thể TextRank nguyên bản (cosine của các vector TF-IDF cũng hoạt động rất tốt). Hệ số tắt dần (damping factor) 0.85 và số vòng lặp là các giá trị mặc định chuẩn mực từ PageRank.

### Bước 2: Tóm tắt trừu tượng với BART

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(long news article text)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN được tinh chỉnh trên tập ngữ liệu tin tức CNN/DailyMail. Nó tạo ra các bản tóm tắt theo phong cách báo chí rất chuẩn xác ngay khi sử dụng. Đối với các miền dữ liệu khác (bài báo khoa học, hội thoại, pháp lý), hãy dùng checkpoint Pegasus tương ứng hoặc tinh chỉnh trên dữ liệu mục tiêu của bạn.

### Bước 3: Đánh giá bằng ROUGE

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

Luôn bật tùy chọn gốc từ `use_stemmer=True`. Nếu không có nó, các từ như "running" và "run" sẽ bị coi là hai từ hoàn toàn khác nhau khiến ROUGE bị tụt điểm sai lệch.

### Vượt qua giới hạn của ROUGE (Đánh giá tóm tắt năm 2026)

ROUGE đã giữ vị trí thống trị trong đánh giá tóm tắt suốt hai mươi năm, nhưng nó không còn đủ nếu đứng độc lập vào năm 2026. Một phân tích tổng hợp quy mô lớn trên các công trình NLG cho thấy:

- **BERTScore** (độ tương đồng vector nhúng theo ngữ cảnh) đã phát triển mạnh mẽ từ năm 2023 và hiện được báo cáo song song với ROUGE trong hầu hết các công bố tóm tắt.
- **BARTScore** coi việc đánh giá như một bài toán tạo sinh: chấm điểm bản tóm tắt dựa trên xác suất mà một mô hình BART huấn luyện sẵn gán cho nó khi biết văn bản nguồn.
- **MoverScore** (Khoảng cách Earth Mover's Distance trên các vector nhúng theo ngữ cảnh) chiếm vị trí dẫn đầu trong các bài đo benchmark năm 2025 nhờ khả năng bắt trọn sự trùng lặp ngữ nghĩa vượt trội hơn ROUGE.
- **FactCC** và **đánh giá độ trung thực dựa trên hỏi đáp (QA-based faithfulness)** rất phổ biến giai đoạn 2021-2023, nay thường được thay thế bằng **G-Eval** (một chuỗi prompt GPT-4 chấm điểm tính mạch lạc, nhất quán, trôi chảy và mức độ liên quan kết hợp lập luận chuỗi suy nghĩ - chain-of-thought).
- **G-Eval** và các phương pháp dùng LLM làm giám khảo đạt độ tương đồng ~80% với đánh giá của con người khi tiêu chí chấm điểm được thiết kế tỉ mỉ.

Khuyến nghị trong sản xuất: Báo cáo ROUGE-L để đối chiếu với các nghiên cứu trước, BERTScore cho mức độ tương đồng ngữ nghĩa, và G-Eval cho tính mạch lạc và độ xác thực. Chuẩn hóa các chỉ số này trên 50-100 bản tóm tắt có nhãn người thẩm định.

### Bước 4: Bài toán độ xác thực dữ kiện (The Factuality Problem)

Bản tóm tắt trừu tượng rất dễ tạo ảo giác. Ngược lại, tóm tắt trích xuất có nguy cơ tạo ảo giác thấp hơn rất nhiều do câu chữ được lấy nguyên văn từ văn bản nguồn, mặc dù chúng vẫn có thể gây hiểu lầm nếu câu nguồn bị tách rời khỏi ngữ cảnh, lỗi thời, hoặc bị trích dẫn sai trật tự logic. Đây là lý do lớn nhất khiến các hệ thống sản xuất thực tế vẫn ưu tiên phương pháp trích xuất cho các nội dung đòi hỏi tuân thủ pháp lý nghiêm ngặt.

Các dạng ảo giác điển hình:

- **Tráo đổi thực thể (Entity swap).** Nguồn nói "John Smith." Bản tóm tắt lại viết "John Brown."
- **Trôi dạt con số (Number drift).** Nguồn ghi "25,000." Bản tóm tắt ghi "25 million."
- **Đảo ngược cực tính (Polarity flip).** Nguồn ghi "rejected the offer" (từ chối lời đề nghị). Bản tóm tắt ghi "accepted the offer" (chấp nhận lời đề nghị).
- **Bịa đặt dữ kiện (Fact invention).** Nguồn không hề nhắc đến CEO. Bản tóm tắt khẳng định CEO đã phê duyệt.

Các phương pháp đánh giá xác thực hiệu quả:

- **FactCC.** Một bộ phân loại nhị phân được huấn luyện dựa trên quan hệ kéo theo (entailment) giữa câu nguồn và câu tóm tắt. Dự đoán đúng dữ kiện / sai dữ kiện.
- **Xác thực dựa trên hỏi đáp (QA-based factuality).** Đặt cho mô hình QA các câu hỏi mà câu trả lời nằm trong nguồn. Nếu bản tóm tắt đưa ra câu trả lời khác biệt, hãy gắn cờ cảnh báo.
- **F1 cấp độ thực thể (Entity-level F1).** So sánh các thực thể có tên trong văn bản nguồn so với bản tóm tắt. Các thực thể chỉ xuất hiện trong bản tóm tắt mà không có trong nguồn là đối tượng đáng ngờ.

Đối với bất kỳ hệ thống nào hướng tới người dùng mà tính chính xác của dữ kiện mang tính sống còn (tin tức, y tế, pháp lý, tài chính), phương pháp trích xuất luôn là lựa chọn mặc định an toàn hơn. Nếu dùng tóm tắt trừu tượng, bắt buộc phải có một cổng kiểm tra tính xác thực (factuality check) trong vòng lặp xử lý.

## Sử dụng trong thực tế (Use It)

Ngăn xếp công nghệ năm 2026:

| Trường hợp sử dụng | Đề xuất |
|-------------------|---------|
| Tin tức, tóm tắt 3-5 câu, tiếng Anh | `facebook/bart-large-cnn` |
| Bài báo nghiên cứu khoa học | `google/pegasus-pubmed` hoặc mô hình T5 đã tinh chỉnh |
| Đa tài liệu, văn bản dài | Bất kỳ LLM nào có ngữ cảnh 32k+ kèm prompt chuyên biệt |
| Tóm tắt hội thoại đối thoại | `philschmid/bart-large-cnn-samsum` |
| Trích xuất, nguy cơ ảo giác bằng 0 theo thiết kế | TextRank hoặc thuật toán LSA / LexRank của thư viện `sumy` |

Năm 2026, các LLM với ngữ cảnh dài thường vượt trội hơn các mô hình chuyên dụng khi tài nguyên tính toán không phải là rào cản. Đánh đổi nằm ở chi phí và tính tái lập; các mô hình chuyên dụng mang lại đầu ra ổn định và nhất quán hơn.

## Đóng gói sản phẩm (Ship It)

Lưu thành `outputs/skill-summary-picker.md`:

```markdown
---
name: summary-picker
description: Lựa chọn phương pháp trích xuất hay trừu tượng, thư viện cụ thể và giải pháp kiểm tra độ xác thực.
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

Dựa trên yêu cầu bài toán (loại tài liệu, yêu cầu tuân thủ pháp lý, độ dài, ngân sách tính toán), bạn đưa ra:

1. Phương pháp tiếp cận: Trích xuất (extractive) hay trừu tượng (abstractive). Giải thích lý do trong một câu.
2. Mô hình / thư viện khởi điểm: Nêu tên cụ thể. `sumy.TextRankSummarizer`, `facebook/bart-large-cnn`, `google/pegasus-pubmed`, hoặc prompt cho LLM.
3. Kế hoạch đánh giá: ROUGE-1, ROUGE-2, ROUGE-L (dùng rouge-score có bật stemming). Bổ sung kiểm tra độ xác thực (factuality check) nếu chọn phương pháp trừu tượng.
4. Một phương thức lỗi cần điều tra: Tráo đổi thực thể là lỗi phổ biến nhất trong tóm tắt tin tức trừu tượng; gắn cờ cảnh báo các mẫu mà thực thể nguồn không xuất hiện trong bản tóm tắt.

Từ chối đề xuất tóm tắt trừu tượng cho nội dung y tế, pháp lý, tài chính hoặc quy định nghiêm ngặt nếu thiếu cổng kiểm tra độ xác thực. Đánh dấu các đầu vào vượt quá cửa sổ ngữ cảnh của mô hình là cần sử dụng chiến lược tóm tắt phân đoạn map-reduce (thay vì chỉ cắt xén văn bản đơn thuần).
```

## Bài tập (Exercises)

1. **Dễ.** Chạy TextRank trên 5 bài báo tin tức. So sánh top-3 câu được chọn với bản tóm tắt tham chiếu chuẩn. Đo điểm ROUGE-L. Bạn sẽ thấy điểm số đạt khoảng 30-45 ROUGE-L trên các bài báo kiểu CNN/DailyMail.
2. **Trung bình.** Cài đặt bước kiểm tra độ xác thực cấp độ thực thể: trích xuất các thực thể có tên từ nguồn và bản tóm tắt (bằng spaCy), tính độ nhạy (recall) của thực thể nguồn trong tóm tắt và độ chính xác (precision) của thực thể tóm tắt so với nguồn. Precision cao và recall thấp nghĩa là bản tóm tắt an toàn nhưng ngắn gọn; precision thấp cảnh báo có thực thể bị ảo giác.
3. **Khó.** So sánh BART-large-CNN với một LLM (Claude hoặc GPT-4) trên 50 bài viết CNN/DailyMail. Báo cáo ROUGE-L, độ xác thực dữ kiện (thông qua F1 thực thể), và chi phí tài chính cho mỗi bản tóm tắt. Ghi nhận tài liệu các trường hợp mỗi bên chiếm ưu thế.

## Thuật ngữ cốt lõi (Key Terms)

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|-----------|-------------------|-----------------|
| Trích xuất (Extractive) | Chọn câu | Trả về các câu nguyên văn trích từ nguồn. Không bao giờ tự tạo ảo giác. |
| Trừu tượng (Abstractive) | Viết lại | Tự sinh ra văn bản mới dựa trên điều kiện của nguồn. Có nguy cơ tạo ảo giác. |
| ROUGE | Điểm tóm tắt | Mức độ trùng lặp n-gram / LCS giữa đầu ra của hệ thống và câu tham chiếu. |
| TextRank | Trích xuất dựa trên đồ thị | Thuật toán PageRank chạy trên đồ thị tương đồng câu. |
| Tính xác thực (Factuality) | Có đúng thực tế không | Liệu các luận điểm trong bản tóm tắt có được văn bản nguồn chứng thực hay không. |
| Ảo giác (Hallucination) | Nội dung tự bịa | Nội dung xuất hiện trong bản tóm tắt mà không hề có căn cứ trong nguồn. |

## Đọc thêm (Further Reading)

- [Mihalcea and Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) — bài báo kinh điển về tóm tắt trích xuất.
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) — bài báo gốc về kiến trúc BART.
- [Zhang et al. (2019). PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777) — bài báo về Pegasus và mục tiêu tiền huấn luyện khuyết câu.
- [Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — bài báo gốc giới thiệu bộ chỉ số ROUGE.
- [Maynez et al. (2020). On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661) — bài báo toàn diện về bức tranh độ trung thực và tính xác thực trong tóm tắt.

