# Đánh giá ngữ cảnh dài — NIAH, RULER, LongBench, MRCR

> Gemini 3 Pro quảng cáo cửa sổ ngữ cảnh 10 triệu token. Nhưng ở mốc 1 triệu token, điểm số MRCR kiểm tra 8 cây kim rớt thẳng xuống còn 26,3%. Thông số quảng cáo ≠ Năng lực thực tế. Đánh giá ngữ cảnh dài (long-context evaluation) cho bạn biết dung lượng thực chiến chuẩn xác của mô hình mà bạn sắp triển khai vào sản xuất.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Giai đoạn 5 · 13 (Hỏi đáp - QA), Giai đoạn 5 · 23 (Chiến lược phân đoạn RAG)
**Time:** ~60 phút

## Vấn đề

Bạn có một bản hợp đồng dài 200 trang. Nhà cung cấp mô hình tuyên bố cửa sổ ngữ cảnh lên tới 1 triệu token. Bạn dán toàn bộ bản hợp đồng vào và đặt câu hỏi: "Điều khoản chấm dứt hợp đồng quy định như thế nào?" Mô hình đưa ra câu trả lời — nhưng câu trả lời lại trích từ trang bìa, bởi vì điều khoản chấm dứt hợp đồng nằm sâu ở vị trí 120.000 token, vượt quá vùng chú ý (attention) thực sự mà mô hình có thể duy trì.

Đây chính là khoảng cách năng lực ngữ cảnh (context-capacity gap) điển hình của năm 2026. Bảng thông số kỹ thuật (spec sheets) hào nhoáng khoe khoang 1M hay 10M token. Thực tế đo lường cho thấy chỉ khoảng 60-70% trong số đó là thực sự dùng được, và định nghĩa "dùng được" phụ thuộc rất lớn vào bản chất của tác vụ:

- **Truy xuất đơn giản (Tìm một cây kim trong đống rơm - Single needle in haystack):** Đạt độ chính xác gần như hoàn hảo tới mức tối đa được quảng cáo trên các mô hình hàng đầu (frontier models).
- **Đa bước suy luận / Tổng hợp thông tin (Multi-hop / aggregation):** Suy giảm nghiêm trọng khi vượt qua mốc ~128k token trên đa số mô hình.
- **Suy luận logic trên các dữ kiện nằm rải rác:** Là tác vụ đầu tiên bị sụp đổ hoàn toàn.

Đánh giá ngữ cảnh dài đo lường trực diện các trục này. Bài học này định danh rõ các bộ benchmark, bản chất thực sự mà mỗi bộ đo lường, và cách tự xây dựng bài kiểm tra tìm kim tùy chỉnh cho miền dữ liệu của riêng bạn.

## Khái niệm

![Đường cơ sở NIAH, đa tác vụ RULER, toàn diện LongBench](../assets/long-context-eval.svg)

**Tìm kim trong đống rơm (Needle-in-a-Haystack - NIAH, 2023).** Cấy một mẩu dữ kiện ("từ khóa bí mật là quả dứa") tại một độ sâu kiểm soát được bên trong một văn bản dài. Yêu cầu mô hình truy xuất lại mẩu dữ kiện đó. Quét lưới độ sâu (depth) $\times$ độ dài ngữ cảnh (length). Đây là benchmark ngữ cảnh dài nguyên thủy. Các mô hình biên giới hiện nay đã bão hòa bài kiểm tra này; nó chỉ còn là điều kiện cần chứ không đủ.

**RULER (Nvidia, 2024).** Gồm 13 loại tác vụ trên 4 danh mục: Truy xuất (đơn khóa / đa khóa / đa giá trị), theo dõi đa bước suy luận (theo dõi biến - variable tracking), tổng hợp (tần suất từ xuất hiện phổ biến), và hỏi đáp (QA). Chiều dài ngữ cảnh cấu hình linh hoạt (từ 4k tới 128k+ token). RULER vạch trần những mô hình đạt điểm tuyệt đối trên NIAH nhưng lại hoàn toàn thất bại ở các tác vụ đa bước suy luận. Trong báo cáo năm 2024, chỉ có một nửa trong số 17 mô hình tuyên bố hỗ trợ ngữ cảnh 32k+ giữ được chất lượng ổn định ở mốc 32k.

**LongBench v2 (2024).** 503 câu hỏi trắc nghiệm, ngữ cảnh từ 8k đến 2 triệu từ, trải rộng trên 6 danh mục tác vụ thực tế: QA đơn tài liệu, QA đa tài liệu, học trong ngữ cảnh dài (long in-context learning), hội thoại dài, kho mã nguồn phần mềm, dữ liệu cấu trúc dài. Đây là bộ benchmark tiêu chuẩn công nghiệp đo lường hành vi ngữ cảnh dài trong đời thực.

**MRCR (Multi-Round Coreference Resolution).** Giải quyết đồng tham chiếu đa lượt thoại ở quy mô lớn. Các biến thể kiểm tra 8 kim, 24 kim và 100 kim. Phơi bày chính xác số lượng dữ kiện tối đa mà một mô hình có thể cùng lúc ghi nhớ trước khi cơ chế chú ý bị quá tải.

**NoLiMa.** "Cây kim phi từ vựng" (Non-lexical needle). Cây kim và câu truy vấn hoàn toàn không chia sẻ bất kỳ từ ngữ trùng lặp nào; việc truy xuất đòi hỏi một bước suy luận ngữ nghĩa thực sự. Khó hơn NIAH rất nhiều.

**HELMET.** Nối ghép nhiều tài liệu khác nhau lại, sau đó đặt câu hỏi bất kỳ từ một tài liệu ngẫu nhiên. Kiểm tra khả năng chú ý chọn lọc.

**BABILong.** Nhúng các chuỗi suy luận bAbI vào sâu bên trong những đống rơm chứa toàn văn bản nhiễu không liên quan. Kiểm tra năng lực "suy luận trong đống rơm" thay vì chỉ truy xuất đơn thuần.

### Cần báo cáo những gì thực chất

- **Cửa sổ ngữ cảnh quảng cáo (Advertised context window):** Con số ghi trên tài liệu tiếp thị.
- **Chiều dài truy xuất hiệu dụng (Effective retrieval length):** Độ dài mà mô hình vượt qua bài test NIAH ở một ngưỡng tin cậy (ví dụ 90%).
- **Chiều dài suy luận hiệu dụng (Effective reasoning length):** Độ dài mà mô hình vượt qua bài test suy luận đa bước hoặc tổng hợp ở cùng ngưỡng đó.
- **Đường cong suy giảm (Degradation curve):** Biểu đồ thể hiện độ chính xác theo độ dài ngữ cảnh, vẽ riêng cho từng loại tác vụ.

Hai con số then chốt cần đưa vào bảng đặc tả kỹ thuật: Năng lực truy xuất hiệu dụng và Năng lực suy luận hiệu dụng. Thông thường, năng lực suy luận hiệu dụng chỉ đạt từ 25-50% kích thước cửa sổ được quảng cáo.

```figure
gx-niah-decay
```

## Thực hành xây dựng

### Bước 1: Tự dựng bài test NIAH cho miền dữ liệu của bạn

Xem mã nguồn khung sườn trong `code/main.py`:

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # Lặp lại văn bản đệm cho đến khi đủ chiều dài thân đống rơm.
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

Quét lưới `depth_ratio` $\in \{0, 0.25, 0.5, 0.75, 1.0\} \times$ `total_tokens` $\in \{1\text{k}, 4\text{k}, 16\text{k}, 64\text{k}\}$. Vẽ biểu đồ nhiệt (heatmap). Đó chính là thẻ đánh giá NIAH chuẩn xác cho mô hình mục tiêu của bạn.

### Bước 2: Biến thể tìm nhiều kim (Multi-needle)

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

Những câu hỏi như "Ba từ khóa ma thuật là gì?" đòi hỏi mô hình phải tìm thấy và kết hợp trọn vẹn cả 3 dữ kiện. Vượt qua bài kiểm tra đơn kim hoàn toàn không đảm bảo thành công ở bài kiểm tra đa kim.

### Bước 3: Theo dõi biến đa bước suy luận (Phong cách RULER)

```python
haystack = """X1 = 42. ... (đoạn đệm) ... X2 = X1 + 10. ... (đoạn đệm) ... X3 = X2 * 2."""
question = "Giá trị của X3 là bao nhiêu?"
```

Câu trả lời đòi hỏi chuỗi liên kết qua ba phép gán biến nằm cách xa nhau. Các mô hình biên giới ở mốc 128k token thường rớt xuống chỉ còn 50-70% độ chính xác trong bài toán này.

### Bước 4: Đánh giá LongBench v2 trên ngăn xếp công nghệ của bạn

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

Báo cáo độ chính xác chi tiết theo từng danh mục. Điểm số tổng hợp chung sẽ che giấu những cách biệt rất lớn giữa các tác vụ khác nhau.

## Cạm bẫy

- **Chỉ đánh giá bằng NIAH.** Đạt điểm tuyệt đối NIAH ở mốc 1 triệu token không nói lên điều gì về khả năng suy luận đa bước. Hãy luôn chạy thêm RULER hoặc bài kiểm tra đa bước tùy biến.
- **Lấy mẫu độ sâu đồng nhất sơ sài.** Rất nhiều đội ngũ kỹ thuật chỉ kiểm tra ở độ sâu giữa `depth=0.5`. Cần kiểm tra đầy đủ các mốc `depth=0, 0.25, 0.5, 0.75, 1.0` — hiện tượng "mất tích ở đoạn giữa" (lost in the middle) là có thật và rất phổ biến.
- **Trùng lặp từ vựng với văn bản đệm.** Nếu cây kim chia sẻ nhiều từ khóa với văn bản đệm xung quanh, việc truy xuất trở nên quá dễ dàng. Hãy dùng các cây kim phi từ vựng dạng NoLiMa.
- **Bỏ qua độ trễ.** Một prompt dài 1 triệu token có thể mất từ 30 đến 120 giây chỉ để xử lý xong pha prefill ban đầu. Hãy đo lường thời gian tới token đầu tiên (time-to-first-token - TTFT) song song với độ chính xác.
- **Tin vào số liệu tự công bố của nhà cung cấp.** OpenAI, Google, Anthropic đều tự công bố các con số rất ấn tượng của họ. Hãy luôn tự mình đo kiểm độc lập trên chính bài toán nghiệp vụ của bạn.

## Sử dụng trong thực tế

Ngăn xếp công nghệ năm 2026:

| Tình huống sử dụng | Bộ benchmark khuyến nghị |
|--------------------|---------------------------|
| Kiểm tra nhanh độ ổn định | Tự xây dựng NIAH tại 3 độ sâu $\times$ 3 độ dài |
| Lựa chọn mô hình cho sản xuất | RULER (13 tác vụ) tại độ dài ngữ cảnh mục tiêu |
| Chất lượng hỏi đáp QA đời thực | LongBench v2 tập con single-doc-QA |
| Năng lực suy luận đa bước | BABILong hoặc bài toán theo dõi biến tùy biến |
| Hội thoại / Chatbot ngữ cảnh dài | MRCR 8 kim tại độ dài ngữ cảnh mục tiêu |
| Kiểm thử thoái lui khi nâng cấp mô hình | Khung kiểm thử cố định NIAH + RULER nội bộ |

Nguyên tắc vàng trong môi trường sản xuất: Tuyệt đối không bao giờ tin tưởng một cửa sổ ngữ cảnh cho đến khi bạn đã tự mình chạy kiểm thử NIAH kèm ít nhất 1 tác vụ suy luận ở đúng độ dài ngữ cảnh dự kiến.

## Đóng gói sản phẩm

Lưu tệp dưới dạng `outputs/skill-long-context-eval.md`:

```markdown
---
name: long-context-eval
description: Design a long-context evaluation battery for a given model and use case.
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

Given a target model, target context length, and use case, output:

1. Tests. NIAH depth × length grid; RULER multi-hop; custom domain task.
2. Sampling. Depths 0, 0.25, 0.5, 0.75, 1.0 at each length.
3. Metrics. Retrieval pass rate; reasoning pass rate; time-to-first-token; cost-per-query.
4. Cutoff. Effective retrieval length (90% pass) and effective reasoning length (70% pass). Report both.
5. Regression. Fixed harness, rerun on every model upgrade, surface deltas.

Refuse to trust a context window from the model card alone. Refuse NIAH-only evaluation for any multi-hop workload. Refuse vendor self-reported long-context scores as independent evidence.
```

## Bài tập

1. **Dễ.** Xây dựng bài kiểm tra NIAH với 3 độ sâu (0.25, 0.5, 0.75) $\times$ 3 độ dài (1k, 4k, 16k token). Chạy trên bất kỳ mô hình nào bạn có. Vẽ tỷ lệ thành công thành biểu đồ nhiệt $3 \times 3$.
2. **Trung bình.** Thêm biến thể tìm 3 cây kim cùng lúc. Đo lường tỷ lệ truy xuất đủ cả 3 kim ở từng độ dài. So sánh với tỷ lệ thành công của bài kiểm tra đơn kim ở cùng độ dài đó.
3. **Khó.** Xây dựng một tác vụ theo dõi biến gồm 3 bước suy luận ($X_1 \to X_2 \to X_3$) nhúng vào 64k token văn bản đệm. Đo lường độ chính xác trên 3 mô hình hàng đầu. Báo cáo chiều dài suy luận hiệu dụng cho từng mô hình.

## Thuật ngữ then chốt

| Thuật ngữ | Cách gọi thông thường | Bản chất kỹ thuật thực sự |
|-----------|------------------------|---------------------------|
| NIAH | Kim trong đống rơm | Cấy một dữ kiện vào văn bản đệm, yêu cầu mô hình truy xuất lại. |
| RULER | NIAH nâng cao | Bộ 13 tác vụ bao quát truy xuất / đa bước / tổng hợp / QA. |
| Ngữ cảnh hiệu dụng (Effective context) | Dung lượng thực tế | Độ dài tối đa mà độ chính xác vẫn duy trì trên ngưỡng chấp nhận được. |
| Mất tích ở đoạn giữa (Lost in the middle) | Thiên kiến độ sâu | Hiện tượng mô hình ít chú ý đến nội dung nằm ở đoạn giữa văn bản dài. |
| Đa kim (Multi-needle) | Nhiều dữ kiện cùng lúc | Cấy nhiều mẩu tin; kiểm tra năng lực điều phối sự chú ý thay vì chỉ tìm kiếm đơn lẻ. |
| MRCR | Đồng tham chiếu đa lượt | Kiểm thử đồng tham chiếu 8, 24 hoặc 100 kim; phơi bày độ bão hòa chú ý. |
| NoLiMa | Kim phi từ vựng | Kim và câu hỏi không trùng từ khóa; bắt buộc phải suy luận ngữ nghĩa. |

## Đọc thêm

- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — kho mã nguồn gốc khởi xướng bài test NIAH.
- [Hsieh et al. (2024). RULER: What's the Real Context Size of Your Long-Context LMs?](https://arxiv.org/abs/2404.06654) — bài báo công bố bộ benchmark đa tác vụ RULER.
- [Bai et al. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) — đánh giá ngữ cảnh dài trong các tình huống thực tế.
- [Modarressi et al. (2024). NoLiMa: Non-lexical needles](https://arxiv.org/abs/2404.06666) — phương pháp kiểm thử tìm kim nâng cao.
- [Kuratov et al. (2024). BABILong](https://arxiv.org/abs/2406.10149) — kiểm thử suy luận logic trong đống rơm.
- [Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — bài báo kinh điển về hiện tượng mất tập trung ở đoạn giữa.

