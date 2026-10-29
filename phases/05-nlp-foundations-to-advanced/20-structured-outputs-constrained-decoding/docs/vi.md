# Đầu ra có Cấu trúc và Giải mã có Ràng buộc (Structured Outputs & Constrained Decoding)

> Yêu cầu LLM trả về JSON. Bạn sẽ nhận được JSON trong phần lớn thời gian. Nhưng trong môi trường sản xuất, "phần lớn" chính là thảm họa. Giải mã có ràng buộc biến "phần lớn" thành "luôn luôn 100%" bằng cách chỉnh sửa trực tiếp phân phối logits trước khi lấy mẫu.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 17 (Chatbot), Phase 5 · 19 (Phân đoạn Từ phụ)
**Time:** ~60 phút

## Vấn đề đặt ra (The Problem)

Một bộ phân loại gửi prompt đến LLM: "Hãy trả về một trong các nhãn {positive, negative, neutral}." Mô hình vui vẻ phản hồi: "Cảm xúc của đoạn văn này là positive — bài đánh giá này cực kỳ tích cực vì khách hàng đã chỉ rõ rằng họ..." Kết quả: Bộ phân tích cú pháp (parser) của bạn bị sập (crash). Điểm F1 của hệ thống phân loại rơi thẳng về 0.0.

Tạo sinh văn bản dạng tự do không phải là một bản hợp đồng kỹ thuật; nó chỉ là một lời gợi ý mang tính may rủi. Một hệ thống phần mềm sản xuất bắt buộc phải có một hợp đồng dữ liệu chuẩn xác tuyệt đối.

Năm 2026 tồn tại ba tầng giải pháp:

1. **Prompting (Nhắc nhở qua câu lệnh).** Yêu cầu mô hình một cách lịch sự: "Chỉ trả về duy nhất đối tượng JSON." Đạt độ tin cậy ~80% trên các mô hình frontier hàng đầu, và thấp hơn rất nhiều trên các mô hình nhỏ.
2. **API đầu ra có cấu trúc gốc của nhà cung cấp.** `response_format` của OpenAI, tool use của Anthropic, JSON mode của Gemini. Rất đáng tin cậy trên các lược đồ (schema) được hỗ trợ. Nhược điểm: Bị khóa chặt vào nhà cung cấp (vendor lock-in).
3. **Giải mã có ràng buộc (Constrained decoding).** Can thiệp trực tiếp vào giá trị logits tại mỗi bước sinh từ để mô hình *về mặt toán học không thể* phát ra bất kỳ token nào không hợp lệ. Chuẩn xác 100% theo đúng thiết kế cấu trúc. Chạy được trên bất kỳ mô hình mã nguồn mở cục bộ nào.

Bài học này xây dựng trực giác cho cả ba phương pháp và chỉ rõ thời điểm nên chọn giải pháp nào.

## Khái niệm (The Concept)

![Giải mã có ràng buộc che mặt nạ các token không hợp lệ tại mỗi bước](../assets/constrained-decoding.svg)

**Cơ chế hoạt động của giải mã có ràng buộc.** Tại mỗi bước tạo sinh, LLM tính toán một vector điểm số logit trên toàn bộ không gian từ vựng (~100.000 token). Một bộ xử lý logit (*logit processor*) đứng ở giữa mô hình và bộ lấy mẫu (sampler). Nó tính toán xem những token nào là hợp lệ dựa trên vị trí hiện tại trong ngữ pháp mục tiêu — JSON Schema, biểu thức chính quy (regex), hoặc ngữ pháp phi ngữ cảnh (CFG) — và gán điểm số logit của tất cả các token không hợp lệ bằng âm vô cực ($-\infty$). Phép tính softmax trên các logit còn lại sẽ chỉ phân bổ xác suất cho các token tiếp nối hợp lệ.

Các bản cài đặt tiêu biểu năm 2026:

- **Outlines.** Biên dịch JSON Schema hoặc Regex thành một máy trạng thái hữu hạn (FSM). Mỗi token có thể tra cứu các token kế tiếp hợp lệ với độ phức tạp $O(1)$. Do dựa trên FSM, các lược đồ đệ quy cần được làm phẳng (flattening).
- **XGrammar / llguidance.** Các công cụ dựa trên ngữ pháp phi ngữ cảnh (CFG). Xử lý mượt mà các JSON Schema đệ quy lồng nhau. Chi phí phụ trội giải mã gần như bằng 0. OpenAI đã ghi nhận đóng góp của `llguidance` trong bản triển khai đầu ra có cấu trúc năm 2025 của họ.
- **vLLM guided decoding.** Hỗ trợ tích hợp sẵn `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar` thông qua các backend Outlines, XGrammar hoặc lm-format-enforcer.
- **Instructor.** Thư viện bọc ngoài dựa trên Pydantic cho mọi LLM. Tự động thử lại (retry) khi kiểm thực schema thất bại. Độc lập với nhà cung cấp, nhưng không can thiệp vào logits — nó dựa vào cơ chế thử lại kết hợp với các prompt tối ưu cho đầu ra có cấu trúc.

### Kết quả trái ngược trực giác

Giải mã có ràng buộc thường chạy *nhanh hơn* so với tạo sinh tự do không ràng buộc. Có hai lý do: Thứ nhất, nó thu hẹp không gian tìm kiếm của token tiếp theo. Thứ hai, các bản cài đặt thông minh có thể bỏ qua hoàn toàn việc tính toán nơ-ron cho các token cố định bắt buộc (các đoạn khung sườn như `{"name": "` — nơi mọi byte đều đã được định đoạt trước).

### Cạm bẫy gây mất mát lớn: Thứ tự các trường

Thứ tự xuất hiện của các trường trong schema quyết định tư duy logic của mô hình. Nếu bạn đặt trường `answer` trước trường `reasoning`, mô hình buộc phải cam kết đưa ra câu trả lời trước khi nó kịp suy nghĩ lập luận. Chuỗi JSON sinh ra hoàn toàn hợp lệ về cú pháp, nhưng câu trả lời bên trong lại hoàn toàn sai lệch. Không một bộ kiểm thực cú pháp nào bắt được lỗi này.

```json
// SAI - Mô hình phải đoán mò đáp án trước khi lập luận
{"answer": "yes", "reasoning": "because ..."}

// ĐÚNG - Mô hình suy nghĩ lập luận trước rồi mới chốt đáp án
{"reasoning": "... therefore ...", "answer": "yes"}
```

Thứ tự các trường trong schema là tư duy logic, không đơn thuần là định dạng hiển thị.

```figure
constrained-decoder
```

## Xây dựng thực tế (Build It)

### Bước 1: Giải mã ràng buộc bằng Regex từ đầu

Xem `code/main.py` để xem bản cài đặt FSM độc lập hoàn chỉnh. Ý tưởng cốt lõi chỉ trong 30 dòng mã:

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

Máy FSM theo dõi phần ngữ pháp nào đã được thỏa mãn cho đến thời điểm hiện tại. Hàm `valid_tokens(state, tokenizer)` tính toán xem những token nào trong từ vựng có thể chuyển tiếp trạng thái FSM mà không đi chệch khỏi đường dẫn chấp nhận hợp lệ.

### Bước 2: Dùng Outlines cho JSON Schema

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

Tỷ lệ lỗi kiểm thực bằng 0 tuyệt đối. Không bao giờ gãy cú pháp. FSM khiến cho các đầu ra sai cú pháp trở thành trạng thái không thể chạm tới về mặt xác suất.

### Bước 3: Dùng Instructor cho Pydantic độc lập nhà cung cấp

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

Cơ chế hoạt động khác biệt: Instructor không can thiệp vào logits. Nó định dạng schema thành prompt, phân tích cú pháp đầu ra, và tự động gọi lại mô hình (mặc định tối đa 3 lần) nếu kiểm thực Pydantic thất bại. Hoạt động với mọi nhà cung cấp mô hình. Nhược điểm: Việc thử lại làm tăng độ trễ và chi phí token. Ưu điểm: Tính linh hoạt, độc lập chuyển đổi giữa các nhà cung cấp mô hình.

### Bước 4: API có cấu trúc gốc của nhà cung cấp

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                      "properties": {"sentiment": {"type": "string",
                                                   "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

Giải mã có ràng buộc được thực thi trực tiếp phía máy chủ (server-side). Độ tin cậy ngang ngửa Outlines cho các schema được hỗ trợ. Không cần tự quản lý mô hình cục bộ. Nhược điểm là bị khóa chặt vào nhà cung cấp.

## Các cạm bẫy cần tránh

- **Schema đệ quy (Recursive schemas).** Outlines làm phẳng đệ quy đến một độ sâu cố định. Các cấu trúc đầu ra dạng cây (bình luận lồng nhau, cây cú pháp trừu tượng AST) bắt buộc phải dùng XGrammar hoặc llguidance (dựa trên CFG).
- **Tập Enum khổng lồ.** Một enum có 10.000 lựa chọn sẽ biên dịch FSM rất chậm hoặc bị timeout. Hãy chuyển sang cơ chế truy xuất: dùng retriever để tìm ra top-k ứng viên trước, rồi ràng buộc enum trong phạm vi top-k đó.
- **Ngữ pháp quá khắt khe.** Ép định dạng ngày tháng theo regex `date: "YYYY-MM-DD"` sẽ khiến mô hình không thể trả về giá trị `"unknown"` khi tài liệu thiếu ngày. Mô hình sẽ bù đắp bằng cách tự bịa ra một ngày giả. Hãy luôn cho phép giá trị `null` hoặc một ký hiệu chỉ định đặc biệt.
- **Cam kết quá sớm (Premature commitment).** Xem lại cạm bẫy thứ tự các trường ở trên: Luôn luôn đặt trường lập luận (`reasoning`) đứng trước trường kết luận (`answer`).
- **Chế độ JSON mode không kèm schema.** Chế độ JSON thuần túy của nhà cung cấp chỉ đảm bảo đầu ra đúng cú pháp JSON hợp lệ, chứ *không đảm bảo khớp với các trường bạn cần*. Luôn luôn cung cấp một schema đầy đủ.

## Sử dụng trong thực tế (Use It)

Ngăn xếp công nghệ năm 2026:

| Tình huống | Lựa chọn |
|------------|----------|
| Dùng mô hình OpenAI/Anthropic/Google, schema đơn giản | Đầu ra có cấu trúc gốc của nhà cung cấp (Native structured output) |
| Đa nhà cung cấp, quy trình làm việc chuẩn Pydantic, chấp nhận độ trễ thử lại | Instructor |
| Mô hình cục bộ mở, cần hợp lệ 100%, schema phẳng | Outlines (FSM) |
| Mô hình cục bộ mở, schema có cấu trúc đệ quy | XGrammar hoặc llguidance |
| Máy chủ suy luận tự lưu trữ (self-hosted) | vLLM guided decoding |
| Xử lý hàng loạt (batch processing) chấp nhận retry | Instructor + mô hình chi phí rẻ nhất |

## Đóng gói sản phẩm (Ship It)

Lưu thành `outputs/skill-structured-output-picker.md`:

```markdown
---
name: structured-output-picker
description: Lựa chọn giải pháp đầu ra có cấu trúc, thiết kế schema và kế hoạch kiểm thực dữ liệu.
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

Dựa trên trường hợp sử dụng (nhà cung cấp, ngân sách độ trễ, độ phức tạp của schema, mức độ chấp nhận lỗi), bạn đưa ra:

1. Cơ chế kỹ thuật: Native vendor structured output, Instructor retries, Outlines FSM, hoặc XGrammar CFG. Giải thích lý do trong một câu.
2. Thiết kế Schema: Thứ tự các trường (lập luận trước, đáp án cuối cùng), các trường cho phép null khi "không rõ thông tin", lựa chọn giữa enum và regex, các trường bắt buộc (required).
3. Chiến lược dự phòng lỗi: Số lần thử lại tối đa, mô hình dự phòng khi lỗi, xử lý êm thuận giá trị `null`, cơ chế từ chối khi dữ liệu nằm ngoài phân phối.
4. Kế hoạch kiểm thực: Tỷ lệ tuân thủ schema (mục tiêu 100%), tính hợp lệ về mặt ngữ nghĩa (LLM giám khảo), tỷ lệ bao phủ các trường, độ trễ p50/p99.

Từ chối bất kỳ thiết kế nào đặt trường `answer` hoặc `decision` đứng trước các trường suy nghĩ lập luận (`reasoning`). Từ chối sử dụng JSON mode trần trụi mà thiếu schema định nghĩa rõ ràng. Cảnh báo các schema đệ quy nếu chạy sau một thư viện chỉ hỗ trợ FSM thuần túy.
```

## Bài tập (Exercises)

1. **Dễ.** Gửi prompt đến một mô hình mở kích thước nhỏ (ví dụ: Llama-3.2-3B) mà không bật giải mã có ràng buộc, yêu cầu trả về `Review(sentiment, confidence, evidence_span)`. Đo lường tỷ lệ phần trăm đầu ra phân tích cú pháp thành công thành JSON hợp lệ trên 100 bài đánh giá.
2. **Trung bình.** Chạy lại trên cùng tập ngữ liệu đó nhưng bật chế độ Outlines JSON mode. So sánh tỷ lệ tuân thủ schema, độ trễ suy luận và độ chính xác ngữ nghĩa.
3. **Khó.** Tự tay lập trình một bộ giải mã ràng buộc bằng Regex từ đầu để bắt buộc định dạng số điện thoại (`\d{3}-\d{3}-\d{4}`). Kiểm chứng rằng mô hình đạt 0% lỗi sai định dạng trên 1.000 mẫu sinh ra liên tiếp.

## Thuật ngữ cốt lõi (Key Terms)

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|-----------|-------------------|-----------------|
| Giải mã có ràng buộc (Constrained decoding) | Ép buộc đầu ra chuẩn | Che mặt nạ (masking) các điểm logit của token không hợp lệ tại từng bước tạo sinh. |
| Bộ xử lý Logit (Logit processor) | Bộ lọc ràng buộc | Hàm biến đổi toán học: `(logits, state) -> masked_logits`. |
| FSM | Máy trạng thái hữu hạn | Biểu diễn ngữ pháp đã biên dịch; tra cứu token kế tiếp hợp lệ với độ phức tạp $O(1)$. |
| CFG | Ngữ pháp phi ngữ cảnh | Ngữ pháp xử lý được các cấu trúc đệ quy lồng nhau; biểu đạt mạnh mẽ hơn FSM. |
| Thứ tự trường Schema | Có quan trọng không? | Tối quan trọng — trường đầu tiên ép mô hình cam kết logic; luôn đặt lập luận trước kết luận. |
| Guided decoding | Tên gọi của vLLM | Cùng một khái niệm giải mã có ràng buộc, được tích hợp trực tiếp vào tầng máy chủ suy luận. |
| JSON mode | Bản sơ khai của OpenAI | Chỉ đảm bảo đúng cú pháp JSON; KHÔNG đảm bảo khớp với schema nghiệp vụ của bạn. |

## Đọc thêm (Further Reading)

- [Willard, Louf (2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702) — bài báo gốc đặt nền móng cho thư viện Outlines.
- [XGrammar paper (2024)](https://arxiv.org/abs/2411.15100) — giải mã có ràng buộc tốc độ cao dựa trên CFG.
- [vLLM — Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) — hướng dẫn tích hợp đầu ra có cấu trúc trên máy chủ suy luận vLLM.
- [OpenAI — Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs) — tài liệu tham khảo API chính thức và các cạm bẫy thường gặp.
- [Instructor library](https://python.useinstructor.com/) — công cụ kết hợp Pydantic và cơ chế tự động thử lại đa nhà cung cấp.
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) — bài đo benchmark so sánh 6 khung giải mã có ràng buộc hàng đầu.

