# Lưu bộ nhớ đệm Lời nhắc và Lưu bộ nhớ đệm Ngữ cảnh (Prompt Caching and Context Caching)

> Lời nhắc hệ thống (system prompt) của bạn dài 4.000 token. Ngữ cảnh RAG của bạn dài 20.000 token. Bạn gửi cả hai theo mọi yêu cầu. Bạn cũng phải trả tiền cho cả hai — trong mọi lần gửi. Kỹ thuật lưu bộ nhớ đệm lời nhắc (prompt caching) cho phép nhà cung cấp giữ cho tiền tố (prefix) đó luôn sẵn sàng (warm) ở phía máy chủ của họ và chỉ tính phí bạn khoảng 10% so với đơn giá thông thường khi tái sử dụng. Khi được áp dụng đúng cách, nó giúp cắt giảm 50–90% chi phí suy luận và giảm 40–85% độ trễ sinh token đầu tiên (first-token latency).

**Type:** Xây dựng  
**Languages:** Python  
**Prerequisites:** Phase 11 · 01 (Kỹ thuật viết lời nhắc - Prompt Engineering), Phase 11 · 05 (Kỹ thuật quản lý ngữ cảnh - Context Engineering), Phase 11 · 11 (Bộ nhớ đệm và Chi phí - Caching and Cost)  
**Time:** ~60 phút  

## Vấn đề Thực tế

Một tác tử lập trình (coding agent) gửi cùng một lời nhắc hệ thống dài 15.000 token đến Claude trong mỗi lượt trò chuyện. Hai mươi lượt tương tác với mức giá 3 USD/triệu token đầu vào sẽ tiêu tốn 0,90 USD chỉ riêng chi phí nạp ngữ cảnh ban đầu — chưa tính đến bất kỳ tin nhắn thực tế nào của người dùng. Nhân con số này với 10.000 cuộc trò chuyện mỗi ngày, hóa đơn sẽ chạm mốc 9.000 USD/ngày chỉ để xử lý đoạn văn bản cố định không bao giờ thay đổi.

Bạn không thể cắt ngắn lời nhắc vì sẽ làm giảm chất lượng phản hồi của mô hình. Bạn cũng không thể không gửi nó — mô hình cần toàn bộ quy tắc chỉ dẫn trong mỗi lượt sinh. Lựa chọn khả dĩ duy nhất là ngừng trả toàn bộ tiền cho một tiền tố văn bản mà nhà cung cấp thực chất đã xử lý trước đó.

Giải pháp ở đây chính là lưu bộ nhớ đệm lời nhắc (prompt caching). Anthropic đã ra mắt tính năng này vào tháng 8 năm 2024 (bổ sung tùy chọn kéo dài thời gian sống TTL lên 1 giờ vào năm 2025), OpenAI tự động hóa nó vào cuối năm đó, Google phát hành cơ chế lưu đệm ngữ cảnh tường minh (context caching) cùng Gemini 1.5, và hiện nay cả ba nhà cung cấp đều hỗ trợ tính năng này như một thành phần hạng nhất trên các mô hình hàng đầu (frontier models) của họ.

## Khái niệm Cốt lõi

![Lưu bộ nhớ đệm lời nhắc: ghi một lần, đọc với giá rẻ](../assets/prompt-caching.svg)

**Cơ chế hoạt động.** Khi tiền tố của một yêu cầu mới khớp hoàn toàn với tiền tố của một yêu cầu gần đây, nhà cung cấp sẽ tái sử dụng bộ đệm khóa-giá trị (KV-cache) từ lần chạy trước thay vì phải mã hóa (encode) lại các token từ đầu. Bạn chỉ phải trả thêm một khoản phí ghi bộ đệm nhỏ (write premium) trong lần đầu tiên và được hưởng mức giảm giá đọc bộ đệm rất lớn (read discount) cho tất cả các lần tiếp theo.

**Ba trường phái của các nhà cung cấp vào năm 2026.**

| Nhà cung cấp | Phong cách API | Mức giảm giá khi trúng cache | Phí phụ trội khi ghi | TTL mặc định | Ngưỡng token tối thiểu để lưu đệm |
|---------|-----------|--------------|---------------|-------------|---------------|
| Anthropic | Đánh dấu tường minh bằng `cache_control` trên các khối nội dung | Giảm 90% chi phí đầu vào | Phụ phí 25% | 5 phút (có thể mở rộng lên 1 giờ) | 1.024 token (Sonnet/Opus), 2.048 (Haiku) |
| OpenAI | Tự động phát hiện tiền tố khớp | Giảm 50% chi phí đầu vào | Không có | Lên tới 1 giờ (nỗ lực tối đa - best-effort) | 1.024 token |
| Google (Gemini) | API tường minh `CachedContent` | Tính phí lưu trữ; đọc với giá ~25% mức thông thường | Phí lưu trữ theo token·giờ | Người dùng tự thiết lập (mặc định 1 giờ) | 4.096 token (Flash), 32.768 (Pro) |

**Bất biến cốt lõi (The invariant).** Cả ba nhà cung cấp đều chỉ lưu bộ nhớ đệm theo tiền tố (prefix-only). Nếu có bất kỳ token nào sai khác giữa hai yêu cầu, toàn bộ các token nằm sau điểm sai khác đầu tiên đó đều bị tính là trượt bộ nhớ đệm (cache miss). Vì vậy, hãy đặt các phần *cố định (stable)* ở phía trên và các phần *biến động (variable)* ở phía dưới.

### Cấu trúc Bố cục Thân thiện với Bộ nhớ Đệm

```text
[system prompt]          <-- lưu đệm phần này
[tool definitions]       <-- lưu đệm phần này
[few-shot examples]      <-- lưu đệm phần này
[retrieved documents]    <-- lưu đệm nếu được tái sử dụng, ngược lại thì không
[conversation history]   <-- lưu đệm cho đến lượt trò chuyện liền trước
[current user message]   <-- tuyệt đối không lưu đệm (thay đổi sau mỗi lượt)
```

Nếu bạn vi phạm trật tự sắp xếp này — ví dụ đặt tin nhắn người dùng lên trên lời nhắc hệ thống, hoặc chèn các kết quả truy xuất động xen kẽ giữa các ví dụ mẫu (few-shot examples) — bộ nhớ đệm sẽ không bao giờ trúng.

### Phép tính Điểm Hòa vốn (Break-even Calculation)

Mức phụ phí ghi 25% của Anthropic (với TTL 5 phút) sẽ tự bù đắp chi phí ngay từ lần đọc đầu tiên, bởi vì một lần đọc từ bộ đệm chỉ tốn 0,1x chi phí. Cụ thể: 1 lần ghi + 1 lần đọc có chi phí trung bình là 0,675x trên mỗi yêu cầu (tiết kiệm 32%); 1 lần ghi + 10 lần đọc có chi phí trung bình là 0,205x (tiết kiệm 80%). Đối với tùy chọn TTL 1 giờ, nhà cung cấp tính phí ghi gấp 2 lần, do đó khối nội dung đó cần ít nhất hai lần đọc để bắt đầu có lãi. Quy tắc ngón tay cái: hãy lưu bộ nhớ đệm bất cứ nội dung nào mà bạn dự kiến sẽ tái sử dụng ít nhất 3 lần trong khoảng thời gian TTL; điều này đảm bảo vượt qua điểm hòa vốn trên cả hai mức cấu hình TTL.

````figure prompt-cache-hit
````

## Phần Thực hành (Build It)

### Bước 1: Lưu Đệm Lời nhắc trên Anthropic với Dấu đánh dấu Tường minh

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "You are a senior Python reviewer. Follow the rubric exactly.\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```

Dấu đánh dấu `cache_control` thông báo cho Anthropic lưu giữ khối nội dung này trong vòng 5 phút. Việc tái sử dụng trong cửa sổ thời gian đó sẽ trúng bộ nhớ đệm; các yêu cầu đến sau khi hết hạn sẽ phải ghi đệm lại từ đầu.

**Các trường sử dụng (usage) trong phản hồi:**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # tính phí ở mức 1.25x
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # tính phí ở mức 0.1x
```

Hãy kiểm tra cả hai trường này trong đường ống CI của bạn — nếu `cache_read_input_tokens` vẫn giữ nguyên bằng 0 qua nhiều yêu cầu, các khóa bộ nhớ đệm (cache keys) của bạn đang bị sai lệch.

### Bước 2: Kéo dài Thời gian Sống (Extended TTL) lên 1 Giờ

Đối với các tác vụ xử lý hàng loạt (batch jobs) chạy dài, thời gian mặc định 5 phút có thể sẽ hết hạn giữa các đợt chạy. Thiết lập tham số `ttl`:

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

TTL 1 giờ có chi phí phụ trội khi ghi gấp đôi (50% so với mức cơ sở thay vì 25%), nhưng sẽ mang lại hiệu quả kinh tế rất nhanh cho bất kỳ tác vụ hàng loạt nào tái sử dụng tiền tố nhiều hơn 5 lần.

### Bước 3: Cơ chế Lưu Đệm Tự động của OpenAI

OpenAI không yêu cầu bạn phải cấu hình bất kỳ tham số nào trong mã nguồn. Bất kỳ tiền tố nào vượt quá 1.024 token khớp với một yêu cầu gần đây sẽ tự động nhận được mức giảm giá 50%.

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # dài và cố định
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # phần token được giảm giá
```

Quy tắc sắp xếp bố cục thân thiện với bộ nhớ đệm vẫn được áp dụng tương tự. Tuy nhiên, có hai yếu tố có thể phá vỡ bộ nhớ đệm của OpenAI mà không ảnh hưởng tới Anthropic: việc thay đổi trường `user` (được dùng như một thành phần của khóa bộ nhớ đệm) và việc xáo trộn thứ tự của danh sách công cụ (tools).

### Bước 4: Lưu Đệm Ngữ cảnh Tường minh trên Gemini

Gemini coi bộ nhớ đệm như một đối tượng hạng nhất mà bạn có thể khởi tạo và đặt tên riêng biệt:

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3.8-flash",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3.8-flash",
    contents=["Review this code:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

Gemini tính phí lưu trữ theo token·giờ trong suốt thời gian bộ đệm tồn tại, và các lượt đọc được tính với mức giá ~25% so với đơn giá đầu vào thông thường. Đây là cấu trúc lý tưởng khi bạn cần tái sử dụng cùng một bộ dữ liệu hoặc lời nhắc khổng lồ qua nhiều phiên làm việc trải dài trong nhiều ngày.

### Bước 5: Đo lường Tỷ lệ Trúng Cache trong Môi trường Sản xuất

Tham khảo tệp `code/main.py` để xem bộ hạch toán mô phỏng ba nhà cung cấp giúp theo dõi số lượt ghi/đọc/trượt và tính toán chi phí kết hợp trên 1.000 yêu cầu. Hãy thiết lập cổng kiểm duyệt triển khai (deployment gate) dựa trên tỷ lệ trúng mục tiêu — hầu hết các hệ thống chạy Anthropic trong sản xuất phải đạt tỷ lệ đọc >80% sau giai đoạn làm nóng (warmup).

## Những Cạm bẫy Phổ biến Vẫn Tồn tại trong Năm 2026

- **Đặt dấu thời gian động (dynamic timestamps) ở đầu lời nhắc.** Đặt dòng chữ `"Current time: 2026-04-22 15:30:02"` ở dòng đầu tiên của lời nhắc hệ thống sẽ khiến mọi yêu cầu đều bị trượt cache. Hãy di chuyển dấu thời gian xuống bên dưới điểm ngắt bộ nhớ đệm (cache breakpoint).
- **Xáo trộn thứ tự công cụ (Tool reordering).** Hãy tuần tự hóa danh sách công cụ theo một trật tự cố định — việc một dictionary bị xáo trộn giữa các lần triển khai mã nguồn sẽ phá hủy toàn bộ các lần trúng bộ nhớ đệm.
- **Văn bản tự do gần giống nhau.** "Bạn là một trợ lý hữu ích." so với "Bạn là một trợ lý AI hữu ích." — chỉ cần sai khác một byte duy nhất cũng dẫn đến trượt cache toàn bộ.
- **Khối dữ liệu quá nhỏ.** Anthropic áp dụng ngưỡng sàn tối thiểu 1.024 token (2.048 đối với dòng Haiku). Các khối dữ liệu nhỏ hơn ngưỡng này sẽ âm thầm bị bỏ qua và không được lưu đệm.
- **Bảng điều khiển chi phí không phân tách.** Không gộp chung "token đầu vào", mà hãy tách biệt rõ ràng giữa token đã lưu đệm và token chưa lưu đệm. Nếu không, sự sụt giảm lưu lượng truy cập thực tế có thể bị hiểu nhầm là chiến thắng của bộ nhớ đệm.

## Ứng dụng Thực tế (Use It)

Ngăn xếp lựa chọn bộ nhớ đệm năm 2026:

| Tình huống | Lựa chọn Đề xuất |
|-----------|------|
| Tác tử có lời nhắc hệ thống >10k token ổn định, nhiều lượt tương tác | Anthropic `cache_control` với TTL 5 phút |
| Tác vụ xử lý hàng loạt tái sử dụng cùng tiền tố trong hơn 30 phút | Anthropic với cấu hình `ttl: "1h"` |
| Điểm cuối Serverless trên GPT-5, không có hạ tầng tùy chỉnh | OpenAI tự động (chỉ cần đảm bảo tiền tố ổn định và đủ dài) |
| Tái sử dụng kho mã nguồn hoặc tài liệu khổng lồ qua nhiều ngày | Google Gemini tường minh qua `CachedContent` |
| Chiến lược dự phòng đa nhà cung cấp (Cross-provider fallback) | Duy trì cấu trúc tiền tố có thể lưu đệm giống hệt nhau giữa các nhà cung cấp để tận dụng cache tối đa |

Kết hợp kỹ thuật này với bộ nhớ đệm ngữ nghĩa (Phase 11 · 11) cho tầng tin nhắn của người dùng: lưu bộ nhớ đệm lời nhắc xử lý các trường hợp tái sử dụng *trùng khớp từng token*, trong khi bộ nhớ đệm ngữ nghĩa xử lý các trường hợp tái sử dụng *trùng khớp về mặt ý nghĩa*.

## Thành phẩm Đóng gói (Ship It)

Lưu tệp `outputs/skill-prompt-caching-planner.md`:

```markdown
---
name: prompt-caching-planner
description: Thiết kế bố cục lời nhắc thân thiện với bộ nhớ đệm và chọn chế độ lưu đệm phù hợp theo nhà cung cấp.
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

Khi nhận được một cấu trúc lời nhắc (hệ thống + công cụ + ví dụ mẫu + truy xuất ngữ cảnh + lịch sử trò chuyện + tin nhắn người dùng) và hồ sơ sử dụng (số yêu cầu/giờ, TTL yêu cầu, nhà cung cấp), hãy xuất ra:

1. Bố cục (Layout): Các phần được sắp xếp lại thứ tự với duy nhất một điểm ngắt bộ nhớ đệm (cache breakpoint) được đánh dấu; giải thích phần nào cố định, phần nào biến động.
2. Chế độ Nhà cung cấp: Anthropic cache_control, OpenAI automatic, hoặc Gemini CachedContent. Giải thích lý do dựa trên TTL và mô hình tái sử dụng.
3. Điểm hòa vốn (Break-even): Số lần đọc dự kiến trên mỗi lần ghi trong thời gian TTL; phân tích chi phí ròng so với khi không dùng bộ đệm kèm công thức tính toán.
4. Kế hoạch kiểm định (Verification plan): Kiểm tra khẳng định (assertion) trong CI rằng `cache_read_input_tokens > 0` ở yêu cầu trùng lặp thứ hai; bảng điều khiển phân tách rõ token đệm và token không đệm.
5. Các kịch bản lỗi (Failure modes): Liệt kê ba nguyên nhân phổ biến nhất khiến bộ đệm bị trượt trong thiết kế này (dấu thời gian động, đổi thứ tự công cụ, văn bản gần giống nhau) và biện pháp phòng ngừa tương ứng.

Từ chối phê duyệt bất kỳ kế hoạch nào đặt một trường dữ liệu động phía trên điểm ngắt bộ nhớ đệm. Từ chối bật TTL 1 giờ nếu số lượt tái sử dụng không đủ bù đắp mức phí phụ trội 2x khi ghi.
```

## Bài tập Thực hành

1. **Mức độ Dễ.** Thực hiện một cuộc trò chuyện 10 lượt với lời nhắc hệ thống dài 5.000 token trên Claude. Chạy thử nghiệm khi không dùng `cache_control` và khi có dùng. Báo cáo chi phí token đầu vào chi tiết cho từng trường hợp.
2. **Mức độ Trung bình.** Xây dựng một khung kiểm thử nhận đầu vào là mẫu lời nhắc và nhật ký yêu cầu, sau đó tính toán tỷ lệ trúng dự kiến và số tiền tiết kiệm được cho từng nhà cung cấp (Anthropic 5m, Anthropic 1h, OpenAI tự động, Gemini tường minh).
3. **Mức độ Khó.** Xây dựng một bộ tối ưu hóa bố cục: nhận đầu vào là một lời nhắc và danh sách các trường được gắn nhãn `stable=True/False`, tự động sắp xếp lại lời nhắc để đặt một điểm ngắt bộ nhớ đệm duy nhất tại vị trí tối ưu nhất có thể mà không làm mất thông tin. Xác thực kết quả trên một điểm cuối Anthropic thực tế.

## Thuật ngữ Then chốt

| Thuật ngữ | Cách mọi người thường nói | Bản chất kỹ thuật thực tế |
|------|-----------------|-----------------------|
| Prompt caching | "Làm cho lời nhắc dài trở nên rẻ" | Tái sử dụng KV-cache ở phía nhà cung cấp cho các tiền tố trùng khớp; giảm 50-90% chi phí cho các token đầu vào lặp lại. |
| `cache_control` | "Dấu đánh dấu của Anthropic" | Thuộc tính của khối nội dung khai báo rằng "mọi thứ cho đến điểm này đều có thể lưu đệm"; `{"type": "ephemeral"}`. |
| Cache write | "Trả phí phụ trội" | Yêu cầu đầu tiên nạp dữ liệu vào bộ nhớ đệm; bị tính phí ở mức ~1,25x giá đầu vào trên Anthropic, miễn phí trên OpenAI. |
| Cache read | "Mức chiết khấu" | Các yêu cầu tiếp theo trùng khớp tiền tố; được tính phí ở mức 10% (Anthropic), 50% (OpenAI), ~25% (Gemini). |
| TTL | "Thời gian sống của đệm" | Số giây bộ nhớ đệm được giữ ấm; Anthropic mặc định 5 phút (mở rộng lên 1 giờ), OpenAI nỗ lực tối đa lên tới 1 giờ, Gemini do người dùng tự đặt. |
| Extended TTL | "Bộ đệm 1 giờ của Anthropic" | `{"type": "ephemeral", "ttl": "1h"}`; phí ghi gấp 2 lần nhưng mang lại hiệu quả cao khi xử lý theo lô. |
| Prefix match | "Lý do trượt bộ nhớ đệm" | Bộ đệm chỉ trúng khi mọi token tính từ đầu văn bản cho đến điểm ngắt trùng khớp chính xác từng byte. |
| Context caching (Gemini) | "Bộ đệm tường minh" | Đối tượng bộ đệm có tên và tính phí lưu trữ của Google; tối ưu cho việc tái sử dụng các kho tài liệu khổng lồ qua nhiều ngày. |

## Đọc thêm

- [Anthropic — Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — tài liệu về `cache_control`, TTL 1 giờ và bảng phân tích điểm hòa vốn.
- [OpenAI — Prompt caching](https://platform.openai.com/docs/guides/prompt-caching) — cơ chế so khớp tiền tố tự động.
- [Google — Context caching](https://ai.google.dev/gemini-api/docs/caching) — API `CachedContent` và bảng giá lưu trữ.
- [Anthropic engineering — Prompt caching for long-context workloads](https://www.anthropic.com/news/prompt-caching) — bài viết công bố tính năng kèm các số liệu đo lường độ trễ thực tế.
- Phase 11 · 05 (Kỹ thuật quản lý ngữ cảnh) — hướng dẫn phân tách lời nhắc để điểm ngắt bộ nhớ đệm phát huy tác dụng.
- Phase 11 · 11 (Bộ nhớ đệm và Chi phí) — kết hợp lưu đệm lời nhắc với bộ nhớ đệm ngữ nghĩa trên các câu hỏi của người dùng.
- [Pope et al., "Efficiently Scaling Transformer Inference" (2022)](https://arxiv.org/abs/2211.05102) — mô hình bộ nhớ KV-cache mà tính năng lưu đệm lời nhắc đưa ra cho người dùng; giải thích lý do tại sao việc đọc lại tiền tố đã lưu đệm rẻ hơn ~10 lần so với việc tính toán lại.
- [Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" (2023)](https://arxiv.org/abs/2308.16369) — giai đoạn prefill là bước mà prompt caching bỏ qua; bài báo này giải thích tại sao thời gian cho token đầu tiên (TTFT) giảm mạnh khi trúng cache trong khi thời gian cho mỗi token đầu ra (TPOT) không bị ảnh hưởng.
- [Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)](https://arxiv.org/abs/2211.17192) — prompt caching kết hợp cùng giải mã suy đoán (speculative decoding), Flash Attention, và MQA/GQA như những đòn bẩy giúp uốn cong đường cong chi phí suy luận.
