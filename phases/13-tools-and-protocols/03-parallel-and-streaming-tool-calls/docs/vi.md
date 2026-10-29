# Gọi công cụ song song và truyền phát luồng cùng công cụ

> Ba lượt tra cứu thời tiết độc lập được thực hiện tuần tự sẽ tốn ba lượt khứ hồi (round trips). Nếu chạy song song, tổng thời gian sẽ co lại bằng thời gian của lượt gọi đơn lẻ chậm nhất. Mọi nhà cung cấp mô hình hàng đầu hiện nay đều có thể phát ra nhiều lượt gọi công cụ trong một lượt xử lý duy nhất. Lợi ích mang lại là có thật; nhưng các cấu trúc ghép nối kỹ thuật bên dưới lại rất tinh tế. Bài học này sẽ dẫn dắt bạn qua cả hai khía cạnh: phát tán song song (parallel fan-out) và tái hợp ráp đối số truyền phát luồng (streamed-argument reassembly), đồng thời nhấn mạnh vào cái bẫy tương quan định danh (id-correlation trap).

**Type:** Build
**Languages:** Python (stdlib, thread pool + streaming harness)
**Prerequisites:** Phase 13 · 02 (function calling deep dive)
**Time:** ~75 phút

## Mục tiêu học tập

- Giải thích lý do vì sao cờ `parallel_tool_calls: true` tồn tại và khi nào nên vô hiệu hóa nó.
- Tương quan (correlate) các đoạn mảnh đối số dạng luồng với đúng id lượt gọi công cụ trong quá trình phát tán song song.
- Tái hợp ráp (reassemble) các chuỗi `arguments` từng phần thành JSON hoàn chỉnh mà không bị phân tích cú pháp quá sớm (parse-early trap).
- Chạy một bài kiểm chuẩn (benchmark) thời tiết ba thành phố để chứng minh độ trễ giữa thực thi tuần tự và song song.

## Vấn đề đặt ra

Nếu không có các lượt gọi song song, một agent khi trả lời câu hỏi "thời tiết ở Bengaluru, Tokyo và Zurich như thế nào" sẽ phải thực hiện như sau:

```text
user -> LLM
LLM -> call get_weather(Bengaluru)
host -> run executor, reply with result
LLM -> call get_weather(Tokyo)
host -> run executor, reply with result
LLM -> call get_weather(Zurich)
host -> run executor, reply with result
LLM -> final text answer
```

Ba lượt khứ hồi tới LLM, mỗi lượt đều phải gánh thêm độ trễ của hàm executor. Tổng thời gian thực tế (wall-clock time) xấp xỉ gấp 4 lần thời gian lý tưởng.

Khi có các lượt gọi song song:

```text
user -> LLM
LLM -> call get_weather(Bengaluru); call get_weather(Tokyo); call get_weather(Zurich)
host -> run all three executors concurrently, reply with three results
LLM -> final text answer
```

Chỉ mất một lượt khứ hồi duy nhất tới LLM. Thời gian chạy executor chỉ bằng thời gian của lệnh gọi lâu nhất trong cả ba, chứ không phải tổng thời gian cộng lại. Các bài kiểm chuẩn thực tế trên OpenAI, Anthropic và Gemini cho thấy mức giảm từ 60 đến 70% thời gian thực tế đối với các khối lượng công việc phát tán (fan-out workloads).

Cái giá phải trả là sự phức tạp trong việc đối chiếu tương quan. Khi ba lượt gọi hoàn thành lệch thứ tự, kết quả của bạn phải mang đúng `tool_call_id` tương ứng để mô hình có thể sắp xếp đối chiếu chúng. Khi kết quả được truyền phát luồng (stream), bạn phải thu thập các mảnh đối số rời rạc thành một chuỗi JSON hoàn chỉnh trước khi thực thi. Gemini 3 đã bổ sung các id duy nhất một phần là để giải quyết bài toán thực tế: hai lượt gọi song song tới cùng một công cụ trước đây không thể phân biệt được với nhau.

## Khái niệm cốt lõi

### Kích hoạt chế độ song song

- **OpenAI.** `parallel_tool_calls: true` được bật mặc định. Đặt thành `false` để ép buộc chạy tuần tự.
- **Anthropic.** Chạy song song thông qua `disable_parallel_tool_use: false` (mặc định trên Claude 3.5 trở lên). Đặt thành `true` để chạy tuần tự.
- **Gemini.** Luôn có khả năng chạy song song; `tool_config.function_calling_config.mode = "AUTO"` cho phép mô hình tự quyết định.

Hãy vô hiệu hóa tính năng song song khi các công cụ có sự phụ thuộc về thứ tự (ví dụ: `create_file` rồi mới đến `write_file`), khi đầu ra của lệnh gọi này cung cấp dữ liệu cho đầu vào của lệnh gọi khác, hoặc khi bộ giới hạn tốc độ (rate limiter) không chịu nổi lưu lượng phát tán đồng thời.

### Đối chiếu tương quan bằng ID (Id correlation)

Mỗi lượt gọi do mô hình phát ra đều có một `id`. Mỗi kết quả do host trả về bắt buộc phải mang đúng id đó. Thiếu điều này, các kết quả sẽ bị mơ hồ, lẫn lộn.

- **OpenAI.** `tool_call_id` trên mỗi thông điệp có vai trò tool.
- **Anthropic.** `tool_use_id` trên mỗi khối `tool_result`.
- **Gemini.** `id` trên mỗi `functionResponse` (từ Gemini 3 trở lên; Gemini 2 đối chiếu theo tên khiến các lượt gọi trùng tên song song bị lỗi).

### Thực thi các lượt gọi đồng thời (Running calls concurrently)

Host sẽ chạy executor của từng lượt gọi trên một luồng (thread), coroutine hoặc worker từ xa riêng biệt. Khung kiểm thử đơn giản nhất sử dụng một pool luồng (thread pool); môi trường sản xuất sử dụng asyncio với `asyncio.gather` hoặc mô hình đồng thời có cấu trúc (structured concurrency). Thứ tự hoàn thành là không thể đoán trước — và id chính là chiếc mỏ neo định danh.

Một lỗi phổ biến: phản hồi các kết quả theo thứ tự danh sách gọi thay vì thứ tự hoàn thành thực tế. Cách làm này thường vẫn chạy được vì mô hình chỉ quan tâm đến `tool_call_id`, nhưng nếu một kết quả bị rớt hoặc bị nhân đôi, danh sách phản hồi bị đổi thứ tự so với thời điểm thực tế sẽ khiến việc truy vết sự cố trong nhật ký log trở nên vô cùng khó khăn. Hãy ưu tiên phản hồi theo thứ tự hoàn thành kèm theo id tường minh.

### Truyền phát luồng các lượt gọi công cụ (Streaming tool calls)

Khi mô hình truyền phát luồng, các `arguments` sẽ đến theo từng mảnh nhỏ. Ba luồng đoạn mảnh riêng biệt của ba lượt gọi song song sẽ đan xen nhau trên đường truyền mạng. Bạn cần một bộ tích lũy (accumulator) cho mỗi id.

Cấu trúc theo từng nhà cung cấp:

- **OpenAI.** Mỗi đoạn mảnh là `choices[0].delta.tool_calls[i].function.arguments` (chuỗi văn bản từng phần). Đoạn mảnh mang theo `index` (vị trí trong danh sách gọi). Bạn tích lũy theo từng index, đọc `id` khi nó xuất hiện lần đầu, và phân tích cú pháp JSON khi nhận được `finish_reason = "tool_calls"`.
- **Anthropic.** Các sự kiện luồng gồm có `message_start`, sau đó là một sự kiện `content_block_start` cho mỗi khối thuộc kiểu `tool_use` (chứa id, name, input rỗng). Các sự kiện `content_block_delta` mang theo các đoạn mảnh `input_json_delta`. Sự kiện `content_block_stop` đóng từng khối.
- **Gemini.** `streamFunctionCallArguments` (từ Gemini 3 trở lên) phát ra các đoạn mảnh kèm theo một `functionCallId` để các lượt gọi đan xen nhau một cách mạch lạc. Trước phiên bản Gemini 3, chế độ truyền phát luồng trả về từng lượt gọi hoàn chỉnh một lúc.

### JSON từng phần và cái bẫy phân tích sớm (Partial JSON and the parse-early trap)

Bạn không thể phân tích cú pháp `arguments` cho đến khi nó được truyền tải hoàn chỉnh. Một đoạn JSON chưa đầy đủ như `{"city": "Beng` không phải là cú pháp hợp lệ và sẽ ném ra ngoại lệ (exception). Cổng kiểm soát chính xác là tín hiệu kết thúc lượt gọi từ nhà cung cấp: `finish_reason = "tool_calls"` của OpenAI, `content_block_stop` của Anthropic, hoặc sự kiện kết thúc luồng của Gemini. Chỉ khi đó mới được gọi `json.loads`. Một giải pháp mạnh mẽ hơn là sử dụng bộ phân tích cú pháp JSON gia tăng (incremental JSON parser) phát ra các sự kiện khi từng cấu trúc hoàn thành; hướng dẫn streaming của OpenAI khuyến nghị cách này cho giao diện hiển thị trạng thái "đang suy nghĩ" trực tiếp. Việc đếm dấu ngoặc nhọn là không đáng tin cậy để kiểm tra tính toàn vẹn (các dấu ngoặc nằm trong chuỗi trích dẫn hoặc ký tự thoát escape sẽ gây ra dương tính giả) và chỉ nên dùng như một giải pháp phỏng đoán tạm thời khi gỡ lỗi.

### Hoàn thành lệch thứ tự (Out-of-order completion)

```text
call_A: API nhanh, trả về đầu tiên
call_B: API chậm, trả về thứ hai
call_C: API trung bình, trả về thứ ba
```

Phản hồi từ phía host vẫn phải nêu rõ các id:

```json
[{"role": "tool", "tool_call_id": "call_A", "content": "..."},
 {"role": "tool", "tool_call_id": "call_B", "content": "..."},
 {"role": "tool", "tool_call_id": "call_C", "content": "..."}]
```

Thứ tự trong phản hồi không ảnh hưởng đến tính đúng đắn trên OpenAI hay Anthropic. Gemini chấp nhận bất kỳ thứ tự nào miễn là các id trùng khớp.

### Kiểm chuẩn: tuần tự so với song song (Benchmark: sequential vs parallel)

Khung kiểm thử trong `code/main.py` mô phỏng ba executor với độ trễ lần lượt là 400, 600 và 800 ms. Thực thi tuần tự mất tổng cộng 1800 ms. Thực thi song song chỉ mất max(400, 600, 800) = 800 ms. Mức chênh lệch mang tính hằng số chứ không phải tỷ lệ, do đó lượng thời gian tiết kiệm được sẽ càng tăng khi số lượng công cụ càng nhiều.

Cảnh báo trong thế giới thực: các lượt gọi song song tạo áp lực lớn lên các API phía hạ nguồn. Việc phát tán đồng thời 10 yêu cầu tới một dịch vụ bị giới hạn tốc độ sẽ gây ra lỗi. Phase 13 · 17 sẽ bao quát cơ chế áp lực ngược (backpressure) ở cấp độ gateway; ngữ nghĩa thử lại (retry semantics) sẽ được thiết kế trong một phase tương lai.

### Thời gian thực tế của việc phát tán dạng luồng (Streaming fan-out wall-clock)

Nếu bản thân mô hình truyền phát dạng luồng, bạn có thể bắt đầu thực thi ngay khi các đối số của một lượt gọi hoàn tất, thay vì phải chờ đợi toàn bộ các lượt gọi kết thúc. Đây là một tối ưu hóa được OpenAI ghi chép trong tài liệu nhưng không phải mọi SDK đều phơi bày. Khung kiểm thử trong bài học này hiện thực hóa điều đó: ngay khi luồng mô phỏng mang lại một đối tượng đối số hoàn chỉnh, host sẽ lập tức khởi chạy lượt gọi đó.

```figure
tp-parallel-fanout
```

## Thực hành (Use It)

Tệp `code/main.py` gồm có hai phần. Phần thứ nhất chạy ba lượt gọi thời tiết mô phỏng theo cách tuần tự và song song sử dụng `concurrent.futures.ThreadPoolExecutor` rồi in ra thời gian thực tế. Phần thứ hai phát lại một phản hồi dạng luồng giả lập — các đoạn mảnh của `arguments` cho ba lượt gọi song song đan xen nhau trên cùng một luồng — và tái hợp ráp chúng theo từng id với `StreamAccumulator`. Không cần LLM, không cần mạng, chỉ có logic tái hợp ráp thuần túy.

Những điểm cần chú ý:

- Bộ đếm thời gian tuần tự chạm mốc 1,8 giây. Bộ đếm thời gian song song chạm mốc 0,8 giây trên cùng các độ trễ giả lập.
- Bộ tích lũy (accumulator) xử lý các đoạn mảnh đến lệch thứ tự bằng cách đệm (buffer) theo từng id và chỉ phân tích cú pháp khi JSON của mỗi lượt gọi đã hoàn chỉnh.
- Hàm executor khởi chạy ngay khi các đối số của một id hoàn tất, chứ không đợi toàn bộ các luồng kết thúc.

## Đóng gói sản phẩm (Ship It)

Bài học này tạo ra artifact `outputs/skill-parallel-call-safety-check.md`. Nhận vào một sổ đăng ký công cụ, kỹ năng này sẽ kiểm toán xem công cụ nào là an toàn để song song hóa, công cụ nào có sự phụ thuộc về thứ tự, và công cụ nào có nguy cơ làm quá tải giới hạn tốc độ hạ nguồn — từ đó trả về một sổ đăng ký đã hiệu chỉnh kèm cờ `parallel_safe` cho từng công cụ.

## Bài tập thực hành (Exercises)

1. Chạy `code/main.py` và thay đổi các độ trễ mô phỏng. Xác nhận rằng tỷ lệ giữa song song và tuần tự xấp xỉ `max/sum` (các lần chạy thực tế có thể chênh lệch nhẹ so với lý thuyết do việc lập lịch luồng, tuần tự hóa và chi phí overhead của khung kiểm thử). Với phân phối độ trễ nào thì việc chạy song song không còn mang lại nhiều ý nghĩa?

2. Mở rộng bộ tích lũy để xử lý trường hợp "lượt gọi bị hủy giữa chừng trong luồng" bằng cách hủy bỏ bộ đệm của nó và phát ra một sự kiện `cancelled`. Nhà cung cấp nào ghi chép tài liệu rõ ràng về trường hợp này? Hãy kiểm tra ngữ nghĩa của sự kiện `content_block_stop` ở Anthropic và hành vi `finish_reason: "length"` ở OpenAI.

3. Thay thế pool luồng bằng `asyncio.gather`. Đo lường kiểm chuẩn cả hai. Bạn sẽ thấy những lợi ích nhỏ với async do chi phí chuyển ngữ cảnh (context-switch) thấp hơn, nhưng chỉ khi các executor thực hiện các thao tác I/O thực sự.

4. Chọn hai công cụ KHÔNG ĐƯỢC PHÉP chạy song song (ví dụ: `create_file` rồi mới đến `write_file`). Thêm một đồ thị phụ thuộc thứ tự `ordering_dependency` vào sổ đăng ký và dùng đồ thị này làm cổng kiểm soát việc phát tán song song. Đây là cấu trúc tối thiểu cho việc lập lịch nhận biết phụ thuộc (dependency-aware scheduling), điều sẽ được chuẩn hóa trong một phase kỹ nghệ agent tương lai.

5. Đọc phần gọi hàm song song của OpenAI và tài liệu về `disable_parallel_tool_use` của Anthropic. Xác định một loại công cụ trong thế giới thực mà Anthropic khuyến nghị nên tắt tính năng song song. (Gợi ý: các đột biến làm biến đổi trạng thái có hệ quả trên cùng một tài nguyên.)

## Thuật ngữ then chốt (Key Terms)

| Thuật ngữ | Cách mọi người thường gọi | Ý nghĩa thực sự |
|-----------|---------------------------|-----------------|
| Gọi công cụ song song (Parallel tool calls) | "Phát tán trong một lượt" | Mô hình phát ra nhiều lượt gọi công cụ trong một thông điệp assistant duy nhất |
| `parallel_tool_calls` | "Cờ của OpenAI" | Bật hoặc tắt tính năng phát ra nhiều lượt gọi |
| `disable_parallel_tool_use` | "Cờ đảo ngược của Anthropic" | Cờ từ chối tham gia (opt-out); mặc định là bật song song |
| ID lượt gọi công cụ (Tool call id) | "Tay cầm tương quan" | Định danh cho từng lượt gọi mà thông điệp kết quả bắt buộc phải phản hồi lại |
| Bộ tích lũy (Accumulator) | "Bộ đệm luồng" | Bộ đệm chuỗi theo từng id để lưu giữ các đoạn mảnh `arguments` từng phần |
| Hoàn thành lệch thứ tự (Out-of-order completion) | "Nhanh nhất về trước" | Các lượt gọi song song hoàn tất theo thứ tự không đoán trước; id là chất kết dính |
| Đồ thị phụ thuộc (Dependency graph) | "Ràng buộc thứ tự" | Các công cụ mà đầu ra của chúng làm đầu vào cho công cụ khác; không thể song song hóa |
| Cái bẫy phân tích sớm (Parse-early trap) | "JSON.parse bị nổ tung" | Nỗ lực phân tích cú pháp một chuỗi `arguments` chưa hoàn chỉnh |
| `streamFunctionCallArguments` | "Tính năng của Gemini 3" | Các đoạn mảnh đối số dạng luồng kèm id duy nhất cho từng lượt gọi |
| Phản hồi theo thứ tự hoàn thành (Completion-order reply) | "Không đợi tất cả" | Phản hồi các kết quả ngay khi chúng đến, được khóa theo id |

## Tài liệu đọc thêm (Further Reading)

- [OpenAI — Parallel function calling](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) — hành vi mặc định và cờ từ chối tham gia
- [Anthropic — Parallel tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use) — `disable_parallel_tool_use` và đóng gói hàng loạt kết quả
- [Google — Gemini function calling parallel section](https://ai.google.dev/gemini-api/docs/function-calling) — các lượt gọi song song tương quan theo id từ Gemini 3
- [OpenAI — Streaming responses with tools](https://platform.openai.com/docs/api-reference/responses-streaming) — tái hợp ráp các đối số theo đoạn mảnh trong luồng dữ liệu OpenAI
- [Anthropic — Streaming messages](https://docs.anthropic.com/en/api/messages-streaming) — sự kiện `content_block_delta` với `input_json_delta`
