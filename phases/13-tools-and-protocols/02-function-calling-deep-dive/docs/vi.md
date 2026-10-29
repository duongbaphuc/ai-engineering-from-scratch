# Đi sâu vào cơ chế gọi hàm — OpenAI, Anthropic, Gemini

> Ba nhà cung cấp mô hình hàng đầu (frontier providers) đã hội tụ về cùng một vòng lặp gọi công cụ (tool-call loop) vào năm 2024 nhưng lại phân kỳ ở hầu như mọi khía cạnh còn lại. OpenAI sử dụng `tools` và `tool_calls`. Anthropic dùng các khối `tool_use` và `tool_result`. Gemini dùng `functionDeclarations` và cơ chế tương quan định danh duy nhất (unique-id correlation). Bài học này sẽ so sánh trực tiếp ba cách tiếp cận đặt cạnh nhau để mã nguồn triển khai trên một nhà cung cấp không bị gãy vỡ khi bạn chuyển đổi (port) sang nền tảng khác.

**Type:** Build
**Languages:** Python (stdlib, schema translators)
**Prerequisites:** Phase 13 · 01 (the tool interface)
**Time:** ~75 phút

## Mục tiêu học tập

- Nêu rõ ba điểm khác biệt về cấu trúc giữa các tải trọng (payloads) gọi hàm của OpenAI, Anthropic và Gemini (khai báo, lượt gọi, kết quả / declaration, call, result).
- Chuyển đổi (translate) một khai báo công cụ qua lại giữa cả ba định dạng nhà cung cấp và dự đoán nơi các ràng buộc ở chế độ nghiêm ngặt (strict mode) sẽ có sự phân kỳ.
- Sử dụng tham số lựa chọn công cụ (`tool_choice`) ở từng nhà cung cấp để ép buộc, cấm chỉ hoặc để mô hình tự động chọn các lượt gọi công cụ.
- Nắm vững các giới hạn cứng (hard limits) của từng nhà cung cấp (số lượng công cụ, độ sâu của lược đồ, độ dài đối số) và các dấu hiệu lỗi đặc trưng (error signatures) mà mỗi bên phát ra khi vi phạm giới hạn.

## Vấn đề đặt ra

Cấu trúc hình học của một yêu cầu gọi hàm (function-calling request) có sự khác biệt giữa các nhà cung cấp. Dưới đây là ba ví dụ cụ thể từ các ngăn xếp sản xuất (production stacks) năm 2026:

**API OpenAI Chat Completions / Responses.** Bạn truyền vào `tools: [{type: "function", function: {name, description, parameters, strict}}]`. Phản hồi của mô hình chứa `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`, trong đó `arguments` là một chuỗi JSON mà bạn phải tự phân tích cú pháp (parse). Chế độ nghiêm ngặt (`strict: true`) thực thi sự tuân thủ lược đồ thông qua kỹ thuật giải mã ràng buộc (constrained decoding).

**API Anthropic Messages.** Bạn truyền vào `tools: [{name, description, input_schema}]`. Phản hồi trả về dưới dạng `content: [{type: "text"}, {type: "tool_use", id, name, input}]`. `input` ở đây đã được phân tích cú pháp sẵn (là một đối tượng object, chứ không phải chuỗi). Bạn phản hồi lại bằng một thông điệp mới của vai trò `user` chứa khối `{type: "tool_result", tool_use_id, content}`.

**API Google Gemini.** Bạn truyền vào `tools: [{functionDeclarations: [{name, description, parameters}]}]` (lồng bên trong `functionDeclarations`). Phản hồi đến dưới dạng `candidates[0].content.parts: [{functionCall: {name, args, id}}]`, trong đó `id` là duy nhất từ phiên bản Gemini 3 trở lên để tương quan các lượt gọi song song. Bạn phản hồi lại với `{functionResponse: {name, id, response}}`.

Cùng một vòng lặp. Nhưng tên trường khác nhau, mức độ lồng nhau khác nhau, quy ước giữa chuỗi và đối tượng khác nhau, cơ chế tương quan khác nhau. Một đội ngũ kỹ sư viết agent dự báo thời tiết trên OpenAI sẽ phải tốn hai ngày để chuyển đổi sang Anthropic và thêm một ngày nữa cho Gemini chỉ vì lớp ghép nối (plumbing) này.

Bài học này xây dựng một bộ chuyển đổi (translator) hợp nhất ba định dạng thành một khai báo công cụ chuẩn tắc (canonical tool declaration) duy nhất và định tuyến tại biên (edge). Phase 13 · 17 sẽ tổng quát hóa mẫu thiết kế này thành một cổng kết nối LLM (LLM gateway).

## Khái niệm cốt lõi

### Cấu trúc chung

Mọi nhà cung cấp đều cần năm thành phần:

1. **Danh sách công cụ (Tool list).** Tên, mô tả và lược đồ đầu vào cho từng công cụ.
2. **Lựa chọn công cụ (Tool choice).** Ép buộc một công cụ cụ thể, cấm dùng công cụ, hoặc để mô hình tự quyết định.
3. **Phát ra lượt gọi (Call emission).** Đầu ra có cấu trúc nêu rõ tên công cụ và các đối số.
4. **Định danh lượt gọi (Call id).** Đối chiếu phản hồi với đúng lượt gọi tương ứng (rất quan trọng đối với các lượt gọi song song).
5. **Đưa kết quả vào (Result injection).** Một thông điệp hoặc khối nội dung gắn kết quả trả về với lượt gọi ban đầu.

### So sánh chi tiết cấu trúc theo từng trường

| Khía cạnh | OpenAI | Anthropic | Gemini |
|-----------|--------|-----------|--------|
| Bao bọc khai báo (Declaration envelope) | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| Trường lược đồ (Schema field) | `parameters` | `input_schema` | `parameters` |
| Vùng chứa phản hồi (Response container) | `tool_calls[]` trên thông điệp assistant | `content[]` thuộc kiểu `tool_use` | `parts[]` thuộc kiểu `functionCall` |
| Kiểu đối số (Arguments type) | Chuỗi JSON (stringified JSON) | Đối tượng đã phân tích cú pháp (parsed object) | Đối tượng đã phân tích cú pháp (parsed object) |
| Định dạng ID (Id format) | `call_...` (do OpenAI sinh ra) | `toolu_...` (Anthropic) | UUID (Gemini 3+) |
| Khối kết quả (Result block) | role `tool`, `tool_call_id` | role `user` với `tool_result`, `tool_use_id` | `functionResponse` với `id` tương ứng |
| Ép buộc một công cụ (Force-a-tool) | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| Cấm gọi công cụ (Forbid tools) | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| Lược đồ nghiêm ngặt (Strict schema) | `strict: true` | `strict: true` theo từng công cụ (tùy chọn kích hoạt) | `responseSchema` ở cấp độ yêu cầu |

### Các giới hạn thực tế bạn sẽ chạm phải

- **OpenAI.** 128 công cụ cho mỗi yêu cầu. Độ sâu lược đồ tối đa 5. Chuỗi đối số <= 8192 byte. Chế độ nghiêm ngặt yêu cầu không có `$ref`, không có `oneOf`/`anyOf`/`allOf` bị chồng chéo, mọi thuộc tính đều phải liệt kê trong `required`.
- **Anthropic.** 64 công cụ cho mỗi yêu cầu. Độ sâu lược đồ về mặt lý thuyết là không giới hạn nhưng ngưỡng thực tế là 10. Chế độ công cụ nghiêm ngặt là tùy chọn kích hoạt với `strict: true` trên định nghĩa công cụ. Nếu không có nó, lược đồ chỉ là một khế ước mang tính hướng dẫn mà mô hình có xu hướng tuân theo, chứ không phải một sự đảm bảo tuyệt đối.
- **Gemini.** 64 hàm cho mỗi yêu cầu. Các kiểu lược đồ là một tập con của OpenAPI 3.0 (có sự khác biệt nhỏ so với JSON Schema 2020-12). Các lượt gọi song song có định danh duy nhất (unique-id) kể từ Gemini 3.

### Hành vi của `tool_choice`

Ba chế độ mà cả ba nhà cung cấp đều hỗ trợ, dù cách đặt tên khác nhau:

- **Auto (Tự động):** Mô hình tự chọn dùng công cụ hoặc trả lời văn bản. Đây là chế độ mặc định.
- **Required / Any (Bắt buộc / Bất kỳ):** Mô hình bắt buộc phải gọi ít nhất một công cụ.
- **None (Không):** Mô hình tuyệt đối không được gọi công cụ.

Kèm theo một chế độ đặc thù của riêng từng nhà cung cấp:

- **OpenAI:** Ép buộc một công cụ cụ thể theo tên.
- **Anthropic:** Ép buộc một công cụ cụ thể theo tên; cờ `disable_parallel_tool_use` tách bạch giữa gọi đơn lẻ và gọi đa công cụ.
- **Gemini:** `mode: "VALIDATED"` định tuyến mọi phản hồi qua một bộ xác thực lược đồ bất kể ý định của mô hình.

### Các lượt gọi song song (Parallel calls)

Thiết lập `parallel_tool_calls: true` của OpenAI (mặc định) phát ra nhiều lượt gọi trong một thông điệp assistant duy nhất. Bạn thực thi tất cả chúng và trả lời bằng một thông điệp vai trò tool được đóng gói hàng loạt (batched), chứa một mục cho mỗi `tool_call_id`. Trong lịch sử, Anthropic từng chỉ hỗ trợ gọi đơn lẻ; `disable_parallel_tool_use: false` (mặc định kể từ Claude 3.5) đã kích hoạt đa lượt gọi. Gemini 2 cho phép gọi song song nhưng không cung cấp id ổn định; Gemini 3 đã bổ sung UUID để các phản hồi trả về lệch thứ tự vẫn có thể tương quan chuẩn xác.

### Truyền phát luồng (Streaming)

Cả ba nhà cung cấp đều hỗ trợ truyền phát luồng các lượt gọi công cụ. Định dạng truyền tải qua mạng (wire format) có sự khác nhau:

- **OpenAI.** Các đoạn mảnh delta của `tool_calls[i].function.arguments` đến dần dần. Bạn tích lũy cho đến khi nhận được `finish_reason: "tool_calls"`.
- **Anthropic.** Các sự kiện bắt đầu khối (block-start) / đoạn mảnh delta (block-delta) / dừng khối (block-stop). Các đoạn mảnh `input_json_delta` mang theo các đối số từng phần.
- **Gemini.** `streamFunctionCallArguments` (mới có trong Gemini 3) phát ra các đoạn mảnh kèm theo một `functionCallId` để nhiều lệnh gọi song song có thể đan xen lẫn nhau.

Phase 13 · 03 sẽ đi sâu vào việc tái hợp ráp (reassembly) khi gọi song song kết hợp truyền phát luồng. Bài học này tập trung vào các cấu trúc khai báo và gọi đơn lẻ.

### Xử lý lỗi và tự sửa chữa (Errors and repair)

Lỗi đối số không hợp lệ cũng biểu hiện khác nhau:

- **OpenAI (không dùng strict).** Mô hình trả về `arguments: "{bad json}"`, việc phân tích cú pháp JSON thất bại, bạn chèn một thông điệp lỗi và gọi lại mô hình.
- **OpenAI (dùng strict).** Việc xác thực diễn ra ngay trong quá trình giải mã (decoding); JSON không hợp lệ là điều bất khả thi nhưng khối từ chối (`refusal`) có thể xuất hiện.
- **Anthropic (không dùng strict).** `input` có thể chứa các trường không mong đợi; lược đồ chỉ đóng vai trò khuyến nghị. Bạn phải xác thực ở phía server. Khi có `strict: true`, việc lấy mẫu (sampling) sẽ bị ràng buộc chặt chẽ theo lược đồ.
- **Gemini.** Điểm dị biệt của OpenAPI 3.0: `enum` trên các trường đối tượng bị bỏ qua trong im lặng; bạn phải tự xác thực lấy.

### Mẫu thiết kế bộ chuyển đổi (The translator pattern)

Một khai báo công cụ chuẩn tắc trong mã nguồn của bạn có dạng như sau (bạn tự định hình cấu trúc):

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

Ba hàm nhỏ sẽ chuyển đổi khai báo này sang định dạng của ba nhà cung cấp tương ứng. Khung kiểm thử trong `code/main.py` thực hiện chính xác điều này, sau đó mô phỏng vòng lặp hoàn chỉnh (round-trip) một lượt gọi công cụ giả lập qua cấu trúc phản hồi của từng nhà cung cấp. Không cần kết nối mạng — bài học này giảng dạy về các cấu trúc định dạng, không phải giao thức HTTP.

Các đội ngũ kỹ sư sản xuất thường bọc bộ chuyển đổi này trong `AbstractToolset` (Pydantic AI), `UniversalToolNode` (LangGraph), hoặc `BaseTool` (LlamaIndex). Phase 13 · 17 sẽ phát hành một gateway phơi bày API chuẩn OpenAI phía trước bất kỳ nhà cung cấp nào trong ba bên trên.

```figure
function-call-args
```

## Thực hành (Use It)

Tệp `code/main.py` định nghĩa một dataclass `Tool` chuẩn tắc duy nhất và ba bộ chuyển đổi để phát ra JSON khai báo theo chuẩn OpenAI, Anthropic và Gemini. Sau đó nó phân tích một phản hồi nhà cung cấp được tạo thủ công của từng cấu trúc thành cùng một đối tượng gọi chuẩn tắc, chứng minh rằng ngữ nghĩa bên dưới hoàn toàn giống nhau. Hãy chạy mã và so sánh ba bản khai báo đặt cạnh nhau.

Những điểm cần chú ý:

- Ba khối khai báo chỉ khác nhau ở lớp vỏ bao bọc (envelope) và tên các trường.
- Ba khối phản hồi khác nhau ở vị trí chứa lượt gọi (ở `tool_calls` cấp cao nhất, khối trong `content[]`, hoặc mục trong `parts[]`).
- Một hàm `canonical_call()` duy nhất trích xuất `{id, name, args}` từ cả ba cấu trúc phản hồi.

## Đóng gói sản phẩm (Ship It)

Bài học này tạo ra artifact `outputs/skill-provider-portability-audit.md`. Khi tiếp nhận một mã tích hợp gọi hàm trên một nhà cung cấp, kỹ năng này sẽ đưa ra một bản kiểm toán tính khả chuyển (portability audit): những giới hạn nào của nhà cung cấp đang bị lệ thuộc vào, những trường nào cần đổi tên, và những gì sẽ bị hỏng khi chuyển mã sang từng nhà cung cấp khác.

## Bài tập thực hành (Exercises)

1. Chạy `code/main.py` và xác minh rằng JSON khai báo của cả ba nhà cung cấp đều tuần tự hóa từ cùng một đối tượng `Tool` bên dưới. Sửa đổi công cụ chuẩn tắc để thêm một tham số enum và xác nhận rằng chỉ có bộ chuyển đổi của Gemini mới cần xử lý điểm dị biệt của OpenAPI.

2. Thêm một bộ phân tích cú pháp `ListToolsResponse` cho từng nhà cung cấp nhằm trích xuất danh sách công cụ mà mô hình trả về sau một lượt gọi `list_tools` hoặc khám phá công cụ (discovery call). Lưu ý rằng OpenAI không có tính năng này một cách bản địa; hãy ghi nhận sự bất đối xứng này.

3. Hiện thực hóa việc chuyển đổi `tool_choice`: ánh xạ một đối tượng chuẩn tắc `ToolChoice(mode="force", tool_name="x")` sang cấu trúc của cả ba nhà cung cấp. Sau đó tiếp tục ánh xạ `mode="any"` và `mode="none"`. Đối chiếu với bảng so sánh trong bài học.

4. Chọn một trong ba nhà cung cấp và đọc tài liệu hướng dẫn gọi hàm từ đầu đến cuối. Tìm một trường trong đặc tả lược đồ của nhà cung cấp đó mà hai bên còn lại không hỗ trợ. Gợi ý: `strict` của OpenAI, `disable_parallel_tool_use` của Anthropic, `function_calling_config.allowed_function_names` của Gemini.

5. Viết một bộ kiểm thử vector: một lượt gọi công cụ có các đối số vi phạm lược đồ đã khai báo. Chạy nó qua bộ xác thực của từng nhà cung cấp (bộ xác thực thư viện chuẩn ở Bài 01 có thể dùng làm đại diện) và ghi lại những lỗi nào được phát ra. Ghi chép lại nhà cung cấp nào bạn sẽ chọn dùng trong môi trường sản xuất về mặt tính nghiêm ngặt.

## Thuật ngữ then chốt (Key Terms)

| Thuật ngữ | Cách mọi người thường gọi | Ý nghĩa thực sự |
|-----------|---------------------------|-----------------|
| Gọi hàm (Function calling) | "Sử dụng công cụ" | API ở cấp độ nhà cung cấp để phát ra lượt gọi công cụ có cấu trúc |
| Khai báo công cụ (Tool declaration) | "Đặc tả công cụ" | Tải trọng gồm tên + mô tả + đầu vào JSON Schema |
| Lựa chọn công cụ (`tool_choice`) | "Bắt buộc / Cấm đoán" | Các chế độ auto / required / none / chỉ định tên cụ thể |
| Chế độ nghiêm ngặt (Strict mode) | "Thực thi lược đồ" | Cờ của OpenAI nhằm ràng buộc quá trình giải mã khớp chính xác với lược đồ |
| Khối `tool_use` (`tool_use` block) | "Cấu trúc gọi của Anthropic" | Khối nội dung nội dòng chứa id, name, input |
| Phần tử `functionCall` (`functionCall` part) | "Cấu trúc gọi của Gemini" | Một mục trong mảng `parts[]` chứa name, args và id |
| Đối số dạng chuỗi (Arguments-as-string) | "JSON hóa chuỗi" | OpenAI trả về các đối số dưới dạng chuỗi JSON thay vì đối tượng |
| Gọi công cụ song song (Parallel tool calls) | "Phát tán trong một lượt" | Nhiều lượt gọi công cụ trong một thông điệp duy nhất của assistant |
| Từ chối (Refusal) | "Mô hình khước từ" | Khối từ chối chỉ xuất hiện ở chế độ strict mode thay cho một lượt gọi |
| Tập con OpenAPI 3.0 (OpenAPI 3.0 subset) | "Điểm dị biệt lược đồ Gemini" | Gemini sử dụng một phương ngữ tương tự JSON Schema nhưng có những khác biệt nhỏ |

## Tài liệu đọc thêm (Further Reading)

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — tài liệu tham khảo chuẩn mực bao gồm chế độ strict mode và các lượt gọi song song
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — ngữ nghĩa của các khối `tool_use` và `tool_result`
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) — các lượt gọi song song, định danh duy nhất và tập con OpenAPI
- [Vertex AI — Function calling reference](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) — bề mặt gọi hàm cấp doanh nghiệp của Gemini
- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — chi tiết về cơ chế thực thi lược đồ trong chế độ strict mode
