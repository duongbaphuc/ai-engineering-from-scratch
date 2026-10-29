# Đầu ra có cấu trúc — JSON Schema, Pydantic, Zod, Giải mã ràng buộc

> Việc "yêu cầu mô hình một cách lịch sự để trả về JSON" thất bại từ 5 đến 15% thời gian, ngay cả trên các mô hình hàng đầu (frontier models). Đầu ra có cấu trúc (structured output) xóa bỏ khoảng cách đó bằng kỹ thuật giải mã ràng buộc (constrained decoding): mô hình bị chặn hoàn toàn về mặt vật lý, không thể phát ra bất kỳ token nào vi phạm lược đồ. Chế độ nghiêm ngặt (strict mode) của OpenAI, tính năng sử dụng công cụ định kiểu theo lược đồ của Anthropic, `responseSchema` của Gemini, `output_type` của Pydantic AI và `.parse` của Zod là năm hình thức biểu hiện bề mặt của cùng một ý tưởng. Bài học này xây dựng bộ xác thực lược đồ và khế ước chế độ nghiêm ngặt mà học viên sẽ sử dụng cho mọi quy trình trích xuất (extraction pipeline) trong môi trường sản xuất.

**Type:** Build
**Languages:** Python (stdlib, JSON Schema 2020-12 subset)
**Prerequisites:** Phase 13 · 02 (function calling deep dive)
**Time:** ~75 phút

## Mục tiêu học tập

- Viết một lược đồ JSON Schema 2020-12 cho một mục tiêu trích xuất bằng cách sử dụng các ràng buộc phù hợp (enum, min/max, required, pattern).
- Giải thích lý do vì sao chế độ nghiêm ngặt (strict mode) và giải mã ràng buộc (constrained decoding) mang lại những đảm bảo khác biệt hoàn toàn so với cách tiếp cận "xác thực sau khi sinh" (validate after generation).
- Phân biệt ba chế độ lỗi (failure modes): lỗi phân tích cú pháp (parse error), vi phạm lược đồ (schema violation) và sự từ chối của mô hình (model refusal).
- Phát hành một quy trình trích xuất hoàn chỉnh với cơ chế sửa chữa có định kiểu (typed repair) và xử lý từ chối có định kiểu (typed refusal handling).

## Vấn đề đặt ra

Một agent khi đọc một email đơn đặt hàng cần phải biến văn bản tự do thành cấu trúc `{customer, line_items, total_usd}`. Có ba cách tiếp cận.

**Cách tiếp cận thứ nhất: nhắc mô hình trả về JSON (prompt for JSON).** "Hãy trả lời bằng JSON với các trường customer, line_items, total_usd." Hoạt động tốt từ 85 đến 95% thời gian trên các mô hình hàng đầu. Thất bại theo sáu dạng: thiếu dấu ngoặc nhọn, dấu phẩy thừa ở cuối, sai kiểu dữ liệu, ảo giác ra các trường không có, bị cắt cụt do chạm giới hạn token, hoặc rò rỉ văn bản đệm như "Đây là JSON của bạn:".

**Cách tiếp cận thứ hai: xác thực sau khi sinh (validate after generation).** Cho phép mô hình sinh tự do, phân tích cú pháp, xác thực theo lược đồ, và thử lại nếu thất bại. Đáng tin cậy hơn nhưng tốn kém — bạn phải trả phí cho mỗi lần thử lại, và các lỗi cắt cụt token tiêu tốn thêm một lượt xử lý cho mỗi lần xuất hiện.

**Cách tiếp cận thứ ba: giải mã ràng buộc (constrained decoding).** Nhà cung cấp thực thi lược đồ ngay tại thời điểm giải mã (decode time). Các token không hợp lệ bị che mặt nạ (masked out) khỏi phân phối lấy mẫu xác suất. Đầu ra được đảm bảo tuyệt đối sẽ phân tích cú pháp thành công và hợp lệ theo lược đồ. Khả năng thất bại chỉ co về duy nhất một chế độ: từ chối (refusal - khi mô hình quyết định rằng đầu vào không thể khớp với lược đồ).

Mọi nhà cung cấp mô hình hàng đầu năm 2026 đều cung cấp một hình thức nào đó của cách tiếp cận thứ ba:

- **OpenAI.** `response_format: {type: "json_schema", strict: true}` kèm theo trường `refusal` trong phản hồi nếu mô hình từ chối.
- **Anthropic.** Thực thi lược đồ trên các đầu vào `tool_use`; không có `stop_reason: "refusal"`, nhưng sự kiện `end_turn` mà không có lượt gọi công cụ nào chính là tín hiệu nhận biết.
- **Gemini.** `responseSchema` ở cấp độ yêu cầu; năm 2026 Gemini cung cấp các ràng buộc ngữ pháp ở cấp độ token cho các kiểu dữ liệu được chọn.
- **Pydantic AI.** `output_type=InvoiceModel` phát ra một `RunResult` có cấu trúc được định kiểu theo `InvoiceModel`.
- **Zod (TypeScript).** Bộ phân tích cú pháp lúc runtime xác thực đầu ra của nhà cung cấp dựa trên lược đồ Zod; kết hợp cùng với `beta.chat.completions.parse` của OpenAI.

Điểm mấu chốt chung: khai báo lược đồ một lần và thực thi nó xuyên suốt từ đầu đến cuối.

## Khái niệm cốt lõi

### JSON Schema 2020-12 — Ngôn ngữ chung

Mọi nhà cung cấp đều chấp nhận JSON Schema 2020-12. Các cấu trúc bạn sẽ sử dụng nhiều nhất:

- `type`: một trong các kiểu `object`, `array`, `string`, `number`, `integer`, `boolean`, `null`.
- `properties`: ánh xạ từ tên trường sang lược đồ con.
- `required`: danh sách tên các trường bắt buộc phải xuất hiện.
- `enum`: tập hợp đóng các giá trị hợp lệ được phép.
- `minimum` / `maximum` (cho số), `minLength` / `maxLength` / `pattern` (cho chuỗi).
- `items`: lược đồ con áp dụng cho mọi phần tử trong mảng.
- `additionalProperties`: đặt `false` để cấm các trường dư thừa không khai báo (mặc định tùy thuộc vào từng chế độ).

Chế độ nghiêm ngặt của OpenAI bổ sung thêm ba yêu cầu: mọi thuộc tính phải được liệt kê trong `required`, `additionalProperties: false` ở mọi cấp, và không có `$ref` nào chưa được giải quyết. Nếu vi phạm các điều kiện này, API sẽ trả về lỗi HTTP 400 ngay tại thời điểm gửi yêu cầu.

### Pydantic, cầu nối trong Python

Pydantic v2 tạo ra JSON Schema từ các mô hình dạng dataclass thông qua phương thức `model_json_schema()`. Pydantic AI bao bọc cơ chế này để bạn chỉ cần viết:

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

và framework agent sẽ tự động dịch chuyển lược đồ này thành chế độ strict mode của OpenAI, `input_schema` của Anthropic hoặc `responseSchema` của Gemini tại biên. Đầu ra của mô hình trả về dưới dạng một thực thể (instance) `Invoice` có định kiểu rõ ràng. Các lỗi xác thực sẽ ném ra ngoại lệ `ValidationError` kèm theo đường dẫn lỗi có định kiểu.

### Zod, cầu nối trong TypeScript

Zod (`z.object({customer: z.string(), ...})`) là giải pháp tương đương trong TypeScript. SDK Node của OpenAI phơi bày `zodResponseFormat(Invoice)` để chuyển đổi trực tiếp sang tải trọng JSON Schema của API.

### Các trường hợp từ chối (Refusals)

Chế độ nghiêm ngặt không thể ép buộc mô hình phải trả lời bằng mọi giá. Nếu đầu vào không thể khớp với lược đồ (ví dụ: "email là một bài thơ, không phải hóa đơn"), mô hình sẽ phát ra một trường `refusal` chứa lý do từ chối. Mã nguồn của bạn phải xử lý trường hợp này như một kết quả chuẩn mực hạng nhất (first-class outcome), chứ không phải một sự cố crash. Việc từ chối cũng rất hữu ích như một tín hiệu an toàn: một mô hình được yêu cầu trích xuất số thẻ tín dụng từ một email chứa nội dung bảo vệ sẽ trả về một lý do từ chối đi kèm giải thích an toàn.

### Giải mã ràng buộc trong thế giới mã nguồn mở

Các bản triển khai mô hình trọng số mở (open-weights) sử dụng ba kỹ thuật:

1. **Giải mã dựa trên ngữ pháp (Grammar-based decoding)** (`outlines`, `guidance`, `lm-format-enforcer`): xây dựng một máy trạng thái hữu hạn tất định (deterministic finite automaton - DFA) từ lược đồ; tại mỗi bước sinh, che mặt nạ (mask) các logit của những token vi phạm FSM.
2. **Che mặt nạ logit bằng bộ phân tích cú pháp JSON (Logit masking with a JSON parser)**: chạy một bộ phân tích cú pháp JSON dạng luồng đồng bộ theo từng bước cùng với mô hình; tại mỗi bước, tính toán tập hợp các token kế tiếp hợp lệ.
3. **Giải mã suy đoán kèm bộ kiểm chứng (Speculative decoding with a verifier)**: một mô hình phác thảo (draft model) gọn nhẹ đề xuất các token, và bộ kiểm chứng sẽ thực thi sự tuân thủ lược đồ.

Các nhà cung cấp thương mại áp dụng một trong những kỹ thuật này bên dưới hệ thống. Trạng thái công nghệ năm 2026 cho tốc độ nhanh hơn cả việc sinh văn bản tự do đối với các đầu ra có cấu trúc ngắn, và có tốc độ tương đương đối với các đầu ra dài.

### Ba chế độ lỗi (The three failure modes)

1. **Lỗi phân tích cú pháp (Parse error).** Đầu ra không phải là JSON hợp lệ. Không thể xảy ra ở chế độ strict mode. Nhưng vẫn có thể xảy ra trên các nhà cung cấp không bật strict.
2. **Vi phạm lược đồ (Schema violation).** Đầu ra phân tích cú pháp thành công nhưng vi phạm các ràng buộc của lược đồ. Không thể xảy ra ở chế độ strict mode. Rất phổ biến khi không bật strict.
3. **Từ chối (Refusal).** Mô hình khước từ trả lời. Phải được xử lý như một kết quả có định kiểu rõ ràng.

### Chiến lược thử lại (Retry strategy)

Khi bạn hoạt động ngoài chế độ nghiêm ngặt (dùng tool use của Anthropic, OpenAI không bật strict, Gemini đời cũ), mẫu thiết kế phục hồi sẽ là:

```text
generate -> parse -> validate -> if fail, inject error and retry, max 3x
```

Một lần thử lại thường là đủ. Ba lần thử lại sẽ khắc phục được các lỗi chập chờn (flakes) của mô hình yếu. Vượt quá ba lần là dấu hiệu của một lược đồ kém chất lượng: mô hình không thể đáp ứng nó cho một số đầu vào nhất định, và lời nhắc hoặc lược đồ cần phải được chỉnh sửa lại.

### Hỗ trợ mô hình kích thước nhỏ (Small-model support)

Giải mã ràng buộc hoạt động rất hiệu quả trên các mô hình nhỏ. Một mô hình mở 3B tham số có thực thi ngữ pháp có thể vượt trội hơn một mô hình 70B tham số chỉ dùng nhắc lệnh thuần túy trong các tác vụ trích xuất có cấu trúc. Đây là lý do cốt lõi vì sao đầu ra có cấu trúc lại mang tính sống còn đối với môi trường sản xuất: nó tách rời độ tin cậy khỏi kích thước mô hình.

```figure
constrained-decoding
```

## Thực hành (Use It)

Tệp `code/main.py` cung cấp một bộ xác thực JSON Schema 2020-12 tối thiểu viết thuần túy bằng thư viện chuẩn (các kiểu dữ liệu, required, enum, min/max, pattern, items, additionalProperties). Nó bao bọc một lược đồ `Invoice` và truyền một đầu ra LLM giả lập qua bộ xác thực, minh họa các đường dẫn lỗi phân tích cú pháp, vi phạm lược đồ và từ chối. Hãy thay thế đầu ra giả lập bằng phản hồi thực tế của bất kỳ nhà cung cấp nào trên môi trường sản xuất.

Những điểm cần chú ý:

- Bộ xác thực trả về một danh sách `[ValidationError]` có định kiểu gồm đường dẫn (path) và thông báo (message). Đây chính là định dạng bạn muốn đưa vào lời nhắc thử lại.
- Nhánh từ chối (refusal branch) KHÔNG thực hiện thử lại. Nó ghi log và trả về một đối tượng từ chối có định kiểu. Phase 14 · 09 sử dụng các lượt từ chối như một tín hiệu an toàn.
- Kiểm tra `additionalProperties: false` được kích hoạt trên đầu vào kiểm thử đối kháng, cho thấy lý do vì sao chế độ nghiêm ngặt triệt tiêu hoàn toàn các trường bị ảo giác.

## Đóng gói sản phẩm (Ship It)

Bài học này tạo ra artifact `outputs/skill-structured-output-designer.md`. Khi tiếp nhận một mục tiêu trích xuất văn bản tự do (hóa đơn, vé hỗ trợ khách hàng, sơ yếu lý lịch, v.v.), kỹ năng này sẽ tạo ra một lược đồ JSON Schema 2020-12 tương thích với chế độ strict mode và một mô hình Pydantic phản chiếu chuẩn xác lược đồ đó, kèm theo các khung xử lý thử lại và từ chối có định kiểu.

## Bài tập thực hành (Exercises)

1. Chạy `code/main.py`. Thêm một trường hợp kiểm thử thứ tư có `total_usd` là một số âm. Xác nhận rằng bộ xác thực từ chối trường hợp này với đường dẫn ràng buộc `minimum`.

2. Mở rộng bộ xác thực để hỗ trợ `oneOf` đi kèm một trường phân biệt (discriminator). Trường hợp thường gặp: `line_item` là sản phẩm hoặc dịch vụ, được gắn thẻ theo trường `kind`. Chế độ nghiêm ngặt có những quy tắc rất tinh vi ở đây; hãy xem hướng dẫn về đầu ra có cấu trúc của OpenAI.

3. Viết cùng lược đồ Invoice đó dưới dạng một Pydantic BaseModel và so sánh đầu ra của `model_json_schema()` với lược đồ viết tay của bạn. Xác định một trường mà Pydantic tự động thiết lập mặc định nhưng phiên bản viết tay lại bỏ qua.

4. Đo lường tỷ lệ từ chối (refusal rates). Xây dựng mười đầu vào không thể trích xuất được (lời bài hát, một chứng minh toán học, một email trống) và chạy chúng qua một nhà cung cấp thực tế có bật strict mode. Đếm số lượt từ chối so với các đầu ra bị ảo giác. Đây là dữ liệu thực tế (ground truth) của bạn cho các lần thử lại nhận biết từ chối.

5. Đọc tài liệu hướng dẫn về đầu ra có cấu trúc của OpenAI từ đầu đến cuối. Xác định một cấu trúc mà nó nghiêm cấm trong chế độ strict mode nhưng JSON Schema thông thường lại cho phép. Sau đó thiết kế một lược đồ sử dụng cấu trúc bị cấm đó theo cách không thiết yếu và tái cấu trúc (refactor) lại để tương thích với strict mode.

## Thuật ngữ then chốt (Key Terms)

| Thuật ngữ | Cách mọi người thường gọi | Ý nghĩa thực sự |
|-----------|---------------------------|-----------------|
| JSON Schema 2020-12 | "Đặc tả lược đồ" | Phương ngữ lược đồ bản thảo IETF mà mọi nhà cung cấp hiện đại đều sử dụng |
| Chế độ nghiêm ngặt (Strict mode) | "Đảm bảo lược đồ" | Cờ của OpenAI thực thi lược đồ thông qua kỹ thuật giải mã ràng buộc |
| Giải mã ràng buộc (Constrained decoding) | "Che mặt nạ logit" | Thực thi tại thời điểm giải mã nhằm che mặt nạ các token kế tiếp không hợp lệ |
| Khước từ / Từ chối (Refusal) | "Mô hình từ chối" | Kết quả có định kiểu khi đầu vào không thể khớp với lược đồ |
| Lỗi phân tích cú pháp (Parse error) | "JSON không hợp lệ" | Đầu ra không thể phân tích cú pháp thành JSON; bất khả thi dưới chế độ strict |
| Vi phạm lược đồ (Schema violation) | "Sai cấu trúc" | Phân tích JSON thành công nhưng vi phạm các ràng buộc kiểu / required / enum / miền giá trị |
| `additionalProperties: false` | "Không cho phép trường thừa" | Cấm các trường lạ không xác định; bắt buộc trong chế độ strict của OpenAI |
| Pydantic BaseModel | "Đầu ra có định kiểu" | Lớp trong Python phát ra và xác thực JSON Schema |
| Lược đồ Zod (Zod schema) | "Kiểu đầu ra TypeScript" | Lược đồ runtime trong TS dùng để xác thực đầu ra của nhà cung cấp |
| Thực thi ngữ pháp (Grammar enforcement) | "Giải mã ràng buộc mã nguồn mở" | Che mặt nạ logit dựa trên máy trạng thái hữu hạn FSM, ví dụ như trong outlines / guidance |

## Tài liệu đọc thêm (Further Reading)

- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — chế độ strict mode, các trường hợp từ chối và yêu cầu lược đồ
- [OpenAI — Introducing structured outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) — bài viết ra mắt tháng 8 năm 2024 giải thích về bảo đảm trong giải mã
- [Pydantic AI — Output](https://ai.pydantic.dev/output/) — liên kết `output_type` có định kiểu tuần tự hóa sang từng nhà cung cấp
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) — đặc tả chuẩn tắc
- [Microsoft — Structured outputs in Azure OpenAI](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) — ghi chú triển khai cho doanh nghiệp và các lưu ý về chế độ strict mode
