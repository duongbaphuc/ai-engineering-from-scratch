# Giao diện công cụ — Vì sao Agent cần I/O có cấu trúc

> Một mô hình ngôn ngữ chỉ tạo ra các token. Một chương trình thì thực hiện các hành động. Khoảng cách giữa hai thế giới đó chính là giao diện công cụ (tool interface): một khế ước (contract) cho phép mô hình yêu cầu một hành động và máy chủ lưu trữ (host) thực thi hành động đó. Mọi ngăn xếp công nghệ năm 2026 — gọi hàm (function calling) trên OpenAI, Anthropic và Gemini; `tools/call` của MCP; các phần nhiệm vụ (task parts) của A2A — đều là những cách mã hóa khác nhau của cùng một vòng lặp 4 bước. Bài học này định danh vòng lặp đó và trình bày cơ chế tối thiểu cần thiết để vận hành nó.

**Type:** Learn
**Languages:** Python (stdlib, no LLM)
**Prerequisites:** Phase 11 (LLM completion APIs)
**Time:** ~45 phút

## Mục tiêu học tập

- Giải thích lý do vì sao một LLM chỉ có thể tạo văn bản thuần túy lại không thể tự mình thực hiện các hành động can thiệp vào thế giới thực.
- Vẽ sơ đồ vòng lặp gọi công cụ (tool-call loop) gồm 4 bước (mô tả → quyết định → thực thi → quan sát / describe → decide → execute → observe) và nêu rõ bên nào nắm quyền sở hữu mỗi bước.
- Viết định nghĩa công cụ gồm 4 thành phần: tên (name), mô tả (description), lược đồ đầu vào JSON Schema (input schema), và một hàm thực thi tất định (deterministic executor function).
- Phân biệt giữa các công cụ thuần túy (pure tools) và công cụ có tác dụng phụ (side-effecting / consequential tools), đồng thời nêu rõ vì sao sự phân tách này có ý nghĩa sống còn đối với tính an toàn (safety).

## Vấn đề đặt ra

Một LLM phát ra một phân phối xác suất trên token kế tiếp. Đó là toàn bộ bề mặt đầu ra của nó. Nếu bạn hỏi một mô hình trò chuyện: "Thời tiết ở Bengaluru lúc này như thế nào?", nó có thể viết ra một câu văn nghe rất hợp lý, nhưng nó không thể tự kết nối vào một API thời tiết. Câu văn đó có thể đúng do ngẫu nhiên trùng hợp hoặc là thông tin đã lỗi thời từ ba ngày trước.

Thu hẹp khoảng cách đó chính là mục đích của giao diện công cụ (tool interface). Chương trình host — runtime của agent, Claude Desktop, ChatGPT, Cursor, hoặc một tập lệnh tự viết — công bố một danh sách các công cụ có thể gọi được cho mô hình. Khi mô hình quyết định cần phải thực hiện một hành động, nó sẽ phát ra một khối dữ liệu có cấu trúc (structured payload) nêu rõ tên công cụ và các đối số (arguments) của nó. Host phân tích cú pháp khối dữ liệu đó, thực sự chạy công cụ trong thực tế, và đưa kết quả phản hồi lại cho mô hình. Vòng lặp này tiếp diễn cho đến khi mô hình quyết định không cần gọi thêm công cụ nào nữa.

Phiên bản đầu tiên của khế ước này được phát hành vào tháng 6 năm 2023 dưới dạng tham số "functions" của OpenAI. Anthropic tiếp bước với các khối `tool_use` trong Claude 2.1. Gemini bổ sung `functionDeclarations` vài tháng sau đó. Mọi nhà cung cấp hiện nay đều phơi bày cùng một cấu trúc: đầu vào là một danh sách công cụ được định kiểu theo JSON Schema, đầu ra là một lượt gọi công cụ chứa tải trọng JSON. Giao thức Ngữ cảnh Mô hình (Model Context Protocol - MCP, tháng 11 năm 2024) đã tổng quát hóa khế ước này để một sổ đăng ký công cụ (tool registry) duy nhất có thể phục vụ cho mọi mô hình. A2A (tháng 4 năm 2026, v1.0) xếp chồng cùng một cấu tử nguyên thủy này cho việc ủy quyền giữa agent với agent (agent-to-agent delegation).

Vòng lặp 4 bước là bất biến (invariant) nền tảng bên dưới tất cả các công nghệ này. Mọi nội dung khác trong Phase 13 đều là sự phát triển chi tiết dựa trên nền tảng đó.

## Khái niệm cốt lõi

### Bước một: mô tả (describe)

Host khai báo từng công cụ cho mô hình với ba trường dữ liệu. Thành phần thứ tư, bộ thực thi (executor), nằm ở phía host và không bao giờ được gửi tới mô hình.

- **Tên (Name).** Một định danh ổn định, máy đọc được. Ví dụ: `get_weather`, chứ không phải "weather thing".
- **Mô tả (Description).** Một bản tóm tắt ngắn bằng ngôn ngữ tự nhiên. "Sử dụng khi người dùng hỏi về điều kiện thời tiết hiện tại của một thành phố cụ thể. Không sử dụng cho dữ liệu lịch sử."
- **Lược đồ đầu vào (Input schema).** Một đối tượng JSON Schema (bản thảo draft 2020-12) mô tả các đối số của công cụ.

Mô hình nhận danh sách này. Các nhà cung cấp hiện đại sẽ tuần tự hóa (serialize) các khai báo này vào lời nhắc hệ thống (system prompt) bằng một mẫu template đặc thù của từng nhà cung cấp, do đó bạn với tư cách là người gọi chỉ cần làm việc với dạng có cấu trúc.

### Bước hai: quyết định (decide)

Dựa vào thông điệp của người dùng và các công cụ sẵn có, mô hình sẽ chọn một trong ba hành vi:

1. **Trả lời trực tiếp** bằng văn bản. Không có lượt gọi công cụ nào.
2. **Gọi một hoặc nhiều công cụ.** Phát ra các đối tượng gọi có cấu trúc. Dưới thiết lập `parallel_tool_calls: true` (mặc định trên OpenAI và Gemini, tùy chọn kích hoạt trên Anthropic), mô hình có thể phát ra nhiều lượt gọi trong một lượt xử lý (turn).
3. **Từ chối (Refuse).** Đầu ra có cấu trúc ở chế độ nghiêm ngặt (strict mode) có thể tạo ra một khối `refusal` có định kiểu thay vì một lượt gọi.

Một tải trọng gọi công cụ (tool call payload) có ba trường ổn định: định danh lượt gọi `id`, tên công cụ `name`, và một đối tượng `arguments` dạng JSON. `id` tồn tại để host có thể đối chiếu kết quả trả về sau đó với lượt gọi cụ thể, điều này đặc biệt quan trọng khi các lệnh gọi song song trả về không theo thứ tự.

### Bước ba: thực thi (execute)

Host nhận lượt gọi, xác thực các đối số theo lược đồ đã khai báo, và chạy hàm thực thi (executor). Đối số không hợp lệ có nghĩa là mô hình đã ảo giác (hallucinated) ra một trường hoặc dùng sai kiểu dữ liệu — một chế độ lỗi rất phổ biến trên các mô hình nhỏ/yếu. Các hệ thống host trong môi trường sản xuất xử lý theo một trong ba cách khi đối số không hợp lệ: thất bại nhanh (fail fast) và chuyển lỗi này cho mô hình, sửa chữa JSON bằng một bộ phân tích cú pháp ràng buộc (constrained parser), hoặc yêu cầu mô hình thử lại kèm theo thông báo lỗi xác thực trong lời nhắc.

Bản thân executor là mã nguồn thông thường. Python, TypeScript, một lệnh shell, một truy vấn cơ sở dữ liệu. Nó tạo ra một kết quả, thường là một chuỗi văn bản nhưng cũng có thể là bất kỳ giá trị JSON nào hoặc một khối nội dung có cấu trúc (văn bản, hình ảnh, hoặc tham chiếu tài nguyên trong MCP). Kết quả này phải tuần tự hóa được (serializable).

### Bước bốn: quan sát (observe)

Host gắn kết quả công cụ vào ngữ cảnh cuộc hội thoại (dưới dạng một thông điệp có vai trò `tool` cùng với `id` khớp tương ứng) và tái gọi mô hình. Lúc này mô hình đã có đầu ra của công cụ trong ngữ cảnh và có thể đưa ra câu trả lời cuối cùng hoặc yêu cầu thêm các lượt gọi khác. Quá trình này tiếp tục cho đến khi mô hình ngừng phát lệnh gọi hoặc host chạm đến giới hạn an toàn về số vòng lặp.

### Phân tách độ tin cậy (The trust split)

Các công cụ có hai phân loại mang ý nghĩa then chốt đối với tính an toàn:

- **Thuần túy (Pure).** Chỉ đọc, tất định, không có tác dụng phụ (no side effects). Ví dụ: `get_weather`, `search_docs`, `get_current_time`. An toàn khi gọi suy đoán (speculatively).
- **Có hệ quả / Thay đổi trạng thái (Consequential).** Làm biến đổi trạng thái, phát sinh chi phí tiền tệ, chạm vào dữ liệu người dùng. Ví dụ: `send_email`, `delete_file`, `execute_trade`. Bắt buộc phải có cổng kiểm soát (gated).

"Quy tắc số hai" (Rule of Two) năm 2026 của Meta về an toàn agent quy định rằng một lượt xử lý duy nhất chỉ được phép kết hợp tối đa hai trong ba yếu tố: đầu vào không đáng tin cậy, dữ liệu nhạy cảm, hành động có hệ quả. Giao diện công cụ chính là nơi bạn thực thi quy tắc đó — bằng cách từ chối các lượt gọi, yêu cầu người dùng xác nhận, hoặc nâng cấp phạm vi quyền hạn (scopes). Xem Phase 13 · 15 để biết toàn bộ chương về bảo mật và Phase 14 · 09 về các chính sách phân quyền ở cấp độ agent.

### Nơi vòng lặp vận hành

| Ngữ cảnh | Bên mô tả | Bên quyết định | Bên thực thi |
|----------|-----------|----------------|--------------|
| Gọi hàm đơn lượt (OpenAI/Anthropic/Gemini) | Nhà phát triển ứng dụng | LLM | Nhà phát triển ứng dụng |
| MCP | MCP server | LLM thông qua MCP client | MCP server |
| A2A | Nhà phát hành Agent Card | Agent gọi | Agent được gọi |
| Trình duyệt web (Agent gọi hàm) | Tiện ích mở rộng trình duyệt / WebMCP | LLM | Runtime trình duyệt |

Ở mọi nơi, đều là cùng 4 bước đó. Tên các cột có thể thay đổi; cấu trúc thì không.

### Tại sao không chỉ đơn giản nhắc mô hình phát ra JSON?

"Yêu cầu mô hình trả lời bằng JSON" là mẫu thiết kế từ thời kỳ trước khi có tính năng gọi hàm bản địa (native function calling). Nó thất bại khoảng 5 đến 15% thời gian trên các mô hình hàng đầu (frontier models) và tỷ lệ lỗi cao hơn rất nhiều trên các mô hình nhỏ hơn. Các dạng lỗi bao gồm thiếu dấu ngoặc nhọn, dấu phẩy thừa ở cuối, trường bị ảo giác và sai kiểu dữ liệu. Khi đó bạn cần một bước sửa lỗi JSON, một lần thử lại, hoặc một bộ giải mã ràng buộc (constrained decoder).

Gọi hàm bản địa vượt trội hơn vì ba lý do. Thứ nhất, nhà cung cấp huấn luyện mô hình theo phương thức đầu-cuối (end-to-end) trên chính xác cấu trúc lượt gọi, do đó tỷ lệ JSON hợp lệ tăng lên 98 đến 99% ở chế độ nghiêm ngặt (strict mode). Thứ hai, tải trọng lượt gọi nằm trong một khe giao thức riêng biệt, không nằm lẫn trong văn bản tự do — nhờ đó lượt gọi công cụ không bao giờ bị rò rỉ vào câu trả lời hiển thị cho người dùng. Thứ ba, các nhà cung cấp thực thi sự tuân thủ lược đồ bằng phương pháp giải mã ràng buộc (constrained decoding - chế độ strict mode của OpenAI, `tool_use` của Anthropic, `responseSchema` của Gemini). Đầu ra được đảm bảo hợp lệ tuyệt đối theo lược đồ.

Phase 13 · 02 sẽ so sánh chi tiết ba API của các nhà cung cấp đặt cạnh nhau. Phase 13 · 04 sẽ đi sâu vào đầu ra có cấu trúc (structured outputs).

### Cơ chế ngắt mạch (Circuit breakers)

Vòng lặp kết thúc khi mô hình ngừng phát ra các lượt gọi hoặc host chạm đến số lượt xử lý tối đa (maximum turn count). Các hệ thống host trên môi trường sản xuất thường đặt giới hạn này trong khoảng từ 5 đến 20 lượt. Vượt quá ngưỡng đó, gần như chắc chắn bạn đang rơi vào một vòng lặp vô hạn mà mô hình không thể tự thoát ra. Claude Code mặc định là 20; OpenAI Assistants là 10; chế độ agent của Cursor là 25.

Giải pháp ngược lại — các vòng lặp không giới hạn — cứ mỗi sáu tháng lại xuất hiện trong các bài mổ xẻ sự cố (post-mortem) kiểu như "agent đã tiêu tốn $400 tiền gọi API chỉ sau một đêm". Tuyệt đối không đưa vào vận hành thực tế mà thiếu đi một ngưỡng giới hạn.

Phase 14 · 12 sẽ đề cập sâu về phục hồi lỗi và khả năng tự chữa lành (self-healing); Phase 17 sẽ bao quát các giới hạn tốc độ (rate limits) trên môi trường sản xuất.

### Hướng đi tiếp theo của Phase 13

- Bài học 02 đến 05 hoàn thiện bề mặt gọi công cụ ở cấp độ nhà cung cấp.
- Bài học 06 đến 14 tổng quát hóa vòng lặp thành MCP.
- Bài học 15 đến 18 bảo vệ vòng lặp trước các server độc hại, người dùng đối kháng và bề mặt xác thực từ xa không có chứng thực.
- Bài học 19 đến 22 mở rộng mô hình này sang phối hợp giữa các agent (agent-to-agent collaboration), khả năng quan sát (observability), định tuyến (routing) và đóng gói (packaging).
- Bài học 23 phát hành một hệ sinh thái hoàn chỉnh sử dụng toàn bộ các cấu tử nguyên thủy.

Mọi bài học còn lại đều là sự phát triển chi tiết từ vòng lặp 4 bước này. Hãy luôn ghi nhớ nó như một bất biến nền tảng.

```figure
tp-tool-loop
```

## Thực hành (Use It)

Tệp `code/main.py` chạy vòng lặp 4 bước mà không cần LLM. Một hàm "decider" giả lập sẽ đóng vai trò của mô hình bằng cách khớp mẫu (pattern matching) trên thông điệp của người dùng; executor, bộ xác thực lược đồ (schema validator) và khung kiểm thử bước quan sát (observe-step harness) đều là mã thật. Hãy chạy nó để quan sát toàn bộ quy trình phối hợp yêu cầu/phản hồi với trạng thái trung gian có thể in ra màn hình, sau đó thay thế bộ decider giả lập bằng bất kỳ nhà cung cấp thực tế nào trong bài học sau.

Những điểm cần chú ý:

- Sổ đăng ký công cụ (tool registry) lưu giữ bốn trường cho mỗi công cụ: tên, mô tả, lược đồ và tham chiếu executor. Chỉ có ba trường đầu tiên được gửi tới mô hình.
- Bộ xác thực là một tập con tối thiểu của JSON Schema (các kiểu dữ liệu, required, enum, min/max) được viết thuần túy bằng thư viện chuẩn (stdlib). Phase 13 · 04 sẽ cung cấp một bộ xác thực hoàn thiện hơn.
- Vòng lặp giới hạn số lượt lặp ở mức năm lượt. Các agent trên môi trường sản xuất cần chính xác loại cơ chế ngắt mạch này.

## Đóng gói sản phẩm (Ship It)

Bài học này tạo ra artifact `outputs/skill-tool-interface-reviewer.md`. Nhận vào một định nghĩa công cụ dự thảo (tên + mô tả + lược đồ + phác thảo executor), kỹ năng (skill) này sẽ kiểm toán xem định nghĩa đó có phù hợp với vòng lặp hay không: tên có ổn định cho máy đọc không, mô tả có phải là bản tóm tắt cách dùng đầy đủ không, lược đồ có sử dụng JSON Schema 2020-12 chuẩn xác không, và việc phân loại giữa công cụ thuần túy và công cụ có hệ quả có được nêu tường minh hay không.

## Bài tập thực hành (Exercises)

1. Thêm một công cụ thứ tư vào `code/main.py` có tên `get_stock_price(ticker)`. Viết phần mô tả là "Sử dụng khi người dùng hỏi về giá cổ phiếu hiện tại theo mã ticker. Không sử dụng cho giá lịch sử hoặc tóm tắt thị trường." Chạy khung kiểm thử và xác nhận rằng bộ decider giả lập định tuyến các truy vấn nhắc đến mã cổ phiếu sang công cụ mới.

2. Phá vỡ bộ xác thực lược đồ. Truyền vào một lượt gọi có đối tượng `arguments` bị thiếu một trường bắt buộc, và xác nhận rằng host từ chối nó trước khi thực thi. Tiếp đó, truyền vào một lượt gọi có thêm một trường lạ không xác định. Đưa ra quyết định: host nên từ chối hay bỏ qua? Bảo vệ sự lựa chọn của bạn bằng một lập luận về tính an toàn.

3. Phân loại từng công cụ trong khung kiểm thử thành công cụ thuần túy (pure) hoặc công cụ có hệ quả (consequential). Thêm cờ `consequential: true` vào các mục trong sổ đăng ký nếu cần, và sửa đổi vòng lặp để in ra dòng "cần xác nhận với người dùng" bất cứ khi nào một công cụ có hệ quả được chọn. Đây chính là hình thái của cổng xác nhận mà mọi hệ thống host trong thực tế đều cần có.

4. Vẽ sơ đồ vòng lặp 4 bước ra giấy với bảng cột nhà cung cấp ở trên được điền thông tin cho client yêu thích của bạn (Claude Desktop, Cursor, ChatGPT, hoặc một ngăn xếp tự xây dựng). Tham chiếu chéo với biến thể đặc thù của MCP trong Phase 13 · 06.

5. Đọc kỹ hướng dẫn gọi hàm của OpenAI từ đầu đến cuối. Xác định một trường xuất hiện trong yêu cầu (request) nhưng không có trong vòng lặp 4 bước được trình bày ở đây. Giải thích trường đó bổ sung điều gì và tại sao nó mang tính tiện lợi hơn là thiết yếu.

## Thuật ngữ then chốt (Key Terms)

| Thuật ngữ | Cách mọi người thường gọi | Ý nghĩa thực sự |
|-----------|---------------------------|-----------------|
| Công cụ (Tool) | "Một thứ mô hình có thể gọi" | Một bộ ba gồm tên + đầu vào định kiểu JSON-Schema + hàm executor |
| Gọi hàm (Function calling) | "Sử dụng công cụ bản địa" | Hỗ trợ API ở cấp độ nhà cung cấp để phát ra các lượt gọi công cụ có cấu trúc thay vì văn bản tự do |
| Lượt gọi công cụ (Tool call) | "Yêu cầu hành động của mô hình" | Một tải trọng JSON gồm `id`, `name`, `arguments` được phát ra bởi mô hình |
| Kết quả công cụ (Tool result) | "Những gì công cụ trả về" | Đầu ra của executor, được bọc trong một thông điệp có vai trò `tool` với id khớp tương ứng |
| Gọi công cụ song song (Parallel tool calls) | "Nhiều lượt gọi cùng lúc" | Nhiều đối tượng gọi trong một lượt xử lý của mô hình, độc lập và có thể định thứ tự theo id |
| Chế độ nghiêm ngặt (Strict mode) | "Đảm bảo JSON hợp lệ" | Giải mã ràng buộc (constrained decoding) ép buộc đầu ra của mô hình phải khớp chuẩn xác với lược đồ đã khai báo |
| Công cụ thuần túy (Pure tool) | "Công cụ chỉ đọc" | Không có tác dụng phụ; an toàn khi chạy lại |
| Công cụ có hệ quả (Consequential tool) | "Công cụ hành động" | Làm thay đổi trạng thái bên ngoài; đòi hỏi cổng kiểm soát, kiểm toán, hoặc xác nhận từ người dùng |
| Vòng lặp 4 bước (Four-step loop) | "Chu trình gọi công cụ" | mô tả → quyết định → thực thi → quan sát (describe → decide → execute → observe) |
| Máy chủ lưu trữ / Runtime (Host) | "Runtime của agent" | Chương trình nắm giữ sổ đăng ký công cụ, gọi mô hình, và chạy hàm executor |

## Tài liệu đọc thêm (Further Reading)

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — tài liệu tham khảo chuẩn mực cho các khai báo công cụ và định dạng lượt gọi kiểu OpenAI
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — định dạng khối `tool_use` / `tool_result` của Claude
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) — ngữ nghĩa của `functionDeclarations` và các lượt gọi song song trong Gemini
- [Model Context Protocol — Specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28) — đặc tả tổng quát hóa phi trạng thái, độc lập với nhà cung cấp hiện tại của giao diện công cụ
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) — phương ngữ lược đồ mà mọi API công cụ hiện đại đều sử dụng
