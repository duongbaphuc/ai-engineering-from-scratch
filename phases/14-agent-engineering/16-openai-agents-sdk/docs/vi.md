# OpenAI Agents SDK: Bàn giao, Rào chắn và Truy vết

> OpenAI Agents SDK là framework đa tác tử gọn nhẹ được xây dựng trên nền tảng Responses API. Năm nguyên thủy cốt lõi: Agent (Tác tử), Handoff (Bàn giao), Guardrail (Rào chắn an toàn), Session (Phiên làm việc), Tracing (Truy vết). Các lượt bàn giao thực chất là công cụ có dạng tên `transfer_to_<agent>`. Rào chắn được kích hoạt ở đầu vào hoặc đầu ra. Cơ chế truy vết được bật mặc định.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 06 (Tool Use)
**Time:** ~75 minutes

## Mục tiêu học tập

- Kể tên năm nguyên thủy cốt lõi của OpenAI Agents SDK.
- Giải thích cơ chế bàn giao (handoffs): tại sao chúng được mô hình hóa dưới dạng công cụ, định dạng tên mà mô hình nhìn thấy, và ngữ cảnh được chuyển giao như thế nào.
- Phân biệt giữa rào chắn đầu vào (input guardrails), rào chắn đầu ra (output guardrails) và rào chắn công cụ (tool guardrails); giải thích chế độ song song (`run_in_parallel`) so với chế độ chặn (blocking mode).
- Triển khai một môi trường thực thi bằng thư viện chuẩn (stdlib) với cơ chế bàn giao + rào chắn + truy vết theo kiểu span.

## Vấn đề

Các tác tử không có khả năng ủy quyền (delegate) rành mạch sẽ kết thúc bằng việc nhồi nhét mọi thứ vào một câu lệnh prompt duy nhất. Các tác tử thiếu rào chắn bảo vệ sẽ làm rò rỉ thông tin nhận dạng cá nhân (PII), sinh đầu ra vi phạm chính sách hoặc rơi vào vòng lặp vô tận. SDK của OpenAI chuẩn hóa ba nguyên thủy giúp việc vận hành hệ thống đa tác tử trở nên khả thi và kiểm soát được.

## Khái niệm

### Năm nguyên thủy

1. **Agent (Tác tử).** LLM + chỉ dẫn (instructions) + công cụ (tools) + cơ chế bàn giao (handoffs).
2. **Handoff (Bàn giao).** Ủy quyền nhiệm vụ cho một tác tử khác. Được biểu diễn trước mô hình như một công cụ mang tên `transfer_to_<agent_name>`.
3. **Guardrail (Rào chắn an toàn).** Xác thực ở đầu vào (chỉ áp dụng cho tác tử đầu tiên), đầu ra (chỉ áp dụng cho tác tử cuối cùng), hoặc khi kích hoạt công cụ (trên từng công cụ hàm).
4. **Session (Phiên làm việc).** Tự động lưu trữ lịch sử hội thoại qua các lượt tương tác.
5. **Tracing (Truy vết).** Các span tích hợp sẵn ghi nhận các lượt sinh phản hồi của LLM, lệnh gọi công cụ, các lượt bàn giao và rào chắn an toàn.

### Bàn giao dưới dạng công cụ

Mô hình nhìn thấy `transfer_to_billing_agent` trong danh sách công cụ của nó. Việc gọi công cụ này sẽ báo hiệu cho môi trường thực thi (runtime) thực hiện:

1. Sao chép ngữ cảnh hội thoại (hoặc thu gọn ngữ cảnh thông qua tính năng beta `nest_handoff_history`).
2. Khởi tạo tác tử đích với bộ chỉ dẫn tương ứng của nó.
3. Tiếp tục lượt chạy với tác tử đích.

Đây chính là mẫu hình giám sát viên (supervisor pattern - Bài 13 / Bài 28) được thương phẩm hóa.

### Rào chắn an toàn

Ba biến thể:

- **Rào chắn đầu vào (Input guardrails).** Chạy trên đầu vào của tác tử đầu tiên. Từ chối các yêu cầu không an toàn hoặc nằm ngoài phạm vi trước khi thực hiện bất kỳ lệnh gọi LLM nào.
- **Rào chắn đầu ra (Output guardrails).** Chạy trên đầu ra của tác tử cuối cùng. Bắt các rò rỉ PII, vi phạm chính sách hoặc các phản hồi sai lệch định dạng.
- **Rào chắn công cụ (Tool guardrails).** Chạy trên từng công cụ hàm (function tool). Xác thực đối số, kiểm tra quyền hạn và kiểm toán quá trình thực thi.

Chế độ hoạt động:

- **Song song (Parallel)** (mặc định). LLM của rào chắn chạy đồng thời cùng LLM chính. Độ trễ đuôi (tail latency) thấp hơn. Nếu rào chắn bị kích hoạt vi phạm, phần công việc của LLM chính sẽ bị hủy bỏ (lãng phí token).
- **Chặn (Blocking)** (`run_in_parallel=False`). LLM của rào chắn chạy trước. Nếu phát hiện vi phạm, sẽ không có token nào bị lãng phí cho lệnh gọi chính.

Cơ chế ngắt vi phạm (tripwires) sẽ ném ra ngoại lệ `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered`.

### Truy vết

Bật theo mặc định. Mọi lần sinh LLM, gọi công cụ, bàn giao và rào chắn an toàn đều phát ra một span. Đặt biến môi trường `OPENAI_AGENTS_DISABLE_TRACING=1` để tắt. Phương thức `add_trace_processor(processor)` phân phối (fan-out) các span tới backend của riêng bạn song song với backend của OpenAI.

### Phiên làm việc

`Session` lưu trữ lịch sử hội thoại vào một backend (SQLite, Redis, hoặc tùy biến). Lệnh gọi `Runner.run(agent, input, session=session)` sẽ tự động tải và thêm dữ liệu vào phiên.

### Những cạm bẫy thường gặp

- **Trôi lệch bàn giao (Handoff drift).** Tác tử A bàn giao cho Tác tử B, rồi B lại bàn giao ngược lại cho A. Cần bổ sung một bộ đếm số bước nhảy (hop counter).
- **Bỏ qua rào chắn (Guardrail bypass).** Rào chắn công cụ chỉ kích hoạt trên các công cụ hàm; các công cụ tích hợp sẵn (đọc tệp, gọi web) cần có chính sách kiểm soát riêng biệt.
- **Lạm dụng truy vết (Over-tracing).** Nội dung nhạy cảm vô tình bị ghi vào các span. Hãy kết hợp với các quy tắc ghi nhận nội dung của OTel GenAI (Bài 23) — lưu trữ bên ngoài và chỉ tham chiếu bằng ID.

```figure
ae-agent-handoff
```

## Xây dựng

`code/main.py` triển khai hình thái SDK chỉ dùng thư viện chuẩn (stdlib):

- `Agent`, `FunctionTool`, `Handoff` (dưới dạng một công cụ hàm mang ngữ nghĩa chuyển giao).
- `Runner` với rào chắn đầu vào/đầu ra/công cụ, điều phối bàn giao và bộ đếm bước nhảy.
- Bộ phát span đơn giản minh họa cấu trúc dấu vết thực thi.
- Một tác tử phân loại (triage agent) sẽ bàn giao công việc sang bộ phận thanh toán (billing) hoặc hỗ trợ (support) dựa trên truy vấn của người dùng; rào chắn sẽ ngắt vi phạm trên một đầu vào cụ thể.

Chạy thử:

```bash
python3 code/main.py
```

Dấu vết thực thi hiển thị hai lần bàn giao thành công, một lần rào chắn đầu vào ngắt vi phạm, và cây span mô phỏng tương tự những gì SDK thật phát ra.

## Sử dụng

- **OpenAI Agents SDK** cho các sản phẩm ưu tiên hệ sinh thái OpenAI.
- **Claude Agent SDK** (Bài 17) cho các sản phẩm ưu tiên hệ sinh thái Claude.
- **LangGraph** (Bài 13) khi bạn cần quản lý trạng thái tường minh và khôi phục bền vững.
- **Tùy biến (Custom)** khi bạn cần toàn quyền kiểm soát (giao tiếp bằng giọng nói, đa nhà cung cấp, triển khai liên kết phân tán).

## Chuyển giao

`outputs/skill-agents-sdk-scaffold.md` tạo khung ứng dụng Agents SDK hoàn chỉnh với tác tử phân loại, các lượt bàn giao, rào chắn đầu vào/đầu ra/công cụ, kho lưu phiên và bộ xử lý truy vết.

## Bài tập

1. Thêm bộ đếm bước nhảy bàn giao: từ chối sau N lần chuyển giao liên tiếp. Ghi nhận hành vi trong dấu vết thực thi.
2. Triển khai tùy chọn `nest_handoff_history` — thu gọn các thông điệp trước đó thành một bản tóm tắt trước khi bàn giao.
3. Viết một rào chắn đầu ra ở chế độ chặn (blocking). So sánh độ trễ đối với các prompt kích hoạt vi phạm so với các prompt vượt qua an toàn.
4. Nối `add_trace_processor` vào một bộ ghi nhật ký JSON (JSON logger). Cấu trúc phát ra cho mỗi span là gì?
5. Đọc tài liệu SDK. Chuyển đổi mã đồ chơi thư viện chuẩn sang thư viện `openai-agents-python`. Bạn đã mô hình hóa sai những điểm nào?

## Thuật ngữ then chốt

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|-----------|-------------------|-----------------|
| Tác tử (Agent) | "LLM + chỉ dẫn" | Kiểu Agent trong SDK; sở hữu các công cụ và bàn giao |
| Bàn giao (Handoff) | "Chuyển giao (Transfer)" | Công cụ mà mô hình gọi để ủy quyền cho tác tử khác |
| Rào chắn (Guardrail) | "Kiểm tra chính sách" | Bước xác thực ở đầu vào / đầu ra / khi gọi công cụ |
| Dây bẫy vi phạm (Tripwire) | "Kích hoạt rào chắn" | Ngoại lệ được ném ra khi rào chắn từ chối nội dung |
| Phiên làm việc (Session) | "Kho lịch sử" | Bộ nhớ hội thoại được lưu bền vững giữa các lượt chạy |
| Truy vết (Tracing) | "Các Span" | Khả năng quan sát tích hợp cho LLM + công cụ + bàn giao + rào chắn |
| Rào chắn chế độ chặn (Blocking guardrail) | "Kiểm tra tuần tự" | Rào chắn chạy trước; không lãng phí token khi có vi phạm |
| Rào chắn song song (Parallel guardrail) | "Kiểm tra đồng thời" | Rào chắn chạy song song; giảm độ trễ, nhưng lãng phí token khi vi phạm |

## Tài liệu đọc thêm

- [Tài liệu OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — các nguyên thủy, bàn giao, rào chắn, truy vết
- [Tổng quan về Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — đối trọng tương đương bên phía Claude
- [Anthropic, Xây dựng các Tác tử Hiệu quả](https://www.anthropic.com/research/building-effective-agents) — khi nào thực sự cần dùng đến cơ chế bàn giao
- [Quy ước ngữ nghĩa OpenTelemetry GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — chuẩn mà các span của Agents SDK ánh xạ vào
