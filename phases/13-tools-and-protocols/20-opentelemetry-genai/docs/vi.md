# OpenTelemetry GenAI — Giám sát Dấu vết các Lệnh gọi Công cụ Đầu-cuối (Tracing Tool Calls End-to-End)

> Một tác tử gọi năm công cụ, ba MCP server và hai tác tử con. Bạn cần một dấu vết truy vết (trace) duy nhất xuyên suốt toàn bộ quá trình đó. Quy ước ngữ nghĩa OpenTelemetry GenAI (OpenTelemetry GenAI semantic conventions — với các thuộc tính ổn định từ v1.37 trở lên) là tiêu chuẩn của năm 2026, được hỗ trợ nguyên bản bởi Datadog, Langfuse, Arize Phoenix, OpenLLMetry và AgentOps. Bài học này xác định các thuộc tính bắt buộc, phân tích cấu trúc phân cấp span (span hierarchy: agent → LLM → tool), và cung cấp một bộ phát span (span emitter) bằng thư viện chuẩn Python mà bạn có thể tích hợp vào bất kỳ bộ trích xuất OTel (OTel exporter) nào.

**Loại:** Xây dựng (Build)
**Ngôn ngữ:** Python (thư viện chuẩn, bộ phát span OTel)
**Yêu cầu tiên quyết:** Phase 13 · 07 (MCP server), Phase 13 · 08 (MCP client)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Gọi tên các thuộc tính OTel GenAI bắt buộc cho một span của LLM và một span thực thi công cụ (tool-execution span).
- Xây dựng cấu trúc phân cấp trace bao quát vòng lặp tác tử (agent loop), lệnh gọi LLM, lệnh gọi công cụ và điều phối client MCP.
- Quyết định nội dung nào cần thu thập (thông qua cơ chế chủ động kích hoạt - opt-in) so với nội dung cần ẩn danh/che giấu (mặc định - redact).
- Phát các span tới một bộ thu thập cục bộ (Jaeger, Langfuse) mà không cần viết lại mã nguồn của công cụ.

## Vấn đề đặt ra

Một phiên gỡ lỗi từ tháng 2 năm 2026: người dùng phản ánh "tác tử của tôi đôi khi mất tới 30 giây để phản hồi; nhưng những lần khác chỉ mất 3 giây." Không có dấu vết truy vết (traces). Nhật ký ghi chép (logs) chỉ hiển thị lệnh gọi LLM, nhưng không ghi nhận việc điều phối công cụ, không có thông tin khứ hồi của MCP server, cũng không có tác tử con. Bạn chỉ có thể đoán mò. Cuối cùng bạn phát hiện: một MCP server thỉnh thoảng bị treo trong quá trình khởi động nguội (cold-start).

Nếu không có cơ chế giám sát dấu vết đầu-cuối (end-to-end tracing), bạn không thể tìm ra nguyên nhân này. OTel GenAI giải quyết triệt để vấn đề đó.

Các quy ước đã được chuẩn hóa trong giai đoạn 2025-2026 thuộc nhóm quy ước ngữ nghĩa của OpenTelemetry (semantic-conventions group). Chúng định nghĩa các tên thuộc tính ổn định để Datadog, Langfuse, Phoenix, OpenLLMetry và AgentOps đều phân tích cú pháp chung một định dạng span. Bạn chỉ cần gắn mã giám sát (instrument) một lần; dữ liệu có thể chuyển đến bất kỳ hạ tầng phụ trợ (backend) nào.

## Khái niệm cốt lõi

### Cấu trúc phân cấp span (Span hierarchy)

```text
agent.invoke_agent  (gốc, span INTERNAL)
  - llm.chat       (span CLIENT)
  - tool.execute   (span INTERNAL)
    - mcp.call     (span CLIENT)
  - llm.chat       (span CLIENT)
  - subagent.invoke (span INTERNAL)
```

Toàn bộ quá trình được lồng ghép dưới một mã định danh vết duy nhất (`trace id`). Các mã định danh span (`span id`) liên kết các mối quan hệ cha - con.

### Các thuộc tính bắt buộc

Theo đặc tả semconv giai đoạn 2025-2026:

- `gen_ai.operation.name` — `"chat"`, `"text_completion"`, `"embeddings"`, `"execute_tool"`, `"invoke_agent"`.
- `gen_ai.provider.name` — `"openai"`, `"anthropic"`, `"google"`, `"azure_openai"`.
- `gen_ai.request.model` — chuỗi định danh mô hình được yêu cầu (ví dụ: `"gpt-4o-2024-08-06"`).
- `gen_ai.response.model` — mô hình thực tế đã xử lý và phản hồi.
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`.
- `gen_ai.response.id` — mã phản hồi của nhà cung cấp dùng để đối chiếu tương quan.

Đối với các span công cụ (tool spans):

- `gen_ai.tool.name` — định danh công cụ.
- `gen_ai.tool.call.id` — định danh lệnh gọi cụ thể.
- `gen_ai.tool.description` — mô tả công cụ (tùy chọn).

Đối với các span tác tử (agent spans):

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`.

### Phân loại span (Span kinds)

- `SpanKind.CLIENT` dành cho các lệnh gọi vượt qua ranh giới tiến trình (nhà cung cấp LLM, MCP server).
- `SpanKind.INTERNAL` dành cho các bước lặp nội bộ và việc thực thi công cụ của chính tác tử.

### Thu thập nội dung theo cơ chế chủ động (Opt-in content capture)

Theo mặc định, các span chỉ mang các số liệu đo lường (metrics) và thời gian thực thi (timing) — không chứa prompt hay nội dung hoàn thành (completions). Các payload kích thước lớn và dữ liệu nhận dạng cá nhân (PII) đều bị tắt theo mặc định. Hãy thiết lập biến môi trường `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` và các biến môi trường thu thập nội dung cụ thể để ghi nhận dữ liệu này. Cần xem xét cẩn trọng trước khi kích hoạt trên môi trường sản xuất (production).

### Sự kiện trên span (Events on spans)

Các sự kiện ở cấp độ token có thể được ghi nhận dưới dạng sự kiện trên span:

- `gen_ai.content.prompt` — các tin nhắn đầu vào.
- `gen_ai.content.completion` — các tin nhắn đầu ra.
- `gen_ai.content.tool_call` — lệnh gọi công cụ được ghi lại.

Các sự kiện được sắp xếp theo trình tự thời gian bên trong span để hỗ trợ tái hiện chi tiết (replay).

### Bộ trích xuất dữ liệu (Exporters)

Các span OTel được xuất tới:

- **Jaeger / Tempo:** Mã nguồn mở, triển khai tại chỗ (on-premise).
- **Langfuse:** Chuyên biệt cho khả năng quan sát LLM (LLM-observability); trực quan hóa mức tiêu thụ token.
- **Arize Phoenix:** Kết hợp đánh giá (evals) và giám sát dấu vết (tracing).
- **Datadog:** Thương mại; phân tích nguyên bản các thuộc tính `gen_ai.*`.
- **Honeycomb:** Cơ sở dữ liệu định hướng cột; tối ưu cho truy vấn phân tích.

Tất cả đều giao tiếp qua OTLP, định dạng truyền tải trên đường truyền. Tầng mã nguồn của bạn hoàn toàn không bị phụ thuộc vào hạ tầng đích.

### Lan truyền ngữ cảnh qua MCP (Propagation across MCP)

Khi một MCP client gọi tới server, hãy chèn tiêu đề W3C `traceparent` vào yêu cầu. Streamable HTTP hỗ trợ các tiêu đề chuẩn. Stdio không hỗ trợ tiêu đề HTTP nguyên bản; lộ trình năm 2026 của đặc tả đề cập đến việc thêm trường `_meta.traceparent` trong các lệnh gọi JSON-RPC.

Cho đến khi tiêu chuẩn đó được phát hành chính thức: hãy đưa `traceparent` vào trường `_meta` của mỗi yêu cầu một cách thủ công. Server sẽ ghi lại trace id vào nhật ký.

### Số liệu đo lường (Metrics)

Song song với các span, quy ước ngữ nghĩa GenAI định nghĩa các số liệu:

- `gen_ai.client.token.usage` — biểu đồ tần suất (histogram).
- `gen_ai.client.operation.duration` — biểu đồ tần suất.
- `gen_ai.tool.execution.duration` — biểu đồ tần suất.

Sử dụng các số liệu này cho các bảng điều khiển (dashboards) không đòi hỏi chi tiết theo từng lệnh gọi riêng lẻ.

### Tầng AgentOps (AgentOps layer)

AgentOps (thành lập năm 2024) chuyên biệt hóa về khả năng quan sát GenAI. Công cụ này bao bọc các framework phổ biến (LangGraph, Pydantic AI, CrewAI) để tự động phát ra các span OTel. Hữu ích nếu ngăn xếp công nghệ của bạn sử dụng một framework được hỗ trợ; nếu không, hãy sử dụng giải pháp gắn mã thủ công (manual instrumentation).

```figure
t3-span-waterfall
```

## Sử dụng trong thực tế (Use It)

Tệp `code/main.py` phát ra các span có cấu trúc chuẩn OTel ra stdout (theo định dạng tương tự OTLP-JSON) cho một tác tử thực hiện gọi một LLM, điều phối hai công cụ và thực hiện một chu trình khứ hồi qua MCP. Không cần bộ trích xuất thực tế — bài học tập trung vào cấu trúc của span và tập hợp thuộc tính. Bạn có thể dán đầu ra vào một công cụ xem tương thích OTLP hoặc đọc trực tiếp.

Những điểm cần quan sát:

- Mã vết `trace id` được chia sẻ xuyên suốt tất cả các span.
- Các liên kết cha - con được mã hóa qua trường `parentSpanId`.
- Các thuộc tính bắt buộc `gen_ai.*` được điền đầy đủ.
- Việc thu thập nội dung mặc định bị tắt; có một kịch bản kích hoạt thông qua biến môi trường.

## Chuyển giao sản phẩm (Ship It)

Bài học này tạo ra `outputs/skill-otel-genai-instrumentation.md`. Với bất kỳ cơ sở mã tác tử nào, kỹ năng này sẽ sinh ra một kế hoạch gắn mã quan sát: vị trí cần thêm span, các thuộc tính cần điền và các bộ trích xuất đích cần hướng tới.

## Bài tập

1. Chạy `code/main.py`. Đếm số lượng span và xác định span nào là CLIENT so với INTERNAL.

2. Bật tính năng thu thập nội dung (thông qua biến môi trường) và xác nhận rằng các sự kiện `gen_ai.content.prompt` và `gen_ai.content.completion` xuất hiện. Phân tích tác động đối với dữ liệu nhận dạng cá nhân (PII).

3. Thêm số liệu đo lường thực thi công cụ `gen_ai.tool.execution.duration` và phát ra mẫu dữ liệu histogram cho mỗi lệnh gọi.

4. Lan truyền `traceparent` từ một span tác tử cha vào trường `_meta.traceparent` của yêu cầu MCP. Xác minh rằng MCP server nhìn thấy cùng một trace id.

5. Đọc đặc tả OTel GenAI semconv. Xác định một thuộc tính được liệt kê trong semconv mà mã nguồn bài học này CHƯA phát ra. Hãy bổ sung nó.

## Thuật ngữ then chốt

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật chuẩn xác |
|-----------|----------------------|---------------------------|
| OTel | "OpenTelemetry" | Chuẩn mở cho dấu vết truy vết, số liệu đo lường và nhật ký |
| GenAI semconv | "Quy ước ngữ nghĩa GenAI" | Tên các thuộc tính ổn định cho các span LLM / công cụ / tác tử |
| `gen_ai.*` | "Không gian tên thuộc tính" | Tất cả thuộc tính GenAI đều chia sẻ tiền tố này |
| Span | "Thao tác có bấm giờ" | Một đơn vị công việc có thời điểm bắt đầu, kết thúc và các thuộc tính |
| Trace | "Dòng dõi xuyên span" | Cây phân cấp các span chia sẻ chung một trace id |
| SpanKind | "CLIENT / SERVER / INTERNAL" | Gợi ý về chiều hướng và ranh giới của span |
| OTLP | "Giao thức OpenTelemetry" | Định dạng đường truyền dành cho các bộ trích xuất (exporters) |
| Nội dung chọn tham gia (Opt-in content) | "Thu thập prompt / completion" | Tắt theo mặc định; dùng biến môi trường để kích hoạt |
| traceparent | "Tiêu đề chuẩn W3C" | Lan truyền ngữ cảnh dấu vết xuyên qua các dịch vụ |
| Bộ trích xuất (Exporter) | "Trình chuyển dữ liệu tới backend" | Thành phần gửi các span tới Jaeger / Datadog / các hệ thống khác |

## Tài liệu đọc thêm

- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — Các quy ước chuẩn cho span, số liệu và sự kiện GenAI
- [OpenTelemetry — GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — Danh sách thuộc tính span cho LLM và thực thi công cụ
- [OpenTelemetry — GenAI agent spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) — Span `invoke_agent` ở cấp độ tác tử
- [open-telemetry/semantic-conventions — GenAI spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) — Nguồn tham chiếu gốc lưu trữ trên GitHub
- [Datadog — LLM OTel semantic convention](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) — Hướng dẫn tích hợp thực tế trên môi trường sản xuất
