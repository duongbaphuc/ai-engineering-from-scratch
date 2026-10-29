# Quy ước ngữ nghĩa GenAI trong OpenTelemetry

> Nhóm chuyên trách GenAI của OpenTelemetry (OpenTelemetry GenAI SIG, thành lập tháng 4/2024) định nghĩa lược đồ tiêu chuẩn cho dữ liệu đo từ xa của tác tử (agent telemetry). Tên khoảng vết (span names), các thuộc tính (attributes), và quy tắc thu thập nội dung (content-capture rules) được chuẩn hóa đồng bộ giữa các nhà cung cấp, giúp các dấu vết thực thi của tác tử (agent traces) mang cùng một ngữ nghĩa nhất quán trên Datadog, Grafana, Jaeger hay Honeycomb.

**Type:** Learn + Build  
**Languages:** Python (stdlib)  
**Prerequisites:** Phase 14 · 13 (LangGraph), Phase 14 · 24 (Observability Platforms)  
**Time:** ~60 phút

## Mục tiêu học tập

- Gọi tên các danh mục khoảng vết (span categories) trong GenAI: mô hình/máy khách (model/client), tác tử (agent), công cụ (tool).
- Phân biệt các khoảng vết `invoke_agent` thuộc loại CLIENT (máy khách) và INTERNAL (nội bộ) cùng ngữ cảnh ứng dụng tương ứng.
- Liệt kê các thuộc tính GenAI cấp cao nhất: tên nhà cung cấp (provider name), mã định danh mô hình yêu cầu (request model), ID nguồn dữ liệu (data-source ID).
- Giải thích giao ước thu thập nội dung: kích hoạt chủ động (opt-in), biến môi trường `OTEL_SEMCONV_STABILITY_OPT_IN`, và khuyến nghị lưu trữ tham chiếu ngoài.

## Vấn đề đặt ra

Mỗi nhà cung cấp công cụ tự sáng tạo ra quy ước đặt tên khoảng vết riêng biệt. Hậu quả là các đội ngũ vận hành hệ thống (Ops) phải xây dựng các bảng điều khiển giám sát riêng lẻ cho từng khung phát triển. Nhóm chuyên trách GenAI của OpenTelemetry đã giải quyết triệt để tình trạng phân mảnh này bằng cách chuẩn hóa một thông số kỹ thuật duy nhất cho toàn bộ hệ sinh thái.

## Khái niệm cốt lõi

### Các danh mục khoảng vết (Span categories)

1. **Khoảng vết mô hình / máy khách (Model / client spans):** Bao bọc các lệnh gọi LLM thô. Được phát ra bởi các SDK chính thức của nhà cung cấp (Anthropic, OpenAI, Bedrock) hoặc các bộ điều hợp mô hình (model adapters) trong khung tác tử.
2. **Khoảng vết tác tử (Agent spans):** Gồm `create_agent` (khi đối tượng tác tử được khởi tạo) và `invoke_agent` (khi tác tử được kích hoạt thực thi).
3. **Khoảng vết công cụ (Tool spans):** Mỗi lần gọi công cụ ứng với một khoảng vết riêng; liên kết trực tiếp với khoảng vết tác tử theo quan hệ cha - con (parent-child).

### Quy ước đặt tên khoảng vết tác tử

- Tên vết: `invoke_agent {gen_ai.agent.name}` nếu tác tử được đặt tên cụ thể; mặc định rút gọn là `invoke_agent`.
- Loại vết (Span kind):
  - **CLIENT** — dùng cho các dịch vụ tác tử từ xa được gọi qua mạng (OpenAI Assistants API, Bedrock Agents).
  - **INTERNAL** — dùng cho các khung tác tử chạy trực tiếp trong tiến trình nội bộ (LangChain, CrewAI, vòng lặp ReAct cục bộ).

### Các thuộc tính then chốt

- `gen_ai.provider.name` — định danh nhà cung cấp: `anthropic`, `openai`, `aws.bedrock`, `google.vertex`.
- `gen_ai.request.model` — mã định danh mô hình được yêu cầu ban đầu.
- `gen_ai.response.model` — mô hình thực tế đã xử lý và phản hồi (có thể khác với mô hình yêu cầu do cơ chế định tuyến thông minh).
- `gen_ai.agent.name` — định danh duy nhất của tác tử.
- `gen_ai.operation.name` — loại thao tác: `chat`, `completion`, `invoke_agent`, `tool_call`.
- `gen_ai.data_source.id` — dùng cho mô hình RAG: định danh tập ngữ liệu hoặc kho lưu trữ đã được truy vấn.

Các quy ước công nghệ chuyên biệt cũng được định nghĩa cho Anthropic, Azure AI Inference, AWS Bedrock và OpenAI.

### Giao ước thu thập nội dung (Content capture)

Quy tắc cốt lõi: Các thư viện giám sát KHÔNG ĐƯỢC PHÉP thu thập dữ liệu đầu vào/đầu ra theo mặc định. Việc ghi lại nội dung bắt buộc phải kích hoạt chủ động (opt-in) thông qua các thuộc tính:

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

Khuyến nghị kiến trúc cho môi trường sản xuất: Lưu trữ toàn bộ nội dung văn bản ra hạ tầng lưu trữ bên ngoài (Amazon S3, hệ thống nhật ký chuyên biệt), trên các khoảng vết chỉ ghi lại các mã định danh tham chiếu (pointer IDs, không chứa văn bản thô). Đây chính là biện pháp phòng thủ chống đầu độc nội dung (content-poisoning defense, xem Bài 27) được tích hợp trực tiếp vào hạ tầng khả năng quan sát (observability).

### Độ ổn định của thông số kỹ thuật

Hầu hết các quy ước đều ở trạng thái thử nghiệm tính đến tháng 3 năm 2026. Bạn có thể kích hoạt chế độ xem trước ổn định bằng biến môi trường:

```bash
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog từ phiên bản v1.37+ đã ánh xạ tự nhiên các thuộc tính GenAI này vào lược đồ LLM Observability của họ. Các nền tảng lưu trữ khác (Grafana, Honeycomb, Jaeger) hỗ trợ tiếp nhận và truy vấn trực tiếp trên các thuộc tính thô.

### Những sai lầm thường gặp

- **Ghi toàn bộ prompt vào khoảng vết:** Dẫn đến rò rỉ thông tin định danh cá nhân (PII), khóa bí mật (secrets), và dữ liệu khách hàng vào hệ thống truy vết mà đội vận hành có thể đọc được. Luôn lưu trữ ra ngoài.
- **Thiếu thuộc tính `gen_ai.provider.name`:** Bảng điều khiển giám sát đa nhà cung cấp sẽ bị gãy vỡ khi không thể phân loại nguồn gốc chi phí và độ trễ.
- **Khoảng vết không liên kết cha-con:** Sinh ra các khoảng vết công cụ mồ côi (orphaned spans). Luôn đảm bảo truyền bá ngữ cảnh (context propagation).
- **Không cấu hình biến môi trường ổn định:** Các thuộc tính có thể bị đổi tên khi nền tảng backend nâng cấp phiên bản.

```figure
ae-genai-span-tree
```

## Thực hành Xây dựng (Build It)

Tệp `code/main.py` hiện thực một bộ phát khoảng vết bằng thư viện chuẩn tuân thủ quy ước GenAI:

- Cấu trúc `Span` hỗ trợ đầy đủ lược đồ thuộc tính GenAI.
- Lớp `Tracer` cung cấp phương thức `start_span` và quản lý ngữ cảnh lồng nhau.
- Một phiên chạy tác tử theo kịch bản phát ra các vết: `create_agent`, `invoke_agent` (INTERNAL), các vết công cụ con, và các vết `chat` cho lệnh gọi LLM.
- Chế độ thu thập nội dung an toàn: lưu prompt vào kho lưu trữ ngoài và chỉ gán ID tham chiếu lên khoảng vết.

Chạy thử nghiệm:

```bash
python3 code/main.py
```

Kết quả: một cây phân cấp vết (span tree) chứa đầy đủ các thuộc tính GenAI theo chuẩn, cùng cấu trúc "kho lưu trữ ngoài" minh họa cơ chế tham chiếu nội dung an toàn.

## Ứng dụng Thực tế (Use It)

- **Datadog LLM Observability** (v1.37+): Ánh xạ tự động các thuộc tính theo chuẩn.
- **Langfuse / Phoenix / Opik:** (Xem Bài 24) Tự động gắn thiết bị giám sát (instrumentation) cho hệ sinh thái tác tử.
- **Jaeger / Honeycomb / Grafana Tempo:** Tiếp nhận các vết OTel thô; xây dựng bảng điều khiển dựa trên các thuộc tính GenAI.
- **Tự lưu trữ (Self-hosted):** Vận hành OTel Collector tích hợp bộ xử lý GenAI chuyên dụng.

## Xuất xưởng Sản phẩm (Ship It)

Tệp `outputs/skill-otel-genai.md` tích hợp các khoảng vết OTel GenAI vào một tác tử sẵn có, thiết lập cấu hình mặc định thu thập nội dung và cơ chế lưu trữ tham chiếu ngoài.

## Bài tập

1. Gắn mã đo kiểm vào vòng lặp ReAct ở Bài 01 với vết `invoke_agent` (INTERNAL) kèm các vết công cụ con. Gửi dữ liệu tới một phiên bản Jaeger đang chạy.
2. Bổ sung chế độ thu thập nội dung chỉ lưu tham chiếu: lưu toàn bộ prompt vào cơ sở dữ liệu SQLite, các thuộc tính của khoảng vết chỉ lưu ID bản ghi.
3. Đọc đặc tả kỹ thuật của `gen_ai.data_source.id`. Tích hợp thuộc tính này vào tác vụ tìm kiếm bộ nhớ Mem0 ở Bài 09.
4. Thiết lập biến môi trường `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` và kiểm chứng các thuộc tính không bị bộ thu thập (collector) đổi tên.
5. Xây dựng bảng điều khiển phân tích: "những lỗi công cụ nào có tương quan chặt chẽ với những dòng mô hình nào" chỉ dựa trên các thuộc tính GenAI.

## Thuật ngữ then chốt

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật chính xác |
|---|---|---|
| GenAI SIG | "Nhóm GenAI OpenTelemetry" | Nhóm làm việc của OTel chịu trách nhiệm định nghĩa chuẩn lược đồ |
| invoke_agent | "Vết tác tử" | Tên chuẩn của khoảng vết đại diện cho một phiên thực thi tác tử |
| Khoảng vết CLIENT | "Lệnh gọi từ xa" | Khoảng vết đại diện cho lệnh gọi đến dịch vụ tác tử từ xa qua mạng |
| Khoảng vết INTERNAL | "Chạy nội bộ" | Khoảng vết đại diện cho phiên chạy tác tử nội bộ trong tiến trình |
| gen_ai.provider.name | "Nhà cung cấp" | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | "Nguồn RAG" | Kho ngữ liệu hoặc kho lưu trữ mà lệnh truy xuất chạm tới |
| Thu thập nội dung (Content capture) | "Ghi nhật ký prompt" | Cơ chế kích hoạt chủ động để lưu thông điệp; nên lưu trữ ngoài trong sản xuất |
| Kích hoạt độ ổn định | "Chế độ xem trước" | Biến môi trường dùng để ghim các quy ước thực nghiệm |

## Đọc thêm

- [Quy ước ngữ nghĩa GenAI của OpenTelemetry](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — tài liệu đặc tả kỹ thuật chính thức
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — hỗ trợ mặc định các khoảng vết GenAI
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — tích hợp sẵn các khoảng vết OTel
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — cơ chế truyền bá ngữ cảnh vết theo chuẩn W3C
