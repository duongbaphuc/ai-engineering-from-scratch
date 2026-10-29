# Đồ án Tốt nghiệp 11 — Bảng điều khiển Giám sát Khả năng Quan sát & Đánh giá LLM

> Langfuse chuyển sang mô hình lõi mở (open-core). Arize Phoenix công bố các ánh xạ quy ước ngữ nghĩa GenAI (GenAI semantic conventions) cho năm 2026. Cả Helicone và Braintrust đều tập trung mạnh vào việc quy kết chi phí trên từng người dùng (per-user cost attribution). OpenLLMetry của Traceloop trở thành bộ công cụ đo kiểm (SDK instrumentation) thực tế chuẩn mực. Cấu trúc kiến trúc trong môi trường sản xuất gồm ClickHouse cho dữ liệu truy vết (traces), Postgres cho siêu dữ liệu (metadata), Next.js cho giao diện người dùng, cùng một loạt tác vụ đánh giá (DeepEval, RAGAS, LLM-judge) vận hành trên các trace được lấy mẫu. Đồ án tốt nghiệp này yêu cầu bạn tự lưu trữ (self-host) một hệ thống hoàn chỉnh, thu nạp dữ liệu từ ít nhất bốn họ SDK khác nhau, và chứng minh khả năng phát hiện sự cố thoái lui (regression) được cấy vào trong thời gian dưới 5 phút.

**Type:** Capstone  
**Languages:** TypeScript (UI), Python / TypeScript (ingest + evals), SQL (ClickHouse)  
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools), Phase 17 (infrastructure), Phase 18 (safety)  
**Phases exercised:** P11 · P13 · P17 · P18  
**Time:** ~25 giờ  

## Vấn đề đặt ra

Mọi đội ngũ AI vận hành lưu lượng thực tế trong môi trường sản xuất vào năm 2026 đều duy trì một mặt phẳng quan sát (observability plane) song hành cùng mô hình: Quy kết chi phí, phát hiện ảo giác, giám sát độ trôi dạt dữ liệu, cảnh báo tấn công bẻ khóa (jailbreak), bảng điều khiển cam kết mức độ dịch vụ (SLO), và cảnh báo rò rỉ thông tin nhận dạng cá nhân (PII). Các giải pháp tham chiếu nguồn mở — Langfuse, Phoenix, OpenLLMetry — đã cùng hội tụ về quy chuẩn ngữ nghĩa GenAI của OpenTelemetry (OpenTelemetry GenAI semantic conventions) làm lược đồ thu nạp chuẩn. Giờ đây bạn có thể đo kiểm các ứng dụng dùng OpenAI, Anthropic, Google, LangChain, LlamaIndex và vLLM chỉ với một bộ SDK duy nhất và phát ra các span tương thích hoàn toàn.

Bạn sẽ xây dựng một bảng điều khiển tự lưu trữ có khả năng thu nạp dữ liệu từ ít nhất bốn họ SDK, thực thi một tập hợp các tác vụ đánh giá trên các vết truy vết được lấy mẫu, phát hiện độ trôi dạt và kích hoạt cảnh báo. Tiêu chuẩn đo lường: khi cấy một lỗi thoái lui chủ động (một lời nhắc bắt đầu làm lộ thông tin PII), bảng điều khiển phải phát hiện và bắn cảnh báo trong thời gian dưới 5 phút.

## Khái niệm cốt lõi

Cơ chế thu nạp sử dụng giao thức OTLP HTTP. SDK sinh ra các span tuân thủ quy ước ngữ nghĩa GenAI: `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.response.id`, `llm.prompts`, `llm.completions`. Các span được nạp vào ClickHouse phục vụ phân tích dữ liệu dạng cột tốc độ cao; siêu dữ liệu (người dùng, phiên làm việc, ứng dụng) được lưu trữ tại Postgres.

Các tác vụ đánh giá (evals) chạy dưới dạng công việc hàng loạt (batch jobs) trên các vết truy vết được lấy mẫu. DeepEval chấm điểm độ trung thực (faithfulness), tính độc hại (toxicity), và độ liên quan của câu trả lời (answer relevance). RAGAS chấm điểm các chỉ số truy xuất khi trace mang theo ngữ cảnh tìm kiếm. Các LLM-judge tùy biến sẽ thực hiện kiểm tra đặc thù theo miền (rò rỉ PII, phản hồi vi phạm chính sách). Kết quả đánh giá được ghi ngược trở lại ClickHouse dưới dạng các span đánh giá (eval spans) liên kết trực tiếp với trace cha.

Cơ chế phát hiện độ trôi dạt theo dõi phân phối trong không gian embedding theo thời gian (sử dụng chỉ số ổn định quần thể - Population Stability Index - PSI hoặc độ phân kỳ KL trên embedding của lời nhắc) kết hợp với xu hướng điểm đánh giá. Cảnh báo được gửi tới Prometheus Alertmanager rồi chuyển tiếp sang Slack / PagerDuty. Giao diện người dùng được xây dựng bằng Next.js 15 kết hợp thư viện biểu đồ Recharts.

## Kiến trúc hệ thống

```text
production apps:
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  OpenTelemetry SDK with GenAI semconv
       |
       v  OTLP HTTP
  collector (ingest, sample, fan-out)
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 archive
   (spans)       (metadata)  (raw events)
       |
       +---> eval jobs (DeepEval, RAGAS, LLM-judge)
       |     sampled or all-trace
       |     write eval spans back
       |
       +---> drift detector (PSI / KL on prompt embeddings)
       |
       +---> Prometheus metrics -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 dashboard (Recharts)
```

## Ngăn xếp công nghệ

- **Thu nạp dữ liệu (Ingest):** OpenTelemetry SDK + quy ước ngữ nghĩa GenAI; phương thức truyền tải OTLP HTTP
- **Bộ thu thập (Collector):** OpenTelemetry Collector tích hợp bộ xử lý lấy mẫu phần đuôi (tail-sampling processor) nhằm tối ưu chi phí
- **Lưu trữ:** ClickHouse cho span, Postgres cho siêu dữ liệu, S3 cho lưu trữ thô sự kiện
- **Đánh giá (Evals):** DeepEval, RAGAS 0.2, gói đánh giá Arize Phoenix, LLM-judge tùy biến
- **Phát hiện trôi dạt (Drift):** Đo PSI / KL trên embedding lời nhắc gộp chung (dùng sentence-transformers) định kỳ hàng tuần
- **Hệ thống cảnh báo:** Prometheus Alertmanager -> Slack / PagerDuty
- **Giao diện (UI):** Next.js 15 App Router + Recharts + server actions
- **SDK hỗ trợ sẵn:** OpenAI, Anthropic, Google GenAI, LangChain, LlamaIndex, vLLM

```figure
ce-otel-drift
```

## Từng bước xây dựng

1. **Cấu hình Collector.** Thiết lập OpenTelemetry Collector với bộ tiếp nhận OTLP HTTP, bộ lấy mẫu phần đuôi giữ lại 100% trace lỗi và 10% trace thành công, cùng các bộ xuất (exporters) đẩy về ClickHouse và S3.

2. **Lược đồ cơ sở dữ liệu ClickHouse.** Tạo bảng `spans` với các cột tương ứng quy ước ngữ nghĩa GenAI: `gen_ai_system`, `gen_ai_request_model`, `input_tokens`, `output_tokens`, `latency_ms`, `prompt_hash`, `trace_id`, `parent_span_id`, kèm một túi JSON chứa các payload dài. Bổ sung các chỉ mục phụ (secondary indexes) theo user_id và app_id.

3. **Kiểm thử độ bao phủ SDK.** Viết một ứng dụng khách mẫu sử dụng từng SDK (OpenAI, Anthropic, Google, LangChain, LlamaIndex, vLLM) tích hợp tự động đo kiểm OpenLLMetry. Xác thực từng SDK đều tạo ra các span GenAI chuẩn mực lưu trữ thành công vào ClickHouse.

4. **Tác vụ đánh giá (Eval jobs).** Thiết lập tác vụ định kỳ đọc các trace được lấy mẫu trong 15 phút gần nhất và chạy đánh giá DeepEval về độ trung thực, tính độc hại và độ liên quan của câu trả lời. Đầu ra là các eval span được liên kết chặt chẽ với trace gốc.

5. **LLM-judge tùy biến.** Xây dựng bộ chấm điểm rò rỉ PII: nhận một phản hồi, gọi LLM đóng vai trò bảo vệ để chấm điểm xác suất rò rỉ PII. Những phản hồi có điểm nguy cơ cao sẽ được đưa vào hàng đợi phân loại xử lý (triage queue).

6. **Phát hiện trôi dạt.** Tác vụ hàng tuần tính toán chỉ số PSI giữa các embedding lời nhắc thu thập trong tuần với đường cơ sở 4 tuần trước đó. Nếu chỉ số PSI vượt ngưỡng, kích hoạt cảnh báo.

7. **Bảng điều khiển giao diện.** Xây dựng ứng dụng Next.js 15 với các trang chức năng: Tổng quan (spans/giây, chi phí/người dùng, độ trễ p95), Dấu vết truy vết (tìm kiếm + biểu đồ thác nước waterfall), Đánh giá (xu hướng độ trung thực, tính độc hại), Độ trôi dạt (biểu đồ PSI theo thời gian), Cảnh báo.

8. **Chuỗi cảnh báo.** Bộ xuất Prometheus đọc các chỉ số tổng hợp điểm đánh giá và phân vị độ trễ; Alertmanager định tuyến cảnh báo mức cảnh báo (warning) về Slack và mức nguy cấp (critical) về PagerDuty.

9. **Kiểm thử thăm dò thoái lui.** Cấy một lỗi chủ động: chatbot được đánh giá bắt đầu làm lộ số an sinh xã hội giả định trong 1% số lượt phản hồi. Đo lường thời gian trung bình để phát hiện (Mean Time To Detect - MTTD): tính từ thời điểm lỗi được triển khai cho đến khi nhận được cảnh báo trên Slack. Việc phát hiện là phần cốt lõi mà bảng điều khiển đảm nhiệm trong chỉ số thời gian trung bình để khắc phục (MTTR).

## Vận hành thực tế

```console
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  accepted 1 trace, 3 spans
[clickhouse] inserted 3 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      weekly PSI 0.08 (below 0.2 threshold)
[ui]         live at https://obs.example.com
```

## Bàn giao sản phẩm

Tệp `outputs/skill-llm-observability.md` mô tả sản phẩm bàn giao. Với một ứng dụng LLM bất kỳ, bảng điều khiển sẽ thu nạp các trace, thực thi đánh giá, cảnh báo khi có trôi dạt dữ liệu, và trực quan hóa chi tiết chi phí trên từng người dùng trên giao diện Next.js.

| Trọng số | Tiêu chí | Phương pháp đo lường |
|:-:|---|---|
| 25 | Độ bao phủ lược đồ Trace | Số lượng họ SDK tạo ra các span GenAI chuẩn hóa (mục tiêu: 6+) |
| 20 | Tính chính xác của bài đánh giá | Điểm DeepEval / RAGAS đối chiếu với tập dữ liệu gán nhãn thủ công |
| 20 | Trải nghiệm người dùng Dashboard | Thời gian MTTD khi xảy ra lỗi thoái lui cấy chủ động (mục tiêu dưới 5 phút) |
| 20 | Chi phí & Quy mô | Duy trì khả năng thu nạp ổn định ở mức 1.000 spans/giây không ứ đọng hàng đợi |
| 15 | Hệ thống cảnh báo & Phát hiện trôi dạt | Chuỗi Prometheus/Alertmanager hoạt động thông suốt từ đầu đến cuối |
| **100** | | |

## Bài tập mở rộng

1. Bổ sung cơ chế đo kiểm tùy biến cho framework Haystack. Xác thực các span chuẩn mực được lưu trữ vào ClickHouse với đầy đủ các thuộc tính `gen_ai.*` chính xác.

2. Hoán đổi DeepEval bằng bộ đánh giá của Phoenix trên cùng các trace dữ liệu. Đo lường độ lệch điểm số giữa hai công cụ đánh giá.

3. Nâng cấp bộ phát hiện trôi dạt: tính toán chỉ số PSI riêng biệt cho từng app-id thay vì tính gộp toàn cục. Hiển thị đường dấu vết trôi dạt riêng của từng ứng dụng.

4. Xây dựng trang "Tác động người dùng" (User Impact): trực quan hóa chi phí trên từng người dùng và tỷ lệ lỗi trên từng người dùng kèm biểu đồ thu nhỏ sparkline.

5. Thiết kế chính sách lấy mẫu phần đuôi giữ lại 100% trace có điểm độc hại (toxicity) > 0.5 cộng thêm 10% lấy mẫu phân tầng của phần còn lại. Đo lường độ thiên vị lấy mẫu phát sinh.

## Thuật ngữ cốt lõi

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật thực tế |
|---|---|---|
| GenAI semconv | "Thuộc tính LLM trong OTel" | Đặc tả OpenTelemetry năm 2025 cho các thuộc tính span LLM (hệ thống, mô hình, số token) |
| Tail sampling | "Lấy mẫu sau khi hoàn tất" | Bộ thu thập quyết định giữ lại hay loại bỏ trace sau khi nó kết thúc (có thể xem xét lỗi) |
| PSI | "Chỉ số ổn định quần thể" | Chỉ số đo độ trôi dạt so sánh hai phân phối; giá trị > 0.2 thường biểu thị trôi dạt đáng kể |
| LLM-judge | "Mô hình làm giám khảo" | Một LLM chấm điểm đầu ra của một LLM khác theo barem định sẵn (độ trung thực, độc hại, PII) |
| Tail-sampling policy | "Quy tắc lưu giữ" | Quy tắc quyết định lưu giữ hay loại bỏ trace; dựa trên lỗi phát sinh và tỷ lệ lấy mẫu |
| Eval span | "Trace đánh giá liên kết" | Span con chứa điểm đánh giá được liên kết trực tiếp với span gọi LLM ban đầu |
| Cost per user | "Kinh tế đơn vị" | Chi phí tài chính tính theo USD quy cho một user_id trong một khung thời gian; chỉ số sản phẩm trọng yếu |

## Tài liệu đọc thêm

- [Langfuse](https://github.com/langfuse/langfuse) — nền tảng quan sát mã nguồn lõi mở tham chiếu
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — giải pháp tham chiếu với khả năng hỗ trợ giám sát độ trôi dạt mạnh mẽ
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry) — bộ SDK tự động đo kiểm
- [Quy ước ngữ nghĩa OpenTelemetry GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — lược đồ thu nạp chuẩn hóa
- [Helicone](https://www.helicone.ai) — nền tảng quan sát dạng hosted
- [Braintrust](https://www.braintrust.dev) — nền tảng tập trung vào đánh giá
- [Tài liệu ClickHouse](https://clickhouse.com/docs) — kho lưu trữ span dạng cột tốc độ cao
- [DeepEval](https://github.com/confident-ai/deepeval) — thư viện đánh giá mô hình
