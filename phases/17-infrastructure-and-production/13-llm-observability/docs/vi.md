# Lựa chọn Ngăn xếp Khả năng Quan sát LLM (LLM Observability Stack Selection)

> Thị trường khả năng quan sát (observability) năm 2026 chia thành hai nhóm chính: Nền tảng phát triển (development platforms - LangSmith, Langfuse, Comet Opik) tích hợp giám sát với đánh giá (evals), quản lý câu lệnh gợi ý (prompt management) và phát lại phiên (session replays). Công cụ cổng kết nối/thu thập dữ liệu đo từ xa (gateway/instrumentation tools - Helicone, SigNoz, OpenLLMetry, Phoenix) tập trung vào dữ liệu đo từ xa (telemetry). Langfuse sở hữu lõi cấp phép MIT với thế cân bằng mã nguồn mở (OSS) mạnh mẽ (50.000 sự kiện/tháng trên cloud miễn phí). Phoenix là công cụ thuần OpenTelemetry theo Giấy phép Elastic License 2.0 — xuất sắc cho trực quan hóa RAG và độ lệch trôi (drift), nhưng không phải là backend lưu trữ dài hạn cho môi trường sản xuất. Arize AX sử dụng tích hợp hồ dữ liệu không sao chép (zero-copy) Iceberg/Parquet với tuyên bố tiết kiệm chi phí gấp 100 lần so với các hệ thống quan sát đơn khối (monolithic). LangSmith dẫn đầu cho hệ sinh thái LangChain/LangGraph, giá $39/người dùng/tháng, chỉ hỗ trợ tự lưu trữ (self-host) ở gói Doanh nghiệp (Enterprise). Helicone hoạt động dựa trên proxy với thời gian thiết lập 15-30 phút, 100.000 yêu cầu/tháng miễn phí, nhưng ít chuyên sâu về dấu vết đa bước của tác tử (agent traces). Mô hình sản xuất phổ biến: Cổng kết nối (Helicone/Portkey) + nền tảng đánh giá (Phoenix/TruLens) được gắn kết thông qua OpenTelemetry.

**Loại:** Tìm hiểu (Learn)  
**Ngôn ngữ:** Python (thư viện chuẩn, bộ mô phỏng lấy mẫu dấu vết đồ chơi - toy trace-sampling simulator)  
**Điều kiện tiên quyết:** Giai đoạn 17 · 08 (Inference Metrics), Giai đoạn 14 (Agent Engineering)  
**Thời lượng:** ~60 phút  

## Mục tiêu Học tập

- Phân biệt giữa nền tảng phát triển (được đóng gói: đánh giá + quản lý prompt + phiên tương tác) và các công cụ cổng kết nối/dữ liệu đo từ xa (chỉ gồm dấu vết [traces] + chỉ số [metrics]).
- Đối chiếu sáu công cụ lớn (Langfuse, LangSmith, Phoenix, Arize AX, Helicone, Opik) theo giấy phép, mô hình định giá và trường hợp sử dụng tối ưu (sweet-spot).
- Giải thích mô hình gắn kết OpenTelemetry cho phép kết hợp công cụ cổng kết nối với một nền tảng đánh giá độc lập.
- Nêu rõ yếu tố phân hóa chi phí năm 2026 (cơ chế zero-copy của Arize AX so với thu nạp đơn khối monolithic) và hệ số tiết kiệm xấp xỉ 100 lần.

## Vấn đề Thực tế

Bạn vừa triển khai một tính năng LLM vào sản xuất. Hệ thống hoạt động tốt. Nhưng bạn hoàn toàn mù tịt về các lỗi prompt, vòng lặp công cụ vô tận, suy giảm độ trễ (latency regressions), chi phí tăng vọt đột ngột, hay tỷ lệ trúng bộ nhớ đệm câu lệnh (prompt-cache hit rate). Bạn tìm kiếm "LLM observability" trên Google và nhận được tám công cụ khác nhau, tất cả đều tuyên bố giải quyết cùng một vấn đề ở ba mức giá khác biệt.

Thực tế chúng không giải quyết cùng một vấn đề. LangSmith trả lời câu hỏi "tại sao lượt chạy LangGraph này lại thất bại?" Phoenix trả lời "đường ống RAG của tôi có đang bị lệch trôi phân phối dữ liệu (drifting) hay không?" Helicone trả lời "ứng dụng nào đang tiêu tốn nhiều token nhất?" Langfuse trả lời "tôi có thể tự lưu trữ (self-host) toàn bộ hệ thống này trên hạ tầng riêng hay không?" Công cụ khác nhau phục vụ các đối tượng khác nhau.

Việc lựa chọn xoay quanh bốn trục: ngăn xếp công nghệ (LangChain? SDK gốc? đa nhà cung cấp?), mức độ chấp nhận giấy phép (chỉ chấp nhận MIT? chấp nhận Elastic? thương mại đóng mã nguồn?), ngân sách (gói miễn phí? $100/tháng? $1000/tháng?), và khả năng tự lưu trữ (bắt buộc? nên có? không bao giờ?).

## Khái niệm

### Hai nhóm công cụ chính

**Nền tảng phát triển (Development platforms)** đóng gói khả năng quan sát cùng các bộ đánh giá (evals), quản lý prompt, lập phiên bản tập dữ liệu (dataset versioning), và phát lại phiên (session replay). Bạn chạy các thử nghiệm, quan sát xem prompt nào hiệu quả nhất, kiểm thử hồi quy tập dữ liệu (dataset-regression) cho prompt mới so với các phiên bản tiền nhiệm chiến thắng. Đại diện tiêu biểu: LangSmith, Langfuse, Comet Opik.

**Công cụ cổng kết nối/dữ liệu đo từ xa (Gateway/telemetry tools)** đo lường và ghi nhận trực tiếp các lượt gọi suy luận — câu lệnh prompt, phản hồi, số token, độ trễ, mô hình, chi phí. Đại diện: Helicone, SigNoz, OpenLLMetry, Phoenix. Tối giản và tinh gọn. Có thể kết hợp với một công cụ đánh giá riêng biệt thông qua OpenTelemetry.

### Langfuse — Cân bằng Mã nguồn mở (OSS)

- Lõi được cấp phép Apache / MIT; hỗ trợ tự lưu trữ (self-host) dễ dàng qua Docker.
- Gói đám mây miễn phí: 50.000 sự kiện/tháng. Gói trả phí: $29/tháng cho nhóm làm việc (team).
- Hỗ trợ đánh giá (evals), quản lý prompt, dấu vết (traces), tập dữ liệu (datasets). Bao quát tốt cả bốn tính năng cốt lõi của nền tảng phát triển.
- Điểm tối ưu: bạn mong muốn các tính năng đẳng cấp tương đương LangSmith nhưng bắt buộc phải tự lưu trữ hoặc duy trì giấy phép nguồn mở.

### Phoenix (Arize) — Ưu tiên Dữ liệu đo từ xa, Thuần OpenTelemetry

- Cấp phép theo Elastic License 2.0; việc tự lưu trữ rất đơn giản.
- Xuất sắc trong việc trực quan hóa RAG và độ lệch trôi (drift). Tích hợp sẵn biểu đồ phân tán trong không gian nhúng (embedding-space scatter plots) như một thành phần hạng nhất.
- Không được thiết kế làm backend lưu trữ bền bỉ trong môi trường sản xuất — chủ yếu phục vụ quan sát trong giai đoạn phát triển.
- Điểm tối ưu: phát triển đường ống RAG, gỡ lỗi độ lệch trôi, ghép nối với một cổng kết nối riêng biệt cho môi trường sản xuất.

### Arize AX — Giải pháp Quy mô lớn

- Sản phẩm thương mại. Tích hợp hồ dữ liệu không sao chép (zero-copy data lake) thông qua định dạng Iceberg/Parquet.
- Tuyên bố rẻ hơn ~100 lần so với các nền tảng quan sát đơn khối (như Datadog) ở quy mô lớn. Bài toán kinh tế: bạn lưu trữ dấu vết trong tệp Parquet của riêng mình trên S3; Arize đọc trực tiếp từ đó mà không cần sao chép hay thu nạp dữ liệu trùng lặp.
- Điểm tối ưu: quy mô >10 triệu dấu vết/ngày, đã có sẵn hồ dữ liệu (data lake), cần bảng điều khiển chuyên biệt cho LLM mà không muốn gánh mức phí cắt cổ của Datadog.

### LangSmith — Ưu tiên Hệ sinh thái LangChain/LangGraph

- Nền tảng thương mại, $39/người dùng/tháng. Chỉ hỗ trợ tự lưu trữ ở gói Doanh nghiệp (Enterprise).
- Dẫn đầu thị trường cho các ngăn xếp sử dụng LangChain và LangGraph. Nếu không dùng hai khung làm việc này, sức hấp dẫn của nó sẽ giảm đi đáng kể.
- Điểm tối ưu: đội ngũ đã cam kết gắn bó với LangChain và sẵn sàng chi trả chi phí bản quyền.

### Helicone — Cổng Tối giản Khả thi (MVP) Dựa trên Proxy

- Thiết lập nhanh chóng trong 15-30 phút chỉ bằng cách chuyển `OPENAI_API_BASE` sang proxy của Helicone.
- Giấy phép MIT; miễn phí 100.000 yêu cầu/tháng, các gói trả phí từ $20/tháng trở lên.
- Tích hợp sẵn cơ chế dự phòng sự cố (failover), bộ nhớ đệm (caching), giới hạn tốc độ (rate limits) — đóng vai trò như một cổng kết nối API (gateway).
- Ít chuyên sâu hơn về dấu vết đa bước phức tạp của tác tử (multi-step agent traces).
- Điểm tối ưu: khởi đầu nhanh, ứng dụng ngăn xếp đơn lẻ, cần kết hợp cả cổng kết nối lẫn khả năng quan sát trong một công cụ.

### Opik (Comet) — Nền tảng Phát triển Mã nguồn mở

- Giấy phép Apache 2.0, hoàn toàn mã nguồn mở (OSS).
- Tập tính năng tương đồng với Langfuse nhưng thừa hưởng nền tảng phân tích từ Comet.
- Điểm tối ưu: các nhóm kỹ sư máy học (ML) đã quen thuộc với Comet và muốn quản lý khả năng quan sát LLM trên cùng một giao diện.

### SigNoz — Giám sát Hiệu năng Ứng dụng Toàn diện (APM) Ưu tiên OpenTelemetry

- Giấy phép Apache 2.0. Xử lý giám sát hiệu năng ứng dụng (APM) tổng quát kết hợp với LLM thông qua OpenTelemetry.
- Điểm tối ưu: khả năng quan sát thống nhất trên toàn bộ các dịch vụ backend và các lượt gọi LLM.

### Lớp keo gắn kết: OpenTelemetry + Quy ước Ngữ nghĩa GenAI

OpenTelemetry đã công bố các quy ước ngữ nghĩa GenAI (GenAI semantic conventions) vào cuối năm 2025 (`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`). Các công cụ tiêu thụ chuẩn OTel có thể tương tác liền mạch với nhau. Mô hình sản xuất chuẩn mực đang hình thành như sau:

1. Phát xuất dữ liệu OTel tuân thủ quy ước GenAI từ mọi lượt gọi LLM.
2. Điều hướng qua cổng kết nối (Helicone / Portkey) cho các tác vụ vận hành thường nhật.
3. Đồng thời gửi bản sao dữ liệu tới nền tảng đánh giá (Phoenix / Langfuse) để kiểm tra các suy thoái chất lượng (regressions).
4. Lưu trữ lâu dài trong hồ dữ liệu (Iceberg) để phân tích chuyên sâu dài hạn thông qua Arize AX hoặc DuckDB.

### Cạm bẫy: Thu thập dữ liệu đo đạc sai tầng kiến trúc

Thu thập dữ liệu đo đạc (instrumenting) ngay bên trong khung tác tử (ví dụ: gắn trực tiếp dấu vết LangSmith vào mã nguồn) sẽ trói buộc ứng dụng của bạn vào khung đó. Thu thập ở tầng HTTP/OpenAI-SDK (thông qua OpenLLMetry hoặc cổng kết nối gateway) đảm bảo tính linh hoạt và khả năng di chuyển mã nguồn cao hơn.

### Lấy mẫu (Sampling) — Không thể lưu giữ mọi thứ

Khi vượt ngưỡng 1 triệu yêu cầu/ngày, chi phí lưu trữ toàn bộ dấu vết (full-trace retention) thậm chí còn tốn kém hơn chi phí gọi LLM. Hãy áp dụng chiến lược lấy mẫu theo quy tắc: 100% lỗi, 100% yêu cầu có chi phí cao, và 5% các yêu cầu thành công thông thường. Luôn giữ lại dữ liệu tổng hợp (aggregates), và chỉ lưu trữ dữ liệu thô cho phần đuôi dài (long tail) của phân phối.

### Các con số cần ghi nhớ

- Gói đám mây miễn phí của Langfuse: 50.000 sự kiện/tháng.
- Chi phí LangSmith: $39/người dùng/tháng.
- Gói miễn phí của Helicone: 100.000 yêu cầu/tháng.
- Tuyên bố của Arize AX: rẻ hơn ~100 lần so với kiến trúc đơn khối ở quy mô lớn.
- Quy ước OpenTelemetry GenAI: phát hành cuối 2025, được ứng dụng rộng rãi trong năm 2026.

```figure
i4-otel-glue
```

## Sử dụng trong Thực tế

Tệp `code/main.py` mô phỏng một ngày tiếp nhận 1 triệu dấu vết qua nhiều chiến lược lưu giữ khác nhau (thu nạp 100%, lấy mẫu ngẫu nhiên, lấy mẫu kết hợp bắt giữ lỗi). Báo cáo chi phí lưu trữ và những thông tin bị mất mát theo từng phương án.

## Sản phẩm Chuyển giao

Bài học này tạo ra tệp kỹ năng `outputs/skill-observability-stack.md`. Dựa trên ngăn xếp công nghệ, quy mô, ngân sách và định hướng cấp phép, công cụ này sẽ đưa ra lựa chọn nền tảng tối ưu nhất cho hệ thống của bạn.

## Bài tập Thực hành

1. Đội ngũ của bạn đang dùng LangChain và muốn có giải pháp quan sát mã nguồn mở tự lưu trữ (self-hosted). Hãy lựa chọn giữa Langfuse hoặc Opik và đưa ra lập luận bảo vệ quyết định.
2. Với lưu lượng 5 triệu dấu vết/ngày và báo giá từ Datadog là $150.000/tháng, hãy tính toán điểm hòa vốn khi chuyển sang Arize AX.
3. Thiết kế một bộ thuộc tính OpenTelemetry GenAI mà tổ chức của bạn nên quy định bắt buộc trên mọi lượt gọi LLM.
4. Hãy lập luận xem liệu chỉ dùng Phoenix có đủ cho môi trường sản xuất hay không. Trong trường hợp nào nó sẽ không đáp ứng được yêu cầu?
5. Helicone tạo ra độ trễ phụ trội (proxy overhead) là 20 ms. Với P99 TTFT là 300 ms, mức trễ này có chấp nhận được không? Điều gì xảy ra nếu SLA yêu cầu độ trễ dưới 100 ms?

## Thuật ngữ Then chốt

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật chuẩn xác |
|---|---|---|
| OpenLLMetry | "OTel cho LLM" | Thư viện đo lường OpenTelemetry mã nguồn mở dành riêng cho LLM |
| GenAI conventions | "Các thuộc tính OTel" | Tên các thuộc tính OTel chuẩn hóa dành cho các lượt gọi LLM |
| LangSmith | "Khả năng quan sát của LangChain" | Nền tảng thương mại tích hợp sâu với hệ sinh thái LangChain |
| Langfuse | "LangSmith mã nguồn mở" | Dự án mã nguồn mở cấp phép MIT với tập tính năng tương tự |
| Phoenix | "Công cụ phát triển của Arize" | Nền tảng đánh giá và phát triển thuần OpenTelemetry |
| Arize AX | "Khả năng quan sát quy mô lớn" | Nền tảng quan sát thương mại zero-copy trên Iceberg/Parquet |
| Helicone | "Khả năng quan sát qua proxy" | Proxy HTTP thu thập dữ liệu đo từ xa LLM kết hợp tính năng gateway |
| Opik | "Comet LLM" | Nền tảng phát triển nguồn mở Apache 2.0 từ Comet |
| Phát lại phiên (Session replay) | "Chạy lại vết (trace rerun)" | Phát lại toàn bộ một phiên làm việc của tác tử cùng các lượt gọi công cụ |
| Đánh giá (Eval) | "Kiểm thử ngoại tuyến (offline test)" | Chạy mô hình/prompt ứng viên trên tập dữ liệu đã gán nhãn |

## Tài liệu Tham khảo thêm

- [SigNoz — Các công cụ quan sát LLM hàng đầu năm 2026](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Phân tích giải pháp thay thế Arize AX](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — Thiết lập Langfuse, LangSmith, Helicone, Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [Quy ước Ngữ nghĩa OpenTelemetry GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Tài liệu hướng dẫn Arize Phoenix](https://docs.arize.com/phoenix)
- [Tài liệu hướng dẫn Helicone](https://docs.helicone.ai/)
