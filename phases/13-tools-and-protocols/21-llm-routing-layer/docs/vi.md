# Lớp Định tuyến LLM — LiteLLM, OpenRouter, Portkey (LLM Routing Layer)

> Sự ràng buộc nhà cung cấp (provider lock-in) rất tốn kém. Các khối lượng công việc gọi công cụ (tool-calling workloads) khác nhau sẽ phù hợp với các mô hình khác nhau. Các cổng định tuyến (routing gateways) cung cấp một bề mặt API duy nhất, cơ chế thử lại (retries), chuyển đổi dự phòng (failover), theo dõi chi phí (cost tracking) và các hàng rào bảo vệ (guardrails). Ba giải pháp tiêu biểu định hình năm 2026: LiteLLM (mã nguồn mở tự lưu trữ), OpenRouter (dịch vụ đám mây được quản lý - managed SaaS), Portkey (chuẩn sản xuất cấp doanh nghiệp, được mở mã nguồn vào tháng 3 năm 2026). Bài học này xác định các tiêu chí ra quyết định và phân tích một cổng định tuyến xây dựng bằng thư viện chuẩn Python.

**Loại:** Học (Learn)
**Ngôn ngữ:** Python (thư viện chuẩn, định tuyến + chuyển đổi dự phòng + theo dõi chi phí)
**Yêu cầu tiên quyết:** Phase 13 · 02 (gọi hàm - function calling), Phase 13 · 17 (cổng API - gateways)
**Thời lượng:** ~45 phút

## Mục tiêu học tập

- Phân biệt các tùy chọn định tuyến: tự lưu trữ (self-hosted), được quản lý (managed) và cấp độ sản xuất (production-grade).
- Triển khai một chuỗi dự phòng (fallback chain) có khả năng thử lại khi nhà cung cấp gặp sự cố theo thứ tự ưu tiên xác định.
- Theo dõi chi phí trên từng yêu cầu và mức sử dụng token xuyên suốt các nhà cung cấp khác nhau.
- Đưa ra quyết định lựa chọn giữa LiteLLM, OpenRouter và Portkey dựa trên các ràng buộc thực tế trong môi trường sản xuất.

## Vấn đề đặt ra

Các kịch bản thực tế mà việc định tuyến nhà cung cấp đóng vai trò quyết định:

1. **Chi phí:** Claude Sonnet có chi phí cao gấp 3 lần Haiku. Đối với tác vụ phân loại sơ bộ (triage task), Haiku là đủ; đối với tác vụ tổng hợp thông tin chuyên sâu (synthesis task), Sonnet mới thực sự xứng đáng. Hãy định tuyến theo từng yêu cầu.

2. **Chuyển đổi dự phòng (Failover):** OpenAI gặp sự cố trong một giờ. Mọi yêu cầu đều thất bại. Bạn muốn tự động chuyển hướng dự phòng sang Anthropic mà không cần tái triển khai hệ thống (redeploy).

3. **Độ trễ (Latency):** Giao diện hội thoại trực tiếp cần thời gian phản hồi token đầu tiên (time-to-first-token) cực nhanh. Tác vụ tóm tắt xử lý theo lô (batch summarizer) thì không cần. Hãy định tuyến theo thỏa thuận mức dịch vụ về độ trễ (latency SLA).

4. **Tuân thủ quy định (Compliance):** Dữ liệu người dùng tại Liên minh Châu Âu (EU) phải được lưu trữ và xử lý trong các khu vực thuộc EU. Hãy định tuyến theo vị trí địa lý.

5. **Thử nghiệm (Experimentation):** Thử nghiệm A/B hai mô hình trên cùng một khối lượng công việc thực tế. Hãy định tuyến theo phân nhóm thử nghiệm.

Việc tự viết mã xử lý những logic này cho từng tích hợp riêng biệt rất tốn công và lặp thừa. Một cổng định tuyến cung cấp một API chuẩn tương thích với OpenAI và tự động đảm nhiệm toàn bộ phần việc còn lại.

## Khái niệm cốt lõi

### Cấu trúc proxy tương thích OpenAI (OpenAI-compatible proxy shape)

Hầu hết hệ sinh thái đều hỗ trợ định dạng của OpenAI. Cổng định tuyến phơi bày điểm cuối `/v1/chat/completions`, chấp nhận lược đồ dữ liệu của OpenAI, và chuyển tiếp nội bộ tới Anthropic / Gemini / Cohere / Ollama / hoặc bất kỳ mô hình nào khác. Phía client không cần quan tâm đến chi tiết này.

### Bí danh mô hình (Model aliases)

Thay vì gắn cứng một mã phiên bản cụ thể (pinned snapshot id), mã nguồn của bạn chỉ cần gọi `our_smart_model`. Cổng định tuyến ánh xạ các bí danh này tới các mô hình thực tế. Khi nhà cung cấp phát hành thế hệ mô hình mới, bạn chỉ cần thay đổi cấu hình bí danh trên máy chủ; toàn bộ mã ứng dụng của bạn được giữ nguyên vẹn.

### Chuỗi chuyển đổi dự phòng (Fallback chains)

```yaml
primary: openai/gpt-4o
on 5xx: anthropic/claude-3-5-sonnet
on 5xx: google/gemini-1.5-pro
on 5xx: refuse
```

Các cổng định tuyến khai báo cấu hình này trong tệp thiết lập. Số lần thử lại được tính toán dựa trên một hạn ngạch ngân sách (budget) để chuỗi dự phòng phân tầng không làm bùng nổ chi phí ngoài kiểm soát.

### Bộ nhớ đệm ngữ nghĩa (Semantic caching)

Các prompt giống hệt hoặc gần như giống nhau sẽ truy xuất thẳng vào bộ nhớ đệm thay vì gửi yêu cầu đến nhà cung cấp. Mức tiết kiệm trong các vòng lặp tác tử lặp đi lặp lại có thể đạt từ 30% đến 60%. Khóa đệm (cache key) được tạo dựa trên vector nhúng (embedding); các prompt có nghĩa gần tương đương sẽ dùng chung một ô nhớ đệm.

### Hàng rào bảo vệ (Guardrails)

Ở cấp độ cổng định tuyến:

- **Ẩn danh PII (PII redaction):** Quét bằng biểu thức chính quy (Regex) hoặc mô hình máy học trước khi gửi prompt đi.
- **Vi phạm chính sách (Policy violations):** Từ chối ngay các prompt chứa nội dung bị nghiêm cấm.
- **Bộ lọc đầu ra (Output filters):** Rà soát và loại bỏ các dữ liệu rò rỉ trong câu trả lời hoàn thành.

Portkey và Kong đều tích hợp sẵn các hàng rào bảo vệ nghiêm ngặt. LiteLLM để tùy chọn cấu hình theo nhu cầu.

### Giới hạn tốc độ theo khóa (Per-key rate limits)

Một khóa API tương ứng với một nhóm phát triển. Ngân sách định mức cho từng khóa ngăn chặn tình trạng một nhóm tiêu thụ cạn kiệt hạn ngạch dùng chung của toàn hệ thống. Hầu hết các cổng định tuyến đều hỗ trợ tính năng này.

### Đánh đổi giữa tự lưu trữ và dịch vụ quản lý

| Tiêu chí | LiteLLM (Tự lưu trữ) | OpenRouter (Quản lý sẵn) | Portkey (Cấp độ sản xuất) |
|----------|----------------------|--------------------------|---------------------------|
| Mã nguồn | Mã nguồn mở, Python | SaaS được quản lý hoàn toàn | Mã nguồn mở (tháng 3/2026) + dịch vụ quản lý |
| Thiết lập | Triển khai một proxy | Đăng ký tài khoản | Hỗ trợ cả hai |
| Nhà cung cấp | Hơn 100 | Hơn 300 | Hơn 100 |
| Thanh toán | Sử dụng khóa riêng của bạn | Tín dụng nạp sẵn của OpenRouter | Sử dụng khóa riêng của bạn |
| Khả năng quan sát | OpenTelemetry | Bảng điều khiển (Dashboard) | Đầy đủ OTel + tự động ẩn danh PII |
| Phù hợp nhất cho | Các đội ngũ muốn kiểm soát toàn diện | Tạo mẫu thử nghiệm nhanh | Môi trường sản xuất đòi hỏi tuân thủ nghiêm ngặt |

LiteLLM chiếm ưu thế khi bạn có đội ngũ kỹ sư đảm bảo độ tin cậy hệ thống (SRE) và yêu cầu chủ quyền dữ liệu. OpenRouter tối ưu nhất khi bạn muốn một gói thuê bao duy nhất và không phải quản lý hạ tầng. Portkey là lựa chọn hàng đầu khi bạn cần hàng rào bảo vệ và khả năng tuân thủ tiêu chuẩn ngay từ đầu.

### Theo dõi chi phí (Cost tracking)

Mỗi yêu cầu đều mang các trường `provider`, `model`, `input_tokens`, `output_tokens`. Hệ thống nhân số lượng token với đơn giá theo từng mô hình (được lấy từ bảng giá mà cổng duy trì liên tục). Dữ liệu được tổng hợp theo người dùng / theo nhóm / hoặc theo dự án.

### Kết hợp MCP và định tuyến

Cổng có thể định tuyến cả lệnh gọi LLM VÀ các yêu cầu lấy mẫu MCP (MCP sampling requests). Khi trường `modelPreferences` trong yêu cầu lấy mẫu ưu tiên một mô hình cụ thể, cổng định tuyến sẽ dịch sang hạ tầng backend thích hợp. Đây là điểm giao thoa nơi cổng MCP (Phase 13 · 17) và cổng định tuyến của bài học này có thể hợp nhất thành một dịch vụ duy nhất.

### Các chiến lược định tuyến

- **Ưu tiên tĩnh (Static priority):** Chọn nhà cung cấp đầu tiên trong danh sách; chuyển sang phương án dự phòng khi có lỗi.
- **Cân bằng tải (Load balancing):** Xoay vòng (round-robin) hoặc phân bổ theo trọng số.
- **Tối ưu chi phí (Cost-aware):** Chọn mô hình có chi phí rẻ nhất nhưng vẫn đáp ứng yêu cầu về độ trễ và chất lượng.
- **Tối ưu độ trễ (Latency-aware):** Chọn mô hình có tốc độ xử lý nhanh nhất trong N phút gần nhất.
- **Định tuyến theo tác vụ (Task-aware):** Bộ phân loại prompt định tuyến các tác vụ lập trình tới một mô hình chuyên biệt, và tác vụ tóm tắt tới một mô hình khác.

```figure
tp-router-failover
```

## Sử dụng trong thực tế (Use It)

Tệp `code/main.py` triển khai một cổng định tuyến trong khoảng 150 dòng mã: chấp nhận các yêu cầu có định dạng của OpenAI, chuyển dịch sang các stub riêng của từng nhà cung cấp, vận hành một chuỗi dự phòng theo thứ tự ưu tiên, theo dõi chi phí trên từng yêu cầu và áp dụng một bước làm sạch PII trên dữ liệu đầu vào. Hãy chạy với ba kịch bản: yêu cầu bình thường, sự cố nhà cung cấp chính kích hoạt dự phòng, và phát hiện rò rỉ PII được che giấu thành công.

Những điểm cần quan sát:

- Từ điển `ROUTES`: bí danh (alias) -> danh sách các nhà cung cấp cụ thể theo thứ tự ưu tiên.
- Vòng lặp dự phòng tự động thử lại khi gặp mã lỗi 5xx.
- Bộ theo dõi chi phí nhân mức sử dụng token với đơn giá của từng mô hình.
- Bộ lọc PII làm sạch các chuỗi có định dạng số An sinh Xã hội (SSN) trước khi chuyển tiếp.

## Chuyển giao sản phẩm (Ship It)

Bài học này tạo ra `outputs/skill-routing-config-designer.md`. Dựa trên hồ sơ yêu cầu của khối lượng công việc (độ trễ, chi phí, tính tuân thủ), kỹ năng này sẽ lựa chọn giữa LiteLLM / OpenRouter / Portkey và tạo ra tệp cấu hình định tuyến hoàn chỉnh.

## Bài tập

1. Chạy `code/main.py`. Kích hoạt kịch bản sự cố; xác nhận rằng cơ chế dự phòng chuyển hướng sang nhà cung cấp thứ hai và chi phí được tính toán chính xác.

2. Thêm bộ nhớ đệm ngữ nghĩa: sử dụng mã băm SHA256 của prompt làm khóa tra cứu; khi trúng đệm (cache hit), phản hồi lập tức được trả về. Đo lường chi phí tiết kiệm được trên một lệnh gọi lặp lại.

3. Bổ sung một bộ phân loại prompt để định tuyến các prompt bắt đầu bằng "code ..." tới một bí danh ưu tiên trí tuệ suy luận và các prompt "summarize ..." tới một bí danh ưu tiên tốc độ.

4. Thiết kế ngân sách cho từng nhóm: mỗi nhóm có một hạn mức chi tiêu hàng tháng; cổng sẽ từ chối yêu cầu khi chạm ngưỡng hạn mức. Chọn mức độ thực thi phù hợp (trên từng yêu cầu hay theo cửa sổ thời gian).

5. Đọc tài liệu của LiteLLM, OpenRouter và Portkey song song với nhau. Nêu tên một tính năng độc nhất mà mỗi bên sở hữu nhưng hai bên còn lại không có.

## Thuật ngữ then chốt

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật chuẩn xác |
|-----------|----------------------|---------------------------|
| Cổng định tuyến (Routing gateway) | "LLM proxy" | Tầng giao diện API hợp nhất đứng trước nhiều nhà cung cấp mô hình |
| Tương thích OpenAI (OpenAI-compatible) | "Nói tiếng OpenAI" | Chấp nhận cấu trúc `/v1/chat/completions`, dịch sang bất kỳ backend nào |
| Bí danh mô hình (Model alias) | "our_smart_model" | Tên trong mã nguồn được cổng định tuyến ánh xạ tới một mô hình cụ thể |
| Chuỗi chuyển đổi dự phòng (Fallback chain) | "Danh sách thử lại" | Danh sách các nhà cung cấp được sắp xếp thứ tự thử nghiệm khi có lỗi |
| Bộ nhớ đệm ngữ nghĩa (Semantic caching) | "Đệm nhúng prompt" | Khóa là vector nhúng của prompt; các prompt gần giống nhau dùng chung kết quả |
| Hàng rào bảo vệ (Guardrails) | "Bộ lọc đầu vào/đầu ra" | Ẩn danh PII, ngăn chặn vi phạm chính sách |
| Giới hạn theo khóa (Per-key rate limit) | "Ngân sách nhóm" | Hạn ngạch tài nguyên được gán theo từng khóa API |
| Theo dõi chi phí (Cost tracking) | "Chi tiêu trên mỗi yêu cầu" | Tổng hợp mức tiêu thụ token nhân với đơn giá của từng mô hình |
| LiteLLM | "Proxy nguồn mở" | Cổng định tuyến mã nguồn mở tự lưu trữ được |
| OpenRouter | "SaaS được quản lý" | Cổng định tuyến lưu trữ trên đám mây với cơ chế thanh toán theo tín dụng |
| Portkey | "Tùy chọn cho sản xuất" | Nền tảng kết hợp mã nguồn mở và dịch vụ quản lý với hàng rào bảo vệ tích hợp |

## Tài liệu đọc thêm

- [LiteLLM — docs](https://docs.litellm.ai/) — Cổng định tuyến tự lưu trữ
- [OpenRouter — quickstart](https://openrouter.ai/docs/quickstart) — Dịch vụ định tuyến SaaS được quản lý
- [Portkey — docs](https://portkey.ai/docs) — Định tuyến cho môi trường sản xuất với hàng rào bảo vệ
- [TrueFoundry — LiteLLM vs OpenRouter](https://www.truefoundry.com/blog/litellm-vs-openrouter) — Hướng dẫn so sánh và đưa ra quyết định
- [Relayplane — LLM gateway comparison 2026](https://relayplane.com/blog/llm-gateway-comparison-2026) — Báo cáo khảo sát thị trường cổng LLM năm 2026
