# FinOps cho LLM — Kinh tế Đơn vị và Phân bổ Chi phí Đa Đối tượng thuê

> Các mô hình FinOps truyền thống bị phá vỡ khi áp dụng cho chi phí LLM. Chi phí AI vận hành theo các giao dịch token (token-transactions) chứ không dựa trên thời gian hoạt động của tài nguyên phần cứng (resource-uptime). Cơ chế gắn thẻ (tags) thông thường không thể ánh xạ hiệu quả — mỗi lệnh gọi API là một giao dịch tức thời, không phải là một tài sản cố định. Các quyết định kỹ thuật (thiết kế prompt, độ dài cửa sổ ngữ cảnh, độ dài phản hồi đầu ra) thực chất chính là các quyết định tài chính. Cẩm nang vận hành năm 2026 xác lập ba chiều phân bổ chi phí (attribution dimensions) cần được tích hợp đo kiểm (instrumentation) ngay từ ngày đầu tiên: theo từng người dùng (`user_id`) để định giá gói ghế sử dụng và mở rộng tài khoản, theo từng tác vụ (`task_id` + `route`) để tính chi phí diện tích bề mặt sản phẩm và sắp xếp thứ tự ưu tiên tính năng, theo từng đối tượng thuê (`tenant_id`) để quản trị kinh tế đơn vị (unit economics) và gia hạn hợp đồng. Bốn tầng token — prompt, công cụ (tool), bộ nhớ (memory), phản hồi (response) — nếu gộp chung vào một giỏ sẽ che giấu nguồn gốc lãng phí ngân sách. Thang thực thi kiểm soát chi phí (enforcement ladder) dành cho các sản phẩm đa đối tượng thuê (multi-tenant): giới hạn tần suất gọi (rate limits) theo đối tượng thuê (gấp 2-3 lần mức đỉnh kỳ vọng, trả về mã 429 kèm tiêu đề retry-after rõ ràng); trần chi tiêu hàng ngày (daily spend cap, gấp 1,5-3 lần trần hợp đồng; kích hoạt siết chặt rate limit và gửi cảnh báo); công tắc ngắt khẩn cấp (kill switch) khi điểm chuẩn hóa chi tiêu z-score > 4 (tự động tạm dừng dịch vụ đối tượng thuê và kích hoạt trang gọi kỹ sư trực nhật). Các mô hình phân bổ chi phí: gắn thẻ và tổng hợp (tag-and-aggregate), ghép nối dữ liệu đo từ xa (telemetry-joiner: ánh xạ trace-ID → hóa đơn thanh toán; độ chính xác cao nhất), lấy mẫu và ngoại suy (sampling-and-extrapolation), phân bổ dựa trên mô hình hồi quy (model-based allocation), tiếp cận hướng sự kiện (event-sourced), phát trực tuyến thời gian thực (real-time streaming). Chỉ số đơn vị sản phẩm chuẩn mực: chi phí trên mỗi truy vấn được giải quyết thành công (cost per resolved query), chi phí trên mỗi tạo tác sinh ra (cost per generated artifact) — không dùng đơn vị $/triệu token của nhà cung cấp. Việc gắn thẻ hồi tố (retroactive tagging) luôn bỏ sót các trường hợp biên; hãy tích hợp định danh ngay tại thời điểm khởi tạo yêu cầu.

**Type:** Learn
**Languages:** Python (stdlib, toy cost-attribution simulator with kill switch)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 14 (Caching)
**Time:** ~60 phút

## Mục tiêu học tập

- Giải thích lý do FinOps truyền thống (dựa trên thẻ tài nguyên + phân tầng dịch vụ) bị phá vỡ trước chi phí LLM và nêu tên ba chiều phân bổ chi phí mới.
- Liệt kê bốn tầng token (prompt, tool, memory, response) và giải thích tại sao việc tính phí gộp một giỏ duy nhất lại làm mù mờ việc tối ưu hóa chi phí.
- Thiết kế một thang thực thi kiểm soát chi phí (rate limit → spend cap → kill switch) cho một sản phẩm đa đối tượng thuê.
- Lựa chọn chỉ số đơn vị chuẩn mực hướng sản phẩm (chi phí trên mỗi truy vấn được giải quyết / tạo tác sinh ra) thay vì đơn vị $/triệu token.

## Vấn đề thực tế

Hóa đơn dịch vụ đám mây cuối tháng gửi về con số 40.000 USD. Bạn hoàn toàn không nắm được:
- Khách hàng (tenant) nào đã tiêu tốn phần lớn ngân sách.
- Tính năng nào trong sản phẩm là nguyên nhân chính thúc đẩy chi phí tăng vọt.
- Liệu có người dùng cá nhân nào đang lạm dụng hệ thống hay không.
- Sự phình to của prompt (prompt bloat), các lệnh gọi công cụ (tool calls), hay sự khuếch đại bộ nhớ ngữ cảnh (memory amplification) là thủ phạm chính.

Phương pháp gắn thẻ và tổng hợp từ phía nhà cung cấp vốn hoạt động rất tốt đối với các tài nguyên đám mây hạ tầng (như EC2, S3) vì thẻ siêu dữ liệu được truyền thẳng vào từng dòng hóa đơn. Nhưng các lệnh gọi API LLM không tự động gắn thẻ — bạn bắt buộc phải đóng dấu `user_id`, `task_id`, `tenant_id` ngay tại điểm gọi mã nguồn và duy trì dữ liệu này xuyên suốt luồng xử lý. Mọi nỗ lực phân bổ chi phí theo phương thức hồi tố (retroactive attribution) đều luôn bỏ sót các trường hợp biên quan trọng.

## Khái niệm then chốt

### Ba chiều phân bổ chi phí

**Theo người dùng** (`user_id`): Xác định chính xác ai đang tiêu tốn bao nhiêu chi phí. Dữ liệu này định hướng chiến lược định giá theo ghế (seat pricing), các cuộc đàm phán mở rộng hợp đồng, và phát hiện những người dùng chuyên sâu (power users).

**Theo tác vụ** (`task_id` + `route`): Xác định bề mặt tính năng nào trong sản phẩm phát sinh chi phí bao nhiêu. Dữ liệu này định hướng ưu tiên phát triển tính năng hoặc đưa ra quyết định loại bỏ các tính năng gây tốn kém không tương xứng.

**Theo đối tượng thuê** (`tenant_id`): Xác định khách hàng doanh nghiệp nào thực sự mang lại lợi nhuận ròng. Dữ liệu này định hướng kinh tế đơn vị (unit economics), giá gia hạn hợp đồng, và ngưỡng phân tầng dịch vụ.

Cần tích hợp đo kiểm cả ba chiều này tại điểm phát sinh yêu cầu ngay từ ngày đầu tiên triển khai. Việc phân bổ hồi tố sau này luôn mang lại kết quả thiếu chính xác.

### Bốn tầng token

| Tầng token | Ví dụ thực tế | Tỷ lệ điển hình trên tổng chi phí |
|---|---|---|
| Prompt | Câu lệnh hệ thống + văn bản nhập của người dùng | 40-60% |
| Tool | Kết quả thực thi công cụ được đưa ngược lại ngữ cảnh | 20-40% (khối lượng công việc agent) |
| Memory | Lịch sử hội thoại trước đó / tài liệu truy xuất RAG | 10-30% |
| Response | Văn bản phản hồi do mô hình sinh ra | 10-30% |

Gộp chung cả bốn tầng này vào một giỏ thanh toán duy nhất sẽ khiến các nỗ lực tối ưu hóa chi phí trở nên mù quáng. Hãy phân tách chúng rõ ràng trong lược đồ phân bổ chi phí của bạn.

### Thang thực thi kiểm soát chi phí

1. **Giới hạn tần suất (Rate limit)** theo từng đối tượng thuê: Thiết lập ở mức gấp 2-3 lần đỉnh tải kỳ vọng. Khi vượt ngưỡng, lập tức trả về mã lỗi 429 kèm tiêu đề `Retry-After`. Khách hàng nhận thấy độ trễ điều tiết mà không gặp phải hóa đơn bất ngờ ngoài ý muốn.

2. **Trần chi tiêu hàng ngày (Daily spend cap)** theo từng đối tượng thuê: Thiết lập ở mức gấp 1,5-3 lần mức trần trong hợp đồng cam kết. Khi chạm ngưỡng kích hoạt: tự động siết chặt giới hạn tần suất gọi và gửi cảnh báo tới đội ngũ chăm sóc khách hàng (Customer Success).

3. **Công tắc ngắt khẩn cấp (Kill switch)** khi điểm chuẩn hóa chi tiêu z-score > 4 so với đường cơ sở của chính đối tượng thuê đó. Hệ thống tự động tạm ngưng dịch vụ của đối tượng thuê; kích hoạt trang báo động kỹ sư trực nhật; báo cáo sự cố khẩn cấp lên đội ngũ vận hành và chăm sóc khách hàng.

### Các mô hình phân bổ chi phí

- **Gắn thẻ và tổng hợp (Tag-and-aggregate)**: Đóng dấu tiêu đề metadata; tổng hợp dữ liệu sau đó qua batch job. Đơn giản nhưng độ chi tiết ở mức tương đối.
- **Ghép nối dữ liệu đo từ xa (Telemetry joiner)**: Ghép các trace quan sát với dữ liệu thanh toán thông qua ID dấu vết (trace ID). Đạt độ chính xác cao nhất; là tiêu chuẩn của các đội ngũ kỹ thuật trưởng thành.
- **Lấy mẫu và ngoại suy (Sampling + extrapolation)**: Lấy mẫu ngẫu nhiên 5-10% số lượng request, sau đó nhân hệ số quy đổi. Tối ưu chi phí xử lý cho các bức tranh tổng quan; nhưng dễ bỏ sót các trường hợp cực trị (tail latency / outliers).
- **Phân bổ dựa trên mô hình (Model-based allocation)**: Áp dụng hồi quy thống kê để suy luận động lực chi phí chính. Phù hợp cho việc phân tích dữ liệu lịch sử chưa được gắn thẻ.
- **Tiếp cận hướng sự kiện (Event-sourced)**: Đẩy các sự kiện chi phí vào một luồng dữ liệu phân tán (Kafka / Kinesis). Đạt độ trễ gần thời gian thực.
- **Phát trực tuyến thời gian thực (Real-time streaming)**: Cập nhật bảng điều khiển chi phí dưới một giây.

### Chi phí trên mỗi đơn vị kết quả (Cost per X) là chỉ số đơn vị cốt lõi

Đơn vị $/triệu token chỉ là ngôn ngữ tiếp thị của các nhà cung cấp nền tảng. Các chỉ số gắn liền với giá trị sản phẩm thực tế phải là:

- Chi phí trên mỗi yêu cầu hỗ trợ khách hàng được giải quyết thành công (Cost per resolved support ticket).
- Chi phí trên mỗi bài viết hoàn chỉnh được khởi tạo (Cost per generated article).
- Chi phí trên mỗi tác vụ agent hoàn thành chuẩn xác (Cost per successful agent task).
- Chi phí trên mỗi phút phiên hoạt động của người dùng (Cost per user-session-minute).

Gắn chặt chi phí kỹ thuật với một kết quả sản phẩm cụ thể. Nếu không, mọi nỗ lực tối ưu hóa đều không có điểm tựa vững chắc.

### Cấu trúc bản ghi dấu vết phân bổ chi phí (Cost attribution trace shape)

```yaml
trace_id: abc123
  user_id: u_42
  tenant_id: t_7
  task_id: task_classify_doc
  route: model_haiku
  layers:
    prompt_tokens: 1800
    tool_tokens: 600
    memory_tokens: 400
    response_tokens: 150
  cost_usd: 0.0135
  cached_input: true
  batch: false
```

Phát sinh bản ghi này trên từng lệnh gọi API. Lưu trữ vào hồ dữ liệu (data lake). Tổng hợp báo cáo theo từng chiều phân tích. Đây chính là nhiệm vụ mà ngăn xếp quan sát được xây dựng ở Phase 17 · 13 đảm nhiệm.

### Ngăn xếp tiết kiệm chi phí cộng gộp

Ngăn xếp giải pháp: Bộ nhớ đệm (Cache) + Xử lý theo lô (Batch) + Điều hướng mô hình (Route) + Cổng kết nối (Gateway). Khi kết hợp đồng thời cả bốn yếu tố:
- Bộ nhớ đệm L2 (Phase 17 · 14): Giảm chi phí token đầu vào khoảng 10 lần.
- Xử lý theo lô (Phase 17 · 15): Giảm thêm 50% đơn giá.
- Điều hướng sang các mô hình nhỏ gọn, tiết kiệm (Phase 17 · 16): Cắt giảm tới 60% chi phí.
- Tối ưu hóa hiệu năng tại cổng kết nối (Phase 17 · 19): Tối ưu dự phòng và xử lý thử lại thông minh.

Trong kịch bản tối ưu nhất khi cộng gộp toàn bộ: Chi phí thực tế chỉ còn khoảng 5-10% so với mức cơ sở ban đầu (naive baseline). Đa số các đội ngũ kỹ thuật mới chỉ kích hoạt 2-3 đòn bẩy; rất ít tổ chức kết hợp trọn vẹn cả bốn.

### Các con số cần ghi nhớ

- Các chiều phân bổ chi phí: Theo người dùng (per-user), theo tác vụ (per-task), theo đối tượng thuê (per-tenant).
- Bốn tầng token: Prompt, công cụ (tool), bộ nhớ (memory), phản hồi (response).
- Công tắc ngắt khẩn cấp: Điểm chuẩn hóa z-score của chi tiêu > 4.
- Chỉ số đơn vị sản phẩm: Chi phí trên mỗi truy vấn được giải quyết, không dùng $/triệu token.
- Tiết kiệm chi phí cộng gộp tối ưu: Giảm xuống chỉ còn ~5-10% so với mức cơ sở ban đầu.

```figure
i4-spend-ladder
```

## Thực hành (Use It)

Tệp `code/main.py` mô phỏng một dịch vụ LLM đa đối tượng thuê tích hợp thang thực thi kiểm soát chi phí ba tầng. Chương trình sẽ tiêm vào một đối tượng thuê có hành vi lạm dụng và minh họa trực quan quá trình kích hoạt công tắc ngắt khẩn cấp.

## Triển khai thực tế (Ship It)

Bài học này xây dựng `outputs/skill-finops-plan.md`. Dựa trên quy mô và đặc thù sản phẩm, tài liệu này thiết kế chi tiết lược đồ phân bổ chi phí và thang thực thi kiểm soát tài chính tự động.

## Bài tập

1. Chạy `code/main.py`. Công tắc ngắt khẩn cấp được kích hoạt ở mức z-score nào? Phương pháp thực tế để lựa chọn ngưỡng phù hợp cho doanh nghiệp của bạn là gì?
2. Thiết kế bảng điều khiển chi phí theo từng đối tượng thuê và từng tác vụ. Đâu là 5 góc nhìn trực quan (views) đầu tiên bạn cần xây dựng?
3. Đối tượng thuê lớn nhất của bạn đang có biên lợi nhuận kinh tế đơn vị âm (unit-economics-negative). Hãy đề xuất ba phương án can thiệp được sắp xếp theo mức độ ảnh hưởng đến trải nghiệm khách hàng từ thấp đến cao.
4. Tính toán chi phí trên mỗi yêu cầu hỗ trợ được giải quyết thành công cho một sản phẩm chăm sóc khách hàng: 3 triệu token/yêu cầu, trung bình ~800 yêu cầu/ngày, toàn bộ đầu vào áp dụng đơn giá lưu đệm của GPT-5 là 0,125 USD/triệu token (giá niêm yết thời điểm ra mắt; hãy kiểm tra bảng giá trực tiếp hiện hành).
5. Phân tích xem việc gắn thẻ hồi tố (retroactive tagging) có bao giờ mang lại kết quả tin cậy không. Khi nào thì phương pháp này có thể chấp nhận được tạm thời?

## Thuật ngữ then chốt

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật thực tế |
|---|---|---|
| Per-user attribution | "chi phí cấp người dùng" | Đóng dấu `user_id` trên mọi lệnh gọi API |
| Per-task attribution | "chi phí tính năng" | `task_id` + `route` định danh chính xác diện tích bề mặt sản phẩm |
| Per-tenant attribution | "chi phí khách hàng" | `tenant_id`; làm cơ sở xác định kinh tế đơn vị |
| Four token layers | "các tầng chi phí" | Prompt + Tool + Memory + Response |
| Rate limit | "rào chắn 429" | Trần tần suất gọi theo đối tượng thuê thực thi tại cổng gateway |
| Daily spend cap | "trần ngân sách ngày" | Hạn mức chi tiêu theo đối tượng thuê kèm cảnh báo tự động |
| Kill switch | "tự động ngắt" | Chi tiêu có z-score > 4 kích hoạt tạm dừng dịch vụ tức thì |
| Cost per resolved | "chỉ số đơn vị sản phẩm" | Chi phí gắn liền với kết quả sản phẩm thực tế, không tính bằng token |
| Telemetry joiner | "ghép trace với hóa đơn" | Mô hình phân bổ chi phí có độ chính xác cao nhất |
| Stacked optimization | "cache+batch+route+gateway" | Cộng gộp các đòn bẩy tiết kiệm để đưa chi phí về ~5-10% mức cơ bản |

## Tài liệu tham khảo

- [FinOps Foundation — FinOps for AI Overview](https://www.finops.org/wg/finops-for-ai-overview/)
- [FinOps School — Cost per Unit 2026 Guide](https://finopsschool.com/blog/cost-per-unit/)
- [Digital Applied — LLM Agent Cost Attribution 2026](https://www.digitalapplied.com/blog/llm-agent-cost-attribution-guide-production-2026)
- [PointFive — Managed LLMs in Azure OpenAI](https://www.pointfive.co/blog/finops-for-ai-economics-of-managed-llms-in-azure-open-ai)
