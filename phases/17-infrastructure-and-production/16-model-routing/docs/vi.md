# Điều hướng Mô hình như một Căn bản Giảm thiểu Chi phí (Model Routing as a Cost-Reduction Primitive)

> Một bộ điều phối động (dynamic broker) đánh giá từng yêu cầu (loại tác vụ, độ dài token, độ tương đồng nhúng, mức độ tin cậy) và gửi các truy vấn đơn giản đến một mô hình giá rẻ, đồng thời leo thang (escalate) các truy vấn phức tạp lên mô hình tiên phong (frontier model). Kỹ thuật này còn được gọi là thác tầng mô hình (model cascading). Các nghiên cứu thực tế trong môi trường sản xuất cho thấy khả năng cắt giảm chi phí 20-60% ở cùng mức chất lượng tương đương (iso-quality) trên khắp các hệ thống tại Mỹ, Anh và EU; việc cải thiện 30% hiệu quả điều hướng trên các dịch vụ SaaS lưu lượng lớn mang lại khoản tiết kiệm hàng trăm nghìn USD mỗi năm. Bối cảnh năm 2026 cho thấy giá suy luận LLM cho một mức năng lực cố định đã giảm khoảng 10 lần mỗi năm (theo ước tính "lạm phát ngược LLM - LLMflation" của a16z); ở phân khúc tương đương GPT-4, tốc độ giảm chậm hơn, từ $20/triệu token xuống còn ~$0,40/triệu token (giảm khoảng 50 lần) từ cuối năm 2022 đến năm 2026. Phần lớn sự sụt giảm này đến từ việc tối ưu hóa ngăn xếp phục vụ suy luận (Giai đoạn 17 · 04-09) thay vì cải tiến phần cứng. Điều hướng mô hình là cách bạn chuyển hóa đà giảm giá đó thành biên lợi nhuận mà không làm suy giảm chất lượng sản phẩm. Rủi ro thất bại tiềm ẩn là hiện tượng lệch trôi do dùng mô hình giá rẻ (cheap-model drift): bộ điều hướng đẩy 40% lưu lượng sang mô hình yếu hơn, chất lượng sụt giảm 3-5% trên các tác vụ suy luận logic mà không ai nhận ra trong suốt một quý. Hãy kiểm soát các lộ trình điều hướng bằng các chỉ số chất lượng trực tuyến (online quality metrics), thay vì chỉ dựa vào các bộ dữ liệu đánh giá ngoại tuyến.

**Loại:** Tìm hiểu (Learn)  
**Ngôn ngữ:** Python (thư viện chuẩn, bộ mô phỏng bộ điều hướng thác tầng đồ chơi - toy cascading router simulator)  
**Điều kiện tiên quyết:** Giai đoạn 17 · 01 (Managed LLM Platforms), Giai đoạn 17 · 19 (AI Gateways)  
**Thời lượng:** ~60 phút  

## Mục tiêu Học tập

- Giải thích cơ chế thác tầng mô hình: ưu tiên mô hình giá rẻ trước kèm kiểm tra độ tin cậy, leo thang lên mô hình cao cấp khi độ tin cậy thấp.
- Liệt kê bốn tín hiệu điều hướng chính (phân loại tác vụ, độ dài prompt, độ tương đồng nhúng với tập tác vụ khó đã biết, độ tự tin từ lượt chạy đầu).
- Tính toán chi phí hỗn hợp (blended cost) dự kiến theo tỷ lệ phân bổ điều hướng mục tiêu và mức dung sai suy giảm chất lượng cho phép.
- Chỉ rõ chỉ số giám sát độ lệch trôi (rào kiểm soát chất lượng trực tuyến) giúp phát hiện sự xâm lấn ngấm ngầm của mô hình giá rẻ.

## Vấn đề Thực tế

Dịch vụ của bạn tiêu tốn $80.000 mỗi tháng khi chạy trên GPT-5. Số liệu phân tích cho thấy 70% truy vấn thực chất rất đơn giản: "bây giờ là mấy giờ ở Paris?" hoặc "hãy diễn đạt lại câu này." Một mô hình thuộc phân khúc Haiku hoàn toàn có thể xử lý hoàn hảo những yêu cầu đó với chỉ 3% chi phí. Chỉ có 30% còn lại thực sự đòi hỏi năng lực suy luận sâu của GPT-5 — như viết mã, giải toán, hoặc lập kế hoạch đa bước.

Nếu bạn điều hướng 70% truy vấn đơn giản sang mô hình giá rẻ và 30% sang mô hình đắt tiền, hóa đơn của bạn sẽ giảm ~65% trong khi chất lượng sản phẩm vẫn được giữ nguyên. Đó chính là bản chất của điều hướng mô hình (routing). Thách thức cốt lõi nằm ở việc xây dựng bộ điều phối mà không làm suy giảm chất lượng đầu ra.

## Khái niệm

### Bốn tín hiệu điều hướng cốt lõi

1. **Phân loại tác vụ (Task classification)**: đơn giản / phức tạp / sinh mã nguồn / giải toán / trò chuyện. Có thể là một bộ phân loại dựa trên luật (rule-based), một LLM nhỏ (phân khúc Haiku ở mức giá $0,25/triệu token), hoặc so sánh độ tương đồng nhúng với các nhóm tác vụ đã gán nhãn. Kết quả đầu ra: lộ trình = giá rẻ (cheap) / cân bằng (balanced) / tiên phong (frontier).

2. **Độ dài câu lệnh (Prompt length)**: các câu lệnh dài >4K token thường cần mô hình tiên phong để duy trì tính mạch lạc. Các câu lệnh ngắn <500 token thường không yêu cầu điều đó.

3. **Độ tương đồng nhúng với tập tác vụ khó đã biết (Embedding similarity to known-hard set)**: nếu truy vấn có độ tương đồng cô-sin gần (cosine > 0,88) với một tập hợp các trường hợp biên/khó đã biết, hãy leo thang trực tiếp lên mô hình tiên phong.

4. **Độ tự tin từ lượt chạy đầu tiên (Self-confidence from first-pass)**: gửi truy vấn tới mô hình giá rẻ; nếu xác suất logarit (log-probs) của mô hình thể hiện độ tin cậy thấp HOẶC mô hình từ chối trả lời HOẶC đưa ra ngôn từ do dự, hãy thử lại ngay trên mô hình tiên phong. Cơ chế này làm tăng độ trễ P95 trên ~10% lưu lượng nhưng tiết kiệm hơn 50% chi phí cho 90% lưu lượng còn lại.

### Ba mô hình kiến trúc

**Định tuyến trước (Pre-route)** (bộ phân loại nằm ở đầu vào): chỉ tăng thêm ~5-10 ms độ trễ; tốc độ tổng thể nhanh nhất.

**Thác tầng (Cascade)** (ưu tiên mô hình rẻ trước, leo thang khi độ tin cậy thấp): độ trễ trung vị gấp ~1,2 lần (do chạy mô hình rẻ cộng thêm bước xác thực), gấp ~2 lần trên các yêu cầu bị leo thang. Đảm bảo sàn chất lượng tốt nhất.

**Định tuyến phối hợp (Ensemble route)** (chạy song song cả mô hình rẻ và tiên phong trên một mẫu thử nghiệm, dùng mô hình chấm điểm [reward model] để chọn): chất lượng cao nhất, chi phí lớn nhất; chỉ nên dùng cho các đợt kiểm thử A/B mang tính sống còn.

### Triển khai thực tế

Các cổng AI Gateway (Giai đoạn 17 · 19) cung cấp sẵn khả năng điều hướng. LiteLLM hỗ trợ cấu hình `router` với cơ chế dự phòng và điều hướng dựa trên chi phí. Portkey cung cấp các lớp bảo vệ (guards) kết hợp điều hướng. Kong AI Gateway sở hữu tính năng điều hướng dựa trên plugin. Sàn giao dịch mô hình của OpenRouter cung cấp API khuyến nghị định tuyến tự động.

Các thư viện mã nguồn mở: RouteLLM (LMSYS), Prompt Mule. Giải pháp thương mại: Not Diamond.

### Đường cong giá cả năm 2026

| Phân khúc mô hình | Cuối 2022 | 2026 | Mức thay đổi |
|---|---|---|---|
| Chất lượng tương đương GPT-4 | ~$20/triệu token | ~$0,40/triệu token | Rẻ hơn 50 lần |
| Mô hình tiên phong (GPT-5, Claude 4) | — | ~$3-10/triệu token | Phân khúc mới |

Hầu hết sự cải thiện đến từ hiệu quả của ngăn xếp phục vụ suy luận — các bài học cốt lõi trong Giai đoạn 17 · 04-09 đã biến thành sự sụt giảm chi phí trực tiếp từ phía nhà cung cấp. Điều hướng mô hình giúp bạn tận dụng tối đa những lợi thế đó ở tầng ứng dụng thay vì phải đợi toàn bộ người dùng tự nguyện chuyển sang gói giá rẻ.

### Hiện tượng lệch trôi là rủi ro thực sự

Bộ điều hướng của bạn chuyển 40% lưu lượng sang mô hình giá rẻ. Trải qua sáu tháng, phân phối tác vụ thay đổi (người dùng trở nên am hiểu hơn, đặt những câu hỏi dài và phức tạp hơn). Bộ điều hướng không nhận biết được điều này vì bộ phân loại của nó chỉ được huấn luyện trên dữ liệu quý 1. Chất lượng suy giảm một cách âm thầm. Không ai phàn nàn đủ lớn để kích hoạt cảnh báo. Cuối cùng, bạn chỉ phát hiện ra điều đó khi thua cuộc trong một bảng xếp hạng chuẩn đối sánh của đối thủ cạnh tranh.

Hãy thiết lập rào chắn cho các lộ trình điều hướng bằng các chỉ số chất lượng trực tuyến:

- Tỷ lệ người dùng bấm thích / không thích (thumbs-up / thumbs-down) trên từng lộ trình.
- Mô hình LLM giám khảo (LLM-judge) tự động đánh giá trên một mẫu đại diện (5%) của mỗi lộ trình.
- Tỷ lệ leo thang (escalation rate): nếu mô hình thác tầng kích hoạt leo thang >30%, điều đó chứng tỏ mô hình giá rẻ đang bị giao quá tải các nhiệm vụ vượt tầm.
- Tỷ lệ từ chối phản hồi trên từng lộ trình.

### Các con số cần ghi nhớ

- Tiết kiệm chi phí nhờ điều hướng năm 2026 ở cùng mức chất lượng: 20-60% theo các nghiên cứu tình huống thực tế.
- Mức giảm giá LLM ở cùng một cấp năng lực: ~10 lần mỗi năm (ước tính lạm phát ngược LLMflation của a16z); giảm khoảng 50 lần ở phân khúc GPT-4 từ 2022 đến 2026.
- Mức giá phân khúc GPT-4 năm 2022 so với 2026: ~$20/triệu token → ~$0,40/triệu token.
- Ảnh hưởng độ trễ của mô hình thác tầng: ~1,2 lần trung vị, ~2 lần cho các yêu cầu bị leo thang (~10% lưu lượng).

```figure
model-cascade-router
```

## Sử dụng trong Thực tế

Tệp `code/main.py` mô phỏng các chiến lược định tuyến trước (pre-route), thác tầng (cascade), và phối hợp (ensemble) trên một khối lượng công việc hỗn hợp. Báo cáo chi phí hỗn hợp, mức suy giảm chất lượng và tỷ lệ leo thang theo từng phương án.

## Sản phẩm Chuyển giao

Bài học này xây dựng tệp kỹ năng `outputs/skill-router-plan.md`. Dựa trên khối lượng công việc và ngân sách chất lượng cho phép, công cụ sẽ đề xuất mô hình kiến trúc điều hướng và các tín hiệu kích hoạt tối ưu.

## Bài tập Thực hành

1. Chạy `code/main.py`. Ở sàn độ chính xác nào thì mô hình thác tầng vượt trội hơn so với định tuyến trước?
2. Cơ sở người dùng của bạn gồm 30% khách hàng doanh nghiệp (truy vấn phức tạp), 70% người dùng miễn phí (truy vấn đơn giản). Hãy thiết kế tỷ lệ phân bổ điều hướng. Chỉ số trực tuyến nào sẽ đóng vai trò rào chắn kiểm soát?
3. Một lộ trình điều hướng làm giảm chất lượng 2% nhưng tiết kiệm được 40% chi phí. Bạn có chấp nhận đưa vào triển khai thực tế không? Quyết định phụ thuộc vào bản chất sản phẩm — hãy lập luận cho cả hai quan điểm.
4. Triển khai bước kiểm tra độ tin cậy bằng cách trích xuất logprobs từ API của OpenAI / Anthropic. Ngưỡng bắt đầu bạn lựa chọn là bao nhiêu?
5. Qua sáu tháng vận hành, tỷ lệ leo thang tăng vọt từ 8% lên 22%. Hãy chẩn đoán ba nguyên nhân cốt lõi và đề xuất giải pháp khắc phục cho từng nguyên nhân.

## Thuật ngữ Then chốt

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật chuẩn xác |
|---|---|---|
| Điều hướng mô hình (Model routing) | "Bộ điều phối chi phí" | Lựa chọn mô hình động và tối ưu trên từng yêu cầu suy luận |
| Thác tầng mô hình (Model cascade) | "Ưu tiên mô hình rẻ trước" | Chạy mô hình rẻ, chuyển tiếp lên mô hình cao cấp nếu độ tin cậy thấp |
| Định tuyến trước (Pre-route) | "Phân loại trước" | Phân loại ngay tại đầu vào; không cần chạy thử nghiệm lại |
| Định tuyến phối hợp (Ensemble route) | "Chọn lọc song song" | Chạy đồng thời nhiều mô hình, dùng mô hình phần thưởng chọn kết quả tốt nhất |
| Tỷ lệ leo thang (Escalation rate) | "Tỷ lệ chuyển tiếp lên" | Tỷ lệ phần trăm các yêu cầu trong mô hình thác tầng phải leo thang lên mô hình cao cấp |
| RouteLLM | "Bộ điều hướng LMSYS" | Thư viện mã nguồn mở chuyên về điều hướng LLM |
| Not Diamond | "Router thương mại" | Sản phẩm SaaS thương mại chuyên về điều hướng mô hình |
| Lệch trôi (Drift) | "Xâm lấn giá rẻ" | Hiện tượng phân phối dữ liệu thay đổi nhưng bộ điều hướng không nhận biết được |
| Rào kiểm soát chất lượng trực tuyến (Online quality gate) | "Kiểm tra trực tiếp" | Dùng mô hình LLM làm giám khảo đánh giá tự động trên luồng dữ liệu thực tế |

## Tài liệu Tham khảo thêm

- [AbhyashSuchi — Thực hành Tốt nhất về Điều hướng Mô hình LLM 2026](https://web.archive.org/web/20260413143335/https://abhyashsuchi.in/model-routing-llm-2026-best-practices/)
- [Lukas Brunner — Sự trỗi dậy của Tối ưu hóa Suy luận 2026](https://dev.to/lukas_brunner/the-rise-of-inference-optimization-the-real-llm-infra-trend-shaping-2026-4e4o)
- [Bài báo khoa học và mã nguồn RouteLLM](https://github.com/lm-sys/RouteLLM)
- [Not Diamond — Điều hướng mô hình](https://www.notdiamond.ai/)
- [OpenRouter](https://openrouter.ai/) — Cổng kết nối đa mô hình tích hợp các nguyên lý điều hướng.
