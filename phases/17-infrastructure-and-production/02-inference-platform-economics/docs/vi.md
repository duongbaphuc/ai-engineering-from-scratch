# Kinh Tế Học Nền Tảng Suy Luận — Fireworks, Together, Baseten, Modal, Replicate, Anyscale

> Thị trường suy luận (inference) năm 2026 không còn đơn thuần là việc thuê giờ GPU. Nó phân nhánh rõ rệt thành bán dẫn tùy biến (custom silicon: Groq, Cerebras, SambaNova), nền tảng GPU (GPU platforms: Baseten, Together, Fireworks, Modal), và chợ mô hình ưu tiên API (API-first marketplaces: Replicate, DeepInfra). Việc Fireworks tăng giá thêm $1/giờ cho mỗi GPU vào ngày 1 tháng 5 năm 2026 và mức định giá 4 tỷ USD trên sản lượng hơn 10 nghìn tỷ token/ngày chứng minh rằng mô hình lấy quy mô sản lượng làm động lực hoàn toàn hiệu quả. Baseten đã hoàn tất vòng gọi vốn Series E trị giá 300 triệu USD ở mức định giá 5 tỷ USD vào tháng 1 năm 2026. Quy tắc định vị cạnh tranh rất đơn giản: Fireworks tối ưu hóa độ trễ, Together tối ưu hóa bề rộng danh mục, Baseten tối ưu hóa sự hoàn thiện cấp doanh nghiệp, Modal tối ưu hóa trải nghiệm nhà phát triển thuần Python (Python-native DX), Replicate tối ưu hóa khả năng tiếp cận đa phương thức (multimodal), và Anyscale tối ưu hóa tính toán Python phân tán. Bài học này cung cấp cho bạn một ma trận so sánh có thể chuyển giao trực tiếp cho một nhà sáng lập công nghệ.

**Type:** Learn
**Languages:** Python (stdlib, bộ so sánh kinh tế chi phí theo lượt gọi mô phỏng)
**Prerequisites:** Phase 17 · 01 (Nền tảng LLM Được Quản lý / Managed LLM Platforms), Phase 17 · 04 (Kiến trúc Nhân Phục vụ / Serving Engine Internals)
**Time:** ~60 phút

## Mục tiêu Học tập

- Gọi tên ba phân khúc thị trường (bán dẫn tùy biến, nền tảng GPU, chợ ưu tiên API) và ánh xạ từng nhà cung cấp vào đúng phân khúc.
- Giải thích lý do tại sao mô hình định giá API "theo từng token" (per-token) chịu áp lực co hẹp tiến sát về đường cong chi phí của nhân phục vụ (serving engine), chứ không phải đường cong chi phí của phần cứng vật lý.
- Tính toán chi phí thực tế cho mỗi lượt yêu cầu (effective cost per request) trên tối thiểu ba nhà cung cấp và giải thích khi nào hình thức tính phí theo phút (Baseten, Modal) đánh bại hình thức tính phí theo token.
- Xác định nền tảng nào là lựa chọn mặc định phù hợp cho một khối lượng công việc nhất định (serverless bùng phát đột biến, thông lượng cao ổn định, các biến thể tinh chỉnh chuyên sâu, đa phương thức).

## Vấn đề Thực tế

Bạn đã đánh giá các nền tảng đám mây lớn được quản lý hoàn toàn. Bạn quyết định rằng mình cần một nhà cung cấp chuyên biệt hơn, nhanh hơn — Fireworks để có độ trễ cực thấp, Together để có danh mục rộng lớn, hoặc Baseten để chạy một mô hình tinh chỉnh (fine-tuned model) tùy biến. Giờ đây bạn đứng trước sáu lựa chọn thực tế và trang biểu phí của họ hoàn toàn không đồng nhất. Fireworks hiển thị $/triệu token; Baseten hiển thị $/phút; Modal hiển thị $/giây; Replicate hiển thị $/lượt dự đoán (prediction). Bạn không thể so sánh trực diện nếu không mô hình hóa khối lượng công việc cụ thể.

Tệ hơn nữa, mô hình kinh doanh đằng sau mỗi trang bảng giá lại hoàn toàn khác nhau. Fireworks chạy nhân tùy biến riêng (FireAttention) trên các GPU dùng chung; đơn giá theo token phản ánh đường cong hiệu suất sử dụng của họ. Baseten cung cấp bộ công cụ Truss kèm các GPU chuyên dụng; mức giá theo phút phản ánh tính độc quyền về tài nguyên phần cứng. Modal là nền tảng serverless thuần túy cho Python — tính phí theo từng giây với khả năng khởi động nguội (cold start) chỉ dưới một giây. Cùng một kết quả đầu ra (câu trả lời từ LLM), nhưng lại có ba hàm chi phí hoàn toàn khác nhau.

Bài học này xây dựng mô hình toán cho cả sáu nhà cung cấp và chỉ ra trường hợp nào mỗi bên sẽ giành chiến thắng.

## Khái niệm

### Ba phân khúc thị trường

**Bán dẫn tùy biến (Custom silicon)** — Groq (LPU), Cerebras (WSE), SambaNova (RDU). Thường đạt tốc độ giải mã (decode) nhanh hơn 5-10 lần so với một cụm máy chủ dùng GPU trên cùng một mô hình. Giá trên mỗi token cao hơn (Groq có giá khoảng ~$0.99/triệu token trên Llama-70B vào cuối năm 2025) nhưng hoàn toàn bất khả chiến bại đối với các trường hợp sử dụng nhạy cảm về độ trễ. Groq là lựa chọn sản xuất hàng đầu cho các tác tử thoại (voice agents) và dịch thuật thời gian thực.

**Nền tảng GPU (GPU platforms)** — Baseten, Together, Fireworks, Modal, Anyscale. Vận hành trên phần cứng NVIDIA (H100, H200, B200 vào năm 2026) hoặc đôi khi là AMD. Đây là lớp kinh tế nằm giữa "thuê GPU thô" (RunPod, Lambda) và "dịch vụ được quản lý của đám mây siêu lớn" (Bedrock).

**Chợ mô hình ưu tiên API (API-first marketplaces)** — Replicate, DeepInfra, OpenRouter, Fal. Danh mục mô hình phong phú, trả tiền theo lượt dự đoán hoặc theo giây, nhấn mạnh vào thời gian để thực hiện lệnh gọi đầu tiên (time-to-first-call).

### Fireworks — nền tảng GPU tối ưu hóa cho độ trễ

- Nhân FireAttention (tự phát triển); được quảng bá giảm 4 lần độ trễ so với vLLM trên cùng cấu hình tương đương.
- Tầng xử lý theo lô (batch tier) với mức giá bằng ~50% so với gói serverless dành cho các khối lượng công việc không đòi hỏi tương tác tức thời.
- Mô hình tinh chỉnh được phục vụ với cùng đơn giá như mô hình gốc (base model) — một yếu tố khác biệt lớn so với các nhà cung cấp thu thêm phụ phí khi bạn nạp trọng số LoRA.
- Giữa năm 2026: tăng giá thuê GPU theo nhu cầu thêm $1/giờ kể từ ngày 1 tháng 5 năm 2026. Mức giá theo sản lượng lớn có thể thương lượng.
- Tín hiệu tài chính: Định giá 4 tỷ USD, xử lý trên 10 nghìn tỷ token mỗi ngày.

### Together — tối ưu hóa cho bề rộng danh mục

- Hơn 200 mô hình, bao gồm việc cập nhật các bản phát hành nguồn mở chỉ trong vòng vài ngày sau khi công bố.
- Rẻ hơn 50-70% so với Replicate trên các mô hình LLM tương đương — định vị "Đám mây AI Bản địa" (AI Native Cloud) dựa trên quy mô sản lượng và danh mục phong phú.
- Tích hợp Suy luận + Tinh chỉnh (fine-tuning) + Huấn luyện (training) trong một API duy nhất.

### Baseten — tối ưu hóa độ hoàn thiện cấp doanh nghiệp

- Framework Truss: đóng gói mô hình (model packaging) kèm các thư viện phụ thuộc, bí mật môi trường (secrets) và cấu hình phục vụ trong một tệp mô tả duy nhất.
- Dải GPU rộng từ T4 tới B200. Tính phí theo phút kèm cơ chế giảm thiểu độ trễ khởi động nguội hợp lý.
- Đạt chứng chỉ SOC 2 Type II, sẵn sàng cho chuẩn HIPAA. Lựa chọn phổ biến của các tổ chức công nghệ tài chính và y tế.
- Định giá 5 tỷ USD, hoàn tất vòng gọi vốn Series E trị giá 300 triệu USD vào tháng 1 năm 2026 (dẫn đầu bởi CapitalG, IVP, NVIDIA).

### Modal — tối ưu hóa trải nghiệm thuần Python

- Hạ tầng dưới dạng mã (Infrastructure-as-code) hoàn toàn bằng Python nguyên bản. Gắn decorator cho hàm bằng `@modal.function(gpu="A100")` và triển khai chỉ bằng một lệnh duy nhất.
- Tính phí chi tiết theo từng giây. Khởi động nguội mất 2-4 giây khi có làm ấm trước (pre-warming); dưới 1 giây đối với các mô hình nhỏ.
- Vòng Series B đạt 87 triệu USD ở mức định giá 1.1 tỷ USD (2025). Đạt điểm số trải nghiệm nhà phát triển cao nhất trong các khảo sát độc lập.

### Replicate — bề rộng đa phương thức

- Trả tiền theo từng lượt dự đoán (pay-per-prediction). Nền tảng mặc định cho các mô hình hình ảnh, video và âm thanh.
- Hệ sinh thái tích hợp phong phú (Zapier, Vercel, các plugin CMS).
- Ít cạnh tranh hơn về đơn giá theo token trên LLM nhưng giành chiến thắng nhờ tính đa dạng của các mô hình đa phương thức.

### Anyscale — gốc rễ từ Ray

- Xây dựng trên nền tảng Ray; RayTurbo là nhân suy luận độc quyền của Anyscale (cạnh tranh trực tiếp với vLLM).
- Tối ưu nhất cho các khối lượng công việc Python phân tán, nơi bước suy luận chỉ là một nút tính toán trong một đồ thị lớn hơn.
- Cụm Ray được quản lý hoàn toàn; tích hợp chặt chẽ với Ray AIR và Ray Serve.

### Tính phí theo token so với theo phút — khi nào mỗi bên thắng thế

Tính phí theo token hợp lý khi khối lượng công việc không nhạy cảm về độ trễ và có lưu lượng bùng phát đột ngột — bạn chỉ trả tiền cho chính xác những gì mình sử dụng. Tính phí theo phút hợp lý khi hiệu suất sử dụng cao và có thể dự đoán được — bạn sẽ đánh bại mức giá theo token một khi làm đầy tải (saturating) GPU.

Quy tắc ước tính: đối với các khối lượng công việc duy trì hiệu suất sử dụng liên tục trên ~30% của một GPU chuyên dụng, tính phí theo phút (Baseten, Modal) bắt đầu rẻ hơn tính phí theo token (Fireworks, Together). Dưới ngưỡng đó, tính phí theo token thắng thế vì bạn tránh được việc phải trả tiền cho thời gian GPU nhàn rỗi (idle).

### Nhân tùy biến là con hào kinh tế thực sự

Mọi nền tảng đứng trên vLLM và SGLang đều tuyên bố sở hữu một nhân phục vụ tùy biến riêng: FireAttention, RayTurbo, ngăn xếp suy luận của Baseten. Những tuyên bố về nhân tùy biến thường mang màu sắc tiếp thị — góc nhìn trung thực là vLLM + SGLang đại diện cho khoảng 80% thị phần suy luận nguồn mở trong môi trường sản xuất, và các yếu tố tạo nên sự khác biệt ở tầng nền tảng nằm ở trải nghiệm nhà phát triển (DX), khả năng phân bổ chi phí và cam kết chất lượng dịch vụ (SLA).

### Những con số bạn cần ghi nhớ

- Mức tăng giá thuê GPU của Fireworks: thêm $1/giờ có hiệu lực từ ngày 1 tháng 5 năm 2026.
- Tuyên bố của Fireworks: độ trễ thấp hơn 4 lần so với vLLM trên cấu hình tương đương.
- Together: Rẻ hơn 50-70% so với Replicate trên các mô hình LLM.
- Định giá Baseten: 5 tỷ USD (Series E, tháng 1 năm 2026, vòng gọi vốn 300 triệu USD).
- Định giá Modal: 1.1 tỷ USD (Series B, 2025).
- Tính phí theo phút đánh bại tính phí theo token khi hiệu suất sử dụng liên tục vượt ngưỡng ~30%.

```figure
cost-per-token
```

## Ứng Dụng Thực Tế (Use It)

Tệp `code/main.py` so sánh cả sáu nhà cung cấp trên một khối lượng công việc mô phỏng xuyên suốt các mô hình định giá. Mã nguồn báo cáo chi phí $/ngày và chi phí thực tế $/triệu token. Hãy chạy mã để tìm ra điểm hòa vốn chính xác giữa phương thức tính phí theo token và theo phút.

## Bàn Giao Thành Phẩm (Ship It)

Bài học này tạo ra tệp artifact `outputs/skill-inference-platform-picker.md`. Dựa trên hồ sơ khối lượng công việc, cam kết SLA và ngân sách, kỹ năng này sẽ chọn ra nền tảng suy luận chính và đề xuất nền tảng dự phòng xếp thứ hai.

## Bài Tập Thực Hành

1. Chạy `code/main.py`. Ở mức hiệu suất sử dụng liên tục nào thì Baseten (tính phí theo phút) đánh bại Fireworks (tính phí theo token) đối với mô hình lớp 70B trên một card H100? Hãy tự mình tính toán điểm giao cắt này và so sánh với quy tắc ước lượng chung.
2. Sản phẩm của bạn phục vụ tính năng sinh hình ảnh kết hợp trò chuyện kèm chuyển giọng nói thành văn bản. Hãy chọn nền tảng tối ưu cho từng phương thức và nêu tên mẫu thiết kế cổng gateway để hợp nhất chúng.
3. Fireworks tăng giá thêm $1/giờ trên mô hình chính của bạn. Hãy mô hình hóa tác động chi phí hỗn hợp nếu 40% lưu lượng truy cập của bạn được chuyển sang tầng xử lý theo lô (batch tier được giảm giá 50%).
4. Một khách hàng thuộc diện quản lý nghiêm ngặt yêu cầu chuẩn SOC 2 Type II + HIPAA + GPU chuyên dụng. Ba nền tảng nào khả thi và nền tảng nào giành chiến thắng về mặt FinOps?
5. So sánh chi phí trên 1.000 lượt dự đoán cho Llama 3.1 70B trên Fireworks serverless, Together on-demand, Baseten dedicated và Replicate API. Lựa chọn nào rẻ nhất ở quy mô 10 lượt dự đoán/ngày? Ở quy mô 10.000 lượt dự đoán/ngày?

## Thuật Ngữ Then Chốt

| Thuật ngữ | Ngôn ngữ thông thường | Ý nghĩa kỹ thuật thực tế |
|-----------|----------------------|--------------------------|
| Bán dẫn tùy biến (Custom silicon) | "Chip không dùng GPU" | Groq LPU, Cerebras WSE, SambaNova RDU — tối ưu hóa chuyên biệt cho pha giải mã |
| FireAttention | "Nhân của Fireworks" | Kernel chú ý tùy biến; được quảng bá giảm 4 lần độ trễ so với vLLM |
| Truss | "Định dạng của Baseten" | Tệp mô tả đóng gói mô hình; gồm thư viện phụ thuộc + bí mật + cấu hình phục vụ |
| Tính phí theo token (Per-token) | "Giá API" | Tính phí theo số lượng token tiêu thụ; không phải trả tiền cho thời gian nhàn rỗi |
| Tính phí theo phút (Per-minute) | "Giá chuyên dụng" | Tính phí theo thời gian thực tế sử dụng GPU; tối ưu chi phí ở mức tải cao |
| Tính phí theo lượt dự đoán (Per-prediction) | "Giá của Replicate" | Tính phí theo từng lần gọi mô hình; phổ biến với mô hình hình ảnh/video |
| RayTurbo | "Nhân của Anyscale" | Nhân suy luận độc quyền chạy trên Ray; cạnh tranh với vLLM trên các cụm Ray |
| Tầng xử lý theo lô (Batch tier) | "Giảm giá 50%" | Hàng đợi phi tương tác với mức phí ưu đãi; phổ biến trên Fireworks, OpenAI |
| Tinh chỉnh theo giá gốc (Fine-tuned at base rate) | "LoRA của Fireworks" | Tính phí các yêu cầu phục vụ qua LoRA bằng đúng giá của mô hình gốc (yếu tố khác biệt) |

## Tài Liệu Đọc Thêm

- [Bảng giá Fireworks](https://fireworks.ai/pricing) — Biểu phí theo token, tầng xử lý theo lô, thuê GPU.
- [Bảng giá Baseten](https://www.baseten.co/pricing/) — Biểu phí theo phút, dung lượng cam kết, các gói doanh nghiệp.
- [Bảng giá Modal](https://modal.com/pricing) — Biểu phí GPU theo giây và gói miễn phí.
- [Bảng giá Together AI](https://www.together.ai/pricing) — Danh mục mô hình và biểu phí theo token.
- [Bảng giá Anyscale](https://www.anyscale.com/pricing) — Chi phí RayTurbo và cụm Ray được quản lý.
- [Northflank — Các giải pháp thay thế Fireworks AI](https://northflank.com/blog/7-best-fireworks-ai-alternatives-for-inference) — Đánh giá so sánh chuyên sâu.
- [Infrabase — Các nhà cung cấp API Suy luận AI 2026](https://infrabase.ai/blog/ai-inference-api-providers-compared) — Toàn cảnh thị trường các nhà cung cấp.
