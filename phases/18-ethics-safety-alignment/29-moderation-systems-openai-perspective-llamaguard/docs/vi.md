# Hệ thống kiểm duyệt: OpenAI, Perspective, Llama Guard (Moderation Systems — OpenAI, Perspective, Llama Guard)

> Các hệ thống kiểm duyệt trong môi trường sản xuất hiện thực hóa các chính sách an toàn đã được định nghĩa tại các Bài 12-16. OpenAI Moderation API: phiên bản `omni-moderation-latest` (2024) được xây dựng trên nền tảng GPT-4o, có khả năng phân loại văn bản + hình ảnh chỉ trong một lệnh gọi duy nhất; cải thiện 42% trên tập kiểm thử đa ngôn ngữ so với phiên bản tiền nhiệm; lược đồ phản hồi (response schema) trả về 13 giá trị boolean tương ứng với 13 danh mục — quấy rối (harassment), quấy rối/đe dọa (harassment/threatening), thù ghét (hate), thù ghét/đe dọa (hate/threatening), bất hợp pháp (illicit), bất hợp pháp/bạo lực (illicit/violent), tự hại (self-harm), tự hại/ý định (self-harm/intent), tự hại/chỉ dẫn (self-harm/instructions), tình dục (sexual), tình dục/trẻ vị thành niên (sexual/minors), bạo lực (violence), bạo lực/đẫm máu (violence/graphic); miễn phí cho hầu hết các nhà phát triển. Mô hình phân tầng: Kiểm duyệt đầu vào (Input moderation - tiền sinh văn bản), Kiểm duyệt đầu ra (Output moderation - hậu sinh văn bản), Kiểm duyệt tùy biến (Custom moderation - quy tắc theo miền nghiệp vụ). Các lệnh gọi song song bất đồng bộ giúp triệt tiêu độ trễ; một phản hồi giữ chỗ có thể lấp đầy thời gian chờ trong khi khâu kiểm duyệt đầu vào hoàn tất. Llama Guard 3/4 (Bài 16): 14 danh mục nguy cơ theo tiêu chuẩn MLCommons, Lạm dụng trình thông dịch mã (Code Interpreter Abuse), 8 ngôn ngữ (v3), hỗ trợ đa hình ảnh (v4). Perspective API (Google Jigsaw): hệ thống chấm điểm độ độc hại (toxicity scoring) ra đời trước làn sóng sử dụng LLM làm bộ kiểm duyệt; chủ yếu chấm điểm độc hại đơn chiều kèm các biến thể con; là baseline kinh điển cho nghiên cứu kiểm duyệt nội dung. Lộ trình ngừng hỗ trợ: Azure Content Moderator ngừng phát triển từ tháng 2/2024, ngừng hoạt động hoàn toàn vào tháng 2/2027, được thay thế bởi Azure AI Content Safety.

**Type:** Build
**Languages:** Python (stdlib, three-layer moderation harness)
**Prerequisites:** Phase 18 · 16 (Llama Guard / Garak / PyRIT)
**Time:** ~60 phút

## Mục tiêu học tập

- Mô tả hệ thống phân loại danh mục (taxonomy) của OpenAI Moderation API và điểm khác biệt so với bộ tiêu chuẩn MLCommons của Llama Guard 3.
- Trình bày mô hình kiểm duyệt ba tầng (đầu vào, đầu ra, tùy biến) và chỉ ra một dạng thức lỗi hỏng (failure mode) điển hình của từng tầng.
- Phân tích vị thế của Perspective API như một baseline tiêu chuẩn thời kỳ tiền LLM và lý do công cụ này vẫn được ứng dụng rộng rãi trong nghiên cứu học thuật.
- Nắm rõ mốc thời gian chuyển đổi và ngừng hỗ trợ (deprecation timeline) của dịch vụ Azure Content Moderator.

## Vấn đề đặt ra

Các Bài 12-16 tập trung phân tích các phương thức tấn công và công cụ phòng thủ. Bài 29 đi sâu vào các hệ thống kiểm duyệt thực tế đang được vận hành để triển khai các biện pháp phòng thủ ngay tại bề mặt tiếp xúc trực tiếp giữa người dùng và sản phẩm. Mô hình kiểm duyệt ba tầng chính là cấu hình mặc định tiêu chuẩn trong môi trường sản xuất năm 2026.

## Khái niệm cốt lõi

### OpenAI Moderation API

Phiên bản `omni-moderation-latest` (2024). Được phát triển trên nền tảng GPT-4o. Có khả năng phân loại đồng thời cả văn bản lẫn hình ảnh trong cùng một lệnh gọi. Miễn phí cho đại đa số nhà phát triển.

Các danh mục kiểm duyệt (13 giá trị boolean trong lược đồ phản hồi):
- harassment (quấy rối), harassment/threatening (quấy rối có tính chất đe dọa)
- hate (thù ghét), hate/threatening (thù ghét có tính chất đe dọa)
- self-harm (tự hại), self-harm/intent (ý định tự hại), self-harm/instructions (chỉ dẫn tự hại)
- sexual (nội dung tình dục), sexual/minors (nội dung tình dục liên quan đến trẻ vị thành niên)
- violence (bạo lực), violence/graphic (bạo lực đẫm máu/chi tiết)
- illicit (hành vi bất hợp pháp), illicit/violent (hành vi bất hợp pháp có bạo lực)

Khả năng hỗ trợ đa phương thức (multimodal) áp dụng đối với `violence`, `self-harm`, và `sexual`, nhưng không áp dụng cho `sexual/minors`; các danh mục còn lại chỉ xử lý trên phương thức văn bản.

Trong khung thực hành mã nguồn tại `code/main.py`, chúng ta sẽ gộp các danh mục phụ như `/threatening`, `/intent`, `/instructions`, và `/graphic` vào danh mục cha tương ứng nhằm phục vụ mục đích minh họa sư phạm một cách tinh gọn. Trong mã nguồn môi trường sản xuất thực tế, bắt buộc phải sử dụng đầy đủ toàn bộ lược đồ 13 danh mục này.

Endpoint kiểm duyệt thế hệ mới này đạt hiệu năng phân loại cao hơn 42% trên tập kiểm thử đa ngôn ngữ so với thế hệ trước đó. API trả về điểm số xác suất cho từng danh mục; các ứng dụng triển khai sẽ tự thiết lập các ngưỡng kích hoạt (thresholds) phù hợp với chính sách của mình.

### Llama Guard 3/4

Đã được giới thiệu chi tiết tại Bài 16. Sử dụng 14 danh mục nguy hại theo tiêu chuẩn MLCommons (được tổ chức theo cấu trúc khác biệt so với 13 giá trị boolean của OpenAI). Hỗ trợ 8 ngôn ngữ (phiên bản v3). Llama Guard 4 (tháng 4/2025) là mô hình 12B hỗ trợ đa phương thức bản địa.

Hệ thống phân loại của OpenAI và Llama Guard có phần giao thoa nhưng cũng có những điểm phân kỳ đáng kể. OpenAI định nghĩa một danh mục khái quát rộng là "illicit" (bất hợp pháp); trong khi Llama Guard tách biệt rõ ràng thành "violent crimes" (tội phạm bạo lực) và "non-violent crimes" (tội phạm phi bạo lực). Tùy thuộc vào chính sách an toàn của tổ chức, hệ thống triển khai sẽ lựa chọn mô hình có hệ phân loại phù hợp nhất.

### Perspective API (Google Jigsaw)

Hệ thống chấm điểm độ độc hại ra đời trước làn sóng ứng dụng LLM làm bộ kiểm duyệt (giai đoạn trước năm 2020). Các danh mục chính gồm: TOXICITY (độ độc hại), SEVERE_TOXICITY (độ độc hại nghiêm trọng), INSULT (xúc phạm), PROFANITY (tục tĩu), THREAT (đe dọa), IDENTITY_ATTACK (tấn công danh tính). Hệ thống sử dụng một điểm số đơn chiều cốt lõi (TOXICITY) đi kèm các điểm số thành phần phụ trợ.

Perspective API được ứng dụng rộng rãi như một chuẩn đối chuẩn (baseline) trong nghiên cứu kiểm duyệt nội dung nhờ vào tính ổn định cao, tài liệu đầy đủ và tập dữ liệu hiệu chuẩn phong phú được tích lũy qua nhiều năm. Tuy nhiên, đối với các trường hợp sử dụng gắn liền với hệ sinh thái LLM hiện đại, Llama Guard hoặc OpenAI Moderation thường là sự lựa chọn phù hợp hơn.

### Mô hình kiểm duyệt ba tầng (The three-layer pattern)

1. **Kiểm duyệt đầu vào (Input moderation):** Phân loại câu lệnh của người dùng trước khi mô hình bắt đầu sinh nội dung. Từ chối yêu cầu ngay lập tức nếu bị gắn cờ. Độ trễ phát sinh: tương đương một lệnh gọi phân loại.
2. **Kiểm duyệt đầu ra (Output moderation):** Phân loại phản hồi của mô hình trước khi gửi về cho người dùng cuối. Thay thế bằng câu từ chối nếu nội dung sinh ra bị gắn cờ độc hại. Độ trễ phát sinh: tương đương một lệnh gọi phân loại sau khi quá trình sinh văn bản hoàn tất.
3. **Kiểm duyệt tùy biến (Custom moderation):** Áp dụng các quy tắc đặc thù theo miền nghiệp vụ (biểu thức chính quy - regex, danh sách trắng/đen - allowlists/denylists, quy chuẩn chính sách doanh nghiệp). Có thể kích hoạt linh hoạt ở tầng đầu vào hoặc tầng đầu ra.

Về mặt thiết kế kiến trúc, ba tầng này hoạt động theo cơ chế tuần tự: việc kiểm duyệt đầu vào bắt buộc phải hoàn thành trước khi chuyển sang khâu sinh nội dung, và việc kiểm duyệt đầu ra chỉ được thực hiện sau khi mô hình đã hoàn tất việc sinh phản hồi. Tuy nhiên, cơ chế song song có thể được áp dụng bên trong mỗi tầng — chạy đồng thời nhiều bộ phân loại (ví dụ: OpenAI Moderation + Llama Guard + Perspective) trên cùng một khối văn bản giúp che giấu và triệt tiêu độ trễ tích lũy của từng bộ phân loại riêng lẻ. Như một kỹ thuật tối ưu hóa tùy chọn, hệ thống có thể hiển thị một phản hồi giữ chỗ cho người dùng ("hệ thống đang xử lý trong giây lát...") trong khi quy trình kiểm duyệt đầu vào đang chạy, nhằm tạm hoãn việc truyền luồng (streaming) token đầu tiên. Hành vi phản hồi khi bị gắn cờ có thể tùy biến linh hoạt: từ chối hoàn toàn, thanh lọc nội dung (sanitize), hoặc chuyển tiếp lên cấp phê duyệt của con người (escalate to human review).

### Các dạng thức lỗi hỏng (Failure modes)

- **Chỉ kiểm duyệt đầu vào (Input only):** Hoàn toàn bất lực trước hiện tượng ảo giác (hallucination) ở đầu ra (các kỹ thuật tấn công mã hóa tại Bài 12-14 có thể dễ dàng đi vòng qua bộ phân loại đầu vào để kích hoạt đầu ra độc hại).
- **Chỉ kiểm duyệt đầu ra (Output only):** Cho phép mọi câu lệnh độc hại tiếp cận mô hình; gây lãng phí chi phí tài nguyên tính toán; làm lộ luồng suy luận nội bộ cho kẻ tấn công trước khi bị chặn.
- **Chỉ kiểm duyệt tùy biến (Custom only):** Không có tính khái quát bền bỉ xuyên suốt các danh mục vi phạm phức tạp; các biểu thức chính quy (regex) rất giòn và dễ bị vô hiệu hóa bởi các biến thể từ ngữ.

Kiểm duyệt phân tầng là kiến trúc mặc định chuẩn mực — tuân thủ nguyên lý phòng thủ nhiều lớp "dây đeo quần kết hợp thắt lưng" (belt-and-suspenders).

### Lộ trình ngừng hỗ trợ của Azure (Azure deprecation)

Dịch vụ Azure Content Moderator: chính thức ngừng phát triển (deprecated) từ tháng 2 năm 2024, và sẽ bị khai tử hoàn toàn (retired) vào tháng 2 năm 2027. Dịch vụ thay thế là Azure AI Content Safety, vốn được xây dựng trên nền tảng LLM và tích hợp sâu với Azure OpenAI. Việc di chuyển nâng cấp giữa hai hệ thống này là một dự án hạ tầng trọng điểm trong giai đoạn 2024-2027 đối với các hệ thống vận hành trên Azure.

### Vị trí của bài học trong Giai đoạn 18

Bài 16 nghiên cứu các công cụ kiểm duyệt trong bối cảnh kiểm thử xâm nhập (red-team). Bài 29 tập trung vào các hệ thống kiểm duyệt khi đưa vào môi trường vận hành thực tế. Bài 30 sẽ khép lại giai đoạn với các bằng chứng thực nghiệm về năng lực lưỡng dụng nguy hiểm.

```figure
an-moderation-layers
```

## Thực hành (Use It)

Tệp `code/main.py` xây dựng một khung kiểm duyệt ba tầng hoàn chỉnh: kiểm duyệt đầu vào (từ khóa + điểm số danh mục), kiểm duyệt đầu ra (bộ phân loại tương tự áp dụng trên văn bản đầu ra), và kiểm duyệt tùy biến (các quy tắc nghiệp vụ). Bạn có thể thử nghiệm truyền các đầu vào khác nhau để quan sát trực tiếp tầng nào sẽ phát hiện và xử lý vi phạm.

## Đóng gói sản phẩm (Ship It)

Bài học này cung cấp sản phẩm `outputs/skill-moderation-stack.md`. Đối với một hệ thống triển khai cụ thể, kỹ năng này sẽ đề xuất cấu hình ngăn xếp kiểm duyệt tối ưu: nên sử dụng bộ phân loại nào ở đầu vào, bộ phân loại nào ở đầu ra, những quy tắc tùy biến nào cần áp dụng, và thiết lập mô hình giám khảo nào cho các trường hợp biên khó phân xử.

## Bài tập

1. Chạy `code/main.py`. Đưa lần lượt một đầu vào lành tính (benign), một đầu vào ở vùng ranh giới (borderline), và một đầu vào độc hại (harmful) qua cả ba tầng kiểm duyệt. Báo cáo cụ thể tầng nào đã kích hoạt xử lý đối với từng trường hợp.

2. Mở rộng khung kiểm duyệt bằng cách tích hợp logic chấm điểm độ độc hại theo phong cách Perspective API trên một danh mục cụ thể. So sánh hành vi phản ứng theo ngưỡng kích hoạt của nó với điểm số danh mục thông thường.

3. Đọc tài liệu OpenAI Moderation API và danh sách phân loại của Llama Guard 3. Ánh xạ từng danh mục của OpenAI sang các danh mục tương ứng gần nhất của Llama Guard. Chỉ ra ba danh mục không thể ánh xạ một-một một cách trọn vẹn.

4. Thiết kế ngăn xếp kiểm duyệt cho một ứng dụng trợ lý lập trình (ví dụ: GitHub Copilot). Xác định những danh mục kiểm duyệt nào có độ liên quan cao nhất, danh mục nào ít liên quan nhất và đề xuất bộ quy tắc tùy biến tương ứng.

5. Dịch vụ Azure Content Moderator sẽ chính thức ngừng hoạt động vào tháng 2 năm 2027. Hãy lập kế hoạch di chuyển sang Azure AI Content Safety và chỉ ra yếu tố rủi ro kỹ thuật cao nhất trong quá trình di chuyển này.

## Thuật ngữ then chốt

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa kỹ thuật thực tế |
|---|---|---|
| OpenAI Moderation | "omni-moderation-latest" | Bộ phân loại 13 danh mục (văn bản) dựa trên GPT-4o với khả năng hỗ trợ đa phương thức một phần |
| Perspective API | "chấm điểm độc hại của Google Jigsaw" | Baseline chấm điểm độ độc hại tiêu chuẩn từ thời kỳ tiền LLM |
| Llama Guard | "bộ phân loại 14 danh mục MLCommons" | Bộ phân loại nguy hại của Meta (v3: 8B văn bản, 8 ngôn ngữ; v4: 12B đa phương thức) |
| Input moderation | "bộ lọc tiền sinh" | Bộ phân loại đánh giá câu lệnh của người dùng trước khi gọi mô hình chính |
| Output moderation | "bộ lọc hậu sinh" | Bộ phân loại đánh giá văn bản do mô hình sinh ra trước khi trả về cho người dùng |
| Custom moderation | "quy tắc nghiệp vụ theo miền" | Các quy tắc riêng của hệ thống triển khai (regex, danh sách trắng, chính sách) |
| Layered moderation | "kiểm duyệt cả ba tầng" | Mô hình kiến trúc triển khai tiêu chuẩn trong môi trường sản xuất |

## Đọc thêm

- [OpenAI Moderation API docs](https://platform.openai.com/docs/api-reference/moderations) — Tài liệu chi tiết về endpoint omni-moderation
- [Meta PurpleLlama + Llama Guard](https://github.com/meta-llama/PurpleLlama) — Kho lưu trữ mã nguồn Llama Guard
- [Google Jigsaw Perspective API](https://perspectiveapi.com/) — Nền tảng chấm điểm độ độc hại
- [Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/) — Nền tảng kiểm duyệt nội dung thế hệ mới của Azure
