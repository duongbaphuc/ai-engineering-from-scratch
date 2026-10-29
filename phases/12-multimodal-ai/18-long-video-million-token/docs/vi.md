# Hiểu Video Dài với Ngữ Cảnh Triệu Token

> Một video 4K dài 1 giờ ở tốc độ 24 FPS sau khi chia mảng ảnh và nhúng sẽ tạo ra khoảng 60 triệu token. Một tập podcast 2 giờ khi được chuyển văn bản chứa khoảng 30.000 token. Một bộ phim điện ảnh chuẩn Blu-ray, ngay cả khi được nén bằng cách gộp mạnh mẽ, vẫn tiêu tốn hàng trăm nghìn token. Gemini 1.5 của Google (tháng 3 năm 2024) đã mở ra kỷ nguyên này với cửa sổ ngữ cảnh 10 triệu token, thực hiện việc truy hồi "tìm kim đáy bể" (needle-in-a-haystack) một cách tin cậy trên các video dài hàng giờ. LWM (Liu et al., tháng 2 năm 2024) đã chỉ ra lộ trình mở rộng quy mô bằng cơ chế chú ý vòng (ring attention). LongVILA và Video-XL tiếp tục mở rộng quy mô nạp dữ liệu. VideoAgent thay thế việc nạp toàn bộ ngữ cảnh thô bằng truy xuất theo hướng tác tử (agentic retrieval). Mỗi phương pháp đại diện cho một sự đánh đổi khác nhau giữa năng lực tính toán, độ chính xác truy hồi và độ phức tạp kỹ thuật. Bài học này sẽ đặt các phương pháp đó lên bàn cân so sánh song song.

**Loại:** Xây dựng (Build)  
**Ngôn ngữ:** Python (thư viện chuẩn stdlib, bộ mô phỏng tìm kim đáy bể + bộ định tuyến truy xuất theo hướng tác tử)  
**Điều kiện tiên quyết:** Giai đoạn 12 · 17 (token thời gian trong video)  
**Thời lượng:** ~180 phút

## Mục tiêu học tập

- Tính toán tổng số lượng token thị giác cho video dài ở các mức FPS và tỷ lệ gộp khác nhau.
- Giải thích bốn lộ trình mở rộng quy mô: ngữ cảnh thuần túy (brute context - Gemini 1.5), chú ý vòng (ring attention - LWM), nén token (token compression - LongVILA / Video-XL), và truy xuất theo hướng tác tử (agentic retrieval - VideoAgent).
- So sánh các mô hình video VLM dùng ngữ cảnh thô với video VLM truy xuất tác tử (VideoAgent) về độ chính xác và độ trễ.
- Thiết kế một bài kiểm tra "tìm kim đáy bể" (needle-in-a-haystack) cho video dài 30 phút và đo lường tỷ lệ truy hồi (recall) tại một phút cụ thể.

## Vấn đề

Một khung hình đơn lẻ với các mảng ảnh kích thước tương đương Qwen2.5-VL ở độ phân giải gốc 384 tạo ra ~729 token. Khi gộp song tuyến tính 3x3, con số này giảm xuống 81 token mỗi khung hình. Một đoạn clip 30 phút ở tốc độ 1 FPS = 1800 khung hình = 145.800 token. Con số này nằm trong tầm xử lý của các VLM nguồn mở năm 2025, nhưng khá sít sao. Ở tốc độ 2 FPS, lượng token lên tới 291.600 — chỉ những mô hình có cửa sổ ngữ cảnh lớn nhất mới có thể chứa vừa.

Một bộ phim dài 2 giờ ở tốc độ 1 FPS tiêu tốn 583k token. Vượt quá giới hạn của hầu hết các mô hình mở năm 2026; đòi hỏi phải dùng đến Gemini 2.5 Pro hoặc phải áp dụng các kỹ thuật gộp khung hình mạnh mẽ hơn nữa.

Bốn lộ trình mở rộng quy mô đã xuất hiện để giải quyết bài toán này.

## Khái niệm

### Hướng tiếp cận 1: Mở rộng ngữ cảnh thuần túy (Brute context - Gemini 1.5, Claude Opus)

Giải quyết bài toán bằng sức mạnh phần cứng: mở rộng cửa sổ ngữ cảnh lên hàng triệu token và xử lý mọi thứ trong một lượt truyền xuôi (forward pass) duy nhất.

Gemini 1.5 Pro ra mắt với ngữ cảnh 1M token; Gemini 1.5 Ultra mở rộng lên 10M; và Gemini 2.5 Pro vào năm 2026 có thể xử lý các video dài nhiều giờ một cách tin cậy. Bài báo kỹ thuật (arXiv:2403.05530) ghi nhận tỷ lệ truy hồi tìm kim đáy bể đạt 99,7% ở quy mô ngữ cảnh lên tới ~9.5M token.

Về mặt kỹ thuật: triển khai cơ chế chú ý tùy biến với hệ thống phân cấp bộ nhớ (cục bộ + toàn cục + thưa - local + global + sparse) kết hợp cùng cơ chế định tuyến chuyên gia MoE để đạt hiệu quả cao trên ngữ cảnh dài. Chi tiết triển khai không được công bố đầy đủ và không phải là mã nguồn mở.

### Hướng tiếp cận 2: Cơ chế chú ý vòng (Ring attention - LWM, LongVILA)

Cơ chế chú ý vòng (Ring attention) phân tán các chuỗi dữ liệu rất dài trên nhiều thiết bị tính toán được kết nối theo dạng "vòng tròn" (ring), trong đó mỗi thiết bị nắm giữ một khối dữ liệu (chunk). Phép tính chú ý trên toàn bộ chuỗi được thực hiện bằng cách: mỗi thiết bị luân chuyển khối dữ liệu của mình sang thiết bị tiếp theo theo cấu trúc vòng, tính toán sự chú ý cục bộ phần dư (partial attention), rồi tổng hợp lại kết quả.

LWM (Liu et al., 2024) đã huấn luyện một mô hình ngữ cảnh 1M token theo cách này. Bộ nhớ trên mỗi thiết bị chỉ tỷ lệ thuận với kích thước khối dữ liệu mà thiết bị đó nắm giữ, vì vậy việc bổ sung thêm thiết bị sẽ giúp kéo dài ngữ cảnh gần như theo đường tuyến tính. Tổng chi phí tính toán attention vẫn tăng theo hàm bậc hai (quadratic) theo độ dài ngữ cảnh; cấu trúc vòng giúp chia sẻ khối lượng công việc đó cho các thiết bị và gối đầu (overlap) việc truyền khóa/giá trị (K/V) với tính toán.

LongVILA (arXiv:2408.10188) đã áp dụng mô hình này cho các VLM. Xử lý video 1400 khung hình ở mức 192 token mỗi khung hình = ngữ cảnh 268k, được huấn luyện bằng ring attention với mức độ song song hóa 8 nhánh.

### Hướng tiếp cận 3: Nén token (Token compression - Video-XL, LongVA)

Tiết kiệm chi phí hơn so với việc mở rộng ngữ cảnh thuần túy: nén thật mạnh lượng token trước khi đưa chuỗi vào LLM.

Video-XL (arXiv:2409.14485) sử dụng một token tóm tắt thị giác (visual summary token): mỗi đoạn clip gồm N khung hình sẽ sản sinh ra một token "tóm tắt" duy nhất có khả năng chú ý trên toàn bộ N khung hình đó. Khi suy luận, LLM chỉ nhìn thấy một token tóm tắt cho mỗi đoạn clip, nhờ đó thu nhỏ đáng kể cửa sổ ngữ cảnh.

LongVA mở rộng ngữ cảnh LLM từ 200k lên 2M thông qua kỹ thuật "chuyển giao ngữ cảnh dài" (long context transfer). Mô hình được huấn luyện trên văn bản ngữ cảnh dài, sau đó chuyển giao sang video ngữ cảnh dài thông qua biểu diễn dùng chung.

Nén token chấp nhận đánh đổi độ chính xác truy hồi tại các mốc thời gian cụ thể để đổi lấy khả năng mở rộng quy mô. Mô hình nắm được bức tranh tổng quan về những gì đã xảy ra nhưng đôi khi bỏ lỡ các khung hình chính xác.

### Hướng tiếp cận 4: Truy xuất theo hướng tác tử (Agentic retrieval - VideoAgent)

Thay vì đưa toàn bộ video vào LLM, hãy xem video như một cơ sở dữ liệu và sử dụng một LLM làm tác tử truy vấn dữ liệu đó.

VideoAgent (arXiv:2403.10517):

1. LLM tiếp nhận và phân tích câu hỏi.
2. LLM yêu cầu công cụ truy xuất tìm kiếm các đoạn clip liên quan ("hãy tìm cho tôi các phân đoạn có xuất hiện con mèo").
3. Công cụ trả về các mốc thời gian của các đoạn clip khớp yêu cầu.
4. LLM đọc các đoạn clip đó thông qua một VLM.
5. LLM tổng hợp câu trả lời hoặc tiếp tục đưa ra các truy vấn bổ sung.

Đây chính là mô hình tác tử LLM (LLM-as-agent) được áp dụng cho bài toán video dài. Chi phí suy luận rẻ hơn nhiều (chỉ mã hóa các đoạn clip thực sự liên quan), nhưng độ phức tạp kỹ thuật tăng lên (chất lượng công cụ truy xuất trở thành nút thắt cổ chai của toàn hệ thống).

### Bộ chuẩn đánh giá "Tìm kim đáy bể" (Needle-in-a-haystack benchmarks)

Bài kiểm tra tiêu chuẩn cho năng lực ngữ cảnh dài: chèn một dấu hiệu thị giác hoặc văn bản độc nhất vào một vị trí ngẫu nhiên trong video, sau đó đặt một câu hỏi yêu cầu mô hình phải nhớ lại chính xác dấu hiệu đó.

Chỉ số đo lường: Tỷ lệ truy hồi (Recall@k) theo độ dài video và vị trí chèn dấu hiệu.

Gemini 2.5 Pro đạt độ chính xác >99% trên các video có thời lượng lên tới 90 phút. Các mô hình mở 72B (Qwen2.5-VL-72B, InternVL3-78B) đạt ~85-90% ở thời lượng 30 phút và suy giảm nhanh khi vượt quá 60 phút.

VideoAgent có thể sánh ngang hoặc vượt qua các mô hình ngữ cảnh thô trên các video dài từ 2 giờ trở lên vì cơ chế truy xuất có thể tìm thấy đúng "cây kim" nếu công cụ tìm kiếm hoạt động hiệu quả.

### Nên lựa chọn hướng đi nào (Which path to pick)

Đối với video dưới 15 phút đòi hỏi độ chính xác cao nhất: mô hình mở 72B + ngữ cảnh nguyên bản thường đáp ứng tốt. Lựa chọn tiêu chuẩn: Qwen2.5-VL-72B.

Đối với nội dung từ 30 phút đến 1 giờ: LongVILA hoặc Video-XL cho giải pháp mã nguồn mở; Gemini 2.5 Pro cho giải pháp thương mại đóng. Ranh giới chất lượng rất quan trọng — các bài toán tiệm cận frontier ưu tiên mô hình đóng.

Đối với nội dung trên 2 giờ: VideoAgent hoặc các mô hình truy xuất tương tự. Ngoài ra có thể tóm tắt video thành các đoạn nhỏ và cung cấp tóm tắt phân cấp.

### Quy trình sản xuất năm 2026 (2026 production pattern)

Trong thực tế, các hệ thống xử lý video dài trên môi trường sản xuất thường kết hợp dạng lai (hybrid):

1. Chạy lấy mẫu FPS động + gộp mạnh mẽ trên toàn bộ video (tạo ra một biểu diễn toàn cục ~100k token).
2. Đưa qua một VLM 72B để tạo bản tóm tắt toàn cục.
3. Nếu người dùng đặt câu hỏi chi tiết, kích hoạt quy trình truy xuất tác tử (agentic retrieval) bằng cách sử dụng bản tóm tắt làm chỉ mục (index).

Sự kết hợp này vừa tận dụng được ngữ cảnh lớn để hiểu bức tranh toàn cảnh, vừa sử dụng truy xuất để nắm bắt chi tiết cục bộ.

````figure
mm-video-token-budget
````

## Thực hành (Use It)

`code/main.py`:

- Tính toán ngân sách token cho các video từ 1 phút đến 3 giờ ở các mức FPS và tỷ lệ gộp khác nhau.
- Mô phỏng một bài kiểm tra "tìm kim đáy bể": chèn một dấu hiệu tại một mốc thời gian ngẫu nhiên, đặt câu hỏi và chấm điểm tỷ lệ truy hồi.
- Tích hợp một bộ định tuyến truy xuất tác tử mô phỏng, có nhiệm vụ chọn lọc các đoạn clip cụ thể để đưa vào VLM hạ nguồn.

Hãy chạy bảng tính ngân sách để cảm nhận trực quan khoảng cách về quy mô dữ liệu.

## Triển khai (Ship It)

Bài học này tạo ra `outputs/skill-long-video-strategy-planner.md`. Với một thời lượng video và độ phức tạp của truy vấn cho trước, kỹ năng này sẽ lựa chọn giữa ngữ cảnh thuần túy (brute-context), nén token hoặc truy xuất tác tử, đồng thời tính toán kỳ vọng về độ trễ và chất lượng.

## Bài tập (Exercises)

1. Một bài giảng dài 45 phút ở tốc độ 1 FPS, 81 token mỗi khung hình. Tổng số token là bao nhiêu? Chuỗi này vừa vặn với cửa sổ ngữ cảnh của những mô hình nào?

2. Thiết kế một bài kiểm tra "tìm kim đáy bể": bạn sẽ chèn dấu hiệu vào phút thứ mấy, và định dạng câu hỏi kiểm tra chính xác là gì?

3. So sánh mô hình ngữ cảnh thô Qwen2.5-VL-72B (ngữ cảnh 80k) với VideoAgent (Claude 3.5 + công cụ truy xuất) trên một video dài 1 giờ. Phương pháp nào giành chiến thắng về độ chính xác truy hồi? Phương pháp nào giành chiến thắng về độ trễ?

4. Bộ nhớ trên mỗi thiết bị của cơ chế chú ý vòng (ring attention) tỷ lệ thuận với độ dài chuỗi chia cho số lượng thiết bị, trong khi tổng tính toán attention vẫn là hàm bậc hai. Hãy giải thích nguyên nhân và điều gì sẽ thất bại nếu bỏ qua pha xoay vòng (ring-rotation phase)?

5. Đọc Mục 5 của bài báo Gemini 1.5 về thử nghiệm tìm kim đáy bể. Bài báo phát hiện thấy điều gì về tỷ lệ truy hồi tại ranh giới giữa 1M và 10M token?

## Thuật ngữ chính (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực sự |
|---|---|---|
| Brute context (Ngữ cảnh thuần túy) | "Cứ tăng thêm token" | Mở rộng cửa sổ ngữ cảnh LLM lên hàng triệu token; xử lý toàn bộ trong một lượt truyền xuôi |
| Ring attention (Cơ chế chú ý vòng) | "Song song kiểu LWM" | Mô hình chú ý phân tán trong đó mỗi thiết bị nắm giữ một khối dữ liệu và luân chuyển theo vòng |
| Token compression (Nén token) | "Token tóm tắt" | Cắt giảm số lượng token trên mỗi clip thông qua một bộ nén học được trước khi đưa vào LLM |
| Needle-in-haystack (Tìm kim đáy bể) | "Kiểm thử NIH" | Chèn một dấu hiệu độc nhất vào vị trí ngẫu nhiên, yêu cầu mô hình nhớ lại chính xác khi kiểm tra |
| Agentic retrieval (Truy xuất theo hướng tác tử) | "LLM lập kế hoạch truy vấn" | LLM yêu cầu công cụ truy xuất tìm clip liên quan, đọc chúng qua VLM, rồi tổng hợp câu trả lời |
| VideoAgent | "Mô hình truy xuất cho video" | Thiết kế truy xuất tác tử chuẩn mực: câu hỏi -> công cụ -> đoạn clip -> câu trả lời |

## Tài liệu tham khảo (Further Reading)

- [Gemini Team — Gemini 1.5 (arXiv:2403.05530)](https://arxiv.org/abs/2403.05530)
- [Liu et al. — LWM / RingAttention (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Xue et al. — LongVILA (arXiv:2408.10188)](https://arxiv.org/abs/2408.10188)
- [Shu et al. — Video-XL (arXiv:2409.14485)](https://arxiv.org/abs/2409.14485)
- [Wang et al. — VideoAgent (arXiv:2403.10517)](https://arxiv.org/abs/2403.10517)
