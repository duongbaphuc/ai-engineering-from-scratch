# Mô hình Video-Ngôn ngữ: Token Thời gian và Định vị Thời gian

> Video không đơn thuần là một chuỗi các bức ảnh xếp chồng lên nhau. Một đoạn clip 5 giây chứa đựng thứ tự nhân quả, động từ hành động và mốc thời gian diễn ra sự kiện mà một mô hình hình ảnh tĩnh không thể biểu diễn được. Video-LLaMA (Zhang et al., tháng 6 năm 2023) là mô hình video-LLM nguồn mở đầu tiên có khả năng định vị thính giác - thị giác (audio-visual grounding). VideoChat và Video-LLaVA tiếp tục mở rộng quy mô mô hình này. Đến năm 2025, cơ chế M-RoPE căn chỉnh theo thời gian (time-aligned M-RoPE) của Qwen2.5-VL đã thu hẹp khoảng cách với các mô hình đóng thương mại hàng đầu. Mỗi hệ thống giải quyết bài toán token thời gian theo một cách riêng — Q-former trên từng đoạn clip, gộp nối tiếp trên từng khung hình, hay M-RoPE căn chỉnh thời gian trên từng token. Bài học này sẽ phân tích các mô hình kiến trúc đó, xây dựng bộ lấy mẫu khung hình đồng đều so với lấy mẫu động, và đánh giá trên các tác vụ định vị thời gian (temporal grounding).

**Loại:** Xây dựng (Build)  
**Ngôn ngữ:** Python (thư viện chuẩn stdlib, bộ lấy mẫu khung hình + bộ đánh giá định vị thời gian)  
**Điều kiện tiên quyết:** Giai đoạn 12 · 08 (LLaVA-OneVision)  
**Thời lượng:** ~180 phút

## Mục tiêu học tập

- Giải thích vì sao việc mã hóa vị trí thời gian (temporal positional encoding) làm thay đổi hiệu năng của video VLM độc lập với bộ mã hóa thị giác (vision encoder).
- So sánh các phương pháp lấy mẫu khung hình đồng đều (uniform), FPS động (dynamic-FPS) và lấy mẫu theo sự kiện (event-driven) về số lượng token trên mỗi giây so với độ chính xác định vị.
- Mô tả các thiết kế: Q-former trên từng đoạn clip (Video-LLaMA) so với gộp trên từng khung hình (Video-LLaVA) so với M-RoPE căn chỉnh theo thời gian trên từng token (Qwen2.5-VL).
- Liệt kê bốn bộ chuẩn đánh giá video: VideoMME, TempCompass, EgoSchema, Video-MMMU.

## Vấn đề

Một video dài 1 phút ở tốc độ 30 FPS chứa 1800 khung hình. Với 196 token thị giác trên mỗi khung hình (ViT-B ở độ phân giải 224), lượng token lên tới 352k — vượt qua bất kỳ cửa sổ ngữ cảnh LLM nào ở kỷ nguyên 2024.

Có ba chiến lược cắt giảm token chính:

1. Lấy mẫu khung hình con (Subsample frames): giảm xuống 1-8 FPS tùy thuộc vào nội dung.
2. Gộp mạnh mẽ các token mảng ảnh của từng khung hình (Pool each frame's patch tokens): dùng gộp song tuyến tính (bilinear pool) 3x3 hoặc 4x4.
3. Nén qua một Q-former: tiếp nhận một đoạn clip 16 khung hình và xuất ra 64 token.

Mỗi phương pháp đều có sự đánh đổi riêng: Lấy mẫu khung hình con làm mất chi tiết thời gian (temporal detail). Gộp token làm mất chi tiết không gian (spatial detail). Q-former làm giảm nhẹ cả hai yếu tố nhưng tiết kiệm đáng kể số lượng token.

Mã hóa vị trí thời gian là một trục quan trọng khác: làm thế nào để mô hình biết khung hình thứ 5 diễn ra trước khung hình thứ 6? Các lựa chọn bao gồm RoPE thời gian 1D đơn giản (Video-LLaMA), nhúng thời gian có thể học được (Video-LLaVA), và M-RoPE căn chỉnh theo thời gian (Qwen2.5-VL, không gian 3D đầy đủ).

## Khái niệm

### Video-LLaMA: Q-former cho từng đoạn clip + nhánh âm thanh (Video-LLaMA: Q-former per clip + audio branch)

Video-LLaMA (2023) là mô hình video-LLM nguồn mở đầu tiên. Kiến trúc gồm:

- Các đoạn clip 16 khung hình ở tốc độ 2 FPS (tương đương 8 giây).
- Đặc trưng ViT trên từng khung hình -> Video Q-former thực hiện cross-attention trên toàn bộ 16 khung hình -> 32 truy vấn học được (learned queries) -> LLM.
- Nhánh âm thanh song song: dạng sóng âm thanh -> bộ mã hóa âm thanh ImageBind -> Audio Q-former -> 32 truy vấn -> LLM.

Điểm mạnh: suy luận kết hợp thính giác - thị giác. Điểm yếu: độ dài clip cố định, không thể định vị mốc thời gian tùy ý.

### VideoChat và Video-LLaVA

VideoChat giữ lại ý tưởng của Video-LLaMA nhưng loại bỏ nhánh âm thanh để đơn giản hóa kiến trúc. Video-LLaVA (Lin et al., 2023) huấn luyện một bộ mã hóa thị giác duy nhất trên cả hình ảnh tĩnh lẫn các khung hình video ("căn chỉnh trước khi chiếu" - alignment before projection), tạo ra một biểu diễn thống nhất. Cả hai đều đi theo mô hình: bộ mã hóa CLIP đóng băng + MLP + LLM.

Cả hai đều chưa xử lý được video dài mà chỉ giới hạn trong các hệ thống 8-16 khung hình.

### Qwen2.5-VL và M-RoPE căn chỉnh theo thời gian (Qwen2.5-VL and time-aligned M-RoPE)

Qwen2.5-VL kế thừa cơ chế M-RoPE (Multimodal Rotary Position Embedding) của Qwen2-VL và căn chỉnh trục thời gian của nó theo thời gian tuyệt đối. Mỗi token mảng ảnh mang một tọa độ vị trí (t, h, w), trong đó t bám sát mốc thời gian thực tế (timestamp), không phải chỉ số khung hình (frame index). Qwen2.5-Omni (Bài học 12.20) sau đó đã gọi phiên bản âm thanh và video của ý tưởng này là TMRoPE (Time-aligned Multimodal RoPE).

Những điểm khác biệt cốt lõi so với nhúng thời gian đơn giản:

- Thời gian tuyệt đối, không phải chỉ số khung hình: Mô hình hiểu "tại giây thứ 4.2" thay vì "ở khung hình thứ 15".
- Phép quay trên từng token, không phải trên từng đoạn clip: Mỗi token thị giác được xoay độc lập theo đúng mốc thời gian của nó.
- Tương thích tự nhiên với FPS động: Nếu bạn lấy mẫu 2 FPS ở đoạn này và 4 FPS ở đoạn khác, M-RoPE căn chỉnh thời gian vẫn xử lý khoảng cách không đều một cách chính xác.

M-RoPE căn chỉnh theo thời gian cho phép giải quyết các truy vấn như "con mèo nhảy lên ở giây thứ mấy?". Mô hình có thể xuất ra trực tiếp "ở giây thứ 4.2", trong khi Video-LLaMA trước đây chỉ có thể trả lời mơ hồ "vào đầu đoạn clip".

### Các chiến lược lấy mẫu khung hình (Frame sampling strategies)

Đồng đều (Uniform): lấy mẫu N khung hình cách đều nhau trên toàn bộ thời lượng. Đơn giản, nhưng dễ bỏ lỡ các đỉnh điểm hành động chuyển động nhanh.

FPS động (Dynamic FPS): lấy mẫu thích ứng dựa trên cường độ chuyển động. Sử dụng dòng quang học (optical flow) hoặc sai phân khung hình (frame differencing) để chọn các phân đoạn có chuyển động mạnh nhằm lấy mẫu dày đặc hơn. Qwen2.5-VL được huấn luyện theo phương pháp này.

Theo sự kiện (Event-driven): chạy một bộ phát hiện gọn nhẹ, lấy mẫu nhiều hơn ở nơi có hành động xảy ra. Được sử dụng bởi VideoAgent.

Khung hình chính + ngữ cảnh (Keyframe + context): lấy mẫu tại các ranh giới chuyển cảnh (shot boundaries) kết hợp cùng vài khung hình liền kề. Phù hợp cho nội dung phim ảnh, điện ảnh.

### Gộp biểu diễn trên từng khung hình (Pooling per frame)

Ở tốc độ 1 FPS với 576 token mỗi khung hình, một đoạn clip 5 phút sẽ tạo ra 172.800 token. Con số này làm tràn ngữ cảnh 128k của Qwen2.5-VL-72B, do đó bạn bắt buộc phải giảm tốc độ khung hình hoặc gộp các token trên từng khung hình trước.

Gộp song tuyến tính 3x3 giúp giảm xuống còn 64 token mỗi khung hình -> 19.200 token cho video 5 phút. Đây là điểm cân bằng lý tưởng (sweet spot) cho hầu hết các tác vụ.

Có thể gộp mạnh tay hơn (6x6 -> 16 token mỗi khung hình) cho các luồng công việc tác tử (agent workflows), nơi chi tiết không gian ít quan trọng hơn so với tiến trình hành động.

### Bốn bộ chuẩn đánh giá video (The four video benchmarks)

- VideoMME: đánh giá toàn diện năng lực hiểu video, từ ngắn, trung bình đến dài.
- TempCompass: kiểm tra suy luận thời gian chi tiết, các câu hỏi về quan hệ "trước" / "sau".
- EgoSchema: suy luận trên video góc nhìn thứ nhất (first-person) thời lượng dài.
- Video-MMMU: các câu hỏi video đa phương thức, đa lĩnh vực chuyên môn.

Một bài đánh giá VLM video toàn diện cần kiểm thử trên cả bốn bộ chuẩn này. Chúng thử thách các khía cạnh khác nhau — TempCompass tập trung vào thứ tự thời gian, EgoSchema thử thách khả năng suy luận trên 3 phút trở lên, còn VideoMME bao quát nhiều độ dài video khác nhau.

### Định dạng đầu ra cho tác vụ định vị (Grounding output formats)

Các định dạng đầu ra cho bài toán định vị thời gian:

- Văn bản tự do (Free text): "Con mèo nhảy lên vào khoảng mốc 4 giây." Dễ đọc nhưng khó bóc tách chính xác bằng mã.
- JSON có cấu trúc (Structured JSON): `{"event": "jump", "start": 4.1, "end": 4.3}`. Qwen2.5-VL được huấn luyện để xuất ra định dạng này.
- Dựa trên token đặc biệt (Token-based): sử dụng các token đặc biệt xen kẽ trong câu trả lời như `<time>4.1</time>`. Một số mô hình video sử dụng cách này; Qwen2.5-VL thay vào đó mã hóa thời gian trong ID vị trí M-RoPE và trả lời bằng văn bản hoặc JSON.

Định dạng token đặc biệt cho độ chính xác cao nhất cho hệ thống hạ nguồn. Đầu ra dạng JSON của Qwen2.5-VL có thể phân tích cú pháp trực tiếp bằng mã.

### Thực hành tốt nhất năm 2026 (2026 best practice)

Đối với các mô hình video VLM vào năm 2026:

- Bộ mã hóa: SigLIP 2 hoặc một ViT hỗ trợ độ phân giải gốc, kết hợp với M-RoPE căn chỉnh theo thời gian bên trong LLM (như Qwen2.5-VL).
- Lấy mẫu khung hình: FPS động (1-4 FPS tùy thuộc vào chuyển động) kèm giới hạn trần số lượng khung hình tối đa.
- Gộp khung hình: song tuyến tính 3x3.
- Đầu ra: JSON có cấu trúc với các trường thời gian và sự kiện.
- Bộ chuẩn đánh giá: VideoMME + TempCompass cho bài toán tổng quát; EgoSchema cho ngữ cảnh dài.

````figure
video-temporal-patches
````

## Thực hành (Use It)

`code/main.py` bao gồm:

- Bộ lấy mẫu khung hình đồng đều và lấy mẫu theo FPS động.
- Bộ đánh giá định vị thời gian mô phỏng: cho trước một sự kiện "chân thực" (ground truth) tại thời điểm T và kết quả dự đoán của mô hình, tính toán điểm số chính xác với sai số cho phép (tolerance).
- So sánh hiệu năng giữa Video-LLaMA (16 khung hình, Q-former), Video-LLaVA (8 khung hình, MLP), và Qwen2.5-VL (FPS động + M-RoPE căn chỉnh theo thời gian).

## Triển khai (Ship It)

Bài học này tạo ra `outputs/skill-video-vlm-frame-planner.md`. Với một nhiệm vụ xử lý video cụ thể (giám sát, nhận dạng hành động, định vị thời gian, tóm tắt video), kỹ năng này sẽ lựa chọn chiến lược lấy mẫu khung hình, hệ số gộp, định dạng đầu ra và phân tầng độ chính xác kỳ vọng.

## Bài tập (Exercises)

1. Đối với một video trình diễn nấu ăn dài 3 phút, hãy lựa chọn giữa lấy mẫu đồng đều và FPS động. Giải thích và chứng minh bằng số lượng token ước tính.

2. M-RoPE căn chỉnh theo thời gian bổ sung năng lực cụ thể nào mà một bảng nhúng thời gian (temporal embedding table) đơn giản không thể thực hiện được?

3. Viết một lược đồ JSON (JSON schema) cho bài toán định vị thời gian mà một VLM có thể học để xuất ra. Hãy tính đến cả các trường hợp lỗi (error cases).

4. Đọc Mục 3 của bài báo Video-LLaVA về "Căn chỉnh trước khi chiếu" (Alignment Before Projection). Vì sao phương pháp này lại vượt trội hơn việc huấn luyện riêng rẽ các bộ mã hóa hình ảnh và video?

5. Dựa trên bảng xếp hạng VideoMME, khoảng cách giữa mô hình nguồn mở dẫn đầu và mô hình đóng thương mại dẫn đầu tính đến năm 2026 là bao nhiêu? Bao nhiêu phần trăm của khoảng cách đó bắt nguồn từ cơ chế mã hóa thời gian so với quy mô của LLM nền tảng?

## Thuật ngữ chính (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực sự |
|---|---|---|
| Temporal grounding (Định vị thời gian) | "Câu trả lời có mốc thời gian" | VLM xuất ra một khoảng thời gian cụ thể (timestamp range) khi một sự kiện diễn ra |
| Time-aligned M-RoPE | "TMRoPE" | Nhúng vị trí quay 3D có trục thời gian bám theo mốc thời gian tuyệt đối; Qwen2.5-VL dùng cho video, và Qwen2.5-Omni định danh phiên bản âm thanh/video là TMRoPE |
| Dynamic FPS (FPS động) | "Lấy mẫu nhận biết chuyển động" | Lấy mẫu nhiều khung hình hơn ở các đoạn có chuyển động nhanh, ít hơn ở các đoạn tĩnh |
| Frame pooling (Gộp khung hình) | "Nén không gian mỗi khung hình" | Giảm số lượng mảng ảnh trên mỗi khung hình bằng phép nội suy song tuyến tính trước khi đưa vào LLM |
| Video Q-former | "Bộ nén đoạn clip" | Nút thắt cross-attention ánh xạ N khung hình thành K truy vấn học được |
| VideoMME | "Bộ chuẩn video" | Bộ chuẩn đánh giá video toàn diện gồm video ngắn, trung bình, dài với hơn 2500 mẫu kiểm thử |

## Tài liệu tham khảo (Further Reading)

- [Zhang et al. — Video-LLaMA (arXiv:2306.02858)](https://arxiv.org/abs/2306.02858)
- [Li et al. — VideoChat (arXiv:2305.06355)](https://arxiv.org/abs/2305.06355)
- [Lin et al. — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Qwen Team — Qwen2.5-VL (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Lin et al. — VILA-1.5 (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
