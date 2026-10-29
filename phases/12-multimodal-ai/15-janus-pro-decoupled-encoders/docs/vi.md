# Janus-Pro: Các Bộ mã hóa Tách rời cho Mô hình Đa phương thức Hợp nhất

> Các mô hình đa phương thức hợp nhất (unified multimodal models) đối mặt với một sự mâu thuẫn tất yếu. Tác vụ hiểu đòi hỏi các đặc trưng ngữ nghĩa (semantic features) — các vector đầu ra từ SigLIP hoặc DINOv2 chứa lượng thông tin phong phú ở cấp độ khái niệm. Tác vụ sinh lại đòi hỏi các mã phù hợp cho việc tái tạo (reconstruction-friendly codes) — các token VQ có khả năng ghép nối lại thành các điểm ảnh sắc nét. Hai mục tiêu này khó có thể dung hòa trong một bộ mã hóa duy nhất. Janus (DeepSeek, tháng 10 năm 2024) và Janus-Pro (DeepSeek, tháng 1 năm 2025) cho rằng giải pháp là hãy ngừng cố gắng gộp chung: hãy tách rời hai bộ mã hóa (decouple the two encoders). Chia sẻ phần thân transformer (shared transformer body) giữa các tác vụ, nhưng định tuyến tác vụ hiểu qua SigLIP và tác vụ sinh qua một bộ token hóa VQ. Ở quy mô 7B tham số, Janus-Pro đánh bại DALL-E 3 trên thang đo GenEval trong khi vẫn sánh ngang LLaVA trên MMMU. Bài học này sẽ phân tích lý do vì sao việc dùng hai bộ mã hóa lại thành công ở nơi mà giải pháp dùng một bộ mã hóa thất bại.

**Loại:** Xây dựng (Build)  
**Ngôn ngữ:** Python (thư viện chuẩn stdlib, định tuyến bộ mã hóa kép + tín hiệu thân dùng chung)  
**Điều kiện tiên quyết:** Giai đoạn 12 · 13 (Transfusion), Giai đoạn 12 · 14 (Show-o)  
**Thời lượng:** ~120 phút

## Mục tiêu học tập

- Giải thích vì sao một bộ mã hóa dùng chung duy nhất buộc phải đánh đổi chất lượng của tác vụ hiểu hoặc tác vụ sinh.
- Mô tả cơ chế định tuyến của Janus-Pro: đặc trưng SigLIP ở phía đầu vào cho tác vụ hiểu, token VQ ở cả đầu vào lẫn đầu ra cho tác vụ sinh.
- Truy vết việc mở rộng tỷ lệ pha trộn dữ liệu (data-mix scaling) giúp Janus-Pro thành công vượt bậc so với phiên bản Janus ban đầu.
- So sánh các kiến trúc tách rời (Janus-Pro), ghép nối liên tục (Transfusion) và ghép nối rời rạc (Show-o).

## Vấn đề

Các mô hình hợp nhất chia sẻ một phần thân transformer cho cả tác vụ hiểu và tác vụ sinh. Những nỗ lực trước đây (Chameleon, Show-o, Transfusion) đều dùng chung một bộ token hóa thị giác duy nhất cho cả hai hướng. Bộ token hóa đó luôn là một sự đánh đổi:

- Tối ưu hóa cho việc tái tạo (tác vụ sinh): VQ-VAE nắm bắt được chi tiết điểm ảnh mịn nhưng sinh ra các token có tính liên kết ngữ nghĩa yếu.
- Tối ưu hóa cho ngữ nghĩa (tác vụ hiểu): Các vector nhúng SigLIP gom nhóm các hình ảnh "con mèo" lại gần các token "con mèo" nhưng không cho phép tái tạo hình ảnh tốt.

Show-o và Transfusion phải trả giá cho điều này bằng sự sụt giảm chất lượng rõ rệt ở một trong hai hướng. Janus-Pro đặt câu hỏi: tại sao lại ép buộc dùng một bộ token hóa duy nhất khi hai bài toán có nhu cầu hoàn toàn khác biệt?

## Khái niệm

### Mã hóa thị giác tách rời (Decoupled visual encoding)

Kiến trúc của Janus-Pro phân tách rạch ròi hai bộ mã hóa:

- Nhánh hiểu (Understanding path): Hình ảnh đầu vào → SigLIP-SO400m → MLP 2 lớp → phần thân transformer.
- Nhánh sinh (Generation path): Hình ảnh đầu vào (nếu điều kiện hóa trên ảnh có sẵn) → bộ token hóa VQ → các ID token → phần thân transformer.
- Nhánh sinh đầu ra (Output generation): Các token hình ảnh do transformer dự đoán → bộ giải mã VQ (VQ decoder) → các điểm ảnh (pixels).

Phần thân transformer được chia sẻ hoàn toàn. Mọi thành phần ở thượng nguồn (upstream) và hạ nguồn (downstream) của phần thân đều được thiết kế chuyên biệt cho từng tác vụ.

Các đầu vào được phân định rõ ràng thông qua định dạng prompt: thẻ `<understand>` sẽ định tuyến qua SigLIP; thẻ `<generate>` sẽ định tuyến qua VQ. Hoặc cơ chế định tuyến được suy ra ngầm định từ bản chất nhiệm vụ.

### Vì sao cơ chế này hoạt động hiệu quả (Why this works)

Hàm mất mát của tác vụ hiểu tiếp nhận các đặc trưng từ SigLIP, vốn đã được tiền huấn luyện theo kiểu CLIP để tối ưu hóa sự tương đồng ngữ nghĩa. Nhờ đó, điểm số đánh giá năng lực thị giác (perception benchmarks) của mô hình vượt qua Show-o / Transfusion vì đặc trưng đầu vào phù hợp hơn nhiều cho bài toán.

Hàm mất mát của tác vụ sinh tiếp nhận các token VQ, vốn được bộ token hóa tối ưu hóa cho bài toán tái tạo ảnh. Chất lượng hình ảnh cải thiện vượt trội so với Show-o vì các mã VQ có thể giải mã ngược lại thành điểm ảnh một cách sắc nét.

Phần thân transformer dùng chung tiếp nhận hai phân phối đầu vào (SigLIP và VQ) và học cách xử lý linh hoạt cả hai. Luận điểm cốt lõi: khi có đủ dữ liệu và đủ tham số, phần thân transformer hoàn toàn có thể hấp thu tốt sự chuyển đổi này.

### Mở rộng quy mô dữ liệu — Janus so với Janus-Pro (Data scaling — Janus vs Janus-Pro)

Janus nguyên bản (arXiv 2410.13848) giới thiệu cơ chế tách rời nhưng ở quy mô khiêm tốn (1.3B tham số, dữ liệu hạn chế). Janus-Pro (arXiv 2501.17811) đã mở rộng quy mô mạnh mẽ:

- 7B tham số (so với 1.3B).
- 90 triệu cặp ảnh-văn bản cho Giai đoạn 1 (căn chỉnh - alignment), tăng từ 72 triệu.
- 72 triệu mẫu cho Giai đoạn 2 (hợp nhất - unified), tăng từ 26 triệu.
- Bổ sung 200 nghìn mẫu chỉ dẫn sinh ảnh cho Giai đoạn 3 (instruction tuning).

Kết quả: Janus-Pro-7B sánh ngang LLaVA trên MMMU (60.3 so với ~58) và đánh bại DALL-E 3 trên GenEval (0.80 so với 0.67). Một mô hình mở duy nhất có năng lực cạnh tranh sòng phẳng ở cả hai phía của quang phổ đa phương thức hợp nhất.

### JanusFlow — Biến thể dòng chảy chỉnh lưu (JanusFlow — the rectified flow variant)

JanusFlow (arXiv 2411.07975) thay thế nhánh sinh dùng VQ bằng nhánh sinh theo dòng chảy chỉnh lưu (rectified flow, trong không gian liên tục). Phân nhánh trở thành SigLIP cho tác vụ hiểu + rectified flow cho tác vụ sinh. Ngưỡng chặn trần về chất lượng được nới rộng hơn nữa. Kiến trúc vẫn trung thành với nguyên lý bộ mã hóa tách rời - thân transformer dùng chung.

### Nhiệm vụ của phần thân dùng chung (The shared body's job)

Phần thân transformer xử lý một chuỗi thống nhất nhưng tiếp nhận hai phân phối đầu vào khác nhau. Nhiệm vụ của nó là:

- Cho tác vụ hiểu: tiếp nhận đặc trưng SigLIP + token văn bản → xuất ra văn bản theo cách tự hồi quy.
- Cho tác vụ sinh: tiếp nhận token văn bản + (các token VQ ảnh tùy chọn) → xuất ra các token VQ hình ảnh theo cách tự hồi quy.

Phần thân không có trọng số chuyên biệt theo phương thức cho từng khối. Đó là một transformer dạng văn bản tiêu chuẩn tương tự như kiến trúc bên trong Qwen hoặc Llama, kết hợp cùng hai bộ điều hợp (adapter) đầu vào.

Đáng chú ý, điều này đồng nghĩa phần thân của Janus-Pro có thể được khởi tạo từ một LLM tiền huấn luyện. Janus-Pro thực tế được khởi tạo từ DeepSeek-MoE-7B. Lựa chọn này tạo ra sự khác biệt lớn: LLM đóng góp năng lực suy luận mà các mô hình hợp nhất huấn luyện hoàn toàn từ đầu (scratch) rất khó đạt tới.

### So sánh với InternVL-U (Compared to InternVL-U)

InternVL-U (Bài học 12.10) là bước tiếp nối của năm 2026. Mô hình này kết hợp:

- Tiền huấn luyện đa phương thức nguyên bản (mạng xương sống InternVL3).
- Định tuyến bộ mã hóa tách rời (SigLIP ở đầu vào, các đầu ra VQ + khuếch tán ở đầu ra).
- Hợp nhất cả hiểu + sinh + chỉnh sửa ảnh.

InternVL-U đã tích hợp lựa chọn kiến trúc của Janus-Pro vào một khuôn khổ rộng lớn hơn. Ý tưởng bộ mã hóa tách rời hiện đã trở thành chuẩn mực mặc định cho các mô hình hợp nhất ở quy mô lớn.

### Những hạn chế (Limitations)

Các bộ mã hóa tách rời làm tăng độ phức tạp của kiến trúc: hai bộ token hóa cần huấn luyện, hai đường dẫn đầu vào cần duy trì, và hai tập hợp các kiểu lỗi tiềm ẩn. Đối với những sản phẩm không có nhu cầu sinh hình ảnh, Janus-Pro là một thiết kế dư thừa — hãy chọn một mô hình thuần hiểu thuộc họ LLaVA.

Đối với những sản phẩm không cần hiểu hình ảnh, Janus-Pro lại có phần quá mức đáp ứng — hãy chọn một mô hình chuyên biệt như Stable Diffusion 3 hoặc Flux.

Với những hệ thống đòi hỏi đồng thời cả hai năng lực, Janus-Pro hiện là kiến trúc nguồn mở tham chiếu tiêu chuẩn.

````figure
l5-janus-decouple
````

## Thực hành (Use It)

`code/main.py` mô phỏng cơ chế định tuyến của Janus-Pro:

- Hai bộ mã hóa giả lập: bộ mã hóa kiểu SigLIP (tạo ra các vector ngữ nghĩa 256 chiều) và bộ mã hóa kiểu VQ (tạo ra các mã số nguyên).
- Một bộ định tuyến câu lệnh (prompt router) lựa chọn bộ mã hóa dựa trên thẻ nhiệm vụ.
- Một phần thân dùng chung (mô hình thay thế) xử lý chuỗi token bất kể chúng được sinh ra từ bộ mã hóa nào.
- Sự chuyển đổi từ lịch trình mẫu có trọng số của Giai đoạn 1 (căn chỉnh) sang Giai đoạn 3 (tinh chỉnh theo chỉ dẫn).

In ra các lộ trình định tuyến cho 3 ví dụ: hỏi đáp hình ảnh (VQA), chuyển văn bản thành ảnh (T2I), và chỉnh sửa hình ảnh.

## Triển khai (Ship It)

Bài học này tạo ra `outputs/skill-decoupled-encoder-picker.md`. Đối với một bài toán sản phẩm yêu cầu hợp nhất tác vụ sinh + hiểu ở chất lượng tiệm cận frontier, kỹ năng này sẽ lựa chọn giữa Janus-Pro, JanusFlow hoặc InternVL-U kèm theo các khuyến nghị cụ thể về quy mô dữ liệu.

## Bài tập (Exercises)

1. Janus-Pro-7B đánh bại DALL-E 3 trên GenEval. Hãy giải thích vì sao một mô hình mở 7B có thể bắt kịp mô hình đóng thương mại hàng đầu về khả năng sinh nhưng lại chưa thể bắt kịp ở năng lực hiểu.

2. Triển khai một hàm định tuyến: cho trước văn bản prompt, phân loại thành `understand` hoặc `generate`. Bạn sẽ xử lý các câu lệnh nhập nhằng như "miêu tả rồi phác thảo bức tranh này" như thế nào?

3. JanusFlow thay thế nhánh VQ bằng dòng chảy chỉnh lưu (rectified flow). Khi đó phần thân transformer sẽ xuất ra giá trị gì, và hàm mất mát thay đổi ra sao?

4. Đề xuất một tác vụ thứ tư mà kiến trúc Janus-Pro có thể xử lý khi bổ sung thêm một bộ mã hóa tách rời nữa. Ví dụ: phân đoạn hình ảnh (kiểu DINO), ước lượng độ sâu (kiểu MiDaS).

5. Đọc Mục 4.2 của bài báo Janus-Pro về mở rộng quy mô dữ liệu. Giai đoạn dữ liệu nào đóng góp nhiều nhất vào mức tăng chất lượng T2I so với Janus?

## Thuật ngữ chính (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực sự |
|---|---|---|
| Decoupled encoding (Mã hóa tách rời) | "Hai bộ mã hóa thị giác" | Bộ token hóa hoặc bộ mã hóa riêng biệt cho từng hướng: ngữ nghĩa cho hiểu, tái tạo cho sinh |
| Shared body (Phần thân dùng chung) | "Một transformer" | Transformer duy nhất xử lý đầu ra của một trong hai bộ mã hóa; không có trọng số chuyên biệt theo phương thức |
| SigLIP for understanding (SigLIP cho tác vụ hiểu) | "Đặc trưng ngữ nghĩa" | Tháp thị giác họ CLIP cung cấp các đặc trưng khái niệm phong phú nhưng khả năng tái tạo điểm ảnh kém |
| VQ for generation (VQ cho tác vụ sinh) | "Mã tái tạo" | Các token lượng tử hóa vector có thể giải mã ngược lại một cách sắc nét thành các điểm ảnh |
| JanusFlow | "Biến thể rectified-flow" | Janus-Pro với đầu sinh flow-matching liên tục thay vì VQ |
| Routing tag (Thẻ định tuyến) | "Thẻ nhiệm vụ" | Dấu hiệu nhận biết trong prompt (`<understand>` / `<generate>`) để lựa chọn bộ mã hóa đầu vào |

## Tài liệu tham khảo (Further Reading)

- [Wu et al. — Janus (arXiv:2410.13848)](https://arxiv.org/abs/2410.13848)
- [Chen et al. — Janus-Pro (arXiv:2501.17811)](https://arxiv.org/abs/2501.17811)
- [Ma et al. — JanusFlow (arXiv:2411.07975)](https://arxiv.org/abs/2411.07975)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Dong et al. — DreamLLM (arXiv:2309.11499)](https://arxiv.org/abs/2309.11499)
