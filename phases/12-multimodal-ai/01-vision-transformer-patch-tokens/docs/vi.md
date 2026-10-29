# Vision Transformer và Cấu trúc Cơ sở Patch-Token

> Trước bất kỳ tác vụ đa phương thức (multimodal) nào, một hình ảnh phải được chuyển đổi thành chuỗi các token mà transformer có thể tiếp nhận. Bài báo ViT năm 2020 đã giải quyết bài toán này bằng các mảng ảnh (patch) 16x16 pixel, một phép chiếu tuyến tính (linear projection) và một nhúng vị trí (position embedding). Năm năm sau, mọi mô hình tiên phong năm 2026 (Claude Opus 4.7 ở độ phân giải gốc 2576px, Gemini 3.1 Pro, Qwen3.5-Omni) vẫn bắt đầu theo cách này — bộ mã hóa (encoder) đã chuyển từ ViT sang DINOv2 rồi SigLIP 2, các token thanh ghi (register tokens) được bổ sung, cơ chế mã hóa vị trí chuyển thành 2D-RoPE, nhưng cấu trúc nguyên thủy này vẫn đứng vững. Bài học này sẽ phân tích toàn bộ quy trình patch-token từ đầu đến cuối và xây dựng nó bằng Python thư viện chuẩn (stdlib) để phần còn lại của Giai đoạn 12 có một mô hình tư duy cụ thể cho "token thị giác" (visual tokens).

**Loại:** Tìm hiểu (Learn)  
**Ngôn ngữ:** Python (stdlib, bộ token hóa mảng ảnh + bộ tính toán hình học)  
**Điều kiện tiên quyết:** Giai đoạn 7 (Transformers), Giai đoạn 4 (Thị giác máy tính - Computer Vision)  
**Thời lượng:** ~120 phút

## Mục tiêu học tập

- Chuyển đổi một hình ảnh HxWx3 thành một chuỗi các token mảng ảnh (patch tokens) với mã hóa vị trí (positional encoding) chính xác.
- Tính toán độ dài chuỗi (sequence length), số lượng tham số (parameter count) và số phép tính dấu phẩy động (FLOPs) cho một ViT với các thông số cho trước (kích thước mảng ảnh, độ phân giải, số chiều ẩn, độ sâu).
- Nêu tên ba nâng cấp đã đưa ViT từ nghiên cứu năm 2020 thành giải pháp sản xuất năm 2026: tiền huấn luyện tự giám sát (self-supervised pretraining - DINO / MAE), token thanh ghi (register tokens) và đóng gói độ phân giải gốc (native-resolution packing).
- Lựa chọn giữa gộp CLS (CLS pooling), gộp trung bình (mean pooling) và token thanh ghi cho một tác vụ hạ nguồn (downstream task).

## Vấn đề

Mô hình Transformer hoạt động trên các chuỗi vector. Văn bản vốn dĩ đã là một chuỗi (byte hoặc token). Một hình ảnh là một lưới 2 chiều của các pixel với ba kênh màu — không phải là một chuỗi. Nếu bạn làm phẳng (flatten) từng pixel, một hình ảnh RGB 224x224 sẽ trở thành 150.528 token, và cơ chế tự chú ý (self-attention) ở độ dài chuỗi đó là hoàn toàn bất khả thi (chi phí bậc hai theo độ dài chuỗi - quadratic in sequence length).

Các phương pháp trước năm 2020 đã ghép thêm một bộ trích xuất đặc trưng CNN (mạng nơ-ron tích chập - convolutional neural network) ở phía trước: ResNet tạo ra một bản đồ đặc trưng (feature map) 7x7 gồm các vector 2048 chiều, rồi đưa 49 token đó vào transformer. Cách tiếp cận này hoạt động nhưng thừa hưởng những thiên kiến quy nạp (inductive biases) của CNN (tính bất biến tịnh tiến - translation equivariance, vùng tiếp nhận cục bộ - local receptive fields) và làm mất đi khả năng mở rộng quy mô (scaling) của transformer.

Dosovitskiy cùng các cộng sự (2020) đã đặt ra câu hỏi thẳng thắn: điều gì sẽ xảy ra nếu chúng ta bỏ qua hoàn toàn CNN? Chia hình ảnh thành các mảng ảnh kích thước cố định (ví dụ: 16x16 pixel), chiếu tuyến tính (linearly project) từng mảng ảnh thành một vector, thêm nhúng vị trí (positional embedding), rồi nạp chuỗi này vào một transformer tiêu chuẩn (vanilla transformer). Vào thời điểm đó, đây được xem là điều kỳ dị — xử lý thị giác mà không dùng phép tích chập (convolution). Với đủ dữ liệu (JFT-300M, sau đó là LAION), mô hình đã đánh bại ResNet trên ImageNet và liên tục cải thiện.

Đến năm 2026, cấu trúc cơ sở ViT (ViT primitive) đã trở thành nền tảng không thể bàn cãi. Tháp thị giác (vision tower) của mọi mô hình thị giác - ngôn ngữ mở (open-weights VLM - Vision-Language Model) đều là hậu duệ của nó (DINOv2, SigLIP 2, CLIP, EVA, InternViT). Câu hỏi không còn là "chúng ta có nên dùng các mảng ảnh không?" mà là "kích thước mảng ảnh nào, lịch trình phân giải (resolution schedule) nào, mục tiêu tiền huấn luyện nào và cơ chế mã hóa vị trí nào".

## Khái niệm

### Mảng ảnh dưới dạng token (Patches as tokens)

Cho một hình ảnh `x` có kích thước `(H, W, 3)` và kích thước mảng ảnh `P`, bạn chia hình ảnh thành một lưới gồm `(H/P) x (W/P)` mảng ảnh không chồng lấn. Mỗi mảng ảnh là một khối pixel `P x P x 3`. Làm phẳng từng khối thành một vector kích thước `3 P^2`. Áp dụng một phép chiếu tuyến tính dùng chung `W_E` có kích thước `(3 P^2, D)` để ánh xạ từng mảng ảnh vào chiều ẩn (hidden dimension) `D` của mô hình.

Đối với cấu hình tiêu chuẩn ViT-B/16:
- Độ phân giải 224, kích thước mảng ảnh 16 → lưới 14x14 → 196 token mảng ảnh.
- Mỗi mảng ảnh là `16 x 16 x 3 = 768` giá trị pixel, được chiếu thành `D = 768`.
- Thêm một token `[CLS]` có thể học được (learnable) → độ dài chuỗi là 197.

Về mặt toán học, phép chiếu mảng ảnh giống hệt một phép tích chập 2D với kích thước nhân (kernel size) `P`, bước nhảy (stride) `P`, và `D` kênh đầu ra. Đó cũng chính là cách mã nguồn sản xuất triển khai trong thực tế — `nn.Conv2d(3, D, kernel_size=P, stride=P)`. Lối diễn giải "chiếu tuyến tính" mang tính khái niệm; còn triển khai bằng nhân tích chập mang lại hiệu năng cao.

### Nhúng vị trí (Positional embeddings)

Các mảng ảnh vốn không có thứ tự nội tại — transformer nhìn nhận chúng như một tập hợp không thứ tự (bag of tokens). Các ViT đời đầu bổ sung một nhúng vị trí 1D có thể học được (mỗi vị trí là một vector 768 chiều, gồm 197 vector). Cách này hiệu quả, nhưng ràng buộc mô hình vào độ phân giải huấn luyện: khi suy luận, bạn phải nội suy (interpolate) bảng vị trí nếu thay đổi kích thước lưới.

Các mạng xương sống thị giác (vision backbones) hiện đại sử dụng 2D-RoPE (nhúng vị trí quay 2 chiều - 2D Rotary Position Embedding, như M-RoPE của Qwen2-VL, mặc định của SigLIP 2) hoặc các vị trí 2D được nhân tử hóa (factorized 2D positions). 2D-RoPE xoay các vector truy vấn (query) và khóa (key) dựa trên chỉ số (hàng, cột) của mảng ảnh, nhờ đó mô hình suy ra vị trí 2D tương đối từ góc quay. Không cần bảng vị trí cố định. Mô hình xử lý được kích thước lưới tùy ý tại thời điểm suy luận.

### Token CLS, đầu ra gộp và token thanh ghi (CLS token, pooled output, and register tokens)

Biểu diễn cấp độ toàn ảnh (image-level representation) là gì? Ba lựa chọn cùng song hành tồn tại:

1. Token `[CLS]`. Đặt thêm một vector có thể học được vào đầu chuỗi mảng ảnh. Sau khi đi qua tất cả các khối transformer, trạng thái ẩn của token CLS đóng vai trò là biểu diễn của toàn bộ hình ảnh. Thừa hưởng từ BERT. Được dùng bởi ViT nguyên bản, CLIP.
2. Gộp trung bình (Mean pool). Lấy trung bình trạng thái ẩn đầu ra của các token mảng ảnh. Được sử dụng bởi SigLIP, DINOv2 và hầu hết các VLM hiện đại.
3. Token thanh ghi (Register tokens). Darcet cùng các cộng sự (2023) nhận thấy rằng các ViT được huấn luyện mà không có một token "hấp thụ" (sink token) rõ ràng sẽ hình thành các mảng ảnh "nhiễu giả" (artifact) có chuẩn (norm) rất cao, chiếm đoạt cơ chế tự chú ý. Việc bổ sung 4–16 token thanh ghi có thể học được sẽ hấp thụ tải này và cải thiện chất lượng dự đoán dày đặc (dense prediction) như phân đoạn (segmentation) hay ước lượng độ sâu (depth). DINOv2 và SigLIP 2 đều được tích hợp sẵn các thanh ghi.

Lựa chọn này đóng vai trò quan trọng cho các tác vụ hạ nguồn. CLS rất phù hợp cho bài toán phân loại. Đối với các VLM đưa trực tiếp các token mảng ảnh vào một LLM (mô hình ngôn ngữ lớn - Large Language Model), bạn bỏ qua hoàn toàn bước gộp — mỗi mảng ảnh trở thành một token đầu vào của LLM. Các token thanh ghi sẽ bị loại bỏ trước khi bàn giao (chúng chỉ đóng vai trò giàn giáo hỗ trợ huấn luyện, không phải nội dung).

### Tiền huấn luyện: có giám sát, tương phản, che mờ, tự chưng cất (Pretraining: supervised, contrastive, masked, self-distilled)

ViT năm 2020 được tiền huấn luyện bằng phương pháp phân loại có giám sát (supervised classification) trên tập dữ liệu JFT-300M. Nhanh chóng sau đó bị thay thế bởi:

- CLIP (2021): học tương phản giữa hình ảnh và văn bản trên 400 triệu cặp (image-text pairs). Bài học 12.02.
- MAE (2021, He et al.): che (mask) 75% các mảng ảnh, tái tạo lại pixel. Tự giám sát (self-supervised), hoạt động thuần túy trên hình ảnh.
- DINO (2021) / DINOv2 (2023): tự chưng cất (self-distillation) với cơ chế học sinh - giáo viên (student-teacher), không cần nhãn, không cần chú thích văn bản (captions). DINOv2 ViT-g/14 năm 2023 là mạng xương sống thuần thị giác mạnh mẽ nhất và là lựa chọn mặc định cho các trường hợp sử dụng "đặc trưng dày đặc" (dense features).
- SigLIP / SigLIP 2 (2023, 2025): CLIP với hàm mất mát sigmoid và NaFlex cho tỷ lệ khung hình gốc. Là tháp thị giác áp đảo trong các VLM nguồn mở năm 2026 (Qwen, Idefics2, LLaVA-OneVision).

Lựa chọn tiền huấn luyện quyết định thế mạnh của mạng xương sống: CLIP/SigLIP phù hợp cho việc so khớp ngữ nghĩa với văn bản, DINOv2 dành cho đặc trưng thị giác dày đặc, MAE là điểm khởi đầu cho tinh chỉnh hạ nguồn.

### Quy luật mở rộng (Scaling laws)

Nghiên cứu mở rộng ViT (Zhai et al. 2022) đã chứng minh rằng chất lượng của ViT tuân theo các quy luật dự đoán được theo kích thước mô hình, dung lượng dữ liệu và lượng tính toán (compute). Ở mức tính toán cố định:
- Mô hình lớn hơn + nhiều dữ liệu hơn → chất lượng tốt hơn.
- Kích thước mảng ảnh là đòn bẩy giữa độ dài chuỗi và độ trung thực (fidelity). Mảng 14 (điển hình cho DINOv2/SigLIP SO400m) tạo ra nhiều token trên mỗi ảnh hơn mảng 16; tốt hơn cho nhận dạng ký tự quang học (OCR) và các tác vụ dày đặc, nhưng xử lý chậm hơn.
- Độ phân giải là đòn bẩy lớn còn lại. Việc tăng từ 224 lên 384 rồi 512 hầu như luôn mang lại cải thiện, với chi phí bậc hai theo FLOPs.

ViT-g/14 (1 tỷ tham số, mảng 14, độ phân giải 224 → 256 token) và SigLIP SO400m/14 (400 triệu tham số, mảng 14) là hai bộ mã hóa chủ lực cho các VLM mở năm 2026.

### Đếm tham số cho một ViT (Parameter count for a ViT)

Toàn bộ phép tính nằm trong `code/main.py`. Với ViT-B/16 ở độ phân giải 224:

```text
patch_embed = 3 * 16 * 16 * 768 + 768  =  591k
cls + pos    = 768 + 197 * 768          =  152k
block        = 4 * 768^2 (QKVO) + 2 * 4 * 768^2 (MLP) + 2 * 2*768 (LN)
             = 12 * 768^2 + 3k          =  7.1M
12 blocks    = 85M
final LN    = 1.5k
total       ≈ 86M
```

Hãy ước tính sơ bộ mọi ViT theo cách này trước khi nạp checkpoint. Kích thước mạng xương sống xác lập ngưỡng VRAM tối thiểu trong bất kỳ VLM hạ nguồn nào.

### Cấu hình sản xuất năm 2026 (2026 production config)

Bộ mã hóa mà hầu hết các VLM mở tích hợp vào năm 2026 là SigLIP 2 SO400m/14 ở độ phân giải gốc (NaFlex). Nó bao gồm:
- 400 triệu tham số (400M parameters).
- Kích thước mảng ảnh 14, độ phân giải mặc định 384 → 729 token mảng ảnh cho mỗi hình ảnh.
- Gộp trung bình cho các tác vụ cấp độ hình ảnh; toàn bộ 729 mảng ảnh được chuyển vào LLM cho tác vụ trả lời câu hỏi bằng hình ảnh (VQA - Visual Question Answering).
- 4 token thanh ghi, được loại bỏ trước khi chuyển sang LLM.
- 2D-RoPE với khả năng co giãn theo cấp độ ảnh cho tỷ lệ khung hình gốc.

Mọi quyết định trong cấu hình đó đều bắt nguồn từ một bài báo nghiên cứu mà bạn có thể đọc.

````figure
image-patch-tokens
````

## Thực hành (Use It)

`code/main.py` là bộ token hóa mảng ảnh và tính toán hình học. Nó nhận đầu vào (chiều cao ảnh H, chiều rộng W, kích thước mảng P, chiều ẩn D, độ sâu L) và báo cáo:

- Hình dạng lưới (grid shape) và độ dài chuỗi sau khi chia mảng.
- Chuỗi token cho một ảnh giả lập đơn giản 8x8 pixel (đi qua quy trình làm phẳng + chiếu tuyến tính).
- Số lượng tham số phân chia theo nhúng mảng ảnh, nhúng vị trí, các khối transformer và đầu ra (head).
- FLOPs cho mỗi lượt truyền xuôi (forward pass) ở độ phân giải mục tiêu.
- Bảng so sánh giữa ViT-B/16 @ 224, ViT-L/14 @ 336, DINOv2 ViT-g/14 @ 224, SigLIP SO400m/14 @ 384.

Hãy chạy nó. Đối chiếu số lượng tham số với các con số đã công bố. Thử nghiệm với kích thước mảng ảnh và độ phân giải để cảm nhận chi phí bùng nổ token.

## Triển khai (Ship It)

Bài học này tạo ra `outputs/skill-patch-geometry-reader.md`. Với cấu hình ViT cho trước (kích thước mảng ảnh, độ phân giải, số chiều ẩn, độ sâu), kỹ năng này sẽ tạo ra ước tính về số lượng token, số lượng tham số và dung lượng VRAM kèm theo giải thích chi tiết. Hãy sử dụng kỹ năng này bất cứ khi nào bạn lựa chọn một mạng xương sống thị giác cho VLM — nó giúp tránh được những sự cố bất ngờ dạng "số lượng token bùng nổ làm tràn ngữ cảnh của LLM".

## Bài tập (Exercises)

1. Tính độ dài chuỗi token mảng ảnh cho Qwen2.5-VL với đầu vào độ phân giải gốc 1280x720 và kích thước mảng ảnh 14. Kết quả này so với biểu diễn chỉ dùng token CLS như thế nào?

2. Một khung hình 1080p (1920x1080) với kích thước mảng 14 tạo ra bao nhiêu token? Ở tốc độ 30 FPS trong một video dài 5 phút, tổng số token thị giác là bao nhiêu? Giải pháp nào giúp tiết kiệm chi phí nhiều nhất: gộp token (pooling), lấy mẫu khung hình (frame sampling), hay hợp nhất token (token merging)?

3. Triển khai phép gộp trung bình (mean pooling) trên các token mảng ảnh bằng Python thuần. Xác minh rằng phép gộp trung bình trên 196 token từ đầu ra của DINOv2 khớp với giá trị mà hàm `forward` của mô hình trả về khi yêu cầu một nhúng đã gộp (pooled embedding).

4. Đọc Mục 3 của bài báo "Vision Transformers Need Registers" (arXiv:2309.16588). Mô tả trong hai câu xem các token thanh ghi hấp thụ loại nhiễu giả nào và tại sao điều đó lại quan trọng đối với tác vụ dự đoán dày đặc ở hạ nguồn.

5. Sửa đổi `code/main.py` để hỗ trợ cơ chế đóng gói mảng ảnh (patch-n'-pack): cho một danh sách các hình ảnh có độ phân giải khác nhau, tạo ra một chuỗi đóng gói duy nhất kèm mặt nạ chú ý dạng khối đường chéo (block-diagonal attention mask). Xác minh lại với Bài học 12.06 khi bạn học đến bài đó.

## Thuật ngữ chính (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực sự |
|---|---|---|
| Patch (Mảng ảnh) | "Ô vuông 16x16 pixel" | Một vùng có kích thước cố định không chồng lấn trên ảnh đầu vào; trở thành một token |
| Patch embedding (Nhúng mảng ảnh) | "Phép chiếu tuyến tính" | Một ma trận trọng số học được dùng chung (hoặc Conv2d với stride=P) ánh xạ các pixel mảng ảnh đã làm phẳng thành vector D chiều |
| CLS token (Token CLS) | "Token phân lớp" | Vector học được chèn vào đầu chuỗi mà trạng thái ẩn cuối cùng của nó đại diện cho toàn bộ hình ảnh; tùy chọn vào năm 2026 |
| Register token (Token thanh ghi) | "Token hấp thụ" | Các token học được bổ sung nhằm hấp thụ các nhiễu giả chú ý có chuẩn cao mà ViT hình thành trong quá trình tiền huấn luyện |
| Position embedding (Nhúng vị trí) | "Thông tin vị trí" | Vector hoặc phép quay theo từng vị trí giúp chuỗi nhận biết được thứ tự; 2D-RoPE là lựa chọn mặc định hiện đại |
| Grid (Lưới mảng ảnh) | "Lưới patch" | Mảng 2 chiều (H/P) x (W/P) chứa các mảng ảnh ứng với độ phân giải và kích thước mảng ảnh cho trước |
| NaFlex | "Độ phân giải linh hoạt gốc" | Tính năng của SigLIP 2: một mô hình duy nhất phục vụ nhiều tỷ lệ khung hình và độ phân giải mà không cần huấn luyện lại |
| Backbone (Mạng xương sống) | "Tháp thị giác" | Bộ mã hóa hình ảnh tiền huấn luyện có các đầu ra token mảng ảnh nạp vào LLM trong một mô hình VLM |
| Pooling (Gộp biểu diễn) | "Tóm tắt cấp độ ảnh" | Chiến lược biến các token mảng ảnh thành một vector duy nhất: CLS, trung bình, gộp chú ý (attention pool), hoặc dựa trên thanh ghi |
| Patch 14 vs 16 | "Lưới mịn hơn vs thô hơn" | Mảng 14 tạo nhiều token trên mỗi ảnh hơn, độ trung thực tốt hơn cho OCR, chậm hơn; mảng 16 là mặc định cổ điển |

## Tài liệu tham khảo (Further Reading)

- [Dosovitskiy et al. — An Image is Worth 16x16 Words (arXiv:2010.11929)](https://arxiv.org/abs/2010.11929) — ViT nguyên bản.
- [He et al. — Masked Autoencoders Are Scalable Vision Learners (arXiv:2111.06377)](https://arxiv.org/abs/2111.06377) — MAE, tiền huấn luyện tự giám sát.
- [Oquab et al. — DINOv2 (arXiv:2304.07193)](https://arxiv.org/abs/2304.07193) — Tự chưng cất ở quy mô lớn, không cần nhãn.
- [Darcet et al. — Vision Transformers Need Registers (arXiv:2309.16588)](https://arxiv.org/abs/2309.16588) — Token thanh ghi và phân tích nhiễu giả.
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — Tháp thị giác mặc định năm 2026.
- [Zhai et al. — Scaling Vision Transformers (arXiv:2106.04560)](https://arxiv.org/abs/2106.04560) — Các quy luật mở rộng thực nghiệm.
