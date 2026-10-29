# Flamingo và Chú Ý Chéo Có Cổng cho VLM Học Ít Mẫu

> Mô hình Flamingo (2022) của DeepMind đã làm được hai điều mang tính tiên phong trước bất kỳ ai khác. Nó chứng minh rằng một mô hình duy nhất có thể xử lý các chuỗi hình ảnh, video và văn bản xen kẽ tùy ý (arbitrarily interleaved sequences). Và nó chứng minh rằng các mô hình thị giác - ngôn ngữ (VLM - Vision-Language Model) có khả năng học trong ngữ cảnh (in-context learning) — chỉ cần cung cấp một câu nhắc học ít mẫu (few-shot prompt) với 3 cặp ví dụ (hình ảnh, chú thích) là mô hình có thể chú thích một hình ảnh mới mà không cần bất kỳ bước cập nhật gradient nào. Cơ chế cốt lõi: các lớp chú ý chéo có cổng (gated cross-attention layers), được chèn xen kẽ giữa các lớp hiện có của một LLM đóng băng, với một cổng hàm tanh học được khởi tạo tại giá trị 0 nhằm bảo toàn hoàn hảo năng lực văn bản của LLM tại thời điểm khởi tạo. Bài học này sẽ phân tích bộ tái lấy mẫu Perceiver (Perceiver resampler) và kiến trúc chú ý chéo có cổng của Flamingo — tiền thân của cơ chế đầu vào xen kẽ trong Gemini và các token thị giác của Idefics2.

**Loại:** Tìm hiểu (Learn)  
**Ngôn ngữ:** Python (stdlib, minh họa cơ chế chú ý chéo có cổng + bộ tái lấy mẫu Perceiver)  
**Điều kiện tiên quyết:** Giai đoạn 12 · 03 (Cầu nối Q-Former của BLIP-2)  
**Thời lượng:** ~120 phút

## Mục tiêu học tập

- Giải thích cách cơ chế chú ý chéo có cổng bảo toàn năng lực văn bản của LLM đóng băng tại thời điểm khởi tạo thông qua `tanh(gate) = 0`.
- Phân tích quy trình của bộ tái lấy mẫu Perceiver: $N$ mảng ảnh → $K$ truy vấn "tiềm ẩn" (latent queries) cố định thông qua chú ý chéo.
- Mô tả cách Flamingo xử lý các chuỗi hình ảnh - văn bản xen kẽ bằng mặt nạ nhân quả (causal masking) tôn trọng vị trí xuất hiện của hình ảnh.
- Tái tạo cấu trúc câu nhắc đa phương thức học ít mẫu (3 ví dụ ảnh - chú thích kèm theo ảnh truy vấn).

## Vấn đề

BLIP-2 đưa 32 token thị giác vào lớp đầu vào của một LLM đóng băng. Cách này hoạt động tốt cho một hình ảnh duy nhất trên mỗi câu nhắc. Nhưng nếu bạn muốn đưa *nhiều* hình ảnh xen kẽ với văn bản, chẳng hạn như "đây là ảnh A, hãy chú thích; đây là ảnh B, hãy chú thích; bây giờ đây là ảnh C, hãy chú thích"? Cơ chế tự chú ý của LLM sẽ phải xử lý cả token hình ảnh và token văn bản trong cùng một luồng duy nhất, và việc kiểm soát xem vị trí nào được phép chú ý đến hình ảnh nào sẽ trở nên vô cùng phức tạp.

Câu trả lời của Flamingo: không thay đổi bất kỳ điều gì trong luồng đầu vào của LLM. Thay vào đó, chèn thêm các lớp chú ý chéo vào giữa các khối hiện có của LLM. Các token văn bản vẫn di chuyển qua phép tự chú ý nhân quả của LLM như bình thường. Cứ sau mỗi vài khối LLM, các token văn bản sẽ thực hiện thêm bước chú ý chéo lên các đặc trưng hình ảnh thông qua một lớp có cổng mới. Chiếc cổng này (được khởi tạo bằng 0) đồng nghĩa với việc tại bước huấn luyện số 0, các lớp mới không gây bất kỳ tác động nào — mô hình hoạt động chính xác như LLM tiền huấn luyện ban đầu. Khi quá trình huấn luyện diễn ra, cổng mở dần và thông tin thị giác bắt đầu chảy vào.

Câu hỏi thứ hai mà Flamingo giải quyết: làm thế nào để xử lý số lượng hình ảnh biến thiên (0, 1 hoặc nhiều ảnh) trong mỗi câu nhắc? Giải pháp là bộ tái lấy mẫu Perceiver (Perceiver resampler) — một mô-đun chú ý chéo nhỏ nhận vào số lượng mảng ảnh bất kỳ và tạo ra một số lượng token tiềm ẩn thị giác cố định cho mỗi ảnh. Tổng số token tăng theo số lượng ảnh, nhưng mỗi token văn bản chỉ đọc các biến tiềm ẩn của một hình ảnh, do đó những gì mà một token văn bản đơn lẻ nhìn thấy trong chú ý chéo luôn có cùng kích thước bất kể câu nhắc chứa bao nhiêu hình ảnh.

## Khái niệm

### LLM đóng băng (The frozen LLM)

Flamingo khởi đầu với một mô hình Chinchilla 70B đóng băng. Toàn bộ 70 tỷ trọng số được giữ nguyên vẹn. Các tầng tự chú ý văn bản và mạng truyền thẳng FFN hiện có vẫn hoạt động bình thường.

### Bộ tái lấy mẫu Perceiver (Perceiver resampler)

Đối với mỗi hình ảnh trong câu nhắc, ViT tạo ra $N$ token mảng ảnh. Bộ tái lấy mẫu Perceiver sở hữu $K$ biến tiềm ẩn học được cố định (Flamingo dùng $K=64$). Mỗi khối tái lấy mẫu bao gồm hai bước phụ:

1. Chú ý chéo: $K$ biến tiềm ẩn chú ý lên $N$ token mảng ảnh (Query lấy từ biến tiềm ẩn, Key/Value lấy từ mảng ảnh).
2. Tự chú ý + FFN bên trong các biến tiềm ẩn.

Sau 6 khối tái lấy mẫu, đầu ra luôn là $K=64$ token thị giác có số chiều 1024, bất kể ViT đã tạo ra bao nhiêu mảng ảnh. Một hình ảnh 224x224 (196 mảng) hay một hình ảnh 480x480 (900 mảng) đều xuất ra đúng 64 token tái lấy mẫu.

Đối với video, bộ tái lấy mẫu được áp dụng theo chiều thời gian: các mảng ảnh của mỗi khung hình tạo ra 64 biến tiềm ẩn, và một cơ chế mã hóa vị trí thời gian giúp mô hình phân biệt $t=0$ với $t=N$. Toàn bộ video sẽ trở thành $T \cdot 64$ token thị giác.

### Chú ý chéo có cổng (Gated cross-attention)

Cứ sau mỗi $M$ lớp của LLM đóng băng (Flamingo chọn $M=4$), một khối chú ý chéo có cổng mới được chèn vào:

```python
x_after_llm_block = llm_block(x_before)
cross = cross_attn(x_after, resampler_output)
gated = tanh(alpha) * cross + x_after
x_before_next_block = gated
```

- `alpha` là một tham số vô hướng học được, khởi tạo bằng 0.
- `tanh(0) = 0`, vì vậy tại thời điểm khởi tạo, nhánh có cổng đóng góp giá trị bằng 0.
- Khi `alpha` dịch chuyển xa khỏi 0, sự đóng góp của chú ý chéo tăng lên một cách mượt mà.
- Kết nối phần dư (residual connection) đảm bảo rằng ngay cả khi cổng mở hoàn toàn, nó cũng không ghi đè lên biểu diễn văn bản của LLM; nó chỉ bổ sung thông tin thị giác lên trên.

Đây là quyết định thiết kế quan trọng nhất trong Flamingo: điều kiện hóa thị giác mang tính cộng dồn (additive), có cổng kiểm soát (gated), và bằng 0 tại thời điểm khởi tạo. Một mô hình Flamingo tại bước 0 là một Chinchilla 70B hoàn hảo trên các đầu vào thuần văn bản.

### Chú ý chéo có mặt nạ cho đầu vào xen kẽ (Masked cross-attention for interleaved inputs)

Trong một câu nhắc như `<image A> chú thích A <image B> chú thích B <image C> ?`, mỗi token văn bản chỉ nên nhìn thấy các hình ảnh xuất hiện trước nó trong chuỗi. Mặt nạ chú ý chéo thực thi quy tắc: token văn bản tại vị trí `t` chỉ chú ý đến các token tái lấy mẫu của hình ảnh `i_t`, trong đó `i_t` là hình ảnh gần nhất xuất hiện trước vị trí `t`; token từ mọi hình ảnh khác đều bị che phủ. Lựa chọn "chỉ nhìn thấy hình ảnh đứng liền trước" hay "nhìn thấy tất cả hình ảnh đứng trước" đều khả thi; Flamingo đã chọn phương án đầu tiên.

### Học ít mẫu trong ngữ cảnh (In-context few-shot learning)

Một câu nhắc của Flamingo có dạng:

```text
<image1> A photo of a cat. <image2> A photo of a dog. <image3> A photo of a
```

Mô hình nhận biết khuôn mẫu hoàn thành câu và xuất ra từ "bird" (hoặc bất cứ thứ gì mà image3 thể hiện). Không có bước cập nhật gradient nào. Khả năng học trong ngữ cảnh của LLM đóng băng được truyền tải thông qua cơ chế chú ý chéo có cổng — đây chính là điểm mấu chốt của bài báo và là lý do kiến trúc này mang tính bước ngoặt.

### Dữ liệu huấn luyện (Training data)

Flamingo được huấn luyện trên ba tập dữ liệu:

1. MultiModal MassiveWeb (M3W): 43 triệu trang web chứa văn bản và hình ảnh xen kẽ, tái tạo lại đúng thứ tự đọc của người dùng.
2. Cặp hình ảnh - văn bản: ALIGN (1,8 tỷ cặp) kết hợp với LTIP (312 triệu cặp), tổng cộng khoảng 2,1 tỷ cặp.
3. Cặp video - văn bản (VTP): 27 triệu đoạn video ngắn.

OBELICS (2023) là tập dữ liệu mở tái hiện lại kho ngữ liệu web xen kẽ này, làm nền tảng huấn luyện cho Idefics, Idefics2 và hầu hết các mô hình mở kiểu Flamingo.

### OpenFlamingo và Otter

- OpenFlamingo (2023) là bản tái tạo mã nguồn mở với kiến trúc giống hệt (bộ tái lấy mẫu Perceiver + chú ý chéo có cổng trên LLaMA hoặc MPT đóng băng). Cung cấp các checkpoint ở quy mô 3B, 4B, 9B. Chất lượng kém hơn Flamingo do LLM gốc nhỏ hơn và ít dữ liệu hơn.
- Otter (2023) xây dựng dựa trên OpenFlamingo kết hợp với tinh chỉnh theo chỉ dẫn trên MIMIC-IT (tập dữ liệu chỉ dẫn đa phương thức), chứng minh rằng chú ý chéo có cổng cũng hoạt động rất tốt cho tác vụ tuân thủ chỉ dẫn (instruction following).

### Các hậu duệ (The descendants)

- Dòng họ Idefics / Idefics2 / Idefics3 của Hugging Face: áp dụng chú ý chéo có cổng nhưng ngày càng được đơn giản hóa (Idefics2 đã loại bỏ bộ tái lấy mẫu để dùng trực tiếp các token mảng ảnh với phép gộp thích ứng).
- Chuyển dịch từ Flamingo sang Chameleon: đến năm 2024, nhiều nhóm nghiên cứu đã chuyển sang cơ chế hòa trộn sớm (early-fusion, Bài học 12.11); tuy nhiên chú ý chéo có cổng kiểu Flamingo vẫn được duy trì trong môi trường sản xuất nơi việc đóng băng mạng xương sống là bắt buộc.
- Đầu vào xen kẽ của Gemini: kế thừa tính linh hoạt của định dạng xen kẽ từ Flamingo, dù cơ chế chi tiết bên trong là độc quyền.

### So sánh với BLIP-2 (Comparison to BLIP-2)

| Đặc điểm | BLIP-2 | Flamingo |
|---|---|---|
| Cầu nối thị giác | Q-Former duy nhất ở đầu vào | Chú ý chéo có cổng ở mỗi $M$ lớp |
| Token thị giác | 32 token mỗi ảnh | 64 token mỗi ảnh trên mỗi lớp chú ý chéo |
| LLM đóng băng | Có | Có |
| Học ít mẫu trong ngữ cảnh | Yếu | Rất mạnh — trọng tâm bài báo |
| Đầu vào xen kẽ | Không hỗ trợ tự nhiên | Có, mục tiêu thiết kế cốt lõi |
| Dữ liệu huấn luyện | 130 triệu cặp | ~2,1 tỷ cặp + 27 triệu clip video + 43 triệu trang xen kẽ |
| Số lượng tham số | 188 triệu tham số được huấn luyện | ~10 tỷ tham số được huấn luyện (các lớp chú ý chéo) |
| Lượng tính toán | Vài ngày trên 8 GPU A100 | Vài tuần trên hàng nghìn chip TPUv4 |

Hãy chọn BLIP-2 cho bài toán VQA ảnh đơn với ngân sách hạn chế. Hãy chọn Flamingo/Idefics2 cho các tác vụ suy luận xen kẽ, học ít mẫu hoặc trên nhiều hình ảnh.

````figure
cross-attention-fusion
````

## Thực hành (Use It)

`code/main.py` minh họa:

1. Một bộ tái lấy mẫu Perceiver trên 36 token mảng ảnh giả lập với 8 biến tiềm ẩn học được (chú ý chéo bằng Python thuần).
2. Một bước chú ý chéo có cổng với `alpha = 0` → đầu ra giống hệt đầu vào (LLM không thay đổi), sau đó `alpha = 2.0` → thông tin thị giác được hòa trộn vào.
3. Bộ xây dựng mặt nạ xen kẽ tạo ra mặt nạ chú ý 2D cho chuỗi "(ảnh 1) (văn bản 1) (ảnh 2) (văn bản 2)".

## Triển khai (Ship It)

Bài học này tạo ra `outputs/skill-gated-bridge-diagnostic.md`. Cho cấu hình của một VLM mở (có dùng bộ tái lấy mẫu hay không, tần suất chú ý chéo, cơ chế cổng), kỹ năng này sẽ nhận diện các yếu tố thuộc dòng dõi Flamingo và giải thích chiến lược đóng băng. Rất hữu ích để chẩn đoán lý do tại sao một lần tinh chỉnh lại làm suy giảm hiệu năng ngôn ngữ của mô hình (nguyên nhân: cổng mở quá rộng và quá nhanh).

## Bài tập (Exercises)

1. Tính toán số lượng tham số thị giác của Flamingo-9B: LLM 9B + 1,4B từ các lớp chú ý chéo có cổng + 64M từ bộ tái lấy mẫu. Tỷ lệ phần trăm tham số được huấn luyện trên tổng số tham số là bao nhiêu?
2. Triển khai nhánh phần dư có cổng `y = tanh(alpha) * cross + x` bằng PyTorch. Hãy chứng minh bằng thực nghiệm rằng với `alpha=0`, `y==x` một cách chính xác tuyệt đối tại thời điểm khởi tạo.
3. Đọc Mục 3.2 của bài báo OpenFlamingo (arXiv:2308.01390) về cách họ xử lý nhiều hình ảnh trong một batch khi mỗi câu nhắc có số lượng ảnh khác nhau. Mô tả chiến lược đệm (padding strategy) của họ.
4. Tại sao mặt nạ chú ý chéo của Flamingo chỉ cho phép một token văn bản chú ý đến *hình ảnh gần nhất đứng trước nó* thay vì tất cả các hình ảnh đứng trước? Đọc Mục 2.4 bài báo Flamingo và giải thích sự đánh đổi này.
5. Học ít mẫu trong ngữ cảnh: xây dựng một câu nhắc gồm 4 ví dụ về "hình ảnh → màu sắc của vật thể chính" cho một biến thể Flamingo mới. Mô tả quy luật độ chính xác kỳ vọng khi bạn thay đổi số lượng ví dụ từ 0 đến 8.

## Thuật ngữ chính (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực sự |
|---|---|---|
| Perceiver resampler (Bộ tái lấy mẫu Perceiver) | "Chú ý chéo biến tiềm ẩn cố định" | Mô-đun tạo ra $K$ token cố định từ một số lượng mảng ảnh đầu vào biến thiên |
| Gated cross-attention (Chú ý chéo có cổng) | "Cầu nối có cổng tanh" | Lớp phần dư $y = \tanh(\alpha) \cdot \text{cross} + x$, với $\alpha$ là tham số học được khởi tạo bằng 0 |
| Interleaved input (Đầu vào xen kẽ) | "Chuỗi hỗn hợp" | Định dạng câu nhắc cho phép hình ảnh và văn bản trộn lẫn tự do theo thứ tự đọc tự nhiên |
| Frozen LLM (LLM đóng băng) | "Không cập nhật gradient LLM" | Trọng số của mô hình ngôn ngữ được giữ nguyên; chỉ huấn luyện bộ tái lấy mẫu và các lớp chú ý chéo |
| Few-shot (Học ít mẫu) | "Các ví dụ trong ngữ cảnh" | Cung cấp một vài cặp (hình ảnh, câu trả lời) trong câu nhắc; mô hình tổng quát hóa mà không cần tinh chỉnh |
| OBELICS | "Kho ngữ liệu web xen kẽ" | Tập dữ liệu mở gồm 141 triệu trang web chứa văn bản và hình ảnh theo thứ tự đọc |
| Chinchilla | "Mô hình nền tảng đóng băng 70B" | LLM văn bản đóng băng của Flamingo, từ bài báo Chinchilla của DeepMind |
| Gate schedule (Lịch trình cổng) | "Tốc độ biến thiên của alpha" | Tốc độ mà cổng chú ý chéo mở ra trong quá trình huấn luyện |
| Cross-attn frequency (Tần suất chú ý chéo) | "Cứ sau mỗi M lớp" | Tần suất một khối chú ý chéo có cổng được chèn vào; Flamingo sử dụng $M=4$ |
| OpenFlamingo | "Bản tái tạo mở" | Checkpoint mã nguồn mở của MosaicML/LAION ở quy mô 3–9B; có kiến trúc giống hệt Flamingo |

## Tài liệu tham khảo (Further Reading)

- [Alayrac et al. — Flamingo (arXiv:2204.14198)](https://arxiv.org/abs/2204.14198) — Bài báo gốc.
- [Awadalla et al. — OpenFlamingo (arXiv:2308.01390)](https://arxiv.org/abs/2308.01390) — Phiên bản tái hiện mã nguồn mở.
- [Laurençon et al. — OBELICS (arXiv:2306.16527)](https://arxiv.org/abs/2306.16527) — Kho ngữ liệu web xen kẽ.
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — Kiến trúc Perceiver tổng quát.
- [Li et al. — Otter (arXiv:2305.03726)](https://arxiv.org/abs/2305.03726) — Hậu duệ Flamingo được tinh chỉnh theo chỉ dẫn.
- [Laurençon et al. — Idefics2 (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246) — Phiên bản đơn giản hóa hiện đại của phương pháp tiếp cận Flamingo.
