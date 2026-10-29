# Thị Giác Độ Phân Giải Tùy Ý: Patch-n'-Pack và NaFlex

> Hình ảnh trong thế giới thực không phải là những hình vuông 224x224. Hóa đơn có tỷ lệ 1:3, biểu đồ có tỷ lệ 16:9, ảnh quét y tế có thể lên tới 4096x4096, ảnh chụp màn hình điện thoại có tỷ lệ 9:19.5. Giải pháp trước năm 2024 của các mô hình thị giác - ngôn ngữ (VLM - Vision-Language Model) — co giãn mọi ảnh về một hình vuông cố định — đã vứt bỏ chính những tín hiệu giúp nhận dạng ký tự quang học (OCR), đọc hiểu tài liệu và phân tích bối cảnh độ phân giải cao hoạt động hiệu quả. NaViT (Google, 2023) đã chứng minh rằng bạn có thể đóng gói các mảng ảnh có độ phân giải biến thiên vào một batch transformer duy nhất nhờ mặt nạ chú ý dạng khối đường chéo (block-diagonal masking). M-RoPE của Qwen2-VL (2024) đã loại bỏ hoàn toàn các bảng vị trí tuyệt đối. AnyRes của LLaVA-NeXT đã chia các hình ảnh độ phân giải cao thành lưới gồm ảnh cơ sở kèm các mảnh ảnh cắt nhỏ. Biến thể NaFlex của SigLIP 2 (2025) hiện là bộ mã hóa mặc định cho các VLM nguồn mở mong muốn một checkpoint duy nhất phục vụ mọi tỷ lệ khung hình. Bài học này sẽ triển khai cơ chế patch-n'-pack từ đầu đến cuối.

**Loại:** Xây dựng (Build)  
**Ngôn ngữ:** Python (stdlib, bộ đóng gói mảng ảnh + mặt nạ khối đường chéo)  
**Điều kiện tiên quyết:** Giai đoạn 12 · 01 (Mảng ảnh ViT), Giai đoạn 12 · 05 (LLaVA)  
**Thời lượng:** ~120 phút

## Mục tiêu học tập

- Đóng gói các mảng ảnh từ một batch gồm các hình ảnh có độ phân giải khác nhau thành một chuỗi duy nhất và xây dựng mặt nạ chú ý dạng khối đường chéo.
- Lựa chọn giữa cơ chế lát gạch AnyRes (LLaVA-NeXT), NaFlex (SigLIP 2) và M-RoPE (Qwen2-VL) cho một tác vụ cụ thể.
- Tính toán ngân sách token cho tác vụ OCR, biểu đồ và ảnh chụp thông thường mà không cần co giãn kích thước ảnh.
- Nêu tên 3 trường hợp thất bại của phương pháp co giãn về hình vuông: văn bản bị méo dạng, nội dung bị cắt xén, token bị lãng phí do đệm (padding).

## Vấn đề

Mô hình Transformer mong đợi dữ liệu dạng chuỗi. Một batch thông thường là một tập hợp các chuỗi có cùng độ dài xếp chồng lên nhau. Nếu hình ảnh của bạn luôn là 224x224, bạn luôn nhận được 196 token mảng ảnh, không cần đệm (padding), bài toán coi như hoàn thành. Huấn luyện trên 224, suy luận trên 224, không bao giờ phải bận tâm về độ phân giải nữa.

Nhưng thế giới thực tế không vận hành như vậy. Tài liệu văn bản thường theo chiều dọc (khổ 8.5x11 inch, tỷ lệ khoảng 2:3). Ảnh chụp màn hình biểu đồ nằm theo chiều ngang (16:9). Hóa đơn thanh toán dài và hẹp (1:3). Ảnh quét y tế xuất ra ở 2048x2048 hoặc lớn hơn. Ảnh chụp màn hình điện thoại di động là 1170x2532 (tỷ lệ 0.46:1).

Ba giải pháp trước năm 2024 và lý do chúng thất bại:

1. Co giãn về một hình vuông cố định (224x224 hoặc 336x336). Thao tác bóp méo hình dạng làm biến dạng văn bản và khuôn mặt. Việc giảm tỷ lệ làm phá hủy nhãn biểu đồ và nội dung OCR. Đây từng là thông lệ phổ biến cho đến khi LLaVA-1.5 ra đời.
2. Cắt cúp (crop) theo một tỷ lệ khung hình cố định. Bạn vứt bỏ phần lớn thông tin của bức ảnh, và việc chọn vùng cắt xén nào lại là một bài toán thị giác nan giải khác.
3. Đệm thêm pixel theo cạnh dài nhất. Giúp tránh méo ảnh nhưng gây lãng phí hơn 50% số token cho phần đệm trống đối với các ảnh dọc. Cơ chế chú ý chịu chi phí bậc hai trên toàn bộ các token đệm vô nghĩa đó.

Lời giải cho giai đoạn 2024–2025: để transformer tiếp nhận các mảng ảnh ở đúng độ phân giải gốc của bức ảnh, và tìm cách đóng gói một batch gồm các ảnh đa dạng kích thước vào một chuỗi duy nhất mà không lãng phí tài nguyên tính toán.

## Khái niệm

### NaViT và kỹ thuật patch-n'-pack

NaViT (Dehghani et al., 2023) là bài báo chứng minh ý tưởng này hoạt động tốt ở quy mô lớn. Cơ chế hoàn toàn mang tính cơ học:

1. Với mỗi hình ảnh trong batch, tính toán lưới mảng ảnh gốc theo kích thước mảng đã chọn (ví dụ: kích thước 14).
2. Làm phẳng các mảng ảnh của từng ảnh thành chuỗi có độ dài biến thiên tương ứng.
3. Nối chuỗi các mảng ảnh của tất cả hình ảnh thành một chuỗi dài duy nhất cho toàn bộ batch.
4. Xây dựng mặt nạ chú ý dạng khối đường chéo (block-diagonal attention mask) để các mảng ảnh của ảnh A chỉ chú ý bên trong ảnh A.
5. Mang theo thông tin vị trí cho từng mảng ảnh (dùng 2D RoPE hoặc các vector nhúng vị trí phân số).

Một batch gồm 3 hình ảnh ở 336x336 (576 token), 224x224 (256 token) và 448x336 (768 token) sẽ trở thành một chuỗi duy nhất dài 1600 token với mặt nạ khối đường chéo 1600x1600. Không có đệm rỗng. Không có chi phí tính toán thừa. Transformer xử lý được tỷ lệ khung hình tùy ý.

NaViT cũng giới thiệu kỹ thuật loại bỏ mảng ảnh theo tỷ lệ (fractional patch dropping) trong quá trình huấn luyện — loại bỏ ngẫu nhiên 50% các mảng ảnh trên toàn bộ batch — vừa giúp điều chuẩn (regularization) vừa gia tốc huấn luyện. SigLIP 2 kế thừa trực tiếp cơ chế này.

### AnyRes (LLaVA-NeXT)

AnyRes của LLaVA-NeXT là giải pháp thay thế mang tính thực dụng cao. Cho một hình ảnh độ phân giải cao và một bộ mã hóa cố định (CLIP hoặc SigLIP ở 336px), tiến hành lát gạch hình ảnh:

1. Chọn một bố cục lưới từ tập cấu hình định sẵn — (1x1), (1x2), (2x1), (1x3), (3x1), (2x2), v.v. — khớp tốt nhất với tỷ lệ khung hình của ảnh.
2. Lát gạch toàn bộ ảnh theo lưới; mỗi mảnh trở thành một ảnh cắt 336x336.
3. Tạo thêm một ảnh thu nhỏ (thumbnail): toàn bộ hình ảnh được co giãn về 336x336 để làm token bối cảnh toàn cục.
4. Mã hóa từng mảnh qua bộ mã hóa 336 đã đóng băng. Nối chuỗi các token mảnh cắt + token ảnh thu nhỏ.

Với một hình ảnh 672x672 chia lưới 2x2 cộng thêm ảnh thu nhỏ: $4 \times 576 + 576 = 2880$ token thị giác. Chi phí token tăng cao nhưng rất hiệu quả — LLM nhìn thấy cả chi tiết cục bộ lẫn bức tranh toàn cảnh.

AnyRes là lựa chọn hàng đầu khi bộ mã hóa của bạn bị đóng băng và chỉ hỗ trợ duy nhất một độ phân giải. Nhược điểm là số lượng token bùng nổ với các ảnh lớn (một ảnh 1344x1344 với lưới 4x4 tốn $9216 + 576 \approx 9800$ token, tràn ngữ cảnh LLM 8k trước khi kịp chèn văn bản).

### M-RoPE (Qwen2-VL)

Qwen2-VL giới thiệu cơ chế Nhúng vị trí quay đa phương thức (Multimodal Rotary Position Embedding - M-RoPE). Thay vì dùng vị trí phân số của NaViT hay cơ chế lát gạch + thumbnail của AnyRes, mỗi mảng ảnh mang một tọa độ vị trí 3D (thời gian, chiều cao, chiều rộng). Phép quay Query/Key xử lý mượt mà chiều cao $H$, chiều rộng $W$ và chiều dài thời gian tùy ý.

M-RoPE mang lại khả năng xử lý độ phân giải động gốc mà không cần huấn luyện lại. Khi suy luận, bạn đưa vào bất kỳ bức ảnh HxW nào, bộ nhúng mảng ảnh tạo ra $H/14 \times W/14$ token, mỗi token nhận tọa độ $(t=0, r=\text{hàng}, c=\text{cột})$, RoPE xoay ma trận chú ý với tần số chuẩn xác. Qwen2.5-VL và Qwen3-VL tiếp tục phát triển hướng đi này. Kiến trúc V2PE của InternVL3 cũng vận dụng ý tưởng tương tự với cơ chế mã hóa biến thiên theo từng phương thức.

Khác với AnyRes, M-RoPE chỉ tiêu tốn $O(H \times W / P^2)$ token ở độ phân giải gốc — không chịu chi phí cấp số nhân do lát gạch. Khác với NaViT, nó vẫn kỳ vọng xử lý từng ảnh đơn lẻ trên mỗi lượt truyền xuôi. Việc ghép batch giữa các độ phân giải khác nhau vẫn cần lớp patch-n'-pack bổ trợ bên trên.

### NaFlex (SigLIP 2)

NaFlex là chế độ độ phân giải linh hoạt gốc (native-flex mode) trong checkpoint SigLIP 2. Một mô hình duy nhất có thể phục vụ nhiều độ dài chuỗi khác nhau (256, 729, 1024 token) tại thời điểm suy luận. Bên trong, nó sử dụng patch-n'-pack kiểu NaViT trong quá trình huấn luyện và vị trí phân số tuyệt đối cho từng mảng ảnh. Điểm bán chạy nhất: chỉ một checkpoint duy nhất, bạn tự chọn ngân sách token tại thời điểm suy luận dựa trên tác vụ.

Với tác vụ ngữ nghĩa (phân loại, truy xuất), dùng 256 token. Với tác vụ OCR hoặc đọc hiểu biểu đồ, dùng 1024 token. Hoàn toàn không cần huấn luyện lại mô hình.

### Mặt nạ đóng gói (The packing mask)

Mặt nạ chú ý khối đường chéo là nơi hầu hết các kỹ sư mới gặp lỗi khi cài đặt. Cho một chuỗi đóng gói có độ dài `N_total` bao gồm các hình ảnh $i=0..B-1$ với độ dài tương ứng $n_i$, mặt nạ $M$ kích thước `(N_total, N_total)` bằng 1 nếu cả hai chỉ số cùng rơi vào khối của một hình ảnh, ngược lại bằng 0. Bạn có thể xây dựng nó từ danh sách độ dài tích lũy:

```python
offsets = [0, n_0, n_0 + n_1, ..., N_total]
# M[i, j] = 1 khi và chỉ khi tồn tại b sao cho offsets[b] <= i < offsets[b+1] và offsets[b] <= j < offsets[b+1]
```

Trong PyTorch, thao tác này có thể thực hiện ngắn gọn bằng `torch.block_diag`. Đường dẫn chuỗi biến thiên của FlashAttention (`cu_seqlens`) thậm chí bỏ qua hoàn toàn ma trận mặt nạ dày đặc và chú ý thẳng trong từng chuỗi thông qua tensor độ dài tích lũy — nhanh hơn khoảng 10 lần so với dùng mặt nạ dày cho các batch thông thường.

### Ngân sách token (Token budgets)

Lựa chọn chiến lược dựa trên đặc thù bài toán:

- OCR / tài liệu: 1024–4096 token. Dùng SigLIP 2 NaFlex ở mức 1024, hoặc AnyRes lưới 3x3 kèm ảnh thu nhỏ.
- Biểu đồ và giao diện người dùng (UI): 729–1024 token ở độ phân giải gốc 384–448. Dùng độ phân giải động của Qwen2.5-VL kèm kẹp giới hạn pixel tối đa (max pixels cap).
- Ảnh chụp tự nhiên: 256–576 token là đủ. LLM hạ nguồn nhìn thấy đầy đủ nội dung. Hãy đầu tư token ở những nơi có mật độ thông tin cao.
- Video: 64–128 token mỗi khung hình sau khi gộp không gian, tốc độ 2–8 FPS. Bài học 12.17 sẽ đề cập chi tiết.

Quy tắc sản xuất năm 2026: chọn mức trần pixel tối đa cho từng tác vụ, mã hóa theo tỷ lệ khung hình gốc lên tới mức trần đó, đóng gói batch và bỏ qua đệm rỗng. Qwen2.5-VL cung cấp sẵn hai núm điều khiển `min_pixels` và `max_pixels` cho chính mục đích này.

````figure
mm-patch-n-pack
````

## Thực hành (Use It)

`code/main.py` triển khai cơ chế patch-n'-pack cho một batch hỗn hợp các hình ảnh với tọa độ pixel nguyên. Mã nguồn sẽ:

- Nhận một danh sách kích thước hình ảnh (H, W).
- Tính toán độ dài chuỗi mảng ảnh cho từng ảnh với kích thước mảng 14.
- Đóng gói chúng vào một chuỗi duy nhất có tổng độ dài `sum(n_i)`.
- Xây dựng mặt nạ chú ý dạng khối đường chéo (dạng ma trận dày để dễ quan sát).
- So sánh chi phí đóng gói với phương pháp co giãn về hình vuông và lát gạch AnyRes.
- In bảng ngân sách token cho một batch hỗn hợp (hóa đơn, biểu đồ, ảnh chụp màn hình, ảnh phong cảnh).

Hãy chạy mã nguồn. Những con số đầu ra chính là lý do tại sao mọi VLM nguồn mở năm 2026 đều ứng dụng patch-n'-pack.

## Triển khai (Ship It)

Bài học này tạo ra `outputs/skill-resolution-budget-planner.md`. Cho một khối lượng công việc đa dạng tỷ lệ khung hình (OCR, biểu đồ, ảnh chụp, khung hình video) và ngân sách tổng token, kỹ năng này sẽ lựa chọn chiến lược phù hợp (NaFlex, AnyRes, M-RoPE, hoặc hình vuông cố định) và xuất cấu hình chi tiết cho từng yêu cầu. Hãy sử dụng kỹ năng này khi định cỡ VLM cho sản phẩm — nó giúp ngăn ngừa việc bùng nổ token âm thầm gấp 10 lần làm phá vỡ ngân sách độ trễ.

## Bài tập (Exercises)

1. Một tờ hóa đơn có kích thước 600x1500 (tỷ lệ 1:2.5). Với kích thước mảng 14, nó tạo ra bao nhiêu token ở độ phân giải gốc? Sau khi co giãn về hình vuông 336x336 thì có bao nhiêu token? Trong thực tế, phương án nào làm sụt giảm độ chính xác OCR nghiêm trọng hơn?
2. Xây dựng mặt nạ khối đường chéo cho một batch gồm 4 hình ảnh có độ dài chuỗi lần lượt là 256, 576, 729, 1024. Xác minh rằng ma trận chú ý có kích thước 2585x2585 và có đúng $256^2 + 576^2 + 729^2 + 1024^2$ phần tử khác 0.
3. Cho một hình ảnh 1792x896 với kích thước mảng 14, hãy so sánh: (a) co giãn về hình vuông 336 rồi mã hóa, (b) AnyRes lưới 2x1 + ảnh thu nhỏ, (c) M-RoPE ở độ phân giải gốc. Phương án nào dùng ít token nhất? Phương án nào bảo toàn chi tiết tốt nhất?
4. Triển khai kỹ thuật loại bỏ mảng ảnh theo tỷ lệ (fractional patch dropping): cho một chuỗi đóng gói, loại bỏ ngẫu nhiên 50% token theo phân phối đều, và cập nhật lại mặt nạ khối đường chéo tương ứng. Đo lường sự thay đổi về độ thưa (sparsity) của mặt nạ.
5. Đọc Mục 3.2 của bài báo Qwen2-VL (arXiv:2409.12191). Mô tả trong hai câu xem `min_pixels` và `max_pixels` kiểm soát điều gì và tại sao việc thiết lập cả hai ngưỡng biên này lại quan trọng.

## Thuật ngữ chính (Key Terms)

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực sự |
|---|---|---|
| Patch-n'-pack | "Đóng gói kiểu NaViT" | Nối các chuỗi mảng ảnh có độ dài biến thiên từ các hình ảnh khác nhau vào cùng một chiều batch |
| Block-diagonal mask (Mặt nạ khối đường chéo) | "Mặt nạ đóng gói" | Mặt nạ chú ý giới hạn các mảng ảnh của mỗi hình ảnh chỉ được chú ý tới chính chúng, không chú ý sang ảnh lân cận trong chuỗi đóng gói |
| AnyRes | "Lát gạch LLaVA-NeXT" | Chia ảnh độ phân giải cao thành lưới các mảnh có kích thước cố định kèm ảnh thu nhỏ toàn cục; mã hóa từng mảnh bằng bộ mã hóa cố định |
| NaFlex | "SigLIP 2 linh hoạt gốc" | Checkpoint SigLIP 2 duy nhất phục vụ các mức ngân sách 256/729/1024 token khi suy luận mà không cần huấn luyện lại |
| M-RoPE | "RoPE đa phương thức" | Cơ chế nhúng vị trí quay 3D (thời gian, hàng, cột) xử lý tùy ý kích thước H, W, T mà không cần bảng vị trí tĩnh |
| cu_seqlens | "Đóng gói FlashAttention" | Tensor độ dài tích lũy mà luồng varlen của FlashAttention sử dụng để thay thế ma trận mặt nạ khối đường chéo dày đặc |
| min_pixels / max_pixels | "Ngưỡng chặn độ phân giải" | Các tham số điều khiển theo từng yêu cầu của Qwen2.5-VL nhằm giới hạn số lượng token trên các đầu vào quá nhỏ hoặc quá lớn |
| Visual token budget (Ngân sách token thị giác) | "Số token trên mỗi ảnh" | Số lượng token mảng ảnh ước tính xuất ra cho mỗi ảnh; quyết định ngân sách câu nhắc và chi phí chú ý của LLM |

## Tài liệu tham khảo (Further Reading)

- [Dehghani et al. — Patch n' Pack: NaViT (arXiv:2307.06304)](https://arxiv.org/abs/2307.06304) — Kỹ thuật đóng gói mảng ảnh NaViT.
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191) — Kiến trúc Qwen2-VL và M-RoPE.
- [Laurençon et al. — What matters when building vision-language models? (Idefics2, arXiv:2405.02246)](https://arxiv.org/abs/2405.02246) — Phân tích thực nghiệm xây dựng VLM của Idefics2.
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — SigLIP 2 và NaFlex.
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923) — Báo cáo kỹ thuật Qwen2.5-VL.
