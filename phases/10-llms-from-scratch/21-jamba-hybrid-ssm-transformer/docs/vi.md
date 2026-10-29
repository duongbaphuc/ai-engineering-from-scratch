# Jamba — Mô hình lai SSM-Transformer (Jamba — Hybrid SSM-Transformer)

> Các mô hình không gian trạng thái (State Space Models - SSM) và Transformer hướng tới những mục tiêu khác biệt. Transformer đánh đổi chi phí bậc hai (quadratic cost) của cơ chế chú ý để lấy chất lượng mô hình vượt trội. SSM mang lại thời gian suy luận tuyến tính và bộ nhớ cố định nhờ cơ chế hồi quy (recurrence), nhưng chất lượng lại tụt lại phía sau. Mô hình Jamba của AI21 (tháng 3/2024) và Jamba 1.5 (tháng 8/2024) đã kết hợp cả hai trong cùng một mô hình: 1 tầng Transformer cho mỗi 7 tầng Mamba, MoE trên mỗi khối xen kẽ, và một cửa sổ ngữ cảnh 256k nằm gọn trên một GPU 80GB duy nhất. Mamba-3 (ICLR 2026) củng cố thêm phía SSM với các không gian trạng thái giá trị phức (complex-valued state spaces) và các phép chiếu MIMO. Bài học này sẽ đọc toàn diện cả hai kiến trúc từ đầu đến cuối và giải thích tại sao công thức kết hợp lai này vẫn đứng vững sau ba năm mở rộng quy mô trong khi các nỗ lực thuần SSM và thuần Transformer cho ngữ cảnh dài đều gặp khó khăn.

**Type:** Learn
**Languages:** Python (stdlib, layer-mix calculator)
**Prerequisites:** Phase 10 · 14 (open-model architectures), Phase 10 · 17 (native sparse attention)
**Time:** ~60 minutes

## Mục tiêu học tập

- Giải thích ba thành phần nguyên thủy trong một khối Jamba — các tầng Transformer, các tầng Mamba, MoE — cùng công thức xen kẽ theo tỷ lệ 1:7:chẵn.
- Nêu rõ cơ chế hồi quy của một SSM ở cấp độ tổng quan và lý do tại sao nó cho phép suy luận với mức tiêu thụ bộ nhớ cố định.
- Tính toán kích thước dấu chân KV cache của một mô hình Jamba ở ngữ cảnh 256k và so sánh với mức mà một mô hình thuần Transformer đòi hỏi.
- Kể tên ba sáng kiến của Mamba-3 (rời rạc hóa hàm mũ - hình thang, cập nhật trạng thái giá trị phức, MIMO) cùng bài toán mà mỗi sáng kiến hướng tới giải quyết.

## Vấn đề đặt ra

Cơ chế chú ý (attention) có độ phức tạp bậc hai theo độ dài chuỗi. Các mô hình không gian trạng thái (SSM) có độ phức tạp tuyến tính. Sự khác biệt đó tăng lũy tiến theo quy mô: ở độ dài 256k token, bản đồ chú ý của Transformer tiêu tốn 65 tỷ phần tử cho mỗi đầu chú ý; trong khi trạng thái hồi quy của một SSM có kích thước cố định bất kể độ dài chuỗi là bao nhiêu.

Các mô hình thuần SSM (Mamba, Mamba-2) đạt độ hỗn loạn (perplexity) ngang ngửa Transformer ở quy mô nhỏ nhưng lại tụt hậu ở các tác vụ theo dõi trạng thái (state tracking) và thất bại ở một số dạng truy xuất trong ngữ cảnh (in-context retrieval). Trực giác ở đây là: SSM nén toàn bộ lịch sử vào một trạng thái cố định, và khi lịch sử quá dài, thông tin sẽ bị rò rỉ. Attention ghi nhớ chính xác mọi thứ nhưng phải trả cái giá tính toán bậc hai.

Giải pháp hiển nhiên: kết hợp cả hai. Đặt các tầng Transformer ở những nơi việc ghi nhớ chính xác đóng vai trò then chốt. Sử dụng các tầng SSM ở những vị trí còn lại. Tinh chỉnh tỷ lệ phân bổ. Jamba là mô hình cấp sản phẩm đầu tiên triển khai công thức lai này ở quy mô lớn (tổng cộng 52B, kích hoạt 12B, ngữ cảnh 256k, chạy trên một GPU 80GB đơn lẻ). Jamba 1.5 mở rộng dòng mô hình này lên 398B tổng cộng / 94B kích hoạt. Mamba-3 (ICLR 2026) là mốc cơ sở thuần SSM tốt nhất hiện nay mà các mô hình lai có thể xây dựng xoay quanh.

Bài học này sẽ đọc toàn bộ ba bài báo khoa học và hình thành mô hình tư duy cho bài toán "chọn tỷ lệ phối hợp chuẩn xác".

## Khái niệm

### Tóm tắt SSM trong một trang

Một mô hình không gian trạng thái xử lý chuỗi $x_1, \dots, x_N$ thông qua một trạng thái kích thước cố định $h$:

```math
\begin{aligned}
h_t &= A h_{t-1} + B x_t \\
y_t &= C h_t
\end{aligned}
```

Tại mỗi bước thời gian, trạng thái tiến hóa thông qua động lực học tuyến tính $A$, nhận đầu vào $B x_t$, và phát ra đầu ra $C h_t$. Các ma trận $A, B, C$ có thể học được. Hãy lưu ý đặc tính tối quan trọng: việc tính toán $y_t$ chỉ cần $h_{t-1}$ và $x_t$, hoàn toàn không cần bất kỳ $x$ nào trước đó. Bộ nhớ là hằng số $O(1)$ trên mỗi token khi suy luận.

Bí quyết nâng cao chất lượng mô hình hóa nằm ở cấu trúc của ma trận $A$. S4 (Gu 2021) sử dụng ma trận có cấu trúc đặc biệt để có thể tính toán hiệu quả dưới dạng phép tích chập dài trong quá trình huấn luyện. Mamba (Gu, Dao 2023) thay thế $A, B, C$ cố định bằng các tham số phụ thuộc vào dữ liệu (phần chọn lọc - "selective"). Mamba-2 (2024) đơn giản hóa cấu trúc hơn nữa. Mamba-3 (2026) bổ sung lại độ phức tạp ở những vị trí cụ thể có chọn lọc.

Đặc tính cốt lõi: đối với một LLM giải mã, một tầng SSM có thể thay thế trực tiếp cho một tầng attention, mang lại trạng thái kích thước cố định trên mỗi tầng thay vì một bộ đệm KV cache phình to liên tục.

### Khối Jamba

Một khối Jamba xen kẽ các tầng dựa trên hai chỉ số:

- $l$: tỷ lệ giữa Attention và Mamba. Jamba sử dụng $l = 8$, nghĩa là 1 tầng Transformer cho mỗi 7 tầng Mamba (7 Mamba + 1 Attention = 8 tầng mỗi nhóm).
- $e$: tần suất áp dụng MoE. Jamba dùng $e = 2$, nghĩa là cứ mỗi tầng xen kẽ sẽ áp dụng MoE.

Trình tự sắp xếp các tầng bên trong một khối:

```text
M  M  M  M  M  M  M  A    (7 Mamba + 1 Attention)
|  M  |  M  |  M  |  M    (trong đó | đánh dấu tầng có áp dụng MoE)
```

Mỗi khối Jamba gồm 8 tầng. Với độ sâu 4 khối (tổng cộng 32 tầng), bạn có 28 tầng Mamba và 4 tầng Attention, trong đó 16 tầng sử dụng MoE.

### Lý do chọn tỷ lệ 1:7

AI21 đã thực hiện các thí nghiệm bóc tách (ablations): tỷ lệ giữa attention và Mamba nào mang lại độ hỗn loạn trên mỗi tham số tốt nhất VÀ khả năng truy xuất trong ngữ cảnh chuẩn xác nhất trên các bộ đánh giá ngữ cảnh dài?

- Quá nhiều attention (1:1): chất lượng tăng lên nhưng bộ nhớ và tốc độ suy giảm nghiêm trọng.
- Quá ít attention (1:15): bộ nhớ cực kỳ tối ưu nhưng khả năng truy xuất trong ngữ cảnh bị thất bại.
- Điểm ngọt (sweet spot): tỷ lệ 1:7 hoặc 1:8.

Trực giác: các tầng Transformer đảm nhiệm việc ghi nhớ chính xác và theo dõi trạng thái. Các tầng Mamba gánh vác phần khối lượng xử lý thông thường với chi phí rất rẻ.

### Mã hóa vị trí

Bản thân các tầng Mamba đã nhận biết được vị trí (thông qua cơ chế hồi quy). Các tầng attention trong các mô hình lai ban đầu không sử dụng RoPE — các tầng SSM tự cung cấp thông tin vị trí. Jamba 1.5 bổ sung RoPE vào các tầng attention để nâng cao khả năng tổng quát hóa trên ngữ cảnh dài hơn, một cải tiến dựa trên đánh giá thực nghiệm ngữ cảnh dài.

### Ngân sách bộ nhớ

Đối với cấu hình kiểu Jamba-1 (32 tầng: 28 Mamba + 4 Attention, hidden 4096, 32 đầu attention):

- KV cache (chỉ trên các tầng attention): `2 * 4 * 32 * 128 * 256k * 2 = 16 GB` ở ngữ cảnh 256k định dạng BF16. Chỉ có 4 tầng attention tạo ra KV cache. Bản Jamba chính thức sử dụng GQA với 8 đầu KV ở các tầng này, giúp giảm dung lượng xuống còn 4 GB, đúng như con số trong bài báo Jamba.
- Trạng thái SSM: `28 * hidden * state_size` trên mỗi tiền tố token, nhưng đây là kích thước cố định trên mỗi tầng, không tăng theo độ dài chuỗi. Trạng thái Mamba điển hình là 16 trên mỗi đặc trưng, chiều ẩn 4096: `28 * 4096 * 16 * 2 = 3.7 MB` tổng cộng.

So sánh với một mô hình thuần Transformer 32 tầng, cùng chiều ẩn, dùng MHA đầy đủ với 32 đầu: `2 * 32 * 32 * 128 * 256k * 2 = 128 GB` ở 256k BF16. Tức là giảm tới 8 lần dung lượng KV cache. Ngay cả khi so với mốc cơ sở GQA(8) mà hầu hết các mô hình 2024 sử dụng (`2 * 32 * 8 * 128 * 256k * 2 = 32 GB`), một mô hình lai 1:7 với 32 đầu KV ở mức 16 GB vẫn nhỏ hơn 2 lần. Bản Jamba chính thức (4 GB) nhỏ hơn tới 8 lần.

Đó chính là ý nghĩa của tuyên bố "ngữ cảnh 256k trên một GPU 80GB đơn lẻ" từ AI21. Bộ nhớ đệm KV của một mô hình thuần Transformer MHA không thể vừa; ngay cả mốc cơ sở GQA cũng không còn chỗ cho trọng số và kích hoạt; nhưng Jamba thì vừa vặn.

### Mamba-3: Mốc cơ sở thuần SSM vào năm 2026

Mamba-3 (ICLR 2026, arXiv:2603.15569) giới thiệu ba cải tiến lớn ở phía thuần SSM:

1. **Rời rạc hóa hàm mũ - hình thang (Exponential-trapezoidal discretization).** Thay thế phép rời rạc hóa bằng phương pháp Euler trong Mamba-2 bằng một cấu trúc hồi quy có sức biểu diễn mạnh mẽ hơn. Phép toán dạng tích chập được áp dụng trên trạng thái đầu vào bên trong lõi hồi quy, thay vì tích chập ngoài trên $x_t$.

2. **Cập nhật trạng thái giá trị phức (Complex-valued state update).** Các phiên bản Mamba trước đã đơn giản hóa ma trận trạng thái từ số phức (S4) xuống đường chéo thực (Mamba) rồi ma trận đơn vị tỷ lệ (Mamba-2). Mamba-3 tái tích hợp các giá trị phức — tương đương với một phép nhúng xoay (rotary embedding) phụ thuộc vào dữ liệu tác động lên trạng thái. Điều này khôi phục khả năng theo dõi trạng thái mà các phép đơn giản hóa số thực trước đây đã làm mất.

3. **Phép chiếu đa đầu vào đa đầu ra (Multi-input multi-output - MIMO).** Thay vì các phép chiếu vô hướng trên từng đặc trưng riêng rẽ, mô hình sử dụng các phép chiếu ma trận. Điều này cải thiện năng lực mô hình hóa và mức độ tận dụng phần cứng khi suy luận mà không làm tăng độ trễ giải mã.

Ở quy mô 1,5B tham số, Mamba-3 cải thiện độ chính xác hạ nguồn trung bình thêm 0,6 điểm so với Gated DeltaNet; biến thể MIMO cộng thêm 1,2 điểm nữa, nâng tổng mức tăng lên 1,8 điểm. Với cùng kích thước trạng thái, Mamba-3 đạt chất lượng tương đương Mamba-2 chỉ với một nửa dung lượng trạng thái.

Mamba-3 hiện chưa được triển khai trong một mô hình lai thương mại quy mô lớn — nhưng nó là ứng cử viên sáng giá nhất cho nhánh SSM của các dòng mô hình lớp Jamba thế hệ tiếp theo.

### Khi nào nên lựa chọn mô hình lai

Mô hình lai chiến thắng khi:

- Ngữ cảnh đủ dài khiến KV cache của Transformer thuần trở thành gánh nặng lớn (từ 64k token trở lên).
- Tác vụ kết hợp cấu trúc ngắn hạn (thế mạnh của SSM) với việc truy xuất phạm vi dài (cần Transformer).
- Bạn muốn triển khai trên ngân sách bộ nhớ của một GPU đơn lẻ, nơi KV cache thuần của Transformer không thể chứa nổi.

Mô hình lai thua cuộc khi:

- Ngữ cảnh ngắn (dưới 16k token). Chi phí phụ trội của SSM bị lãng phí; Transformer thuần hoạt động tốt hơn.
- Tác vụ đòi hỏi cơ chế chú ý mọi nơi đến mọi nơi (everywhere-to-everywhere attention), chẳng hạn như suy luận logic sâu hoặc đối chiếu chéo nhiều tài liệu. Sự thưa thớt của các tầng attention trong mô hình lai sẽ làm giảm hiệu năng.
- Bạn đang mở rộng quy mô lên các mô hình ranh giới nghìn tỷ tham số. Mô hình thuần Transformer kết hợp MLA và MoE (như kiểu DeepSeek-V3) hiện đang dẫn đầu cuộc đua về năng lực tổng thể.

### Bức tranh toàn cảnh cạnh tranh

| Mô hình | Dòng mô hình | Quy mô | Điểm nhấn độc đáo |
|---|---|---|---|
| Mamba-2 | Thuần SSM | 3B | Thời gian tuyến tính, bộ nhớ cố định |
| Jamba | Lai (Hybrid) | 52B/12B | Ngữ cảnh 256k trên GPU 80GB |
| Jamba 1.5 Large | Lai (Hybrid) | 398B/94B | Ngữ cảnh dài cấp doanh nghiệp |
| Mamba-3 | Thuần SSM | 1.5B (bài báo) | Khôi phục khả năng theo dõi trạng thái |
| DeepSeek-V3 | Thuần Transformer + MoE | 671B/37B | Năng lực ranh giới tiên phong |

Bức tranh công nghệ 2026: Mô hình thuần Transformer MoE thống trị ranh giới tiên phong, nhưng các mô hình lai làm chủ phân khúc ngữ cảnh siêu dài từ 256k trở lên. Những cải tiến theo dõi trạng thái của Mamba-3 có thể đẩy tỷ lệ mô hình lai xuống thấp hơn nữa (nhiều SSM hơn, ít attention hơn) trong các thế hệ tiếp theo.

```figure
swiglu-ffn
```

## Sử dụng trong thực tế

Tệp `code/main.py` là bộ tính toán bộ nhớ cho các kiến trúc lai. Khi nhận vào tỷ lệ SSM-Transformer cùng cấu hình số chiều ẩn / số tầng, nó tính toán:

- Dung lượng KV cache ở ngữ cảnh mục tiêu.
- Bộ nhớ trạng thái SSM.
- Tổng bộ nhớ tại ngữ cảnh $N$ cho nhiều dạng hình thái mô hình khác nhau.

Bộ tính toán hỗ trợ:

- Mốc cơ sở thuần Transformer (KV cache tăng tuyến tính theo $N$).
- Mô hình lai tỷ lệ 1:7 kiểu Jamba.
- Thuần SSM (hoàn toàn không có KV cache).

Các con số được trích xuất trực tiếp từ các bài báo Jamba-1 và Jamba-1.5 đối với các cấu hình đã phát hành, đồng thời ngoại suy cho các biến thể giả định.

Những điểm cần lưu ý khi triển khai trên hệ thống thực tế:

- Hầu hết các máy chủ suy luận môi trường sản xuất (vLLM, SGLang) đều đã hỗ trợ Jamba và Mamba. Cần kiểm tra kỹ phiên bản cụ thể.
- Ở ngữ cảnh 256k, ưu thế bộ nhớ của Jamba thể hiện rõ qua thông lượng xử lý các yêu cầu đồng thời (concurrent-request throughput). Trên cùng dung lượng VRAM, bạn có thể chứa được nhiều chuỗi Jamba hơn chuỗi Transformer.
- Mamba-3 dưới dạng một mô hình độc lập hiện chưa được triển khai thương mại — mới dừng ở mức bản xem trước nghiên cứu ở kích thước 1,5B.

## Đóng gói sản phẩm

Bài học này xây dựng `outputs/skill-hybrid-picker.md`. Khi tiếp nhận thông số khối lượng công việc (đặc thù độ dài ngữ cảnh, cơ cấu tác vụ, ngân sách bộ nhớ), kỹ năng này sẽ đưa ra khuyến nghị lựa chọn giữa thuần Transformer, mô hình lai kiểu Jamba hay thuần SSM, đi kèm lập luận phân tích chi tiết về sự đánh đổi giữa bộ nhớ và chất lượng.

## Bài tập

1. Chạy `code/main.py` để tính toán dung lượng KV cache ở ngữ cảnh 256k cho một mô hình thuần Transformer 32 tầng (chiều ẩn 4096, 32 đầu) và cho mô hình lai Jamba-1 có cấu trúc tương đương. Xác minh mức giảm bộ nhớ xấp xỉ 8 lần như tuyên bố trong bài báo của AI21.

2. Điều chỉnh bộ tính toán để mô phỏng mô hình lai 1:3 (4 Mamba : 1 Attention) và mô hình lai 1:15 (14 Mamba : 1 Attention). Vẽ biểu đồ dung lượng KV cache theo tỷ lệ. Tại tỷ lệ nào thì bộ nhớ KV cache bằng với bộ nhớ trạng thái SSM?

3. Đọc Mục 3 của bài báo Jamba (arXiv:2403.19887). Giải thích tại sao AI21 lại sử dụng Mamba-1 thay vì Mamba-2 dù Mamba-2 có tốc độ nhanh hơn. Gợi ý: phần phân tích bóc tách mô hình lai có ghi chép chi tiết điều này.

4. Tính toán chi phí phụ trội tham số của cấu hình MoE-trên-từng-tầng-xen-kẽ trong Jamba 1.5 Large (tổng 398B, kích hoạt 94B). So sánh tỷ lệ kích hoạt này với DeepSeek-V3 (37B/671B) và giải thích tại sao kiến trúc của Jamba lại đẩy tỷ lệ kích hoạt lên mức cao hơn.

5. Đọc Mục 3 của bài báo Mamba-3 (arXiv:2603.15569). Giải thích trong ba câu tại sao việc cập nhật trạng thái giá trị phức lại tương đương với một phép nhúng xoay (rotary embedding) phụ thuộc vào dữ liệu. Liên hệ câu trả lời với bài suy dẫn RoPE trong Giai đoạn 7 · Bài học 04.

## Thuật ngữ then chốt

| Thuật ngữ | Mọi người thường nói | Ý nghĩa thực tế |
|---|---|---|
| Mô hình không gian trạng thái (SSM) | "Mô hình hồi quy với trạng thái cố định" | Tầng xử lý với cơ chế hồi quy có thể học được $h_t = A h_{t-1} + B x_t$; bộ nhớ hằng số trên mỗi token |
| SSM chọn lọc (Selective SSM) | "Bí quyết của Mamba" | Các tham số A, B, C phụ thuộc vào dữ liệu giúp mô hình có khả năng chọn lọc kiểu cổng kích hoạt ở thời gian tuyến tính |
| Tỷ lệ Attention-to-Mamba | "Bao nhiêu tầng attention" | Trong Jamba, $l = 8$ nghĩa là 1 tầng attention trên mỗi 7 tầng Mamba |
| Khối Jamba (Jamba block) | "Nhóm 8 tầng" | Một tầng attention + bảy tầng Mamba + MoE ở các vị trí xen kẽ |
| Trạng thái SSM (SSM state) | "Bộ đệm ẩn" | Trạng thái kích thước cố định trên mỗi tầng thay thế cho KV cache đối với các tầng Mamba |
| Ngữ cảnh 256k (256k context) | "Con số biểu tượng của Jamba" | Độ dài chuỗi mà Jamba-1 có thể chạy vừa trên một GPU 80GB đơn lẻ; Transformer thuần không thể làm được ở kích thước đó |
| Mamba-3 | "Thuần SSM năm 2026" | Kiến trúc thuần SSM tốt nhất hiện nay với trạng thái số phức + MIMO; mốc cơ sở để các mô hình lai tái xây dựng xoay quanh |
| MIMO | "Đa đầu vào đa đầu ra" | Sáng kiến của Mamba-3 sử dụng các phép chiếu ma trận thay vì phép chiếu vô hướng trên từng đặc trưng |
| Rời rạc hóa hàm mũ - hình thang | "Cơ chế hồi quy của Mamba-3" | Phép hồi quy giàu sức biểu đạt hơn, bao hàm phép rời rạc hóa Euler của Mamba-2 |
| Kiến trúc lai (Hybrid architecture) | "Trộn lẫn attention và SSM" | Bất kỳ mô hình nào xen kẽ các tầng Transformer và SSM; Jamba là hình mẫu tiêu biểu trong môi trường sản xuất |

## Đọc thêm

- [Lieber et al. — Jamba: A Hybrid Transformer-Mamba Language Model (arXiv:2403.19887)](https://arxiv.org/abs/2403.19887) — bài báo gốc về Jamba, phân tích bóc tách tỷ lệ, tuyên bố ngữ cảnh 256k
- [AI21 — Jamba 1.5: Hybrid Transformer-Mamba at Scale (arXiv:2408.12570)](https://arxiv.org/abs/2408.12570) — dòng mô hình mở rộng quy mô, các bản phát hành công khai 398B/94B và 12B/52B
- [Gu, Dao — Mamba: Linear-Time Sequence Modeling with Selective State Spaces (arXiv:2312.00752)](https://arxiv.org/abs/2312.00752) — bài báo về SSM chọn lọc làm nền tảng cho Jamba
- [Dao, Gu — Mamba-2 (arXiv:2405.21060)](https://arxiv.org/abs/2405.21060) — thế hệ kế nhiệm đơn giản hóa không gian trạng thái có cấu trúc
- [Lahoti et al. — Mamba-3 (arXiv:2603.15569, ICLR 2026)](https://arxiv.org/abs/2603.15569) — trạng thái giá trị phức, MIMO, ranh giới thuần SSM năm 2026
- [Gu et al. — Efficiently Modeling Long Sequences with Structured State Spaces (arXiv:2111.00396)](https://arxiv.org/abs/2111.00396) — bài báo S4, điểm khởi đầu cho phả hệ SSM trong các mô hình ngôn ngữ lớn
