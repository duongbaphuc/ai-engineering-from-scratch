# Song song hóa DualPipe (DualPipe Parallelism)

> DeepSeek-V3 được huấn luyện trên 2.048 GPU H800 với các chuyên gia MoE nằm rải rác trên nhiều node. Giao tiếp all-to-all giữa các chuyên gia xuyên node tiêu tốn 1 GPU-giờ truyền thông cho mỗi 1 GPU-giờ tính toán. Các GPU bị nhàn rỗi một nửa thời gian. DualPipe (DeepSeek, tháng 12/2024) là một đường ống hai chiều (bidirectional pipeline) chồng lấn (overlap) tính toán lượt truyền xuôi (forward) và truyền ngược (backward) với các giao tiếp all-to-all do chúng kích hoạt. Hiện tượng bóng bóng (bubble) giảm mạnh, thông lượng tăng vọt, và việc duy trì hai bản sao tham số mô hình (phần "dual" tạo nên tên gọi) có chi phí rất rẻ một khi Song song hóa Chuyên gia (Expert Parallelism - EP) vốn dĩ đã phân tán các chuyên gia trên khắp các rank. Bài học này là bài học dạng Tìm hiểu (Learn) giải thích cặn kẽ những gì DualPipe thực sự thực hiện và lý do tại sao tinh chỉnh DualPipeV của Sea AI Lab loại bỏ chi phí tham số 2x với sự đánh đổi là bong bóng đường ống rộng hơn một chút.

**Type:** Learn
**Languages:** Python (stdlib, schedule simulator)
**Prerequisites:** Phase 10 · 05 (distributed training, FSDP, DeepSpeed), Phase 10 · 14 (open-model architectures and MoE)
**Time:** ~60 minutes

## Mục tiêu học tập

- Nêu tên bốn thành phần của một khối (chunk) forward-backward trong DualPipe và lý do tại sao mỗi thành phần lại có cửa sổ chồng lấn (overlap window) riêng biệt.
- Giải thích vấn đề bong bóng đường ống (pipeline bubble) ở quy mô lớn, và ý nghĩa thực tế của thuật ngữ "không bong bóng" (bubble-free) so với quảng bá tiếp thị.
- Lần vết (trace) thủ công một lịch trình DualPipe cho 8 rank song song đường ống (PP ranks) và 16 micro-batch, xác nhận các luồng truyền xuôi và truyền ngược lấp đầy các khoảng thời gian nhàn rỗi (idle slots) của nhau.
- Trình bày sự đánh đổi mà DualPipeV (Sea AI Lab, 2025) thực hiện: loại bỏ việc nhân bản tham số 2x với cái giá là bong bóng lớn hơn đôi chút khi Song song hóa Chuyên gia không hoạt động.

## Vấn đề đặt ra

Việc huấn luyện một mô hình MoE 671B trên 2.000 GPU H800 vấp phải ba nút thắt cổ chai (bottlenecks) cộng dồn:

1. **Áp lực bộ nhớ (Memory pressure).** Mỗi GPU chỉ giữ một phần của mô hình. Bộ nhớ kích hoạt (activation memory) ở độ dài chuỗi 8k trên 61 tầng với 128 đầu chú ý là khổng lồ.
2. **Bong bóng đường ống (Pipeline bubbles).** Song song hóa đường ống truyền thống (GPipe, 1F1B) khiến các GPU nhàn rỗi trong lúc chờ dữ liệu đầu vào hoặc gradient của tầng tương ứng. Với 8 chặng (stages), khoảng 12% thời gian GPU có thể là bong bóng ngay cả khi lập lịch 1F1B.
3. **Giao tiếp all-to-all xuyên node (Cross-node all-to-all).** MoE kết hợp song song hóa chuyên gia phân tán các chuyên gia trên nhiều node. Mỗi lượt truyền xuôi kích hoạt một lệnh all-to-all để điều phối token đến các chuyên gia tương ứng, và một lệnh all-to-all khác để thu thập kết quả. Với 2.000 GPU, tỷ lệ tính toán trên truyền thông (compute-to-comm ratio) dễ dàng chạm mức 1:1.

Mỗi vấn đề này đều có các giải pháp riêng rẽ: lưu điểm kiểm tra gradient (gradient checkpointing) cho bộ nhớ, Zero Bubble (Sea AI Lab, 2023) cho bong bóng đường ống, các kernel truyền thông song song chuyên gia cho all-to-all. Điểm đột phá của DualPipe là kết hợp chúng hoạt động nhịp nhàng cùng nhau. Bộ lập lịch chồng lấn tính toán và truyền thông ngay bên trong một khối forward-backward đơn lẻ, bơm các micro-batch từ cả hai đầu của đường ống cùng lúc, và tận dụng lịch trình tạo thành để ẩn các lệnh all-to-all vào bên trong các cửa sổ tính toán.

Kết quả được công bố: gần như triệt tiêu hoàn toàn bong bóng đường ống, đạt mức tận dụng GPU (GPU utilization) trên 95% trong đợt huấn luyện 14,8 nghìn tỷ (14.8T) token của DeepSeek-V3.

## Khái niệm

### Nhắc lại về song song hóa đường ống

Chia một mô hình gồm $N$ tầng trên $P$ thiết bị. Thiết bị $i$ nắm giữ các tầng từ $i \times N/P$ đến $(i+1) \times N/P - 1$. Một micro-batch chạy xuôi qua các thiết bị từ 0 đến $P-1$, sau đó chạy ngược từ $P-1$ về 0. Mỗi thiết bị chỉ có thể bắt đầu chặng truyền xuôi khi thiết bị liền trước gửi đầu ra của nó, và chỉ có thể bắt đầu truyền ngược khi thiết bị hạ nguồn gửi gradient thượng nguồn về.

GPipe (Huang et al., 2019) lập lịch từng micro-batch một, gây lãng phí phần lớn thời gian GPU. 1F1B (Narayanan et al., 2021) xen kẽ các lượt truyền xuôi và truyền ngược cho nhiều micro-batch. Zero Bubble (Qi et al., 2023) chia lượt truyền ngược thành hai phần — truyền ngược cho đầu vào (backward-for-input, ký hiệu B) và truyền ngược cho trọng số (backward-for-weights, ký hiệu W) — rồi lập lịch chúng để lấp đầy bong bóng. Sau Zero Bubble, đường ống gần như đã khít.

DualPipe là bước tiến tiếp theo, bổ sung thêm hai ý tưởng cốt lõi:

### Ý tưởng 1: Phân rã khối công việc (Chunk decomposition)

Mỗi khối truyền xuôi được chia thành bốn thành phần:

- **Attention.** Các phép chiếu Q/K/V, cơ chế chú ý (attention), phép chiếu đầu ra (output projection).
- **All-to-all dispatch (Điều phối all-to-all).** Giao tiếp xuyên node gửi các token đến các chuyên gia được phân công.
- **MLP.** Phần tính toán của các chuyên gia MoE.
- **All-to-all combine (Hợp nhất all-to-all).** Giao tiếp xuyên node thu nhận đầu ra của các chuyên gia trở về.

Một khối truyền ngược bổ sung các phiên bản tính gradient tương ứng cho từng thành phần này. DualPipe lập lịch sao cho việc điều phối all-to-all diễn ra song song với phép tính attention của khối tiếp theo, và hợp nhất all-to-all diễn ra song song với phép tính MLP của khối kế sau đó.

### Ý tưởng 2: Lập lịch hai chiều (Bidirectional scheduling)

Hầu hết các bộ lập lịch đường ống đều bơm các micro-batch từ chặng 0 và luân chuyển dần về chặng $P-1$. DualPipe bơm các micro-batch từ CẢ HAI ĐẦU. Chặng 0 xử lý các micro-batch truyền xuôi xuất phát từ đó; chặng $P-1$ cũng xử lý các micro-batch truyền xuôi xuất phát từ chính nó. Hai luồng dữ liệu gặp nhau ở giữa đường ống.

Để cơ chế này hoạt động, thiết bị $i$ phải lưu trữ CẢ tầng đầu đường ống $i$ LẪN tầng cuối đường ống $P - 1 - i$. Đó chính là phần "dual" (kép) của DualPipe: mỗi thiết bị duy trì hai bản sao của các tầng mô hình mà nó cần phục vụ (một bản sao cho mỗi hướng). Ở quy mô của DeepSeek-V3, đây là chi phí nhân đôi tham số (2x parameter replication cost). Chi phí này hoàn toàn chấp nhận được vì Song song hóa Chuyên gia (Expert Parallelism) vốn đã phân tán các chuyên gia MoE ra rất mỏng, khiến việc nhân bản hai lần các tầng không phải chuyên gia (non-expert layers) chỉ chiếm một phần rất nhỏ.

Quan trọng nhất, luồng truyền xuôi theo một hướng và luồng truyền ngược theo hướng ngược lại chồng lấn chính xác vào những vị trí lẽ ra sẽ xuất hiện bong bóng trong lịch trình đơn hướng. Nhờ đó, các bong bóng biến mất.

### Lần vết lịch trình thủ công

Xét $P = 4$ rank, 8 micro-batch, chia thành 4 lượt xuôi / 4 lượt ngược. Thời gian trôi từ trái sang phải; các hàng đại diện cho rank thiết bị.

```text
           Thời gian →
rank 0:  F1 F2 F3 F4  F5R F6R F7R F8R  B1 B2 B3 B4  ...
rank 1:     F1 F2 F3  F4/F5R F6R F7R   B1 B2 ...
rank 2:        F1 F2  F3/F5R F4/F6R    B1 ...
rank 3:           F1  F2/F5R F3/F6R    ...
```

Cách đọc ký hiệu "F4/F5R": rank 1 đang thực thi lượt truyền xuôi của micro-batch 4 (chạy từ trái sang phải trong đường ống) VÀ lượt truyền xuôi của micro-batch 5 (chạy từ phải sang trái) trong cùng một khoảng thời gian. Đó chính là ý nghĩa vận hành của khái niệm "hai chiều" (bidirectional).

Tại rank 2, các luồng giao nhau chồng lấn sớm hơn; tại rank 0 và $P-1$, chúng chồng lấn muộn nhất. Trong giai đoạn ổn định ở giữa của lịch trình, mọi rank đều thực thi lượt truyền xuôi theo hướng X chồng lấn với lượt truyền ngược theo hướng Y. Năng lực tính toán luôn bận rộn. Các lệnh điều phối all-to-all cho lượt truyền xuôi ẩn mình bên trong tính toán truyền ngược. Các lệnh hợp nhất all-to-all ẩn mình bên trong tính toán truyền xuôi. Các bong bóng bị ép tiêu biến hoàn toàn.

### Hạch toán bong bóng

Bong bóng đường ống chuẩn của 1F1B (thời gian lãng phí trên mỗi rank):

```text
bubble_1F1B = (P - 1) * forward_chunk_time
```

Cải tiến Zero Bubble kéo giảm đại lượng này nhưng không thể đưa về 0. DualPipe, trong giai đoạn ổn định, đạt mức bong bóng bằng không nếu số lượng micro-batch chia hết cho 2 lần độ sâu đường ống. Ngoài giai đoạn ổn định (giai đoạn làm nóng - warmup và giai đoạn hạ nhiệt - cooldown), vẫn tồn tại một số bong bóng nhưng chúng không tăng theo số lượng micro-batch — một đặc tính quan trọng được bài báo nhấn mạnh.

Theo ngôn từ tiếp thị: "không bong bóng" (bubble-free). Theo thuật ngữ kỹ thuật: kích thước bong bóng không tăng theo số lượng micro-batch. Phân tích tiếp nối của Sea AI Lab (DualPipeV / Cut-in-half) chỉ ra rằng hiện tượng bong bóng hoàn toàn bằng không chỉ xuất hiện khi Song song hóa Chuyên gia không phải là nút thắt cổ chai; với all-to-all do EP thúc đẩy, việc phải thỏa hiệp lập lịch ở một mức độ nào đó luôn luôn tồn tại.

### DualPipeV — Cải tiến tinh chỉnh

Sea AI Lab (2025) nhận thấy rằng việc nhân đôi tham số 2x là lãng phí khi mục tiêu chính không phải là chồng lấn giao tiếp EP. Lịch trình DualPipeV của họ gập việc bơm hai chiều thành một lịch trình hình chữ V ("V-shape") chạy trên một bản sao tham số duy nhất. Bong bóng rộng hơn một chút so với DualPipe, nhưng tiết kiệm bộ nhớ đáng kể. DeepSeek đã tích hợp DualPipeV vào bản cài đặt mã nguồn mở của DualPipe như một chế độ tắt EP (EP-off mode).

Bảng so sánh đánh đổi:

| Tính năng | DualPipe | DualPipeV | 1F1B | Zero Bubble |
|---|---|---|---|---|
| Bản sao tham số trên mỗi thiết bị | 2 | 1 | 1 | 1 |
| Bong bóng so với số micro-batch | hằng số | tăng nhẹ | tăng | tăng |
| Chồng lấn tính toán - truyền thông | toàn phần | một phần | tối thiểu | một phần |
| Áp dụng khi nào | MoE phụ thuộc nặng vào EP | mô hình dày đặc (dense) hoặc ít EP | đường cơ sở (baseline) | bất kỳ đường ống nào |

### Ý nghĩa đối với đợt huấn luyện 14,8T token

Quá trình tiền huấn luyện (pre-training) của DeepSeek-V3 tiêu tốn 14,8 nghìn tỷ token trên 2.048 GPU H800 trong khoảng 2,8 triệu GPU-giờ. Nếu dùng 1F1B ngây thơ, họ sẽ mất khoảng 12-15% tổng thời gian đó cho bong bóng đường ống — tương đương 340.000 đến 420.000 GPU-giờ, đủ để huấn luyện hoàn chỉnh một mô hình 70B. DualPipe đã thu hồi phần lớn con số đó. Rất khó để định lượng chính xác đóng góp cụ thể nếu không có nhật ký nội bộ, nhưng công bố trong bài báo khẳng định mức tận dụng GPU trung bình đạt trên 95% trong suốt quá trình huấn luyện.

Đối với các đợt huấn luyện nhỏ hơn (dưới 1.000 GPU), DualPipe là quá mức cần thiết — bong bóng đường ống chiếm tỷ trọng nhỏ so với tổng chi phí, và việc huấn luyện mô hình dày đặc hiếm khi chạm đến nút thắt all-to-all. Nhưng đối với việc huấn luyện mô hình MoE tiên phong ở quy mô nhiều nghìn GPU, kỹ thuật này trên thực tế là điều bắt buộc.

### Vị trí trong ngăn xếp công nghệ

- Bổ trợ cho **FSDP** (Phase 10 · 05). FSDP phân mảnh (shard) các tham số mô hình trên các rank; DualPipe lập lịch tính toán trên các rank. Chúng kết hợp hoàn hảo với nhau.
- Tương thích với kỹ thuật phân mảnh gradient **ZeRO-3**. Việc theo dõi sổ sách cho việc nhân bản hai bản sao cần phối hợp nhịp nhàng với các gradient được phân mảnh của ZeRO.
- Đòi hỏi các **kernel all-to-all tùy chỉnh** được tinh chỉnh cho cấu trúc liên kết mạng (topology) cụ thể của cụm máy. Các kernel mã nguồn mở của DeepSeek là bản cài đặt tham chiếu chuẩn mực.

```figure
expert-capacity
```

## Sử dụng trong thực tế

Tệp `code/main.py` là một trình mô phỏng lịch trình đường ống. Tệp nhận vào `(P, n_micro_batches, schedule)` và in ra mức độ tận dụng trong giai đoạn ổn định cho từng phương pháp: 1F1B, Zero Bubble, DualPipe và DualPipeV. Đây là một công cụ giảng dạy — các con số phản ánh đúng các nhận định định tính trong các bài báo khoa học, chứ không phải là tuyên bố về tốc độ tăng tốc thực tế đo được trên môi trường sản xuất.

Giá trị của trình mô phỏng: chạy thử với các giá trị $P$ và số lượng micro-batch khác nhau để quan sát tỷ lệ bong bóng tăng lên ra sao đối với 1F1B nhưng giữ nguyên đối với DualPipe.

Các lưu ý khi tích hợp vào một đợt huấn luyện thực tế:

- Chọn độ sâu song song đường ống chia hết cho số lượng micro-batch của bạn.
- Đảm bảo lưới song song chuyên gia (expert-parallel mesh) hỗ trợ all-to-all hai chiều. Các kernel của DeepSeek là tài liệu tham khảo chính.
- Dự kiến sẽ mất khoảng một tuần gỡ lỗi (debug) lịch trình trong lần đầu tiên thiết lập. Công tác ghi chép sổ sách và điều phối rất phức tạp.
- Giám sát mức độ tận dụng GPU trên từng rank, không chỉ nhìn vào mức trung bình tổng thể. Lợi ích của DualPipe đến từ việc ép chặt các rank chạy chậm nhất (stragglers).

## Đóng gói sản phẩm

Bài học này xây dựng `outputs/skill-dualpipe-planner.md`. Khi được cung cấp thông số kỹ thuật cụm huấn luyện (số lượng GPU, cấu trúc liên kết, đường truyền kết nối, cấu trúc mô hình), kỹ năng này sẽ đề xuất chiến lược song song hóa đường ống, thuật toán lập lịch phù hợp, và tỷ lệ bong bóng dự kiến ở quy mô mục tiêu.

## Bài tập

1. Chạy `code/main.py` với `(P=8, micro_batches=16, schedule=dualpipe)` và `(P=8, micro_batches=16, schedule=1f1b)`. Tính toán chênh lệch mức tận dụng GPU và biểu diễn dưới dạng số GPU-giờ tiết kiệm được trên mỗi triệu token huấn luyện.

2. Phác thảo bảng lịch trình cho `(P=4, micro_batches=8, schedule=dualpipe)` bằng tay. Đánh dấu từng khe thời gian với mã định danh micro-batch và chiều di chuyển. Xác định khe thời gian đầu tiên không còn xuất hiện bong bóng.

3. Đọc Hình 5 trong báo cáo kỹ thuật của DeepSeek-V3 (arXiv:2412.19437). Xác định cửa sổ chồng lấn cho điều phối all-to-all bên trong một khối truyền xuôi DualPipe. Giải thích cách lịch trình tính toán ẩn được thao tác này.

4. Tính chi phí phụ trội tham số 2x của DualPipe đối với mô hình dày đặc 70B có $P=8$ chặng đường ống và mô hình MoE 671B có $P=16$ chặng đường ống. Chứng minh tại sao phần trăm phụ trội trong trường hợp MoE lại nhỏ hơn đáng kể (hầu hết tham số nằm ở các chuyên gia, vốn đã được phân mảnh trên một nhóm EP lớn).

5. So sánh DualPipe với Chimera (bộ lập lịch hai chiều cạnh tranh từ năm 2021). Xác định hai đặc tính cụ thể mà DualPipe bổ sung nhưng Chimera không có, sử dụng Phần 3.4 của bài báo làm tài liệu tham khảo.

## Thuật ngữ then chốt

| Thuật ngữ | Mọi người thường nói | Ý nghĩa thực tế |
|---|---|---|
| Bong bóng đường ống (Pipeline bubble) | "Thời gian nhàn rỗi mỗi rank" | Các chu kỳ GPU bị lãng phí do một chặng đường ống phải chờ đầu vào hoặc gradient |
| 1F1B | "Lịch trình đường ống mặc định" | Lập lịch xen kẽ một lượt xuôi / một lượt ngược; mốc cơ sở mà DualPipe vượt qua |
| Zero Bubble | "Công trình Sea AI Lab 2023" | Tách lượt truyền ngược thành B (gradient đầu vào) và W (gradient trọng số); gần như làm khít hoàn toàn đường ống |
| DualPipe | "Lịch trình DeepSeek-V3" | Đường ống hai chiều kết hợp chồng lấn tính toán - truyền thông; bong bóng không tăng theo số lượng micro-batch |
| DualPipeV | "Cắt đôi (Cut-in-half)" | Tinh chỉnh dạng chữ V giúp loại bỏ nhân đôi tham số 2x với cái giá là bong bóng lớn hơn một chút |
| Khối (Chunk) | "Đơn vị công việc đường ống" | Một lượt truyền xuôi hoặc truyền ngược của một micro-batch qua một chặng đường ống |
| Điều phối all-to-all (All-to-all dispatch) | "Gửi token đến chuyên gia" | Giao tiếp xuyên node định tuyến các token đến các chuyên gia MoE được chỉ định |
| Hợp nhất all-to-all (All-to-all combine) | "Thu nhận đầu ra chuyên gia về" | Giao tiếp xuyên node thu thập đầu ra của các chuyên gia sau khối MLP |
| Song song hóa Chuyên gia (Expert Parallelism - EP) | "Chuyên gia rải trên nhiều GPU" | Phân mảnh các chuyên gia MoE trên các rank để các GPU khác nhau nắm giữ các chuyên gia khác nhau |
| Song song hóa Đường ống (Pipeline Parallelism - PP) | "Các tầng rải trên nhiều GPU" | Phân mảnh các tầng mô hình trên các rank; chiều không gian mà DualPipe lập lịch |
| Tỷ lệ bong bóng (Bubble fraction) | "Thời gian GPU lãng phí" | (thời gian bong bóng / tổng thời gian); tỷ lệ mà DualPipe kéo về gần mức 0 |

## Đọc thêm

- [DeepSeek-AI — Báo cáo kỹ thuật DeepSeek-V3 (arXiv:2412.19437), Mục 3.3.2 và Hình 5](https://arxiv.org/abs/2412.19437) — tài liệu tham khảo chính về DualPipe
- [DeepSeek — Kho mã nguồn DualPipe trên GitHub](https://github.com/deepseek-ai/DualPipe) — bản cài đặt tham chiếu mã nguồn mở, bao gồm chế độ DualPipeV (Cut-in-half)
- [Qi et al. — Zero Bubble Pipeline Parallelism (arXiv:2401.10241, Sea AI Lab 2023)](https://arxiv.org/abs/2401.10241) — công trình tiền thân Zero Bubble
- [Sea AI Lab — DualPipe could be better without the Dual](https://sail.sea.com/blog/articles/63) — phân tích DualPipeV làm nền tảng cho chế độ tắt EP của DeepSeek
- [Narayanan et al. — PipeDream / 1F1B (arXiv:1806.03377, 2018-2021)](https://arxiv.org/abs/1806.03377) — lịch trình 1F1B mà DualPipe so sánh đối chiếu
- [Huang et al. — GPipe (arXiv:1811.06965, 2018)](https://arxiv.org/abs/1811.06965) — bài báo gốc về song song hóa đường ống và vấn đề bong bóng
