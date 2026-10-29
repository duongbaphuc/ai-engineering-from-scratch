# Biên dịch suy luận chuyên biệt hóa theo phần cứng — FP8 và NVFP4 trên Blackwell

> Biên dịch suy luận chuyên biệt hóa theo phần cứng (hardware-specialized inference compilation) đánh đổi tính khả chuyển (portability) lấy thông lượng (throughput), và TensorRT-LLM — công cụ dành riêng cho NVIDIA, được tinh chỉnh tối ưu cho Blackwell — là ví dụ rõ ràng nhất cho thấy sự đánh đổi này mang lại hiệu quả vượt bậc. Trên hệ thống GB200 NVL72 kết hợp tầng điều phối Dynamo, SemiAnalysis InferenceX ghi nhận mức chi phí chỉ 0,012 USD trên một triệu token cho mô hình 120B vào quý 1–quý 2 năm 2026, so với 0,09 USD/triệu token trên H100 + vLLM — chênh lệch kinh tế tới 7 lần. Ngăn xếp này kết hợp đồng thời hai định dạng dấu phẩy động cùng hai kỹ thuật phục vụ: FP8 giữ vai trò then chốt cho KV cache và các nhân chú ý (attention kernels) vì cung cấp dải động (dynamic range) cần thiết; NVFP4 (vi định tỉ lệ 4-bit / 4-bit microscaling) đảm nhiệm trọng số (weights) và kích hoạt (activations); dự đoán đa token (multi-token prediction - MTP) cùng kiến trúc phân tách tiền điền/giải mã (disaggregated prefill/decode) gia tăng thêm từ 2 đến 3 lần hiệu năng. Hỗ trợ mô hình Day-0 tải trực tiếp trọng số FP4 mà không cần chuyển đổi hậu huấn luyện (post-training conversion). Điểm cần lưu ý cho các đội ngũ kỹ thuật năm 2026: TRT-LLM tuy là mã nguồn mở nhưng gắn chặt độc quyền với NVIDIA — tối ưu hóa sâu cho CUDA và Blackwell — do đó việc triển khai nó đồng nghĩa với việc đánh đổi tính khả chuyển để lấy thông lượng tối đa. Hãy phân tích kỹ bài toán kinh tế trên cơ cấu mô hình và hạ tầng phần cứng trước khi quyết định cam kết.

**Phân loại:** Học tập (Learn)
**Ngôn ngữ:** Python (thư viện chuẩn, công cụ tính toán bộ nhớ và chi phí FP8/NVFP4 mô phỏng)
**Điều kiện tiên quyết:** Phase 17 · 04 (Serving Engine Internals), Phase 10 · 13 (Quantization)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Giải thích lý do FP8 vẫn là ngưỡng sàn tối thiểu cho KV cache và cơ chế chú ý ngay cả khi trọng số đã chuyển sang NVFP4.
- Tính toán dung lượng bộ nhớ HBM chiếm dụng của một mô hình biên giới (frontier model) dưới các định dạng BF16, FP8 và NVFP4, đồng thời phân tích nguồn gốc của mức tiết kiệm tài nguyên.
- Liệt kê các tính năng chuyên biệt của kiến trúc Blackwell mà TRT-LLM khai thác (FP4 day-0, MTP, phục vụ phân tách, các hàm nguyên thủy all-to-all).
- Đưa ra quyết định thời điểm việc phụ thuộc vào hệ sinh thái NVIDIA của TRT-LLM là xứng đáng so với mức chênh lệch chi phí 7 lần trước vLLM trên Hopper.

## Vấn đề

Ranh giới kinh tế trong suy luận mô hình năm 2026 xoay quanh câu hỏi "thu được bao nhiêu token trên mỗi đô-la chi phí". Câu trả lời phụ thuộc vào bốn lựa chọn xếp chồng lên nhau: thế hệ phần cứng (Hopper H100/H200 so với Blackwell B200/GB200), độ chính xác số học (BF16 → FP8 → NVFP4), động cơ phục vụ (serving engine: vLLM so với SGLang so với TRT-LLM), và tầng điều phối (cơ bản so với phân tách so với Dynamo).

Trên Hopper chạy vLLM, mô hình hỗn hợp chuyên gia (Mixture-of-Experts - MoE) kích thước 120B tiêu tốn khoảng ~0,09 USD trên một triệu token. Trên Blackwell chạy TRT-LLM + Dynamo, cùng mô hình đó chỉ tốn ~0,012 USD — rẻ hơn 7 lần. Một phần khoảng cách này đến từ phần cứng thuần túy (Blackwell đạt thông lượng LLM trên mỗi GPU gấp 11–15 lần Hopper). Phần còn lại đến từ ngăn xếp phần mềm: trọng số FP4, mô hình phác thảo MTP, phân tách các pha tiền điền/giải mã (prefill/decode), và kết nối all-to-all qua NVLink 5 để truyền thông tin giữa các chuyên gia MoE.

Bạn không thể tái lập hiệu năng này bên ngoài ngăn xếp công nghệ của NVIDIA. Đó chính là sự đánh đổi — từ bỏ tính khả chuyển đa nền tảng để đổi lấy hiệu quả kinh tế vượt trội. Hiểu rõ từng thành phần trong ngăn xếp đóng góp bao nhiêu phần trăm vào khoảng cách này là trọng tâm của bài học.

## Khái niệm

### Tại sao FP8 vẫn là ngưỡng sàn cho KV cache

Một sai lầm phổ biến vào năm 2026: giả định rằng NVFP4 có thể áp dụng cho mọi thành phần. Thực tế không phải vậy. KV cache bắt buộc phải dùng FP8 (dấu phẩy động 8-bit) vì nó lưu trữ các vector khóa (keys) và giá trị (values) của cơ chế chú ý trải dài trên một dải động rất rộng. Việc lượng tử hóa KV cache xuống FP4 dẫn đến suy giảm độ chính xác nghiêm trọng — phần đuôi của phân phối giá trị bị cắt cụt và điểm chú ý (attention scores) bị sụp đổ. Các bit số mũ (exponent bits) của FP8 mang lại cho KV cache dải giá trị cần thiết.

NVFP4 (giai đoạn 2025–2026) được áp dụng cho trọng số và kích hoạt. Cơ chế vi định tỉ lệ (microscaling): mỗi khối trọng số nhỏ sở hữu một hệ số tỉ lệ (scale factor) riêng biệt, giúp các khối nhỏ này có thể bao quát các dải động khác nhau mà không làm mất mát độ chính xác theo tỉ lệ trên toàn tensor. Đối với kích hoạt, FP4 vẫn duy trì tốt độ chính xác vì các kích hoạt chỉ biến thiên trong phạm vi hẹp bên trong từng tầng.

Cấu hình điển hình trên Blackwell:

- Trọng số: NVFP4 (vi định tỉ lệ 4-bit).
- Kích hoạt: NVFP4.
- KV cache: FP8.
- Bộ tích lũy chú ý (Attention accumulator): FP32 (đảm bảo tính ổn định số học cho softmax).

### Các hàm nguyên thủy chuyên biệt cho Blackwell mà TRT-LLM khai thác

- **Trọng số FP4 Day-0**: các nhà cung cấp mô hình phát hành trực tiếp trọng số ở định dạng FP4; TRT-LLM nạp trực tiếp mà không cần chuyển đổi hậu huấn luyện. Không cần chạy thêm bước AWQ hay GPTQ cho FP4.
- **Dự đoán đa token (Multi-token prediction - MTP)**: cùng ý tưởng với EAGLE (Phase 17 · 05) nhưng được tích hợp trực tiếp vào bản dựng nội tại của TRT-LLM.
- **Phục vụ phân tách (Disaggregated serving)**: tách riêng các nhóm GPU chuyên xử lý tiền điền (prefill) và giải mã (decode), truyền KV cache thông qua NVLink hoặc InfiniBand. Cùng ý tưởng với Dynamo (Phase 17 · 20).
- **Các hàm nguyên thủy truyền thông all-to-all**: NVLink 5 giúp giảm độ trễ truyền thông giữa các chuyên gia MoE xuống 3 lần so với Hopper. Các nhân tính toán MoE của TRT-LLM được tinh chỉnh chuyên biệt cho cơ chế này.
- **Vi định tỉ lệ NVFP4 + MXFP8**: xử lý hệ số tỉ lệ được tăng tốc trực tiếp bằng phần cứng trên các lõi Blackwell Tensor Cores.

### Các con số cần ghi nhớ

- HGX B200 đạt 0,02 USD/M token trên mô hình GPT-OSS-120B thông qua TRT-LLM.
- GB200 NVL72 đạt 0,012 USD/M token thông qua Dynamo (điều phối TRT-LLM).
- Cấu hình H100 + vLLM tiêu tốn khoảng ~0,09 USD/M token trên khối lượng công việc tương đương.
- Tăng 2,8 lần thông lượng chỉ sau ba tháng cập nhật các phiên bản của TRT-LLM (năm 2026).
- Thông lượng LLM trên mỗi GPU tăng 11–15 lần khi so sánh Blackwell với Hopper.
- MLPerf Inference v6.0 (tháng 4 năm 2026): Blackwell chiếm ưu thế áp đảo ở tất cả các tác vụ dự thi.

### Cái giá thực tế về chất lượng khi dùng FP4

NVFP4 là một bước lượng tử hóa rất mạnh mẽ. Trên các khối lượng công việc đòi hỏi suy luận logic nặng (chuỗi suy nghĩ / chain-of-thought, toán học, sinh mã lệnh với ngữ cảnh dài), trọng số FP4 bị suy giảm chất lượng rõ rệt. Việc hiệu chuẩn theo từng khối (per-block calibration) giúp giảm thiểu nhưng không thể loại bỏ hoàn toàn hiện tượng này. Các đội ngũ triển khai mô hình suy luận chuyên sâu thường áp dụng giải pháp dung hòa: trọng số FP8 + kích hoạt FP4, hoặc giữ nguyên phần cứng H200 với FP8 toàn diện.

Nguyên tắc bắt buộc: luôn đánh giá chất lượng tác vụ trên tập dữ liệu kiểm thử (eval set) của riêng bạn trước khi cam kết triển khai trọng số NVFP4 vào môi trường production.

### Tại sao đây là quyết định phụ thuộc độc quyền NVIDIA

TRT-LLM được viết bằng C++ + CUDA + các nhân tính toán đóng mã nguồn. Các mô hình bắt buộc phải được biên dịch cho từng mã SKU GPU cụ thể. Không hỗ trợ AMD, không Intel, không ARM. Nếu chiến lược hạ tầng của bạn là đa nhà cung cấp (multi-vendor), TRT-LLM không thể áp dụng cho toàn bộ các tầng phục vụ — bạn vẫn có thể phục vụ từ vLLM trên các phần cứng hỗn hợp. Nhưng nếu bạn hoàn toàn sử dụng NVIDIA, khoảng cách chi phí 7 lần hoàn toàn bù đắp cho sự phụ thuộc này.

### Công thức thực hành năm 2026

Với hóa đơn suy luận hàng năm từ 100 triệu USD trở lên, việc tiếp tục vận hành trên Hopper + vLLM đồng nghĩa với việc lãng phí gấp 7 đến 10 lần ngân sách. Hãy chuyển dịch các khối lượng công việc chiếm tỉ trọng chi phí lớn sang Blackwell + TRT-LLM + Dynamo. Duy trì tầng thử nghiệm trên H100 + vLLM để giữ tốc độ lặp nhanh của các mô hình mới. Kiểm định nghiêm ngặt chất lượng trên từng mô hình chuyển đổi sang NVFP4 trước khi đưa lên production.

### Lợi thế cộng hưởng từ phân tách kiến trúc phục vụ

Khả năng phục vụ phân tách của TRT-LLM (tách riêng nhóm prefill và decode) được trình bày chi tiết trong Phase 17 · 20. Trên Blackwell, hệ số nhân được khuếch đại theo cấp số nhân: trọng số FP4 × tăng tốc MTP × bố trí phân tách tài nguyên × định tuyến nhận biết bộ nhớ đệm (cache-aware routing). Con số chênh lệch 7 lần giả định rằng toàn bộ ngăn xếp này được kích hoạt đồng thời.

```figure
pipeline-parallel
```

## Sử dụng

Tệp `code/main.py` tính toán dấu chân bộ nhớ HBM, thông lượng giải mã (trong chế độ giới hạn băng thông bộ nhớ / memory-bound regime), và chi phí USD trên mỗi triệu token cho một mô hình trên ba ngăn xếp công nghệ: H100 + BF16 + vLLM, H100 + FP8 + vLLM, B200 + NVFP4/FP8 + TRT-LLM. Hãy chạy chương trình để thấy hiệu ứng tích lũy và tỷ lệ đóng góp của từng thay đổi vào khoảng cách hiệu năng chung.

## Phát hành

Bài học này cung cấp tệp `outputs/skill-trtllm-blackwell-advisor.md`. Dựa trên khối lượng công việc, kích thước mô hình và sản lượng token hàng năm, công cụ này sẽ đưa ra quyết định liệu việc chuyển sang ngăn xếp Blackwell + TRT-LLM có đáng để chấp nhận sự phụ thuộc độc quyền vào NVIDIA hay không.

## Bài tập

1. Chạy `code/main.py`. Trên mô hình MoE 120B với 30% tham số hoạt động, tính toán thông lượng giải mã bị giới hạn bởi băng thông bộ nhớ trên H100 BF16, H100 FP8, và B200 NVFP4/FP8. Bước nhảy vọt lớn nhất đến từ đâu?
2. Một khách hàng chi 2 triệu USD mỗi năm cho hệ thống H100 + vLLM. Số lượng GPU Blackwell tối thiểu cần đầu tư để hòa vốn chi phí di chuyển sang TRT-LLM trong vòng 12 tháng là bao nhiêu, xét trên khoảng cách kinh tế 7 lần?
3. Bạn nhận thấy điểm số suy giảm 3 điểm trên tập đánh giá MATH sau khi chuyển đổi trọng số sang NVFP4. Hãy nêu hai lộ trình phục hồi: một lộ trình ưu tiên chất lượng (giữ trọng số FP8), và một lộ trình ưu tiên chi phí (hiệu chuẩn với dữ liệu nội miền / in-domain data).
4. Đọc kết quả suy luận từ báo cáo MLPerf v6.0. Tác vụ nào có khoảng cách chênh lệch giữa Blackwell và Hopper nhỏ nhất, và giải thích lý do tại sao?
5. Tính dung lượng HBM cần thiết cho một mô hình 405B ở mức trọng số NVFP4 + KV cache FP8 với độ dài ngữ cảnh 128k token. Mô hình này có nằm vừa trên một node đơn GB200 NVL72 hay không?

## Thuật ngữ then chốt

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|---|---|---|
| FP8 | "số thực 8-bit" | Định dạng dấu phẩy động 8-bit; dùng cho KV cache và cơ chế chú ý nhờ dải động rộng |
| NVFP4 | "vi lượng tử 4-bit" | Định dạng FP vi định tỉ lệ 4-bit của NVIDIA; áp dụng cho trọng số và kích hoạt trên Blackwell |
| MXFP8 | "MX 8-bit" | Biến thể FP8 vi định tỉ lệ; được tăng tốc phần cứng trên các lõi Blackwell Tensor Cores |
| Day-0 FP4 | "phát hành trọng số FP4" | Nhà cung cấp mô hình phát hành sẵn trọng số ở định dạng FP4; không cần bước chuyển đổi hậu huấn luyện |
| MTP | "dự đoán đa token" | Bộ tạo phác thảo giải mã suy đoán tích hợp sẵn trong TRT-LLM (Phase 17 · 05) |
| Disaggregated serving | "tách prefill/decode" | Tách prefill và decode ra các cụm GPU riêng biệt; truyền KV cache qua NVLink/InfiniBand |
| All-to-all | "truyền thông chuyên gia MoE" | Mẫu truyền thông định tuyến token tới các GPU chứa chuyên gia; NVLink 5 giảm độ trễ 3 lần |
| InferenceX | "chuẩn đo lường suy luận SemiAnalysis" | Bộ đo lường chi phí trên mỗi token được toàn ngành công nhận trong năm 2026 |

## Tài liệu đọc thêm

- [NVIDIA — Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) — Kết quả đo kiểm MLPerf tháng 4 năm 2026.
- [NVIDIA — MoE Inference on Blackwell](https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/) — Cơ chế NVLink 5 all-to-all và các nhân tính toán MoE.
- [TensorRT-LLM Overview](https://nvidia.github.io/TensorRT-LLM/overview.html) — Tài liệu chính thức về công cụ phục vụ suy luận.
- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) — Khung điều phối suy luận phân tán độ trễ thấp phía trên TRT-LLM.
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — Bộ tiêu chuẩn đo lường hiệu năng công bố các chỉ số của Blackwell.
