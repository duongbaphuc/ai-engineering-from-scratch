# Lượng tử hóa trong môi trường Production — AWQ, GPTQ, GGUF K-quants, FP8, MXFP4/NVFP4

> Định dạng lượng tử hóa (quantization format) không phải là một lựa chọn áp dụng chung cho mọi bài toán — nó phụ thuộc chặt chẽ vào phần cứng, động cơ phục vụ (serving engine) và khối lượng công việc thực tế. GGUF Q4_K_M hoặc Q5_K_M thống trị môi trường CPU và thiết bị biên (edge), được triển khai thông qua llama.cpp và Ollama. GPTQ chiếm ưu thế bên trong vLLM khi bạn cần chạy cơ chế đa LoRA (multi-LoRA) trên cùng một mô hình nền tảng. AWQ kết hợp với các nhân tính toán Marlin-AWQ mang lại thông lượng ~741 tok/s trên mô hình lớp 7B cùng chỉ số Pass@1 tốt nhất ở định dạng INT4 — trở thành cấu hình mặc định cho các trung tâm dữ liệu trong năm 2026. FP8 giữ vai trò trung gian cân bằng trên Hopper, Ada và Blackwell — gần như không làm suy giảm chất lượng và được hỗ trợ rộng rãi. NVFP4 và MXFP4 (cơ chế vi định tỉ lệ / microscaling trên Blackwell) là những định dạng lượng tử hóa rất sâu và đòi hỏi phải thẩm định độ chính xác trên từng khối. Có hai cái bẫy thường gặp khiến các đội ngũ kỹ thuật trả giá đắt: tập dữ liệu hiệu chuẩn (calibration dataset) bắt buộc phải khớp với miền dữ liệu triển khai thực tế, và dung lượng KV cache tách biệt hoàn toàn với lượng tử hóa trọng số — bài học từ AWQ cho thấy việc "mô hình giờ chỉ còn 4 GB" thường quên mất khoản 10–30 GB bộ nhớ dành riêng cho KV cache khi vận hành ở kích thước lô thực tế.

**Phân loại:** Học tập (Learn)
**Ngôn ngữ:** Python (thư viện chuẩn, công cụ so sánh bộ nhớ và thông lượng mô phỏng giữa các định dạng)
**Điều kiện tiên quyết:** Phase 10 · 13 (Nền tảng lượng tử hóa), Phase 17 · 04 (Serving Engine Internals)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Nêu tên sáu định dạng lượng tử hóa phổ biến trong môi trường production và vùng ứng dụng tối ưu (sweet spots) của từng định dạng vào năm 2026.
- Lựa chọn định dạng chuẩn xác dựa trên phần cứng (CPU so với GPU, Hopper so với Blackwell), động cơ phục vụ (vLLM, TRT-LLM, llama.cpp), và loại tải công việc (trò chuyện thông thường, suy luận logic, đa LoRA).
- Tính toán lượng bộ nhớ trọng số tiết kiệm được và dung lượng bộ nhớ đệm KV cache không bị ảnh hưởng đối với một định dạng cụ thể.
- Chỉ ra cạm bẫy của tập dữ liệu hiệu chuẩn làm suy giảm chất lượng của mô hình lượng tử hóa trên lưu lượng truy cập chuyên biệt theo miền.

## Vấn đề

Lượng tử hóa giúp giảm đáng kể dung lượng bộ nhớ và tiết kiệm băng thông HBM — đây chính xác là những gì pha giải mã (decode) đang cần. Một mô hình 70B ở định dạng FP16 chiếm tới 140 GB trọng số. Lượng tử hóa trọng số xuống INT4 (bằng AWQ hoặc GPTQ) giúp mô hình giảm xuống chỉ còn 35 GB — vừa vặn trên một card H100 duy nhất và vẫn còn đủ không gian cho KV cache. Điều này cực kỳ quan trọng vì khi chạy 128 chuỗi đồng thời với độ dài ngữ cảnh 2k token, riêng KV cache đã ngốn từ 20 đến 30 GB.

Tuy nhiên, lượng tử hóa không phải là giải pháp miễn phí. Việc lượng tử hóa quá sâu sẽ làm suy giảm chất lượng đầu ra, đặc biệt là ở các tác vụ đòi hỏi suy luận logic nặng. Các định dạng khác nhau tương thích với các động cơ phục vụ khác nhau. Các kiến trúc phần cứng khác nhau hỗ trợ các mức độ chính xác số học khác nhau ở cấp độ phần cứng. Bức tranh định dạng năm 2026 rất đa dạng và bạn không thể sao chép nguyên xi lựa chọn của người khác — bạn phải đưa ra quyết định dựa trên chính ngăn xếp công nghệ của mình.

## Khái niệm

### Sáu định dạng lượng tử hóa

| Định dạng | Số bit | Điểm tối ưu (Sweet spot) | Động cơ phục vụ (Engines) |
|---|---|---|---|
| GGUF Q4_K_M / Q5_K_M | 4–5 | CPU, thiết bị biên, máy tính xách tay | llama.cpp, Ollama |
| GPTQ | 4–8 | Đa LoRA trên vLLM | vLLM, TGI |
| AWQ | 4 | GPU production tại trung tâm dữ liệu | vLLM (Marlin-AWQ), TGI |
| FP8 | 8 | Trung tâm dữ liệu Hopper/Ada/Blackwell | vLLM, TRT-LLM, SGLang |
| MXFP4 | 4 | Đa người dùng trên Blackwell | TRT-LLM |
| NVFP4 | 4 | Đa người dùng trên Blackwell | TRT-LLM |

### GGUF — Chuẩn mặc định cho CPU/Edge

GGUF thực chất là một định dạng tệp chứ không đơn thuần là một phương pháp lượng tử hóa — nó đóng gói các biến thể K-quant (Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0) trong cùng một vùng chứa. Q4_K_M và Q5_K_M là các chuẩn mặc định trong môi trường production — đạt chất lượng xấp xỉ BF16 ở mức 4–5 bit. Đây là lựa chọn tối ưu nhất khi phục vụ trên CPU hoặc thiết bị biên vì llama.cpp là động cơ suy luận trên CPU nhanh nhất hiện nay.

Tổn thất thông lượng khi chạy trong vLLM: chỉ đạt ~93 tok/s trên mô hình 7B — định dạng này không được tối ưu hóa cho các nhân tính toán của GPU. Chỉ sử dụng GGUF khi mục tiêu triển khai là CPU hoặc thiết bị biên. Không dùng cho các trường hợp khác.

### GPTQ — Phục vụ đa LoRA trong vLLM

GPTQ là thuật toán lượng tử hóa hậu huấn luyện (post-training quantization) có sử dụng một lượt hiệu chuẩn. Các nhân tính toán Marlin giúp tăng tốc GPTQ đáng kể trên GPU (tăng 2,6 lần tốc độ so với GPTQ không dùng Marlin), đạt khoảng ~712 tok/s trên mô hình 7B.

Lợi thế độc nhất: GPTQ-Int4 hỗ trợ các bộ chuyển đổi LoRA (LoRA adapters) trực tiếp trong vLLM. Nếu bạn đang phục vụ một mô hình nền tảng cùng với 10–50 biến thể tinh chỉnh khác nhau (mỗi biến thể là một LoRA), GPTQ là con đường duy nhất dành cho bạn. Tính đến đầu năm 2026, NVFP4 vẫn chưa hỗ trợ cơ chế LoRA này.

### AWQ — Chuẩn mặc định cho GPU trung tâm dữ liệu

AWQ (Activation-aware Weight Quantization - Lượng tử hóa trọng số nhận biết kích hoạt) bảo vệ khoảng ~1% các trọng số quan trọng nhất (salient weights) trong quá trình lượng tử hóa. Các nhân tính toán Marlin-AWQ mang lại tốc độ vượt trội gấp 10,9 lần so với cách triển khai cơ bản, đạt ~741 tok/s trên mô hình 7B và đạt điểm Pass@1 cao nhất trong số các định dạng INT4.

Hãy ưu tiên chọn AWQ cho các hệ thống phục vụ GPU mới, trừ khi bạn bắt buộc phải dùng đa LoRA (chuyển sang GPTQ) hoặc muốn tối ưu hóa sâu trên Blackwell FP4 (chuyển sang NVFP4).

### FP8 — Điểm cân bằng đáng tin cậy

Định dạng dấu phẩy động 8-bit (FP8) gần như không làm mất mát độ chính xác và được hỗ trợ rộng rãi. Các lõi Hopper Tensor Cores tăng tốc FP8 ở cấp độ phần cứng nguyên bản. Blackwell kế thừa và phát huy tối đa lợi thế này. FP8 là lựa chọn an toàn và mặc định trong năm 2026 khi chất lượng là yếu tố sống còn không thể thỏa hiệp (suy luận logic, y tế, sinh mã lệnh). Mức tiết kiệm bộ nhớ chỉ bằng một nửa so với INT4, nhưng rủi ro suy giảm chất lượng là cực kỳ thấp.

### MXFP4 / NVFP4 — Lượng tử hóa sâu trên Blackwell

Định dạng vi định tỉ lệ FP4 (Microscaling FP4). Mỗi khối trọng số nhỏ sở hữu một hệ số tỉ lệ riêng biệt. Đây là phương pháp lượng tử hóa sâu nhưng được tăng tốc trực tiếp bằng phần cứng trên các lõi Blackwell Tensor Cores. Giảm một nửa số byte trên mỗi token so với FP8 — đây chính là nguồn gốc tạo nên ưu thế kinh tế đã nêu trong Phase 17 · 07.

Các điểm cần lưu ý:
- Chưa hỗ trợ LoRA (tính đến đầu năm 2026).
- Sự suy giảm chất lượng thể hiện rõ trên các khối lượng công việc đòi hỏi suy luận logic nặng.
- Bắt buộc phải đánh giá kiểm định trên tập eval set riêng cho từng mô hình cụ thể.

### Bẫy dữ liệu hiệu chuẩn

AWQ và GPTQ đều cần một tập dữ liệu hiệu chuẩn (calibration dataset) — thông thường người ta hay dùng C4 hoặc WikiText. Đối với các mô hình chuyên biệt theo miền (mã lệnh, y tế, pháp lý), việc hiệu chuẩn trên văn bản web thông thường sẽ khiến thuật toán nhận diện sai các trọng số quan trọng cần bảo vệ. Kết quả là điểm Pass@1 trên HumanEval có thể tụt dốc vài điểm phần trăm.

Giải pháp khắc phục: luôn hiệu chuẩn trên dữ liệu thực tế thuộc đúng miền nghiệp vụ. Chỉ cần vài trăm mẫu dữ liệu đặc thù là đủ. Kiểm thử kỹ lưỡng trên tập eval set trước khi đưa vào vận hành.

### Bẫy bộ nhớ đệm KV cache

AWQ nén trọng số mô hình xuống 4-bit. Nhưng KV cache là một cấu trúc bộ nhớ độc lập và vẫn duy trì ở mức FP16/FP8. Xét một mô hình 70B được nén bằng AWQ:

- Trọng số: ~35 GB (từ mức 140 GB của INT4).
- KV cache khi phục vụ 128 chuỗi đồng thời × ngữ cảnh 2k token: ~20 GB.
- Kích hoạt (Activations): ~5 GB.
- Tổng cộng: ~60 GB — vừa vặn trên một card H100 80GB.

Lập luận ngây thơ rằng "tôi đã nén mô hình xuống còn 4 GB" thường bỏ quên mất 30–50 GB bộ nhớ còn lại. Hãy luôn lập ngân sách bộ nhớ HBM một cách toàn diện.

Ngoài ra, lượng tử hóa KV cache (FP8 KV hoặc INT8 KV) là một quyết định hoàn toàn độc lập với các đánh đổi riêng — nó ảnh hưởng trực tiếp đến độ chính xác của cơ chế chú ý và không phải là một giải pháp hoàn toàn miễn phí.

### AWQ INT4 tiềm ẩn rủi ro cho các tác vụ suy luận logic

Chuỗi suy nghĩ (chain-of-thought), giải toán, sinh mã lệnh với ngữ cảnh dài — đây là những tác vụ bị suy giảm rõ rệt khi lượng tử hóa sâu. AWQ INT4 thường làm mất từ 3 đến 5 điểm trên tập đánh giá MATH. Đối với các khối lượng công việc đặt nặng tính suy luận logic, hãy triển khai FP8 hoặc BF16 và chấp nhận chi phí bộ nhớ cao hơn.

### Cẩm nang lựa chọn năm 2026

- Phục vụ trên CPU/thiết bị biên: Dùng GGUF Q4_K_M.
- Phục vụ trên GPU, hội thoại thông thường, không dùng LoRA: Dùng AWQ.
- Phục vụ trên GPU, cần chạy đa LoRA: Dùng GPTQ với nhân Marlin.
- Tải công việc suy luận logic chuyên sâu: Dùng FP8.
- Trung tâm dữ liệu Blackwell, đã kiểm định chất lượng: Dùng NVFP4 + FP8 KV.
- Trường hợp phân vân: Chạy bài đánh giá trên tập mẫu 1.000 trường hợp cho từng định dạng ứng viên.

```figure
gpu-memory-breakdown
```

## Sử dụng

Tệp `code/main.py` tính toán dấu chân bộ nhớ (trọng số + KV cache + kích hoạt) và thông lượng tương đối giữa sáu định dạng trên nhiều kích thước mô hình khác nhau. Chương trình sẽ chỉ ra thời điểm KV cache chiếm ưu thế tuyệt đối, trường hợp nén trọng số phát huy hiệu quả kinh tế cao nhất, và lý do FP8 luôn là lựa chọn an toàn.

Mô hình trực quan [Bên trong GPU phục vụ LLM](https://aiengineeringfromscratch.com/inside-an-llm-serving-gpu?dtype=fp8&users=64) minh họa định dạng FP8 trên NVIDIA L4. Dung lượng trọng số giảm từ 15,28 GB xuống 8,81 GB, dung lượng vùng đệm KV cache tăng từ 24.094 lên 142.369 token, và sản lượng đầu ra đo được ở mức 64 người dùng tăng vọt từ 250,85 lên 735,91 token mỗi giây.

## Phát hành

Bài học này cung cấp tệp `outputs/skill-quantization-picker.md`. Dựa trên cấu hình phần cứng, kích thước mô hình, loại tải công việc và ngưỡng dung sai chất lượng, công cụ này sẽ lựa chọn định dạng tối ưu và lập kế hoạch hiệu chuẩn/kiểm định chi tiết.

## Bài tập

1. Chạy `code/main.py`. Với mô hình 70B phục vụ 128 chuỗi đồng thời với ngữ cảnh 2k token, hãy tính tổng dung lượng HBM cho từng định dạng. Định dạng nào cho phép bạn vận hành vừa vặn trên một GPU H100 80GB duy nhất?
2. Bạn sở hữu một mô hình sinh mã lập trình 7B. Hãy chọn định dạng lượng tử hóa phù hợp và giải thích lý do. Nếu phán đoán về dung sai chất lượng của bạn bị sai, giải pháp khắc phục là gì?
3. Tính toán kích thước tập dữ liệu hiệu chuẩn cần thiết để chạy AWQ cho một mô hình chuyên ngành y tế. Tại sao việc đưa quá nhiều dữ liệu vào pha hiệu chuẩn không phải lúc nào cũng mang lại kết quả tốt hơn?
4. Đọc bài báo khoa học hoặc ghi chú phát hành của nhân Marlin-AWQ. Hãy giải thích ngắn gọn trong ba câu lý do AWQ đạt thông lượng 741 tok/s trên mô hình 7B trong khi GPTQ thuần túy chỉ đạt ~712 tok/s.
5. Khi nào việc kết hợp trọng số AWQ với KV cache FP8 mang lại lợi ích rõ rệt so với việc giữ KV cache ở định dạng BF16?

## Thuật ngữ then chốt

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|---|---|---|
| GGUF | "định dạng llama.cpp" | Định dạng tệp đóng gói các biến thể K-quant; chuẩn mặc định cho CPU/Edge |
| Q4_K_M | "Q4 K M" | K-quant 4-bit mức trung bình (medium); chuẩn production mặc định của GGUF |
| GPTQ | "lượng tử hóa GPTQ" | Lượng tử hóa INT4 hậu huấn luyện có hiệu chuẩn; hỗ trợ đa LoRA trong vLLM |
| AWQ | "lượng tử hóa AWQ" | INT4 nhận biết kích hoạt; dùng nhân Marlin; đạt điểm Pass@1 tốt nhất ở INT4 |
| Marlin kernels | "nhân tính toán INT4 nhanh" | Các nhân CUDA tùy biến cho INT4 trên Hopper; tăng tốc tới 10 lần |
| FP8 | "số thực 8-bit" | Chuẩn độ chính xác an toàn mặc định trên Hopper/Ada/Blackwell |
| MXFP4 / NVFP4 | "vi lượng tử 4-bit" | Định dạng FP 4-bit của Blackwell với hệ số tỉ lệ theo từng khối riêng biệt |
| Calibration dataset | "dữ liệu hiệu chuẩn" | Văn bản đầu vào dùng để xác định tham số lượng tử hóa; bắt buộc phải khớp miền nghiệp vụ |
| KV cache quantization | "lượng tử hóa KV" | Quyết định độc lập với trọng số; ảnh hưởng trực tiếp đến độ chính xác của cơ chế chú ý |

## Tài liệu đọc thêm

- [VRLA Tech — LLM Quantization 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/) — Các bài đo điểm chuẩn so sánh chi tiết.
- [Jarvis Labs — vLLM Quantization Complete Guide](https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks) — Số liệu thông lượng theo từng định dạng.
- [PremAI — GGUF vs AWQ vs GPTQ vs bitsandbytes 2026](https://blog.premai.io/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026/) — Cẩm nang lựa chọn chi tiết từng định dạng.
- [vLLM docs — Quantization](https://docs.vllm.ai/en/latest/features/quantization/index.html) — Danh sách các định dạng và cờ lệnh được hỗ trợ.
- [AWQ paper (arXiv:2306.00978)](https://arxiv.org/abs/2306.00978) — Bài báo khoa học gốc đề xuất thuật toán AWQ.
- [GPTQ paper (arXiv:2210.17323)](https://arxiv.org/abs/2210.17323) — Bài báo khoa học gốc đề xuất thuật toán GPTQ.
